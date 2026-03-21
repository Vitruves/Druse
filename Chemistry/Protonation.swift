import Foundation

/// Standard amino acid pKa values for Henderson-Hasselbalch protonation prediction.
enum Protonation {

    struct TitratableGroup {
        let residueName: String
        let atomName: String // e.g., "NZ" for Lys, "OD1" for Asp
        let pKa: Float
        let chargeWhenProtonated: Int
        let chargeWhenDeprotonated: Int
    }

    /// Standard pKa values for titratable amino acid sidechains.
    static let sidechainGroups: [TitratableGroup] = [
        // Acidic sidechains (lose H+ to become negative)
        TitratableGroup(residueName: "ASP", atomName: "OD1", pKa: 3.65, chargeWhenProtonated: 0, chargeWhenDeprotonated: -1),
        TitratableGroup(residueName: "GLU", atomName: "OE1", pKa: 4.25, chargeWhenProtonated: 0, chargeWhenDeprotonated: -1),
        TitratableGroup(residueName: "CYS", atomName: "SG",  pKa: 8.18, chargeWhenProtonated: 0, chargeWhenDeprotonated: -1),
        TitratableGroup(residueName: "TYR", atomName: "OH",  pKa: 10.07, chargeWhenProtonated: 0, chargeWhenDeprotonated: -1),

        // Basic sidechains (gain H+ to become positive)
        // HIS: both ND1 and NE2 are titratable (tautomers). At pH < pKa, both are protonated (HIP, +1).
        // At pH > pKa, one nitrogen is protonated (HID or HIE, charge 0). We model both sites.
        TitratableGroup(residueName: "HIS", atomName: "ND1", pKa: 6.00, chargeWhenProtonated: 1, chargeWhenDeprotonated: 0),
        TitratableGroup(residueName: "HIS", atomName: "NE2", pKa: 6.00, chargeWhenProtonated: 1, chargeWhenDeprotonated: 0),
        TitratableGroup(residueName: "LYS", atomName: "NZ",  pKa: 10.53, chargeWhenProtonated: 1, chargeWhenDeprotonated: 0),
        TitratableGroup(residueName: "ARG", atomName: "NH1", pKa: 12.48, chargeWhenProtonated: 1, chargeWhenDeprotonated: 0),
    ]

    /// Backbone terminus pKa values.
    static let nTermPKa: Float = 9.69
    static let cTermPKa: Float = 2.34

    /// Kept for backward compatibility.
    static let titratableGroups: [TitratableGroup] = sidechainGroups + [
        TitratableGroup(residueName: "*N-term*", atomName: "N", pKa: nTermPKa, chargeWhenProtonated: 1, chargeWhenDeprotonated: 0),
        TitratableGroup(residueName: "*C-term*", atomName: "OXT", pKa: cTermPKa, chargeWhenProtonated: 0, chargeWhenDeprotonated: -1),
    ]

    /// Henderson-Hasselbalch: fraction protonated = 1 / (1 + 10^(pH - pKa))
    static func fractionProtonated(pH: Float, pKa: Float) -> Float {
        1.0 / (1.0 + powf(10.0, pH - pKa))
    }

    /// Predict formal charges for titratable residues at a given pH.
    /// Returns a list of (atomIndex, predicted formal charge) tuples.
    static func predictCharges(atoms: [Atom], pH: Float = 7.4) -> [(atomIndex: Int, charge: Int)] {
        var results: [(atomIndex: Int, charge: Int)] = []

        // Build chain termini lookup: for each chain, find the smallest and largest
        // residue sequence numbers to identify N-terminal and C-terminal residues.
        var chainMinSeq: [String: Int] = [:]
        var chainMaxSeq: [String: Int] = [:]
        var cTermHasOXT: Set<String> = []
        for atom in atoms {
            guard !atom.isHetAtom else { continue }
            let seq = atom.residueSeq
            if chainMinSeq[atom.chainID] == nil || seq < chainMinSeq[atom.chainID]! {
                chainMinSeq[atom.chainID] = seq
            }
            if chainMaxSeq[atom.chainID] == nil || seq > chainMaxSeq[atom.chainID]! {
                chainMaxSeq[atom.chainID] = seq
            }
            if atom.name.trimmingCharacters(in: .whitespaces) == "OXT" {
                cTermHasOXT.insert("\(atom.chainID):\(seq)")
            }
        }

        for (i, atom) in atoms.enumerated() {
            let trimmedName = atom.name.trimmingCharacters(in: .whitespaces)

            // Sidechain titratable groups
            for group in sidechainGroups {
                guard group.residueName == atom.residueName else { continue }
                guard trimmedName == group.atomName else { continue }

                let frac = fractionProtonated(pH: pH, pKa: group.pKa)
                let charge = frac > 0.5 ? group.chargeWhenProtonated : group.chargeWhenDeprotonated
                results.append((atomIndex: i, charge: charge))
            }

            // N-terminus: backbone "N" atom in the first residue of each chain
            if trimmedName == "N" && !atom.isHetAtom {
                if atom.residueSeq == chainMinSeq[atom.chainID] {
                    let frac = fractionProtonated(pH: pH, pKa: nTermPKa)
                    let charge = frac > 0.5 ? 1 : 0  // protonated = +1, deprotonated = 0
                    results.append((atomIndex: i, charge: charge))
                }
            }

            // C-terminus: prefer OXT when present, otherwise fall back to backbone O.
            if !atom.isHetAtom && (trimmedName == "OXT" || trimmedName == "O") {
                if atom.residueSeq == chainMaxSeq[atom.chainID] {
                    let cTermKey = "\(atom.chainID):\(atom.residueSeq)"
                    if trimmedName == "O", cTermHasOXT.contains(cTermKey) {
                        continue
                    }
                    let frac = fractionProtonated(pH: pH, pKa: cTermPKa)
                    let charge = frac > 0.5 ? 0 : -1  // protonated = 0, deprotonated = -1
                    results.append((atomIndex: i, charge: charge))
                }
            }
        }

        return results
    }

    /// Apply predicted protonation charges to atoms array, returning modified copy.
    static func applyProtonation(atoms: [Atom], pH: Float = 7.4) -> [Atom] {
        let charges = predictCharges(atoms: atoms, pH: pH)
        var result = atoms

        for entry in charges {
            guard entry.atomIndex < result.count else { continue }
            result[entry.atomIndex] = Atom(
                id: result[entry.atomIndex].id,
                element: result[entry.atomIndex].element,
                position: result[entry.atomIndex].position,
                name: result[entry.atomIndex].name,
                residueName: result[entry.atomIndex].residueName,
                residueSeq: result[entry.atomIndex].residueSeq,
                chainID: result[entry.atomIndex].chainID,
                charge: Float(entry.charge),
                formalCharge: entry.charge,
                isHetAtom: result[entry.atomIndex].isHetAtom,
                occupancy: result[entry.atomIndex].occupancy,
                tempFactor: result[entry.atomIndex].tempFactor,
                altLoc: result[entry.atomIndex].altLoc
            )
        }

        return result
    }
}
