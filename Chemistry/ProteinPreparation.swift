import Foundation
import simd

enum ProteinPreparation {

    struct DockingPreparationReport: Sendable {
        var protonationUpdates: Int = 0
        var hydrogensAdded: Int = 0
        var hydrogenMethod: String = "none"
        var rdkitChargeMatches: Int = 0
        var fallbackChargeAtoms: Int = 0
        var nonZeroChargeAtoms: Int = 0
    }

    struct PreparationReport: Sendable {
        var waterCount: Int = 0
        var altConfsRemoved: Int = 0
        var missingResidues: [(chainID: String, gapStart: Int, gapEnd: Int)] = []
        var chainSummary: [(chainID: String, residueCount: Int, type: String)] = []
        var hetGroups: [(name: String, count: Int)] = []
        var clashCount: Int = 0
        var totalAtoms: Int = 0
        var heavyAtoms: Int = 0
    }

    // MARK: - Remove Waters

    static func removeWaters(atoms: [Atom], bonds: [Bond]) -> (atoms: [Atom], bonds: [Bond], removedCount: Int) {
        let waterNames: Set<String> = ["HOH", "WAT", "DOD", "H2O"]
        var indexMap: [Int: Int] = [:]
        var newAtoms: [Atom] = []
        var removedCount = 0

        for (oldIdx, atom) in atoms.enumerated() {
            if waterNames.contains(atom.residueName) {
                removedCount += 1
                continue
            }
            let newIdx = newAtoms.count
            indexMap[oldIdx] = newIdx
            newAtoms.append(Atom(
                id: newIdx,
                element: atom.element,
                position: atom.position,
                name: atom.name,
                residueName: atom.residueName,
                residueSeq: atom.residueSeq,
                chainID: atom.chainID,
                charge: atom.charge,
                formalCharge: atom.formalCharge,
                isHetAtom: atom.isHetAtom,
                occupancy: atom.occupancy,
                tempFactor: atom.tempFactor,
                altLoc: atom.altLoc
            ))
        }

        var newBonds: [Bond] = []
        for bond in bonds {
            if let a = indexMap[bond.atomIndex1], let b = indexMap[bond.atomIndex2] {
                newBonds.append(Bond(id: newBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
            }
        }

        return (newAtoms, newBonds, removedCount)
    }

    // MARK: - Detect Missing Residues

    static func detectMissingResidues(in atoms: [Atom]) -> [(chainID: String, gapStart: Int, gapEnd: Int)] {
        // Group residue sequence numbers by chain
        var chainResidues: [String: Set<Int>] = [:]
        for atom in atoms {
            guard !atom.isHetAtom else { continue }
            chainResidues[atom.chainID, default: []].insert(atom.residueSeq)
        }

        var gaps: [(chainID: String, gapStart: Int, gapEnd: Int)] = []

        for (chainID, seqNums) in chainResidues.sorted(by: { $0.key < $1.key }) {
            let sorted = seqNums.sorted()
            guard sorted.count > 1 else { continue }

            for i in 0..<(sorted.count - 1) {
                let diff = sorted[i + 1] - sorted[i]
                if diff > 1 {
                    gaps.append((chainID: chainID, gapStart: sorted[i] + 1, gapEnd: sorted[i + 1] - 1))
                }
            }
        }

        return gaps
    }

    // MARK: - Analyze Structure

    static func analyze(atoms: [Atom], bonds: [Bond], waterCount: Int = 0) -> PreparationReport {
        var report = PreparationReport()
        report.waterCount = waterCount
        report.totalAtoms = atoms.count
        report.heavyAtoms = atoms.filter { $0.element != .H }.count

        // Chain summary
        var chainResCount: [String: Set<Int>] = [:]
        var chainTypes: [String: String] = [:]
        for atom in atoms {
            chainResCount[atom.chainID, default: []].insert(atom.residueSeq)
            if chainTypes[atom.chainID] == nil {
                chainTypes[atom.chainID] = atom.isHetAtom ? "Ligand" : "Protein"
            }
        }
        report.chainSummary = chainResCount.keys.sorted().map { chainID in
            (chainID: chainID,
             residueCount: chainResCount[chainID]?.count ?? 0,
             type: chainTypes[chainID] ?? "Unknown")
        }

        // HETATM groups
        var hetGroupCounts: [String: Int] = [:]
        for atom in atoms where atom.isHetAtom {
            hetGroupCounts[atom.residueName, default: 0] += 1
        }
        report.hetGroups = hetGroupCounts.map { (name: $0.key, count: $0.value) }
            .sorted { $0.count > $1.count }

        // Missing residues
        report.missingResidues = detectMissingResidues(in: atoms)

        return report
    }

    // MARK: - Shared Docking Preparation

    /// Prepare a receptor copy for docking without mutating the displayed model.
    /// This is intentionally conservative: keep the heavy-atom geometry fixed,
    /// add missing polar hydrogens, merge whatever RDKit heavy-atom charges we can
    /// recover from the raw PDB, and fall back to protonation-derived formal charges
    /// when no partial charge is available.
    static func prepareForDocking(
        atoms: [Atom],
        bonds: [Bond],
        rawPDBContent: String? = nil,
        pH: Float = 7.4
    ) -> (atoms: [Atom], bonds: [Bond], report: DockingPreparationReport) {
        var report = DockingPreparationReport()

        let protonated = Protonation.applyProtonation(atoms: atoms, pH: pH)
        report.protonationUpdates = zip(atoms, protonated).filter { $0.formalCharge != $1.formalCharge }.count

        var workingAtoms = protonated
        var workingBonds = bonds

        if protonated.contains(where: { $0.element == .H }) {
            report.hydrogenMethod = "existing"
        } else {
            let hydrogenated = addPolarHydrogens(atoms: protonated, bonds: bonds)
            workingAtoms = hydrogenated.atoms
            workingBonds = hydrogenated.bonds
            report.hydrogensAdded = max(workingAtoms.count - protonated.count, 0)
            report.hydrogenMethod = "geometry fallback"
        }

        if let pdbContent = rawPDBContent,
           let chargeData = RDKitBridge.computeChargesPDB(pdbContent: pdbContent) {
            let merged = mergeProteinAtoms(currentAtoms: workingAtoms, sourceAtoms: chargeData.atoms) { current, source in
                current.charge = source.charge
            }
            workingAtoms = merged.atoms
            report.rdkitChargeMatches = merged.matchedCount
        }

        let beforeFallback = workingAtoms
        workingAtoms = applyElectrostaticFallback(to: workingAtoms)
        report.fallbackChargeAtoms = zip(beforeFallback, workingAtoms).filter {
            abs($0.charge) <= 0.0001 && abs($1.charge) > 0.0001
        }.count
        report.nonZeroChargeAtoms = workingAtoms.filter { abs($0.charge) > 0.001 }.count

        return (workingAtoms, workingBonds, report)
    }

    // MARK: - Add Polar Hydrogens (preserves all PDB residue info)

    /// Add polar hydrogens (N-H, O-H) to protein backbone and sidechains.
    /// Places H atoms at standard bond distances along estimated bond vectors.
    /// Preserves all existing atom properties (residueName, chainID, residueSeq, etc).
    static func addPolarHydrogens(atoms: [Atom], bonds: [Bond]) -> (atoms: [Atom], bonds: [Bond]) {
        // Build adjacency map
        var neighbors: [Int: [Int]] = [:]
        for bond in bonds {
            neighbors[bond.atomIndex1, default: []].append(bond.atomIndex2)
            neighbors[bond.atomIndex2, default: []].append(bond.atomIndex1)
        }

        var newAtoms = atoms
        var newBonds = bonds
        var addedCount = 0

        // Standard N-H bond distance: 1.01 Å, O-H: 0.96 Å
        let nhDist: Float = 1.01
        let ohDist: Float = 0.96

        for (idx, atom) in atoms.enumerated() {
            let trimmedName = atom.name.trimmingCharacters(in: .whitespaces)

            // Backbone amide N-H (every residue except PRO N-terminus)
            if trimmedName == "N" && atom.element == .N && atom.residueName != "PRO" {
                let nNeighbors = (neighbors[idx] ?? []).filter { atoms[$0].element != .H }
                // Backbone N has 2 heavy neighbors (CA and C of previous residue) or 1 if N-terminus
                if nNeighbors.count <= 2 {
                    let hPos = placeHydrogen(on: atom.position, awayFrom: nNeighbors.map { atoms[$0].position }, distance: nhDist)
                    let hAtom = Atom(
                        id: newAtoms.count, element: .H, position: hPos,
                        name: "H", residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: 0, formalCharge: 0, isHetAtom: atom.isHetAtom
                    )
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: idx, atomIndex2: newAtoms.count, order: .single))
                    newAtoms.append(hAtom)
                    addedCount += 1
                }
            }

            // Sidechain N-H: NZ (Lys), NE/NH1/NH2 (Arg), ND2 (Asn), NE2 (Gln), ND1/NE2 (His)
            if atom.element == .N && !["N", ""].contains(trimmedName) {
                let donors: Set<String> = ["NZ", "NE", "NH1", "NH2", "ND2", "NE2", "ND1", "NE1"]
                if donors.contains(trimmedName) {
                    let nNeighbors = (neighbors[idx] ?? []).filter { atoms[$0].element != .H }
                    let hPos = placeHydrogen(on: atom.position, awayFrom: nNeighbors.map { atoms[$0].position }, distance: nhDist)
                    let hAtom = Atom(
                        id: newAtoms.count, element: .H, position: hPos,
                        name: "HN", residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: 0, formalCharge: 0, isHetAtom: atom.isHetAtom
                    )
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: idx, atomIndex2: newAtoms.count, order: .single))
                    newAtoms.append(hAtom)
                    addedCount += 1
                }
            }

            // Hydroxyl O-H: OG (Ser), OG1 (Thr), OH (Tyr)
            if atom.element == .O {
                let hydroxyls: Set<String> = ["OG", "OG1", "OH"]
                if hydroxyls.contains(trimmedName) {
                    let oNeighbors = (neighbors[idx] ?? []).filter { atoms[$0].element != .H }
                    let hPos = placeHydrogen(on: atom.position, awayFrom: oNeighbors.map { atoms[$0].position }, distance: ohDist)
                    let hAtom = Atom(
                        id: newAtoms.count, element: .H, position: hPos,
                        name: "HO", residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: 0, formalCharge: 0, isHetAtom: atom.isHetAtom
                    )
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: idx, atomIndex2: newAtoms.count, order: .single))
                    newAtoms.append(hAtom)
                    addedCount += 1
                }
            }
        }

        return (newAtoms, newBonds)
    }

    /// Place a hydrogen atom at `distance` from `center`, in the direction opposite to the average
    /// of neighbor positions (so it points away from existing bonds).
    private static func placeHydrogen(on center: SIMD3<Float>, awayFrom neighbors: [SIMD3<Float>], distance: Float) -> SIMD3<Float> {
        if neighbors.isEmpty {
            // No neighbors — place along +Y
            return center + SIMD3<Float>(0, distance, 0)
        }

        // Average direction of neighbors from center
        let avg = neighbors.reduce(SIMD3<Float>.zero) { $0 + ($1 - center) } / Float(neighbors.count)
        let len = simd_length(avg)
        if len < 1e-6 {
            return center + SIMD3<Float>(0, distance, 0)
        }

        // Place H opposite to average neighbor direction
        let dir = -simd_normalize(avg)
        return center + dir * distance
    }

    private struct ProteinAtomIdentity: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let atomName: String
        let atomicNumber: Int
        let isHetAtom: Bool
    }

    private static func proteinAtomIdentity(for atom: Atom) -> ProteinAtomIdentity {
        ProteinAtomIdentity(
            chainID: atom.chainID,
            residueSeq: atom.residueSeq,
            residueName: atom.residueName,
            atomName: atom.name.trimmingCharacters(in: .whitespaces),
            atomicNumber: atom.element.rawValue,
            isHetAtom: atom.isHetAtom
        )
    }

    private static func mergeProteinAtoms(
        currentAtoms: [Atom],
        sourceAtoms: [Atom],
        update: (inout Atom, Atom) -> Void
    ) -> (atoms: [Atom], matchedCount: Int) {
        var buckets: [ProteinAtomIdentity: [Atom]] = [:]
        for atom in sourceAtoms {
            buckets[proteinAtomIdentity(for: atom), default: []].append(atom)
        }

        var offsets: [ProteinAtomIdentity: Int] = [:]
        var updated = currentAtoms
        var matchedCount = 0

        for index in updated.indices {
            let identity = proteinAtomIdentity(for: updated[index])
            let offset = offsets[identity, default: 0]
            guard let bucket = buckets[identity], offset < bucket.count else { continue }
            update(&updated[index], bucket[offset])
            offsets[identity] = offset + 1
            matchedCount += 1
        }

        return (updated, matchedCount)
    }

    private static func applyElectrostaticFallback(to atoms: [Atom]) -> [Atom] {
        atoms.map { atom in
            var atom = atom
            if abs(atom.charge) <= 0.0001 {
                atom.charge = Float(atom.formalCharge)
            }
            return atom
        }
    }
}
