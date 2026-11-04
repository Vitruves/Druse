// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

/// Writer for V2000 SDF (Structure-Data File) format.
/// Used for exporting ligand databases and docking results.
enum SDFWriter {

    /// Write a single molecule as a V2000 MOL block. The returned text always ends
    /// with a newline, so blocks can be concatenated directly into a multi-molecule SDF.
    static func molBlock(
        name: String,
        atoms: [Atom],
        bonds: [Bond],
        properties: [String: String] = [:],
        includeTerminator: Bool = true
    ) -> String {
        var lines: [String] = []

        // Header block (3 lines)
        // Molecule name: a single line (an embedded newline would shift the
        // line-positional header and corrupt the counts line).
        lines.append(String(name.split(whereSeparator: \.isNewline).first ?? ""))
        lines.append("  Druse   3D")                   // program/timestamp line
        lines.append("")                               // comment

        // Counts line
        let atomCount = atoms.count
        let bondCount = bonds.count
        lines.append(String(format: "%3d%3d  0  0  0  0  0  0  0  0999 V2000", atomCount, bondCount))

        // Atom block
        for atom in atoms {
            let sym = atom.element.symbol
            let fc = atom.formalCharge
            // V2000 charge encoding: 0=0, 1=+3, 2=+2, 3=+1, 4=doublet, 5=-1, 6=-2, 7=-3
            let chargeCode: Int
            switch fc {
            case 3:  chargeCode = 1
            case 2:  chargeCode = 2
            case 1:  chargeCode = 3
            case -1: chargeCode = 5
            case -2: chargeCode = 6
            case -3: chargeCode = 7
            default: chargeCode = 0
            }

            let paddedSym = sym.padding(toLength: 3, withPad: " ", startingAt: 0)
            lines.append(String(format: "%10.4f%10.4f%10.4f %@ 0%3d  0  0  0  0  0  0  0  0  0  0",
                                atom.position.x, atom.position.y, atom.position.z,
                                paddedSym, chargeCode))
        }

        // Bond block
        for bond in bonds {
            let a1 = bond.atomIndex1 + 1  // 1-based
            let a2 = bond.atomIndex2 + 1
            let bondType: Int
            switch bond.order {
            case .single:   bondType = 1
            case .double:   bondType = 2
            case .triple:   bondType = 3
            case .aromatic: bondType = 4
            }
            lines.append(String(format: "%3d%3d%3d  0  0  0  0", a1, a2, bondType))
        }

        // M CHG lines for non-zero formal charges
        let chargedAtoms = atoms.enumerated().filter { $0.element.formalCharge != 0 }
        if !chargedAtoms.isEmpty {
            // Write in groups of 8
            for chunk in stride(from: 0, to: chargedAtoms.count, by: 8) {
                let end = min(chunk + 8, chargedAtoms.count)
                let group = chargedAtoms[chunk..<end]
                var chgLine = "M  CHG  \(group.count)"
                for (idx, atom) in group {
                    chgLine += String(format: " %3d %3d", idx + 1, atom.formalCharge)
                }
                lines.append(chgLine)
            }
        }

        lines.append("M  END")

        // Properties (SDF data items)
        for (key, value) in properties.sorted(by: { $0.key < $1.key }) {
            lines.append("> <\(key)>")
            lines.append(value)
            lines.append("")
        }

        if includeTerminator {
            lines.append("$$$$")
        }

        return lines.joined(separator: "\n") + "\n"
    }

    /// Write a single Molecule object as V2000 SDF.
    @MainActor
    static func writeMolecule(_ mol: Molecule) -> String {
        molBlock(name: mol.name, atoms: mol.atoms, bonds: mol.bonds)
    }

    /// Write all docking result poses as a multi-molecule SDF with score properties.
    @MainActor
    static func writeDockingResults(_ results: [DockingResult], ligand: Molecule) -> String {
        let template = heavyAtomTemplate(ligand)
        return writeDockingResults(results, ligandName: ligand.name, atoms: template.atoms, bonds: template.bonds)
    }

    /// Docked poses as a multi-molecule SDF: the heavy-atom ligand at each pose's coordinates,
    /// named `<ligand>_pose<rank>`, with every field of `DockingPoseRecord` (score in the run's
    /// units, score terms, cluster, validity, refinement) as SDF properties. `rank` is the pose's
    /// 1-based rank in the full results list, so filtered exports keep their original ranks.
    @MainActor
    static func writeDockingPoses(
        _ poses: [(rank: Int, result: DockingResult)],
        ligand: Molecule,
        method: ScoringMethod
    ) -> String {
        let template = heavyAtomTemplate(ligand)
        return poses.map { pose in
            var atoms = template.atoms
            for j in 0..<min(atoms.count, pose.result.transformedAtomPositions.count) {
                atoms[j].position = pose.result.transformedAtomPositions[j]
            }
            var properties = Dictionary(
                DockingPoseRecord.fields(rank: pose.rank, result: pose.result, method: method)
                    .filter { !$0.value.isEmpty }
                    .map { ($0.name, $0.value) },
                uniquingKeysWith: { first, _ in first }
            )
            if let smiles = ligand.smiles, !smiles.isEmpty { properties["SMILES"] = smiles }
            return molBlock(name: "\(ligand.name)_pose\(pose.rank)", atoms: atoms, bonds: template.bonds,
                            properties: properties)
        }.joined()
    }

    /// The ligand's heavy atoms and the bonds between them, re-indexed onto that list.
    /// Docked poses carry heavy-atom positions only, in this order.
    @MainActor
    static func heavyAtomTemplate(_ ligand: Molecule) -> (atoms: [Atom], bonds: [Bond]) {
        var heavyIndex: [Int: Int] = [:]
        var heavyAtoms: [Atom] = []
        for (i, atom) in ligand.atoms.enumerated() where atom.element != .H {
            heavyIndex[i] = heavyAtoms.count
            heavyAtoms.append(atom)
        }
        var heavyBonds: [Bond] = []
        for bond in ligand.bonds {
            guard let a = heavyIndex[bond.atomIndex1], let b = heavyIndex[bond.atomIndex2] else { continue }
            heavyBonds.append(Bond(id: heavyBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
        }
        return (heavyAtoms, heavyBonds)
    }

    /// Write multiple molecules to an SDF string.
    static func write(molecules: [(name: String, atoms: [Atom], bonds: [Bond], properties: [String: String])]) -> String {
        molecules.map { mol in
            molBlock(name: mol.name, atoms: mol.atoms, bonds: mol.bonds, properties: mol.properties)
        }.joined()
    }

    /// Write docking results as SDF.
    static func writeDockingResults(_ results: [DockingResult], ligandName: String, atoms: [Atom], bonds: [Bond]) -> String {
        var blocks: [String] = []

        for (i, result) in results.enumerated() {
            // Build atoms at docked positions
            var dockedAtoms = atoms
            for j in 0..<min(atoms.count, result.transformedAtomPositions.count) {
                dockedAtoms[j] = Atom(
                    id: j, element: atoms[j].element,
                    position: result.transformedAtomPositions[j],
                    name: atoms[j].name,
                    residueName: atoms[j].residueName,
                    residueSeq: atoms[j].residueSeq,
                    chainID: atoms[j].chainID,
                    charge: atoms[j].charge,
                    formalCharge: atoms[j].formalCharge,
                    isHetAtom: atoms[j].isHetAtom
                )
            }

            let props: [String: String] = [
                "Rank": "\(i + 1)",
                "Energy": String(format: "%.2f", result.energy),
                "VdW_Energy": String(format: "%.2f", result.stericEnergy),
                "Hydrophobic_Energy": String(format: "%.2f", result.hydrophobicEnergy),
                "HBond_Energy": String(format: "%.2f", result.hbondEnergy),
                "Torsion_Energy": String(format: "%.2f", result.torsionPenalty),
                "Cluster": "\(result.clusterID)",
                "Generation": "\(result.generation)"
            ]

            blocks.append(molBlock(
                name: "\(ligandName)_pose\(i + 1)",
                atoms: dockedAtoms, bonds: bonds,
                properties: props
            ))
        }

        return blocks.joined()
    }

    /// Export ligand database entries as SDF.
    static func writeLigandDatabase(_ entries: [LigandEntry]) -> String {
        var blocks: [String] = []

        for entry in entries {
            var props: [String: String] = [
                "SMILES": entry.smiles,
                "Prepared": entry.isPrepared ? "yes" : "no"
            ]
            if let d = entry.descriptors {
                props["MW"] = String(format: "%.2f", d.molecularWeight)
                props["LogP"] = String(format: "%.2f", d.logP)
                props["TPSA"] = String(format: "%.1f", d.tpsa)
                props["HBD"] = "\(d.hbd)"
                props["HBA"] = "\(d.hba)"
                props["RotBonds"] = "\(d.rotatableBonds)"
                props["Lipinski"] = d.lipinski ? "PASS" : "FAIL"
            }

            blocks.append(molBlock(
                name: entry.name,
                atoms: entry.atoms, bonds: entry.bonds,
                properties: props
            ))
        }

        return blocks.joined()
    }

    /// Write SDF string to a file URL.
    static func save(_ sdfContent: String, to url: URL) throws {
        try sdfContent.write(to: url, atomically: true, encoding: .utf8)
    }
}

// MARK: - Docking pose record

/// The exported description of one docked pose, shared by the pose SDF properties and the
/// results CSV so both name and format every field identically. Optional quantities that a run
/// did not compute (strain, GFN2 refinement, validity, chemical form), and non-finite values, are empty strings.
enum DockingPoseRecord {

    /// Ordered (name, value) fields of a pose. Energy-based scoring (Vina, Drusina, PIGNet2)
    /// reports the intermolecular terms, torsion penalty and `Intra_and_Other` (ligand internal
    /// energy plus penalties and corrections, so the terms sum to `Energy`); affinity scoring
    /// (DruseAF) reports pKd and confidence instead, since its stored "terms" are not energies.
    static func fields(rank: Int, result r: DockingResult, method: ScoringMethod) -> [(name: String, value: String)] {
        let f3 = { (value: Float) in String(format: "%.3f", value) }
        let optional = { (value: Float?) in value.flatMap { $0.isFinite ? f3($0) : nil } ?? "" }
        var fields: [(name: String, value: String)] = [
            ("Rank", "\(rank)"),
            ("Scoring", method.rawValue),
            ("Score", f3(r.displayScore(method: method))),
            ("Score_Unit", method.unitLabel)
        ]
        if method.isAffinityScore {
            fields += [
                ("pKd", f3(r.displayScore(method: method))),
                ("Confidence", f3(r.afConfidence))
            ]
        } else {
            fields += [
                ("Energy", f3(r.energy)),
                ("Steric", f3(r.stericEnergy)),
                ("Hydrophobic", f3(r.hydrophobicEnergy)),
                ("HBond", f3(r.hbondEnergy)),
                ("Torsion", f3(r.torsionPenalty)),
                ("Intra_and_Other", f3(r.intraAndOtherEnergy)),
                ("Drusina_Correction", f3(r.drusinaCorrection)),
                ("Constraint_Penalty", f3(r.constraintPenalty))
            ]
        }
        fields += [
            ("Cluster", "\(r.clusterID)"),
            ("Cluster_Rank", "\(r.clusterRank)"),
            ("Generation", "\(r.generation)"),
            ("Valid", r.validity.map { $0.passed ? "yes" : "no" } ?? ""),
            ("Validity_Failures", r.validity?.failures.joined(separator: "; ") ?? ""),
            ("Min_Protein_Distance", r.validity.flatMap { $0.minProteinDistance.isFinite ? f3($0.minProteinDistance) : nil } ?? ""),
            ("Strain_Energy", optional(r.strainEnergy)),
            ("Refinement_Energy", optional(r.refinementEnergy)),
            ("GFN2_Energy", optional(r.gfn2Energy)),
            ("GFN2_Dispersion", optional(r.gfn2DispersionEnergy)),
            ("GFN2_Solvation", optional(r.gfn2SolvationEnergy)),
            ("Form", r.formLabel ?? ""),
            ("Form_Population", optional(r.formPopulation))
        ]
        return fields
    }
}
