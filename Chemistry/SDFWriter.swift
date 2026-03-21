import Foundation
import simd

/// Writer for V2000 SDF (Structure-Data File) format.
/// Used for exporting ligand databases and docking results.
enum SDFWriter {

    /// Write a single molecule as a V2000 MOL block.
    static func molBlock(
        name: String,
        atoms: [Atom],
        bonds: [Bond],
        properties: [String: String] = [:],
        includeTerminator: Bool = true
    ) -> String {
        var lines: [String] = []

        // Header block (3 lines)
        lines.append(name)                             // molecule name
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

        return lines.joined(separator: "\n")
    }

    /// Write a single Molecule object as V2000 SDF.
    @MainActor
    static func writeMolecule(_ mol: Molecule) -> String {
        molBlock(name: mol.name, atoms: mol.atoms, bonds: mol.bonds)
    }

    /// Write all docking result poses as a multi-molecule SDF with score properties.
    @MainActor
    static func writeDockingResults(_ results: [DockingResult], ligand: Molecule) -> String {
        writeDockingResults(results, ligandName: ligand.name,
                           atoms: ligand.atoms.filter { $0.element != .H },
                           bonds: ligand.bonds)
    }

    /// Write multiple molecules to an SDF string.
    static func write(molecules: [(name: String, atoms: [Atom], bonds: [Bond], properties: [String: String])]) -> String {
        molecules.map { mol in
            molBlock(name: mol.name, atoms: mol.atoms, bonds: mol.bonds, properties: mol.properties)
        }.joined(separator: "\n")
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
                "VdW_Energy": String(format: "%.2f", result.vdwEnergy),
                "Elec_Energy": String(format: "%.2f", result.elecEnergy),
                "HBond_Energy": String(format: "%.2f", result.hbondEnergy),
                "Desolv_Energy": String(format: "%.2f", result.desolvEnergy),
                "Cluster": "\(result.clusterID)",
                "Generation": "\(result.generation)"
            ]

            blocks.append(molBlock(
                name: "\(ligandName)_pose\(i + 1)",
                atoms: dockedAtoms, bonds: bonds,
                properties: props
            ))
        }

        return blocks.joined(separator: "\n")
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

        return blocks.joined(separator: "\n")
    }

    /// Write SDF string to a file URL.
    static func save(_ sdfContent: String, to url: URL) throws {
        try sdfContent.write(to: url, atomically: true, encoding: .utf8)
    }
}
