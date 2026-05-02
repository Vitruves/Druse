// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

/// Writer for the Protein Data Bank (PDB) ATOM/HETATM record format.
/// Used to export a docked pose (protein + transformed ligand) as a single .pdb file.
enum PDBWriter {

    /// Build a PDB string containing the protein atoms followed by the ligand pose
    /// (HETATM records) using the supplied transformed positions.
    @MainActor
    static func combinedPose(
        protein: Molecule?,
        ligandTemplate: Molecule?,
        ligandPositions: [SIMD3<Float>],
        ligandResidueName: String = "LIG",
        ligandChainID: String = "X",
        ligandResidueSeq: Int = 1,
        title: String? = nil
    ) -> String {
        var lines: [String] = []

        if let title {
            for chunk in titleChunks(title) {
                lines.append(chunk)
            }
        }

        var serial = 1
        var lastChain: String? = nil

        if let protein {
            for atom in protein.atoms {
                if let prev = lastChain, prev != atom.chainID {
                    lines.append(terRecord(serial: serial,
                                           residueName: atom.residueName,
                                           chainID: prev,
                                           residueSeq: atom.residueSeq))
                    serial += 1
                }
                lines.append(atomRecord(record: atom.isHetAtom ? "HETATM" : "ATOM",
                                        serial: serial,
                                        atom: atom,
                                        position: atom.position))
                serial += 1
                lastChain = atom.chainID
            }
            if let last = lastChain, let lastAtom = protein.atoms.last {
                lines.append(terRecord(serial: serial,
                                       residueName: lastAtom.residueName,
                                       chainID: last,
                                       residueSeq: lastAtom.residueSeq))
                serial += 1
            }
        }

        // Ligand HETATM records — use the heavy-atom template combined with the
        // transformed positions produced by the docking engine.
        if let ligandTemplate {
            let heavyAtoms = ligandTemplate.atoms.filter { $0.element != .H }
            let count = min(heavyAtoms.count, ligandPositions.count)
            for i in 0..<count {
                let template = heavyAtoms[i]
                let atomName = ligandAtomName(template: template, index: i)
                let synth = Atom(
                    id: serial,
                    element: template.element,
                    position: ligandPositions[i],
                    name: atomName,
                    residueName: ligandResidueName,
                    residueSeq: ligandResidueSeq,
                    chainID: ligandChainID,
                    charge: template.charge,
                    formalCharge: template.formalCharge,
                    isHetAtom: true,
                    occupancy: 1.0,
                    tempFactor: 0.0
                )
                lines.append(atomRecord(record: "HETATM",
                                        serial: serial,
                                        atom: synth,
                                        position: ligandPositions[i]))
                serial += 1
            }
        }

        lines.append("END")
        return lines.joined(separator: "\n") + "\n"
    }

    static func save(_ content: String, to url: URL) throws {
        try content.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: - Record formatting

    private static func atomRecord(record: String, serial: Int, atom: Atom, position: SIMD3<Float>) -> String {
        // PDB fixed-column ATOM/HETATM record (80 cols).
        let recordField = record.padding(toLength: 6, withPad: " ", startingAt: 0)
        let serialStr = String(format: "%5d", min(serial, 99999))
        let nameField = formatAtomName(atom.name, element: atom.element.symbol)
        let altLoc = atom.altLoc.isEmpty ? " " : String(atom.altLoc.prefix(1))
        let resName = atom.residueName.padding(toLength: 3, withPad: " ", startingAt: 0)
        let chain = atom.chainID.isEmpty ? " " : String(atom.chainID.prefix(1))
        let resSeq = String(format: "%4d", atom.residueSeq)
        let iCode = atom.insertionCode.isEmpty ? " " : String(atom.insertionCode.prefix(1))
        let x = String(format: "%8.3f", clampCoord(position.x))
        let y = String(format: "%8.3f", clampCoord(position.y))
        let z = String(format: "%8.3f", clampCoord(position.z))
        let occ = String(format: "%6.2f", atom.occupancy)
        let temp = String(format: "%6.2f", atom.tempFactor)
        let element = padLeft(atom.element.symbol.uppercased(), width: 2)
        let chargeField = chargeString(atom.formalCharge)

        // Columns: 1-6 record, 7-11 serial, 12 space, 13-16 name, 17 altLoc,
        // 18-20 resName, 21 space, 22 chain, 23-26 resSeq, 27 iCode,
        // 28-30 spaces, 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 temp,
        // 67-76 spaces, 77-78 element, 79-80 charge.
        return recordField
            + serialStr
            + " "
            + nameField
            + altLoc
            + resName
            + " "
            + chain
            + resSeq
            + iCode
            + "   "
            + x + y + z
            + occ + temp
            + "          "
            + element
            + chargeField
    }

    private static func terRecord(serial: Int, residueName: String, chainID: String, residueSeq: Int) -> String {
        let serialStr = String(format: "%5d", min(serial, 99999))
        let resName = residueName.padding(toLength: 3, withPad: " ", startingAt: 0)
        let chain = chainID.isEmpty ? " " : String(chainID.prefix(1))
        let resSeq = String(format: "%4d", residueSeq)
        return "TER   " + serialStr + "      " + resName + " " + chain + resSeq
    }

    /// Format the atom-name field (columns 13-16) per PDB convention:
    /// 2-letter element symbols start in column 13; 1-letter elements start in 14.
    private static func formatAtomName(_ name: String, element: String) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            // Fall back to the element symbol so the record is still parseable.
            let sym = element.uppercased()
            if sym.count >= 2 { return (sym + "  ").prefix(4).description }
            return " " + sym.padding(toLength: 3, withPad: " ", startingAt: 0)
        }
        if trimmed.count >= 4 {
            return String(trimmed.prefix(4))
        }
        if element.count >= 2 {
            return trimmed.padding(toLength: 4, withPad: " ", startingAt: 0)
        }
        // 1-letter element: leading space, name in cols 14-16.
        return " " + trimmed.padding(toLength: 3, withPad: " ", startingAt: 0)
    }

    private static func ligandAtomName(template: Atom, index: Int) -> String {
        let trimmed = template.name.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { return trimmed }
        let sym = template.element.symbol.uppercased()
        let suffix = String(index + 1)
        let combined = sym + suffix
        return String(combined.prefix(4))
    }

    private static func chargeString(_ charge: Int) -> String {
        guard charge != 0 else { return "  " }
        let mag = min(abs(charge), 9)
        return charge > 0 ? "\(mag)+" : "\(mag)-"
    }

    private static func padLeft(_ s: String, width: Int) -> String {
        if s.count >= width { return String(s.suffix(width)) }
        return String(repeating: " ", count: width - s.count) + s
    }

    private static func clampCoord(_ value: Float) -> Float {
        if value.isNaN || value.isInfinite { return 0 }
        return max(-9999.999, min(9999.999, value))
    }

    private static func titleChunks(_ title: String) -> [String] {
        let cleaned = title.replacingOccurrences(of: "\n", with: " ")
        let maxLen = 70
        var chunks: [String] = []
        var remaining = cleaned[...]
        var index = 1
        while !remaining.isEmpty {
            let take = remaining.prefix(maxLen)
            remaining = remaining.dropFirst(take.count)
            let cont = index == 1 ? "  " : String(format: "%2d", index)
            chunks.append("TITLE " + cont + " " + take)
            index += 1
        }
        return chunks
    }
}
