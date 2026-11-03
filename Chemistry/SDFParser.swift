// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

enum SDFParser {

    static func parse(_ content: String) -> [MoleculeData] {
        let blocks = normalizeLineEndings(content).components(separatedBy: "$$$$")
        var molecules: [MoleculeData] = []

        for (idx, rawBlock) in blocks.enumerated() {
            guard rawBlock.contains(where: { !$0.isWhitespace }) else { continue }
            // The header block is line-positional (name, program, comment, counts):
            // a blank name line is legal and must not be trimmed away. Only drop the
            // remainder of the preceding "$$$$" line and trailing whitespace.
            var block = Substring(rawBlock)
            if idx > 0, let newline = block.firstIndex(of: "\n") {
                block = block[block.index(after: newline)...]
            }
            while let last = block.last, last.isWhitespace {
                block = block.dropLast()
            }
            if let mol = parseMolBlock(String(block), index: idx) {
                molecules.append(mol)
            }
        }

        Task { @MainActor in ActivityLog.shared.info("[Parser] SDF parse complete: \(molecules.count) molecules from \(blocks.count - 1) blocks", category: .molecule) }
        return molecules
    }

    static func parse(url: URL) throws -> [MoleculeData] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return parse(content)
    }

    // MARK: - Single MOL Block

    static func parseMolBlock(_ block: String, index: Int = 0) -> MoleculeData? {
        let lines = normalizeLineEndings(block).components(separatedBy: "\n")
        guard lines.count >= 4 else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF mol block \(index) has fewer than 4 lines, skipped", category: .molecule) }
            return nil
        }

        // Line 0: molecule name
        let name = lines[0].trimmingCharacters(in: .whitespaces)

        // Line 3: counts line
        let countsLine = lines[3]
        guard countsLine.count >= 6 else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF mol block \(index) counts line too short (\(countsLine.count) chars)", category: .molecule) }
            return nil
        }

        // Detect V3000 format
        if countsLine.contains("V3000") {
            return parseV3000(lines: lines, name: name, index: index)
        }

        return parseV2000(lines: lines, name: name, index: index)
    }

    /// Convert CRLF / CR line endings to LF so fixed line positions and blank-line
    /// terminators of SDF data items are recognised.
    private static func normalizeLineEndings(_ text: String) -> String {
        guard text.contains("\r") else { return text }
        return text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }

    // MARK: - V2000 Format

    private static func parseV2000(lines: [String], name: String, index: Int) -> MoleculeData? {
        let countsLine = lines[3]

        let atomCountStr = String(countsLine.prefix(3)).trimmingCharacters(in: .whitespaces)
        let bondCountStr = String(countsLine.dropFirst(3).prefix(3)).trimmingCharacters(in: .whitespaces)

        guard let atomCount = Int(atomCountStr),
              let bondCount = Int(bondCountStr),
              atomCount > 0
        else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V2000 block '\(name)' has invalid atom/bond counts", category: .molecule) }
            return nil
        }

        guard lines.count >= 4 + atomCount + bondCount else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V2000 block '\(name)' truncated: expected \(4 + atomCount + bondCount) lines, got \(lines.count)", category: .molecule) }
            return nil
        }

        // Parse atom block. Bonds and M  CHG entries address atoms by their line
        // position, so a skipped atom line would silently shift every later index.
        var atoms: [Atom] = []
        for i in 0..<atomCount {
            let line = lines[4 + i]
            guard let atom = parseAtomLineV2000(line, id: i) else {
                Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V2000 '\(name)' has unparseable atom line \(i), molecule skipped", category: .molecule) }
                return nil
            }
            atoms.append(atom)
        }

        // Parse bond block
        var bonds: [Bond] = []
        for i in 0..<bondCount {
            let line = lines[4 + atomCount + i]
            guard let bond = parseBondLineV2000(line, id: bonds.count),
                  atoms.indices.contains(bond.atomIndex1),
                  atoms.indices.contains(bond.atomIndex2),
                  bond.atomIndex1 != bond.atomIndex2 else {
                Task { @MainActor in ActivityLog.shared.debug("[Parser] SDF V2000 '\(name)' skipped invalid bond line \(i)", category: .molecule) }
                continue
            }
            bonds.append(bond)
        }

        // Parse M  CHG properties until M  END.
        // CTfile spec: any M  CHG line supersedes all atom-block charges.
        var mEndLine = lines.count
        var sawChargeProperty = false
        for i in (4 + atomCount + bondCount)..<lines.count {
            let line = lines[i]
            if line.hasPrefix("M  CHG") {
                if !sawChargeProperty {
                    sawChargeProperty = true
                    for j in atoms.indices { atoms[j].formalCharge = 0 }
                }
                parseCharges(line, atoms: &atoms)
            } else if line.hasPrefix("M  END") {
                mEndLine = i
                break
            }
        }

        // Parse SDF data blocks after M  END
        let properties = parseDataBlocks(lines: lines, startAfter: mEndLine)

        let molName = name.isEmpty ? "Molecule \(index + 1)" : name

        return MoleculeData(
            name: molName,
            title: "",
            atoms: atoms,
            bonds: bonds,
            properties: properties,
            molBlock: rawMolBlock(lines: lines, mEndLine: mEndLine)
        )
    }

    /// Raw mol block (lines 0 through M  END) for direct RDKit parsing.
    /// Returns nil when the block has no M  END line (RDKit requires it).
    private static func rawMolBlock(lines: [String], mEndLine: Int) -> String? {
        guard mEndLine < lines.count else { return nil }
        return lines[0...mEndLine].joined(separator: "\n")
    }

    // MARK: - V3000 Format

    private static func parseV3000(lines: [String], name: String, index: Int) -> MoleculeData? {
        // Collect logical "M  V30" records: strip the prefix and join continuation
        // lines (a record ending in "-" continues on the next M  V30 line).
        var records: [String] = []
        var pending: String?
        var mEndLine = lines.count
        for (i, line) in lines.enumerated() {
            if line.hasPrefix("M  END") {
                mEndLine = i
                break
            }
            let content: Substring
            if line.hasPrefix("M  V30 ") {
                content = line.dropFirst(7)
            } else if line.hasPrefix("M V30 ") {
                content = line.dropFirst(6)
            } else {
                continue
            }
            var text: String = (pending ?? "") + String(content)
            while text.last?.isWhitespace == true { text.removeLast() }
            if text.hasSuffix("-") {
                text.removeLast()
                pending = text
                continue
            }
            pending = nil
            records.append(text.trimmingCharacters(in: .whitespaces))
        }

        enum Section { case none, atom, bond }
        var section = Section.none
        var atoms: [Atom] = []
        // V3000 bonds reference atoms by their index field, not by line position.
        var atomIndexByID: [Int: Int] = [:]
        var bondFields: [(type: Int, a1: Int, a2: Int)] = []

        for record in records {
            if record.hasPrefix("BEGIN ATOM") { section = .atom; continue }
            if record.hasPrefix("BEGIN BOND") { section = .bond; continue }
            if record.hasPrefix("END ") { section = .none; continue }

            let parts = record.split(whereSeparator: \.isWhitespace)
            switch section {
            case .atom:
                // index type x y z aamap [CHG=n] [RAD=n] ...
                guard parts.count >= 5,
                      let atomID = Int(parts[0]),
                      let x = Float(parts[2]),
                      let y = Float(parts[3]),
                      let z = Float(parts[4])
                else {
                    Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V3000 block '\(name)' has an unparseable atom record, molecule skipped", category: .molecule) }
                    return nil
                }

                let element = Element.from(symbol: String(parts[1])) ?? .C
                var formalCharge = 0
                for part in parts.dropFirst(5) where part.hasPrefix("CHG=") {
                    formalCharge = Int(part.dropFirst(4)) ?? 0
                }

                let newIndex = atoms.count
                atomIndexByID[atomID] = newIndex
                atoms.append(Atom(
                    id: newIndex,
                    element: element,
                    position: SIMD3<Float>(x, y, z),
                    name: "\(element.symbol)\(newIndex + 1)",
                    residueName: "LIG",
                    residueSeq: 1,
                    chainID: "A",
                    formalCharge: formalCharge,
                    isHetAtom: true
                ))

            case .bond:
                // index type atom1 atom2 [properties...]
                guard parts.count >= 4,
                      let typeInt = Int(parts[1]),
                      let a1 = Int(parts[2]),
                      let a2 = Int(parts[3])
                else { continue }
                bondFields.append((typeInt, a1, a2))

            case .none:
                continue
            }
        }

        guard !atoms.isEmpty else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V3000 block '\(name)' missing or empty atom block", category: .molecule) }
            return nil
        }

        var bonds: [Bond] = []
        for field in bondFields {
            guard let i1 = atomIndexByID[field.a1],
                  let i2 = atomIndexByID[field.a2],
                  i1 != i2 else { continue }
            let order: BondOrder
            switch field.type {
            case 2: order = .double
            case 3: order = .triple
            case 4: order = .aromatic
            default: order = .single
            }
            bonds.append(Bond(id: bonds.count, atomIndex1: i1, atomIndex2: i2, order: order))
        }

        // Parse SDF data blocks after M  END
        let properties = parseDataBlocks(lines: lines, startAfter: mEndLine)

        let molName = name.isEmpty ? "Molecule \(index + 1)" : name

        return MoleculeData(
            name: molName,
            title: "",
            atoms: atoms,
            bonds: bonds,
            properties: properties,
            molBlock: rawMolBlock(lines: lines, mEndLine: mEndLine)
        )
    }

    // MARK: - SDF Data Blocks

    /// Parse SDF data blocks: lines like "> <PropertyName>" followed by value lines, ending with blank line.
    private static func parseDataBlocks(lines: [String], startAfter mEndLine: Int) -> [String: String] {
        var properties: [String: String] = [:]
        var i = mEndLine + 1

        while i < lines.count {
            let line = lines[i]

            // Match "> <PropertyName>" pattern
            if line.hasPrefix(">") {
                var propName: String?
                if let openAngle = line.firstIndex(of: "<") {
                    let searchStart = line.index(after: openAngle)
                    if let closeAngle = line[searchStart...].firstIndex(of: ">") {
                        propName = String(line[searchStart..<closeAngle])
                    }
                }

                // Collect value lines until blank line
                i += 1
                var valueLines: [String] = []
                while i < lines.count {
                    let valLine = lines[i].trimmingCharacters(in: .whitespaces)
                    if valLine.isEmpty { break }
                    valueLines.append(valLine)
                    i += 1
                }

                if let key = propName, !key.isEmpty, !valueLines.isEmpty {
                    properties[key] = valueLines.joined(separator: "\n")
                }
            }
            i += 1
        }

        return properties
    }

    // MARK: - V2000 Atom Line

    private static func parseAtomLineV2000(_ line: String, id: Int) -> Atom? {
        // SDF V2000 atom line: xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee
        guard line.count >= 34 else { return nil }

        let chars = Array(line)

        func extract(_ start: Int, _ len: Int) -> String {
            let end = min(start + len, chars.count)
            guard start < chars.count else { return "" }
            return String(chars[start..<end]).trimmingCharacters(in: .whitespaces)
        }

        guard let x = Float(extract(0, 10)),
              let y = Float(extract(10, 10)),
              let z = Float(extract(20, 10))
        else { return nil }

        let elemStr = extract(31, 3)
        let element = Element.from(symbol: elemStr) ?? .C

        // Charge from atom block (V2000 encoding)
        var formalCharge = 0
        if chars.count >= 39 {
            let chargeCode = Int(extract(36, 3)) ?? 0
            switch chargeCode {
            case 1: formalCharge = 3
            case 2: formalCharge = 2
            case 3: formalCharge = 1
            case 4: formalCharge = 0 // doublet radical
            case 5: formalCharge = -1
            case 6: formalCharge = -2
            case 7: formalCharge = -3
            default: formalCharge = 0
            }
        }

        return Atom(
            id: id,
            element: element,
            position: SIMD3<Float>(x, y, z),
            name: "\(element.symbol)\(id + 1)",
            residueName: "LIG",
            residueSeq: 1,
            chainID: "A",
            formalCharge: formalCharge,
            isHetAtom: true
        )
    }

    // MARK: - V2000 Bond Line

    private static func parseBondLineV2000(_ line: String, id: Int) -> Bond? {
        // Bond line: 111222tttsssxxxrrrccc
        guard line.count >= 9 else { return nil }

        let chars = Array(line)

        func extract(_ start: Int, _ len: Int) -> String {
            let end = min(start + len, chars.count)
            guard start < chars.count else { return "" }
            return String(chars[start..<end]).trimmingCharacters(in: .whitespaces)
        }

        guard let a1 = Int(extract(0, 3)),
              let a2 = Int(extract(3, 3)),
              let typeInt = Int(extract(6, 3))
        else { return nil }

        let order: BondOrder
        switch typeInt {
        case 1: order = .single
        case 2: order = .double
        case 3: order = .triple
        case 4: order = .aromatic
        default: order = .single
        }

        // Convert from 1-based to 0-based
        return Bond(id: id, atomIndex1: a1 - 1, atomIndex2: a2 - 1, order: order)
    }

    // MARK: - M  CHG Property

    private static func parseCharges(_ line: String, atoms: inout [Atom]) {
        // M  CHG  n  aaa vvv  aaa vvv ...
        let parts = line.split(whereSeparator: \.isWhitespace)
        guard parts.count >= 4, let count = Int(parts[2]) else { return }

        var idx = 3
        for _ in 0..<count {
            guard idx + 1 < parts.count,
                  let atomNum = Int(parts[idx]),
                  let charge = Int(parts[idx + 1])
            else { break }

            let atomIdx = atomNum - 1
            if atoms.indices.contains(atomIdx) {
                atoms[atomIdx].formalCharge = charge
            }
            idx += 2
        }
    }
}
