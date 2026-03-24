import Foundation
import simd

enum SDFParser {

    static func parse(_ content: String) -> [MoleculeData] {
        let blocks = content.components(separatedBy: "$$$$")
        var molecules: [MoleculeData] = []

        for (idx, block) in blocks.enumerated() {
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if let mol = parseMolBlock(trimmed, index: idx) {
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
        let lines = block.components(separatedBy: "\n")
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

        // Parse atom block
        var atoms: [Atom] = []
        for i in 0..<atomCount {
            let line = lines[4 + i]
            guard let atom = parseAtomLineV2000(line, id: i) else {
                Task { @MainActor in ActivityLog.shared.debug("[Parser] SDF V2000 '\(name)' skipped unparseable atom line \(i)", category: .molecule) }
                continue
            }
            atoms.append(atom)
        }

        // Parse bond block
        var bonds: [Bond] = []
        for i in 0..<bondCount {
            let line = lines[4 + atomCount + i]
            guard let bond = parseBondLineV2000(line, id: i) else {
                Task { @MainActor in ActivityLog.shared.debug("[Parser] SDF V2000 '\(name)' skipped unparseable bond line \(i)", category: .molecule) }
                continue
            }
            bonds.append(bond)
        }

        // Parse M  CHG properties until M  END
        var mEndLine = lines.count
        for i in (4 + atomCount + bondCount)..<lines.count {
            let line = lines[i]
            if line.hasPrefix("M  CHG") {
                parseCharges(line, atoms: &atoms)
            } else if line.hasPrefix("M  END") {
                mEndLine = i
                break
            }
        }

        // Parse SDF data blocks after M  END
        let properties = parseDataBlocks(lines: lines, startAfter: mEndLine)

        let molName = name.isEmpty ? "Molecule \(index + 1)" : name

        // Extract raw mol block (lines 0 through M  END) for direct RDKit parsing
        let molBlockText = lines[0...mEndLine].joined(separator: "\n")

        return MoleculeData(
            name: molName,
            title: "",
            atoms: atoms,
            bonds: bonds,
            properties: properties,
            molBlock: molBlockText
        )
    }

    // MARK: - V3000 Format

    private static func parseV3000(lines: [String], name: String, index: Int) -> MoleculeData? {
        // Find COUNTS line: M  V30 COUNTS natoms nbonds ...
        var atomCount = 0
        var atomBlockStart = -1
        var bondBlockStart = -1
        var atomBlockEnd = -1
        var bondBlockEnd = -1
        var mEndLine = lines.count

        for (i, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("M  V30 COUNTS") || trimmed.hasPrefix("M V30 COUNTS") {
                let parts = trimmed.split(separator: " ")
                // M V30 COUNTS natoms nbonds ...
                if parts.count >= 5 {
                    atomCount = Int(parts[3]) ?? 0
                    _ = Int(parts[4]) ?? 0  // bond count parsed but not used directly
                }
            } else if trimmed.contains("V30 BEGIN ATOM") {
                atomBlockStart = i + 1
            } else if trimmed.contains("V30 END ATOM") {
                atomBlockEnd = i
            } else if trimmed.contains("V30 BEGIN BOND") {
                bondBlockStart = i + 1
            } else if trimmed.contains("V30 END BOND") {
                bondBlockEnd = i
            } else if trimmed.hasPrefix("M  END") {
                mEndLine = i
                break
            }
        }

        guard atomCount > 0, atomBlockStart > 0, atomBlockEnd > atomBlockStart else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V3000 block '\(name)' missing or empty atom block", category: .molecule) }
            return nil
        }

        // Parse V3000 atom block
        // Format: M  V30 index type x y z aamap [properties...]
        var atoms: [Atom] = []
        var atomID = 0
        for i in atomBlockStart..<atomBlockEnd {
            let line = lines[i].trimmingCharacters(in: .whitespaces)
            // Handle line continuation (trailing -)
            guard !line.isEmpty else { continue }

            // Strip "M  V30 " or "M V30 " prefix
            let content: String
            if line.hasPrefix("M  V30 ") {
                content = String(line.dropFirst(7))
            } else if line.hasPrefix("M V30 ") {
                content = String(line.dropFirst(6))
            } else {
                continue
            }

            let parts = content.split(separator: " ")
            // index type x y z aamap [CHG=n] [RAD=n] ...
            guard parts.count >= 5,
                  let x = Float(parts[2]),
                  let y = Float(parts[3]),
                  let z = Float(parts[4])
            else { continue }

            let elemStr = String(parts[1])
            let element = Element.from(symbol: elemStr) ?? .C

            // Extract formal charge from CHG=n
            var formalCharge = 0
            for part in parts[5...] where part.hasPrefix("CHG=") {
                formalCharge = Int(part.dropFirst(4)) ?? 0
            }

            atoms.append(Atom(
                id: atomID,
                element: element,
                position: SIMD3<Float>(x, y, z),
                name: "\(element.symbol)\(atomID + 1)",
                residueName: "LIG",
                residueSeq: 1,
                chainID: "A",
                formalCharge: formalCharge,
                isHetAtom: true
            ))
            atomID += 1
        }

        // Parse V3000 bond block
        // Format: M  V30 index type atom1 atom2 [properties...]
        var bonds: [Bond] = []
        var bondID = 0
        if bondBlockStart > 0, bondBlockEnd > bondBlockStart {
            for i in bondBlockStart..<bondBlockEnd {
                let line = lines[i].trimmingCharacters(in: .whitespaces)
                guard !line.isEmpty else { continue }

                let content: String
                if line.hasPrefix("M  V30 ") {
                    content = String(line.dropFirst(7))
                } else if line.hasPrefix("M V30 ") {
                    content = String(line.dropFirst(6))
                } else {
                    continue
                }

                let parts = content.split(separator: " ")
                // index type atom1 atom2 [properties...]
                guard parts.count >= 4,
                      let typeInt = Int(parts[1]),
                      let a1 = Int(parts[2]),
                      let a2 = Int(parts[3])
                else { continue }

                let order: BondOrder
                switch typeInt {
                case 1: order = .single
                case 2: order = .double
                case 3: order = .triple
                case 4: order = .aromatic
                default: order = .single
                }

                // V3000 atom indices are 1-based
                bonds.append(Bond(id: bondID, atomIndex1: a1 - 1, atomIndex2: a2 - 1, order: order))
                bondID += 1
            }
        }

        guard !atoms.isEmpty else {
            Task { @MainActor in ActivityLog.shared.warn("[Parser] SDF V3000 block '\(name)' produced no atoms after parsing", category: .molecule) }
            return nil
        }

        // Parse SDF data blocks after M  END
        let properties = parseDataBlocks(lines: lines, startAfter: mEndLine)

        let molName = name.isEmpty ? "Molecule \(index + 1)" : name

        // Extract raw mol block (lines 0 through M  END) for direct RDKit parsing
        let molBlockText = lines[0...mEndLine].joined(separator: "\n")

        return MoleculeData(
            name: molName,
            title: "",
            atoms: atoms,
            bonds: bonds,
            properties: properties,
            molBlock: molBlockText
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
        let parts = line.split(separator: " ").map(String.init)
        guard parts.count >= 4, let count = Int(parts[2]) else { return }

        var idx = 3
        for _ in 0..<count {
            guard idx + 1 < parts.count,
                  let atomNum = Int(parts[idx]),
                  let charge = Int(parts[idx + 1])
            else { break }

            let atomIdx = atomNum - 1
            if atomIdx >= 0 && atomIdx < atoms.count {
                atoms[atomIdx] = Atom(
                    id: atoms[atomIdx].id,
                    element: atoms[atomIdx].element,
                    position: atoms[atomIdx].position,
                    name: atoms[atomIdx].name,
                    residueName: atoms[atomIdx].residueName,
                    residueSeq: atoms[atomIdx].residueSeq,
                    chainID: atoms[atomIdx].chainID,
                    charge: atoms[atomIdx].charge,
                    formalCharge: charge,
                    isHetAtom: atoms[atomIdx].isHetAtom,
                    occupancy: atoms[atomIdx].occupancy,
                    tempFactor: atoms[atomIdx].tempFactor
                )
            }
            idx += 2
        }
    }
}
