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

        return molecules
    }

    static func parse(url: URL) throws -> [MoleculeData] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return parse(content)
    }

    // MARK: - Single MOL Block

    static func parseMolBlock(_ block: String, index: Int = 0) -> MoleculeData? {
        let lines = block.components(separatedBy: "\n")
        guard lines.count >= 4 else { return nil }

        // Line 0: molecule name
        let name = lines[0].trimmingCharacters(in: .whitespaces)

        // Line 1: program/timestamp (skip)
        // Line 2: comment (skip)

        // Line 3: counts line
        let countsLine = lines[3]
        guard countsLine.count >= 6 else { return nil }

        let atomCountStr = String(countsLine.prefix(3)).trimmingCharacters(in: .whitespaces)
        let bondCountStr = String(countsLine.dropFirst(3).prefix(3)).trimmingCharacters(in: .whitespaces)

        guard let atomCount = Int(atomCountStr),
              let bondCount = Int(bondCountStr),
              atomCount > 0
        else { return nil }

        // Check for V3000
        if countsLine.contains("V3000") {
            return nil // V3000 not supported
        }

        guard lines.count >= 4 + atomCount + bondCount else { return nil }

        // Parse atom block
        var atoms: [Atom] = []
        for i in 0..<atomCount {
            let line = lines[4 + i]
            guard let atom = parseAtomLine(line, id: i) else { continue }
            atoms.append(atom)
        }

        // Parse bond block
        var bonds: [Bond] = []
        for i in 0..<bondCount {
            let line = lines[4 + atomCount + i]
            guard let bond = parseBondLine(line, id: i) else { continue }
            bonds.append(bond)
        }

        // Parse properties (M  CHG, etc.)
        for i in (4 + atomCount + bondCount)..<lines.count {
            let line = lines[i]
            if line.hasPrefix("M  CHG") {
                parseCharges(line, atoms: &atoms)
            } else if line.hasPrefix("M  END") {
                break
            }
        }

        let molName = name.isEmpty ? "Molecule \(index + 1)" : name

        return MoleculeData(
            name: molName,
            title: "",
            atoms: atoms,
            bonds: bonds
        )
    }

    // MARK: - Atom Line

    private static func parseAtomLine(_ line: String, id: Int) -> Atom? {
        // SDF V2000 atom line: xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee
        // x: cols 0-9 (10.4), y: cols 10-19, z: cols 20-29
        // element: cols 31-33
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

    // MARK: - Bond Line

    private static func parseBondLine(_ line: String, id: Int) -> Bond? {
        // Bond line: 111222tttsssxxxrrrccc
        // atom1: cols 0-2, atom2: cols 3-5, type: cols 6-8
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
