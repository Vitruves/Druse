import Foundation

enum MMCIFParser {

    enum MMCIFError: Error, LocalizedError {
        case emptyFile
        case parseFailed(String)
        case noAtoms

        var errorDescription: String? {
            switch self {
            case .emptyFile:
                return "mmCIF file is empty or could not be decoded."
            case .parseFailed(let reason):
                return "mmCIF parsing failed: \(reason)"
            case .noAtoms:
                return "No atoms found in mmCIF file."
            }
        }
    }

    static func parse(url: URL) throws -> (name: String, atoms: [Atom], bonds: [Bond], title: String) {
        let data = try Data(contentsOf: url)
        let content = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .ascii)
            ?? ""
        guard !content.isEmpty else {
            throw MMCIFError.emptyFile
        }
        return try parse(content: content, fileName: url.deletingPathExtension().lastPathComponent)
    }

    static func parse(
        content: String,
        fileName: String = "mmCIF"
    ) throws -> (name: String, atoms: [Atom], bonds: [Bond], title: String) {
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MMCIFError.emptyFile
        }

        do {
            let structure = try GemmiBridge.parseStructure(content: content, fileName: fileName)
            let resolved = ProteinPreparation.selectPreferredAltConformers(
                atoms: structure.atoms,
                bonds: structure.bonds
            )

            let selectedIndices = preferredAtomIndices(in: resolved.atoms)
            let substructure = ProteinPreparation.remapSubstructure(
                atoms: resolved.atoms,
                bonds: resolved.bonds,
                selectedIndices: selectedIndices
            )

            guard !substructure.atoms.isEmpty else {
                throw MMCIFError.noAtoms
            }

            let bonds = substructure.bonds.isEmpty
                ? BondPerception.perceiveBonds(in: substructure.atoms)
                : substructure.bonds

            let structureName = structure.name.isEmpty ? fileName : structure.name
            return (name: structureName, atoms: substructure.atoms, bonds: bonds, title: fileName)
        } catch let error as GemmiBridge.GemmiError {
            if let fallback = parseSimpleAtomSiteLoop(content: content, fileName: fileName) {
                return fallback
            }
            throw MMCIFError.parseFailed(error.localizedDescription)
        } catch let error as MMCIFError {
            throw error
        } catch {
            throw MMCIFError.parseFailed(error.localizedDescription)
        }
    }

    private static func preferredAtomIndices(in atoms: [Atom]) -> [Int] {
        let proteinIndices = atoms.indices.filter { atomIndex in
            let atom = atoms[atomIndex]
            return !atom.isHetAtom || isWaterResidue(atom.residueName)
        }
        return proteinIndices.isEmpty ? Array(atoms.indices) : proteinIndices
    }

    private static func isWaterResidue(_ residueName: String) -> Bool {
        ["HOH", "WAT", "DOD", "H2O"].contains(residueName.uppercased())
    }

    private static func parseSimpleAtomSiteLoop(
        content: String,
        fileName: String
    ) -> (name: String, atoms: [Atom], bonds: [Bond], title: String)? {
        let lines = content
            .split(whereSeparator: \.isNewline)
            .map(String.init)

        var structureName = fileName
        if let dataLine = lines.first(where: { $0.hasPrefix("data_") }) {
            let candidate = String(dataLine.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !candidate.isEmpty {
                structureName = candidate
            }
        }

        var lineIndex = 0
        while lineIndex < lines.count {
            let line = lines[lineIndex].trimmingCharacters(in: .whitespacesAndNewlines)
            guard line == "loop_" else {
                lineIndex += 1
                continue
            }

            var columnNames: [String] = []
            var cursor = lineIndex + 1
            while cursor < lines.count {
                let header = lines[cursor].trimmingCharacters(in: .whitespacesAndNewlines)
                guard header.hasPrefix("_") else { break }
                columnNames.append(header)
                cursor += 1
            }

            guard columnNames.contains("_atom_site.Cartn_x"),
                  columnNames.contains("_atom_site.Cartn_y"),
                  columnNames.contains("_atom_site.Cartn_z"),
                  columnNames.contains("_atom_site.label_atom_id") else {
                lineIndex = cursor
                continue
            }

            let columnMap = Dictionary(uniqueKeysWithValues: columnNames.enumerated().map { ($0.element, $0.offset) })
            var atoms: [Atom] = []

            while cursor < lines.count {
                let rawLine = lines[cursor].trimmingCharacters(in: .whitespacesAndNewlines)
                if rawLine.isEmpty || rawLine == "#" || rawLine == "loop_" || rawLine.hasPrefix("data_") {
                    break
                }

                let fields = tokenizeMMCIFRow(lines[cursor])
                if fields.count >= columnNames.count,
                   let atom = parseSimpleAtomSiteRow(fields: fields, columnMap: columnMap, atomID: atoms.count) {
                    atoms.append(atom)
                }
                cursor += 1
            }

            guard !atoms.isEmpty else { return nil }

            let selectedIndices = preferredAtomIndices(in: atoms)
            let substructure = ProteinPreparation.remapSubstructure(
                atoms: atoms,
                bonds: [],
                selectedIndices: selectedIndices
            )
            guard !substructure.atoms.isEmpty else { return nil }

            let bonds = BondPerception.perceiveBonds(in: substructure.atoms)
            return (name: structureName, atoms: substructure.atoms, bonds: bonds, title: fileName)
        }

        return nil
    }

    private static func parseSimpleAtomSiteRow(
        fields: [String],
        columnMap: [String: Int],
        atomID: Int
    ) -> Atom? {
        func value(_ column: String) -> String? {
            guard let index = columnMap[column], index < fields.count else { return nil }
            let token = fields[index]
            return token == "?" || token == "." ? nil : token
        }

        guard let x = value("_atom_site.Cartn_x").flatMap(Float.init),
              let y = value("_atom_site.Cartn_y").flatMap(Float.init),
              let z = value("_atom_site.Cartn_z").flatMap(Float.init) else {
            return nil
        }

        let atomName = value("_atom_site.label_atom_id") ?? value("_atom_site.auth_atom_id") ?? "X"
        let residueName = value("_atom_site.label_comp_id") ?? value("_atom_site.auth_comp_id") ?? "UNK"
        let chainID = value("_atom_site.label_asym_id") ?? value("_atom_site.auth_asym_id") ?? "A"
        let residueSeq = value("_atom_site.label_seq_id").flatMap(Int.init)
            ?? value("_atom_site.auth_seq_id").flatMap(Int.init)
            ?? 1
        let group = value("_atom_site.group_PDB")?.uppercased() ?? "ATOM"
        let elementSymbol = value("_atom_site.type_symbol")
            ?? Element.from(symbol: atomName.trimmingCharacters(in: .letters.inverted))?.symbol
            ?? "C"
        let element = Element.from(symbol: elementSymbol) ?? .C

        return Atom(
            id: atomID,
            element: element,
            position: SIMD3<Float>(x, y, z),
            name: atomName,
            residueName: residueName,
            residueSeq: residueSeq,
            chainID: chainID,
            isHetAtom: group == "HETATM"
        )
    }

    private static func tokenizeMMCIFRow(_ line: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var quote: Character?

        for character in line {
            if let activeQuote = quote, character == activeQuote {
                fields.append(current)
                current = ""
                quote = nil
                continue
            }

            if quote != nil {
                current.append(character)
                continue
            }

            if character == "'" || character == "\"" {
                quote = character
                continue
            }

            if character.isWhitespace {
                if !current.isEmpty {
                    fields.append(current)
                    current = ""
                }
                continue
            }

            current.append(character)
        }

        if !current.isEmpty {
            fields.append(current)
        }

        return fields
    }
}
