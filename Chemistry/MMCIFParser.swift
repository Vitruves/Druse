// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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

            let columnMap = Dictionary(columnNames.enumerated().map { ($0.element, $0.offset) },
                                       uniquingKeysWith: { first, _ in first })
            let modelColumn = columnMap["_atom_site.pdbx_PDB_model_num"]
            var firstModel: String?
            var atoms: [Atom] = []

            while cursor < lines.count {
                let rawLine = lines[cursor].trimmingCharacters(in: .whitespacesAndNewlines)
                if rawLine.isEmpty || rawLine == "#" || rawLine == "loop_" || rawLine.hasPrefix("data_") {
                    break
                }

                let fields = tokenizeMMCIFRow(lines[cursor])
                cursor += 1
                guard fields.count >= columnNames.count else { continue }

                // Multi-model entries (NMR ensembles): keep only the first model,
                // matching the Gemmi path.
                if let modelColumn {
                    let model = fields[modelColumn]
                    if firstModel == nil { firstModel = model }
                    guard model == firstModel else { continue }
                }

                if let atom = parseSimpleAtomSiteRow(fields: fields, columnMap: columnMap, atomID: atoms.count) {
                    atoms.append(atom)
                }
            }

            guard !atoms.isEmpty else { return nil }

            let resolved = ProteinPreparation.selectPreferredAltConformers(atoms: atoms, bonds: [])
            let selectedIndices = preferredAtomIndices(in: resolved.atoms)
            let substructure = ProteinPreparation.remapSubstructure(
                atoms: resolved.atoms,
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

        // Prefer author numbering/chain IDs (what gemmi and PDB-format files use);
        // label_seq_id is "." for every non-polymer atom, which would otherwise mix
        // label numbering for the protein with author numbering for ligands/waters.
        let atomName = value("_atom_site.auth_atom_id") ?? value("_atom_site.label_atom_id") ?? "X"
        let residueName = value("_atom_site.auth_comp_id") ?? value("_atom_site.label_comp_id") ?? "UNK"
        let chainID = value("_atom_site.auth_asym_id") ?? value("_atom_site.label_asym_id") ?? "A"
        let residueSeq = value("_atom_site.auth_seq_id").flatMap(Int.init)
            ?? value("_atom_site.label_seq_id").flatMap(Int.init)
            ?? 1
        let altLoc = value("_atom_site.label_alt_id") ?? ""
        let insertionCode = value("_atom_site.pdbx_PDB_ins_code") ?? ""
        let occupancy = value("_atom_site.occupancy").flatMap(Float.init) ?? 1.0
        let tempFactor = value("_atom_site.B_iso_or_equiv").flatMap(Float.init) ?? 0.0
        let formalCharge = value("_atom_site.pdbx_formal_charge").flatMap(Int.init) ?? 0
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
            formalCharge: formalCharge,
            isHetAtom: group == "HETATM",
            occupancy: occupancy,
            tempFactor: tempFactor,
            altLoc: altLoc,
            insertionCode: insertionCode
        )
    }

    /// Split a CIF data row into tokens. Per the CIF grammar a quote only opens a
    /// token at a token start and only closes it when followed by whitespace or the
    /// end of the line, so primes inside names (e.g. `"O5'"`, unquoted `C1'`) survive.
    private static func tokenizeMMCIFRow(_ line: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var quote: Character?
        let characters = Array(line)

        for (index, character) in characters.enumerated() {
            if let activeQuote = quote {
                let nextIsBoundary = index + 1 >= characters.count || characters[index + 1].isWhitespace
                if character == activeQuote && nextIsBoundary {
                    fields.append(current)
                    current = ""
                    quote = nil
                } else {
                    current.append(character)
                }
                continue
            }

            if current.isEmpty && (character == "'" || character == "\"") {
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
