// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

enum ProteinResidueReferenceTemplateStore {

    struct AtomTemplate: Sendable {
        let atomName: String
        let altAtomName: String?
        let element: Element
        let formalCharge: Int
        let idealPosition: SIMD3<Float>
        let isHydrogen: Bool
        let isBackbone: Bool
        let isNTerminal: Bool
        let isCTerminal: Bool
        let isLeaving: Bool
    }

    struct BondTemplate: Sendable {
        let atomName1: String
        let atomName2: String
        let order: BondOrder
    }

    struct Template: Sendable {
        let residueName: String
        let atomOrder: [String]
        let atomsByName: [String: AtomTemplate]
        let altNameToCanonicalName: [String: String]
        let bonds: [BondTemplate]
        let bondedAtomNamesByAtom: [String: [String]]

        var heavyAtomNames: [String] {
            atomOrder.filter { atomsByName[$0]?.isHydrogen == false }
        }

        func atom(named atomName: String) -> AtomTemplate? {
            let normalized = ProteinResidueTemplateStore.normalizeAtomName(atomName)
            if let atom = atomsByName[normalized] {
                return atom
            }
            guard let canonical = altNameToCanonicalName[normalized] else { return nil }
            return atomsByName[canonical]
        }

        func bondedAtomNames(to atomName: String) -> [String] {
            let normalized = canonicalAtomName(atomName)
            return bondedAtomNamesByAtom[normalized, default: []]
        }

        func hydrogenAtomNames(
            bondedTo parentAtomName: String,
            externalBondedAtoms: Set<String>
        ) -> [String] {
            let canonicalParent = canonicalAtomName(parentAtomName)
            let filtered = Set(filteredAtomNames(externalBondedAtoms: externalBondedAtoms, includeHydrogens: true))
            return bondedAtomNames(to: canonicalParent).filter { bondedAtomName in
                guard filtered.contains(bondedAtomName) else { return false }
                return atomsByName[bondedAtomName]?.isHydrogen == true
            }
        }

        func filteredAtomNames(
            externalBondedAtoms: Set<String>,
            includeHydrogens: Bool
        ) -> [String] {
            var skip = Set<String>()
            skip.formUnion(supplementalLinkedAtomNames(externalBondedAtoms: externalBondedAtoms))

            for atomName in atomOrder {
                guard let atom = atomsByName[atomName], atom.isLeaving else { continue }
                let shouldSkip = bondedAtomNames(to: atomName).contains { bondedAtomName in
                    externalBondedAtoms.contains(bondedAtomName)
                }
                if shouldSkip {
                    skip.insert(atomName)
                }
            }

            var changed = true
            while changed {
                changed = false
                for atomName in atomOrder {
                    guard let atom = atomsByName[atomName], atom.isLeaving, !skip.contains(atomName) else { continue }
                    if bondedAtomNames(to: atomName).contains(where: skip.contains) {
                        skip.insert(atomName)
                        changed = true
                    }
                }
            }

            return atomOrder.filter { atomName in
                guard !skip.contains(atomName) else { return false }
                if includeHydrogens {
                    return true
                }
                return atomsByName[atomName]?.isHydrogen == false
            }
        }

        private func supplementalLinkedAtomNames(externalBondedAtoms: Set<String>) -> Set<String> {
            switch residueName {
            case "ACE":
                return externalBondedAtoms.contains("C") ? ["H"] : []
            default:
                return []
            }
        }

        func canonicalAtomName(_ atomName: String) -> String {
            let normalized = ProteinResidueTemplateStore.normalizeAtomName(atomName)
            return altNameToCanonicalName[normalized] ?? normalized
        }
    }

    static func template(for residueName: String) -> Template? {
        templates[ProteinResidueTemplateStore.normalizeAtomName(residueName)]
    }

    private static let templates: [String: Template] = {
        var parsed: [String: Template] = [:]
        for (residueName, cif) in ProteinResidueReferenceCIFData.cifByResidueName {
            if let template = parseTemplate(cif: cif, residueName: residueName) {
                parsed[template.residueName] = template
            }
        }
        return parsed
    }()

    private static func parseTemplate(cif: String, residueName: String) -> Template? {
        let atomRows = loopRows(in: cif, headerPrefix: "_chem_comp_atom.")
        let bondRows = loopRows(in: cif, headerPrefix: "_chem_comp_bond.")
        guard !atomRows.isEmpty, !bondRows.isEmpty else { return nil }

        var atomOrder: [String] = []
        var atomsByName: [String: AtomTemplate] = [:]
        var altNameToCanonicalName: [String: String] = [:]

        for row in atomRows {
            guard let rawAtomName = normalizedField(row["_chem_comp_atom.atom_id"]),
                  let typeSymbol = normalizedField(row["_chem_comp_atom.type_symbol"]),
                  let element = Element.from(symbol: typeSymbol),
                  let position = idealPosition(from: row) else {
                continue
            }

            let atomName = ProteinResidueTemplateStore.normalizeAtomName(rawAtomName)
            let altAtomName = normalizedField(row["_chem_comp_atom.alt_atom_id"]).map(ProteinResidueTemplateStore.normalizeAtomName)
            let atom = AtomTemplate(
                atomName: atomName,
                altAtomName: altAtomName,
                element: element,
                formalCharge: Int(row["_chem_comp_atom.charge"] ?? "") ?? 0,
                idealPosition: position,
                isHydrogen: element == .H,
                isBackbone: booleanFlag(row["_chem_comp_atom.pdbx_backbone_atom_flag"]),
                isNTerminal: booleanFlag(row["_chem_comp_atom.pdbx_n_terminal_atom_flag"]),
                isCTerminal: booleanFlag(row["_chem_comp_atom.pdbx_c_terminal_atom_flag"]),
                isLeaving: booleanFlag(row["_chem_comp_atom.pdbx_leaving_atom_flag"])
            )

            atomOrder.append(atomName)
            atomsByName[atomName] = atom
            if let altAtomName, altAtomName != atomName {
                altNameToCanonicalName[altAtomName] = atomName
            }
        }

        for (alias, canonical) in supplementalAliases(for: residueName) {
            guard atomsByName[canonical] != nil else { continue }
            altNameToCanonicalName[ProteinResidueTemplateStore.normalizeAtomName(alias)] = canonical
        }

        var bonds: [BondTemplate] = []
        var bondedAtomNamesByAtom: [String: [String]] = [:]
        for row in bondRows {
            guard let atomName1 = normalizedField(row["_chem_comp_bond.atom_id_1"]).map(ProteinResidueTemplateStore.normalizeAtomName),
                  let atomName2 = normalizedField(row["_chem_comp_bond.atom_id_2"]).map(ProteinResidueTemplateStore.normalizeAtomName),
                  atomsByName[atomName1] != nil,
                  atomsByName[atomName2] != nil else {
                continue
            }

            let orderToken = (row["_chem_comp_bond.value_order"] ?? "SING").uppercased()
            let aromatic = (row["_chem_comp_bond.pdbx_aromatic_flag"] ?? "N").uppercased() == "Y"
            let order: BondOrder
            if aromatic || orderToken == "AROM" || orderToken == "DELO" {
                order = .aromatic
            } else {
                switch orderToken {
                case "DOUB":
                    order = .double
                case "TRIP":
                    order = .triple
                default:
                    order = .single
                }
            }

            bonds.append(.init(atomName1: atomName1, atomName2: atomName2, order: order))
            bondedAtomNamesByAtom[atomName1, default: []].append(atomName2)
            bondedAtomNamesByAtom[atomName2, default: []].append(atomName1)
        }

        guard !atomOrder.isEmpty else { return nil }
        return Template(
            residueName: ProteinResidueTemplateStore.normalizeAtomName(residueName),
            atomOrder: atomOrder,
            atomsByName: atomsByName,
            altNameToCanonicalName: altNameToCanonicalName,
            bonds: bonds,
            bondedAtomNamesByAtom: bondedAtomNamesByAtom
        )
    }

    private static func idealPosition(from row: [String: String]) -> SIMD3<Float>? {
        let xField = normalizedField(row["_chem_comp_atom.pdbx_model_Cartn_x_ideal"]) ?? normalizedField(row["_chem_comp_atom.model_Cartn_x"])
        let yField = normalizedField(row["_chem_comp_atom.pdbx_model_Cartn_y_ideal"]) ?? normalizedField(row["_chem_comp_atom.model_Cartn_y"])
        let zField = normalizedField(row["_chem_comp_atom.pdbx_model_Cartn_z_ideal"]) ?? normalizedField(row["_chem_comp_atom.model_Cartn_z"])
        guard let x = xField.flatMap(Float.init),
              let y = yField.flatMap(Float.init),
              let z = zField.flatMap(Float.init) else {
            return nil
        }
        return SIMD3<Float>(x, y, z)
    }

    private static func booleanFlag(_ rawValue: String?) -> Bool {
        (rawValue ?? "N").trimmingCharacters(in: .whitespacesAndNewlines).uppercased() == "Y"
    }

    private static func supplementalAliases(for residueName: String) -> [String: String] {
        switch ProteinResidueTemplateStore.normalizeAtomName(residueName) {
        case "NME":
            return [
                "CH3": "C",
                "H": "HN2",
            ]
        default:
            return [:]
        }
    }

    private static func loopRows(in cif: String, headerPrefix: String) -> [[String: String]] {
        let lines = cif.components(separatedBy: .newlines)
        var rows: [[String: String]] = []
        var index = 0

        while index < lines.count {
            let trimmed = lines[index].trimmingCharacters(in: .whitespaces)
            guard trimmed == "loop_" else {
                index += 1
                continue
            }

            index += 1
            var headers: [String] = []
            while index < lines.count {
                let header = lines[index].trimmingCharacters(in: .whitespaces)
                guard header.hasPrefix("_") else { break }
                headers.append(header)
                index += 1
            }

            let isTargetLoop = !headers.isEmpty && headers.allSatisfy { $0.hasPrefix(headerPrefix) }
            while index < lines.count {
                let line = lines[index]
                let row = line.trimmingCharacters(in: .whitespaces)
                if row.isEmpty || row == "#" {
                    index += 1
                    if row == "#" {
                        break
                    }
                    continue
                }
                if row == "loop_" || row.hasPrefix("_") || row.hasPrefix("data_") {
                    break
                }

                if isTargetLoop {
                    let values = tokenizeCIFRow(row)
                    if values.count >= headers.count {
                        var dict: [String: String] = [:]
                        for (header, value) in zip(headers, values) {
                            dict[header] = value
                        }
                        rows.append(dict)
                    }
                }
                index += 1
            }
        }

        return rows
    }

    private static func tokenizeCIFRow(_ row: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        var quote: Character?

        for character in row {
            if let activeQuote = quote {
                if character == activeQuote {
                    quote = nil
                } else {
                    current.append(character)
                }
                continue
            }

            if character == "\"" || character == "'" {
                quote = character
                continue
            }

            if character.isWhitespace {
                if !current.isEmpty {
                    tokens.append(current)
                    current.removeAll(keepingCapacity: true)
                }
                continue
            }

            current.append(character)
        }

        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }

    private static func normalizedField(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != "?", trimmed != "." else { return nil }
        return trimmed
    }
}
