import Foundation

// MARK: - Search Result

struct PDBSearchResult: Identifiable, Sendable {
    let id: String          // PDB ID
    var title: String
    var resolution: Float?
    var experimentMethod: String?
    var releaseDate: String?
}

// MARK: - Service Errors

enum PDBServiceError: LocalizedError {
    case invalidPDBID(String)
    case networkError(Error)
    case httpError(Int)
    case notFound(String)
    case parseError(String)

    var errorDescription: String? {
        switch self {
        case .invalidPDBID(let id): "Invalid PDB ID: \(id)"
        case .networkError(let err): "Network error: \(err.localizedDescription)"
        case .httpError(let code): "HTTP error \(code)"
        case .notFound(let id): "PDB entry \(id) not found"
        case .parseError(let msg): "Parse error: \(msg)"
        }
    }
}

// MARK: - PDB Service

actor PDBService {
    static let shared = PDBService()
    private let session = URLSession.shared

    // MARK: - Fetch PDB File

    func fetchPDBFile(id: String) async throws -> String {
        let cleanID = id.trimmingCharacters(in: .whitespaces).uppercased()
        guard cleanID.count == 4, cleanID.allSatisfy({ $0.isLetter || $0.isNumber }) else {
            throw PDBServiceError.invalidPDBID(id)
        }

        let url = URL(string: "https://files.rcsb.org/download/\(cleanID).pdb")!
        let (data, response) = try await session.data(from: url)

        guard let http = response as? HTTPURLResponse else {
            throw PDBServiceError.networkError(URLError(.badServerResponse))
        }

        if http.statusCode == 404 {
            throw PDBServiceError.notFound(cleanID)
        }

        guard (200..<300).contains(http.statusCode) else {
            throw PDBServiceError.httpError(http.statusCode)
        }

        guard let content = String(data: data, encoding: .utf8), !content.isEmpty else {
            throw PDBServiceError.parseError("Empty response for \(cleanID)")
        }

        // Prefetch CCD templates for any non-water HET groups so downstream parsing
        // can restore chemically correct bond orders and formal charges.
        await ChemicalComponentStore.prefetchTemplates(referencedIn: content)

        return content
    }

    // MARK: - Text Search

    func search(query: String, maxResults: Int = 20) async throws -> [PDBSearchResult] {
        let url = URL(string: "https://search.rcsb.org/rcsbsearch/v2/query")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "query": [
                "type": "terminal",
                "service": "full_text",
                "parameters": ["value": query]
            ],
            "return_type": "entry",
            "request_options": [
                "paginate": ["start": 0, "rows": maxResults],
                "results_content_type": ["experimental"],
                "sort": [["sort_by": "score", "direction": "desc"]]
            ]
        ]

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw PDBServiceError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let resultSet = json["result_set"] as? [[String: Any]]
        else {
            return []
        }

        // Extract PDB IDs from results
        let pdbIDs = resultSet.compactMap { $0["identifier"] as? String }

        // Fetch metadata for each result
        return await withTaskGroup(of: PDBSearchResult?.self, returning: [PDBSearchResult].self) { group in
            for pdbID in pdbIDs {
                group.addTask { [self] in
                    try? await self.fetchSearchMetadata(id: pdbID)
                }
            }

            var results: [PDBSearchResult] = []
            for await result in group {
                if let r = result { results.append(r) }
            }
            // Preserve original search order
            return results.sorted { a, b in
                (pdbIDs.firstIndex(of: a.id) ?? 0) < (pdbIDs.firstIndex(of: b.id) ?? 0)
            }
        }
    }

    // MARK: - Entry Metadata

    private func fetchSearchMetadata(id: String) async throws -> PDBSearchResult {
        let url = URL(string: "https://data.rcsb.org/rest/v1/core/entry/\(id)")!
        let (data, response) = try await session.data(from: url)

        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            return PDBSearchResult(id: id, title: id)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return PDBSearchResult(id: id, title: id)
        }

        let struct_ = json["struct"] as? [String: Any]
        let title = (struct_?["title"] as? String) ?? id

        let cell = json["cell"] as? [String: Any]
        _ = cell // reserved for future use

        let reflns = json["rcsb_entry_info"] as? [String: Any]
        let resolution = reflns?["resolution_combined"] as? [Double]
        let method = reflns?["experimental_method"] as? String

        let releaseDate = (json["rcsb_accession_info"] as? [String: Any])?["initial_release_date"] as? String

        return PDBSearchResult(
            id: id,
            title: title.prefix(200).description,
            resolution: resolution?.first.map { Float($0) },
            experimentMethod: method,
            releaseDate: releaseDate?.prefix(10).description
        )
    }
}

// MARK: - Chemical Component Templates

struct ChemicalComponentTemplate: Sendable {
    struct AtomTemplate: Sendable {
        var atomID: String
        var altAtomID: String?
        var element: Element
        var formalCharge: Int
    }

    struct BondTemplate: Sendable {
        var atomID1: String
        var atomID2: String
        var order: BondOrder
    }

    var componentID: String
    var canonicalSmiles: String?
    var atomsByName: [String: AtomTemplate]
    var bonds: [BondTemplate]
}

enum ChemicalComponentStore {
    private static let session = URLSession.shared
    private static let lock = NSLock()
    nonisolated(unsafe) private static var templates: [String: ChemicalComponentTemplate] = [:]
    nonisolated(unsafe) private static var inFlight: Set<String> = []

    static func prefetchTemplates(referencedIn pdbContent: String) async {
        let componentIDs = extractComponentIDs(from: pdbContent)
        guard !componentIDs.isEmpty else { return }

        await withTaskGroup(of: Void.self) { group in
            for componentID in componentIDs where beginFetchIfNeeded(componentID) {
                group.addTask {
                    await fetchAndCacheTemplate(componentID: componentID)
                }
            }
        }
    }

    static func template(for componentID: String) -> ChemicalComponentTemplate? {
        let key = componentID.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard !key.isEmpty else { return nil }

        lock.lock()
        if let cached = templates[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let url = cacheFileURL(for: key)
        guard let cif = try? String(contentsOf: url, encoding: .utf8),
              let parsed = parseTemplate(cif: cif, componentID: key) else {
            return nil
        }

        lock.lock()
        templates[key] = parsed
        lock.unlock()
        return parsed
    }

    private static func beginFetchIfNeeded(_ componentID: String) -> Bool {
        let key = componentID.uppercased()
        lock.lock()
        defer { lock.unlock() }
        if templates[key] != nil || inFlight.contains(key) {
            return false
        }
        inFlight.insert(key)
        return true
    }

    private static func finishFetch(_ componentID: String, template: ChemicalComponentTemplate?, rawCIF: String?) {
        let key = componentID.uppercased()
        lock.lock()
        if let template {
            templates[key] = template
        }
        inFlight.remove(key)
        lock.unlock()

        guard let rawCIF else { return }
        let url = cacheFileURL(for: key)
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
        try? rawCIF.write(to: url, atomically: true, encoding: .utf8)
    }

    private static func fetchAndCacheTemplate(componentID: String) async {
        defer { finishFetch(componentID, template: nil, rawCIF: nil) }

        guard let url = URL(string: "https://files.rcsb.org/ligands/download/\(componentID).cif") else { return }
        guard let (data, response) = try? await session.data(from: url),
              let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode),
              let cif = String(data: data, encoding: .utf8),
              let parsed = parseTemplate(cif: cif, componentID: componentID) else {
            return
        }

        finishFetch(componentID, template: parsed, rawCIF: cif)
    }

    private static func cacheFileURL(for componentID: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("DruseChemicalComponents", isDirectory: true)
            .appendingPathComponent("\(componentID).cif")
    }

    private static func extractComponentIDs(from pdbContent: String) -> [String] {
        var ids: Set<String> = []
        for rawLine in pdbContent.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            guard line.hasPrefix("HETATM"), line.count >= 20 else { continue }
            let chars = Array(line)
            let start = 17
            let end = min(20, chars.count)
            let compID = String(chars[start..<end]).trimmingCharacters(in: .whitespaces).uppercased()
            guard !compID.isEmpty,
                  compID != "HOH",
                  compID != "WAT",
                  compID != "DOD" else {
                continue
            }
            ids.insert(compID)
        }
        return ids.sorted()
    }

    private static func parseTemplate(cif: String, componentID: String) -> ChemicalComponentTemplate? {
        let atomRows = loopRows(in: cif, headerPrefix: "_chem_comp_atom.")
        let bondRows = loopRows(in: cif, headerPrefix: "_chem_comp_bond.")
        let descriptorRows = loopRows(in: cif, headerPrefix: "_pdbx_chem_comp_descriptor.")
        guard !atomRows.isEmpty, !bondRows.isEmpty else { return nil }

        var atomsByName: [String: ChemicalComponentTemplate.AtomTemplate] = [:]
        for row in atomRows {
            guard let atomID = row["_chem_comp_atom.atom_id"],
                  let typeSymbol = row["_chem_comp_atom.type_symbol"],
                  let element = Element.from(symbol: typeSymbol) else {
                continue
            }

            let altAtomID = normalizedField(row["_chem_comp_atom.alt_atom_id"])
            let formalCharge = Int(row["_chem_comp_atom.charge"] ?? "") ?? 0
            let template = ChemicalComponentTemplate.AtomTemplate(
                atomID: atomID,
                altAtomID: altAtomID,
                element: element,
                formalCharge: formalCharge
            )

            atomsByName[atomID] = template
            if let altAtomID {
                atomsByName[altAtomID] = template
            }
        }

        var bonds: [ChemicalComponentTemplate.BondTemplate] = []
        for row in bondRows {
            guard let atomID1 = row["_chem_comp_bond.atom_id_1"],
                  let atomID2 = row["_chem_comp_bond.atom_id_2"] else {
                continue
            }

            let orderToken = (row["_chem_comp_bond.value_order"] ?? "SING").uppercased()
            let aromatic = (row["_chem_comp_bond.pdbx_aromatic_flag"] ?? "N").uppercased() == "Y"
            let order: BondOrder
            if aromatic || orderToken == "AROM" || orderToken == "DELO" {
                order = .aromatic
            } else {
                switch orderToken {
                case "DOUB": order = .double
                case "TRIP": order = .triple
                default:     order = .single
                }
            }

            bonds.append(.init(atomID1: atomID1, atomID2: atomID2, order: order))
        }

        let canonicalSmiles = descriptorRows.first {
            ($0["_pdbx_chem_comp_descriptor.type"] ?? "").uppercased() == "SMILES_CANONICAL"
        }?["_pdbx_chem_comp_descriptor.descriptor"]

        guard !atomsByName.isEmpty, !bonds.isEmpty else { return nil }
        return ChemicalComponentTemplate(
            componentID: componentID.uppercased(),
            canonicalSmiles: canonicalSmiles,
            atomsByName: atomsByName,
            bonds: bonds
        )
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
                    if row == "#" { break }
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

        for ch in row {
            if let activeQuote = quote {
                if ch == activeQuote {
                    quote = nil
                } else {
                    current.append(ch)
                }
                continue
            }

            if ch == "\"" || ch == "'" {
                quote = ch
                continue
            }

            if ch.isWhitespace {
                if !current.isEmpty {
                    tokens.append(current)
                    current.removeAll(keepingCapacity: true)
                }
                continue
            }

            current.append(ch)
        }

        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }

    private static func normalizedField(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty || trimmed == "?" || trimmed == "." ? nil : trimmed
    }
}
