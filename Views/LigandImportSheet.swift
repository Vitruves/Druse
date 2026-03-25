import SwiftUI
import AppKit

// MARK: - LigandDatabaseWindow Import Extension
// Import with column mapping: open file dialogs, build previews for CSV/SMI/SDF,
// perform mapped imports, and column-target suggestion logic.

extension LigandDatabaseWindow {

    // MARK: - Import with Column Mapping

    func openImportWithMapping(_ fileType: ImportFileType) {
        guard let url = FileImportHandler.showBatchOpenPanel() else { return }
        do {
            let preview = try buildImportPreview(url: url, fileType: fileType)
            importPreview = preview
            showImportMapping = true
        } catch {
            viewModel.log.error("Failed to read file: \(error.localizedDescription)", category: .molecule)
        }
    }

    func buildImportPreview(url: URL, fileType: ImportFileType) throws -> ImportPreview {
        let content = try String(contentsOf: url, encoding: .utf8)

        switch fileType {
        case .csv:
            return buildCSVPreview(content: content, url: url)
        case .smi:
            return buildSMIPreview(content: content, url: url)
        case .sdf:
            return buildSDFPreview(content: content, url: url)
        }
    }

    func buildCSVPreview(content: String, url: URL) -> ImportPreview {
        let rows = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !rows.isEmpty else { return ImportPreview(columns: [], rowCount: 0, fileURL: url, fileType: .csv) }

        let separator: Character = rows[0].contains("\t") ? "\t" : ","
        let headerCols = rows[0].split(separator: separator, omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        let dataRows = Array(rows.dropFirst())

        var columns: [ImportColumnMapping] = []
        for (i, header) in headerCols.enumerated() {
            let samples = dataRows.prefix(3).compactMap { row -> String? in
                let cols = row.split(separator: separator, omittingEmptySubsequences: false).map {
                    $0.trimmingCharacters(in: .whitespaces)
                }
                return i < cols.count ? cols[i] : nil
            }
            var mapping = ImportColumnMapping(sourceHeader: header, sampleValues: samples)
            // Auto-suggest based on header name
            mapping.target = suggestTarget(for: header)
            columns.append(mapping)
        }

        return ImportPreview(columns: columns, rowCount: dataRows.count, fileURL: url, fileType: .csv)
    }

    func buildSMIPreview(content: String, url: URL) -> ImportPreview {
        let lines = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        // Detect separator
        let firstLine = lines.first ?? ""
        let separator: Character = firstLine.contains("\t") ? "\t" : " "
        let parts = firstLine.split(separator: separator, maxSplits: 10).map(String.init)
        let colCount = max(parts.count, 2)

        var columns: [ImportColumnMapping] = []
        for i in 0..<colCount {
            let samples = lines.prefix(3).compactMap { line -> String? in
                let cols = line.split(separator: separator, maxSplits: 10).map(String.init)
                return i < cols.count ? cols[i] : nil
            }
            let header = "Column \(i + 1)"
            var mapping = ImportColumnMapping(sourceHeader: header, sampleValues: samples)
            // Auto-suggest: first col with SMILES-like content → SMILES, other → Name
            if i == 0 && samples.first.map({ LigandDatabase.looksLikeSMILES($0) }) == true {
                mapping.target = .smiles
            } else if i == 1 && columns.first?.target == .smiles {
                mapping.target = .name
            }
            columns.append(mapping)
        }

        return ImportPreview(columns: columns, rowCount: lines.count, fileURL: url, fileType: .smi)
    }

    func buildSDFPreview(content: String, url: URL) -> ImportPreview {
        let mols = SDFParser.parse(content)
        guard !mols.isEmpty else { return ImportPreview(columns: [], rowCount: 0, fileURL: url, fileType: .sdf) }

        // Collect all unique property keys across molecules
        var allKeys: [String] = []
        var keySamples: [String: [String]] = [:]
        for mol in mols {
            for (key, value) in mol.properties {
                if !allKeys.contains(key) { allKeys.append(key) }
                keySamples[key, default: []].append(value)
            }
        }

        var columns: [ImportColumnMapping] = []
        for key in allKeys {
            let samples = Array((keySamples[key] ?? []).prefix(3))
            var mapping = ImportColumnMapping(sourceHeader: key, sampleValues: samples)
            mapping.target = suggestTarget(for: key)
            columns.append(mapping)
        }

        return ImportPreview(columns: columns, rowCount: mols.count, fileURL: url, fileType: .sdf)
    }

    /// Suggest a target field based on a column header name.
    func suggestTarget(for header: String) -> ImportTargetField {
        let h = header.lowercased().trimmingCharacters(in: .whitespaces)
        if h == "smiles" || h == "molecule" || h == "structure" || h == "canonical_smiles" { return .smiles }
        if h == "name" || h == "id" || h == "title" || h == "compound_name" || h == "mol_name" { return .name }
        if h == "ki" || h == "ki_nm" || h == "ki (nm)" || h == "ki_value" { return .ki }
        if h == "pki" || h == "p_ki" || h == "pki_value" { return .pKi }
        if h == "ic50" || h == "ic50_nm" || h == "ic50 (nm)" || h == "ic50_value" { return .ic50 }
        return .none
    }

    // MARK: - Execute Mapped Import

    func performMappedImport(_ preview: ImportPreview) {
        do {
            switch preview.fileType {
            case .csv:
                try performCSVImport(preview)
            case .smi:
                try performSMIImport(preview)
            case .sdf:
                try performSDFImport(preview)
            }
        } catch {
            viewModel.log.error("Import failed: \(error.localizedDescription)", category: .molecule)
        }
    }

    func performCSVImport(_ preview: ImportPreview) throws {
        let content = try String(contentsOf: preview.fileURL, encoding: .utf8)
        let rows = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard rows.count > 1 else { return }

        let separator: Character = rows[0].contains("\t") ? "\t" : ","
        let dataRows = Array(rows.dropFirst())

        let smilesIdx = preview.columns.firstIndex { $0.target == .smiles }
        let nameIdx = preview.columns.firstIndex { $0.target == .name }
        let kiIdx = preview.columns.firstIndex { $0.target == .ki }
        let pKiIdx = preview.columns.firstIndex { $0.target == .pKi }
        let ic50Idx = preview.columns.firstIndex { $0.target == .ic50 }

        var count = 0
        let baseIndex = db.entries.count
        for (offset, row) in dataRows.enumerated() {
            let cols = row.split(separator: separator, omittingEmptySubsequences: false).map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            let smiles = smilesIdx.flatMap { $0 < cols.count ? cols[$0] : nil } ?? ""
            let name = nameIdx.flatMap { $0 < cols.count && !cols[$0].isEmpty ? cols[$0] : nil } ?? "Mol_\(baseIndex + offset + 1)"
            let ki = kiIdx.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
            let pKi = pKiIdx.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
            let ic50 = ic50Idx.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }

            guard !smiles.isEmpty || !name.isEmpty else { continue }
            let entry = LigandEntry(name: name, smiles: smiles, ki: ki, pKi: pKi, ic50: ic50)
            db.add(entry)
            count += 1
        }
        viewModel.log.success("Imported \(count) ligands from CSV", category: .molecule)
    }

    func performSMIImport(_ preview: ImportPreview) throws {
        let content = try String(contentsOf: preview.fileURL, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        let firstLine = lines.first ?? ""
        let separator: Character = firstLine.contains("\t") ? "\t" : " "

        let smilesIdx = preview.columns.firstIndex { $0.target == .smiles }
        let nameIdx = preview.columns.firstIndex { $0.target == .name }
        let kiIdx = preview.columns.firstIndex { $0.target == .ki }
        let pKiIdx = preview.columns.firstIndex { $0.target == .pKi }
        let ic50Idx = preview.columns.firstIndex { $0.target == .ic50 }

        var count = 0
        let baseIndex = db.entries.count
        for (offset, line) in lines.enumerated() {
            let cols = line.split(separator: separator, maxSplits: 10).map(String.init)
            let smiles = smilesIdx.flatMap { $0 < cols.count ? cols[$0] : nil } ?? ""
            let name = nameIdx.flatMap { $0 < cols.count && !cols[$0].isEmpty ? cols[$0] : nil } ?? "Mol_\(baseIndex + offset + 1)"
            let ki = kiIdx.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
            let pKi = pKiIdx.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
            let ic50 = ic50Idx.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }

            guard !smiles.isEmpty else { continue }
            let entry = LigandEntry(name: name, smiles: smiles, ki: ki, pKi: pKi, ic50: ic50)
            db.add(entry)
            count += 1
        }
        viewModel.log.success("Imported \(count) ligands from SMI", category: .molecule)
    }

    func performSDFImport(_ preview: ImportPreview) throws {
        let mols = try SDFParser.parse(url: preview.fileURL)

        // Build mapping: SDF property key → target field
        var propertyMapping: [String: ImportTargetField] = [:]
        for col in preview.columns where col.target != .none {
            propertyMapping[col.sourceHeader] = col.target
        }

        // Derive SMILES on background thread (RDKit can be slow + crash-prone on malformed data)
        let molData = mols  // capture for Task
        let mapping = propertyMapping
        viewModel.log.info("Importing \(mols.count) molecules from SDF...", category: .molecule)

        Task {
            var count = 0
            for mol in molData {
                var ki: Float?
                var pKi: Float?
                var ic50: Float?
                var name = mol.name

                var smilesFromProperty: String?
                for (key, value) in mol.properties {
                    guard let target = mapping[key] else { continue }
                    let trimmed = value.trimmingCharacters(in: .whitespaces)
                    switch target {
                    case .ki:     ki = Float(trimmed)
                    case .pKi:    pKi = Float(trimmed)
                    case .ic50:   ic50 = Float(trimmed)
                    case .name:   name = trimmed
                    case .smiles: smilesFromProperty = trimmed
                    default: break
                    }
                }

                // Use SMILES from property if available, otherwise derive from mol block/atoms
                let molBlockText = mol.molBlock
                let molAtoms = mol.atoms
                let molBonds = mol.bonds
                let smiles: String
                if let s = smilesFromProperty, !s.isEmpty {
                    smiles = s
                } else {
                    smiles = await Task.detached {
                        // Prefer mol block path (handles 2D/3D correctly)
                        if let mb = molBlockText, let s = RDKitBridge.smilesFromMolBlock(mb) {
                            return s
                        }
                        // Fallback to atom/bond reconstruction
                        return RDKitBridge.atomsBondsToSMILES(atoms: molAtoms, bonds: molBonds)
                    }.value ?? ""
                }

                let entry = LigandEntry(
                    name: name, smiles: smiles, atoms: mol.atoms, bonds: mol.bonds,
                    isPrepared: true, ki: ki, pKi: pKi, ic50: ic50
                )
                db.add(entry)
                count += 1
            }
            viewModel.log.success("Imported \(count) ligands from SDF", category: .molecule)
        }
    }
}
