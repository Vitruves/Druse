// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import AppKit
import UniformTypeIdentifiers
import simd

/// Legacy conformer type — kept for backward compat with WorkspaceState mapping.
typealias LigandConformer = Conformer3D

/// An entry in the ligand database.
///
/// Flat model: each entry is one dockable entity (one chemical form with its conformers).
/// - Raw import: `isPrepared == false`, no 3D coords, no conformers.
/// - After Prepare (dominant): `isPrepared == true`, dominant form at target pH, conformers generated.
/// - After Enumerate: parent entry is replaced by N rows (one per tautomer/protomer),
///   each with `parentName` set and `populationWeight` assigned.
///
/// One row = one docking job. No hierarchy, no ensemble machinery.
struct LigandEntry: Identifiable, Sendable {
    let id: UUID
    var name: String
    var originalSMILES: String              // SMILES as originally imported (or dominant form after prep)
    var atoms: [Atom]                       // best conformer (for docking/rendering)
    var bonds: [Bond]                       // best conformer
    var descriptors: LigandDescriptors?
    var isPrepared: Bool
    var preparationDate: Date?
    var conformerCount: Int                 // number of 3D conformers stored
    var conformers: [Conformer3D] = []      // all conformers (atoms/bonds are best conformer)

    // Experimental binding affinity data (optional)
    var ki: Float?       // Ki in nM
    var pKi: Float?      // -log10(Ki in M)
    var ic50: Float?     // IC50 in nM

    /// Source chain ID from the protein (for co-crystallized ligands).
    var sourceChainID: String?

    /// Arbitrary user-imported properties — extra columns from CSV/SMI/SDF imports
    /// that don't map to a built-in field. Keys are user-chosen column names
    /// (defaulted to the source CSV header). Values are stored as raw strings;
    /// the table renderer parses them on display when shown as numeric columns.
    var userProperties: [String: String] = [:]

    // --- Enumeration metadata (set when Enumerate expands a molecule into forms) ---

    /// Parent molecule name. Nil for standalone/raw/dominant entries.
    /// Set to the original molecule name when this entry was produced by enumeration.
    /// Used to group sibling entries in the UI (visual grouping, not functional).
    var parentName: String?

    /// Boltzmann population weight at target pH. Nil = not enumerated (show "—" in table).
    /// A value like 0.803 means this form is 80.3% populated at equilibrium.
    var populationWeight: Double?

    /// What kind of chemical form this is. Nil = raw/dominant (not from enumeration).
    var formKind: ChemicalFormKind?

    /// Relative energy vs best form in the enumerated set (kcal/mol). Nil = not enumerated.
    var relativeEnergy: Double?

    /// Preparation error message, if preparation failed.
    var preparationError: String?

    // MARK: - Computed

    /// Active SMILES (always originalSMILES in flat model).
    var smiles: String {
        get { originalSMILES }
        set { originalSMILES = newValue }
    }

    /// Whether this entry came from enumeration (has siblings in the table).
    var isEnumerated: Bool { parentName != nil }

    /// Computed pKi from whatever affinity data is available (pKi > Ki > IC50).
    var effectivePKi: Float? {
        if let pk = pKi { return pk }
        if let k = ki, k > 0 { return -log10(k * 1e-9) }
        if let ic = ic50, ic > 0 { return -log10(ic * 1e-9) }
        return nil
    }

    // MARK: - Backward compat shims (accessed by old code, will be removed)

    /// Old code reads `forms` — return empty. Will be removed after full migration.
    var forms: [ChemicalForm] { [] }
    var bestFormIndex: Int { 0 }
    var totalFormCount: Int { isEnumerated ? 1 : 0 }
    var variantCount: Int { 0 }
    var totalConformerCount: Int { conformerCount }
    var parentID: UUID? { nil }

    // MARK: - Init

    init(name: String, smiles: String, atoms: [Atom] = [], bonds: [Bond] = [],
         descriptors: LigandDescriptors? = nil, isPrepared: Bool = false,
         conformerCount: Int = 0, ki: Float? = nil, pKi: Float? = nil, ic50: Float? = nil,
         sourceChainID: String? = nil) {
        self.id = UUID()
        self.name = name
        self.originalSMILES = smiles
        self.atoms = atoms
        self.bonds = bonds
        self.descriptors = descriptors
        self.isPrepared = isPrepared
        self.preparationDate = isPrepared ? Date() : nil
        self.conformerCount = conformerCount
        self.ki = ki
        self.pKi = pKi
        self.ic50 = ic50
        self.sourceChainID = sourceChainID
    }
}

extension LigandDescriptors {
    /// Crippen logP to one decimal. Values that round to zero print as "0.0":
    /// `String(format:)` keeps the sign of e.g. -0.04 and would show "-0.0".
    var logPText: String {
        let rounded = (logP * 10).rounded() / 10
        return String(format: "%.1f", rounded == 0 ? 0 : rounded)
    }
}

/// In-memory ligand database with search, filter, and persistent JSON storage.
@Observable
@MainActor
final class LigandDatabase {
    var entries: [LigandEntry] = [] {
        didSet { scheduleSave() }
    }
    var isProcessing: Bool = false
    var processingMessage: String = ""
    var processingProgress: Double = 0

    var count: Int { entries.count }

    /// Debounce timer for auto-save
    private var saveTask: Task<Void, Never>?
    private let saveDebounceInterval: Duration = .milliseconds(500)

    /// The preparation pipeline (enumerate / dominant / basic) that writes into this database.
    /// Shared by the Ligand Database window and the console so both see one run, one progress
    /// and one set of session defaults.
    @ObservationIgnored private(set) lazy var preparation = LigandPreparationService(database: self)

    /// Auto-save is only allowed while the in-memory entries are the authoritative copy of the
    /// default database file: the file did not exist at launch, or it was loaded/saved explicitly.
    /// Otherwise the first edit of a fresh session (or a project load) would overwrite the user's
    /// saved database with a partial set of entries.
    private var autoSaveEnabled: Bool

    init() {
        // Don't auto-load — user explicitly loads via File > Load Database
        autoSaveEnabled = !FileManager.default.fileExists(atPath: Self.defaultURL.path)
    }

    // MARK: - Helpers

    /// All entries (flat model — every entry is top-level). Backward compat alias.
    var topLevelEntries: [LigandEntry] { entries }

    /// Remove an entry by ID.
    func removeWithChildren(id: UUID) {
        entries.removeAll { $0.id == id }
    }

    /// All sibling entries that share the same parentName (for visual grouping).
    func siblings(of entry: LigandEntry) -> [LigandEntry] {
        guard let pn = entry.parentName else { return [] }
        return entries.filter { $0.parentName == pn && $0.id != entry.id }
    }

    /// Whether this entry has enumerated children in the database.
    func hasEnumeratedChildren(_ entry: LigandEntry) -> Bool {
        entries.contains { $0.parentName == entry.name }
    }

    /// Remove an entry and all its enumerated siblings (same parentName).
    func removeWithSiblings(parentName: String) {
        entries.removeAll { $0.parentName == parentName }
    }

    /// Perform multiple mutations on the entries array as a single batch.
    /// This triggers only one `didSet` (and thus one debounced save) instead of N.
    func batchMutate(_ block: (inout [LigandEntry]) -> Void) {
        block(&entries)
    }

    // MARK: - Add

    func add(_ entry: LigandEntry) {
        entries.append(entry)
    }

    func addFromSMILES(_ smiles: String, name: String) {
        var entry = LigandEntry(name: name, smiles: smiles)
        if let desc = RDKitBridge.computeDescriptors(smiles: smiles) {
            entry.descriptors = desc
        }
        entries.append(entry)
    }

    // MARK: - Remove

    func remove(at offsets: IndexSet) {
        entries.remove(atOffsets: offsets)
    }

    func remove(id: UUID) {
        entries.removeAll { $0.id == id }
    }

    func remove(ids: Set<UUID>) {
        entries.removeAll { ids.contains($0.id) }
    }

    func removeAll() {
        entries.removeAll()
    }

    // MARK: - Update

    func update(_ entry: LigandEntry) {
        if let idx = entries.firstIndex(where: { $0.id == entry.id }) {
            entries[idx] = entry
        }
    }

    // MARK: - Batch Import from SMILES list

    typealias ImportedAffinity = (ki: Float?, pKi: Float?, ic50: Float?)

    /// - Parameter affinities: experimental affinity data keyed by index into `lines`.
    func importSMILES(_ lines: [(smiles: String, name: String)], prepare: Bool = true,
                       addH: Bool = true, minimize: Bool = true, charges: Bool = true,
                       affinities: [Int: ImportedAffinity] = [:]) {
        isProcessing = true
        processingMessage = "Importing \(lines.count) molecules..."
        processingProgress = 0

        Task {
            // Build all entries (RDKit preparation + descriptors) in one background pass,
            // then insert them with a single mutation (one didSet / one debounced save).
            let newEntries = await Task.detached(priority: .userInitiated) { () -> [LigandEntry] in
                let results = prepare
                    ? RDKitBridge.batchProcess(entries: lines, addHydrogens: addH,
                                               minimize: minimize, computeCharges: charges)
                    : []
                return lines.enumerated().map { i, line in
                    var entry = LigandEntry(name: line.name, smiles: line.smiles)
                    if prepare {
                        if i < results.count, let mol = results[i].molecule {
                            entry.atoms = mol.atoms
                            entry.bonds = mol.bonds
                            entry.isPrepared = true
                            entry.preparationDate = Date()
                            entry.conformerCount = 1
                        }
                        entry.descriptors = RDKitBridge.computeDescriptors(smiles: line.smiles)
                    }
                    if let aff = affinities[i] {
                        entry.ki = aff.ki
                        entry.pKi = aff.pKi
                        entry.ic50 = aff.ic50
                    }
                    return entry
                }
            }.value

            batchMutate { entries in
                entries.reserveCapacity(entries.count + newEntries.count)
                entries.append(contentsOf: newEntries)
            }
            processingProgress = 1

            isProcessing = false
            processingMessage = ""
            ActivityLog.shared.success("Imported \(lines.count) ligands", category: .molecule)
        }
    }

    // MARK: - Import from .smi file

    func importSMIFile(url: URL, prepare: Bool = true) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        importSMILES(Self.smiLines(content, baseIndex: entries.count), prepare: prepare)
    }

    /// Parses a SMILES list: one molecule per line as `SMILES name` (tab or space separated,
    /// either column order), `#` comments ignored; unnamed molecules get `Mol_<n>`.
    nonisolated static func smiLines(_ content: String, baseIndex: Int) -> [(smiles: String, name: String)] {
        content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .enumerated()
            .compactMap { offset, line -> (smiles: String, name: String)? in
                // Split by tab first, then space
                let parts: [String]
                let tabParts = line.split(separator: "\t", maxSplits: 1).map(String.init)
                if tabParts.count > 1 {
                    parts = tabParts
                } else {
                    parts = line.split(separator: " ", maxSplits: 1).map(String.init)
                }
                if parts.isEmpty { return nil }

                if parts.count > 1 {
                    // Two columns: figure out which is SMILES; default to column 0
                    if !looksLikeSMILES(parts[0]) && looksLikeSMILES(parts[1]) {
                        return (parts[1], parts[0])
                    }
                    return (parts[0], parts[1])
                }
                return (parts[0], "Mol_\(baseIndex + offset + 1)")
            }
    }

    // MARK: - Import from CSV

    func importCSV(url: URL, smilesColumn: Int? = nil, nameColumn: Int? = nil, prepare: Bool = true) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let rows = Self.parseDelimitedRows(content)

        guard !rows.isEmpty else { return }

        // Detect header and column indices
        let header = rows[0].joined(separator: ",").lowercased()
        let hasHeader = header.contains("smiles") || header.contains("name") || header.contains("molecule")
            || header.contains("ki") || header.contains("ic50") || header.contains("pki")
        let dataRows = hasHeader ? Array(rows.dropFirst()) : rows

        // Auto-detect SMILES column if not specified
        let detectedSmilesCol: Int
        let detectedNameCol: Int
        var kiCol: Int? = nil
        var pKiCol: Int? = nil
        var ic50Col: Int? = nil

        if let sc = smilesColumn {
            detectedSmilesCol = sc
            detectedNameCol = nameColumn ?? (sc == 0 ? 1 : 0)
        } else if hasHeader {
            let headerCols = rows[0].map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            detectedSmilesCol = headerCols.firstIndex(where: { $0.contains("smiles") || $0.contains("molecule") || $0.contains("structure") }) ?? 0
            detectedNameCol = nameColumn ?? headerCols.firstIndex(where: { $0.contains("name") || $0.contains("id") || $0.contains("title") }) ?? (detectedSmilesCol == 0 ? 1 : 0)

            // Detect affinity columns
            kiCol = headerCols.firstIndex(where: { $0 == "ki" || $0 == "ki_nm" || $0 == "ki (nm)" })
            pKiCol = headerCols.firstIndex(where: { $0 == "pki" || $0 == "p_ki" || $0 == "pki_value" })
            ic50Col = headerCols.firstIndex(where: { $0 == "ic50" || $0 == "ic50_nm" || $0 == "ic50 (nm)" })
        } else {
            // No header: auto-detect by checking which column looks like SMILES
            let firstRow = dataRows.first ?? []
            detectedSmilesCol = firstRow.firstIndex(where: { Self.looksLikeSMILES($0) }) ?? 0
            detectedNameCol = nameColumn ?? (detectedSmilesCol == 0 ? 1 : 0)
        }

        let hasAffinity = kiCol != nil || pKiCol != nil || ic50Col != nil

        let baseIndex = entries.count
        var lines: [(smiles: String, name: String)] = []
        // Affinity data keyed by index into `lines`; applied when the entries are built
        // (importSMILES inserts asynchronously, so it cannot be patched in afterwards).
        var affinities: [Int: ImportedAffinity] = [:]
        for (offset, cols) in dataRows.enumerated() {
            guard detectedSmilesCol < cols.count else { continue }
            let smi = cols[detectedSmilesCol]
            let name = detectedNameCol < cols.count ? cols[detectedNameCol] : "Mol_\(baseIndex + offset + 1)"
            if hasAffinity {
                let ki = kiCol.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
                let pKi = pKiCol.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
                let ic50 = ic50Col.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
                if ki != nil || pKi != nil || ic50 != nil {
                    affinities[lines.count] = (ki: ki, pKi: pKi, ic50: ic50)
                }
            }
            lines.append((smi, name))
        }

        importSMILES(lines, prepare: prepare, affinities: affinities)

        if !affinities.isEmpty {
            ActivityLog.shared.info("Imported affinity data for \(affinities.count) ligands (Ki/pKi/IC50)", category: .molecule)
        }
    }

    nonisolated static func parseDelimitedRows(_ content: String, separator explicitSeparator: Character? = nil) -> [[String]] {
        let separator = explicitSeparator ?? detectDelimiter(in: content)
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false
        var index = content.startIndex

        func finishField() {
            row.append(field.trimmingCharacters(in: .whitespaces))
            field.removeAll(keepingCapacity: true)
        }

        func finishRow() {
            finishField()
            if row.contains(where: { !$0.isEmpty }) {
                rows.append(row)
            }
            row.removeAll(keepingCapacity: true)
        }

        while index < content.endIndex {
            let ch = content[index]
            let next = content.index(after: index)

            if ch == "\"" {
                if inQuotes, next < content.endIndex, content[next] == "\"" {
                    field.append("\"")
                    index = content.index(after: next)
                    continue
                }
                inQuotes.toggle()
            } else if ch == separator, !inQuotes {
                finishField()
            } else if (ch == "\n" || ch == "\r"), !inQuotes {
                finishRow()
                if ch == "\r", next < content.endIndex, content[next] == "\n" {
                    index = content.index(after: next)
                    continue
                }
            } else {
                field.append(ch)
            }

            index = next
        }

        if !field.isEmpty || !row.isEmpty {
            finishRow()
        }

        return rows
    }

    nonisolated private static func detectDelimiter(in content: String) -> Character {
        let firstLine = content.split(whereSeparator: { $0 == "\n" || $0 == "\r" }).first ?? ""
        return firstLine.contains("\t") ? "\t" : ","
    }

    /// Heuristic: does this string look like a SMILES notation?
    nonisolated static func looksLikeSMILES(_ s: String) -> Bool {
        guard s.count >= 2 else { return false }
        // SMILES have organic atoms and structural chars; scores/names don't
        let smilesChars = CharacterSet(charactersIn: "CNOSPFIBrcnospfi[]()=#@+-/\\%0123456789")
        let allSmilesChars = s.unicodeScalars.allSatisfy { smilesChars.contains($0) }
        let hasOrganic = s.contains(where: { "CNOScnos".contains($0) })
        // Reject pure numbers (scores like "1.8")
        let isNumber = Float(s) != nil
        return allSmilesChars && hasOrganic && !isNumber
    }

    // MARK: - Filter

    func filtered(minMW: Float? = nil, maxMW: Float? = nil,
                  maxLogP: Float? = nil, lipinski: Bool? = nil) -> [LigandEntry] {
        entries.filter { entry in
            guard let d = entry.descriptors else { return true }
            if let min = minMW, d.molecularWeight < min { return false }
            if let max = maxMW, d.molecularWeight > max { return false }
            if let max = maxLogP, d.logP > max { return false }
            if let lip = lipinski, d.lipinski != lip { return false }
            return true
        }
    }

    // MARK: - Persistence

    /// ~/Library/Application Support/Druse/ligand_database.json (Save / Load Database).
    static var defaultURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let druseDir = appSupport.appendingPathComponent("Druse", isDirectory: true)
        try? FileManager.default.createDirectory(at: druseDir, withIntermediateDirectories: true)
        return druseDir.appendingPathComponent("ligand_database.json")
    }

    /// Save the database to ~/Library/Application Support/Druse/ligand_database.json.
    /// Returns false (after logging the error) when the file could not be written.
    @discardableResult
    func save() -> Bool {
        do {
            let data = try Self.encode(entries)
            try data.write(to: Self.defaultURL, options: .atomic)
            autoSaveEnabled = true
            return true
        } catch {
            ActivityLog.shared.error("Failed to save ligand database: \(error.localizedDescription)", category: .system)
            return false
        }
    }

    /// Load the database from disk (File > Load Ligand Database).
    /// Returns false when there is no saved database or it could not be read (the error is logged).
    @discardableResult
    func load() -> Bool {
        guard FileManager.default.fileExists(atPath: Self.defaultURL.path) else { return false }
        do {
            let data = try Data(contentsOf: Self.defaultURL)
            entries = try Self.decodeEntries(from: data)
            autoSaveEnabled = true
            ActivityLog.shared.info("Loaded \(entries.count) ligands from database", category: .system)
            return true
        } catch {
            ActivityLog.shared.error("Failed to load ligand database: \(error.localizedDescription)", category: .system)
            return false
        }
    }

    /// Schedule a debounced save. Cancels any pending save and waits before writing.
    /// JSON encoding and the file write run off the main actor; writes are serialized so an
    /// older snapshot can never land on disk after a newer one.
    private func scheduleSave() {
        guard autoSaveEnabled else { return }
        saveTask?.cancel()
        let previousSave = saveTask
        let debounce = saveDebounceInterval
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: debounce)
            guard !Task.isCancelled, let snapshot = self?.entries else { return }
            await previousSave?.value
            let url = Self.defaultURL
            let failure: String? = await Task.detached(priority: .utility) {
                do {
                    try Self.encode(snapshot).write(to: url, options: .atomic)
                    return nil
                } catch {
                    return error.localizedDescription
                }
            }.value
            if let failure {
                ActivityLog.shared.error("Failed to save ligand database: \(failure)", category: .system)
            }
        }
    }

    /// Encode the database (or `subset` of it) in the on-disk JSON format (project save, JSON export).
    func encodeToData(_ subset: [LigandEntry]? = nil) throws -> Data {
        try Self.encode(subset ?? entries)
    }

    /// Replace the current entries with ones decoded from a project's ligand file.
    /// The project's ligands are not the default database, so auto-save is suspended
    /// until the user explicitly saves or loads the default database.
    func replaceWithProjectEntries(_ projectEntries: [LigandEntry]) {
        autoSaveEnabled = false
        saveTask?.cancel()
        entries = projectEntries
    }

    /// Export as SDF to a user-selected file.
    func exportSDF() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "sdf")].compactMap { $0 }
        panel.nameFieldStringValue = "ligands.sdf"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        do {
            let sdf = SDFWriter.writeLigandDatabase(entries)
            try SDFWriter.save(sdf, to: url)
            ActivityLog.shared.success("Exported \(entries.count) ligands to SDF", category: .molecule)
        } catch {
            ActivityLog.shared.error("SDF export failed: \(error.localizedDescription)", category: .molecule)
        }
    }

    // MARK: - Codable Helpers

    /// Current on-disk format version.
    nonisolated private static let currentFormatVersion = 6

    /// v3: Serialized conformer within a chemical form.
    private struct SerializableConformer: Codable {
        let positions: [Float]   // flat [x,y,z,x,y,z,...] for all atoms
        let energy: Double
    }

    /// v3: Serialized chemical form within a ligand entry.
    private struct SerializableForm: Codable {
        let smiles: String
        let kind: Int            // ChemicalFormKind.rawValue
        let label: String
        let boltzmannWeight: Double
        let relativeEnergy: Double
        let atomElements: [Int]?   // shared across conformers
        let atomNames: [String]?   // shared across conformers
        let bondData: [Int]?       // shared across conformers
        let conformers: [SerializableConformer]
    }

    /// Legacy serializable variant (v1 nested model).
    private struct SerializableVariant: Codable {
        let smiles: String
        let relativeEnergy: Double
        let kind: Int           // 0=tautomer, 1=protomer
        let label: String
        let isPrepared: Bool
        let conformerCount: Int
        let atomData: [Float]?
        let atomElements: [Int]?
        let atomNames: [String]?
        let bondData: [Int]?
    }

    private struct SerializableEntry: Codable {
        let name: String
        let smiles: String
        let isPrepared: Bool
        let conformerCount: Int?
        let mw: Float?
        let logP: Float?
        let tpsa: Float?
        let hbd: Int?
        let hba: Int?
        let rotBonds: Int?
        let rings: Int?
        let aromaticRings: Int?
        let heavyAtomCount: Int?
        let fractionCSP3: Float?
        let lipinski: Bool?
        let veber: Bool?
        let ki: Float?
        let pKi: Float?
        let ic50: Float?
        let atomData: [Float]?
        let atomElements: [Int]?
        let atomNames: [String]?
        let bondData: [Int]?
        // Legacy v1 variants
        let variants: [SerializableVariant]?
        // Legacy v2 hierarchy
        let parentID: String?
        let variantKind: Int?
        let relativeEnergy: Double?
        let conformerPositions: [[Float]]?
        let conformerEnergies: [Double]?
        // --- v3 fields ---
        let forms: [SerializableForm]?
        // --- v4 flat model fields ---
        let parentName: String?
        let populationWeight: Double?
        let formKind: Int?          // ChemicalFormKind.rawValue
        let formRelativeEnergy: Double?
        let entryConformers: [SerializableConformer]?
        // --- v5 fields ---
        /// User-imported custom properties from CSV/SMI/SDF columns that don't
        /// map to a built-in field. Optional for backward compat with v3/v4.
        let userProperties: [String: String]?
        // --- v6 fields ---
        /// Per-atom partial and formal charges (nil when all zero / pre-v6 files).
        let atomCharges: [Float]?
        let atomFormalCharges: [Int]?
    }

    /// Top-level wrapper for versioned on-disk format.
    private struct DatabaseWrapper: Codable {
        let version: Int
        let entries: [SerializableEntry]
    }

    nonisolated private static func encodeAtoms(_ atoms: [Atom]) -> (data: [Float]?, elements: [Int]?, names: [String]?,
                                                                      charges: [Float]?, formalCharges: [Int]?) {
        guard !atoms.isEmpty else { return (nil, nil, nil, nil, nil) }
        return (
            atoms.flatMap { [$0.position.x, $0.position.y, $0.position.z] },
            atoms.map { $0.element.rawValue },
            atoms.map(\.name),
            atoms.contains { $0.charge != 0 } ? atoms.map(\.charge) : nil,
            atoms.contains { $0.formalCharge != 0 } ? atoms.map(\.formalCharge) : nil
        )
    }

    nonisolated private static func encodeBonds(_ bonds: [Bond]) -> [Int]? {
        guard !bonds.isEmpty else { return nil }
        return bonds.flatMap { [$0.atomIndex1, $0.atomIndex2, $0.order == .single ? 1 : $0.order == .double ? 2 : $0.order == .triple ? 3 : 4] }
    }

    private static func decodeAtoms(data: [Float]?, elements: [Int]?, names: [String]?,
                                    charges: [Float]? = nil, formalCharges: [Int]? = nil) -> [Atom] {
        guard let ad = data, let ae = elements, let an = names else { return [] }
        var atoms: [Atom] = []
        for i in stride(from: 0, to: ad.count - 2, by: 3) {
            let idx = i / 3
            guard idx < ae.count, idx < an.count else { break }
            atoms.append(Atom(
                id: idx, element: Element(rawValue: ae[idx]) ?? .C,
                position: SIMD3(ad[i], ad[i+1], ad[i+2]),
                name: an[idx], residueName: "LIG", residueSeq: 1, chainID: "L",
                charge: charges.flatMap { idx < $0.count ? $0[idx] : nil } ?? 0,
                formalCharge: formalCharges.flatMap { idx < $0.count ? $0[idx] : nil } ?? 0,
                isHetAtom: true
            ))
        }
        return atoms
    }

    private static func decodeBonds(data: [Int]?) -> [Bond] {
        guard let bd = data else { return [] }
        var bonds: [Bond] = []
        for i in stride(from: 0, to: bd.count - 2, by: 3) {
            let order: BondOrder = switch bd[i+2] { case 2: .double; case 3: .triple; case 4: .aromatic; default: .single }
            bonds.append(Bond(id: bonds.count, atomIndex1: bd[i], atomIndex2: bd[i+1], order: order))
        }
        return bonds
    }

    /// Decode conformers from serialized flat position arrays and energies.
    /// Uses the entry's bonds as the template (conformers share the same bonding).
    private static func decodeConformers(positions: [[Float]]?, energies: [Double]?,
                                         templateElements: [Int]?, templateNames: [String]?,
                                         templateBonds: [Int]?) -> [LigandConformer] {
        guard let posArrays = positions, !posArrays.isEmpty else { return [] }
        let energyList = energies ?? Array(repeating: 0.0, count: posArrays.count)
        let bonds = decodeBonds(data: templateBonds)

        return posArrays.enumerated().compactMap { confIdx, flatPos -> LigandConformer? in
            // Reconstruct atoms from the flat position array + template element/name info
            guard let elems = templateElements, let names = templateNames else { return nil }
            var atoms: [Atom] = []
            for i in stride(from: 0, to: flatPos.count - 2, by: 3) {
                let aIdx = i / 3
                guard aIdx < elems.count, aIdx < names.count else { break }
                atoms.append(Atom(
                    id: aIdx, element: Element(rawValue: elems[aIdx]) ?? .C,
                    position: SIMD3(flatPos[i], flatPos[i+1], flatPos[i+2]),
                    name: names[aIdx], residueName: "LIG", residueSeq: 1, chainID: "L",
                    charge: 0, formalCharge: 0, isHetAtom: true
                ))
            }
            let energy = confIdx < energyList.count ? energyList[confIdx] : 0.0
            return LigandConformer(id: confIdx, atoms: atoms, bonds: bonds, energy: energy)
        }
    }

    // MARK: - Encode (v4 flat format)

    nonisolated private static func encode(_ entries: [LigandEntry]) throws -> Data {
        // Only encode top-level entries (skip legacy children)
        let topLevel = entries.filter { $0.parentID == nil }
        let serializable = topLevel.map { entry -> SerializableEntry in
            let (atomData, atomElements, atomNames, atomCharges, atomFormalCharges) = Self.encodeAtoms(entry.atoms)
            let bondData = Self.encodeBonds(entry.bonds)

            // Encode per-entry conformers for the flat model
            let entryConfs: [SerializableConformer]? = entry.conformers.isEmpty ? nil :
                entry.conformers.map { conf in
                    let positions = conf.atoms.flatMap { [$0.position.x, $0.position.y, $0.position.z] }
                    return SerializableConformer(positions: positions, energy: conf.energy)
                }

            return SerializableEntry(
                name: entry.name, smiles: entry.originalSMILES, isPrepared: entry.isPrepared,
                conformerCount: entry.conformerCount,
                mw: entry.descriptors?.molecularWeight, logP: entry.descriptors?.logP,
                tpsa: entry.descriptors?.tpsa, hbd: entry.descriptors?.hbd, hba: entry.descriptors?.hba,
                rotBonds: entry.descriptors?.rotatableBonds, rings: entry.descriptors?.rings,
                aromaticRings: entry.descriptors?.aromaticRings, heavyAtomCount: entry.descriptors?.heavyAtomCount,
                fractionCSP3: entry.descriptors?.fractionCSP3, lipinski: entry.descriptors?.lipinski,
                veber: entry.descriptors?.veber,
                ki: entry.ki, pKi: entry.pKi, ic50: entry.ic50,
                atomData: atomData, atomElements: atomElements, atomNames: atomNames, bondData: bondData,
                variants: nil, parentID: nil, variantKind: nil, relativeEnergy: nil,
                conformerPositions: nil, conformerEnergies: nil,
                forms: nil,
                parentName: entry.parentName,
                populationWeight: entry.populationWeight,
                formKind: entry.formKind?.rawValue,
                formRelativeEnergy: entry.relativeEnergy,
                entryConformers: entryConfs,
                userProperties: entry.userProperties.isEmpty ? nil : entry.userProperties,
                atomCharges: atomCharges,
                atomFormalCharges: atomFormalCharges
            )
        }
        let wrapper = DatabaseWrapper(version: Self.currentFormatVersion, entries: serializable)
        return try JSONEncoder().encode(wrapper)
    }

    // MARK: - Decode (v1 + v2 + v3 format)

    static func decodeEntries(from data: Data) throws -> [LigandEntry] {
        if let wrapper = try? JSONDecoder().decode(DatabaseWrapper.self, from: data) {
            if wrapper.version >= 3 {
                return decodeEntriesV3(wrapper.entries)
            }
            // v2 format: flat parent+child hierarchy → migrate to forms
            return decodeEntriesV2(wrapper.entries)
        }
        // Fall back to legacy v1 format: plain [SerializableEntry] array
        let decoded = try JSONDecoder().decode([SerializableEntry].self, from: data)
        return decodeEntriesV2(decoded)
    }

    /// Decode v3/v4 entries (flat model with optional legacy forms migration).
    private static func decodeEntriesV3(_ serializableEntries: [SerializableEntry]) -> [LigandEntry] {
        serializableEntries.map { s in
            let atoms = decodeAtoms(data: s.atomData, elements: s.atomElements, names: s.atomNames,
                                    charges: s.atomCharges, formalCharges: s.atomFormalCharges)
            let bonds = decodeBonds(data: s.bondData)
            let desc = decodeDescriptors(s)

            var entry = LigandEntry(
                name: s.name, smiles: s.smiles, atoms: atoms, bonds: bonds,
                descriptors: desc, isPrepared: s.isPrepared,
                conformerCount: s.conformerCount ?? 0,
                ki: s.ki, pKi: s.pKi, ic50: s.ic50
            )

            // v4 flat model fields
            entry.parentName = s.parentName
            entry.populationWeight = s.populationWeight
            entry.formKind = s.formKind.flatMap { ChemicalFormKind(rawValue: $0) }
            entry.relativeEnergy = s.formRelativeEnergy

            // v5 user-imported properties (empty for v3/v4 entries)
            entry.userProperties = s.userProperties ?? [:]

            // Decode per-entry conformers
            if let serConfs = s.entryConformers, !serConfs.isEmpty {
                entry.conformers = serConfs.enumerated().map { idx, sc in
                    var confAtoms = atoms  // same topology, different positions
                    let positions = sc.positions
                    for i in confAtoms.indices {
                        let base = i * 3
                        if base + 2 < positions.count {
                            confAtoms[i].position = SIMD3(positions[base], positions[base+1], positions[base+2])
                        }
                    }
                    return Conformer3D(id: idx, atoms: confAtoms, bonds: bonds, energy: sc.energy)
                }
            }

            return entry
        }
    }

    /// Decode v2 entries (flat parent+child hierarchy) and migrate to flat v4 model.
    /// Legacy children are converted to flat entries with `parentName` set.
    private static func decodeEntriesV2(_ serializableEntries: [SerializableEntry]) -> [LigandEntry] {
        // Track which serializable entries are children (have a parentID) so we can
        // resolve parentID → parent name in a second pass.
        struct DecodedV2 {
            var entry: LigandEntry
            var legacyParentID: UUID?
            var legacyVariantKind: VariantKind?
            var legacyLabel: String?
        }

        var decoded: [DecodedV2] = []

        for s in serializableEntries {
            let atoms = decodeAtoms(data: s.atomData, elements: s.atomElements, names: s.atomNames)
            let bonds = decodeBonds(data: s.bondData)
            let desc = decodeDescriptors(s)
            let legacyParentID: UUID? = s.parentID.flatMap { UUID(uuidString: $0) }
            let variantKind: VariantKind? = s.variantKind.flatMap { VariantKind(rawValue: $0) }
            let conformers = decodeConformers(
                positions: s.conformerPositions, energies: s.conformerEnergies,
                templateElements: s.atomElements, templateNames: s.atomNames, templateBonds: s.bondData
            )

            var entry = LigandEntry(
                name: s.name, smiles: s.smiles, atoms: atoms, bonds: bonds,
                descriptors: desc, isPrepared: s.isPrepared,
                conformerCount: s.conformerCount ?? 0,
                ki: s.ki, pKi: s.pKi, ic50: s.ic50,
                sourceChainID: nil
            )
            entry.conformers = conformers
            entry.relativeEnergy = s.relativeEnergy

            decoded.append(DecodedV2(
                entry: entry, legacyParentID: legacyParentID,
                legacyVariantKind: variantKind, legacyLabel: nil
            ))

            // Promote v1 nested variants to flat entries
            for sv in (s.variants ?? []) {
                let vAtoms = decodeAtoms(data: sv.atomData, elements: sv.atomElements, names: sv.atomNames)
                let vBonds = decodeBonds(data: sv.bondData)
                let vKind = VariantKind(rawValue: sv.kind) ?? .tautomer

                var child = LigandEntry(
                    name: "\(s.name)_\(sv.label)",
                    smiles: sv.smiles, atoms: vAtoms, bonds: vBonds,
                    isPrepared: sv.isPrepared, conformerCount: sv.conformerCount,
                    sourceChainID: nil
                )
                child.relativeEnergy = sv.relativeEnergy

                decoded.append(DecodedV2(
                    entry: child, legacyParentID: entry.id,
                    legacyVariantKind: vKind, legacyLabel: "\(sv.label) of \(s.name)"
                ))
            }
        }

        // Build a lookup from legacy UUID → parent name for children
        let idToName: [UUID: String] = Dictionary(
            decoded.map { ($0.entry.id, $0.entry.name) },
            uniquingKeysWith: { first, _ in first }
        )

        // Convert to flat entries: children get parentName / formKind set
        return decoded.map { d in
            var entry = d.entry
            if let pid = d.legacyParentID {
                entry.parentName = idToName[pid] ?? "Unknown"
                let kind: ChemicalFormKind = switch d.legacyVariantKind {
                case .tautomer: .tautomer
                case .protomer: .protomer
                case nil: .parent
                }
                entry.formKind = kind
            }
            return entry
        }
    }

    private static func decodeDescriptors(_ s: SerializableEntry) -> LigandDescriptors? {
        guard let mw = s.mw else { return nil }
        return LigandDescriptors(
            molecularWeight: mw, exactMW: mw, logP: s.logP ?? 0, tpsa: s.tpsa ?? 0,
            hbd: s.hbd ?? 0, hba: s.hba ?? 0, rotatableBonds: s.rotBonds ?? 0,
            rings: s.rings ?? 0, aromaticRings: s.aromaticRings ?? 0,
            heavyAtomCount: s.heavyAtomCount ?? 0, fractionCSP3: s.fractionCSP3 ?? 0,
            lipinski: s.lipinski ?? false, veber: s.veber ?? false
        )
    }
}

// MARK: - Dockable entries

extension LigandDatabase {
    /// Entries that are one docking job each: prepared, and not a source molecule whose
    /// enumerated forms are in the database (those forms are docked instead).
    var dockableEntries: [LigandEntry] {
        let parentsWithForms = Set(entries.compactMap(\.parentName))
        return entries.filter { $0.isPrepared && !($0.parentName == nil && parentsWithForms.contains($0.name)) }
    }
}

// MARK: - Reading ligand files (console imports)

/// A ligand file that could not be read, with a message naming the problem.
struct LigandImportError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// How to read a delimited ligand table (CSV/TSV). Column specs are a header name
/// (case-insensitive) or a 1-based column number; nil means auto-detect.
struct LigandTableImportOptions: Sendable {
    var smilesColumn: String?
    var nameColumn: String?
    var kiColumn: String?
    var pKiColumn: String?
    var ic50Column: String?
    /// nil: tab if the first line contains one, else comma.
    var delimiter: Character?
    /// nil: a header is assumed when the first row names a SMILES/name/affinity column.
    var hasHeader: Bool?
    /// Keep unmapped columns as user properties (table columns in the Ligand Database window).
    var keepExtraColumns = true
}

extension LigandDatabase {

    /// userProperties keys the Ligand Database window shows affinity data under.
    nonisolated static let kiPropertyKey = "Ki (nM)"
    nonisolated static let pKiPropertyKey = "pKi"
    nonisolated static let ic50PropertyKey = "IC50 (nM)"

    /// Stores affinity values on the entry and mirrors them into its user properties, as the
    /// window's mapped imports do, so the table shows them as columns.
    nonisolated static func setAffinity(_ entry: inout LigandEntry, ki: Float?, pKi: Float?, ic50: Float?) {
        if let ki {
            entry.ki = ki
            entry.userProperties[kiPropertyKey] = String(format: "%g", ki)
        }
        if let pKi {
            entry.pKi = pKi
            entry.userProperties[pKiPropertyKey] = String(format: "%g", pKi)
        }
        if let ic50 {
            entry.ic50 = ic50
            entry.userProperties[ic50PropertyKey] = String(format: "%g", ic50)
        }
    }

    /// Reads a SMILES list into unprepared entries with descriptors.
    nonisolated static func readSMIFile(_ url: URL, baseIndex: Int) throws -> [LigandEntry] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return smiLines(content, baseIndex: baseIndex).map { line in
            LigandEntry(name: line.name, smiles: line.smiles,
                        descriptors: RDKitBridge.computeDescriptors(smiles: line.smiles))
        }
    }

    /// Reads a CSV/TSV table into unprepared entries with descriptors, affinity data and
    /// (optionally) every other column as user properties.
    nonisolated static func readTableFile(_ url: URL, options: LigandTableImportOptions,
                                          baseIndex: Int) throws -> [LigandEntry] {
        let content = try String(contentsOf: url, encoding: .utf8)
        let rows = parseDelimitedRows(content, separator: options.delimiter ?? tableDelimiter(in: content))
        guard let first = rows.first else { return [] }
        let normalized = first.map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
        let looksLikeHeader = normalized.contains {
            $0.contains("smiles") || $0.contains("name") || $0.contains("molecule")
                || ["ki", "pki", "ic50", "id", "title"].contains($0)
        } || !first.contains(where: looksLikeSMILES)
        let hasHeader = options.hasHeader ?? looksLikeHeader
        let headers = hasHeader ? first.map { $0.trimmingCharacters(in: .whitespaces) } : []
        let dataRows = hasHeader ? Array(rows.dropFirst()) : rows
        let columnCount = rows.map(\.count).max() ?? 0

        func resolve(_ spec: String?, role: String, auto: () -> Int?) throws -> Int? {
            guard let spec else { return auto() }
            if let number = Int(spec) {
                guard (1...max(columnCount, 1)).contains(number) else {
                    throw LigandImportError(message: "\(role) column \(number) is out of range (the table has \(columnCount) columns)")
                }
                return number - 1
            }
            guard hasHeader else {
                throw LigandImportError(message: "\(role) column '\(spec)' given by name, but the table has no header row — use a column number")
            }
            guard let index = headers.firstIndex(where: { $0.lowercased() == spec.lowercased() }) else {
                throw LigandImportError(message: "no column named '\(spec)' for \(role); columns: \(headers.joined(separator: ", "))")
            }
            return index
        }
        func header(matching names: [String]) -> Int? {
            normalized.indices.first { hasHeader && names.contains(normalized[$0]) }
        }

        let smilesIndex = try resolve(options.smilesColumn, role: "SMILES") {
            if hasHeader, let index = normalized.firstIndex(where: {
                $0.contains("smiles") || $0.contains("molecule") || $0.contains("structure")
            }) { return index }
            return dataRows.first?.firstIndex(where: looksLikeSMILES)
        }
        guard let smilesIndex else {
            throw LigandImportError(message: "could not find a SMILES column — name it with smiles-col=")
        }
        let nameIndex = try resolve(options.nameColumn, role: "name") {
            guard hasHeader else { return smilesIndex == 0 && columnCount > 1 ? 1 : (smilesIndex > 0 ? 0 : nil) }
            return normalized.indices.first { $0 != smilesIndex
                && (normalized[$0].contains("name") || normalized[$0] == "id" || normalized[$0].contains("title")) }
        }
        let kiIndex = try resolve(options.kiColumn, role: "Ki") { header(matching: ["ki", "ki_nm", "ki (nm)"]) }
        let pKiIndex = try resolve(options.pKiColumn, role: "pKi") { header(matching: ["pki", "p_ki", "pki_value"]) }
        let ic50Index = try resolve(options.ic50Column, role: "IC50") {
            header(matching: ["ic50", "ic50_nm", "ic50 (nm)", "ic50_value"])
        }
        let mapped = Set([smilesIndex, nameIndex, kiIndex, pKiIndex, ic50Index].compactMap { $0 })

        return dataRows.enumerated().compactMap { offset, cells -> LigandEntry? in
            func cell(_ index: Int?) -> String? {
                guard let index, index < cells.count, !cells[index].isEmpty else { return nil }
                return cells[index]
            }
            guard let smiles = cell(smilesIndex) else { return nil }
            var entry = LigandEntry(name: cell(nameIndex) ?? "Mol_\(baseIndex + offset + 1)", smiles: smiles,
                                    descriptors: RDKitBridge.computeDescriptors(smiles: smiles))
            setAffinity(&entry, ki: cell(kiIndex).flatMap(Float.init), pKi: cell(pKiIndex).flatMap(Float.init),
                        ic50: cell(ic50Index).flatMap(Float.init))
            if options.keepExtraColumns && hasHeader {
                for (index, key) in headers.enumerated() where !mapped.contains(index) && !key.isEmpty {
                    if let value = cell(index) { entry.userProperties[key] = value }
                }
            }
            return entry
        }
    }

    /// The most frequent of tab, semicolon and comma on the first line (comma when none occurs).
    nonisolated private static func tableDelimiter(in content: String) -> Character {
        let firstLine = content.split(whereSeparator: { $0 == "\n" || $0 == "\r" }).first ?? ""
        let counts: [(Character, Int)] = ["\t", ";", ","].map { delimiter in
            (delimiter, firstLine.filter { $0 == delimiter }.count)
        }
        return counts.max { $0.1 < $1.1 }.flatMap { $0.1 > 0 ? $0.0 : nil } ?? ","
    }

    /// SD fields Druse writes on export that are recomputed on import rather than kept as properties.
    nonisolated private static let derivedSDFields: Set<String> = [
        "prepared", "mw", "logp", "tpsa", "hbd", "hba", "rotbonds", "lipinski", "parent", "formkind",
        "population", "relativeenergy", "conformer", "mmff_energy"
    ]

    /// Reads SDF/MOL/MOL2 molecules with their file coordinates. SD data fields named like
    /// Ki/pKi/IC50 become affinity data, the rest user properties. Entries are marked prepared
    /// when `coordinatesArePrepared` (the file's 3D structures are used for docking as they are).
    nonisolated static func readStructureFile(_ url: URL, coordinatesArePrepared: Bool) throws -> [LigandEntry] {
        let molecules: [MoleculeData]
        switch url.pathExtension.lowercased() {
        case "mol2":
            molecules = try MOL2Parser.parse(url: url).map {
                MoleculeData(name: $0.name, title: $0.title, atoms: $0.atoms, bonds: $0.bonds)
            }
        default:
            molecules = try SDFParser.parse(url: url)
        }
        return molecules.map { molecule in
            var ki: Float?, pKi: Float?, ic50: Float?
            var smiles = molecule.smiles ?? ""
            var properties: [String: String] = [:]
            for (key, raw) in molecule.properties {
                let value = raw.trimmingCharacters(in: .whitespaces)
                switch key.lowercased() {
                case "ki", "ki_nm", "ki (nm)": ki = Float(value)
                case "pki", "p_ki", "pki_value": pKi = Float(value)
                case "ic50", "ic50_nm", "ic50 (nm)", "ic50_value": ic50 = Float(value)
                case "smiles" where smiles.isEmpty: smiles = value
                case let name where derivedSDFields.contains(name) || name == "smiles": break
                default: if !value.isEmpty { properties[key] = value }
                }
            }
            if smiles.isEmpty {
                smiles = molecule.molBlock.flatMap(RDKitBridge.smilesFromMolBlock)
                    ?? RDKitBridge.atomsBondsToSMILES(atoms: molecule.atoms, bonds: molecule.bonds) ?? ""
            }
            var entry = LigandEntry(
                name: molecule.name.isEmpty ? url.deletingPathExtension().lastPathComponent : molecule.name,
                smiles: smiles, atoms: molecule.atoms, bonds: molecule.bonds,
                descriptors: smiles.isEmpty ? nil : RDKitBridge.computeDescriptors(smiles: smiles),
                isPrepared: coordinatesArePrepared, conformerCount: coordinatesArePrepared ? 1 : 0
            )
            entry.userProperties = properties
            setAffinity(&entry, ki: ki, pKi: pKi, ic50: ic50)
            return entry
        }
    }
}
