import Foundation
import AppKit
import UniformTypeIdentifiers
import simd

/// An entry in the ligand database.
struct LigandEntry: Identifiable, Sendable {
    let id: UUID
    var name: String
    var smiles: String
    var atoms: [Atom]
    var bonds: [Bond]
    var descriptors: LigandDescriptors?
    var isPrepared: Bool
    var preparationDate: Date?
    var conformerCount: Int

    init(name: String, smiles: String, atoms: [Atom] = [], bonds: [Bond] = [],
         descriptors: LigandDescriptors? = nil, isPrepared: Bool = false,
         conformerCount: Int = 0) {
        self.id = UUID()
        self.name = name
        self.smiles = smiles
        self.atoms = atoms
        self.bonds = bonds
        self.descriptors = descriptors
        self.isPrepared = isPrepared
        self.preparationDate = isPrepared ? Date() : nil
        self.conformerCount = conformerCount
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

    init() {
        // Don't auto-load — user explicitly loads via File > Load Database
    }

    // MARK: - Add

    func add(_ entry: LigandEntry) {
        entries.append(entry)
    }

    func addFromSMILES(_ smiles: String, name: String) {
        let entry = LigandEntry(name: name, smiles: smiles)
        entries.append(entry)
    }

    // MARK: - Remove

    func remove(at offsets: IndexSet) {
        entries.remove(atOffsets: offsets)
    }

    func remove(id: UUID) {
        entries.removeAll { $0.id == id }
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

    // MARK: - Prepare Single

    func prepareEntry(at index: Int, addH: Bool = true, minimize: Bool = true, charges: Bool = true) {
        guard index < entries.count else { return }
        let smiles = entries[index].smiles
        let name = entries[index].name

        Task {
            let (mol, desc, error) = await Task.detached {
                RDKitBridge.prepareLigand(smiles: smiles, name: name,
                                          addHydrogens: addH, minimize: minimize, computeCharges: charges)
            }.value

            if let mol {
                entries[index].atoms = mol.atoms
                entries[index].bonds = mol.bonds
                entries[index].descriptors = desc
                entries[index].isPrepared = true
                entries[index].preparationDate = Date()
                entries[index].conformerCount = 1
            }
            if let error {
                ActivityLog.shared.error("Failed to prepare \(name): \(error)", category: .prep)
            }
        }
    }

    // MARK: - Batch Import from SMILES list

    func importSMILES(_ lines: [(smiles: String, name: String)], prepare: Bool = true,
                       addH: Bool = true, minimize: Bool = true, charges: Bool = true) {
        isProcessing = true
        processingMessage = "Importing \(lines.count) molecules..."
        processingProgress = 0

        Task {
            if prepare {
                let results = await Task.detached {
                    RDKitBridge.batchProcess(entries: lines, addHydrogens: addH,
                                             minimize: minimize, computeCharges: charges)
                }.value

                for (i, result) in results.enumerated() {
                    let line = lines[i]
                    var entry = LigandEntry(name: line.name, smiles: line.smiles)
                    if let mol = result.molecule {
                        entry.atoms = mol.atoms
                        entry.bonds = mol.bonds
                        entry.isPrepared = true
                        entry.preparationDate = Date()
                        entry.conformerCount = 1
                    }
                    // Compute descriptors
                    if let desc = await Task.detached(operation: { RDKitBridge.computeDescriptors(smiles: line.smiles) }).value {
                        entry.descriptors = desc
                    }
                    entries.append(entry)
                    processingProgress = Double(i + 1) / Double(results.count)
                }
            } else {
                for line in lines {
                    entries.append(LigandEntry(name: line.name, smiles: line.smiles))
                }
            }

            isProcessing = false
            processingMessage = ""
            ActivityLog.shared.success("Imported \(lines.count) ligands", category: .molecule)
        }
    }

    // MARK: - Import from .smi file

    func importSMIFile(url: URL, prepare: Bool = true) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let baseIndex = entries.count
        let lines = content.components(separatedBy: .newlines)
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
                    let spaceParts = line.split(separator: " ", maxSplits: 1).map(String.init)
                    parts = spaceParts
                }
                if parts.isEmpty { return nil }

                if parts.count > 1 {
                    // Two columns: figure out which is SMILES
                    let col0IsSMILES = Self.looksLikeSMILES(parts[0])
                    let col1IsSMILES = Self.looksLikeSMILES(parts[1])
                    if col0IsSMILES {
                        return (parts[0], parts[1])
                    } else if col1IsSMILES {
                        return (parts[1], parts[0])
                    }
                    // Default: assume column 0 is SMILES
                    return (parts[0], parts[1])
                } else {
                    return (parts[0], "Mol_\(baseIndex + offset + 1)")
                }
            }

        importSMILES(lines, prepare: prepare)
    }

    // MARK: - Import from CSV

    func importCSV(url: URL, smilesColumn: Int? = nil, nameColumn: Int? = nil, prepare: Bool = true) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let rows = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !rows.isEmpty else { return }

        // Detect header and column indices
        let header = rows[0].lowercased()
        let hasHeader = header.contains("smiles") || header.contains("name") || header.contains("molecule")
        let dataRows = hasHeader ? Array(rows.dropFirst()) : rows

        // Auto-detect SMILES column if not specified
        let detectedSmilesCol: Int
        let detectedNameCol: Int

        if let sc = smilesColumn {
            detectedSmilesCol = sc
            detectedNameCol = nameColumn ?? (sc == 0 ? 1 : 0)
        } else if hasHeader {
            let headerCols = rows[0].components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            detectedSmilesCol = headerCols.firstIndex(where: { $0.contains("smiles") || $0.contains("molecule") || $0.contains("structure") }) ?? 0
            detectedNameCol = nameColumn ?? headerCols.firstIndex(where: { $0.contains("name") || $0.contains("id") || $0.contains("title") }) ?? (detectedSmilesCol == 0 ? 1 : 0)
        } else {
            // No header: auto-detect by checking which column looks like SMILES
            let firstRow = dataRows.first.map { $0.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) } } ?? []
            detectedSmilesCol = firstRow.firstIndex(where: { Self.looksLikeSMILES($0) }) ?? 0
            detectedNameCol = nameColumn ?? (detectedSmilesCol == 0 ? 1 : 0)
        }

        let baseIndex = entries.count
        let lines = dataRows.enumerated().compactMap { offset, row -> (smiles: String, name: String)? in
            let cols = row.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            guard detectedSmilesCol < cols.count else { return nil }
            let smi = cols[detectedSmilesCol]
            let name = detectedNameCol < cols.count ? cols[detectedNameCol] : "Mol_\(baseIndex + offset + 1)"
            return (smi, name)
        }

        importSMILES(lines, prepare: prepare)
    }

    /// Heuristic: does this string look like a SMILES notation?
    private static func looksLikeSMILES(_ s: String) -> Bool {
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

    private static var defaultURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let druseDir = appSupport.appendingPathComponent("Druse", isDirectory: true)
        try? FileManager.default.createDirectory(at: druseDir, withIntermediateDirectories: true)
        return druseDir.appendingPathComponent("ligand_database.json")
    }

    /// Save the database to ~/Library/Application Support/Druse/ligand_database.json
    func save() {
        do {
            let data = try encodeToDisk()
            try data.write(to: Self.defaultURL, options: .atomic)
        } catch {
            ActivityLog.shared.error("Failed to save ligand database: \(error.localizedDescription)", category: .system)
        }
    }

    /// Load the database from disk. Called automatically on init.
    func load() {
        guard FileManager.default.fileExists(atPath: Self.defaultURL.path) else { return }
        do {
            let data = try Data(contentsOf: Self.defaultURL)
            try decodeFromDisk(data)
            ActivityLog.shared.info("Loaded \(entries.count) ligands from database", category: .system)
        } catch {
            ActivityLog.shared.error("Failed to load ligand database: \(error.localizedDescription)", category: .system)
        }
    }

    /// Schedule a debounced save. Cancels any pending save and waits before writing.
    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: self?.saveDebounceInterval ?? .milliseconds(500))
            guard !Task.isCancelled else { return }
            self?.save()
        }
    }

    // Backward compatibility aliases
    func saveToDefault() { save() }
    func loadFromDefault() { load() }

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
        // Atom coordinates stored as flat [x,y,z,x,y,z,...] for compactness
        let atomData: [Float]?
        let atomElements: [Int]?
        let atomNames: [String]?
        let bondData: [Int]?  // [a1,a2,order, a1,a2,order, ...]
    }

    private func encodeToDisk() throws -> Data {
        let serializable = entries.map { entry -> SerializableEntry in
            let atomData: [Float]? = entry.atoms.isEmpty ? nil : entry.atoms.flatMap { [$0.position.x, $0.position.y, $0.position.z] }
            let atomElements: [Int]? = entry.atoms.isEmpty ? nil : entry.atoms.map { $0.element.rawValue }
            let atomNames: [String]? = entry.atoms.isEmpty ? nil : entry.atoms.map(\.name)
            let bondData: [Int]? = entry.bonds.isEmpty ? nil : entry.bonds.flatMap { [$0.atomIndex1, $0.atomIndex2, $0.order == .single ? 1 : $0.order == .double ? 2 : $0.order == .triple ? 3 : 4] }

            return SerializableEntry(
                name: entry.name, smiles: entry.smiles, isPrepared: entry.isPrepared,
                conformerCount: entry.conformerCount,
                mw: entry.descriptors?.molecularWeight, logP: entry.descriptors?.logP,
                tpsa: entry.descriptors?.tpsa, hbd: entry.descriptors?.hbd, hba: entry.descriptors?.hba,
                rotBonds: entry.descriptors?.rotatableBonds, rings: entry.descriptors?.rings,
                aromaticRings: entry.descriptors?.aromaticRings, heavyAtomCount: entry.descriptors?.heavyAtomCount,
                fractionCSP3: entry.descriptors?.fractionCSP3, lipinski: entry.descriptors?.lipinski,
                veber: entry.descriptors?.veber,
                atomData: atomData, atomElements: atomElements, atomNames: atomNames, bondData: bondData
            )
        }
        return try JSONEncoder().encode(serializable)
    }

    private func decodeFromDisk(_ data: Data) throws {
        let decoded = try JSONDecoder().decode([SerializableEntry].self, from: data)
        // Temporarily suppress auto-save while bulk-loading
        let loaded: [LigandEntry] = decoded.map { s in
            var atoms: [Atom] = []
            if let ad = s.atomData, let ae = s.atomElements, let an = s.atomNames {
                for i in stride(from: 0, to: ad.count - 2, by: 3) {
                    let idx = i / 3
                    guard idx < ae.count, idx < an.count else { break }
                    atoms.append(Atom(
                        id: idx, element: Element(rawValue: ae[idx]) ?? .C,
                        position: SIMD3(ad[i], ad[i+1], ad[i+2]),
                        name: an[idx], residueName: "LIG", residueSeq: 1, chainID: "L",
                        charge: 0, formalCharge: 0, isHetAtom: true
                    ))
                }
            }
            var bonds: [Bond] = []
            if let bd = s.bondData {
                for i in stride(from: 0, to: bd.count - 2, by: 3) {
                    let order: BondOrder = switch bd[i+2] { case 2: .double; case 3: .triple; case 4: .aromatic; default: .single }
                    bonds.append(Bond(id: bonds.count, atomIndex1: bd[i], atomIndex2: bd[i+1], order: order))
                }
            }
            let desc: LigandDescriptors? = s.mw != nil ? LigandDescriptors(
                molecularWeight: s.mw!, exactMW: s.mw!, logP: s.logP ?? 0, tpsa: s.tpsa ?? 0,
                hbd: s.hbd ?? 0, hba: s.hba ?? 0, rotatableBonds: s.rotBonds ?? 0,
                rings: s.rings ?? 0, aromaticRings: s.aromaticRings ?? 0,
                heavyAtomCount: s.heavyAtomCount ?? 0, fractionCSP3: s.fractionCSP3 ?? 0,
                lipinski: s.lipinski ?? false, veber: s.veber ?? false
            ) : nil
            return LigandEntry(name: s.name, smiles: s.smiles, atoms: atoms, bonds: bonds,
                              descriptors: desc, isPrepared: s.isPrepared,
                              conformerCount: s.conformerCount ?? 0)
        }
        entries = loaded
    }
}
