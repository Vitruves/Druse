import Foundation
import AppKit
import UniformTypeIdentifiers
import simd

/// A tautomer or protomer variant of a ligand.
struct MolecularVariant: Identifiable, Sendable {
    let id: UUID
    var smiles: String
    var atoms: [Atom]
    var bonds: [Bond]
    var relativeEnergy: Double     // kcal/mol (MMFF energy)
    var kind: VariantKind          // .tautomer or .protomer
    var label: String              // human-readable description
    var isPrepared: Bool
    var conformerCount: Int

    init(smiles: String, atoms: [Atom] = [], bonds: [Bond] = [],
         relativeEnergy: Double = 0, kind: VariantKind, label: String,
         isPrepared: Bool = false, conformerCount: Int = 0) {
        self.id = UUID()
        self.smiles = smiles
        self.atoms = atoms
        self.bonds = bonds
        self.relativeEnergy = relativeEnergy
        self.kind = kind
        self.label = label
        self.isPrepared = isPrepared
        self.conformerCount = conformerCount
    }
}

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

    // Experimental binding affinity data (optional)
    var ki: Float?       // Ki in nM
    var pKi: Float?      // -log10(Ki in M)
    var ic50: Float?     // IC50 in nM

    // Tautomer and protomer variants
    var variants: [MolecularVariant] = []

    /// Computed pKi from whatever affinity data is available (pKi > Ki > IC50).
    var effectivePKi: Float? {
        if let pk = pKi { return pk }
        if let k = ki, k > 0 { return -log10(k * 1e-9) }
        if let ic = ic50, ic > 0 { return -log10(ic * 1e-9) }
        return nil
    }

    init(name: String, smiles: String, atoms: [Atom] = [], bonds: [Bond] = [],
         descriptors: LigandDescriptors? = nil, isPrepared: Bool = false,
         conformerCount: Int = 0, ki: Float? = nil, pKi: Float? = nil, ic50: Float? = nil,
         variants: [MolecularVariant] = []) {
        self.id = UUID()
        self.name = name
        self.smiles = smiles
        self.atoms = atoms
        self.bonds = bonds
        self.descriptors = descriptors
        self.isPrepared = isPrepared
        self.preparationDate = isPrepared ? Date() : nil
        self.conformerCount = conformerCount
        self.ki = ki
        self.pKi = pKi
        self.ic50 = ic50
        self.variants = variants
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
            let headerCols = rows[0].components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            detectedSmilesCol = headerCols.firstIndex(where: { $0.contains("smiles") || $0.contains("molecule") || $0.contains("structure") }) ?? 0
            detectedNameCol = nameColumn ?? headerCols.firstIndex(where: { $0.contains("name") || $0.contains("id") || $0.contains("title") }) ?? (detectedSmilesCol == 0 ? 1 : 0)

            // Detect affinity columns
            kiCol = headerCols.firstIndex(where: { $0 == "ki" || $0 == "ki_nm" || $0 == "ki (nm)" })
            pKiCol = headerCols.firstIndex(where: { $0 == "pki" || $0 == "p_ki" || $0 == "pki_value" })
            ic50Col = headerCols.firstIndex(where: { $0 == "ic50" || $0 == "ic50_nm" || $0 == "ic50 (nm)" })
        } else {
            // No header: auto-detect by checking which column looks like SMILES
            let firstRow = dataRows.first.map { $0.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) } } ?? []
            detectedSmilesCol = firstRow.firstIndex(where: { Self.looksLikeSMILES($0) }) ?? 0
            detectedNameCol = nameColumn ?? (detectedSmilesCol == 0 ? 1 : 0)
        }

        let hasAffinity = kiCol != nil || pKiCol != nil || ic50Col != nil

        let baseIndex = entries.count
        let lines = dataRows.enumerated().compactMap { offset, row -> (smiles: String, name: String)? in
            let cols = row.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            guard detectedSmilesCol < cols.count else { return nil }
            let smi = cols[detectedSmilesCol]
            let name = detectedNameCol < cols.count ? cols[detectedNameCol] : "Mol_\(baseIndex + offset + 1)"
            return (smi, name)
        }

        // Parse affinity data per row before importing SMILES
        var affinityMap: [String: (ki: Float?, pKi: Float?, ic50: Float?)] = [:]
        if hasAffinity {
            for (offset, row) in dataRows.enumerated() {
                let cols = row.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                guard detectedSmilesCol < cols.count else { continue }
                let name: String
                if detectedNameCol < cols.count {
                    name = cols[detectedNameCol]
                } else {
                    name = "Mol_\(baseIndex + offset + 1)"
                }
                let ki = kiCol.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
                let pKi = pKiCol.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
                let ic50 = ic50Col.flatMap { $0 < cols.count ? Float(cols[$0]) : nil }
                if ki != nil || pKi != nil || ic50 != nil {
                    affinityMap[name] = (ki: ki, pKi: pKi, ic50: ic50)
                }
            }
        }

        importSMILES(lines, prepare: prepare)

        // Apply affinity data to imported entries
        if !affinityMap.isEmpty {
            for i in (entries.count - lines.count)..<entries.count {
                guard i >= 0, i < entries.count else { continue }
                if let aff = affinityMap[entries[i].name] {
                    entries[i].ki = aff.ki
                    entries[i].pKi = aff.pKi
                    entries[i].ic50 = aff.ic50
                }
            }
            let affCount = affinityMap.count
            ActivityLog.shared.info("Imported affinity data for \(affCount) ligands (Ki/pKi/IC50)", category: .molecule)
        }
    }

    /// Heuristic: does this string look like a SMILES notation?
    static func looksLikeSMILES(_ s: String) -> Bool {
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

    /// Encode the entire database to Data (for project save).
    func encodeToData() -> Data {
        (try? encodeToDisk()) ?? Data()
    }

    /// Decode database from Data (for project load). Replaces current entries.
    func decodeFromData(_ data: Data) {
        try? decodeFromDisk(data)
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
        // Experimental binding affinity
        let ki: Float?       // nM
        let pKi: Float?
        let ic50: Float?     // nM
        // Atom coordinates stored as flat [x,y,z,x,y,z,...] for compactness
        let atomData: [Float]?
        let atomElements: [Int]?
        let atomNames: [String]?
        let bondData: [Int]?  // [a1,a2,order, a1,a2,order, ...]
        // Tautomer/protomer variants
        let variants: [SerializableVariant]?
    }

    private static func encodeAtoms(_ atoms: [Atom]) -> (data: [Float]?, elements: [Int]?, names: [String]?) {
        guard !atoms.isEmpty else { return (nil, nil, nil) }
        return (
            atoms.flatMap { [$0.position.x, $0.position.y, $0.position.z] },
            atoms.map { $0.element.rawValue },
            atoms.map(\.name)
        )
    }

    private static func encodeBonds(_ bonds: [Bond]) -> [Int]? {
        guard !bonds.isEmpty else { return nil }
        return bonds.flatMap { [$0.atomIndex1, $0.atomIndex2, $0.order == .single ? 1 : $0.order == .double ? 2 : $0.order == .triple ? 3 : 4] }
    }

    private static func decodeAtoms(data: [Float]?, elements: [Int]?, names: [String]?) -> [Atom] {
        guard let ad = data, let ae = elements, let an = names else { return [] }
        var atoms: [Atom] = []
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

    private func encodeToDisk() throws -> Data {
        let serializable = entries.map { entry -> SerializableEntry in
            let (atomData, atomElements, atomNames) = Self.encodeAtoms(entry.atoms)
            let bondData = Self.encodeBonds(entry.bonds)

            let serializedVariants: [SerializableVariant]? = entry.variants.isEmpty ? nil : entry.variants.map { v in
                let (vad, vae, van) = Self.encodeAtoms(v.atoms)
                return SerializableVariant(
                    smiles: v.smiles, relativeEnergy: v.relativeEnergy,
                    kind: v.kind.rawValue, label: v.label,
                    isPrepared: v.isPrepared, conformerCount: v.conformerCount,
                    atomData: vad, atomElements: vae, atomNames: van,
                    bondData: Self.encodeBonds(v.bonds)
                )
            }

            return SerializableEntry(
                name: entry.name, smiles: entry.smiles, isPrepared: entry.isPrepared,
                conformerCount: entry.conformerCount,
                mw: entry.descriptors?.molecularWeight, logP: entry.descriptors?.logP,
                tpsa: entry.descriptors?.tpsa, hbd: entry.descriptors?.hbd, hba: entry.descriptors?.hba,
                rotBonds: entry.descriptors?.rotatableBonds, rings: entry.descriptors?.rings,
                aromaticRings: entry.descriptors?.aromaticRings, heavyAtomCount: entry.descriptors?.heavyAtomCount,
                fractionCSP3: entry.descriptors?.fractionCSP3, lipinski: entry.descriptors?.lipinski,
                veber: entry.descriptors?.veber,
                ki: entry.ki, pKi: entry.pKi, ic50: entry.ic50,
                atomData: atomData, atomElements: atomElements, atomNames: atomNames, bondData: bondData,
                variants: serializedVariants
            )
        }
        return try JSONEncoder().encode(serializable)
    }

    private func decodeFromDisk(_ data: Data) throws {
        let decoded = try JSONDecoder().decode([SerializableEntry].self, from: data)
        let loaded: [LigandEntry] = decoded.map { s in
            let atoms = Self.decodeAtoms(data: s.atomData, elements: s.atomElements, names: s.atomNames)
            let bonds = Self.decodeBonds(data: s.bondData)
            let desc: LigandDescriptors? = s.mw != nil ? LigandDescriptors(
                molecularWeight: s.mw!, exactMW: s.mw!, logP: s.logP ?? 0, tpsa: s.tpsa ?? 0,
                hbd: s.hbd ?? 0, hba: s.hba ?? 0, rotatableBonds: s.rotBonds ?? 0,
                rings: s.rings ?? 0, aromaticRings: s.aromaticRings ?? 0,
                heavyAtomCount: s.heavyAtomCount ?? 0, fractionCSP3: s.fractionCSP3 ?? 0,
                lipinski: s.lipinski ?? false, veber: s.veber ?? false
            ) : nil
            let variants: [MolecularVariant] = (s.variants ?? []).map { sv in
                MolecularVariant(
                    smiles: sv.smiles,
                    atoms: Self.decodeAtoms(data: sv.atomData, elements: sv.atomElements, names: sv.atomNames),
                    bonds: Self.decodeBonds(data: sv.bondData),
                    relativeEnergy: sv.relativeEnergy,
                    kind: VariantKind(rawValue: sv.kind) ?? .tautomer,
                    label: sv.label,
                    isPrepared: sv.isPrepared,
                    conformerCount: sv.conformerCount
                )
            }
            return LigandEntry(name: s.name, smiles: s.smiles, atoms: atoms, bonds: bonds,
                              descriptors: desc, isPrepared: s.isPrepared,
                              conformerCount: s.conformerCount ?? 0,
                              ki: s.ki, pKi: s.pKi, ic50: s.ic50,
                              variants: variants)
        }
        entries = loaded
    }
}
