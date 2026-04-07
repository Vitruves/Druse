import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - pKa Method

/// Controls how ionizable-site pKa values are determined during Populate & Prepare.
enum PKaMethod: String, CaseIterable {
    /// GNN-predicted pKa via Metal GPU (fast + accurate, ~0.1ms per molecule).
    case gnn = "GNN"
    /// Use the built-in SMARTS lookup table (fast, ~ms per molecule).
    case table = "Table"

    var description: String {
        switch self {
        case .gnn: return "GNN-predicted pKa (Metal GPU, MAE ~1.0)"
        case .table: return "SMARTS lookup table (~230 rules)"
        }
    }
}

// MARK: - Ligand Database Manager Window

struct LigandDatabaseWindow: View {
    @Environment(AppViewModel.self) var viewModel
    @Environment(\.dismiss) var dismiss

    // Selection state (shift+click multi-select)
    @State var selectedIDs: Set<UUID> = []
    @State var lastClickedID: UUID?

    // Batch operation state
    @State var batchProgress: (current: Int, total: Int) = (0, 0)
    @State var isBatchProcessing: Bool = false

    // Search / filter
    @State var searchText: String = ""
    @State var lipinskiFilter: Bool = false

    // Inline SMILES entry
    @State var showSMILESEntry: Bool = false
    @State var smilesInput: String = ""
    @State var nameInput: String = ""

    // Processing state
    @State var isProcessing: Bool = false
    @State var processingMessage: String = ""

    // Draw structure
    @State var showChemDrawer: Bool = false
    @State var editingEntryID: UUID? = nil

    // Scaffold-based analog generator
    @State var showScaffoldAnalogSheet: Bool = false

    // Sorting
    enum SortField: String { case name, smiles, mw, logP, hbd, hba, tpsa, rotB, atoms }
    @State var sortField: SortField? = nil
    @State var sortAscending: Bool = true

    // Detail panel: selected entry for inspection
    @State var inspectedEntry: LigandEntry?

    // Inline rename
    @State var renamingEntryID: UUID? = nil
    @State var renamingText: String = ""

    // Preparation options (used in detail panel)
    @State var prepAddHydrogens = true
    @State var prepMinimize = true
    @State var prepComputeCharges = true
    @State var prepNumConformers = 50

    // Import mapping sheet
    @State var importPreview: ImportPreview?
    @State var showImportMapping: Bool = false

    // Conformer state
    @State var conformers: [ConformerEntry] = []
    @State var selectedConformerIndex: Int = 0
    @State var isGeneratingConformers = false

    // Cancellable populate task
    @State var populateTask: Task<Void, Never>?

    // Preparation options
    @State var isGeneratingVariants = false
    @State var variantPH: Double = 7.4
    @State var variantMaxTautomers: Int = 10
    @State var variantMaxProtomers: Int = 8
    @State var variantEnergyCutoff: Double = 10.0
    @State var variantPkaThreshold: Double = 2.0
    @State var variantMinPopulation: Double = 1.0
    @State var pkaMethod: PKaMethod = .gnn

    // 2D/3D preview toggle
    @State var show3DPreview: Bool = false
    @State var miniRenderer: Renderer? = nil

    struct ConformerEntry: Identifiable {
        let id: Int
        let molecule: MoleculeData
        let energy: Double
    }

    // Prepared-only filter
    @State var showOnlyPrepared: Bool = false

    var db: LigandDatabase { viewModel.ligandDB }

    /// Filtered entries (search + Lipinski + prepared-only filters, sorted).
    var filteredEntries: [LigandEntry] {
        var entries = db.entries
        if !searchText.isEmpty {
            entries = entries.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.originalSMILES.localizedCaseInsensitiveContains(searchText)
            }
        }
        if lipinskiFilter {
            entries = entries.filter { $0.descriptors?.lipinski == true }
        }
        if showOnlyPrepared {
            entries = entries.filter { $0.isPrepared }
        }
        if let field = sortField {
            entries.sort { a, b in
                let result: Bool
                switch field {
                case .name:  result = a.name.localizedCompare(b.name) == .orderedAscending
                case .smiles: result = a.smiles < b.smiles
                case .mw:    result = (a.descriptors?.molecularWeight ?? 0) < (b.descriptors?.molecularWeight ?? 0)
                case .logP:  result = (a.descriptors?.logP ?? 0) < (b.descriptors?.logP ?? 0)
                case .hbd:   result = (a.descriptors?.hbd ?? 0) < (b.descriptors?.hbd ?? 0)
                case .hba:   result = (a.descriptors?.hba ?? 0) < (b.descriptors?.hba ?? 0)
                case .tpsa:  result = (a.descriptors?.tpsa ?? 0) < (b.descriptors?.tpsa ?? 0)
                case .rotB:  result = (a.descriptors?.rotatableBonds ?? 0) < (b.descriptors?.rotatableBonds ?? 0)
                case .atoms: result = a.atoms.count < b.atoms.count
                }
                return sortAscending ? result : !result
            }
        }
        return entries
    }

    // Tab for the bottom options panel
    @State var bottomTab: BottomTab = .properties
    enum BottomTab: String, CaseIterable {
        case properties = "Properties"
        case populateAndPrepare = "Populate & Prepare"
    }

    // 2D structure preview (RDKit SVG depiction, pre-rasterized to NSImage)
    @State var ligand2DCoords: RDKitBridge.Coords2D?
    @State var ligand2DImage: NSImage?
    @State var isComputing2D = false

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            toolbar
            Divider()

            // Main content: VSplitView { table (top) | preview+options (bottom) }
            if db.entries.isEmpty {
                emptyState
            } else {
                VSplitView {
                    // TOP: Ligand table with variants
                    tableView
                        .frame(minHeight: 200)

                    // BOTTOM: always visible — shows detail, batch panel, or placeholder
                    Group {
                        if selectedIDs.count > 1 {
                            ScrollView {
                                batchActionPanel
                            }
                        } else if let entry = inspectedEntry {
                            bottomDetailArea(entry)
                                .id("\(entry.id)-\(entry.isPrepared)-\(entry.conformerCount)-\(entry.originalSMILES.hashValue)")
                        } else {
                            VStack {
                                Spacer()
                                Text("Select a ligand to view details")
                                    .font(.body)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }
                    .frame(minHeight: 250)
                }
                .onAppear {
                    // Auto-select first row when window opens
                    if selectedIDs.isEmpty, let first = filteredEntries.first {
                        selectedIDs = [first.id]
                    }
                }
            }

            Divider()
            statusBar
        }
        .background(.background)
        .sheet(isPresented: $showImportMapping) {
            ImportMappingSheet(preview: $importPreview) { finalPreview in
                performMappedImport(finalPreview)
            }
        }
        .sheet(isPresented: $showScaffoldAnalogSheet) {
            ScaffoldAnalogSheet()
                .environment(viewModel)
        }
        .sheet(isPresented: $showChemDrawer) {
            let editEntry = editingEntryID.flatMap { id in db.entries.first { $0.id == id } }
            ChemDrawerSheet(
                title: editEntry != nil ? "Edit Structure — \(editEntry!.name)" : "Draw Molecule",
                initialSmiles: editEntry?.smiles ?? "",
                mode: .molecule
            ) { smiles, _ in
                guard !smiles.isEmpty else { return }
                if let id = editingEntryID,
                   let idx = db.entries.firstIndex(where: { $0.id == id }) {
                    // Update existing entry
                    db.entries[idx].originalSMILES = smiles
                    db.entries[idx].isPrepared = false
                    db.entries[idx].atoms = []
                    db.entries[idx].bonds = []
                    db.entries[idx].conformers = []
                    db.entries[idx].conformerCount = 0
                    db.entries[idx].descriptors = nil
                    prepareSingleEntry(db.entries[idx])
                    if inspectedEntry?.id == id { inspectedEntry = db.entries[idx] }
                } else {
                    // New molecule
                    let name = "Drawn_\(db.count + 1)"
                    addAndPrepare(smiles: smiles, name: name)
                }
                editingEntryID = nil
            }
        }
        .onChange(of: selectedIDs) { _, newIDs in
            if newIDs.count == 1, let id = newIDs.first {
                if let entry = db.entries.first(where: { $0.id == id }) {
                    inspectedEntry = entry
                    selectedConformerIndex = 0
                    conformers = []
                    compute2DPreview(smiles: entry.originalSMILES)
                    if show3DPreview, !entry.atoms.isEmpty {
                        updateMiniRenderer(atoms: entry.atoms, bonds: entry.bonds)
                    }
                }
            }
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 8) {
            // Add SMILES (adds raw molecule)
            Button(action: { showSMILESEntry.toggle() }) {
                Label("Add Molecule", systemImage: "plus")
            }
            .controlSize(.small)
            .help("Add a molecule from SMILES string")

            Button(action: { editingEntryID = nil; showChemDrawer = true }) {
                Label("Draw", systemImage: "pencil.and.outline")
            }
            .controlSize(.small)
            .help("Draw a molecule structure")

            Button(action: { showScaffoldAnalogSheet = true }) {
                Label("Generate", systemImage: "atom")
            }
            .controlSize(.small)
            .help("Generate analogs from a drawn scaffold (R-group decoration or whole-molecule transforms)")

            // Import buttons (import raw molecules)
            Menu {
                Button("Import .smi file") { openImportWithMapping(.smi) }
                Button("Import .csv file") { openImportWithMapping(.csv) }
                Button("Import .sdf file") { openImportWithMapping(.sdf) }
            } label: {
                Label("Import", systemImage: "square.and.arrow.down")
            }
            .controlSize(.small)
            .help("Import molecules from file")

            Divider().frame(height: 16)

            // Prepare — dominant form at target pH, docking-ready
            if isBatchProcessing {
                Button(action: { cancelPopulateAndPrepare() }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .controlSize(.small)
                .foregroundStyle(.orange)
                .help("Cancel preparation")
            } else {
                Button(action: {
                    bottomTab = .populateAndPrepare
                    let selected = db.entries.filter { selectedIDs.contains($0.id) }
                    if !selected.isEmpty {
                        runPopulateAndPrepare(entries: selected)
                    } else if let entry = inspectedEntry {
                        runPopulateAndPrepare(entries: [entry])
                    }
                }) {
                    Label({
                        let n = db.entries.filter { selectedIDs.contains($0.id) }.count
                        return n > 1 ? "Prepare \(n) Selected" : "Prepare Selected"
                    }(), systemImage: "wand.and.stars")
                }
                .controlSize(.small)
                .disabled(selectedIDs.isEmpty || isProcessing)
                .help("Prepare dominant form: add H → minimize → charges → conformers")

                // Enumerate — expand to all tautomers/protomers as separate rows
                Button(action: {
                    bottomTab = .populateAndPrepare
                    let selected = db.entries.filter { selectedIDs.contains($0.id) && !$0.isEnumerated }
                    if !selected.isEmpty {
                        runEnumerate(entries: selected)
                    } else if let entry = inspectedEntry, !entry.isEnumerated {
                        runEnumerate(entries: [entry])
                    }
                }) {
                    Label({
                        let n = db.entries.filter { selectedIDs.contains($0.id) && !$0.isEnumerated }.count
                        return n > 1 ? "Enumerate \(n) Selected" : "Enumerate Selected"
                    }(), systemImage: "arrow.triangle.branch")
                }
                .controlSize(.small)
                .disabled(selectedIDs.isEmpty || isProcessing)
                .help("Enumerate all tautomers & protomers as separate entries")
            }

            Button(action: { deleteSelected() }) {
                Label("Delete", systemImage: "trash")
            }
            .controlSize(.small)
            .foregroundStyle(.red)
            .disabled(selectedIDs.isEmpty)
            .help("Delete selected molecules and their variants")

            Divider().frame(height: 16)

            // Use for docking
            Button(action: { useSelectedForDocking() }) {
                let nPrepared = db.entries.filter { selectedIDs.contains($0.id) && $0.isPrepared }.count
                Label(nPrepared > 1
                      ? "Send \(nPrepared) to Docking"
                      : nPrepared == 1 ? "Send to Docking" : "No Prepared",
                      systemImage: "arrow.right.circle")
            }
            .controlSize(.small)
            .disabled(selectedIDs.isEmpty)
            .help("Send selected entries to docking")

            Spacer()

            // Search
            TextField("Search...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 140)
                .font(.subheadline)

            Toggle("Lipinski", isOn: $lipinskiFilter)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .font(.footnote)

            // Save/Load/Export
            Menu {
                Button("Save Database") {
                    db.save()
                    viewModel.log.success("Saved \(db.count) entries", category: .molecule)
                }
                Button("Load Database") {
                    db.load()
                }
                Divider()
                Button("Export All as SDF") { db.exportSDF() }
                Divider()
                Button("Clear All", role: .destructive) {
                    db.removeAll()
                    selectedIDs.removeAll()
                    inspectedEntry = nil
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .controlSize(.small)

        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - SMILES Entry (inline, shown below toolbar)

    @ViewBuilder
    var smilesEntryBar: some View {
        if showSMILESEntry {
            HStack(spacing: 8) {
                PastableTextField(text: $smilesInput, placeholder: "SMILES", font: .monospacedSystemFont(ofSize: 11, weight: .regular))
                    .frame(height: 22)

                TextField("Name", text: $nameInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.subheadline)
                    .frame(width: 120)

                Button("Add") {
                    let name = nameInput.isEmpty ? "Ligand_\(db.count + 1)" : nameInput
                    db.addFromSMILES(smilesInput, name: name)
                    smilesInput = ""
                    nameInput = ""
                }
                .controlSize(.small)
                .disabled(smilesInput.isEmpty)

                Button("Add & Prepare") {
                    let name = nameInput.isEmpty ? "Ligand_\(db.count + 1)" : nameInput
                    addAndPrepare(smiles: smilesInput, name: name)
                    smilesInput = ""
                    nameInput = ""
                }
                .controlSize(.small)
                .disabled(smilesInput.isEmpty)

                Button(action: { showSMILESEntry = false }) {
                    Image(systemName: "xmark")
                        .font(.footnote)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(nsColor: .controlBackgroundColor))
        }
    }

    // MARK: - Table

    private var tableView: some View {
        VStack(spacing: 0) {
            smilesEntryBar

            // Processing indicator
            if isProcessing {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(processingMessage)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
                .frame(maxWidth: .infinity)
                .background(Color.accentColor.opacity(0.05))
            }

            // Table header — fixed columns, same for every row
            HStack(spacing: 0) {
                // Select all / none checkbox
                Button(action: { toggleSelectAll() }) {
                    Image(systemName: selectedIDs.count == filteredEntries.count && !filteredEntries.isEmpty
                          ? "checkmark.square.fill" : "square")
                        .font(.subheadline)
                        .foregroundStyle(selectedIDs.count == filteredEntries.count && !filteredEntries.isEmpty
                                         ? Color.accentColor : Color.secondary)
                }
                .buttonStyle(.plain)
                .frame(width: 24)
                .help("Select all / none")
                headerCell("Name", width: 170, field: .name)
                headerCell("SMILES", width: nil, field: .smiles)
                headerCell("Pop%", width: 48, field: nil, alignment: .trailing)
                headerCell("ΔE", width: 44, field: nil, alignment: .trailing)
                headerCell("Conf", width: 38, field: nil, alignment: .trailing)
                headerCell("MW", width: 52, field: .mw, alignment: .trailing)
                headerCell("LogP", width: 42, field: .logP, alignment: .trailing)
                headerCell("HBD", width: 30, field: .hbd, alignment: .trailing)
                headerCell("HBA", width: 30, field: .hba, alignment: .trailing)
                headerCell("TPSA", width: 42, field: .tpsa, alignment: .trailing)
                headerCell("RotB", width: 30, field: .rotB, alignment: .trailing)
                headerCell("Lip.", width: 26, field: nil, alignment: .center)
                headerCell("Atoms", width: 38, field: .atoms, alignment: .trailing)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // Table body — flat: one row per entry, no hierarchy
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(filteredEntries) { entry in
                        tableRow(entry)
                        Divider().opacity(0.3)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func headerCell(_ title: String, width: CGFloat?, field: SortField?, alignment: Alignment = .leading) -> some View {
        let isSorted = field != nil && sortField == field
        Group {
            if let field {
                Button(action: {
                    if sortField == field {
                        sortAscending.toggle()
                    } else {
                        sortField = field
                        sortAscending = true
                    }
                }) {
                    HStack(spacing: 2) {
                        Text(title)
                        if isSorted {
                            Image(systemName: sortAscending ? "chevron.up" : "chevron.down")
                                .font(.footnote)
                        }
                    }
                }
                .buttonStyle(.plain)
            } else {
                Text(title)
            }
        }
        .font(.footnote.weight(isSorted ? .bold : .semibold))
        .foregroundStyle(isSorted ? .primary : .secondary)
        .frame(width: width, alignment: alignment)
        .frame(maxWidth: width == nil ? .infinity : nil, alignment: alignment)
    }

    // MARK: - Table Row (flat — one row per entry)

    @ViewBuilder
    private func tableRow(_ entry: LigandEntry) -> some View {
        let isSelected = selectedIDs.contains(entry.id)
        let isInspected = inspectedEntry?.id == entry.id
        let isActive = viewModel.molecules.ligand?.name == entry.name
        let isExpandedParent = db.hasEnumeratedChildren(entry)
        let kindColor: Color? = entry.formKind.map { kind in
            switch kind {
            case .parent: .green
            case .tautomer: .cyan
            case .protomer: .orange
            case .tautomerProtomer: .purple
            }
        }

        HStack(spacing: 0) {
            // Checkbox
            Button {
                if selectedIDs.contains(entry.id) { selectedIDs.remove(entry.id) }
                else { selectedIDs.insert(entry.id) }
            } label: {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .font(.subheadline)
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
            }
            .buttonStyle(.plain)
            .frame(width: 24)

            // Name (with optional form kind badge and active indicator)
            HStack(spacing: 4) {
                if isActive {
                    Circle().fill(.green).frame(width: 5, height: 5)
                }
                if entry.isEnumerated {
                    Text("↳")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                }
                if let kind = entry.formKind, let color = kindColor {
                    Text(kind.symbol)
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(width: 16, height: 14)
                        .background(RoundedRectangle(cornerRadius: 3).fill(color))
                }
                if renamingEntryID == entry.id {
                    TextField("Name", text: $renamingText, onCommit: {
                        commitRename(entryID: entry.id)
                    })
                    .textFieldStyle(.roundedBorder)
                    .font(.footnote)
                    .onExitCommand { renamingEntryID = nil }
                } else {
                    Text(entry.name)
                        .font(.footnote.weight(isActive ? .semibold : .regular))
                        .foregroundStyle(isExpandedParent ? .tertiary : .primary)
                        .lineLimit(1)
                        .onTapGesture(count: 2) {
                            renamingEntryID = entry.id
                            renamingText = entry.name
                        }
                }
            }
            .frame(width: 170, alignment: .leading)

            // SMILES
            Text(entry.originalSMILES)
                .font(.footnote.monospaced())
                .foregroundStyle(isExpandedParent ? .tertiary : .secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(entry.originalSMILES)

            // Pop% — nil = not enumerated (show —), value = Boltzmann weight
            if let pop = entry.populationWeight {
                Text(String(format: "%.1f%%", pop * 100))
                    .font(.footnote.monospaced().weight(.medium))
                    .foregroundStyle(pop > 0.3 ? .green : pop > 0.1 ? .yellow : .secondary)
                    .frame(width: 48, alignment: .trailing)
            } else {
                Text("—").font(.footnote).foregroundStyle(.tertiary).frame(width: 48, alignment: .trailing)
            }

            // ΔE — nil = not enumerated
            if let dE = entry.relativeEnergy {
                if dE < 0.01 {
                    Text("best").font(.caption.weight(.medium)).foregroundStyle(.green)
                        .frame(width: 44, alignment: .trailing)
                } else {
                    Text(String(format: "+%.1f", dE))
                        .font(.caption.monospaced()).foregroundStyle(.orange)
                        .frame(width: 44, alignment: .trailing)
                }
            } else {
                Text("—").font(.footnote).foregroundStyle(.tertiary).frame(width: 44, alignment: .trailing)
            }

            // Conf
            if entry.conformerCount > 0 {
                Text("\(entry.conformerCount)")
                    .font(.footnote.monospaced())
                    .frame(width: 38, alignment: .trailing)
            } else {
                Text("—").font(.footnote).foregroundStyle(.tertiary).frame(width: 38, alignment: .trailing)
            }

            // Descriptors (ADMET)
            if let d = entry.descriptors {
                Text(String(format: "%.0f", d.molecularWeight)).font(.footnote.monospaced()).frame(width: 52, alignment: .trailing)
                Text(String(format: "%.1f", d.logP)).font(.footnote.monospaced()).frame(width: 42, alignment: .trailing)
                Text("\(d.hbd)").font(.footnote.monospaced()).frame(width: 30, alignment: .trailing)
                Text("\(d.hba)").font(.footnote.monospaced()).frame(width: 30, alignment: .trailing)
                Text(String(format: "%.0f", d.tpsa)).font(.footnote.monospaced()).frame(width: 42, alignment: .trailing)
                Text("\(d.rotatableBonds)").font(.footnote.monospaced()).frame(width: 30, alignment: .trailing)
                Image(systemName: d.lipinski ? "checkmark" : "xmark")
                    .font(.footnote).foregroundStyle(d.lipinski ? .green : .red.opacity(0.7)).frame(width: 26)
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 52, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 42, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 30, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 30, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 42, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 30, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 26, alignment: .center)
            }

            // Atom count
            Text(entry.atoms.isEmpty ? "—" : "\(entry.atoms.count)")
                .font(.footnote.monospaced())
                .foregroundStyle(entry.atoms.isEmpty ? .tertiary : .primary)
                .frame(width: 38, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isInspected ? Color.accentColor.opacity(0.2) :
                     isSelected ? Color.accentColor.opacity(0.1) :
                     isActive ? Color.green.opacity(0.05) :
                     isExpandedParent ? Color.secondary.opacity(0.06) :
                     entry.isEnumerated ? Color.secondary.opacity(0.03) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture {
            handleRowClick(entry.id, shiftKey: NSEvent.modifierFlags.contains(.shift),
                          cmdKey: NSEvent.modifierFlags.contains(.command))
            inspectedEntry = entry
            selectedConformerIndex = 0
            // Defer conformer building — only needed when bottom panel is visible
            conformers = []
            compute2DPreview(smiles: entry.originalSMILES)
            if show3DPreview, !entry.atoms.isEmpty {
                updateMiniRenderer(atoms: entry.atoms, bonds: entry.bonds)
            }
        }
        .contextMenu {
            Button("Send to Docking") { useEntryForDocking(entry) }
                .disabled(entry.atoms.isEmpty || isExpandedParent)
            Button("Prepare") { prepareSingleEntry(entry) }
                .disabled(entry.smiles.isEmpty)
            Button("Edit Structure") {
                editingEntryID = entry.id
                showChemDrawer = true
            }
            .disabled(entry.smiles.isEmpty)
            Button("Rename") {
                renamingEntryID = entry.id
                renamingText = entry.name
            }
            Divider()
            if entry.isEnumerated, let pn = entry.parentName {
                Button("Delete All \(pn) Forms", role: .destructive) {
                    db.removeWithSiblings(parentName: pn)
                    if inspectedEntry?.parentName == pn { inspectedEntry = nil; conformers = [] }
                    selectedIDs.subtract(db.entries.filter { $0.parentName == pn }.map(\.id))
                }
            }
            Button("Delete", role: .destructive) {
                if inspectedEntry?.id == entry.id { inspectedEntry = nil; conformers = [] }
                db.removeWithChildren(id: entry.id)
                selectedIDs.remove(entry.id)
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "tray")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No ligands in database")
                .font(.title3.weight(.medium))
                .foregroundStyle(.secondary)
            Text("Add SMILES, or import .smi / .csv / .sdf files")
                .font(.callout)
                .foregroundStyle(.secondary)

            Button(action: { showSMILESEntry = true }) {
                Label("Add SMILES", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)

            smilesEntryBar
            Spacer()
        }
    }

    // MARK: - Status Bar

    @ViewBuilder
    private var statusBar: some View {
        let entries = filteredEntries
        let nTotal = entries.count
        let nPrepared = entries.filter(\.isPrepared).count
        let nEnumerated = entries.filter(\.isEnumerated).count
        let nConformers = entries.reduce(0) { $0 + $1.conformerCount }
        let nSelected = selectedIDs.count
        HStack(spacing: 16) {
            HStack(spacing: 4) {
                Text("\(nTotal) entries")
                    .font(.footnote)
                if nPrepared > 0 {
                    Text("(\(nPrepared) prepared)")
                        .font(.footnote)
                        .foregroundStyle(.green)
                }
                if nEnumerated > 0 {
                    Text("(\(nEnumerated) enumerated)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if nConformers > 0 {
                    Image(systemName: "arrow.right").font(.caption).foregroundStyle(.secondary)
                    Text("\(nConformers) conformers")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Text("Selected: \(nSelected)")
                .font(.footnote)
                .foregroundStyle(Color.accentColor)

            Spacer()

            if isBatchProcessing {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.mini)
                    Text("Processing \(batchProgress.current)/\(batchProgress.total)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button(action: { cancelPopulateAndPrepare() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Cancel Populate & Prepare")
                }
            }

            if let lig = viewModel.molecules.ligand {
                HStack(spacing: 4) {
                    Circle().fill(.green).frame(width: 5, height: 5)
                    Text("Active: \(lig.name)")
                        .font(.footnote.weight(.medium))
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Selection

    private func handleRowClick(_ id: UUID, shiftKey: Bool, cmdKey: Bool) {
        if shiftKey, let lastID = lastClickedID {
            let allIDs = filteredEntries.map(\.id)
            if let startIdx = allIDs.firstIndex(of: lastID),
               let endIdx = allIDs.firstIndex(of: id) {
                let range = min(startIdx, endIdx)...max(startIdx, endIdx)
                for i in range {
                    selectedIDs.insert(allIDs[i])
                }
            }
        } else if cmdKey {
            if selectedIDs.contains(id) {
                selectedIDs.remove(id)
            } else {
                selectedIDs.insert(id)
            }
        } else {
            selectedIDs = [id]
        }
        lastClickedID = id
    }

    private func toggleSelectAll() {
        let entries = filteredEntries
        if selectedIDs.count == entries.count && !entries.isEmpty {
            selectedIDs.removeAll()
        } else {
            selectedIDs = Set(entries.map(\.id))
        }
    }

    // MARK: - Actions

    private func prepareSelected() {
        let selected = db.entries.filter { selectedIDs.contains($0.id) }
        guard !selected.isEmpty else { return }

        let toProcess = selected.filter { !$0.smiles.isEmpty || !$0.name.isEmpty }
        guard !toProcess.isEmpty else { return }

        isProcessing = true
        processingMessage = "Preparing \(toProcess.count) ligands..."

        Task {
            var successCount = 0
            for (i, entry) in toProcess.enumerated() {
                processingMessage = "Preparing \(i+1)/\(toProcess.count): \(entry.name)"

                let candidates: [(smiles: String, name: String)]
                if !entry.smiles.isEmpty {
                    candidates = [
                        (entry.smiles, entry.name),
                        (entry.name, entry.smiles)
                    ]
                } else {
                    candidates = [(entry.name, "Ligand_\(i+1)")]
                }

                var prepared = false
                for candidate in candidates {
                    let smi = candidate.smiles
                    let nm = candidate.name
                    let ah = prepAddHydrogens, mn = prepMinimize, cc = prepComputeCharges
                    let (mol, desc, canonSMILES, _) = await Task.detached { @Sendable in
                        RDKitBridge.prepareLigand(smiles: smi, name: nm,
                                                  numConformers: 1, addHydrogens: ah,
                                                  minimize: mn, computeCharges: cc)
                    }.value

                    if let mol {
                        var updated = entry
                        if smi == entry.name && !entry.smiles.isEmpty {
                            updated.smiles = entry.name
                            updated.name = entry.smiles
                        }
                        updated.atoms = mol.atoms
                        updated.bonds = mol.bonds
                        updated.descriptors = desc
                        updated.isPrepared = true
                        updated.conformerCount = 1
                        if let canonSMILES { updated.originalSMILES = canonSMILES }
                        db.update(updated)
                        successCount += 1
                        prepared = true
                        break
                    }
                }

                if !prepared {
                    viewModel.log.error("Failed to prepare \(entry.name): invalid SMILES", category: .molecule)
                }
            }
            isProcessing = false
            processingMessage = ""
            // Refresh inspected entry if it was prepared
            if let inspID = inspectedEntry?.id,
               let refreshed = db.entries.first(where: { $0.id == inspID }) {
                inspectedEntry = refreshed
                compute2DPreview(smiles: refreshed.originalSMILES)
                if show3DPreview, !refreshed.atoms.isEmpty {
                    updateMiniRenderer(atoms: refreshed.atoms, bonds: refreshed.bonds)
                }
            }
            if successCount > 0 {
                viewModel.log.success("Prepared \(successCount)/\(toProcess.count) ligands", category: .molecule)
            } else {
                viewModel.log.error("All \(toProcess.count) preparations failed — check SMILES validity", category: .molecule)
            }
        }
    }

    func deleteSelected() {
        let toDelete = selectedIDs
        guard !toDelete.isEmpty else { return }

        if inspectedEntry.map({ toDelete.contains($0.id) }) == true {
            inspectedEntry = nil
            conformers = []
        }
        db.batchMutate { entries in
            entries.removeAll { toDelete.contains($0.id) }
        }
        let count = selectedIDs.count
        selectedIDs.removeAll()
        viewModel.log.info("Deleted \(count) entries", category: .molecule)
    }

    func useSelectedForDocking() {
        // Exclude parent entries that have enumerated children (would duplicate docking)
        let selected = db.entries.filter {
            selectedIDs.contains($0.id) && $0.isPrepared && !db.hasEnumeratedChildren($0)
        }
        guard !selected.isEmpty else {
            viewModel.log.warn("No prepared entries in selection — prepare them first", category: .dock)
            return
        }

        if selected.count == 1 {
            useEntryForDocking(selected[0])
        } else {
            viewModel.queueLigandsForBatchDocking(selected)
        }
        viewModel.log.success("Queued \(selected.count) entries for docking", category: .dock)
    }

    // (Form-level helpers removed — flat model has no forms within entries)

    func prepareSingleEntry(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty || !entry.name.isEmpty else { return }
        isProcessing = true
        processingMessage = "Preparing \(entry.name)..."

        Task {
            let candidates: [(smiles: String, name: String)]
            if !entry.smiles.isEmpty {
                candidates = [(entry.smiles, entry.name), (entry.name, entry.smiles)]
            } else {
                candidates = [(entry.name, "Ligand")]
            }

            let ah = prepAddHydrogens, mn = prepMinimize, cc = prepComputeCharges

            var prepared = false
            for candidate in candidates {
                let smi = candidate.smiles
                let nm = candidate.name
                let (mol, desc, _, _) = await Task.detached { @Sendable in
                    RDKitBridge.prepareLigand(smiles: smi, name: nm,
                                              numConformers: 1, addHydrogens: ah,
                                              minimize: mn, computeCharges: cc)
                }.value

                if let mol {
                    var updated = entry
                    if smi == entry.name && !entry.smiles.isEmpty {
                        updated.smiles = entry.name
                        updated.name = entry.smiles
                    }
                    updated.atoms = mol.atoms
                    updated.bonds = mol.bonds
                    updated.descriptors = desc
                    updated.isPrepared = true
                    updated.conformerCount = 1
                    db.update(updated)
                    viewModel.log.success("Prepared \(updated.name)", category: .molecule)
                    prepared = true
                    // Refresh inspected entry
                    if inspectedEntry?.id == entry.id {
                        inspectedEntry = db.entries.first { $0.id == entry.id }
                    }
                    break
                }
            }

            if !prepared {
                viewModel.log.error("Failed to prepare \(entry.name): invalid SMILES", category: .molecule)
            }
            isProcessing = false
            processingMessage = ""
        }
    }

    func loadConformers(entry: LigandEntry) {
        conformers = entry.conformers.enumerated().map { idx, conf in
            ConformerEntry(id: idx,
                           molecule: MoleculeData(name: "\(entry.name)_conf\(idx)",
                                                  title: entry.smiles,
                                                  atoms: conf.atoms, bonds: conf.bonds),
                           energy: conf.energy)
        }
        selectedConformerIndex = 0
    }

    func useEntryForDocking(_ entry: LigandEntry) {
        guard !entry.atoms.isEmpty else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare first", category: .molecule)
            return
        }
        let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds,
                           title: entry.smiles, smiles: entry.smiles)
        viewModel.setLigandForDocking(mol, entryID: entry.id)
    }

    private func commitRename(entryID: UUID) {
        let newName = renamingText.trimmingCharacters(in: .whitespacesAndNewlines)
        renamingEntryID = nil
        guard !newName.isEmpty,
              let idx = db.entries.firstIndex(where: { $0.id == entryID }),
              db.entries[idx].name != newName else { return }
        db.entries[idx].name = newName
        if inspectedEntry?.id == entryID {
            inspectedEntry?.name = newName
        }
    }

    private func addAndPrepare(smiles: String, name: String) {
        db.addFromSMILES(smiles, name: name)
        if let entry = db.entries.last, entry.smiles == smiles {
            prepareSingleEntry(entry)
        }
    }

    // MARK: - Prepare (Dominant Form)

    /// Prepare entries: add H → adjust protonation at target pH → minimize → charges → conformers.
    /// Uses the C++ ensemble pipeline with maxTautomers=1 so protonation is adjusted but
    /// tautomers are not enumerated. Keeps the dominant (most populated) protomer only.
    func runPopulateAndPrepare(entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.originalSMILES.isEmpty || !$0.name.isEmpty }
        guard !toProcess.isEmpty else { return }

        populateTask?.cancel()
        isProcessing = true
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        let ph = variantPH
        let confsPerForm = prepNumConformers
        let pkaThreshold = variantPkaThreshold
        let energyCutoff = variantEnergyCutoff
        let useGNN = pkaMethod == .gnn

        populateTask = Task {
            var successCount = 0
            var totalConformers = 0
            var completedCount = 0

            let maxConcurrency = max(ProcessInfo.processInfo.activeProcessorCount - 1, 1)
            await withTaskGroup(of: (LigandEntry, RDKitBridge.EnsembleResult)?.self) { group in
                var enqueued = 0
                for entry in toProcess {
                    if enqueued >= maxConcurrency {
                        if let item = await group.next(), let (entry, result) = item {
                            completedCount += 1
                            batchProgress = (completedCount, toProcess.count)
                            processingMessage = "Preparing \(completedCount)/\(toProcess.count): \(entry.name)"
                            let ok = processPrepareResult(entry: entry, result: result)
                            successCount += ok
                        }
                    }
                    guard !Task.isCancelled else { break }

                    let smi = !entry.originalSMILES.isEmpty ? entry.originalSMILES : entry.name
                    let nm = entry.name
                    group.addTask { [entry] in
                        guard !Task.isCancelled else { return nil }
                        let result = await Task.detached(priority: .userInitiated) {
                            let predictions = useGNN ? PKaGNNPredictor.predict(smiles: smi) : []
                            return RDKitBridge.prepareEnsembleWithSites(
                                smiles: smi, name: nm,
                                pH: ph, pkaThreshold: pkaThreshold,
                                maxTautomers: 1, maxProtomers: 1,
                                energyCutoff: energyCutoff, conformersPerForm: confsPerForm,
                                sites: predictions
                            )
                        }.value
                        return (entry, result)
                    }
                    enqueued += 1
                }

                for await item in group {
                    guard !Task.isCancelled else { break }
                    guard let (entry, result) = item else { continue }
                    completedCount += 1
                    batchProgress = (completedCount, toProcess.count)
                    processingMessage = "Preparing \(completedCount)/\(toProcess.count): \(entry.name)"
                    successCount += processPrepareResult(entry: entry, result: result)
                }
            }

            isProcessing = false
            isBatchProcessing = false
            processingMessage = ""
            populateTask = nil

            if let inspID = inspectedEntry?.id {
                inspectedEntry = db.entries.first { $0.id == inspID }
            }

            if !Task.isCancelled {
                viewModel.log.success("Prepared \(successCount)/\(toProcess.count) molecules", category: .molecule)
            }
        }
    }

    /// Process a single prepare result — update entry in-place with dominant form. Returns 1 on success.
    private func processPrepareResult(entry: LigandEntry, result: RDKitBridge.EnsembleResult) -> Int {
        guard result.success, !result.members.isEmpty else {
            var failed = entry
            failed.preparationError = result.errorMessage
            db.update(failed)
            viewModel.log.error("Failed to prepare \(entry.name): \(result.errorMessage)", category: .molecule)
            return 0
        }

        let forms = RDKitBridge.ensembleResultToForms(result)
        guard let bestForm = forms.first else { return 0 }

        var updated = entry
        updated.originalSMILES = bestForm.smiles
        updated.atoms = bestForm.atoms
        updated.bonds = bestForm.bonds
        updated.isPrepared = true
        updated.preparationDate = Date()
        updated.conformerCount = bestForm.conformerCount
        updated.conformers = bestForm.conformers
        updated.preparationError = nil
        if let desc = RDKitBridge.computeDescriptors(smiles: bestForm.smiles) {
            updated.descriptors = desc
        }
        db.update(updated)
        return 1
    }

    // MARK: - Enumerate (Full Tautomer/Protomer Expansion)

    /// Enumerate entries: run full tautomer + protomer expansion at target pH.
    /// Inserts N new rows after the original entry (which is kept as-is).
    func runEnumerate(entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.originalSMILES.isEmpty || !$0.name.isEmpty }
        guard !toProcess.isEmpty else { return }

        populateTask?.cancel()
        isProcessing = true
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        let ph = variantPH
        let maxTauto = variantMaxTautomers
        let maxProto = variantMaxProtomers
        let energyCutoff = variantEnergyCutoff
        let confsPerForm = prepNumConformers
        let pkaThreshold = variantPkaThreshold
        let minPop = variantMinPopulation / 100.0
        let useGNN = pkaMethod == .gnn

        populateTask = Task {
            var totalNewEntries = 0
            var totalConformers = 0
            var completedCount = 0

            let maxConcurrency = max(ProcessInfo.processInfo.activeProcessorCount - 1, 1)
            await withTaskGroup(of: (LigandEntry, RDKitBridge.EnsembleResult, [PKaGNNPredictor.SitePrediction])?.self) { group in
                var enqueued = 0
                for entry in toProcess {
                    if enqueued >= maxConcurrency {
                        if let item = await group.next() {
                            if let (entry, result, pkaResults) = item {
                                completedCount += 1
                                batchProgress = (completedCount, toProcess.count)
                                processingMessage = "Enumerating \(completedCount)/\(toProcess.count): \(entry.name)"
                                processEnumerateResult(entry: entry, result: result, pkaResults: pkaResults,
                                                       minPop: minPop,
                                                       totalNewEntries: &totalNewEntries,
                                                       totalConformers: &totalConformers)
                            }
                        }
                    }

                    guard !Task.isCancelled else { break }

                    let smi = !entry.originalSMILES.isEmpty ? entry.originalSMILES : entry.name
                    let nm = entry.name
                    group.addTask { [entry] in
                        guard !Task.isCancelled else { return nil }
                        let (result, pkaResults) = await Task.detached(priority: .userInitiated) {
                            let predictions = useGNN ? PKaGNNPredictor.predict(smiles: smi) : []
                            let result = RDKitBridge.prepareEnsembleWithSites(
                                smiles: smi, name: nm,
                                pH: ph, pkaThreshold: pkaThreshold,
                                maxTautomers: maxTauto, maxProtomers: maxProto,
                                energyCutoff: energyCutoff, conformersPerForm: confsPerForm,
                                sites: predictions
                            )
                            return (result, predictions)
                        }.value
                        return (entry, result, pkaResults)
                    }
                    enqueued += 1
                }

                for await item in group {
                    guard !Task.isCancelled else { break }
                    guard let (entry, result, pkaResults) = item else { continue }
                    completedCount += 1
                    batchProgress = (completedCount, toProcess.count)
                    processingMessage = "Enumerating \(completedCount)/\(toProcess.count): \(entry.name)"
                    processEnumerateResult(entry: entry, result: result, pkaResults: pkaResults,
                                           minPop: minPop,
                                           totalNewEntries: &totalNewEntries,
                                           totalConformers: &totalConformers)
                }
            }

            isProcessing = false
            isBatchProcessing = false
            processingMessage = ""
            populateTask = nil

            if let inspID = inspectedEntry?.id {
                inspectedEntry = db.entries.first { $0.id == inspID }
            }

            if !Task.isCancelled {
                viewModel.log.success(
                    "Enumerate: \(toProcess.count) molecules → \(totalNewEntries) new entries, \(totalConformers) conformers",
                    category: .molecule
                )
            }
        }
    }

    /// Process a single enumerate result: insert N new entries after the original (which is kept).
    private func processEnumerateResult(
        entry: LigandEntry, result: RDKitBridge.EnsembleResult,
        pkaResults: [PKaGNNPredictor.SitePrediction],
        minPop: Double,
        totalNewEntries: inout Int, totalConformers: inout Int
    ) {
        if !pkaResults.isEmpty {
            let pkaLog = pkaResults.map { p in
                "\(p.isAcid ? "acid" : "base")[\(p.atomIdx)]: \(String(format: "%.1f", p.pKa)) (p=\(String(format: "%.2f", p.ionizableProbability)))"
            }.joined(separator: ", ")
            viewModel.log.info("pKa(\(entry.name)): \(pkaLog)", category: .molecule)
        }

        guard result.success, !result.members.isEmpty else {
            viewModel.log.error("Failed to enumerate \(entry.name): \(result.errorMessage)", category: .molecule)
            return
        }

        var forms = RDKitBridge.ensembleResultToForms(result)
        guard !forms.isEmpty else { return }

        // Filter by minimum population (always keep best form)
        if minPop > 0 && forms.count > 1 {
            forms = forms.enumerated().filter { idx, form in
                idx == 0 || form.boltzmannWeight >= minPop
            }.map(\.element)
            let weightSum = forms.reduce(0.0) { $0 + $1.boltzmannWeight }
            if weightSum > 0 {
                for i in forms.indices { forms[i].boltzmannWeight /= weightSum }
            }
        }

        // Build new entries from forms (parent form keeps original name, variants get suffix)
        var newEntries: [LigandEntry] = []
        for form in forms {
            let formName = form.kind == .parent
                ? entry.name
                : "\(entry.name)_\(form.label)"
            var newEntry = LigandEntry(
                name: formName,
                smiles: form.smiles,
                atoms: form.atoms,
                bonds: form.bonds,
                isPrepared: true,
                conformerCount: form.conformerCount
            )
            newEntry.preparationDate = Date()
            newEntry.parentName = entry.name
            newEntry.populationWeight = form.boltzmannWeight
            newEntry.formKind = form.kind
            newEntry.relativeEnergy = form.relativeEnergy
            newEntry.conformers = form.conformers
            if let desc = RDKitBridge.computeDescriptors(smiles: form.smiles) {
                newEntry.descriptors = desc
            }
            newEntry.ki = entry.ki
            newEntry.pKi = entry.pKi
            newEntry.ic50 = entry.ic50
            newEntries.append(newEntry)
        }

        guard !newEntries.isEmpty else { return }

        // Remove any previous enumerated siblings of this entry, then insert new ones after it
        db.batchMutate { entries in
            entries.removeAll { $0.parentName == entry.name }
            if let idx = entries.firstIndex(where: { $0.id == entry.id }) {
                entries.insert(contentsOf: newEntries, at: idx + 1)
            } else {
                entries.append(contentsOf: newEntries)
            }
        }

        totalNewEntries += newEntries.count
        totalConformers += newEntries.reduce(0) { $0 + $1.conformerCount }
    }

    func cancelPopulateAndPrepare() {
        populateTask?.cancel()
        populateTask = nil
        isProcessing = false
        isBatchProcessing = false
        processingMessage = ""
        viewModel.log.info("Cancelled", category: .molecule)
    }
}

// MARK: - PastableTextField (NSViewRepresentable for reliable paste)

/// An `NSTextField` wrapper that reliably updates the binding on paste, drag-drop,
/// and programmatic text changes -- unlike SwiftUI's `TextField` which can miss
/// paste events until the next interaction.
struct PastableTextField: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String = ""
    var font: NSFont = .systemFont(ofSize: 12)
    var onSubmit: (() -> Void)?

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField()
        field.placeholderString = placeholder
        field.font = font
        field.isBordered = true
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.lineBreakMode = .byTruncatingTail
        field.cell?.isScrollable = false
        field.cell?.truncatesLastVisibleLine = true
        field.delegate = context.coordinator
        field.stringValue = text
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        if nsView.stringValue != text {
            nsView.stringValue = text
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text, onSubmit: onSubmit)
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var text: Binding<String>
        var onSubmit: (() -> Void)?

        init(text: Binding<String>, onSubmit: (() -> Void)?) {
            self.text = text
            self.onSubmit = onSubmit
        }

        func controlTextDidChange(_ obj: Notification) {
            guard let field = obj.object as? NSTextField else { return }
            text.wrappedValue = field.stringValue
        }

        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                onSubmit?()
                return true
            }
            return false
        }

    }
}
