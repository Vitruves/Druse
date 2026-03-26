import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - pKa Method

/// Controls how ionizable-site pKa values are determined during Populate & Prepare.
enum PKaMethod: String, CaseIterable {
    /// Use the built-in SMARTS lookup table (fast, ~ms per molecule).
    case table = "Table"
    /// Compute pKa via GFN2-xTB single-point energies on protonated/deprotonated forms (slow, ~seconds per site).
    case gfn2 = "GFN2-xTB"

    var description: String {
        switch self {
        case .table: return "Fast lookup table (~345 SMARTS rules)"
        case .gfn2: return "GFN2-xTB quantum chemistry (accurate but slow)"
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
    @State var expandedEntryIDs: Set<UUID> = []

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

    // Sorting
    enum SortField: String { case name, smiles, mw, logP, hbd, hba, tpsa, rotB, atoms }
    @State var sortField: SortField? = nil
    @State var sortAscending: Bool = true

    // Detail panel: selected entry for inspection
    @State var inspectedEntry: LigandEntry?

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

    // Variant (tautomer/protomer) state
    @State var isGeneratingVariants = false
    @State var variantPH: Double = 7.4
    @State var variantMaxTautomers: Int = 10
    @State var variantMaxProtomers: Int = 8
    @State var variantEnergyCutoff: Double = 10.0
    @State var variantPkaThreshold: Double = 2.0
    @State var variantMinPopulation: Double = 1.0  // minimum Boltzmann population % to keep a form
    @State var pkaMethod: PKaMethod = .table
    @State var selectedVariantID: UUID?
    @State var conformerBudgetPerVariant: Int = 20
    @State var selectedFormIndex: Int = 0
    @State var selectedFormConformerIndex: Int = 0

    // 2D/3D preview toggle
    @State var show3DPreview: Bool = false
    @State var miniRenderer: Renderer? = nil

    struct ConformerEntry: Identifiable {
        let id: Int
        let molecule: MoleculeData
        let energy: Double
    }

    // A single row in the flat table: one chemical form (or one raw molecule if no forms)
    struct FormRow: Identifiable {
        let id: UUID              // form.id or entry.id
        let entryID: UUID         // parent LigandEntry.id
        let formIndex: Int?       // index into entry.forms[], nil = raw molecule
        let parentName: String
        let form: ChemicalForm?   // nil for raw/unprepared molecules
        let entry: LigandEntry

        var name: String {
            if let form { return parentName + (form.kind == .parent ? "" : "_\(form.label)") }
            return parentName
        }
        var smiles: String { form?.smiles ?? entry.originalSMILES }
        var kind: ChemicalFormKind? { form?.kind }
        var population: Double { form?.boltzmannWeight ?? 0 }
        var relativeEnergy: Double { form?.relativeEnergy ?? 0 }
        var conformerCount: Int { form?.conformerCount ?? 0 }
        var atoms: [Atom] { form?.atoms ?? entry.atoms }
        var bonds: [Bond] { form?.bonds ?? entry.bonds }
        var isPrepared: Bool { form != nil ? !atoms.isEmpty : entry.isPrepared }
        var descriptors: LigandDescriptors? { entry.descriptors }
    }

    // Type filter for the table
    enum FormTypeFilter: String, CaseIterable {
        case all = "All"
        case parents = "Parents"
        case tautomers = "Tautomers"
        case protomers = "Protomers"
    }
    @State var formTypeFilter: FormTypeFilter = .all

    var db: LigandDatabase { viewModel.ligandDB }

    /// Flat list of rows: one per chemical form (or one per raw entry if no forms).
    var flatRows: [FormRow] {
        var rows: [FormRow] = []
        for entry in db.topLevelEntries {
            // Search filter on parent name/SMILES
            if !searchText.isEmpty {
                let match = entry.name.localizedCaseInsensitiveContains(searchText) ||
                    entry.originalSMILES.localizedCaseInsensitiveContains(searchText)
                if !match { continue }
            }
            if lipinskiFilter, entry.descriptors?.lipinski != true { continue }

            if entry.forms.isEmpty {
                // Raw molecule — no forms yet
                rows.append(FormRow(id: entry.id, entryID: entry.id, formIndex: nil,
                                    parentName: entry.name, form: nil, entry: entry))
            } else {
                for (idx, form) in entry.forms.enumerated() {
                    // Type filter
                    switch formTypeFilter {
                    case .all: break
                    case .parents: if form.kind != .parent { continue }
                    case .tautomers: if form.kind != .tautomer && form.kind != .tautomerProtomer { continue }
                    case .protomers: if form.kind != .protomer && form.kind != .tautomerProtomer { continue }
                    }
                    rows.append(FormRow(id: form.id, entryID: entry.id, formIndex: idx,
                                        parentName: entry.name, form: form, entry: entry))
                }
            }
        }

        if let field = sortField {
            rows.sort { a, b in
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
        return rows
    }

    // Tab for the bottom options panel
    @State var bottomTab: BottomTab = .properties
    enum BottomTab: String, CaseIterable {
        case properties = "Properties"
        case populateAndPrepare = "Populate & Prepare"
        case ensemble = "Ensemble"
    }

    // 2D structure preview (RDKit SVG depiction)
    @State var ligand2DCoords: RDKitBridge.Coords2D?
    @State var ligand2DSVG: String?
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
                    if selectedIDs.isEmpty, let first = flatRows.first {
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
        .onChange(of: selectedIDs) { _, newIDs in
            if newIDs.count == 1, let id = newIDs.first {
                // Find the FormRow for this selection
                if let row = flatRows.first(where: { $0.id == id }) {
                    inspectedEntry = row.entry
                    conformers = []
                    selectedConformerIndex = 0
                    selectedFormConformerIndex = 0
                    selectedVariantID = nil

                    if let fi = row.formIndex {
                        selectedFormIndex = fi
                        loadFormConformers(entry: row.entry, formIndex: fi)
                        compute2DPreview(smiles: row.smiles)
                    } else {
                        selectedFormIndex = 0
                        compute2DPreview(smiles: row.smiles)
                    }
                    if show3DPreview, !row.atoms.isEmpty {
                        updateMiniRenderer(atoms: row.atoms, bonds: row.bonds)
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

            // Populate & Prepare — the core action: protonate, minimize, charges,
            // enumerate tautomers/protomers/conformers → produce docking-ready ligands
            if isBatchProcessing {
                Button(action: { cancelPopulateAndPrepare() }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .controlSize(.small)
                .foregroundStyle(.orange)
                .help("Cancel Populate & Prepare")
            } else {
                Button(action: {
                    bottomTab = .populateAndPrepare
                    let selected = db.entries.filter { selectedIDs.contains($0.id) && $0.parentID == nil }
                    if !selected.isEmpty {
                        runPopulateAndPrepare(entries: selected)
                    } else if let entry = inspectedEntry, entry.parentID == nil {
                        runPopulateAndPrepare(entries: [entry])
                    }
                }) {
                    Label("Populate & Prepare", systemImage: "wand.and.stars")
                }
                .controlSize(.small)
                .disabled(selectedIDs.isEmpty || isProcessing)
                .help("Full pipeline: add polar H → MMFF94 minimize → charges → enumerate tautomers/protomers → conformers")
            }

            Button(action: { deleteSelected() }) {
                Label("Delete", systemImage: "trash")
            }
            .controlSize(.small)
            .foregroundStyle(.red)
            .disabled(selectedIDs.isEmpty)
            .help("Delete selected molecules and their variants")

            Divider().frame(height: 16)

            // Use for docking — sends each selected form as a docking candidate
            Button(action: { useSelectedForDocking() }) {
                let nPrepared = flatRows.filter { selectedIDs.contains($0.id) && $0.isPrepared }.count
                Label(nPrepared > 1
                      ? "Dock \(nPrepared) Forms"
                      : nPrepared == 1 ? "Use for Docking" : "No Prepared Forms",
                      systemImage: "arrow.right.circle")
            }
            .controlSize(.small)
            .disabled(selectedIDs.isEmpty)
            .help("Send selected chemical forms to docking")

            Spacer()

            // Type filter
            Picker("", selection: $formTypeFilter) {
                ForEach(FormTypeFilter.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .controlSize(.mini)
            .frame(width: 220)

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

            // Select all / none
            Button(action: { toggleSelectAll() }) {
                Image(systemName: selectedIDs.count == flatRows.count && !flatRows.isEmpty
                      ? "checkmark.square.fill" : "square")
                    .font(.callout)
            }
            .buttonStyle(.plain)
            .help("Select all / none")
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

            // Table header
            HStack(spacing: 0) {
                headerCell("", width: 24, field: nil)           // checkbox
                headerCell("Type", width: 28, field: nil)       // kind badge
                headerCell("Name", width: 140, field: .name)
                headerCell("SMILES", width: nil, field: .smiles)
                headerCell("Pop%", width: 42, field: nil, alignment: .trailing)
                headerCell("ΔE", width: 42, field: nil, alignment: .trailing)
                headerCell("Confs", width: 36, field: nil, alignment: .trailing)
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

            // Table body — flat: one row per chemical form
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(flatRows) { row in
                        formTableRow(row)
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

    // MARK: - Form Table Row (flat table — one row per chemical form)

    @ViewBuilder
    private func formTableRow(_ row: FormRow) -> some View {
        let isSelected = selectedIDs.contains(row.id)
        let isInspected = (inspectedEntry?.id == row.entryID) &&
            (row.formIndex == nil || selectedFormIndex == row.formIndex)

        HStack(spacing: 0) {
            // Checkbox
            Button {
                if selectedIDs.contains(row.id) { selectedIDs.remove(row.id) }
                else { selectedIDs.insert(row.id) }
            } label: {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .font(.subheadline)
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
            }
            .buttonStyle(.plain)
            .frame(width: 24)

            // Type badge
            if let kind = row.kind {
                let kindColor: Color = switch kind {
                case .parent: .green
                case .tautomer: .cyan
                case .protomer: .orange
                case .tautomerProtomer: .purple
                }
                Text(kind.symbol)
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(width: 18, height: 14)
                    .background(RoundedRectangle(cornerRadius: 4).fill(kindColor))
                    .frame(width: 28)
            } else {
                Text("—").font(.caption).foregroundStyle(.secondary).frame(width: 28)
            }

            // Name
            Text(row.name)
                .font(.footnote)
                .lineLimit(1)
                .frame(width: 140, alignment: .leading)

            // SMILES
            Text(row.smiles)
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(row.smiles)

            // Population %
            if row.population > 0 {
                Text(String(format: "%.0f%%", row.population * 100))
                    .font(.footnote.monospaced())
                    .foregroundStyle(row.population > 0.3 ? .green : row.population > 0.1 ? .yellow : .secondary)
                    .frame(width: 42, alignment: .trailing)
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 42, alignment: .trailing)
            }

            // Relative energy
            if row.form != nil {
                if row.relativeEnergy < 0.01 {
                    Text("best").font(.caption.weight(.medium)).foregroundStyle(.green).frame(width: 42, alignment: .trailing)
                } else {
                    Text(String(format: "+%.1f", row.relativeEnergy))
                        .font(.caption.monospaced()).foregroundStyle(.orange).frame(width: 42, alignment: .trailing)
                }
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 42, alignment: .trailing)
            }

            // Conformer count
            Text(row.conformerCount > 0 ? "\(row.conformerCount)" : "—")
                .font(.footnote.monospaced())
                .foregroundStyle(row.conformerCount > 0 ? .primary : .tertiary)
                .frame(width: 36, alignment: .trailing)

            // Descriptors
            if let d = row.descriptors {
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
            Text(row.atoms.isEmpty ? "—" : "\(row.atoms.count)")
                .font(.footnote.monospaced())
                .foregroundStyle(row.atoms.isEmpty ? .tertiary : .primary)
                .frame(width: 38, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isInspected ? Color.accentColor.opacity(0.2) :
                     isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture {
            handleRowClick(row.id, shiftKey: NSEvent.modifierFlags.contains(.shift),
                          cmdKey: NSEvent.modifierFlags.contains(.command))
        }
        .contextMenu {
            Button("Use for Docking") { useFormForDocking(row) }
                .disabled(row.atoms.isEmpty)
            if row.formIndex != nil {
                Button("Remove Form") { deleteFormRow(row) }
            } else {
                Button("Delete", role: .destructive) { deleteEntryByRow(row) }
            }
        }
    }

    @ViewBuilder
    private func tableRow(_ entry: LigandEntry) -> some View {
        let isSelected = selectedIDs.contains(entry.id)
        let isActive = viewModel.molecules.ligand?.name == entry.name
        let isInspected = inspectedEntry?.id == entry.id

        HStack(spacing: 0) {
            // Disclosure triangle for form expansion
            if entry.forms.count > 1 {
                Button {
                    if expandedEntryIDs.contains(entry.id) {
                        expandedEntryIDs.remove(entry.id)
                    } else {
                        expandedEntryIDs.insert(entry.id)
                    }
                } label: {
                    Image(systemName: expandedEntryIDs.contains(entry.id) ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .frame(width: 14)
            } else {
                Spacer().frame(width: 14)
            }

            // Checkbox — toggles selection without needing Cmd key
            Button {
                if selectedIDs.contains(entry.id) {
                    selectedIDs.remove(entry.id)
                } else {
                    selectedIDs.insert(entry.id)
                }
            } label: {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .font(.subheadline)
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
            }
            .buttonStyle(.plain)
            .frame(width: 24)

            // Name
            HStack(spacing: 4) {
                if isActive {
                    Circle().fill(.green).frame(width: 5, height: 5)
                }
                Text(entry.name)
                    .font(.footnote.weight(isActive ? .semibold : .regular))
                    .lineLimit(1)
            }
            .frame(width: 140, alignment: .leading)

            // SMILES
            Text(entry.smiles)
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(entry.smiles)

            // Descriptors
            if let d = entry.descriptors {
                Text(String(format: "%.0f", d.molecularWeight))
                    .font(.footnote.monospaced())
                    .frame(width: 60, alignment: .trailing)
                Text(String(format: "%.1f", d.logP))
                    .font(.footnote.monospaced())
                    .frame(width: 50, alignment: .trailing)
                Text("\(d.hbd)")
                    .font(.footnote.monospaced())
                    .frame(width: 36, alignment: .trailing)
                Text("\(d.hba)")
                    .font(.footnote.monospaced())
                    .frame(width: 36, alignment: .trailing)
                Text(String(format: "%.0f", d.tpsa))
                    .font(.footnote.monospaced())
                    .frame(width: 50, alignment: .trailing)
                Text("\(d.rotatableBonds)")
                    .font(.footnote.monospaced())
                    .frame(width: 36, alignment: .trailing)
                Image(systemName: d.lipinski ? "checkmark" : "xmark")
                    .font(.footnote)
                    .foregroundStyle(d.lipinski ? .green : .red.opacity(0.7))
                    .frame(width: 30)
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 60, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 50, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 36, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 36, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 50, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 36, alignment: .trailing)
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 30, alignment: .center)
            }

            // Affinity data (Ki, pKi, IC50)
            if let ki = entry.ki {
                Text(String(format: "%.1f", ki))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.purple)
                    .frame(width: 55, alignment: .trailing)
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 55, alignment: .trailing)
            }
            if let pKi = entry.pKi ?? entry.effectivePKi {
                Text(String(format: "%.2f", pKi))
                    .font(.footnote.monospaced())
                    .foregroundStyle(entry.pKi != nil ? .purple : .purple.opacity(0.5))
                    .frame(width: 45, alignment: .trailing)
                    .help(entry.pKi != nil ? "Stored pKi" : "Computed from \(entry.ki != nil ? "Ki" : "IC50")")
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 45, alignment: .trailing)
            }
            if let ic50 = entry.ic50 {
                Text(String(format: "%.1f", ic50))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.purple)
                    .frame(width: 55, alignment: .trailing)
            } else {
                Text("—").font(.footnote).foregroundStyle(.secondary).frame(width: 55, alignment: .trailing)
            }

            // Prepared status
            Image(systemName: entry.isPrepared ? "checkmark.circle.fill" : "circle")
                .font(.footnote)
                .foregroundColor(entry.isPrepared ? .green : .gray)
                .frame(width: 36)

            // Atom count
            Text(entry.atoms.isEmpty ? "—" : "\(entry.atoms.count)")
                .font(.footnote.monospaced())
                .foregroundStyle(entry.atoms.isEmpty ? .tertiary : .primary)
                .frame(width: 44, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isInspected ? Color.accentColor.opacity(0.2) :
                     isSelected ? Color.accentColor.opacity(0.1) :
                     isActive ? Color.green.opacity(0.05) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture {
            handleRowClick(entry.id, shiftKey: NSEvent.modifierFlags.contains(.shift),
                          cmdKey: NSEvent.modifierFlags.contains(.command))
        }
        .contextMenu {
            Button("Inspect") {
                inspectedEntry = entry
                conformers = []
                selectedConformerIndex = 0
            }
            Divider()
            Button("Use for Docking") { useEntryForDocking(entry) }
                .disabled(entry.atoms.isEmpty)
            Button("Prepare") { prepareSingleEntry(entry) }
                .disabled(entry.smiles.isEmpty)
            Divider()
            Button("Delete", role: .destructive) {
                if inspectedEntry?.id == entry.id { inspectedEntry = nil }
                db.removeWithChildren(id: entry.id)
                selectedIDs.remove(entry.id)
            }
        }
    }

    // MARK: - Chemical Form Sub-Row

    @ViewBuilder
    private func chemicalFormSubRow(entry: LigandEntry, form: ChemicalForm, index: Int) -> some View {
        let kindColor: Color = switch form.kind {
        case .parent: .green
        case .tautomer: .cyan
        case .protomer: .orange
        case .tautomerProtomer: .purple
        }
        let isBest = index == entry.bestFormIndex

        HStack(spacing: 4) {
            Spacer().frame(width: 38) // indent past disclosure + checkbox

            // Kind badge
            Text(form.kind.symbol)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white)
                .frame(width: 18, height: 16)
                .background(RoundedRectangle(cornerRadius: 4).fill(kindColor))

            // Form label
            Text(form.label)
                .font(.footnote.weight(isBest ? .semibold : .regular))
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)

            // SMILES (shows difference from parent)
            Text(form.smiles)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Population %
            Text(form.populationString)
                .font(.caption.monospaced().weight(.medium))
                .foregroundStyle(form.boltzmannWeight > 0.3 ? .green :
                                 form.boltzmannWeight > 0.1 ? .yellow : .secondary)
                .frame(width: 42, alignment: .trailing)

            // Relative energy
            if form.relativeEnergy < 0.01 {
                Text("best")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.green)
                    .frame(width: 50, alignment: .trailing)
            } else {
                Text(String(format: "+%.1f kcal", form.relativeEnergy))
                    .font(.caption.monospaced())
                    .foregroundStyle(.orange)
                    .frame(width: 50, alignment: .trailing)
            }

            // Conformer count
            Text("\(form.conformerCount) conf")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)

            // Use for docking button
            if !form.atoms.isEmpty {
                Button {
                    let mol = Molecule(
                        name: "\(entry.name)_\(form.label)",
                        atoms: form.atoms,
                        bonds: form.bonds,
                        title: form.smiles,
                        smiles: form.smiles
                    )
                    viewModel.setLigandForDocking(mol, entryID: entry.id, forms: entry.forms)
                } label: {
                    Image(systemName: "arrow.right.circle")
                        .font(.footnote)
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)
                .help("Dock \(form.label)")
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(selectedFormIndex == index && inspectedEntry?.id == entry.id
                     ? Color.accentColor.opacity(0.15)
                     : isBest ? Color.green.opacity(0.04) : Color.secondary.opacity(0.03))
        .contentShape(Rectangle())
        .onTapGesture {
            // Select this form: update preview to show this form's SMILES and conformers
            selectedFormIndex = index
            selectedFormConformerIndex = 0
            inspectedEntry = entry
            selectedIDs = [entry.id]
            // Update 2D preview with this form's SMILES
            compute2DPreview(smiles: form.smiles)
            // Load this form's conformers into the carousel
            loadFormConformers(entry: entry, formIndex: index)
            // Update 3D if active
            if show3DPreview, !form.atoms.isEmpty {
                updateMiniRenderer(atoms: form.atoms, bonds: form.bonds)
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
        let rows = flatRows
        let nMolecules = db.topLevelEntries.count
        let nForms = rows.count
        let nVariants = rows.filter { $0.kind != nil && $0.kind != .parent }.count
        let nConformers = rows.reduce(0) { $0 + $1.conformerCount }
        let nSelected = selectedIDs.count
        HStack(spacing: 16) {
            HStack(spacing: 4) {
                Text("\(nMolecules) molecules")
                    .font(.footnote)
                if nForms > nMolecules || nConformers > 0 {
                    Image(systemName: "arrow.right").font(.caption).foregroundStyle(.secondary)
                    Text("\(nForms) forms")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.green)
                    if nVariants > 0 {
                        Text("(\(nVariants) variants)")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
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
            let allIDs = flatRows.map(\.id)
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
        if selectedIDs.count == flatRows.count && !flatRows.isEmpty {
            selectedIDs.removeAll()
        } else {
            selectedIDs = Set(flatRows.map(\.id))
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
                    let (mol, desc, _) = await Task.detached { @Sendable in
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
            if let inspID = inspectedEntry?.id {
                inspectedEntry = db.entries.first { $0.id == inspID }
            }
            if successCount > 0 {
                viewModel.log.success("Prepared \(successCount)/\(toProcess.count) ligands", category: .molecule)
            } else {
                viewModel.log.error("All \(toProcess.count) preparations failed — check SMILES validity", category: .molecule)
            }
        }
    }

    func deleteSelected() {
        let rows = flatRows.filter { selectedIDs.contains($0.id) }
        guard !rows.isEmpty else { return }

        // Group by entry to handle form-level vs entry-level deletion
        let byEntry = Dictionary(grouping: rows, by: \.entryID)
        for (entryID, entryRows) in byEntry {
            guard var entry = db.entries.first(where: { $0.id == entryID }) else { continue }

            if entry.forms.isEmpty || entryRows.contains(where: { $0.formIndex == nil }) {
                // Raw molecule or entire entry selected — remove whole entry
                db.removeWithChildren(id: entryID)
            } else {
                // Remove specific forms
                let formIndicesToRemove = Set(entryRows.compactMap(\.formIndex))
                entry.forms = entry.forms.enumerated().filter { !formIndicesToRemove.contains($0.offset) }.map(\.element)

                if entry.forms.isEmpty {
                    db.removeWithChildren(id: entryID)
                } else {
                    // Re-normalize Boltzmann weights
                    let wSum = entry.forms.reduce(0.0) { $0 + $1.boltzmannWeight }
                    if wSum > 0 { for i in entry.forms.indices { entry.forms[i].boltzmannWeight /= wSum } }
                    entry.bestFormIndex = 0
                    entry.atoms = entry.forms[0].atoms
                    entry.bonds = entry.forms[0].bonds
                    entry.conformerCount = entry.forms.reduce(0) { $0 + $1.conformerCount }
                    db.update(entry)
                }
            }
        }

        if inspectedEntry.map({ byEntry[$0.id] != nil }) == true {
            inspectedEntry = nil
            conformers = []
        }
        let count = selectedIDs.count
        selectedIDs.removeAll()
        viewModel.log.info("Deleted \(count) forms", category: .molecule)
    }

    func useSelectedForDocking() {
        // Each selected row IS a docking candidate (one form = one unique SMILES)
        let selectedRows = flatRows.filter { selectedIDs.contains($0.id) && $0.isPrepared }
        guard !selectedRows.isEmpty else {
            viewModel.log.warn("No prepared forms in selection — prepare them first", category: .dock)
            return
        }

        var entries: [LigandEntry] = []
        for row in selectedRows {
            var e = row.entry
            e.atoms = row.atoms
            e.bonds = row.bonds
            e.originalSMILES = row.smiles
            e.name = row.name
            entries.append(e)
        }

        if entries.count == 1 {
            useEntryForDocking(entries[0])
        } else {
            viewModel.queueLigandsForBatchDocking(entries)
        }
        viewModel.log.success("Queued \(entries.count) forms for docking", category: .dock)
    }

    // MARK: - Form-level helpers

    func useFormForDocking(_ row: FormRow) {
        guard !row.atoms.isEmpty else {
            viewModel.log.warn("\(row.name) has no 3D coordinates", category: .molecule)
            return
        }
        let mol = Molecule(name: row.name, atoms: row.atoms, bonds: row.bonds,
                           title: row.smiles, smiles: row.smiles)
        let forms = db.entries.first(where: { $0.id == row.entryID })?.forms ?? []
        viewModel.setLigandForDocking(mol, entryID: row.entryID, forms: forms)
    }

    func deleteFormRow(_ row: FormRow) {
        guard let fi = row.formIndex,
              var entry = db.entries.first(where: { $0.id == row.entryID }),
              fi < entry.forms.count, entry.forms.count > 1 else { return }
        entry.forms.remove(at: fi)
        let wSum = entry.forms.reduce(0.0) { $0 + $1.boltzmannWeight }
        if wSum > 0 { for i in entry.forms.indices { entry.forms[i].boltzmannWeight /= wSum } }
        entry.bestFormIndex = 0
        entry.atoms = entry.forms[0].atoms
        entry.bonds = entry.forms[0].bonds
        entry.conformerCount = entry.forms.reduce(0) { $0 + $1.conformerCount }
        db.update(entry)
        selectedIDs.remove(row.id)
    }

    func deleteEntryByRow(_ row: FormRow) {
        if inspectedEntry?.id == row.entryID { inspectedEntry = nil; conformers = [] }
        db.removeWithChildren(id: row.entryID)
        selectedIDs.remove(row.id)
    }

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
                let (mol, desc, _) = await Task.detached { @Sendable in
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

    func loadFormConformers(entry: LigandEntry, formIndex: Int) {
        guard formIndex < entry.forms.count else {
            conformers = []
            return
        }
        let form = entry.forms[formIndex]
        conformers = form.conformers.enumerated().map { idx, conf in
            ConformerEntry(id: idx,
                           molecule: MoleculeData(name: "\(entry.name)_\(form.label)_conf\(idx)",
                                                  title: form.smiles,
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
        viewModel.setLigandForDocking(mol, entryID: entry.id, forms: entry.forms)
    }

    private func addAndPrepare(smiles: String, name: String) {
        db.addFromSMILES(smiles, name: name)
        if let entry = db.entries.last, entry.smiles == smiles {
            prepareSingleEntry(entry)
        }
    }

    // MARK: - Populate & Prepare (Unified Pipeline)

    /// Full preparation pipeline for one or more entries:
    /// 1. Add polar hydrogens → MMFF94 minimization → 3D coordinates
    /// 2. Gasteiger charge computation
    /// 3. Enumerate tautomers and protomers at target pH
    /// 4. Generate conformers for each chemical form
    /// 5. Filter by Boltzmann population and energy cutoff
    /// 6. Store all forms with their conformer ensembles on each entry
    ///
    /// Runs molecules in parallel via TaskGroup. Cancellable via `cancelPopulateAndPrepare()`.
    func runPopulateAndPrepare(entries: [LigandEntry]) {
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
        let minPop = variantMinPopulation / 100.0  // convert % to fraction
        let useGFN2 = pkaMethod == .gfn2

        populateTask = Task {
            var totalForms = 0
            var totalConformers = 0
            var completedCount = 0

            // GFN2-xTB: limit to 2 (GPU-heavy per site); Table: scale with CPU cores
            let maxConcurrency = useGFN2 ? 2 : max(ProcessInfo.processInfo.activeProcessorCount - 1, 1)
            await withTaskGroup(of: (LigandEntry, RDKitBridge.EnsembleResult, [LigandpKaPredictor.SitePKa])?.self) { group in
                var enqueued = 0
                for entry in toProcess {
                    // Throttle: wait for a result before adding more tasks past the limit
                    if enqueued >= maxConcurrency {
                        if let item = await group.next() {
                            if let (entry, result, pkaResults) = item {
                                completedCount += 1
                                batchProgress = (completedCount, toProcess.count)
                                processingMessage = "Populating \(completedCount)/\(toProcess.count): \(entry.name)"
                                processPopulateResult(entry: entry, result: result, pkaResults: pkaResults,
                                                      minPop: minPop, totalForms: &totalForms,
                                                      totalConformers: &totalConformers)
                            }
                        }
                    }

                    guard !Task.isCancelled else { break }

                    let smi = !entry.originalSMILES.isEmpty ? entry.originalSMILES : entry.name
                    let nm = entry.name
                    group.addTask { [entry] in
                        guard !Task.isCancelled else { return nil }

                        // Run the heavy C++ work off the main actor
                        let (result, pkaResults) = await Task.detached(priority: .userInitiated) {
                            // Phase 1: pKa — either GFN2-xTB (slow, accurate) or table lookup (fast)
                            var pkaResults: [LigandpKaPredictor.SitePKa] = []
                            var computedPKa: [Double] = []
                            if useGFN2 {
                                pkaResults = await LigandpKaPredictor.predictpKa(smiles: smi)
                                computedPKa = LigandpKaPredictor.pKaArray(from: pkaResults)
                            }

                            // Phase 2: Ensemble generation (C++ RDKit — CPU-heavy)
                            let result = RDKitBridge.prepareEnsembleWithPKa(
                                smiles: smi, name: nm,
                                pH: ph, pkaThreshold: pkaThreshold,
                                maxTautomers: maxTauto, maxProtomers: maxProto,
                                energyCutoff: energyCutoff, conformersPerForm: confsPerForm,
                                sitePKa: computedPKa
                            )
                            return (result, pkaResults)
                        }.value
                        return (entry, result, pkaResults)
                    }
                    enqueued += 1
                }

                // Drain remaining results
                for await item in group {
                    guard !Task.isCancelled else { break }
                    guard let (entry, result, pkaResults) = item else { continue }
                    completedCount += 1
                    batchProgress = (completedCount, toProcess.count)
                    processingMessage = "Populating \(completedCount)/\(toProcess.count): \(entry.name)"
                    processPopulateResult(entry: entry, result: result, pkaResults: pkaResults,
                                          minPop: minPop, totalForms: &totalForms,
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
                    "Populate & Prepare: \(toProcess.count) molecules → \(totalForms) forms, \(totalConformers) conformers",
                    category: .molecule
                )
            }
        }
    }

    /// Process a single completed Populate & Prepare result — shared by throttle drain and final drain.
    private func processPopulateResult(
        entry: LigandEntry, result: RDKitBridge.EnsembleResult,
        pkaResults: [LigandpKaPredictor.SitePKa],
        minPop: Double, totalForms: inout Int, totalConformers: inout Int
    ) {
        // Log pKa predictions
        if !pkaResults.isEmpty {
            let pkaLog = pkaResults.map { p in
                "\(p.groupName)[\(p.atomIdx)]: \(p.converged ? String(format: "%.1f", p.computedPKa) : "fallback \(String(format: "%.1f", p.defaultPKa))")"
            }.joined(separator: ", ")
            viewModel.log.info("pKa(\(entry.name)): \(pkaLog)", category: .molecule)
        }

        guard result.success, !result.members.isEmpty else {
            viewModel.log.error("Failed to populate \(entry.name): \(result.errorMessage)", category: .molecule)
            return
        }

        // Convert flat ensemble members → hierarchical ChemicalForm array
        var forms = RDKitBridge.ensembleResultToForms(result)
        guard !forms.isEmpty else { return }

        // Filter forms by minimum Boltzmann population (always keep parent/best form)
        if minPop > 0 && forms.count > 1 {
            let preCount = forms.count
            forms = forms.enumerated().filter { idx, form in
                idx == 0 || form.boltzmannWeight >= minPop
            }.map(\.element)

            let weightSum = forms.reduce(0.0) { $0 + $1.boltzmannWeight }
            if weightSum > 0 {
                for i in forms.indices {
                    forms[i].boltzmannWeight /= weightSum
                }
            }

            if forms.count < preCount {
                viewModel.log.info("\(entry.name): filtered \(preCount) → \(forms.count) forms (>\(String(format: "%.1f", minPop * 100))% population)", category: .molecule)
            }
        }

        // Update the entry with forms and best conformer
        var updated = entry
        updated.forms = forms
        updated.bestFormIndex = 0
        updated.atoms = forms[0].atoms
        updated.bonds = forms[0].bonds
        updated.originalSMILES = entry.originalSMILES
        updated.isPrepared = true
        updated.preparationDate = Date()
        updated.conformerCount = forms.reduce(0) { $0 + $1.conformerCount }

        if let desc = RDKitBridge.computeDescriptors(smiles: forms[0].smiles) {
            updated.descriptors = desc
        }

        db.batchMutate { entries in
            entries.removeAll { $0.parentID == entry.id }
        }
        db.update(updated)

        totalForms += forms.count
        totalConformers += updated.conformerCount
        if forms.count > 1 {
            expandedEntryIDs.insert(entry.id)
        }
    }

    func cancelPopulateAndPrepare() {
        populateTask?.cancel()
        populateTask = nil
        isProcessing = false
        isBatchProcessing = false
        processingMessage = ""
        viewModel.log.info("Populate & Prepare cancelled", category: .molecule)
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

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField()
        field.placeholderString = placeholder
        field.font = font
        field.isBordered = true
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.lineBreakMode = .byTruncatingTail
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
        Coordinator(text: $text)
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var text: Binding<String>

        init(text: Binding<String>) {
            self.text = text
        }

        func controlTextDidChange(_ obj: Notification) {
            guard let field = obj.object as? NSTextField else { return }
            text.wrappedValue = field.stringValue
        }
    }
}
