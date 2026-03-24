import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Ligand Database Manager Window

struct LigandDatabaseWindow: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.dismiss) private var dismiss

    // Selection state (shift+click multi-select)
    @State private var selectedIDs: Set<UUID> = []
    @State private var lastClickedID: UUID?
    @State private var expandedEntryIDs: Set<UUID> = []

    // Batch operation state
    @State private var batchProgress: (current: Int, total: Int) = (0, 0)
    @State private var isBatchProcessing: Bool = false

    // Search / filter
    @State private var searchText: String = ""
    @State private var lipinskiFilter: Bool = false

    // Inline SMILES entry
    @State private var showSMILESEntry: Bool = false
    @State private var smilesInput: String = ""
    @State private var nameInput: String = ""

    // Processing state
    @State private var isProcessing: Bool = false
    @State private var processingMessage: String = ""

    // Sorting
    enum SortField: String { case name, smiles, mw, logP, hbd, hba, tpsa, rotB, atoms }
    @State private var sortField: SortField? = nil
    @State private var sortAscending: Bool = true

    // Detail panel: selected entry for inspection
    @State private var inspectedEntry: LigandEntry?

    // Preparation options (used in detail panel)
    @State private var prepAddHydrogens = true
    @State private var prepMinimize = true
    @State private var prepComputeCharges = true
    @State private var prepNumConformers = 50

    // Import mapping sheet
    @State private var importPreview: ImportPreview?
    @State private var showImportMapping: Bool = false

    // Conformer state
    @State private var conformers: [ConformerEntry] = []
    @State private var selectedConformerIndex: Int = 0
    @State private var isGeneratingConformers = false

    // Variant (tautomer/protomer) state
    @State private var isGeneratingVariants = false
    @State private var variantPH: Double = 7.4
    @State private var variantMaxTautomers: Int = 10
    @State private var variantMaxProtomers: Int = 8
    @State private var variantEnergyCutoff: Double = 10.0
    @State private var variantPkaThreshold: Double = 2.0
    @State private var selectedVariantID: UUID?
    @State private var conformerBudgetPerVariant: Int = 20

    struct ConformerEntry: Identifiable {
        let id: Int
        let molecule: MoleculeData
        let energy: Double
    }

    private var db: LigandDatabase { viewModel.ligandDB }

    private var filteredEntries: [LigandEntry] {
        var entries = db.entries
        if !searchText.isEmpty {
            entries = entries.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.smiles.localizedCaseInsensitiveContains(searchText)
            }
        }
        if lipinskiFilter {
            entries = entries.filter { $0.descriptors?.lipinski == true }
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

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            toolbar
            Divider()

            // Main content: table + detail panel
            if db.entries.isEmpty {
                emptyState
            } else {
                HSplitView {
                    tableView
                        .frame(minWidth: 600)

                    if selectedIDs.count > 1 {
                        ScrollView {
                            batchActionPanel
                        }
                        .frame(minWidth: 320, idealWidth: 400, maxWidth: 500)
                    } else if let entry = inspectedEntry {
                        detailPanel(entry)
                            .frame(minWidth: 320, idealWidth: 400, maxWidth: 500)
                    }
                }
            }

            Divider()
            statusBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .sheet(isPresented: $showImportMapping) {
            ImportMappingSheet(preview: $importPreview) { finalPreview in
                performMappedImport(finalPreview)
            }
        }
        .onChange(of: selectedIDs) { _, newIDs in
            // Auto-inspect when a single entry is selected
            if newIDs.count == 1, let id = newIDs.first {
                if inspectedEntry?.id != id {
                    inspectedEntry = db.entries.first { $0.id == id }
                    conformers = []
                    selectedConformerIndex = 0
                    selectedVariantID = nil
                }
            }
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 8) {
            // Add SMILES
            Button(action: { showSMILESEntry.toggle() }) {
                Label("Add SMILES", systemImage: "plus")
            }
            .controlSize(.small)

            // Import buttons
            Menu {
                Button("Import .smi file") { openImportWithMapping(.smi) }
                Button("Import .csv file") { openImportWithMapping(.csv) }
                Button("Import .sdf file") { openImportWithMapping(.sdf) }
            } label: {
                Label("Import", systemImage: "square.and.arrow.down")
            }
            .controlSize(.small)

            Divider().frame(height: 16)

            // Bulk actions (enabled when selection exists)
            Button(action: { prepareSelected() }) {
                Label("Prepare", systemImage: "wand.and.stars")
            }
            .controlSize(.small)
            .disabled(selectedIDs.isEmpty || isProcessing)
            .help("Prepare selected ligands (3D + minimize + charges)")

            Button(action: { deleteSelected() }) {
                Label("Delete", systemImage: "trash")
            }
            .controlSize(.small)
            .foregroundStyle(.red)
            .disabled(selectedIDs.isEmpty)
            .help("Delete selected ligands")

            Button(action: { useSelectedForDocking() }) {
                Label(selectedIDs.count > 1 ? "Queue \(selectedIDs.count) for Docking" : "Use for Docking",
                      systemImage: "arrow.right.circle")
            }
            .controlSize(.small)
            .disabled(selectedIDs.isEmpty)
            .help(selectedIDs.count > 1
                  ? "Queue \(selectedIDs.count) ligands for batch docking (launch from Docking tab)"
                  : "Set selected ligand as active for docking")

            Spacer()

            // Search
            TextField("Search...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 180)
                .font(.system(size: 11))

            Toggle("Lipinski", isOn: $lipinskiFilter)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .font(.system(size: 10))

            // Save/Load
            Menu {
                Button("Save Database") {
                    db.save()
                    viewModel.log.success("Saved \(db.count) ligands", category: .molecule)
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
                Image(systemName: selectedIDs.count == filteredEntries.count && !filteredEntries.isEmpty
                      ? "checkmark.square.fill" : "square")
                    .font(.system(size: 12))
            }
            .buttonStyle(.plain)
            .help("Select all / none")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }

    // MARK: - SMILES Entry (inline, shown below toolbar)

    @ViewBuilder
    private var smilesEntryBar: some View {
        if showSMILESEntry {
            HStack(spacing: 8) {
                PastableTextField(text: $smilesInput, placeholder: "SMILES", font: .monospacedSystemFont(ofSize: 11, weight: .regular))
                    .frame(height: 22)

                TextField("Name", text: $nameInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11))
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
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
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
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
                .frame(maxWidth: .infinity)
                .background(Color.accentColor.opacity(0.05))
            }

            // Table header (click to sort)
            HStack(spacing: 0) {
                headerCell("", width: 30, field: nil)
                headerCell("Name", width: 140, field: .name)
                headerCell("SMILES", width: nil, field: .smiles)
                headerCell("MW", width: 60, field: .mw, alignment: .trailing)
                headerCell("LogP", width: 50, field: .logP, alignment: .trailing)
                headerCell("HBD", width: 36, field: .hbd, alignment: .trailing)
                headerCell("HBA", width: 36, field: .hba, alignment: .trailing)
                headerCell("TPSA", width: 50, field: .tpsa, alignment: .trailing)
                headerCell("RotB", width: 36, field: .rotB, alignment: .trailing)
                headerCell("Lip.", width: 30, field: nil, alignment: .center)
                headerCell("Ki", width: 55, field: nil, alignment: .trailing)
                headerCell("pKi", width: 45, field: nil, alignment: .trailing)
                headerCell("IC50", width: 55, field: nil, alignment: .trailing)
                headerCell("Prep", width: 36, field: nil, alignment: .center)
                headerCell("Atoms", width: 44, field: .atoms, alignment: .trailing)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))

            Divider()

            // Table body
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(filteredEntries) { entry in
                        tableRow(entry)
                        // Expandable variant sub-rows
                        if expandedEntryIDs.contains(entry.id) && !entry.variants.isEmpty {
                            ForEach(Array(entry.variants.enumerated()), id: \.element.id) { idx, variant in
                                variantSubRow(parent: entry, variant: variant, index: idx)
                            }
                        }
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
                                .font(.system(size: 9))
                        }
                    }
                }
                .buttonStyle(.plain)
            } else {
                Text(title)
            }
        }
        .font(.system(size: 9, weight: isSorted ? .bold : .semibold))
        .foregroundStyle(isSorted ? .primary : .secondary)
        .frame(width: width, alignment: alignment)
        .frame(maxWidth: width == nil ? .infinity : nil, alignment: alignment)
    }

    @ViewBuilder
    private func tableRow(_ entry: LigandEntry) -> some View {
        let isSelected = selectedIDs.contains(entry.id)
        let isActive = viewModel.molecules.ligand?.name == entry.name
        let isInspected = inspectedEntry?.id == entry.id

        HStack(spacing: 0) {
            // Disclosure triangle for variant expansion
            if !entry.variants.isEmpty {
                Button {
                    if expandedEntryIDs.contains(entry.id) {
                        expandedEntryIDs.remove(entry.id)
                    } else {
                        expandedEntryIDs.insert(entry.id)
                    }
                } label: {
                    Image(systemName: expandedEntryIDs.contains(entry.id) ? "chevron.down" : "chevron.right")
                        .font(.system(size: 8, weight: .semibold))
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
                    .font(.system(size: 11))
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
                    .font(.system(size: 10, weight: isActive ? .semibold : .regular))
                    .lineLimit(1)
            }
            .frame(width: 140, alignment: .leading)

            // SMILES
            Text(entry.smiles)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(entry.smiles)

            // Descriptors
            if let d = entry.descriptors {
                Text(String(format: "%.0f", d.molecularWeight))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 60, alignment: .trailing)
                Text(String(format: "%.1f", d.logP))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 50, alignment: .trailing)
                Text("\(d.hbd)")
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
                Text("\(d.hba)")
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
                Text(String(format: "%.0f", d.tpsa))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 50, alignment: .trailing)
                Text("\(d.rotatableBonds)")
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
                Image(systemName: d.lipinski ? "checkmark" : "xmark")
                    .font(.system(size: 10))
                    .foregroundStyle(d.lipinski ? .green : .red.opacity(0.7))
                    .frame(width: 30)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 60, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 50, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 36, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 36, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 50, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 36, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 30, alignment: .center)
            }

            // Affinity data (Ki, pKi, IC50)
            if let ki = entry.ki {
                Text(String(format: "%.1f", ki))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.purple)
                    .frame(width: 55, alignment: .trailing)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 55, alignment: .trailing)
            }
            if let pKi = entry.pKi ?? entry.effectivePKi {
                Text(String(format: "%.2f", pKi))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(entry.pKi != nil ? .purple : .purple.opacity(0.5))
                    .frame(width: 45, alignment: .trailing)
                    .help(entry.pKi != nil ? "Stored pKi" : "Computed from \(entry.ki != nil ? "Ki" : "IC50")")
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 45, alignment: .trailing)
            }
            if let ic50 = entry.ic50 {
                Text(String(format: "%.1f", ic50))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.purple)
                    .frame(width: 55, alignment: .trailing)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 55, alignment: .trailing)
            }

            // Prepared status
            Image(systemName: entry.isPrepared ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 9))
                .foregroundColor(entry.isPrepared ? .green : .gray)
                .frame(width: 36)

            // Atom count
            Text(entry.atoms.isEmpty ? "—" : "\(entry.atoms.count)")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(entry.atoms.isEmpty ? .tertiary : .primary)
                .frame(width: 44, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
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
                db.remove(id: entry.id)
                selectedIDs.remove(entry.id)
            }
        }
    }

    // MARK: - Variant Sub-Row

    @ViewBuilder
    private func variantSubRow(parent: LigandEntry, variant: MolecularVariant, index: Int) -> some View {
        let kindSymbol = variant.kind == .tautomer ? "T" : "P"
        let kindColor: Color = variant.kind == .tautomer ? .cyan : .mint
        let displayName = "\(parent.name)_\(kindSymbol)\(index + 1)"

        HStack(spacing: 4) {
            Spacer().frame(width: 38) // indent past disclosure + checkbox

            // Kind badge
            Text(kindSymbol)
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 16, height: 16)
                .background(Circle().fill(kindColor))

            // Name
            Text(displayName)
                .font(.system(size: 9, weight: .medium))
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)

            // SMILES
            Text(variant.smiles)
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Relative energy
            if variant.relativeEnergy > 0 {
                Text(String(format: "+%.1f", variant.relativeEnergy))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.orange)
                    .frame(width: 40, alignment: .trailing)
            } else {
                Text("best")
                    .font(.system(size: 8, weight: .medium))
                    .foregroundStyle(.green)
                    .frame(width: 40, alignment: .trailing)
            }

            // Conformer count
            if variant.conformerCount > 0 {
                Text("C:\(variant.conformerCount)")
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 30, alignment: .trailing)
            }

            // Use for docking button
            if !variant.atoms.isEmpty {
                Button {
                    let mol = Molecule(
                        name: displayName,
                        atoms: variant.atoms,
                        bonds: variant.bonds,
                        title: variant.smiles,
                        smiles: variant.smiles
                    )
                    viewModel.setLigandForDocking(mol)
                    // Prefer ball-and-stick for variant preview
                    if viewModel.workspace.renderMode != .ballAndStick {
                        viewModel.workspace.renderMode = .ballAndStick
                    }
                } label: {
                    Image(systemName: "arrow.right.circle")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)
                .help("Use \(displayName) for docking")
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Color.secondary.opacity(0.03))
        .contentShape(Rectangle())
        .onTapGesture {
            // Inspect parent entry and select this variant
            inspectedEntry = parent
            selectedVariantID = variant.id
        }
    }

    // MARK: - Batch Action Panel

    @ViewBuilder
    private var batchActionPanel: some View {
        let selectedEntries = db.entries.filter { selectedIDs.contains($0.id) }

        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("\(selectedEntries.count) entries selected", systemImage: "checkmark.square.fill")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Deselect All") { selectedIDs.removeAll() }
                    .font(.system(size: 9))
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }

            if isBatchProcessing {
                VStack(spacing: 4) {
                    ProgressView(value: Double(batchProgress.current), total: Double(max(batchProgress.total, 1)))
                        .controlSize(.small)
                    Text("Processing \(batchProgress.current)/\(batchProgress.total)...")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            // Batch parameters (shared with single-entry panel)
            VStack(alignment: .leading, spacing: 6) {
                Label("Batch Preparation", systemImage: "wand.and.stars")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                Toggle("Add Hydrogens", isOn: $prepAddHydrogens)
                    .toggleStyle(.switch).controlSize(.small)
                Toggle("MMFF94 Minimization", isOn: $prepMinimize)
                    .toggleStyle(.switch).controlSize(.small)

                HStack {
                    Text("Conformers:")
                        .font(.system(size: 10))
                    Picker("", selection: $prepNumConformers) {
                        ForEach([1, 10, 50, 100], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                }
            }

            HStack(spacing: 8) {
                Button(action: { prepareBatchEntries(selectedEntries) }) {
                    Label("Prepare All", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.small)
                .disabled(isBatchProcessing)

                Button(action: { generateConformersBatch(selectedEntries) }) {
                    Label("Conformers", systemImage: "cube.transparent")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(isBatchProcessing)
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Label("Batch Variants", systemImage: "arrow.triangle.branch")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max tautomers:")
                        .font(.system(size: 10))
                    Picker("", selection: $variantMaxTautomers) {
                        ForEach([5, 10, 25], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                }

                HStack {
                    Text("Max protomers:")
                        .font(.system(size: 10))
                    Picker("", selection: $variantMaxProtomers) {
                        ForEach([4, 8, 16], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                }
            }

            HStack(spacing: 8) {
                Button(action: { generateTautomersBatch(selectedEntries) }) {
                    Label("Tautomers", systemImage: "arrow.2.squarepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(isBatchProcessing)

                Button(action: { generateProtomersBatch(selectedEntries) }) {
                    Label("Protomers", systemImage: "bolt")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(isBatchProcessing)
            }
        }
        .padding()
    }

    // MARK: - Detail Panel (structure viz, preparation, conformers)

    @ViewBuilder
    private func detailPanel(_ entry: LigandEntry) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Header
                HStack {
                    Label(entry.name, systemImage: "hexagon.fill")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Button(action: { inspectedEntry = nil; conformers = [] }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }

                // SMILES
                if !entry.smiles.isEmpty {
                    Text(entry.smiles)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(3)
                }

                Divider()

                // Structure visualization
                structureView(entry)

                Divider()

                // Descriptors
                if let d = entry.descriptors {
                    descriptorCard(d)
                    Divider()
                }

                // Binding Affinity Data
                affinitySection(entry)
                Divider()

                // Preparation options & action
                if !entry.smiles.isEmpty {
                    preparationSection(entry)
                    Divider()
                }

                // Tautomer / Protomer variants
                if !entry.smiles.isEmpty {
                    variantsSection(entry)
                    Divider()
                }

                // Conformer browser
                if !conformers.isEmpty {
                    conformerBrowser(entry)
                    Divider()
                }

                // Actions
                actionButtons(entry)
            }
            .padding(12)
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    // MARK: - Structure Visualization

    @ViewBuilder
    private func structureView(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Structure", systemImage: "atom")
                .font(.system(size: 11, weight: .medium))

            if !entry.atoms.isEmpty {
                // 3D atom stats
                let heavyAtoms = entry.atoms.filter { $0.element != .H }
                HStack(spacing: 12) {
                    statBadge("Atoms", "\(entry.atoms.count)")
                    statBadge("Heavy", "\(heavyAtoms.count)")
                    statBadge("Bonds", "\(entry.bonds.count)")
                    if entry.conformerCount > 1 {
                        statBadge("Conf.", "\(entry.conformerCount)")
                    }
                }

                // Simple 2D projection of 3D coords
                Canvas { context, size in
                    drawMolecule2D(entry: entry, context: context, size: size)
                }
                .frame(height: 180)
                .background(Color.black.opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 9))
                        .foregroundStyle(.orange)
                    Text("No 3D coordinates — prepare this ligand first")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(Color.orange.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private func drawMolecule2D(entry: LigandEntry, context: GraphicsContext, size: CGSize) {
        let atoms = entry.atoms
        let bonds = entry.bonds
        guard !atoms.isEmpty else { return }

        // Project 3D → 2D using XY plane with padding
        let xs = atoms.map { CGFloat($0.position.x) }
        let ys = atoms.map { CGFloat($0.position.y) }
        let minX = xs.min()!, maxX = xs.max()!
        let minY = ys.min()!, maxY = ys.max()!
        let rangeX = max(maxX - minX, 0.1)
        let rangeY = max(maxY - minY, 0.1)
        let padding: CGFloat = 20
        let drawW = size.width - padding * 2
        let drawH = size.height - padding * 2
        let scale = min(drawW / rangeX, drawH / rangeY)
        let cx = size.width / 2
        let cy = size.height / 2
        let midX = (minX + maxX) / 2
        let midY = (minY + maxY) / 2

        func project(_ atom: Atom) -> CGPoint {
            CGPoint(
                x: cx + (CGFloat(atom.position.x) - midX) * scale,
                y: cy - (CGFloat(atom.position.y) - midY) * scale  // flip Y
            )
        }

        // Draw bonds
        for bond in bonds {
            guard bond.atomIndex1 < atoms.count, bond.atomIndex2 < atoms.count else { continue }
            let p1 = project(atoms[bond.atomIndex1])
            let p2 = project(atoms[bond.atomIndex2])
            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(.gray.opacity(0.6)), lineWidth: 1.5)
        }

        // Draw atoms (only heavy atoms to keep it clean)
        for atom in atoms {
            if atom.element == .H { continue }
            let p = project(atom)
            let radius: CGFloat = 4
            let color = elementColor(atom.element)
            context.fill(Circle().path(in: CGRect(x: p.x - radius, y: p.y - radius, width: radius * 2, height: radius * 2)),
                        with: .color(color))
        }
    }

    private func elementColor(_ element: Element) -> Color {
        switch element {
        case .C:  return .white.opacity(0.8)
        case .N:  return .blue
        case .O:  return .red
        case .S:  return .yellow
        case .P:  return .orange
        case .F:  return .green
        case .Cl: return .green.opacity(0.7)
        case .Br: return .brown
        default:  return .gray
        }
    }

    // MARK: - Descriptor Card

    @ViewBuilder
    private func descriptorCard(_ d: LigandDescriptors) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Properties", systemImage: "list.bullet.rectangle")
                .font(.system(size: 11, weight: .medium))

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 12) {
                    propBadge("MW", String(format: "%.1f", d.molecularWeight))
                    propBadge("LogP", String(format: "%.2f", d.logP))
                    propBadge("TPSA", String(format: "%.1f", d.tpsa))
                }
                HStack(spacing: 12) {
                    propBadge("HBD", "\(d.hbd)")
                    propBadge("HBA", "\(d.hba)")
                    propBadge("RotB", "\(d.rotatableBonds)")
                    propBadge("Rings", "\(d.rings)")
                }
                HStack(spacing: 8) {
                    ruleTag("Lipinski", d.lipinski)
                    ruleTag("Veber", d.veber)
                    Text("fCSP3: \(String(format: "%.2f", d.fractionCSP3))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    // MARK: - Affinity Section

    @ViewBuilder
    private func affinitySection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Binding Affinity", systemImage: "chart.line.uptrend.xyaxis")
                .font(.system(size: 11, weight: .medium))

            VStack(alignment: .leading, spacing: 4) {
                affinityField("Ki (nM)", value: entry.ki, entryId: entry.id) { newVal in
                    if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                        db.entries[idx].ki = newVal
                        inspectedEntry = db.entries[idx]
                    }
                }
                affinityField("pKi", value: entry.pKi, entryId: entry.id) { newVal in
                    if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                        db.entries[idx].pKi = newVal
                        inspectedEntry = db.entries[idx]
                    }
                }
                affinityField("IC50 (nM)", value: entry.ic50, entryId: entry.id) { newVal in
                    if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                        db.entries[idx].ic50 = newVal
                        inspectedEntry = db.entries[idx]
                    }
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            if let ePKi = entry.effectivePKi {
                HStack(spacing: 4) {
                    Text("Effective pKi:")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.2f", ePKi))
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(.purple)
                }
            }
        }
    }

    @ViewBuilder
    private func affinityField(_ label: String, value: Float?, entryId: UUID, onUpdate: @escaping (Float?) -> Void) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 10))
                .frame(width: 65, alignment: .leading)

            let textBinding = Binding<String>(
                get: { value.map { String(format: "%.2f", $0) } ?? "" },
                set: { newText in
                    let trimmed = newText.trimmingCharacters(in: .whitespaces)
                    onUpdate(trimmed.isEmpty ? nil : Float(trimmed))
                }
            )
            TextField("—", text: textBinding)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 10, design: .monospaced))
                .frame(width: 90)

            if value != nil {
                Button(action: { onUpdate(nil) }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Preparation Section

    @ViewBuilder
    private func preparationSection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Preparation", systemImage: "wand.and.stars")
                .font(.system(size: 11, weight: .medium))

            Toggle("Add Hydrogens", isOn: $prepAddHydrogens)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.system(size: 11))

            Toggle("MMFF94 Minimization", isOn: $prepMinimize)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.system(size: 11))

            Toggle("Gasteiger Charges", isOn: $prepComputeCharges)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.system(size: 11))

            HStack {
                Text("Conformers:")
                    .font(.system(size: 11))
                Picker("", selection: $prepNumConformers) {
                    Text("1").tag(1)
                    Text("10").tag(10)
                    Text("50").tag(50)
                    Text("100").tag(100)
                    Text("200").tag(200)
                }
                .pickerStyle(.segmented)
                .frame(width: 200)
            }

            HStack(spacing: 8) {
                Button(action: { prepareInspectedEntry(entry) }) {
                    Label("Prepare", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(entry.smiles.isEmpty || isProcessing || isGeneratingConformers)

                if entry.isPrepared && prepNumConformers > 1 {
                    Button(action: { generateConformersForEntry(entry) }) {
                        Label("Generate \(prepNumConformers) Conformers", systemImage: "cube.transparent")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(isGeneratingConformers || isProcessing)
                }
            }

            if isProcessing || isGeneratingConformers {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(isGeneratingConformers ? "Generating conformers..." : "Preparing...")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Conformer Browser

    @ViewBuilder
    private func conformerBrowser(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Conformers (\(conformers.count))", systemImage: "cube.transparent")
                    .font(.system(size: 11, weight: .medium))

                Spacer()

                Button("Use Best") { applyBestConformer(entry) }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(conformers.isEmpty)

                Button("Use Selected") { applySelectedConformer(entry) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
                    .disabled(conformers.isEmpty)
            }

            // Conformer visualization: show energy-sorted bars
            ScrollView(.vertical) {
                VStack(spacing: 2) {
                    ForEach(conformers) { conf in
                        conformerRow(conf)
                    }
                }
            }
            .frame(maxHeight: 180)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    @ViewBuilder
    private func conformerRow(_ conf: ConformerEntry) -> some View {
        let isSelected = conf.id == selectedConformerIndex
        let isBest = conf.id == (conformers.first?.id ?? -1)

        HStack(spacing: 8) {
            Text("#\(conf.id + 1)")
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .frame(width: 30, alignment: .trailing)

            // Energy bar
            let minEnergy = conformers.map(\.energy).min() ?? 0
            let maxEnergy = conformers.map(\.energy).max() ?? 1
            let range = max(maxEnergy - minEnergy, 0.1)
            let normalized = (conf.energy - minEnergy) / range

            GeometryReader { geo in
                RoundedRectangle(cornerRadius: 2)
                    .fill(energyColor(normalized))
                    .frame(width: max(4, geo.size.width * CGFloat(1.0 - normalized)))
            }
            .frame(height: 8)

            Text(String(format: "%.2f", conf.energy))
                .font(.system(size: 10, design: .monospaced))
                .frame(width: 60, alignment: .trailing)

            Text("kcal/mol")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)

            if isBest {
                Text("Best")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.green)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture {
            selectedConformerIndex = conf.id
            // Update the structure preview with this conformer's coords
            if var updated = inspectedEntry {
                updated.atoms = conf.molecule.atoms
                updated.bonds = conf.molecule.bonds
                inspectedEntry = updated
            }
        }
    }

    private func energyColor(_ normalized: Double) -> Color {
        if normalized < 0.3 { return .green }
        if normalized < 0.6 { return .yellow }
        return .orange
    }

    // MARK: - Variants Section (Tautomers & Protomers)

    @ViewBuilder
    private func variantsSection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Label("Variants", systemImage: "arrow.triangle.branch")
                    .font(.system(size: 11, weight: .medium))
                Spacer()
                if !entry.variants.isEmpty {
                    let tCount = entry.variants.filter { $0.kind == .tautomer }.count
                    let pCount = entry.variants.filter { $0.kind == .protomer }.count
                    if tCount > 0 {
                        Text("\(tCount)T")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 4).padding(.vertical, 1)
                            .background(Color.purple.opacity(0.15))
                            .foregroundStyle(.purple)
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                    if pCount > 0 {
                        Text("\(pCount)P")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 4).padding(.vertical, 1)
                            .background(Color.orange.opacity(0.15))
                            .foregroundStyle(.orange)
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                }
            }

            // Generation parameters
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text("pH")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.1f", variantPH))
                        .font(.system(size: 9, design: .monospaced))
                        .frame(width: 28)
                    Slider(value: $variantPH, in: 5.0...9.0, step: 0.1)
                        .controlSize(.mini)
                }
                HStack(spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Max tautomers").font(.system(size: 9)).foregroundStyle(.secondary)
                        Picker("", selection: $variantMaxTautomers) {
                            Text("5").tag(5); Text("10").tag(10); Text("25").tag(25); Text("50").tag(50)
                        }
                        .pickerStyle(.segmented)
                        .controlSize(.mini)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Max protomers").font(.system(size: 9)).foregroundStyle(.secondary)
                        Picker("", selection: $variantMaxProtomers) {
                            Text("4").tag(4); Text("8").tag(8); Text("16").tag(16)
                        }
                        .pickerStyle(.segmented)
                        .controlSize(.mini)
                    }
                }
                HStack(spacing: 8) {
                    Text("Energy cutoff")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.0f", variantEnergyCutoff))
                        .font(.system(size: 9, design: .monospaced))
                    Stepper("", value: $variantEnergyCutoff, in: 1...50, step: 1)
                        .labelsHidden()
                        .controlSize(.mini)
                    Text("kcal/mol")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Generate buttons
            HStack(spacing: 6) {
                Button(action: { generateTautomers(entry) }) {
                    Label("Tautomers", systemImage: "arrow.2.squarepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(isGeneratingVariants || entry.smiles.isEmpty)

                Button(action: { generateProtomers(entry) }) {
                    Label("Protomers", systemImage: "bolt.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(isGeneratingVariants || entry.smiles.isEmpty)
            }

            if isGeneratingVariants {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Enumerating variants...")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }

            // Variant list
            if !entry.variants.isEmpty {
                ScrollView {
                    LazyVStack(spacing: 2) {
                        ForEach(entry.variants) { variant in
                            variantRow(variant, entry: entry)
                        }
                    }
                }
                .frame(maxHeight: 180)

                // Conformer generation for variants
                Divider()
                HStack(spacing: 6) {
                    Text("Conformers per variant:")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                    Picker("", selection: $conformerBudgetPerVariant) {
                        Text("5").tag(5); Text("10").tag(10); Text("20").tag(20); Text("50").tag(50)
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.mini)
                    .frame(width: 160)
                }

                let totalMolecules = entry.variants.count * conformerBudgetPerVariant
                Text("Total: \(entry.variants.count) variants \u{00d7} \(conformerBudgetPerVariant) conformers = \(totalMolecules) molecules")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)

                Button(action: { generateConformersForAllVariants(entry) }) {
                    Label("Generate Conformers for All Variants", systemImage: "square.stack.3d.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(isGeneratingVariants || isGeneratingConformers)

                // Clear all variants
                Button(action: { clearVariants(entry) }) {
                    Label("Clear All Variants", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .foregroundStyle(.red)
            }
        }
    }

    @ViewBuilder
    private func variantRow(_ variant: MolecularVariant, entry: LigandEntry) -> some View {
        let isSelected = selectedVariantID == variant.id
        HStack(spacing: 6) {
            // Kind badge
            Text(variant.kind.symbol)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .frame(width: 16, height: 16)
                .background(variant.kind == .tautomer ? Color.purple.opacity(0.2) : Color.orange.opacity(0.2))
                .foregroundStyle(variant.kind == .tautomer ? .purple : .orange)
                .clipShape(RoundedRectangle(cornerRadius: 3))

            // Label + SMILES
            VStack(alignment: .leading, spacing: 1) {
                Text(variant.label)
                    .font(.system(size: 9, weight: .medium))
                    .lineLimit(1)
                Text(variant.smiles)
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            // Energy
            if !variant.relativeEnergy.isNaN {
                Text(String(format: "%.1f", variant.relativeEnergy))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text("kcal")
                    .font(.system(size: 8))
                    .foregroundStyle(.tertiary)
            }

            // Conformer count
            if variant.conformerCount > 0 {
                Text("\(variant.conformerCount)c")
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.green)
            }

            // Delete button
            Button(action: { deleteVariant(variant.id, from: entry) }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(.red.opacity(0.6))
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture {
            selectedVariantID = isSelected ? nil : variant.id
            // When selecting a variant, show its structure in the preview
            if let idx = db.entries.firstIndex(where: { $0.id == entry.id }),
               selectedVariantID != nil, !variant.atoms.isEmpty {
                inspectedEntry = db.entries[idx]
            }
        }
    }

    // MARK: - Variant Actions

    private func generateTautomers(_ entry: LigandEntry) {
        guard let idx = db.entries.firstIndex(where: { $0.id == entry.id }) else { return }
        let smi = entry.smiles
        let nm = entry.name
        let maxT = variantMaxTautomers
        let cutoff = variantEnergyCutoff
        isGeneratingVariants = true

        Task {
            let results = await Task.detached { @Sendable in
                RDKitBridge.enumerateTautomers(smiles: smi, name: nm,
                                                maxTautomers: maxT, energyCutoff: cutoff)
            }.value

            let newVariants: [MolecularVariant] = results.map { r in
                MolecularVariant(
                    smiles: r.smiles,
                    atoms: r.molecule.atoms,
                    bonds: r.molecule.bonds,
                    relativeEnergy: r.score,
                    kind: .tautomer,
                    label: r.label,
                    isPrepared: !r.molecule.atoms.isEmpty
                )
            }

            // Remove existing tautomers, add new ones
            db.entries[idx].variants.removeAll { $0.kind == .tautomer }
            db.entries[idx].variants.append(contentsOf: newVariants)
            inspectedEntry = db.entries[idx]
            isGeneratingVariants = false
            ActivityLog.shared.success("Generated \(newVariants.count) tautomer(s) for \(nm)", category: .molecule)
        }
    }

    private func generateProtomers(_ entry: LigandEntry) {
        guard let idx = db.entries.firstIndex(where: { $0.id == entry.id }) else { return }
        let smi = entry.smiles
        let nm = entry.name
        let maxP = variantMaxProtomers
        let ph = variantPH
        let threshold = variantPkaThreshold
        isGeneratingVariants = true

        Task {
            let results = await Task.detached { @Sendable in
                RDKitBridge.enumerateProtomers(smiles: smi, name: nm,
                                                maxProtomers: maxP, pH: ph, pkaThreshold: threshold)
            }.value

            let newVariants: [MolecularVariant] = results.map { r in
                MolecularVariant(
                    smiles: r.smiles,
                    atoms: r.molecule.atoms,
                    bonds: r.molecule.bonds,
                    relativeEnergy: r.score,
                    kind: .protomer,
                    label: r.label,
                    isPrepared: !r.molecule.atoms.isEmpty
                )
            }

            // Remove existing protomers, add new ones
            db.entries[idx].variants.removeAll { $0.kind == .protomer }
            db.entries[idx].variants.append(contentsOf: newVariants)
            inspectedEntry = db.entries[idx]
            isGeneratingVariants = false
            ActivityLog.shared.success("Generated \(newVariants.count) protomer(s) for \(nm) at pH \(String(format: "%.1f", ph))", category: .molecule)
        }
    }

    private func deleteVariant(_ variantID: UUID, from entry: LigandEntry) {
        guard let idx = db.entries.firstIndex(where: { $0.id == entry.id }) else { return }
        db.entries[idx].variants.removeAll { $0.id == variantID }
        if selectedVariantID == variantID { selectedVariantID = nil }
        inspectedEntry = db.entries[idx]
    }

    private func clearVariants(_ entry: LigandEntry) {
        guard let idx = db.entries.firstIndex(where: { $0.id == entry.id }) else { return }
        db.entries[idx].variants.removeAll()
        selectedVariantID = nil
        inspectedEntry = db.entries[idx]
    }

    private func generateConformersForAllVariants(_ entry: LigandEntry) {
        guard let idx = db.entries.firstIndex(where: { $0.id == entry.id }) else { return }
        let budget = conformerBudgetPerVariant
        isGeneratingVariants = true

        Task {
            for vi in 0..<db.entries[idx].variants.count {
                let varSmi = db.entries[idx].variants[vi].smiles
                let varName = entry.name

                let confs = await Task.detached { @Sendable in
                    RDKitBridge.generateConformers(smiles: varSmi, name: varName,
                                                    count: budget, minimize: true)
                }.value

                if let best = confs.first {
                    db.entries[idx].variants[vi].atoms = best.molecule.atoms
                    db.entries[idx].variants[vi].bonds = best.molecule.bonds
                    db.entries[idx].variants[vi].isPrepared = true
                    db.entries[idx].variants[vi].conformerCount = confs.count
                }
            }

            inspectedEntry = db.entries[idx]
            isGeneratingVariants = false
            let total = db.entries[idx].variants.reduce(0) { $0 + $1.conformerCount }
            ActivityLog.shared.success("Generated conformers for \(db.entries[idx].variants.count) variant(s): \(total) total", category: .molecule)
        }
    }

    // MARK: - Action Buttons

    @ViewBuilder
    private func actionButtons(_ entry: LigandEntry) -> some View {
        HStack(spacing: 8) {
            Button(action: { viewInViewport(entry) }) {
                Label("View in 3D", systemImage: "cube")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(entry.atoms.isEmpty)

            Button(action: { useEntryForDocking(entry) }) {
                Label("Use for Docking", systemImage: "arrow.right.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(entry.atoms.isEmpty)
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "tray")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("No ligands in database")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.secondary)
            Text("Add SMILES, or import .smi / .csv / .sdf files")
                .font(.system(size: 12))
                .foregroundStyle(.tertiary)

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

    private var statusBar: some View {
        HStack(spacing: 16) {
            Text("Total: \(db.count)")
                .font(.system(size: 10))
            Text("Prepared: \(db.entries.filter(\.isPrepared).count)")
                .font(.system(size: 10))
                .foregroundStyle(.green)
            Text("Selected: \(selectedIDs.count)")
                .font(.system(size: 10))
                .foregroundStyle(Color.accentColor)

            Spacer()

            if let lig = viewModel.molecules.ligand {
                HStack(spacing: 4) {
                    Circle().fill(.green).frame(width: 5, height: 5)
                    Text("Active: \(lig.name)")
                        .font(.system(size: 10, weight: .medium))
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
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
        if selectedIDs.count == filteredEntries.count && !filteredEntries.isEmpty {
            selectedIDs.removeAll()
        } else {
            selectedIDs = Set(filteredEntries.map(\.id))
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

    private func deleteSelected() {
        let count = selectedIDs.count
        if let inspID = inspectedEntry?.id, selectedIDs.contains(inspID) {
            inspectedEntry = nil
            conformers = []
        }
        db.remove(ids: selectedIDs)
        selectedIDs.removeAll()
        viewModel.log.info("Deleted \(count) ligands", category: .molecule)
    }

    private func useSelectedForDocking() {
        let selectedEntries = db.entries.filter { selectedIDs.contains($0.id) }
        guard !selectedEntries.isEmpty else { return }

        if selectedEntries.count == 1 {
            useEntryForDocking(selectedEntries[0])
        } else {
            viewModel.queueLigandsForBatchDocking(selectedEntries)
        }
    }

    /// Resolve which atoms/bonds/smiles to use: selected variant or entry itself.
    private func resolveActiveStructure(_ entry: LigandEntry) -> (atoms: [Atom], bonds: [Bond], smiles: String, name: String)? {
        if let vid = selectedVariantID, let variant = entry.variants.first(where: { $0.id == vid }) {
            guard !variant.atoms.isEmpty else { return nil }
            return (variant.atoms, variant.bonds, variant.smiles, "\(entry.name) (\(variant.label))")
        }
        guard !entry.atoms.isEmpty else { return nil }
        return (entry.atoms, entry.bonds, entry.smiles, entry.name)
    }

    private func viewInViewport(_ entry: LigandEntry) {
        guard let active = resolveActiveStructure(entry) else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare first", category: .molecule)
            return
        }
        let mol = Molecule(name: active.name, atoms: active.atoms, bonds: active.bonds, title: active.smiles, smiles: active.smiles)
        viewModel.setLigandForDocking(mol)
        if !conformers.isEmpty {
            viewModel.setLigandConformers(conformers.map { (atoms: $0.molecule.atoms, bonds: $0.molecule.bonds, energy: $0.energy) })
        }
        viewModel.fitToLigand()
    }

    private func useEntryForDocking(_ entry: LigandEntry) {
        guard let active = resolveActiveStructure(entry) else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare first", category: .molecule)
            return
        }
        let mol = Molecule(name: active.name, atoms: active.atoms, bonds: active.bonds, title: active.smiles, smiles: active.smiles)
        viewModel.setLigandForDocking(mol)
        if !conformers.isEmpty {
            viewModel.setLigandConformers(conformers.map { (atoms: $0.molecule.atoms, bonds: $0.molecule.bonds, energy: $0.energy) })
        }
    }

    private func prepareSingleEntry(_ entry: LigandEntry) {
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

    private func prepareInspectedEntry(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isProcessing = true
        processingMessage = "Preparing \(entry.name)..."
        conformers = []
        selectedConformerIndex = 0

        Task {
            let smi = entry.smiles, nm = entry.name
            let nc = prepNumConformers, ah = prepAddHydrogens
            let mn = prepMinimize, cc = prepComputeCharges
            let (mol, desc, err) = await Task.detached { @Sendable in
                RDKitBridge.prepareLigand(smiles: smi, name: nm,
                                          numConformers: nc, addHydrogens: ah,
                                          minimize: mn, computeCharges: cc)
            }.value

            if let mol {
                var updated = entry
                updated.atoms = mol.atoms
                updated.bonds = mol.bonds
                updated.descriptors = desc
                updated.isPrepared = true
                updated.conformerCount = 1
                db.update(updated)
                inspectedEntry = db.entries.first { $0.id == entry.id }
                viewModel.log.success("Prepared \(nm)", category: .molecule)
            } else if let err {
                viewModel.log.error("Preparation failed: \(err)", category: .molecule)
            }
            isProcessing = false
            processingMessage = ""
        }
    }

    private func generateConformersForEntry(_ entry: LigandEntry) {
        // Use selected variant's SMILES if one is active
        let activeSmi: String
        if let vid = selectedVariantID, let variant = entry.variants.first(where: { $0.id == vid }) {
            activeSmi = variant.smiles
        } else {
            activeSmi = entry.smiles
        }
        guard !activeSmi.isEmpty else { return }
        isGeneratingConformers = true

        Task {
            let smi = activeSmi, nm = entry.name
            let nc = prepNumConformers, mn = prepMinimize
            let results = await Task.detached { @Sendable in
                RDKitBridge.generateConformers(smiles: smi, name: nm, count: nc, minimize: mn)
            }.value

            conformers = results.enumerated().map { idx, pair in
                ConformerEntry(id: idx, molecule: pair.molecule, energy: pair.energy)
            }

            // Auto-select best conformer and update entry
            if let best = conformers.first {
                selectedConformerIndex = best.id
                var updated = entry
                updated.atoms = best.molecule.atoms
                updated.bonds = best.molecule.bonds
                updated.conformerCount = conformers.count
                db.update(updated)
                inspectedEntry = db.entries.first { $0.id == entry.id }
            }

            // Sync conformers to viewport if this is the active ligand
            if viewModel.molecules.ligand?.name == nm {
                viewModel.setLigandConformers(conformers.map { (atoms: $0.molecule.atoms, bonds: $0.molecule.bonds, energy: $0.energy) })
            }

            isGeneratingConformers = false
            viewModel.log.success("Generated \(conformers.count) conformers for \(nm)", category: .molecule)
        }
    }

    private func applyBestConformer(_ entry: LigandEntry) {
        guard let best = conformers.first else { return }
        selectedConformerIndex = best.id
        applyConformer(best, to: entry)
    }

    private func applySelectedConformer(_ entry: LigandEntry) {
        guard selectedConformerIndex < conformers.count else { return }
        let conf = conformers[selectedConformerIndex]
        applyConformer(conf, to: entry)
    }

    private func applyConformer(_ conf: ConformerEntry, to entry: LigandEntry) {
        var updated = entry
        updated.atoms = conf.molecule.atoms
        updated.bonds = conf.molecule.bonds
        db.update(updated)
        inspectedEntry = db.entries.first { $0.id == entry.id }
        viewModel.log.info("Applied conformer #\(conf.id + 1) to \(entry.name)", category: .molecule)
    }

    private func addAndPrepare(smiles: String, name: String) {
        db.addFromSMILES(smiles, name: name)
        if let entry = db.entries.last, entry.smiles == smiles {
            prepareSingleEntry(entry)
        }
    }

    // MARK: - Import with Column Mapping

    private func openImportWithMapping(_ fileType: ImportFileType) {
        guard let url = FileImportHandler.showBatchOpenPanel() else { return }
        do {
            let preview = try buildImportPreview(url: url, fileType: fileType)
            importPreview = preview
            showImportMapping = true
        } catch {
            viewModel.log.error("Failed to read file: \(error.localizedDescription)", category: .molecule)
        }
    }

    private func buildImportPreview(url: URL, fileType: ImportFileType) throws -> ImportPreview {
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

    private func buildCSVPreview(content: String, url: URL) -> ImportPreview {
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

    private func buildSMIPreview(content: String, url: URL) -> ImportPreview {
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

    private func buildSDFPreview(content: String, url: URL) -> ImportPreview {
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
    private func suggestTarget(for header: String) -> ImportTargetField {
        let h = header.lowercased().trimmingCharacters(in: .whitespaces)
        if h == "smiles" || h == "molecule" || h == "structure" || h == "canonical_smiles" { return .smiles }
        if h == "name" || h == "id" || h == "title" || h == "compound_name" || h == "mol_name" { return .name }
        if h == "ki" || h == "ki_nm" || h == "ki (nm)" || h == "ki_value" { return .ki }
        if h == "pki" || h == "p_ki" || h == "pki_value" { return .pKi }
        if h == "ic50" || h == "ic50_nm" || h == "ic50 (nm)" || h == "ic50_value" { return .ic50 }
        return .none
    }

    // MARK: - Execute Mapped Import

    private func performMappedImport(_ preview: ImportPreview) {
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

    private func performCSVImport(_ preview: ImportPreview) throws {
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

    private func performSMIImport(_ preview: ImportPreview) throws {
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

    private func performSDFImport(_ preview: ImportPreview) throws {
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

    // MARK: - Helpers

    private func statBadge(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    private func propBadge(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
        }
    }

    private func ruleTag(_ name: String, _ passes: Bool) -> some View {
        HStack(spacing: 3) {
            Image(systemName: passes ? "checkmark.circle.fill" : "xmark.circle")
                .font(.system(size: 10))
                .foregroundStyle(passes ? .green : .red)
            Text(name)
                .font(.system(size: 10))
        }
    }

    // MARK: - Batch Operations

    private func prepareBatchEntries(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let ah = prepAddHydrogens, mn = prepMinimize, cc = prepComputeCharges, nc = prepNumConformers
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let (mol, desc, _) = await Task.detached { @Sendable in
                    RDKitBridge.prepareLigand(smiles: smi, name: nm, numConformers: nc,
                                              addHydrogens: ah, minimize: mn, computeCharges: cc)
                }.value

                if let mol {
                    var updated = entry
                    updated.atoms = mol.atoms
                    updated.bonds = mol.bonds
                    updated.descriptors = desc
                    updated.isPrepared = true
                    updated.conformerCount = 1
                    db.update(updated)
                }
            }
            viewModel.log.success("Batch prepared \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }

    private func generateConformersBatch(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let nc = prepNumConformers, mn = prepMinimize
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let results = await Task.detached { @Sendable in
                    RDKitBridge.generateConformers(smiles: smi, name: nm, count: nc, minimize: mn)
                }.value

                if let best = results.first {
                    var updated = entry
                    updated.atoms = best.molecule.atoms
                    updated.bonds = best.molecule.bonds
                    updated.conformerCount = results.count
                    updated.isPrepared = true
                    db.update(updated)
                }
            }
            viewModel.log.success("Batch generated conformers for \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }

    private func generateTautomersBatch(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let maxT = variantMaxTautomers
            let cutoff = variantEnergyCutoff
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let results = await Task.detached { @Sendable in
                    RDKitBridge.enumerateTautomers(smiles: smi, name: nm,
                                                    maxTautomers: maxT, energyCutoff: cutoff)
                }.value

                if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                    db.entries[idx].variants.removeAll { $0.kind == .tautomer }
                    let newVariants = results.map { r in
                        MolecularVariant(
                            smiles: r.smiles,
                            atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                            relativeEnergy: r.score, kind: .tautomer,
                            label: r.label, isPrepared: !r.molecule.atoms.isEmpty,
                            conformerCount: 0
                        )
                    }
                    db.entries[idx].variants.append(contentsOf: newVariants)
                    // Auto-expand so user sees them
                    expandedEntryIDs.insert(entry.id)
                }
            }
            viewModel.log.success("Batch generated tautomers for \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }

    private func generateProtomersBatch(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let maxP = variantMaxProtomers
            let pH = variantPH
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let results = await Task.detached { @Sendable in
                    RDKitBridge.enumerateProtomers(smiles: smi, name: nm,
                                                    maxProtomers: maxP, pH: pH)
                }.value

                if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                    db.entries[idx].variants.removeAll { $0.kind == .protomer }
                    let newVariants = results.map { r in
                        MolecularVariant(
                            smiles: r.smiles,
                            atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                            relativeEnergy: r.score, kind: .protomer,
                            label: r.label, isPrepared: !r.molecule.atoms.isEmpty,
                            conformerCount: 0
                        )
                    }
                    db.entries[idx].variants.append(contentsOf: newVariants)
                    expandedEntryIDs.insert(entry.id)
                }
            }
            viewModel.log.success("Batch generated protomers for \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
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
