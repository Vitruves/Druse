import SwiftUI
import AppKit
import UniformTypeIdentifiers

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

    // Variant (tautomer/protomer) state
    @State var isGeneratingVariants = false
    @State var variantPH: Double = 7.4
    @State var variantMaxTautomers: Int = 10
    @State var variantMaxProtomers: Int = 8
    @State var variantEnergyCutoff: Double = 10.0
    @State var variantPkaThreshold: Double = 2.0
    @State var selectedVariantID: UUID?
    @State var conformerBudgetPerVariant: Int = 20

    // 2D/3D preview toggle
    @State var show3DPreview: Bool = false
    @State var miniRenderer: Renderer? = nil

    struct ConformerEntry: Identifiable {
        let id: Int
        let molecule: MoleculeData
        let energy: Double
    }

    var db: LigandDatabase { viewModel.ligandDB }

    var filteredEntries: [LigandEntry] {
        var entries = db.entries.filter { $0.parentID == nil }
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

    // Tab for the bottom options panel
    @State var bottomTab: BottomTab = .properties
    enum BottomTab: String, CaseIterable {
        case properties = "Properties"
        case prepare = "Prepare"
        case variants = "Variants"
        case conformers = "Conformers"
    }

    // 2D structure preview (RDKit-based, not 3D projection)
    @State var ligand2DCoords: RDKitBridge.Coords2D?
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
                                    .font(.system(size: 13))
                                    .foregroundStyle(.tertiary)
                                Spacer()
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }
                    .frame(minHeight: 250)
                }
                .onAppear {
                    // Auto-select first entry when window opens
                    if selectedIDs.isEmpty, let first = filteredEntries.first {
                        selectedIDs = [first.id]
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
            if newIDs.count == 1, let id = newIDs.first {
                if inspectedEntry?.id != id {
                    inspectedEntry = db.entries.first { $0.id == id }
                    conformers = []
                    selectedConformerIndex = 0
                    selectedVariantID = nil
                    // Compute 2D coords for preview
                    compute2DPreview(smiles: inspectedEntry?.smiles ?? "")
                    // Update 3D viewport if active
                    if show3DPreview, let entry = inspectedEntry, !entry.atoms.isEmpty {
                        updateMiniRenderer(atoms: entry.atoms, bonds: entry.bonds)
                    }
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
    var smilesEntryBar: some View {
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
                        // Expandable child entry sub-rows (flat variant model)
                        let children = db.children(of: entry.id)
                        if expandedEntryIDs.contains(entry.id) && !children.isEmpty {
                            ForEach(Array(children.enumerated()), id: \.element.id) { idx, child in
                                childEntrySubRow(parent: entry, child: child, index: idx)
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
            if !db.children(of: entry.id).isEmpty {
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
                db.removeWithChildren(id: entry.id)
                selectedIDs.remove(entry.id)
            }
        }
    }

    // MARK: - Child Entry Sub-Row

    @ViewBuilder
    private func childEntrySubRow(parent: LigandEntry, child: LigandEntry, index: Int) -> some View {
        let kindSymbol = child.variantKind == .tautomer ? "T" : "P"
        let kindColor: Color = child.variantKind == .tautomer ? .cyan : .mint
        let displayName = child.name

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
            Text(child.smiles)
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Relative energy
            if let energy = child.relativeEnergy, energy > 0 {
                Text(String(format: "+%.1f", energy))
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
            if child.conformerCount > 0 {
                Text("C:\(child.conformerCount)")
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 30, alignment: .trailing)
            }

            // Use for docking button
            if !child.atoms.isEmpty {
                Button {
                    let mol = Molecule(
                        name: displayName,
                        atoms: child.atoms,
                        bonds: child.bonds,
                        title: child.smiles,
                        smiles: child.smiles
                    )
                    viewModel.setLigandForDocking(mol, entryID: child.id)
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
            // Inspect child entry: update detail panel, conformers, and preview
            inspectVariantChild(child)
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

    @ViewBuilder
    private var statusBar: some View {
        let topLevel = db.topLevelEntries.count
        let totalWithChildren = db.count
        HStack(spacing: 16) {
            if totalWithChildren != topLevel {
                Text("Ligands: \(topLevel) (+\(totalWithChildren - topLevel) variants)")
                    .font(.system(size: 10))
            } else {
                Text("Total: \(topLevel)")
                    .font(.system(size: 10))
            }
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
        for id in selectedIDs {
            db.removeWithChildren(id: id)
        }
        selectedIDs.removeAll()
        viewModel.log.info("Deleted \(count) ligands (with children)", category: .molecule)
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

    func useEntryForDocking(_ entry: LigandEntry) {
        guard !entry.atoms.isEmpty else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare first", category: .molecule)
            return
        }
        let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds,
                           title: entry.smiles, smiles: entry.smiles)
        viewModel.setLigandForDocking(mol, entryID: entry.id)
    }

    private func addAndPrepare(smiles: String, name: String) {
        db.addFromSMILES(smiles, name: name)
        if let entry = db.entries.last, entry.smiles == smiles {
            prepareSingleEntry(entry)
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
