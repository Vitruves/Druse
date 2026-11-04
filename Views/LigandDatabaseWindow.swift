// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Ligand Database Manager Window

struct LigandDatabaseWindow: View {
    @Environment(AppViewModel.self) var viewModel

    // Selection state (shift+click multi-select)
    @State var selectedIDs: Set<UUID> = []
    @State var lastClickedID: UUID?

    // Search / filter
    @State var searchText: String = ""
    @State var lipinskiFilter: Bool = false
    @State var showOnlyPrepared: Bool = false

    /// Parent compounds whose enumerated forms are folded away in the table.
    @State var collapsedParents: Set<String> = []

    // Inline SMILES entry
    @State var showSMILESEntry: Bool = false
    @State var smilesInput: String = ""
    @State var nameInput: String = ""

    // Draw structure
    @State var showChemDrawer: Bool = false
    @State var editingEntryID: UUID? = nil

    // Scaffold-based analog generator
    @State var showScaffoldAnalogSheet: Bool = false

    // Inspector column + destructive confirmation
    @State var showInspector: Bool = true
    @State var showClearAllConfirmation: Bool = false

    // Sorting
    enum SortField: String { case name, smiles, mw, logP, hbd, hba, tpsa, rotB, atoms }
    @State var sortField: SortField? = nil
    @State var sortAscending: Bool = true

    /// Dynamic column manifest (built-in + user-imported). Persists order and
    /// visibility via @AppStorage.
    @State var columnState = LigandColumnState()

    /// Column id currently being dragged (header → header reorder).
    @State var draggedColumnID: String? = nil
    /// Column id the user is hovering over while dragging — used to draw a
    /// vertical insertion indicator on its leading edge.
    @State var dropTargetColumnID: String? = nil
    /// Horizontal distance the table has scrolled from its resting position —
    /// the leading checkbox cell of every row is shifted back by this amount so
    /// it stays pinned to the viewport's leading edge.
    @State var tableHorizontalScroll: CGFloat = 0

    // Inspector: selected entry for inspection
    @State var inspectedEntry: LigandEntry?

    // Inline rename
    @State var renamingEntryID: UUID? = nil
    @State var renamingText: String = ""

    // Import mapping sheet
    @State var importPreview: ImportPreview?
    @State var showImportMapping: Bool = false

    // Enumeration sheet
    @State var showEnumerationSheet: Bool = false
    @State var enumerationSheetEntries: [LigandEntry] = []

    // Conformer state (3D viewport in the inspector)
    @State var conformers: [ConformerEntry] = []
    @State var selectedConformerIndex: Int = 0

    struct ConformerEntry: Identifiable {
        let id: Int
        let molecule: MoleculeData
        let energy: Double
    }

    /// Width of the sticky leading checkbox gutter (padding + checkbox + padding).
    private let checkboxGutter: CGFloat = 40

    var db: LigandDatabase { viewModel.ligandDB }

    // Preparation state lives in the shared service (also driven by the console).
    var prep: LigandPreparationService { db.preparation }
    var isProcessing: Bool { prep.isRunning }
    /// A cancellable multi-ligand run (dominant or enumerate) rather than a single basic prepare.
    var isBatchProcessing: Bool { prep.isRunning && prep.runningMode != .basic }
    var batchProgress: (current: Int, total: Int) { prep.progress }
    var processingMessage: String { prep.message }

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
        // Folded forms stay hidden unless a search is looking for them.
        if searchText.isEmpty && !collapsedParents.isEmpty {
            entries = entries.filter { entry in
                entry.parentName.map { !collapsedParents.contains($0) } ?? true
            }
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

    private var selectedEntries: [LigandEntry] {
        db.entries.filter { selectedIDs.contains($0.id) }
    }

    var body: some View {
        mainContent
            .safeAreaInset(edge: .top, spacing: 0) { smilesEntryBar }
            .safeAreaInset(edge: .bottom, spacing: 0) { statusBar }
            .inspector(isPresented: $showInspector) {
                inspectorPane
                    .inspectorColumnWidth(min: 300, ideal: 360, max: 480)
            }
            .toolbar { windowToolbar }
            .searchable(text: $searchText, placement: .toolbar, prompt: "Filter by name or SMILES")
            .frame(minWidth: 1040, minHeight: 680)
            .onDeleteCommand {
                if !selectedIDs.isEmpty { deleteSelected() }
            }
            .confirmationDialog(
                "Remove all \(db.count) compounds from the library?",
                isPresented: $showClearAllConfirmation
            ) {
                Button("Clear All", role: .destructive) {
                    db.removeAll()
                    selectedIDs.removeAll()
                    inspectedEntry = nil
                    conformers = []
                }
            } message: {
                Text("This cannot be undone. Save the database first if you want to keep a copy.")
            }
            .sheet(isPresented: $showImportMapping) {
                ImportMappingSheet(preview: $importPreview) { finalPreview in
                    performMappedImport(finalPreview)
                }
            }
            .sheet(isPresented: $showEnumerationSheet) {
                let settings = Bindable(prep).settings
                EnumerationSheet(
                    ph: settings.pH,
                    pkaMethod: settings.pkaMethod,
                    pkaThreshold: settings.pkaThreshold,
                    numConformers: settings.conformersPerForm,
                    maxTautomers: settings.maxTautomers,
                    maxProtomers: settings.maxProtomers,
                    energyCutoff: settings.energyCutoff,
                    minPopulation: settings.minPopulationPercent,
                    entryCount: enumerationSheetEntries.count,
                    isProcessing: isProcessing,
                    progressMessage: processingMessage,
                    progressCurrent: batchProgress.current,
                    progressTotal: batchProgress.total,
                    onRun: {
                        runEnumerate(entries: enumerationSheetEntries)
                    },
                    onCancelRun: {
                        cancelPopulateAndPrepare()
                    }
                )
            }
            .sheet(isPresented: $showScaffoldAnalogSheet) {
                ScaffoldAnalogSheet()
                    .environment(viewModel)
            }
            .sheet(isPresented: $showChemDrawer) {
                let editEntry = editingEntryID.flatMap { id in db.entries.first { $0.id == id } }
                ChemDrawerSheet(
                    title: editEntry.map { "Edit Structure — \($0.name)" } ?? "Draw Molecule",
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
            .onAppear {
                // Auto-select first row when window opens
                if selectedIDs.isEmpty, let first = filteredEntries.first {
                    selectedIDs = [first.id]
                }
                // Discover any user-imported columns from the loaded DB
                columnState.discoverUserColumns(from: db.entries)
            }
            .onChange(of: db.entries.count) { _, _ in
                // New entries (e.g. from an import) may carry new
                // userProperties keys — register them as columns.
                columnState.discoverUserColumns(from: db.entries)
            }
            .onChange(of: selectedIDs) { _, newIDs in
                guard newIDs.count == 1, let id = newIDs.first,
                      let entry = db.entries.first(where: { $0.id == id }) else { return }
                inspectedEntry = entry
                selectedConformerIndex = 0
                conformers = entry.conformers.map { c in
                    ConformerEntry(
                        id: c.id,
                        molecule: MoleculeData(name: entry.name, title: entry.smiles,
                                               atoms: c.atoms, bonds: c.bonds),
                        energy: c.energy
                    )
                }
                // The 3D viewport updates itself via .onChange(of: atoms.count)
                // inside LigandMini3DView; the 2D depiction keys on SMILES.
            }
    }

    @ViewBuilder
    private var mainContent: some View {
        if db.entries.isEmpty {
            emptyState
        } else {
            tableView
        }
    }

    // MARK: - Toolbar

    private var enumerableSelection: [LigandEntry] {
        selectedEntries.filter { !$0.isEnumerated }
    }

    private var preparedSelectionCount: Int {
        selectedEntries.filter { $0.isPrepared && !db.hasEnumeratedChildren($0) }.count
    }

    @ToolbarContentBuilder
    private var windowToolbar: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Menu {
                addMenuItems
            } label: {
                Label("Add", systemImage: "plus")
            }
            .help("Add compounds from SMILES, a drawing or a file")
        }

        ToolbarItemGroup(placement: .primaryAction) {
            if isBatchProcessing {
                Button(action: cancelPopulateAndPrepare) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .help("Cancel preparation")
            } else {
                Button {
                    if !selectedEntries.isEmpty {
                        runPopulateAndPrepare(entries: selectedEntries)
                    } else if let entry = inspectedEntry {
                        runPopulateAndPrepare(entries: [entry])
                    }
                } label: {
                    Label(selectedEntries.count > 1 ? "Prepare \(selectedEntries.count)" : "Prepare",
                          systemImage: "wand.and.stars")
                }
                .labelStyle(.titleAndIcon)
                .disabled(selectedIDs.isEmpty || isProcessing)
                .help("Prepare the dominant form: add H → protonate at pH → minimize → charges → conformers")

                Button {
                    enumerationSheetEntries = enumerableSelection
                    showEnumerationSheet = !enumerationSheetEntries.isEmpty
                } label: {
                    Label("Enumerate…", systemImage: "arrow.triangle.branch")
                }
                .labelStyle(.titleAndIcon)
                .disabled(enumerableSelection.isEmpty || isProcessing)
                .help("Configure and run tautomer / protomer enumeration")
            }

            Button {
                showScaffoldAnalogSheet = true
            } label: {
                Label("Analogs…", systemImage: "atom")
            }
            .labelStyle(.titleAndIcon)
            .help("Generate analogs from a drawn scaffold (R-group decoration or whole-molecule transforms)")
        }

        ToolbarSpacer(.fixed, placement: .primaryAction)

        ToolbarItem(placement: .primaryAction) {
            Button(action: useSelectedForDocking) {
                Label(preparedSelectionCount > 1 ? "Dock \(preparedSelectionCount)" : "Use for Docking",
                      systemImage: "arrow.right.circle")
            }
            .labelStyle(.titleAndIcon)
            .disabled(preparedSelectionCount == 0)
            .help(preparedSelectionCount == 0
                  ? "Prepare the selected compounds first, then send them to docking"
                  : "Send the prepared compounds in the selection to docking")
        }

        ToolbarSpacer(.fixed, placement: .primaryAction)

        ToolbarItemGroup(placement: .primaryAction) {
            Menu {
                Toggle("Lipinski-compliant only", isOn: $lipinskiFilter)
                Toggle("Prepared only", isOn: $showOnlyPrepared)
            } label: {
                Label("Filter",
                      systemImage: lipinskiFilter || showOnlyPrepared
                        ? "line.3.horizontal.decrease.circle.fill"
                        : "line.3.horizontal.decrease.circle")
            }
            .help("Filter the table")

            Menu {
                Button("Export All as SDF…") { db.exportSDF() }
                    .disabled(db.entries.isEmpty)
                Divider()
                Button("Save Database") {
                    db.save()
                    viewModel.log.success("Saved \(db.count) entries", category: .molecule)
                }
                Button("Load Saved Database") { db.load() }
                Divider()
                Button(selectedIDs.count > 1 ? "Delete \(selectedIDs.count) Compounds" : "Delete Selected",
                       role: .destructive, action: deleteSelected)
                    .disabled(selectedIDs.isEmpty)
                Button("Clear All…", role: .destructive) { showClearAllConfirmation = true }
                    .disabled(db.entries.isEmpty)
            } label: {
                Label("Library", systemImage: "ellipsis.circle")
            }
            .help("Export, save or load the library; delete compounds (⌫)")
        }

        ToolbarItem(placement: .primaryAction) {
            Button {
                showInspector.toggle()
            } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
            }
            .help(showInspector ? "Hide Inspector" : "Show Inspector")
        }
    }

    /// Shared items for the toolbar Add menu and the empty-state Add menu.
    @ViewBuilder
    private var addMenuItems: some View {
        Button("SMILES…", systemImage: "character.cursor.ibeam") { showSMILESEntry = true }
        Button("Draw Structure…", systemImage: "pencil.and.outline") {
            editingEntryID = nil
            showChemDrawer = true
        }
        Divider()
        Section("Import File with Column Mapping") {
            Button("SMILES List (.smi)…") { openImportWithMapping(.smi) }
            Button("CSV Table (.csv)…") { openImportWithMapping(.csv) }
            Button("Structure File (.sdf)…") { openImportWithMapping(.sdf) }
        }
    }

    // MARK: - SMILES Entry (inline bar under the toolbar)

    @ViewBuilder
    var smilesEntryBar: some View {
        if showSMILESEntry {
            VStack(spacing: 0) {
                HStack(spacing: DS.Space.m) {
                    Image(systemName: "character.cursor.ibeam")
                        .foregroundStyle(.secondary)
                    PastableTextField(
                        text: $smilesInput,
                        placeholder: "SMILES",
                        font: .monospacedSystemFont(ofSize: NSFont.systemFontSize(for: .small), weight: .regular),
                        onSubmit: submitSMILESEntry
                    )

                    TextField("Name (optional)", text: $nameInput)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 160)

                    Button("Add") {
                        let name = nameInput.isEmpty ? "Ligand_\(db.count + 1)" : nameInput
                        db.addFromSMILES(smilesInput, name: name)
                        smilesInput = ""
                        nameInput = ""
                    }
                    .disabled(smilesInput.isEmpty)
                    .help("Add without preparing")

                    Button("Add & Prepare", action: submitSMILESEntry)
                        .buttonStyle(.borderedProminent)
                        .disabled(smilesInput.isEmpty)
                        .help("Add and generate 3D coordinates (↩)")

                    Button {
                        showSMILESEntry = false
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .help("Close")
                    .accessibilityLabel("Close SMILES entry")
                }
                .controlSize(.small)
                .font(DS.Font.label)
                .padding(.horizontal, DS.Space.l)
                .padding(.vertical, DS.Space.m)
                Divider()
            }
            .background(.bar)
        }
    }

    private func submitSMILESEntry() {
        guard !smilesInput.isEmpty else { return }
        let name = nameInput.isEmpty ? "Ligand_\(db.count + 1)" : nameInput
        addAndPrepare(smiles: smilesInput, name: name)
        smilesInput = ""
        nameInput = ""
    }

    // MARK: - Table

    private var tableBackground: Color { Color(nsColor: .controlBackgroundColor) }

    private var tableView: some View {
        // 2D scrolling table: horizontal for column overflow, vertical for
        // row overflow. Header is pinned via Section so it stays visible
        // while rows scroll vertically and tracks the rows when they
        // scroll horizontally. Each row owns its checkbox cell (so vertical
        // alignment is structural); the cell is counter-offset by the
        // horizontal scroll distance to stay pinned on the leading edge.
        GeometryReader { geo in
            let widths = computeColumnWidths(viewportWidth: geo.size.width - DS.Space.xl - checkboxGutter)
            let viewportHeight = geo.size.height
            ScrollView([.horizontal, .vertical], showsIndicators: true) {
                LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                    Section {
                        ForEach(filteredEntries) { entry in
                            tableRow(entry, widths: widths)
                            Divider().opacity(0.5)
                        }
                    } header: {
                        tableHeaderRow(widths: widths)
                    }
                }
                // Claim at least the viewport height with top alignment so
                // short tables don't float in the middle of the viewport.
                .frame(minHeight: viewportHeight, alignment: .topLeading)
                .background(alignment: .topLeading) {
                    columnSeparatorOverlay(widths: widths)
                }
            }
            .onScrollGeometryChange(for: CGFloat.self) { geo in
                // contentOffset is measured from the content origin, so a
                // leading content inset makes the resting offset negative.
                geo.contentOffset.x + geo.contentInsets.leading
            } action: { _, newValue in
                tableHorizontalScroll = newValue
            }
        }
        .background(tableBackground)
    }

    /// Row tint for selection / inspection / active / enumeration state.
    private func rowBackground(_ entry: LigandEntry) -> Color {
        if inspectedEntry?.id == entry.id { return DS.Color.accent.opacity(0.22) }
        if selectedIDs.contains(entry.id) { return DS.Color.accentSoft }
        if viewModel.activeLigandEntryID == entry.id { return DS.Color.ligand.opacity(0.08) }
        if db.hasEnumeratedChildren(entry) { return DS.Color.controlFill }
        if entry.isEnumerated { return DS.Color.cardFill }
        return .clear
    }

    private func checkboxImage(isOn: Bool) -> some View {
        Image(systemName: isOn ? "checkmark.square.fill" : "square")
            .font(DS.Font.body)
            .foregroundStyle(isOn ? DS.Color.accent : Color.secondary)
    }

    /// Header checkbox: select all / none of the selectable (non-parent) rows.
    @ViewBuilder
    private var headerCheckboxCell: some View {
        let selectableIDs = Set(filteredEntries.lazy
            .filter { !db.hasEnumeratedChildren($0) }
            .map(\.id))
        let allSelected = !selectableIDs.isEmpty && selectableIDs.isSubset(of: selectedIDs)

        Button(action: toggleSelectAll) {
            checkboxImage(isOn: allSelected)
                .frame(width: checkboxGutter, height: DS.Size.rowHeight + DS.Space.s)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background { stickyCellBackground(Color.clear, material: true) }
        .offset(x: tableHorizontalScroll)
        .zIndex(1)
        .help("Select all / none")
        .accessibilityLabel(allSelected ? "Deselect all" : "Select all")
    }

    /// Per-row checkbox, pinned to the leading edge during horizontal scroll.
    @ViewBuilder
    private func rowCheckboxCell(_ entry: LigandEntry) -> some View {
        let isOn = selectedIDs.contains(entry.id)
        Button {
            if isOn { selectedIDs.remove(entry.id) } else { selectedIDs.insert(entry.id) }
        } label: {
            checkboxImage(isOn: isOn)
                .frame(width: checkboxGutter, height: DS.Size.rowHeight)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background { stickyCellBackground(rowBackground(entry), material: false) }
        .offset(x: tableHorizontalScroll)
        .zIndex(1)
        .accessibilityLabel(isOn ? "Deselect \(entry.name)" : "Select \(entry.name)")
    }

    /// Opaque backing for a pinned checkbox cell so scrolled-under cells don't
    /// show through, plus the trailing hairline separating it from the columns.
    private func stickyCellBackground(_ tint: Color, material: Bool) -> some View {
        ZStack {
            tableBackground
            if material { Rectangle().fill(.bar) }
            tint
        }
        .overlay(alignment: .trailing) {
            // Explicit 0.5 pt rule: a Divider here would default to horizontal.
            Rectangle()
                .fill(DS.Color.hairline)
                .frame(width: 0.5)
        }
    }

    /// Continuous vertical separator lines drawn at every column boundary,
    /// from the top of the table to the bottom.
    @ViewBuilder
    private func columnSeparatorOverlay(widths: [String: CGFloat]) -> some View {
        let gutter = checkboxGutter
        let columns = columnState.visibleColumns
        Canvas { context, size in
            // Leading edge: outer padding + sticky checkbox gutter
            var x: CGFloat = gutter + DS.Space.m
            let lineColor = GraphicsContext.Shading.color(DS.Color.hairline)
            for col in columns {
                x += widths[col.id] ?? 60
                var path = Path()
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: size.height))
                context.stroke(path, with: lineColor, lineWidth: 0.5)
            }
        }
        .allowsHitTesting(false)
    }

    /// Compute per-column display widths for the current viewport.
    /// - Built-in columns with a fixed `defaultWidth` use that.
    /// - Flexible columns (SMILES) split the leftover viewport space, with a
    ///   minimum of `minFlex`. When the total exceeds the viewport, the table
    ///   overflows horizontally and the user can scroll.
    private func computeColumnWidths(viewportWidth: CGFloat) -> [String: CGFloat] {
        let cols = columnState.visibleColumns
        var widths: [String: CGFloat] = [:]
        var fixedTotal: CGFloat = 0
        var flexIDs: [String] = []
        let minFlex: CGFloat = 240

        for col in cols {
            if let w = col.kind.defaultWidth {
                widths[col.id] = w
                fixedTotal += w
            } else {
                flexIDs.append(col.id)
            }
        }

        if flexIDs.isEmpty { return widths }

        let leftover = viewportWidth - fixedTotal
        let perFlex = leftover / CGFloat(flexIDs.count)
        for id in flexIDs {
            widths[id] = max(minFlex, perFlex)
        }
        return widths
    }

    /// Header row — dynamic, driven by columnState.visibleColumns. Widths
    /// match the row cells. Drag-and-drop reorder with a custom preview chip
    /// and leading-edge insertion indicator while dragging. The leading cell
    /// is the pinned select-all checkbox.
    @ViewBuilder
    private func tableHeaderRow(widths: [String: CGFloat]) -> some View {
        HStack(spacing: 0) {
            headerCheckboxCell
            HStack(spacing: 0) {
                ForEach(columnState.visibleColumns) { col in
                    headerCellWithDrag(col: col, width: widths[col.id])
                }
            }
            .padding(.horizontal, DS.Space.m)
        }
        .frame(height: DS.Size.rowHeight + DS.Space.s)
        .background(.bar)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    /// One header cell wrapped in drag/drop modifiers. The drag preview is an
    /// accent chip showing the column name, and the drop indicator is a 2-pt
    /// accent bar on the leading edge of the hovered column.
    @ViewBuilder
    private func headerCellWithDrag(col: LigandColumn, width: CGFloat?) -> some View {
        let isDragSource = draggedColumnID == col.id
        let isDropTarget = dropTargetColumnID == col.id && draggedColumnID != nil && draggedColumnID != col.id

        headerCell(
            col.kind.title,
            width: width,
            field: col.kind.sortField,
            alignment: col.kind.alignment
        )
        .opacity(isDragSource ? 0.35 : 1.0)
        .overlay(alignment: .leading) {
            if isDropTarget {
                Rectangle()
                    .fill(DS.Color.accent)
                    .frame(width: 2)
                    .frame(maxHeight: .infinity)
            }
        }
        .contextMenu {
            columnHeaderMenu(for: col)
        }
        .onDrag({
            draggedColumnID = col.id
            return NSItemProvider(object: col.id as NSString)
        }, preview: {
            Text(col.kind.title)
                .font(DS.Font.secondary.weight(.semibold))
                .foregroundStyle(DS.Color.accentOnFill)
                .padding(.horizontal, DS.Space.m)
                .padding(.vertical, DS.Space.xs)
                .background(Capsule().fill(DS.Color.accent))
        })
        .onDrop(
            of: [.text],
            delegate: ColumnDropDelegate(
                targetID: col.id,
                columnState: columnState,
                onDragEnter: { dropTargetColumnID = col.id },
                onDragExit: { if dropTargetColumnID == col.id { dropTargetColumnID = nil } },
                onDrop: {
                    draggedColumnID = nil
                    dropTargetColumnID = nil
                }
            )
        )
    }

    @ViewBuilder
    private func headerCell(_ title: String, width: CGFloat?, field: SortField?, alignment: Alignment = .leading) -> some View {
        let isSorted = field != nil && sortField == field
        // Whole cell is hit-testable so drag works on empty space inside the
        // cell, not just on the title text. Sort fires via tap on the whole frame.
        ZStack(alignment: alignment) {
            Color.clear
            HStack(spacing: DS.Space.xxs) {
                Text(title)
                if isSorted {
                    Image(systemName: sortAscending ? "chevron.up" : "chevron.down")
                        .font(DS.Font.caption.weight(.bold))
                        .foregroundStyle(DS.Color.accent)
                }
            }
            .padding(.horizontal, DS.Space.xs)
        }
        .font(DS.Font.caption.weight(.semibold))
        .foregroundStyle(isSorted ? .primary : .secondary)
        .lineLimit(1)
        .frame(width: width, alignment: alignment)
        .frame(maxWidth: width == nil ? .infinity : nil, alignment: alignment)
        .contentShape(Rectangle())
        .onTapGesture {
            guard let field else { return }
            if sortField == field {
                sortAscending.toggle()
            } else {
                sortField = field
                sortAscending = true
            }
        }
        .accessibilityAddTraits(field != nil ? .isButton : [])
        .help(field != nil ? "Sort by \(title) · drag to reorder · right-click for columns" : "Drag to reorder · right-click for columns")
    }

    // MARK: - Table Row (flat — one row per entry)

    @ViewBuilder
    private func tableRow(_ entry: LigandEntry, widths: [String: CGFloat]) -> some View {
        let isActive = viewModel.activeLigandEntryID == entry.id
        let isExpandedParent = db.hasEnumeratedChildren(entry)

        HStack(spacing: 0) {
            rowCheckboxCell(entry)

            // Dynamic columns. Vertical separators between columns are drawn
            // by the table-level column separator overlay (not per cell).
            HStack(spacing: 0) {
                ForEach(columnState.visibleColumns) { col in
                    cellView(for: col, entry: entry,
                             width: widths[col.id],
                             isActive: isActive, isExpandedParent: isExpandedParent)
                }
            }
            .padding(.horizontal, DS.Space.m)
        }
        .frame(height: DS.Size.rowHeight)
        .background(rowBackground(entry))
        .overlay(alignment: .leading) {
            // Ligand-coloured edge marks the active docking ligand, pinned like the checkbox.
            if isActive {
                Rectangle()
                    .fill(DS.Color.ligand)
                    .frame(width: 3)
                    .offset(x: tableHorizontalScroll)
                    .allowsHitTesting(false)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            // Selection update only — the .onChange(of: selectedIDs) handler
            // is the single source of truth for inspectedEntry and conformers.
            handleRowClick(entry.id, shiftKey: NSEvent.modifierFlags.contains(.shift),
                           cmdKey: NSEvent.modifierFlags.contains(.command))
        }
        .contextMenu {
            Button("Use for Docking") { useEntryForDocking(entry) }
                .disabled(entry.atoms.isEmpty || isExpandedParent)
            Button("Prepare") { prepareSingleEntry(entry) }
                .disabled(entry.smiles.isEmpty || isProcessing)
            Button("Edit Structure…") {
                editingEntryID = entry.id
                showChemDrawer = true
            }
            .disabled(entry.smiles.isEmpty)
            Button("Rename") {
                renamingEntryID = entry.id
                renamingText = entry.name
            }
            Button("Copy SMILES") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(entry.originalSMILES, forType: .string)
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

    // MARK: - Cell Dispatch

    /// Render a single cell for `entry` in column `col`. Width is supplied by
    /// the caller (computed from viewport size); the contents are switched per
    /// kind. The Name column has the special inline-rename + form-badge behavior.
    @ViewBuilder
    private func cellView(
        for col: LigandColumn,
        entry: LigandEntry,
        width: CGFloat?,
        isActive: Bool,
        isExpandedParent: Bool
    ) -> some View {
        let alignment = col.kind.alignment

        Group {
            switch col.kind {
            case .name:
                nameCell(entry, isActive: isActive, isExpandedParent: isExpandedParent)
            case .smiles:
                Text(entry.originalSMILES)
                    .font(DS.Font.monoSmall)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(entry.originalSMILES)
            case .popPercent:
                if let pop = entry.populationWeight {
                    // Whole percents above 10 % keep "100%" inside the narrow column.
                    Text(String(format: pop >= 0.0995 ? "%.0f%%" : "%.1f%%", pop * 100))
                        .font(DS.Font.secondary.weight(.medium))
                        .monospacedDigit()
                        .foregroundStyle(Self.populationTone(pop).color)
                        .help(String(format: "Boltzmann population %.1f %% at the target pH", pop * 100))
                } else {
                    dashCell
                }
            case .deltaE:
                if let dE = entry.relativeEnergy {
                    // Numbers only in a numeric column; the lowest form reads 0.0 in the success tone.
                    Text(dE < 0.01 ? "0.0" : String(format: "+%.1f", dE))
                        .font(DS.Font.secondary)
                        .monospacedDigit()
                        .foregroundStyle(dE < 0.01 ? DS.Color.success : DS.Color.warning)
                        .help(dE < 0.01 ? "Lowest-energy form"
                              : String(format: "%.1f kcal/mol above the lowest-energy form", dE))
                } else {
                    dashCell
                }
            case .conf:
                if entry.conformerCount > 0 {
                    numberText("\(entry.conformerCount)")
                } else {
                    dashCell
                }
            case .mw:
                if let d = entry.descriptors { numberText(String(format: "%.0f", d.molecularWeight)) } else { dashCell }
            case .logP:
                if let d = entry.descriptors { numberText(d.logPText) } else { dashCell }
            case .hbd:
                if let d = entry.descriptors { numberText("\(d.hbd)") } else { dashCell }
            case .hba:
                if let d = entry.descriptors { numberText("\(d.hba)") } else { dashCell }
            case .tpsa:
                if let d = entry.descriptors { numberText(String(format: "%.0f", d.tpsa)) } else { dashCell }
            case .rotB:
                if let d = entry.descriptors { numberText("\(d.rotatableBonds)") } else { dashCell }
            case .lipinski:
                if let d = entry.descriptors {
                    Image(systemName: d.lipinski ? "checkmark" : "xmark")
                        .font(DS.Font.caption.weight(.semibold))
                        .foregroundStyle(d.lipinski ? DS.Color.success : DS.Color.danger)
                        .accessibilityLabel(d.lipinski ? "Lipinski pass" : "Lipinski fail")
                } else { dashCell }
            case .atoms:
                if entry.atoms.isEmpty { dashCell } else { numberText("\(entry.atoms.count)") }
            case .userProperty(let key):
                if let value = entry.userProperties[key], !value.isEmpty {
                    Text(value)
                        .font(DS.Font.monoSmall)
                        .lineLimit(1)
                        .help(value)
                } else {
                    dashCell
                }
            }
        }
        .padding(.horizontal, DS.Space.xs)
        .frame(width: width, alignment: alignment)
        .frame(maxWidth: width == nil ? .infinity : nil, alignment: alignment)
        .clipped()
    }

    private func numberText(_ value: String) -> some View {
        Text(value)
            .font(DS.Font.secondary)
            .monospacedDigit()
    }

    /// Em-dash placeholder used by every cell when its data is missing.
    private var dashCell: some View {
        Text("—").font(DS.Font.secondary).foregroundStyle(.tertiary)
    }

    /// The Name column cell: fold chevron for compounds with enumerated forms, an
    /// indent for the forms themselves, the form-kind badge, inline rename and the
    /// active-docking-ligand marker.
    @ViewBuilder
    private func nameCell(_ entry: LigandEntry, isActive: Bool, isExpandedParent: Bool) -> some View {
        HStack(spacing: DS.Space.xs) {
            if isExpandedParent {
                let isCollapsed = collapsedParents.contains(entry.name)
                Button {
                    withAnimation(DS.Motion.snappy) {
                        if isCollapsed { collapsedParents.remove(entry.name) } else { collapsedParents.insert(entry.name) }
                    }
                } label: {
                    Image(systemName: "chevron.right")
                        .font(DS.Font.caption.weight(.semibold))
                        .rotationEffect(.degrees(isCollapsed ? 0 : 90))
                        .foregroundStyle(.secondary)
                        .frame(width: Self.disclosureWidth, height: DS.Size.rowHeight)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isCollapsed ? "Show the enumerated forms" : "Hide the enumerated forms")
                .accessibilityLabel(isCollapsed ? "Show forms of \(entry.name)" : "Hide forms of \(entry.name)")
            } else {
                // Children sit one step in from their parent; plain compounds keep the parents' text column.
                Color.clear.frame(width: entry.isEnumerated ? Self.disclosureWidth * 2 : Self.disclosureWidth, height: 1)
            }
            if let kind = entry.formKind {
                DSTag(kind.symbol, tint: Self.formKindColor(kind))
                    .help(kind.label)
            }
            if renamingEntryID == entry.id {
                TextField("Name", text: $renamingText, onCommit: {
                    commitRename(entryID: entry.id)
                })
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .onExitCommand { renamingEntryID = nil }
            } else {
                Text(entry.name)
                    .font(DS.Font.secondary.weight(isActive || isExpandedParent ? .semibold : .regular))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .layoutPriority(1)
                    .help(entry.name)
                    .onTapGesture(count: 2) {
                        renamingEntryID = entry.id
                        renamingText = entry.name
                    }
            }
            if isActive {
                Image(systemName: "scope")
                    .font(DS.Font.caption.weight(.semibold))
                    .foregroundStyle(DS.Color.ligand)
                    .help("Active docking ligand")
                    .accessibilityLabel("Active docking ligand")
            }
            if entry.preparationError != nil {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(DS.Font.caption)
                    .foregroundStyle(DS.Color.danger)
                    .help(entry.preparationError ?? "")
            }
        }
    }

    /// Width of the fold chevron in the Name column (and of one indent step).
    private static let disclosureWidth: CGFloat = 12

    // MARK: - Column Header Context Menu

    /// Right-click menu on any column header: hide this column, toggle other
    /// hideable columns, reset to defaults.
    @ViewBuilder
    private func columnHeaderMenu(for col: LigandColumn) -> some View {
        if col.kind.isHideable {
            Button("Hide \"\(col.kind.title)\"") {
                columnState.setVisible(false, id: col.id)
            }
            Divider()
        }

        Menu("Show / Hide Columns") {
            ForEach(columnState.hideableColumns) { hideable in
                Toggle(hideable.kind.title, isOn: Binding(
                    get: { hideable.visible },
                    set: { _ in columnState.toggleVisible(id: hideable.id) }
                ))
            }
        }

        Divider()

        Button("Reset Columns to Defaults") {
            columnState.resetToDefaults()
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        DSEmptyState(
            "No Compounds Yet",
            systemImage: "hexagon",
            description: "Add a SMILES string, draw a structure, or import .smi, .csv or .sdf files."
        ) {
            HStack(spacing: DS.Space.m) {
                Button {
                    showSMILESEntry = true
                } label: {
                    Label("Add SMILES", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)

                Button {
                    editingEntryID = nil
                    showChemDrawer = true
                } label: {
                    Label("Draw Structure…", systemImage: "pencil.and.outline")
                }

                Menu("Import File") {
                    Button("SMILES List (.smi)…") { openImportWithMapping(.smi) }
                    Button("CSV Table (.csv)…") { openImportWithMapping(.csv) }
                    Button("Structure File (.sdf)…") { openImportWithMapping(.sdf) }
                }
                .fixedSize()
            }
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(tableBackground)
    }

    // MARK: - Status Bar

    @ViewBuilder
    private var statusBar: some View {
        let entries = filteredEntries
        let nPrepared = entries.filter(\.isPrepared).count
        let nEnumerated = entries.filter(\.isEnumerated).count
        let nConformers = entries.reduce(0) { $0 + $1.conformerCount }
        let nSelected = selectedIDs.count
        let isFiltered = !searchText.isEmpty || lipinskiFilter || showOnlyPrepared
        let isBusy = isProcessing || isBatchProcessing || db.isProcessing

        DSWindowStatusBar {
            Text(isFiltered ? "\(entries.count) of \(db.count) compounds"
                 : entries.count == 1 ? "1 compound" : "\(entries.count) compounds")
            if nPrepared > 0 {
                statusSeparator
                Label("\(nPrepared) prepared", systemImage: "checkmark.circle.fill")
                    .labelStyle(StatusLabelStyle(tint: DS.Color.success))
            }
            if nEnumerated > 0 {
                statusSeparator
                Text("\(nEnumerated) enumerated forms")
            }
            if nConformers > 0 {
                statusSeparator
                Text("\(nConformers) conformers")
            }
        } trailing: {
            if isBusy {
                HStack(spacing: DS.Space.s) {
                    ProgressView().controlSize(.mini)
                    Text(statusProgressText)
                        .truncationMode(.middle)
                    if isBatchProcessing {
                        Button(action: cancelPopulateAndPrepare) {
                            Image(systemName: "xmark.circle.fill")
                        }
                        .buttonStyle(.borderless)
                        .help("Cancel preparation")
                        .accessibilityLabel("Cancel preparation")
                    }
                }
                if nSelected > 0 || viewModel.molecules.ligand != nil { statusSeparator }
            }
            if nSelected > 0 {
                Text("\(nSelected) selected")
                    .foregroundStyle(DS.Color.accent)
            }
            if let lig = viewModel.molecules.ligand {
                if nSelected > 0 { statusSeparator }
                Label("Docking ligand: \(lig.name)", systemImage: "scope")
                    .labelStyle(StatusLabelStyle(tint: DS.Color.ligand))
            }
        }
    }

    private var statusProgressText: String {
        if db.isProcessing && !db.processingMessage.isEmpty { return db.processingMessage }
        if isBatchProcessing {
            return processingMessage.isEmpty
                ? "Processing \(batchProgress.current)/\(batchProgress.total)"
                : processingMessage
        }
        return processingMessage.isEmpty ? "Working…" : processingMessage
    }

    private var statusSeparator: some View {
        Text("·").foregroundStyle(.tertiary)
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
        // Exclude grayed-out parent rows that have enumerated children — those
        // would duplicate downstream work, and the table grays them visually.
        let selectable = filteredEntries.filter { !db.hasEnumeratedChildren($0) }
        let selectableIDs = Set(selectable.map(\.id))
        if selectableIDs.isSubset(of: selectedIDs) && !selectableIDs.isEmpty {
            selectedIDs.subtract(selectableIDs)
        } else {
            selectedIDs.formUnion(selectableIDs)
        }
    }

    // MARK: - Actions

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

    /// Basic RDKit preparation of one entry (hydrogens, MMFF94, Gasteiger charges, one conformer).
    func prepareSingleEntry(_ entry: LigandEntry) {
        let run = prep.prepare([entry], mode: .basic)
        Task {
            _ = await run.value
            refreshInspectedEntry()
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

    // MARK: - Prepare (Dominant Form) / Enumerate

    /// Prepare entries: add H → adjust protonation at target pH → minimize → charges → conformers,
    /// keeping the dominant (most populated) form of each, in place.
    func runPopulateAndPrepare(entries: [LigandEntry]) {
        let run = prep.prepare(entries, mode: .dominant)
        Task {
            _ = await run.value
            refreshInspectedEntry()
        }
    }

    /// Enumerate entries: full tautomer + protomer expansion at target pH. Inserts one new row
    /// per form after the original entry (which is kept as-is), then closes the sheet.
    func runEnumerate(entries: [LigandEntry]) {
        let run = prep.prepare(entries, mode: .enumerate)
        Task {
            _ = await run.value
            refreshInspectedEntry()
            if showEnumerationSheet {
                showEnumerationSheet = false
                enumerationSheetEntries = []
            }
        }
    }

    func cancelPopulateAndPrepare() {
        prep.cancel()
    }

    /// Re-reads the inspected entry after a preparation wrote new coordinates or forms.
    private func refreshInspectedEntry() {
        if let id = inspectedEntry?.id, let refreshed = db.entries.first(where: { $0.id == id }) {
            inspectedEntry = refreshed
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

// MARK: - Status bar label style

/// Small tinted icon + secondary title, sized for `DSWindowStatusBar`.
private struct StatusLabelStyle: LabelStyle {
    let tint: Color

    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: DS.Space.xs) {
            configuration.icon
                .font(DS.Font.caption)
                .imageScale(.small)
                .foregroundStyle(tint)
            configuration.title
        }
    }
}
