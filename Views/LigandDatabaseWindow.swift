// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Ligand Database Manager Window

struct LigandDatabaseWindow: View {
    @Environment(AppViewModel.self) var viewModel

    // Table selection (native click / ⇧-click / ⌘-click / ⌘A)
    @State var selectedIDs: Set<UUID> = []

    // Search / filter
    @State var searchText: String = ""
    @State var lipinskiFilter: Bool = false
    @State var showOnlyPrepared: Bool = false

    /// Parent compounds whose enumerated forms are folded away in the table.
    @State var collapsedParents: Set<String> = []

    /// Sort columns, most significant first. Empty keeps library order.
    @State var sortOrder: [TableSort<LigandEntry>] = []
    /// Column order, widths and visibility (header right-click menu), persisted.
    @State var columnCustomization: TableColumnCustomization<LigandEntry> = TableColumnStore.load(Self.columnStoreKey)
    private static let columnStoreKey = "ui.ligandTable.columns"

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

    var db: LigandDatabase { viewModel.ligandDB }

    // Preparation state lives in the shared service (also driven by the console).
    var prep: LigandPreparationService { db.preparation }
    var isProcessing: Bool { prep.isRunning }
    /// A cancellable multi-ligand run (dominant or enumerate) rather than a single basic prepare.
    var isBatchProcessing: Bool { prep.isRunning && prep.runningMode != .basic }
    var batchProgress: (current: Int, total: Int) { prep.progress }
    var processingMessage: String { prep.message }

    var isFiltered: Bool { !searchText.isEmpty || lipinskiFilter || showOnlyPrepared }

    /// Filtered entries (search + Lipinski + prepared-only filters), in table sort order.
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
        guard !sortOrder.isEmpty else { return entries }
        // Rebuild each comparator from the current data: a stored comparator
        // captured the docking results of the moment its header was clicked.
        let comparators = sortOrder.map { sorter($0.id, order: $0.order) }
        return entries.sorted(using: comparators)
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
            }
            .onChange(of: columnCustomization) { _, newValue in
                TableColumnStore.save(newValue, key: Self.columnStoreKey)
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
            ligandTable
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

    /// A top-level row and the enumerated forms folded under it.
    struct LigandTableNode: Identifiable {
        let entry: LigandEntry
        let forms: [LigandEntry]
        var id: UUID { entry.id }
    }

    /// Filtered, sorted entries grouped as parent → forms. A form whose parent
    /// is filtered out (or deleted) is shown at the top level.
    private func tableNodes(_ entries: [LigandEntry]) -> [LigandTableNode] {
        let shownNames = Set(entries.map(\.name))
        var forms: [String: [LigandEntry]] = [:]
        for entry in entries {
            if let parent = entry.parentName, parent != entry.name, shownNames.contains(parent) {
                forms[parent, default: []].append(entry)
            }
        }
        return entries.compactMap { entry in
            if let parent = entry.parentName, parent != entry.name, shownNames.contains(parent) { return nil }
            return LigandTableNode(entry: entry, forms: forms[entry.name] ?? [])
        }
    }

    /// Keys of user-imported properties, one column each.
    private var userPropertyKeys: [String] {
        Array(Set(db.entries.flatMap(\.userProperties.keys))).sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }

    /// The comparator of column `id`, built from the current data.
    func sorter(_ id: String, order: SortOrder = .forward) -> TableSort<LigandEntry> {
        let best = viewModel.bestDockingResultByLigand
        let method = viewModel.docking.scoringMethod
        let key: @Sendable (LigandEntry) -> TableSortValue
        switch id {
        case "name":      key = { .text($0.name) }
        case "status":    key = { .number(LigandPrepStatus($0).sortRank) }
        case "smiles":    key = { .text($0.originalSMILES) }
        case "score":     key = { .number(best[$0.name]?.displayScore(method: method)) }
        case "le":        key = { .number(Self.ligandEfficiency($0, best: best[$0.name], method: method)) }
        case "mw":        key = { .number($0.descriptors?.molecularWeight) }
        case "logP":      key = { .number($0.descriptors?.logP) }
        case "tpsa":      key = { .number($0.descriptors?.tpsa) }
        case "hbd":       key = { .number($0.descriptors?.hbd) }
        case "hba":       key = { .number($0.descriptors?.hba) }
        case "rotB":      key = { .number($0.descriptors?.rotatableBonds) }
        case "heavy":     key = { .number($0.descriptors?.heavyAtomCount) }
        case "fsp3":      key = { .number($0.descriptors?.fractionCSP3) }
        case "ro5":       key = { .number($0.descriptors.map { $0.lipinski ? 1 : 0 }) }
        case "pKi":       key = { .number($0.effectivePKi) }
        case "lle":       key = { .number(Self.lipophilicEfficiency($0)) }
        case "pop":       key = { .number($0.populationWeight) }
        case "deltaE":    key = { .number($0.relativeEnergy) }
        case "conformers": key = { .number($0.conformerCount > 0 ? $0.conformerCount : nil) }
        default:
            // "user.<key>" — imported property, numeric when it parses.
            let property = String(id.dropFirst("user.".count))
            key = { .parsed($0.userProperties[property]) }
        }
        return TableSort(id, order: order, key: key)
    }

    private nonisolated static func ligandEfficiency(_ entry: LigandEntry, best: DockingResult?, method: ScoringMethod) -> Float? {
        guard let best else { return nil }
        let heavyAtoms = entry.descriptors?.heavyAtomCount ?? best.transformedAtomPositions.count
        return LigandMetrics.ligandEfficiency(score: best.displayScore(method: method), method: method, heavyAtoms: heavyAtoms)
    }

    private nonisolated static func lipophilicEfficiency(_ entry: LigandEntry) -> Float? {
        guard let pK = entry.effectivePKi, let d = entry.descriptors else { return nil }
        return LigandMetrics.lipophilicEfficiency(pK: pK, logP: d.logP)
    }

    @ViewBuilder
    private var ligandTable: some View {
        let entries = filteredEntries
        let nodes = tableNodes(entries)
        let best = viewModel.bestDockingResultByLigand
        let method = viewModel.docking.scoringMethod
        // Context columns appear only when the library has that kind of data.
        let hasDocking = db.entries.contains { best[$0.name] != nil }
        let hasAffinity = db.entries.contains { $0.effectivePKi != nil }
        let hasForms = db.entries.contains(where: \.isEnumerated)

        Table(of: LigandEntry.self,
              selection: $selectedIDs,
              sortOrder: $sortOrder,
              columnCustomization: $columnCustomization) {
            Group {
                TableColumn("Name", sortUsing: sorter("name")) { entry in
                    nameCell(entry)
                }
                .width(min: 140, ideal: 200)
                .customizationID("name")
                .disabledCustomizationBehavior(.visibility)

                TableColumn("Status", sortUsing: sorter("status")) { entry in
                    LigandPrepStatusCell(entry: entry)
                }
                .width(min: 80, ideal: 100, max: 130)
                .customizationID("status")

                TableColumn("SMILES", sortUsing: sorter("smiles")) { entry in
                    Text(entry.originalSMILES)
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .help(entry.originalSMILES)
                }
                .width(min: 120, ideal: 240)
                .customizationID("smiles")

                if hasDocking {
                    TableColumn(method.scoreColumnTitle, sortUsing: sorter("score")) { entry in
                        TableScoreCell(value: best[entry.name]?.displayScore(method: method), method: method)
                            .help("Best docking score of this compound")
                    }
                    .width(min: 70, ideal: 100, max: 130)
                    .alignment(.trailing)
                    .customizationID("score")

                    TableColumn("LE", sortUsing: sorter("le")) { entry in
                        let le = Self.ligandEfficiency(entry, best: best[entry.name], method: method)
                        TableNumberCell(text: le.map(TableFormat.ligandEfficiency),
                                        tone: le.map { DSScore.tone(ligandEfficiency: $0) },
                                        help: "Ligand efficiency of the best pose (kcal/mol per heavy atom)")
                    }
                    .width(min: 40, ideal: 50, max: 70)
                    .alignment(.trailing)
                    .customizationID("le")
                }
            }

            Group {
                descriptorColumn("MW", id: "mw", help: "Molecular weight (Da)") { TableFormat.mw($0.molecularWeight) }
                descriptorColumn("cLogP", id: "logP", help: "Crippen logP") { $0.logPText }
                descriptorColumn("TPSA", id: "tpsa", help: "Topological polar surface area (Å²)") { TableFormat.tpsa($0.tpsa) }
                descriptorColumn("HBD", id: "hbd", help: "H-bond donors") { "\($0.hbd)" }
                descriptorColumn("HBA", id: "hba", help: "H-bond acceptors") { "\($0.hba)" }
                descriptorColumn("RotB", id: "rotB", help: "Rotatable bonds") { "\($0.rotatableBonds)" }
                descriptorColumn("HA", id: "heavy", help: "Heavy atoms") { "\($0.heavyAtomCount)" }
                descriptorColumn("Fsp3", id: "fsp3", help: "Fraction of sp3 carbons") { String(format: "%.2f", $0.fractionCSP3) }
                    .defaultVisibility(.hidden)

                TableColumn("Ro5", sortUsing: sorter("ro5")) { entry in
                    if let d = entry.descriptors {
                        Image(systemName: d.lipinski ? "checkmark" : "xmark")
                            .font(DS.Font.caption.weight(.semibold))
                            .foregroundStyle(d.lipinski ? DS.Color.success : DS.Color.danger)
                            .help(d.lipinski ? "Passes Lipinski's rule of five" : "Fails Lipinski's rule of five")
                            .accessibilityLabel(d.lipinski ? "Lipinski pass" : "Lipinski fail")
                    } else {
                        TableDashCell()
                    }
                }
                .width(min: 32, ideal: 36, max: 48)
                .alignment(.center)
                .customizationID("ro5")
            }

            Group {
                if hasAffinity {
                    TableColumn("Exp. pKi", sortUsing: sorter("pKi")) { entry in
                        let pK = entry.effectivePKi
                        TableNumberCell(text: pK.map(TableFormat.pK), tone: pK.map { DSScore.tone(pK: $0) },
                                        help: "Experimental pKi (from pKi, Ki or IC50)")
                    }
                    .width(min: 50, ideal: 64, max: 90)
                    .alignment(.trailing)
                    .customizationID("pKi")

                    TableColumn("LLE", sortUsing: sorter("lle")) { entry in
                        let lle = Self.lipophilicEfficiency(entry)
                        TableNumberCell(text: lle.map(TableFormat.pK), tone: lle.map { DSScore.tone(lipophilicEfficiency: $0) },
                                        help: "Lipophilic ligand efficiency: pKi − cLogP")
                    }
                    .width(min: 40, ideal: 50, max: 70)
                    .alignment(.trailing)
                    .customizationID("lle")
                }

                if hasForms {
                    TableColumn("Pop.", sortUsing: sorter("pop")) { entry in
                        if let pop = entry.populationWeight {
                            Text(TableFormat.percent(pop))
                                .font(DS.Font.secondary.weight(.medium))
                                .monospacedDigit()
                                .foregroundStyle(Self.populationTone(pop).color)
                                .help(String(format: "Boltzmann population %.1f %% at the target pH", pop * 100))
                        } else {
                            TableDashCell()
                        }
                    }
                    .width(min: 44, ideal: 52, max: 70)
                    .alignment(.trailing)
                    .customizationID("pop")

                    TableColumn("ΔE", sortUsing: sorter("deltaE")) { entry in
                        if let dE = entry.relativeEnergy {
                            // The lowest form reads 0.0 in the success tone.
                            Text(dE < 0.01 ? "0.0" : String(format: "+%.1f", dE))
                                .font(DS.Font.secondary)
                                .monospacedDigit()
                                .foregroundStyle(dE < 0.01 ? DS.Color.success : DS.Color.warning)
                                .help(dE < 0.01 ? "Lowest-energy form"
                                      : String(format: "%.1f kcal/mol above the lowest-energy form", dE))
                        } else {
                            TableDashCell()
                        }
                    }
                    .width(min: 40, ideal: 48, max: 70)
                    .alignment(.trailing)
                    .customizationID("deltaE")
                }

                TableColumn("Conf.", sortUsing: sorter("conformers")) { entry in
                    TableNumberCell(text: entry.conformerCount > 0 ? "\(entry.conformerCount)" : nil,
                                    help: "3D conformers generated by preparation")
                }
                .width(min: 36, ideal: 44, max: 60)
                .alignment(.trailing)
                .customizationID("conformers")
            }

            TableColumnForEach(userPropertyKeys, id: \.self) { key in
                TableColumn(key, sortUsing: sorter("user.\(key)")) { entry in
                    if let value = entry.userProperties[key], !value.isEmpty {
                        Text(value)
                            .font(DS.Font.secondary)
                            .monospacedDigit()
                            .lineLimit(1)
                            .help(value)
                    } else {
                        TableDashCell()
                    }
                }
                .width(min: 50, ideal: 80)
                .customizationID("user.\(key)")
            }
        } rows: {
            ForEach(nodes) { node in
                if node.forms.isEmpty {
                    TableRow(node.entry)
                } else {
                    DisclosureTableRow(node.entry, isExpanded: formsExpanded(node.entry.name)) {
                        ForEach(node.forms) { form in
                            TableRow(form)
                        }
                    }
                }
            }
        }
        .contextMenu(forSelectionType: UUID.self) { ids in
            rowContextMenu(ids)
        } primaryAction: { ids in
            // Double-click / ↩ on one dockable compound makes it the docking ligand.
            guard ids.count == 1, let id = ids.first,
                  let entry = db.entries.first(where: { $0.id == id }),
                  !entry.atoms.isEmpty, !db.hasEnumeratedChildren(entry) else { return }
            useEntryForDocking(entry)
        }
        .overlay {
            if entries.isEmpty {
                noMatchesState
            }
        }
    }

    /// A trailing-aligned numeric descriptor column (em dash before preparation computes descriptors).
    private func descriptorColumn(
        _ title: String, id: String, help: String,
        format: @escaping (LigandDescriptors) -> String
    ) -> some TableColumnContent<LigandEntry, TableSort<LigandEntry>> {
        TableColumn(title, sortUsing: sorter(id)) { entry in
            TableNumberCell(text: entry.descriptors.map(format), help: help)
        }
        .width(min: 36, ideal: 48, max: 80)
        .alignment(.trailing)
        .customizationID(id)
    }

    /// Forms are shown by default; a search always reveals them.
    private func formsExpanded(_ parentName: String) -> Binding<Bool> {
        Binding(
            get: { !searchText.isEmpty || !collapsedParents.contains(parentName) },
            set: { expanded in
                if expanded { collapsedParents.remove(parentName) } else { collapsedParents.insert(parentName) }
            }
        )
    }

    /// Name cell: form-kind badge, inline rename and the active-docking-ligand marker.
    @ViewBuilder
    private func nameCell(_ entry: LigandEntry) -> some View {
        let isActive = viewModel.activeLigandEntryID == entry.id
        HStack(spacing: DS.Space.xs) {
            if let kind = entry.formKind {
                DSTag(kind.symbol, tint: Self.formKindColor(kind))
                    .help(kind.label)
            }
            if renamingEntryID == entry.id {
                TextField("Name", text: $renamingText)
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.small)
                    .onSubmit { commitRename(entryID: entry.id) }
                    .onExitCommand { renamingEntryID = nil }
            } else {
                Text(entry.name)
                    .font(DS.Font.secondary.weight(isActive || db.hasEnumeratedChildren(entry) ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(entry.name)
            }
            if isActive {
                Image(systemName: "scope")
                    .font(DS.Font.caption.weight(.semibold))
                    .foregroundStyle(DS.Color.ligand)
                    .help("Active docking ligand")
                    .accessibilityLabel("Active docking ligand")
            }
        }
    }

    // MARK: - Row Context Menu

    /// Acts on the clicked rows (the selection when the click is inside it).
    @ViewBuilder
    private func rowContextMenu(_ ids: Set<UUID>) -> some View {
        let entries = db.entries.filter { ids.contains($0.id) }
        let single = entries.count == 1 ? entries.first : nil
        let dockable = entries.filter { $0.isPrepared && !db.hasEnumeratedChildren($0) }
        let preparable = entries.filter { !$0.isEnumerated && !$0.smiles.isEmpty }

        if !entries.isEmpty {
            Button(dockable.count > 1 ? "Queue \(dockable.count) for Docking" : "Use for Docking") {
                dockEntries(dockable)
            }
            .disabled(dockable.isEmpty)

            Button(preparable.count > 1 ? "Prepare \(preparable.count) Compounds" : "Prepare") {
                runPopulateAndPrepare(entries: preparable)
            }
            .disabled(preparable.isEmpty || isProcessing)

            Button("Enumerate Forms…") {
                enumerationSheetEntries = preparable
                showEnumerationSheet = true
            }
            .disabled(preparable.isEmpty || isProcessing)

            Divider()

            if let entry = single {
                Button("Edit Structure…") {
                    editingEntryID = entry.id
                    showChemDrawer = true
                }
                .disabled(entry.smiles.isEmpty)
                Button("Rename") {
                    renamingEntryID = entry.id
                    renamingText = entry.name
                }
            }
            Button(entries.count > 1 ? "Copy \(entries.count) SMILES" : "Copy SMILES") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(entries.map(\.originalSMILES).joined(separator: "\n"), forType: .string)
            }

            Divider()

            if let entry = single, entry.isEnumerated, let parent = entry.parentName {
                Button("Delete All \(parent) Forms", role: .destructive) {
                    delete(Set(db.entries.filter { $0.parentName == parent }.map(\.id)))
                }
            }
            Button(entries.count > 1 ? "Delete \(entries.count) Compounds" : "Delete", role: .destructive) {
                delete(ids)
            }
        }
    }

    // MARK: - Empty States

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
        .background(Color(nsColor: .controlBackgroundColor))
    }

    /// Shown over the table when search or filters hide every compound.
    private var noMatchesState: some View {
        DSEmptyState(
            "No Matching Compounds",
            systemImage: "line.3.horizontal.decrease.circle",
            description: searchText.isEmpty
                ? "No compound passes the active filters."
                : "No compound name or SMILES contains “\(searchText)”."
        ) {
            Button("Clear Search and Filters") {
                searchText = ""
                lipinskiFilter = false
                showOnlyPrepared = false
            }
        }
    }

    // MARK: - Status Bar

    @ViewBuilder
    private var statusBar: some View {
        let entries = filteredEntries
        let nPrepared = entries.filter(\.isPrepared).count
        let nEnumerated = entries.filter(\.isEnumerated).count
        let nConformers = entries.reduce(0) { $0 + $1.conformerCount }
        let nSelected = selectedIDs.count
        let isBusy = isProcessing || isBatchProcessing || db.isProcessing

        DSWindowStatusBar {
            Text(isFiltered ? "\(entries.count) of \(db.count) compounds"
                 : entries.count == 1 ? "1 compound" : "\(entries.count) compounds")
            if nPrepared > 0 {
                StatusSeparator()
                Label("\(nPrepared) prepared", systemImage: "checkmark.circle.fill")
                    .labelStyle(StatusLabelStyle(tint: DS.Color.success))
            }
            if nEnumerated > 0 {
                StatusSeparator()
                Text("\(nEnumerated) enumerated forms")
            }
            if nConformers > 0 {
                StatusSeparator()
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
                if nSelected > 0 || viewModel.molecules.ligand != nil { StatusSeparator() }
            }
            if nSelected > 0 {
                Text("\(nSelected) selected")
                    .foregroundStyle(DS.Color.accent)
            }
            if let lig = viewModel.molecules.ligand {
                if nSelected > 0 { StatusSeparator() }
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

    // MARK: - Actions

    func deleteSelected() {
        delete(selectedIDs)
    }

    /// Removes the given compounds, clearing the inspector and selection that pointed at them.
    func delete(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        if inspectedEntry.map({ ids.contains($0.id) }) == true {
            inspectedEntry = nil
            conformers = []
        }
        db.batchMutate { entries in
            entries.removeAll { ids.contains($0.id) }
        }
        selectedIDs.subtract(ids)
        viewModel.log.info("Deleted \(ids.count) entries", category: .molecule)
    }

    func useSelectedForDocking() {
        // Exclude parent entries that have enumerated children (would duplicate docking)
        dockEntries(db.entries.filter {
            selectedIDs.contains($0.id) && $0.isPrepared && !db.hasEnumeratedChildren($0)
        })
    }

    /// One prepared compound becomes the docking ligand; several are queued for batch docking.
    private func dockEntries(_ entries: [LigandEntry]) {
        guard !entries.isEmpty else {
            viewModel.log.warn("No prepared entries in selection — prepare them first", category: .dock)
            return
        }
        if entries.count == 1 {
            useEntryForDocking(entries[0])
        } else {
            viewModel.queueLigandsForBatchDocking(entries)
        }
        viewModel.log.success("Queued \(entries.count) entries for docking", category: .dock)
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
