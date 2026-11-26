// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import Charts

// MARK: - Results Database Window

/// Secondary window: batch ligands in the sidebar, the sortable pose table in
/// the content column and the selected pose's details in the inspector.
///
/// The table selection is the shared pose selection (`selectedPoseIndices`,
/// also ticked in the Results step): one selected pose is shown in 3D and
/// inspected, several can be overlaid or compared.
struct ResultsDatabaseWindow: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    @State private var sortOrder: [TableSort<PoseTableRow>] = []
    @State private var columnCustomization: TableColumnCustomization<PoseTableRow> = TableColumnStore.load(Self.columnStoreKey)
    private static let columnStoreKey = "ui.resultsTable.columns"
    @State private var clusterLeadersOnly = false
    @State private var validPosesOnly = false
    @State private var selectedBatchLigand: String? = nil
    @State private var showCorrelation: Bool = false
    @State private var showMultiPoseAnalysis: Bool = false
    @State private var showInspector: Bool = true
    @State private var showSidebar: Bool = true
    @AppStorage("ui.resultsSidebarWidth") private var sidebarWidth = 240.0
    @AppStorage("ui.resultsInspectorWidth") private var inspectorWidth = 320.0
    @State private var filterText: String = ""

    private var method: ScoringMethod { viewModel.docking.scoringMethod }
    private var results: [DockingResult] { viewModel.docking.dockingResults }
    private var batch: [(ligandName: String, results: [DockingResult])] { viewModel.docking.batchResults }

    private var isFiltered: Bool {
        clusterLeadersOnly || validPosesOnly || (batch.isEmpty && !filterText.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    /// Heavy atoms of the docked ligand, for ligand efficiency.
    private var heavyAtomCount: Int {
        results.first?.transformedAtomPositions.count ?? 0
    }

    /// Poses in table order. A row's id is its index in `dockingResults` (the index every action uses).
    private var tableRows: [PoseTableRow] {
        let query = filterText.trimmingCharacters(in: .whitespaces).lowercased()
        let reference = results.first?.transformedAtomPositions ?? []
        let heavyAtoms = heavyAtomCount
        let method = self.method
        let rows = results.enumerated().compactMap { index, result -> PoseTableRow? in
            if clusterLeadersOnly && !(result.clusterID >= 0 && result.clusterRank == 0) { return nil }
            if validPosesOnly && result.validity?.passed != true { return nil }
            if batch.isEmpty, !query.isEmpty,
               !("#\(index + 1)".hasPrefix(query) || "c\(result.clusterID)" == query) { return nil }
            let score = result.displayScore(method: method)
            return PoseTableRow(
                id: index,
                result: result,
                score: score,
                ligandEfficiency: LigandMetrics.ligandEfficiency(score: score, method: method, heavyAtoms: heavyAtoms),
                rmsdToBest: LigandMetrics.rmsd(result.transformedAtomPositions, reference)
            )
        }
        return sortOrder.isEmpty ? rows : rows.sorted(using: sortOrder)
    }

    /// The pose shown in the inspector: the selection when it is a single pose.
    private var inspectedIndex: Int? {
        let selection = viewModel.docking.selectedPoseIndices
        guard selection.count == 1, let idx = selection.first, idx < results.count else { return nil }
        return idx
    }

    var body: some View {
        Group {
            if results.isEmpty && batch.isEmpty {
                DSEmptyState("No Docking Results",
                             systemImage: "chart.bar.doc.horizontal",
                             description: "Run docking in step 5 to browse poses, interactions and correlations here.") {
                    Button("Go to Docking") {
                        SidebarTab.dock.request(in: viewModel)
                        openWindow(id: "main")
                    }
                    .help("Show the Docking step in the main window (⌘5)")
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Integrated panes, like the main window: NavigationSplitView's floating
                // sidebar and `.inspector` mis-place their content in this window on macOS 26.
                HStack(spacing: 0) {
                    if showSidebar {
                        sidebar
                            .frame(width: Self.clamped(sidebarWidth, to: Self.sidebarRange))
                            .overlay(alignment: .trailing) {
                                PaneResizeHandle(width: $sidebarWidth, range: Self.sidebarRange, edge: .trailing)
                            }
                        Divider()
                    }
                    poseTable
                        .safeAreaInset(edge: .bottom, spacing: 0) { statusBar }
                        .frame(maxWidth: .infinity)
                        .layoutPriority(1)
                    if showInspector {
                        Divider()
                        inspector
                            .frame(width: Self.clamped(inspectorWidth, to: Self.inspectorRange))
                            .background(.background)
                            .overlay(alignment: .leading) {
                                PaneResizeHandle(width: $inspectorWidth, range: Self.inspectorRange, edge: .leading)
                            }
                    }
                }
                .overlay(alignment: .top) { Divider() }
                .clipped()
                .searchable(text: $filterText, placement: .toolbar,
                            prompt: batch.isEmpty ? "Filter poses (#3, C2)" : "Filter ligands")
            }
        }
        .navigationSubtitle(windowSubtitle)
        .toolbar { toolbarContent }
        .onChange(of: columnCustomization) { _, newValue in
            TableColumnStore.save(newValue, key: Self.columnStoreKey)
        }
        .sheet(isPresented: $showMultiPoseAnalysis) {
            MultiPoseAnalysisSheet()
                .environment(viewModel)
        }
        .sheet(isPresented: $showCorrelation) {
            correlationAnalysisSheet
                .frame(minWidth: 640, idealWidth: 720, minHeight: 540, idealHeight: 700)
        }
    }

    private var windowSubtitle: String {
        if results.isEmpty && batch.isEmpty { return "" }
        if !batch.isEmpty { return "\(batch.count) ligands · \(results.count) poses" }
        if let name = viewModel.molecules.ligand?.name, !name.isEmpty { return "\(name) · \(results.count) poses" }
        return "\(results.count) poses"
    }

    // MARK: - Toolbar

    private static let sidebarRange: ClosedRange<CGFloat> = 200...340
    private static let inspectorRange: ClosedRange<CGFloat> = DS.Size.inspectorMin...480

    private static func clamped(_ width: Double, to range: ClosedRange<CGFloat>) -> CGFloat {
        min(max(CGFloat(width), range.lowerBound), range.upperBound)
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Button { showSidebar.toggle() } label: {
                Label("Sidebar", systemImage: "sidebar.leading")
            }
            .disabled(results.isEmpty && batch.isEmpty)
            .help(showSidebar ? "Hide Sidebar" : "Show Sidebar")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Button { showCorrelation = true } label: {
                Label("Correlation…", systemImage: "chart.line.uptrend.xyaxis")
            }
            .labelStyle(.titleAndIcon)
            .disabled(batch.isEmpty)
            .help("Correlate docking scores with experimental affinity (batch results)")

            Button {
                if let idx = inspectedIndex { openDiagram(idx) }
            } label: {
                Label("Diagram", systemImage: "point.3.connected.trianglepath.dotted")
            }
            .labelStyle(.titleAndIcon)
            .disabled(inspectedIndex == nil)
            .help("Open the 2D interaction diagram for the selected pose")

            Menu {
                Toggle("Cluster representatives only", isOn: $clusterLeadersOnly)
                Toggle("PoseBusters-valid only", isOn: $validPosesOnly)
                    .disabled(!results.contains { $0.validity != nil })
            } label: {
                Label("Filter",
                      systemImage: clusterLeadersOnly || validPosesOnly
                        ? "line.3.horizontal.decrease.circle.fill"
                        : "line.3.horizontal.decrease.circle")
            }
            .disabled(results.isEmpty)
            .help("Filter the pose table")

            Menu {
                Button("Export Poses as SDF…") { viewModel.exportDockingResultsSDF() }
                Button("Export Table as CSV…") { viewModel.exportDockingResultsCSV() }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(results.isEmpty)
            .help("Export poses (SDF) or the results table (CSV)")

            Button { showInspector.toggle() } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
            }
            .disabled(results.isEmpty && batch.isEmpty)
            .help(showInspector ? "Hide Inspector" : "Show Inspector")
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: $selectedBatchLigand) {
            if batch.isEmpty {
                Section("Current Run") {
                    HStack(spacing: DS.Space.s) {
                        Circle().fill(DS.Color.ligand).frame(width: 8, height: 8)
                        Text(viewModel.molecules.ligand?.name.nonEmpty ?? "Docking run")
                            .font(DS.Font.label.weight(.medium))
                            .lineLimit(1)
                        Spacer(minLength: DS.Space.xs)
                        DSTag("\(results.count)")
                    }
                }
            } else {
                Section("Batch Ligands") {
                    ForEach(Array(filteredBatch.enumerated()), id: \.element.ligandName) { _, entry in
                        batchLigandRow(entry: entry)
                            .tag(entry.ligandName)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .onChange(of: selectedBatchLigand) { _, name in
            guard let name, let entry = batch.first(where: { $0.ligandName == name }) else { return }
            viewModel.docking.selectedPoseIndices = []
            ResultsActions.loadBatchLigand(named: name, results: entry.results, viewModel: viewModel)
        }
    }

    private var filteredBatch: [(ligandName: String, results: [DockingResult])] {
        let query = filterText.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return batch }
        return batch.filter { $0.ligandName.localizedCaseInsensitiveContains(query) }
    }

    private func batchLigandRow(entry: (ligandName: String, results: [DockingResult])) -> some View {
        let rank = (batch.firstIndex(where: { $0.ligandName == entry.ligandName }) ?? 0) + 1
        let best = entry.results.first
        let affinity = ligandAffinityData(for: entry.ligandName)
        return HStack(spacing: DS.Space.s) {
            Text(verbatim: "#\(rank)")
                .font(DS.Font.monoSmall.weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(rank == 1 ? AnyShapeStyle(DS.Color.success) : AnyShapeStyle(.secondary))
                .frame(minWidth: 26, alignment: .trailing)
            VStack(alignment: .leading, spacing: 1) {
                Text(entry.ligandName)
                    .font(DS.Font.label.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: DS.Space.s) {
                    if let best {
                        DSScoreBadge(best.displayScore(method: method), method: method)
                        if let le = LigandMetrics.ligandEfficiency(score: best.displayScore(method: method), method: method,
                                                                   heavyAtoms: best.transformedAtomPositions.count) {
                            Text("LE \(TableFormat.ligandEfficiency(le))")
                                .font(DS.Font.caption)
                                .monospacedDigit()
                                .foregroundStyle(DSScore.tone(ligandEfficiency: le).cellColor)
                                .help("Ligand efficiency of the best pose (kcal/mol per heavy atom)")
                        }
                    }
                    Text(entry.results.count == 1 ? "1 pose" : "\(entry.results.count) poses")
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.tertiary)
                    if let ki = affinity?.ki {
                        Text(String(format: "Ki %.1f nM", ki))
                            .font(DS.Font.caption)
                            .monospacedDigit()
                            .foregroundStyle(DS.Color.pocket)
                    }
                }
            }
        }
        .padding(.vertical, DS.Space.xxs)
    }

    // MARK: - Pose table

    /// The shared pose selection; picking exactly one pose shows it in 3D.
    private var selection: Binding<Set<Int>> {
        Binding(
            get: { viewModel.docking.selectedPoseIndices },
            set: { newValue in
                viewModel.docking.selectedPoseIndices = newValue
                if newValue.count == 1, let idx = newValue.first { viewModel.showDockingPose(at: idx) }
            }
        )
    }

    @ViewBuilder
    private var poseTable: some View {
        let rows = tableRows
        let method = self.method
        let hasValidity = results.contains { $0.validity != nil }
        let hasStrain = results.contains { $0.strainEnergy != nil }
        let hasXTB = results.contains { $0.gfn2Energy != nil }
        let hasConstraints = results.contains { $0.constraintPenalty > 0.01 }
        let hasForms = results.contains { $0.formLabel != nil }

        Table(of: PoseTableRow.self,
              selection: selection,
              sortOrder: $sortOrder,
              columnCustomization: $columnCustomization) {
            Group {
                TableColumn("#", sortUsing: TableSort<PoseTableRow>("rank") { .number($0.id) }) { row in
                    Text(verbatim: "#\(row.id + 1)")
                        .font(DS.Font.monoSmall.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .width(min: 36, ideal: 48, max: 64)
                .customizationID("rank")
                .disabledCustomizationBehavior(.visibility)

                TableColumn(method.scoreColumnTitle, sortUsing: TableSort<PoseTableRow>("score") { .number($0.score) }) { row in
                    TableScoreCell(value: row.score, method: method)
                }
                .width(min: 70, ideal: 104, max: 130)
                .alignment(.trailing)
                .customizationID("score")

                TableColumn("LE", sortUsing: TableSort<PoseTableRow>("le") { .number($0.ligandEfficiency) }) { row in
                    TableNumberCell(text: row.ligandEfficiency.map(TableFormat.ligandEfficiency),
                                    tone: row.ligandEfficiency.map { DSScore.tone(ligandEfficiency: $0) },
                                    help: "Ligand efficiency (kcal/mol per heavy atom)")
                }
                .width(min: 40, ideal: 50, max: 70)
                .alignment(.trailing)
                .customizationID("le")

                TableColumn("RMSD (Å)", sortUsing: TableSort<PoseTableRow>("rmsd") { .number($0.rmsdToBest) }) { row in
                    TableNumberCell(text: row.rmsdToBest.map(TableFormat.rmsd),
                                    help: "Heavy-atom RMSD to pose #1 (no symmetry correction)")
                }
                .width(min: 50, ideal: 66, max: 90)
                .alignment(.trailing)
                .customizationID("rmsd")

                TableColumn("Cluster", sortUsing: TableSort<PoseTableRow>("cluster") {
                    $0.result.clusterID >= 0 ? .number($0.result.clusterID) : .missing
                }) { row in
                    clusterCell(row.result)
                }
                .width(min: 48, ideal: 60, max: 80)
                .customizationID("cluster")

                TableColumn("Rank in Cluster", sortUsing: TableSort<PoseTableRow>("clusterRank") {
                    $0.result.clusterID >= 0 ? .number($0.result.clusterRank + 1) : .missing
                }) { row in
                    TableNumberCell(text: row.result.clusterID >= 0 ? "\(row.result.clusterRank + 1)" : nil)
                }
                .width(min: 40, ideal: 60, max: 90)
                .alignment(.trailing)
                .customizationID("clusterRank")
                .defaultVisibility(.hidden)

                if hasValidity {
                    TableColumn("PB", sortUsing: TableSort<PoseTableRow>("validity") {
                        .number($0.result.validity.map { $0.passed ? 1 : 0 })
                    }) { row in
                        validityCell(row.result.validity)
                    }
                    .width(min: 28, ideal: 34, max: 48)
                    .alignment(.center)
                    .customizationID("validity")
                }
            }

            if method.isAffinityScore {
                TableColumn("Confidence", sortUsing: TableSort<PoseTableRow>("confidence") { .number($0.result.afConfidence) }) { row in
                    TableNumberCell(text: String(format: "%.2f", row.result.afConfidence),
                                    help: "Druse Affinity model confidence (0–1)")
                }
                .width(min: 56, ideal: 76, max: 100)
                .alignment(.trailing)
                .customizationID("confidence")
            } else {
                Group {
                    energyTermColumn("vdW", id: "vdw", help: "Steric (gauss + repulsion), kcal/mol") { $0.stericEnergy }
                    energyTermColumn("Hydrophobic", id: "hydrophobic", help: "Hydrophobic contact term, kcal/mol") { $0.hydrophobicEnergy }
                    energyTermColumn("H-bond", id: "hbond", help: "Hydrogen-bond term, kcal/mol") { $0.hbondEnergy }
                    energyTermColumn("Torsion", id: "torsion", help: "Torsional entropy penalty, kcal/mol") { $0.torsionPenalty }
                    energyTermColumn("Intra+", id: "intra",
                                     help: "Ligand intramolecular energy, box and constraint penalties and scoring corrections, kcal/mol") { $0.intraAndOtherEnergy }
                        .defaultVisibility(.hidden)
                }
            }

            Group {
                if hasStrain {
                    TableColumn("Strain", sortUsing: TableSort<PoseTableRow>("strain") { .number($0.result.strainEnergy) }) { row in
                        let strain = row.result.strainEnergy
                        TableNumberCell(text: strain.map(TableFormat.energyTerm),
                                        tone: strain.map { $0 > 10 ? .danger : $0 > 4 ? .warning : .neutral },
                                        help: "MMFF94 ligand strain energy (docked − relaxed), kcal/mol")
                    }
                    .width(min: 44, ideal: 56, max: 80)
                    .alignment(.trailing)
                    .customizationID("strain")
                }
                if hasXTB {
                    TableColumn("GFN2", sortUsing: TableSort<PoseTableRow>("gfn2") { .number($0.result.gfn2Energy) }) { row in
                        TableNumberCell(text: row.result.gfn2Energy.map(TableFormat.energyTerm),
                                        help: "GFN2-xTB energy, kcal/mol")
                    }
                    .width(min: 50, ideal: 64, max: 90)
                    .alignment(.trailing)
                    .customizationID("gfn2")
                }
                if hasConstraints {
                    TableColumn("Constraint", sortUsing: TableSort<PoseTableRow>("constraint") { .number($0.result.constraintPenalty) }) { row in
                        let penalty = row.result.constraintPenalty
                        TableNumberCell(text: TableFormat.energyTerm(penalty),
                                        tone: penalty > 0.01 ? .warning : .neutral,
                                        help: "Pharmacophore constraint penalty (0 = all constraints satisfied)")
                    }
                    .width(min: 50, ideal: 70, max: 90)
                    .alignment(.trailing)
                    .customizationID("constraint")
                }
                if hasForms {
                    TableColumn("Form", sortUsing: TableSort<PoseTableRow>("form") { row in
                        row.result.formLabel.map(TableSortValue.text) ?? .missing
                    }) { row in
                        if let label = row.result.formLabel {
                            Text(row.result.formPopulation.map { "\(label) · \(TableFormat.percent(Double($0)))" } ?? label)
                                .font(DS.Font.secondary)
                                .monospacedDigit()
                                .lineLimit(1)
                                .help("Chemical form and its Boltzmann population")
                        } else {
                            TableDashCell()
                        }
                    }
                    .width(min: 60, ideal: 100)
                    .customizationID("form")
                }
                TableColumn("Gen", sortUsing: TableSort<PoseTableRow>("generation") { .number($0.result.generation) }) { row in
                    TableNumberCell(text: "\(row.result.generation)", help: "Search generation that produced this pose")
                }
                .width(min: 36, ideal: 44, max: 64)
                .alignment(.trailing)
                .customizationID("generation")
                .defaultVisibility(.hidden)
            }
        } rows: {
            ForEach(rows) { row in
                TableRow(row)
            }
        }
        .contextMenu(forSelectionType: Int.self) { ids in
            poseContextMenu(ids)
        } primaryAction: { ids in
            // Double-click / ↩ opens the interaction diagram of one pose.
            if ids.count == 1, let idx = ids.first { openDiagram(idx) }
        }
        .overlay {
            if rows.isEmpty && !results.isEmpty {
                DSEmptyState("No Matching Poses",
                             systemImage: "line.3.horizontal.decrease.circle",
                             description: "No pose passes the search and filters.") {
                    Button("Clear Search and Filters") {
                        filterText = ""
                        clusterLeadersOnly = false
                        validPosesOnly = false
                    }
                }
            } else if results.isEmpty {
                DSEmptyState("No Poses Loaded",
                             systemImage: "list.number",
                             description: "Pick a ligand in the sidebar to browse its poses.")
            }
        }
    }

    private func energyTermColumn(
        _ title: String, id: String, help: String,
        value: @escaping @Sendable (DockingResult) -> Float
    ) -> some TableColumnContent<PoseTableRow, TableSort<PoseTableRow>> {
        TableColumn(title, sortUsing: TableSort<PoseTableRow>(id) { .number(value($0.result)) }) { row in
            TableNumberCell(text: TableFormat.energyTerm(value(row.result)), help: help)
        }
        .width(min: 44, ideal: 60, max: 90)
        .alignment(.trailing)
        .customizationID(id)
    }

    /// Cluster id; the cluster representative (best pose of its binding mode) reads in full weight.
    @ViewBuilder
    private func clusterCell(_ result: DockingResult) -> some View {
        if result.clusterID >= 0 {
            let isLeader = result.clusterRank == 0
            Text(verbatim: "C\(result.clusterID)")
                .font(DS.Font.monoSmall.weight(isLeader ? .semibold : .regular))
                .foregroundStyle(isLeader ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .help(isLeader
                      ? "Representative of cluster \(result.clusterID): the best pose of this binding mode"
                      : "Cluster \(result.clusterID), rank \(result.clusterRank + 1): poses sharing an ID occupy the same binding mode (RMSD < 2.0 Å)")
        } else {
            TableDashCell()
        }
    }

    @ViewBuilder
    private func validityCell(_ validity: PoseValidity?) -> some View {
        if let validity {
            Image(systemName: validity.passed ? "checkmark" : "xmark")
                .font(DS.Font.caption.weight(.semibold))
                .foregroundStyle(validity.passed ? DS.Color.success : DS.Color.warning)
                .help(validity.passed ? "Passes the PoseBusters-style checks"
                      : "Fails: " + validity.failures.joined(separator: ", "))
                .accessibilityLabel(validity.passed ? "PoseBusters pass" : "PoseBusters fail")
        } else {
            TableDashCell()
        }
    }

    /// Acts on the clicked rows (the selection when the click is inside it).
    @ViewBuilder
    private func poseContextMenu(_ ids: Set<Int>) -> some View {
        let indices = ids.filter { $0 < results.count }.sorted()
        if let idx = indices.first, indices.count == 1 {
            Button("Show in 3D Viewport") { select(idx) }
            Button("Interaction Diagram") { openDiagram(idx) }
            Divider()
            Button("Save Pose as PDB…") { viewModel.exportDockingPosePDB(at: idx) }
                .disabled(results[idx].transformedAtomPositions.isEmpty)
            if let ligand = viewModel.docking.originalDockingLigand ?? viewModel.molecules.ligand {
                Button("Use as Lead-Opt Reference") {
                    viewModel.selectReferenceForOptimization(result: results[idx], ligand: ligand)
                }
            }
        } else if indices.count > 1 {
            Button("Overlay \(indices.count) Poses in 3D") {
                viewModel.docking.selectedPoseIndices = Set(indices)
                viewModel.showSelectedPoses()
            }
            Button("Compare \(indices.count) Poses…") {
                viewModel.docking.selectedPoseIndices = Set(indices)
                showMultiPoseAnalysis = true
            }
            Divider()
            Button("Export \(indices.count) Poses as SDF…") {
                viewModel.docking.selectedPoseIndices = Set(indices)
                viewModel.exportSelectedDockingPosesSDF()
            }
        }
    }

    private func select(_ index: Int) {
        viewModel.docking.selectedPoseIndices = [index]
        viewModel.showDockingPose(at: index)
    }

    private func openDiagram(_ index: Int) {
        select(index)
        viewModel.docking.interactionDiagramPoseIndex = index
        openWindow(id: "interaction-diagram")
    }

    // MARK: - Status bar

    private var statusBar: some View {
        let shown = tableRows.count
        let clusters = Set(results.map(\.clusterID).filter { $0 >= 0 }).count
        let selected = viewModel.docking.selectedPoseIndices.filter { $0 < results.count }
        return DSWindowStatusBar {
            Text(isFiltered ? "\(shown) of \(results.count) poses" : "\(results.count) poses")
            if clusters > 0 {
                StatusSeparator()
                Text(clusters == 1 ? "1 cluster" : "\(clusters) clusters")
            }
            if let best = results.first {
                StatusSeparator()
                Text("Best \(TableFormat.score(best.displayScore(method: method))) \(method.unitLabel)")
            }
            if !batch.isEmpty {
                StatusSeparator()
                Text("\(batch.count) ligands in batch")
            }
        } trailing: {
            if let idx = inspectedIndex {
                Text("Pose #\(idx + 1) · \(TableFormat.score(results[idx].displayScore(method: method))) \(method.unitLabel)")
            } else if selected.count > 1 {
                Text("\(selected.count) selected")
                    .foregroundStyle(DS.Color.accent)
            }
        }
    }

    // MARK: - Inspector

    @ViewBuilder
    private var inspector: some View {
        if let idx = inspectedIndex {
            let result = results[idx]
            ScrollView {
                VStack(alignment: .leading, spacing: DS.Space.l) {
                    HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                        Text(verbatim: "Pose #\(idx + 1)")
                            .font(DS.Font.stepTitle)
                        if result.clusterID >= 0 {
                            DSTag("C\(result.clusterID)", monospaced: true)
                        }
                        Spacer(minLength: DS.Space.xs)
                        DSScoreBadge(result.displayScore(method: method), method: method)
                    }
                    energyBreakdownSection(result: result)
                    interactionsSummary
                    miniInteractionDiagram
                    VStack(spacing: DS.Space.s) {
                        Button { openDiagram(idx) } label: {
                            Label("2D Interaction Diagram", systemImage: "point.3.connected.trianglepath.dotted")
                        }
                        .buttonStyle(.dsSecondary)
                        .help("Open the 2D interaction diagram for this pose")
                        Button {
                            viewModel.showDockingPose(at: idx)
                            bringMainViewportToFront()
                        } label: {
                            Label("Show in 3D Viewport", systemImage: "cube")
                        }
                        .buttonStyle(.dsSecondary)
                        .help("Display this pose in the 3D viewer and bring the main window forward")
                    }
                }
                .padding(DS.Space.l)
            }
        } else if viewModel.sortedSelectedPoseIndices.count > 1 {
            multiSelectionPanel
        } else {
            // Inspector-sized placeholder (the large step empty state overflows a 240 pt column).
            VStack(spacing: DS.Space.s) {
                Image(systemName: "cube")
                    .font(.system(size: 28, weight: .light))
                    .foregroundStyle(.tertiary)
                Text("No Selection")
                    .font(DS.Font.sectionTitle)
                    .foregroundStyle(.secondary)
                Text("Click a pose to show it in 3D and inspect its energy terms and contacts. ⌘-click several to overlay or compare them.")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(DS.Space.l)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    /// Several poses selected: list them (best first) with the overlay / compare actions.
    private var multiSelectionPanel: some View {
        let indices = viewModel.sortedSelectedPoseIndices
        return ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                VStack(alignment: .leading, spacing: DS.Space.xxs) {
                    Text("\(indices.count) Poses Selected")
                        .font(DS.Font.stepTitle)
                    Text("Click a pose to inspect it on its own.")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                }
                VStack(spacing: 0) {
                    ForEach(Array(indices.enumerated()), id: \.element) { position, idx in
                        if position > 0 { Divider() }
                        Button { select(idx) } label: {
                            HStack(spacing: DS.Space.s) {
                                Text(verbatim: "#\(idx + 1)")
                                    .font(DS.Font.monoSmall.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .frame(minWidth: 36, alignment: .leading)
                                DSScoreBadge(results[idx].displayScore(method: method), method: method)
                                Spacer(minLength: DS.Space.xs)
                                if results[idx].clusterID >= 0 {
                                    Text(verbatim: "C\(results[idx].clusterID)")
                                        .font(DS.Font.monoSmall)
                                        .foregroundStyle(.tertiary)
                                }
                                Image(systemName: "chevron.right")
                                    .font(DS.Font.caption.weight(.semibold))
                                    .foregroundStyle(.tertiary)
                            }
                            .padding(.horizontal, DS.Space.m)
                            .frame(height: DS.Size.rowHeight + DS.Space.s)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help("Inspect pose \(idx + 1)")
                        .accessibilityLabel("Inspect pose \(idx + 1)")
                    }
                }
                .dsCard(padding: 0)
                VStack(spacing: DS.Space.s) {
                    Button { viewModel.showSelectedPoses() } label: {
                        Label("Overlay in 3D Viewport", systemImage: "square.stack.3d.up")
                    }
                    .buttonStyle(.dsPrimary)
                    .help("Overlay the selected poses in the 3D viewport (best as primary, others as ghosts)")
                    Button { showMultiPoseAnalysis = true } label: {
                        Label("Compare Poses…", systemImage: "square.grid.3x3")
                    }
                    .buttonStyle(.dsSecondary)
                    .help("Pairwise RMSD and interaction fingerprints of the selected poses")
                    Button { viewModel.exportSelectedDockingPosesSDF() } label: {
                        Label("Export Selected as SDF…", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.dsSecondary)
                    .help("Export the selected poses as an SDF file")
                }
            }
            .padding(DS.Space.l)
        }
    }

    private func energyBreakdownSection(result: DockingResult) -> some View {
        let score = result.displayScore(method: method)
        let le = LigandMetrics.ligandEfficiency(score: score, method: method, heavyAtoms: heavyAtomCount)
        let rmsd = LigandMetrics.rmsd(result.transformedAtomPositions, results.first?.transformedAtomPositions ?? [])
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader(method.isAffinityScore ? "Score" : "Energy Breakdown")
            Grid(alignment: .leading, horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.xs) {
                GridRow {
                    Text(method.isAffinityScore ? "pKi" : "Total")
                        .font(DS.Font.label.weight(.medium))
                    DSScoreBadge(score, method: method)
                        .gridColumnAlignment(.trailing)
                }
                if method.isAffinityScore {
                    GridRow {
                        Text("Confidence").foregroundStyle(.secondary)
                        Text(String(format: "%.2f", result.afConfidence))
                    }
                    .font(DS.Font.label)
                    .monospacedDigit()
                } else {
                    breakdownRow("Steric (vdW)", result.stericEnergy)
                    breakdownRow("Hydrophobic", result.hydrophobicEnergy)
                    breakdownRow("H-bond", result.hbondEnergy)
                    breakdownRow("Torsion penalty", result.torsionPenalty)
                    breakdownRow("Intramolecular & other", result.intraAndOtherEnergy)
                }
                if let refine = result.refinementEnergy {
                    GridRow {
                        Text("Refined (OpenMM)").foregroundStyle(DS.Color.pocket)
                        Text(String(format: "%.2f", refine)).foregroundStyle(DS.Color.pocket)
                    }
                    .font(DS.Font.label)
                    .monospacedDigit()
                }
            }
            Divider()
            Grid(alignment: .leading, horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.xs) {
                if let le {
                    GridRow {
                        Text("Ligand efficiency").foregroundStyle(.secondary)
                        Text("\(TableFormat.ligandEfficiency(le)) kcal/mol/HA")
                            .foregroundStyle(DSScore.tone(ligandEfficiency: le).cellColor)
                    }
                }
                if let rmsd {
                    GridRow {
                        Text("RMSD to pose #1").foregroundStyle(.secondary)
                        Text("\(TableFormat.rmsd(rmsd)) Å")
                    }
                }
                if result.clusterID >= 0 {
                    GridRow {
                        Text("Cluster").foregroundStyle(.secondary)
                        Text("C\(result.clusterID) · rank \(result.clusterRank + 1)")
                    }
                }
            }
            .font(DS.Font.label)
            .monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard()
    }

    private func breakdownRow(_ label: String, _ value: Float) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(String(format: "%.2f", value))
        }
        .font(DS.Font.label)
        .monospacedDigit()
    }

    private var interactionsSummary: some View {
        let interactions = viewModel.docking.currentInteractions
        let grouped = Dictionary(grouping: interactions, by: \.type)
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Interactions", count: interactions.count)
            if interactions.isEmpty {
                PanelHint(text: "No contacts detected for this pose under the current scoring method.",
                          icon: "info.circle")
            } else {
                ForEach(MolecularInteraction.InteractionType.allCases, id: \.rawValue) { type in
                    if let group = grouped[type] {
                        HStack(spacing: DS.Space.m) {
                            Capsule().fill(type.themeColor).frame(width: 14, height: 4)
                            Text(type.label).font(DS.Font.label)
                            Spacer()
                            Text("\(group.count)")
                                .font(DS.Font.monoSmall.weight(.medium))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard()
    }

    // MARK: - Mini Interaction Map

    @ViewBuilder
    private var miniInteractionDiagram: some View {
        let interactions = viewModel.docking.currentInteractions
        if let protein = viewModel.molecules.protein, !interactions.isEmpty {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                DSSectionHeader("Interaction Map")
                Canvas { context, size in
                    drawMiniDiagram(context: context, size: size,
                                    interactions: interactions,
                                    proteinAtoms: protein.atoms)
                }
                .frame(height: 220)
                .background(
                    RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                        .fill(DS.Color.diagramPaper)
                )
                .clipShape(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous))
            }
            .dsCard()
        }
    }

    private func drawMiniDiagram(context: GraphicsContext, size: CGSize,
                                 interactions: [MolecularInteraction],
                                 proteinAtoms: [Atom]) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let radius = min(size.width, size.height) * 0.38

        // Group interactions by residue
        var residueInteractions: [String: [MolecularInteraction]] = [:]
        var residueClasses: [String: DiagramResidueClass] = [:]
        for inter in interactions {
            guard inter.proteinAtomIndex < proteinAtoms.count else { continue }
            let pa = proteinAtoms[inter.proteinAtomIndex]
            let key = "\(pa.residueName)\(pa.residueSeq)"
            residueInteractions[key, default: []].append(inter)
            residueClasses[key] = DiagramResidueClass(residueName: pa.residueName)
        }

        let residues = Array(residueInteractions.keys).sorted()
        guard !residues.isEmpty else { return }

        // Ligand at the center
        let ligandRect = CGRect(x: center.x - 14, y: center.y - 14, width: 28, height: 28)
        context.fill(Circle().path(in: ligandRect), with: .color(DS.Color.ligand.opacity(0.22)))
        context.stroke(Circle().path(in: ligandRect), with: .color(DS.Color.ligand), lineWidth: 1.5)
        context.draw(Text("Lig").font(DS.Font.canvasLabel.weight(.bold)).foregroundColor(DS.Color.ligand), at: center)

        // Residues arranged radially
        for (i, resName) in residues.enumerated() {
            let angle = CGFloat(i) / CGFloat(residues.count) * 2.0 * .pi - .pi / 2.0
            let resCenter = CGPoint(x: center.x + cos(angle) * radius,
                                    y: center.y + sin(angle) * radius)
            guard let inters = residueInteractions[resName] else { continue }

            for inter in inters {
                var path = Path()
                path.move(to: center)
                path.addLine(to: resCenter)
                let dashPattern: [CGFloat] = inter.type == .hbond ? [3, 2] : []
                context.stroke(path, with: .color(inter.type.themeColor.opacity(0.7)),
                               style: StrokeStyle(lineWidth: 1.5, dash: dashPattern))
            }

            let bubbleRadius: CGFloat = 19
            let bubbleRect = CGRect(x: resCenter.x - bubbleRadius, y: resCenter.y - bubbleRadius,
                                    width: bubbleRadius * 2, height: bubbleRadius * 2)
            let bubbleColor = (residueClasses[resName] ?? .hydrophobic).themeColor
            context.fill(Circle().path(in: bubbleRect), with: .color(DS.Color.diagramPaper))
            context.fill(Circle().path(in: bubbleRect), with: .color(bubbleColor.opacity(0.15)))
            context.stroke(Circle().path(in: bubbleRect), with: .color(bubbleColor), lineWidth: 1)
            context.draw(
                Text(resName).font(DS.Font.canvasLabel).foregroundColor(DS.Color.diagramInk),
                at: resCenter
            )

            // One dot per interaction type under the bubble
            let uniqueTypes = Array(Set(inters.map(\.type))).sorted(by: { $0.rawValue < $1.rawValue })
            let dotSpacing: CGFloat = 7
            let startX = resCenter.x - CGFloat(uniqueTypes.count - 1) * dotSpacing / 2
            let dotY = resCenter.y + bubbleRadius + 5
            for (di, iType) in uniqueTypes.enumerated() {
                let dotRect = CGRect(x: startX + CGFloat(di) * dotSpacing - 2.5, y: dotY - 2.5, width: 5, height: 5)
                context.fill(Circle().path(in: dotRect), with: .color(iType.themeColor))
            }
        }
    }

    // MARK: - Correlation Analysis Sheet

    private var correlationAnalysisSheet: some View {
        let pairs = affinityDockingPairs()
        return DSSheetScaffold(
            title: "Docking vs. Experimental Affinity",
            subtitle: "Rank agreement between each ligand's best docking score and its measured affinity.",
            systemImage: "chart.line.uptrend.xyaxis"
        ) {
            if pairs.isEmpty {
                DSEmptyState("No Affinity Data",
                             systemImage: "chart.line.uptrend.xyaxis",
                             description: "Import Ki, pKi or IC50 values in the Ligand Database to compute rank correlation with docking scores.")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: DS.Space.xl) {
                        correlationStatistics(pairs: pairs)
                        correlationScatter(pairs: pairs)
                        rankComparisonTable(pairs: pairs)
                        clusterConsensusSection
                    }
                    .padding(DS.Space.xl)
                }
            }
        } footer: {
            Button("Done") { showCorrelation = false }
                .keyboardShortcut(.defaultAction)
        }
    }

    private func correlationStatistics(pairs: [(ligand: String, dockingEnergy: Float, pKi: Float)]) -> some View {
        let dockingRanks = rankValues(pairs.map { $0.dockingEnergy })
        let affinityRanks = rankValues(pairs.map { -$0.pKi }) // higher pKi = better binder = lower rank
        let spearman = spearmanCorrelation(dockingRanks, affinityRanks)
        let pearson = pearsonCorrelation(pairs.map { Double($0.dockingEnergy) }, pairs.map { Double($0.pKi) })
        let kendall = kendallTau(dockingRanks, affinityRanks)

        return VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Correlation Statistics", count: pairs.count)
            DSMetricGrid(columns: 4) {
                DSMetric("Spearman ρ", value: String(format: "%.3f", spearman),
                         tone: correlationTone(spearman, good: 0.7, fair: 0.4))
                DSMetric("Pearson r", value: String(format: "%.3f", pearson),
                         tone: correlationTone(pearson, good: 0.7, fair: 0.4))
                DSMetric("Kendall τ", value: String(format: "%.3f", kendall),
                         tone: correlationTone(kendall, good: 0.6, fair: 0.3))
                DSMetric("R²", value: String(format: "%.3f", pearson * pearson))
            }
        }
    }

    private func correlationTone(_ value: Double, good: Double, fair: Double) -> DSTone {
        abs(value) > good ? .success : abs(value) > fair ? .warning : .danger
    }

    private func correlationScatter(pairs: [(ligand: String, dockingEnergy: Float, pKi: Float)]) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Score vs. pKi")
            Chart {
                ForEach(Array(pairs.enumerated()), id: \.offset) { _, pair in
                    PointMark(
                        x: .value("pKi", pair.pKi),
                        y: .value("Docking score", pair.dockingEnergy)
                    )
                    .foregroundStyle(DS.Color.ligand)
                    .symbolSize(44)
                    .accessibilityLabel(pair.ligand)
                }
            }
            .chartXAxisLabel("Experimental pKi")
            .chartYAxisLabel("Best docking score (kcal/mol)")
            .chartXAxis {
                AxisMarks { _ in
                    AxisGridLine().foregroundStyle(DS.Color.hairline)
                    AxisValueLabel().font(DS.Font.caption)
                }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisGridLine().foregroundStyle(DS.Color.hairline)
                    AxisValueLabel().font(DS.Font.caption)
                }
            }
            .frame(height: 220)
            .dsCard()
        }
    }

    private func rankComparisonTable(pairs: [(ligand: String, dockingEnergy: Float, pKi: Float)]) -> some View {
        let sortedByDocking = pairs.sorted { $0.dockingEnergy < $1.dockingEnergy }
        let sortedByAffinity = pairs.sorted { $0.pKi > $1.pKi }

        return VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Rank Comparison")
            VStack(spacing: 0) {
                HStack(spacing: 0) {
                    rankHeader("Dock rank", width: 72)
                    rankHeader("Ligand", width: 160)
                    rankHeader("Energy", width: 72)
                    rankHeader("pKi", width: 60)
                    rankHeader("Aff. rank", width: 72)
                    rankHeader("Δ rank", width: 60)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, DS.Space.s)
                .padding(.vertical, DS.Space.xs)
                Divider()
                ForEach(Array(sortedByDocking.enumerated()), id: \.offset) { dockRank, pair in
                    let affRank = sortedByAffinity.firstIndex(where: { $0.ligand == pair.ligand }).map { $0 + 1 } ?? 0
                    let delta = abs(dockRank + 1 - affRank)
                    let energyTone = DSScore.tone(energy: pair.dockingEnergy)
                    HStack(spacing: 0) {
                        Text(verbatim: "#\(dockRank + 1)").fontWeight(.semibold).frame(width: 72, alignment: .leading)
                        Text(pair.ligand)
                            .font(DS.Font.label)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(width: 160, alignment: .leading)
                        Text(String(format: "%.1f", pair.dockingEnergy))
                            .foregroundStyle(energyTone == .danger ? DS.Color.scorePoor : energyTone.color)
                            .frame(width: 72, alignment: .leading)
                        Text(String(format: "%.1f", pair.pKi)).frame(width: 60, alignment: .leading)
                        Text(verbatim: "#\(affRank)").frame(width: 72, alignment: .leading)
                        Text(delta == 0 ? "=" : "±\(delta)")
                            .fontWeight(.medium)
                            .foregroundStyle(delta == 0 ? DS.Color.success : delta <= 2 ? DS.Color.warning : DS.Color.scorePoor)
                            .frame(width: 60, alignment: .leading)
                        Spacer(minLength: 0)
                    }
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .padding(.horizontal, DS.Space.s)
                    .padding(.vertical, DS.Space.xxs)
                    .background(dockRank % 2 == 1 ? DS.Color.cardFill : Color.clear)
                }
            }
            .dsCard(padding: DS.Space.xs)
        }
    }

    private func rankHeader(_ title: String, width: CGFloat) -> some View {
        Text(title)
            .font(DS.Font.secondary.weight(.semibold))
            .foregroundStyle(.secondary)
            .frame(width: width, alignment: .leading)
    }

    private var clusterConsensusSection: some View {
        let clusterCounts = computeClusterConsensus()
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Binding Site Consensus")
            Text("The most populated cluster across all docked ligands identifies the consensus binding site.")
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
            if clusterCounts.isEmpty {
                PanelHint(text: "No cluster data available.", icon: "info.circle")
            } else {
                ForEach(clusterCounts.prefix(5), id: \.cluster) { entry in
                    HStack(spacing: DS.Space.m) {
                        DSTag("C\(entry.cluster)", tone: .accent, monospaced: true)
                            .frame(width: 52, alignment: .leading)
                        ProgressView(value: Double(entry.count), total: Double(clusterCounts.first?.count ?? 1))
                            .progressViewStyle(.linear)
                            .tint(DS.Color.accent)
                        Text("\(entry.count) ligands")
                            .font(DS.Font.monoSmall)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Window Helpers

    /// Brings the main 3D viewport window (the one hosting the Metal renderer)
    /// to the front. The Results Database window is opened as a separate window
    /// so updating the renderer's pose otherwise has no visible effect for the
    /// user — they need the main window raised above this one.
    private func bringMainViewportToFront() {
        for window in NSApp.windows {
            guard window.isVisible else { continue }
            if containsMTKView(window.contentView) {
                window.makeKeyAndOrderFront(nil)
                NSApp.activate(ignoringOtherApps: true)
                return
            }
        }
    }

    private func containsMTKView(_ view: NSView?) -> Bool {
        guard let view else { return false }
        if String(describing: type(of: view)).contains("MTKView") { return true }
        for subview in view.subviews {
            if containsMTKView(subview) { return true }
        }
        return false
    }

    // MARK: - Data Helpers

    private struct AffinityData {
        var ki: Float?      // nM
        var pKi: Float?
        var ic50: Float?    // nM
    }

    private func ligandAffinityData(for name: String) -> AffinityData? {
        guard let entry = viewModel.ligandDB.entries.first(where: { $0.name == name }) else { return nil }
        if entry.ki == nil && entry.pKi == nil && entry.ic50 == nil { return nil }
        return AffinityData(ki: entry.ki, pKi: entry.pKi, ic50: entry.ic50)
    }

    /// Build pairs of (ligand, docking energy, pKi) for correlation analysis.
    private func affinityDockingPairs() -> [(ligand: String, dockingEnergy: Float, pKi: Float)] {
        var pairs: [(ligand: String, dockingEnergy: Float, pKi: Float)] = []

        for batch in viewModel.docking.batchResults {
            guard let bestEnergy = batch.results.first?.energy else { continue }
            if let entry = viewModel.ligandDB.entries.first(where: { $0.name == batch.ligandName }) {
                let pKi: Float?
                if let pk = entry.pKi {
                    pKi = pk
                } else if let ki = entry.ki, ki > 0 {
                    pKi = -log10(ki * 1e-9) // nM to pKi
                } else if let ic50 = entry.ic50, ic50 > 0 {
                    pKi = -log10(ic50 * 1e-9) // approximate pKi from IC50
                } else {
                    pKi = nil
                }

                if let pk = pKi {
                    pairs.append((ligand: batch.ligandName, dockingEnergy: bestEnergy, pKi: pk))
                }
            }
        }

        return pairs
    }

    private func computeClusterConsensus() -> [(cluster: Int, count: Int)] {
        var counts: [Int: Int] = [:]
        for batch in viewModel.docking.batchResults {
            if let best = batch.results.first, best.clusterID >= 0 {
                counts[best.clusterID, default: 0] += 1
            }
        }
        return counts.map { (cluster: $0.key, count: $0.value) }
            .sorted { $0.count > $1.count }
    }

    // MARK: - Statistics Helpers

    private func rankValues(_ values: [Float]) -> [Double] {
        let indexed = values.enumerated().sorted { $0.element < $1.element }
        var ranks = [Double](repeating: 0, count: values.count)
        for (rank, entry) in indexed.enumerated() {
            ranks[entry.offset] = Double(rank + 1)
        }
        return ranks
    }

    private func spearmanCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count > 1 else { return 0 }
        let n = Double(x.count)
        var d2: Double = 0
        for i in 0..<x.count {
            let diff = x[i] - y[i]
            d2 += diff * diff
        }
        return 1 - (6 * d2) / (n * (n * n - 1))
    }

    private func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count > 1 else { return 0 }
        let n = Double(x.count)
        let mx = x.reduce(0.0, +) / n
        let my = y.reduce(0.0, +) / n
        var cov: Double = 0
        var sx2: Double = 0
        var sy2: Double = 0
        for i in 0..<x.count {
            let dx = x[i] - mx
            let dy = y[i] - my
            cov += dx * dy
            sx2 += dx * dx
            sy2 += dy * dy
        }
        let sx = sqrt(sx2)
        let sy = sqrt(sy2)
        guard sx > 0, sy > 0 else { return 0 }
        return cov / (sx * sy)
    }

    private func kendallTau(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count > 1 else { return 0 }
        var concordant = 0
        var discordant = 0
        for i in 0..<x.count {
            for j in (i+1)..<x.count {
                let xd = x[i] - x[j]
                let yd = y[i] - y[j]
                if xd * yd > 0 { concordant += 1 }
                else if xd * yd < 0 { discordant += 1 }
            }
        }
        let n = x.count
        let denom = n * (n - 1) / 2
        guard denom > 0 else { return 0 }
        return Double(concordant - discordant) / Double(denom)
    }
}

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}

/// One row of the Results Database pose table; `id` is the pose's index in `dockingResults`.
private struct PoseTableRow: Identifiable, Sendable {
    let id: Int
    let result: DockingResult
    let score: Float
    let ligandEfficiency: Float?
    let rmsdToBest: Float?
}
