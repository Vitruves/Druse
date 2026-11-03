// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import Charts
import simd

/// Step 6 — analyze & export poses. Summary first, then the pose list, batch
/// and screening results; exports and the hand-off to Lead Opt live in the
/// pinned footer.
struct ResultsTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    @State private var showMultiPoseAnalysis = false
    @State private var multiPoseInitialMode: MultiPoseAnalysisSheet.Mode = .rmsd
    @State private var activePoseIndex: Int?

    private static let poseListLimit = 50
    private static let topMeanCount = 10
    private static let batchListLimit = 10

    var body: some View {
        Group {
            if isEmpty {
                DSStepScaffold { emptyState }
            } else {
                DSStepScaffold {
                    summaryCard
                    if showsBatchCard { batchCard }
                    if showsPosesCard { posesCard }
                    if !viewModel.docking.screeningHits.isEmpty { screeningCard }
                } footer: {
                    footer
                }
            }
        }
        .onChange(of: viewModel.docking.screeningHits.count) { _, _ in
            initializeCutoffsIfNeeded()
        }
        .onChange(of: viewModel.docking.showInteractionDiagram) { _, show in
            if show {
                openWindow(id: "interaction-diagram")
                viewModel.docking.showInteractionDiagram = false
            }
        }
        .onChange(of: viewModel.docking.dockingResults.count) { _, _ in
            activePoseIndex = nil
        }
        .sheet(isPresented: $showMultiPoseAnalysis) {
            MultiPoseAnalysisSheet(initialMode: multiPoseInitialMode)
                .environment(viewModel)
        }
    }

    // MARK: - State helpers

    private var method: ScoringMethod { viewModel.docking.scoringMethod }
    private var results: [DockingResult] { viewModel.docking.dockingResults }

    private var isScreeningRunning: Bool {
        switch viewModel.docking.screeningState {
        case .preparing, .docking, .reranking: true
        default: false
        }
    }

    private var isEmpty: Bool {
        results.isEmpty
            && viewModel.docking.batchResults.isEmpty
            && viewModel.docking.screeningHits.isEmpty
            && !viewModel.docking.isBatchDocking
    }

    private var hasDockingSummary: Bool { !results.isEmpty && !viewModel.docking.isDocking }
    private var showsBatchCard: Bool { viewModel.docking.isBatchDocking || !viewModel.docking.batchResults.isEmpty }
    private var showsPosesCard: Bool { hasDockingSummary && !viewModel.docking.isBatchDocking }

    private func tone(_ value: Float) -> DSTone {
        method.isAffinityScore ? DSScore.tone(pK: value) : DSScore.tone(energy: value)
    }

    // MARK: - Empty state

    @ViewBuilder
    private var emptyState: some View {
        Group {
            if viewModel.docking.isDocking || isScreeningRunning {
                DSEmptyState("Run in Progress",
                             systemImage: "hourglass",
                             description: "Results appear here as soon as the run finishes.") {
                    openDatabaseButton
                }
            } else {
                DSEmptyState("No Results Yet",
                             systemImage: "chart.bar.doc.horizontal",
                             description: "Ranked poses, scores and interactions appear here after a docking run. Earlier runs stay available in the Results Database.") {
                    Button {
                        SidebarTab.dock.request(in: viewModel)
                    } label: {
                        Label("Go to Docking", systemImage: SidebarTab.dock.icon)
                    }
                    .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                    openDatabaseButton
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, DS.Space.xxl)
    }

    private var openDatabaseButton: some View {
        Button { openWindow(id: "results-database") } label: {
            Label("Open Results Database", systemImage: "tablecells")
        }
        .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
        .help("Browse every pose with scores, interactions and correlation analysis (⇧⌘R)")
        .accessibilityIdentifier(AccessibilityID.resultsOpenDB)
    }

    // MARK: - Summary

    private var summaryCard: some View {
        PanelCard("Summary", icon: "chart.bar.doc.horizontal", accessory: {
            Button { openWindow(id: "results-database") } label: {
                Label("Database", systemImage: "tablecells")
            }
            .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
            .help("Open Results Database (⇧⌘R)")
            .accessibilityIdentifier(AccessibilityID.resultsOpenDB)
        }) {
            if hasDockingSummary {
                dockingSummary
            } else if !viewModel.docking.batchResults.isEmpty {
                batchSummary
            } else if !viewModel.docking.screeningHits.isEmpty {
                screeningMetrics
            } else if viewModel.docking.isBatchDocking {
                PanelHint(text: "Batch docking is running — ligands appear below as they finish.",
                          icon: "hourglass")
            } else {
                PanelHint(text: "Docking is running — the summary updates when the run finishes.",
                          icon: "hourglass")
            }
        }
    }

    @ViewBuilder
    private var dockingSummary: some View {
        let scores = results.map { $0.displayScore(method: method) }
        let best = scores.first ?? 0
        // Mean over the top poses, not the whole population: unconverged GA individuals
        // (positive, clashing energies) dominate an all-pose mean and make it meaningless.
        let topScores = scores.prefix(Self.topMeanCount)
        let topMean = topScores.isEmpty ? 0 : topScores.reduce(0, +) / Float(topScores.count)
        let clusters = Set(results.map(\.clusterID).filter { $0 >= 0 }).count

        VStack(alignment: .leading, spacing: DS.Space.m) {
            if let name = ligandDisplayName {
                DSTag(name, tone: .ligand, systemImage: "hexagon")
            }
            DSMetricGrid(columns: 2) {
                DSMetric("Best", value: String(format: "%.1f", best), unit: method.unitLabel, tone: tone(best))
                DSMetric("Top-\(topScores.count) mean", value: String(format: "%.1f", topMean), unit: method.unitLabel)
                DSMetric("Poses", value: "\(results.count)")
                DSMetric("Clusters", value: "\(clusters)")
            }
            runDetails
            poseBustersRow
            if let top = results.first {
                bestPoseTerms(top)
            }
            if results.count > 1 {
                ScoreLandscapeChart(
                    scores: Array(scores.prefix(Self.poseListLimit)),
                    isAffinity: method.isAffinityScore,
                    unit: method.unitLabel,
                    activeIndex: activePoseIndex
                ) { index in viewPose(index) }
            }
        }
    }

    private var ligandDisplayName: String? {
        guard let ligand = viewModel.molecules.ligand, !ligand.name.isEmpty else { return nil }
        if let entry = viewModel.ligandDB.entries.first(where: { $0.name == ligand.name || $0.smiles == (ligand.smiles ?? "") }),
           let parent = entry.parentName {
            return "\(ligand.name) ◂ \(parent)"
        }
        return ligand.name
    }

    /// "10 runs × 200 gen × 300" for multi-run GA, matching the preset and the live progress readout.
    private func samplingText(_ docking: DockingCoordinator) -> String {
        let config = docking.dockingConfig
        let total = docking.dockingTotalGenerations
        if config.numRuns > 1, config.generationsPerRun > 0, total == config.numGenerations {
            return "\(config.numRuns) runs × \(config.generationsPerRun) gen × \(config.populationSize)"
        }
        return "\(total) gen × \(config.populationSize)"
    }

    private var runDetails: some View {
        let docking = viewModel.docking
        return Grid(alignment: .leading, horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.xxs) {
            detailRow("Search", docking.searchMethod.rawValue)
            detailRow("Scoring", method.shortLabel)
            if docking.dockingDuration > 0 {
                detailRow("Wall time", formatDuration(docking.dockingDuration))
            }
            detailRow("Sampling", samplingText(docking))
            if docking.dockingPosesEvaluated > 0 {
                // Final-population poses kept after DockingEngine's Vina-style output selection
                // (energy window, duplicate removal, outputMaxClusters modes).
                detailRow("Kept", "\(results.count) of \(docking.dockingPosesEvaluated) poses")
            }
            if let lastGen = results.map(\.generation).max() {
                detailRow("Last improvement", "Gen \(lastGen)")
            }
        }
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(.secondary)
            Text(value)
                .monospacedDigit()
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .font(DS.Font.secondary)
    }

    @ViewBuilder
    private var poseBustersRow: some View {
        if results.contains(where: { $0.validity != nil }) {
            let validCount = results.filter { $0.validity?.passed == true }.count
            let allPassed = validCount == results.count
            let failures = results.first(where: { $0.validity?.passed == false })?.validity?.failures ?? []
            PanelHighlightRow(color: allPassed ? DS.Color.success : DS.Color.warning) {
                HStack(spacing: DS.Space.s) {
                    Image(systemName: allPassed ? "checkmark.seal.fill" : "exclamationmark.shield.fill")
                        .foregroundStyle(allPassed ? DS.Color.success : DS.Color.warning)
                    Text("PoseBusters \(validCount)/\(results.count) passed")
                        .font(DS.Font.secondary.weight(.medium))
                        .monospacedDigit()
                    Spacer(minLength: DS.Space.xs)
                    if !allPassed, !failures.isEmpty {
                        Text(failures.prefix(2).joined(separator: ", "))
                            .font(DS.Font.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .help(failures.joined(separator: "\n"))
                    }
                }
            }
        }
    }

    private func bestPoseTerms(_ best: DockingResult) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xs) {
            PanelSubheader(title: "Best pose terms")
            HStack(spacing: DS.Space.s) {
                termCell("vdW", best.stericEnergy)
                termCell("Hydro", best.hydrophobicEnergy)
                termCell("H-bond", best.hbondEnergy)
                termCell("Torsion", best.torsionPenalty)
                if !method.isAffinityScore {
                    termCell("Intra+", best.intraAndOtherEnergy)
                        .help("Ligand intramolecular energy, boundary and constraint penalties, and scoring corrections. The five terms sum to the score.")
                }
            }
        }
    }

    private func termCell(_ label: String, _ value: Float) -> some View {
        VStack(spacing: 1) {
            Text(String(format: "%.1f", value))
                .font(DS.Font.monoSmall.weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(value < 0 ? DS.Color.success : DS.Color.scorePoor)
            Text(label)
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var batchSummary: some View {
        let batch = viewModel.docking.batchResults
        VStack(alignment: .leading, spacing: DS.Space.m) {
            if let top = batch.first {
                HStack(spacing: DS.Space.s) {
                    Text("Top hit")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                    DSTag(top.ligandName, tone: .ligand, systemImage: "trophy")
                }
            }
            DSMetricGrid(columns: 2) {
                if let bestScore = batch.first?.results.first?.displayScore(method: method) {
                    DSMetric("Best", value: String(format: "%.1f", bestScore), unit: method.unitLabel, tone: tone(bestScore))
                }
                DSMetric("Ligands", value: "\(batch.count)")
            }
        }
    }

    private var screeningMetrics: some View {
        DSMetricGrid(columns: 3) {
            DSMetric("Hits", value: "\(filteredScreeningHits.count)")
            DSMetric("Screened", value: "\(totalScreened)")
            if let best = filteredScreeningHits.first {
                DSMetric("Best", value: String(format: "%.1f", best.compositeScore), tone: tone(best.compositeScore))
            }
        }
    }

    // MARK: - Poses

    private var posesCard: some View {
        PanelCard("Poses", icon: "list.number", accessory: {
            HStack(spacing: DS.Space.s) {
                DSTag("\(results.count)")
                if viewModel.molecules.ligand != nil {
                    Button { viewModel.clearPosesFromView() } label: {
                        Image(systemName: "eye.slash")
                    }
                    .buttonStyle(.borderless)
                    .help("Clear poses and interaction lines from the 3D viewport")
                    .accessibilityLabel("Clear Poses from View")
                    .accessibilityIdentifier(AccessibilityID.resultsClearPoses)
                }
            }
        }) {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                if !viewModel.docking.selectedPoseIndices.isEmpty {
                    selectionBar
                }
                let reference = results.first?.transformedAtomPositions ?? []
                let canOptimize = (viewModel.docking.originalDockingLigand ?? viewModel.molecules.ligand) != nil
                VStack(spacing: DS.Space.xxs) {
                    ForEach(Array(results.prefix(Self.poseListLimit).enumerated()), id: \.offset) { idx, result in
                        PoseRow(
                            index: idx,
                            result: result,
                            method: method,
                            rmsd: rmsd(result.transformedAtomPositions, reference),
                            isChecked: viewModel.docking.selectedPoseIndices.contains(idx),
                            isActive: activePoseIndex == idx,
                            canOptimize: canOptimize,
                            onToggle: { viewModel.togglePoseSelection(at: idx) },
                            onView: { viewPose(idx) },
                            onDiagram: { openDiagram(idx) },
                            onSave: { viewModel.exportDockingPosePDB(at: idx) },
                            onOptimize: { optimize(result) }
                        )
                    }
                }
                if results.count > Self.poseListLimit {
                    PanelHint(text: "+ \(results.count - Self.poseListLimit) more poses — open the Results Database to browse all.",
                              icon: "ellipsis.circle")
                }
            }
        }
    }

    private var selectionBar: some View {
        let count = viewModel.docking.selectedPoseIndices.count
        return PanelHighlightRow(color: DS.Color.accent) {
            HStack(spacing: DS.Space.m) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(DS.Color.accent)
                Text("\(count) selected")
                    .font(DS.Font.secondary.weight(.medium))
                    .monospacedDigit()
                Spacer(minLength: DS.Space.xs)
                Button("View") { viewModel.showSelectedPoses() }
                    .help("Show selected poses in the 3D viewport (best as primary, others as ghosts)")
                    .accessibilityIdentifier(AccessibilityID.resultsViewSelected)
                Menu("Export") {
                    Button("Export as SDF…") { viewModel.exportSelectedDockingPosesSDF() }
                    Button("Export as PDB…") { viewModel.exportSelectedDockingPosesPDB() }
                    Button("Export as CSV…") { viewModel.exportSelectedDockingPosesCSV() }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Export selected poses (SDF / PDB / CSV)")
                Button("Clear") { viewModel.docking.selectedPoseIndices.removeAll() }
                    .help("Clear the pose selection")
                    .accessibilityIdentifier(AccessibilityID.resultsClearSelection)
            }
            .font(DS.Font.secondary)
            .buttonStyle(.borderless)
        }
    }

    private func viewPose(_ index: Int) {
        activePoseIndex = index
        viewModel.showDockingPose(at: index)
    }

    private func openDiagram(_ index: Int) {
        viewPose(index)
        viewModel.docking.interactionDiagramPoseIndex = index
        openWindow(id: "interaction-diagram")
    }

    private func optimize(_ result: DockingResult) {
        guard let ligand = viewModel.docking.originalDockingLigand ?? viewModel.molecules.ligand else { return }
        viewModel.selectReferenceForOptimization(result: result, ligand: ligand)
    }

    /// Heavy-atom RMSD to the top-ranked pose (same atom order, no symmetry correction).
    private func rmsd(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float? {
        guard !a.isEmpty, a.count == b.count else { return nil }
        var sumSq: Float = 0
        for k in 0..<a.count {
            let d = a[k] - b[k]
            sumSq += simd_dot(d, d)
        }
        return (sumSq / Float(a.count)).squareRoot()
    }

    // MARK: - Batch

    private var batchCard: some View {
        let batch = viewModel.docking.batchResults
        return PanelCard("Batch Docking", icon: "square.stack.3d.down.right", accessory: {
            DSTag("\(batch.count) ligands")
        }) {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                if viewModel.docking.isBatchDocking {
                    batchProgress
                }
                VStack(spacing: DS.Space.xxs) {
                    ForEach(Array(batch.prefix(Self.batchListLimit).enumerated()), id: \.offset) { rank, entry in
                        batchRow(rank: rank, ligandName: entry.ligandName, results: entry.results)
                    }
                }
                if batch.count > Self.batchListLimit {
                    PanelHint(text: "+ \(batch.count - Self.batchListLimit) more ligands — open the Results Database to browse all.",
                              icon: "ellipsis.circle")
                }
            }
        }
    }

    private var batchProgress: some View {
        let (current, total) = viewModel.docking.batchProgress
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            ProgressView(value: total > 0 ? Double(current) / Double(total) : 0) {
                HStack {
                    Text(viewModel.workspace.statusMessage)
                        .lineLimit(1)
                    Spacer(minLength: DS.Space.xs)
                    Text("\(current)/\(total)")
                        .monospacedDigit()
                }
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
            }
            .controlSize(.small)
            PanelSecondaryButton(title: "Cancel Batch", icon: "xmark", tint: DS.Color.danger,
                                 help: "Stop batch docking after the current ligand") {
                viewModel.cancelBatchDocking()
            }
            .accessibilityIdentifier(AccessibilityID.resultsCancelBatch)
        }
    }

    private func batchRow(rank: Int, ligandName: String, results: [DockingResult]) -> some View {
        let isActive = viewModel.molecules.ligand?.name == ligandName && !self.results.isEmpty
        let best = results.first?.displayScore(method: method)
        return Button {
            ResultsActions.loadBatchLigand(named: ligandName, results: results, viewModel: viewModel)
        } label: {
            HStack(spacing: DS.Space.s) {
                Text(verbatim: "#\(rank + 1)")
                    .font(DS.Font.monoSmall.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(rank == 0 ? AnyShapeStyle(DS.Color.success) : AnyShapeStyle(.secondary))
                    .frame(minWidth: 24, alignment: .trailing)
                Text(ligandName)
                    .font(DS.Font.label.weight(rank == 0 ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: DS.Space.xs)
                if let best { DSScoreBadge(best, method: method) }
                Text("\(results.count)p")
                    .font(DS.Font.caption)
                    .monospacedDigit()
                    .foregroundStyle(.tertiary)
                    .help("\(results.count) poses")
            }
            .padding(.horizontal, DS.Space.m)
            .frame(minHeight: DS.Size.rowHeight + 4)
            .background(
                RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                    .fill(isActive ? DS.Color.accentSoft : DS.Color.controlFill.opacity(0.5))
            )
            .contentShape(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous))
        }
        .buttonStyle(.plain)
        .help("Load \(ligandName)'s poses")
    }

    // MARK: - Screening

    private var screeningCard: some View {
        @Bindable var vm = viewModel
        let hits = filteredScreeningHits
        return PanelCard("Screening Hits", icon: "flask", accessory: {
            Text("\(hits.count)/\(viewModel.docking.screeningHits.count)")
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .help("Hits passing the filters / all hits")
        }) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if hasDockingSummary || !viewModel.docking.batchResults.isEmpty {
                    screeningMetrics
                }
                PanelLabeledDivider(title: "Filters", icon: "line.3.horizontal.decrease")
                PanelToggleRow(title: "Lipinski compliant only", isOn: $vm.docking.resultsLipinskiFilter)
                PanelSliderRow(
                    label: "Energy ≤",
                    value: $vm.docking.resultsEnergyCutoff,
                    range: -20...5, step: 0.5,
                    format: { String(format: "%.1f \(vm.docking.scoringMethod.unitLabel)", $0) }
                )
                PanelSliderRow(
                    label: "ML score ≥",
                    value: $vm.docking.resultsMLScoreCutoff,
                    range: 0...15, step: 0.5,
                    format: { String(format: "%.1f", $0) }
                )
                PanelLabeledDivider(title: "Hits")
                VStack(spacing: DS.Space.xs) {
                    ForEach(Array(hits.prefix(Self.poseListLimit).enumerated()), id: \.element.id) { idx, hit in
                        screeningHitRow(index: idx, hit: hit)
                    }
                }
                if hits.count > Self.poseListLimit {
                    PanelHint(text: "+ \(hits.count - Self.poseListLimit) more hits — export to see all.",
                              icon: "ellipsis.circle")
                }
                PanelSecondaryButton(
                    title: "Export Hits (CSV + SDF)",
                    icon: "square.and.arrow.up",
                    isDisabled: viewModel.docking.screeningHits.isEmpty,
                    help: "Export screening hits as CSV and SDF files"
                ) { viewModel.exportScreeningHits() }
                .accessibilityIdentifier(AccessibilityID.resultsExportHits)
            }
        }
    }

    private func screeningHitRow(index: Int, hit: VirtualScreeningPipeline.ScreeningHit) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                Text(verbatim: "#\(index + 1)")
                    .font(DS.Font.monoSmall.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 24, alignment: .trailing)
                VStack(alignment: .leading, spacing: 1) {
                    Text(hit.name)
                        .font(DS.Font.label.weight(.medium))
                        .lineLimit(1)
                    Text(hit.smiles)
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: DS.Space.xs)
                VStack(alignment: .trailing, spacing: 1) {
                    DSScoreBadge(hit.compositeScore, method: method)
                    if let ml = hit.mlScore {
                        Text(String(format: "ML %.1f", ml))
                            .font(DS.Font.caption)
                            .monospacedDigit()
                            .foregroundStyle(DS.Color.pocket)
                            .help("ML affinity score (pKd)")
                    }
                }
            }
            HStack(spacing: DS.Space.s) {
                if let desc = hit.descriptors {
                    Text(String(format: "MW %.0f · cLogP %.1f", desc.molecularWeight, desc.logP))
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    if desc.lipinski {
                        DSTag("Lipinski", tone: .success)
                    }
                }
                Spacer(minLength: DS.Space.xs)
                Button {
                    viewModel.selectReferenceFromHit(hit)
                } label: {
                    Label("Optimize", systemImage: "arrow.triangle.branch")
                }
                .buttonStyle(.borderless)
                .font(DS.Font.caption)
                .help("Use as reference for lead optimization")
            }
            .padding(.leading, 30)
        }
        .padding(.vertical, DS.Space.xxs)
    }

    // MARK: - Footer

    private var footer: some View {
        let hasPoses = !results.isEmpty
        let selected = viewModel.docking.selectedPoseIndices.count
        return VStack(spacing: DS.Space.s) {
            HStack(spacing: DS.Space.s) {
                PanelSecondaryButton(title: "Export SDF", icon: "doc.text",
                                     isDisabled: !hasPoses,
                                     help: "Export all poses as an SDF file") {
                    viewModel.exportDockingResultsSDF()
                }
                .accessibilityIdentifier(AccessibilityID.resultsExportSDF)
                PanelSecondaryButton(title: "Export CSV", icon: "tablecells",
                                     isDisabled: !hasPoses,
                                     help: "Export the results table as CSV") {
                    viewModel.exportDockingResultsCSV()
                }
                .accessibilityIdentifier(AccessibilityID.resultsExportCSV)
            }
            HStack(spacing: DS.Space.m) {
                Button {
                    multiPoseInitialMode = selected >= 2 ? .rmsd : .interactions
                    showMultiPoseAnalysis = true
                } label: {
                    Label("Multi-Pose Analysis…", systemImage: "square.stack.3d.up")
                        .font(DS.Font.label)
                }
                .buttonStyle(.borderless)
                .disabled(selected < 1)
                .help(selected < 1
                      ? "Tick poses in the list to compare RMSD and interaction fingerprints"
                      : "Pairwise RMSD and per-residue interaction fingerprints of the selected poses")
                Spacer(minLength: DS.Space.xs)
                DSNextStepButton(.leadOptimization)
            }
        }
    }

    // MARK: - Helpers

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        return "\(Int(seconds) / 60) m \(Int(seconds) % 60) s"
    }

    private var filteredScreeningHits: [VirtualScreeningPipeline.ScreeningHit] {
        viewModel.docking.screeningHits.filter { hit in
            if viewModel.docking.resultsLipinskiFilter {
                guard hit.descriptors?.lipinski == true else { return false }
            }
            if viewModel.docking.resultsEnergyCutoff < 0 {
                guard hit.compositeScore <= viewModel.docking.resultsEnergyCutoff else { return false }
            }
            if viewModel.docking.resultsMLScoreCutoff > 0 {
                if let ml = hit.mlScore {
                    guard ml >= viewModel.docking.resultsMLScoreCutoff else { return false }
                }
            }
            return true
        }
    }

    private var totalScreened: Int {
        if case .complete(_, let total) = viewModel.docking.screeningState {
            return total
        }
        return viewModel.docking.screeningHits.count
    }

    private func initializeCutoffsIfNeeded() {
        guard !viewModel.docking.resultsHasInitializedCutoffs,
              !viewModel.docking.screeningHits.isEmpty else { return }
        viewModel.docking.resultsHasInitializedCutoffs = true
        let energies = viewModel.docking.screeningHits.map(\.compositeScore).sorted()
        viewModel.docking.resultsEnergyCutoff = energies[energies.count / 2]
    }
}

// MARK: - Shared results actions

/// Results actions shared by the Results step, the Results Database and the
/// Interaction Diagram window.
@MainActor
enum ResultsActions {
    /// Makes a batch ligand's poses the current docking results and shows its best pose.
    static func loadBatchLigand(named name: String, results: [DockingResult], viewModel: AppViewModel) {
        viewModel.docking.dockingResults = results
        viewModel.docking.dockingPosesEvaluated = 0
        guard let entry = viewModel.ligandDB.entries.first(where: { $0.name == name }) else { return }
        let molecule = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds,
                                title: entry.smiles, smiles: entry.smiles)
        viewModel.setLigandForDocking(molecule)
        viewModel.docking.dockingResults = results
        viewModel.showDockingPose(at: 0)
    }
}

// MARK: - Score badge

extension DSScoreBadge {
    /// Score badge in the convention of `method`: pK for affinity models, energy otherwise.
    init(_ value: Float, method: ScoringMethod) {
        if method.isAffinityScore {
            self.init(pK: value, label: method.unitLabel)
        } else {
            self.init(energy: value, unit: method.unitLabel)
        }
    }
}

// MARK: - Pose row

private struct PoseRow: View {
    let index: Int
    let result: DockingResult
    let method: ScoringMethod
    let rmsd: Float?
    let isChecked: Bool
    let isActive: Bool
    let canOptimize: Bool
    let onToggle: () -> Void
    let onView: () -> Void
    let onDiagram: () -> Void
    let onSave: () -> Void
    let onOptimize: () -> Void

    @State private var isHovering = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            HStack(spacing: DS.Space.s) {
                Toggle("Select pose \(index + 1)", isOn: Binding(get: { isChecked }, set: { _ in onToggle() }))
                    .toggleStyle(.checkbox)
                    .labelsHidden()
                    .help("Select for multi-pose overlay and analysis")
                    .accessibilityIdentifier(AccessibilityID.resultsCheckbox(index))
                Text(verbatim: "#\(index + 1)")
                    .font(DS.Font.monoSmall.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 24, alignment: .trailing)
                DSScoreBadge(result.displayScore(method: method), method: method)
                if result.clusterID >= 0 {
                    DSTag("C\(result.clusterID)", monospaced: true)
                        .help("Cluster \(result.clusterID) — RMSD-based binding-mode group")
                }
                Spacer(minLength: DS.Space.xs)
                if let rmsd {
                    Text(String(format: "%.1f Å", rmsd))
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .help("Heavy-atom RMSD to pose #1")
                }
            }
            HStack(spacing: DS.Space.xs) {
                terms
                Spacer(minLength: DS.Space.xs)
                actions
            }
            .padding(.leading, 22)
        }
        .padding(.horizontal, DS.Space.s)
        .padding(.vertical, DS.Space.xs)
        .background(shape.fill(isActive ? DS.Color.accentSoft : (isHovering ? DS.Color.controlFill : Color.clear)))
        .overlay(shape.strokeBorder(isChecked ? DS.Color.accent.opacity(0.5) : Color.clear, lineWidth: 1))
        .contentShape(shape)
        .onTapGesture { onView() }
        .onHover { isHovering = $0 }
    }

    private var terms: some View {
        FlowLayout(spacing: DS.Space.xs) {
            term("vdW", result.stericEnergy)
            term("hyd", result.hydrophobicEnergy)
            term("hb", result.hbondEnergy)
            if let strain = result.strainEnergy, strain > 4.0 {
                DSTag(String(format: "str %.1f", strain), tone: strain > 10.0 ? .danger : .warning, monospaced: true)
                    .help("Ligand strain energy (kcal/mol)")
            }
            if let gfn2 = result.gfn2Energy {
                DSTag(String(format: "xtb %.1f", gfn2), tone: .pocket, monospaced: true)
                    .help("GFN2-xTB energy (kcal/mol)")
            }
            if result.constraintPenalty > 0.01 {
                DSTag(String(format: "cst %.1f", result.constraintPenalty), tone: .warning, monospaced: true)
                    .help("Pharmacophore constraint penalty")
            }
        }
    }

    private func term(_ label: String, _ value: Float) -> some View {
        HStack(spacing: DS.Space.xxs) {
            Text(label).foregroundStyle(.tertiary)
            Text(String(format: "%.1f", value)).foregroundStyle(.secondary)
        }
        .font(DS.Font.monoSmall)
        .monospacedDigit()
    }

    private var actions: some View {
        HStack(spacing: DS.Space.xxs) {
            actionButton("Show in 3D", systemImage: "eye", id: AccessibilityID.resultsViewPose(index), action: onView)
            actionButton("2D Interaction Diagram", systemImage: "point.3.connected.trianglepath.dotted",
                         id: AccessibilityID.resultsDiagram(index), action: onDiagram)
            actionButton("Save pose (protein + ligand) as PDB", systemImage: "square.and.arrow.down",
                         isDisabled: result.transformedAtomPositions.isEmpty, action: onSave)
            if canOptimize {
                actionButton("Use as lead-optimization reference", systemImage: "arrow.triangle.branch",
                             id: AccessibilityID.resultsOptimize(index), action: onOptimize)
            }
        }
    }

    private func actionButton(_ help: String, systemImage: String, id: String? = nil,
                              isDisabled: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(DS.Font.secondary.weight(.medium))
                .frame(width: 22, height: 20)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(isDisabled)
        .help(help)
        .accessibilityLabel(help)
        .modifier(OptionalAccessibilityID(id: id))
    }
}

/// Conditional accessibility identifier modifier.
private struct OptionalAccessibilityID: ViewModifier {
    let id: String?
    func body(content: Content) -> some View {
        if let id { content.accessibilityIdentifier(id) } else { content }
    }
}

// MARK: - Score landscape chart

/// Bar chart of the ranked poses' scores; click a bar to show that pose.
private struct ScoreLandscapeChart: View {
    let scores: [Float]
    let isAffinity: Bool
    let unit: String
    let activeIndex: Int?
    let onSelect: (Int) -> Void

    var body: some View {
        // Clamp outliers (1.5 × IQR) so one bad pose doesn't flatten the chart.
        let sorted = scores.sorted()
        let q1 = sorted[sorted.count / 4]
        let q3 = sorted[sorted.count * 3 / 4]
        let iqr = q3 - q1
        let low = q1 - iqr * 1.5
        let high = q3 + iqr * 1.5

        VStack(alignment: .leading, spacing: DS.Space.xs) {
            HStack {
                PanelSubheader(title: isAffinity ? "Affinity landscape" : "Energy landscape")
                Spacer()
                if let activeIndex, activeIndex < scores.count {
                    Text(String(format: "Pose #%d · %.2f %@", activeIndex + 1, scores[activeIndex], unit))
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
            Chart {
                ForEach(Array(scores.enumerated()), id: \.offset) { index, score in
                    BarMark(
                        x: .value("Pose", "\(index + 1)"),
                        y: .value("Score", max(low, min(high, score)))
                    )
                    .foregroundStyle(barColor(index: index, score: score))
                    .cornerRadius(1.5)
                }
            }
            .chartXAxis(.hidden)
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { _ in
                    AxisGridLine().foregroundStyle(DS.Color.hairline)
                    AxisValueLabel().font(DS.Font.caption)
                }
            }
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded { value in
                            guard let plot = proxy.plotFrame else { return }
                            let x = value.location.x - geo[plot].origin.x
                            if let label: String = proxy.value(atX: x), let pose = Int(label) {
                                onSelect(pose - 1)
                            }
                        })
                }
            }
            .frame(height: 72)
            .accessibilityLabel("\(isAffinity ? "Affinity" : "Energy") landscape of \(scores.count) poses")
        }
        .padding(DS.Space.m)
        .background(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous).fill(DS.Color.cardFill))
    }

    private func barColor(index: Int, score: Float) -> Color {
        if index == activeIndex { return DS.Color.accent }
        let tone = isAffinity ? DSScore.tone(pK: score) : DSScore.tone(energy: score)
        return (tone == .danger ? DS.Color.scorePoor : tone.color).opacity(0.75)
    }
}
