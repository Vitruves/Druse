// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

struct ResultsTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    @State private var showMultiPoseAnalysis = false
    @State private var multiPoseInitialMode: MultiPoseAnalysisSheet.Mode = .rmsd
    @State private var showExportMenu = false

    var body: some View {
        VStack(alignment: .leading, spacing: PanelStyle.cardSpacing) {
            // Always-visible "Open Results Database" button when results exist.
            if hasAnyResults {
                openDatabaseButton
            }

            if viewModel.docking.isBatchDocking {
                batchProgressCard
            }

            if !viewModel.docking.batchResults.isEmpty && !viewModel.docking.isBatchDocking {
                batchResultsCard
            }

            if !viewModel.docking.dockingResults.isEmpty
                && !viewModel.docking.isDocking
                && viewModel.docking.batchResults.isEmpty {
                dockingResultsCard
                exportCard
            }

            if !viewModel.docking.screeningHits.isEmpty {
                screeningResultsCard
                screeningFilterCard
                screeningExportCard
            }

            if viewModel.docking.dockingResults.isEmpty
                && viewModel.docking.screeningHits.isEmpty
                && !viewModel.docking.isDocking {
                emptyStateCard
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .onChange(of: viewModel.docking.screeningHits.count) { _, _ in
            initializeCutoffsIfNeeded()
        }
        .onChange(of: viewModel.docking.showInteractionDiagram) { _, show in
            if show {
                openWindow(id: "interaction-diagram")
                viewModel.docking.showInteractionDiagram = false
            }
        }
        .sheet(isPresented: $showMultiPoseAnalysis) {
            MultiPoseAnalysisSheet(initialMode: multiPoseInitialMode)
                .environment(viewModel)
        }
    }

    private var hasAnyResults: Bool {
        !viewModel.docking.dockingResults.isEmpty || !viewModel.docking.batchResults.isEmpty
    }

    private var openDatabaseButton: some View {
        PanelRunButton(
            title: "Open Results Database",
            icon: "tablecells",
            color: .accentColor
        ) {
            openWindow(id: "results-database")
        }
        .accessibilityIdentifier(AccessibilityID.resultsOpenDB)
    }

    // MARK: - Docking results card

    @ViewBuilder
    private var dockingResultsCard: some View {
        let results = viewModel.docking.dockingResults
        PanelCard(
            "Docking Results",
            icon: "chart.bar.xaxis",
            accessory: { ligandNameChip }
        ) {
            VStack(alignment: .leading, spacing: 10) {
                if let best = results.first {
                    summaryStrip(bestResult: best)
                }
                if results.count > 1 {
                    energyLandscapeChart
                }
                if viewModel.docking.selectedPoseIndices.count > 1 {
                    multiPoseActions
                }
                if viewModel.molecules.ligand != nil {
                    PanelSecondaryButton(
                        title: "Clear Poses from View",
                        icon: "eye.slash",
                        help: "Remove all displayed poses and interaction lines from the 3D viewport"
                    ) { viewModel.clearPosesFromView() }
                    .accessibilityIdentifier(AccessibilityID.resultsClearPoses)
                }
                PanelLabeledDivider(title: "Poses")
                ForEach(Array(results.prefix(50).enumerated()), id: \.offset) { idx, result in
                    poseRow(index: idx, result: result)
                }
                if results.count > 50 {
                    Text("+ \(results.count - 50) more poses")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
        }
    }

    @ViewBuilder
    private var ligandNameChip: some View {
        if let ligand = viewModel.molecules.ligand, !ligand.name.isEmpty {
            let displayName: String = {
                if let entry = viewModel.ligandDB.entries.first(where: { $0.name == ligand.name || $0.smiles == (ligand.smiles ?? "") }),
                   let parent = entry.parentName {
                    return "\(ligand.name) ◂ \(parent)"
                }
                return ligand.name
            }()
            Text(displayName)
                .font(PanelStyle.smallFont.weight(.medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.accentColor.opacity(0.12)))
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private func summaryStrip(bestResult: DockingResult) -> some View {
        let method = viewModel.docking.scoringMethod
        let bestValue = bestResult.displayScore(method: method)
        let color = scoreColor(bestValue, method: method)
        let results = viewModel.docking.dockingResults
        let clusters = Set(results.map(\.clusterID)).count
        HStack(spacing: 8) {
            PanelStat(label: "Best",
                      value: String(format: "%.1f", bestValue),
                      unit: method.unitLabel,
                      color: color)
            PanelStat(label: "Poses", value: "\(results.count)")
            PanelStat(label: "Clusters", value: "\(clusters)", color: .cyan)
        }
        .padding(8)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.green.opacity(0.08))
        )
    }

    @ViewBuilder
    private var multiPoseActions: some View {
        let count = viewModel.docking.selectedPoseIndices.count
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.blue)
                Text("\(count) poses selected")
                    .font(PanelStyle.smallFont.weight(.medium))
                Spacer()
                Button("Clear") {
                    viewModel.docking.selectedPoseIndices.removeAll()
                }
                .font(PanelStyle.smallFont)
                .buttonStyle(.plain)
                .foregroundStyle(.red)
                .accessibilityIdentifier(AccessibilityID.resultsClearSelection)
            }
            PanelChoiceGrid(columns: 2) {
                PanelSecondaryButton(
                    title: "View 3D", icon: "eye",
                    help: "Show selected poses in the 3D viewport (best as primary, others as ghosts)"
                ) { viewModel.showSelectedPoses() }
                .accessibilityIdentifier(AccessibilityID.resultsViewSelected)

                Menu {
                    Button("Export as SDF") { viewModel.exportSelectedDockingPosesSDF() }
                    Button("Export as PDB") { viewModel.exportSelectedDockingPosesPDB() }
                    Button("Export as CSV") { viewModel.exportSelectedDockingPosesCSV() }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 11, weight: .medium))
                        Text("Export…")
                            .font(PanelStyle.bodyFont)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: PanelStyle.buttonHeight)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(Color.primary.opacity(PanelStyle.chipFillOpacity))
                    )
                    .contentShape(Rectangle())
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .help("Export selected poses (SDF / PDB / CSV)")
            }
            PanelChoiceGrid(columns: 2) {
                PanelSecondaryButton(
                    title: "RMSD", icon: "square.grid.3x3",
                    isDisabled: count < 2,
                    help: "Pairwise heavy-atom RMSD between selected poses"
                ) {
                    multiPoseInitialMode = .rmsd
                    showMultiPoseAnalysis = true
                }
                PanelSecondaryButton(
                    title: "Fingerprints", icon: "circle.hexagongrid.fill",
                    isDisabled: count < 1,
                    help: "Per-residue interaction fingerprint heatmap"
                ) {
                    multiPoseInitialMode = .interactions
                    showMultiPoseAnalysis = true
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.blue.opacity(0.08))
        )
    }

    // MARK: - Energy landscape

    @ViewBuilder
    private var energyLandscapeChart: some View {
        let results = viewModel.docking.dockingResults
        let method = viewModel.docking.scoringMethod
        let scores: [Float] = results.map { $0.displayScore(method: method) }
        let selectedIdx = viewModel.docking.selectedPoseIndices.count == 1
            ? viewModel.docking.selectedPoseIndices.first
            : nil
        let title = method.isAffinityScore ? "Affinity Landscape" : "Energy Landscape"

        let sorted = scores.sorted()
        let q1 = sorted[sorted.count / 4]
        let q3 = sorted[sorted.count * 3 / 4]
        let iqr = q3 - q1
        let clampLow = q1 - iqr * 1.5
        let clampHigh = q3 + iqr * 1.5
        let clampedScores = scores.map { max(clampLow, min(clampHigh, $0)) }
        let minE = clampedScores.min() ?? 0
        let maxE = clampedScores.max() ?? 0
        let range = max(abs(maxE - minE), 0.1)

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(PanelStyle.smallFont.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if let sel = selectedIdx, sel < results.count {
                    Text("Pose \(sel + 1): \(String(format: "%.2f", scores[sel])) \(method.unitLabel)")
                        .font(PanelStyle.monoSmall.weight(.medium))
                }
            }
            GeometryReader { geo in
                let barCount = min(results.count, 50)
                let spacing: CGFloat = 1
                let totalSpacing = spacing * CGFloat(barCount - 1)
                let barWidth = max(2, (geo.size.width - totalSpacing) / CGFloat(barCount))
                let chartHeight = geo.size.height
                HStack(alignment: .bottom, spacing: spacing) {
                    ForEach(0..<barCount, id: \.self) { idx in
                        let score = clampedScores[idx]
                        let normalized: CGFloat = CGFloat((score - minE) / range)
                        let barHeight = max(3, (1.0 - normalized) * (chartHeight - 4) + 4)
                        let isSelected = selectedIdx == idx
                        Rectangle()
                            .fill(barColor(energy: scores[idx], isSelected: isSelected))
                            .frame(width: barWidth, height: barHeight)
                            .clipShape(RoundedRectangle(cornerRadius: 1.5))
                            .overlay(
                                isSelected
                                    ? RoundedRectangle(cornerRadius: 1.5)
                                        .stroke(Color.primary.opacity(0.8), lineWidth: 1.5)
                                    : nil
                            )
                            .onTapGesture { viewModel.showDockingPose(at: idx) }
                    }
                }
            }
            .frame(height: 56)
            HStack {
                Text(String(format: "%.1f", minE))
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.green)
                Spacer()
                Text(method.unitLabel)
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f", maxE))
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(maxE > 0 ? .red : .yellow)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.primary.opacity(0.04))
        )
    }

    private func barColor(energy: Float, isSelected: Bool) -> Color {
        if isSelected { return .white }
        if energy < -8 { return .green }
        if energy < -6 { return .green.opacity(0.7) }
        if energy < -4 { return .yellow.opacity(0.7) }
        if energy < -2 { return .orange.opacity(0.7) }
        return .red.opacity(0.7)
    }

    // MARK: - Pose row

    @ViewBuilder
    private func poseRow(index: Int, result: DockingResult) -> some View {
        let isSelected = viewModel.docking.selectedPoseIndices.contains(index)
        let method = viewModel.docking.scoringMethod
        let score = result.displayScore(method: method)
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button(action: { viewModel.togglePoseSelection(at: index) }) {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 14))
                        .foregroundStyle(isSelected ? Color.blue : Color.gray.opacity(0.4))
                }
                .buttonStyle(.plain)
                .help("Select for multi-pose overlay")

                Text("#\(index + 1)")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 24, alignment: .trailing)

                Text(String(format: "%.2f", score))
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(scoreColor(score, method: method))

                Text(method.unitLabel)
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)

                Spacer()

                if result.clusterID >= 0 {
                    Text("C\(result.clusterID)")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.cyan.opacity(0.18)))
                        .foregroundStyle(.cyan)
                        .help("Cluster \(result.clusterID) — RMSD-based binding-mode group.")
                }
            }

            FlowLayout(spacing: 4) {
                energyTag("vdW", result.vdwEnergy)
                energyTag("elec", result.elecEnergy)
                energyTag("hb", result.hbondEnergy)
                if let strain = result.strainEnergy, strain > 4.0 {
                    energyTag("str", strain, color: strain > 10.0 ? .red : .orange)
                }
                if let gfn2 = result.gfn2Energy {
                    energyTag("xtb", gfn2, color: .purple)
                }
                if result.constraintPenalty > 0.01 {
                    energyTag("cst", result.constraintPenalty, color: .orange)
                }
            }

            HStack(spacing: 4) {
                Spacer()
                poseIconButton(icon: "circle.hexagongrid",
                               help: "2D Interaction Diagram",
                               id: AccessibilityID.resultsDiagram(index)) {
                    viewModel.showDockingPose(at: index)
                    viewModel.docking.interactionDiagramPoseIndex = index
                    viewModel.docking.showInteractionDiagram = true
                }
                poseIconButton(icon: "eye",
                               help: "Show this pose in 3D viewport",
                               id: AccessibilityID.resultsViewPose(index)) {
                    viewModel.showDockingPose(at: index)
                }
                poseIconButton(icon: "square.and.arrow.down",
                               help: "Save this pose (protein + ligand) as a .pdb file",
                               isDisabled: result.transformedAtomPositions.isEmpty) {
                    viewModel.exportDockingPosePDB(at: index)
                }
                if let lig = viewModel.docking.originalDockingLigand ?? viewModel.molecules.ligand {
                    poseIconButton(icon: "arrow.triangle.branch",
                                   help: "Use as reference for lead optimization",
                                   id: AccessibilityID.resultsOptimize(index)) {
                        viewModel.selectReferenceForOptimization(result: result, ligand: lig)
                    }
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(isSelected ? Color.blue.opacity(0.08) : Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(isSelected ? Color.blue.opacity(0.3) : Color.clear, lineWidth: 1)
        )
    }

    @ViewBuilder
    private func poseIconButton(
        icon: String,
        help: String,
        id: String? = nil,
        isDisabled: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .frame(width: 28, height: 22)
                .background(
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .fill(Color.primary.opacity(0.08))
                )
                .foregroundStyle(.primary)
                .opacity(isDisabled ? 0.4 : 1.0)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .help(help)
        .modifier(OptionalAccessibilityID(id: id))
    }

    @ViewBuilder
    private func energyTag(_ label: String, _ value: Float, color: Color = .secondary) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(color.opacity(0.8))
            Text(String(format: "%.1f", value))
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(color)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
        .background(
            RoundedRectangle(cornerRadius: 4, style: .continuous)
                .fill(Color.primary.opacity(0.05))
        )
    }

    // MARK: - Export card

    @ViewBuilder
    private var exportCard: some View {
        PanelCard("Export", icon: "square.and.arrow.up") {
            PanelChoiceGrid(columns: 2) {
                PanelSecondaryButton(
                    title: "Export SDF", icon: "doc.text",
                    help: "Export all poses as SDF file"
                ) { viewModel.exportDockingResultsSDF() }
                .accessibilityIdentifier(AccessibilityID.resultsExportSDF)

                PanelSecondaryButton(
                    title: "Export CSV", icon: "tablecells",
                    help: "Export results table as CSV"
                ) { viewModel.exportDockingResultsCSV() }
                .accessibilityIdentifier(AccessibilityID.resultsExportCSV)
            }
        }
    }

    // MARK: - Screening results card

    @ViewBuilder
    private var screeningResultsCard: some View {
        PanelCard("Screening Results", icon: "flask") {
            VStack(alignment: .leading, spacing: 10) {
                screeningSummaryStrip
                ForEach(Array(filteredScreeningHits.prefix(50).enumerated()), id: \.element.id) { idx, hit in
                    screeningHitRow(index: idx, hit: hit)
                }
                if filteredScreeningHits.count > 50 {
                    Text("+ \(filteredScreeningHits.count - 50) more hits")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
        }
    }

    @ViewBuilder
    private var screeningSummaryStrip: some View {
        HStack(spacing: 8) {
            PanelStat(label: "Hits", value: "\(filteredScreeningHits.count)", color: .blue)
            PanelStat(label: "Screened", value: "\(totalScreened)")
            if let best = filteredScreeningHits.first {
                let sm = viewModel.docking.scoringMethod
                PanelStat(label: "Best",
                          value: String(format: "%.1f", best.compositeScore),
                          unit: sm.unitLabel,
                          color: scoreColor(best.compositeScore, method: sm))
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.blue.opacity(0.08))
        )
    }

    @ViewBuilder
    private func screeningHitRow(index: Int, hit: VirtualScreeningPipeline.ScreeningHit) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("#\(index + 1)")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .frame(width: 26, alignment: .trailing)
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(hit.name)
                        .font(PanelStyle.bodyFont.weight(.medium))
                        .lineLimit(1)
                    Text(hit.smiles)
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 1) {
                    Text(String(format: "%.1f", hit.compositeScore))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(scoreColor(hit.compositeScore, method: viewModel.docking.scoringMethod))
                    if let ml = hit.mlScore {
                        Text(String(format: "ML:%.1f", ml))
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.purple)
                    }
                }
            }
            if let desc = hit.descriptors {
                HStack(spacing: 6) {
                    Text(String(format: "MW:%.0f", desc.molecularWeight))
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                    Text(String(format: "cLogP:%.1f", desc.logP))
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                    if desc.lipinski {
                        Text("Lipinski")
                            .font(.system(size: 10, weight: .medium))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Capsule().fill(Color.green.opacity(0.18)))
                            .foregroundStyle(.green)
                    }
                    Spacer()
                    PanelSecondaryButton(title: "Optimize", icon: "arrow.triangle.branch",
                                         help: "Use as reference for lead optimization") {
                        viewModel.selectReferenceFromHit(hit)
                    }
                    .frame(width: 96)
                }
                .padding(.leading, 30)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Screening filters card

    @ViewBuilder
    private var screeningFilterCard: some View {
        @Bindable var vm = viewModel
        PanelCard(
            "Filters",
            icon: "line.3.horizontal.decrease",
            accessory: {
                Text("\(filteredScreeningHits.count)/\(viewModel.docking.screeningHits.count)")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                PanelToggleRow(
                    title: "Lipinski compliant only",
                    isOn: $vm.docking.resultsLipinskiFilter
                )
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
            }
        }
    }

    // MARK: - Screening export card

    @ViewBuilder
    private var screeningExportCard: some View {
        PanelCard("Export Hits", icon: "square.and.arrow.up") {
            PanelSecondaryButton(
                title: "Export CSV + SDF",
                icon: "square.and.arrow.up",
                isDisabled: viewModel.docking.screeningHits.isEmpty,
                help: "Export screening hits as CSV and SDF files"
            ) { viewModel.exportScreeningHits() }
            .accessibilityIdentifier(AccessibilityID.resultsExportHits)
        }
    }

    // MARK: - Batch progress card

    @ViewBuilder
    private var batchProgressCard: some View {
        PanelCard("Batch Docking", icon: "arrow.triangle.merge") {
            VStack(alignment: .leading, spacing: 8) {
                let (current, total) = viewModel.docking.batchProgress
                ProgressView(value: total > 0 ? Double(current) / Double(total) : 0) {
                    Text(viewModel.workspace.statusMessage)
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                }
                .controlSize(.small)
                if !viewModel.docking.batchResults.isEmpty {
                    Text("\(viewModel.docking.batchResults.count) ligands docked so far")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                }
                PanelSecondaryButton(title: "Cancel", icon: "xmark", tint: .red) {
                    viewModel.cancelBatchDocking()
                }
                .accessibilityIdentifier(AccessibilityID.resultsCancelBatch)
            }
        }
    }

    // MARK: - Batch results card

    @ViewBuilder
    private var batchResultsCard: some View {
        PanelCard(
            "Batch Results",
            icon: "chart.bar.xaxis",
            accessory: {
                Text("\(viewModel.docking.batchResults.count) ligands")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
            }
        ) {
            VStack(alignment: .leading, spacing: 10) {
                if let best = viewModel.docking.batchResults.first,
                   let bestPose = best.results.first {
                    let sm = viewModel.docking.scoringMethod
                    let bpScore = bestPose.displayScore(method: sm)
                    HStack(spacing: 8) {
                        PanelStat(label: "Top Hit", value: best.ligandName, color: .green)
                        PanelStat(label: "Score",
                                  value: String(format: "%.1f", bpScore),
                                  unit: sm.unitLabel,
                                  color: scoreColor(bpScore, method: sm))
                        PanelStat(label: "Ligands", value: "\(viewModel.docking.batchResults.count)")
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(Color.cyan.opacity(0.08))
                    )
                }
                ForEach(Array(viewModel.docking.batchResults.prefix(10).enumerated()), id: \.offset) { rank, entry in
                    batchResultRow(rank: rank, ligandName: entry.ligandName, results: entry.results)
                }
                if viewModel.docking.batchResults.count > 10 {
                    Text("+ \(viewModel.docking.batchResults.count - 10) more ligands")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
        }
    }

    @ViewBuilder
    private func batchResultRow(rank: Int, ligandName: String, results: [DockingResult]) -> some View {
        let bestEnergy = results.first?.energy ?? .infinity
        Button(action: {
            viewModel.docking.dockingResults = results
            if let entry = viewModel.ligandDB.entries.first(where: { $0.name == ligandName }),
               results.first != nil {
                let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)
                viewModel.setLigandForDocking(mol)
                viewModel.docking.dockingResults = results
                viewModel.showDockingPose(at: 0)
            }
        }) {
            HStack(spacing: 6) {
                Text("#\(rank + 1)")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(rank == 0 ? .green : .secondary)
                    .frame(width: 24, alignment: .trailing)
                Text(ligandName)
                    .font(PanelStyle.smallFont.weight(rank == 0 ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Text(String(format: "%.1f", bestEnergy))
                    .font(PanelStyle.monoSmall.weight(.medium))
                    .foregroundStyle(energyColor(bestEnergy))
                Text("\(results.count) poses")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .padding(.horizontal, 8)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(Color.primary.opacity(0.04))
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Empty state

    @ViewBuilder
    private var emptyStateCard: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("No Results Yet")
                .font(PanelStyle.titleFont)
                .foregroundStyle(.secondary)
            Text("Run docking or virtual screening to see results here.")
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: PanelStyle.cardCornerRadius, style: .continuous)
                .fill(Color.primary.opacity(PanelStyle.cardFillOpacity))
        )
    }

    // MARK: - Helpers

    private func energyColor(_ energy: Float) -> Color {
        if energy < -8 { return .green }
        if energy < -4 { return .yellow }
        if energy < 0  { return .orange }
        return .red
    }

    private func scoreColor(_ value: Float, method: ScoringMethod) -> Color {
        if method.isAffinityScore {
            if value > 8 { return .green }
            if value > 5 { return .yellow }
            if value > 3 { return .orange }
            return .red
        }
        return energyColor(value)
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
        if let median = energies[safe: energies.count / 2] {
            viewModel.docking.resultsEnergyCutoff = median
        }
    }
}

// Conditional accessibility identifier modifier
private struct OptionalAccessibilityID: ViewModifier {
    let id: String?
    func body(content: Content) -> some View {
        if let id { content.accessibilityIdentifier(id) } else { content }
    }
}

// Safe array subscript
private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
