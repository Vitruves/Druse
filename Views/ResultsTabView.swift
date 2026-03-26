import SwiftUI
import AppKit

struct ResultsTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    // Screening hit filters (stored in viewModel to persist across tab switches)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Results Database button always on top when results exist
            if !viewModel.docking.dockingResults.isEmpty || !viewModel.docking.batchResults.isEmpty {
                Button(action: { openWindow(id: "results-database") }) {
                    Label("Open Results Database", systemImage: "tablecells")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .help("Open full results database window for detailed analysis")
                .accessibilityIdentifier(AccessibilityID.resultsOpenDB)
                Divider()
            }

            // Batch docking progress
            if viewModel.docking.isBatchDocking {
                batchProgressSection
            }

            // Batch docking results (ranked across all ligands)
            if !viewModel.docking.batchResults.isEmpty && !viewModel.docking.isBatchDocking {
                batchResultsSection
                Divider()
            }

            if !viewModel.docking.dockingResults.isEmpty && !viewModel.docking.isDocking && viewModel.docking.batchResults.isEmpty {
                dockingResultsSection
                Divider()
                dockingExportSection
            }

            // Show screening results section when available
            if !viewModel.docking.screeningHits.isEmpty {
                if !viewModel.docking.dockingResults.isEmpty { Divider() }
                screeningResultsSection
                Divider()
                screeningFilterSection
                Divider()
                screeningExportSection
            }

            // Empty state
            if viewModel.docking.dockingResults.isEmpty && viewModel.docking.screeningHits.isEmpty && !viewModel.docking.isDocking {
                emptyState
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .onChange(of: viewModel.docking.screeningHits.count) { _, _ in
            initializeCutoffsIfNeeded()
        }
        .sheet(isPresented: Binding(
            get: { viewModel.docking.showInteractionDiagram },
            set: { viewModel.docking.showInteractionDiagram = $0 }
        )) {
            if let ligand = viewModel.molecules.ligand,
               let protein = viewModel.molecules.protein,
               viewModel.docking.interactionDiagramPoseIndex < viewModel.docking.dockingResults.count {
                let idx = viewModel.docking.interactionDiagramPoseIndex
                let result = viewModel.docking.dockingResults[idx]
                InteractionDiagramView(
                    interactions: viewModel.docking.currentInteractions,
                    ligandAtoms: ligand.atoms.filter { $0.element != .H },
                    ligandBonds: ligand.bonds,
                    proteinAtoms: protein.atoms.filter { $0.element != .H },
                    ligandSmiles: ligand.smiles ?? ligand.title,
                    poseEnergy: result.energy,
                    poseIndex: idx
                )
                .frame(minWidth: 650, minHeight: 550)
            }
        }
    }

    // MARK: - Docking Results

    @ViewBuilder
    private var dockingResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Docking Results", systemImage: "chart.bar.xaxis")
                    .font(.callout.weight(.semibold))
                if let ligand = viewModel.molecules.ligand, !ligand.name.isEmpty {
                    // Show ligand name with variant lineage if available
                    let displayName: String = {
                        if let entry = viewModel.ligandDB.entries.first(where: { $0.name == ligand.name || $0.smiles == (ligand.smiles ?? "") }) {
                            if let lineage = entry.variantLineage {
                                return "\(ligand.name) (\(lineage))"
                            }
                        }
                        return ligand.name
                    }()
                    Text(displayName)
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor.opacity(0.1)))
                        .lineLimit(1)
                }
            }

            // Summary card
            if let best = viewModel.docking.dockingResults.first {
                let bestValue = best.energy
                let color: Color = bestValue < -6 ? .green : bestValue < 0 ? .yellow : .red
                summaryCard(
                    bestDisplay: String(format: "%.1f", bestValue),
                    bestUnit: "kcal/mol",
                    bestColor: color,
                    poseCount: viewModel.docking.dockingResults.count,
                    clusterCount: Set(viewModel.docking.dockingResults.map(\.clusterID)).count
                )
            }

            // Energy landscape bar chart
            if viewModel.docking.dockingResults.count > 1 {
                energyLandscapeChart
            }

            // Multi-pose action bar
            if viewModel.docking.selectedPoseIndices.count > 1 {
                HStack {
                    Text("\(viewModel.docking.selectedPoseIndices.count) poses selected")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("View Selected") {
                        viewModel.showSelectedPoses()
                    }
                    .controlSize(.small)
                    .buttonStyle(.borderedProminent)
                    .accessibilityIdentifier(AccessibilityID.resultsViewSelected)
                    Button("Clear") {
                        viewModel.docking.selectedPoseIndices.removeAll()
                    }
                    .controlSize(.small)
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier(AccessibilityID.resultsClearSelection)
                }
                .padding(.vertical, 4)
            }

            // Clear poses from view
            if viewModel.molecules.ligand != nil && !viewModel.docking.dockingResults.isEmpty {
                HStack {
                    Spacer()
                    Button {
                        viewModel.clearPosesFromView()
                    } label: {
                        Label("Clear Poses from View", systemImage: "eye.slash")
                            .font(.footnote)
                    }
                    .controlSize(.mini)
                    .buttonStyle(.bordered)
                    .help("Remove all displayed poses and interaction lines from the 3D viewport")
                    .accessibilityIdentifier(AccessibilityID.resultsClearPoses)
                }
            }

            // Scrollable pose list (up to 50)
            ForEach(Array(viewModel.docking.dockingResults.prefix(50).enumerated()), id: \.offset) { idx, result in
                dockingPoseRow(index: idx, result: result)
            }

            if viewModel.docking.dockingResults.count > 50 {
                Text("+ \(viewModel.docking.dockingResults.count - 50) more poses")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }

    @ViewBuilder
    private func summaryCard(bestDisplay: String, bestUnit: String, bestColor: Color, poseCount: Int, clusterCount: Int, confidence: Float? = nil) -> some View {
        HStack(spacing: 12) {
            statBadge("Best", bestDisplay, unit: bestUnit, color: bestColor)
            statBadge("Poses", "\(poseCount)")
            statBadge("Clusters", "\(clusterCount)")
            if let conf = confidence {
                statBadge("Conf", String(format: "%.0f%%", conf * 100), color: conf > 0.7 ? .green : conf > 0.4 ? .yellow : .red)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.green.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.green.opacity(0.15), lineWidth: 1)
        )
    }

    // MARK: - Energy Landscape Bar Chart

    @ViewBuilder
    private var energyLandscapeChart: some View {
        let results = viewModel.docking.dockingResults
        let scores: [Float] = results.map(\.energy)
        let selectedIdx = viewModel.docking.selectedPoseIndices.count == 1 ? viewModel.docking.selectedPoseIndices.first : nil
        let chartTitle = "Energy Landscape"
        let unitLabel = "kcal/mol"

        // Clamp outliers: use interquartile range to determine useful display range
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
                Text(chartTitle)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if let sel = selectedIdx, sel < results.count {
                    let val = scores[sel]
                    let display = String(format: "%.2f", val)
                    Text("Pose \(sel + 1): \(display) \(unitLabel)")
                        .font(.footnote.monospaced().weight(.medium))
                        .foregroundStyle(.primary)
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
                                isSelected ?
                                    RoundedRectangle(cornerRadius: 1.5)
                                        .stroke(Color.primary.opacity(0.8), lineWidth: 1.5)
                                    : nil
                            )
                            .onTapGesture {
                                viewModel.showDockingPose(at: idx)
                            }
                    }
                }
            }
            .frame(height: 60)

            HStack {
                Text(String(format: "%.1f", minE))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.green)
                Spacer()
                Text(unitLabel)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f", maxE))
                    .font(.footnote.monospaced())
                    .foregroundStyle(maxE > 0 ? .red : .yellow)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
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

    @ViewBuilder
    private func dockingPoseRow(index: Int, result: DockingResult) -> some View {
        let isSelected = viewModel.docking.selectedPoseIndices.contains(index)

        VStack(alignment: .leading, spacing: 4) {
            // Row 1: Selection toggle, rank, energy, cluster badge
            HStack(spacing: 8) {
                Button(action: { viewModel.togglePoseSelection(at: index) }) {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.body)
                        .foregroundStyle(isSelected ? Color.blue : Color.gray.opacity(0.4))
                }
                .buttonStyle(.plain)
                .help("Select for multi-pose overlay")

                Text("#\(index + 1)")
                    .font(.subheadline.monospaced().weight(.bold))
                    .foregroundStyle(.secondary)
                    .frame(width: 24, alignment: .trailing)

                Text(String(format: "%.2f", result.energy))
                    .font(.callout.monospaced().weight(.semibold))
                    .foregroundStyle(energyColor(result.energy))
                Text("kcal/mol")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                Spacer()

                if result.clusterID >= 0 {
                    Text("C\(result.clusterID)")
                        .font(.footnote.monospaced().weight(.medium))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.cyan.opacity(0.15)))
                        .foregroundStyle(.cyan)
                }
            }

            // Row 2: Energy decomposition (compact grid-like layout)
            HStack(spacing: 0) {
                energyTag("vdW", result.vdwEnergy, color: .secondary)
                energyTag("elec", result.elecEnergy, color: .secondary)
                energyTag("hb", result.hbondEnergy, color: .secondary)
                if let strain = result.strainEnergy, strain > 4.0 {
                    energyTag("str", strain, color: strain > 10.0 ? .red : .orange)
                        .help("MMFF94 strain energy: \(String(format: "%.1f", strain)) kcal/mol")
                }
                if let gfn2 = result.gfn2Energy {
                    energyTag("xtb", gfn2, color: .purple)
                        .help(String(format: "GFN2-xTB: %.1f kcal/mol (D4:%.1f, solv:%.1f)%@",
                                     gfn2,
                                     result.gfn2DispersionEnergy ?? 0,
                                     result.gfn2SolvationEnergy ?? 0,
                                     result.gfn2Converged == true ? "" : " [not converged]"))
                }
                if result.constraintPenalty > 0.01 {
                    energyTag("cst", result.constraintPenalty, color: .orange)
                }
                Spacer()
            }

            // Row 3: Action buttons (aligned right)
            HStack(spacing: 4) {
                Spacer()

                Button {
                    viewModel.showDockingPose(at: index)
                    viewModel.docking.interactionDiagramPoseIndex = index
                    viewModel.docking.showInteractionDiagram = true
                } label: {
                    Label("Diagram", systemImage: "circle.hexagongrid")
                        .font(.footnote)
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)
                .help("2D Interaction Diagram")
                .accessibilityIdentifier(AccessibilityID.resultsDiagram(index))

                Button {
                    viewModel.showDockingPose(at: index)
                } label: {
                    Label("View", systemImage: "eye")
                        .font(.footnote)
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)
                .help("Show this pose in 3D viewport")
                .accessibilityIdentifier(AccessibilityID.resultsViewPose(index))

                if let lig = viewModel.docking.originalDockingLigand ?? viewModel.molecules.ligand {
                    Button {
                        viewModel.selectReferenceForOptimization(result: result, ligand: lig)
                    } label: {
                        Label("Optimize", systemImage: "arrow.triangle.branch")
                            .font(.footnote)
                    }
                    .controlSize(.mini)
                    .buttonStyle(.bordered)
                    .help("Use as reference for lead optimization")
                    .accessibilityIdentifier(AccessibilityID.resultsOptimize(index))
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.blue.opacity(0.06) : Color.primary.opacity(0.02))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(isSelected ? Color.blue.opacity(0.2) : Color.clear, lineWidth: 1)
        )
    }

    /// Compact energy decomposition tag
    @ViewBuilder
    private func energyTag(_ label: String, _ value: Float, color: Color) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .font(.caption.weight(.medium))
                .foregroundStyle(color.opacity(0.7))
            Text(String(format: "%.1f", value))
                .font(.footnote.monospaced().weight(.medium))
                .foregroundStyle(color)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
        .background(Color.primary.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .padding(.trailing, 4)
    }

    // MARK: - Docking Export

    @ViewBuilder
    private var dockingExportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Export", systemImage: "square.and.arrow.up")
                .font(.callout.weight(.semibold))

            HStack(spacing: 8) {
                Button(action: { viewModel.exportDockingResultsSDF() }) {
                    Label("Export SDF", systemImage: "doc.text")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Export all poses as SDF file")
                .accessibilityIdentifier(AccessibilityID.resultsExportSDF)

                Button(action: { viewModel.exportDockingResultsCSV() }) {
                    Label("Export CSV", systemImage: "tablecells")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Export results table as CSV")
                .accessibilityIdentifier(AccessibilityID.resultsExportCSV)
            }
        }
    }

    // MARK: - Screening Results

    @ViewBuilder
    private var screeningResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Screening Results", systemImage: "flask")
                .font(.callout.weight(.semibold))

            // Summary
            HStack(spacing: 12) {
                statBadge("Hits", "\(filteredScreeningHits.count)")
                statBadge("Screened", "\(totalScreened)", color: .secondary)
                if let best = filteredScreeningHits.first {
                    statBadge("Best", String(format: "%.1f", best.compositeScore), unit: "kcal/mol",
                               color: best.compositeScore < -6 ? .green : .yellow)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.blue.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.blue.opacity(0.15), lineWidth: 1)
            )

            // Hit list
            ForEach(Array(filteredScreeningHits.prefix(50).enumerated()), id: \.element.id) { idx, hit in
                screeningHitRow(index: idx, hit: hit)
            }

            if filteredScreeningHits.count > 50 {
                Text("+ \(filteredScreeningHits.count - 50) more hits")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }

    @ViewBuilder
    private func screeningHitRow(index: Int, hit: VirtualScreeningPipeline.ScreeningHit) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text("#\(index + 1)")
                    .font(.footnote.monospaced().weight(.bold))
                    .frame(width: 26, alignment: .trailing)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    Text(hit.name)
                        .font(.subheadline.weight(.medium))
                        .lineLimit(1)
                    Text(hit.smiles)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f", hit.compositeScore))
                        .font(.subheadline.monospaced().weight(.semibold))
                        .foregroundStyle(energyColor(hit.compositeScore))
                    if let ml = hit.mlScore {
                        Text(String(format: "ML:%.1f", ml))
                            .font(.footnote.monospaced().weight(.medium))
                            .foregroundStyle(.purple)
                    }
                }
            }

            // Descriptor badges
            if let desc = hit.descriptors {
                HStack(spacing: 8) {
                    Text(String(format: "MW:%.0f", desc.molecularWeight))
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                    Text(String(format: "cLogP:%.1f", desc.logP))
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                    if desc.lipinski {
                        Text("Lipinski")
                            .font(.footnote.weight(.medium))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.green.opacity(0.15)))
                            .foregroundStyle(.green)
                    }
                }
                .padding(.leading, 30)
            }

            HStack {
                Spacer()
                Button("Optimize") {
                    viewModel.selectReferenceFromHit(hit)
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)
                .help("Use as reference for lead optimization")
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Screening Filters

    @ViewBuilder
    private var screeningFilterSection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 8) {
            Label("Filters", systemImage: "line.3.horizontal.decrease")
                .font(.callout.weight(.semibold))

            Toggle("Lipinski compliant only", isOn: $vm.docking.resultsLipinskiFilter)
                .font(.subheadline)

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Energy cutoff")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.1f kcal/mol", vm.docking.resultsEnergyCutoff))
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $vm.docking.resultsEnergyCutoff, in: -20...5, step: 0.5)
                    .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("ML score cutoff")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.1f", vm.docking.resultsMLScoreCutoff))
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $vm.docking.resultsMLScoreCutoff, in: 0...15, step: 0.5)
                    .controlSize(.small)
            }

            Text("\(filteredScreeningHits.count) / \(viewModel.docking.screeningHits.count) hits pass filters")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Screening Export

    @ViewBuilder
    private var screeningExportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: { viewModel.exportScreeningHits() }) {
                Label("Export Hits (CSV + SDF)", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.docking.screeningHits.isEmpty)
            .help("Export screening hits as CSV and SDF files")
            .accessibilityIdentifier(AccessibilityID.resultsExportHits)
        }
    }

    // MARK: - Analog Generation

    // MARK: - Batch Docking Results

    @ViewBuilder
    private var batchProgressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Batch Docking", systemImage: "arrow.triangle.merge")
                .font(.callout.weight(.semibold))

            let (current, total) = viewModel.docking.batchProgress
            ProgressView(value: total > 0 ? Double(current) / Double(total) : 0) {
                Text("\(viewModel.workspace.statusMessage)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .controlSize(.small)

            if !viewModel.docking.batchResults.isEmpty {
                Text("\(viewModel.docking.batchResults.count) ligands docked so far")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Button("Cancel") {
                viewModel.cancelBatchDocking()
            }
            .controlSize(.small)
            .buttonStyle(.bordered)
            .foregroundStyle(.red)
            .accessibilityIdentifier(AccessibilityID.resultsCancelBatch)
        }
    }

    @ViewBuilder
    private var batchResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Batch Results", systemImage: "chart.bar.xaxis")
                    .font(.callout.weight(.semibold))
                Spacer()
                Text("\(viewModel.docking.batchResults.count) ligands")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }

            // Summary: best ligand
            if let best = viewModel.docking.batchResults.first, let bestPose = best.results.first {
                HStack(spacing: 12) {
                    statBadge("Best", best.ligandName, color: .green)
                    statBadge("Energy", String(format: "%.1f", bestPose.energy), unit: "kcal/mol",
                               color: bestPose.energy < -6 ? .green : bestPose.energy < 0 ? .yellow : .red)
                    statBadge("Ligands", "\(viewModel.docking.batchResults.count)")
                }
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.cyan.opacity(0.06)))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan.opacity(0.15), lineWidth: 1))
            }

            // Ranked ligand list (top 10)
            ForEach(Array(viewModel.docking.batchResults.prefix(10).enumerated()), id: \.offset) { rank, entry in
                batchResultRow(rank: rank, ligandName: entry.ligandName, results: entry.results)
            }

            if viewModel.docking.batchResults.count > 10 {
                Text("+ \(viewModel.docking.batchResults.count - 10) more ligands")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }

            // Open Results Database for full analysis
            Button(action: { openWindow(id: "results-database") }) {
                Label("Open Results Database", systemImage: "tablecells")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private func batchResultRow(rank: Int, ligandName: String, results: [DockingResult]) -> some View {
        let bestEnergy = results.first?.energy ?? .infinity
        let poseCount = results.count

        Button(action: {
            // Load this ligand's results into the main docking results view
            viewModel.docking.dockingResults = results
            if let bestEntry = viewModel.ligandDB.entries.first(where: { $0.name == ligandName }),
               results.first != nil {
                let mol = Molecule(name: bestEntry.name, atoms: bestEntry.atoms, bonds: bestEntry.bonds, title: bestEntry.smiles, smiles: bestEntry.smiles)
                viewModel.setLigandForDocking(mol)
                viewModel.docking.dockingResults = results
                viewModel.showDockingPose(at: 0)
            }
        }) {
            HStack(spacing: 6) {
                Text("#\(rank + 1)")
                    .font(.footnote.monospaced().weight(.bold))
                    .foregroundStyle(rank == 0 ? .green : .secondary)
                    .frame(width: 24, alignment: .trailing)

                Text(ligandName)
                    .font(.footnote.weight(rank == 0 ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)

                Spacer()

                Text(String(format: "%.1f", bestEnergy))
                    .font(.footnote.monospaced().weight(.medium))
                    .foregroundStyle(energyColor(bestEnergy))

                Text("\(poseCount) poses")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .padding(.horizontal, 8)
            .background(RoundedRectangle(cornerRadius: 4).fill(Color.primary.opacity(0.03)))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Empty State

    @ViewBuilder
    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("No Results Yet")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            Text("Run docking or virtual screening to see results here.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - Helpers

    @ViewBuilder
    private func statBadge(_ label: String, _ value: String, unit: String = "", color: Color = .primary) -> some View {
        VStack(spacing: 2) {
            HStack(spacing: 2) {
                Text(value)
                    .font(.subheadline.monospaced().weight(.medium))
                    .foregroundStyle(color)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func configRow<Content: View>(_ label: String, @ViewBuilder control: () -> Content) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
            Spacer()
            control()
        }
    }

    private func energyColor(_ energy: Float) -> Color {
        if energy < -8 { return .green }
        if energy < -4 { return .yellow }
        if energy < 0 { return .orange }
        return .red
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
        guard !viewModel.docking.resultsHasInitializedCutoffs, !viewModel.docking.screeningHits.isEmpty else { return }
        viewModel.docking.resultsHasInitializedCutoffs = true
        // Set energy cutoff to median hit energy
        let energies = viewModel.docking.screeningHits.map(\.compositeScore).sorted()
        if let median = energies[safe: energies.count / 2] {
            viewModel.docking.resultsEnergyCutoff = median
        }
    }
}

// Safe array subscript
private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
