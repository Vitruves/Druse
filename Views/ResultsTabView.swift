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
                    .font(.system(size: 12, weight: .semibold))
                if let ligandName = viewModel.molecules.ligand?.name, !ligandName.isEmpty {
                    Text(ligandName)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor.opacity(0.1)))
                }
            }

            // Summary card
            if let best = viewModel.docking.dockingResults.first {
                let isML = viewModel.docking.scoringMethod == .druseAffinity
                let bestValue = isML ? (best.mlPKd ?? best.energy) : best.energy
                let unit = isML ? viewModel.docking.affinityDisplayUnit.unitLabel : "kcal/mol"
                let display = isML ? viewModel.docking.affinityDisplayUnit.format(best.mlPKd ?? 0) : String(format: "%.1f", best.energy)
                let color: Color = isML
                    ? ((best.mlPKd ?? 0) > 8 ? .green : (best.mlPKd ?? 0) > 5 ? .yellow : .red)
                    : (bestValue < -6 ? .green : bestValue < 0 ? .yellow : .red)
                summaryCard(
                    bestDisplay: display,
                    bestUnit: unit,
                    bestColor: color,
                    poseCount: viewModel.docking.dockingResults.count,
                    clusterCount: Set(viewModel.docking.dockingResults.map(\.clusterID)).count,
                    confidence: isML ? best.mlPoseConfidence : nil
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
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("View Selected") {
                        viewModel.showSelectedPoses()
                    }
                    .controlSize(.small)
                    .buttonStyle(.borderedProminent)
                    Button("Clear") {
                        viewModel.docking.selectedPoseIndices.removeAll()
                    }
                    .controlSize(.small)
                    .buttonStyle(.bordered)
                }
                .padding(.vertical, 4)
            }

            // Scrollable pose list (up to 50)
            ForEach(Array(viewModel.docking.dockingResults.prefix(50).enumerated()), id: \.offset) { idx, result in
                dockingPoseRow(index: idx, result: result)
            }

            if viewModel.docking.dockingResults.count > 50 {
                Text("+ \(viewModel.docking.dockingResults.count - 50) more poses")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
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
        let isML = viewModel.docking.scoringMethod == .druseAffinity
        let scores: [Float] = isML ? results.map { $0.mlDockingScore ?? $0.energy } : results.map(\.energy)
        let minE = scores.min() ?? 0
        let maxE = scores.max() ?? 0
        let range = max(abs(maxE - minE), 0.1)
        let selectedIdx = viewModel.docking.selectedPoseIndices.count == 1 ? viewModel.docking.selectedPoseIndices.first : nil
        let chartTitle = isML ? "Scoring Landscape" : "Energy Landscape"
        let unitLabel = isML ? viewModel.docking.affinityDisplayUnit.unitLabel : "kcal/mol"

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(chartTitle)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if let sel = selectedIdx, sel < results.count {
                    let val = scores[sel]
                    let display = isML ? viewModel.docking.affinityDisplayUnit.format(results[sel].mlPKd ?? val) : String(format: "%.2f", val)
                    Text("Pose \(sel + 1): \(display) \(unitLabel)")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
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
                        let score = scores[idx]
                        let normalized: CGFloat = isML
                            ? CGFloat((score - minE) / range)  // ML: higher = better, tall bars on left
                            : CGFloat((score - minE) / range)
                        let barHeight = max(3, (isML ? normalized : (1.0 - normalized)) * (chartHeight - 4) + 4)
                        let isSelected = selectedIdx == idx

                        Rectangle()
                            .fill(barColor(energy: isML ? -score : score, isSelected: isSelected))
                            .frame(width: barWidth, height: barHeight)
                            .clipShape(RoundedRectangle(cornerRadius: 1.5))
                            .overlay(
                                isSelected ?
                                    RoundedRectangle(cornerRadius: 1.5)
                                        .stroke(Color.white.opacity(0.8), lineWidth: 1.5)
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
                Text(String(format: "%.1f", isML ? maxE : minE))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.green)
                Spacer()
                Text(unitLabel)
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                Spacer()
                Text(String(format: "%.1f", isML ? minE : maxE))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(isML ? .red : (maxE > 0 ? .red : .yellow))
            }
        }
        .padding(10)
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
        HStack(spacing: 6) {
            // Multi-pose selection toggle
            Button(action: { viewModel.togglePoseSelection(at: index) }) {
                Image(systemName: viewModel.docking.selectedPoseIndices.contains(index)
                      ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 13))
                    .foregroundStyle(viewModel.docking.selectedPoseIndices.contains(index) ? Color.blue : Color.gray.opacity(0.4))
            }
            .buttonStyle(.plain)
            .help("Select for multi-pose overlay")

            Text("#\(index + 1)")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .frame(width: 24, alignment: .trailing)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    if viewModel.docking.scoringMethod == .druseAffinity, let pKd = result.mlPKd {
                        Text(viewModel.docking.affinityDisplayUnit.format(pKd))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(pKd > 8 ? Color.green : pKd > 5 ? .yellow : .red)
                        Text(viewModel.docking.affinityDisplayUnit.unitLabel)
                            .font(.system(size: 9))
                            .foregroundStyle(.tertiary)
                        if let conf = result.mlPoseConfidence {
                            Text(String(format: "%.0f%%", conf * 100))
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                .foregroundStyle(conf > 0.7 ? Color.green.opacity(0.8) : .secondary)
                        }
                    } else {
                        Text(String(format: "%.2f", result.energy))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(energyColor(result.energy))
                        Text("kcal/mol")
                            .font(.system(size: 9))
                            .foregroundStyle(.tertiary)
                    }
                }

                HStack(spacing: 6) {
                    Text(String(format: "vdW:%.1f", result.vdwEnergy))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(String(format: "elec:%.1f", result.elecEnergy))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(String(format: "hb:%.1f", result.hbondEnergy))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                    if let strain = result.strainEnergy, strain > 4.0 {
                        Text(String(format: "str:%.0f", strain))
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(strain > 10.0 ? .red : .orange)
                            .help("MMFF94 strain energy: \(String(format: "%.1f", strain)) kcal/mol")
                    }
                    if result.constraintPenalty > 0.01 {
                        Text(String(format: "cst:%.1f", result.constraintPenalty))
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(.orange)
                    }
                    if result.clusterID >= 0 {
                        Text("C\(result.clusterID)")
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.cyan.opacity(0.15)))
                            .foregroundStyle(.cyan)
                    }
                }
            }

            Spacer()

            Button {
                viewModel.showDockingPose(at: index)
                viewModel.docking.interactionDiagramPoseIndex = index
                viewModel.docking.showInteractionDiagram = true
            } label: {
                Image(systemName: "circle.hexagongrid")
                    .font(.system(size: 11))
            }
            .controlSize(.small)
            .buttonStyle(.bordered)
            .help("2D Interaction Diagram")

            Button("View") {
                viewModel.showDockingPose(at: index)
            }
            .controlSize(.small)
            .buttonStyle(.bordered)

            if let lig = viewModel.docking.originalDockingLigand ?? viewModel.molecules.ligand {
                Button {
                    viewModel.selectReferenceForOptimization(result: result, ligand: lig)
                } label: {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.system(size: 11))
                }
                .controlSize(.small)
                .buttonStyle(.bordered)
                .help("Optimize — use as reference for lead optimization")
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Docking Export

    @ViewBuilder
    private var dockingExportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Export", systemImage: "square.and.arrow.up")
                .font(.system(size: 12, weight: .semibold))

            HStack(spacing: 6) {
                Button(action: { viewModel.exportDockingResultsSDF() }) {
                    Label("Export SDF", systemImage: "doc.text")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(action: { viewModel.exportDockingResultsCSV() }) {
                    Label("Export CSV", systemImage: "tablecells")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Screening Results

    @ViewBuilder
    private var screeningResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Screening Results", systemImage: "flask")
                .font(.system(size: 12, weight: .semibold))

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
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }

    @ViewBuilder
    private func screeningHitRow(index: Int, hit: VirtualScreeningPipeline.ScreeningHit) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text("#\(index + 1)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .frame(width: 26, alignment: .trailing)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    Text(hit.name)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                    Text(hit.smiles)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f", hit.compositeScore))
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(energyColor(hit.compositeScore))
                    if let ml = hit.mlScore {
                        Text(String(format: "ML:%.1f", ml))
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(.purple)
                    }
                }
            }

            // Descriptor badges
            if let desc = hit.descriptors {
                HStack(spacing: 8) {
                    Text(String(format: "MW:%.0f", desc.molecularWeight))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(String(format: "cLogP:%.1f", desc.logP))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                    if desc.lipinski {
                        Text("Lipinski")
                            .font(.system(size: 9, weight: .medium))
                            .padding(.horizontal, 5)
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
                .font(.system(size: 12, weight: .semibold))

            Toggle("Lipinski compliant only", isOn: $vm.docking.resultsLipinskiFilter)
                .font(.system(size: 11))

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Energy cutoff")
                        .font(.system(size: 11))
                    Spacer()
                    Text(String(format: "%.1f kcal/mol", vm.docking.resultsEnergyCutoff))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Slider(value: $vm.docking.resultsEnergyCutoff, in: -20...5, step: 0.5)
                    .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("ML score cutoff")
                        .font(.system(size: 11))
                    Spacer()
                    Text(String(format: "%.1f", vm.docking.resultsMLScoreCutoff))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Slider(value: $vm.docking.resultsMLScoreCutoff, in: 0...15, step: 0.5)
                    .controlSize(.small)
            }

            Text("\(filteredScreeningHits.count) / \(viewModel.docking.screeningHits.count) hits pass filters")
                .font(.system(size: 9))
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
        }
    }

    // MARK: - Analog Generation

    // MARK: - Batch Docking Results

    @ViewBuilder
    private var batchProgressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Batch Docking", systemImage: "arrow.triangle.merge")
                .font(.system(size: 12, weight: .semibold))

            let (current, total) = viewModel.docking.batchProgress
            ProgressView(value: total > 0 ? Double(current) / Double(total) : 0) {
                Text("\(viewModel.workspace.statusMessage)")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            .controlSize(.small)

            if !viewModel.docking.batchResults.isEmpty {
                Text("\(viewModel.docking.batchResults.count) ligands docked so far")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }

            Button("Cancel") {
                viewModel.cancelBatchDocking()
            }
            .controlSize(.small)
            .buttonStyle(.bordered)
            .foregroundStyle(.red)
        }
    }

    @ViewBuilder
    private var batchResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Batch Results", systemImage: "chart.bar.xaxis")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Text("\(viewModel.docking.batchResults.count) ligands")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.tertiary)
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
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
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
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(rank == 0 ? .green : .secondary)
                    .frame(width: 24, alignment: .trailing)

                Text(ligandName)
                    .font(.system(size: 10, weight: rank == 0 ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)

                Spacer()

                Text(String(format: "%.1f", bestEnergy))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(energyColor(bestEnergy))

                Text("\(poseCount) poses")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 3)
            .padding(.horizontal, 6)
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
                .foregroundStyle(.tertiary)
            Text("No Results Yet")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
            Text("Run docking or virtual screening to see results here.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
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
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(color)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func configRow<Content: View>(_ label: String, @ViewBuilder control: () -> Content) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11))
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
