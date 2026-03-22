import SwiftUI
import AppKit

struct ResultsTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    // Screening hit filters (stored in viewModel to persist across tab switches)

    // Analog generation
    @State private var analogRefSmiles: String = ""
    @State private var analogRefName: String = ""
    @State private var analogCount: Double = 50
    @State private var analogSimilarity: Double = 0.7
    @State private var analogKeepScaffold: Bool = true
    @State private var showAnalogSection: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Results Database button always on top when results exist
            if !viewModel.dockingResults.isEmpty || !viewModel.batchResults.isEmpty {
                Button(action: { openWindow(id: "results-database") }) {
                    Label("Open Results Database", systemImage: "tablecells")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                Divider()
            }

            // Batch docking progress
            if viewModel.isBatchDocking {
                batchProgressSection
            }

            // Batch docking results (ranked across all ligands)
            if !viewModel.batchResults.isEmpty && !viewModel.isBatchDocking {
                batchResultsSection
                Divider()
            }

            if !viewModel.dockingResults.isEmpty && !viewModel.isDocking && viewModel.batchResults.isEmpty {
                dockingResultsSection
                Divider()
                dockingExportSection
            }

            // Show screening results section when available
            if !viewModel.screeningHits.isEmpty {
                if !viewModel.dockingResults.isEmpty { Divider() }
                screeningResultsSection
                Divider()
                screeningFilterSection
                Divider()
                screeningExportSection
            }

            // Analog generation section
            if showAnalogSection || !viewModel.dockingResults.isEmpty || !viewModel.screeningHits.isEmpty {
                Divider()
                analogGenerationSection
            }

            // Empty state
            if viewModel.dockingResults.isEmpty && viewModel.screeningHits.isEmpty && !viewModel.isDocking {
                emptyState
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .onChange(of: viewModel.screeningHits.count) { _, _ in
            initializeCutoffsIfNeeded()
        }
        .sheet(isPresented: Binding(
            get: { viewModel.showInteractionDiagram },
            set: { viewModel.showInteractionDiagram = $0 }
        )) {
            if let ligand = viewModel.ligand,
               let protein = viewModel.protein,
               viewModel.interactionDiagramPoseIndex < viewModel.dockingResults.count {
                let idx = viewModel.interactionDiagramPoseIndex
                let result = viewModel.dockingResults[idx]
                InteractionDiagramView(
                    interactions: viewModel.currentInteractions,
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
                if let ligandName = viewModel.ligand?.name, !ligandName.isEmpty {
                    Text(ligandName)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor.opacity(0.1)))
                }
            }

            // Summary card
            if let best = viewModel.dockingResults.first {
                summaryCard(
                    bestEnergy: best.energy,
                    poseCount: viewModel.dockingResults.count,
                    clusterCount: Set(viewModel.dockingResults.map(\.clusterID)).count
                )
            }

            // Energy landscape bar chart
            if viewModel.dockingResults.count > 1 {
                energyLandscapeChart
            }

            // Multi-pose action bar
            if viewModel.selectedPoseIndices.count > 1 {
                HStack {
                    Text("\(viewModel.selectedPoseIndices.count) poses selected")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("View Selected") {
                        viewModel.showSelectedPoses()
                    }
                    .controlSize(.small)
                    .buttonStyle(.borderedProminent)
                    Button("Clear") {
                        viewModel.selectedPoseIndices.removeAll()
                    }
                    .controlSize(.small)
                    .buttonStyle(.bordered)
                }
                .padding(.vertical, 4)
            }

            // Scrollable pose list (up to 50)
            ForEach(Array(viewModel.dockingResults.prefix(50).enumerated()), id: \.offset) { idx, result in
                dockingPoseRow(index: idx, result: result)
            }

            if viewModel.dockingResults.count > 50 {
                Text("+ \(viewModel.dockingResults.count - 50) more poses")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }

    @ViewBuilder
    private func summaryCard(bestEnergy: Float, poseCount: Int, clusterCount: Int) -> some View {
        HStack(spacing: 12) {
            statBadge("Best", String(format: "%.1f", bestEnergy), unit: "kcal/mol",
                       color: bestEnergy < -6 ? .green : bestEnergy < 0 ? .yellow : .red)
            statBadge("Poses", "\(poseCount)")
            statBadge("Clusters", "\(clusterCount)")
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
        let results = viewModel.dockingResults
        let energies = results.map(\.energy)
        let minE = energies.min() ?? 0
        let maxE = energies.max() ?? 0
        let range = max(abs(maxE - minE), 0.1)
        let selectedIdx = viewModel.selectedPoseIndices.count == 1 ? viewModel.selectedPoseIndices.first : nil

        VStack(alignment: .leading, spacing: 4) {
            // Axis labels
            HStack {
                Text("Energy Landscape")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if let sel = selectedIdx, sel < results.count {
                    Text(String(format: "Pose %d: %.2f kcal/mol", sel + 1, results[sel].energy))
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.primary)
                }
            }

            // Bar chart
            GeometryReader { geo in
                let barCount = min(results.count, 50)
                let spacing: CGFloat = 1
                let totalSpacing = spacing * CGFloat(barCount - 1)
                let barWidth = max(2, (geo.size.width - totalSpacing) / CGFloat(barCount))
                let chartHeight = geo.size.height

                HStack(alignment: .bottom, spacing: spacing) {
                    ForEach(0..<barCount, id: \.self) { idx in
                        let energy = results[idx].energy
                        let normalized = CGFloat((energy - minE) / range)
                        let barHeight = max(3, (1.0 - normalized) * (chartHeight - 4) + 4)
                        let isSelected = selectedIdx == idx

                        Rectangle()
                            .fill(barColor(energy: energy, isSelected: isSelected))
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

            // Scale
            HStack {
                Text(String(format: "%.1f", minE))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.green)
                Spacer()
                Text("kcal/mol")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                Spacer()
                Text(String(format: "%.1f", maxE))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(maxE > 0 ? .red : .yellow)
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
                Image(systemName: viewModel.selectedPoseIndices.contains(index)
                      ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 13))
                    .foregroundStyle(viewModel.selectedPoseIndices.contains(index) ? Color.blue : Color.gray.opacity(0.4))
            }
            .buttonStyle(.plain)
            .help("Select for multi-pose overlay")

            Text("#\(index + 1)")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .frame(width: 24, alignment: .trailing)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(String(format: "%.2f", result.energy))
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(energyColor(result.energy))
                    Text("kcal/mol")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
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
                viewModel.interactionDiagramPoseIndex = index
                viewModel.showInteractionDiagram = true
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

            // "Use as reference" button for analog generation
            HStack {
                Spacer()
                Button("Analogs") {
                    analogRefSmiles = hit.smiles
                    analogRefName = hit.name
                    showAnalogSection = true
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)
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

            Toggle("Lipinski compliant only", isOn: $vm.resultsLipinskiFilter)
                .font(.system(size: 11))

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Energy cutoff")
                        .font(.system(size: 11))
                    Spacer()
                    Text(String(format: "%.1f kcal/mol", vm.resultsEnergyCutoff))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Slider(value: $vm.resultsEnergyCutoff, in: -20...5, step: 0.5)
                    .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("ML score cutoff")
                        .font(.system(size: 11))
                    Spacer()
                    Text(String(format: "%.1f", vm.resultsMLScoreCutoff))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Slider(value: $vm.resultsMLScoreCutoff, in: 0...15, step: 0.5)
                    .controlSize(.small)
            }

            Text("\(filteredScreeningHits.count) / \(viewModel.screeningHits.count) hits pass filters")
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
            .disabled(viewModel.screeningHits.isEmpty)
        }
    }

    // MARK: - Analog Generation

    @ViewBuilder
    private var analogGenerationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Generate Analogs", systemImage: "arrow.triangle.branch")
                .font(.system(size: 12, weight: .semibold))

            // Reference SMILES input
            VStack(alignment: .leading, spacing: 4) {
                Text("Reference SMILES")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                TextField("SMILES...", text: $analogRefSmiles)
                    .font(.system(size: 10, design: .monospaced))
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.small)

                if !analogRefName.isEmpty {
                    Text("Ref: \(analogRefName)")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }

                // Quick-fill from docking result
                if !viewModel.dockingResults.isEmpty, let lig = viewModel.ligand {
                    Button("Use best docking pose") {
                        analogRefSmiles = lig.title  // title typically stores SMILES
                        analogRefName = lig.name
                    }
                    .controlSize(.mini)
                    .buttonStyle(.bordered)
                    .disabled(lig.title.isEmpty)
                }
            }

            // Options
            VStack(alignment: .leading, spacing: 4) {
                Text("Options")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                configRow("Analogs to generate") {
                    Picker("", selection: $analogCount) {
                        Text("10").tag(Double(10))
                        Text("25").tag(Double(25))
                        Text("50").tag(Double(50))
                        Text("100").tag(Double(100))
                        Text("500").tag(Double(500))
                        Text("1000").tag(Double(1000))
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(width: 70)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Similarity threshold")
                            .font(.system(size: 11))
                        Spacer()
                        Text(String(format: "%.2f", analogSimilarity))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $analogSimilarity, in: 0.5...0.9, step: 0.05)
                        .controlSize(.small)
                }

                Toggle("Keep scaffold (modify R-groups)", isOn: $analogKeepScaffold)
                    .font(.system(size: 11))
            }

            // Generate button
            Button(action: {
                viewModel.generateAnalogs(
                    referenceSmiles: analogRefSmiles,
                    referenceName: analogRefName.isEmpty ? "ref" : analogRefName,
                    count: Int(analogCount),
                    similarityThreshold: Float(analogSimilarity),
                    keepScaffold: analogKeepScaffold
                )
            }) {
                Label("Generate", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(analogRefSmiles.isEmpty || viewModel.isGeneratingAnalogs)

            // Progress
            if viewModel.isGeneratingAnalogs {
                ProgressView(value: Double(viewModel.analogGenerationProgress))
                    .progressViewStyle(.linear)
                HStack {
                    Text("Generating...")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.0f%%", viewModel.analogGenerationProgress * 100))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Batch Docking Results

    @ViewBuilder
    private var batchProgressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Batch Docking", systemImage: "arrow.triangle.merge")
                .font(.system(size: 12, weight: .semibold))

            let (current, total) = viewModel.batchProgress
            ProgressView(value: total > 0 ? Double(current) / Double(total) : 0) {
                Text("\(viewModel.statusMessage)")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            .controlSize(.small)

            if !viewModel.batchResults.isEmpty {
                Text("\(viewModel.batchResults.count) ligands docked so far")
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
                Text("\(viewModel.batchResults.count) ligands")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            // Summary: best ligand
            if let best = viewModel.batchResults.first, let bestPose = best.results.first {
                HStack(spacing: 12) {
                    statBadge("Best", best.ligandName, color: .green)
                    statBadge("Energy", String(format: "%.1f", bestPose.energy), unit: "kcal/mol",
                               color: bestPose.energy < -6 ? .green : bestPose.energy < 0 ? .yellow : .red)
                    statBadge("Ligands", "\(viewModel.batchResults.count)")
                }
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.cyan.opacity(0.06)))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan.opacity(0.15), lineWidth: 1))
            }

            // Ranked ligand list (top 10)
            ForEach(Array(viewModel.batchResults.prefix(10).enumerated()), id: \.offset) { rank, entry in
                batchResultRow(rank: rank, ligandName: entry.ligandName, results: entry.results)
            }

            if viewModel.batchResults.count > 10 {
                Text("+ \(viewModel.batchResults.count - 10) more ligands")
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
            viewModel.dockingResults = results
            if let bestEntry = viewModel.ligandDB.entries.first(where: { $0.name == ligandName }),
               results.first != nil {
                let mol = Molecule(name: bestEntry.name, atoms: bestEntry.atoms, bonds: bestEntry.bonds, title: bestEntry.smiles, smiles: bestEntry.smiles)
                viewModel.setLigandForDocking(mol)
                viewModel.dockingResults = results
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
        viewModel.screeningHits.filter { hit in
            if viewModel.resultsLipinskiFilter {
                guard hit.descriptors?.lipinski == true else { return false }
            }
            if viewModel.resultsEnergyCutoff < 0 {
                guard hit.compositeScore <= viewModel.resultsEnergyCutoff else { return false }
            }
            if viewModel.resultsMLScoreCutoff > 0 {
                if let ml = hit.mlScore {
                    guard ml >= viewModel.resultsMLScoreCutoff else { return false }
                }
            }
            return true
        }
    }

    private var totalScreened: Int {
        if case .complete(_, let total) = viewModel.screeningState {
            return total
        }
        return viewModel.screeningHits.count
    }

    private func initializeCutoffsIfNeeded() {
        guard !viewModel.resultsHasInitializedCutoffs, !viewModel.screeningHits.isEmpty else { return }
        viewModel.resultsHasInitializedCutoffs = true
        // Set energy cutoff to median hit energy
        let energies = viewModel.screeningHits.map(\.compositeScore).sorted()
        if let median = energies[safe: energies.count / 2] {
            viewModel.resultsEnergyCutoff = median
        }
    }
}

// Safe array subscript
private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
