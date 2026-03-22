import SwiftUI
import AppKit

// MARK: - Results Database Window

struct ResultsDatabaseWindow: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.dismiss) private var dismiss

    @State private var selectedPoseIndex: Int? = nil
    @State private var showInteractionDiagram: Bool = false
    @State private var sortField: SortField = .energy
    @State private var sortAscending: Bool = true

    // Batch mode: which ligand is being inspected
    @State private var selectedBatchLigand: String? = nil

    // Correlation analysis
    @State private var showCorrelation: Bool = false

    enum SortField: String, CaseIterable {
        case rank = "Rank"
        case energy = "Energy"
        case vdw = "VdW"
        case hbond = "H-bond"
        case cluster = "Cluster"
    }

    // Current results to display (either single docking or selected batch ligand)
    private var displayedResults: [DockingResult] {
        if let batchLigand = selectedBatchLigand,
           let entry = viewModel.batchResults.first(where: { $0.ligandName == batchLigand }) {
            return entry.results
        }
        return viewModel.dockingResults
    }

    private var sortedResults: [DockingResult] {
        let results = displayedResults
        return results.sorted { a, b in
            let cmp: Bool
            switch sortField {
            case .rank:    cmp = a.id < b.id
            case .energy:  cmp = a.energy < b.energy
            case .vdw:     cmp = a.vdwEnergy < b.vdwEnergy
            case .hbond:   cmp = a.hbondEnergy < b.hbondEnergy
            case .cluster: cmp = a.clusterID < b.clusterID
            }
            return sortAscending ? cmp : !cmp
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()

            if viewModel.dockingResults.isEmpty && viewModel.batchResults.isEmpty {
                emptyState
            } else {
                HSplitView {
                    // Left: batch ligand list (if batch results)
                    if !viewModel.batchResults.isEmpty {
                        batchLigandList
                            .frame(minWidth: 180, idealWidth: 220, maxWidth: 260)
                    }

                    // Center: pose table
                    poseTableView
                        .frame(minWidth: 500)

                    // Right: detail panel (interaction diagram + info)
                    if let idx = selectedPoseIndex, idx < displayedResults.count {
                        detailPanel(result: displayedResults[idx], index: idx)
                            .frame(minWidth: 350, idealWidth: 420, maxWidth: 550)
                    }
                }
            }

            Divider()
            statusBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .sheet(isPresented: $showInteractionDiagram) {
            if let idx = selectedPoseIndex, idx < displayedResults.count,
               let ligand = viewModel.ligand, let protein = viewModel.protein {
                let result = displayedResults[idx]
                InteractionDiagramView(
                    interactions: viewModel.currentInteractions,
                    ligandAtoms: ligand.atoms.filter { $0.element != .H },
                    ligandBonds: ligand.bonds,
                    proteinAtoms: protein.atoms.filter { $0.element != .H },
                    ligandSmiles: ligand.smiles ?? ligand.title,
                    poseEnergy: result.energy,
                    poseIndex: idx
                )
                .frame(minWidth: 700, minHeight: 600)
            }
        }
        .sheet(isPresented: $showCorrelation) {
            correlationAnalysisSheet
                .frame(minWidth: 600, minHeight: 500)
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 12) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)

            Text("Results Database")
                .font(.system(size: 13, weight: .semibold))

            if let ligandName = viewModel.ligand?.name, !ligandName.isEmpty {
                Text(ligandName)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.accentColor.opacity(0.1)))
            }

            Spacer()

            // Sort controls
            HStack(spacing: 4) {
                Text("Sort:")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                Picker("", selection: $sortField) {
                    ForEach(SortField.allCases, id: \.self) { field in
                        Text(field.rawValue).tag(field)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 90)
                .controlSize(.small)

                Button(action: { sortAscending.toggle() }) {
                    Image(systemName: sortAscending ? "arrow.up" : "arrow.down")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
            }

            Divider().frame(height: 18)

            // Correlation analysis button (when batch results with affinity data exist)
            if !viewModel.batchResults.isEmpty {
                Button(action: { showCorrelation = true }) {
                    Label("Correlation", systemImage: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // Export
            HStack(spacing: 4) {
                Button(action: { viewModel.exportDockingResultsSDF() }) {
                    Label("SDF", systemImage: "doc.text")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(action: { viewModel.exportDockingResultsCSV() }) {
                    Label("CSV", systemImage: "tablecells")
                        .font(.system(size: 10))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    // MARK: - Batch Ligand List

    private var batchLigandList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Label("Ligands", systemImage: "tray.full")
                    .font(.system(size: 11, weight: .semibold))
                Spacer()
                Text("\(viewModel.batchResults.count)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
            .padding(8)

            Divider()

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(Array(viewModel.batchResults.enumerated()), id: \.offset) { rank, entry in
                        let isSelected = selectedBatchLigand == entry.ligandName
                        let bestEnergy = entry.results.first?.energy ?? .infinity
                        let affinityData = ligandAffinityData(for: entry.ligandName)

                        HStack(spacing: 6) {
                            Text("#\(rank + 1)")
                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                .foregroundStyle(rank == 0 ? .green : .secondary)
                                .frame(width: 24, alignment: .trailing)

                            VStack(alignment: .leading, spacing: 1) {
                                Text(entry.ligandName)
                                    .font(.system(size: 10, weight: .medium))
                                    .lineLimit(1)
                                HStack(spacing: 6) {
                                    Text(String(format: "%.1f", bestEnergy))
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundStyle(energyColor(bestEnergy))
                                    Text("\(entry.results.count)p")
                                        .font(.system(size: 8, design: .monospaced))
                                        .foregroundStyle(.tertiary)
                                    if let ki = affinityData?.ki {
                                        Text(String(format: "Ki:%.1f", ki))
                                            .font(.system(size: 8, design: .monospaced))
                                            .foregroundStyle(.purple)
                                    }
                                }
                            }
                            Spacer()
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            selectedBatchLigand = entry.ligandName
                            selectedPoseIndex = nil
                            // Load this ligand's results into the main view model
                            viewModel.dockingResults = entry.results
                            if let dbEntry = viewModel.ligandDB.entries.first(where: { $0.name == entry.ligandName }) {
                                let mol = Molecule(name: dbEntry.name, atoms: dbEntry.atoms,
                                                   bonds: dbEntry.bonds, title: dbEntry.smiles, smiles: dbEntry.smiles)
                                viewModel.setLigandForDocking(mol)
                                viewModel.dockingResults = entry.results
                            }
                        }
                    }
                }
            }
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    // MARK: - Pose Table

    private var poseTableView: some View {
        VStack(spacing: 0) {
            // Multi-pose action bar
            if viewModel.selectedPoseIndices.count > 1 {
                HStack {
                    Text("\(viewModel.selectedPoseIndices.count) poses selected")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("View Selected Poses") {
                        viewModel.showSelectedPoses()
                    }
                    .controlSize(.small)
                    .buttonStyle(.borderedProminent)
                    Button("Clear Selection") {
                        viewModel.selectedPoseIndices.removeAll()
                    }
                    .controlSize(.small)
                    .buttonStyle(.bordered)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.accentColor.opacity(0.05))
            }

            // Table header
            HStack(spacing: 0) {
                tableHeader("", width: 24)  // checkbox column
                tableHeader("#", width: 40)
                tableHeader("Energy", width: 80)
                tableHeader("VdW", width: 65)
                tableHeader("Elec", width: 65)
                tableHeader("H-bond", width: 65)
                tableHeader("Torsion", width: 65)
                tableHeader("Cluster", width: 60)
                tableHeader("Gen", width: 45)
                Spacer()
                tableHeader("Actions", width: 100)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

            Divider()

            // Pose rows
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(Array(sortedResults.enumerated()), id: \.offset) { idx, result in
                        let origIdx = displayedResults.firstIndex(where: { $0.id == result.id }) ?? idx
                        poseRow(index: origIdx, displayIndex: idx, result: result)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func tableHeader(_ title: String, width: CGFloat) -> some View {
        Text(title)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .foregroundStyle(.secondary)
            .frame(width: width, alignment: .leading)
    }

    @ViewBuilder
    private func poseRow(index: Int, displayIndex: Int, result: DockingResult) -> some View {
        let isSelected = selectedPoseIndex == index
        HStack(spacing: 0) {
            // Multi-pose selection checkbox
            Button(action: { viewModel.togglePoseSelection(at: index) }) {
                Image(systemName: viewModel.selectedPoseIndices.contains(index)
                      ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 10))
                    .foregroundStyle(viewModel.selectedPoseIndices.contains(index) ? Color.blue : Color.gray.opacity(0.3))
            }
            .buttonStyle(.plain)
            .frame(width: 24)

            Text("#\(index + 1)")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .leading)

            Text(String(format: "%.2f", result.energy))
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(energyColor(result.energy))
                .frame(width: 80, alignment: .leading)

            Text(String(format: "%.1f", result.vdwEnergy))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 65, alignment: .leading)

            Text(String(format: "%.1f", result.elecEnergy))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 65, alignment: .leading)

            Text(String(format: "%.1f", result.hbondEnergy))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 65, alignment: .leading)

            Text(String(format: "%.1f", result.torsionPenalty))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 65, alignment: .leading)

            if result.clusterID >= 0 {
                Text("C\(result.clusterID)")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(.cyan)
                    .frame(width: 60, alignment: .leading)
            } else {
                Text("-")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .frame(width: 60, alignment: .leading)
            }

            Text("\(result.generation)")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.tertiary)
                .frame(width: 45, alignment: .leading)

            Spacer()

            // Actions
            HStack(spacing: 4) {
                Button("View") {
                    selectedPoseIndex = index
                    viewModel.showDockingPose(at: index)
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)

                Button {
                    selectedPoseIndex = index
                    viewModel.showDockingPose(at: index)
                    viewModel.interactionDiagramPoseIndex = index
                    showInteractionDiagram = true
                } label: {
                    Image(systemName: "circle.hexagongrid")
                        .font(.system(size: 9))
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)
                .help("2D Interaction Diagram")
            }
            .frame(width: 100, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(isSelected ? Color.accentColor.opacity(0.12) : (displayIndex % 2 == 0 ? Color.clear : Color.primary.opacity(0.02)))
        .contentShape(Rectangle())
        .onTapGesture {
            selectedPoseIndex = index
            viewModel.showDockingPose(at: index)
        }
    }

    // MARK: - Detail Panel

    @ViewBuilder
    private func detailPanel(result: DockingResult, index: Int) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Label("Pose #\(index + 1)", systemImage: "cube")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Text(String(format: "%.2f kcal/mol", result.energy))
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(energyColor(result.energy))
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    // Energy breakdown
                    energyBreakdownSection(result: result)

                    Divider()

                    // Interactions summary
                    interactionsSummary

                    Divider()

                    // Action buttons
                    VStack(spacing: 6) {
                        Button(action: {
                            viewModel.showDockingPose(at: index)
                            viewModel.interactionDiagramPoseIndex = index
                            showInteractionDiagram = true
                        }) {
                            Label("2D Interaction Diagram", systemImage: "circle.hexagongrid")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        Button(action: {
                            viewModel.showDockingPose(at: index)
                        }) {
                            Label("Show in 3D Viewport", systemImage: "cube")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
                .padding(10)
            }
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    @ViewBuilder
    private func energyBreakdownSection(result: DockingResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Energy Breakdown")
                .font(.system(size: 11, weight: .semibold))

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow {
                    Text("Total").font(.system(size: 10, weight: .medium))
                    Text(String(format: "%.2f kcal/mol", result.energy))
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(energyColor(result.energy))
                }
                GridRow {
                    Text("Steric (vdW)").font(.system(size: 10)).foregroundStyle(.secondary)
                    Text(String(format: "%.2f", result.stericEnergy))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                GridRow {
                    Text("Hydrophobic").font(.system(size: 10)).foregroundStyle(.secondary)
                    Text(String(format: "%.2f", result.hydrophobicEnergy))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                GridRow {
                    Text("H-bond").font(.system(size: 10)).foregroundStyle(.secondary)
                    Text(String(format: "%.2f", result.hbondEnergy))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                GridRow {
                    Text("Torsion penalty").font(.system(size: 10)).foregroundStyle(.secondary)
                    Text(String(format: "%.2f", result.torsionPenalty))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                if let refine = result.refinementEnergy {
                    GridRow {
                        Text("Refined (OpenMM)").font(.system(size: 10)).foregroundStyle(.purple)
                        Text(String(format: "%.2f", refine))
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.purple)
                    }
                }
            }

            // Cluster info
            if result.clusterID >= 0 {
                HStack(spacing: 4) {
                    Text("Cluster \(result.clusterID)")
                        .font(.system(size: 10, weight: .medium))
                    Text("(rank \(result.clusterRank) within cluster)")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }
                .padding(4)
                .background(Capsule().fill(Color.cyan.opacity(0.1)))
            }
        }
    }

    @ViewBuilder
    private var interactionsSummary: some View {
        let interactions = viewModel.currentInteractions
        VStack(alignment: .leading, spacing: 6) {
            Text("Interactions")
                .font(.system(size: 11, weight: .semibold))

            if interactions.isEmpty {
                Text("Select a pose to see interactions")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            } else {
                let grouped = Dictionary(grouping: interactions, by: \.type)
                ForEach(MolecularInteraction.InteractionType.allCases, id: \.rawValue) { type in
                    if let group = grouped[type] {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(interactionColor(type))
                                .frame(width: 6, height: 6)
                            Text(interactionLabel(type))
                                .font(.system(size: 10))
                            Spacer()
                            Text("\(group.count)")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Correlation Analysis Sheet

    @ViewBuilder
    private var correlationAnalysisSheet: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Docking vs. Experimental Affinity", systemImage: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Button("Done") { showCorrelation = false }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            .padding(12)

            Divider()

            let pairs = affinityDockingPairs()

            if pairs.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 36))
                        .foregroundStyle(.tertiary)
                    Text("No Affinity Data Available")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text("Import Ki, pKi, or IC50 values in the Ligand Database\nto compute rank correlation with docking scores.")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        // Statistics
                        correlationStatistics(pairs: pairs)

                        Divider()

                        // Rank comparison table
                        rankComparisonTable(pairs: pairs)

                        Divider()

                        // Cluster consensus
                        clusterConsensusSection
                    }
                    .padding(16)
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    @ViewBuilder
    private func correlationStatistics(pairs: [(ligand: String, dockingEnergy: Float, pKi: Float)]) -> some View {
        let n = pairs.count
        let dockingRanks = rankValues(pairs.map { $0.dockingEnergy })
        let affinityRanks = rankValues(pairs.map { -$0.pKi }) // higher pKi = better binder = lower rank

        // Spearman rank correlation
        let spearman = spearmanCorrelation(dockingRanks, affinityRanks)

        // Pearson correlation
        let dE = pairs.map { Double($0.dockingEnergy) }
        let pKi = pairs.map { Double($0.pKi) }
        let pearson = pearsonCorrelation(dE, pKi)

        // Kendall's tau
        let kendall = kendallTau(dockingRanks, affinityRanks)

        VStack(alignment: .leading, spacing: 8) {
            Text("Correlation Statistics (n=\(n))")
                .font(.system(size: 12, weight: .semibold))

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                GridRow {
                    Text("Spearman ρ").font(.system(size: 11))
                    Text(String(format: "%.3f", spearman))
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(abs(spearman) > 0.7 ? .green : abs(spearman) > 0.4 ? .yellow : .red)
                }
                GridRow {
                    Text("Pearson r").font(.system(size: 11))
                    Text(String(format: "%.3f", pearson))
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(abs(pearson) > 0.7 ? .green : abs(pearson) > 0.4 ? .yellow : .red)
                }
                GridRow {
                    Text("Kendall τ").font(.system(size: 11))
                    Text(String(format: "%.3f", kendall))
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(abs(kendall) > 0.6 ? .green : abs(kendall) > 0.3 ? .yellow : .red)
                }
                GridRow {
                    Text("R²").font(.system(size: 11))
                    Text(String(format: "%.3f", pearson * pearson))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.accentColor.opacity(0.05)))
        }
    }

    @ViewBuilder
    private func rankComparisonTable(pairs: [(ligand: String, dockingEnergy: Float, pKi: Float)]) -> some View {
        let sortedByDocking = pairs.sorted { $0.dockingEnergy < $1.dockingEnergy }
        let sortedByAffinity = pairs.sorted { $0.pKi > $1.pKi }

        VStack(alignment: .leading, spacing: 8) {
            Text("Rank Comparison")
                .font(.system(size: 12, weight: .semibold))

            // Header
            HStack(spacing: 0) {
                Text("Dock Rank").font(.system(size: 9, weight: .semibold)).frame(width: 70, alignment: .leading)
                Text("Ligand").font(.system(size: 9, weight: .semibold)).frame(width: 120, alignment: .leading)
                Text("Energy").font(.system(size: 9, weight: .semibold)).frame(width: 70, alignment: .leading)
                Text("pKi").font(.system(size: 9, weight: .semibold)).frame(width: 60, alignment: .leading)
                Text("Aff Rank").font(.system(size: 9, weight: .semibold)).frame(width: 70, alignment: .leading)
                Text("Δ Rank").font(.system(size: 9, weight: .semibold)).frame(width: 60, alignment: .leading)
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)

            ForEach(Array(sortedByDocking.enumerated()), id: \.offset) { dockRank, pair in
                let affRank = sortedByAffinity.firstIndex(where: { $0.ligand == pair.ligand }).map { $0 + 1 } ?? 0
                let delta = abs(dockRank + 1 - affRank)

                HStack(spacing: 0) {
                    Text("#\(dockRank + 1)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .frame(width: 70, alignment: .leading)

                    Text(pair.ligand)
                        .font(.system(size: 10))
                        .lineLimit(1)
                        .frame(width: 120, alignment: .leading)

                    Text(String(format: "%.1f", pair.dockingEnergy))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(energyColor(pair.dockingEnergy))
                        .frame(width: 70, alignment: .leading)

                    Text(String(format: "%.1f", pair.pKi))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.purple)
                        .frame(width: 60, alignment: .leading)

                    Text("#\(affRank)")
                        .font(.system(size: 10, design: .monospaced))
                        .frame(width: 70, alignment: .leading)

                    Text(delta == 0 ? "=" : "±\(delta)")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(delta == 0 ? .green : delta <= 2 ? .yellow : .red)
                        .frame(width: 60, alignment: .leading)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(dockRank % 2 == 0 ? Color.clear : Color.primary.opacity(0.02))
            }
        }
    }

    @ViewBuilder
    private var clusterConsensusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Binding Site Consensus")
                .font(.system(size: 12, weight: .semibold))

            Text("Most populated cluster across all docked ligands identifies the consensus binding site.")
                .font(.system(size: 10))
                .foregroundStyle(.secondary)

            // Count ligands per cluster (using best pose of each ligand)
            let clusterCounts = computeClusterConsensus()
            if clusterCounts.isEmpty {
                Text("No cluster data available")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(clusterCounts.prefix(5), id: \.cluster) { entry in
                    HStack(spacing: 8) {
                        Text("Cluster \(entry.cluster)")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.cyan)
                        ProgressView(value: Double(entry.count), total: Double(clusterCounts.first?.count ?? 1))
                            .progressViewStyle(.linear)
                            .tint(.cyan)
                        Text("\(entry.count) ligands")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)
            Text("No Docking Results")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
            Text("Run docking from the Docking tab to see results here.")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack {
            Text("\(displayedResults.count) poses")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)

            if let best = displayedResults.first {
                Text("•")
                    .foregroundStyle(.tertiary)
                Text(String(format: "Best: %.1f kcal/mol", best.energy))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            if !viewModel.batchResults.isEmpty {
                Text("•")
                    .foregroundStyle(.tertiary)
                Text("\(viewModel.batchResults.count) ligands in batch")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if let sel = selectedPoseIndex, sel < displayedResults.count {
                Text(String(format: "Selected: Pose #%d (%.2f kcal/mol)",
                           sel + 1, displayedResults[sel].energy))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
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

        for batch in viewModel.batchResults {
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
        for batch in viewModel.batchResults {
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

    // MARK: - Styling Helpers

    private func energyColor(_ energy: Float) -> Color {
        if energy < -8 { return .green }
        if energy < -4 { return .yellow }
        if energy < 0 { return .orange }
        return .red
    }

    private func interactionColor(_ type: MolecularInteraction.InteractionType) -> Color {
        switch type {
        case .hbond:       .blue
        case .hydrophobic: .green
        case .saltBridge:  .red
        case .piStack:     .purple
        case .piCation:    .orange
        case .halogen:     .mint
        case .metalCoord:  .yellow
        case .chPi:        .teal
        }
    }

    private func interactionLabel(_ type: MolecularInteraction.InteractionType) -> String {
        switch type {
        case .hbond:       "H-bond"
        case .hydrophobic: "Hydrophobic"
        case .saltBridge:  "Salt bridge"
        case .piStack:     "π-π stacking"
        case .piCation:    "π-cation"
        case .halogen:     "Halogen bond"
        case .metalCoord:  "Metal coord."
        case .chPi:        "CH-π"
        }
    }
}
