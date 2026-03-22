import SwiftUI

struct DockingTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var showPreDockSheet = false

    // Grid box state is in viewModel so it persists across tab switches

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // 1. Active ligand summary
            ligandSummarySection
            Divider()
            // 2. Pocket detection
            pocketDetectionSection
            Divider()
            // 3. Grid box (always available)
            gridBoxSection
            Divider()
            // 4. Docking parameters
            dockingConfigSection
            Divider()
            // 5. Run button
            dockingControlSection
            if viewModel.isDocking {
                Divider()
                dockingProgressSection
            }
            if !viewModel.dockingResults.isEmpty && !viewModel.isDocking {
                Divider()
                dockingResultsSection
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .sheet(isPresented: $showPreDockSheet) {
            PreDockSheet()
                .environment(viewModel)
        }
        .onChange(of: viewModel.selectedPocket?.id) { _, newID in
            if let newID, let pocket = viewModel.detectedPockets.first(where: { $0.id == newID }) {
                syncGridFromPocket(pocket)
                viewModel.showGridBoxForPocket(pocket)
            } else if newID == nil {
                viewModel.renderer?.clearGridBox()
            }
        }
        .onAppear {
            if !viewModel.gridInitialized, let prot = viewModel.protein {
                initializeGridAtProteinCenter(prot)
            }
        }
        .onChange(of: viewModel.protein?.name) { _, _ in
            // Only reset grid to protein center if no pocket or docking results exist
            if let prot = viewModel.protein, viewModel.selectedPocket == nil, viewModel.dockingResults.isEmpty {
                initializeGridAtProteinCenter(prot)
            }
        }
    }

    // MARK: - Ligand Summary

    @ViewBuilder
    private var ligandSummarySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label(viewModel.batchQueue.count > 1 ? "Ligands" : "Ligand",
                      systemImage: viewModel.batchQueue.count > 1 ? "tray.full" : "hexagon")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button(action: { openWindow(id: "ligand-database") }) {
                    Image(systemName: "tablecells")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Open Ligand Database (Cmd+L)")
            }

            if viewModel.batchQueue.count > 1 {
                // Batch mode: show general docking info
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Image(systemName: "tray.full.fill")
                            .font(.system(size: 10))
                            .foregroundStyle(.cyan)
                        Text("\(viewModel.batchQueue.count) ligands to dock")
                            .font(.system(size: 11, weight: .medium))
                        Spacer()
                        Button(action: {
                            viewModel.batchQueue = []
                            viewModel.clearLigand()
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Clear batch queue")
                    }

                    // MW range summary
                    let mws = viewModel.batchQueue.compactMap { $0.descriptors?.molecularWeight }
                    if let minMW = mws.min(), let maxMW = mws.max() {
                        Text(String(format: "MW range: %.0f – %.0f", minMW, maxMW))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }

                    let prepCount = viewModel.batchQueue.filter(\.isPrepared).count
                    Text("\(prepCount)/\(viewModel.batchQueue.count) prepared")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(prepCount == viewModel.batchQueue.count ? .green : .orange)
                }
                .padding(6)
                .background(Color.cyan.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 5))

                // Reopen Ligand Database button
                Button(action: { openWindow(id: "ligand-database") }) {
                    Label("Open Ligand Database", systemImage: "tablecells")
                        .font(.system(size: 10))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            } else if let lig = viewModel.ligand {
                // Single ligand mode
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 10))
                        .foregroundStyle(.green)
                    Text(lig.name)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                    Spacer()
                    Text("\(lig.atomCount) atoms")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)

                    Button(action: { viewModel.clearLigand() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Remove active ligand")
                }
                .padding(6)
                .background(Color.green.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 5))
            } else {
                Button(action: { openWindow(id: "ligand-database") }) {
                    HStack(spacing: 4) {
                        Image(systemName: "plus.circle")
                            .font(.system(size: 9))
                        Text("Open Database to select ligand")
                            .font(.system(size: 10))
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
        }
    }

    // MARK: - Pocket Detection

    @ViewBuilder
    private var pocketDetectionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Binding Site", systemImage: "target")
                .font(.system(size: 12, weight: .semibold))

            // Detection methods
            HStack(spacing: 4) {
                Button(action: { detectPocketsAuto() }) {
                    Label("Auto", systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(viewModel.protein == nil)
                .help("Alpha-sphere + DBSCAN pocket detection")

                Button(action: { viewModel.detectPocketsML() }) {
                    Label("ML", systemImage: "brain")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(viewModel.protein == nil || !viewModel.pocketDetectorML.isAvailable)
                .help("ML-based pocket detection (GNN)")

                Button(action: { detectFromLigand() }) {
                    Label("Ligand", systemImage: "hexagon")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(viewModel.protein == nil || viewModel.ligand == nil)
                .help("Define pocket around current ligand")
            }
            HStack(spacing: 4) {
                Button(action: { pocketFromSelection() }) {
                    Label("Selection", systemImage: "hand.tap")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(viewModel.protein == nil || viewModel.selectedResidueIndices.isEmpty)
                .help("Define pocket from selected residues")
            }

            // Detected pockets list
            if !viewModel.detectedPockets.isEmpty {
                ForEach(viewModel.detectedPockets) { pocket in
                    pocketRow(pocket)
                }
            }

            // Selected pocket badge
            if let pocket = viewModel.selectedPocket {
                selectedPocketBadge(pocket)
            }
        }
    }

    @ViewBuilder
    private func pocketRow(_ pocket: BindingPocket) -> some View {
        let isSelected = viewModel.selectedPocket?.id == pocket.id
        HStack(spacing: 6) {
            Circle()
                .fill(isSelected ? .green : .gray)
                .frame(width: 6, height: 6)

            Text("Pocket")
                .font(.system(size: 10, weight: .medium))

            Spacer()

            Text(String(format: "%.0f A\u{00B3}", pocket.volume))
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)

            Text("\(pocket.residueIndices.count) res")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(4)
        .background(isSelected ? Color.green.opacity(0.06) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.selectedPocket = pocket
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private func selectedPocketBadge(_ pocket: BindingPocket) -> some View {
        HStack(spacing: 12) {
            statBadge("Vol", String(format: "%.0f", pocket.volume), unit: "A\u{00B3}")
            statBadge("Bur", String(format: "%.0f%%", pocket.buriedness * 100))
            statBadge("Res", "\(pocket.residueIndices.count)")
        }
    }

    // MARK: - Grid Box (always available — no pocket required)

    @ViewBuilder
    private var gridBoxSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Grid Box", systemImage: "cube.transparent")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                // Show grid toggle
                Button(action: { applyGridBoxFromSliders() }) {
                    Image(systemName: "arrow.right.circle")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .help("Apply grid box to viewport")
            }

            // Center controls
            VStack(alignment: .leading, spacing: 3) {
                Text("Center")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                gridAxisRow("X", value: gridCenterBinding(\.x), range: -200...200)
                gridAxisRow("Y", value: gridCenterBinding(\.y), range: -200...200)
                gridAxisRow("Z", value: gridCenterBinding(\.z), range: -200...200)
            }

            // Half-size controls
            VStack(alignment: .leading, spacing: 3) {
                Text("Half-size")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                gridAxisRow("X", value: gridHalfBinding(\.x), range: 2...50)
                gridAxisRow("Y", value: gridHalfBinding(\.y), range: 2...50)
                gridAxisRow("Z", value: gridHalfBinding(\.z), range: 2...50)
            }

            // Quick placement buttons
            HStack(spacing: 3) {
                Button(action: { placeGridAtProteinCenter() }) {
                    Label("Protein", systemImage: "building.2")
                        .font(.system(size: 9))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(viewModel.protein == nil)
                .help("Center grid on protein centroid")

                if viewModel.ligand != nil {
                    Button(action: { placeGridAtLigand() }) {
                        Label("Ligand", systemImage: "hexagon")
                            .font(.system(size: 9))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }

                if !viewModel.selectedResidueIndices.isEmpty {
                    Button(action: { placeGridAtSelection() }) {
                        Label("Selection", systemImage: "hand.tap")
                            .font(.system(size: 9))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }

                if viewModel.selectedPocket != nil {
                    Button(action: { resetGridToPocket() }) {
                        Label("Pocket", systemImage: "arrow.counterclockwise")
                            .font(.system(size: 9))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }
        }
    }

    @ViewBuilder
    private func gridAxisRow(_ axis: String, value: Binding<Float>, range: ClosedRange<Float>) -> some View {
        HStack(spacing: 4) {
            Text(axis)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .frame(width: 12, alignment: .trailing)
                .foregroundStyle(.secondary)
            Slider(value: value, in: range, step: 0.5)
                .controlSize(.mini)
                .onChange(of: value.wrappedValue) { _, _ in
                    applyGridBoxFromSliders()
                }
            Text(String(format: "%.1f \u{00C5}", value.wrappedValue))
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .trailing)
        }
    }

    // MARK: - Docking Configuration

    @ViewBuilder
    private var dockingConfigSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Configuration", systemImage: "gearshape")
                .font(.system(size: 12, weight: .semibold))

            // Presets row
            HStack(spacing: 4) {
                Text("Preset")
                    .font(.system(size: 11))
                Spacer()
                ForEach(["Fast", "Standard", "Thorough"], id: \.self) { preset in
                    let isActive = isPresetActive(preset)
                    Button(preset) {
                        applyPreset(preset)
                    }
                    .font(.system(size: 9, weight: isActive ? .bold : .regular))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(isActive ? Color.accentColor.opacity(0.25) : Color.secondary.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .buttonStyle(.plain)
                }
            }
            .help("Fast: quick scan. Standard: balanced. Thorough: exhaustive search")

            configRow("Population") {
                intField(value: Binding(
                    get: { viewModel.dockingConfig.populationSize },
                    set: { viewModel.dockingConfig.populationSize = max(10, $0) }
                ))
            }
            .help("Number of candidate poses per generation")

            configRow("Generations") {
                intField(value: Binding(
                    get: { viewModel.dockingConfig.generationsPerRun },
                    set: { viewModel.dockingConfig.generationsPerRun = max(10, $0) }
                ))
            }
            .help("Number of GA evolution cycles per run")

            configRow("Runs") {
                intField(value: Binding(
                    get: { viewModel.dockingConfig.numRuns },
                    set: { viewModel.dockingConfig.numRuns = max(1, $0) }
                ))
            }
            .help("Independent docking runs (more = better sampling)")

            Toggle("Ligand Flexibility", isOn: Binding(
                get: { viewModel.dockingConfig.enableFlexibility },
                set: { viewModel.dockingConfig.enableFlexibility = $0 }
            ))
            .font(.system(size: 11))
            .help("Allow rotatable bonds to flex during docking")

            configRow("Grid Spacing") {
                Picker("", selection: Binding(
                    get: { viewModel.dockingConfig.gridSpacing },
                    set: { viewModel.dockingConfig.gridSpacing = $0 }
                )) {
                    Text("0.375 \u{00C5}").tag(Float(0.375))
                    Text("0.500 \u{00C5}").tag(Float(0.500))
                    Text("0.750 \u{00C5}").tag(Float(0.750))
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 80)
            }
            .help("Energy grid resolution (smaller = more accurate but slower)")

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Mutation Rate")
                        .font(.system(size: 11))
                    Spacer()
                    Text(String(format: "%.3f", viewModel.dockingConfig.mutationRate))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { viewModel.dockingConfig.mutationRate },
                        set: { viewModel.dockingConfig.mutationRate = $0 }
                    ),
                    in: 0.01...0.10,
                    step: 0.005
                )
                .controlSize(.small)
            }
        }
    }

    private func applyPreset(_ preset: String) {
        switch preset {
        case "Fast":
            viewModel.dockingConfig.populationSize = 50
            viewModel.dockingConfig.generationsPerRun = 30
            viewModel.dockingConfig.numRuns = 3
        case "Standard":
            viewModel.dockingConfig.populationSize = 150
            viewModel.dockingConfig.generationsPerRun = 80
            viewModel.dockingConfig.numRuns = 5
        case "Thorough":
            viewModel.dockingConfig.populationSize = 300
            viewModel.dockingConfig.generationsPerRun = 200
            viewModel.dockingConfig.numRuns = 10
        default: break
        }
    }

    private func isPresetActive(_ preset: String) -> Bool {
        let c = viewModel.dockingConfig
        switch preset {
        case "Fast":     return c.populationSize == 50 && c.generationsPerRun == 30 && c.numRuns == 3
        case "Standard": return c.populationSize == 150 && c.generationsPerRun == 80 && c.numRuns == 5
        case "Thorough": return c.populationSize == 300 && c.generationsPerRun == 200 && c.numRuns == 10
        default: return false
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

    @ViewBuilder
    private func intField(value: Binding<Int>) -> some View {
        TextField("", value: value, format: .number)
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 10, design: .monospaced))
            .frame(width: 70)
            .multilineTextAlignment(.trailing)
    }

    // MARK: - Docking Control

    @ViewBuilder
    private var dockingControlSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            let hasGrid = viewModel.selectedPocket != nil ||
                          (viewModel.gridHalfSize.x > 2 && viewModel.gridHalfSize.y > 2 && viewModel.gridHalfSize.z > 2)
            let canDock = hasGrid
                && viewModel.ligand != nil
                && viewModel.protein != nil
                && !viewModel.isDocking

            // ML re-ranking toggle (visible when DruseScore model available)
            if viewModel.druseScore.isAvailable {
                @Bindable var vm = viewModel
                Toggle(isOn: $vm.useDruseScoreReranking) {
                    HStack(spacing: 4) {
                        Image(systemName: "brain")
                            .font(.system(size: 10))
                        Text("ML re-ranking")
                            .font(.system(size: 11))
                    }
                }
                .toggleStyle(.checkbox)
                .controlSize(.mini)
                .help("Re-rank poses with DruseScore neural network after docking")
            }

            Button(action: {
                ensurePocketFromGrid()
                if viewModel.batchQueue.count > 1 {
                    // Batch mode: dock all queued ligands
                    viewModel.dockEntries(viewModel.batchQueue)
                    viewModel.batchQueue = []
                } else {
                    // Single ligand: show pre-dock confirmation
                    showPreDockSheet = true
                }
            }) {
                Label(viewModel.batchQueue.count > 1
                      ? "Dock \(viewModel.batchQueue.count) Ligands"
                      : "Run Docking...",
                      systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(!canDock)

            if !canDock && !viewModel.isDocking {
                VStack(alignment: .leading, spacing: 2) {
                    if viewModel.protein == nil {
                        requirementLabel("Load a protein")
                    }
                    if viewModel.ligand == nil {
                        requirementLabel("Set an active ligand")
                    }
                    if !hasGrid {
                        requirementLabel("Configure grid box or detect pocket")
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func requirementLabel(_ text: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 8))
                .foregroundStyle(.red.opacity(0.6))
            Text(text)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Docking Progress

    @ViewBuilder
    private var dockingProgressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Running", systemImage: "bolt.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.yellow)

            // Overall batch progress bar (ligand X/Y)
            if viewModel.isBatchDocking {
                let (current, total) = viewModel.batchProgress
                ProgressView(
                    value: total > 0 ? Double(current) / Double(total) : 0
                )
                .progressViewStyle(.linear)
                .tint(.cyan)

                Text("Ligand \(current + 1)/\(total)")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.cyan)
            }

            // Per-ligand generation progress
            ProgressView(
                value: min(Double(viewModel.dockingGeneration),
                           Double(max(viewModel.dockingTotalGenerations, 1))),
                total: Double(max(viewModel.dockingTotalGenerations, 1))
            )
            .progressViewStyle(.linear)

            HStack {
                Text("Gen \(viewModel.dockingGeneration)/\(viewModel.dockingTotalGenerations)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.2f kcal/mol", viewModel.dockingBestEnergy))
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(viewModel.dockingBestEnergy < 0 ? .green : .orange)
            }

            Button(action: {
                if viewModel.isBatchDocking {
                    viewModel.cancelBatchDocking()
                } else {
                    viewModel.stopDocking()
                }
            }) {
                Label(viewModel.isBatchDocking ? "Stop All" : "Stop", systemImage: "stop.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .tint(.red)
        }
    }

    // MARK: - Docking Statistics (replaces results preview)

    @ViewBuilder
    private var dockingResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Docking Statistics", systemImage: "chart.bar.xaxis")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                if viewModel.dockingDuration > 0 {
                    Text(formatDuration(viewModel.dockingDuration))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            // Key statistics grid
            let results = viewModel.dockingResults
            let energies = results.map(\.energy)
            let clusterIDs = Set(results.map(\.clusterID))

            HStack(spacing: 8) {
                statCell("Best", String(format: "%.1f", energies.min() ?? 0),
                         color: (energies.min() ?? 0) < -6 ? .green : .yellow)
                statCell("Mean", String(format: "%.1f", energies.isEmpty ? 0 : energies.reduce(0, +) / Float(energies.count)),
                         color: .secondary)
                statCell("Poses", "\(results.count)", color: .blue)
                statCell("Clusters", "\(clusterIDs.count)", color: .cyan)
            }
            .padding(6)
            .frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 5).fill(Color.green.opacity(0.06)))

            // Best pose scoring breakdown
            if let best = results.first {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Best Pose Breakdown")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        scoreItem("vdW", best.vdwEnergy)
                        scoreItem("Elec", best.elecEnergy)
                        scoreItem("H-bond", best.hbondEnergy)
                        scoreItem("Torsion", best.torsionPenalty)
                    }
                }

                // Quick action
                Button(action: { viewModel.showDockingPose(at: 0) }) {
                    Label("View Best Pose", systemImage: "eye")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // GA/search statistics
            VStack(alignment: .leading, spacing: 3) {
                Text("Search Summary")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    statMini("Generations", "\(viewModel.dockingTotalGenerations)")
                    statMini("Pop. Size", "\(viewModel.dockingConfig.populationSize)")
                    if let maxGen = results.map(\.generation).max() {
                        statMini("Last Improv.", "Gen \(maxGen)")
                    }
                }
            }

            // Open Results Database button
            Button(action: { openWindow(id: "results-database") }) {
                Label("Open Results Database", systemImage: "tablecells")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private func statCell(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func scoreItem(_ label: String, _ value: Float) -> some View {
        VStack(spacing: 0) {
            Text(String(format: "%.1f", value))
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(value < 0 ? .green : .red)
            Text(label)
                .font(.system(size: 7))
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func statMini(_ label: String, _ value: String) -> some View {
        HStack(spacing: 2) {
            Text(label + ":")
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.system(size: 8, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return "\(m)m \(s)s"
    }

    // MARK: - Helpers

    @ViewBuilder
    private func statBadge(_ label: String, _ value: String, unit: String = "") -> some View {
        VStack(spacing: 1) {
            HStack(spacing: 1) {
                Text(value)
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                if !unit.isEmpty {
                    Text(unit)
                        .font(.system(size: 7))
                        .foregroundStyle(.tertiary)
                }
            }
            Text(label)
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Actions

    private func detectPocketsAuto() {
        guard let prot = viewModel.protein else { return }
        let excluded = viewModel.hiddenChainIDs
        let chainMsg = excluded.isEmpty ? "" : " (excluding chains: \(excluded.sorted().joined(separator: ", ")))"
        viewModel.log.info("Detecting binding pockets\(chainMsg)...", category: .dock)

        Task {
            let pockets = BindingSiteDetector.detectPockets(protein: prot, excludedChainIDs: excluded)
            viewModel.detectedPockets = pockets
            if let first = pockets.first {
                viewModel.selectedPocket = first
            }
            viewModel.log.success("Found \(pockets.count) pocket\(pockets.count == 1 ? "" : "s")", category: .dock)
        }
    }

    private func detectFromLigand() {
        guard let prot = viewModel.protein, let lig = viewModel.ligand else { return }
        let excluded = viewModel.hiddenChainIDs
        viewModel.log.info("Detecting pocket from ligand position...", category: .dock)

        if let pocket = BindingSiteDetector.ligandGuidedPocket(protein: prot, ligand: lig, excludedChainIDs: excluded) {
            viewModel.detectedPockets = [pocket]
            viewModel.selectedPocket = pocket
            viewModel.log.success("Ligand-guided pocket: \(pocket.residueIndices.count) residues, \(Int(pocket.volume)) A\u{00B3}", category: .dock)
        } else {
            viewModel.log.warn("Could not define pocket from ligand", category: .dock)
        }
    }

    private func pocketFromSelection() {
        guard let prot = viewModel.protein else { return }
        let resIndices = Array(viewModel.selectedResidueIndices)
        let pocket = BindingSiteDetector.pocketFromResidues(protein: prot, residueIndices: resIndices)
        viewModel.detectedPockets = [pocket]
        viewModel.selectedPocket = pocket
        viewModel.log.success("Manual pocket from \(resIndices.count) residues", category: .dock)
    }

    // MARK: - Grid Box Actions

    private func initializeGridAtProteinCenter(_ prot: Molecule) {
        // Only use visible chains for center computation
        let positions: [SIMD3<Float>]
        if viewModel.hiddenChainIDs.isEmpty {
            positions = prot.atoms.map(\.position)
        } else {
            positions = prot.atoms.filter { !viewModel.hiddenChainIDs.contains($0.chainID) }.map(\.position)
        }
        guard !positions.isEmpty else { return }
        let center = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
        viewModel.gridCenter = center
        viewModel.gridHalfSize = SIMD3<Float>(repeating: 10)
        viewModel.gridInitialized = true
    }

    private func syncGridFromPocket(_ pocket: BindingPocket) {
        viewModel.gridCenter = pocket.center
        viewModel.gridHalfSize = pocket.size
    }

    private func applyGridBoxFromSliders() {
        viewModel.updateGridBoxVisualization(center: viewModel.gridCenter, halfSize: viewModel.gridHalfSize)

        // Keep selected pocket in sync if one exists
        if var pocket = viewModel.selectedPocket {
            pocket.center = viewModel.gridCenter
            pocket.size = viewModel.gridHalfSize
            viewModel.selectedPocket = pocket
        }
    }

    // Bindings to individual components of viewModel.gridCenter
    private func gridCenterBinding(_ keyPath: WritableKeyPath<SIMD3<Float>, Float>) -> Binding<Float> {
        Binding(
            get: { viewModel.gridCenter[keyPath: keyPath] },
            set: { viewModel.gridCenter[keyPath: keyPath] = $0 }
        )
    }

    // Bindings to individual components of viewModel.gridHalfSize
    private func gridHalfBinding(_ keyPath: WritableKeyPath<SIMD3<Float>, Float>) -> Binding<Float> {
        Binding(
            get: { viewModel.gridHalfSize[keyPath: keyPath] },
            set: { viewModel.gridHalfSize[keyPath: keyPath] = $0 }
        )
    }

    private func placeGridAtProteinCenter() {
        guard let prot = viewModel.protein else { return }
        initializeGridAtProteinCenter(prot)
        applyGridBoxFromSliders()
    }

    private func placeGridAtLigand() {
        guard let lig = viewModel.ligand else { return }
        let positions = lig.atoms.map(\.position)
        guard !positions.isEmpty else { return }
        let center = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
        var maxDist: Float = 0
        for pos in positions {
            let d = abs(pos - center)
            maxDist = max(maxDist, max(d.x, max(d.y, d.z)))
        }
        let margin: Float = 4.0
        let halfSize = SIMD3<Float>(repeating: maxDist + margin)

        viewModel.gridCenter = center
        viewModel.gridHalfSize = halfSize
        applyGridBoxFromSliders()
    }

    private func placeGridAtSelection() {
        guard let prot = viewModel.protein else { return }
        let resIndices = viewModel.selectedResidueIndices
        guard !resIndices.isEmpty else { return }

        var positions: [SIMD3<Float>] = []
        for resIdx in resIndices {
            if resIdx < prot.residues.count {
                for atomIdx in prot.residues[resIdx].atomIndices {
                    if atomIdx < prot.atoms.count {
                        positions.append(prot.atoms[atomIdx].position)
                    }
                }
            }
        }
        guard !positions.isEmpty else { return }

        let center = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
        var maxDist: Float = 0
        for pos in positions { let d = abs(pos - center); maxDist = max(maxDist, max(d.x, max(d.y, d.z))) }
        let halfSize = SIMD3<Float>(repeating: maxDist + 4.0)

        viewModel.gridCenter = center
        viewModel.gridHalfSize = halfSize
        applyGridBoxFromSliders()
    }

    private func resetGridToPocket() {
        guard let pocket = viewModel.selectedPocket else { return }
        syncGridFromPocket(pocket)
        applyGridBoxFromSliders()
    }

    /// Create a pocket from current grid sliders so the docking engine has what it needs.
    private func ensurePocketFromGrid() {
        if viewModel.selectedPocket != nil { return }
        let center = viewModel.gridCenter
        let size = viewModel.gridHalfSize
        let pocket = BindingPocket(
            id: 0, center: center, size: size,
            volume: size.x * size.y * size.z * 8,
            buriedness: 0.5, polarity: 0.5, druggability: 0.5,
            residueIndices: [], probePositions: []
        )
        viewModel.detectedPockets = [pocket]
        viewModel.selectedPocket = pocket
    }
}
