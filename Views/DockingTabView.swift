import SwiftUI

struct DockingTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var showPreDockSheet = false

    /// Available scoring methods: Vina, Drusina, DruseAF (Metal), and PIGNet2 if weights present.
    private var scoringMethodsAvailable: [ScoringMethod] {
        var methods: [ScoringMethod] = [.vina, .drusina, .druseAffinity]
        if Bundle.main.url(forResource: "PIGNet2", withExtension: "weights") != nil {
            methods.append(.pignet2)
        }
        return methods
    }

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
            // 3b. Pocket view / Z-slab controls
            pocketViewSection
            Divider()
            // 4. Docking parameters
            dockingConfigSection
            // 4b. Pharmacophore constraints
            Divider()
            constraintSummarySection
            // 4c. Flexible residues (induced fit)
            if viewModel.docking.selectedPocket != nil {
                Divider()
                flexibleResidueSection
            }
            Divider()
            // 5. Run button
            dockingControlSection
            if viewModel.docking.isDocking {
                Divider()
                dockingProgressSection
            }
            if !viewModel.docking.dockingResults.isEmpty && !viewModel.docking.isDocking {
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
        .onChange(of: viewModel.docking.selectedPocket?.id) { _, newID in
            if let newID, let pocket = viewModel.docking.detectedPockets.first(where: { $0.id == newID }) {
                syncGridFromPocket(pocket)
                viewModel.showGridBoxForPocket(pocket)
            } else if newID == nil {
                viewModel.renderer?.clearGridBox()
            }
        }
        .onAppear {
            if !viewModel.docking.gridInitialized, let prot = viewModel.molecules.protein {
                initializeGridAtProteinCenter(prot)
            }
        }
        .onChange(of: viewModel.molecules.protein?.name) { _, _ in
            // Only reset grid to protein center if no pocket or docking results exist
            if let prot = viewModel.molecules.protein, viewModel.docking.selectedPocket == nil, viewModel.docking.dockingResults.isEmpty {
                initializeGridAtProteinCenter(prot)
            }
        }
    }

    // MARK: - Ligand Summary

    @ViewBuilder
    private var ligandSummarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(viewModel.docking.batchQueue.count > 1 ? "Ligands" : "Ligand",
                      systemImage: viewModel.docking.batchQueue.count > 1 ? "tray.full" : "hexagon")
                    .font(.callout.weight(.semibold))
                Spacer()
                Button(action: { openWindow(id: "ligand-database") }) {
                    Image(systemName: "tablecells")
                        .font(.callout)
                        .padding(4)
                        .background(Color.accentColor.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Open Ligand Database (Cmd+L)")
                .plainButtonAccessibility(AccessibilityID.dockOpenLigandDB)
            }

            if viewModel.docking.batchQueue.count > 1 {
                // Batch mode: show general docking info
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Image(systemName: "tray.full.fill")
                            .font(.footnote)
                            .foregroundStyle(.cyan)
                        Text("\(viewModel.docking.batchQueue.count) ligands to dock")
                            .font(.subheadline.weight(.medium))
                        Spacer()
                        Button(action: {
                            viewModel.docking.batchQueue = []
                            viewModel.clearLigand()
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.body)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Clear batch queue")
                        .plainButtonAccessibility(AccessibilityID.dockClearBatch)
                    }

                    // MW range summary
                    let mws = viewModel.docking.batchQueue.compactMap { $0.descriptors?.molecularWeight }
                    if let minMW = mws.min(), let maxMW = mws.max() {
                        Text(String(format: "MW range: %.0f \u{2013} %.0f", minMW, maxMW))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }

                    let prepCount = viewModel.docking.batchQueue.filter(\.isPrepared).count
                    Text("\(prepCount)/\(viewModel.docking.batchQueue.count) prepared")
                        .font(.footnote.monospaced())
                        .foregroundStyle(prepCount == viewModel.docking.batchQueue.count ? .green : .orange)
                }
                .padding(8)
                .background(Color.cyan.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))

                // Reopen Ligand Database button
                Button(action: { openWindow(id: "ligand-database") }) {
                    Label("Open Ligand Database", systemImage: "tablecells")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else if let lig = viewModel.molecules.ligand {
                // Single ligand mode
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.footnote)
                        .foregroundStyle(.green)
                    Text(lig.name)
                        .font(.subheadline.weight(.medium))
                        .lineLimit(1)
                    Spacer()
                    Text("\(lig.atomCount) atoms")
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)

                    Button(action: { viewModel.removeLigandFromView() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Remove active ligand")
                    .plainButtonAccessibility(AccessibilityID.dockRemoveLigand)
                }
                .padding(8)
                .background(Color.green.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                Button(action: { openWindow(id: "ligand-database") }) {
                    Label("Open Database to select ligand", systemImage: "plus.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Pocket Detection

    @ViewBuilder
    private var pocketDetectionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Binding Site", systemImage: "target")
                .font(.callout.weight(.semibold))

            // Detection methods
            HStack(spacing: 4) {
                Button(action: { detectPocketsAuto() }) {
                    Label("Auto", systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.molecules.protein == nil)
                .help("Hybrid pocket detection: ML candidates plus geometric fallback")
                .accessibilityIdentifier(AccessibilityID.dockDetectAuto)

                Button(action: { viewModel.detectPocketsML() }) {
                    Label("ML", systemImage: "brain")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.molecules.protein == nil || !viewModel.pocketDetectorML.isAvailable)
                .help("ML-based pocket detection (GNN)")
                .accessibilityIdentifier(AccessibilityID.dockDetectML)

                Button(action: { detectFromLigand() }) {
                    Label("Ligand", systemImage: "hexagon")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.molecules.protein == nil || viewModel.molecules.ligand == nil)
                .help("Define pocket around current ligand")
                .accessibilityIdentifier(AccessibilityID.dockDetectLigand)
            }
            HStack(spacing: 4) {
                Button(action: { pocketFromSelection() }) {
                    Label("Selection", systemImage: "hand.tap")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.molecules.protein == nil || viewModel.workspace.selectedResidueIndices.isEmpty)
                .help("Define pocket from selected residues")
                .accessibilityIdentifier(AccessibilityID.dockDetectSelection)
            }

            // Detected pockets list
            if !viewModel.docking.detectedPockets.isEmpty {
                ForEach(viewModel.docking.detectedPockets) { pocket in
                    pocketRow(pocket)
                }
            }

            // Selected pocket badge
            if let pocket = viewModel.docking.selectedPocket {
                selectedPocketBadge(pocket)
            }
        }
    }

    @ViewBuilder
    private func pocketRow(_ pocket: BindingPocket) -> some View {
        let isSelected = viewModel.docking.selectedPocket?.id == pocket.id
        HStack(spacing: 8) {
            Circle()
                .fill(isSelected ? .green : .gray)
                .frame(width: 6, height: 6)

            Text("Pocket")
                .font(.footnote.weight(.medium))

            Spacer()

            Text(String(format: "%.0f \u{00C5}\u{00B3}", pocket.volume))
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)

            Text("\(pocket.residueIndices.count) res")
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
        }
        .padding(4)
        .background(isSelected ? Color.green.opacity(0.06) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.docking.selectedPocket = pocket
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private func selectedPocketBadge(_ pocket: BindingPocket) -> some View {
        HStack(spacing: 12) {
            statBadge("Vol", String(format: "%.0f", pocket.volume), unit: "A\u{00B3}")
            statBadge("Bur", String(format: "%.0f%%", pocket.buriedness * 100))
            statBadge("Res", "\(pocket.residueIndices.count)")

            Button(action: { viewModel.focusOnPocket(pocket) }) {
                Label("Focus", systemImage: "scope")
                    .font(.footnote.weight(.medium))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Zoom to pocket with Z-clipping slab")
            .accessibilityIdentifier(AccessibilityID.dockFocusPocket)
        }
    }

    // MARK: - Grid Box (always available — no pocket required)

    @ViewBuilder
    private var gridBoxSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Grid Box", systemImage: "cube.transparent")
                    .font(.callout.weight(.semibold))
                Spacer()
                // Show grid toggle
                Button(action: { applyGridBoxFromSliders() }) {
                    Image(systemName: "arrow.right.circle")
                        .font(.callout)
                        .padding(3)
                        .background(Color.accentColor.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .help("Apply grid box to viewport")
                .plainButtonAccessibility(AccessibilityID.dockApplyGrid)
            }

            // Center controls
            VStack(alignment: .leading, spacing: 3) {
                Text("Center")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                gridAxisRow("X", value: gridCenterBinding(\.x), range: -200...200)
                gridAxisRow("Y", value: gridCenterBinding(\.y), range: -200...200)
                gridAxisRow("Z", value: gridCenterBinding(\.z), range: -200...200)
            }

            // Half-size controls
            VStack(alignment: .leading, spacing: 3) {
                Text("Half-size")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                gridAxisRow("X", value: gridHalfBinding(\.x), range: 2...50)
                gridAxisRow("Y", value: gridHalfBinding(\.y), range: 2...50)
                gridAxisRow("Z", value: gridHalfBinding(\.z), range: 2...50)
            }

            // Quick placement buttons
            Text("Center on")
                .font(.footnote.weight(.medium))
                .foregroundStyle(.secondary)

            HStack(spacing: 4) {
                Button(action: { placeGridAtProteinCenter() }) {
                    Label("Protein", systemImage: "building.2")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.molecules.protein == nil)
                .help("Center grid on protein centroid")
                .accessibilityIdentifier(AccessibilityID.dockGridProtein)

                if viewModel.molecules.ligand != nil {
                    Button(action: { placeGridAtLigand() }) {
                        Label("Ligand", systemImage: "hexagon")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Center grid on ligand position")
                    .accessibilityIdentifier(AccessibilityID.dockGridLigand)
                }

                if !viewModel.workspace.selectedResidueIndices.isEmpty {
                    Button(action: { placeGridAtSelection() }) {
                        Label("Selection", systemImage: "hand.tap")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Center grid on selected residues")
                    .accessibilityIdentifier(AccessibilityID.dockGridSelection)
                }

                if viewModel.docking.selectedPocket != nil {
                    Button(action: { resetGridToPocket() }) {
                        Label("Pocket", systemImage: "arrow.counterclockwise")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Reset grid to detected pocket")
                    .accessibilityIdentifier(AccessibilityID.dockGridPocket)
                }
            }
        }
    }

    @ViewBuilder
    private func gridAxisRow(_ axis: String, value: Binding<Float>, range: ClosedRange<Float>) -> some View {
        HStack(spacing: 4) {
            Text(axis)
                .font(.footnote.bold().monospaced())
                .frame(width: 12, alignment: .trailing)
                .foregroundStyle(.secondary)
            Slider(value: value, in: range, step: 0.5)
                .controlSize(.mini)
                .onChange(of: value.wrappedValue) { _, _ in
                    applyGridBoxFromSliders()
                }
            Text(String(format: "%.1f \u{00C5}", value.wrappedValue))
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .trailing)
        }
    }

    // MARK: - Pocket View / Z-Slab

    @ViewBuilder
    private var pocketViewSection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Pocket View", systemImage: "rectangle.split.3x1")
                    .font(.callout.weight(.semibold))
                Spacer()
                Toggle("", isOn: Binding(
                    get: { vm.workspace.enableClipping },
                    set: { newValue in
                        vm.workspace.enableClipping = newValue
                        vm.renderer?.enableClipping = newValue
                        if newValue, let pocket = vm.docking.selectedPocket {
                            // Zoom camera to pocket and set slab clipping
                            viewModel.focusOnPocket(pocket)
                            // Auto-enable molecular surface for cavity visualization
                            if !vm.workspace.showSurface {
                                viewModel.toggleSurface()
                            }
                            // Set semi-transparent surface so ligand is visible inside
                            if vm.workspace.surfaceOpacity > 0.7 {
                                vm.workspace.surfaceOpacity = 0.55
                                vm.renderer?.surfaceOpacity = 0.55
                            }
                        } else if !newValue {
                            // Disable: remove surface if we auto-enabled it, restore full view
                            if vm.workspace.showSurface {
                                viewModel.toggleSurface()
                            }
                            viewModel.fitToView()
                        }
                    }
                ))
                .toggleStyle(.switch)
                .controlSize(.mini)
            }

            if vm.workspace.enableClipping {
                VStack(alignment: .leading, spacing: 4) {
                    // Slab thickness slider
                    HStack(spacing: 4) {
                        Text("Thickness")
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(.secondary)
                            .frame(width: 52, alignment: .leading)
                        Slider(
                            value: Binding(
                                get: { vm.workspace.slabThickness },
                                set: { newVal in
                                    vm.workspace.slabThickness = newVal
                                    vm.renderer?.slabHalfThickness = newVal / 2.0
                                }
                            ),
                            in: 2...40,
                            step: 0.5
                        )
                        .controlSize(.mini)
                        Text(String(format: "%.1f \u{00C5}", vm.workspace.slabThickness))
                            .font(.footnote.monospaced())
                            .foregroundStyle(vm.workspace.slabThickness < 10 ? .orange : .secondary)
                            .frame(width: 44, alignment: .trailing)
                    }

                    // Slab offset slider
                    HStack(spacing: 4) {
                        Text("Offset")
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(.secondary)
                            .frame(width: 52, alignment: .leading)
                        Slider(
                            value: Binding(
                                get: { vm.workspace.slabOffset },
                                set: { newVal in
                                    vm.workspace.slabOffset = newVal
                                    vm.renderer?.slabOffset = newVal
                                }
                            ),
                            in: -20...20,
                            step: 0.5
                        )
                        .controlSize(.mini)
                        Text(String(format: "%+.1f \u{00C5}", vm.workspace.slabOffset))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                            .frame(width: 44, alignment: .trailing)
                    }

                    // Quick presets
                    HStack(spacing: 4) {
                        ForEach(["Tight", "Medium", "Wide"], id: \.self) { preset in
                            Button(preset) {
                                applySlabPreset(preset)
                            }
                            .font(.footnote)
                            .controlSize(.mini)
                            .buttonStyle(.bordered)
                            .help("Z-slab \(preset.lowercased()) clipping around the pocket")
                            .accessibilityIdentifier(
                                preset == "Tight" ? AccessibilityID.dockSlabTight :
                                preset == "Medium" ? AccessibilityID.dockSlabMedium :
                                AccessibilityID.dockSlabWide
                            )
                        }
                    }
                }
            } else {
                Text("Enable to clip the view around the pocket for clean binding site visualization")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func applySlabPreset(_ preset: String) {
        guard let pocket = viewModel.docking.selectedPocket else { return }
        let pocketRadius = max(pocket.size.x, max(pocket.size.y, pocket.size.z))

        let thickness: Float
        switch preset {
        case "Tight": thickness = pocketRadius * 1.5
        case "Medium": thickness = pocketRadius * 2.5
        default: thickness = pocketRadius * 4.0 // Wide
        }

        viewModel.workspace.slabThickness = thickness
        viewModel.workspace.slabOffset = 0
        viewModel.renderer?.slabCenter = pocket.center
        viewModel.renderer?.slabHalfThickness = thickness / 2.0
        viewModel.renderer?.slabOffset = 0
        // Re-focus camera on pocket for the chosen slab width
        viewModel.focusOnPocket(pocket)
    }

    // MARK: - Docking Configuration

    @ViewBuilder
    private var dockingConfigSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Configuration", systemImage: "gearshape")
                .font(.callout.weight(.semibold))

            // Presets row
            HStack(spacing: 4) {
                Text("Preset")
                    .font(.subheadline)
                Spacer()
                ForEach(["Auto", "Fast", "Standard", "Thorough"], id: \.self) { preset in
                    let isActive = isPresetActive(preset)
                    Button(preset) {
                        applyPreset(preset)
                    }
                    .font(.footnote.weight(isActive ? .bold : .regular))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(isActive ? Color.accentColor.opacity(0.25) : Color.secondary.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .buttonStyle(.plain)
                }
            }
            .help("Auto: adapts to protein/ligand complexity. Fast/Standard/Thorough: fixed presets")

            if viewModel.docking.dockingConfig.autoMode {
                HStack(spacing: 4) {
                    Image(systemName: "wand.and.stars")
                        .font(.footnote)
                        .foregroundStyle(.purple)
                    Text("Parameters adapt to protein size, pocket shape, and ligand flexibility")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
            }

            configRow("Population") {
                intField(value: Binding(
                    get: { viewModel.docking.dockingConfig.populationSize },
                    set: { viewModel.docking.dockingConfig.populationSize = max(10, $0) }
                ))
            }
            .help("Number of candidate poses per generation")

            configRow("Generations") {
                intField(value: Binding(
                    get: { viewModel.docking.dockingConfig.generationsPerRun },
                    set: { viewModel.docking.dockingConfig.generationsPerRun = max(10, $0) }
                ))
            }
            .help("Number of GA evolution cycles per run")

            configRow("Runs") {
                intField(value: Binding(
                    get: { viewModel.docking.dockingConfig.numRuns },
                    set: { viewModel.docking.dockingConfig.numRuns = max(1, $0) }
                ))
            }
            .help("Independent docking runs (more = better sampling)")

            Toggle("Ligand Flexibility", isOn: Binding(
                get: { viewModel.docking.dockingConfig.enableFlexibility },
                set: { viewModel.docking.dockingConfig.enableFlexibility = $0 }
            ))
            .font(.subheadline)
            .help("Allow rotatable bonds to flex during docking")

            configRow("Grid Spacing") {
                Picker("", selection: Binding(
                    get: { viewModel.docking.dockingConfig.gridSpacing },
                    set: { viewModel.docking.dockingConfig.gridSpacing = $0 }
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
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.3f", viewModel.docking.dockingConfig.mutationRate))
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { viewModel.docking.dockingConfig.mutationRate },
                        set: { viewModel.docking.dockingConfig.mutationRate = $0 }
                    ),
                    in: 0.01...0.25,
                    step: 0.005
                )
                .controlSize(.small)
            }

            DisclosureGroup("Exploration") {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Exploration Phase Ratio")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", viewModel.docking.dockingConfig.explorationPhaseRatio))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: Binding(
                            get: { viewModel.docking.dockingConfig.explorationPhaseRatio },
                            set: { viewModel.docking.dockingConfig.explorationPhaseRatio = $0 }
                        ),
                        in: 0.2...0.8,
                        step: 0.05
                    )
                    .controlSize(.small)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Local Search Freq.")
                            .font(.subheadline)
                        Spacer()
                        Text("every \(viewModel.docking.dockingConfig.explorationLocalSearchFrequency)th gen")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Stepper(
                        value: Binding(
                            get: { viewModel.docking.dockingConfig.explorationLocalSearchFrequency },
                            set: { viewModel.docking.dockingConfig.explorationLocalSearchFrequency = $0 }
                        ),
                        in: 1...10
                    ) {
                        EmptyView()
                    }
                    .controlSize(.small)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("MC Temperature")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.1f", viewModel.docking.dockingConfig.mcTemperature))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: Binding(
                            get: { viewModel.docking.dockingConfig.mcTemperature },
                            set: { viewModel.docking.dockingConfig.mcTemperature = $0 }
                        ),
                        in: 0.5...4.0,
                        step: 0.1
                    )
                    .controlSize(.small)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Exploration Mutation")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", viewModel.docking.dockingConfig.explorationMutationRate))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: Binding(
                            get: { viewModel.docking.dockingConfig.explorationMutationRate },
                            set: { viewModel.docking.dockingConfig.explorationMutationRate = $0 }
                        ),
                        in: 0.10...0.50,
                        step: 0.01
                    )
                    .controlSize(.small)
                }
            }
        }
    }

    private func applyPreset(_ preset: String) {
        viewModel.docking.dockingConfig.autoMode = false
        switch preset {
        case "Auto":
            viewModel.docking.dockingConfig.autoMode = true
            // Display values are placeholders — actual values computed at docking launch
            viewModel.docking.dockingConfig.populationSize = 200
            viewModel.docking.dockingConfig.generationsPerRun = 150
            viewModel.docking.dockingConfig.numRuns = 3
        case "Fast":
            viewModel.docking.dockingConfig.populationSize = 50
            viewModel.docking.dockingConfig.generationsPerRun = 30
            viewModel.docking.dockingConfig.numRuns = 3
        case "Standard":
            viewModel.docking.dockingConfig.populationSize = 150
            viewModel.docking.dockingConfig.generationsPerRun = 80
            viewModel.docking.dockingConfig.numRuns = 5
        case "Thorough":
            viewModel.docking.dockingConfig.populationSize = 300
            viewModel.docking.dockingConfig.generationsPerRun = 200
            viewModel.docking.dockingConfig.numRuns = 10
        default: break
        }
    }

    private func isPresetActive(_ preset: String) -> Bool {
        let c = viewModel.docking.dockingConfig
        switch preset {
        case "Auto":     return c.autoMode
        case "Fast":     return !c.autoMode && c.populationSize == 50 && c.generationsPerRun == 30 && c.numRuns == 3
        case "Standard": return !c.autoMode && c.populationSize == 150 && c.generationsPerRun == 80 && c.numRuns == 5
        case "Thorough": return !c.autoMode && c.populationSize == 300 && c.generationsPerRun == 200 && c.numRuns == 10
        default: return false
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

    @ViewBuilder
    private func intField(value: Binding<Int>) -> some View {
        TextField("", value: value, format: .number)
            .textFieldStyle(.roundedBorder)
            .font(.footnote.monospaced())
            .frame(width: 70)
            .multilineTextAlignment(.trailing)
    }

    // MARK: - Pharmacophore Constraints

    // MARK: - Flexible Residue Section

    @ViewBuilder
    private var flexibleResidueSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Receptor Flexibility", systemImage: "figure.flexibility")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if !viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.isEmpty
                    && !viewModel.docking.flexibleResidueConfig.autoFlex {
                    Button("Clear") {
                        viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll()
                    }
                    .font(.footnote)
                    .buttonStyle(.plain)
                    .foregroundStyle(.red)
                    .help("Remove all flexible residues (use rigid receptor)")
                }
            }

            // Auto-flex toggle
            @Bindable var flexVM = viewModel
            Toggle(isOn: $flexVM.docking.flexibleResidueConfig.autoFlex) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Soft Receptor Flexibility")
                        .font(.footnote.weight(.medium))
                    Text("Pocket-lining sidechains adjust gently during docking")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.switch)
            .controlSize(.mini)
            .onChange(of: viewModel.docking.flexibleResidueConfig.autoFlex) { _, isAuto in
                if isAuto {
                    viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll()
                }
            }

            if viewModel.docking.flexibleResidueConfig.autoFlex {
                // Show preview of which residues would be auto-selected
                if let prot = viewModel.molecules.protein,
                   let pocket = viewModel.docking.selectedPocket {
                    let autoIndices = FlexibleResidueConfig.autoSelectResidues(
                        protein: prot.atoms,
                        pocket: (center: pocket.center, residueIndices: pocket.residueIndices)
                    )
                    if autoIndices.isEmpty {
                        Text("No rotatable residues lining this pocket")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    } else {
                        let names = autoIndices.compactMap { seq -> String? in
                            prot.atoms.first(where: { $0.residueSeq == seq }).map { "\($0.residueName)\(seq)" }
                        }
                        HStack(spacing: 4) {
                            ForEach(names, id: \.self) { name in
                                Text(name)
                                    .font(.footnote.weight(.medium).monospaced())
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(Color.purple.opacity(0.1))
                                    .foregroundStyle(.purple.opacity(0.8))
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }

                        let totalChi = autoIndices.compactMap { seq -> Int? in
                            prot.atoms.first(where: { $0.residueSeq == seq }).flatMap {
                                RotamerLibrary.rotamers(for: $0.residueName)?.chiAngles.count
                            }
                        }.reduce(0, +)

                        Text("\(autoIndices.count) residue(s), \(totalChi) chi angle(s) — soft weight, small perturbations")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                // Manual residue selection (existing behavior)
                if viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.isEmpty {
                    Text("Or select residues manually in the sequence panel or 3D viewport for full induced-fit docking.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    let flexIndices = viewModel.docking.flexibleResidueConfig.flexibleResidueIndices
                    let residueNames = flexIndices.compactMap { seq -> String? in
                        guard let prot = viewModel.molecules.protein else { return nil }
                        if let atom = prot.atoms.first(where: { $0.residueSeq == seq }) {
                            return "\(atom.residueName)\(seq)"
                        }
                        return "?\(seq)"
                    }

                    HStack(spacing: 4) {
                        ForEach(residueNames, id: \.self) { name in
                            Text(name)
                                .font(.footnote.weight(.medium).monospaced())
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.purple.opacity(0.15))
                                .foregroundStyle(.purple)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }

                    let totalChi = flexIndices.compactMap { seq -> Int? in
                        guard let prot = viewModel.molecules.protein,
                              let atom = prot.atoms.first(where: { $0.residueSeq == seq }) else { return nil }
                        return RotamerLibrary.rotamers(for: atom.residueName)?.chiAngles.count
                    }.reduce(0, +)

                    Text("\(flexIndices.count) residue(s), \(totalChi) chi angle(s)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.count >= FlexibleResidueConfig.maxFlexibleResidues {
                    Text("Maximum \(FlexibleResidueConfig.maxFlexibleResidues) flexible residues reached")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    @ViewBuilder
    private var constraintSummarySection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Constraints (\(viewModel.docking.pharmacophoreConstraints.count))",
                      systemImage: "scope")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if !viewModel.docking.pharmacophoreConstraints.isEmpty {
                    Button("Clear All") {
                        viewModel.docking.pharmacophoreConstraints.removeAll()
                    }
                    .font(.footnote)
                    .buttonStyle(.plain)
                    .foregroundStyle(.red)
                    .help("Remove all pharmacophore constraints")
                }
            }

            // Pharmacophore editor button
            Button {
                viewModel.docking.showPharmacophoreEditor = true
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "circle.hexagongrid.circle")
                        .font(.footnote)
                    Text("Pharmacophore Editor...")
                        .font(.footnote)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 3)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Create constraints from a reference ligand's pharmacophoric features")
            .sheet(isPresented: $vm.docking.showPharmacophoreEditor) {
                PharmacophoreEditorView()
                    .environment(viewModel)
            }

            ForEach(Array(viewModel.docking.pharmacophoreConstraints.enumerated()), id: \.element.id) { idx, constraint in
                HStack(spacing: 8) {
                    Image(systemName: constraint.interactionType.icon)
                        .font(.footnote)
                        .frame(width: 14)
                        .foregroundColor(Color(
                            red: Double(constraint.interactionType.color.x),
                            green: Double(constraint.interactionType.color.y),
                            blue: Double(constraint.interactionType.color.z)
                        ))

                    Text(constraint.targetLabel)
                        .font(.footnote.monospaced())
                        .lineLimit(1)

                    Spacer()

                    Text(constraint.interactionType.rawValue)
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Text(constraint.strength.isHard ? "Hard" : "Soft")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(constraint.strength.isHard ? Color.red.opacity(0.2) : Color.orange.opacity(0.2))
                        .clipShape(RoundedRectangle(cornerRadius: 4))

                    Button(action: {
                        viewModel.docking.pharmacophoreConstraints.remove(at: idx)
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Remove this constraint")
                }
            }
        }
    }

    // MARK: - Search Method Options

    @ViewBuilder
    private var searchMethodOptionsSection: some View {
        @Bindable var vm = viewModel
        let method = viewModel.docking.dockingConfig.searchMethod

        if method == .fragmentBased {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("Beam Width")
                        .font(.footnote)
                    Spacer()
                    Text("\(viewModel.docking.dockingConfig.fragment.beamWidth)")
                        .font(.footnote.monospaced())
                }
                Slider(
                    value: Binding(
                        get: { Float(viewModel.docking.dockingConfig.fragment.beamWidth) },
                        set: { viewModel.docking.dockingConfig.fragment.beamWidth = Int($0) }
                    ),
                    in: 4...256, step: 4
                )
                .controlSize(.mini)

                HStack(spacing: 4) {
                    Button {
                        viewModel.docking.showScaffoldInput = true
                    } label: {
                        HStack(spacing: 3) {
                            Image(systemName: "pencil.and.outline")
                                .font(.footnote)
                            Text(viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil ? "Edit Scaffold" : "Enforce Scaffold")
                                .font(.footnote)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 3)
                    }
                    .buttonStyle(.bordered)
                    .tint(viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil ? .orange : .secondary)

                    if viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil {
                        Button {
                            viewModel.docking.dockingConfig.fragment.scaffoldSMARTS = nil
                            viewModel.docking.dockingConfig.fragment.scaffoldMode = .auto
                        } label: {
                            Image(systemName: "xmark")
                                .font(.footnote)
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    }
                }

                if let scaffold = viewModel.docking.dockingConfig.fragment.scaffoldSMARTS {
                    Text("Scaffold: \(scaffold)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.orange)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .padding(.top, 2)
        }

        if method == .parallelTempering {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("Replicas")
                        .font(.footnote)
                    Spacer()
                    Picker("", selection: Binding(
                        get: { viewModel.docking.dockingConfig.replicaExchange.numReplicas },
                        set: { viewModel.docking.dockingConfig.replicaExchange.numReplicas = $0 }
                    )) {
                        Text("4").tag(4)
                        Text("8").tag(8)
                        Text("12").tag(12)
                        Text("16").tag(16)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 140)
                }

                HStack {
                    Text("T range")
                        .font(.footnote)
                    Spacer()
                    Text("\(String(format: "%.1f", viewModel.docking.dockingConfig.replicaExchange.minTemperature))–\(String(format: "%.1f", viewModel.docking.dockingConfig.replicaExchange.maxTemperature)) kcal/mol")
                        .font(.footnote.monospaced())
                }
            }
            .padding(.top, 2)
        }

        if method == .diffusionGuided {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("Denoising Steps")
                        .font(.footnote)
                    Spacer()
                    Text("\(viewModel.docking.dockingConfig.diffusion.numDenoisingSteps)")
                        .font(.footnote.monospaced())
                }
                Slider(
                    value: Binding(
                        get: { Float(viewModel.docking.dockingConfig.diffusion.numDenoisingSteps) },
                        set: { viewModel.docking.dockingConfig.diffusion.numDenoisingSteps = Int($0) }
                    ),
                    in: 10...100, step: 5
                )
                .controlSize(.mini)

                HStack {
                    Text("Schedule")
                        .font(.footnote)
                    Spacer()
                    Picker("", selection: Binding(
                        get: { viewModel.docking.dockingConfig.diffusion.noiseSchedule },
                        set: { viewModel.docking.dockingConfig.diffusion.noiseSchedule = $0 }
                    )) {
                        ForEach(DiffusionDockingConfig.NoiseSchedule.allCases, id: \.self) { sched in
                            Text(sched.rawValue).tag(sched)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(minWidth: 200, maxWidth: 260)
                }

                Text("Requires DruseAF weights")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 2)
        }
    }

    // MARK: - Docking Control

    @ViewBuilder
    private var dockingControlSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            let hasGrid = viewModel.docking.selectedPocket != nil ||
                          (viewModel.docking.gridHalfSize.x > 2 && viewModel.docking.gridHalfSize.y > 2 && viewModel.docking.gridHalfSize.z > 2)
            let canDock = hasGrid
                && viewModel.molecules.ligand != nil
                && viewModel.molecules.protein != nil
                && !viewModel.docking.isDocking

            // Search method selector
            @Bindable var vm = viewModel

            Divider()
                .padding(.vertical, 2)

            VStack(alignment: .leading, spacing: 6) {
                Text("Search Method")
                    .font(.subheadline.weight(.semibold))

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 4) {
                    ForEach(SearchMethod.allCases, id: \.self) { method in
                        Button {
                            vm.docking.dockingConfig.searchMethod = method
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: method.icon)
                                    .font(.subheadline)
                                Text(method.shortLabel)
                                    .font(.subheadline.weight(.medium))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 4)
                        }
                        .buttonStyle(.bordered)
                        .tint(vm.docking.dockingConfig.searchMethod == method ? .accentColor : .secondary)
                        .opacity(vm.docking.dockingConfig.searchMethod == method ? 1 : 0.6)
                        .help(method.description)
                    }
                }

                Text(viewModel.docking.dockingConfig.searchMethod.description)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                // Search method specific options
                searchMethodOptionsSection
            }

            Divider()
                .padding(.vertical, 2)

            // Scoring function selector
            VStack(alignment: .leading, spacing: 6) {
                Text("Scoring Function")
                    .font(.subheadline.weight(.semibold))

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 4) {
                    ForEach(scoringMethodsAvailable, id: \.self) { method in
                        Button {
                            vm.docking.scoringMethod = method
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: method.icon)
                                    .font(.subheadline)
                                Text(method.shortLabel)
                                    .font(.subheadline.weight(.medium))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 4)
                        }
                        .buttonStyle(.bordered)
                        .tint(vm.docking.scoringMethod == method ? .accentColor : .secondary)
                        .opacity(vm.docking.scoringMethod == method ? 1 : 0.6)
                        .help(method.description)
                    }
                }

                Text(viewModel.docking.scoringMethod.description)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Divider()
                .padding(.vertical, 2)

            // Post-docking refinement options
            VStack(alignment: .leading, spacing: 4) {
                Text("Post-docking Refinement")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)

                // DruseAF v4 neural rescoring
                Toggle(isOn: Binding(
                    get: { viewModel.docking.dockingConfig.useAFv4Rescore },
                    set: { viewModel.docking.dockingConfig.useAFv4Rescore = $0 }
                )) {
                    HStack(spacing: 4) {
                        Image(systemName: "brain")
                            .font(.footnote)
                        Text("DruseAF rescoring")
                            .font(.subheadline)
                    }
                }
                .toggleStyle(.checkbox)
                .controlSize(.mini)
                .help("Re-rank top poses with DruseAF v4 pairwise geometric network (pKd prediction + pose confidence). ~0.1ms/pose.")

                // GFN2-xTB rescoring
                Toggle(isOn: Binding(
                    get: { viewModel.docking.dockingConfig.gfn2Refinement.enabled },
                    set: { viewModel.docking.dockingConfig.gfn2Refinement.enabled = $0 }
                )) {
                    HStack(spacing: 4) {
                        Image(systemName: "atom")
                            .font(.footnote)
                        Text("GFN2-xTB rescoring")
                            .font(.subheadline)
                    }
                }
                .toggleStyle(.checkbox)
                .controlSize(.mini)
                .help("Geometry optimization of top poses with semi-empirical QM: D4 dispersion (π-stacking, CH-π) + implicit solvation (ALPB). ~15ms/pose for top 20.")

                // GFN2 options (shown when enabled)
                if viewModel.docking.dockingConfig.gfn2Refinement.enabled {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Solvation")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Picker("", selection: Binding(
                                get: { viewModel.docking.dockingConfig.gfn2Refinement.solvation },
                                set: { viewModel.docking.dockingConfig.gfn2Refinement.solvation = $0 }
                            )) {
                                ForEach(GFN2SolvationMode.allCases, id: \.self) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(.menu)
                            .labelsHidden()
                            .frame(width: 110)
                            .controlSize(.mini)
                        }
                        .help("Implicit solvation: ALPB (recommended) or GBSA for aqueous environment")

                        HStack {
                            Text("Opt Level")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Picker("", selection: Binding(
                                get: { viewModel.docking.dockingConfig.gfn2Refinement.optLevel },
                                set: { viewModel.docking.dockingConfig.gfn2Refinement.optLevel = $0 }
                            )) {
                                ForEach(GFN2OptLevel.allCases, id: \.self) { level in
                                    Text(level.rawValue).tag(level)
                                }
                            }
                            .pickerStyle(.menu)
                            .labelsHidden()
                            .frame(width: 80)
                            .controlSize(.mini)
                        }
                        .help("Convergence: Crude (~5ms/pose), Normal (~15ms), Tight (~30ms)")

                        HStack {
                            Text("Top poses")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Spacer()
                            TextField("", value: Binding(
                                get: { viewModel.docking.dockingConfig.gfn2Refinement.topPosesToRefine },
                                set: { viewModel.docking.dockingConfig.gfn2Refinement.topPosesToRefine = max(1, $0) }
                            ), format: .number)
                            .textFieldStyle(.roundedBorder)
                            .font(.footnote.monospaced())
                            .frame(width: 45)
                        }
                        .help("Number of top-ranked poses to optimize with GFN2-xTB")

                        HStack {
                            Text("Blend")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Slider(
                                value: Binding(
                                    get: { viewModel.docking.dockingConfig.gfn2Refinement.blendWeight },
                                    set: { viewModel.docking.dockingConfig.gfn2Refinement.blendWeight = $0 }
                                ),
                                in: 0...1.0, step: 0.05
                            )
                            .controlSize(.mini)
                            Text(String(format: "%.0f%%", viewModel.docking.dockingConfig.gfn2Refinement.blendWeight * 100))
                                .font(.footnote.monospaced())
                                .foregroundStyle(.secondary)
                                .frame(width: 28)
                        }
                        .help("How much GFN2 energy influences final ranking: 0% = scoring function only, 100% = GFN2 only")
                    }
                    .padding(.leading, 20)
                }
            }

            Button(action: {
                ensurePocketFromGrid()
                if viewModel.docking.batchQueue.count > 1 {
                    // Batch mode: dock all queued ligands
                    viewModel.dockEntries(viewModel.docking.batchQueue)
                    viewModel.docking.batchQueue = []
                } else {
                    // Single ligand: show pre-dock confirmation
                    showPreDockSheet = true
                }
            }) {
                Label(viewModel.docking.batchQueue.count > 1
                      ? "Dock \(viewModel.docking.batchQueue.count) Ligands"
                      : "Run Docking...",
                      systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(!canDock)
            .help("Launch GPU-accelerated molecular docking with the configured scoring method and GA parameters")
            .accessibilityIdentifier(AccessibilityID.dockStartButton)

            // VRAM usage warning
            if let engine = viewModel.docking.dockingEngine, let pocket = viewModel.docking.selectedPocket {
                let estimate = engine.estimateVRAMUsage(
                    gridDims: SIMD3(
                        UInt32(ceil((pocket.size.x + 7) * 2 / viewModel.docking.dockingConfig.gridSpacing)) + 1,
                        UInt32(ceil((pocket.size.y + 7) * 2 / viewModel.docking.dockingConfig.gridSpacing)) + 1,
                        UInt32(ceil((pocket.size.z + 7) * 2 / viewModel.docking.dockingConfig.gridSpacing)) + 1
                    ),
                    numAffinityTypes: 10,
                    populationSize: viewModel.docking.dockingConfig.populationSize,
                    numLigandAtoms: viewModel.molecules.ligand?.heavyAtomCount ?? 30,
                    numTorsions: 8,
                    numProteinAtoms: viewModel.molecules.protein?.heavyAtomCount ?? 0
                )
                if estimate.usageRatio > 0.7 {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                        Text("VRAM: \(String(format: "%.0f", estimate.totalMB))MB / \(String(format: "%.0f", estimate.deviceBudgetMB))MB")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                    .help("High GPU memory usage — grid spacing may be coarsened automatically")
                }
            }

            if !canDock && !viewModel.docking.isDocking {
                VStack(alignment: .leading, spacing: 2) {
                    if viewModel.molecules.protein == nil {
                        requirementLabel("Load a protein")
                    }
                    if viewModel.molecules.ligand == nil {
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
                .font(.footnote)
                .foregroundStyle(.red.opacity(0.7))
            Text(text)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Docking Progress

    @ViewBuilder
    private var dockingProgressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Docking in progress", systemImage: "bolt.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.yellow)
                Spacer()
                // Compact generation counter
                let gen = viewModel.docking.dockingGeneration + 1
                let total = viewModel.docking.dockingTotalGenerations
                Text("Gen \(gen)/\(total)")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }

            // Compact progress bar
            ProgressView(
                value: min(Double(viewModel.docking.dockingGeneration + 1),
                           Double(max(viewModel.docking.dockingTotalGenerations, 1))),
                total: Double(max(viewModel.docking.dockingTotalGenerations, 1))
            )
            .progressViewStyle(.linear)
            .tint(.cyan)

            Text("Live scores visible in viewport (top-right)")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Button(action: {
                if viewModel.docking.isBatchDocking {
                    viewModel.cancelBatchDocking()
                } else {
                    viewModel.stopDocking()
                }
            }) {
                Label(viewModel.docking.isBatchDocking ? "Stop All" : "Stop", systemImage: "stop.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .tint(.red)
            .help("Cancel the active docking run and keep the best pose found so far")
            .accessibilityIdentifier(AccessibilityID.dockCancelButton)
        }
    }

    // MARK: - Docking Statistics (replaces results preview)

    @ViewBuilder
    private var dockingResultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Docking Statistics", systemImage: "chart.bar.xaxis")
                    .font(.callout.weight(.semibold))
                Spacer()
                if viewModel.docking.dockingDuration > 0 {
                    Text(formatDuration(viewModel.docking.dockingDuration))
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            // Key statistics grid
            let results = viewModel.docking.dockingResults
            let clusterIDs = Set(results.map(\.clusterID))

            HStack(spacing: 8) {
                let energies = results.map(\.energy)
                statCell("Best", String(format: "%.1f", energies.min() ?? 0),
                         color: (energies.min() ?? 0) < -6 ? .green : .yellow)
                statCell("Mean", String(format: "%.1f", energies.isEmpty ? 0 : energies.reduce(0, +) / Float(energies.count)),
                         color: .secondary)
                statCell("Poses", "\(results.count)", color: .blue)
                statCell("Clusters", "\(clusterIDs.count)", color: .cyan)
            }
            .padding(8)
            .frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.green.opacity(0.06)))

            // Best pose scoring breakdown
            if let best = results.first {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Best Pose Breakdown")
                        .font(.footnote.weight(.medium))
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
                .help("Display the top-ranked docking pose in the 3D viewport")
            }

            // GA/search statistics
            VStack(alignment: .leading, spacing: 3) {
                Text("Search Summary")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    statMini("Generations", "\(viewModel.docking.dockingTotalGenerations)")
                    statMini("Pop. Size", "\(viewModel.docking.dockingConfig.populationSize)")
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
            .help("Browse all docking poses with scores, interactions, and export options")
        }
    }

    @ViewBuilder
    private func statCell(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.subheadline.weight(.medium).monospaced())
                .foregroundStyle(color)
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func scoreItem(_ label: String, _ value: Float) -> some View {
        VStack(spacing: 1) {
            Text(String(format: "%.1f", value))
                .font(.footnote.weight(.medium).monospaced())
                .foregroundStyle(value < 0 ? .green : .red)
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func statMini(_ label: String, _ value: String) -> some View {
        HStack(spacing: 3) {
            Text(label + ":")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.footnote.weight(.medium).monospaced())
                .foregroundStyle(.primary)
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
        VStack(spacing: 2) {
            HStack(spacing: 2) {
                Text(value)
                    .font(.subheadline.weight(.medium).monospaced())
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

    // MARK: - Actions

    private func detectPocketsAuto() {
        guard viewModel.molecules.protein != nil else { return }
        let excluded = viewModel.workspace.hiddenChainIDs
        let chainMsg = excluded.isEmpty ? "" : " (excluding chains: \(excluded.sorted().joined(separator: ", ")))"
        viewModel.log.info("Detecting binding pockets\(chainMsg)...", category: .dock)
        viewModel.detectPockets(excludedChainIDs: excluded)
    }

    private func detectFromLigand() {
        guard let prot = viewModel.molecules.protein, let lig = viewModel.molecules.ligand else { return }
        let excluded = viewModel.workspace.hiddenChainIDs
        viewModel.log.info("Detecting pocket from ligand position...", category: .dock)

        if let pocket = BindingSiteDetector.ligandGuidedPocket(protein: prot, ligand: lig, excludedChainIDs: excluded) {
            viewModel.docking.detectedPockets = [pocket]
            viewModel.docking.selectedPocket = pocket
            viewModel.log.success("Ligand-guided pocket: \(pocket.residueIndices.count) residues, \(Int(pocket.volume)) A\u{00B3}", category: .dock)
        } else {
            viewModel.log.warn("Could not define pocket from ligand", category: .dock)
        }
    }

    private func pocketFromSelection() {
        guard let prot = viewModel.molecules.protein else { return }
        let resIndices = Array(viewModel.workspace.selectedResidueIndices)
        let pocket = BindingSiteDetector.pocketFromResidues(protein: prot, residueIndices: resIndices)
        viewModel.docking.detectedPockets = [pocket]
        viewModel.docking.selectedPocket = pocket
        viewModel.log.success("Manual pocket from \(resIndices.count) residues", category: .dock)
    }

    // MARK: - Grid Box Actions

    private func initializeGridAtProteinCenter(_ prot: Molecule) {
        // Only use visible chains for center computation
        let positions: [SIMD3<Float>]
        if viewModel.workspace.hiddenChainIDs.isEmpty {
            positions = prot.atoms.map(\.position)
        } else {
            positions = prot.atoms.filter { !viewModel.workspace.hiddenChainIDs.contains($0.chainID) }.map(\.position)
        }
        guard !positions.isEmpty else { return }
        let center = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
        viewModel.docking.gridCenter = center
        viewModel.docking.gridHalfSize = SIMD3<Float>(repeating: 10)
        viewModel.docking.gridInitialized = true
    }

    private func syncGridFromPocket(_ pocket: BindingPocket) {
        viewModel.docking.gridCenter = pocket.center
        viewModel.docking.gridHalfSize = pocket.size
    }

    private func applyGridBoxFromSliders() {
        viewModel.updateGridBoxVisualization(center: viewModel.docking.gridCenter, halfSize: viewModel.docking.gridHalfSize)

        // Keep selected pocket in sync if one exists
        if var pocket = viewModel.docking.selectedPocket {
            pocket.center = viewModel.docking.gridCenter
            pocket.size = viewModel.docking.gridHalfSize
            viewModel.docking.selectedPocket = pocket
        }
    }

    // Bindings to individual components of viewModel.docking.gridCenter
    private func gridCenterBinding(_ keyPath: WritableKeyPath<SIMD3<Float>, Float>) -> Binding<Float> {
        Binding(
            get: { viewModel.docking.gridCenter[keyPath: keyPath] },
            set: { viewModel.docking.gridCenter[keyPath: keyPath] = $0 }
        )
    }

    // Bindings to individual components of viewModel.docking.gridHalfSize
    private func gridHalfBinding(_ keyPath: WritableKeyPath<SIMD3<Float>, Float>) -> Binding<Float> {
        Binding(
            get: { viewModel.docking.gridHalfSize[keyPath: keyPath] },
            set: { viewModel.docking.gridHalfSize[keyPath: keyPath] = $0 }
        )
    }

    private func placeGridAtProteinCenter() {
        guard let prot = viewModel.molecules.protein else { return }
        initializeGridAtProteinCenter(prot)
        applyGridBoxFromSliders()
    }

    private func placeGridAtLigand() {
        guard let lig = viewModel.molecules.ligand else { return }
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

        viewModel.docking.gridCenter = center
        viewModel.docking.gridHalfSize = halfSize
        applyGridBoxFromSliders()
    }

    private func placeGridAtSelection() {
        guard let prot = viewModel.molecules.protein else { return }
        let resIndices = viewModel.workspace.selectedResidueIndices
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

        viewModel.docking.gridCenter = center
        viewModel.docking.gridHalfSize = halfSize
        applyGridBoxFromSliders()
    }

    private func resetGridToPocket() {
        guard let pocket = viewModel.docking.selectedPocket else { return }
        syncGridFromPocket(pocket)
        applyGridBoxFromSliders()
    }

    /// Create a pocket from current grid sliders so the docking engine has what it needs.
    private func ensurePocketFromGrid() {
        if viewModel.docking.selectedPocket != nil { return }
        let center = viewModel.docking.gridCenter
        let size = viewModel.docking.gridHalfSize
        let pocket = BindingPocket(
            id: 0, center: center, size: size,
            volume: size.x * size.y * size.z * 8,
            buriedness: 0.5, polarity: 0.5, druggability: 0.5,
            residueIndices: [], probePositions: []
        )
        viewModel.docking.detectedPockets = [pocket]
        viewModel.docking.selectedPocket = pocket
    }
}
