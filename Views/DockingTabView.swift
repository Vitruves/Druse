// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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

    var body: some View {
        VStack(alignment: .leading, spacing: PanelStyle.cardSpacing) {
            ligandCard
            pocketCard
            gridBoxCard
            searchCard
            constraintsCard
            if viewModel.docking.selectedPocket != nil {
                flexibilityCard
            }
            refinementCard
            runCard
            if viewModel.docking.isDocking {
                progressCard
            }
            if !viewModel.docking.dockingResults.isEmpty && !viewModel.docking.isDocking {
                resultsCard
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .sheet(isPresented: $showPreDockSheet) {
            PreDockSheet().environment(viewModel)
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
            if let prot = viewModel.molecules.protein,
               viewModel.docking.selectedPocket == nil,
               viewModel.docking.dockingResults.isEmpty {
                initializeGridAtProteinCenter(prot)
            }
        }
    }

    // MARK: - Ligand card

    private var ligandCard: some View {
        let isBatch = viewModel.docking.batchQueue.count > 1
        return PanelCard(
            isBatch ? "Ligands" : "Ligand",
            icon: isBatch ? "tray.full" : "hexagon",
            accessory: {
                Button(action: { openWindow(id: "ligand-database") }) {
                    Image(systemName: "tablecells")
                        .font(.system(size: 11, weight: .medium))
                        .frame(width: 22, height: 22)
                        .background(
                            RoundedRectangle(cornerRadius: 5, style: .continuous)
                                .fill(Color.accentColor.opacity(0.14))
                        )
                        .foregroundStyle(Color.accentColor)
                }
                .buttonStyle(.plain)
                .help("Open Ligand Database (Cmd+L)")
                .plainButtonAccessibility(AccessibilityID.dockOpenLigandDB)
            }
        ) {
            ligandCardContent
        }
    }

    @ViewBuilder
    private var ligandCardContent: some View {
        if viewModel.docking.batchQueue.count > 1 {
            batchLigandSummary
        } else if let lig = viewModel.molecules.ligand {
            singleLigandSummary(lig)
        } else {
            PanelSecondaryButton(title: "Open Database to select ligand",
                                 icon: "plus.circle") {
                openWindow(id: "ligand-database")
            }
        }
    }

    @ViewBuilder
    private var batchLigandSummary: some View {
        PanelHighlightRow(color: .cyan) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Image(systemName: "tray.full.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.cyan)
                    Text("\(viewModel.docking.batchQueue.count) ligands queued")
                        .font(PanelStyle.bodyFont.weight(.medium))
                    Spacer()
                    Button(action: {
                        viewModel.docking.batchQueue = []
                        viewModel.clearLigand()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear batch queue")
                    .plainButtonAccessibility(AccessibilityID.dockClearBatch)
                }
                let mws = viewModel.docking.batchQueue.compactMap { $0.descriptors?.molecularWeight }
                if let minMW = mws.min(), let maxMW = mws.max() {
                    Text(String(format: "MW range: %.0f – %.0f Da", minMW, maxMW))
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
                let prepCount = viewModel.docking.batchQueue.filter(\.isPrepared).count
                Text("\(prepCount)/\(viewModel.docking.batchQueue.count) prepared")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(prepCount == viewModel.docking.batchQueue.count ? .green : .orange)
            }
        }
    }

    @ViewBuilder
    private func singleLigandSummary(_ lig: Molecule) -> some View {
        PanelHighlightRow(color: .green) {
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.green)
                Text(lig.name)
                    .font(PanelStyle.bodyFont.weight(.medium))
                    .lineLimit(1)
                Spacer()
                Text("\(lig.atomCount) atoms")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
                Button(action: { viewModel.removeLigandFromView() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Remove active ligand")
                .plainButtonAccessibility(AccessibilityID.dockRemoveLigand)
            }
        }
    }

    // MARK: - Pocket card (detection + selected pocket + pocket view)

    private var pocketCard: some View {
        PanelCard("Binding Site", icon: "scope") {
            VStack(alignment: .leading, spacing: 10) {
                pocketDetectionContent
                if !viewModel.docking.detectedPockets.isEmpty {
                    Divider().opacity(0.4)
                    VStack(spacing: 4) {
                        ForEach(viewModel.docking.detectedPockets) { pocket in
                            pocketRow(pocket)
                        }
                    }
                }
                if let pocket = viewModel.docking.selectedPocket {
                    Divider().opacity(0.4)
                    selectedPocketStats(pocket)
                }
                Divider().opacity(0.4)
                pocketViewControls
            }
        }
    }

    @ViewBuilder
    private var pocketDetectionContent: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Detect")
            PanelChoiceGrid(columns: 2) {
                PanelChoiceButton(
                    title: "Auto", icon: "sparkles", isSelected: false,
                    isDisabled: viewModel.molecules.protein == nil,
                    help: "Hybrid pocket detection: ML candidates plus geometric fallback"
                ) { detectPocketsAuto() }
                .accessibilityIdentifier(AccessibilityID.dockDetectAuto)

                PanelChoiceButton(
                    title: "ML", icon: "brain", isSelected: false,
                    isDisabled: viewModel.molecules.protein == nil || !viewModel.pocketDetectorML.isAvailable,
                    help: "ML-based pocket detection (GNN)"
                ) { viewModel.detectPocketsML() }
                .accessibilityIdentifier(AccessibilityID.dockDetectML)

                PanelChoiceButton(
                    title: "From Ligand", icon: "hexagon", isSelected: false,
                    isDisabled: viewModel.molecules.protein == nil || viewModel.molecules.ligand == nil,
                    help: "Define pocket around current ligand"
                ) { detectFromLigand() }
                .accessibilityIdentifier(AccessibilityID.dockDetectLigand)

                PanelChoiceButton(
                    title: "From Selection", icon: "hand.tap", isSelected: false,
                    isDisabled: viewModel.molecules.protein == nil || viewModel.workspace.selectedResidueIndices.isEmpty,
                    help: "Define pocket from selected residues"
                ) { pocketFromSelection() }
                .accessibilityIdentifier(AccessibilityID.dockDetectSelection)
            }
        }
    }

    @ViewBuilder
    private func pocketRow(_ pocket: BindingPocket) -> some View {
        let isSelected = viewModel.docking.selectedPocket?.id == pocket.id
        HStack(spacing: 8) {
            Circle()
                .fill(isSelected ? Color.green : Color.secondary.opacity(0.5))
                .frame(width: 6, height: 6)
            Text("Pocket #\(pocket.id)")
                .font(PanelStyle.smallFont.weight(.medium))
            Spacer()
            Text(String(format: "%.0f Å³", pocket.volume))
                .font(PanelStyle.monoSmall)
                .foregroundStyle(.secondary)
            Text("\(pocket.residueIndices.count) res")
                .font(PanelStyle.monoSmall)
                .foregroundStyle(.secondary)
            Button(action: { removePocket(pocket) }) {
                Image(systemName: "trash")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Remove this pocket and its grid box")
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(isSelected ? Color.green.opacity(0.10) : Color.clear)
        )
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.docking.selectedPocket = pocket
            ActivityLog.shared.info(
                "Pocket #\(pocket.id) selected (vol \(String(format: "%.0f", pocket.volume)) Å³, druggability \(String(format: "%.2f", pocket.druggability)))",
                category: .dock
            )
        }
    }

    @ViewBuilder
    private func selectedPocketStats(_ pocket: BindingPocket) -> some View {
        HStack(spacing: 8) {
            PanelStat(label: "Volume", value: String(format: "%.0f", pocket.volume), unit: "Å³")
            PanelStat(label: "Buried", value: String(format: "%.0f%%", pocket.buriedness * 100))
            PanelStat(label: "Residues", value: "\(pocket.residueIndices.count)")
            PanelSecondaryButton(title: "Focus", icon: "scope", help: "Zoom to pocket with Z-clipping slab") {
                viewModel.focusOnPocket(pocket)
            }
            .accessibilityIdentifier(AccessibilityID.dockFocusPocket)
            .frame(width: 80)
        }
    }

    @ViewBuilder
    private var pocketViewControls: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 8) {
            PanelToggleRow(
                title: "Pocket View",
                subtitle: vm.workspace.enableClipping ? nil : "Clip view around the binding site",
                icon: "rectangle.split.3x1",
                isOn: Binding(
                    get: { vm.workspace.enableClipping },
                    set: { newValue in setPocketClipping(newValue) }
                )
            )
            .accessibilityIdentifier(AccessibilityID.dockPocketViewToggle)

            if vm.workspace.enableClipping {
                PanelSliderRow(
                    label: "Thickness",
                    value: Binding(
                        get: { vm.workspace.slabThickness },
                        set: { newVal in
                            vm.workspace.slabThickness = newVal
                            vm.renderer?.slabHalfThickness = newVal / 2.0
                            vm.renderer?.setNeedsRedraw()
                        }
                    ),
                    range: 2...40,
                    step: 0.5,
                    format: { String(format: "%.1f Å", $0) },
                    valueColor: vm.workspace.slabThickness < 10 ? .orange : .secondary
                )
                PanelSliderRow(
                    label: "Offset",
                    value: Binding(
                        get: { vm.workspace.slabOffset },
                        set: { newVal in
                            vm.workspace.slabOffset = newVal
                            vm.renderer?.slabOffset = newVal
                            vm.renderer?.setNeedsRedraw()
                        }
                    ),
                    range: -20...20,
                    step: 0.5,
                    format: { String(format: "%+.1f Å", $0) }
                )
                PanelChoiceGrid(columns: 3) {
                    ForEach(["Tight", "Medium", "Wide"], id: \.self) { preset in
                        PanelChoiceButton(
                            title: preset, isSelected: false,
                            help: "Z-slab \(preset.lowercased()) clipping around the pocket"
                        ) { applySlabPreset(preset) }
                        .accessibilityIdentifier(
                            preset == "Tight" ? AccessibilityID.dockSlabTight :
                            preset == "Medium" ? AccessibilityID.dockSlabMedium :
                            AccessibilityID.dockSlabWide
                        )
                    }
                }
            }
        }
    }

    private func setPocketClipping(_ newValue: Bool) {
        let vm = viewModel
        vm.workspace.enableClipping = newValue
        vm.renderer?.enableClipping = newValue
        if newValue, let pocket = vm.docking.selectedPocket {
            vm.focusOnPocket(pocket)
            if !vm.workspace.showSurface {
                vm.toggleSurface()
            }
            if vm.workspace.surfaceOpacity > 0.7 {
                vm.workspace.surfaceOpacity = 0.55
                vm.renderer?.surfaceOpacity = 0.55
            }
        } else if !newValue {
            if vm.workspace.showSurface {
                vm.toggleSurface()
            }
            vm.renderer?.slabCenter = nil
            vm.fitToView()
        }
        vm.renderer?.setNeedsRedraw()
    }

    // MARK: - Grid box card

    private var gridBoxCard: some View {
        PanelCard(
            "Grid Box",
            icon: "cube.transparent",
            accessory: {
                Button(action: { applyGridBoxFromSliders() }) {
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(Color.accentColor)
                }
                .buttonStyle(.plain)
                .help("Apply grid box to viewport")
                .plainButtonAccessibility(AccessibilityID.dockApplyGrid)
            }
        ) {
            VStack(alignment: .leading, spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    PanelSubheader(title: "Center")
                    PanelAxisSlider(axis: "X", value: gridCenterBinding(\.x), range: -200...200) { _ in applyGridBoxFromSliders() }
                    PanelAxisSlider(axis: "Y", value: gridCenterBinding(\.y), range: -200...200) { _ in applyGridBoxFromSliders() }
                    PanelAxisSlider(axis: "Z", value: gridCenterBinding(\.z), range: -200...200) { _ in applyGridBoxFromSliders() }
                }
                VStack(alignment: .leading, spacing: 4) {
                    PanelSubheader(title: "Half-size")
                    PanelAxisSlider(axis: "X", value: gridHalfBinding(\.x), range: 2...50) { _ in applyGridBoxFromSliders() }
                    PanelAxisSlider(axis: "Y", value: gridHalfBinding(\.y), range: 2...50) { _ in applyGridBoxFromSliders() }
                    PanelAxisSlider(axis: "Z", value: gridHalfBinding(\.z), range: 2...50) { _ in applyGridBoxFromSliders() }
                }
                VStack(alignment: .leading, spacing: 4) {
                    PanelSubheader(title: "Center on")
                    PanelChoiceGrid(columns: 2) {
                        PanelChoiceButton(
                            title: "Protein", icon: "building.2",
                            isSelected: false,
                            isDisabled: viewModel.molecules.protein == nil,
                            help: "Center grid on protein centroid"
                        ) { placeGridAtProteinCenter() }
                        .accessibilityIdentifier(AccessibilityID.dockGridProtein)

                        PanelChoiceButton(
                            title: "Ligand", icon: "hexagon",
                            isSelected: false,
                            isDisabled: viewModel.molecules.ligand == nil,
                            help: "Center grid on ligand position"
                        ) { placeGridAtLigand() }
                        .accessibilityIdentifier(AccessibilityID.dockGridLigand)

                        PanelChoiceButton(
                            title: "Selection", icon: "hand.tap",
                            isSelected: false,
                            isDisabled: viewModel.workspace.selectedResidueIndices.isEmpty,
                            help: "Center grid on selected residues"
                        ) { placeGridAtSelection() }
                        .accessibilityIdentifier(AccessibilityID.dockGridSelection)

                        PanelChoiceButton(
                            title: "Pocket", icon: "arrow.counterclockwise",
                            isSelected: false,
                            isDisabled: viewModel.docking.selectedPocket == nil,
                            help: "Reset grid to detected pocket"
                        ) { resetGridToPocket() }
                        .accessibilityIdentifier(AccessibilityID.dockGridPocket)
                    }
                }
            }
        }
    }

    // MARK: - Search card (preset + method + scoring + budget + advanced)

    private var searchCard: some View {
        return PanelCard("Search", icon: "magnifyingglass") {
            VStack(alignment: .leading, spacing: 12) {
                presetSection
                searchMethodSection
                scoringFunctionSection
                searchBudgetSection
                PanelLabeledDivider(title: "Advanced", icon: "slider.horizontal.3")
                    .padding(.top, 4)
                VStack(alignment: .leading, spacing: 18) {
                    searchBehaviorSection
                    searchMethodOptionsSection
                    VStack(alignment: .leading, spacing: 8) {
                        PanelSubheader(title: "Exploration Phase", icon: "sparkles")
                        explorationPhaseSection
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var presetSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Preset")
            PanelChoiceGrid(columns: 4) {
                ForEach(["Auto", "Fast", "Standard", "Thorough"], id: \.self) { preset in
                    PanelChoiceButton(
                        title: preset,
                        isSelected: isPresetActive(preset),
                        help: presetHelp(preset)
                    ) { applyPreset(preset) }
                }
            }
            if viewModel.docking.dockingConfig.autoMode {
                PanelHint(text: "Parameters adapt to protein size, pocket shape, and ligand flexibility",
                          icon: "wand.and.stars",
                          color: .purple)
            }
        }
    }

    @ViewBuilder
    private var searchMethodSection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Method")
            PanelChoiceGrid(columns: 2) {
                ForEach(SearchMethod.allCases, id: \.self) { method in
                    PanelChoiceButton(
                        title: method.shortLabel,
                        icon: method.icon,
                        badge: method.isExperimental ? "BETA" : nil,
                        isSelected: vm.docking.dockingConfig.searchMethod == method,
                        help: method.description
                    ) {
                        vm.docking.dockingConfig.searchMethod = method
                        if method == .diffusionGuided && vm.docking.scoringMethod != .druseAffinity {
                            vm.docking.scoringMethod = .druseAffinity
                        }
                    }
                }
            }
            Text(viewModel.docking.dockingConfig.searchMethod.description)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var scoringFunctionSection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Scoring")
            PanelChoiceGrid(columns: 2) {
                ForEach(scoringMethodsAvailable, id: \.self) { method in
                    PanelChoiceButton(
                        title: method.shortLabel,
                        icon: method.icon,
                        isSelected: vm.docking.scoringMethod == method,
                        help: method.description
                    ) { vm.docking.scoringMethod = method }
                }
            }
            Text(viewModel.docking.scoringMethod.description)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ScoringInteractionsBadgeRow(method: viewModel.docking.scoringMethod)
        }
    }

    @ViewBuilder
    private var searchBudgetSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Budget", icon: "speedometer")
            PanelLabeledRow("Population", help: "Candidate poses per generation") {
                PanelNumberField(
                    value: Binding(
                        get: { viewModel.docking.dockingConfig.populationSize },
                        set: { viewModel.docking.dockingConfig.populationSize = max(10, $0) }
                    ),
                    minimum: 10
                )
            }
            PanelLabeledRow("Generations", help: "GA evolution cycles within each independent run") {
                PanelNumberField(
                    value: Binding(
                        get: { viewModel.docking.dockingConfig.generationsPerRun },
                        set: { viewModel.docking.dockingConfig.generationsPerRun = max(10, $0) }
                    ),
                    minimum: 10
                )
            }
            PanelLabeledRow("Runs", help: "Independent restarts (more = better sampling)") {
                PanelNumberField(
                    value: Binding(
                        get: { viewModel.docking.dockingConfig.numRuns },
                        set: { viewModel.docking.dockingConfig.numRuns = max(1, $0) }
                    ),
                    minimum: 1
                )
            }
            HStack {
                Text("Total evaluations")
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(totalEvaluationsString)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
            }
            .padding(.top, 2)
        }
    }

    @ViewBuilder
    private var searchBehaviorSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            PanelSubheader(title: "Behavior")
            PanelSliderRow(
                label: "Mutation Rate",
                value: Binding(
                    get: { viewModel.docking.dockingConfig.mutationRate },
                    set: { viewModel.docking.dockingConfig.mutationRate = $0 }
                ),
                range: 0.01...0.25,
                step: 0.005,
                format: { String(format: "%.3f", $0) }
            )
            PanelToggleRow(
                title: "Ligand Flexibility",
                subtitle: "Allow rotatable bonds to flex",
                isOn: Binding(
                    get: { viewModel.docking.dockingConfig.enableFlexibility },
                    set: { viewModel.docking.dockingConfig.enableFlexibility = $0 }
                )
            )
            PanelLabeledRow("Grid Spacing",
                            help: "Energy grid resolution (smaller = more accurate but slower)") {
                Picker("", selection: Binding(
                    get: { viewModel.docking.dockingConfig.gridSpacing },
                    set: { viewModel.docking.dockingConfig.gridSpacing = $0 }
                )) {
                    Text("0.375 Å").tag(Float(0.375))
                    Text("0.500 Å").tag(Float(0.500))
                    Text("0.750 Å").tag(Float(0.750))
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .frame(width: 96)
            }
        }
    }

    @ViewBuilder
    private var explorationPhaseSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSliderRow(
                label: "Phase Ratio",
                value: Binding(
                    get: { viewModel.docking.dockingConfig.explorationPhaseRatio },
                    set: { viewModel.docking.dockingConfig.explorationPhaseRatio = $0 }
                ),
                range: 0.2...0.8, step: 0.05,
                format: { String(format: "%.2f", $0) }
            )
            PanelSliderRow(
                label: "Local Search",
                value: Binding(
                    get: { Float(viewModel.docking.dockingConfig.explorationLocalSearchFrequency) },
                    set: { viewModel.docking.dockingConfig.explorationLocalSearchFrequency = max(1, Int($0.rounded())) }
                ),
                range: 1...10, step: 1,
                format: { v in
                    let n = Int(v.rounded())
                    return "every \(n)\(ordinalSuffix(n))"
                }
            )
            PanelSliderRow(
                label: "MC Temperature",
                value: Binding(
                    get: { viewModel.docking.dockingConfig.mcTemperature },
                    set: { viewModel.docking.dockingConfig.mcTemperature = $0 }
                ),
                range: 0.5...4.0, step: 0.1,
                format: { String(format: "%.1f", $0) }
            )
            PanelSliderRow(
                label: "Mutation",
                value: Binding(
                    get: { viewModel.docking.dockingConfig.explorationMutationRate },
                    set: { viewModel.docking.dockingConfig.explorationMutationRate = $0 }
                ),
                range: 0.10...0.50, step: 0.01,
                format: { String(format: "%.2f", $0) }
            )
        }
    }

    @ViewBuilder
    private var searchMethodOptionsSection: some View {
        let method = viewModel.docking.dockingConfig.searchMethod
        if method == .fragmentBased {
            fragmentMethodOptions
        } else if method == .parallelTempering {
            parallelTemperingOptions
        } else if method == .diffusionGuided {
            diffusionMethodOptions
        }
    }

    @ViewBuilder
    private var fragmentMethodOptions: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Fragment", icon: "puzzlepiece")
            PanelSliderRow(
                label: "Beam Width",
                value: Binding(
                    get: { Float(viewModel.docking.dockingConfig.fragment.beamWidth) },
                    set: { viewModel.docking.dockingConfig.fragment.beamWidth = Int($0) }
                ),
                range: 4...256, step: 4,
                format: { String(Int($0)) }
            )
            HStack(spacing: 6) {
                PanelSecondaryButton(
                    title: viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil ? "Edit Scaffold" : "Enforce Scaffold",
                    icon: "pencil.and.outline",
                    tint: viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil ? .orange : nil
                ) { viewModel.docking.showScaffoldInput = true }
                if viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil {
                    PanelSecondaryButton(title: "", icon: "xmark", tint: .red) {
                        viewModel.docking.dockingConfig.fragment.scaffoldSMARTS = nil
                        viewModel.docking.dockingConfig.fragment.scaffoldMode = .auto
                    }
                    .frame(width: 28)
                }
            }
            if let scaffold = viewModel.docking.dockingConfig.fragment.scaffoldSMARTS {
                Text("Scaffold: \(scaffold)")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.orange)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    @ViewBuilder
    private var parallelTemperingOptions: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Replica Exchange", icon: "thermometer")
            PanelLabeledRow("Replicas") {
                Picker("", selection: Binding(
                    get: { viewModel.docking.dockingConfig.replicaExchange.numReplicas },
                    set: { viewModel.docking.dockingConfig.replicaExchange.numReplicas = $0 }
                )) {
                    ForEach([4, 8, 12, 16], id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
            }
            HStack {
                Text("Temperature range")
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(String(format: "%.1f", viewModel.docking.dockingConfig.replicaExchange.minTemperature))–\(String(format: "%.1f", viewModel.docking.dockingConfig.replicaExchange.maxTemperature)) kcal/mol")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var diffusionMethodOptions: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Diffusion", icon: "waveform.path.ecg")
            PanelSliderRow(
                label: "Denoising",
                value: Binding(
                    get: { Float(viewModel.docking.dockingConfig.diffusion.numDenoisingSteps) },
                    set: { viewModel.docking.dockingConfig.diffusion.numDenoisingSteps = Int($0) }
                ),
                range: 10...100, step: 5,
                format: { "\(Int($0)) steps" }
            )
            PanelLabeledRow("Schedule") {
                Picker("", selection: Binding(
                    get: { viewModel.docking.dockingConfig.diffusion.noiseSchedule },
                    set: { viewModel.docking.dockingConfig.diffusion.noiseSchedule = $0 }
                )) {
                    ForEach(DiffusionDockingConfig.NoiseSchedule.allCases, id: \.self) { sched in
                        Text(sched.rawValue).tag(sched)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 200)
            }
            PanelHint(
                text: "Experimental. DruseAF was trained for affinity prediction, not denoising; the gradient is heuristic. Use GA or Fragment for production work.",
                icon: "exclamationmark.triangle.fill",
                color: .orange
            )
        }
    }

    // MARK: - Constraints card

    @ViewBuilder
    private var constraintsCard: some View {
        @Bindable var vm = viewModel
        let count = viewModel.docking.pharmacophoreConstraints.count
        PanelCard(
            "Constraints" + (count > 0 ? " (\(count))" : ""),
            icon: "circle.hexagongrid.circle",
            accessory: {
                if count > 0 {
                    Button("Clear") {
                        viewModel.docking.pharmacophoreConstraints.removeAll()
                    }
                    .font(PanelStyle.smallFont)
                    .buttonStyle(.plain)
                    .foregroundStyle(.red)
                    .help("Remove all pharmacophore constraints")
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                PanelSecondaryButton(title: "Pharmacophore Editor…",
                                     icon: "circle.hexagongrid.circle",
                                     help: "Create constraints from a reference ligand's pharmacophoric features") {
                    viewModel.docking.showPharmacophoreEditor = true
                }
                ForEach(Array(viewModel.docking.pharmacophoreConstraints.enumerated()), id: \.element.id) { idx, constraint in
                    constraintRow(idx: idx, constraint: constraint)
                }
            }
        }
        .sheet(isPresented: $vm.docking.showPharmacophoreEditor) {
            PharmacophoreEditorView().environment(viewModel)
        }
    }

    @ViewBuilder
    private func constraintRow(idx: Int, constraint: PharmacophoreConstraintDef) -> some View {
        HStack(spacing: 6) {
            Image(systemName: constraint.interactionType.icon)
                .font(.system(size: 11))
                .frame(width: 14)
                .foregroundColor(Color(
                    red: Double(constraint.interactionType.color.x),
                    green: Double(constraint.interactionType.color.y),
                    blue: Double(constraint.interactionType.color.z)
                ))
            Text(constraint.targetLabel)
                .font(PanelStyle.monoSmall)
                .lineLimit(1)
            Spacer()
            Text(constraint.interactionType.rawValue)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
            Text(constraint.strength.isHard ? "Hard" : "Soft")
                .font(.system(size: 10, weight: .medium))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill((constraint.strength.isHard ? Color.red : Color.orange).opacity(0.18))
                )
            Button(action: {
                viewModel.docking.pharmacophoreConstraints.remove(at: idx)
            }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Flexibility card

    @ViewBuilder
    private var flexibilityCard: some View {
        @Bindable var vm = viewModel
        PanelCard(
            "Receptor Flexibility",
            icon: "figure.flexibility",
            accessory: {
                if !viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.isEmpty
                    && !viewModel.docking.flexibleResidueConfig.autoFlex {
                    Button("Clear") {
                        viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll()
                    }
                    .font(PanelStyle.smallFont)
                    .buttonStyle(.plain)
                    .foregroundStyle(.red)
                    .help("Remove all flexible residues (use rigid receptor)")
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                PanelToggleRow(
                    title: "Soft Flexibility",
                    subtitle: "Pocket-lining sidechains adjust gently during docking",
                    isOn: $vm.docking.flexibleResidueConfig.autoFlex
                )
                .onChange(of: viewModel.docking.flexibleResidueConfig.autoFlex) { _, isAuto in
                    if isAuto {
                        viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll()
                    }
                }

                if viewModel.docking.flexibleResidueConfig.autoFlex {
                    autoFlexPreview
                } else {
                    manualFlexResidues
                }
            }
        }
    }

    @ViewBuilder
    private var autoFlexPreview: some View {
        if let prot = viewModel.molecules.protein,
           let pocket = viewModel.docking.selectedPocket {
            let autoIndices = FlexibleResidueConfig.autoSelectResidues(
                protein: prot.atoms,
                pocket: (center: pocket.center, residueIndices: pocket.residueIndices)
            )
            if autoIndices.isEmpty {
                PanelHint(text: "No rotatable residues lining this pocket", color: .orange)
            } else {
                let names = autoIndices.compactMap { seq -> String? in
                    prot.atoms.first(where: { $0.residueSeq == seq }).map { "\($0.residueName)\(seq)" }
                }
                FlowLayout(spacing: 4) {
                    ForEach(names, id: \.self) { name in
                        PanelChip(text: name, color: .purple)
                    }
                }
                let totalChi = autoIndices.compactMap { seq -> Int? in
                    prot.atoms.first(where: { $0.residueSeq == seq }).flatMap {
                        RotamerLibrary.rotamers(for: $0.residueName)?.chiAngles.count
                    }
                }.reduce(0, +)
                Text("\(autoIndices.count) residue(s), \(totalChi) chi angle(s) — soft weight")
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var manualFlexResidues: some View {
        let flexIndices = viewModel.docking.flexibleResidueConfig.flexibleResidueIndices
        if flexIndices.isEmpty {
            PanelHint(text: "Select residues in the sequence panel or 3D viewport for full induced-fit docking.")
        } else {
            let residueNames = flexIndices.compactMap { seq -> String? in
                guard let prot = viewModel.molecules.protein else { return nil }
                if let atom = prot.atoms.first(where: { $0.residueSeq == seq }) {
                    return "\(atom.residueName)\(seq)"
                }
                return "?\(seq)"
            }
            FlowLayout(spacing: 4) {
                ForEach(residueNames, id: \.self) { name in
                    PanelChip(text: name, color: .purple)
                }
            }
            let totalChi = flexIndices.compactMap { seq -> Int? in
                guard let prot = viewModel.molecules.protein,
                      let atom = prot.atoms.first(where: { $0.residueSeq == seq }) else { return nil }
                return RotamerLibrary.rotamers(for: atom.residueName)?.chiAngles.count
            }.reduce(0, +)
            Text("\(flexIndices.count) residue(s), \(totalChi) chi angle(s)")
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
            if flexIndices.count >= FlexibleResidueConfig.maxFlexibleResidues {
                PanelHint(text: "Maximum \(FlexibleResidueConfig.maxFlexibleResidues) flexible residues reached", color: .orange)
            }
        }
    }

    // MARK: - Refinement card

    @ViewBuilder
    private var refinementCard: some View {
        let afOn = viewModel.docking.dockingConfig.useAFv4Rescore
        let xtbOn = viewModel.docking.dockingConfig.gfn2Refinement.enabled
        let summary: String = {
            if afOn && xtbOn { return "DruseAF + GFN2-xTB" }
            if afOn { return "DruseAF" }
            if xtbOn { return "GFN2-xTB" }
            return "Off"
        }()
        PanelCard(
            "Post-docking Refinement",
            icon: "wand.and.rays",
            accessory: {
                Text(summary)
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(afOn || xtbOn ? Color.accentColor : .secondary)
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                PanelToggleRow(
                    title: "DruseAF rescoring",
                    subtitle: "Re-rank top poses with DruseAF v4 (~0.1 ms/pose)",
                    icon: "brain",
                    isOn: Binding(
                        get: { viewModel.docking.dockingConfig.useAFv4Rescore },
                        set: { viewModel.docking.dockingConfig.useAFv4Rescore = $0 }
                    )
                )
                PanelToggleRow(
                    title: "GFN2-xTB rescoring",
                    subtitle: "Semi-empirical QM optimization of top poses (~15 ms/pose)",
                    icon: "atom",
                    isOn: Binding(
                        get: { viewModel.docking.dockingConfig.gfn2Refinement.enabled },
                        set: { viewModel.docking.dockingConfig.gfn2Refinement.enabled = $0 }
                    )
                )
                if viewModel.docking.dockingConfig.gfn2Refinement.enabled {
                    gfn2Options
                }
            }
        }
    }

    @ViewBuilder
    private var gfn2Options: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelLabeledRow("Solvation",
                            help: "Implicit solvation: ALPB (recommended) or GBSA for aqueous environment") {
                Picker("", selection: Binding(
                    get: { viewModel.docking.dockingConfig.gfn2Refinement.solvation },
                    set: { viewModel.docking.dockingConfig.gfn2Refinement.solvation = $0 }
                )) {
                    ForEach(GFN2SolvationMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .frame(width: 110)
            }
            PanelLabeledRow("Opt Level",
                            help: "Crude (~5 ms/pose), Normal (~15 ms), Tight (~30 ms)") {
                Picker("", selection: Binding(
                    get: { viewModel.docking.dockingConfig.gfn2Refinement.optLevel },
                    set: { viewModel.docking.dockingConfig.gfn2Refinement.optLevel = $0 }
                )) {
                    ForEach(GFN2OptLevel.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .frame(width: 96)
            }
            PanelLabeledRow("Top poses",
                            help: "Number of top-ranked poses to optimize with GFN2-xTB") {
                PanelNumberField(
                    value: Binding(
                        get: { viewModel.docking.dockingConfig.gfn2Refinement.topPosesToRefine },
                        set: { viewModel.docking.dockingConfig.gfn2Refinement.topPosesToRefine = max(1, $0) }
                    ),
                    width: 56
                )
            }
            PanelSliderRow(
                label: "Blend",
                value: Binding(
                    get: { viewModel.docking.dockingConfig.gfn2Refinement.blendWeight },
                    set: { viewModel.docking.dockingConfig.gfn2Refinement.blendWeight = $0 }
                ),
                range: 0...1.0, step: 0.05,
                format: { String(format: "%.0f%%", $0 * 100) }
            )
        }
        .padding(.leading, 18)
    }

    // MARK: - Run card

    @ViewBuilder
    private var runCard: some View {
        let hasGrid = viewModel.docking.selectedPocket != nil ||
            (viewModel.docking.gridHalfSize.x > 2 && viewModel.docking.gridHalfSize.y > 2 && viewModel.docking.gridHalfSize.z > 2)
        let canDock = hasGrid
            && viewModel.molecules.ligand != nil
            && viewModel.molecules.protein != nil
            && !viewModel.docking.isDocking

        VStack(alignment: .leading, spacing: 8) {
            PanelRunButton(
                title: viewModel.docking.batchQueue.count > 1
                    ? "Dock \(viewModel.docking.batchQueue.count) Ligands"
                    : "Run Docking",
                icon: "play.fill",
                isDisabled: !canDock
            ) {
                ensurePocketFromGrid()
                if viewModel.docking.batchQueue.count > 1 {
                    viewModel.dockEntries(viewModel.docking.batchQueue)
                    viewModel.docking.batchQueue = []
                } else {
                    showPreDockSheet = true
                }
            }
            .accessibilityIdentifier(AccessibilityID.dockStartButton)

            if let warning = vramWarning() {
                PanelHint(text: warning, icon: "exclamationmark.triangle.fill", color: .orange)
            }
            if !canDock && !viewModel.docking.isDocking {
                VStack(alignment: .leading, spacing: 3) {
                    if viewModel.molecules.protein == nil {
                        PanelRequirement(text: "Load a protein")
                    }
                    if viewModel.molecules.ligand == nil {
                        PanelRequirement(text: "Set an active ligand")
                    }
                    if !hasGrid {
                        PanelRequirement(text: "Configure grid box or detect pocket")
                    }
                }
            }
        }
    }

    private func vramWarning() -> String? {
        guard let engine = viewModel.docking.dockingEngine,
              let pocket = viewModel.docking.selectedPocket else { return nil }
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
        guard estimate.usageRatio > 0.7 else { return nil }
        return String(format: "High VRAM: %.0f MB / %.0f MB — grid spacing may be coarsened automatically",
                      estimate.totalMB, estimate.deviceBudgetMB)
    }

    // MARK: - Progress card

    @ViewBuilder
    private var progressCard: some View {
        PanelCard(
            "Docking In Progress",
            icon: "bolt.fill",
            accessory: {
                let gen = viewModel.docking.dockingGeneration + 1
                let total = viewModel.docking.dockingTotalGenerations
                Text("Gen \(gen)/\(total)")
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(
                    value: min(Double(viewModel.docking.dockingGeneration + 1),
                               Double(max(viewModel.docking.dockingTotalGenerations, 1))),
                    total: Double(max(viewModel.docking.dockingTotalGenerations, 1))
                )
                .progressViewStyle(.linear)
                .tint(.cyan)

                PanelHint(text: "Live scores visible in viewport (top-right)")

                PanelSecondaryButton(
                    title: viewModel.docking.isBatchDocking ? "Stop All" : "Stop",
                    icon: "stop.fill",
                    tint: .red,
                    help: "Cancel the active docking run and keep the best pose found so far"
                ) {
                    if viewModel.docking.isBatchDocking {
                        viewModel.cancelBatchDocking()
                    } else {
                        viewModel.stopDocking()
                    }
                }
                .accessibilityIdentifier(AccessibilityID.dockCancelButton)
            }
        }
    }

    // MARK: - Results card

    @ViewBuilder
    private var resultsCard: some View {
        PanelCard(
            "Docking Statistics",
            icon: "chart.bar.xaxis",
            accessory: {
                if viewModel.docking.dockingDuration > 0 {
                    Text(formatDuration(viewModel.docking.dockingDuration))
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 10) {
                resultsStatGrid
                posebustersBadge
                if let best = viewModel.docking.dockingResults.first {
                    bestPoseBreakdown(best)
                }
                searchSummary
                PanelSecondaryButton(title: "Open Results Database",
                                     icon: "tablecells",
                                     help: "Browse all docking poses with scores and exports") {
                    openWindow(id: "results-database")
                }
            }
        }
    }

    @ViewBuilder
    private var resultsStatGrid: some View {
        let results = viewModel.docking.dockingResults
        let energies = results.map(\.energy)
        let clusterIDs = Set(results.map(\.clusterID))
        HStack(spacing: 8) {
            PanelStat(label: "Best", value: String(format: "%.1f", energies.min() ?? 0),
                      color: (energies.min() ?? 0) < -6 ? .green : .yellow)
            PanelStat(label: "Mean",
                      value: String(format: "%.1f", energies.isEmpty ? 0 : energies.reduce(0, +) / Float(energies.count)))
            PanelStat(label: "Poses", value: "\(results.count)", color: .blue)
            PanelStat(label: "Clusters", value: "\(clusterIDs.count)", color: .cyan)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.green.opacity(0.08))
        )
    }

    @ViewBuilder
    private var posebustersBadge: some View {
        let results = viewModel.docking.dockingResults
        if results.contains(where: { $0.validity != nil }) {
            let validCount = results.filter { $0.validity?.passed == true }.count
            let total = results.count
            let allPassed = validCount == total
            HStack(spacing: 6) {
                Image(systemName: allPassed ? "checkmark.seal.fill" : "exclamationmark.shield")
                    .foregroundStyle(allPassed ? .green : .orange)
                    .font(.system(size: 11))
                Text("PoseBusters: \(validCount)/\(total) passed")
                    .font(PanelStyle.smallFont.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if !allPassed,
                   let firstBad = results.first(where: { $0.validity?.passed == false }),
                   let v = firstBad.validity, !v.failures.isEmpty {
                    Text(v.failures.prefix(2).joined(separator: ", "))
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.orange)
                        .lineLimit(1)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill((allPassed ? Color.green : Color.orange).opacity(0.10))
            )
        }
    }

    @ViewBuilder
    private func bestPoseBreakdown(_ best: DockingResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Best Pose Breakdown")
            HStack(spacing: 8) {
                scoreItem("vdW", best.vdwEnergy)
                scoreItem("Elec", best.elecEnergy)
                scoreItem("H-bond", best.hbondEnergy)
                scoreItem("Torsion", best.torsionPenalty)
            }
            PanelSecondaryButton(title: "View Best Pose",
                                 icon: "eye",
                                 help: "Display the top-ranked docking pose in the 3D viewport") {
                viewModel.showDockingPose(at: 0)
            }
        }
    }

    @ViewBuilder
    private var searchSummary: some View {
        VStack(alignment: .leading, spacing: 4) {
            PanelSubheader(title: "Search Summary")
            HStack(spacing: 12) {
                statMini("Generations", "\(viewModel.docking.dockingTotalGenerations)")
                statMini("Population", "\(viewModel.docking.dockingConfig.populationSize)")
                if let maxGen = viewModel.docking.dockingResults.map(\.generation).max() {
                    statMini("Last Improv.", "Gen \(maxGen)")
                }
            }
        }
    }

    @ViewBuilder
    private func scoreItem(_ label: String, _ value: Float) -> some View {
        VStack(spacing: 1) {
            Text(String(format: "%.1f", value))
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(value < 0 ? .green : .red)
            Text(label)
                .font(PanelStyle.captionFont)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func statMini(_ label: String, _ value: String) -> some View {
        HStack(spacing: 3) {
            Text("\(label):")
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
            Text(value)
                .font(PanelStyle.monoSmall.weight(.medium))
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return "\(m)m \(s)s"
    }

    // MARK: - Helpers (presets, ordinals)

    private func applySlabPreset(_ preset: String) {
        guard let pocket = viewModel.docking.selectedPocket else { return }
        let pocketRadius = max(pocket.size.x, max(pocket.size.y, pocket.size.z))
        let thickness: Float
        switch preset {
        case "Tight":  thickness = pocketRadius * 1.5
        case "Medium": thickness = pocketRadius * 2.5
        default:       thickness = pocketRadius * 4.0
        }
        viewModel.workspace.slabThickness = thickness
        viewModel.workspace.slabOffset = 0
        viewModel.renderer?.slabCenter = pocket.center
        viewModel.renderer?.slabHalfThickness = thickness / 2.0
        viewModel.renderer?.slabOffset = 0
        viewModel.focusOnPocket(pocket)
    }

    private func applyPreset(_ preset: String) {
        viewModel.docking.dockingConfig.autoMode = false
        switch preset {
        case "Auto":
            viewModel.docking.dockingConfig.autoMode = true
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

    private func presetHelp(_ preset: String) -> String {
        switch preset {
        case "Auto": return "Adapts parameters to protein/ligand complexity at launch"
        case "Fast": return "Quick exploration: 50 × 30 × 3"
        case "Standard": return "Balanced: 150 × 80 × 5"
        case "Thorough": return "Deep search: 300 × 200 × 10"
        default: return ""
        }
    }

    private func ordinalSuffix(_ n: Int) -> String {
        let mod100 = n % 100
        if (11...13).contains(mod100) { return "th" }
        switch n % 10 {
        case 1: return "st"
        case 2: return "nd"
        case 3: return "rd"
        default: return "th"
        }
    }

    private var totalEvaluationsString: String {
        let cfg = viewModel.docking.dockingConfig
        let f = NumberFormatter()
        f.numberStyle = .decimal
        let total: Int
        switch cfg.searchMethod {
        case .diffusionGuided:
            total = cfg.populationSize * max(cfg.diffusion.numDenoisingSteps, 1)
        case .fragmentBased:
            return "≈ beam × frags"
        default:
            total = cfg.populationSize * cfg.generationsPerRun * cfg.numRuns
        }
        return f.string(from: NSNumber(value: total)) ?? "\(total)"
    }

    // MARK: - Pocket actions

    private func detectPocketsAuto() {
        guard viewModel.molecules.protein != nil else { return }
        let excluded = viewModel.workspace.hiddenChainIDs
        let chainMsg = excluded.isEmpty ? "" : " (excluding chains: \(excluded.sorted().joined(separator: ", ")))"
        viewModel.log.info("Detecting binding pockets\(chainMsg)...", category: .dock)
        viewModel.detectPockets(excludedChainIDs: excluded)
    }

    private func detectFromLigand() {
        guard viewModel.molecules.protein != nil, viewModel.molecules.ligand != nil else { return }
        viewModel.log.info("Detecting pocket from ligand position...", category: .dock)
        viewModel.detectLigandGuidedPocket()
        if let pocket = viewModel.docking.selectedPocket {
            syncGridFromPocket(pocket)
            applyGridBoxFromSliders()
        }
    }

    private func pocketFromSelection() {
        guard viewModel.molecules.protein != nil,
              !viewModel.workspace.selectedResidueIndices.isEmpty else { return }
        viewModel.pocketFromSelection()
        if let pocket = viewModel.docking.selectedPocket {
            syncGridFromPocket(pocket)
            applyGridBoxFromSliders()
        }
    }

    private func removePocket(_ pocket: BindingPocket) {
        viewModel.docking.detectedPockets.removeAll { $0.id == pocket.id }
        if viewModel.docking.selectedPocket?.id == pocket.id {
            viewModel.docking.selectedPocket = nil
            viewModel.renderer?.clearGridBox()
            viewModel.renderer?.setNeedsRedraw()
        }
        ActivityLog.shared.info("Removed pocket #\(pocket.id)", category: .dock)
    }

    // MARK: - Grid box actions

    private func initializeGridAtProteinCenter(_ prot: Molecule) {
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
        viewModel.updateGridBoxVisualization(center: viewModel.docking.gridCenter,
                                             halfSize: viewModel.docking.gridHalfSize)
        if var pocket = viewModel.docking.selectedPocket {
            pocket.center = viewModel.docking.gridCenter
            pocket.size = viewModel.docking.gridHalfSize
            viewModel.docking.selectedPocket = pocket
        }
    }

    private func gridCenterBinding(_ keyPath: WritableKeyPath<SIMD3<Float>, Float>) -> Binding<Float> {
        Binding(
            get: { viewModel.docking.gridCenter[keyPath: keyPath] },
            set: { viewModel.docking.gridCenter[keyPath: keyPath] = $0 }
        )
    }

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
        viewModel.docking.gridCenter = center
        viewModel.docking.gridHalfSize = SIMD3<Float>(repeating: maxDist + margin)
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
        for pos in positions {
            let d = abs(pos - center)
            maxDist = max(maxDist, max(d.x, max(d.y, d.z)))
        }
        viewModel.docking.gridCenter = center
        viewModel.docking.gridHalfSize = SIMD3<Float>(repeating: maxDist + 4.0)
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

// MARK: - Scoring interactions badge row

private struct ScoringInteractionsBadgeRow: View {
    let method: ScoringMethod

    private var types: [MolecularInteraction.InteractionType] {
        let allowed = method.accountedInteractionTypes
        return MolecularInteraction.InteractionType.allCases.filter { allowed.contains($0) }
    }

    var body: some View {
        if types.isEmpty {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 4) {
                PanelSubheader(title: "Accounted interactions")
                FlowLayout(spacing: 4) {
                    ForEach(types, id: \.self) { t in
                        HStack(spacing: 3) {
                            Circle()
                                .fill(Color(red: Double(t.color.x),
                                            green: Double(t.color.y),
                                            blue: Double(t.color.z))
                                        .opacity(Double(t.color.w)))
                                .frame(width: 6, height: 6)
                            Text(t.label)
                                .font(.system(size: 10))
                        }
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.12)))
                    }
                }
            }
        }
    }
}

// MARK: - Flow layout for chips / badges

struct FlowLayout: Layout {
    var spacing: CGFloat = 4
    var lineSpacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        var widestLine: CGFloat = 0
        for sv in subviews {
            let sz = sv.sizeThatFits(.unspecified)
            if x > 0 && x + sz.width > maxWidth {
                widestLine = max(widestLine, x - spacing)
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            x += sz.width + spacing
            lineHeight = max(lineHeight, sz.height)
        }
        widestLine = max(widestLine, x - spacing)
        return CGSize(width: maxWidth.isFinite ? maxWidth : widestLine,
                      height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize,
                       subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        for sv in subviews {
            let sz = sv.sizeThatFits(.unspecified)
            if x > 0 && x + sz.width > maxWidth {
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            sv.place(at: CGPoint(x: bounds.minX + x, y: bounds.minY + y),
                     proposal: ProposedViewSize(sz))
            x += sz.width + spacing
            lineHeight = max(lineHeight, sz.height)
        }
    }
}
