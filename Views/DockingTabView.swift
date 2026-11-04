// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Docking step (step 5)

/// Inputs → Search → Constraints → Flexibility → Refinement, with the run summary,
/// Run / Stop controls pinned in the step footer. Live progress lives in the
/// toolbar activity view and the canvas HUD; results live in the Results step.
struct DockingTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    @AppStorage("ui.confirmBeforeDocking") private var confirmBeforeDocking = true
    @AppStorage("ui.docking.advancedExpanded") private var advancedExpanded = false
    @AppStorage("ui.docking.flexibilityExpanded") private var flexibilityExpanded = false
    @AppStorage("ui.docking.refinementExpanded") private var refinementExpanded = false
    @State private var showPreDockSheet = false

    private var site: DockingSite { DockingSite(viewModel: viewModel) }

    /// Available scoring methods: Vina, Drusina, DruseAF (Metal), and PIGNet2 if weights present.
    private var scoringMethodsAvailable: [ScoringMethod] {
        var methods: [ScoringMethod] = [.vina, .drusina, .druseAffinity]
        if Bundle.main.url(forResource: "PIGNet2", withExtension: "weights") != nil {
            methods.append(.pignet2)
        }
        return methods
    }

    private var isRunning: Bool { viewModel.docking.isDocking || viewModel.docking.isBatchDocking }
    private var isBatchQueued: Bool { viewModel.docking.batchQueue.count > 1 }

    var body: some View {
        @Bindable var vm = viewModel
        DSStepScaffold {
            if viewModel.molecules.protein == nil {
                DSEmptyState(
                    "No Protein Loaded",
                    systemImage: SidebarTab.dock.icon,
                    description: "Load and prepare a protein, then choose ligands and a binding site to configure a docking run."
                ) {
                    Button {
                        site.goTo(.search)
                    } label: {
                        Label("Go to Search", systemImage: SidebarTab.search.icon)
                    }
                    .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                }
                .padding(.top, DS.Space.xxl)
            } else {
                if !isRunning && !viewModel.docking.dockingResults.isEmpty {
                    completionBanner
                }
                inputsCard
                searchCard
                constraintsCard
                if viewModel.docking.selectedPocket != nil {
                    flexibilityCard
                }
                refinementCard
            }
        } footer: {
            if isRunning {
                runningFooter
            } else {
                runFooter
            }
        }
        .sheet(isPresented: $showPreDockSheet) {
            PreDockSheet().environment(viewModel)
        }
        .sheet(isPresented: $vm.docking.showPharmacophoreEditor) {
            PharmacophoreEditorView().environment(viewModel)
        }
        .dockingSiteSync()
    }

    // MARK: - Completion banner

    @ViewBuilder
    private var completionBanner: some View {
        let results = viewModel.docking.dockingResults
        let method = viewModel.docking.scoringMethod
        let clusters = Set(results.map(\.clusterID)).count
        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(DS.Color.success)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 0) {
                    Text("Docking complete")
                        .font(DS.Font.label.weight(.semibold))
                    Text("\(results.count) poses · \(clusters) clusters")
                        .font(DS.Font.secondary)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: DS.Space.xs)
                if let best = results.first {
                    DSScoreBadge(best.displayScore(method: method), method: method)
                        .help("Best pose score")
                }
            }
            FlowLayout(spacing: DS.Space.xs) {
                if viewModel.docking.dockingDuration > 0 {
                    DSTag(formatDuration(viewModel.docking.dockingDuration), systemImage: "clock")
                        .help("Wall time")
                }
                poseBustersTag(results)
            }
            HStack {
                Spacer(minLength: 0)
                DSNextStepButton(.results)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: DS.Space.l)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private func poseBustersTag(_ results: [DockingResult]) -> some View {
        if results.contains(where: { $0.validity != nil }) {
            let validCount = results.filter { $0.validity?.passed == true }.count
            let allPassed = validCount == results.count
            let failures = results.first(where: { $0.validity?.passed == false })?.validity?.failures ?? []
            DSTag("PoseBusters \(validCount)/\(results.count)",
                  tone: allPassed ? .success : .warning,
                  systemImage: allPassed ? "checkmark.shield" : "exclamationmark.shield")
                .help(failures.isEmpty ? "All poses pass PoseBusters-style validity checks"
                                       : "First failure: " + failures.prefix(2).joined(separator: ", "))
        }
    }

    // MARK: - Inputs card

    private var inputsCard: some View {
        PanelCard("Inputs", icon: "tray.2") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                proteinInputRow
                Divider()
                ligandInputRow
                Divider()
                siteInputRow
            }
        }
    }

    private var proteinInputRow: some View {
        let prot = viewModel.molecules.protein
        return DockInputRow(
            kind: "Protein",
            tint: DS.Color.protein,
            title: prot?.name,
            detail: prot.map { prot in
                // Amino acids only, matching the Sequence overview and Inspector (no waters or caps).
                let aminoAcids = prot.residues.filter(\.isStandard)
                let chains = Set(aminoAcids.map(\.chainID)).count
                return "\(aminoAcids.count) residues · \(chains) chain\(chains == 1 ? "" : "s")"
            },
            missingText: "Load a protein",
            step: .search
        )
    }

    private var ligandInputRow: some View {
        let queue = viewModel.docking.batchQueue
        let lig = viewModel.molecules.ligand
        let title: String?
        let detail: String?
        if isBatchQueued {
            let prepared = queue.filter(\.isPrepared).count
            let mws = queue.compactMap { $0.descriptors?.molecularWeight }
            var parts = ["\(prepared)/\(queue.count) prepared"]
            if let lo = mws.min(), let hi = mws.max() {
                parts.append(String(format: "MW %.0f–%.0f Da", lo, hi))
            }
            title = "\(queue.count) ligands queued"
            detail = parts.joined(separator: " · ")
        } else if let lig {
            title = lig.name
            detail = String(format: "%d heavy atoms · MW %.1f Da", lig.heavyAtomCount, RDKitBridge.parentMolecularWeight(of: lig))
        } else {
            title = nil
            detail = nil
        }
        return DockInputRow(
            kind: isBatchQueued ? "Ligands" : "Ligand",
            tint: DS.Color.ligand,
            title: title,
            detail: detail,
            missingText: "Choose a ligand",
            step: .ligands
        ) {
            Button("Manage…") { openWindow(id: "ligand-database") }
                .buttonStyle(.borderless)
                .font(DS.Font.secondary.weight(.medium))
                .help("Open Ligand Database (⌘L)")
                .accessibilityIdentifier(AccessibilityID.dockOpenLigandDB)
            if isBatchQueued {
                clearButton(help: "Clear batch queue") {
                    viewModel.docking.batchQueue = []
                    viewModel.clearLigand()
                }
                .accessibilityIdentifier(AccessibilityID.dockClearBatch)
            } else if lig != nil {
                clearButton(help: "Remove active ligand") { viewModel.removeLigandFromView() }
                    .accessibilityIdentifier(AccessibilityID.dockRemoveLigand)
            }
        }
    }

    private var siteInputRow: some View {
        let half = viewModel.docking.gridHalfSize
        let detail: String? = {
            guard site.hasGrid else { return nil }
            if let pocket = viewModel.docking.selectedPocket, !pocket.residueIndices.isEmpty {
                return String(format: "%.0f Å³ · %d residues", pocket.volume, pocket.residueIndices.count)
            }
            return DockingSite.boxDimensions(half)
        }()
        return DockInputRow(
            kind: "Binding Site",
            tint: DS.Color.pocket,
            title: site.siteTitle,
            detail: detail,
            missingText: "Define a binding site",
            step: .bindingSite
        ) {
            if site.hasGrid {
                Button("Edit") { site.goTo(.bindingSite) }
                    .buttonStyle(.borderless)
                    .font(DS.Font.secondary.weight(.medium))
                    .help("Adjust the pocket and grid box in the Binding Site step")
            }
        }
    }

    private func clearButton(help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: "xmark.circle.fill")
                .font(DS.Font.label)
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.borderless)
        .help(help)
        .accessibilityLabel(help)
    }

    // MARK: - Search card

    private var searchCard: some View {
        PanelCard("Search", icon: "magnifyingglass") {
            VStack(alignment: .leading, spacing: DS.Space.xl) {
                presetSection
                searchMethodSection
                scoringFunctionSection
                searchBudgetSection
                Divider()
                PanelDisclosure("Advanced", icon: "slider.horizontal.3", isExpanded: $advancedExpanded) {
                    VStack(alignment: .leading, spacing: DS.Space.xl) {
                        searchBehaviorSection
                        if viewModel.docking.dockingConfig.enableFlexibility {
                            torsionSearchSection
                        }
                        searchMethodOptionsSection
                        explorationPhaseSection
                    }
                }
            }
        }
    }

    private var presetSection: some View {
        let config = viewModel.docking.dockingConfig
        let active = DockingPreset.allCases.first { $0.isActive(in: config) }
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(spacing: DS.Space.s) {
                PanelSubheader(title: "Preset")
                Spacer(minLength: DS.Space.xs)
                if active == nil {
                    DSTag("Custom", tone: .accent)
                        .help("The budget no longer matches a preset. Pick a preset to reset it.")
                }
            }
            DockPresetPicker(active: active) { preset in
                preset.apply(to: &viewModel.docking.dockingConfig)
            }
            if config.autoMode {
                PanelHint(text: "Parameters adapt to protein size, pocket shape, and ligand flexibility",
                          icon: "wand.and.stars",
                          color: DS.Color.accent)
            } else if let active {
                descriptionText(active.help)
            } else {
                descriptionText("Custom budget: \(budgetSummary) (population × generations × runs)")
            }
        }
    }

    private var searchMethodSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelSubheader(title: "Method")
            // Stable methods fill a 2×2 grid; experimental ones get a full-width row so the BETA tag never truncates the label.
            PanelChoiceGrid(columns: 2) {
                ForEach(SearchMethod.allCases.filter { !$0.isExperimental }, id: \.self) { method in
                    searchMethodButton(method)
                }
            }
            ForEach(SearchMethod.allCases.filter(\.isExperimental), id: \.self) { method in
                searchMethodButton(method)
            }
            descriptionText(viewModel.docking.dockingConfig.searchMethod.description)
        }
    }

    private func searchMethodButton(_ method: SearchMethod) -> some View {
        PanelChoiceButton(
            title: method.shortLabel,
            icon: method.icon,
            isSelected: viewModel.docking.dockingConfig.searchMethod == method,
            help: method.isExperimental ? "Experimental. " + method.description : method.description
        ) {
            viewModel.docking.dockingConfig.searchMethod = method
            if method == .diffusionGuided && viewModel.docking.scoringMethod != .druseAffinity {
                viewModel.docking.scoringMethod = .druseAffinity
            }
        }
        .overlay(alignment: .trailing) {
            if method.isExperimental {
                DSTag("BETA", tone: .warning)
                    .padding(.trailing, DS.Space.m)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
    }

    private var scoringFunctionSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelSubheader(title: "Scoring")
            PanelChoiceGrid(columns: 2) {
                ForEach(scoringMethodsAvailable, id: \.self) { method in
                    PanelChoiceButton(
                        title: method.shortLabel,
                        icon: method.icon,
                        isSelected: viewModel.docking.scoringMethod == method,
                        help: method.description
                    ) { viewModel.docking.scoringMethod = method }
                }
            }
            VStack(alignment: .leading, spacing: DS.Space.s) {
                descriptionText(viewModel.docking.scoringMethod.description)
                ScoringInteractionsBadgeRow(method: viewModel.docking.scoringMethod)
            }
        }
    }

    private var searchBudgetSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
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
            HStack(alignment: .firstTextBaseline) {
                Text("Total evaluations")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(totalEvaluationsString)
                    .font(DS.Font.mono.weight(.semibold))
                    .monospacedDigit()
            }
            .padding(.top, DS.Space.xxs)
        }
    }

    private var searchBehaviorSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
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
                Picker("Grid Spacing", selection: Binding(
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
                .fixedSize()
            }
        }
    }

    private var torsionSearchSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            PanelSubheader(title: "Torsion Search", icon: "arrow.triangle.2.circlepath")
            PanelChoiceGrid(columns: 2) {
                ForEach(TorsionSearchPreset.allCases, id: \.self) { preset in
                    PanelChoiceButton(
                        title: preset.rawValue,
                        isSelected: viewModel.docking.dockingConfig.torsionSearchPreset == preset,
                        help: preset.help
                    ) {
                        viewModel.docking.dockingConfig.applyTorsionSearchPreset(preset)
                    }
                }
            }
            torsionSlider("Exact", \.torsionExactFraction, range: 0...1, step: 0.05, clamp: 0...1) {
                String(format: "%.0f%%", $0 * 100)
            }
            torsionSlider("Local", \.torsionLocalFraction, range: 0...1, step: 0.05, clamp: 0...1) {
                String(format: "%.0f%%", $0 * 100)
            }
            torsionSlider("Local Width", \.torsionLocalAmplitude, range: 0...1.5, step: 0.05, clamp: 0...Float.greatestFiniteMagnitude) {
                String(format: "%.2f rad", $0)
            }
            torsionSlider("Reset", \.torsionRandomResetProbability, range: 0...0.5, step: 0.01, clamp: 0...1) {
                String(format: "%.0f%%", $0 * 100)
            }
            torsionSlider("Step Scale", \.torsionPerturbationScale, range: 0...1.5, step: 0.05, clamp: 0...Float.greatestFiniteMagnitude) {
                String(format: "%.2fx", $0)
            }
        }
    }

    /// Torsion slider: editing any value switches the torsion preset to Manual.
    private func torsionSlider(
        _ label: String,
        _ keyPath: WritableKeyPath<DockingConfig, Float>,
        range: ClosedRange<Float>,
        step: Float,
        clamp: ClosedRange<Float>,
        format: @escaping (Float) -> String
    ) -> some View {
        PanelSliderRow(
            label: label,
            value: Binding(
                get: { viewModel.docking.dockingConfig[keyPath: keyPath] },
                set: {
                    viewModel.docking.dockingConfig.torsionSearchPreset = .manual
                    viewModel.docking.dockingConfig[keyPath: keyPath] = min(max($0, clamp.lowerBound), clamp.upperBound)
                }
            ),
            range: range, step: step,
            format: format
        )
    }

    private var explorationPhaseSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            PanelSubheader(title: "Exploration Phase", icon: "sparkles")
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
        switch viewModel.docking.dockingConfig.searchMethod {
        case .fragmentBased:     fragmentMethodOptions
        case .parallelTempering: parallelTemperingOptions
        case .diffusionGuided:   diffusionMethodOptions
        default:                 EmptyView()
        }
    }

    private var fragmentMethodOptions: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
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
            PanelLabeledRow("Scaffold", help: "SMARTS pattern that fragment growth must preserve (leave empty for automatic)") {
                TextField("Auto", text: Binding(
                    get: { viewModel.docking.dockingConfig.fragment.scaffoldSMARTS ?? "" },
                    set: { newValue in
                        let trimmed = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
                        viewModel.docking.dockingConfig.fragment.scaffoldSMARTS = trimmed.isEmpty ? nil : trimmed
                        viewModel.docking.dockingConfig.fragment.scaffoldMode = trimmed.isEmpty ? .auto : .manual
                    }
                ))
                .textFieldStyle(.roundedBorder)
                .font(DS.Font.monoSmall)
                .controlSize(.small)
            }
            if viewModel.docking.dockingConfig.fragment.scaffoldSMARTS != nil {
                PanelHint(text: "Scaffold enforced: poses must contain this substructure.",
                          icon: "pencil.and.outline", color: DS.Color.warning)
            }
        }
    }

    private var parallelTemperingOptions: some View {
        let rex = viewModel.docking.dockingConfig.replicaExchange
        return VStack(alignment: .leading, spacing: DS.Space.m) {
            PanelSubheader(title: "Replica Exchange", icon: "thermometer")
            PanelLabeledRow("Replicas") {
                Picker("Replicas", selection: Binding(
                    get: { viewModel.docking.dockingConfig.replicaExchange.numReplicas },
                    set: { viewModel.docking.dockingConfig.replicaExchange.numReplicas = $0 }
                )) {
                    ForEach([4, 8, 12, 16], id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
            }
            HStack(alignment: .firstTextBaseline) {
                Text("Temperature range")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f–%.1f kcal/mol", rex.minTemperature, rex.maxTemperature))
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var diffusionMethodOptions: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
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
                Picker("Schedule", selection: Binding(
                    get: { viewModel.docking.dockingConfig.diffusion.noiseSchedule },
                    set: { viewModel.docking.dockingConfig.diffusion.noiseSchedule = $0 }
                )) {
                    ForEach(DiffusionDockingConfig.NoiseSchedule.allCases, id: \.self) { sched in
                        Text(sched.rawValue).tag(sched)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
            }
            PanelHint(
                text: "Experimental. DruseAF was trained for affinity prediction, not denoising; the gradient is heuristic. Use GA or Fragment for production work.",
                icon: "exclamationmark.triangle.fill",
                color: DS.Color.warning
            )
        }
    }

    // MARK: - Constraints card

    private var constraintsCard: some View {
        let constraints = viewModel.docking.pharmacophoreConstraints
        return PanelCard(
            "Constraints",
            icon: "circle.hexagongrid.circle",
            accessory: {
                if !constraints.isEmpty {
                    HStack(spacing: DS.Space.s) {
                        DSTag("\(constraints.count)", tone: .accent, monospaced: true)
                        Button("Clear") { viewModel.docking.pharmacophoreConstraints.removeAll() }
                            .buttonStyle(.borderless)
                            .font(DS.Font.secondary.weight(.medium))
                            .foregroundStyle(DS.Color.danger)
                            .help("Remove all pharmacophore constraints")
                    }
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if constraints.isEmpty {
                    PanelHint(text: "Optional. Right-click an atom or residue in the viewport to add one, or build a pharmacophore query.",
                              icon: "info.circle")
                } else {
                    VStack(spacing: DS.Space.xs) {
                        ForEach(Array(constraints.enumerated()), id: \.element.id) { idx, constraint in
                            constraintRow(idx: idx, constraint: constraint)
                        }
                    }
                }
                PanelSecondaryButton(title: "Pharmacophore Editor…",
                                     icon: "circle.hexagongrid.circle",
                                     help: "Create constraints from a reference ligand's pharmacophoric features") {
                    viewModel.docking.showPharmacophoreEditor = true
                }
            }
        }
    }

    private func constraintRow(idx: Int, constraint: PharmacophoreConstraintDef) -> some View {
        HStack(spacing: DS.Space.s) {
            Image(systemName: constraint.interactionType.icon)
                .font(DS.Font.label)
                .foregroundStyle(constraint.interactionType.dsColor)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 0) {
                Text(constraint.targetLabel)
                    .font(DS.Font.monoSmall)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(constraint.interactionType.rawValue)
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: DS.Space.xs)
            DSTag(constraint.strength.isHard ? "Hard" : "Soft",
                  tone: constraint.strength.isHard ? .danger : .warning)
            Button {
                viewModel.docking.pharmacophoreConstraints.remove(at: idx)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(DS.Font.label)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Remove constraint")
        }
        .padding(.vertical, DS.Space.xxs)
    }

    // MARK: - Flexibility card

    private var flexibilityCard: some View {
        @Bindable var vm = viewModel
        let config = viewModel.docking.flexibleResidueConfig
        let summary: String = {
            if config.autoFlex { return "Soft" }
            if config.flexibleResidueIndices.isEmpty { return "Rigid" }
            let n = config.flexibleResidueIndices.count
            return "\(n) residue\(n == 1 ? "" : "s")"
        }()
        return DockCollapsibleCard(
            "Receptor Flexibility",
            icon: "figure.flexibility",
            isExpanded: $flexibilityExpanded,
            accessory: {
                if flexibilityExpanded && !config.flexibleResidueIndices.isEmpty && !config.autoFlex {
                    Button("Clear") {
                        viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll()
                    }
                    .buttonStyle(.borderless)
                    .font(DS.Font.secondary.weight(.medium))
                    .foregroundStyle(DS.Color.danger)
                    .help("Remove all flexible residues (use rigid receptor)")
                } else {
                    DSTag(summary, tone: summary == "Rigid" ? .neutral : .pocket)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
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

                if config.autoFlex {
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
                PanelHint(text: "No rotatable residues lining this pocket", color: DS.Color.warning)
            } else {
                flexResidueSummary(autoIndices, protein: prot, suffix: " — soft weight")
            }
        }
    }

    @ViewBuilder
    private var manualFlexResidues: some View {
        let flexIndices = viewModel.docking.flexibleResidueConfig.flexibleResidueIndices
        if flexIndices.isEmpty {
            PanelHint(text: "Select residues in the sequence panel or 3D viewport for full induced-fit docking.")
        } else if let prot = viewModel.molecules.protein {
            flexResidueSummary(flexIndices, protein: prot, suffix: "")
            if flexIndices.count >= FlexibleResidueConfig.maxFlexibleResidues {
                PanelHint(text: "Maximum \(FlexibleResidueConfig.maxFlexibleResidues) flexible residues reached",
                          color: DS.Color.warning)
            }
        }
    }

    @ViewBuilder
    private func flexResidueSummary(_ indices: [Int], protein prot: Molecule, suffix: String) -> some View {
        let residueNames: [Int: String] = Dictionary(
            indices.compactMap { seq in
                prot.atoms.first(where: { $0.residueSeq == seq }).map { (seq, $0.residueName) }
            },
            uniquingKeysWith: { first, _ in first }
        )
        let totalChi = indices.compactMap { seq in
            residueNames[seq].flatMap { RotamerLibrary.rotamers(for: $0)?.chiAngles.count }
        }.reduce(0, +)
        FlowLayout(spacing: DS.Space.xs) {
            ForEach(indices, id: \.self) { seq in
                DSTag("\(residueNames[seq] ?? "?")\(seq)", tone: .pocket, monospaced: true)
            }
        }
        Text("\(indices.count) residue(s), \(totalChi) chi angle(s)\(suffix)")
            .font(DS.Font.secondary)
            .monospacedDigit()
            .foregroundStyle(.secondary)
    }

    // MARK: - Refinement card

    private var refinementCard: some View {
        let afOn = viewModel.docking.dockingConfig.useAFv4Rescore
        let xtbOn = viewModel.docking.dockingConfig.gfn2Refinement.enabled
        let summary: String = {
            if afOn && xtbOn { return "DruseAF + xTB" }
            if afOn { return "DruseAF" }
            if xtbOn { return "GFN2-xTB" }
            return "Off"
        }()
        return DockCollapsibleCard(
            "Post-docking Refinement",
            icon: "wand.and.rays",
            isExpanded: $refinementExpanded,
            accessory: { DSTag(summary, tone: afOn || xtbOn ? .accent : .neutral) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
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
                if xtbOn {
                    gfn2Options
                }
            }
        }
    }

    private var gfn2Options: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelLabeledRow("Solvation",
                            help: "Implicit solvation: ALPB (recommended) or GBSA for aqueous environment") {
                Picker("Solvation", selection: Binding(
                    get: { viewModel.docking.dockingConfig.gfn2Refinement.solvation },
                    set: { viewModel.docking.dockingConfig.gfn2Refinement.solvation = $0 }
                )) {
                    ForEach(GFN2SolvationMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .fixedSize()
            }
            PanelLabeledRow("Opt Level",
                            help: "Crude (~5 ms/pose), Normal (~15 ms), Tight (~30 ms)") {
                Picker("Opt Level", selection: Binding(
                    get: { viewModel.docking.dockingConfig.gfn2Refinement.optLevel },
                    set: { viewModel.docking.dockingConfig.gfn2Refinement.optLevel = $0 }
                )) {
                    ForEach(GFN2OptLevel.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .fixedSize()
            }
            PanelLabeledRow("Top poses",
                            help: "Number of top-ranked poses to optimize with GFN2-xTB") {
                PanelNumberField(
                    value: Binding(
                        get: { viewModel.docking.dockingConfig.gfn2Refinement.topPosesToRefine },
                        set: { viewModel.docking.dockingConfig.gfn2Refinement.topPosesToRefine = max(1, $0) }
                    )
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
        .padding(.leading, DS.Space.xl)
    }

    // MARK: - Footer: run summary + Run

    private var canDock: Bool {
        site.hasGrid
            && viewModel.molecules.ligand != nil
            && viewModel.molecules.protein != nil
            && !viewModel.docking.isDocking
    }

    @ViewBuilder
    private var runFooter: some View {
        if viewModel.molecules.protein != nil {
            runSummary
        }
        if let warning = vramWarning() {
            PanelHint(text: warning, icon: "exclamationmark.triangle.fill", color: DS.Color.warning)
        }
        PanelRunButton(
            title: isBatchQueued
                ? "Dock \(viewModel.docking.batchQueue.count) Ligands"
                : (confirmBeforeDocking ? "Run Docking…" : "Run Docking"),
            icon: "play.fill",
            isDisabled: !canDock
        ) { startDocking() }
        .help(confirmBeforeDocking && !isBatchQueued
              ? "Review the run, then start docking"
              : "Start docking now")
        .accessibilityIdentifier(AccessibilityID.dockStartButton)
    }

    /// One-line recap of what will run; the Inputs card above holds the details.
    private var runSummary: some View {
        let config = viewModel.docking.dockingConfig
        let ligand: String = isBatchQueued
            ? "\(viewModel.docking.batchQueue.count) ligands"
            : (viewModel.molecules.ligand?.name ?? "No ligand")
        let siteName = site.siteTitle ?? "No site"
        let search = "\(config.searchMethod.shortLabel)·\(viewModel.docking.scoringMethod.shortLabel) \(budgetSummary)"
        let line = [ligand, siteName, search].joined(separator: " · ")
        return Text(line)
            .font(DS.Font.secondary)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.middle)
            .frame(maxWidth: .infinity, alignment: .leading)
            .help("\(line)\nTotal evaluations: \(totalEvaluationsString)")
            .accessibilityLabel("Run summary: \(line)")
    }

    private var budgetSummary: String {
        let c = viewModel.docking.dockingConfig
        switch c.searchMethod {
        case .fragmentBased:   return "beam \(c.fragment.beamWidth)"
        case .diffusionGuided: return "\(c.diffusion.numDenoisingSteps) steps"
        default:               return "\(c.populationSize)×\(c.generationsPerRun)×\(c.numRuns)"
        }
    }

    private func startDocking() {
        site.ensurePocketFromGrid()
        if isBatchQueued {
            viewModel.dockEntries(viewModel.docking.batchQueue)
            viewModel.docking.batchQueue = []
        } else if confirmBeforeDocking {
            showPreDockSheet = true
        } else {
            viewModel.runDocking()
        }
    }

    private func vramWarning() -> String? {
        guard let engine = viewModel.docking.dockingEngine,
              let pocket = viewModel.docking.selectedPocket else { return nil }
        let spacing = viewModel.docking.dockingConfig.gridSpacing
        let estimate = engine.estimateVRAMUsage(
            gridDims: SIMD3(
                UInt32(ceil((pocket.size.x + 7) * 2 / spacing)) + 1,
                UInt32(ceil((pocket.size.y + 7) * 2 / spacing)) + 1,
                UInt32(ceil((pocket.size.z + 7) * 2 / spacing)) + 1
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

    // MARK: - Footer: running

    /// Single-ligand progress is shown once, in the canvas HUD (and the toolbar capsule);
    /// the footer only swaps Run for Stop. Batch runs keep their ligand-level bar here because
    /// the HUD tracks the ligand currently being docked.
    @ViewBuilder
    private var runningFooter: some View {
        let isBatch = viewModel.docking.isBatchDocking
        if isBatch {
            let p = viewModel.docking.batchProgress
            let label = "Ligand \(min(p.current + 1, max(p.total, 1))) of \(p.total)"
            HStack(spacing: DS.Space.s) {
                ProgressView()
                    .controlSize(.mini)
                Text("Batch docking…")
                    .font(DS.Font.label.weight(.medium))
                Spacer(minLength: DS.Space.xs)
                Text(label)
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: Double(min(p.current, p.total)), total: Double(max(p.total, 1)))
                .progressViewStyle(.linear)
                .accessibilityLabel(label)
        }
        PanelRunButton(
            title: isBatch ? "Stop All" : "Stop Docking",
            icon: "stop.fill",
            color: DS.Color.danger
        ) {
            if isBatch {
                viewModel.cancelBatchDocking()
            } else {
                viewModel.stopDocking()
            }
        }
        .help("Cancel the active docking run and keep the best pose found so far")
        .accessibilityIdentifier(AccessibilityID.dockCancelButton)
    }

    // MARK: - Helpers

    private func descriptionText(_ text: String) -> some View {
        Text(text)
            .font(DS.Font.secondary)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        return "\(Int(seconds) / 60)m \(Int(seconds) % 60)s"
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
        let total: Int
        switch cfg.searchMethod {
        case .diffusionGuided:
            total = cfg.populationSize * max(cfg.diffusion.numDenoisingSteps, 1)
        case .fragmentBased:
            return "≈ beam × frags"
        default:
            total = cfg.populationSize * cfg.generationsPerRun * cfg.numRuns
        }
        return total.formatted(.number)
    }
}

// MARK: - Inputs row

/// One Inputs row: eyebrow + entity dot + name + detail, or a requirement with a link to its step.
private struct DockInputRow<Accessory: View>: View {
    @Environment(AppViewModel.self) private var viewModel

    let kind: String
    let tint: Color
    let title: String?
    let detail: String?
    let missingText: String
    let step: SidebarTab
    @ViewBuilder let accessory: () -> Accessory

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            Text(kind.uppercased())
                .font(DS.Font.eyebrow)
                .tracking(0.6)
                .foregroundStyle(.tertiary)
            if let title {
                HStack(spacing: DS.Space.s) {
                    Circle()
                        .fill(tint)
                        .frame(width: 8, height: 8)
                    Text(title)
                        .font(DS.Font.label.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: DS.Space.xs)
                    accessory()
                }
                if let detail {
                    Text(detail)
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .padding(.leading, DS.Space.l + DS.Space.xxs)
                }
            } else {
                HStack(spacing: DS.Space.m) {
                    PanelRequirement(text: missingText)
                    Spacer(minLength: DS.Space.xs)
                    goToStepButton
                    accessory()
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var goToStepButton: some View {
        Button {
            step.request(in: viewModel)
        } label: {
            HStack(spacing: DS.Space.xxs) {
                Text(step.rawValue)
                Image(systemName: "arrow.right")
            }
            .font(DS.Font.secondary.weight(.medium))
        }
        .buttonStyle(.borderless)
        .help("Go to step: \(step.rawValue)")
    }
}

extension DockInputRow where Accessory == EmptyView {
    init(kind: String, tint: Color, title: String?, detail: String?, missingText: String, step: SidebarTab) {
        self.init(kind: kind, tint: tint, title: title, detail: detail,
                  missingText: missingText, step: step, accessory: { EmptyView() })
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
        if !types.isEmpty {
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                Text("Accounts for")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                FlowLayout(spacing: DS.Space.xs) {
                    ForEach(types, id: \.self) { t in
                        HStack(spacing: DS.Space.xs) {
                            Circle()
                                .fill(color(for: t))
                                .frame(width: 6, height: 6)
                            Text(t.label)
                        }
                        .font(DS.Font.caption)
                        .padding(.horizontal, DS.Space.s)
                        .padding(.vertical, DS.Space.xxs)
                        .background(RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous).fill(DS.Color.controlFill))
                    }
                }
            }
        }
    }

    /// Interaction palette tokens; types without a token keep their renderer colour.
    private func color(for type: MolecularInteraction.InteractionType) -> Color {
        switch type {
        case .hbond:       DS.Color.interactionHBond
        case .hydrophobic: DS.Color.interactionHydrophobic
        case .saltBridge:  DS.Color.interactionSaltBridge
        case .piStack:     DS.Color.interactionPiStack
        case .piCation:    DS.Color.interactionPiCation
        case .halogen:     DS.Color.interactionHalogen
        case .metalCoord:  DS.Color.interactionMetal
        default:
            Color(red: Double(type.color.x), green: Double(type.color.y), blue: Double(type.color.z))
        }
    }
}

// MARK: - Collapsible card

/// PanelCard chrome whose body folds away behind its header; the accessory (usually a
/// state summary tag) stays visible so a collapsed card still says what it will do.
private struct DockCollapsibleCard<Accessory: View, Content: View>: View {
    let title: String
    let icon: String
    @Binding var isExpanded: Bool
    let accessory: Accessory
    let content: Content
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(_ title: String, icon: String, isExpanded: Binding<Bool>,
         @ViewBuilder accessory: () -> Accessory, @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self._isExpanded = isExpanded
        self.accessory = accessory()
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            HStack(spacing: DS.Space.s) {
                Button {
                    withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) { isExpanded.toggle() }
                } label: {
                    HStack(spacing: DS.Space.s) {
                        Image(systemName: icon)
                            .font(.system(size: DS.Size.toolIcon, weight: .medium))
                            .symbolRenderingMode(.hierarchical)
                            .foregroundStyle(.secondary)
                            .frame(width: 16, height: 16)
                        Text(title)
                            .font(DS.Font.sectionTitle)
                            .lineLimit(1)
                        Image(systemName: "chevron.right")
                            .font(DS.Font.caption.weight(.semibold))
                            .foregroundStyle(.tertiary)
                            .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        Spacer(minLength: DS.Space.xs)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isExpanded ? "Collapse \(title)" : "Show \(title) options")
                .accessibilityLabel(title)
                .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
                accessory
            }
            if isExpanded {
                content
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: PanelStyle.cardPadding)
    }
}

// MARK: - Preset picker

/// One-row segmented picker for the search-budget presets. Built from plain buttons (not a
/// `Picker`) so each preset stays an AXButton named after the preset; no segment is selected
/// while the budget is custom.
private struct DockPresetPicker: View {
    let active: DockingPreset?
    let onSelect: (DockingPreset) -> Void

    var body: some View {
        let outer = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        HStack(spacing: DS.Space.xxs) {
            ForEach(DockingPreset.allCases, id: \.self) { preset in
                DockPresetSegment(preset: preset, isSelected: preset == active) { onSelect(preset) }
            }
        }
        .padding(DS.Space.xxs)
        .background(outer.fill(DS.Color.controlFill))
        .overlay(outer.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
    }
}

private struct DockPresetSegment: View {
    let preset: DockingPreset
    let isSelected: Bool
    let action: () -> Void
    @State private var isHovering = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
        Button(action: action) {
            Text(preset.rawValue)
                .font(DS.Font.label.weight(isSelected ? .medium : .regular))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .foregroundStyle(isSelected ? AnyShapeStyle(DS.Color.accent) : AnyShapeStyle(.primary))
                .padding(.horizontal, DS.Space.s)
                .frame(maxWidth: .infinity)
                .frame(height: DS.Size.buttonHeight - 4)
                .background(shape.fill(isSelected ? AnyShapeStyle(DS.Color.accentSoft)
                                                  : AnyShapeStyle(isHovering ? DS.Color.controlFill : Color.clear)))
                .overlay(shape.strokeBorder(isSelected ? DS.Color.accent : Color.clear, lineWidth: 1))
                .contentShape(shape)
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help(preset.help)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
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
