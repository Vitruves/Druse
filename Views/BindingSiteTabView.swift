// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Binding Site step (step 4)

/// Pocket detection, pocket list, pocket view (Z-slab presets) and docking grid box.
struct BindingSiteTabView: View {
    @Environment(AppViewModel.self) private var viewModel

    private var site: DockingSite { DockingSite(viewModel: viewModel) }

    var body: some View {
        DSStepScaffold {
            if viewModel.molecules.protein == nil {
                DSEmptyState(
                    "No Protein Loaded",
                    systemImage: "scope",
                    description: "Load a protein to detect binding pockets and place the docking grid box."
                ) {
                    Button("Go to Search") { site.goTo(.search) }
                }
                .padding(.top, DS.Space.xxl)
            } else {
                detectCard
                if !viewModel.docking.detectedPockets.isEmpty {
                    pocketsCard
                }
                pocketViewCard
                gridBoxCard
            }
        } footer: {
            footer
        }
        .dockingSiteSync()
    }

    // MARK: - Detect

    private var detectCard: some View {
        PanelCard("Detect", icon: "sparkle.magnifyingglass") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                PanelChoiceGrid(columns: 2) {
                    PanelChoiceButton(
                        title: "Auto", icon: "sparkles", isSelected: false,
                        help: "Hybrid pocket detection: ML candidates plus geometric fallback"
                    ) { detectPocketsAuto() }
                    .accessibilityIdentifier(AccessibilityID.dockDetectAuto)

                    PanelChoiceButton(
                        title: "ML", icon: "brain", isSelected: false,
                        isDisabled: !viewModel.pocketDetectorML.isAvailable,
                        help: "ML-based pocket detection (GNN)"
                    ) { viewModel.detectPocketsML() }
                    .accessibilityIdentifier(AccessibilityID.dockDetectML)

                    PanelChoiceButton(
                        title: "From Ligand", icon: "hexagon", isSelected: false,
                        isDisabled: viewModel.molecules.ligand == nil,
                        help: "Define pocket around current ligand"
                    ) { detectFromLigand() }
                    .accessibilityIdentifier(AccessibilityID.dockDetectLigand)

                    PanelChoiceButton(
                        title: "From Selection", icon: "hand.tap", isSelected: false,
                        isDisabled: viewModel.workspace.selectedResidueIndices.isEmpty,
                        help: "Define pocket from selected residues"
                    ) { pocketFromSelection() }
                    .accessibilityIdentifier(AccessibilityID.dockDetectSelection)
                }
                if !viewModel.pocketDetectorML.isAvailable {
                    PanelHint(text: "ML pocket model not loaded — Auto uses geometric detection.",
                              icon: "info.circle")
                }
            }
        }
    }

    // MARK: - Pockets

    private var pocketsCard: some View {
        let pockets = viewModel.docking.detectedPockets
        return PanelCard(
            "Pockets",
            icon: "circle.hexagongrid",
            accessory: { DSTag("\(pockets.count)", tone: .pocket, monospaced: true) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                VStack(spacing: DS.Space.xs) {
                    ForEach(pockets, id: \.id) { pocket in
                        pocketRow(pocket)
                    }
                }
                Divider()
                selectedPocketSummary
            }
        }
    }

    private func pocketRow(_ pocket: BindingPocket) -> some View {
        let isSelected = viewModel.docking.selectedPocket?.id == pocket.id
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        return HStack(alignment: .top, spacing: DS.Space.m) {
            Text("\(pocket.number)")
                .font(DS.Font.caption.weight(.bold))
                .monospacedDigit()
                .foregroundStyle(isSelected ? DS.Color.accentOnFill : DS.Color.pocket)
                .frame(width: 20, height: 20)
                .background(Circle().fill(isSelected ? DS.Color.pocket : DS.Color.pocket.opacity(0.14)))

            VStack(alignment: .leading, spacing: DS.Space.xs) {
                HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                    Text(pocket.displayName)
                        .font(DS.Font.label.weight(.medium))
                    Spacer(minLength: DS.Space.xs)
                    Text(String(format: "%.0f Å³", pocket.volume))
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                FlowLayout(spacing: DS.Space.xs) {
                    DSTag("\(pocket.residueIndices.count) res", monospaced: true)
                    DSTag(String(format: "Buried %.0f%%", pocket.buriedness * 100),
                          tone: pocket.buriedness >= 0.6 ? .pocket : .neutral)
                    // `druggability` is each detector's ranking score (unbounded, not a 0–1
                    // probability), so it is shown neutrally as a rank score.
                    DSTag(String(format: "Score %.1f", pocket.druggability), monospaced: true)
                        .help("Detector ranking score — compare between pockets of one detection run")
                }
            }

            Button { removePocket(pocket) } label: {
                Image(systemName: "trash")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Remove this pocket and its grid box")
        }
        .padding(DS.Space.s)
        .background(shape.fill(isSelected ? DS.Color.pocket.opacity(0.12) : Color.clear))
        .overlay(shape.strokeBorder(isSelected ? DS.Color.pocket.opacity(0.45) : Color.clear, lineWidth: 1))
        .contentShape(shape)
        .onTapGesture { selectPocket(pocket) }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(pocket.displayName), \(Int(pocket.volume)) cubic ångströms")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityAction { selectPocket(pocket) }
    }

    @ViewBuilder
    private var selectedPocketSummary: some View {
        let selected = viewModel.docking.selectedPocket
        if let pocket = selected {
            DSMetricGrid(columns: 3) {
                DSMetric("Volume", value: String(format: "%.0f", pocket.volume), unit: "Å³", tone: .pocket)
                DSMetric("Buried", value: String(format: "%.0f", pocket.buriedness * 100), unit: "%")
                DSMetric("Residues", value: "\(pocket.residueIndices.count)")
            }
        } else {
            PanelHint(text: "Select a pocket to inspect it and place the grid box.", icon: "hand.point.up.left")
        }
        PanelSecondaryButton(
            title: "Focus on Pocket",
            icon: "scope",
            tint: DS.Color.pocket,
            isDisabled: selected == nil,
            help: "Zoom to the pocket with Z-clipping slab"
        ) {
            if let pocket = selected { viewModel.focusOnPocket(pocket) }
        }
        .accessibilityIdentifier(AccessibilityID.dockFocusPocket)
    }

    // MARK: - Pocket view

    private var pocketViewCard: some View {
        let hasPocket = viewModel.docking.selectedPocket != nil
        return PanelCard("Pocket View", icon: "rectangle.split.3x1") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                PanelToggleRow(
                    title: "Clip around pocket",
                    subtitle: hasPocket ? "Slab view with a translucent surface" : "Select a pocket first",
                    isOn: Binding(
                        get: { viewModel.workspace.enableClipping },
                        set: { setPocketClipping($0) }
                    )
                )
                .accessibilityIdentifier(AccessibilityID.dockPocketViewToggle)

                PanelChoiceGrid(columns: 3) {
                    slabPresetButton("Tight", id: AccessibilityID.dockSlabTight, isDisabled: !hasPocket)
                    slabPresetButton("Medium", id: AccessibilityID.dockSlabMedium, isDisabled: !hasPocket)
                    slabPresetButton("Wide", id: AccessibilityID.dockSlabWide, isDisabled: !hasPocket)
                }
                PanelHint(text: "Fine-tune thickness from the clipping control in the viewport.",
                          icon: "slider.horizontal.below.rectangle")
            }
        }
    }

    private func slabPresetButton(_ preset: String, id: String, isDisabled: Bool) -> some View {
        PanelChoiceButton(
            title: preset, isSelected: false, isDisabled: isDisabled,
            help: "Z-slab \(preset.lowercased()) clipping around the pocket"
        ) {
            if !viewModel.workspace.enableClipping { setPocketClipping(true) }
            applySlabPreset(preset)
        }
        .accessibilityIdentifier(id)
    }

    // MARK: - Grid box

    private var gridBoxCard: some View {
        let half = viewModel.docking.gridHalfSize
        return PanelCard(
            "Grid Box",
            icon: "cube.transparent",
            accessory: {
                Button { site.applyGridBoxFromSliders() } label: {
                    Label("Apply", systemImage: "arrow.clockwise")
                        .font(DS.Font.secondary.weight(.medium))
                        .foregroundStyle(DS.Color.accent)
                }
                .buttonStyle(.borderless)
                .help("Apply grid box to viewport")
                .accessibilityIdentifier(AccessibilityID.dockApplyGrid)
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    PanelSubheader(title: "Center on")
                    PanelChoiceGrid(columns: 2) {
                        PanelChoiceButton(
                            title: "Protein", icon: "building.2", isSelected: false,
                            help: "Center grid on protein centroid"
                        ) { placeGridAtProteinCenter() }
                        .accessibilityIdentifier(AccessibilityID.dockGridProtein)

                        PanelChoiceButton(
                            title: "Ligand", icon: "hexagon", isSelected: false,
                            isDisabled: viewModel.molecules.ligand == nil,
                            help: "Center grid on ligand position"
                        ) { placeGridAtLigand() }
                        .accessibilityIdentifier(AccessibilityID.dockGridLigand)

                        PanelChoiceButton(
                            title: "Selection", icon: "hand.tap", isSelected: false,
                            isDisabled: viewModel.workspace.selectedResidueIndices.isEmpty,
                            help: "Center grid on selected residues"
                        ) { placeGridAtSelection() }
                        .accessibilityIdentifier(AccessibilityID.dockGridSelection)

                        PanelChoiceButton(
                            title: "Pocket", icon: "arrow.counterclockwise", isSelected: false,
                            isDisabled: viewModel.docking.selectedPocket == nil,
                            help: "Reset grid to detected pocket"
                        ) { resetGridToPocket() }
                        .accessibilityIdentifier(AccessibilityID.dockGridPocket)
                    }
                }
                VStack(alignment: .leading, spacing: DS.Space.xs) {
                    PanelSubheader(title: "Center")
                    PanelAxisSlider(axis: "X", value: gridCenterBinding(\.x), range: -200...200, onChange: { _ in site.applyGridBoxFromSliders() })
                    PanelAxisSlider(axis: "Y", value: gridCenterBinding(\.y), range: -200...200, onChange: { _ in site.applyGridBoxFromSliders() })
                    PanelAxisSlider(axis: "Z", value: gridCenterBinding(\.z), range: -200...200, onChange: { _ in site.applyGridBoxFromSliders() })
                }
                VStack(alignment: .leading, spacing: DS.Space.xs) {
                    PanelSubheader(title: "Half-size")
                    PanelAxisSlider(axis: "X", value: gridHalfBinding(\.x), range: 2...50, onChange: { _ in site.applyGridBoxFromSliders() })
                    PanelAxisSlider(axis: "Y", value: gridHalfBinding(\.y), range: 2...50, onChange: { _ in site.applyGridBoxFromSliders() })
                    PanelAxisSlider(axis: "Z", value: gridHalfBinding(\.z), range: 2...50, onChange: { _ in site.applyGridBoxFromSliders() })
                }
                HStack(spacing: DS.Space.xs) {
                    Image(systemName: "cube")
                        .foregroundStyle(DS.Color.pocket)
                    Text(DockingSite.boxDimensions(half))
                    Spacer(minLength: DS.Space.xs)
                    Text(String(format: "%.0f Å³", half.x * half.y * half.z * 8))
                }
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        // Side by side when it fits; otherwise the summary gets its own line so it is never truncated.
        ViewThatFits(in: .horizontal) {
            HStack(spacing: DS.Space.m) {
                footerSummary.fixedSize()
                Spacer(minLength: DS.Space.xs)
                footerAction
            }
            VStack(alignment: .leading, spacing: DS.Space.s) {
                footerSummary
                HStack {
                    Spacer(minLength: 0)
                    footerAction
                }
            }
        }
    }

    private var footerSummary: some View {
        HStack(spacing: DS.Space.m) {
            Circle()
                .fill(site.hasGrid ? DS.Color.pocket : Color.secondary.opacity(0.4))
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                Text(site.siteTitle ?? "No binding site defined")
                    .font(DS.Font.label.weight(.medium))
                    .lineLimit(1)
                Text(site.hasGrid
                     ? DockingSite.boxDimensions(viewModel.docking.gridHalfSize)
                     : "Detect a pocket or place the grid box")
                    .font(DS.Font.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }

    @ViewBuilder
    private var footerAction: some View {
        if site.hasGrid {
            DSNextStepButton(.dock)
                .fixedSize()
        }
    }

    // MARK: - Pocket actions

    private func selectPocket(_ pocket: BindingPocket) {
        viewModel.docking.selectedPocket = pocket
        ActivityLog.shared.info(
            "\(pocket.displayName) selected (vol \(String(format: "%.0f", pocket.volume)) Å³, druggability \(String(format: "%.2f", pocket.druggability)))",
            category: .dock
        )
    }

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
            site.syncGridFromPocket(pocket)
            site.applyGridBoxFromSliders()
        }
    }

    private func pocketFromSelection() {
        guard viewModel.molecules.protein != nil,
              !viewModel.workspace.selectedResidueIndices.isEmpty else { return }
        viewModel.pocketFromSelection()
        if let pocket = viewModel.docking.selectedPocket {
            site.syncGridFromPocket(pocket)
            site.applyGridBoxFromSliders()
        }
    }

    private func removePocket(_ pocket: BindingPocket) {
        viewModel.docking.detectedPockets.removeAll { $0.id == pocket.id }
        if viewModel.docking.selectedPocket?.id == pocket.id {
            viewModel.docking.selectedPocket = nil
            viewModel.renderer?.clearGridBox()
            viewModel.renderer?.setNeedsRedraw()
        }
        ActivityLog.shared.info("Removed \(pocket.displayName)", category: .dock)
    }

    // MARK: - Pocket view actions

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

    // MARK: - Grid box actions

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
        site.initializeGridAtProteinCenter(prot)
        site.applyGridBoxFromSliders()
    }

    private func placeGridAtLigand() {
        guard let lig = viewModel.molecules.ligand else { return }
        placeGrid(around: lig.atoms.map(\.position))
    }

    private func placeGridAtSelection() {
        guard let prot = viewModel.molecules.protein else { return }
        var positions: [SIMD3<Float>] = []
        for resIdx in viewModel.workspace.selectedResidueIndices where resIdx < prot.residues.count {
            for atomIdx in prot.residues[resIdx].atomIndices where atomIdx < prot.atoms.count {
                positions.append(prot.atoms[atomIdx].position)
            }
        }
        placeGrid(around: positions)
    }

    /// Centers the grid on the centroid of `positions`, sized to enclose them plus a 4 Å margin.
    private func placeGrid(around positions: [SIMD3<Float>]) {
        guard !positions.isEmpty else { return }
        let center = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
        var maxDist: Float = 0
        for pos in positions {
            let d = abs(pos - center)
            maxDist = max(maxDist, max(d.x, max(d.y, d.z)))
        }
        viewModel.docking.gridCenter = center
        viewModel.docking.gridHalfSize = SIMD3<Float>(repeating: maxDist + 4.0)
        site.applyGridBoxFromSliders()
    }

    private func resetGridToPocket() {
        guard let pocket = viewModel.docking.selectedPocket else { return }
        site.syncGridFromPocket(pocket)
        site.applyGridBoxFromSliders()
    }
}

// MARK: - Shared binding-site / grid logic (Binding Site + Docking steps)

/// Grid-box and site helpers shared by the Binding Site and Docking steps.
@MainActor
struct DockingSite {
    let viewModel: AppViewModel

    /// A docking site exists when a pocket is selected or the grid box has been placed.
    var hasGrid: Bool {
        let half = viewModel.docking.gridHalfSize
        return viewModel.docking.selectedPocket != nil
            || (viewModel.docking.gridInitialized && half.x > 2 && half.y > 2 && half.z > 2)
    }

    /// Short display name of the current site, or nil when none is defined.
    var siteTitle: String? {
        if let pocket = viewModel.docking.selectedPocket {
            return pocket.displayName
        }
        return hasGrid ? "Custom grid box" : nil
    }

    static func boxDimensions(_ half: SIMD3<Float>) -> String {
        String(format: "%.1f × %.1f × %.1f Å", half.x * 2, half.y * 2, half.z * 2)
    }

    func goTo(_ tab: SidebarTab) {
        viewModel.workspace.requestedTab = tab
        viewModel.workspace.requestedTabToken &+= 1
    }

    func initializeGridAtProteinCenter(_ prot: Molecule) {
        let hidden = viewModel.workspace.hiddenChainIDs
        let positions = hidden.isEmpty
            ? prot.atoms.map(\.position)
            : prot.atoms.filter { !hidden.contains($0.chainID) }.map(\.position)
        guard !positions.isEmpty else { return }
        viewModel.docking.gridCenter = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
        viewModel.docking.gridHalfSize = SIMD3<Float>(repeating: 10)
        viewModel.docking.gridInitialized = true
    }

    func syncGridFromPocket(_ pocket: BindingPocket) {
        viewModel.docking.gridCenter = pocket.center
        viewModel.docking.gridHalfSize = pocket.size
    }

    func applyGridBoxFromSliders() {
        viewModel.updateGridBoxVisualization(center: viewModel.docking.gridCenter,
                                             halfSize: viewModel.docking.gridHalfSize)
        if var pocket = viewModel.docking.selectedPocket {
            pocket.center = viewModel.docking.gridCenter
            pocket.size = viewModel.docking.gridHalfSize
            viewModel.docking.selectedPocket = pocket
        }
    }

    /// Create a pocket from the current grid so the docking engine has what it needs.
    func ensurePocketFromGrid() {
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

/// Keeps the grid box in sync with the selected pocket and the loaded protein.
private struct DockingSiteSync: ViewModifier {
    @Environment(AppViewModel.self) private var viewModel

    func body(content: Content) -> some View {
        let site = DockingSite(viewModel: viewModel)
        content
            .onChange(of: viewModel.docking.selectedPocket?.id) { _, newID in
                if let newID, let pocket = viewModel.docking.detectedPockets.first(where: { $0.id == newID }) {
                    site.syncGridFromPocket(pocket)
                    viewModel.showGridBoxForPocket(pocket)
                } else if newID == nil {
                    viewModel.renderer?.clearGridBox()
                }
            }
            .onAppear {
                if !viewModel.docking.gridInitialized, let prot = viewModel.molecules.protein {
                    site.initializeGridAtProteinCenter(prot)
                }
            }
            .onChange(of: viewModel.molecules.protein?.name) { _, _ in
                if let prot = viewModel.molecules.protein,
                   viewModel.docking.selectedPocket == nil,
                   viewModel.docking.dockingResults.isEmpty {
                    site.initializeGridAtProteinCenter(prot)
                }
            }
    }
}

extension View {
    /// Grid ↔ pocket synchronisation shared by the Binding Site and Docking steps.
    func dockingSiteSync() -> some View { modifier(DockingSiteSync()) }
}

extension BindingPocket {
    /// 1-based rank shown to the user. Detectors assign `id` 0…n-1 in ranking order, and the
    /// ligand-guided / residue-selection paths produce a single pocket with id 0.
    var number: Int { id + 1 }

    /// User-facing name, shared by the pocket list, canvas chip, step header and Docking inputs.
    /// A pocket synthesised from a hand-placed grid box has id 0 and no lining residues.
    var displayName: String {
        id == 0 && residueIndices.isEmpty ? "Custom grid box" : "Pocket \(number)"
    }
}
