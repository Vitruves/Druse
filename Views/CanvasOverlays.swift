// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Canvas stage

/// The centre of the main window: the Metal viewport plus everything that floats on it
/// (molecule chips, camera cluster, legends, display cluster, docking HUD, welcome, demo).
/// `MetalView` is created once with a stable identity; overlays never wrap it.
struct CanvasStage: View {
    /// Step shown in the navigator: the interaction legend only appears where poses are reviewed.
    let currentStep: SidebarTab
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var metalTimedOut = false

    private var hasContent: Bool {
        viewModel.molecules.protein != nil || viewModel.molecules.ligand != nil || viewModel.isDemoRunning
    }
    private var metalUnavailable: Bool { viewModel.renderer == nil && metalTimedOut }

    var body: some View {
        viewport
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay { overlays }
            .clipped()
            .task {
                try? await Task.sleep(for: .seconds(3))
                metalTimedOut = true
            }
    }

    @ViewBuilder
    private var viewport: some View {
        if let renderer = viewModel.renderer {
            MetalView(
                renderer: renderer,
                onAtomSelected: { atomIndex, isOptionClick in
                    viewModel.selectAtom(atomIndex, toggle: isOptionClick)
                },
                onAtomDoubleClicked: { atomIndex in
                    viewModel.selectChainOfAtom(atomIndex)
                },
                onRenderModeChanged: { mode in
                    viewModel.setRenderMode(mode)
                },
                onToggleHydrogens: {
                    viewModel.toggleHydrogens()
                },
                onToggleLighting: {
                    viewModel.toggleLighting()
                },
                onBoxSelection: { atomIndices, addToExisting in
                    viewModel.handleBoxSelection(atomIndices: atomIndices, addToExisting: addToExisting)
                },
                onRibbonResidueSelected: { proteinAtomID, isOptionClick in
                    viewModel.selectAtom(proteinAtomID, toggle: isOptionClick)
                },
                onFitToView: {
                    viewModel.fitToView()
                },
                onContextMenu: { event, view in
                    viewModel.showContextMenu(event: event, view: view)
                },
                onDeselectAll: {
                    viewModel.deselectAll()
                }
            )
            .id("viewport")
        } else {
            ZStack {
                DS.Color.canvasBackground
                if metalUnavailable {
                    DSEmptyState(
                        "Metal Unavailable",
                        systemImage: "exclamationmark.triangle",
                        description: "Druse could not start its GPU renderer on this Mac."
                    ) {
                        Button("Report Bug…") { openWindow(id: "bug-report") }
                            .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                    }
                } else {
                    DSEmptyState("Starting Metal…", systemImage: "cube.transparent") {
                        ProgressView().controlSize(.small)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var overlays: some View {
        ZStack {
            if hasContent {
                VStack(alignment: .leading, spacing: DS.Space.m) {
                    HStack(alignment: .top, spacing: DS.Space.l) {
                        MoleculeChips()
                        Spacer(minLength: DS.Space.l)
                        CameraCluster()
                    }
                    Spacer(minLength: DS.Space.l)
                    // Legends sit in the bottom-left corner, sized to their content, on their own
                    // row above the HUD and display cluster so they never cover them.
                    legends
                    VStack(spacing: DS.Space.m) {
                        if viewModel.docking.isDocking && !viewModel.isDemoRunning {
                            DockingHUD()
                                .transition(.move(edge: .bottom).combined(with: .opacity))
                        }
                        DisplayCluster()
                    }
                    .frame(maxWidth: .infinity)
                }
                .padding(DS.Space.l)
                .animation(DS.Motion.panel, value: viewModel.docking.isDocking)
            } else if !metalUnavailable {
                WelcomeScreen()
                    .transition(.opacity)
            }

            DemoOverlay()
        }
    }

    private var showsInteractionLegend: Bool {
        switch currentStep {
        case .dock, .results, .leadOptimization: !viewModel.docking.currentInteractions.isEmpty
        default: false
        }
    }

    @ViewBuilder
    private var legends: some View {
        let surfaceLegend = viewModel.workspace.showSurface ? viewModel.workspace.surfaceLegend : nil
        if surfaceLegend != nil || showsInteractionLegend {
            HStack(alignment: .bottom, spacing: DS.Space.m) {
                if let surfaceLegend {
                    SurfaceLegendView(mode: viewModel.workspace.surfaceColorMode, legend: surfaceLegend)
                }
                if showsInteractionLegend {
                    InteractionLegendView(interactions: viewModel.docking.currentInteractions)
                }
            }
            .fixedSize()
            // Opaque material: glass would smear the atoms behind a legend.
            .dsFloatingStyle(.solid)
            .transition(.opacity)
        }
    }
}

// MARK: - Molecule chips (top-left)

private struct MoleculeChips: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        HStack(spacing: DS.Space.s) {
            if let protein = viewModel.molecules.protein {
                DSMoleculeChip(kind: .protein, name: protein.name, detail: "\(protein.atomCount.formatted()) atoms") {
                    viewModel.molecules.protein = nil
                    viewModel.pushToRenderer()
                }
            }
            if let ligand = viewModel.molecules.ligand {
                DSMoleculeChip(kind: .ligand, name: ligand.name, detail: "\(ligand.atomCount.formatted()) atoms") {
                    viewModel.removeLigandFromView()
                }
            }
            if let pocket = viewModel.docking.selectedPocket {
                DSMoleculeChip(kind: .pocket, name: pocket.displayName,
                               detail: String(format: "%.0f Å³", pocket.volume))
            }
        }
    }
}

// MARK: - Camera cluster (top-right)

private struct CameraCluster: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        DSCanvasToolCluster(axis: .vertical) {
            DSCanvasToolButton("Fit to View", systemImage: "arrow.up.left.and.arrow.down.right",
                               help: "Fit to view (⌘0)") { viewModel.fitToView() }
                .accessibilityIdentifier(AccessibilityID.renderFitToView)
            DSCanvasToolButton("Center on Ligand", systemImage: "viewfinder",
                               help: "Center on ligand") { viewModel.fitToLigand() }
                .disabled(viewModel.molecules.ligand == nil)
                .accessibilityIdentifier(AccessibilityID.renderFitToLigand)
            DSCanvasToolButton("Center on Protein", systemImage: "cube",
                               help: "Center on protein") { viewModel.fitToProtein() }
                .disabled(viewModel.molecules.protein == nil)
                .accessibilityIdentifier(AccessibilityID.renderFitToProtein)
            DSCanvasToolButton("Reset Camera", systemImage: "arrow.counterclockwise",
                               help: "Reset camera") {
                viewModel.renderer?.camera.reset()
                viewModel.pushToRenderer()
            }
            .accessibilityIdentifier(AccessibilityID.renderResetCamera)
        }
    }
}

// MARK: - Display cluster (bottom-centre)

private struct DisplayCluster: View {
    @Environment(AppViewModel.self) private var viewModel

    @State private var showSideChains = false
    @State private var showColorScheme = false
    @State private var showSurfaceColor = false
    @State private var showClipOptions = false
    @State private var showProbe = false
    @State private var showOpacity = false
    @State private var showMore = false

    var body: some View {
        // Collapses the "Look" and opacity groups into a single "…" when the canvas is narrow.
        // Four groups: representation | visibility | colour | lighting & view.
        ViewThatFits(in: .horizontal) {
            DSCanvasToolCluster {
                styleGroup
                DSCanvasToolDivider()
                showGroup
                DSCanvasToolDivider()
                colorGroup
                DSCanvasToolDivider()
                lookGroup
                opacityButton
            }
            DSCanvasToolCluster {
                styleGroup
                DSCanvasToolDivider()
                showGroup
                DSCanvasToolDivider()
                colorGroup
                DSCanvasToolDivider()
                DSCanvasToolButton("More", systemImage: "ellipsis", isOn: showMore,
                                   help: "Lighting, clipping, probe and opacity") { showMore.toggle() }
                    .popover(isPresented: $showMore, arrowEdge: .top) {
                        CanvasPopover("View Options") { LookOptionsPanel() }
                    }
            }
        }
    }

    // A. Style
    @ViewBuilder
    private var styleGroup: some View {
        ForEach(Array(RenderMode.allCases.enumerated()), id: \.element) { index, mode in
            DSCanvasToolButton(mode.rawValue, systemImage: mode.icon,
                               isOn: viewModel.workspace.renderMode == mode,
                               help: "\(mode.rawValue) (⌃⌘\(index + 1))") {
                viewModel.setRenderMode(mode)
            }
            .accessibilityIdentifier(AccessibilityID.renderMode(mode.rawValue))
        }
    }

    // B. Visibility
    @ViewBuilder
    private var showGroup: some View {
        let workspace = viewModel.workspace
        let hAvailable = viewModel.proteinHasHydrogens
        CanvasGlyphButton(
            "Hydrogens",
            glyph: "H",
            isOn: workspace.showHydrogens && hAvailable,
            help: hAvailable
                ? (workspace.showHydrogens ? "Hide hydrogens (⌃⌘H)" : "Show hydrogens (⌃⌘H)")
                : "No hydrogens — add them in Preparation"
        ) { viewModel.toggleHydrogens() }
            .disabled(!hAvailable)
            .accessibilityIdentifier(AccessibilityID.renderHydrogens)

        let isRibbon = workspace.renderMode == .ribbon
        DSCanvasToolButton("Side Chains", systemImage: workspace.sideChainDisplay.icon,
                           isOn: isRibbon && workspace.sideChainDisplay != .none,
                           help: isRibbon
                               ? "Side chains: \(workspace.sideChainDisplay.rawValue)"
                               : "Side chains (Ribbon mode only)") { showSideChains.toggle() }
            .disabled(!isRibbon)
            .popover(isPresented: $showSideChains, arrowEdge: .top) {
                CanvasPopover("Side Chains") {
                    Picker("Side chains", selection: Binding(
                        get: { viewModel.workspace.sideChainDisplay },
                        set: { viewModel.workspace.sideChainDisplay = $0; viewModel.pushToRenderer() }
                    )) {
                        ForEach(SideChainDisplay.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.radioGroup)
                    .labelsHidden()
                }
            }

        DSCanvasToolButton("Surface", systemImage: "circle.hexagongrid", isOn: workspace.showSurface,
                           help: workspace.showSurface ? "Hide molecular surface" : "Show molecular surface") {
            viewModel.toggleSurface()
        }
        .disabled(viewModel.molecules.protein == nil || workspace.isGeneratingSurface)
        .accessibilityIdentifier(AccessibilityID.renderSurface)

        DSCanvasToolButton("Probe Radius", systemImage: "circle.dashed",
                           isOn: workspace.showSurface && workspace.surfaceProbeRadius != 1.4,
                           help: workspace.showSurface
                               ? String(format: "Surface probe %.1f Å, grid %.1f Å",
                                        workspace.surfaceProbeRadius, workspace.surfaceGridSpacing)
                               : "Surface probe (show the surface first)") { showProbe.toggle() }
            .disabled(!workspace.showSurface)
            .popover(isPresented: $showProbe, arrowEdge: .top) {
                CanvasPopover("Surface Probe") { ProbeControls() }
            }
    }

    // C. Colour
    @ViewBuilder
    private var colorGroup: some View {
        let workspace = viewModel.workspace
        DSCanvasToolButton("Surface Coloring", systemImage: surfaceColorIcon,
                           isOn: workspace.showSurface && workspace.surfaceColorMode != .uniform,
                           help: workspace.showSurface
                               ? "Surface coloring: \(workspace.surfaceColorMode.rawValue)"
                               : "Surface coloring (show the surface first)") { showSurfaceColor.toggle() }
            .disabled(!workspace.showSurface)
            .accessibilityIdentifier(AccessibilityID.renderSurfaceColor)
            .popover(isPresented: $showSurfaceColor, arrowEdge: .top) {
                CanvasPopover("Surface Coloring") {
                    Picker("Surface coloring", selection: Binding(
                        get: { viewModel.workspace.surfaceColorMode },
                        set: { viewModel.setSurfaceColorMode($0) }
                    )) {
                        ForEach(SurfaceColorMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.radioGroup)
                    .labelsHidden()
                }
            }

        DSCanvasToolButton("Color Scheme", systemImage: "paintpalette",
                           isOn: workspace.colorScheme != .element,
                           help: "Color scheme: \(workspace.colorScheme.rawValue)") { showColorScheme.toggle() }
            .accessibilityIdentifier(AccessibilityID.renderColorScheme)
            .popover(isPresented: $showColorScheme, arrowEdge: .top) {
                CanvasPopover("Color Scheme") {
                    Picker("Color scheme", selection: Binding(
                        get: { viewModel.workspace.colorScheme },
                        set: { viewModel.workspace.colorScheme = $0; viewModel.pushToRenderer() }
                    )) {
                        ForEach(WorkspaceState.MoleculeColorScheme.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.radioGroup)
                    .labelsHidden()
                }
            }
    }

    // D. Lighting & view
    @ViewBuilder
    private var lookGroup: some View {
        let workspace = viewModel.workspace
        DSCanvasToolButton("Lighting", systemImage: "sun.max", isOn: workspace.useDirectionalLighting,
                           help: workspace.useDirectionalLighting ? "Uniform lighting" : "Directional lighting") {
            viewModel.toggleLighting()
        }
        .accessibilityIdentifier(AccessibilityID.renderLighting)

        HStack(spacing: 0) {
            DSCanvasToolButton("Clipping", systemImage: "square.split.diagonal", isOn: workspace.enableClipping,
                               help: workspace.enableClipping ? "Disable Z-slab clipping" : "Enable Z-slab clipping") {
                viewModel.workspace.enableClipping.toggle()
                CanvasActions.syncClipping(viewModel)
            }
            .accessibilityIdentifier(AccessibilityID.renderClipping)

            CanvasDisclosureButton(help: "Clipping thickness and offset") { showClipOptions.toggle() }
                .popover(isPresented: $showClipOptions, arrowEdge: .top) {
                    CanvasPopover("Z-Slab Clipping") { ClippingControls() }
                }
        }
    }

    private var opacityButton: some View {
        DSCanvasToolButton("Opacity", systemImage: "circle.lefthalf.filled", isOn: showOpacity,
                           help: "Background and surface opacity") { showOpacity.toggle() }
            .popover(isPresented: $showOpacity, arrowEdge: .top) {
                CanvasPopover("Opacity") { OpacityControls() }
            }
    }

    private var surfaceColorIcon: String {
        switch viewModel.workspace.surfaceColorMode {
        case .uniform:        "paintbrush"
        case .esp:            "bolt"
        case .hydrophobicity: "drop"
        case .pharmacophore:  "circle.grid.cross"
        }
    }
}

// MARK: - Popover panels

/// Shared popover chrome for canvas controls.
private struct CanvasPopover<Content: View>: View {
    let title: String
    let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            Text(title)
                .font(DS.Font.sectionTitle)
            content
        }
        .padding(DS.Space.l)
        .frame(minWidth: 240, alignment: .leading)
    }
}

/// Compact "…" panel: everything from the Look and Opacity groups.
private struct LookOptionsPanel: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            Toggle("Directional lighting", isOn: Binding(
                get: { viewModel.workspace.useDirectionalLighting },
                set: { if $0 != viewModel.workspace.useDirectionalLighting { viewModel.toggleLighting() } }
            ))
            Toggle("Z-slab clipping", isOn: Binding(
                get: { viewModel.workspace.enableClipping },
                set: { viewModel.workspace.enableClipping = $0; CanvasActions.syncClipping(viewModel) }
            ))
            ClippingControls()
            Divider()
            ProbeControls()
                .disabled(!viewModel.workspace.showSurface)
            Divider()
            OpacityControls()
        }
        .toggleStyle(.switch)
        .controlSize(.small)
        .font(DS.Font.label)
    }
}

private struct ClippingControls: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            sliderRow("Thickness", value: viewModel.workspace.slabThickness, range: 2...40) {
                viewModel.workspace.slabThickness = $0
            }
            sliderRow("Offset", value: viewModel.workspace.slabOffset, range: -20...20) {
                viewModel.workspace.slabOffset = $0
            }
            if !viewModel.workspace.enableClipping {
                Text("Turn clipping on to apply the slab.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func sliderRow(_ title: String, value: Float, range: ClosedRange<Float>,
                           set: @escaping (Float) -> Void) -> some View {
        CanvasSliderRow(title: title, value: Binding(get: { value }, set: {
            set($0)
            CanvasActions.syncClipping(viewModel)
        }), range: range, format: "%.0f Å")
    }
}

private struct ProbeControls: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var probe: Float = 1.4

    private struct Preset: Hashable {
        let title: String
        let radius: Float
    }

    private static let presets: [Preset] = [
        Preset(title: "Water", radius: 1.4), Preset(title: "Small molecule", radius: 1.8),
        Preset(title: "Ligand access", radius: 2.2), Preset(title: "Large pocket", radius: 3.0),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(spacing: DS.Space.m) {
                Text("Probe")
                    .foregroundStyle(.secondary)
                    .frame(width: 64, alignment: .leading)
                Slider(value: snappedSliderBinding($probe, range: 1.0...4.0, step: 0.1), in: 1.0...4.0) { editing in
                    if !editing { applyProbe(probe) }
                }
                Text(String(format: "%.1f Å", probe))
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .frame(width: 44, alignment: .trailing)
            }
            HStack(spacing: DS.Space.xs) {
                ForEach(Self.presets, id: \.self) { preset in
                    Button(preset.title) {
                        probe = preset.radius
                        applyProbe(preset.radius)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .help(String(format: "%@ probe (%.1f Å)", preset.title, preset.radius))
                }
            }
            Picker("Grid", selection: Binding(
                get: { viewModel.workspace.surfaceGridSpacing },
                set: { viewModel.workspace.surfaceGridSpacing = $0; viewModel.regenerateSurface() }
            )) {
                Text("Fine 0.3 Å").tag(Float(0.3))
                Text("Normal 0.5 Å").tag(Float(0.5))
                Text("Coarse 0.7 Å").tag(Float(0.7))
            }
            .pickerStyle(.segmented)
            .controlSize(.small)
        }
        .font(DS.Font.label)
        .onAppear { probe = viewModel.workspace.surfaceProbeRadius }
    }

    private func applyProbe(_ radius: Float) {
        guard radius != viewModel.workspace.surfaceProbeRadius else { return }
        viewModel.workspace.surfaceProbeRadius = radius
        viewModel.regenerateSurface()
    }
}

private struct OpacityControls: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            CanvasSliderRow(title: "Surface", value: Binding(
                get: { viewModel.workspace.surfaceOpacity },
                set: { viewModel.workspace.surfaceOpacity = $0; viewModel.renderer?.surfaceOpacity = $0 }
            ), range: 0.1...1.0, format: "%.0f%%", scale: 100)
            .disabled(!viewModel.workspace.showSurface)

            CanvasSliderRow(title: "Background", value: Binding(
                get: { viewModel.workspace.backgroundOpacity },
                set: { viewModel.workspace.backgroundOpacity = $0; viewModel.renderer?.backgroundOpacity = $0 }
            ), range: 0...1, format: "%.0f%%", scale: 100)

            Text("Background at 0% gives a plain white or black backdrop for figures.")
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(DS.Font.label)
    }
}

private struct CanvasSliderRow: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String
    var scale: Float = 1

    var body: some View {
        HStack(spacing: DS.Space.m) {
            Text(title)
                .foregroundStyle(.secondary)
                .frame(width: 64, alignment: .leading)
            Slider(value: $value, in: range)
                .controlSize(.small)
            Text(String(format: format, value * scale))
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .frame(width: 44, alignment: .trailing)
        }
    }
}

// MARK: - Docking HUD

private struct DockingHUD: View {
    @Environment(AppViewModel.self) private var viewModel

    /// Live readout only: stopping lives with the Run button in the Docking step footer and
    /// in the toolbar activity capsule, so the HUD carries no third Stop button.
    var body: some View {
        let docking = viewModel.docking
        let progress = DockingProgress(docking)
        let method = docking.scoringMethod

        HStack(alignment: .center, spacing: DS.Space.xl) {
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                    Text(phaseTitle)
                        .font(DS.Font.label.weight(.semibold))
                        .lineLimit(1)
                    Spacer(minLength: DS.Space.s)
                    Text(progress.percentText)
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                ProgressView(value: progress.fraction)
                    .progressViewStyle(.linear)
                    .tint(DS.Color.accent)
                    .accessibilityLabel("Docking progress")
                    .accessibilityValue(progress.summary)
                Text("\(progress.summary) · \((progress.step * docking.dockingConfig.populationSize).formatted()) poses")
                    .font(DS.Font.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .frame(width: 260)

            VStack(alignment: .trailing, spacing: DS.Space.xxs) {
                Text("Best so far")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
                if method.isAffinityScore, let pKi = docking.dockingBestPKi {
                    DSScoreBadge(pK: pKi)
                } else if docking.dockingBestEnergy < .infinity {
                    DSScoreBadge(energy: docking.dockingBestEnergy, unit: method.unitLabel)
                } else {
                    Text("—").font(DS.Font.label).foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.horizontal, DS.Space.l)
        .padding(.vertical, DS.Space.m)
        .dsFloating()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Docking progress")
    }

    private var phaseTitle: String {
        let docking = viewModel.docking
        let batch = docking.batchProgress
        if docking.isBatchDocking && batch.total > 1 {
            return "Docking ligand \(batch.current + 1) of \(batch.total)"
        }
        return "Docking"
    }
}

// MARK: - Canvas tool helpers

/// Same chrome as `DSCanvasToolButton`, with a text glyph instead of a symbol.
private struct CanvasGlyphButton: View {
    let title: String
    let glyph: String
    let isOn: Bool
    let help: String
    let action: () -> Void
    @State private var isHovering = false
    @Environment(\.isEnabled) private var isEnabled

    init(_ title: String, glyph: String, isOn: Bool, help: String, action: @escaping () -> Void) {
        self.title = title
        self.glyph = glyph
        self.isOn = isOn
        self.help = help
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(glyph)
                .font(.system(size: DS.Size.toolIcon, weight: .semibold, design: .rounded))
                .frame(width: 16, height: 16)
                .dsCanvasToolChrome(isOn: isOn, isHovering: isHovering, isEnabled: isEnabled)
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help(help)
        .accessibilityLabel(title)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

/// Narrow chevron that opens a tool's options popover.
private struct CanvasDisclosureButton: View {
    let help: String
    let action: () -> Void
    @State private var isHovering = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
        Button(action: action) {
            Image(systemName: "chevron.down")
                .font(DS.Font.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 14, height: DS.Size.toolButton)
                .background(shape.fill(isHovering ? DS.Color.controlFill : Color.clear))
                .contentShape(shape)
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help(help)
        .accessibilityLabel(help)
    }
}

/// Renderer-facing canvas actions shared by the display cluster and its popovers.
@MainActor
private enum CanvasActions {
    static func syncClipping(_ viewModel: AppViewModel) {
        let workspace = viewModel.workspace
        guard let renderer = viewModel.renderer else { return }
        renderer.enableClipping = workspace.enableClipping
        renderer.clipNearZ = workspace.clipNearZ
        renderer.clipFarZ = workspace.clipFarZ
        renderer.slabHalfThickness = workspace.slabThickness / 2.0
        renderer.slabOffset = workspace.slabOffset
        // Without a focused pocket, release any stale slab anchor so the slab
        // follows the camera's look-at point (Renderer falls back to camera.target).
        if !workspace.enableClipping {
            renderer.slabCenter = nil
        }
        renderer.setNeedsRedraw()
    }
}
