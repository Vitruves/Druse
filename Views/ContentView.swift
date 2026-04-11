import SwiftUI
import MetalKit

struct ContentView: View {
    @Environment(AppViewModel.self) private var viewModel
    @AppStorage("appTheme") private var appTheme: String = AppTheme.dark.rawValue
    @Environment(\.colorScheme) private var colorScheme
    @State private var showInspector = false
    @State private var showConsole = false
    @State private var consoleHeight: CGFloat = 130
    @State private var isDragTargeted = false
    @State private var pipelineTab: SidebarTab = .search
    @State private var pipelinePanelOpen: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            mainContentArea
            StatusStripView(showConsole: $showConsole, consoleHeight: $consoleHeight)
        }
        .toolbar {
            // Pipeline steps — centered in the toolbar
            ToolbarItem(placement: .principal) {
                PipelineBar(selectedTab: $pipelineTab, panelOpen: $pipelinePanelOpen)
            }

            // Inspector toggle only (console moved to status strip)
            ToolbarItemGroup(placement: .primaryAction) {
                Toggle(isOn: $showInspector) {
                    Label("Inspector", systemImage: "sidebar.right")
                }
                .help("Toggle inspector panel")
            }
        }
        .onChange(of: ActivityLog.shared.entries.count) { _, _ in
            // Auto-expand console on error
            if let last = ActivityLog.shared.entries.last, last.level == .error, !showConsole {
                withAnimation(.easeInOut(duration: 0.2)) { showConsole = true }
            }
        }
        .onChange(of: viewModel.molecules.protein?.atomCount) { old, new in
            // Auto-show inspector when first molecule is loaded
            if old == nil && new != nil && !showInspector {
                withAnimation(.easeInOut(duration: 0.2)) { showInspector = true }
            }
        }
        .onChange(of: viewModel.demoStep) { _, step in
            // Auto-advance pipeline tabs during guided demo
            withAnimation(.easeInOut(duration: 0.2)) {
                switch step {
                case .fetching, .parsing:
                    pipelineTab = .search
                    pipelinePanelOpen = false
                case .overview, .ribbon:
                    pipelineTab = .search
                    pipelinePanelOpen = false
                case .pocketScan, .pocketFound, .gridSetup:
                    pipelineTab = .dock
                    pipelinePanelOpen = false
                case .dockingStart, .dockingRun, .dockingConverge:
                    pipelineTab = .dock
                    pipelinePanelOpen = false
                case .scoring, .bestPose, .interactions:
                    pipelineTab = .results
                    pipelinePanelOpen = false
                case .complete, .idle:
                    break
                }
            }
        }
        .onChange(of: colorScheme) { _, newScheme in
            viewModel.renderer?.themeMode = (newScheme == .light) ? 1 : 0
        }
        .onAppear {
            let theme = AppTheme(rawValue: appTheme) ?? .dark
            let isLight: Bool
            switch theme {
            case .light: isLight = true
            case .dark: isLight = false
            case .auto: isLight = (colorScheme == .light)
            }
            viewModel.renderer?.themeMode = isLight ? 1 : 0
            viewModel.renderer?.backgroundOpacity = viewModel.workspace.backgroundOpacity
            viewModel.renderer?.surfaceOpacity = viewModel.workspace.surfaceOpacity

            // No auto-open — welcome screen handles first interaction
        }
        .sheet(isPresented: Binding(
            get: { viewModel.workspace.showingConstraintSheet },
            set: { viewModel.workspace.showingConstraintSheet = $0 }
        )) {
            if let ctx = viewModel.workspace.constraintSheetContext {
                ConstraintConfigSheet(context: ctx)
                    .environment(viewModel)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $isDragTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url, FileImportHandler.canHandle(url: url) else { return }
                Task { @MainActor in
                    viewModel.loadFromFile(url: url)
                }
            }
            return true
        }
        .overlay {
            if isDragTargeted {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8, 4]))
                    .background(Color.accentColor.opacity(0.08))
                    .overlay {
                        VStack(spacing: 8) {
                            Image(systemName: "arrow.down.doc")
                                .font(.title)
                                .foregroundStyle(.secondary)
                            Text("Drop to import")
                                .font(.body.weight(.medium))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .allowsHitTesting(false)
            }
        }
    }

    // MARK: - Main Content Area (extracted to help type checker)

    private var mainContentArea: some View {
        HStack(spacing: 0) {
            // Pipeline panel — takes its own space so Metal viewport shrinks naturally
            if pipelinePanelOpen {
                PipelineContentPanel(
                    selectedTab: $pipelineTab,
                    panelOpen: $pipelinePanelOpen
                )
                .transition(.move(edge: .leading).combined(with: .opacity))

                Divider()
            }

            // Metal viewport with overlays — auto-centers in remaining space
            ZStack {
                metalViewport

                if viewModel.molecules.protein == nil && !viewModel.isDemoRunning {
                    WelcomeScreen(
                        pipelineTab: $pipelineTab,
                        pipelinePanelOpen: $pipelinePanelOpen
                    )
                    .transition(.opacity)
                } else {
                    viewportTopBottom
                }

                // Demo narration overlay (always on top when demo is active)
                DemoOverlay()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if showInspector {
                Divider()
                InspectorPanel(showInspector: $showInspector)
            }
        }
    }

    private var viewportTopBottom: some View {
        VStack {
            HStack(alignment: .top) {
                moleculeBadges
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    if viewModel.workspace.showSurface, let legend = viewModel.workspace.surfaceLegend {
                        SurfaceLegendView(mode: viewModel.workspace.surfaceColorMode, legend: legend)
                    }
                    if !viewModel.docking.currentInteractions.isEmpty {
                        InteractionLegendView(interactions: viewModel.docking.currentInteractions)
                    }
                    if viewModel.docking.isDocking && !viewModel.isDemoRunning {
                        dockingHUD
                    }
                }
            }
            .padding(12)

            Spacer()

            renderControls
                .padding(12)
        }
    }

    // MARK: - Metal Viewport

    @ViewBuilder
    private var metalViewport: some View {
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
        } else {
            ZStack {
                Color(nsColor: .windowBackgroundColor)
                VStack(spacing: 12) {
                    Image(systemName: "cube.transparent")
                        .font(.title)
                        .foregroundStyle(.tertiary)
                    Text("Initializing Metal...")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                    Text("Load a protein from the Search tab or drag a PDB file here to begin")
                        .font(.subheadline)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 260)
                }
            }
        }
    }

    // MARK: - Molecule Badges (top overlay)

    @ViewBuilder
    private var moleculeBadges: some View {
        HStack(spacing: 8) {
            if let prot = viewModel.molecules.protein {
                badge(prot.name, icon: "cube", color: .cyan, count: prot.atomCount)
            }
            if let lig = viewModel.molecules.ligand {
                badge(lig.name, icon: "hexagon", color: .green, count: lig.atomCount)
            }
            Spacer()
        }
    }

    // MARK: - Render Controls (bottom overlay)

    // MARK: - Docking HUD (live progress overlay on Metal viewport)

    @ViewBuilder
    private var dockingHUD: some View {
        let gen = viewModel.docking.dockingGeneration
        let totalGen = viewModel.docking.dockingTotalGenerations
        let bestE = viewModel.docking.dockingBestEnergy
        let bestPKi = viewModel.docking.dockingBestPKi
        let isBatch = viewModel.docking.isBatchDocking
        let bp = viewModel.docking.batchProgress

        VStack(alignment: .trailing, spacing: 4) {
            // Batch progress (if applicable)
            if isBatch && bp.total > 1 {
                Text("Ligand \(bp.current + 1)/\(bp.total)")
                    .font(.footnote.weight(.semibold).monospaced())
                    .foregroundStyle(.cyan)
            }

            // Generation progress bar
            HStack(spacing: 8) {
                Text("Gen \(gen + 1)/\(totalGen)")
                    .font(.footnote.weight(.medium).monospaced())
                    .foregroundStyle(.primary.opacity(0.9))

                ProgressView(value: Double(gen + 1), total: Double(max(totalGen, 1)))
                    .progressViewStyle(.linear)
                    .frame(width: 80)
                    .tint(.cyan)
            }

            // Best score (kcal/mol or pKi)
            HStack(spacing: 4) {
                let method = viewModel.docking.scoringMethod
                if method.isAffinityScore, let pKi = bestPKi {
                    Text("pKi")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.1f", pKi))
                        .font(.body.weight(.bold).monospaced())
                        .foregroundStyle(pKi > 6 ? .green : pKi > 4 ? .yellow : .orange)
                } else if bestE < .infinity {
                    Text(String(format: "%.1f", bestE))
                        .font(.body.weight(.bold).monospaced())
                        .foregroundStyle(bestE < -6 ? .green : bestE < -3 ? .yellow : .orange)
                    Text(method.unitLabel)
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                }
            }

            // Poses tested count
            let posesPerGen = viewModel.docking.dockingConfig.populationSize
            let totalPoses = (gen + 1) * posesPerGen
            Text("\(totalPoses) poses")
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(.ultraThinMaterial)
                .shadow(color: Color(nsColor: .shadowColor).opacity(0.3), radius: 4, y: 2)
        )
    }

    private var renderControls: some View {
        HStack(spacing: 4) {
            // Render mode buttons
            ForEach(RenderMode.allCases, id: \.self) { mode in
                let isActive = viewModel.workspace.renderMode == mode
                Button(action: { viewModel.setRenderMode(mode) }) {
                    Image(systemName: mode.icon)
                        .font(.callout)
                        .frame(width: 28, height: 28)
                        .background(isActive ? Color.accentColor.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(isActive ? Color.accentColor.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: isActive ? 1 : 0.5)
                        )
                }
                .buttonStyle(.plain)
                .foregroundStyle(isActive ? .primary : .secondary)
                .nativeTooltip(mode.rawValue)
                .plainButtonAccessibility(AccessibilityID.renderMode(mode.rawValue))
            }

            // Side chain display in ribbon mode
            if viewModel.workspace.renderMode == .ribbon {
                let scActive = viewModel.workspace.sideChainDisplay != .none
                Menu {
                    ForEach(SideChainDisplay.allCases, id: \.self) { mode in
                        Button(action: {
                            viewModel.workspace.sideChainDisplay = mode
                            viewModel.pushToRenderer()
                        }) {
                            HStack {
                                Text(mode.rawValue)
                                if viewModel.workspace.sideChainDisplay == mode {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    Image(systemName: viewModel.workspace.sideChainDisplay.icon)
                        .font(.callout)
                        .frame(width: 28, height: 28)
                        .background(scActive ? Color.purple.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(scActive ? Color.purple.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: scActive ? 1 : 0.5)
                        )
                }
                .menuStyle(.borderlessButton)
                .frame(width: 34)
                .foregroundStyle(scActive ? .purple : .secondary)
                .nativeTooltip("Side chains: \(viewModel.workspace.sideChainDisplay.rawValue)")
            }

            Divider()
                .frame(height: 24)
                .padding(.horizontal, 2)

            // Hydrogen toggle
            let hActive = viewModel.workspace.showHydrogens
            let hAvailable = viewModel.proteinHasHydrogens
            Button(action: { viewModel.toggleHydrogens() }) {
                Text("H")
                    .font(.callout.weight(.bold).monospaced())
                    .frame(width: 28, height: 28)
                    .background(hActive && hAvailable ? Color.accentColor.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(hActive && hAvailable ? Color.accentColor.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: hActive && hAvailable ? 1 : 0.5)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(hAvailable ? (hActive ? .primary : .secondary) : .tertiary)
            .disabled(!hAvailable)
            .nativeTooltip(hAvailable ? (hActive ? "Hide hydrogens" : "Show hydrogens") : "No hydrogens — add them in Preparation")
            .plainButtonAccessibility(AccessibilityID.renderHydrogens)

            // Color scheme picker (ligand focus coloring)
            let isLigandFocus = viewModel.workspace.colorScheme == .ligandFocus
            Menu {
                ForEach(WorkspaceState.MoleculeColorScheme.allCases, id: \.self) { scheme in
                    Button(action: {
                        viewModel.workspace.colorScheme = scheme
                        viewModel.pushToRenderer()
                    }) {
                        HStack {
                            Text(scheme.rawValue)
                            if viewModel.workspace.colorScheme == scheme {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                Image(systemName: "paintpalette")
                    .font(.body)
                    .frame(width: 28, height: 28)
                    .background(isLigandFocus ? Color.yellow.opacity(0.3) : Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(isLigandFocus ? Color.yellow.opacity(0.5) : Color(nsColor: .separatorColor), lineWidth: isLigandFocus ? 1 : 0.5)
                    )
            }
            .menuStyle(.borderlessButton)
            .frame(width: 34)
            .foregroundStyle(isLigandFocus ? .yellow : .secondary)
            .nativeTooltip("Color scheme: \(viewModel.workspace.colorScheme.rawValue)")

            // Molecular surface toggle
            let surfActive = viewModel.workspace.showSurface
            Button(action: { viewModel.toggleSurface() }) {
                Image(systemName: "drop.halffull")
                    .font(.body)
                    .frame(width: 28, height: 28)
                    .background(surfActive ? Color.accentColor.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(surfActive ? Color.accentColor.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: surfActive ? 1 : 0.5)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(surfActive ? .primary : .secondary)
            .disabled(viewModel.molecules.protein == nil || viewModel.workspace.isGeneratingSurface)
            .nativeTooltip(surfActive ? "Hide molecular surface" : "Show molecular surface")
            .plainButtonAccessibility(AccessibilityID.renderSurface)

            // Surface color mode picker and opacity (only when surface is visible)
            if viewModel.workspace.showSurface {
                let scmActive = viewModel.workspace.surfaceColorMode != .uniform
                Menu {
                    ForEach(SurfaceColorMode.allCases, id: \.self) { mode in
                        Button(action: { viewModel.setSurfaceColorMode(mode) }) {
                            HStack {
                                Text(mode.rawValue)
                                if viewModel.workspace.surfaceColorMode == mode {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    Image(systemName: surfaceColorIcon)
                        .font(.callout)
                        .frame(width: 28, height: 28)
                        .background(scmActive ? Color.yellow.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(scmActive ? Color.yellow.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: scmActive ? 1 : 0.5)
                        )
                }
                .menuStyle(.borderlessButton)
                .frame(width: 34)
                .foregroundStyle(scmActive ? .yellow : .secondary)
                .nativeTooltip("Surface coloring: \(viewModel.workspace.surfaceColorMode.rawValue)")

                // Surface opacity slider
                Slider(value: Binding(
                    get: { viewModel.workspace.surfaceOpacity },
                    set: { viewModel.workspace.surfaceOpacity = $0; viewModel.renderer?.surfaceOpacity = $0 }
                ), in: 0.1...1.0)
                .frame(width: 60)
                .controlSize(.small)
                .nativeTooltip("Surface opacity: \(Int(viewModel.workspace.surfaceOpacity * 100))%")

                // Probe radius control (affects channel/pocket size visibility)
                Menu {
                    Button("Water (1.4 Å)") { viewModel.workspace.surfaceProbeRadius = 1.4; viewModel.regenerateSurface() }
                    Button("Small molecule (1.8 Å)") { viewModel.workspace.surfaceProbeRadius = 1.8; viewModel.regenerateSurface() }
                    Button("Ligand access (2.2 Å)") { viewModel.workspace.surfaceProbeRadius = 2.2; viewModel.regenerateSurface() }
                    Button("Large pocket (3.0 Å)") { viewModel.workspace.surfaceProbeRadius = 3.0; viewModel.regenerateSurface() }
                    Divider()
                    Button("Fine grid (0.3 Å)") { viewModel.workspace.surfaceGridSpacing = 0.3; viewModel.regenerateSurface() }
                    Button("Normal grid (0.5 Å)") { viewModel.workspace.surfaceGridSpacing = 0.5; viewModel.regenerateSurface() }
                    Button("Coarse grid (0.7 Å)") { viewModel.workspace.surfaceGridSpacing = 0.7; viewModel.regenerateSurface() }
                } label: {
                    Image(systemName: "scope")
                        .font(.subheadline)
                        .foregroundStyle(viewModel.workspace.surfaceProbeRadius != 1.4 ? .cyan : .secondary)
                }
                .menuStyle(.borderlessButton)
                .frame(width: 22)
                .nativeTooltip("Probe: \(String(format: "%.1f", viewModel.workspace.surfaceProbeRadius)) Å, Grid: \(String(format: "%.1f", viewModel.workspace.surfaceGridSpacing)) Å")
            }

            // Lighting toggle
            let lightActive = viewModel.workspace.useDirectionalLighting
            Button(action: { viewModel.toggleLighting() }) {
                Image(systemName: lightActive ? "sun.max.fill" : "sun.min")
                    .font(.body)
                    .frame(width: 28, height: 28)
                    .background(lightActive ? Color.accentColor.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(lightActive ? Color.accentColor.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: lightActive ? 1 : 0.5)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(lightActive ? .primary : .secondary)
            .nativeTooltip(lightActive ? "Uniform lighting" : "Directional lighting")
            .plainButtonAccessibility(AccessibilityID.renderLighting)

            Divider()
                .frame(height: 24)
                .padding(.horizontal, 2)

            // Z-slab clipping toggle
            let clipActive = viewModel.workspace.enableClipping
            Button(action: {
                viewModel.workspace.enableClipping.toggle()
                syncClipping()
            }) {
                Image(systemName: "scissors")
                    .font(.body)
                    .frame(width: 28, height: 28)
                    .background(clipActive ? Color.orange.opacity(0.5) : Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(clipActive ? Color.orange.opacity(0.6) : Color(nsColor: .separatorColor), lineWidth: clipActive ? 1 : 0.5)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(clipActive ? .orange : .secondary)
            .nativeTooltip("Z-slab clipping")
            .plainButtonAccessibility(AccessibilityID.renderClipping)

            if viewModel.workspace.enableClipping {
                // Thickness / Offset sliders (object-space slab)
                VStack(spacing: 2) {
                    Slider(value: Binding(
                        get: { viewModel.workspace.slabThickness },
                        set: { viewModel.workspace.slabThickness = $0; syncClipping() }
                    ), in: 2...40)
                    .frame(width: 90)
                    .controlSize(.small)

                    Slider(value: Binding(
                        get: { viewModel.workspace.slabOffset },
                        set: { viewModel.workspace.slabOffset = $0; syncClipping() }
                    ), in: -20...20)
                    .frame(width: 90)
                    .controlSize(.small)
                }

                Text(String(format: "%.0f \u{00C5}", viewModel.workspace.slabThickness))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }

            // Camera controls (moved from toolbar)
            Divider()
                .frame(height: 24)
                .padding(.horizontal, 2)

            cameraButtons

            Divider()
                .frame(height: 24)
                .padding(.horizontal, 2)

            // Current mode label
            Text(viewModel.workspace.renderMode.rawValue)
                .font(.subheadline.weight(.medium).monospaced())
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(nsColor: .separatorColor).opacity(0.3))
                .clipShape(Capsule())
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
    }

    private var cameraButtons: some View {
        HStack(spacing: 4) {
            Button(action: { viewModel.fitToView() }) {
                Image(systemName: "viewfinder")
                    .font(.callout)
                    .frame(width: 28, height: 28)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .nativeTooltip("Fit to view (Space)")
            .plainButtonAccessibility(AccessibilityID.renderFitToView)

            Button(action: { viewModel.fitToLigand() }) {
                Image(systemName: "scope")
                    .font(.callout)
                    .frame(width: 28, height: 28)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .disabled(viewModel.molecules.ligand == nil)
            .nativeTooltip("Center on ligand")
            .plainButtonAccessibility(AccessibilityID.renderFitToLigand)

            Button(action: { viewModel.fitToProtein() }) {
                Image(systemName: "atom")
                    .font(.callout)
                    .frame(width: 28, height: 28)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .disabled(viewModel.molecules.protein == nil)
            .nativeTooltip("Center on protein")
            .plainButtonAccessibility(AccessibilityID.renderFitToProtein)

            Button(action: { viewModel.renderer?.camera.reset(); viewModel.pushToRenderer() }) {
                Image(systemName: "arrow.counterclockwise")
                    .font(.callout)
                    .frame(width: 28, height: 28)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .nativeTooltip("Reset camera")
            .plainButtonAccessibility(AccessibilityID.renderResetCamera)
        }
    }

    private func syncClipping() {
        viewModel.renderer?.enableClipping = viewModel.workspace.enableClipping
        viewModel.renderer?.clipNearZ = viewModel.workspace.clipNearZ
        viewModel.renderer?.clipFarZ = viewModel.workspace.clipFarZ
        viewModel.renderer?.slabHalfThickness = viewModel.workspace.slabThickness / 2.0
        viewModel.renderer?.slabOffset = viewModel.workspace.slabOffset
        // When the user toggles clipping from the central banner without a
        // pocket focused, release any stale slab anchor so the slab follows
        // the camera's look-at point (Renderer falls back to camera.target).
        if !viewModel.workspace.enableClipping {
            viewModel.renderer?.slabCenter = nil
        }
        viewModel.renderer?.setNeedsRedraw()
    }

    private func badge(_ name: String, icon: String, color: Color, count: Int) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.subheadline)
            Text(name)
                .font(.callout.weight(.medium))
            Text("\(count) atoms")
                .font(.footnote.monospaced())
                .foregroundStyle(color.opacity(0.7))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(color.opacity(0.15))
        .foregroundStyle(color)
        .clipShape(Capsule())
    }

    private var surfaceColorIcon: String {
        switch viewModel.workspace.surfaceColorMode {
        case .uniform: "paintpalette"
        case .esp: "bolt.fill"
        case .hydrophobicity: "drop.fill"
        case .pharmacophore: "circle.hexagongrid"
        }
    }
}
