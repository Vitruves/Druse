import SwiftUI
import MetalKit

struct ContentView: View {
    @Environment(AppViewModel.self) private var viewModel
    @AppStorage("appTheme") private var appTheme: String = AppTheme.dark.rawValue
    @Environment(\.colorScheme) private var colorScheme
    @State private var showInspector = true
    @State private var showConsole = true
    @State private var consoleHeight: CGFloat = 54
    @State private var isDragTargeted = false

    var body: some View {
        VStack(spacing: 0) {
            // Main content area
            HStack(spacing: 0) {
                // Left sidebar
                SidebarView()

                Divider()

                // Center: Metal viewport with overlays
                ZStack {
                    metalViewport

                    // Top overlay: badges left, mode label right
                    VStack {
                        HStack(alignment: .top) {
                            moleculeBadges
                            Spacer()
                            // Interaction legend (top-right, only when interactions present)
                            if !viewModel.docking.currentInteractions.isEmpty {
                                InteractionLegendView(interactions: viewModel.docking.currentInteractions)
                            }
                        }
                        .padding(10)

                        Spacer()

                        // Bottom overlay: render controls
                        renderControls
                            .padding(10)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if showInspector {
                    Divider()
                    InspectorPanel()
                }
            }

            // Bottom status strip + expandable console
            StatusStripView(showConsole: $showConsole, consoleHeight: $consoleHeight)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: { viewModel.fitToView() }) {
                    Label("Fit to View", systemImage: "viewfinder")
                }
                .help("Fit molecule to view (Space)")

                Button(action: { viewModel.fitToLigand() }) {
                    Label("Center to Ligand", systemImage: "scope")
                }
                .help("Center camera on ligand")
                .disabled(viewModel.molecules.ligand == nil)

                Button(action: { viewModel.renderer?.camera.reset(); viewModel.pushToRenderer() }) {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .help("Reset camera")

                Divider()

                Toggle(isOn: $showInspector) {
                    Label("Inspector", systemImage: "sidebar.right")
                }
                .help("Toggle inspector panel")

                Toggle(isOn: $showConsole) {
                    Label("Console", systemImage: "terminal")
                }
                .help("Toggle console")
            }
        }
        .onChange(of: ActivityLog.shared.entries.count) { _, _ in
            // Auto-expand console on error
            if let last = ActivityLog.shared.entries.last, last.level == .error, !showConsole {
                withAnimation(.easeInOut(duration: 0.2)) { showConsole = true }
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
                                .font(.system(size: 32))
                                .foregroundStyle(.secondary)
                            Text("Drop to import")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .allowsHitTesting(false)
            }
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
                    viewModel.focusOnAtom(atomIndex)
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
                        .font(.system(size: 40))
                        .foregroundStyle(.tertiary)
                    Text("Initializing Metal...")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text("Load a protein from the Search tab or drag a PDB file here to begin")
                        .font(.system(size: 11))
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
        HStack(spacing: 6) {
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

    @ViewBuilder
    private var renderControls: some View {
        HStack(spacing: 4) {
            // Render mode buttons
            ForEach(RenderMode.allCases, id: \.self) { mode in
                let isActive = viewModel.workspace.renderMode == mode
                Button(action: { viewModel.setRenderMode(mode) }) {
                    Image(systemName: mode.icon)
                        .font(.system(size: 14))
                        .frame(width: 34, height: 34)
                        .background(isActive ? Color.accentColor.opacity(0.5) : Color.white.opacity(0.05))
                        .clipShape(RoundedRectangle(cornerRadius: 7))
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .stroke(isActive ? Color.accentColor.opacity(0.6) : Color.clear, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
                .foregroundStyle(isActive ? .primary : .secondary)
                .help(mode.rawValue)
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
                        .font(.system(size: 14))
                        .frame(width: 34, height: 34)
                        .background(scActive ? Color.purple.opacity(0.5) : Color.white.opacity(0.05))
                        .clipShape(RoundedRectangle(cornerRadius: 7))
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .stroke(scActive ? Color.purple.opacity(0.6) : Color.clear, lineWidth: 1)
                        )
                }
                .menuStyle(.borderlessButton)
                .frame(width: 34)
                .foregroundStyle(scActive ? .purple : .secondary)
                .help("Side chains: \(viewModel.workspace.sideChainDisplay.rawValue)")
            }

            Divider()
                .frame(height: 24)
                .padding(.horizontal, 2)

            // Hydrogen toggle
            let hActive = viewModel.workspace.showHydrogens
            let hAvailable = viewModel.proteinHasHydrogens
            Button(action: { viewModel.toggleHydrogens() }) {
                Text("H")
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .frame(width: 34, height: 34)
                    .background(hActive && hAvailable ? Color.accentColor.opacity(0.5) : Color.white.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .stroke(hActive && hAvailable ? Color.accentColor.opacity(0.6) : Color.clear, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(hAvailable ? (hActive ? .primary : .secondary) : .tertiary)
            .disabled(!hAvailable)
            .help(hAvailable ? (hActive ? "Hide hydrogens" : "Show hydrogens") : "No hydrogens — add them in Preparation")

            // Molecular surface toggle
            let surfActive = viewModel.workspace.showSurface
            Button(action: { viewModel.toggleSurface() }) {
                Image(systemName: "drop.halffull")
                    .font(.system(size: 14))
                    .frame(width: 34, height: 34)
                    .background(surfActive ? Color.accentColor.opacity(0.5) : Color.white.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .stroke(surfActive ? Color.accentColor.opacity(0.6) : Color.clear, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(surfActive ? .primary : .secondary)
            .disabled(viewModel.molecules.protein == nil || viewModel.workspace.isGeneratingSurface)
            .help(surfActive ? "Hide molecular surface" : "Show molecular surface")

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
                        .font(.system(size: 14))
                        .frame(width: 34, height: 34)
                        .background(scmActive ? Color.yellow.opacity(0.5) : Color.white.opacity(0.05))
                        .clipShape(RoundedRectangle(cornerRadius: 7))
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .stroke(scmActive ? Color.yellow.opacity(0.6) : Color.clear, lineWidth: 1)
                        )
                }
                .menuStyle(.borderlessButton)
                .frame(width: 34)
                .foregroundStyle(scmActive ? .yellow : .secondary)
                .help("Surface coloring: \(viewModel.workspace.surfaceColorMode.rawValue)")

                // Surface opacity slider
                Slider(value: Binding(
                    get: { viewModel.workspace.surfaceOpacity },
                    set: { viewModel.workspace.surfaceOpacity = $0; viewModel.renderer?.surfaceOpacity = $0 }
                ), in: 0.1...1.0)
                .frame(width: 60)
                .controlSize(.small)
                .help("Surface opacity: \(Int(viewModel.workspace.surfaceOpacity * 100))%")
            }

            // Lighting toggle
            let lightActive = viewModel.workspace.useDirectionalLighting
            Button(action: { viewModel.toggleLighting() }) {
                Image(systemName: lightActive ? "sun.max.fill" : "sun.min")
                    .font(.system(size: 14))
                    .frame(width: 34, height: 34)
                    .background(lightActive ? Color.accentColor.opacity(0.5) : Color.white.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .stroke(lightActive ? Color.accentColor.opacity(0.6) : Color.clear, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(lightActive ? .primary : .secondary)
            .help(lightActive ? "Uniform lighting" : "Directional lighting")

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
                    .font(.system(size: 14))
                    .frame(width: 34, height: 34)
                    .background(clipActive ? Color.orange.opacity(0.5) : Color.white.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .stroke(clipActive ? Color.orange.opacity(0.6) : Color.clear, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .foregroundStyle(clipActive ? .orange : .secondary)
            .help("Z-slab clipping")

            if viewModel.workspace.enableClipping {
                // Near/Far sliders
                VStack(spacing: 2) {
                    Slider(value: Binding(
                        get: { viewModel.workspace.clipNearZ },
                        set: { viewModel.workspace.clipNearZ = $0; syncClipping() }
                    ), in: 0...200)
                    .frame(width: 90)
                    .controlSize(.small)

                    Slider(value: Binding(
                        get: { viewModel.workspace.clipFarZ },
                        set: { viewModel.workspace.clipFarZ = $0; syncClipping() }
                    ), in: 0...200)
                    .frame(width: 90)
                    .controlSize(.small)
                }

                Text(String(format: "%.0f\u{2013}%.0f \u{00C5}", viewModel.workspace.clipNearZ, viewModel.workspace.clipFarZ))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            // Current mode label
            Text(viewModel.workspace.renderMode.rawValue)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.white.opacity(0.05))
                .clipShape(Capsule())
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func syncClipping() {
        viewModel.renderer?.enableClipping = viewModel.workspace.enableClipping
        viewModel.renderer?.clipNearZ = viewModel.workspace.clipNearZ
        viewModel.renderer?.clipFarZ = viewModel.workspace.clipFarZ
    }

    private func badge(_ name: String, icon: String, color: Color, count: Int) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 11))
            Text(name)
                .font(.system(size: 12, weight: .medium))
            Text("\(count) atoms")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(color.opacity(0.7))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
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
