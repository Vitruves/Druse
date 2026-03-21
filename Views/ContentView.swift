import SwiftUI
import MetalKit

struct ContentView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var showInspector = true
    @State private var showConsole = true
    @State private var consoleHeight: CGFloat = 140
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
                            if !viewModel.currentInteractions.isEmpty {
                                InteractionLegendView(interactions: viewModel.currentInteractions)
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

            // Bottom console
            LogConsoleView(isExpanded: $showConsole, consoleHeight: $consoleHeight)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: { viewModel.fitToView() }) {
                    Label("Fit to View", systemImage: "viewfinder")
                }
                .help("Fit molecule to view (Space)")

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
                Color(nsColor: NSColor(red: 0.08, green: 0.09, blue: 0.12, alpha: 1.0))
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
            if let prot = viewModel.protein {
                badge(prot.name, icon: "cube", color: .cyan, count: prot.atomCount)
            }
            if let lig = viewModel.ligand {
                badge(lig.name, icon: "hexagon", color: .green, count: lig.atomCount)
            }
            Spacer()
        }
    }

    // MARK: - Render Controls (bottom overlay)

    @ViewBuilder
    private var renderControls: some View {
        HStack(spacing: 6) {
            // Render mode buttons
            ForEach(RenderMode.allCases, id: \.self) { mode in
                Button(action: { viewModel.setRenderMode(mode) }) {
                    Image(systemName: mode.icon)
                        .font(.system(size: 12))
                        .frame(width: 28, height: 28)
                        .background(viewModel.renderMode == mode ? Color.accentColor.opacity(0.3) : Color.clear)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .foregroundStyle(viewModel.renderMode == mode ? .primary : .secondary)
                .help(mode.rawValue)
            }

            Divider()
                .frame(height: 20)

            // Hydrogen toggle
            Button(action: { viewModel.toggleHydrogens() }) {
                Text("H")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .frame(width: 28, height: 28)
                    .background(viewModel.showHydrogens ? Color.accentColor.opacity(0.3) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .foregroundStyle(viewModel.showHydrogens ? .primary : .secondary)
            .help(viewModel.showHydrogens ? "Hide hydrogens" : "Show hydrogens")

            // Molecular surface toggle
            Button(action: { viewModel.toggleSurface() }) {
                Image(systemName: "drop.halffull")
                    .font(.system(size: 12))
                    .frame(width: 28, height: 28)
                    .background(viewModel.showSurface ? Color.accentColor.opacity(0.3) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .foregroundStyle(viewModel.showSurface ? .primary : .secondary)
            .disabled(viewModel.protein == nil || viewModel.isGeneratingSurface)
            .help(viewModel.showSurface ? "Hide molecular surface" : "Show molecular surface")

            // ESP coloring toggle (only when surface is visible)
            if viewModel.showSurface {
                Button(action: { viewModel.toggleESPColoring() }) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 12))
                        .frame(width: 28, height: 28)
                        .background(viewModel.surfaceColorByESP ? Color.yellow.opacity(0.3) : Color.clear)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .foregroundStyle(viewModel.surfaceColorByESP ? .yellow : .secondary)
                .help(viewModel.surfaceColorByESP ? "Uniform surface color" : "Color by electrostatic potential")
            }

            // Lighting toggle
            Button(action: { viewModel.toggleLighting() }) {
                Image(systemName: viewModel.useDirectionalLighting ? "sun.max.fill" : "sun.min")
                    .font(.system(size: 12))
                    .frame(width: 28, height: 28)
                    .background(viewModel.useDirectionalLighting ? Color.accentColor.opacity(0.3) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .foregroundStyle(viewModel.useDirectionalLighting ? .primary : .secondary)
            .help(viewModel.useDirectionalLighting ? "Uniform lighting" : "Directional lighting")

            Divider()
                .frame(height: 20)

            // Z-slab clipping toggle
            Button(action: {
                viewModel.enableClipping.toggle()
                syncClipping()
            }) {
                Image(systemName: "scissors")
                    .font(.system(size: 12))
                    .frame(width: 28, height: 28)
                    .background(viewModel.enableClipping ? Color.orange.opacity(0.3) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .foregroundStyle(viewModel.enableClipping ? .orange : .secondary)
            .help("Z-slab clipping")

            if viewModel.enableClipping {
                // Near/Far sliders (compact)
                VStack(spacing: 1) {
                    Slider(value: Binding(
                        get: { viewModel.clipNearZ },
                        set: { viewModel.clipNearZ = $0; syncClipping() }
                    ), in: 0...200)
                    .frame(width: 80)
                    .controlSize(.mini)

                    Slider(value: Binding(
                        get: { viewModel.clipFarZ },
                        set: { viewModel.clipFarZ = $0; syncClipping() }
                    ), in: 0...200)
                    .frame(width: 80)
                    .controlSize(.mini)
                }

                Text(String(format: "%.0f–%.0f", viewModel.clipNearZ, viewModel.clipFarZ))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            // Current mode label
            Text(viewModel.renderMode.rawValue)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func syncClipping() {
        viewModel.renderer?.enableClipping = viewModel.enableClipping
        viewModel.renderer?.clipNearZ = viewModel.clipNearZ
        viewModel.renderer?.clipFarZ = viewModel.clipFarZ
    }

    private func badge(_ name: String, icon: String, color: Color, count: Int) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 9))
            Text(name)
                .font(.system(size: 10, weight: .medium))
            Text("\(count)")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.15))
        .foregroundStyle(color)
        .clipShape(Capsule())
    }
}
