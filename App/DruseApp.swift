import SwiftUI
import MetalKit

@main
struct DruseApp: App {
    @State private var viewModel = AppViewModel()
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
                .preferredColorScheme(.dark)
                .onAppear {
                    initializeRenderer()
                }
                .frame(minWidth: 1100, minHeight: 700)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1400, height: 900)
        .commands {
            // File menu
            CommandGroup(after: .newItem) {
                Button("Open...") { viewModel.importFile() }
                    .keyboardShortcut("o", modifiers: .command)
                Divider()
                Button("Load Ligand Database") {
                    viewModel.ligandDB.load()
                }
                Button("Save Ligand Database") {
                    viewModel.ligandDB.save()
                    viewModel.log.success("Saved \(viewModel.ligandDB.count) ligands", category: .molecule)
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                Divider()
                Button("Load Caffeine") { viewModel.loadCaffeine() }
                    .keyboardShortcut("1", modifiers: [.command, .shift])
                Button("Load Ala Dipeptide") { viewModel.loadAlanineDipeptide() }
                    .keyboardShortcut("2", modifiers: [.command, .shift])
                Button("Load Both") { viewModel.loadBoth() }
                    .keyboardShortcut("3", modifiers: [.command, .shift])
            }

            // View menu additions
            CommandGroup(after: .toolbar) {
                Divider()
                Button("Ball & Stick") { viewModel.setRenderMode(.ballAndStick) }
                    .keyboardShortcut("1", modifiers: .command)
                Button("Space Filling") { viewModel.setRenderMode(.spaceFilling) }
                    .keyboardShortcut("2", modifiers: .command)
                Button("Wireframe") { viewModel.setRenderMode(.wireframe) }
                    .keyboardShortcut("3", modifiers: .command)
                Divider()
                Button("Toggle Hydrogens") { viewModel.toggleHydrogens() }
                    .keyboardShortcut("h", modifiers: .command)
                Button("Recenter") {
                    viewModel.renderer?.fitToContent()
                }
                .keyboardShortcut(" ", modifiers: [])
                Divider()
                Button("Ligand Database") {
                    openWindow(id: "ligand-database")
                }
                .keyboardShortcut("l", modifiers: .command)
            }
        }

        // Ligand Database window — shares the same viewModel (and thus LigandDatabase) as the main window
        Window("Ligand Database", id: "ligand-database") {
            LigandDatabaseWindow()
                .environment(viewModel)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1200, height: 700)
        .windowStyle(.titleBar)

    }

    private func initializeRenderer() {
        guard viewModel.renderer == nil else { return }
        guard let device = MTLCreateSystemDefaultDevice() else {
            viewModel.log.error("Metal is not supported on this device", category: .system)
            return
        }

        // Create a temporary MTKView to initialize the renderer
        let tempView = MTKView(frame: CGRect(x: 0, y: 0, width: 800, height: 600), device: device)
        tempView.colorPixelFormat = .bgra8Unorm
        tempView.depthStencilPixelFormat = .depth32Float
        tempView.sampleCount = 4

        guard let renderer = Renderer(mtkView: tempView) else {
            viewModel.log.error("Failed to initialize Metal renderer", category: .render)
            return
        }

        viewModel.renderer = renderer
        viewModel.log.success("Metal renderer initialized (\(device.name))", category: .render)
        viewModel.log.info("GPU: \(device.name), Unified Memory: \(device.hasUnifiedMemory)", category: .system)

        // Load ML models in the background (non-blocking)
        viewModel.loadMLModels()
    }
}
