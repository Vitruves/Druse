import SwiftUI
import MetalKit

@main
struct DruseApp: App {
    @State private var viewModel = AppViewModel()
    @Environment(\.openWindow) private var openWindow
    @AppStorage("appTheme") private var appTheme: String = AppTheme.dark.rawValue

    private var selectedColorScheme: ColorScheme? {
        (AppTheme(rawValue: appTheme) ?? .dark).colorScheme
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
                .onAppear {
                    initializeRenderer()
                    // Speed up tooltip display (default macOS delay is ~1.5 seconds)
                    UserDefaults.standard.set(300, forKey: "NSInitialToolTipDelay")
                }
                .frame(minWidth: 1100, minHeight: 700)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1400, height: 900)
        .defaultLaunchBehavior(.automatic)
        .commands {
            // File menu
            CommandGroup(after: .newItem) {
                Button("Open...") { viewModel.importFile() }
                    .keyboardShortcut("o", modifiers: .command)
                Divider()
                Button("Save Project...") { saveProject() }
                    .keyboardShortcut("s", modifiers: .command)
                Button("Open Project...") { openProject() }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                Divider()
                Button("Load Ligand Database") {
                    viewModel.ligandDB.load()
                }
                Button("Save Ligand Database") {
                    viewModel.ligandDB.save()
                    viewModel.log.success("Saved \(viewModel.ligandDB.count) ligands", category: .molecule)
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
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
                Button("Results Database") {
                    openWindow(id: "results-database")
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }

            // Help menu
            CommandGroup(replacing: .help) {
                Button("Druse Help") {
                    openWindow(id: "druse-help")
                }
                .keyboardShortcut("?", modifiers: .command)
            }
        }

        Settings {
            SettingsView()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
        }

        // Ligand Database window
        Window("Ligand Database", id: "ligand-database") {
            LigandDatabaseWindow()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
        }
        .defaultSize(width: 1200, height: 700)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        // Help window
        Window("Druse Help", id: "druse-help") {
            DruseHelpView()
                .preferredColorScheme(selectedColorScheme)
        }
        .defaultSize(width: 860, height: 680)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        // Results Database window — full pose analysis, interaction diagrams, correlation
        Window("Results Database", id: "results-database") {
            ResultsDatabaseWindow()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
        }
        .defaultSize(width: 1300, height: 800)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

    }

    private func saveProject() {
        let panel = NSSavePanel()
        panel.title = "Save Druse Project"
        panel.allowedContentTypes = [.folder]
        panel.nameFieldStringValue = "\(viewModel.molecules.protein?.name ?? "Untitled").druse"
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                do {
                    try DruseProjectIO.save(to: url, viewModel: viewModel)
                } catch {
                    viewModel.log.error("Save failed: \(error.localizedDescription)", category: .system)
                }
            }
        }
    }

    private func openProject() {
        let panel = NSOpenPanel()
        panel.title = "Open Druse Project"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowedContentTypes = [.folder]
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                do {
                    try await DruseProjectIO.load(from: url, into: viewModel)
                } catch {
                    viewModel.log.error("Load failed: \(error.localizedDescription)", category: .system)
                }
            }
        }
    }

    private let terminationObserver = NotificationCenter.default.addObserver(
        forName: NSApplication.willTerminateNotification,
        object: nil, queue: .main
    ) { _ in
        MainActor.assumeIsolated {
            ActivityLog.shared.shutdown()
        }
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

        // Apply persisted theme immediately so renderer doesn't flash dark on relaunch
        let theme = AppTheme(rawValue: appTheme) ?? .dark
        switch theme {
        case .light: renderer.themeMode = 1
        case .dark:  renderer.themeMode = 0
        case .auto:  renderer.themeMode = (selectedColorScheme == .light) ? 1 : 0
        }
        renderer.backgroundOpacity = viewModel.workspace.backgroundOpacity
        renderer.surfaceOpacity = viewModel.workspace.surfaceOpacity

        viewModel.renderer = renderer
        viewModel.log.success("Metal renderer initialized (\(device.name))", category: .render)
        viewModel.log.info("GPU: \(device.name), Unified Memory: \(device.hasUnifiedMemory)", category: .system)
        if let logURL = ActivityLog.shared.currentLogFileURL {
            viewModel.log.info("Log file: \(logURL.path)", category: .system)
        }

        // Load ML models in the background (non-blocking)
        viewModel.loadMLModels()

        // Initialize GPU-accelerated xTB if available
        if let xtbGPU = XTBMetalAccelerator() {
            XTBMetalAccelerator.shared = xtbGPU
            xtbGPU.installGPUContext()
            viewModel.log.info("GFN2-xTB Metal GPU acceleration enabled", category: .system)
        }
    }
}
