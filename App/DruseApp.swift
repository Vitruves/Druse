// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import MetalKit

/// Launched from App/main.swift, which routes `--headless` invocations to HeadlessRunner instead.
struct DruseApp: App {
    @NSApplicationDelegateAdaptor(DruseAppDelegate.self) private var appDelegate
    @State private var viewModel: AppViewModel
    /// The single DruseScript session, shown either in the main window's bottom panel or in
    /// its own Terminal window, so scrollback, history and a running command follow it.
    @State private var consoleSession: ConsoleSession
    @AppStorage("appTheme") private var appTheme: String = AppTheme.auto.rawValue

    /// Applies the CPU core limit before any work starts (headless runs apply it in HeadlessRunner).
    init() {
        AppSettings.shared.applyThreadLimits()
        let viewModel = AppViewModel()
        _viewModel = State(initialValue: viewModel)
        _consoleSession = State(initialValue: ConsoleSession(viewModel: viewModel))
    }

    private var selectedColorScheme: ColorScheme? {
        (AppTheme(rawValue: appTheme) ?? .auto).colorScheme
    }

    var body: some Scene {
        Window("Druse", id: "main") {
            ContentView()
                .environment(viewModel)
                .environment(consoleSession)
                .preferredColorScheme(selectedColorScheme)
                .onAppear {
                    appDelegate.viewModel = viewModel
                    initializeRenderer()
                    // Speed up tooltip display (default macOS delay is ~1.5 seconds)
                    UserDefaults.standard.set(300, forKey: "NSInitialToolTipDelay")
                }
                .frame(minWidth: DS.Size.windowMin.width, minHeight: DS.Size.windowMin.height - DS.Size.toolbarHeight)
        }
        .windowToolbarStyle(.unified(showsTitle: false))
        .defaultSize(DS.Size.windowDefault)
        .defaultLaunchBehavior(.automatic)
        .commands {
            DruseCommands(viewModel: viewModel, consoleSession: consoleSession)
        }

        Settings {
            SettingsView()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
        }

        Window("Ligand Database", id: "ligand-database") {
            LigandDatabaseWindow()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
                .frame(minWidth: 1040, minHeight: 680)
        }
        .defaultSize(width: 1280, height: 820)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        Window("Druse Help", id: "druse-help") {
            DruseHelpView()
                .preferredColorScheme(selectedColorScheme)
        }
        .defaultSize(width: 960, height: 820)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        Window("Report Bug", id: "bug-report") {
            BugReportWindow()
                .preferredColorScheme(selectedColorScheme)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 720, height: 780)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        // 2D protein–ligand interaction view
        Window("Interaction Diagram", id: "interaction-diagram") {
            InteractionDiagramWindow()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
        }
        .defaultSize(width: 1050, height: 800)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        // The DruseScript terminal, detached from the main window's bottom panel
        Window("Druse Terminal", id: ConsoleWindow.id) {
            ConsoleWindow()
                .environment(consoleSession)
                .preferredColorScheme(selectedColorScheme)
                .frame(minWidth: 480, minHeight: 240)
        }
        .defaultSize(width: 900, height: 560)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        // Full pose analysis, interaction diagrams, correlation
        Window("Results Database", id: "results-database") {
            ResultsDatabaseWindow()
                .environment(viewModel)
                .preferredColorScheme(selectedColorScheme)
                .frame(minWidth: 1000, minHeight: 640)
        }
        .defaultSize(width: 1320, height: 820)
        .windowStyle(.titleBar)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)
    }

    /// The single Save Project implementation (File menu and toolbar).
    @MainActor
    static func saveProject(_ viewModel: AppViewModel) {
        let panel = NSSavePanel()
        panel.title = "Save Druse Project"
        panel.allowedContentTypes = [.folder]
        panel.nameFieldStringValue = "\(viewModel.molecules.protein?.name ?? "Untitled").druse"
        panel.canCreateDirectories = true
        panel.directoryURL = AppSettings.shared.defaultProjectDirectoryURL
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

        // Theme is synced by ContentView (onAppear / colorScheme changes).
        renderer.backgroundOpacity = viewModel.workspace.backgroundOpacity
        renderer.surfaceOpacity = viewModel.workspace.surfaceOpacity

        viewModel.renderer = renderer
        viewModel.log.success("Metal renderer initialized (\(device.name))", category: .render)
        viewModel.log.info("GPU: \(device.name), Unified Memory: \(device.hasUnifiedMemory)", category: .system)
        if let logURL = ActivityLog.shared.currentLogFileURL {
            viewModel.log.info("Log file: \(logURL.path)", category: .system)
        }

        Self.initializeComputeServices(viewModel)
    }

    /// Computation services shared by the GUI and headless mode (no renderer needed):
    /// ML models and GPU-accelerated GFN2-xTB. The CPU thread limit is already applied by then.
    @MainActor
    static func initializeComputeServices(_ viewModel: AppViewModel) {
        let settings = AppSettings.shared
        viewModel.log.info("CPU thread limit: \(settings.appliedThreadLimit) of \(AppSettings.availableCores) cores"
                           + (settings.sessionCoreOverride != nil ? " (--cores)" : ""), category: .system)

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

// MARK: - Menu commands

private struct DruseCommands: Commands {
    let viewModel: AppViewModel
    let consoleSession: ConsoleSession
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @FocusedValue(\.selectedStep) private var selectedStep
    @AppStorage("ui.navigatorOpen") private var showNavigator = true
    @AppStorage("ui.inspectorOpen") private var showInspector = false
    @AppStorage("ui.consoleOpen") private var showConsole = false
    @AppStorage(BottomPanel.storageKey) private var bottomPanel = BottomPanel.terminal

    private var currentStep: SidebarTab { selectedStep?.wrappedValue ?? .search }

    var body: some Commands {
        // File ▸ New Window would create a second window sharing one renderer.
        CommandGroup(replacing: .newItem) { }

        CommandGroup(after: .newItem) {
            Button("Open…") { viewModel.importFile() }
                .keyboardShortcut("o", modifiers: .command)
            Button("Open Project…") { viewModel.openProject() }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            Button("Save Project") { DruseApp.saveProject(viewModel) }
                .keyboardShortcut("s", modifiers: .command)
            Divider()
            Button("Import Ligand Database…") {
                viewModel.ligandDB.load()
            }
            Button("Export Ligand Database…") {
                viewModel.ligandDB.save()
                viewModel.log.success("Saved \(viewModel.ligandDB.count) ligands", category: .molecule)
            }
        }

        CommandMenu("Workflow") {
            ForEach(SidebarTab.allCases, id: \.self) { tab in
                // Tools (Sequence) are set apart from the seven pipeline steps.
                if !tab.isPipelineStep { Divider() }
                Button { tab.request(in: viewModel) } label: {
                    Label(tab.isPipelineStep ? "\(tab.shortcutNumber). \(tab.rawValue)" : tab.rawValue,
                          systemImage: tab.icon)
                }
                .keyboardShortcut(KeyEquivalent(Character(String(tab.shortcutNumber))), modifiers: .command)
            }
            Divider()
            Button("Next Step") { currentStep.nextStep?.request(in: viewModel) }
                .keyboardShortcut("]", modifiers: .command)
                .disabled(currentStep.nextStep == nil)
            Button("Previous Step") { currentStep.previousStep?.request(in: viewModel) }
                .keyboardShortcut("[", modifiers: .command)
                .disabled(currentStep.previousStep == nil)
        }

        CommandGroup(after: .toolbar) {
            Divider()
            ForEach(Array(RenderMode.allCases.enumerated()), id: \.element) { index, mode in
                Button(mode.rawValue) { viewModel.setRenderMode(mode) }
                    .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: [.control, .command])
            }
            Divider()
            Button("Toggle Hydrogens") { viewModel.toggleHydrogens() }
                .keyboardShortcut("h", modifiers: [.control, .command])
            Button("Fit to View") { viewModel.fitToView() }
                .keyboardShortcut("0", modifiers: .command)
            Divider()
            Button(showNavigator ? "Hide Navigator" : "Show Navigator") { showNavigator.toggle() }
                .keyboardShortcut("s", modifiers: [.control, .command])
            Button(showInspector ? "Hide Inspector" : "Show Inspector") { showInspector.toggle() }
                .keyboardShortcut("0", modifiers: [.option, .command])
            if consoleSession.isDetached {
                Button("Close Terminal Window") { dismissWindow(id: ConsoleWindow.id) }
                    .keyboardShortcut("`", modifiers: .control)
            } else {
                panelToggle(.terminal)
                    .keyboardShortcut("`", modifiers: .control)
            }
            Button("Open Terminal in Separate Window") { openWindow(id: ConsoleWindow.id) }
            panelToggle(.logs)
                .keyboardShortcut("y", modifiers: [.shift, .command])
            Divider()
            Button("Ligand Database") { openWindow(id: "ligand-database") }
                .keyboardShortcut("l", modifiers: .command)
            Button("Results Database") { openWindow(id: "results-database") }
                .keyboardShortcut("r", modifiers: [.command, .shift])
        }

        CommandGroup(replacing: .help) {
            Button("Druse Help") { openWindow(id: "druse-help") }
                .keyboardShortcut("?", modifiers: .command)
            Divider()
            Button("Report Bug…") { openWindow(id: "bug-report") }
        }
    }

    /// View ▸ Show/Hide Terminal or Logs: shows that surface of the bottom panel, or hides the
    /// panel when the surface is already visible.
    private func panelToggle(_ surface: BottomPanel) -> some View {
        let isVisible = showConsole && bottomPanel == surface
        return Button(isVisible ? "Hide \(surface.title)" : "Show \(surface.title)") {
            if isVisible {
                showConsole = false
            } else {
                bottomPanel = surface
                showConsole = true
            }
        }
    }
}

/// Receives files opened from Finder or `open -a Druse …`: `.druse` scripts are queued for the
/// terminal. Files can arrive before the main window exists, so they wait for the view model.
@MainActor
final class DruseAppDelegate: NSObject, NSApplicationDelegate {
    weak var viewModel: AppViewModel? {
        didSet { flushPendingScripts() }
    }
    private var pendingScripts: [URL] = []

    func application(_ application: NSApplication, open urls: [URL]) {
        pendingScripts += urls.filter { $0.pathExtension.lowercased() == "druse" }
        flushPendingScripts()
    }

    private func flushPendingScripts() {
        guard let viewModel, !pendingScripts.isEmpty else { return }
        viewModel.workspace.pendingScripts += pendingScripts
        pendingScripts.removeAll()
    }
}
