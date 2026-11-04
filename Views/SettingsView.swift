// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

enum AppTheme: String, CaseIterable {
    case auto = "Auto"     // rawValues are persisted in "appTheme"; keep them stable
    case light = "Light"
    case dark = "Dark"

    var title: String {
        switch self {
        case .auto: "Automatic"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .auto: nil
        case .light: .light
        case .dark: .dark
        }
    }

    var icon: String {
        switch self {
        case .auto: "circle.lefthalf.filled"
        case .light: "sun.max"
        case .dark: "moon"
        }
    }
}

struct SettingsView: View {
    @AppStorage("appTheme") private var appTheme: String = AppTheme.auto.rawValue
    @AppStorage("ui.confirmBeforeDocking") private var confirmBeforeDocking: Bool = true
    @AppStorage("ui.docking.advancedExpanded") private var dockingAdvancedExpanded: Bool = false
    @Environment(AppViewModel.self) private var viewModel
    @Bindable private var settings = AppSettings.shared

    // Logging settings (local state synced with LogSettings)
    @AppStorage("logFileEnabled") private var logFileEnabled: Bool = true
    @AppStorage("logFileDirectory") private var logFileDirectory: String = LogSettings.defaultLogDirectory
    @AppStorage("logFileRetentionDays") private var logFileRetentionDays: Int = 7

    private var selectedTheme: AppTheme {
        AppTheme(rawValue: appTheme) ?? .auto
    }

    var body: some View {
        TabView {
            generalPane
                .tabItem { Label("General", systemImage: "gearshape") }
            viewportPane
                .tabItem { Label("Viewport", systemImage: "cube.transparent") }
            dockingPane
                .tabItem { Label("Docking", systemImage: "arrow.down.right.and.arrow.up.left") }
            loggingPane
                .tabItem { Label("Logging", systemImage: "doc.text.magnifyingglass") }
            PerformanceSettingsPane()
                .tabItem { Label("Performance", systemImage: "cpu") }
            ConsoleSettingsPane()
                .tabItem { Label("Console", systemImage: "text.and.command.macwindow") }
            CommandLineSettingsPane()
                .tabItem { Label("Command Line", systemImage: "apple.terminal") }
            ShellCompletionsSettingsPane()
                .tabItem { Label("Shell Completions", systemImage: "text.insert") }
        }
        .frame(width: Self.paneSize.width, height: Self.paneSize.height)
    }

    /// One size for every tab, so switching tabs never makes the window jump; longer panes scroll.
    static let paneSize = CGSize(width: 620, height: 520)

    // MARK: - General

    private var generalPane: some View {
        Form {
            Section {
                Picker("Appearance", selection: $appTheme) {
                    ForEach(AppTheme.allCases, id: \.rawValue) { theme in
                        Label(theme.title, systemImage: theme.icon)
                            .tag(theme.rawValue)
                    }
                }
                .pickerStyle(.segmented)
            } footer: {
                Text(themeDescription)
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                FolderSettingRow(title: "PDB download cache", path: $settings.pdbCacheDirectory,
                                 defaultPath: AppSettings.defaultPDBCacheDirectory)
                FolderSettingRow(title: "Project folder", path: $settings.defaultProjectDirectory,
                                 defaultPath: AppSettings.defaultProjectDirectoryPath)
            } header: {
                Text("Folders")
            } footer: {
                Text("Downloaded PDB entries are kept for 7 days. Save Project starts in the project folder.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .settingsPane()
    }

    // MARK: - Viewport

    private var viewportPane: some View {
        Form {
            Section {
                LabeledContent("Gradient intensity") {
                    Slider(value: Binding(
                        get: { viewModel.workspace.backgroundOpacity },
                        set: {
                            viewModel.workspace.backgroundOpacity = $0
                            viewModel.renderer?.backgroundOpacity = $0
                        }
                    ), in: 0...1)
                }
                LabeledContent("Surface opacity") {
                    Slider(value: Binding(
                        get: { viewModel.workspace.surfaceOpacity },
                        set: {
                            viewModel.workspace.surfaceOpacity = $0
                            viewModel.renderer?.surfaceOpacity = $0
                        }
                    ), in: 0.1...1)
                }
            } header: {
                Text("Background & Surface")
            } footer: {
                Text("Set the gradient to 0 for a plain white (light) or black (dark) backdrop, useful for figures.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Grid Box") {
                LabeledContent("Line thickness") {
                    HStack(spacing: DS.Space.m) {
                        Slider(value: Binding(
                            get: { viewModel.workspace.gridLineWidth },
                            set: {
                                viewModel.workspace.gridLineWidth = $0
                                viewModel.renderer?.gridLineWidth = $0
                            }
                        ), in: 1...8, step: 0.5)
                        Text(String(format: "%.1f", viewModel.workspace.gridLineWidth))
                            .font(DS.Font.monoSmall)
                            .monospacedDigit()
                            .frame(width: 30, alignment: .trailing)
                    }
                }

                ColorPicker("Color", selection: Binding(
                    get: {
                        let c = viewModel.workspace.gridColor
                        return Color(.sRGB, red: Double(c.x), green: Double(c.y), blue: Double(c.z), opacity: Double(c.w))
                    },
                    set: { newColor in
                        if let components = NSColor(newColor).usingColorSpace(.deviceRGB) {
                            viewModel.workspace.gridColor = SIMD4<Float>(
                                Float(components.redComponent),
                                Float(components.greenComponent),
                                Float(components.blueComponent),
                                Float(components.alphaComponent)
                            )
                        }
                    }
                ), supportsOpacity: true)
            }
        }
        .settingsPane()
    }

    // MARK: - Docking

    private var dockingPane: some View {
        Form {
            Section {
                Toggle("Confirm before docking", isOn: $confirmBeforeDocking)
            } header: {
                Text("Running")
            } footer: {
                Text("Shows a review sheet with the protein, ligands, site and method before each run. Turn off to start docking straight from Run.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                Toggle("Show advanced settings", isOn: $dockingAdvancedExpanded)
            } header: {
                Text("Docking Step")
            } footer: {
                Text("Opens the Docking step with its Advanced section expanded. Expanding or collapsing it in the step changes this setting too.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .settingsPane()
    }

    // MARK: - Logging

    private var loggingPane: some View {
        Form {
            Section {
                Toggle("Write logs to file", isOn: $logFileEnabled)
                    .onChange(of: logFileEnabled) { _, enabled in
                        LogSettings.isFileLoggingEnabled = enabled
                        if enabled {
                            ActivityLog.shared.startFileLogging()
                        } else {
                            ActivityLog.shared.stopFileLogging()
                        }
                    }
            } header: {
                Text("File Logging")
            } footer: {
                Text("Each session creates a timestamped log file with all console output.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Log Directory") {
                HStack {
                    TextField("Path", text: $logFileDirectory)
                        .textFieldStyle(.roundedBorder)
                        .font(DS.Font.mono)
                        .labelsHidden()
                        .onChange(of: logFileDirectory) { _, newPath in
                            LogSettings.logDirectory = newPath
                        }
                    Button("Choose…") { chooseLogDirectory() }
                }

                HStack(spacing: DS.Space.l) {
                    Button("Reset to Default") {
                        logFileDirectory = LogSettings.defaultLogDirectory
                        LogSettings.logDirectory = LogSettings.defaultLogDirectory
                    }
                    Button("Open in Finder") {
                        ActivityLog.shared.revealLogDirectoryInFinder()
                    }
                }
                .controlSize(.small)

                if let url = ActivityLog.shared.currentLogFileURL {
                    LabeledContent {
                        Button("Reveal") { ActivityLog.shared.revealLogInFinder() }
                            .controlSize(.small)
                    } label: {
                        Label {
                            Text("Current: \(url.lastPathComponent)")
                                .font(DS.Font.monoSmall)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        } icon: {
                            Image(systemName: "doc.text")
                                .foregroundStyle(DS.Color.success)
                        }
                    }
                }
            }

            Section {
                NumberStepperRow(title: "Keep logs for", value: $logFileRetentionDays, range: 1...90,
                                 unit: logFileRetentionDays == 1 ? "day" : "days")
                    .onChange(of: logFileRetentionDays) { _, days in
                        LogSettings.retentionDays = days
                    }

                LabeledContent {
                    Button("Prune Old Logs Now") {
                        Task.detached(priority: .utility) {
                            LogSettings.pruneOldLogs()
                        }
                        viewModel.log.info("Pruned log files older than \(logFileRetentionDays) day(s)", category: .system)
                    }
                    .controlSize(.small)
                } label: {
                    logDirectoryStats
                }
            } header: {
                Text("Retention")
            } footer: {
                Text("Log files older than this are deleted automatically on launch.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .settingsPane()
    }

    // MARK: - Helpers

    private var themeDescription: String {
        switch selectedTheme {
        case .auto: "Follows your macOS system appearance."
        case .light: "Always use the light appearance."
        case .dark: "Always use the dark appearance."
        }
    }

    @ViewBuilder
    private var logDirectoryStats: some View {
        let dir = URL(fileURLWithPath: LogSettings.logDirectory, isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey],
            options: .skipsHiddenFiles
        ))?.filter { $0.pathExtension == "log" } ?? []

        let totalSize = files.compactMap {
            try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize
        }.reduce(0, +)

        HStack(spacing: DS.Space.xs) {
            Text("\(files.count) log file\(files.count == 1 ? "" : "s")")
                .foregroundStyle(.secondary)
            Text("(\(ByteSize.string(totalSize)))")
                .foregroundStyle(.tertiary)
        }
        .font(DS.Font.secondary)
        .monospacedDigit()
    }

    private func chooseLogDirectory() {
        let panel = NSOpenPanel()
        panel.title = "Choose Log Directory"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = URL(fileURLWithPath: logFileDirectory)
        if panel.runModal() == .OK, let url = panel.url {
            logFileDirectory = url.path
            LogSettings.logDirectory = url.path
            // Restart file logging with new path
            ActivityLog.shared.startFileLogging()
        }
    }
}

// MARK: - Folders

/// A folder path with Choose… and Reset buttons (General ▸ Folders).
private struct FolderSettingRow: View {
    let title: String
    @Binding var path: String
    let defaultPath: String

    var body: some View {
        LabeledContent(title) {
            HStack {
                Text((path as NSString).abbreviatingWithTildeInPath)
                    .font(DS.Font.monoSmall)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Button("Choose…") { choose() }
                Button("Reset") { path = defaultPath }
                    .disabled(path == defaultPath)
            }
            .controlSize(.small)
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}

// MARK: - Performance

/// CPU core budget (applied to the C++ core and parallel Swift work) and read-only machine info.
private struct PerformanceSettingsPane: View {
    @Bindable private var settings = AppSettings.shared
    private let machine = MachineInfo.current

    var body: some View {
        Form {
            Section {
                LabeledContent("CPU cores") {
                    HStack(spacing: DS.Space.m) {
                        Slider(value: Binding(
                            get: { Double(settings.maxCPUCores) },
                            set: { settings.maxCPUCores = AppSettings.clampCores(Int($0.rounded())) }
                        ), in: 1...Double(max(AppSettings.availableCores, 2)), step: 1)
                        Stepper(value: $settings.maxCPUCores, in: 1...AppSettings.availableCores) {
                            Text("\(settings.maxCPUCores) of \(AppSettings.availableCores)")
                                .font(DS.Font.monoSmall)
                                .monospacedDigit()
                        }
                    }
                }
                LabeledContent("In effect") {
                    Text("\(settings.appliedThreadLimit) threads")
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Default") {
                    if settings.maxCPUCores == AppSettings.defaultCPUCores {
                        Label("\(AppSettings.defaultCPUCores) cores (in use)", systemImage: "checkmark")
                            .font(DS.Font.secondary)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    } else {
                        Button("Use \(AppSettings.defaultCPUCores) Cores") {
                            settings.maxCPUCores = AppSettings.defaultCPUCores
                        }
                        .controlSize(.small)
                    }
                }
            } header: {
                Text("CPU")
            } footer: {
                Text("Caps RDKit/TBB ligand preparation, GFN2-xTB, OpenMM and parallel batch work. The default keeps one performance core free for the interface. Terminal runs can override it with 'druse --cores N'.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            Section("This Mac") {
                LabeledContent("Processor", value: machine.cpuSummary)
                LabeledContent("GPU", value: machine.gpuSummary)
                LabeledContent("Memory", value: machine.memorySummary)
            }
        }
        .settingsPane()
    }
}

// MARK: - Console

/// In-app console behaviour and the default log verbosity of terminal runs.
private struct ConsoleSettingsPane: View {
    @Bindable private var settings = AppSettings.shared

    var body: some View {
        Form {
            Section {
                NumberStepperRow(title: "History length", value: $settings.consoleHistoryLength,
                                 range: AppSettings.historyLengthRange, step: 50, unit: "commands")
                Toggle("Suggest completions while typing", isOn: $settings.consoleAutosuggestions)
            } header: {
                Text("Console")
            } footer: {
                Text("Tab completes commands, subcommands and options; ↑ and ↓ walk the history, ⌃R searches it. Activity-log entries appear in the Logs panel (⇧⌘Y), never in the terminal (⌃`).")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                Picker("Log level", selection: $settings.headlessLogLevel) {
                    ForEach(LogVerbosity.allCases, id: \.self) { level in
                        Text(level.rawValue).tag(level)
                    }
                }
            } header: {
                Text("Terminal Runs")
            } footer: {
                Text("Activity-log lines 'druse' prints on stderr. Override per run with --log-level, or in a script with 'log level <level>'.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .settingsPane()
    }
}

// MARK: - Command Line

/// Installs the `druse` terminal command and reports whether the user's shell can find it.
private struct CommandLineSettingsPane: View {
    private let installer = CommandLineInstaller.current
    private let shell = CommandLineInstaller.loginShell
    @State private var userStatus: CommandLineInstaller.ToolStatus = .notInstalled
    @State private var systemStatus: CommandLineInstaller.ToolStatus = .notInstalled
    /// nil until the login shell has reported its PATH.
    @State private var shellPath: [String]?
    @State private var message: (text: String, isError: Bool)?

    var body: some View {
        Form {
            Section {
                locationRow(.user, title: "For me", subtitle: "~/.local/bin — no password needed", status: userStatus)
                locationRow(.system, title: "For all users", subtitle: "/usr/local/bin — asks for an administrator password", status: systemStatus)
                if let message {
                    Text(message.text)
                        .font(DS.Font.caption)
                        .foregroundStyle(message.isError ? DS.Color.danger : DS.Color.success)
                        .textSelection(.enabled)
                }
            } header: {
                Text("Terminal Command")
            } footer: {
                Text("Runs DruseScript from a terminal without opening a window: 'druse script.druse', 'druse dock method=vina', or 'druse' alone for an interactive prompt. See 'druse --help'.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }

            if let shell {
                Section("Your Shell") {
                    pathStatus(shell)
                }
            }
        }
        .settingsPane()
        .task { await refresh() }
    }

    @ViewBuilder
    private func locationRow(_ location: CommandLineInstaller.Location, title: String, subtitle: String,
                             status: CommandLineInstaller.ToolStatus) -> some View {
        LabeledContent {
            HStack {
                switch status {
                case .notInstalled:
                    Button("Install") { perform { try install(location) } }
                case .installed:
                    Button("Uninstall") { perform { try installer.uninstall(from: location) } }
                case .stale:
                    Button("Update") { perform { try install(location) } }
                    Button("Uninstall") { perform { try installer.uninstall(from: location) } }
                case .foreign:
                    EmptyView()
                }
            }
            .controlSize(.small)
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(subtitle).font(DS.Font.caption).foregroundStyle(.secondary)
                statusLabel(status, path: installer.toolURL(for: location).path)
            }
        }
    }

    private func statusLabel(_ status: CommandLineInstaller.ToolStatus, path: String) -> some View {
        let (text, color): (String, Color) = switch status {
        case .notInstalled: ("Not installed", .secondary)
        case .installed: ("Installed: \(path)", DS.Color.success)
        case .stale(let appPath): ("Points to another Druse.app (\(appPath))", DS.Color.warning)
        case .foreign: ("\(path) exists but was not installed by Druse", DS.Color.warning)
        }
        return Text(text).font(DS.Font.caption).foregroundStyle(color).lineLimit(2)
    }

    @ViewBuilder
    private func pathStatus(_ shell: CompletionShell) -> some View {
        let directory = installer.directory(for: .user)
        if let shellPath {
            if shellPath.contains(directory.path) {
                Label("\(installer.displayPath(directory)) is in your \(shell.displayName) PATH", systemImage: "checkmark.circle")
                    .foregroundStyle(DS.Color.success)
            } else {
                let setup = installer.pathSetup(for: shell, directory: directory)
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    Label("\(installer.displayPath(directory)) is not in your \(shell.displayName) PATH. Add this line (\(setup.file)):",
                          systemImage: "exclamationmark.triangle")
                        .foregroundStyle(DS.Color.warning)
                    CopyableCommand(text: setup.line)
                }
            }
        } else {
            Label("Checking your \(shell.displayName) PATH…", systemImage: "hourglass")
                .foregroundStyle(.secondary)
        }
    }

    private func install(_ location: CommandLineInstaller.Location) throws {
        switch location {
        case .user: try installer.installForUser()
        case .system: try installer.installForAllUsers()
        }
    }

    private func perform(_ action: () throws -> Void) {
        do {
            try action()
            message = ("Done.", false)
        } catch {
            message = (error.localizedDescription, true)
        }
        Task { await refresh() }
    }

    private func refresh() async {
        userStatus = installer.status(at: .user)
        systemStatus = installer.status(at: .system)
        if let shell {
            shellPath = await CommandLineInstaller.shellPATH(for: shell)
        }
    }
}

// MARK: - Shell completions

/// Installs per-shell completion scripts generated from the console's command registry.
private struct ShellCompletionsSettingsPane: View {
    @Environment(AppViewModel.self) private var viewModel
    private let installer = CommandLineInstaller.current
    @State private var scripts: [CompletionShell: String] = [:]
    @State private var statuses: [CompletionShell: CommandLineInstaller.CompletionStatus] = [:]
    @State private var errorMessage: String?

    var body: some View {
        Form {
            Section {
                ForEach(CompletionShell.allCases, id: \.self) { shell in
                    shellRow(shell)
                }
                if let errorMessage {
                    Text(errorMessage)
                        .font(DS.Font.caption)
                        .foregroundStyle(DS.Color.danger)
                }
            } header: {
                Text("Tab Completion for druse")
            } footer: {
                Text("Completes commands, subcommands, options and their values, and .druse files. Install again after updating Druse to pick up new commands.")
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .settingsPane()
        .onAppear(perform: refresh)
    }

    @ViewBuilder
    private func shellRow(_ shell: CompletionShell) -> some View {
        let status = statuses[shell] ?? .notInstalled
        VStack(alignment: .leading, spacing: DS.Space.s) {
            LabeledContent {
                HStack {
                    switch status {
                    case .notInstalled:
                        Button("Install") { perform { try install(shell) } }
                    case .current:
                        Button("Uninstall") { perform { try installer.uninstallCompletions(for: shell) } }
                    case .outdated:
                        Button("Update") { perform { try install(shell) } }
                        Button("Uninstall") { perform { try installer.uninstallCompletions(for: shell) } }
                    case .foreign:
                        EmptyView()
                    }
                }
                .controlSize(.small)
            } label: {
                VStack(alignment: .leading, spacing: 2) {
                    Text(shell.displayName + (shell == CommandLineInstaller.loginShell ? " (your shell)" : ""))
                    Text(installer.displayPath(installer.completionURL(for: shell)))
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                    Text(statusText(status))
                        .font(DS.Font.caption)
                        .foregroundStyle(status == .current ? DS.Color.success : status == .outdated ? DS.Color.warning : .secondary)
                }
            }
            if status == .current || status == .outdated, let hint = installer.completionSetupHint(for: shell) {
                Text(hint)
                    .font(DS.Font.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func statusText(_ status: CommandLineInstaller.CompletionStatus) -> String {
        switch status {
        case .notInstalled: "Not installed"
        case .current: "Installed"
        case .outdated: "Installed, from an older Druse: update it"
        case .foreign: "Another file is there (not generated by Druse); left alone"
        }
    }

    private func install(_ shell: CompletionShell) throws {
        try installer.installCompletions(for: shell, script: scripts[shell] ?? "")
    }

    private func perform(_ action: () throws -> Void) {
        do {
            try action()
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
        refresh()
    }

    private func refresh() {
        if scripts.isEmpty {
            let completions = ShellCompletions(viewModel: viewModel)
            scripts = Dictionary(uniqueKeysWithValues: CompletionShell.allCases.map { ($0, completions.script(for: $0)) })
        }
        statuses = Dictionary(uniqueKeysWithValues: CompletionShell.allCases.map {
            ($0, installer.completionStatus(for: $0, script: scripts[$0] ?? ""))
        })
    }
}

/// A shell line in monospace with a Copy button.
private struct CopyableCommand: View {
    let text: String

    var body: some View {
        HStack {
            Text(text)
                .font(DS.Font.monoSmall)
                .textSelection(.enabled)
                .lineLimit(2)
            Spacer()
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
            }
            .controlSize(.small)
        }
    }
}

/// A labelled number: the label on the left, then an editable field, a stepper and the unit,
/// so the value scans in its own column instead of being buried in the label.
private struct NumberStepperRow: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    var step = 1
    let unit: String

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: DS.Space.s) {
                TextField(title, value: Binding(
                    get: { value },
                    set: { value = min(max($0, range.lowerBound), range.upperBound) }
                ), format: .number.grouping(.never))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .monospacedDigit()
                    .frame(width: DS.Size.numberField)
                Stepper(title, value: $value, in: range, step: step)
                    .labelsHidden()
                Text(unit)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 64, alignment: .leading)
            }
        }
    }
}

private extension View {
    /// Grouped form filling the fixed-size Settings window (it scrolls when a pane is taller).
    func settingsPane() -> some View {
        formStyle(.grouped)
    }
}
