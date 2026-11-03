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
    @Environment(AppViewModel.self) private var viewModel

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
        }
        .frame(width: 560)
    }

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
            } footer: {
                Text("Shows a review sheet with the protein, ligands, site and method before each run. Turn off to start docking straight from Run.")
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
                Stepper("Keep logs for \(logFileRetentionDays) day\(logFileRetentionDays == 1 ? "" : "s")",
                        value: $logFileRetentionDays, in: 1...90)
                    .monospacedDigit()
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
            Text("(\(ByteCountFormatter.string(fromByteCount: Int64(totalSize), countStyle: .file)))")
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

private extension View {
    /// Grouped form sized to its content, so each Settings tab resizes the window to fit
    /// instead of leaving a mostly empty pane.
    func settingsPane() -> some View {
        formStyle(.grouped)
            .scrollDisabled(true)
            .fixedSize(horizontal: false, vertical: true)
    }
}
