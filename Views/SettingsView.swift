// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

enum AppTheme: String, CaseIterable {
    case auto = "Auto"
    case light = "Light"
    case dark = "Dark"

    var colorScheme: ColorScheme? {
        switch self {
        case .auto: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }

    var icon: String {
        switch self {
        case .auto: return "circle.lefthalf.filled"
        case .light: return "sun.max.fill"
        case .dark: return "moon.fill"
        }
    }
}

struct SettingsView: View {
    @AppStorage("appTheme") private var appTheme: String = AppTheme.dark.rawValue
    @Environment(AppViewModel.self) private var viewModel

    // Logging settings (local state synced with LogSettings)
    @AppStorage("logFileEnabled") private var logFileEnabled: Bool = true
    @AppStorage("logFileDirectory") private var logFileDirectory: String = LogSettings.defaultLogDirectory
    @AppStorage("logFileRetentionDays") private var logFileRetentionDays: Int = 7

    private var selectedTheme: AppTheme {
        AppTheme(rawValue: appTheme) ?? .dark
    }

    var body: some View {
        TabView {
            appearanceSettings
                .tabItem {
                    Label("Appearance", systemImage: "paintbrush")
                }

            loggingSettings
                .tabItem {
                    Label("Logging", systemImage: "doc.text")
                }
        }
        .frame(width: 500, height: 520)
    }

    // MARK: - Appearance Tab

    private var appearanceSettings: some View {
        Form {
            Section("Theme") {
                Picker("Appearance", selection: $appTheme) {
                    ForEach(AppTheme.allCases, id: \.rawValue) { theme in
                        Label(theme.rawValue, systemImage: theme.icon)
                            .tag(theme.rawValue)
                    }
                }
                .pickerStyle(.segmented)

                Text(themeDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Viewport Background") {
                HStack {
                    Text("Gradient intensity")
                    Slider(value: Binding(
                        get: { viewModel.workspace.backgroundOpacity },
                        set: {
                            viewModel.workspace.backgroundOpacity = $0
                            viewModel.renderer?.backgroundOpacity = $0
                        }
                    ), in: 0...1)
                }
                Text("Slide to 0 for a plain white (light) or black (dark) background, useful for presentations.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Grid Box") {
                HStack {
                    Text("Line thickness")
                    Slider(value: Binding(
                        get: { viewModel.workspace.gridLineWidth },
                        set: {
                            viewModel.workspace.gridLineWidth = $0
                            viewModel.renderer?.gridLineWidth = $0
                        }
                    ), in: 1...8, step: 0.5)
                    Text(String(format: "%.1f", viewModel.workspace.gridLineWidth))
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 30)
                }

                HStack {
                    Text("Color")
                    Spacer()
                    ColorPicker("", selection: Binding(
                        get: {
                            let c = viewModel.workspace.gridColor
                            return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z), opacity: Double(c.w))
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
                    .labelsHidden()
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Logging Tab

    private var loggingSettings: some View {
        Form {
            Section("File Logging") {
                Toggle("Write logs to file", isOn: $logFileEnabled)
                    .onChange(of: logFileEnabled) { _, enabled in
                        LogSettings.isFileLoggingEnabled = enabled
                        if enabled {
                            ActivityLog.shared.startFileLogging()
                        } else {
                            ActivityLog.shared.stopFileLogging()
                        }
                    }

                Text("When enabled, each session creates a timestamped log file with all console output.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Log Directory") {
                HStack {
                    TextField("Path", text: $logFileDirectory)
                        .textFieldStyle(.roundedBorder)
                        .font(.subheadline.monospaced())
                        .onChange(of: logFileDirectory) { _, newPath in
                            LogSettings.logDirectory = newPath
                        }

                    Button("Choose...") {
                        chooseLogDirectory()
                    }
                    .controlSize(.small)
                }

                HStack(spacing: 12) {
                    Button("Reset to Default") {
                        logFileDirectory = LogSettings.defaultLogDirectory
                        LogSettings.logDirectory = LogSettings.defaultLogDirectory
                    }
                    .controlSize(.small)

                    Button("Open in Finder") {
                        ActivityLog.shared.revealLogDirectoryInFinder()
                    }
                    .controlSize(.small)
                }

                if let url = ActivityLog.shared.currentLogFileURL {
                    HStack(spacing: 4) {
                        Image(systemName: "doc.text.fill")
                            .font(.footnote)
                            .foregroundStyle(.green)
                        Text("Current: \(url.lastPathComponent)")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        Spacer()
                        Button("Reveal") {
                            ActivityLog.shared.revealLogInFinder()
                        }
                        .controlSize(.mini)
                    }
                }
            }

            Section("Retention") {
                Stepper("Keep logs for \(logFileRetentionDays) day\(logFileRetentionDays == 1 ? "" : "s")",
                        value: $logFileRetentionDays, in: 1...90)
                    .onChange(of: logFileRetentionDays) { _, days in
                        LogSettings.retentionDays = days
                    }

                Text("Log files older than this are automatically deleted on app launch.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Button("Prune Old Logs Now") {
                    Task.detached(priority: .utility) {
                        LogSettings.pruneOldLogs()
                    }
                    viewModel.log.info("Pruned log files older than \(logFileRetentionDays) day(s)", category: .system)
                }
                .controlSize(.small)

                logDirectoryStats
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Helpers

    private var themeDescription: String {
        switch selectedTheme {
        case .auto: return "Follows your macOS system appearance."
        case .light: return "Always use light appearance."
        case .dark: return "Always use dark appearance."
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

        HStack {
            Text("\(files.count) log file\(files.count == 1 ? "" : "s")")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("(\(ByteCountFormatter.string(fromByteCount: Int64(totalSize), countStyle: .file)))")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
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
