// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - settings, cli
//
// Software settings (the same store as Settings in the GUI) and installation of the `druse`
// terminal command and its shell completions. `log level` (BuiltinCommands) changes the session's
// log verbosity; `set onerror=…` (BuiltinCommands) is reported here with the other session flags.

extension DruseInterpreter {

    static func settingsCommands() -> [ConsoleCommand] {
        [settingsCommand, cliCommand]
    }

    // MARK: - settings

    /// One user-facing setting: its option name in `settings set` / `settings reset`, and how to
    /// show, default and reset it.
    private struct SettingDescriptor {
        let name: String
        let summary: String
        let type: ConsoleValueType
        /// Accepted bounds of numeric settings (checked by the binder).
        var range: ClosedRange<Double>?
        let value: @MainActor (AppSettings) -> String
        let defaultValue: @MainActor () -> String
        let reset: @MainActor (AppSettings) -> Void
    }

    @MainActor
    private static var settingDescriptors: [SettingDescriptor] {
        let available = AppSettings.availableCores
        let history = AppSettings.historyLengthRange
        return [
            SettingDescriptor(
                name: "cores", summary: "CPU cores Druse may use at once; applied immediately", type: .int,
                range: 1...Double(available),
                value: { "\($0.maxCPUCores)" }, defaultValue: { "\(AppSettings.defaultCPUCores)" },
                reset: { $0.maxCPUCores = AppSettings.defaultCPUCores }),
            SettingDescriptor(
                name: "log-level", summary: "Default activity-log verbosity on stderr for headless runs",
                type: .choice(LogVerbosity.allCases.map(\.rawValue)),
                value: { $0.headlessLogLevel.rawValue }, defaultValue: { LogVerbosity.warn.rawValue },
                reset: { $0.headlessLogLevel = .warn }),
            SettingDescriptor(
                name: "history", summary: "Console commands remembered", type: .int,
                range: Double(history.lowerBound)...Double(history.upperBound),
                value: { "\($0.consoleHistoryLength)" }, defaultValue: { "\(AppSettings.defaultHistoryLength)" },
                reset: { $0.consoleHistoryLength = AppSettings.defaultHistoryLength }),
            SettingDescriptor(
                name: "autosuggest", summary: "Inline completion suggestions in the app's console", type: .bool,
                value: { $0.consoleAutosuggestions ? "on" : "off" }, defaultValue: { "on" },
                reset: { $0.consoleAutosuggestions = true }),
            SettingDescriptor(
                name: "pdb-cache", summary: "Folder for downloaded PDB entries (kept 7 days)", type: .path,
                value: { $0.pdbCacheDirectory }, defaultValue: { AppSettings.defaultPDBCacheDirectory },
                reset: { $0.pdbCacheDirectory = AppSettings.defaultPDBCacheDirectory }),
            SettingDescriptor(
                name: "project-dir", summary: "Folder the Save Project panel starts in", type: .path,
                value: { $0.defaultProjectDirectory }, defaultValue: { AppSettings.defaultProjectDirectoryPath },
                reset: { $0.defaultProjectDirectory = AppSettings.defaultProjectDirectoryPath })
        ]
    }

    private static var settingsCommand: ConsoleCommand {
        ConsoleCommand(
            name: "settings",
            summary: "Show or change software settings (CPU cores, logging, console, folders)",
            category: .settings,
            forms: [
                ConsoleForm(path: [], summary: "Show every setting, its default, the session flags and the machine") { inv in
                    inv.emit(.output, settingsReport(inv))
                },
                ConsoleForm(
                    path: ["set"],
                    summary: "Change one or more settings; they are saved and shared with the GUI and the terminal",
                    options: settingDescriptors.map { ConsoleParameter(name: $0.name, type: $0.type, summary: $0.summary, range: $0.range) }
                ) { inv in
                    try applySettings(inv)
                },
                ConsoleForm(
                    path: ["reset"],
                    summary: "Restore one setting, or all of them, to the default",
                    arguments: [ConsoleParameter(name: "setting", type: .choice(settingDescriptors.map(\.name)),
                                                 summary: "Setting to reset (all when left out)", isOptional: true)]
                ) { inv in
                    let settings = AppSettings.shared
                    if let name = inv.string("setting"), let descriptor = settingDescriptors.first(where: { $0.name == name }) {
                        descriptor.reset(settings)
                        inv.emit(.success, "\(name) reset to \(descriptor.value(settings))")
                    } else {
                        settings.resetToDefaults()
                        inv.emit(.success, "All settings reset to their defaults")
                    }
                }
            ],
            examples: [
                "settings",
                "settings set cores=4",
                "settings set log-level=info history=1000",
                "settings set pdb-cache=~/Data/pdb project-dir=~/Projects/druse",
                "settings reset cores",
                "settings reset",
                "set onerror=continue      # session flag: keep running a script after a failing statement"
            ]
        )
    }

    /// Validates and saves `settings set` values; the change is shared with the GUI and later headless runs.
    private static func applySettings(_ inv: ConsoleInvocation) throws {
        guard !inv.values.isEmpty else {
            throw ConsoleError("nothing to change; usage: settings set cores=N log-level=… history=N autosuggest=on|off "
                               + "pdb-cache=path project-dir=path")
        }
        let settings = AppSettings.shared
        var changes: [String] = []
        if let cores = inv.int("cores") {
            settings.maxCPUCores = cores
            changes.append("cores=\(cores) (thread limit now \(settings.appliedThreadLimit))")
        }
        if let history = inv.int("history") {
            settings.consoleHistoryLength = history
            changes.append("history=\(history)")
        }
        if let level = inv.string("log-level").flatMap(LogVerbosity.init(name:)) {
            settings.headlessLogLevel = level
            changes.append("log-level=\(level.rawValue)")
        }
        if let autosuggest = inv.bool("autosuggest") {
            settings.consoleAutosuggestions = autosuggest
            changes.append("autosuggest=\(autosuggest ? "on" : "off")")
        }
        if let url = inv.url("pdb-cache") {
            settings.pdbCacheDirectory = url.path
            changes.append("pdb-cache=\(url.path)")
        }
        if let url = inv.url("project-dir") {
            settings.defaultProjectDirectory = url.path
            changes.append("project-dir=\(url.path)")
            if settings.defaultProjectDirectoryURL == nil {
                inv.emit(.warning, "\(url.path) does not exist yet; panels use it once it does")
            }
        }
        inv.emit(.success, "Saved " + changes.joined(separator: ", "))
    }

    /// Every setting with its value and default, the session flags and this Mac's cores.
    private static func settingsReport(_ inv: ConsoleInvocation) -> String {
        let settings = AppSettings.shared
        let rows = settingDescriptors.map { [$0.name, $0.value(settings), $0.defaultValue(), $0.summary] }
        var lines = [ConsoleTable.render(["Setting", "Value", "Default", "Description"], rows), ""]

        let onError = inv.interpreter.variables[SessionVariable.onError] ?? "stop"
        let logLevel = inv.interpreter.logLevel
        let session: [[String]] = [
            ["cores in effect", "\(settings.effectiveCores)"
                + (settings.sessionCoreOverride != nil ? " (--cores for this run)" : "")
                + ", C++ thread limit \(settings.appliedThreadLimit)"],
            ["onerror", "\(onError)  — 'set onerror=continue' keeps running a script after a failing statement, 'stop' (default) ends it"],
            ["log level", "\(logLevel.rawValue)  — \(inv.isHeadless ? "stderr of this run" : "entries shown in the Logs panel"); 'log level <level>' changes it"]
        ]
        lines.append(ConsoleTable.render(["Session", "Value"], session))
        lines.append("")

        let machine = MachineInfo.current
        lines.append(ConsoleTable.render(["Machine", ""], [
            ["CPU", machine.cpuSummary],
            ["GPU", machine.gpuSummary],
            ["Memory", machine.memorySummary]
        ]))
        lines.append("")
        lines.append("Change with 'settings set name=value', restore with 'settings reset [name]'. The same settings are in Druse ▸ Settings.")
        return lines.joined(separator: "\n")
    }

    // MARK: - cli

    private static var cliCommand: ConsoleCommand {
        let location = ConsoleParameter(name: "location", type: .choice(CommandLineInstaller.Location.allCases.map(\.rawValue)),
                                        summary: "user: ~/.local/bin (no password); system: /usr/local/bin (administrator prompt, app only)",
                                        defaultValue: "user")
        let shells = CompletionShell.allCases.map(\.rawValue)
        let shellOption = ConsoleParameter(name: "shell", type: .choice(shells + ["all"]),
                                           summary: "fish, zsh, bash or all", defaultValue: "your login shell")
        return ConsoleCommand(
            name: "cli",
            summary: "Install the 'druse' terminal command and its shell completions",
            category: .settings,
            forms: [
                ConsoleForm(path: ["status"], summary: "Show where 'druse' and the completion scripts are installed") { inv in
                    inv.emit(.output, await cliStatusReport(commands: inv.interpreter.commands))
                },
                ConsoleForm(path: ["install"], summary: "Install the 'druse' command (runs this Druse.app headless)",
                            options: [location]) { inv in
                    let installer = CommandLineInstaller.current
                    let target = inv.string("location").flatMap(CommandLineInstaller.Location.init(rawValue:)) ?? .user
                    switch target {
                    case .user:
                        let url = try wrapInstallError { try installer.installForUser() }
                        inv.emit(.success, "Installed \(url.path)")
                        if let shell = CommandLineInstaller.loginShell,
                           let path = await CommandLineInstaller.shellPATH(for: shell),
                           !path.contains(url.deletingLastPathComponent().path) {
                            let setup = installer.pathSetup(for: shell, directory: url.deletingLastPathComponent())
                            inv.emit(.warning, "\(installer.displayPath(url.deletingLastPathComponent())) is not in your \(shell.displayName) PATH; "
                                     + "add (\(setup.file)):  \(setup.line)")
                        }
                    case .system:
                        guard !inv.isHeadless else {
                            throw ConsoleError("location=system needs the administrator prompt: install it from Druse ▸ Settings ▸ Command Line, "
                                               + "or use location=user")
                        }
                        try wrapInstallError { try installer.installForAllUsers() }
                        inv.emit(.success, "Installed \(installer.toolURL(for: .system).path)")
                    }
                },
                ConsoleForm(path: ["uninstall"], summary: "Remove the 'druse' command Druse installed (other files are left alone)",
                            options: [location]) { inv in
                    let target = inv.string("location").flatMap(CommandLineInstaller.Location.init(rawValue:)) ?? .user
                    if target == .system && inv.isHeadless {
                        throw ConsoleError("location=system needs the administrator prompt: uninstall it from Druse ▸ Settings ▸ Command Line")
                    }
                    let installer = CommandLineInstaller.current
                    guard installer.status(at: target) != .notInstalled else {
                        inv.emit(.output, "'druse' is not installed in \(installer.directory(for: target).path)")
                        return
                    }
                    try wrapInstallError { try installer.uninstall(from: target) }
                    inv.emit(.success, "Removed \(installer.toolURL(for: target).path)")
                },
                ConsoleForm(path: ["completions", "print"], summary: "Print the completion script for a shell",
                            arguments: [ConsoleParameter(name: "shell", type: .choice(shells), summary: "fish, zsh or bash")]) { inv in
                    let shell = CompletionShell(rawValue: inv.requiredString("shell")) ?? .zsh
                    inv.emit(.output, ShellCompletions(commands: inv.interpreter.commands).script(for: shell))
                },
                ConsoleForm(path: ["completions", "install"],
                            summary: "Write the completion script to the shell's standard per-user folder",
                            options: [shellOption]) { inv in
                    let installer = CommandLineInstaller.current
                    let completions = ShellCompletions(commands: inv.interpreter.commands)
                    for shell in try selectedShells(inv) {
                        let url = try wrapInstallError {
                            try installer.installCompletions(for: shell, script: completions.script(for: shell))
                        }
                        inv.emit(.success, "Installed \(shell.displayName) completions: \(url.path)")
                        if let hint = installer.completionSetupHint(for: shell) { inv.emit(.output, "  " + hint) }
                    }
                },
                ConsoleForm(path: ["completions", "uninstall"], summary: "Remove installed completion scripts",
                            options: [shellOption]) { inv in
                    let installer = CommandLineInstaller.current
                    for shell in try selectedShells(inv) {
                        try wrapInstallError { try installer.uninstallCompletions(for: shell) }
                        inv.emit(.success, "Removed \(shell.displayName) completions (\(installer.completionURL(for: shell).path))")
                    }
                }
            ],
            examples: [
                "cli status",
                "cli install",
                "cli install location=system",
                "cli completions install",
                "cli completions install shell=all",
                "cli completions print zsh",
                "cli uninstall"
            ]
        )
    }

    /// The shells a `cli completions` command targets: `shell=`, all of them, or the login shell.
    private static func selectedShells(_ inv: ConsoleInvocation) throws -> [CompletionShell] {
        switch inv.string("shell") {
        case "all": return CompletionShell.allCases
        case let name?: return CompletionShell(rawValue: name).map { [$0] } ?? []
        case nil:
            guard let shell = CommandLineInstaller.loginShell else {
                throw ConsoleError("your login shell is not fish, zsh or bash; pass shell=fish|zsh|bash|all")
            }
            return [shell]
        }
    }

    /// Installer failures become console errors with their message.
    @discardableResult
    private static func wrapInstallError<T>(_ body: () throws -> T) throws -> T {
        do {
            return try body()
        } catch let error as ConsoleError {
            throw error
        } catch {
            throw ConsoleError(error.localizedDescription)
        }
    }

    /// Where the `druse` command and each completion script are installed, and whether they are current.
    private static func cliStatusReport(commands: [ConsoleCommand]) async -> String {
        let installer = CommandLineInstaller.current
        var rows: [[String]] = []
        for location in CommandLineInstaller.Location.allCases {
            let state: String
            switch installer.status(at: location) {
            case .notInstalled: state = "not installed"
            case .installed: state = "installed"
            case .stale(let appPath): state = "points to another Druse.app (\(appPath)); install again"
            case .foreign: state = "another file is there (not installed by Druse)"
            }
            rows.append(["druse (\(location.rawValue))", installer.toolURL(for: location).path, state])
        }
        let completions = ShellCompletions(commands: commands)
        for shell in CompletionShell.allCases {
            let state: String
            switch installer.completionStatus(for: shell, script: completions.script(for: shell)) {
            case .notInstalled: state = "not installed"
            case .current: state = "installed"
            case .outdated: state = "outdated; install again"
            case .foreign: state = "another file is there (not generated by Druse)"
            }
            rows.append(["\(shell.displayName) completions", installer.completionURL(for: shell).path, state])
        }
        var lines = [ConsoleTable.render(["Item", "Path", "Status"], rows)]
        if let shell = CommandLineInstaller.loginShell {
            let userDirectory = installer.directory(for: .user)
            if let path = await CommandLineInstaller.shellPATH(for: shell) {
                let onPath = path.contains(userDirectory.path)
                lines.append("")
                lines.append("Login shell: \(shell.displayName); \(installer.displayPath(userDirectory)) is "
                             + (onPath ? "in PATH" : "not in PATH"))
                if !onPath {
                    let setup = installer.pathSetup(for: shell, directory: userDirectory)
                    lines.append("  add (\(setup.file)):  \(setup.line)")
                }
            }
        }
        return lines.joined(separator: "\n")
    }
}
