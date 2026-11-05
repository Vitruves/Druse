// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - Session commands: help, apropos, set, unset, vars, echo, log, history, wait, run, status, version, clear

extension DruseInterpreter {

    /// Commands that act on the console session itself rather than on molecules.
    static func builtinCommands() -> [ConsoleCommand] {
        [helpCommand(), aproposCommand(), completeCommand(), setCommand(), unsetCommand(), varsCommand(), echoCommand(),
         logCommand(), historyCommand(), waitCommand(), runCommand(), statusCommand(), versionCommand(), clearCommand()]
    }

    // MARK: help / apropos

    /// `help`: the category listing, or the generated reference of one command (forms, types, ranges, defaults, examples).
    private static func helpCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "help",
            summary: "List commands by category, or show every form, argument and option of one command",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Without a topic: all commands; with one: usage, types, ranges, defaults and examples",
                            arguments: [ConsoleParameter(name: "topic", type: .string,
                                                         summary: "Command, optionally followed by subcommand words (help ligand db)",
                                                         isOptional: true, isVariadic: true)]) { inv in
                    let commands = inv.interpreter.commands
                    guard let topic = inv.string("topic") else {
                        inv.emit(.output, ConsoleHelp.overview(commands, style: inv.interpreter.textStyle))
                        return
                    }
                    let words = topic.split(separator: " ").map { $0.lowercased() }
                    guard let name = words.first, let command = inv.interpreter.command(named: name) else {
                        var message = "no command '\(words.first ?? topic)'"
                        if let hint = ConsoleSuggest.closest(words.first ?? topic, in: commands.map(\.name)) {
                            message += " — did you mean '\(hint)'?"
                        }
                        throw ConsoleError(message + "; 'apropos <word>' searches every summary")
                    }
                    let path = Array(words.dropFirst())
                    guard command.forms.contains(where: { $0.path.starts(with: path) }) else {
                        var message = "'\(command.name)' has no subcommand '\(path.joined(separator: " "))'"
                        let candidates = Array(Set(command.forms.compactMap(\.path.first))).sorted()
                        if let hint = ConsoleSuggest.closest(path[0], in: candidates) {
                            message += " — did you mean 'help \(command.name) \(hint)'?"
                        } else if !candidates.isEmpty {
                            message += "; subcommands: \(command.subcommandPaths.joined(separator: ", "))"
                        }
                        throw ConsoleError(message)
                    }
                    inv.emit(.output, ConsoleHelp.detail(command, path: path, style: inv.interpreter.textStyle))
                }
            ],
            examples: ["help", "help dock", "help ligand db", "help set"]
        )
    }

    /// `apropos`: finds command forms by a word in their names or summaries, for users who do not know the verb.
    private static func aproposCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "apropos",
            summary: "Search command, subcommand, argument and option summaries for a word",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "List every command form whose name or descriptions mention the word",
                            arguments: [ConsoleParameter(name: "word", type: .string,
                                                         summary: "Text to look for (case-insensitive; several words are searched as one phrase)",
                                                         isVariadic: true)]) { inv in
                    let word = inv.requiredString("word")
                    guard let matches = ConsoleHelp.apropos(word, in: inv.interpreter.commands, style: inv.interpreter.textStyle) else {
                        inv.emit(.output, "Nothing mentions '\(word)' — 'help' lists every command")
                        return
                    }
                    inv.emit(.output, matches)
                }
            ],
            examples: ["apropos charge", "apropos tautomer", "apropos exhaustiveness"]
        )
    }

    /// `complete`: prints the candidates Tab would offer, so completion can be inspected from scripts and tests.
    private static func completeCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "complete",
            summary: "List the completions the console offers for a partial command line",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "One candidate per line: completed line, kind and description (tab-separated)",
                            arguments: [ConsoleParameter(name: "line", type: .string,
                                                         summary: "Partial line, quoted to keep a trailing space (\"dock \")")]) { inv in
                    let line = inv.requiredString("line")
                    let candidates = inv.interpreter.completionCandidates(for: line)
                    guard !candidates.isEmpty else {
                        inv.emit(.output, "No completions")
                        return
                    }
                    inv.emit(.output, candidates.map { "\($0.applied(to: line))\t\($0.kind)\t\($0.summary)" }
                        .joined(separator: "\n"))
                }
            ],
            examples: ["complete \"dock \"", "complete \"prep pro\"", "complete \"load ~/Doc\""]
        )
    }

    // MARK: Variables

    /// `set`: defines a `$variable` or changes a session setting (`onerror`, `script_dir`).
    private static func setCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "set",
            summary: "Define a $variable, or change a session setting such as onerror",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Assign a value (spaces around '=' are optional; quote values containing spaces)",
                            arguments: [ConsoleParameter(name: "name", type: .string,
                                                         summary: "Variable name: letters, digits and '_', not starting with a digit"),
                                        ConsoleParameter(name: "value", type: .string,
                                                         summary: "Value; $other variables are substituted first")],
                            // `set` statements never reach the binder (the parser turns them into
                            // assignments); these options document the session variables for help
                            // and completion of `set onerror=…`.
                            options: [ConsoleParameter(name: SessionVariable.onError, type: .choice(["stop", "continue"]),
                                                       summary: "stop: a failed statement ends the script; continue: report it and run the next one",
                                                       defaultValue: "stop"),
                                      ConsoleParameter(name: SessionVariable.scriptDirectory, type: .path,
                                                       summary: "Folder relative paths resolve against (scripts set it to their own folder)")],
                            syntax: "set <name> = <value>   |   set onerror=stop|continue   |   set script_dir=<path>") { inv in
                    let name = inv.requiredString("name")
                    var value = inv.requiredString("value")
                    if name == SessionVariable.onError {
                        value = value.lowercased()
                        guard value == "stop" || value == "continue" else {
                            throw ConsoleError("onerror must be stop or continue, got '\(value)'")
                        }
                    }
                    inv.interpreter.variables[name] = value
                }
            ],
            examples: ["set ph = 7.4", "set receptor=\"~/targets/1hsg.pdb\"", "set onerror=continue", "echo \"pH $ph\""]
        )
    }

    /// `unset`: forgets a variable; unsetting a session setting restores its default.
    private static func unsetCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "unset",
            summary: "Remove a variable (removing onerror restores the default, stop)",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Forget the variable",
                            arguments: [ConsoleParameter(name: "name", type: .string, summary: "Variable name, without '$'")]) { inv in
                    let name = inv.requiredString("name")
                    guard inv.interpreter.variables.removeValue(forKey: name) != nil else {
                        throw ConsoleError("no variable '\(name)' — 'vars' lists them")
                    }
                }
            ],
            examples: ["unset ph", "unset onerror"]
        )
    }

    /// `vars`: lists variables and session settings with their current values.
    private static func varsCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "vars",
            summary: "List variables and session settings with their values",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Show every variable and its value") { inv in
                    let variables = inv.interpreter.variables
                    guard !variables.isEmpty else {
                        inv.emit(.output, "No variables — define one with 'set name = value'")
                        return
                    }
                    let rows = variables.keys.sorted().map { [$0, variables[$0] ?? ""] }
                    inv.emit(.output, ConsoleTable.render(["Name", "Value"], rows))
                }
            ],
            examples: ["set ph = 7.0", "vars"]
        )
    }

    /// `echo`: prints text after variable substitution (progress markers in scripts).
    private static func echoCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "echo",
            summary: "Print text, with $variables substituted",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Print the words, separated by spaces",
                            arguments: [ConsoleParameter(name: "text", type: .string, summary: "Text to print",
                                                         isOptional: true, isVariadic: true)]) { inv in
                    inv.emit(.output, inv.string("text") ?? "")
                }
            ],
            examples: ["echo \"Docking $ligand at pH $ph\""]
        )
    }

    // MARK: log / history

    /// `log`: the session log threshold, recent activity-log entries, and the log file's path.
    private static func logCommand() -> ConsoleCommand {
        let levels = LogVerbosity.allCases.map(\.rawValue)
        let categories = LogCategory.allCases.map { $0.rawValue.lowercased() }
        return ConsoleCommand(
            name: "log",
            summary: "Set how much activity-log output is shown, list recent entries, or locate the log file",
            category: .session,
            forms: [
                ConsoleForm(path: ["level"], summary: "Show or set the log threshold (GUI: entries shown in the Logs panel; headless: stderr, for the rest of the run)",
                            arguments: [ConsoleParameter(name: "level", type: .choice(levels),
                                                         summary: "debug, info, warn, error or none; omit to show the current level",
                                                         isOptional: true)]) { inv in
                    let interpreter = inv.interpreter
                    guard let name = inv.string("level"), let level = LogVerbosity(name: name) else {
                        inv.emit(.output, "Log level: \(interpreter.logLevel.rawValue) — \(interpreter.logLevel.summary)")
                        return
                    }
                    interpreter.logLevel = level
                    inv.emit(.success, "Log level \(level.rawValue): \(level.summary)")
                },
                ConsoleForm(path: ["show"], summary: "Print the most recent activity-log entries",
                            arguments: [ConsoleParameter(name: "count", type: .int, summary: "Number of entries",
                                                         isOptional: true, defaultValue: "20", range: 1...2000)],
                            options: [ConsoleParameter(name: "level", type: .choice(levels),
                                                       summary: "Minimum severity (default: the current log level)"),
                                      ConsoleParameter(name: "category", type: .choice(categories),
                                                       summary: "Only entries of this category")]) { inv in
                    let threshold = inv.string("level").flatMap(LogVerbosity.init(name:)) ?? inv.interpreter.logLevel
                    let category = inv.string("category")
                    let entries = ActivityLog.shared.entries.filter { entry in
                        threshold.levels.contains(entry.level) && (category.map { entry.category.rawValue.lowercased() == $0 } ?? true)
                    }
                    guard !entries.isEmpty else {
                        inv.emit(.output, "No matching log entries")
                        return
                    }
                    inv.emit(.output, entries.suffix(inv.int("count") ?? 20)
                        .map { "\($0.formattedTime)  \($0.level.tag.padding(toLength: 4, withPad: " ", startingAt: 0))  [\($0.category.rawValue)]  \($0.message)" }
                        .joined(separator: "\n"))
                },
                ConsoleForm(path: ["file"], summary: "Print the path of this session's log file") { inv in
                    guard let url = ActivityLog.shared.currentLogFileURL else {
                        throw ConsoleError("file logging is off (enable it in Settings)")
                    }
                    inv.emit(.output, url.path)
                }
            ],
            examples: ["log level info", "log level", "log show 50 level=warn", "log show category=dock", "log file"]
        )
    }

    /// `history`: lists or clears the persisted console command history.
    private static func historyCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "history",
            summary: "Show or clear the console's command history",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "List the most recent command lines, oldest first",
                            arguments: [ConsoleParameter(name: "count", type: .int, summary: "Number of entries",
                                                         isOptional: true, defaultValue: "20", range: 1...10_000)]) { inv in
                    let history = try requireHistory(inv)
                    let entries = history.entries
                    guard !entries.isEmpty else {
                        inv.emit(.output, "History is empty")
                        return
                    }
                    let count = min(inv.int("count") ?? 20, entries.count)
                    let first = entries.count - count
                    let width = String(entries.count).count
                    inv.emit(.output, entries[first...].enumerated().map { offset, line in
                        let number = String(first + offset + 1)
                        return String(repeating: " ", count: width - number.count) + number + "  "
                            + line.replacingOccurrences(of: "\n", with: " ⏎ ")
                    }.joined(separator: "\n"))
                },
                ConsoleForm(path: ["clear"], summary: "Forget every remembered command line") { inv in
                    try requireHistory(inv).clear()
                    inv.emit(.success, "History cleared")
                }
            ],
            examples: ["history", "history 50", "history clear"]
        )
    }

    /// The session's history, or a clear error where none is kept.
    private static func requireHistory(_ inv: ConsoleInvocation) throws -> ConsoleHistory {
        guard let history = inv.interpreter.history else {
            throw ConsoleError("no command history is kept here (history belongs to the in-app console)")
        }
        return history
    }

    // MARK: wait / run / status / version / clear

    /// `wait`: a cancellable pause, for scripts that pace GUI demonstrations.
    private static func waitCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "wait",
            summary: "Pause for a number of seconds",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Sleep (Stop / Ctrl-C cancels it)",
                            arguments: [ConsoleParameter(name: "seconds", type: .float, summary: "Duration in seconds",
                                                         range: 0...86_400)]) { inv in
                    try await Task.sleep(for: .milliseconds(Int((inv.double("seconds") ?? 0) * 1000)))
                }
            ],
            examples: ["wait 2.5"]
        )
    }

    /// `run`: runs another script file with the caller's `onerror` policy and the file's folder as `script_dir`.
    private static func runCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "run",
            summary: "Run a .druse script (relative to the current script's folder)",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Run every statement of the file; honours onerror like the calling script",
                            arguments: [ConsoleParameter(name: "file", type: .path, summary: "Script file")]) { inv in
                    try await inv.interpreter.runFile(inv.requiredURL("file"))
                }
            ],
            examples: ["run prepare.druse", "run ~/scripts/screen_kinases.druse"]
        )
    }

    /// `status`: one-screen summary of the workspace (protein, ligand, pocket, results, analogs, settings).
    private static func statusCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "status",
            summary: "Show what is loaded: protein, ligand, pocket, results, analogs and key settings",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Summarize the workspace") { inv in
                    inv.emit(.output, statusReport(inv.viewModel))
                }
            ],
            examples: ["status"]
        )
    }

    /// `version`: the running app's version and build, for bug reports and script logs.
    private static func versionCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "version",
            summary: "Print the Druse version and build number",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Show the version of the running app") { inv in
                    let info = Bundle.main.infoDictionary
                    let version = info?["CFBundleShortVersionString"] as? String ?? "unknown"
                    let build = info?["CFBundleVersion"] as? String ?? "unknown"
                    inv.emit(.output, "Druse \(version) (build \(build))")
                }
            ],
            examples: ["version"]
        )
    }

    /// `clear`: empties the console transcript (no effect on the log or the workspace).
    private static func clearCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "clear",
            summary: "Clear the console transcript",
            category: .session,
            forms: [
                ConsoleForm(path: [], summary: "Empty the console view (the activity log and its file are kept)") { inv in
                    guard !inv.isHeadless else {
                        inv.emit(.warning, "'clear' only applies to the interactive console — ignored")
                        return
                    }
                    inv.interpreter.onClear?()
                }
            ],
            examples: ["clear"]
        )
    }

    /// The `status` table: one row per workspace aspect.
    private static func statusReport(_ vm: AppViewModel) -> String {
        var rows: [[String]] = []

        if let protein = vm.molecules.protein {
            let residues = protein.residues.filter(\.isStandard).count
            let chains = protein.chainIDs.joined(separator: ",")
            let state = vm.molecules.proteinPrepared ? "prepared" : "not prepared"
            rows.append(["Protein", "\(protein.name) — \(protein.atomCount) atoms, \(residues) residues, chains \(chains) (\(state))"])
        } else {
            rows.append(["Protein", "none"])
        }

        if let ligand = vm.molecules.ligand {
            rows.append(["Ligand", "\(ligand.name) — \(ligand.atomCount) atoms (\(ligand.heavyAtomCount) heavy)"])
        } else {
            rows.append(["Ligand", "none"])
        }

        let pockets = vm.docking.detectedPockets
        if let pocket = vm.docking.selectedPocket {
            let index = pockets.firstIndex { $0.id == pocket.id }.map { "#\($0 + 1) of \(pockets.count)" } ?? "custom box"
            rows.append(["Pocket", "\(index) — \(ConsoleFormat.vector(pocket.center)), box \(ConsoleFormat.box(pocket.size)) Å, \(String(format: "%.0f", pocket.volume)) Å³"])
        } else {
            rows.append(["Pocket", pockets.isEmpty ? "none" : "\(pockets.count) detected, none selected"])
        }

        let method = vm.docking.scoringMethod
        if vm.docking.isBatchDocking {
            rows.append(["Docking", "batch running \(vm.docking.batchProgress.current)/\(vm.docking.batchProgress.total)"])
        } else if vm.docking.isDocking {
            rows.append(["Docking", "running, generation \(vm.docking.dockingGeneration)/\(vm.docking.dockingTotalGenerations)"])
        } else if let best = vm.docking.dockingResults.first {
            let clusters = Set(vm.docking.dockingResults.map(\.clusterID)).count
            rows.append(["Results", "\(vm.docking.dockingResults.count) poses, \(clusters) clusters, best \(ConsoleFormat.score(best, method))"])
        } else {
            rows.append(["Results", "none"])
        }
        if !vm.docking.batchResults.isEmpty {
            rows.append(["Batch", "\(vm.docking.batchResults.count) ligands ranked"])
        }

        let entries = vm.ligandDB.entries
        rows.append(["Ligand DB", "\(entries.count) entries (\(entries.filter { !$0.atoms.isEmpty }.count) with 3D coordinates)"])

        if vm.leadOpt.hasReference || !vm.leadOpt.analogs.isEmpty {
            let reference = vm.leadOpt.hasReference ? vm.leadOpt.referenceName : "none"
            rows.append(["Lead opt", "reference \(reference), \(vm.leadOpt.analogs.count) analogs (\(vm.leadOpt.dockedAnalogCount) docked)"])
        }

        let config = vm.docking.dockingConfig
        let preset = DockingPreset.allCases.first { $0.isActive(in: config) }?.rawValue.lowercased() ?? "custom"
        rows.append(["Settings", "search=\(ConsoleNames.search(config.searchMethod)) scoring=\(ConsoleNames.scoring(method)) "
                     + "preset=\(preset) charges=\(ConsoleNames.charge(vm.docking.chargeMethod)) "
                     + "pH=\(String(format: "%.1f", vm.molecules.protonationPH))"])

        let width = rows.map { $0[0].count }.max() ?? 0
        return rows.map { $0[0].padding(toLength: width + 2, withPad: " ", startingAt: 0) + $0[1] }
            .joined(separator: "\n")
    }
}

// MARK: - Shared formatting and option-name mappings

/// Number and vector formatting shared by command output.
enum ConsoleFormat {
    static func vector(_ v: SIMD3<Float>) -> String {
        String(format: "(%.1f, %.1f, %.1f)", v.x, v.y, v.z)
    }

    /// Full box edge lengths from half-widths.
    static func box(_ halfSize: SIMD3<Float>) -> String {
        String(format: "%.1f×%.1f×%.1f", halfSize.x * 2, halfSize.y * 2, halfSize.z * 2)
    }

    static func score(_ result: DockingResult, _ method: ScoringMethod) -> String {
        String(format: "%.2f %@", result.displayScore(method: method), method.unitLabel)
    }

    static func seconds(_ interval: TimeInterval) -> String {
        interval < 60 ? String(format: "%.1f s", interval) : "\(Int(interval) / 60) min \(Int(interval) % 60) s"
    }
}

/// Console spellings of the app's enums (option values are short lower-case words).
enum ConsoleNames {
    static let searchMethods: [(String, SearchMethod)] = [
        ("ga", .genetic), ("fragment", .fragmentBased), ("remc", .parallelTempering),
        ("diffusion", .diffusionGuided), ("auto", .auto)
    ]
    static let scoringMethods: [(String, ScoringMethod)] = [
        ("vina", .vina), ("drusina", .drusina), ("druseaf", .druseAffinity), ("pignet2", .pignet2)
    ]
    static let chargeMethods: [(String, ChargeMethod)] = [
        ("gasteiger", .gasteiger), ("eem", .eem), ("qeq", .qeq), ("xtb", .xtb)
    ]
    static let renderModes: [(String, RenderMode)] = [
        ("ballstick", .ballAndStick), ("space", .spaceFilling), ("wire", .wireframe), ("ribbon", .ribbon)
    ]
    static let presets: [(String, DockingPreset)] = DockingPreset.allCases.map { ($0.rawValue.lowercased(), $0) }

    static func lookup<T>(_ name: String?, in table: [(String, T)]) -> T? {
        guard let name else { return nil }
        return table.first { $0.0 == name }?.1
    }

    static func names<T>(_ table: [(String, T)]) -> [String] { table.map(\.0) }

    static func search(_ method: SearchMethod) -> String {
        searchMethods.first { $0.1 == method }?.0 ?? method.rawValue
    }

    static func scoring(_ method: ScoringMethod) -> String {
        scoringMethods.first { $0.1 == method }?.0 ?? method.rawValue
    }

    static func charge(_ method: ChargeMethod) -> String {
        chargeMethods.first { $0.1 == method }?.0 ?? method.rawValue
    }
}
