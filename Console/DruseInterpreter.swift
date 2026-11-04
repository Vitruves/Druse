// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Observation

/// Runs DruseScript against an `AppViewModel`, for the GUI console and for headless runs.
///
/// Statements run one at a time; every operation command awaits the real completion of the
/// view-model operation it starts. The first failing statement stops the script unless the
/// session variable `onerror` is `continue` (then the failure is reported, the next statement
/// runs, and `run` still returns false). Each command statement is echoed as `.input` before it runs.
@MainActor
final class DruseInterpreter {
    /// Whether commands run for the in-app terminal or a headless run (GUI-only commands adapt their behaviour).
    enum Mode: Sendable { case interactive, headless }

    let viewModel: AppViewModel
    let mode: Mode

    /// Script variables (`set name = value`, `$name`). Headless `--var key=value` seeds them;
    /// `script_dir`, when set, is the directory relative paths resolve against.
    var variables: [String: String] = [:]

    /// Called by the `clear` command (interactive mode) to empty the transcript.
    var onClear: (() -> Void)?

    /// Activity-log threshold set by `log level <level>`. Headless, it is the stderr mirror's
    /// verbosity kept by `AppSettings` (seeded from `--log-level` or the saved default); in the
    /// GUI it is the minimum level of the Logs panel, which learns it through
    /// `onLogLevelChange` (default `debug`: every entry is shown).
    var logLevel: LogVerbosity {
        get {
            guard mode == .headless else { return interactiveLogLevel }
            let settings = AppSettings.shared
            return settings.sessionLogVerbosity ?? settings.headlessLogLevel
        }
        set {
            if mode == .headless {
                AppSettings.shared.applyHeadlessLogVerbosity(newValue)
            } else {
                interactiveLogLevel = newValue
            }
            onLogLevelChange?(newValue)
        }
    }

    /// Notified after every change of `logLevel`.
    var onLogLevelChange: ((LogVerbosity) -> Void)?

    private var interactiveLogLevel = LogVerbosity.debug

    /// Command-line history shown and cleared by the `history` command; nil where no history is
    /// kept (headless scripts), in which case `history` reports that.
    var history: ConsoleHistory?

    private(set) var isRunning = false

    private weak var output: (any ConsoleOutput)?
    private(set) var commands: [ConsoleCommand] = []
    private var cancelRequested = false
    /// "Cancelled" is shown once, not once per nested script level.
    private var cancellationReported = false
    private var currentStatement: Task<Void, Error>?
    /// Directory of each script being run (innermost last), for relative paths and nested `run`.
    private var directoryStack: [URL] = []
    /// Nested `run` depth guard against scripts that run themselves.
    private static let maxScriptDepth = 16

    init(viewModel: AppViewModel, output: any ConsoleOutput, mode: Mode) {
        self.viewModel = viewModel
        self.output = output
        self.mode = mode
        commands = Self.registry()
    }

    /// Every command declaration, in help order. Side-effect free (nothing runs and no view
    /// model is needed): this is the entry point for generated documentation and shell
    /// completions, which read names, forms, parameters, types, choices and summaries from it.
    static func registry() -> [ConsoleCommand] {
        let groups: [[ConsoleCommand]] = [
            builtinCommands(),
            structureCommands(),
            selectionCommands(),
            preparationCommands(),
            ligandCommands(),
            dockingCommands(),
            analysisCommands(),
            exportCommands(),
            leadOptimizationCommands(),
            viewCommands(),
            settingsCommands(),
        ]
        return groups.flatMap { $0 }
    }

    /// True when `set onerror=continue` is in effect: failed statements are reported and skipped.
    var continuesOnError: Bool { variables[SessionVariable.onError] == "continue" }

    // MARK: - Running

    /// Parses and runs source; returns true only if every statement succeeded. A failure stops
    /// the script unless `onerror` is `continue`; cancellation always stops it.
    /// `directory` is where relative paths resolve (pass a script file's folder); by default the
    /// `script_dir` variable, else the working directory (headless) or the home folder (GUI).
    func run(_ source: String, origin: String, directory: URL? = nil) async -> Bool {
        guard !isRunning else {
            emit(.error, "Another command is still running")
            return false
        }
        isRunning = true
        cancelRequested = false
        cancellationReported = false
        defer {
            isRunning = false
            cancelRequested = false
            directoryStack.removeAll()
        }
        return await runScript(source, origin: origin, directory: directory ?? defaultDirectory)
    }

    /// Stops the running statement (and the operation it awaits); `run` then returns false.
    func cancel() {
        guard isRunning else { return }
        cancelRequested = true
        currentStatement?.cancel()
        stopRunningOperations()
    }

    /// Runs a nested script file (the `run` command).
    func runFile(_ url: URL) async throws {
        guard directoryStack.count < Self.maxScriptDepth else {
            throw ConsoleError("scripts nested more than \(Self.maxScriptDepth) deep (does \(url.lastPathComponent) run itself?)")
        }
        let source: String
        do {
            source = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw ConsoleError("cannot read \(url.path): \(error.localizedDescription)")
        }
        guard await runScript(source, origin: url.lastPathComponent, directory: url.deletingLastPathComponent()) else {
            if cancelRequested { throw CancellationError() }
            throw ConsoleError.reported
        }
    }

    /// Directory relative paths resolve against.
    var currentDirectory: URL { directoryStack.last ?? defaultDirectory }

    private var defaultDirectory: URL {
        if let dir = variables[SessionVariable.scriptDirectory], !dir.isEmpty {
            return URL(fileURLWithPath: (dir as NSString).expandingTildeInPath, isDirectory: true)
        }
        switch mode {
        case .headless: return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        case .interactive: return FileManager.default.homeDirectoryForCurrentUser
        }
    }

    /// Parses and runs a script, honouring `onerror`; returns false when a statement failed.
    private func runScript(_ source: String, origin: String, directory: URL) async -> Bool {
        let statements: [ScriptStatement]
        switch DruseScriptParser.parse(source) {
        case .failure(let error):
            emit(.error, Self.prefix(origin, error.location) + "syntax error: " + error.message)
            return false
        case .success(let parsed):
            statements = parsed
        }
        directoryStack.append(directory.standardizedFileURL)
        defer { directoryStack.removeLast() }

        var failed = false
        for statement in statements {
            if cancelRequested { return false }
            if await execute(statement, origin: origin) { continue }
            if cancelRequested || !continuesOnError { return false }
            failed = true
        }
        return !failed && !cancelRequested
    }

    /// Runs one statement and reports its error with the script location; returns false on failure.
    private func execute(_ statement: ScriptStatement, origin: String) async -> Bool {
        let location: ScriptLocation
        do {
            switch statement {
            case .assignment(let name, let value, let at):
                location = at
                try await assign(name, try substitute(value))
            case .command(let command):
                location = command.location
                emit(.input, command.source)
                try await dispatch(command)
                if cancelRequested { throw CancellationError() }
            }
            return true
        } catch let error as ConsoleError {
            if !error.isReported {
                emit(.error, Self.prefix(origin, error.location ?? location) + error.message)
            }
            return false
        } catch is CancellationError {
            if !cancellationReported {
                emit(.warning, "Cancelled")
                cancellationReported = true
            }
            return false
        } catch {
            emit(.error, Self.prefix(origin, statementLocation(statement)) + error.localizedDescription)
            return false
        }
    }

    /// The source location of a statement, for error messages.
    private func statementLocation(_ statement: ScriptStatement) -> ScriptLocation {
        switch statement {
        case .assignment(_, _, let location): location
        case .command(let command): command.location
        }
    }

    /// Typed console lines need no position; script errors read "file.druse:12:5: message".
    private static func prefix(_ origin: String, _ location: ScriptLocation) -> String {
        origin == "console" ? "" : "\(origin):\(location): "
    }

    /// Finds the command for a statement, binds its arguments and options to a form, and runs it.
    private func dispatch(_ statement: ScriptCommand) async throws {
        guard let command = command(named: statement.verb) else {
            var message = "unknown command '\(statement.verb)'"
            if let hint = ConsoleSuggest.closest(statement.verb, in: commands.map(\.name)) {
                message += " — did you mean '\(hint)'?"
            } else {
                message += " — type 'help' for the list of commands"
            }
            throw ConsoleError(message, at: statement.location)
        }
        let substituted = try substitute(statement)
        let (form, values) = try ConsoleBinder.bind(substituted, to: command, baseDirectory: currentDirectory)
        if form.isGUIOnly && mode == .headless {
            emit(.warning, "'\(ConsoleBinder.display(command, form))' only affects the 3D view — ignored in headless mode")
            return
        }
        let invocation = ConsoleInvocation(interpreter: self, values: values)
        let run = form.run
        let task = Task { try await run(invocation) }
        currentStatement = task
        defer { currentStatement = nil }
        try await task.value
    }

    /// Runs an assignment through the registry's `set` command, which validates session
    /// variables such as `onerror` before storing the value.
    private func assign(_ name: String, _ value: ScriptWord) async throws {
        guard let set = command(named: "set")?.forms.first else {
            preconditionFailure("'set' is registered by builtinCommands()")
        }
        try await set.run(ConsoleInvocation(interpreter: self, values: ["name": .text(name), "value": .text(value.text)]))
    }

    /// The command with this name (case-insensitive), if registered.
    func command(named name: String) -> ConsoleCommand? {
        commands.first { $0.name == name.lowercased() }
    }

    /// Writes to the console (GUI transcript or headless stdout/stderr).
    func emit(_ kind: ConsoleOutputKind, _ text: String) {
        output?.emit(kind, text)
    }

    /// Sets or (nil) clears the live status line of the running command (see `ConsoleOutput.progress`).
    func progress(_ text: String?) {
        output?.progress(text)
    }

    // MARK: - Cancellation

    /// Cancels whatever long operation a cancelled command started (docking, preparation, screening…).
    private func stopRunningOperations() {
        let vm = viewModel
        if vm.docking.isBatchDocking {
            vm.cancelBatchDocking()
        } else if vm.docking.isDocking {
            vm.stopDocking()
        }
        if vm.leadOpt.isGenerating || vm.leadOpt.isDocking {
            vm.cancelLeadOptimization()
        }
        if vm.docking.screeningState.isActive {
            vm.cancelScreening()
        }
    }

    // MARK: - Variables

    /// Replaces `$name` and `${name}` in every argument and option value before the statement runs.
    private func substitute(_ statement: ScriptCommand) throws -> ScriptCommand {
        ScriptCommand(
            verb: statement.verb,
            arguments: try statement.arguments.map(substitute),
            options: try statement.options.mapValues(substitute),
            location: statement.location,
            source: statement.source
        )
    }

    /// Replaces `$name` and `${name}`; a `$` not followed by a name is kept literally.
    private func substitute(_ word: ScriptWord) throws -> ScriptWord {
        guard word.text.contains("$") else { return word }
        let characters = Array(word.text)
        var result = ""
        var i = 0
        func isNameStart(_ c: Character) -> Bool { c == "_" || (c.isASCII && c.isLetter) }
        func isNameChar(_ c: Character) -> Bool { isNameStart(c) || (c.isASCII && c.isNumber) }

        while i < characters.count {
            let c = characters[i]
            guard c == "$", i + 1 < characters.count else {
                result.append(c)
                i += 1
                continue
            }
            var name = ""
            var end = i + 1
            if characters[end] == "{" {
                guard let close = characters[end...].firstIndex(of: "}") else {
                    throw ConsoleError("unterminated ${…} in '\(word.text)'", at: word.location)
                }
                name = String(characters[(end + 1)..<close])
                end = close + 1
            } else if isNameStart(characters[end]) {
                while end < characters.count, isNameChar(characters[end]) {
                    name.append(characters[end])
                    end += 1
                }
            } else {
                result.append(c)
                i += 1
                continue
            }
            guard let value = variables[name] else {
                var message = "unknown variable '$\(name)'"
                if let hint = ConsoleSuggest.closest(name, in: Array(variables.keys)) {
                    message += " — did you mean '$\(hint)'?"
                } else {
                    message += " — define it with 'set \(name) = …'"
                }
                throw ConsoleError(message, at: word.location)
            }
            result += value
            i = end
        }
        return ScriptWord(text: result, isQuoted: word.isQuoted, location: word.location)
    }

    // MARK: - Completion

    /// Structured completions for the word at the end of `line` (registry words, `$variables`
    /// and file paths relative to the current directory), for rich UIs such as the in-app popup.
    func completionCandidates(for line: String) -> [ConsoleCompletion] {
        ConsoleCompleter(commands: commands, variables: variables, directory: currentDirectory)
            .completions(for: line)
    }

    /// Plain variant of `completionCandidates(for:)`: each candidate as the full completed line.
    func completions(for line: String) -> [String] {
        completionCandidates(for: line).map { $0.applied(to: line) }
    }

    /// See `ConsoleCompleter.isAtCommandBoundary(_:)`.
    func isAtCommandBoundary(_ line: String) -> Bool {
        ConsoleCompleter(commands: commands, variables: variables, directory: currentDirectory)
            .isAtCommandBoundary(line)
    }
}

// MARK: - Session settings

/// Names of the variables that configure the session rather than hold user data.
enum SessionVariable {
    /// `continue` or `stop` (default): whether a failed statement stops the script.
    static let onError = "onerror"
    /// Folder relative paths resolve against when no script file sets it.
    static let scriptDirectory = "script_dir"
}

extension LogVerbosity {
    /// One-line description for `log level` output.
    var summary: String {
        switch self {
        case .debug: "everything, including debug diagnostics"
        case .info: "progress and results (no debug)"
        case .warn: "warnings and errors only"
        case .error: "errors only"
        case .none: "no log output"
        }
    }
}

extension VirtualScreeningPipeline.ScreeningState {
    var isActive: Bool {
        switch self {
        case .preparing, .docking, .reranking: true
        case .idle, .complete, .failed: false
        }
    }
}

// MARK: - Awaiting observable state

/// Lets commands await a change of observable app state instead of polling it.
@MainActor
enum ConsoleObservation {
    /// Suspends until `condition` holds, re-checking whenever observable state it reads changes.
    /// Returns early when the calling task is cancelled (Ctrl-C, console stop), so a command never
    /// stays stuck on state that does not change; callers check `Task.isCancelled` afterwards.
    static func wait(until condition: @escaping @MainActor () -> Bool) async {
        while !condition() && !Task.isCancelled {
            let gate = ResumeOnce()
            await withTaskCancellationHandler {
                await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                    gate.arm(continuation)
                    withObservationTracking {
                        _ = condition()
                    } onChange: {
                        gate.resume()
                    }
                    if Task.isCancelled { gate.resume() }
                }
            } onCancel: {
                gate.resume()
            }
        }
    }

    /// Resumes a continuation at most once, whichever of a state change or cancellation comes first.
    private final class ResumeOnce: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Void, Never>?
        private var fired = false

        func arm(_ continuation: CheckedContinuation<Void, Never>) {
            lock.lock()
            if fired {
                lock.unlock()
                continuation.resume()
                return
            }
            self.continuation = continuation
            lock.unlock()
        }

        func resume() {
            lock.lock()
            fired = true
            let pending = continuation
            continuation = nil
            lock.unlock()
            pending?.resume()
        }
    }
}
