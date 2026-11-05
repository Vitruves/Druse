// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit

// MARK: - Headless DruseScript runner
//
// `Druse --headless script.druse`, `Druse --headless -c "cmd; cmd"` (alias `--eval`), `Druse --headless <command>
// [args…]` (one statement given as shell words), or `Druse --headless` reading stdin (an interactive prompt with
// line editing, history and Tab completion on a terminal; one script otherwise). `-i` keeps the prompt open after a
// script, `-c` or command, with everything it loaded. `Druse --completions <shell>` prints
// a shell completion script. Runs on the main actor with no SwiftUI scene and no Dock icon.
// Exit codes: 0 success, 1 a command failed, 2 usage or syntax error, 130 interrupted.

/// The command-line flags of headless mode: the single source for `--help` and the shell completion scripts.
enum HeadlessCLI {
    /// One command-line flag: its names, value placeholder, completion values and summary.
    struct Flag: Sendable {
        let names: [String]
        /// Placeholder of the value the flag takes; nil for switches.
        let valueName: String?
        /// Values offered by shell completion (empty: free text).
        var valueChoices: [String] = []
        let summary: String
    }

    static let flags: [Flag] = [
        Flag(names: ["-c", "-e", "--eval"], valueName: "source",
             summary: "Run the given statements (';' or newline separated)"),
        Flag(names: ["-i", "--interactive"], valueName: nil,
             summary: "After the script, -c or command, stay in the interactive prompt with its state"),
        Flag(names: ["--var"], valueName: "name=value",
             summary: "Predefine $name (repeatable)"),
        Flag(names: ["--cores"], valueName: "N",
             valueChoices: (1...AppSettings.availableCores).map(String.init),
             summary: "CPU cores for this run (overrides Settings ▸ Performance)"),
        Flag(names: ["--log-level"], valueName: "level",
             valueChoices: LogVerbosity.allCases.map(\.rawValue),
             summary: "Activity log mirrored to stderr (default from Settings, normally warn)"),
        Flag(names: ["--echo"], valueName: nil,
             summary: "Echo each statement to stderr before running it"),
        Flag(names: ["--completions"], valueName: "shell",
             valueChoices: CompletionShell.allCases.map(\.rawValue),
             summary: "Print the shell completion script for fish, zsh or bash"),
        Flag(names: ["-h", "--help"], valueName: nil,
             summary: "Show this help")
    ]

    /// The `--help` text: invocation forms, the two modes, every command by category, the flags,
    /// examples and exit codes.
    @MainActor
    static func usage(style: ConsoleTextStyle) -> String {
        var lines = [style.command("druse") + " — run Druse from the terminal: DruseScript commands, no window",
                     style.dim("('druse' is 'Druse.app/Contents/MacOS/Druse --headless')")]

        lines += ["", style.heading("Usage")]
        lines += style.columns([
            ("druse <command> [args…] [key=value…]", "Run one command, then exit"),
            ("druse <script.druse>", "Run a script file, then exit"),
            ("druse -c \"command; command\"", "Run inline statements, then exit"),
            ("druse", "Open the interactive prompt (REPL)"),
            ("druse -i <script | -c \"…\" | command>", "Run first, then open the prompt with everything it loaded"),
            ("… | druse   or   druse -", "Run the statements read from stdin, then exit")
        ], label: style.command)

        lines += ["", style.heading("Modes")]
        lines += style.columns([
            ("One-shot", "A command, script, -c or piped stdin. Each run starts from an empty session and exits "
             + "when done, so load, prepare and dock in the same script or -c. Results go to stdout; errors, "
             + "warnings and progress to stderr, so output can be piped or redirected."),
            ("Interactive", "'druse' alone on a terminal, or -i. A 'druse›' prompt keeps the session between "
             + "statements, with line editing, history (~/.druse/console_history) and Tab completion. "
             + "Ctrl-C cancels the running command; Ctrl-D or 'exit' leaves.")
        ], label: style.heading)
        lines.append("")
        lines += style.paragraph("The first word is a script when it ends in .druse, contains a '/', or names an "
                                 + "existing file that is not a command; otherwise it is a command and the "
                                 + "remaining words form one statement.", indent: 2)

        lines += ["", style.heading("Commands") + style.dim("   'druse help <command>' for details")]
        lines += ConsoleHelp.commandList(DruseInterpreter.registry(), style: style, indent: 2)

        lines += ["", style.heading("Options")]
        lines += style.columns(flags.map { flag in
            let names = flag.names.joined(separator: ", ")
            return (flag.valueName.map { "\(names) <\($0)>" } ?? names, flag.summary)
        }, label: style.command)

        lines += ["", style.heading("Examples")]
        let examples = [
            ("druse help dock", "Every docking option, with types, defaults and examples"),
            ("""
             druse -c 'fetch 1HSG; prep; pocket detect
                       ligand smiles "CC(=O)Oc1ccccc1C(=O)O" name=aspirin
                       dock; export results results.csv'
             """, "Full pipeline in one run: fetch, prepare, find the pocket, dock aspirin, write the ranking"),
            ("druse screen.druse --var library=hits.smi --cores 8", "A script with $library predefined, on 8 cores"),
            ("druse -i -c 'fetch 1HSG; prep'", "Prepare a protein, then explore it at the prompt"),
            ("druse --completions fish > ~/.config/fish/completions/druse.fish",
             "Install Tab completion for fish (also zsh and bash)")
        ]
        for (example, note) in examples {
            lines += example.split(separator: "\n").map { "  " + style.command(String($0)) }
            lines += style.paragraph(note, indent: 6)
        }

        lines += ["", style.heading("Exit codes")]
        lines += style.columns([
            ("0", "success"),
            ("1", "a command failed"),
            ("2", "usage or syntax error (nothing ran)"),
            ("130", "interrupted with Ctrl-C")
        ], label: style.command)
        lines += style.paragraph("With -i, the exit code is that of the interactive session (normally 0).", indent: 2)
        return lines.joined(separator: "\n")
    }
}

/// Runs DruseScript without a window: parses the flags, feeds the interpreter, writes output to stdout/stderr and exits with a status code.
@MainActor
final class HeadlessRunner: ConsoleOutput {

    /// Process exit statuses documented in `--help`.
    enum ExitCode {
        static let success: Int32 = 0
        static let failure: Int32 = 1
        static let usage: Int32 = 2
        static let interrupted: Int32 = 130
    }

    /// Where the statements come from.
    private enum Source {
        case file(String)
        case eval(String)
        /// Shell words after the options: a script path or a single statement (resolved at run time).
        case words([String])
        case stdin
    }

    /// The parsed command-line options of one run.
    private struct Options {
        var source: Source = .stdin
        var variables: [String: String] = [:]
        /// nil: the Settings default (`AppSettings.headlessLogLevel`).
        var logLevel: LogVerbosity?
        var cores: Int?
        var echo = false
        /// `-i`: open the interactive prompt once the source has run.
        var interactive = false
    }

    /// What the command line asks for: a run, the help, a completion script, or a usage error.
    private enum ParseOutcome {
        case run(Options)
        case help
        case completions(CompletionShell)
        case usageError(String)
    }

    /// True when the launch arguments ask for headless mode instead of the GUI.
    nonisolated static func isRequested(by arguments: [String]) -> Bool {
        let headlessFlags: Set<String> = ["--headless", "-c", "-e", "--eval", "-i", "--interactive", "--completions"]
        return arguments.contains(where: headlessFlags.contains)
    }

    /// Parses the arguments, schedules the run on the main actor and keeps the main run loop alive
    /// (so MainActor tasks, main-queue blocks and run-loop timers all fire). Exits with the script's code.
    static func start(arguments: [String]) -> Never {
        switch parse(arguments) {
        case .help:
            writeLine(HeadlessCLI.usage(style: .terminal(STDOUT_FILENO)), to: .standardOutput)
            exit(ExitCode.success)
        case .completions(let shell):
            write(ShellCompletions(viewModel: AppViewModel()).script(for: shell), to: .standardOutput)
            exit(ExitCode.success)
        case .usageError(let message):
            let style = ConsoleTextStyle.terminal(STDERR_FILENO)
            writeLine(style.error("error:") + " \(message)\n" + style.dim("Run 'druse -h' for usage."), to: .standardError)
            exit(ExitCode.usage)
        case .run(let options):
            // Line-buffer C stdio so `print` from engine code interleaves in order with our
            // unbuffered FileHandle writes when stdout is a pipe or file.
            setvbuf(stdout, nil, _IOLBF, 0)
            NSApplication.shared.setActivationPolicy(.prohibited)
            let runner = HeadlessRunner(options: options)
            Task { @MainActor in
                runner.terminate(await runner.execute())
            }
            RunLoop.main.add(NSMachPort(), forMode: .default)
            while true { RunLoop.main.run() }
        }
    }

    private let options: Options
    private let viewModel = AppViewModel()
    private var interpreter: DruseInterpreter?
    private var interruptSource: DispatchSourceSignal?
    /// Set by the first Ctrl-C while a statement runs; a second one exits.
    private var interrupted = false
    /// Present while the interactive prompt is active, so the terminal can be restored on exit.
    private var lineEditor: TerminalLineEditor?
    /// True while a live status line is drawn on stderr (see `progress(_:)`).
    private var progressShown = false

    private init(options: Options) {
        self.options = options
    }

    // MARK: - ConsoleOutput

    /// Help and other formatted text go to stdout: coloured and wrapped when it is a terminal.
    var textStyle: ConsoleTextStyle { .terminal(STDOUT_FILENO) }

    /// Decoration of the prompt, errors, warnings and echoed statements, which go to stderr.
    private var stderrStyle: ConsoleTextStyle { .terminal(STDERR_FILENO) }

    func emit(_ kind: ConsoleOutputKind, _ text: String) {
        clearProgressLine()
        switch kind {
        case .input:
            if options.echo { Self.writeLine(stderrStyle.dim("druse› \(text)"), to: .standardError) }
        case .output:
            Self.writeLine(text, to: .standardOutput)
        case .success:
            Self.writeLine(textStyle.success(text), to: .standardOutput)
        case .warning:
            Self.writeLine(stderrStyle.warning("warning:") + " " + text, to: .standardError)
        case .error:
            Self.writeLine(stderrStyle.error("error:") + " " + text, to: .standardError)
        }
    }

    /// Live status line on an interactive stderr: redrawn in place with a carriage return and
    /// erase-line, cut to the terminal width so it never wraps. Pipes and files get nothing, so
    /// logs and captured output stay free of control characters.
    func progress(_ text: String?) {
        guard Self.stderrIsTerminal else { return }
        guard let text, !text.isEmpty else {
            clearProgressLine()
            return
        }
        let width = ConsoleTextStyle.columns(of: STDERR_FILENO)
        let line = text.count >= width ? String(text.prefix(max(width - 2, 1))) + "…" : text
        Self.write("\r\u{1B}[2K" + line, to: .standardError)
        progressShown = true
    }

    /// Erases the live status line before anything else is written to the terminal.
    private func clearProgressLine() {
        guard progressShown else { return }
        progressShown = false
        Self.write("\r\u{1B}[2K", to: .standardError)
    }

    private nonisolated static let stderrIsTerminal = isatty(STDERR_FILENO) != 0


    // MARK: - Execution

    /// Applies the options (cores, log level, variables) and runs the source; returns the exit code.
    private func execute() async -> Int32 {
        let settings = AppSettings.shared
        settings.applyHeadlessLogVerbosity(options.logLevel ?? settings.headlessLogLevel)
        if let cores = options.cores {
            settings.overrideCores(forSession: cores)
        } else {
            settings.applyThreadLimits()
        }
        DruseApp.initializeComputeServices(viewModel)

        let interpreter = DruseInterpreter(viewModel: viewModel, output: self, mode: .headless)
        interpreter.variables.merge(options.variables) { _, new in new }
        self.interpreter = interpreter
        installInterruptHandler()

        if case .stdin = options.source {
            return await runStandardInput()
        }
        let code = await runSource()
        guard options.interactive else { return code }
        interrupted = false
        return await runStandardInput()
    }

    /// Runs the script, `-c` source or command given on the command line; returns its exit code.
    private func runSource() async -> Int32 {
        switch options.source {
        case .eval(let source):
            return await runScript(source, origin: "<eval>")
        case .file(let path):
            return await runFile(path)
        case .words(let words):
            if Self.isScriptPath(words[0]) {
                guard words.count == 1 else {
                    emit(.error, "only one script can be given (got '\(words[1])' after \(words[0]))")
                    return ExitCode.usage
                }
                return await runFile(words[0])
            }
            return await runScript(DruseScriptQuoting.statement(fromShellWords: words), origin: "<args>")
        case .stdin:
            return await runStandardInput()
        }
    }

    /// The interactive prompt on a terminal; otherwise all of stdin run as one script.
    private func runStandardInput() async -> Int32 {
        if isatty(STDIN_FILENO) != 0 {
            return await runREPL()
        }
        return await runScript(await Self.readStandardInput(), origin: "<stdin>")
    }

    /// Runs a script file with its folder as `script_dir`.
    private func runFile(_ path: String) async -> Int32 {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard let source = try? String(contentsOf: url, encoding: .utf8) else {
            emit(.error, "cannot read script '\(path)'")
            return ExitCode.usage
        }
        interpreter?.variables["script_dir"] = url.deletingLastPathComponent().path
        return await runScript(source, origin: url.lastPathComponent)
    }

    /// Whether the first word is a script rather than a command: it ends in `.druse`, contains a
    /// `/`, or names an existing file that is not a command verb (so `druse status` in a folder
    /// holding a file called `status` still runs the command).
    private static func isScriptPath(_ word: String) -> Bool {
        if word.lowercased().hasSuffix(".druse") || word.contains("/") { return true }
        guard !commandVerbs.contains(word.lowercased()) else { return false }
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: (word as NSString).expandingTildeInPath, isDirectory: &isDirectory)
        return exists && !isDirectory.boolValue
    }

    /// Names of every registered command, for telling a command word from a script file.
    private static let commandVerbs = Set(DruseInterpreter.registry().map { $0.name.lowercased() })

    /// Syntax errors are reported (exit 2) before anything runs, so a typo on line 40
    /// never leaves a half-executed script behind.
    private func runScript(_ source: String, origin: String) async -> Int32 {
        if case .failure(let error) = DruseScriptParser.parse(source) {
            emit(.error, "\(origin):\(error)")
            return ExitCode.usage
        }
        guard let interpreter else { return ExitCode.failure }
        let succeeded = await interpreter.run(source, origin: origin)
        if interrupted { return ExitCode.interrupted }
        return succeeded ? ExitCode.success : ExitCode.failure
    }

    /// Interactive prompt on a terminal, with readline editing, persisted history and Tab completion.
    private func runREPL() async -> Int32 {
        guard let interpreter else { return ExitCode.failure }
        writeBanner()
        lineEditor = TerminalLineEditor.make(
            historyFile: AppSettings.shellHome.appendingPathComponent(".druse/console_history"),
            historyLength: AppSettings.shared.consoleHistoryLength
        ) { line in
            DispatchQueue.main.sync {
                MainActor.assumeIsolated { interpreter.completions(for: line) }
            }
        }
        while let source = await readStatement() {
            let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed == "exit" || trimmed == "quit" { break }
            if case .failure(let error) = DruseScriptParser.parse(source) {
                emit(.error, "\(error)")
                continue
            }
            _ = await interpreter.run(source, origin: "<stdin>")
            interrupted = false
        }
        Self.writeLine("", to: .standardError)
        return ExitCode.success
    }

    /// Greeting of the interactive prompt (on stderr, so stdout carries only command output).
    /// The prompt itself stays uncoloured: macOS's libedit prints every escape sequence of a
    /// prompt before its text, so colours would not land where they belong.
    private func writeBanner() {
        let style = stderrStyle
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
        Self.writeLine(style.heading("Druse" + (version.map { " " + $0 } ?? "")) + style.dim(" — interactive DruseScript"),
                       to: .standardError)
        Self.writeLine(style.dim("Type ") + style.command("help") + style.dim(" for commands, ")
                       + style.command("help <command>") + style.dim(" for details, ")
                       + style.command("apropos <word>") + style.dim(" to search. Tab completes; Ctrl-D or ")
                       + style.command("exit") + style.dim(" quits."), to: .standardError)
    }

    /// One logical statement from the terminal: lines ending in `\` continue on the next line.
    /// Returns nil at end of input (Ctrl-D).
    private func readStatement() async -> String? {
        var lines: [String] = []
        while true {
            let prompt = lines.isEmpty ? "druse› " : "   … "
            guard let line = await readLine(prompt: prompt) else {
                return lines.isEmpty ? nil : lines.joined(separator: "\n")
            }
            lines.append(line)
            if !line.hasSuffix("\\") { return lines.joined(separator: "\n") }
        }
    }

    /// Blocking reads happen off the main thread so MainActor work and Ctrl-C keep being serviced.
    private func readLine(prompt: String) async -> String? {
        let editor = lineEditor
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                if let editor {
                    continuation.resume(returning: editor.readLine(prompt: prompt))
                } else {
                    Self.write(prompt, to: .standardError)
                    continuation.resume(returning: Swift.readLine(strippingNewline: true))
                }
            }
        }
    }

    // MARK: - Interrupts

    /// Routes SIGINT to `handleInterrupt` instead of killing the process.
    private func installInterruptHandler() {
        signal(SIGINT, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        source.setEventHandler { [weak self] in
            MainActor.assumeIsolated { self?.handleInterrupt() }
        }
        source.resume()
        interruptSource = source
    }

    /// First Ctrl-C cancels the running command; a second one exits with code 130.
    private func handleInterrupt() {
        guard let interpreter, interpreter.isRunning, !interrupted else {
            terminate(ExitCode.interrupted)
        }
        interrupted = true
        interpreter.cancel()
        Self.writeLine("\ninterrupted: cancelling (press Ctrl-C again to exit)", to: .standardError)
    }

    /// Clears the progress line, restores the terminal, shuts the activity log down and exits with the given code.
    private func terminate(_ code: Int32) -> Never {
        clearProgressLine()
        lineEditor?.restoreTerminal()
        ActivityLog.shared.shutdown()
        exit(code)
    }

    // MARK: - Argument parsing

    /// Parses the launch arguments; flags may come before or after a script path.
    private static func parse(_ arguments: [String]) -> ParseOutcome {
        var options = Options()
        var words: [String] = []
        var evalSources: [String] = []
        var readsStdin = false
        var index = 0

        func nextValue() -> String? {
            index += 1
            return index < arguments.count ? arguments[index] : nil
        }

        while index < arguments.count {
            let argument = arguments[index]
            // After a command word, everything belongs to the statement (`druse dock method=vina`);
            // after a script path, flags such as `--log-level` may still follow.
            if let first = words.first, !isScriptPath(first) {
                words.append(argument)
                index += 1
                continue
            }
            switch argument {
            case "--headless":
                break
            case "-h", "--help":
                return .help
            case "--echo":
                options.echo = true
            case "-i", "--interactive":
                options.interactive = true
            case "--completions":
                guard let name = nextValue(), let shell = CompletionShell(rawValue: name.lowercased()) else {
                    return .usageError("--completions expects fish, zsh or bash")
                }
                return .completions(shell)
            case "-c", "-e", "--eval":
                guard let source = nextValue() else { return .usageError("\(argument) needs a value") }
                evalSources.append(source)
            case "--var":
                guard let assignment = nextValue(),
                      let equals = assignment.firstIndex(of: "="),
                      equals != assignment.startIndex else {
                    return .usageError("--var expects name=value")
                }
                options.variables[String(assignment[..<equals])] = String(assignment[assignment.index(after: equals)...])
            case "--cores":
                guard let text = nextValue(), let cores = Int(text), cores >= 1 else {
                    return .usageError("--cores expects a whole number of at least 1")
                }
                options.cores = cores
            case "--log-level":
                guard let name = nextValue(), let level = LogVerbosity(name: name) else {
                    return .usageError("--log-level expects debug, info, warn, error or none")
                }
                options.logLevel = level
            case "-":
                guard !readsStdin else { return .usageError("only one script can be given") }
                readsStdin = true
            default:
                if argument.hasPrefix("-NS") || argument.hasPrefix("-Apple") {
                    // Cocoa user-default overrides added by Xcode/launchd (e.g. -NSDocumentRevisionsDebugMode YES).
                    index += 1
                } else if argument.hasPrefix("-") {
                    return .usageError("unknown option '\(argument)'")
                } else if readsStdin {
                    return .usageError("give either '-' (stdin) or a script/command, not both")
                } else {
                    words.append(argument)
                }
            }
            index += 1
        }

        switch (words.isEmpty, evalSources.isEmpty) {
        case (false, false):
            return .usageError("give either a script/command or --eval, not both")
        case (false, true):
            options.source = .words(words)
        case (true, false):
            guard !readsStdin else { return .usageError("give either '-' (stdin) or --eval, not both") }
            options.source = .eval(evalSources.joined(separator: "\n"))
        case (true, true):
            options.source = .stdin
        }
        return .run(options)
    }

    // MARK: - Standard streams

    private nonisolated static func write(_ text: String, to handle: FileHandle) {
        handle.write(Data(text.utf8))
    }

    private nonisolated static func writeLine(_ text: String, to handle: FileHandle) {
        write(text + "\n", to: handle)
    }

    /// Reads all of stdin off the main thread, for piped scripts.
    private nonisolated static func readStandardInput() async -> String {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let data = FileHandle.standardInput.readDataToEndOfFile()
                continuation.resume(returning: String(decoding: data, as: UTF8.self))
            }
        }
    }
}

// MARK: - Shell words → DruseScript

/// Rebuilds DruseScript source from words the shell has already split and unquoted
/// (`druse load "my protein.pdb" name=x`), re-quoting the ones the parser would otherwise split.
enum DruseScriptQuoting {

    /// One statement: the verb as typed, each other word quoted when needed; `key=value` words keep
    /// their key bare so they still bind as options.
    static func statement(fromShellWords words: [String]) -> String {
        guard let verb = words.first else { return "" }
        let rest = words.dropFirst().map { word -> String in
            if let equals = word.firstIndex(of: "="), isOptionKey(word[..<equals]) {
                let value = String(word[word.index(after: equals)...])
                return String(word[...equals]) + (needsQuoting(value) ? quoted(value) : value)
            }
            return needsQuoting(word) ? quoted(word) : word
        }
        return ([verb] + rest).joined(separator: " ")
    }

    private static func isOptionKey(_ key: Substring) -> Bool {
        guard let first = key.first, first.isLetter || first == "_" else { return false }
        return key.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
    }

    private static func needsQuoting(_ word: String) -> Bool {
        word.isEmpty || word.hasPrefix("#")
            || word.contains { $0.isWhitespace || $0 == ";" || $0 == "\"" || $0 == "\\" }
    }

    /// Writes a shell word as a DruseScript string literal, so a command given as shell words re-parses to the same words.
    private static func quoted(_ word: String) -> String {
        var result = "\""
        for character in word {
            switch character {
            case "\\": result += "\\\\"
            case "\"": result += "\\\""
            case "\n": result += "\\n"
            case "\t": result += "\\t"
            default: result.append(character)
            }
        }
        return result + "\""
    }
}

// MARK: - Terminal line editing

/// Line editing for the headless prompt on a terminal: cursor movement and the usual Emacs keys,
/// ↑/↓ and Ctrl-R history persisted in `~/.druse/console_history`, and Tab completion from the
/// command registry.
///
/// It drives the readline-compatible API of macOS's libedit (`/usr/lib/libedit.3.dylib`, part of
/// every macOS install), resolved with `dlopen` so the app links nothing extra and no bridging
/// header is needed. libedit reads `~/.editrc` (prefix bindings with `druse:`). When the library
/// cannot be loaded, `make` returns nil and the REPL falls back to plain line reads.
///
/// All libedit calls, including the completion callback, happen on the one background thread that
/// is inside `readLine`; reads are strictly sequential.
final class TerminalLineEditor: @unchecked Sendable {
    private typealias ReadlineFunction = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
    private typealias HistoryFileFunction = @convention(c) (UnsafePointer<CChar>?) -> Int32
    private typealias StifleFunction = @convention(c) (Int32) -> Void
    private typealias VoidFunction = @convention(c) () -> Void
    private typealias CompletionFunction = @convention(c) (UnsafePointer<CChar>?, Int32, Int32)
        -> UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?

    private let readlineFunction: ReadlineFunction
    private let addHistory: HistoryFileFunction
    private let writeHistory: HistoryFileFunction
    private let lineBuffer: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
    private let historyPath: String
    private let completer: @Sendable (String) -> [String]
    private var savedTerminal = termios()

    /// The editor the C completion callback reports to (a C function pointer cannot capture).
    nonisolated(unsafe) private static var active: TerminalLineEditor?

    /// Loads libedit and configures completion and history; nil when libedit is unavailable.
    /// `completer` maps the line up to the cursor to full replacement lines
    /// (`DruseInterpreter.completions(for:)`); it is called on the reading thread.
    static func make(historyFile: URL, historyLength: Int,
                     completer: @escaping @Sendable (String) -> [String]) -> TerminalLineEditor? {
        guard let handle = dlopen("/usr/lib/libedit.3.dylib", RTLD_NOW) else { return nil }
        func symbol<T>(_ name: String, as type: T.Type) -> T? {
            dlsym(handle, name).map { unsafeBitCast($0, to: type) }
        }
        func variable<T>(_ name: String, as type: T.Type) -> UnsafeMutablePointer<T>? {
            dlsym(handle, name)?.assumingMemoryBound(to: type)
        }
        guard let readline = symbol("readline", as: ReadlineFunction.self),
              let addHistory = symbol("add_history", as: HistoryFileFunction.self),
              let readHistory = symbol("read_history", as: HistoryFileFunction.self),
              let writeHistory = symbol("write_history", as: HistoryFileFunction.self),
              let stifle = symbol("stifle_history", as: StifleFunction.self),
              let usingHistory = symbol("using_history", as: VoidFunction.self),
              let lineBuffer = variable("rl_line_buffer", as: UnsafeMutablePointer<CChar>?.self),
              let attempted = variable("rl_attempted_completion_function", as: CompletionFunction?.self) else {
            return nil
        }
        let editor = TerminalLineEditor(readline: readline, addHistory: addHistory, writeHistory: writeHistory,
                                        lineBuffer: lineBuffer, historyPath: historyFile.path, completer: completer)
        tcgetattr(STDIN_FILENO, &editor.savedTerminal)
        active = editor
        // libedit measures the prompt and input by the C locale's character set: use the
        // terminal's (UTF-8) so '›' and non-ASCII paths take one column each.
        setlocale(LC_CTYPE, "")

        variable("rl_readline_name", as: UnsafeMutablePointer<CChar>?.self)?.pointee = strdup("druse")
        // Words break only at blanks and ';' so `key=value` and paths complete as one word.
        variable("rl_completer_word_break_characters", as: UnsafeMutablePointer<CChar>?.self)?.pointee = strdup(" \t;")
        variable("rl_basic_word_break_characters", as: UnsafeMutablePointer<CChar>?.self)?.pointee = strdup(" \t;")
        // Candidates carry their own trailing space ("load ") or none ("method="); no filename fallback.
        variable("rl_completion_append_character", as: Int32.self)?.pointee = 0
        variable("rl_attempted_completion_over", as: Int32.self)?.pointee = 1
        // The prompt and editing go to stderr so stdout stays clean for command output.
        variable("rl_outstream", as: UnsafeMutablePointer<FILE>?.self)?.pointee = stderr
        attempted.pointee = { text, _, end in
            TerminalLineEditor.active?.complete(word: text.map { String(cString: $0) } ?? "", cursor: Int(end))
        }

        usingHistory()
        try? FileManager.default.createDirectory(at: historyFile.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        _ = readHistory(historyFile.path)
        stifle(Int32(historyLength))
        return editor
    }

    private init(readline: ReadlineFunction, addHistory: HistoryFileFunction, writeHistory: HistoryFileFunction,
                 lineBuffer: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>, historyPath: String,
                 completer: @escaping @Sendable (String) -> [String]) {
        self.readlineFunction = readline
        self.addHistory = addHistory
        self.writeHistory = writeHistory
        self.lineBuffer = lineBuffer
        self.historyPath = historyPath
        self.completer = completer
    }

    /// Reads one edited line (blocking); nil at end of input. Non-blank lines join the history.
    func readLine(prompt: String) -> String? {
        guard let raw = readlineFunction(prompt) else { return nil }
        defer { free(raw) }
        let line = String(cString: raw)
        if !line.trimmingCharacters(in: .whitespaces).isEmpty {
            _ = addHistory(raw)
            _ = writeHistory(historyPath)
        }
        return line
    }

    /// Puts the terminal back in the mode it had before editing (for exits from inside a read).
    func restoreTerminal() {
        tcsetattr(STDIN_FILENO, TCSANOW, &savedTerminal)
    }

    /// Builds readline's match array for the word being completed: element 0 is the text to insert
    /// (the common prefix, or the only match), then every candidate, then NULL.
    private func complete(word: String, cursor: Int) -> UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>? {
        let buffer = lineBuffer.pointee.map { String(cString: $0) } ?? ""
        let line = String(String.UnicodeScalarView(buffer.unicodeScalars.prefix(cursor)))
        let candidates = completer(line).map(Self.lastWord)
        var seen = Set<String>()
        let unique = candidates.filter { $0.hasPrefix(word) && seen.insert($0).inserted }
        guard !unique.isEmpty else { return nil }

        let insertion: String
        if unique.count == 1 {
            insertion = unique[0]
        } else {
            insertion = unique.dropFirst().reduce(unique[0]) { common, next in String(common.commonPrefix(with: next)) }
                .trimmingCharacters(in: .whitespaces)
        }
        let entries = unique.count == 1 ? [insertion] : [insertion] + unique
        // libedit releases the array and its strings with free(), so both come from malloc.
        let size = MemoryLayout<UnsafeMutablePointer<CChar>?>.stride * (entries.count + 1)
        guard let raw = malloc(size) else { return nil }
        let matches = raw.bindMemory(to: UnsafeMutablePointer<CChar>?.self, capacity: entries.count + 1)
        for (index, entry) in entries.enumerated() { matches[index] = strdup(entry) }
        matches[entries.count] = nil
        return matches
    }

    /// The final word of a full completion line, keeping its trailing space ("dock batch " → "batch ").
    private static func lastWord(_ completion: String) -> String {
        let trailing = completion.hasSuffix(" ") ? " " : ""
        let body = completion.hasSuffix(" ") ? String(completion.dropLast()) : completion
        let start = body.lastIndex { $0 == " " || $0 == ";" }.map { body.index(after: $0) } ?? body.startIndex
        return String(body[start...]) + trailing
    }
}
