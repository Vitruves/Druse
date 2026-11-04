// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import Foundation
import UniformTypeIdentifiers

/// One block of terminal scrollback: an echoed command line or what a command printed.
/// Entries never change once written, so the terminal view can append them incrementally.
struct ConsoleEntry: Identifiable, Sendable {
    /// How the terminal colours the entry (an ANSI-style colour per output kind).
    enum Style: Sendable {
        /// A command line as typed after the prompt.
        case command
        case output
        case success
        case warning
        case error
        /// Terminal chatter that is not command output (welcome banner, `^C`).
        case note
    }

    let id = UUID()
    let style: Style
    let text: String
    /// A command line abandoned with ⌃C before it ran; shown with a trailing `^C`.
    let interrupted: Bool

    /// The entry as it reads in a terminal, for copying: prompt before commands, `warning:` /
    /// `error:` before diagnostics (the same prefixes `Druse --headless` prints on stderr).
    var plainText: String {
        switch style {
        case .command: "\(ConsoleSession.prompt) \(text)\(interrupted ? "^C" : "")"
        case .warning: "warning: \(text)"
        case .error: "error: \(text)"
        case .output, .success, .note: text
        }
    }

    fileprivate init(style: Style, text: String, interrupted: Bool = false) {
        self.style = style
        self.text = text
        self.interrupted = interrupted
    }

    /// Maps an interpreter output kind to its terminal style.
    fileprivate init(_ kind: ConsoleOutputKind, _ text: String) {
        let style: Style = switch kind {
        case .input: .command
        case .output: .output
        case .success: .success
        case .warning: .warning
        case .error: .error
        }
        self.init(style: style, text: text)
    }
}

/// Remembered command lines, oldest first, optionally persisted in UserDefaults. Shared by the
/// terminal's ↑/↓ navigation, its ⌃R search, its autosuggestions and the `history` command.
@MainActor
final class ConsoleHistory {
    private(set) var entries: [String]
    private let defaultsKey: String?
    /// Entries kept; applied on the next `record`, so a changed setting takes effect without a restart.
    var limit: Int

    /// `defaultsKey` nil keeps the history in memory only.
    init(defaultsKey: String?, limit: Int) {
        self.defaultsKey = defaultsKey
        self.limit = limit
        entries = defaultsKey.flatMap { UserDefaults.standard.stringArray(forKey: $0) } ?? []
    }

    /// Appends a submitted line (an immediate repeat is stored once) and trims to the limit.
    func record(_ line: String) {
        if entries.last != line { entries.append(line) }
        if entries.count > limit { entries.removeFirst(entries.count - limit) }
        save()
    }

    func clear() {
        entries.removeAll()
        save()
    }

    /// The newest single-line entry that starts with `prefix` and is longer than it: the
    /// fish-style autosuggestion for a partially typed line.
    func latest(extending prefix: String) -> String? {
        entries.last { $0.count > prefix.count && $0.hasPrefix(prefix) && !$0.contains("\n") }
    }

    /// Index of the newest entry older than `index` (all entries when nil) that contains `query`,
    /// ignoring case: one step of a ⌃R reverse search.
    func index(containing query: String, before index: Int?) -> Int? {
        let end = min(index ?? entries.count, entries.count)
        return entries[..<end].lastIndex { query.isEmpty || $0.localizedCaseInsensitiveContains(query) }
    }

    private func save() {
        if let defaultsKey { UserDefaults.standard.set(entries, forKey: defaultsKey) }
    }
}

/// State of a ⌃R reverse-incremental history search (bash/zsh style).
struct ConsoleHistorySearch: Equatable, Sendable {
    /// What the user typed after ⌃R.
    var query = ""
    /// Index in the history of the line found, nil before a match.
    var matchIndex: Int?
    /// The line found (empty when there is none).
    var match = ""
    /// True when the query matches nothing (older): shown as "failing reverse-i-search".
    var failed = false
}

/// State of the window's DruseScript terminal: the interactive interpreter, its scrollback,
/// command history, the command line being edited with its completion list and inline
/// suggestion, the running command's live progress line and ⌃R search.
///
/// The terminal only ever shows this session (commands, their output and errors); activity-log
/// entries belong to the Logs panel. The session is created once per window, so scrollback,
/// draft and running commands survive hiding the panel or switching to the Logs panel or to
/// another workflow step.
@Observable
@MainActor
final class ConsoleSession: ConsoleOutput {
    nonisolated static let prompt = "druse›"

    private static let historyKey = "console.history"
    /// Scrollback entries kept; older ones are dropped so long sessions stay responsive.
    private static let scrollbackLimit = 5000

    // MARK: Scrollback

    private(set) var transcript: [ConsoleEntry] = []
    /// Bumped by every clear, so the terminal view knows to drop what it rendered.
    private(set) var clearCount = 0
    private(set) var isRunning = false
    /// Live status line of the running command (`ConsoleOutput.progress`), nil when none.
    private(set) var progressText: String?

    // MARK: Command line

    /// The command line being edited (kept here so a draft survives closing the panel).
    /// The terminal view reports typing through `edit(_:)`.
    private(set) var input = ""
    /// Ghost text shown after the caret: the rest of the newest matching history line, else
    /// what registry completion would add. Accepted with → / End (all) or ⌥→ (one word).
    private(set) var suggestion = ""
    /// Candidates of the completion list under the terminal; empty when the list is hidden.
    private(set) var completionItems: [ConsoleCompletion] = []
    /// Highlighted row of the completion list.
    private(set) var selectedCompletion = 0
    /// Active ⌃R search, which replaces the prompt line until accepted or cancelled.
    private(set) var historySearch: ConsoleHistorySearch?

    /// Threshold set by `log level <level>`; the Logs panel hides entries below it.
    /// Mirrors the interpreter's `logLevel` (through `onLogLevelChange`).
    private(set) var logLevel: LogVerbosity = .debug

    // MARK: History

    @ObservationIgnored let history = ConsoleHistory(defaultsKey: ConsoleSession.historyKey,
                                                     limit: AppSettings.shared.consoleHistoryLength)
    private var historyIndex: Int?
    private var historyDraft = ""

    /// Echoed lines the interpreter will echo again as `.input`; those repeats are dropped.
    private var pendingEchoes: [String] = []

    @ObservationIgnored private var interpreter: DruseInterpreter!

    init(viewModel: AppViewModel) {
        interpreter = DruseInterpreter(viewModel: viewModel, output: self, mode: .interactive)
        logLevel = interpreter.logLevel
        interpreter.history = history
        interpreter.onClear = { [weak self] in self?.clearScrollback() }
        interpreter.onLogLevelChange = { [weak self] level in self?.logLevel = level }
        append(ConsoleEntry(style: .note, text: Self.banner))
    }

    /// Two short lines that fit a narrow panel; the full key list is in `help` and the Help window.
    private static let banner = "DruseScript terminal · type 'help' for commands · logs are in the Logs panel (⇧⌘Y)\n"
        + "Tab complete · → accept suggestion · ↑↓ history · ⌃R search · ⌃C cancel · ⌘K clear"

    // MARK: ConsoleOutput

    func emit(_ kind: ConsoleOutputKind, _ text: String) {
        if kind == .input {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            if let index = pendingEchoes.firstIndex(of: trimmed) {
                pendingEchoes.remove(at: index)
                return
            }
        }
        // A result line takes the place of the progress line that preceded it.
        progressText = nil
        append(ConsoleEntry(kind, text))
    }

    func progress(_ text: String?) {
        progressText = text.flatMap { $0.isEmpty ? nil : $0 }
    }

    /// Every scrollback entry as plain text (Copy All).
    var plainTranscript: String {
        transcript.map(\.plainText).joined(separator: "\n")
    }

    // MARK: Running

    /// Runs the current command line (Return) and empties it.
    func submitInput() {
        let line = input
        setInput("")
        dismissCompletions()
        Task { await submit(line) }
    }

    /// Runs a typed (or pasted, possibly multi-line) command line; several lines run as one script.
    func submit(_ line: String) async {
        let source = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty else {
            append(ConsoleEntry(style: .command, text: ""))
            return
        }
        guard !isRunning else {
            append(ConsoleEntry(.warning, "A command is still running — press ⌃C to cancel it."))
            return
        }
        recordHistory(source)
        let lines = source.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        for echoed in lines { append(ConsoleEntry(style: .command, text: echoed)) }
        pendingEchoes = lines
        await execute(source, origin: "console")
    }

    /// Asks for a `.druse` / text file and runs it as a script.
    func chooseAndRunScript() {
        guard !isRunning else { return }
        let panel = NSOpenPanel()
        panel.title = "Run DruseScript"
        panel.prompt = "Run"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType(filenameExtension: "druse"), .plainText].compactMap { $0 }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await runScript(at: url) }
    }

    /// Runs a script file, announcing its start and outcome in the scrollback.
    func runScript(at url: URL) async {
        guard !isRunning else { return }
        let source: String
        do {
            source = try String(contentsOf: url, encoding: .utf8)
        } catch {
            append(ConsoleEntry(.error, "Cannot read \(url.path): \(error.localizedDescription)"))
            return
        }
        let name = url.lastPathComponent
        append(ConsoleEntry(.output, "Running script \(name)…"))
        if await execute(source, origin: name, directory: url.deletingLastPathComponent()) {
            append(ConsoleEntry(.success, "Script \(name) finished."))
        } else {
            append(ConsoleEntry(.error, "Script \(name) stopped."))
        }
    }

    /// ⌃C: cancels the running command (printing `^C`, as a terminal does); with nothing running
    /// it abandons the line being edited and starts a fresh prompt.
    func interrupt() {
        if historySearch != nil {
            cancelHistorySearch()
            return
        }
        if isRunning {
            progressText = nil
            append(ConsoleEntry(style: .note, text: "^C"))
            interpreter.cancel()
            return
        }
        append(ConsoleEntry(style: .command, text: input, interrupted: true))
        historyIndex = nil
        dismissCompletions()
        setInput("")
    }

    /// Runs one submitted input through the interpreter, marking the session busy until it finishes.
    @discardableResult
    private func execute(_ source: String, origin: String, directory: URL? = nil) async -> Bool {
        isRunning = true
        suggestion = ""
        defer {
            isRunning = false
            progressText = nil
            pendingEchoes.removeAll()
            refreshSuggestion()
        }
        return await interpreter.run(source, origin: origin, directory: directory)
    }

    // MARK: Scrollback

    /// ⌘K, ⌃L and the `clear` command: empties the scrollback; the line being edited, history
    /// and any running command are kept.
    func clearScrollback() {
        transcript.removeAll()
        clearCount &+= 1
    }

    /// Adds a transcript entry, dropping the oldest ones beyond the scrollback limit.
    private func append(_ entry: ConsoleEntry) {
        transcript.append(entry)
        if transcript.count > Self.scrollbackLimit {
            transcript.removeFirst(transcript.count - Self.scrollbackLimit)
        }
    }

    // MARK: Editing

    /// The user changed the command line. An open completion list follows the typing; typing a
    /// space right after a known command or subcommand opens it.
    func edit(_ text: String) {
        let typedSpace = text == input + " "
        input = text
        if !completionItems.isEmpty || (typedSpace && interpreter.isAtCommandBoundary(text)) {
            refreshCompletions()
        }
        refreshSuggestion()
    }

    private func setInput(_ text: String) {
        input = text
        refreshSuggestion()
    }

    // MARK: History (↑ / ↓)

    /// ↑: replaces the line with the previous history entry, remembering the line being typed.
    func showPreviousHistory() {
        let entries = history.entries
        guard !entries.isEmpty else { return }
        if let index = historyIndex {
            historyIndex = max(0, min(index, entries.count) - 1)
        } else {
            historyDraft = input
            historyIndex = entries.count - 1
        }
        dismissCompletions()
        setInput(entries[historyIndex ?? 0])
    }

    /// ↓: moves forward in history, restoring the line being typed after the newest entry.
    func showNextHistory() {
        guard let index = historyIndex else { return }
        let entries = history.entries
        dismissCompletions()
        if index + 1 < entries.count {
            historyIndex = index + 1
            setInput(entries[index + 1])
        } else {
            historyIndex = nil
            setInput(historyDraft)
        }
    }

    /// Remembers a submitted line in the persisted history.
    private func recordHistory(_ source: String) {
        historyIndex = nil
        historyDraft = ""
        history.limit = AppSettings.shared.consoleHistoryLength
        history.record(source)
    }

    // MARK: Reverse search (⌃R)

    /// ⌃R: starts a reverse search, or (already searching) moves to the next older match.
    func searchHistoryBackward() {
        guard !isRunning else { return }
        dismissCompletions()
        guard var search = historySearch else {
            historySearch = ConsoleHistorySearch()
            return
        }
        if let older = history.index(containing: search.query, before: search.matchIndex) {
            search.matchIndex = older
            search.match = history.entries[older]
            search.failed = false
        } else {
            search.failed = true
        }
        historySearch = search
    }

    /// Changes the search text; the newest line containing it becomes the match.
    func updateHistorySearch(query: String) {
        guard historySearch != nil else { return }
        var search = ConsoleHistorySearch(query: query)
        if !query.isEmpty {
            if let index = history.index(containing: query, before: nil) {
                search.matchIndex = index
                search.match = history.entries[index]
            } else {
                search.failed = true
            }
        }
        historySearch = search
    }

    /// Ends the search with the match on the command line (Esc, arrows, Tab), or runs it (Return).
    func acceptHistorySearch(run: Bool) {
        guard let search = historySearch else { return }
        historySearch = nil
        if !search.match.isEmpty { setInput(search.match) }
        if run { submitInput() }
    }

    /// Ends the search and restores the line that was being edited (⌃G, ⌃C).
    func cancelHistorySearch() {
        historySearch = nil
    }

    // MARK: Completion list

    /// Tab: accepts the highlighted candidate when the list is open; otherwise completes a
    /// single candidate directly, or extends the line to the candidates' common prefix and
    /// opens the list.
    func complete() {
        if !completionItems.isEmpty {
            acceptCompletion()
            return
        }
        let items = interpreter.completionCandidates(for: input)
        guard let first = items.first else { return }
        if items.count == 1 {
            accept(first)
            return
        }
        let lines = items.map { $0.applied(to: input) }
        let common = lines.dropFirst().reduce(lines[0]) { String($0.commonPrefix(with: $1)) }
        if common.count > input.count, common.hasPrefix(input) {
            setInput(common)
        }
        refreshCompletions()
    }

    /// Accepts the candidate at `index` (default: the highlighted one). Accepting a command,
    /// subcommand, `option=` or folder keeps the list open on what can follow it.
    func acceptCompletion(at index: Int? = nil) {
        let index = index ?? selectedCompletion
        guard completionItems.indices.contains(index) else { return }
        accept(completionItems[index])
    }

    /// Puts a completion candidate on the command line and closes the list.
    private func accept(_ completion: ConsoleCompletion) {
        setInput(completion.applied(to: input))
        let continues = completion.kind == .command || completion.kind == .subcommand
            || completion.text.hasSuffix("=") || completion.text.hasSuffix("/")
        if continues {
            refreshCompletions()
        } else {
            dismissCompletions()
        }
    }

    /// ↑ / ↓ (and ⇧Tab) inside the open list (wraps around).
    func moveCompletionSelection(by offset: Int) {
        guard !completionItems.isEmpty else { return }
        let count = completionItems.count
        selectedCompletion = ((selectedCompletion + offset) % count + count) % count
    }

    func dismissCompletions() {
        completionItems = []
        selectedCompletion = 0
    }

    /// Recomputes the list; a candidate that would only re-type what is already there (an exact
    /// match plus its trailing space) is left out so Return still runs the line.
    private func refreshCompletions() {
        let typed = input.trimmingCharacters(in: .whitespaces)
        completionItems = interpreter.completionCandidates(for: input).filter {
            $0.applied(to: input).trimmingCharacters(in: .whitespaces) != typed
        }
        selectedCompletion = 0
    }

    // MARK: Autosuggestion

    /// Accepts the ghost text: all of it, or (`wordOnly`) up to the end of its next word.
    /// Returns false when there is nothing to accept, so the key keeps its normal meaning.
    func acceptSuggestion(wordOnly: Bool) -> Bool {
        guard !suggestion.isEmpty else { return false }
        var accepted = suggestion
        if wordOnly {
            let start = suggestion.firstIndex { $0 != " " } ?? suggestion.endIndex
            let end = suggestion[start...].firstIndex(of: " ") ?? suggestion.endIndex
            accepted = String(suggestion[..<end])
        }
        dismissCompletions()
        setInput(input + accepted)
        return true
    }

    /// Recomputes the ghost text (off when Settings ▸ Console disables suggestions).
    private func refreshSuggestion() {
        guard AppSettings.shared.consoleAutosuggestions, !isRunning, !input.isEmpty, !input.contains("\n") else {
            suggestion = ""
            return
        }
        if let match = history.latest(extending: input) {
            suggestion = String(match.dropFirst(input.count))
            return
        }
        let lines = interpreter.completions(for: input).filter { $0.count > input.count && $0.hasPrefix(input) }
        guard let first = lines.first else {
            suggestion = ""
            return
        }
        let common = lines.dropFirst().reduce(first) { String($0.commonPrefix(with: $1)) }
        let extra = String(common.dropFirst(input.count))
        suggestion = extra.allSatisfy { $0 == " " } ? "" : extra
    }
}
