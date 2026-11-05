// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Observation

// MARK: - DruseScript syntax tree
//
// DruseScript is Druse's console and headless language: one command per line (or `;`-separated),
// `verb [subverb] [positional…] [key=value…]`, `# comments`, "quoted strings", `\` line continuation,
// `set name = value` assignments and `$name` / `${name}` substitution at execution time.
// The parser (DruseScriptParser) produces these types; the interpreter (DruseInterpreter) runs them.

/// 1-based position in the source, for error messages ("script.druse:12:5: unknown option").
struct ScriptLocation: Sendable, Equatable, CustomStringConvertible {
    let line: Int
    let column: Int

    var description: String { "\(line):\(column)" }
}

/// One argument word. `isQuoted` keeps "dock" (a literal string) apart from dock (a bare word),
/// so quoted text is never re-read as a subcommand or option.
struct ScriptWord: Sendable, Equatable {
    let text: String
    let isQuoted: Bool
    let location: ScriptLocation
}

/// One command statement: verb, positional words, `key=value` options, location and source text.
struct ScriptCommand: Sendable, Equatable {
    /// Lower-cased first word, e.g. "dock".
    let verb: String
    /// Remaining bare and quoted words, in order (subcommands are resolved by the interpreter).
    let arguments: [ScriptWord]
    /// `key=value` options (keys lower-cased; the last occurrence wins).
    let options: [String: ScriptWord]
    let location: ScriptLocation
    /// The statement's source text, for echoing in the console and in headless logs.
    let source: String
}

/// A parsed statement: a command, or a `set` assignment.
enum ScriptStatement: Sendable, Equatable {
    case command(ScriptCommand)
    /// `set name = value`
    case assignment(name: String, value: ScriptWord, location: ScriptLocation)
}

/// A lexing or parsing error with its location; a script with one does not run at all.
struct ScriptSyntaxError: Error, Sendable, Equatable, CustomStringConvertible {
    let message: String
    let location: ScriptLocation

    var description: String { "\(location): \(message)" }
}

// MARK: - Output channel

/// How a line of command output is presented: echoed input, plain output, success, warning, error…
enum ConsoleOutputKind: Sendable {
    /// The command being run (echoed input).
    case input
    /// Plain command output (tables, values, help).
    case output
    case success
    case warning
    case error
}

/// Where the interpreter writes. The GUI console and the headless runner each provide one.
@MainActor
protocol ConsoleOutput: AnyObject {
    func emit(_ kind: ConsoleOutputKind, _ text: String)
    /// Shows `text` as the live status line of the running command, replacing the previous one;
    /// nil removes it. The line is transient: it never enters the transcript, and the next
    /// `emit` replaces it (so a command's result line takes the place of its progress line).
    func progress(_ text: String?)
    /// How help and other formatted text should be decorated for this output.
    var textStyle: ConsoleTextStyle { get }
}

extension ConsoleOutput {
    /// Outputs without a live status line (files, discarding sinks) ignore progress.
    func progress(_ text: String?) {}
    /// Outputs other than a terminal get plain, unwrapped text.
    var textStyle: ConsoleTextStyle { .plain }
}

// MARK: - Live progress

extension ConsoleInvocation {
    /// Runs `body` while showing a live status line built by `describe`, which is re-evaluated
    /// whenever the observable state it reads changes (no polling: changes are coalesced by
    /// `Observations`). The line is cleared when `body` returns or throws. A nil description
    /// shows no line, e.g. before the operation has reported anything.
    func withProgress<T>(
        _ describe: @escaping @MainActor @Sendable () -> String?,
        during body: () async throws -> T
    ) async rethrows -> T {
        let interpreter = interpreter
        let observer = Task { @MainActor in
            var shown: String?
            for await text in Observations(describe) {
                // A value delivered after the body finished must not resurrect the cleared line.
                guard !Task.isCancelled else { break }
                // Unrelated changes to the observed models re-evaluate to the same line.
                guard text != shown else { continue }
                shown = text
                interpreter.progress(text)
            }
        }
        defer {
            observer.cancel()
            interpreter.progress(nil)
        }
        return try await body()
    }
}
