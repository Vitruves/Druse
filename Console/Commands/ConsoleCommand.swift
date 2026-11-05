// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - Command registry types
//
// Every console command is declared as data: its forms (`dock`, `dock stop`, `ligand db import`),
// their positional arguments and `key=value` options with value types. The interpreter binds
// parsed words against these declarations, and `help`, `apropos`, usage lines, "did you mean"
// suggestions, in-app completion and the generated shell completions are all derived from them.
// `DruseInterpreter.registry()` enumerates the declarations without running anything.

/// Section a command is listed under in `help` and in generated documentation.
/// Declaration order is display order.
enum ConsoleCategory: String, CaseIterable, Sendable {
    case session = "Session"
    case structure = "Structures"
    case selection = "Selection"
    case preparation = "Protein preparation"
    case ligand = "Ligands"
    case docking = "Docking"
    case analysis = "Analysis"
    case export = "Export"
    case leadOptimization = "Lead optimization"
    case view = "View"
    case settings = "Settings"
    case general = "Other"
}

/// Type of a positional argument or option value.
enum ConsoleValueType {
    case string
    case int
    case float
    /// on/off, true/false, yes/no, 1/0
    case bool
    /// File or directory; relative paths resolve against the current script directory.
    case path
    /// "x,y,z"
    case vec3
    /// One of a fixed set of lower-case words.
    case choice([String])

    /// Short value placeholder used in usage lines (`exhaustiveness=N`).
    var placeholder: String {
        switch self {
        case .string: "text"
        case .int: "N"
        case .float: "x"
        case .bool: "on|off"
        case .path: "path"
        case .vec3: "x,y,z"
        case .choice(let values): values.joined(separator: "|")
        }
    }

    /// Readable type name used in detailed help ("integer", "one of: a, b").
    var typeDescription: String {
        switch self {
        case .string: "text"
        case .int: "integer"
        case .float: "number"
        case .bool: "on|off"
        case .path: "path"
        case .vec3: "x,y,z vector"
        case .choice(let values): "one of " + values.joined(separator: "|")
        }
    }

    /// The fixed values a completion can offer (empty when any value is allowed).
    var completions: [String] {
        switch self {
        case .bool: ["on", "off"]
        case .choice(let values): values
        default: []
        }
    }

    /// True for values that name files or folders, so completion lists the file system.
    var isPath: Bool {
        if case .path = self { return true }
        return false
    }
}

/// A positional argument or a `key=value` option of a command form.
struct ConsoleParameter {
    /// Argument name shown as `<name>`, or the option key (lower-case words joined with '-').
    let name: String
    let type: ConsoleValueType
    /// One-line description shown in help, completion lists and shell completions.
    let summary: String
    /// Positional arguments only: may be left out.
    var isOptional = false
    /// Last positional argument only: takes every remaining word, joined with spaces.
    var isVariadic = false
    /// Value used when the parameter is left out, as the user would type it (shown in help only;
    /// the form's `run` closure applies the actual default).
    var defaultValue: String?
    /// Accepted bounds for `.int` and `.float` values; the binder rejects values outside them.
    var range: ClosedRange<Double>?
}

/// One invocable form of a command, e.g. `dock`, `dock stop` or `ligand db import`.
struct ConsoleForm {
    /// Subcommand words after the verb; empty for the bare form.
    let path: [String]
    /// One-line description of what this form does.
    let summary: String
    var arguments: [ConsoleParameter] = []
    var options: [ConsoleParameter] = []
    /// Forms that only change the 3D view: skipped with a warning in headless mode.
    var isGUIOnly = false
    /// Usage line to show instead of the generated one, for forms the parser reads specially
    /// (`set name = value`).
    var syntax: String?
    let run: @MainActor (ConsoleInvocation) async throws -> Void
}

/// A console verb with all of its forms, its help category and usage examples.
struct ConsoleCommand {
    let name: String
    /// One-line description shown in `help` and completion lists.
    let summary: String
    /// Section of `help` the command is listed under.
    var category: ConsoleCategory = .general
    let forms: [ConsoleForm]
    /// Complete statements shown under "Examples" in `help <command>`.
    var examples: [String] = []

    /// Every subcommand path, space-joined ("db import"); the bare form is left out.
    var subcommandPaths: [String] {
        forms.map { $0.path.joined(separator: " ") }.filter { !$0.isEmpty }
    }
}

/// A failed statement. The message is shown as-is; the location (when known) points at the word.
struct ConsoleError: Error {
    let message: String
    var location: ScriptLocation?
    /// Already shown to the user (a nested script's own failure): stop without a second message.
    var isReported = false

    init(_ message: String, at location: ScriptLocation? = nil) {
        self.message = message
        self.location = location
    }

    static var reported: ConsoleError {
        var error = ConsoleError("")
        error.isReported = true
        return error
    }
}

// MARK: - Bound values

/// A bound argument or option value, already converted to its declared type.
enum ConsoleValue {
    case text(String)
    case int(Int)
    case float(Double)
    case bool(Bool)
    case url(URL)
    case vec3(SIMD3<Float>)
}

/// A statement bound to one command form, handed to the form's `run` closure.
@MainActor
struct ConsoleInvocation {
    let interpreter: DruseInterpreter
    let values: [String: ConsoleValue]

    var viewModel: AppViewModel { interpreter.viewModel }
    var isHeadless: Bool { interpreter.mode == .headless }

    func emit(_ kind: ConsoleOutputKind, _ text: String) {
        interpreter.emit(kind, text)
    }

    func has(_ name: String) -> Bool { values[name] != nil }

    func string(_ name: String) -> String? {
        if case .text(let value) = values[name] { return value }
        return nil
    }

    func int(_ name: String) -> Int? {
        if case .int(let value) = values[name] { return value }
        return nil
    }

    func double(_ name: String) -> Double? {
        if case .float(let value) = values[name] { return value }
        return nil
    }

    func bool(_ name: String) -> Bool? {
        if case .bool(let value) = values[name] { return value }
        return nil
    }

    func url(_ name: String) -> URL? {
        if case .url(let value) = values[name] { return value }
        return nil
    }

    func vec3(_ name: String) -> SIMD3<Float>? {
        if case .vec3(let value) = values[name] { return value }
        return nil
    }

    // Required positional arguments: binding guarantees they are present.

    func requiredURL(_ name: String) -> URL {
        guard let value = url(name) else { preconditionFailure("required <\(name)> not bound") }
        return value
    }

    func requiredString(_ name: String) -> String {
        guard let value = string(name) else { preconditionFailure("required <\(name)> not bound") }
        return value
    }

    func requiredInt(_ name: String) -> Int {
        guard let value = int(name) else { preconditionFailure("required <\(name)> not bound") }
        return value
    }
}

// MARK: - Binding

/// Matches parsed statements to command forms and converts their words to typed values.
enum ConsoleBinder {

    /// Picks the form a statement addresses and converts its words to typed values.
    static func bind(
        _ statement: ScriptCommand,
        to command: ConsoleCommand,
        baseDirectory: URL
    ) throws -> (form: ConsoleForm, values: [String: ConsoleValue]) {
        let arguments = statement.arguments
        let form = try selectForm(statement, command: command)
        let positional = Array(arguments.dropFirst(form.path.count))
        var values: [String: ConsoleValue] = [:]

        let usage = "usage: \(ConsoleHelp.usage(command: command.name, form: form))"
        let isVariadic = form.arguments.last?.isVariadic == true
        if positional.count > form.arguments.count && !isVariadic {
            let extra = positional[form.arguments.count]
            var message = "unexpected argument '\(extra.text)' for '\(display(command, form))'"
            let subcommands = command.subcommandPaths
            if form.arguments.isEmpty, !extra.isQuoted,
               let hint = ConsoleSuggest.closest(extra.text.lowercased(), in: command.forms.compactMap(\.path.first)) {
                message += " — did you mean '\(command.name) \(hint)'?"
            } else if form.path.isEmpty, form.arguments.isEmpty, !subcommands.isEmpty {
                message += "; subcommands: \(subcommands.joined(separator: ", "))"
            } else {
                message += "; \(usage)"
            }
            throw ConsoleError(message, at: extra.location)
        }
        for (index, parameter) in form.arguments.enumerated() {
            guard index < positional.count else {
                if parameter.isOptional { continue }
                throw ConsoleError("missing <\(parameter.name)>; \(usage)", at: statement.location)
            }
            if parameter.isVariadic {
                values[parameter.name] = .text(positional[index...].map(\.text).joined(separator: " "))
            } else {
                values[parameter.name] = try convert(positional[index], for: parameter, baseDirectory: baseDirectory)
            }
        }

        for key in statement.options.keys.sorted() {
            guard let word = statement.options[key] else { continue }
            guard let parameter = form.options.first(where: { $0.name == key }) else {
                var message = "unknown option '\(key)' for '\(display(command, form))'"
                if form.options.isEmpty {
                    message += " (it takes no options)"
                } else if let hint = ConsoleSuggest.closest(key, in: form.options.map(\.name)) {
                    message += " — did you mean '\(hint)='?"
                } else {
                    message += "; options: \(form.options.map(\.name).joined(separator: ", "))"
                }
                throw ConsoleError(message, at: word.location)
            }
            values[key] = try convert(word, for: parameter, baseDirectory: baseDirectory)
        }
        return (form, values)
    }

    /// "ligand db import": the words that address a form, for messages.
    static func display(_ command: ConsoleCommand, _ form: ConsoleForm) -> String {
        ([command.name] + form.path).joined(separator: " ")
    }

    /// Longest subcommand path matching the leading bare words wins; otherwise the bare form.
    private static func selectForm(_ statement: ScriptCommand, command: ConsoleCommand) throws -> ConsoleForm {
        let leading = statement.arguments.prefix { !$0.isQuoted }.map { $0.text.lowercased() }
        let matching = command.forms
            .filter { !$0.path.isEmpty && leading.starts(with: $0.path) }
            .max { $0.path.count < $1.path.count }
        if let matching { return matching }
        if let bare = command.forms.first(where: { $0.path.isEmpty }) { return bare }

        let subcommands = command.subcommandPaths
        guard let first = statement.arguments.first else {
            throw ConsoleError("'\(command.name)' needs a subcommand: \(subcommands.joined(separator: ", "))",
                               at: statement.location)
        }
        // A partial multi-word path ("ligand db") or a typo ("ligand lsit").
        let typed = leading.prefix(2).joined(separator: " ")
        let candidates = subcommands.filter { $0.hasPrefix(typed + " ") }
        if !typed.isEmpty, !candidates.isEmpty {
            throw ConsoleError("'\(command.name) \(typed)' needs one of: \(candidates.joined(separator: ", "))",
                               at: first.location)
        }
        var message = "unknown subcommand '\(first.text)' for '\(command.name)'"
        let firstWords = Array(Set(command.forms.compactMap(\.path.first))).sorted()
        if let hint = ConsoleSuggest.closest(first.text.lowercased(), in: firstWords) {
            message += " — did you mean '\(command.name) \(hint)'?"
        } else {
            message += "; expected: \(subcommands.joined(separator: ", "))"
        }
        throw ConsoleError(message, at: first.location)
    }

    /// Converts a word to the parameter's type and enforces its declared numeric range.
    static func convert(_ word: ScriptWord, for parameter: ConsoleParameter, baseDirectory: URL) throws -> ConsoleValue {
        let value = try convert(word, as: parameter.type, name: parameter.name, baseDirectory: baseDirectory)
        guard let range = parameter.range else { return value }
        let number: Double
        switch value {
        case .int(let int): number = Double(int)
        case .float(let float): number = float
        default: return value
        }
        guard range.contains(number) else {
            throw ConsoleError("\(parameter.name) must be between \(ConsoleHelp.number(range.lowerBound)) and "
                               + "\(ConsoleHelp.number(range.upperBound)), got '\(word.text)'", at: word.location)
        }
        return value
    }

    /// Converts a word to a typed value (no range check); errors name the parameter and point at the word.
    static func convert(_ word: ScriptWord, as type: ConsoleValueType,
                        name: String, baseDirectory: URL) throws -> ConsoleValue {
        let text = word.text
        switch type {
        case .string:
            return .text(text)
        case .int:
            guard let value = Int(text) else {
                throw ConsoleError("\(name) expects a whole number, got '\(text)'", at: word.location)
            }
            return .int(value)
        case .float:
            guard let value = Double(text), value.isFinite else {
                throw ConsoleError("\(name) expects a number, got '\(text)'", at: word.location)
            }
            return .float(value)
        case .bool:
            switch text.lowercased() {
            case "on", "true", "yes", "1": return .bool(true)
            case "off", "false", "no", "0": return .bool(false)
            default:
                throw ConsoleError("\(name) expects on or off, got '\(text)'", at: word.location)
            }
        case .path:
            return .url(resolvePath(text, relativeTo: baseDirectory))
        case .vec3:
            let parts = text.split(separator: ",").map { Float($0.trimmingCharacters(in: .whitespaces)) }
            guard parts.count == 3, let x = parts[0], let y = parts[1], let z = parts[2] else {
                throw ConsoleError("\(name) expects x,y,z (three numbers, no spaces), got '\(text)'", at: word.location)
            }
            return .vec3(SIMD3(x, y, z))
        case .choice(let values):
            let lowered = text.lowercased()
            guard values.contains(lowered) else {
                var message = "unknown value '\(text)' for \(name) (expected \(values.joined(separator: "|")))"
                if let hint = ConsoleSuggest.closest(lowered, in: values) {
                    message += " — did you mean '\(hint)'?"
                }
                throw ConsoleError(message, at: word.location)
            }
            return .text(lowered)
        }
    }

    /// Expands `~` and resolves relative paths against the current script directory.
    static func resolvePath(_ text: String, relativeTo base: URL) -> URL {
        let expanded = (text as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return base.appendingPathComponent(expanded).standardizedFileURL
    }
}

// MARK: - Suggestions

/// "Did you mean" suggestions for mistyped commands, subcommands, options and values.
enum ConsoleSuggest {
    /// The closest candidate within a small edit distance (or sharing the typed prefix).
    static func closest(_ word: String, in candidates: [String]) -> String? {
        guard !word.isEmpty else { return nil }
        if let prefixed = candidates.first(where: { $0.hasPrefix(word) }) { return prefixed }
        let limit = max(1, min(3, word.count / 3 + 1))
        let scored = candidates.map { ($0, distance(word, $0)) }.filter { $0.1 <= limit }
        return scored.min { $0.1 < $1.1 }?.0
    }

    /// Levenshtein distance.
    static func distance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var previous = Array(0...b.count)
        var current = [Int](repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            current[0] = i
            for j in 1...b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            }
            swap(&previous, &current)
        }
        return previous[b.count]
    }
}

// MARK: - Help text

/// Generates every piece of user-facing help from the registry: the categorized overview,
/// per-command and per-subcommand detail, usage lines and `apropos` search results.
enum ConsoleHelp {

    /// Forms with more options than this show `[option=value…]` in their usage line; the
    /// options are listed one per line underneath in `help`.
    private static let maxInlineOptions = 4

    /// One usage line: `dock run <ligand> [seeds=N] [out=path]`.
    static func usage(command: String, form: ConsoleForm) -> String {
        if let syntax = form.syntax { return syntax }
        var parts = [command] + form.path
        for argument in form.arguments {
            let name = argument.isVariadic ? argument.name + "…" : argument.name
            parts.append(argument.isOptional ? "[\(name)]" : "<\(name)>")
        }
        if form.options.count > maxInlineOptions {
            parts.append("[option=value…]")
        } else {
            for option in form.options {
                parts.append("[\(option.name)=\(option.type.placeholder)]")
            }
        }
        return parts.joined(separator: " ")
    }

    /// `help` with no argument: commands grouped by category with their one-line summaries,
    /// followed by a reminder of the statement syntax.
    static func overview(_ commands: [ConsoleCommand], style: ConsoleTextStyle) -> String {
        var lines = [style.heading("DruseScript commands")
                     + style.dim(" — 'help <command>' for details, 'apropos <word>' to search")]
        lines += commandList(commands, style: style)
        lines.append("")
        lines.append(style.heading("Syntax"))
        lines += style.columns([
            ("verb [subcommand] [args…] [key=value…]", "one statement per line, or several separated by ';'"),
            ("# comment", "ignored to the end of the line"),
            ("\"quoted text\"", "one word containing spaces"),
            ("\\", "at the end of a line, continues the statement on the next one"),
            ("set name = value", "defines a variable, used later as $name or ${name}"),
            ("set onerror=continue", "keeps going after a failed statement (default: stop)"),
            ("log level <level>", "filters the activity-log output")
        ], label: style.command)
        return lines.joined(separator: "\n")
    }

    /// Every command under its category heading, names aligned on one column across categories.
    /// Headings start `indent` columns in; commands two further.
    static func commandList(_ commands: [ConsoleCommand], style: ConsoleTextStyle, indent: Int = 0) -> [String] {
        let width = commands.map(\.name.count).max() ?? 0
        let margin = String(repeating: " ", count: indent)
        var lines: [String] = []
        for category in ConsoleCategory.allCases {
            let members = commands.filter { $0.category == category }
            guard !members.isEmpty else { continue }
            lines.append("")
            lines.append(margin + style.heading(category.rawValue))
            lines += style.columns(members.map { ($0.name, $0.summary) }, indent: indent + 2, labelWidth: width,
                                   label: style.command)
        }
        return lines
    }

    /// `help <command> [subcommand…]`: every addressed form with its arguments and options
    /// (type, range, default), followed by the matching examples.
    /// `path` narrows the output to forms whose subcommand path starts with it.
    static func detail(_ command: ConsoleCommand, path: [String] = [], style: ConsoleTextStyle) -> String {
        let forms = command.forms.filter { $0.path.starts(with: path) }
        let title = ([command.name] + path).joined(separator: " ")
        let summary = path.isEmpty ? command.summary : forms.first { $0.path == path }?.summary ?? command.summary
        var lines = [style.command(title) + " — " + summary]
        if path.isEmpty, command.category != .general {
            lines[0] += style.dim("  [\(command.category.rawValue)]")
        }
        for form in forms {
            lines.append("")
            let verb = ([command.name] + form.path).joined(separator: " ")
            let usageLine = usage(command: command.name, form: form)
            lines.append("  " + (usageLine.hasPrefix(verb)
                                 ? style.command(verb) + String(usageLine.dropFirst(verb.count))
                                 : usageLine))
            lines += style.paragraph(form.summary + (form.isGUIOnly ? " " + style.dim("(GUI only; ignored in headless mode)") : ""),
                                     indent: 6)
            let parameters = form.arguments.map { ($0, "<\($0.name)>") } + form.options.map { ($0, "\($0.name)=") }
            lines += style.columns(parameters.map { parameter, label in
                (label, parameter.summary + " " + style.dim("(" + facts(parameter) + ")"))
            }, indent: 8, label: { $0 })
        }
        let prefix = title + " "
        let examples = path.isEmpty ? command.examples : command.examples.filter { $0.hasPrefix(prefix) || $0 == title }
        if !examples.isEmpty {
            lines.append("")
            lines.append(style.heading("Examples"))
            lines.append(contentsOf: examples.map { "  " + $0 })
        }
        return lines.joined(separator: "\n")
    }

    /// "integer 1–64, default 8": type, bounds, default and optionality of one parameter.
    static func facts(_ parameter: ConsoleParameter) -> String {
        var facts = [parameter.type.typeDescription]
        if let range = parameter.range {
            facts[0] += " \(number(range.lowerBound))–\(number(range.upperBound))"
        }
        if let value = parameter.defaultValue {
            facts.append("default \(value)")
        } else if parameter.isOptional {
            facts.append("optional")
        }
        if parameter.isVariadic {
            facts.append("takes the rest of the line")
        }
        return facts.joined(separator: ", ")
    }

    /// `apropos <word>`: every form whose command name, subcommand words, summary, or parameter
    /// names and summaries contain the word (case-insensitive), one per line. A command whose
    /// own summary matches but none of whose forms do is listed once, by name, so a broad
    /// command summary does not flood the result with all of its forms.
    static func apropos(_ word: String, in commands: [ConsoleCommand], style: ConsoleTextStyle) -> String? {
        let needle = word.lowercased()
        func matches(_ text: String) -> Bool { text.lowercased().contains(needle) }
        var rows: [[String]] = []
        for command in commands {
            let nameMatches = matches(command.name)
            var formRows: [[String]] = []
            for form in command.forms {
                let parameters = form.arguments + form.options
                let hit = parameters.first { matches($0.name) || matches($0.summary) }
                let formMatches = matches(form.summary) || form.path.contains(where: matches)
                guard nameMatches || formMatches || hit != nil else { continue }
                var summary = form.summary
                if let hit, !(nameMatches || formMatches) {
                    summary += " — \(form.options.contains { $0.name == hit.name } ? hit.name + "=" : "<" + hit.name + ">"): \(hit.summary)"
                }
                formRows.append([ConsoleBinder.display(command, form), summary])
            }
            if formRows.isEmpty, matches(command.summary) {
                formRows = [[command.name, command.summary]]
            }
            rows += formRows
        }
        guard !rows.isEmpty else { return nil }
        return style.columns(rows.map { ($0[0], $0[1]) }, indent: 0, label: style.command).joined(separator: "\n")
    }

    /// Formats a range bound without a needless ".0".
    static func number(_ value: Double) -> String {
        value == value.rounded() && abs(value) < 1e15 ? String(Int(value)) : String(value)
    }
}

// MARK: - Plain-text tables

/// Aligned plain-text tables for command output (readable in the console and in a terminal).
enum ConsoleTable {
    /// Columns separated by two spaces; columns listed in `rightAligned` are right-aligned.
    static func render(_ headers: [String], _ rows: [[String]], rightAligned: Set<Int> = []) -> String {
        var widths = headers.map(\.count)
        for row in rows {
            for (i, cell) in row.enumerated() where i < widths.count {
                widths[i] = max(widths[i], cell.count)
            }
        }
        func line(_ cells: [String]) -> String {
            var text = cells.enumerated().map { i, cell in
                let pad = String(repeating: " ", count: max(0, widths[i] - cell.count))
                return rightAligned.contains(i) ? pad + cell : cell + pad
            }
            .joined(separator: "  ")
            while text.last == " " { text.removeLast() }
            return text
        }
        let separator = widths.map { String(repeating: "─", count: $0) }.joined(separator: "  ")
        return ([line(headers), separator] + rows.map(line)).joined(separator: "\n")
    }
}
