// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - Completion candidates

/// One completion for the word being typed at the end of a command line.
///
/// `range` addresses the line the completion was computed for; `text` replaces it (it already
/// carries the trailing space, `=` or `/` that lets the user keep typing). Rich UIs show `display`,
/// `kind` and `summary`; plain UIs only need `applied(to:)`.
struct ConsoleCompletion: Equatable, Sendable {
    /// What the candidate completes; drives the icon in the in-app popup.
    enum Kind: Sendable {
        case command, subcommand, option, value, variable, path
    }

    /// Text that replaces `range`.
    let text: String
    /// Label shown in lists ("exhaustiveness=", "ligands/", "$ph").
    let display: String
    let kind: Kind
    /// One-line description taken from the registry (or the value of a variable, or "directory").
    let summary: String
    /// Range of the partial word being replaced, in the line passed to the completer.
    let range: Range<String.Index>

    /// The full line with this completion applied.
    func applied(to line: String) -> String {
        line.replacingCharacters(in: range, with: text)
    }
}

// MARK: - Completer

/// Computes completions for the last statement of a command line from the command registry,
/// the defined `$variables` and the file system (for path-typed parameters). Pure apart from
/// reading directory listings, so it can serve the in-app console, a headless REPL or a
/// `druse` shell-completion helper alike.
struct ConsoleCompleter {
    let commands: [ConsoleCommand]
    let variables: [String: String]
    /// Directory relative path completions resolve against.
    let directory: URL

    /// Directory listings are capped so completing in a huge folder stays instant.
    private static let maxPathCandidates = 200

    /// Candidates for the word at the end of `line`, in display order (empty inside a comment).
    func completions(for line: String) -> [ConsoleCompletion] {
        completions(in: line[...])
    }

    /// Completions for a slice of the line; ranges index the underlying line, which lets
    /// `help <topic>` complete its topic as if it were typed at the start of a statement.
    private func completions(in line: Substring) -> [ConsoleCompletion] {
        guard let context = CompletionContext(line) else { return [] }
        let partial = context.partial

        if let variable = variableCompletions(partial) { return variable }

        guard let verbWord = context.words.first else {
            let typed = partial.text.lowercased()
            return commands.filter { $0.name.hasPrefix(typed) }.map {
                ConsoleCompletion(text: $0.name + " ", display: $0.name, kind: .command, summary: $0.summary, range: partial.range)
            }
        }
        guard !verbWord.isQuoted, let command = commands.first(where: { $0.name == verbWord.text.lowercased() }) else { return [] }

        if command.name == "help" {
            let topicStart = context.words.dropFirst().first?.range.lowerBound ?? partial.range.lowerBound
            return completions(in: line[topicStart...]).filter { $0.kind == .command || $0.kind == .subcommand }
        }

        let afterVerb = Array(context.words.dropFirst())
        let positional = afterVerb.filter { $0.isQuoted || !$0.text.contains("=") }
        let bareWords = positional.prefix { !$0.isQuoted }.map { $0.text.lowercased() }
        let form = Self.addressedForm(command, bareWords: bareWords)

        // key=partial-value
        if !partial.isQuoted, let equals = partial.text.firstIndex(of: "=") {
            let key = partial.text[..<equals].lowercased()
            let options = form?.options ?? command.forms.flatMap(\.options)
            guard let option = options.first(where: { $0.name == key }) else { return [] }
            let valueStart = line.base.index(partial.range.lowerBound, offsetBy: key.count + 1)
            let value = Word(text: String(partial.text[partial.text.index(after: equals)...]),
                             range: valueStart..<partial.range.upperBound, isQuoted: false)
            return valueCompletions(for: option, typed: value)
        }

        var candidates: [ConsoleCompletion] = []
        let typed = partial.text.lowercased()

        // Subcommand words continuing the bare words typed so far.
        if !partial.isQuoted, positional.count == bareWords.count {
            var seen = Set<String>()
            for candidate in command.forms where candidate.path.count > bareWords.count && candidate.path.starts(with: bareWords) {
                let next = candidate.path[bareWords.count]
                guard next.hasPrefix(typed), seen.insert(next).inserted else { continue }
                let path = bareWords + [next]
                let summary = command.forms.first { $0.path == path }?.summary
                    ?? "subcommands: " + command.forms.filter { $0.path.starts(with: path) }
                        .map { $0.path.dropFirst(path.count).joined(separator: " ") }.joined(separator: ", ")
                candidates.append(ConsoleCompletion(text: next + " ", display: next, kind: .subcommand,
                                                    summary: summary, range: partial.range))
            }
        }

        guard let form else { return candidates }

        // Value of the positional argument being typed.
        let index = positional.count - form.path.count
        if index >= 0, let argument = form.arguments.indices.contains(index) ? form.arguments[index]
            : (form.arguments.last?.isVariadic == true ? form.arguments.last : nil) {
            candidates += valueCompletions(for: argument, typed: partial)
        }

        // Option keys of the addressed form, minus the ones already given.
        if !partial.isQuoted {
            let used = Set(afterVerb.compactMap { word in
                word.text.firstIndex(of: "=").map { word.text[..<$0].lowercased() }
            })
            for option in form.options where !used.contains(option.name) && option.name.hasPrefix(typed) {
                candidates.append(ConsoleCompletion(text: option.name + "=", display: option.name + "=", kind: .option,
                                                    summary: option.summary, range: partial.range))
            }
        }

        // `set` also offers the variables already defined.
        if command.name == "set", afterVerb.isEmpty, !partial.isQuoted {
            let reserved = Set(form.options.map(\.name))
            for name in variables.keys.sorted() where !reserved.contains(name) && name.hasPrefix(partial.text) {
                candidates.append(ConsoleCompletion(text: name + " = ", display: name, kind: .variable,
                                                    summary: "= " + (variables[name] ?? ""), range: partial.range))
            }
        }
        return candidates
    }

    /// True when the line ends with a space right after a known verb or a subcommand prefix
    /// ("dock ", "ligand db "): the moment the in-app console opens its completion list without
    /// waiting for Tab.
    func isAtCommandBoundary(_ line: String) -> Bool {
        guard line.last == " ", let context = CompletionContext(line[...]), context.partial.text.isEmpty,
              let verb = context.words.first, !verb.isQuoted,
              let command = commands.first(where: { $0.name == verb.text.lowercased() }) else { return false }
        let rest = context.words.dropFirst()
        guard rest.allSatisfy({ !$0.isQuoted && !$0.text.contains("=") }) else { return false }
        let path = rest.map { $0.text.lowercased() }
        return command.forms.contains { $0.path.starts(with: path) }
    }

    // MARK: Forms

    /// The form the bare words address: the longest matching subcommand path, else the bare form.
    private static func addressedForm(_ command: ConsoleCommand, bareWords: [String]) -> ConsoleForm? {
        command.forms
            .filter { bareWords.starts(with: $0.path) }
            .max { $0.path.count < $1.path.count }
    }

    // MARK: Values

    /// Fixed values (choices, on/off) or file-system entries for one parameter.
    private func valueCompletions(for parameter: ConsoleParameter, typed: Word) -> [ConsoleCompletion] {
        if parameter.type.isPath {
            return pathCompletions(typed)
        }
        guard !typed.isQuoted else { return [] }
        let lowered = typed.text.lowercased()
        return parameter.type.completions.filter { $0.hasPrefix(lowered) }.map {
            ConsoleCompletion(text: $0 + " ", display: $0, kind: .value,
                              summary: parameter.summary, range: typed.range)
        }
    }

    /// `$na…` / `${na…` at the end of the word: the matching variable names, keeping the text before `$`.
    private func variableCompletions(_ word: Word) -> [ConsoleCompletion]? {
        guard let dollar = word.text.lastIndex(of: "$") else { return nil }
        var name = word.text[word.text.index(after: dollar)...]
        let braced = name.hasPrefix("{")
        if braced { name = name.dropFirst() }
        guard name.allSatisfy({ $0 == "_" || ($0.isASCII && ($0.isLetter || $0.isNumber)) }) else { return nil }
        let before = word.text[..<dollar]
        let quote = word.isQuoted ? "\"" : ""
        return variables.keys.sorted().filter { $0.hasPrefix(name) }.map { variable in
            let reference = braced ? "${\(variable)}" : "$\(variable)"
            return ConsoleCompletion(text: quote + before + reference, display: reference, kind: .variable,
                                     summary: "= " + (variables[variable] ?? ""), range: word.range)
        }
    }

    /// Entries of the directory being typed whose names start with the typed last component.
    /// Folders end in `/` so completion can continue into them; names needing quotes get quoted.
    private func pathCompletions(_ word: Word) -> [ConsoleCompletion] {
        let typed = word.text
        let typedDirectory: String
        let namePrefix: String
        if typed.isEmpty || typed.hasSuffix("/") {
            typedDirectory = typed
            namePrefix = ""
        } else if let slash = typed.lastIndex(of: "/") {
            typedDirectory = String(typed[...slash])
            namePrefix = String(typed[typed.index(after: slash)...])
        } else if typed == "~" {
            typedDirectory = "~/"
            namePrefix = ""
        } else {
            typedDirectory = ""
            namePrefix = typed
        }
        let directoryURL = typedDirectory.isEmpty
            ? directory
            : ConsoleBinder.resolvePath(typedDirectory, relativeTo: directory)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directoryURL.path) else { return [] }

        let showHidden = namePrefix.hasPrefix(".")
        let matches = names
            .filter { (showHidden || !$0.hasPrefix(".")) && $0.lowercased().hasPrefix(namePrefix.lowercased()) }
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
            .prefix(Self.maxPathCandidates)

        return matches.map { name in
            var isDirectory: ObjCBool = false
            FileManager.default.fileExists(atPath: directoryURL.appendingPathComponent(name).path, isDirectory: &isDirectory)
            let path = typedDirectory + name
            let needsQuotes = word.isQuoted || path.contains { " ;#\"\\=".contains($0) }
            let text: String
            if isDirectory.boolValue {
                text = (needsQuotes ? "\"" : "") + path + "/"
            } else {
                text = needsQuotes ? "\"\(path)\" " : path + " "
            }
            return ConsoleCompletion(text: text, display: name + (isDirectory.boolValue ? "/" : ""), kind: .path,
                                     summary: isDirectory.boolValue ? "folder" : "file", range: word.range)
        }
    }
}

// MARK: - Line scanning

/// A word of the statement being completed. `text` is unquoted; `range` covers the raw text
/// including an opening quote.
private struct Word {
    let text: String
    let range: Range<String.Index>
    let isQuoted: Bool
}

/// The last statement of a line split into finished words plus the partial word at the end
/// (empty when the line ends in a blank). Tolerates an unterminated quote; nil inside a comment.
private struct CompletionContext {
    let words: [Word]
    let partial: Word

    init?(_ line: Substring) {
        var words: [Word] = []
        var index = line.startIndex
        var current: (start: String.Index, text: String, quoted: Bool, inQuote: Bool)?

        func finish(at end: String.Index) {
            if let word = current {
                words.append(Word(text: word.text, range: word.start..<end, isQuoted: word.quoted))
            }
            current = nil
        }

        while index < line.endIndex {
            let c = line[index]
            if var word = current, word.inQuote {
                if c == "\"" {
                    word.inQuote = false
                } else if c == "\\", line.index(after: index) < line.endIndex {
                    index = line.index(after: index)
                    word.text.append(line[index])
                } else {
                    word.text.append(c)
                }
                current = word
            } else if c == " " || c == "\t" || c == "\n" || c == "\r" {
                finish(at: index)
                if c == "\n" || c == "\r" { words.removeAll() }
            } else if c == ";" {
                finish(at: index)
                words.removeAll()
            } else if c == "#" && current == nil {
                return nil
            } else if c == "\"" {
                if current == nil {
                    current = (index, "", true, true)
                } else {
                    current?.inQuote = true
                }
            } else {
                if current == nil { current = (index, "", false, false) }
                current?.text.append(c)
            }
            index = line.index(after: index)
        }

        if let word = current {
            partial = Word(text: word.text, range: word.start..<line.endIndex, isQuoted: word.quoted)
        } else {
            partial = Word(text: "", range: line.endIndex..<line.endIndex, isQuoted: false)
        }
        self.words = words
    }
}
