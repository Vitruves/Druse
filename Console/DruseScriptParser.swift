// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - DruseScript parser
//
// Turns DruseScript source into `[ScriptStatement]`. Lexing rules:
// - statements end at a newline or `;` (outside quotes); a `\` at the end of a line continues the statement;
// - `#` at the start of a word begins a comment that runs to the end of the line (`a#b` stays one word);
// - "double-quoted" strings support `\"`, `\\`, `\n` and `\t`, and cannot span lines;
// - `key=value` options take a bare or quoted value (`key="a b"`), with no spaces around `=`; a bare word
//   whose text before `=` is not an option name (e.g. the SMILES `CC(=O)O`) stays a positional argument.
// `$name` / `${name}` are kept verbatim: the interpreter substitutes them at run time.

/// Turns DruseScript source into statements; see the lexing rules above.
enum DruseScriptParser {
    /// Parses a whole script; the first syntax error fails the parse so nothing runs.
    static func parse(_ source: String) -> Result<[ScriptStatement], ScriptSyntaxError> {
        var lexer = Lexer(source)
        var statements: [ScriptStatement] = []
        do {
            while let raw = try lexer.nextStatement() {
                if let statement = try makeStatement(raw) {
                    statements.append(statement)
                }
            }
        } catch let error as ScriptSyntaxError {
            return .failure(error)
        } catch {
            return .failure(ScriptSyntaxError(message: "\(error)", location: ScriptLocation(line: 1, column: 1)))
        }
        return .success(statements)
    }

    // MARK: Statements

    /// Builds a statement from the tokens of one line: the verb, then arguments and `key=value` options.
    private static func makeStatement(_ raw: RawStatement) throws -> ScriptStatement? {
        guard let first = raw.tokens.first else { return nil }
        guard case .word(let verbWord) = first, !verbWord.isQuoted, !verbWord.text.contains("=") else {
            throw ScriptSyntaxError(
                message: "expected a command name, found '\(first.display)' (assignments are written 'set name = value')",
                location: first.location)
        }
        let verb = verbWord.text.lowercased()
        let rest = Array(raw.tokens.dropFirst())
        if verb == "set" {
            return try makeAssignment(rest, setLocation: verbWord.location)
        }

        var arguments: [ScriptWord] = []
        var options: [String: ScriptWord] = [:]
        for token in rest {
            switch token {
            case .quotedOption(let key, _, let value):
                options[key.lowercased()] = value
            case .word(let word):
                // A word whose text before `=` cannot be an option name (an unquoted SMILES such as
                // `CC(=O)O`) stays a positional argument.
                guard !word.isQuoted, let split = try splitOption(word), isOptionName(split.key) else {
                    arguments.append(word)
                    continue
                }
                guard !split.value.text.isEmpty else {
                    throw ScriptSyntaxError(
                        message: "missing value for option '\(split.key)' (write \(split.key)=\"\" for an empty value)",
                        location: split.value.location)
                }
                options[split.key.lowercased()] = split.value
            }
        }
        return .command(ScriptCommand(
            verb: verb, arguments: arguments, options: options,
            location: verbWord.location, source: raw.source.trimmingCharacters(in: .whitespaces)))
    }

    /// Accepts `set name = value`, `set name=value`, `set name =value` and `set name= value`.
    private static func makeAssignment(_ tokens: [Token], setLocation: ScriptLocation) throws -> ScriptStatement {
        guard let nameToken = tokens.first else {
            throw ScriptSyntaxError(message: "'set' needs a variable name: set name = value", location: setLocation)
        }
        let name: String
        let nameLocation = nameToken.location
        var valueToken: Token?
        let remainder: ArraySlice<Token>

        switch nameToken {
        case .quotedOption(let key, _, let value):
            name = key
            valueToken = .word(value)
            remainder = tokens.dropFirst(1)
        case .word(let word) where word.isQuoted:
            throw ScriptSyntaxError(message: "variable name must not be quoted", location: word.location)
        case .word(let word):
            if let split = try splitOption(word) {
                name = split.key
                if split.value.text.isEmpty {
                    valueToken = tokens.dropFirst(1).first
                    remainder = tokens.dropFirst(2)
                } else {
                    valueToken = .word(split.value)
                    remainder = tokens.dropFirst(1)
                }
            } else {
                name = word.text
                guard tokens.count > 1, case .word(let equals) = tokens[1], !equals.isQuoted, equals.text.hasPrefix("=") else {
                    throw ScriptSyntaxError(
                        message: "expected '=' after 'set \(word.text)'",
                        location: tokens.count > 1 ? tokens[1].location : word.location)
                }
                if equals.text == "=" {
                    valueToken = tokens.dropFirst(2).first
                    remainder = tokens.dropFirst(3)
                } else {
                    valueToken = .word(ScriptWord(
                        text: String(equals.text.dropFirst()), isQuoted: false,
                        location: ScriptLocation(line: equals.location.line, column: equals.location.column + 1)))
                    remainder = tokens.dropFirst(2)
                }
            }
        }

        guard isIdentifier(name) else {
            throw ScriptSyntaxError(
                message: "invalid variable name '\(name)' (use letters, digits and '_', not starting with a digit)",
                location: nameLocation)
        }
        guard let valueToken else {
            throw ScriptSyntaxError(message: "missing value after '=' in 'set \(name)'", location: nameLocation)
        }
        guard case .word(let value) = valueToken, value.isQuoted || !value.text.contains("=") else {
            throw ScriptSyntaxError(
                message: "unexpected '\(valueToken.display)' as value; quote values that contain '='",
                location: valueToken.location)
        }
        if let extra = remainder.first {
            throw ScriptSyntaxError(
                message: "unexpected '\(extra.display)' after the value; quote values that contain spaces",
                location: extra.location)
        }
        return .assignment(name: name, value: value, location: setLocation)
    }

    /// Splits a bare `key=value` word at its first `=`; nil when there is no `=`.
    private static func splitOption(_ word: ScriptWord) throws -> (key: String, value: ScriptWord)? {
        guard let equals = word.text.firstIndex(of: "=") else { return nil }
        let key = String(word.text[..<equals])
        guard !key.isEmpty else {
            throw ScriptSyntaxError(message: "'=' without an option name before it", location: word.location)
        }
        let valueLocation = ScriptLocation(line: word.location.line, column: word.location.column + key.count + 1)
        let value = ScriptWord(text: String(word.text[word.text.index(after: equals)...]), isQuoted: false, location: valueLocation)
        return (key, value)
    }

    private static func isIdentifier(_ text: String) -> Bool {
        guard let first = text.unicodeScalars.first, first == "_" || isASCIILetter(first) else { return false }
        return text.unicodeScalars.allSatisfy { $0 == "_" || isASCIILetter($0) || ("0"..."9").contains($0) }
    }

    /// Option names: a letter, then letters, digits, `_` or `-`.
    private static func isOptionName(_ text: String) -> Bool {
        guard let first = text.unicodeScalars.first, isASCIILetter(first) else { return false }
        return text.unicodeScalars.allSatisfy { $0 == "_" || $0 == "-" || isASCIILetter($0) || ("0"..."9").contains($0) }
    }

    private static func isASCIILetter(_ scalar: Unicode.Scalar) -> Bool {
        ("a"..."z").contains(scalar) || ("A"..."Z").contains(scalar)
    }
}

// MARK: - Lexer

/// A lexed word (bare or quoted) or a `key="quoted value"` option.
private enum Token {
    /// A bare or quoted word (bare words may still be `key=value`, split by the parser).
    case word(ScriptWord)
    /// `key="quoted value"`.
    case quotedOption(key: String, keyLocation: ScriptLocation, value: ScriptWord)

    var location: ScriptLocation {
        switch self {
        case .word(let word): return word.location
        case .quotedOption(_, let location, _): return location
        }
    }

    var display: String {
        switch self {
        case .word(let word): return word.isQuoted ? "\"\(word.text)\"" : word.text
        case .quotedOption(let key, _, let value): return "\(key)=\"\(value.text)\""
        }
    }
}

/// The tokens and source text of one statement, before it is interpreted.
private struct RawStatement {
    let tokens: [Token]
    let source: String
}

/// Splits source into statements of tokens, handling quotes, comments, `;` and line continuations.
private struct Lexer {
    private let source: String
    private var index: String.Index
    private var line = 1
    private var column = 1
    /// Source text of the statement being lexed (comments dropped, continuations folded to a space).
    private var text = ""

    init(_ source: String) {
        self.source = source
        index = source.startIndex
    }

    private var here: ScriptLocation { ScriptLocation(line: line, column: column) }
    private var current: Character? { index < source.endIndex ? source[index] : nil }

    private static func isNewline(_ c: Character) -> Bool { c == "\n" || c == "\r" || c == "\r\n" }
    private static func isBlank(_ c: Character) -> Bool { c == " " || c == "\t" }

    /// Moves past the current character without recording it.
    private mutating func skip() {
        guard let c = current else { return }
        source.formIndex(after: &index)
        if Self.isNewline(c) { line += 1; column = 1 } else { column += 1 }
    }

    /// Records the current character into the statement text, then moves past it.
    private mutating func advance() {
        if let c = current { text.append(c) }
        skip()
    }

    /// The next statement's tokens (possibly none, for blank or comment lines); nil at end of input.
    mutating func nextStatement() throws -> RawStatement? {
        guard current != nil else { return nil }
        text = ""
        var tokens: [Token] = []
        while let c = current {
            if Self.isNewline(c) || c == ";" {
                skip()
                break
            } else if Self.isBlank(c) {
                advance()
            } else if c == "#" {
                while let next = current, !Self.isNewline(next) { skip() }
            } else if c == "\\" {
                try lexContinuation()
            } else if c == "\"" {
                let start = here
                tokens.append(.word(ScriptWord(text: try lexQuoted(), isQuoted: true, location: start)))
                try expectWordEnd()
            } else {
                tokens.append(try lexBare())
            }
        }
        return RawStatement(tokens: tokens, source: text)
    }

    /// `\` followed only by blanks and a newline joins the next line to this statement.
    private mutating func lexContinuation() throws {
        let start = here
        skip()
        while let c = current, Self.isBlank(c) { skip() }
        guard let c = current, Self.isNewline(c) else {
            throw ScriptSyntaxError(
                message: current == nil
                    ? "line continuation '\\' at end of input"
                    : "stray '\\' (it only continues a statement at the end of a line; use it inside quotes as \\\\)",
                location: start)
        }
        skip()
        text.append(" ")
    }

    /// Reads a "quoted string" starting at the opening quote and returns its unescaped contents.
    private mutating func lexQuoted() throws -> String {
        let start = here
        advance()
        var value = ""
        while true {
            guard let c = current, !Self.isNewline(c) else {
                throw ScriptSyntaxError(message: "unterminated string (missing closing '\"')", location: start)
            }
            if c == "\"" {
                advance()
                return value
            }
            if c == "\\" {
                let escapeLocation = here
                advance()
                switch current {
                case "\"": value.append("\"")
                case "\\": value.append("\\")
                case "n": value.append("\n")
                case "t": value.append("\t")
                default:
                    throw ScriptSyntaxError(
                        message: "invalid escape '\\\(current.map(String.init) ?? "")' in string (use \\\", \\\\, \\n or \\t)",
                        location: escapeLocation)
                }
            } else {
                value.append(c)
            }
            advance()
        }
    }

    private func expectWordEnd() throws {
        guard let c = current, !Self.isBlank(c), !Self.isNewline(c), c != ";", c != "#", c != "\\" else { return }
        throw ScriptSyntaxError(message: "unexpected '\(c)' right after a closing quote; separate words with a space", location: here)
    }

    /// Reads a bare word, or a `key="quoted value"` option when a quote follows `key=`.
    private mutating func lexBare() throws -> Token {
        let start = here
        var word = ""
        while let c = current, !Self.isBlank(c), !Self.isNewline(c), c != ";", c != "\\", c != "\"" {
            word.append(c)
            advance()
        }
        guard current == "\"", word != "=" else {
            return .word(ScriptWord(text: word, isQuoted: false, location: start))
        }
        guard word.hasSuffix("="), word.firstIndex(of: "=") == word.index(before: word.endIndex) else {
            throw ScriptSyntaxError(message: "unexpected '\"' inside a word; quote the whole word", location: here)
        }
        let key = String(word.dropLast())
        guard !key.isEmpty else {
            throw ScriptSyntaxError(message: "'=' without an option name before it", location: start)
        }
        let valueStart = here
        let value = ScriptWord(text: try lexQuoted(), isQuoted: true, location: valueStart)
        try expectWordEnd()
        return .quotedOption(key: key, keyLocation: start, value: value)
    }
}

// MARK: - Syntax highlighting

/// Role of a span of DruseScript source, for colouring the console input and echoed commands.
enum ScriptHighlightKind: Sendable, Equatable {
    /// First word of a statement (`dock`, `set`).
    case verb
    /// Any other bare word: subcommand or positional argument.
    case argument
    /// The key of `key=value`.
    case optionKey
    /// A bare option value after `=`.
    case optionValue
    /// A "quoted string", quotes included (also an unterminated one, to the end of the line).
    case string
    /// `$name` or `${name}`, inside bare words and strings alike.
    case variable
    /// `# …` to the end of the line.
    case comment
    /// `=`, `;` and a continuation `\`.
    case punctuation
}

/// One coloured span. Spans are emitted in source order; a `.variable` span lies inside the
/// word or string span before it and should be applied after it.
struct ScriptHighlight: Sendable, Equatable {
    let kind: ScriptHighlightKind
    let range: Range<String.Index>
}

/// Tolerant lexical colouring of DruseScript: never fails, so half-typed input (an open quote,
/// a trailing `=`) still gets coloured. It mirrors the parser's word rules without validating them.
enum DruseScriptHighlighter {
    /// Colour spans for the console's syntax highlighting; tolerant of incomplete input.
    static func highlight(_ source: String) -> [ScriptHighlight] {
        var spans: [ScriptHighlight] = []
        var index = source.startIndex
        var atStatementStart = true

        func isBreak(_ c: Character) -> Bool {
            c == " " || c == "\t" || c == "\n" || c == "\r" || c == "\r\n" || c == ";" || c == "\\" || c == "\""
        }

        while index < source.endIndex {
            let c = source[index]
            let next = source.index(after: index)
            switch c {
            case "\n", "\r", "\r\n":
                atStatementStart = true
                index = next
            case " ", "\t":
                index = next
            case ";":
                spans.append(ScriptHighlight(kind: .punctuation, range: index..<next))
                atStatementStart = true
                index = next
            case "\\":
                spans.append(ScriptHighlight(kind: .punctuation, range: index..<next))
                index = next
            case "#":
                let end = source[index...].firstIndex { $0 == "\n" || $0 == "\r" || $0 == "\r\n" } ?? source.endIndex
                spans.append(ScriptHighlight(kind: .comment, range: index..<end))
                index = end
            case "\"":
                let end = stringEnd(in: source, openingQuote: index)
                spans.append(ScriptHighlight(kind: .string, range: index..<end))
                spans += variables(in: source, index..<end)
                atStatementStart = false
                index = end
            default:
                let end = source[index...].firstIndex(where: isBreak) ?? source.endIndex
                let word = index..<end
                if atStatementStart {
                    spans.append(ScriptHighlight(kind: .verb, range: word))
                } else if let equals = source[word].firstIndex(of: "="), equals > index {
                    let afterEquals = source.index(after: equals)
                    spans.append(ScriptHighlight(kind: .optionKey, range: index..<equals))
                    spans.append(ScriptHighlight(kind: .punctuation, range: equals..<afterEquals))
                    if afterEquals < end {
                        spans.append(ScriptHighlight(kind: .optionValue, range: afterEquals..<end))
                    }
                } else {
                    spans.append(ScriptHighlight(kind: .argument, range: word))
                }
                spans += variables(in: source, word)
                atStatementStart = false
                index = end
            }
        }
        return spans
    }

    /// End of a quoted string: after its closing quote, or at the end of the line when unterminated.
    private static func stringEnd(in source: String, openingQuote: String.Index) -> String.Index {
        var index = source.index(after: openingQuote)
        while index < source.endIndex {
            let c = source[index]
            if c == "\n" || c == "\r" || c == "\r\n" { return index }
            if c == "\"" { return source.index(after: index) }
            if c == "\\" {
                index = source.index(after: index)
                if index == source.endIndex { break }
            }
            index = source.index(after: index)
        }
        return source.endIndex
    }

    /// `$name` and `${name}` references inside a span.
    private static func variables(in source: String, _ range: Range<String.Index>) -> [ScriptHighlight] {
        var spans: [ScriptHighlight] = []
        var index = range.lowerBound
        func isName(_ c: Character) -> Bool { c == "_" || (c.isASCII && (c.isLetter || c.isNumber)) }
        while index < range.upperBound {
            guard source[index] == "$" else {
                index = source.index(after: index)
                continue
            }
            var end = source.index(after: index)
            if end < range.upperBound, source[end] == "{" {
                end = source[end..<range.upperBound].firstIndex(of: "}").map { source.index(after: $0) } ?? range.upperBound
            } else {
                while end < range.upperBound, isName(source[end]) { end = source.index(after: end) }
            }
            if source.distance(from: index, to: end) > 1 {
                spans.append(ScriptHighlight(kind: .variable, range: index..<end))
            }
            index = end
        }
        return spans
    }
}
