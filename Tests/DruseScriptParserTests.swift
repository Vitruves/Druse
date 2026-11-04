// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import Druse

/// DruseScript lexer/parser tests — pure Foundation, no GPU, no app state.
final class DruseScriptParserTests: XCTestCase {

    // MARK: - Helpers

    private func statements(_ source: String, file: StaticString = #filePath, line: UInt = #line) -> [ScriptStatement] {
        switch DruseScriptParser.parse(source) {
        case .success(let statements): return statements
        case .failure(let error):
            XCTFail("unexpected syntax error \(error)", file: file, line: line)
            return []
        }
    }

    private func commands(_ source: String, file: StaticString = #filePath, line: UInt = #line) -> [ScriptCommand] {
        statements(source, file: file, line: line).compactMap {
            if case .command(let command) = $0 { return command }
            return nil
        }
    }

    private func syntaxError(_ source: String, file: StaticString = #filePath, line: UInt = #line) -> ScriptSyntaxError? {
        if case .failure(let error) = DruseScriptParser.parse(source) { return error }
        XCTFail("expected a syntax error for \(source.debugDescription)", file: file, line: line)
        return nil
    }

    private func loc(_ line: Int, _ column: Int) -> ScriptLocation { ScriptLocation(line: line, column: column) }

    // MARK: - Commands

    func testCommandWithArgumentsAndOptions() {
        let parsed = commands("Dock run Exhaustiveness=8 out=\"my file.sdf\"")
        XCTAssertEqual(parsed.count, 1)
        let command = parsed[0]
        XCTAssertEqual(command.verb, "dock")
        XCTAssertEqual(command.location, loc(1, 1))
        XCTAssertEqual(command.arguments, [ScriptWord(text: "run", isQuoted: false, location: loc(1, 6))])
        XCTAssertEqual(command.options["exhaustiveness"], ScriptWord(text: "8", isQuoted: false, location: loc(1, 25)))
        XCTAssertEqual(command.options["out"], ScriptWord(text: "my file.sdf", isQuoted: true, location: loc(1, 31)))
        XCTAssertEqual(command.source, "Dock run Exhaustiveness=8 out=\"my file.sdf\"")
    }

    func testLastDuplicateOptionWins() {
        XCTAssertEqual(commands("dock seeds=1 SEEDS=2").first?.options["seeds"]?.text, "2")
    }

    func testStatementSeparatorsCommentsAndBlankLines() {
        let parsed = commands("\n# header\nload pdb 1HSG   # trailing comment\nprepare; dock run;;\r\nstatus")
        XCTAssertEqual(parsed.map(\.verb), ["load", "prepare", "dock", "status"])
        XCTAssertEqual(parsed[0].source, "load pdb 1HSG")
        XCTAssertEqual(parsed[1].location, loc(4, 1))
        XCTAssertEqual(parsed[2].location, loc(4, 10))
        XCTAssertEqual(parsed[3].location, loc(5, 1))
    }

    func testHashInsideWordIsNotAComment() {
        XCTAssertEqual(commands("echo a#b").first?.arguments.map(\.text), ["a#b"])
    }

    func testQuotedStringsAndEscapes() {
        let command = commands("echo \"a\\\"b\\\\c\" \"tab\\there\" bare").first
        XCTAssertEqual(command?.arguments.map(\.text), ["a\"b\\c", "tab\there", "bare"])
        XCTAssertEqual(command?.arguments.map(\.isQuoted), [true, true, false])
    }

    func testQuotedWordIsNotAnOption() {
        let command = commands("echo \"a=b\"").first
        XCTAssertEqual(command?.options.isEmpty, true)
        XCTAssertEqual(command?.arguments.first?.text, "a=b")
    }

    func testVariablesStayVerbatim() {
        let command = commands("load $file name=${target} \"$quoted\"").first
        XCTAssertEqual(command?.arguments.map(\.text), ["$file", "$quoted"])
        XCTAssertEqual(command?.options["name"]?.text, "${target}")
    }

    func testLineContinuation() {
        let command = commands("dock run \\\n    seeds=3").first
        XCTAssertEqual(command?.arguments.map(\.text), ["run"])
        XCTAssertEqual(command?.options["seeds"]?.location, loc(2, 11))
        XCTAssertFalse(command?.source.contains("\\") ?? true)
    }

    // MARK: - Assignments

    func testAssignmentForms() {
        let parsed = statements("set x = 5\nset y=\"a b\"\nset z= 3\nset w =4\nset _a1=$x")
        XCTAssertEqual(parsed, [
            .assignment(name: "x", value: ScriptWord(text: "5", isQuoted: false, location: loc(1, 9)), location: loc(1, 1)),
            .assignment(name: "y", value: ScriptWord(text: "a b", isQuoted: true, location: loc(2, 7)), location: loc(2, 1)),
            .assignment(name: "z", value: ScriptWord(text: "3", isQuoted: false, location: loc(3, 8)), location: loc(3, 1)),
            .assignment(name: "w", value: ScriptWord(text: "4", isQuoted: false, location: loc(4, 8)), location: loc(4, 1)),
            .assignment(name: "_a1", value: ScriptWord(text: "$x", isQuoted: false, location: loc(5, 9)), location: loc(5, 1)),
        ])
    }

    func testSetIsCaseInsensitiveAndAllowsEmptyQuotedValue() {
        XCTAssertEqual(statements("SET name = \"\""), [
            .assignment(name: "name", value: ScriptWord(text: "", isQuoted: true, location: loc(1, 12)), location: loc(1, 1)),
        ])
    }

    func testUnquotedSMILESStaysPositional() {
        XCTAssertEqual(statements("ligand smiles CC(=O)O name=acid"), [
            .command(ScriptCommand(
                verb: "ligand",
                arguments: [
                    ScriptWord(text: "smiles", isQuoted: false, location: loc(1, 8)),
                    ScriptWord(text: "CC(=O)O", isQuoted: false, location: loc(1, 15)),
                ],
                options: ["name": ScriptWord(text: "acid", isQuoted: false, location: loc(1, 28))],
                location: loc(1, 1), source: "ligand smiles CC(=O)O name=acid")),
        ])
    }

    // MARK: - Errors

    func testErrors() {
        let cases: [(String, ScriptLocation, String)] = [
            ("echo \"abc", loc(1, 6), "unterminated string"),
            ("echo =3", loc(1, 6), "'=' without an option name"),
            ("echo key=", loc(1, 10), "missing value for option 'key'"),
            ("set", loc(1, 1), "'set' needs a variable name"),
            ("set x", loc(1, 5), "expected '='"),
            ("set x =", loc(1, 5), "missing value"),
            ("set 1x = 3", loc(1, 5), "invalid variable name '1x'"),
            ("set x = 1 2", loc(1, 11), "after the value"),
            ("set \"x\" = 1", loc(1, 5), "must not be quoted"),
            ("echo a\\b", loc(1, 7), "stray '\\'"),
            ("echo trailing \\", loc(1, 15), "end of input"),
            ("echo \"a\"b", loc(1, 9), "after a closing quote"),
            ("echo a\"b\"", loc(1, 7), "inside a word"),
            ("echo \"bad \\q\"", loc(1, 11), "invalid escape"),
            ("\"dock\" run", loc(1, 1), "expected a command name"),
            ("x=1", loc(1, 1), "expected a command name"),
            ("help\n  echo \"open", loc(2, 8), "unterminated string"),
        ]
        for (source, location, fragment) in cases {
            guard let error = syntaxError(source) else { continue }
            XCTAssertEqual(error.location, location, "location for \(source.debugDescription)")
            XCTAssertTrue(error.message.contains(fragment), "\(source.debugDescription): \(error.message)")
        }
    }

    func testEmptySourceParsesToNothing() {
        XCTAssertEqual(statements(""), [])
        XCTAssertEqual(statements("  # just a comment\n\n"), [])
    }

    // MARK: - Session settings syntax

    func testSessionSettingAssignments() {
        XCTAssertEqual(statements("set onerror=continue\nset onerror = stop"), [
            .assignment(name: "onerror", value: ScriptWord(text: "continue", isQuoted: false, location: loc(1, 13)), location: loc(1, 1)),
            .assignment(name: "onerror", value: ScriptWord(text: "stop", isQuoted: false, location: loc(2, 15)), location: loc(2, 1)),
        ])
    }

    func testLogLevelIsAnOrdinaryCommand() {
        let command = commands("log level warn").first
        XCTAssertEqual(command?.verb, "log")
        XCTAssertEqual(command?.arguments.map(\.text), ["level", "warn"])
    }

    // MARK: - Highlighting

    private func spans(_ source: String) -> [(ScriptHighlightKind, String)] {
        DruseScriptHighlighter.highlight(source).map { ($0.kind, String(source[$0.range])) }
    }

    func testHighlightCommandWordsOptionsAndComments() {
        let result = spans("dock run seeds=3 out=\"a b.sdf\" # note")
        XCTAssertEqual(result.map(\.0), [.verb, .argument, .optionKey, .punctuation, .optionValue,
                                         .optionKey, .punctuation, .string, .comment])
        XCTAssertEqual(result.map(\.1), ["dock", "run", "seeds", "=", "3", "out", "=", "\"a b.sdf\"", "# note"])
    }

    func testHighlightVariablesInsideWordsAndStrings() {
        let result = spans("echo $ph \"at ${target}\"")
        XCTAssertEqual(result.filter { $0.0 == .variable }.map(\.1), ["$ph", "${target}"])
    }

    func testHighlightNewStatementAfterSemicolonAndNewline() {
        let verbs = spans("load a; prepare\nstatus").filter { $0.0 == .verb }.map(\.1)
        XCTAssertEqual(verbs, ["load", "prepare", "status"])
    }

    func testHighlightToleratesUnterminatedString() {
        let result = spans("echo \"open")
        XCTAssertEqual(result.last?.0, .string)
        XCTAssertEqual(result.last?.1, "\"open")
    }

    func testHighlightHashInsideWordIsNotAComment() {
        XCTAssertEqual(spans("echo a#b").map(\.0), [.verb, .argument])
    }

    // MARK: - Completion

    /// A tiny registry with a subcommand tree, options, choices and a path argument.
    @MainActor
    private func completer(variables: [String: String] = [:], directory: URL = FileManager.default.temporaryDirectory) -> ConsoleCompleter {
        let commands = [
            ConsoleCommand(name: "dock", summary: "Dock", forms: [
                ConsoleForm(path: [], summary: "Run docking",
                            options: [ConsoleParameter(name: "search", type: .choice(["ga", "remc"]), summary: "Search"),
                                      ConsoleParameter(name: "seeds", type: .int, summary: "Seeds")]) { _ in },
                ConsoleForm(path: ["stop"], summary: "Stop docking") { _ in },
            ]),
            ConsoleCommand(name: "help", summary: "Help", forms: [
                ConsoleForm(path: [], summary: "Show help",
                            arguments: [ConsoleParameter(name: "topic", type: .string, summary: "Topic",
                                                         isOptional: true, isVariadic: true)]) { _ in },
            ]),
            ConsoleCommand(name: "load", summary: "Load", forms: [
                ConsoleForm(path: [], summary: "Load a file",
                            arguments: [ConsoleParameter(name: "file", type: .path, summary: "File")]) { _ in },
            ]),
        ]
        return ConsoleCompleter(commands: commands, variables: variables, directory: directory)
    }

    @MainActor
    func testCompletesCommandNames() {
        let result = completer().completions(for: "do")
        XCTAssertEqual(result.map(\.text), ["dock "])
        XCTAssertEqual(result.first?.kind, .command)
        XCTAssertEqual(result.first?.applied(to: "do"), "dock ")
    }

    @MainActor
    func testCompletesSubcommandsAndOptions() {
        let result = completer().completions(for: "dock ")
        XCTAssertEqual(result.map(\.display), ["stop", "search=", "seeds="])
        XCTAssertEqual(result.map(\.kind), [.subcommand, .option, .option])
        XCTAssertEqual(completer().completions(for: "dock seeds=2 s").map(\.display), ["stop", "search="])
    }

    @MainActor
    func testCompletesChoiceValues() {
        let result = completer().completions(for: "dock search=r")
        XCTAssertEqual(result.map { $0.applied(to: "dock search=r") }, ["dock search=remc "])
        XCTAssertEqual(result.first?.kind, .value)
    }

    @MainActor
    func testCompletesVariablesAndStatementsAfterSemicolon() {
        let completer = completer(variables: ["ligand": "x.sdf", "lig2": "y"])
        XCTAssertEqual(completer.completions(for: "load ${li").map(\.text), ["${lig2}", "${ligand}"])
        XCTAssertEqual(completer.completions(for: "status; do").map { $0.applied(to: "status; do") }, ["status; dock "])
        XCTAssertTrue(completer.completions(for: "dock # se").isEmpty)
    }

    @MainActor
    func testCompletesPaths() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("sub dir"), withIntermediateDirectories: true)
        try Data().write(to: directory.appendingPathComponent("ligand.sdf"))
        defer { try? FileManager.default.removeItem(at: directory) }
        let completer = completer(directory: directory)
        XCTAssertEqual(completer.completions(for: "load li").map(\.text), ["ligand.sdf "])
        XCTAssertEqual(completer.completions(for: "load su").map(\.text), ["\"sub dir/"])
    }

    @MainActor
    func testHelpTopicCompletesCommandsAndSubcommands() {
        XCTAssertEqual(completer().completions(for: "help d").map { $0.applied(to: "help d") }, ["help dock "])
        XCTAssertEqual(completer().completions(for: "help dock ").map(\.display), ["stop"])
    }

    @MainActor
    func testCommandBoundary() {
        XCTAssertTrue(completer().isAtCommandBoundary("dock "))
        XCTAssertFalse(completer().isAtCommandBoundary("dock"))
        XCTAssertFalse(completer().isAtCommandBoundary("nosuch "))
        XCTAssertFalse(completer().isAtCommandBoundary("load file.pdb "))
    }
}
