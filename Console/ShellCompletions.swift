// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - Shell completion scripts
//
// `druse --completions fish|zsh|bash` prints a completion script for the `druse` terminal command.
// The scripts are generated from the interpreter's command registry, so every command, subcommand,
// option and option value that `help` documents is also offered by Tab in the shell, together with
// the CLI flags (`HeadlessCLI.flags`) and `.druse` script files.
//
// All three scripts share one completion model, flattened into shell `case`/`switch` tables:
// - a *context* is a command word plus the subcommand words typed after it ("ligand", "ligand db");
//   each context lists the subcommand words that may follow and, when it is a complete form,
//   that form's `key=` options and the kind of its first positional argument;
// - an *option value* table maps "context|option" to its value words (`.choice`, on/off) or to
//   file completion (`.path`).
// At Tab time the shell function skips CLI flags, extends the context with subcommand words for as
// long as they match, then offers values (after `key=`), subcommands, options and arguments.

/// The shells `druse --completions` can generate a script for.
enum CompletionShell: String, CaseIterable, Sendable {
    case fish, zsh, bash

    var displayName: String {
        switch self {
        case .fish: "fish"
        case .zsh: "zsh"
        case .bash: "bash"
        }
    }
}

/// Generates shell completion scripts for `druse` from a command registry.
struct ShellCompletions {

    /// What Tab offers for a positional argument or an option value.
    private enum Values: Equatable {
        case words([String])
        case paths
    }

    /// A completion candidate and its description.
    private struct Item {
        let word: String
        let summary: String
    }

    /// One completion context (see the file header).
    private struct Context {
        let key: String
        var subcommands: [Item] = []
        var options: [Item] = []
        var argument: Values?
    }

    private let verbs: [Item]
    private let contexts: [Context]
    /// Keyed "context|option".
    private let optionValues: [(key: String, values: Values)]

    /// Builds the completion model from the registry (`DruseInterpreter.commands`).
    init(commands: [ConsoleCommand]) {
        let verbs = commands.map { Item(word: $0.name, summary: $0.summary) }

        var contexts: [String: Context] = [:]
        var order: [String] = []
        var optionValues: [(String, Values)] = []

        func context(_ key: String) -> Context {
            if contexts[key] == nil {
                contexts[key] = Context(key: key)
                order.append(key)
            }
            return contexts[key]!
        }

        for command in commands {
            for form in command.forms {
                for depth in 0...form.path.count {
                    let key = ([command.name] + form.path.prefix(depth)).joined(separator: " ")
                    var entry = context(key)
                    if depth < form.path.count {
                        let word = form.path[depth]
                        if !entry.subcommands.contains(where: { $0.word == word }) {
                            let summary = depth + 1 == form.path.count ? form.summary : "\(command.name) \(word) …"
                            entry.subcommands.append(Item(word: word, summary: summary))
                        }
                    } else {
                        for option in form.options where !entry.options.contains(where: { $0.word == option.name + "=" }) {
                            entry.options.append(Item(word: option.name + "=", summary: option.summary))
                            if let values = Self.values(for: option.type) {
                                optionValues.append(("\(key)|\(option.name)", values))
                            }
                        }
                        if entry.argument == nil, let first = form.arguments.first {
                            entry.argument = Self.values(for: first.type)
                        }
                    }
                    contexts[key] = entry
                }
            }
        }
        self.verbs = verbs
        self.contexts = order.compactMap { contexts[$0] }
        self.optionValues = optionValues
    }

    /// Builds the model from a fresh interpreter's registry (the same declarations `help` reads).
    @MainActor
    init(viewModel: AppViewModel) {
        let sink = DiscardingConsoleOutput()
        self.init(commands: DruseInterpreter(viewModel: viewModel, output: sink, mode: .headless).commands)
    }

    /// The values a shell can offer for a parameter type; nil for free-form numbers and text.
    private static func values(for type: ConsoleValueType) -> Values? {
        switch type {
        case .path: .paths
        case .bool, .choice: .words(type.completions)
        case .string, .int, .float, .vec3: nil
        }
    }

    /// The completion script for `shell`, ready to be written to the shell's completion directory.
    func script(for shell: CompletionShell) -> String {
        switch shell {
        case .fish: fishScript()
        case .zsh: zshScript()
        case .bash: bashScript()
        }
    }

    /// Text every generated script carries in its header; `CommandLineInstaller` only replaces or
    /// removes completion files that contain it, so a hand-written script is never touched.
    static let signature = "# druse shell completions, generated by Druse"

    private var header: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
        return Self.signature + " \(version) from its command registry.\n"
            + "# Regenerate after updating Druse: Settings ▸ Shell Completions, or 'druse --completions <shell>'."
    }

    // MARK: - fish

    /// The fish completion script.
    private func fishScript() -> String {
        func q(_ text: String) -> String { Self.fishQuote(text) }
        func itemLines(_ items: [Item]) -> [String] {
            items.map { "            printf '%s\\t%s\\n' \(q($0.word)) \(q(Self.oneLine($0.summary)))" }
        }
        var lines = [header, "# Install: druse --completions fish > ~/.config/fish/completions/druse.fish", ""]

        lines.append("function __druse_subcommands --description 'Subcommand words after a druse context'")
        lines.append("    switch $argv[1]")
        lines.append("        case ''")
        lines += itemLines(verbs)
        for context in contexts where !context.subcommands.isEmpty {
            lines.append("        case \(q(context.key))")
            lines += itemLines(context.subcommands)
        }
        lines += ["    end", "end", ""]

        lines.append("function __druse_options --description 'key= options of a druse command form'")
        lines.append("    switch $argv[1]")
        for context in contexts where !context.options.isEmpty {
            lines.append("        case \(q(context.key))")
            lines += itemLines(context.options)
        }
        lines += ["    end", "end", ""]

        lines.append("function __druse_is_context --description 'True when the words name a druse command context'")
        lines.append("    switch $argv[1]")
        lines.append("        case " + contexts.map { q($0.key) }.joined(separator: " "))
        lines.append("            return 0")
        lines += ["    end", "    return 1", "end", ""]

        lines.append("function __druse_values --description 'Values of an argument or option; @path means files'")
        lines.append("    switch $argv[1]")
        for (key, values) in valueTable {
            lines.append("        case \(q(key))")
            lines.append("            printf '%s\\n' " + Self.valueWords(values).map(q).joined(separator: " "))
        }
        lines += ["    end", "end", ""]

        lines.append(contentsOf: """
        function __druse_complete --description 'Complete a druse command line'
            set -l tokens (commandline -opc)
            set -e tokens[1]
            set -l cur (commandline -ct)
            set -l prev ''
            test (count $tokens) -gt 0; and set prev $tokens[-1]
            switch $prev
        \(fishFlagValueCases())
            end
            if string match -q -- '-*' $cur
        \(HeadlessCLI.flags.flatMap { flag in flag.names.map { "        printf '%s\\t%s\\n' \(q($0)) \(q(flag.summary))" } }.joined(separator: "\n"))
                return
            end
            set -l words
            set -l skip 0
            for token in $tokens
                if test $skip -eq 1
                    set skip 0
                    continue
                end
                switch $token
                    case \(HeadlessCLI.flags.filter { $0.valueName != nil }.flatMap(\.names).joined(separator: " "))
                        set skip 1
                    case '-*'
                    case '*'
                        set -a words $token
                end
            end
            if test (count $words) -eq 0
                __druse_subcommands ''
                __fish_complete_path $cur | string match -r -- '^[^\\t]*(?:\\.druse|/)(?:\\t.*)?$'
                return
            end
            set -l ctx (string lower -- $words[1])
            string match -q -- '*.druse' $ctx; and return
            set -l positional 0
            for word in $words[2..-1]
                string match -q -- '*=*' $word; and continue
                set -l lowered (string lower -- $word)
                if test $positional -eq 0; and __druse_is_context "$ctx $lowered"
                    set ctx "$ctx $lowered"
                else
                    set positional 1
                end
            end
            if string match -q -- '*=*' $cur
                set -l key (string split -m 1 = -- $cur)[1]
                set -l typed (string split -m 1 = -- $cur)[2]
                set -l values (__druse_values "$ctx|$key")
                if test "$values" = '@path'
                    for path in (__fish_complete_path $typed)
                        echo "$key=$path"
                    end
                else
                    for value in $values
                        echo "$key=$value"
                    end
                end
                return
            end
            test $positional -eq 0; and __druse_subcommands $ctx
            __druse_options $ctx
            set -l argument (__druse_values "$ctx|")
            if test "$argument" = '@path'
                __fish_complete_path $cur
            else if test -n "$argument"
                printf '%s\\n' $argument
            end
        end

        complete -c druse -e
        complete -c druse -f -a '(__druse_complete)'

        """.components(separatedBy: "\n"))
        return lines.joined(separator: "\n")
    }

    /// fish completions for the values of the headless flags.
    private func fishFlagValueCases() -> String {
        var cases: [String] = []
        for flag in HeadlessCLI.flags where flag.valueName != nil {
            let names = flag.names.joined(separator: " ")
            if flag.valueChoices.isEmpty {
                cases.append("        case \(names)\n            return")
            } else {
                let choices = flag.valueChoices.map(Self.fishQuote).joined(separator: " ")
                cases.append("        case \(names)\n            printf '%s\\n' \(choices)\n            return")
            }
        }
        return cases.joined(separator: "\n")
    }

    // MARK: - zsh

    /// The zsh completion function (`_druse`).
    private func zshScript() -> String {
        func q(_ text: String) -> String { Self.posixQuote(text) }
        func described(_ items: [Item]) -> String {
            items.map { q($0.word.replacingOccurrences(of: ":", with: "\\:") + ":" + Self.oneLine($0.summary)) }
                .joined(separator: " ")
        }
        var lines = ["#compdef druse", header,
                     "# Install: druse --completions zsh > ~/.zfunc/_druse, with 'fpath=(~/.zfunc $fpath)' before compinit in ~/.zshrc.",
                     ""]

        lines.append("_druse_subcommands() {")
        lines.append("    case $1 in")
        lines.append("        ('') reply=(\(described(verbs))) ;;")
        for context in contexts where !context.subcommands.isEmpty {
            lines.append("        (\(q(context.key))) reply=(\(described(context.subcommands))) ;;")
        }
        lines += ["        (*) reply=() ;;", "    esac", "}", ""]

        lines.append("_druse_options() {")
        lines.append("    case $1 in")
        for context in contexts where !context.options.isEmpty {
            lines.append("        (\(q(context.key))) reply=(\(described(context.options))) ;;")
        }
        lines += ["        (*) reply=() ;;", "    esac", "}", ""]
        lines += posixContextAndValueFunctions(quote: q, arm: { "(\($0))" })

        let valueFlags = HeadlessCLI.flags.filter { $0.valueName != nil }.flatMap(\.names).joined(separator: "|")
        lines.append(contentsOf: """
        _druse() {
            local cur=$words[CURRENT] prev=$words[CURRENT-1]
            local -a reply tokens args flags
            case $prev in
        \(posixFlagValueCases(quote: q, indent: "        ", compadd: "compadd -- "))
            esac
            if [[ $cur == -* ]]; then
                flags=(\(HeadlessCLI.flags.flatMap { flag in flag.names.map { q($0 + ":" + flag.summary) } }.joined(separator: " ")))
                _describe -t flags 'druse option' flags
                return
            fi
            tokens=("${(@)words[2,CURRENT-1]}")
            local skip=0 token
            for token in $tokens; do
                if (( skip )); then skip=0; continue; fi
                case $token in
                    (\(valueFlags)) skip=1 ;;
                    (-*) ;;
                    (*) args+=("$token") ;;
                esac
            done
            if (( ${#args} == 0 )); then
                _druse_subcommands ''
                _describe -t commands 'druse command' reply
                _files -g '*.druse'
                return
            fi
            local ctx=${(L)args[1]}
            [[ $ctx == *.druse ]] && return
            local positional=0 word
            for word in "${(@)args[2,-1]}"; do
                [[ $word == *=* ]] && continue
                if (( ! positional )) && _druse_is_context "$ctx ${(L)word}"; then
                    ctx="$ctx ${(L)word}"
                else
                    positional=1
                fi
            done
            if [[ $cur == *=* ]]; then
                local key=${cur%%=*}
                _druse_values "$ctx|$key"
                compset -P '*='
                if [[ $reply[1] == @path ]]; then
                    _files
                else
                    compadd -- $reply
                fi
                return
            fi
            if (( ! positional )); then
                _druse_subcommands "$ctx"
                (( ${#reply} )) && _describe -t subcommands 'subcommand' reply
            fi
            _druse_options "$ctx"
            (( ${#reply} )) && _describe -t options 'option' reply -S ''
            _druse_values "$ctx|"
            if [[ $reply[1] == @path ]]; then
                _files
            elif (( ${#reply} )); then
                compadd -- $reply
            fi
        }

        if [[ $funcstack[1] == _druse ]]; then
            _druse "$@"
        else
            compdef _druse druse
        fi

        """.components(separatedBy: "\n"))
        return lines.joined(separator: "\n")
    }

    // MARK: - bash

    /// The bash completion function, for bash-completion 2.
    private func bashScript() -> String {
        func q(_ text: String) -> String { Self.posixQuote(text) }
        func words(_ items: [Item]) -> String { q(items.map(\.word).joined(separator: " ")) }
        var lines = [header,
                     "# Install: druse --completions bash > ~/.local/share/bash-completion/completions/druse",
                     "# (loaded on demand by bash-completion 2), or source it from ~/.bash_profile. Works with bash 3.2.",
                     ""]

        lines.append("_druse_subcommands() {")
        lines.append("    case $1 in")
        lines.append("        '') reply=\(words(verbs)) ;;")
        for context in contexts where !context.subcommands.isEmpty {
            lines.append("        \(q(context.key))) reply=\(words(context.subcommands)) ;;")
        }
        lines += ["        *) reply='' ;;", "    esac", "}", ""]

        lines.append("_druse_options() {")
        lines.append("    case $1 in")
        for context in contexts where !context.options.isEmpty {
            lines.append("        \(q(context.key))) reply=\(words(context.options)) ;;")
        }
        lines += ["        *) reply='' ;;", "    esac", "}", ""]
        lines += posixContextAndValueFunctions(quote: q, arm: { "\($0))" }, bash: true)

        let flagWords = HeadlessCLI.flags.flatMap(\.names).joined(separator: " ")
        let valueFlags = HeadlessCLI.flags.filter { $0.valueName != nil }.flatMap(\.names).joined(separator: "|")
        lines.append(contentsOf: """
        # Adds candidates matching $cur to COMPREPLY. Readline splits words at '=' (COMP_WORDBREAKS),
        # so for 'key=value' candidates only the part after the last '=' typed is inserted.
        _druse_add() {
            local candidate strip=''
            if [[ $COMP_WORDBREAKS == *=* && $cur == *=* ]]; then strip=${cur%"${cur##*=}"}; fi
            for candidate in "$@"; do
                [[ $candidate == "$cur"* ]] || continue
                COMPREPLY+=("${candidate#"$strip"}")
                [[ $candidate == *= ]] && type compopt >/dev/null 2>&1 && compopt -o nospace
            done
        }

        _druse_files() {
            local typed=$1 prefix=$2 path
            type compopt >/dev/null 2>&1 && compopt -o filenames
            while IFS= read -r path; do
                cur=$prefix$typed _druse_add "$prefix$path"
            done < <(compgen -f -- "$typed")
        }

        _druse() {
            COMPREPLY=()
            local line=${COMP_LINE:0:$COMP_POINT} cur='' prev='' reply
            [[ $line == *[[:space:]] ]] || cur=${line##*[[:space:]]}
            local -a tokens args
            read -r -a tokens <<< "${line%"$cur"}"
            tokens=("${tokens[@]:1}")
            (( ${#tokens[@]} )) && prev=${tokens[${#tokens[@]}-1]}
            case $prev in
        \(posixFlagValueCases(quote: q, indent: "        ", compadd: "_druse_add "))
            esac
            if [[ $cur == -* ]]; then
                _druse_add \(flagWords)
                return
            fi
            local skip=0 token
            for token in "${tokens[@]}"; do
                if (( skip )); then skip=0; continue; fi
                case $token in
                    \(valueFlags)) skip=1 ;;
                    -*) ;;
                    *) args+=("$token") ;;
                esac
            done
            if (( ${#args[@]} == 0 )); then
                _druse_subcommands ''
                _druse_add $reply
                local file
                while IFS= read -r file; do _druse_add "$file"; done < <(compgen -f -X '!*.druse' -- "$cur"; compgen -d -- "$cur")
                type compopt >/dev/null 2>&1 && compopt -o filenames
                return
            fi
            local ctx word lowered positional=0
            ctx=$(printf '%s' "${args[0]}" | tr '[:upper:]' '[:lower:]')
            [[ $ctx == *.druse ]] && return
            for word in "${args[@]:1}"; do
                [[ $word == *=* ]] && continue
                lowered=$(printf '%s' "$word" | tr '[:upper:]' '[:lower:]')
                if (( ! positional )) && _druse_is_context "$ctx $lowered"; then
                    ctx="$ctx $lowered"
                else
                    positional=1
                fi
            done
            if [[ $cur == *=* ]]; then
                local key=${cur%%=*} value values=()
                _druse_values "$ctx|$key"
                if [[ $reply == @path ]]; then
                    _druse_files "${cur#*=}" "$key="
                else
                    for value in $reply; do values+=("$key=$value"); done
                    _druse_add "${values[@]}"
                fi
                return
            fi
            if (( ! positional )); then
                _druse_subcommands "$ctx"
                _druse_add $reply
            fi
            _druse_options "$ctx"
            _druse_add $reply
            _druse_values "$ctx|"
            if [[ $reply == @path ]]; then
                _druse_files "$cur" ''
            else
                _druse_add $reply
            fi
        }

        complete -F _druse druse

        """.components(separatedBy: "\n"))
        return lines.joined(separator: "\n")
    }

    // MARK: - Shared table emitters (zsh and bash)

    /// `_druse_is_context` and `_druse_values` for POSIX-style shells; `reply` receives the
    /// values (an array in zsh, a space-separated word list in bash).
    private func posixContextAndValueFunctions(quote q: (String) -> String,
                                               arm: (String) -> String,
                                               bash: Bool = false) -> [String] {
        var lines = ["_druse_is_context() {", "    case $1 in"]
        lines.append("        \(arm(contexts.map { q($0.key) }.joined(separator: "|"))) return 0 ;;")
        lines += ["    esac", "    return 1", "}", ""]
        lines.append("_druse_values() {")
        lines.append("    case $1 in")
        for (key, values) in valueTable {
            let words = Self.valueWords(values)
            let reply = bash ? q(words.joined(separator: " ")) : "(" + words.map(q).joined(separator: " ") + ")"
            lines.append("        \(arm(q(key))) reply=\(reply) ;;")
        }
        lines.append("        \(arm("*")) reply=\(bash ? "''" : "()") ;;")
        lines += ["    esac", "}", ""]
        return lines
    }

    /// zsh/bash `case` branches completing the values of the headless flags.
    private func posixFlagValueCases(quote q: (String) -> String, indent: String, compadd: String) -> String {
        HeadlessCLI.flags.compactMap { flag -> String? in
            guard flag.valueName != nil else { return nil }
            let pattern = flag.names.joined(separator: "|")
            let body = flag.valueChoices.isEmpty ? "return" : compadd + flag.valueChoices.map(q).joined(separator: " ") + "; return"
            return "\(indent)(\(pattern)) \(body) ;;"
        }
        .joined(separator: "\n")
    }

    /// Argument values ("context|") followed by option values ("context|option").
    private var valueTable: [(String, Values)] {
        contexts.compactMap { context in context.argument.map { ("\(context.key)|", $0) } } + optionValues
    }

    /// The literal words of a value set (empty for paths, which shells complete themselves).
    private static func valueWords(_ values: Values) -> [String] {
        switch values {
        case .words(let words): words
        case .paths: ["@path"]
        }
    }

    // MARK: - Quoting

    private static func oneLine(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ")
    }

    /// Single-quoted word for sh, bash and zsh.
    private static func posixQuote(_ text: String) -> String {
        "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Single-quoted word for fish (only `\` and `'` are special inside).
    private static func fishQuote(_ text: String) -> String {
        "'" + text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'") + "'"
    }
}

/// A console output that drops everything: used when an interpreter is built only to read its
/// command registry (completion scripts, Settings).
final class DiscardingConsoleOutput: ConsoleOutput {
    func emit(_ kind: ConsoleOutputKind, _ text: String) {}
}
