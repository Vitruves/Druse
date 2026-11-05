// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// How console text is decorated and laid out: plain and unwrapped for the in-app console, pipes
/// and files; ANSI colours and wrapping to the window width for a terminal. Help text and the
/// headless prompt, errors and warnings go through it, so captured output never contains escape
/// sequences.
struct ConsoleTextStyle: Sendable {
    /// True when ANSI escape sequences may be written.
    let usesColor: Bool
    /// Columns to wrap long descriptions to; nil leaves lines unwrapped (the view wraps them).
    let width: Int?

    static let plain = ConsoleTextStyle(usesColor: false, width: nil)

    /// Styling for output written to `descriptor`: colours on a terminal unless `NO_COLOR` is set
    /// (https://no-color.org) or `TERM=dumb`, and wrapping to the terminal's width.
    static func terminal(_ descriptor: Int32) -> ConsoleTextStyle {
        guard isatty(descriptor) != 0 else { return .plain }
        let environment = ProcessInfo.processInfo.environment
        let usesColor = environment["NO_COLOR"].map(\.isEmpty) ?? true && environment["TERM"] != "dumb"
        // One column short of the window, so a full line never triggers the terminal's own wrap.
        return ConsoleTextStyle(usesColor: usesColor, width: columns(of: descriptor) - 1)
    }

    /// Section titles.
    func heading(_ text: String) -> String { sgr("1", text) }
    /// Command names and literal words to type.
    func command(_ text: String) -> String { sgr("1;36", text) }
    /// Placeholders, defaults and secondary notes.
    func dim(_ text: String) -> String { sgr("2", text) }
    func error(_ text: String) -> String { sgr("1;31", text) }
    func warning(_ text: String) -> String { sgr("1;33", text) }
    func success(_ text: String) -> String { sgr("32", text) }

    private func sgr(_ code: String, _ text: String) -> String {
        usesColor && !text.isEmpty ? "\u{1B}[\(code)m\(text)\u{1B}[0m" : text
    }

    /// Lays out `rows` as two columns: the label (styled after padding, so escape sequences do not
    /// skew the alignment) and its description, wrapped to `width` (when set) with a hanging indent.
    /// `labelWidth` aligns several tables on one column; by default the longest label sets it.
    func columns(_ rows: [(label: String, text: String)], indent: Int = 2, labelWidth: Int? = nil,
                 label style: (String) -> String) -> [String] {
        let labelWidth = labelWidth ?? rows.map(\.label.count).max() ?? 0
        let textColumn = indent + labelWidth + 2
        let margin = String(repeating: " ", count: indent)
        let hanging = String(repeating: " ", count: textColumn)
        return rows.flatMap { row in
            let padding = String(repeating: " ", count: max(labelWidth - row.label.count, 0) + 2)
            let lines = width.map { Self.wrap(row.text, width: max($0 - textColumn, 24)) } ?? [row.text]
            return [margin + style(row.label) + padding + lines[0]] + lines.dropFirst().map { hanging + $0 }
        }
    }

    /// `text` indented by `indent` columns and wrapped to `width` (when set).
    func paragraph(_ text: String, indent: Int) -> [String] {
        let margin = String(repeating: " ", count: indent)
        let lines = width.map { Self.wrap(text, width: max($0 - indent, 24)) } ?? [text]
        return lines.map { margin + $0 }
    }

    /// Greedy word wrap of one paragraph, measuring words without their escape sequences.
    static func wrap(_ text: String, width: Int) -> [String] {
        var lines: [String] = []
        var current = ""
        var currentWidth = 0
        for word in text.split(separator: " ") {
            let wordWidth = visibleCount(word)
            if currentWidth > 0, currentWidth + 1 + wordWidth > width {
                lines.append(current)
                current = ""
                currentWidth = 0
            }
            current += currentWidth == 0 ? String(word) : " " + word
            currentWidth += (currentWidth == 0 ? 0 : 1) + wordWidth
        }
        return lines + [current]
    }

    /// Characters of `text` that take a column: everything except `ESC [ … m` sequences.
    private static func visibleCount(_ text: Substring) -> Int {
        var count = 0
        var inEscape = false
        for character in text {
            if inEscape {
                if character == "m" { inEscape = false }
            } else if character == "\u{1B}" {
                inEscape = true
            } else {
                count += 1
            }
        }
        return count
    }

    /// Columns of the terminal behind `descriptor` (80 when it does not report a size).
    static func columns(of descriptor: Int32) -> Int {
        var size = winsize()
        guard ioctl(descriptor, TIOCGWINSZ, &size) == 0, size.ws_col > 0 else { return 80 }
        return Int(size.ws_col)
    }
}
