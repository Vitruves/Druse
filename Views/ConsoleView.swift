// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

extension ConsoleCompletion {
    /// SF Symbol for the completion list row.
    fileprivate var icon: String {
        switch kind {
        case .command:    "terminal"
        case .subcommand: "arrow.turn.down.right"
        case .option:     "slider.horizontal.3"
        case .value:      "list.bullet"
        case .variable:   "dollarsign"
        case .path:       display.hasSuffix("/") ? "folder" : "doc"
        }
    }
}

/// The Terminal surface of the bottom panel: the DruseScript terminal (scrollback with the
/// prompt inline at its end) and, while Tab completion offers several candidates, a compact
/// completion list floating next to the word being typed, kept inside the terminal. It shows
/// only the session — activity-log entries live in the Logs surface (`LogPanelView`).
struct ConsoleView: View {
    let session: ConsoleSession
    @State private var anchor = TerminalCaretAnchor()
    @State private var size: CGSize = .zero

    var body: some View {
        TerminalTextView(session: session, transcript: session.transcript,
                         clearCount: session.clearCount, live: TerminalLiveLine(session), anchor: anchor)
            .onGeometryChange(for: CGSize.self) { $0.size } action: { size = $0 }
            .overlay(alignment: .topLeading) {
                ConsoleCompletionList(session: session, anchor: anchor, bounds: size)
            }
            .clipped()
            .background(Color(nsColor: TerminalStyle.background))
    }
}

// MARK: - Window

/// The terminal in its own window: the same `ConsoleSession` as the bottom panel, under a
/// header with the terminal's tools and a button moving it back to the main window. While it
/// is open the session is marked detached, so the bottom panel never shows it at the same time;
/// closing the window hands the session back to the panel.
struct ConsoleWindow: View {
    static let id = "terminal"

    @Environment(ConsoleSession.self) private var session
    @Environment(\.dismissWindow) private var dismissWindow
    @AppStorage("ui.consoleOpen") private var showConsole = false
    @AppStorage(BottomPanel.storageKey) private var bottomPanel = BottomPanel.terminal

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: DS.Space.m) {
                if session.isRunning {
                    ProgressView().controlSize(.mini)
                        .help("A terminal command is running")
                }
                Spacer(minLength: DS.Space.m)
                TerminalTools(session: session)
                Divider().frame(height: 12)
                BarToolButton(title: "Move to Main Window", systemImage: "rectangle.bottomhalf.inset.filled",
                              help: "Show the terminal in the main window's bottom panel again",
                              id: "console.attach") {
                    bottomPanel = .terminal
                    showConsole = true
                    dismissWindow(id: Self.id)
                }
            }
            .padding(.horizontal, DS.Space.l)
            .frame(height: DS.Size.debugBar)
            .background(.bar)
            Divider()
            if session.isDetached {
                ConsoleView(session: session)
            }
        }
        .onAppear { session.isDetached = true }
        .onDisappear { session.isDetached = false }
    }
}

// MARK: - Completion list

/// Candidates for the word being typed: kind icon, text and the registry's one-line description
/// (dimmed). Placed under the caret's line when it fits, else above it, never wider than
/// `maxWidth` nor outside the terminal. Keyboard handling lives in the terminal (↑/↓ or ⇧Tab
/// move, Tab/Return accept, Esc closes); a click accepts a row without taking focus from it.
private struct ConsoleCompletionList: View {
    let session: ConsoleSession
    let anchor: TerminalCaretAnchor
    let bounds: CGSize

    private static let rowHeight: CGFloat = 22
    private static let maxRows = 8
    private static let maxWidth: CGFloat = 520
    private static let inset: CGFloat = DS.Space.xs
    private static let margin: CGFloat = DS.Space.s
    /// Icon column plus its spacing: rows are shifted left by it so the candidate text lines up
    /// with the word being completed.
    private static let leadOffset: CGFloat = DS.Space.s + 16 + DS.Space.s + inset
    private static let textFont = Font(TerminalStyle.font)

    var body: some View {
        let items = session.completionItems
        if !items.isEmpty, let line = anchor.wordLine, bounds.width > 0 {
            let frame = placement(for: items, line: line)
            list(items, rows: frame.rows)
                .frame(width: frame.rect.width, height: frame.rect.height)
                .background(
                    RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                        .fill(Color(nsColor: TerminalStyle.popup))
                        .shadow(color: .black.opacity(0.18), radius: 8, y: 3)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                        .strokeBorder(DS.Color.hairline, lineWidth: 0.5)
                )
                .clipShape(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous))
                .offset(x: frame.rect.minX, y: frame.rect.minY)
                .accessibilityElement(children: .contain)
                .accessibilityLabel("Completions")
                .accessibilityIdentifier("console.completions")
        }
    }

    private func list(_ items: [ConsoleCompletion], rows: Int) -> some View {
        let textColumn = Self.textColumnWidth(items)
        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(items.indices, id: \.self) { index in
                        row(items[index], textColumn: textColumn, isSelected: index == session.selectedCompletion)
                            .id(index)
                            .onTapGesture { session.acceptCompletion(at: index) }
                    }
                }
                .padding(Self.inset)
            }
            .scrollIndicators(items.count > rows ? .automatic : .never)
            .onChange(of: session.selectedCompletion) { _, index in proxy.scrollTo(index) }
        }
    }

    private func row(_ item: ConsoleCompletion, textColumn: CGFloat, isSelected: Bool) -> some View {
        HStack(spacing: DS.Space.s) {
            Image(systemName: item.icon)
                .font(DS.Font.caption)
                .foregroundStyle(isSelected ? DS.Color.accent : .secondary)
                .frame(width: 16)
            Text(item.display)
                .font(Self.textFont)
                .foregroundStyle(Color(nsColor: TerminalStyle.foreground))
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(width: textColumn, alignment: .leading)
            Text(item.summary)
                .font(DS.Font.secondary)
                .foregroundStyle(Color(nsColor: TerminalStyle.dim))
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, DS.Space.s)
        .frame(height: Self.rowHeight)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                .fill(isSelected ? DS.Color.accentSoft : .clear)
        )
        .contentShape(Rectangle())
        .help(item.summary)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// Width of the longest candidate in the terminal font, between 80 and 200 pt.
    private static func textColumnWidth(_ items: [ConsoleCompletion]) -> CGFloat {
        let widest = items.lazy
            .map { ceil(($0.display as NSString).size(withAttributes: [.font: TerminalStyle.font]).width) }
            .max() ?? 0
        return min(200, max(80, widest + DS.Space.xs))
    }

    /// Below the caret's line when all visible rows fit there (or there is more room below),
    /// else above it; clamped inside the terminal.
    private func placement(for items: [ConsoleCompletion], line: CGRect) -> (rect: CGRect, rows: Int) {
        let wanted = min(items.count, Self.maxRows)
        let chrome = 2 * Self.inset
        func rowsFitting(_ space: CGFloat) -> Int { max(0, Int((space - chrome) / Self.rowHeight)) }
        let below = rowsFitting(bounds.height - line.maxY - Self.margin)
        let above = rowsFitting(line.minY - Self.margin)
        let placeBelow = below >= wanted || below >= above
        let rows = max(1, min(wanted, placeBelow ? below : above))
        let height = CGFloat(rows) * Self.rowHeight + chrome
        let width = min(Self.maxWidth, bounds.width - 2 * Self.margin)
        let x = min(max(Self.margin, line.minX - Self.leadOffset), bounds.width - width - Self.margin)
        let y = placeBelow ? line.maxY + DS.Space.xxs : line.minY - DS.Space.xxs - height
        return (CGRect(x: x, y: max(0, y), width: width, height: height), rows)
    }
}
