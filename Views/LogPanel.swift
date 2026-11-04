// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Filters and selection of the Logs surface. Owned by the bottom panel for the window's
/// lifetime, so switching to the terminal and back keeps them.
@Observable
@MainActor
final class LogPanelState {
    /// Exact level picked in the bar (nil = every level that passes the `log level` threshold).
    var levelFilter: LogLevel?
    var categoryFilter: LogCategory?
    /// Case-insensitive text the message or category must contain.
    var searchText = ""
    var selectedIDs: Set<UUID> = []
    private var lastClickedID: UUID?

    /// The entries the panel shows: at or above `threshold` (set by the terminal's `log level`),
    /// then narrowed by the level, category and search filters.
    func visibleEntries(_ entries: [LogEntry], threshold: LogVerbosity) -> [LogEntry] {
        let allowed = threshold.levels
        let query = searchText.trimmingCharacters(in: .whitespaces)
        return entries.filter { entry in
            allowed.contains(entry.level)
                && (levelFilter.map { entry.level == $0 } ?? true)
                && (categoryFilter.map { entry.category == $0 } ?? true)
                && (query.isEmpty || entry.message.localizedCaseInsensitiveContains(query)
                    || entry.category.rawValue.localizedCaseInsensitiveContains(query))
        }
    }

    /// Click selects one entry; shift-click extends from the last clicked one; clicking the only
    /// selected entry deselects it.
    func handleClick(on entry: LogEntry, in entries: [LogEntry], extending: Bool) {
        if extending, let anchorID = lastClickedID,
           let anchor = entries.firstIndex(where: { $0.id == anchorID }),
           let clicked = entries.firstIndex(where: { $0.id == entry.id }) {
            selectedIDs = Set(entries[min(anchor, clicked)...max(anchor, clicked)].map(\.id))
        } else if selectedIDs == [entry.id] {
            clearSelection()
        } else {
            selectedIDs = [entry.id]
            lastClickedID = entry.id
        }
    }

    func clearSelection() {
        selectedIDs.removeAll()
        lastClickedID = nil
    }

    /// True when the level, category or search filter hides some entries.
    var hasFilters: Bool {
        levelFilter != nil || categoryFilter != nil || !searchText.trimmingCharacters(in: .whitespaces).isEmpty
    }

    func clearFilters() {
        levelFilter = nil
        categoryFilter = nil
        searchText = ""
    }

    /// The selected entries among `entries`, or all of them when nothing is selected, one per
    /// line in the log-file format.
    func copyText(of entries: [LogEntry]) -> String {
        let copied = selectedIDs.isEmpty ? entries : entries.filter { selectedIDs.contains($0.id) }
        return copied.map(Self.plainText).joined(separator: "\n")
    }

    private static func plainText(_ entry: LogEntry) -> String {
        "\(entry.formattedTime)  [\(entry.level.rawValue)]  [\(entry.category.rawValue)]  \(entry.message)"
    }
}

/// The Logs surface of the bottom panel: the window's `ActivityLog` as a filterable, searchable
/// list (time, level, category, message). Filters live in the panel bar; this view shows the
/// result, handles selection and ⌘C, and follows new entries.
struct LogPanelView: View {
    let state: LogPanelState
    /// Minimum verbosity from the terminal's `log level` command.
    let threshold: LogVerbosity
    private let log = ActivityLog.shared

    var body: some View {
        let entries = state.visibleEntries(log.entries, threshold: threshold)
        VStack(spacing: 0) {
            if state.hasFilters && !entries.isEmpty {
                filterSummary(shown: entries.count)
                Divider()
            }
            if entries.isEmpty {
                emptyState
            } else {
                list(entries)
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .onCopyCommand { [NSItemProvider(object: state.copyText(of: entries) as NSString)] }
        .accessibilityIdentifier("logs.list")
    }

    /// Says how much the filters hide, with a one-click way back to every entry.
    private func filterSummary(shown: Int) -> some View {
        HStack(spacing: DS.Space.s) {
            Image(systemName: "line.3.horizontal.decrease.circle.fill")
                .foregroundStyle(DS.Color.accent)
            Text("Showing \(shown) of \(log.entries.count) entries")
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Spacer(minLength: DS.Space.s)
            Button("Show All", action: state.clearFilters)
                .buttonStyle(.link)
                .help("Clear the level, category and search filters")
        }
        .font(DS.Font.secondary)
        .padding(.horizontal, DS.Space.l)
        .frame(height: DS.Size.rowHeight)
    }

    private var emptyState: some View {
        VStack(spacing: DS.Space.s) {
            Image(systemName: log.entries.isEmpty ? "text.alignleft" : "line.3.horizontal.decrease.circle")
                .font(.system(size: 20, weight: .light))
                .foregroundStyle(.tertiary)
            Text(log.entries.isEmpty ? "No log entries yet" : "No log entries match the filters")
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
            if !log.entries.isEmpty && state.hasFilters {
                Button("Show All Entries", action: state.clearFilters)
                    .controlSize(.small)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func list(_ entries: [LogEntry]) -> some View {
        let selected = state.selectedIDs
        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(entries) { entry in
                        LogRow(entry: entry, isSelected: selected.contains(entry.id))
                            .onTapGesture {
                                state.handleClick(on: entry, in: entries, extending: NSEvent.modifierFlags.contains(.shift))
                            }
                            .id(entry.id)
                    }
                }
                .padding(.horizontal, DS.Space.m)
                .padding(.vertical, DS.Space.s)
            }
            .onChange(of: entries.last?.id) { _, _ in scrollToLatest(proxy, entries) }
            .onAppear { scrollToLatest(proxy, entries) }
            .onKeyPress(.escape) {
                state.clearSelection()
                return .handled
            }
        }
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy, _ entries: [LogEntry]) {
        guard let last = entries.last else { return }
        DispatchQueue.main.async { proxy.scrollTo(last.id, anchor: .bottom) }
    }
}

/// One log entry: time (seconds; milliseconds in the tooltip and copies), level icon, category
/// and the selectable message.
private struct LogRow: View {
    let entry: LogEntry
    let isSelected: Bool

    private static let timeColumn: CGFloat = 56
    private static let iconColumn: CGFloat = 14
    private static let categoryColumn: CGFloat = 68

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.m) {
            Text(Self.timeFormatter.string(from: entry.timestamp))
                .foregroundStyle(.tertiary)
                .frame(width: Self.timeColumn, alignment: .leading)
                .help(entry.formattedTime)
            Image(systemName: entry.level.icon)
                .foregroundStyle(entry.level.dsColor)
                .frame(width: Self.iconColumn)
                .accessibilityLabel(entry.level.rawValue)
            Text(entry.category.rawValue)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)
                .frame(width: Self.categoryColumn, alignment: .leading)
                .lineLimit(1)
            Text(entry.message)
                .foregroundStyle(entry.level == .error ? AnyShapeStyle(DS.Color.danger) : AnyShapeStyle(.primary))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .font(DS.Font.monoSmall)
        .padding(.vertical, DS.Space.xxs + 1)
        .padding(.horizontal, DS.Space.xs)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                .fill(isSelected ? DS.Color.accentSoft : .clear)
        )
        .contentShape(Rectangle())
    }
}
