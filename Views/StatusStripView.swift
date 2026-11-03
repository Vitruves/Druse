// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

extension LogLevel {
    /// Design-system tone for log icons (readable in Light and Dark).
    var dsColor: Color {
        switch self {
        case .info:    .secondary
        case .success: DS.Color.success
        case .warning: DS.Color.warning
        case .error:   DS.Color.danger
        case .debug:   DS.Color.pocket
        }
    }
}

/// Bottom debug area of the main window: a 24 pt bar (console toggle, next-step hint,
/// warning/error counters, console tools) above a resizable log console.
/// Live status, progress and stop live in the toolbar's `ActivityView`.
struct StatusStripView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var showConsole: Bool
    @Binding var consoleHeight: CGFloat
    /// Step shown in the navigator; the hint never points at the step the user is already on.
    let currentStep: SidebarTab
    private let log = ActivityLog.shared

    @State private var filterLevel: LogLevel? = nil
    @GestureState private var dragStartHeight: CGFloat = 0

    // Console selection state
    @State private var selectedEntryIDs: Set<UUID> = []
    @State private var lastClickedID: UUID? = nil

    // Error handling: the console opens by itself for one error only; after that
    // (until the user opens the console again) new errors just pulse the badge.
    @State private var autoOpenArmed = true
    @State private var isAutoOpening = false
    @State private var errorPulse = 0

    private static let consoleHeightRange: ClosedRange<CGFloat> = 80...500

    private var filteredEntries: [LogEntry] {
        guard let level = filterLevel else { return log.entries }
        return log.entries.filter { $0.level == level }
    }

    private var nextStep: SidebarTab? { currentStep.suggestedNextStep(in: viewModel) }

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            bar
            if showConsole {
                Divider()
                console
            }
        }
        .onChange(of: log.entries.count) { _, _ in
            guard let last = log.entries.last, last.level == .error else { return }
            if !showConsole && autoOpenArmed {
                autoOpenArmed = false
                isAutoOpening = true
                withAnimation(DS.Motion.respecting(reduceMotion)) { showConsole = true }
            } else if !showConsole {
                errorPulse &+= 1
            }
        }
        .onChange(of: showConsole) { _, isOpen in
            // A user-driven open re-arms the one-shot auto-open.
            if isAutoOpening {
                isAutoOpening = false
            } else if isOpen {
                autoOpenArmed = true
            }
        }
    }

    // MARK: - Bar

    private var bar: some View {
        HStack(spacing: DS.Space.m) {
            Button(action: toggleConsole) {
                HStack(spacing: DS.Space.xs) {
                    Image(systemName: showConsole ? "chevron.down" : "chevron.up")
                        .font(DS.Font.caption.weight(.semibold))
                    Text("Console")
                        .font(DS.Font.secondary)
                }
                .foregroundStyle(showConsole ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .help(showConsole ? "Hide Console ⇧⌘Y" : "Show Console ⇧⌘Y")
            .accessibilityIdentifier(AccessibilityID.statusToggleConsole)

            if let next = nextStep {
                Divider().frame(height: 12)
                Button { next.request(in: viewModel) } label: {
                    Label {
                        Text("Up next: \(next.rawValue) — \(next.subtitle)").lineLimit(1)
                    } icon: {
                        Image(systemName: "lightbulb")
                    }
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .labelStyle(.titleAndIcon)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Go to step \(next.shortcutNumber): \(next.rawValue) (⌘\(next.shortcutNumber))")
            }

            Spacer(minLength: DS.Space.m)

            counters

            if showConsole {
                Divider().frame(height: 12)
                consoleTools
            }
        }
        .padding(.horizontal, DS.Space.l)
        .frame(height: DS.Size.debugBar)
        .background(.bar)
        .overlay(alignment: .top) {
            if showConsole { resizeHandle }
        }
    }

    @ViewBuilder
    private var counters: some View {
        let warnings = log.entries.lazy.filter { $0.level == .warning }.count
        let errors = log.entries.lazy.filter { $0.level == .error }.count
        if warnings > 0 {
            counterButton(count: warnings, level: .warning, tone: .warning)
        }
        if errors > 0 {
            counterButton(count: errors, level: .error, tone: .danger)
                .symbolEffect(.bounce, value: errorPulse)
        }
    }

    private func counterButton(count: Int, level: LogLevel, tone: DSTone) -> some View {
        Button {
            filterLevel = level
            if !showConsole { toggleConsole() }
        } label: {
            DSTag("\(count)", tone: tone, systemImage: level.icon)
        }
        .buttonStyle(.plain)
        .help("Show \(count) \(level.rawValue.lowercased()) message\(count == 1 ? "" : "s")")
    }

    @ViewBuilder
    private var consoleTools: some View {
        Picker("Filter", selection: $filterLevel) {
            Text("All").tag(nil as LogLevel?)
            ForEach(LogLevel.allCases, id: \.self) { level in
                Image(systemName: level.icon)
                    .accessibilityLabel(level.rawValue)
                    .tag(level as LogLevel?)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .controlSize(.small)
        .fixedSize()
        .help("Filter console by log level")

        if !selectedEntryIDs.isEmpty {
            Button(action: copySelected) {
                Label("\(selectedEntryIDs.count)", systemImage: "doc.on.doc")
                    .monospacedDigit()
            }
            .buttonStyle(.borderless)
            .help("Copy selected lines ⌘C")
            .accessibilityIdentifier(AccessibilityID.statusCopySelected)
        }

        toolButton("Copy All", systemImage: "doc.on.clipboard", help: "Copy all console output",
                   id: AccessibilityID.statusCopyAll, action: copyAll)

        toolButton("Clear", systemImage: "trash", help: "Clear all log entries",
                   id: AccessibilityID.statusClearLog) {
            log.clear()
            selectedEntryIDs.removeAll()
            lastClickedID = nil
        }

        if log.currentLogFileURL != nil {
            toolButton("Reveal Log", systemImage: "doc.text.magnifyingglass", help: "Reveal log file in Finder",
                       id: AccessibilityID.statusRevealLog) { log.revealLogInFinder() }
        }
    }

    private func toolButton(_ title: String, systemImage: String, help: String, id: String,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.borderless)
        .help(help)
        .accessibilityIdentifier(id)
    }

    private var resizeHandle: some View {
        Color.clear
            .frame(height: 5)
            .contentShape(Rectangle())
            .pointerStyle(.rowResize)
            .gesture(
                DragGesture(minimumDistance: 1)
                    .updating($dragStartHeight) { _, state, _ in
                        if state == 0 { state = consoleHeight }
                    }
                    .onChanged { value in
                        let initial = dragStartHeight != 0 ? dragStartHeight : consoleHeight
                        let range = Self.consoleHeightRange
                        consoleHeight = min(range.upperBound, max(range.lowerBound, initial - value.translation.height))
                    }
            )
            .offset(y: -2)
            .accessibilityHidden(true)
    }

    // MARK: - Console

    private var console: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(filteredEntries) { entry in
                        logEntryRow(entry)
                            .id(entry.id)
                    }
                }
                .padding(.horizontal, DS.Space.m)
                .padding(.vertical, DS.Space.xs)
            }
            .onChange(of: log.entries.count) { _, _ in
                if let last = filteredEntries.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
            .onChange(of: filterLevel) { _, _ in scrollToLatest(proxy) }
            .onAppear { scrollToLatest(proxy) }
            .onKeyPress(.escape) {
                selectedEntryIDs.removeAll()
                lastClickedID = nil
                return .handled
            }
        }
        .frame(height: min(Self.consoleHeightRange.upperBound, max(Self.consoleHeightRange.lowerBound, consoleHeight)))
        .background(Color(nsColor: .textBackgroundColor))
        .copyable(formatEntries(selectedEntryIDs.isEmpty ? filteredEntries : selectedEntries))
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy) {
        guard let last = filteredEntries.last else { return }
        DispatchQueue.main.async {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    @ViewBuilder
    private func logEntryRow(_ entry: LogEntry) -> some View {
        let isSelected = selectedEntryIDs.contains(entry.id)
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.m) {
            Text(entry.formattedTime)
                .foregroundStyle(.secondary)
                .frame(width: 84, alignment: .leading)
            Image(systemName: entry.level.icon)
                .foregroundStyle(entry.level.dsColor)
                .frame(width: 14)
            Text(entry.category.rawValue)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)
                .lineLimit(1)
            Text(entry.message)
                .foregroundStyle(.primary)
                .lineLimit(2)
                .textSelection(.enabled)
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
        .onTapGesture {
            handleEntryClick(entry, shiftKey: NSEvent.modifierFlags.contains(.shift))
        }
    }

    // MARK: - Actions

    private func toggleConsole() {
        withAnimation(DS.Motion.respecting(reduceMotion)) { showConsole.toggle() }
    }

    private var selectedEntries: [LogEntry] {
        filteredEntries.filter { selectedEntryIDs.contains($0.id) }
    }

    private func handleEntryClick(_ entry: LogEntry, shiftKey: Bool) {
        let entries = filteredEntries
        if shiftKey, let anchorID = lastClickedID,
           let anchorIdx = entries.firstIndex(where: { $0.id == anchorID }),
           let clickIdx = entries.firstIndex(where: { $0.id == entry.id }) {
            let range = min(anchorIdx, clickIdx)...max(anchorIdx, clickIdx)
            selectedEntryIDs = Set(entries[range].map(\.id))
        } else if selectedEntryIDs.contains(entry.id) && selectedEntryIDs.count == 1 {
            selectedEntryIDs.removeAll()
            lastClickedID = nil
        } else {
            selectedEntryIDs = [entry.id]
            lastClickedID = entry.id
        }
    }

    private func formatEntries(_ entries: [LogEntry]) -> [String] {
        let text = entries.map { entry in
            "\(entry.formattedTime)  [\(entry.level.rawValue)]  [\(entry.category.rawValue)]  \(entry.message)"
        }.joined(separator: "\n")
        return [text]
    }

    private func copySelected() {
        copyToPasteboard(formatEntries(selectedEntries).first ?? "")
    }

    private func copyAll() {
        copyToPasteboard(formatEntries(filteredEntries).first ?? "")
    }

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
