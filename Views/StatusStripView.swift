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

/// The two surfaces of the bottom panel. The terminal shows only the DruseScript session; the
/// logs show only `ActivityLog` entries.
enum BottomPanel: String, CaseIterable, Sendable {
    case terminal
    case logs

    /// `@AppStorage` key of the surface last shown (the panel's visibility is `ui.consoleOpen`).
    static let storageKey = "ui.bottomPanel"

    var title: String {
        switch self {
        case .terminal: "Terminal"
        case .logs: "Logs"
        }
    }

    var systemImage: String {
        switch self {
        case .terminal: "terminal"
        case .logs: "list.bullet.rectangle"
        }
    }

    /// Menu shortcut, as shown in help tags (View ▸ Show Terminal / Show Logs).
    var shortcut: String {
        switch self {
        case .terminal: "⌃`"
        case .logs: "⇧⌘Y"
        }
    }
}

/// Bottom debug area of the main window: a 24 pt bar (Terminal / Logs tabs, next-step hint,
/// warning/error counters, terminal tools) above a resizable panel showing either the
/// DruseScript terminal or the activity log (whose filters get their own header row, so the bar
/// never runs out of width). Live status, progress and stop live in the toolbar's `ActivityView`.
struct StatusStripView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(ConsoleSession.self) private var session
    @Environment(\.openWindow) private var openWindow
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var showPanel: Bool
    @Binding var panel: BottomPanel
    @Binding var panelHeight: CGFloat
    /// Step shown in the navigator; the hint never points at the step the user is already on.
    let currentStep: SidebarTab
    private let log = ActivityLog.shared

    @State private var logState = LogPanelState()
    @State private var dragStartHeight: CGFloat?
    @State private var isHoveringResize = false
    /// True while the queue of externally opened scripts is being drained.
    @State private var isRunningPendingScripts = false

    // Error handling: the Logs panel opens by itself for one error only; after that
    // (until the user opens the panel again) new errors just pulse the badge.
    @State private var autoOpenArmed = true
    @State private var isAutoOpening = false
    @State private var errorPulse = 0

    private static let panelHeightRange: ClosedRange<CGFloat> = 80...500

    private var nextStep: SidebarTab? { currentStep.suggestedNextStep(in: viewModel) }

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            bar
            if showPanel, panel == .logs || !session.isDetached {
                Divider()
                if panel == .logs {
                    logToolbar
                    Divider()
                }
                Group {
                    switch panel {
                    case .terminal: ConsoleView(session: session)
                    case .logs: LogPanelView(state: logState, threshold: session.logLevel)
                    }
                }
                .frame(height: min(Self.panelHeightRange.upperBound, max(Self.panelHeightRange.lowerBound, panelHeight)))
            }
        }
        .onAppear { runPendingScripts() }
        .onChange(of: session.isDetached) { _, detached in
            // The terminal moved to its own window: the panel no longer has it to show.
            if detached, showPanel, panel == .terminal {
                withAnimation(DS.Motion.respecting(reduceMotion)) { showPanel = false }
            }
        }
        .onChange(of: viewModel.workspace.pendingScripts) { _, _ in runPendingScripts() }
        .onChange(of: log.entries.count) { _, _ in
            guard let last = log.entries.last, last.level == .error else { return }
            if !showPanel && autoOpenArmed {
                autoOpenArmed = false
                isAutoOpening = true
                show(.logs)
            } else if !showPanel || panel != .logs {
                errorPulse &+= 1
            }
        }
        .onChange(of: showPanel) { _, isOpen in
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
            HStack(spacing: DS.Space.xxs) {
                ForEach(BottomPanel.allCases, id: \.self) { tab in
                    tabButton(tab)
                }
            }
            .fixedSize()

            if session.isRunning {
                ProgressView().controlSize(.mini)
                    .help("A terminal command is running")
            }

            if let next = nextStep {
                // Full hint, then the step name alone, then nothing: never a truncated "Prep…".
                ViewThatFits(in: .horizontal) {
                    upNextButton(next, text: "Up next: \(next.rawValue) — \(next.subtitle)")
                    upNextButton(next, text: "Up next: \(next.rawValue)")
                    Color.clear.frame(width: 0, height: 0)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Spacer(minLength: DS.Space.m)
            }

            counters

            if showPanel, panel == .terminal, !session.isDetached {
                Divider().frame(height: 12)
                TerminalTools(session: session)
                toolButton("Move to Window", systemImage: "macwindow.on.rectangle",
                           help: "Show the terminal in its own window", id: "console.detach") {
                    openWindow(id: ConsoleWindow.id)
                }
            }
        }
        .padding(.horizontal, DS.Space.l)
        .frame(height: DS.Size.debugBar)
        .background(.bar)
        .overlay(alignment: .top) {
            if showPanel { resizeHandle }
        }
    }

    private func upNextButton(_ next: SidebarTab, text: String) -> some View {
        Button { next.request(in: viewModel) } label: {
            HStack(spacing: DS.Space.m) {
                Divider().frame(height: 12)
                Label(text, systemImage: "lightbulb")
                    .labelStyle(.titleAndIcon)
                    .lineLimit(1)
            }
            .font(DS.Font.secondary)
            .foregroundStyle(.secondary)
            .fixedSize()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Go to step \(next.shortcutNumber): \(next.rawValue) (⌘\(next.shortcutNumber))")
    }

    /// Terminal / Logs tab: selects that surface, or hides the panel when it is already shown.
    /// While the terminal has its own window, the Terminal tab brings that window forward.
    private func tabButton(_ tab: BottomPanel) -> some View {
        let inWindow = tab == .terminal && session.isDetached
        let isActive = showPanel && panel == tab && !inWindow
        return Button {
            if inWindow {
                openWindow(id: ConsoleWindow.id)
            } else if isActive {
                withAnimation(DS.Motion.respecting(reduceMotion)) { showPanel = false }
            } else {
                show(tab)
            }
        } label: {
            Label(tab.title, systemImage: inWindow ? "macwindow" : tab.systemImage)
                .labelStyle(.titleAndIcon)
                .lineLimit(1)
                .fixedSize()
                .font(DS.Font.secondary)
                .foregroundStyle(isActive ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .padding(.horizontal, DS.Space.s)
                .padding(.vertical, DS.Space.xxs)
                .background(
                    RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                        .fill(isActive ? DS.Color.controlFill : .clear)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(inWindow ? "Bring the Terminal window forward"
              : isActive ? "Hide \(tab.title) \(tab.shortcut)" : "Show \(tab.title) \(tab.shortcut)")
        .accessibilityAddTraits(isActive ? .isSelected : [])
        .accessibilityIdentifier(tab == .terminal ? AccessibilityID.statusToggleConsole : "status_toggleLogs")
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
            logState.levelFilter = level
            show(.logs)
        } label: {
            DSTag("\(count)", tone: tone, systemImage: level.icon)
        }
        .buttonStyle(.plain)
        .help("\(count) \(level.rawValue.lowercased())\(count == 1 ? "" : "s") — click to show \(count == 1 ? "it" : "them") in Logs")
        .accessibilityLabel("\(count) \(level.rawValue.lowercased())\(count == 1 ? "" : "s")")
        .accessibilityHint("Shows them in the logs")
    }

    // MARK: - Log tools

    /// Header row of the Logs surface: search, level and category filters, then the log actions.
    private var logToolbar: some View {
        HStack(spacing: DS.Space.m) {
            logTools
        }
        .padding(.horizontal, DS.Space.l)
        .frame(height: DS.Size.debugBar + DS.Space.s)
        .background(.bar)
    }

    @ViewBuilder
    private var logTools: some View {
        @Bindable var logState = logState

        TextField("Search logs", text: $logState.searchText)
            .textFieldStyle(.roundedBorder)
            .controlSize(.small)
            .frame(minWidth: 120, idealWidth: 220, maxWidth: 260)
            .help("Show only entries whose message or category contains this text")
            .accessibilityIdentifier("logs.search")

        Picker("Filter", selection: $logState.levelFilter) {
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
        .help("Filter log entries by level")

        Picker("Category", selection: $logState.categoryFilter) {
            Text("All Categories").tag(nil as LogCategory?)
            Divider()
            ForEach(LogCategory.allCases, id: \.self) { category in
                Text(category.rawValue).tag(category as LogCategory?)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .controlSize(.small)
        .fixedSize()
        .help("Filter log entries by category")

        Spacer(minLength: DS.Space.m)

        if !logState.selectedIDs.isEmpty {
            Button { copyToPasteboard(logState.copyText(of: visibleLogEntries)) } label: {
                Label("\(logState.selectedIDs.count)", systemImage: "doc.on.doc")
                    .monospacedDigit()
            }
            .buttonStyle(.borderless)
            .help("Copy selected entries ⌘C")
            .accessibilityIdentifier(AccessibilityID.statusCopySelected)
        }

        toolButton("Copy All", systemImage: "doc.on.clipboard", help: "Copy every entry shown",
                   id: "logs.copyAll") {
            logState.clearSelection()
            copyToPasteboard(logState.copyText(of: visibleLogEntries))
        }

        toolButton("Clear", systemImage: "trash", help: "Clear the activity log (the log file is kept)",
                   id: AccessibilityID.statusClearLog) {
            logState.clearSelection()
            log.clear()
        }

        if log.currentLogFileURL != nil {
            toolButton("Reveal Log", systemImage: "doc.text.magnifyingglass", help: "Reveal log file in Finder",
                       id: AccessibilityID.statusRevealLog) { log.revealLogInFinder() }
        }
    }

    private var visibleLogEntries: [LogEntry] {
        logState.visibleEntries(log.entries, threshold: session.logLevel)
    }

    private func toolButton(_ title: String, systemImage: String, help: String, id: String,
                            action: @escaping () -> Void) -> some View {
        BarToolButton(title: title, systemImage: systemImage, help: help, id: id, action: action)
    }

    /// Drag strip along the bar's top edge. It stays inside the bar (the Metal view above would
    /// swallow events) and measures in window coordinates, because the bar itself moves while the
    /// panel grows. A highlighted hairline shows it on hover and while dragging.
    private var resizeHandle: some View {
        Color.clear
            .frame(height: DS.Space.s)
            .contentShape(Rectangle())
            .overlay(alignment: .top) {
                Rectangle()
                    .fill(DS.Color.accent)
                    .frame(height: 2)
                    .opacity(isHoveringResize || dragStartHeight != nil ? 0.8 : 0)
                    .animation(DS.Motion.snappy, value: isHoveringResize)
            }
            .pointerStyle(.rowResize)
            .onHover { isHoveringResize = $0 }
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { value in
                        let initial = dragStartHeight ?? panelHeight
                        dragStartHeight = initial
                        let range = Self.panelHeightRange
                        panelHeight = min(range.upperBound, max(range.lowerBound, initial - value.translation.height))
                    }
                    .onEnded { _ in dragStartHeight = nil }
            )
            .help("Drag to resize the panel")
            .accessibilityHidden(true)
    }

    // MARK: - Actions

    /// Runs the scripts opened from outside the app one after another in the terminal.
    private func runPendingScripts() {
        guard !isRunningPendingScripts, !viewModel.workspace.pendingScripts.isEmpty else { return }
        isRunningPendingScripts = true
        if session.isDetached {
            openWindow(id: ConsoleWindow.id)
        } else {
            show(.terminal)
        }
        Task {
            while !viewModel.workspace.pendingScripts.isEmpty {
                let url = viewModel.workspace.pendingScripts.removeFirst()
                await session.runScript(at: url)
            }
            isRunningPendingScripts = false
        }
    }

    private func show(_ surface: BottomPanel) {
        panel = surface
        guard !showPanel else { return }
        withAnimation(DS.Motion.respecting(reduceMotion)) { showPanel = true }
    }
}

func copyToPasteboard(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}

/// Icon-only borderless button of the debug bar and the Terminal window's header.
struct BarToolButton: View {
    let title: String
    let systemImage: String
    let help: String
    let id: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.borderless)
        .help(help)
        .accessibilityIdentifier(id)
    }
}

/// Run Script, Stop (while a command runs), Copy All and Clear: the terminal's tools, shown in
/// the debug bar or in the Terminal window's header.
struct TerminalTools: View {
    let session: ConsoleSession

    var body: some View {
        BarToolButton(title: "Run Script…", systemImage: "play.rectangle", help: "Run a DruseScript file (.druse)",
                      id: "console.runScript", action: session.chooseAndRunScript)
            .disabled(session.isRunning)

        if session.isRunning {
            BarToolButton(title: "Stop", systemImage: "stop.fill", help: "Cancel the running command ⌃C",
                          id: "console.stop", action: session.interrupt)
                .foregroundStyle(DS.Color.danger)
        }

        BarToolButton(title: "Copy All", systemImage: "doc.on.clipboard", help: "Copy the whole terminal scrollback",
                      id: AccessibilityID.statusCopyAll) { copyToPasteboard(session.plainTranscript) }

        BarToolButton(title: "Clear", systemImage: "clear", help: "Clear the terminal scrollback ⌘K",
                      id: "console.clear", action: session.clearScrollback)
    }
}
