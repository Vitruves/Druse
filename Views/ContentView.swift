// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Main window: navigator pane | canvas over the debug bar | inspector pane, laid out as one
/// integrated window under a unified toolbar. Both side panes are part of the window (window
/// background, hairline dividers), resizable from their inner edge, and collapsible.
struct ContentView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(ConsoleSession.self) private var consoleSession
    @Environment(\.openWindow) private var openWindow
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @AppStorage("ui.navigatorOpen") private var showNavigator = true
    @AppStorage("ui.inspectorOpen") private var showInspector = false
    @AppStorage("ui.consoleOpen") private var showConsole = false
    @AppStorage(BottomPanel.storageKey) private var bottomPanel = BottomPanel.terminal
    @AppStorage("ui.navigatorWidth") private var navigatorWidth = Double(DS.Size.navigatorIdeal)
    @AppStorage("ui.inspectorWidth") private var inspectorWidth = Double(DS.Size.inspectorIdeal)
    @State private var consoleHeight: CGFloat = 130
    @State private var isDragTargeted = false
    @State private var pipelineTab: SidebarTab = .search
    @State private var windowWidth: CGFloat = DS.Size.windowDefault.width
    @State private var stepBarWidth: CGFloat = 0
    /// Set when the window got too narrow and the shell hid the inspector; widening the window
    /// again brings it back. A user toggle clears it.
    @State private var inspectorAutoCollapsed = false

    /// The inspector opens by itself on first load only when there is room for it.
    private static let inspectorAutoOpenMinWidth: CGFloat = 1400
    /// Below this width the inspector steps aside so the canvas keeps a comfortable size.
    private static let inspectorAutoCollapseWidth: CGFloat = 1300
    /// Side panes give way (down to their minimum widths) before the canvas gets narrower than this.
    private static let canvasMinWidth: CGFloat = 520

    var body: some View {
        let widths = paneWidths
        HStack(spacing: 0) {
            if showNavigator {
                NavigatorColumn(selection: pipelineTab)
                    .frame(width: widths.navigator)
                    .background(.background)
                    .overlay(alignment: .trailing) {
                        PaneResizeHandle(width: $navigatorWidth, range: DS.Size.navigatorMin...DS.Size.navigatorMax, edge: .trailing)
                    }
                    .transition(.move(edge: .leading))
                Divider()
            }

            VStack(spacing: 0) {
                CanvasStage(currentStep: pipelineTab)
                StatusStripView(showPanel: $showConsole, panel: $bottomPanel, panelHeight: $consoleHeight, currentStep: pipelineTab)
            }
            .frame(maxWidth: .infinity)
            .layoutPriority(1)

            if showInspector {
                Divider()
                InspectorPanel()
                    .frame(width: widths.inspector)
                    .background(.background)
                    .overlay(alignment: .leading) {
                        PaneResizeHandle(width: $inspectorWidth, range: DS.Size.inspectorMin...DS.Size.inspectorMax, edge: .leading)
                    }
                    .transition(.move(edge: .trailing))
            }
        }
        // Hairline under the toolbar so the panes read as sitting below it.
        .overlay(alignment: .top) { Divider() }
        .clipped()
        .animation(DS.Motion.respecting(reduceMotion), value: showNavigator)
        .animation(DS.Motion.respecting(reduceMotion), value: showInspector)
        .onGeometryChange(for: CGFloat.self, of: \.size.width) { _, new in
            // `windowWidth` starts at the default width, so a window that opens narrow also
            // counts as a crossing and starts without the inspector.
            autoCollapseInspector(from: windowWidth, to: new)
            windowWidth = new
        }
        // Showing the inspector by any means (toolbar, menu, auto-open) ends the auto-collapse.
        .onChange(of: showInspector) { _, isShown in
            if isShown { inspectorAutoCollapsed = false }
        }
        .toolbar { toolbarContent }
        .focusedSceneValue(\.selectedStep, $pipelineTab)
        .onChange(of: viewModel.workspace.requestedTabToken) { _, _ in
            // View-model–driven step switch (welcome cards, "Continue to …", menu ⌘1…⌘8,
            // "Optimize" on a pose). Always reveals the navigator.
            guard let target = viewModel.workspace.requestedTab else { return }
            withAnimation(DS.Motion.respecting(reduceMotion)) {
                pipelineTab = target
                showNavigator = true
            }
        }
        .onChange(of: viewModel.molecules.protein?.atomCount) { old, new in
            guard old == nil, new != nil, !showInspector,
                  (NSApp.keyWindow?.frame.width ?? 0) >= Self.inspectorAutoOpenMinWidth else { return }
            withAnimation(DS.Motion.respecting(reduceMotion)) { showInspector = true }
        }
        .onChange(of: viewModel.demoStep) { _, step in
            // Follow the guided demo through the pipeline steps.
            let target: SidebarTab? = switch step {
            case .fetching, .parsing, .overview, .ribbon: .search
            case .pocketScan, .pocketFound, .gridSetup: .bindingSite
            case .dockingStart, .dockingRun, .dockingConverge: .dock
            case .scoring, .bestPose, .interactions: .results
            case .complete, .idle: nil
            }
            guard let target else { return }
            withAnimation(DS.Motion.respecting(reduceMotion)) { pipelineTab = target }
        }
        // Renderer theme follows the effective appearance — the only place it is set.
        .onAppear { syncRendererTheme() }
        .onChange(of: colorScheme) { _, _ in syncRendererTheme() }
        .onChange(of: viewModel.renderer == nil) { _, _ in syncRendererTheme() }
        .sheet(isPresented: Binding(
            get: { viewModel.workspace.showingConstraintSheet },
            set: { viewModel.workspace.showingConstraintSheet = $0 }
        )) {
            if let ctx = viewModel.workspace.constraintSheetContext {
                ConstraintConfigSheet(context: ctx)
                    .environment(viewModel)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $isDragTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url, FileImportHandler.canHandle(url: url) else { return }
                Task { @MainActor in
                    viewModel.loadFromFile(url: url)
                }
            }
            return true
        }
        .overlay {
            if isDragTargeted { dropTargetOverlay }
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Toggle(isOn: $showNavigator) {
                Label("Navigator", systemImage: "sidebar.leading")
            }
            .help(showNavigator ? "Hide Navigator ⌃⌘S" : "Show Navigator ⌃⌘S")
            .accessibilityIdentifier(AccessibilityID.navigatorToggle)
        }

        ToolbarSpacer(.fixed, placement: .navigation)

        ToolbarItemGroup(placement: .navigation) {
            Button { viewModel.importFile() } label: {
                Label("Open File…", systemImage: "doc.badge.plus")
            }
            .help("Open a structure or ligand file (⌘O)")

            Button { viewModel.openProject() } label: {
                Label("Open Project…", systemImage: "folder")
            }
            .help("Open a saved Druse project (⇧⌘O)")
            .accessibilityIdentifier(AccessibilityID.statusOpenProject)

            Button { DruseApp.saveProject(viewModel) } label: {
                Label("Save Project", systemImage: "square.and.arrow.down")
            }
            .help("Save the current workspace as a Druse project (⌘S)")
            .accessibilityIdentifier(AccessibilityID.statusSaveProject)
        }

        // The one step switcher: centred and always visible, even with the navigator collapsed.
        // It draws its own capsule so it can pick a density that fits the space it is given.
        ToolbarItem(placement: .principal) {
            PipelineStepBar(selection: pipelineTab, availableWidth: toolbarLayout.stepBar, usedWidth: $stepBarWidth) { tab in
                withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) {
                    pipelineTab = tab
                    showNavigator = true
                }
            }
        }
        .sharedBackgroundVisibility(.hidden)

        ToolbarItem(placement: .primaryAction) {
            ActivityView(width: toolbarLayout.activity)
        }

        ToolbarSpacer(.fixed, placement: .primaryAction)

        ToolbarItemGroup(placement: .primaryAction) {
            if consoleSession.isDetached {
                Button { openWindow(id: ConsoleWindow.id) } label: {
                    Label("Terminal", systemImage: "macwindow")
                }
                .help("Bring the Terminal window forward")
            } else {
                Toggle(isOn: Binding(
                    get: { showConsole && bottomPanel == .terminal },
                    set: { newValue in
                        if newValue { bottomPanel = .terminal }
                        withAnimation(DS.Motion.respecting(reduceMotion)) { showConsole = newValue }
                    }
                )) {
                    Label("Terminal", systemImage: BottomPanel.terminal.systemImage)
                }
                .help(showConsole && bottomPanel == .terminal ? "Hide Terminal ⌃`" : "Show Terminal ⌃`")
            }

            Toggle(isOn: $showInspector) {
                Label("Inspector", systemImage: "sidebar.trailing")
            }
            .help(showInspector ? "Hide Inspector ⌥⌘0" : "Show Inspector ⌥⌘0")
            .accessibilityIdentifier(AccessibilityID.inspectorToggle)

            SettingsLink {
                Label("Settings", systemImage: "gearshape")
            }
            .help("Settings: CPU cores, terminal, command line (⌘,)")
        }
    }

    // MARK: - Layout

    /// Hides the inspector when the window shrinks past `inspectorAutoCollapseWidth` and restores
    /// it when the window grows past it again, if the shell was the one that hid it. Acts on
    /// crossings only, so an inspector the user opens in a narrow window stays open.
    private func autoCollapseInspector(from old: CGFloat, to new: CGFloat) {
        let threshold = Self.inspectorAutoCollapseWidth
        if old >= threshold, new < threshold, showInspector {
            withAnimation(DS.Motion.respecting(reduceMotion)) { showInspector = false }
            inspectorAutoCollapsed = true
        } else if old < threshold, new >= threshold, inspectorAutoCollapsed {
            withAnimation(DS.Motion.respecting(reduceMotion)) { showInspector = true }
        }
    }

    /// Toolbar budget. The step bar is centred in the window and has priority: it may use what is
    /// left between the two sides with the activity capsule at its minimum. The capsule then
    /// takes the trailing space the bar's chosen density leaves free.
    private var toolbarLayout: (stepBar: CGFloat, activity: CGFloat) {
        // Traffic lights + navigator toggle + file group; console/inspector/settings group plus the
        // glass padding around the capsule; minimum gap NSToolbar keeps between neighbouring items.
        let leading: CGFloat = 256, trailingChrome: CGFloat = 164, gap: CGFloat = 20
        let activityRange: ClosedRange<CGFloat> = 132...320
        let stepBar = max(windowWidth - 2 * (max(leading, trailingChrome + activityRange.lowerBound) + gap), 0)
        let free = (windowWidth - min(stepBarWidth, stepBar)) / 2 - trailingChrome - gap
        return (stepBar, min(max(free, activityRange.lowerBound), activityRange.upperBound))
    }

    /// Stored pane widths, clamped to their ranges and shrunk towards their minimums when the
    /// window is too narrow to keep `canvasMinWidth` for the canvas.
    private var paneWidths: (navigator: CGFloat, inspector: CGFloat) {
        var nav = min(max(CGFloat(navigatorWidth), DS.Size.navigatorMin), DS.Size.navigatorMax)
        var insp = min(max(CGFloat(inspectorWidth), DS.Size.inspectorMin), DS.Size.inspectorMax)
        let excess = (showNavigator ? nav : 0) + (showInspector ? insp : 0) + Self.canvasMinWidth - windowWidth
        if excess > 0 {
            let navSlack = showNavigator ? nav - DS.Size.navigatorMin : 0
            let inspSlack = showInspector ? insp - DS.Size.inspectorMin : 0
            let slack = navSlack + inspSlack
            if slack > 0 {
                let take = min(excess, slack) / slack
                nav -= navSlack * take
                insp -= inspSlack * take
            }
        }
        return (nav, insp)
    }

    // MARK: - Drop target

    private var dropTargetOverlay: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.floating, style: .continuous)
        return shape
            .strokeBorder(DS.Color.accent, style: StrokeStyle(lineWidth: 2, dash: [8, 5]))
            .background(shape.fill(DS.Color.accentSoft.opacity(0.5)))
            .padding(DS.Space.m)
            .overlay {
                Label("Drop to import", systemImage: "arrow.down.doc")
                    .font(DS.Font.sectionTitle)
                    .padding(.horizontal, DS.Space.xl)
                    .padding(.vertical, DS.Space.l)
                    .dsFloating()
            }
            .allowsHitTesting(false)
    }

    // MARK: - Renderer theme

    private func syncRendererTheme() {
        viewModel.renderer?.themeMode = colorScheme == .light ? 1 : 0
    }
}

// MARK: - Pane resize handle

/// Invisible drag strip on a side pane's inner edge that resizes the pane within `range` and
/// persists the width. It straddles the hairline divider but stays mostly inside the pane: the
/// canvas's Metal view would swallow the events on the other side. Also used by the
/// Results Database window.
struct PaneResizeHandle: View {
    @Binding var width: Double
    let range: ClosedRange<CGFloat>
    /// The pane edge the handle sits on: `.trailing` for the navigator, `.leading` for the inspector.
    let edge: HorizontalEdge
    @State private var dragStartWidth: Double?

    var body: some View {
        Color.clear
            .frame(width: 6)
            .contentShape(Rectangle())
            .offset(x: edge == .trailing ? 2 : -2)
            .pointerStyle(.frameResize(position: edge == .trailing ? .trailing : .leading))
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { value in
                        let start = dragStartWidth ?? width
                        dragStartWidth = start
                        let delta = edge == .trailing ? value.translation.width : -value.translation.width
                        width = Double(min(max(CGFloat(start) + delta, range.lowerBound), range.upperBound))
                    }
                    .onEnded { _ in dragStartWidth = nil }
            )
            .accessibilityHidden(true)
    }
}
