// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Pipeline steps in processing order, plus utility tools (stepNumber == nil).
/// Declaration order defines `allCases`.
enum SidebarTab: String, CaseIterable {
    case search           = "Search"
    case preparation      = "Preparation"
    case ligands          = "Ligands"
    case bindingSite      = "Binding Site"
    case dock             = "Docking"
    case results          = "Results"
    case leadOptimization = "Lead Opt"
    case sequence         = "Sequence"      // utility, not a numbered step

    var icon: String {
        switch self {
        case .search:           "magnifyingglass"
        case .preparation:      "wrench.and.screwdriver"
        case .ligands:          "hexagon"
        case .bindingSite:      "scope"
        case .dock:             "arrow.down.right.and.arrow.up.left"
        case .results:          "chart.bar.doc.horizontal"
        case .leadOptimization: "arrow.triangle.branch"
        case .sequence:         "textformat.abc"
        }
    }

    /// Label used in the toolbar step bar.
    var shortTitle: String {
        switch self {
        case .search:           "Search"
        case .preparation:      "Prepare"
        case .ligands:          "Ligands"
        case .bindingSite:      "Site"
        case .dock:             "Dock"
        case .results:          "Results"
        case .leadOptimization: "Lead Opt"
        case .sequence:         "Sequence"
        }
    }

    var subtitle: String {
        switch self {
        case .search:           "Load or fetch a protein"
        case .preparation:      "Clean, repair & protonate"
        case .ligands:          "Build the compound set"
        case .bindingSite:      "Find the pocket & grid box"
        case .dock:             "Configure & run"
        case .results:          "Analyze & export poses"
        case .leadOptimization: "Generate & rank analogs"
        case .sequence:         "Inspect chains & residues"
        }
    }

    var stepNumber: Int? {
        switch self {
        case .search:           1
        case .preparation:      2
        case .ligands:          3
        case .bindingSite:      4
        case .dock:             5
        case .results:          6
        case .leadOptimization: 7
        case .sequence:         nil
        }
    }

    var isPipelineStep: Bool { stepNumber != nil }

    var nextStep: SidebarTab? {
        switch self {
        case .search:           .preparation
        case .preparation:      .ligands
        case .ligands:          .bindingSite
        case .bindingSite:      .dock
        case .dock:             .results
        case .results:          .leadOptimization
        case .leadOptimization, .sequence: nil
        }
    }

    /// The step to suggest after this one, shared by the step footers' "Continue to …" links and
    /// the status strip's "Up next" hint so they always agree. Input steps that are already
    /// satisfied (a ligand is loaded, a site is defined) are skipped; Docking, Results and Lead Opt
    /// are never skipped, since "done" there only means output exists, not that it was reviewed.
    @MainActor
    func suggestedNextStep(in viewModel: AppViewModel) -> SidebarTab? {
        var candidate = nextStep
        while let step = candidate, step.isSkippableWhenDone, step.status(in: viewModel) == .done {
            candidate = step.nextStep
        }
        return candidate
    }

    private var isSkippableWhenDone: Bool {
        switch self {
        case .search, .preparation, .ligands, .bindingSite: true
        case .dock, .results, .leadOptimization, .sequence: false
        }
    }

    var previousStep: SidebarTab? {
        SidebarTab.allCases.first { $0.nextStep == self }
    }

    /// Menu / shortcut position: steps 1…7, then Sequence as 8.
    var shortcutNumber: Int { stepNumber ?? SidebarTab.allCases.count }

    /// Asks the shell to show this step (handled by `ContentView`, which also reveals the navigator).
    @MainActor
    func request(in viewModel: AppViewModel) {
        viewModel.workspace.requestedTab = self
        viewModel.workspace.requestedTabToken &+= 1
    }
}

// MARK: - Step status & summary

extension SidebarTab {
    /// Progress of this step for the toolbar step-bar badges and the debug-bar hint. `nil` for tools.
    @MainActor
    func status(in viewModel: AppViewModel) -> DSStepStatus? {
        let molecules = viewModel.molecules
        let docking = viewModel.docking
        let hasProtein = molecules.protein != nil
        let hasDockingOutput = !docking.dockingResults.isEmpty || !docking.batchResults.isEmpty
            || !docking.screeningHits.isEmpty

        switch self {
        case .search:
            return hasProtein ? .done : .ready
        case .preparation:
            if molecules.proteinPrepared { return .done }
            guard hasProtein else { return .blocked }
            if let report = molecules.preparationReport,
               !report.missingResidues.isEmpty || !report.chainBreaks.isEmpty || report.clashCount > 0 {
                return .attention
            }
            return .ready
        case .ligands:
            return molecules.ligand != nil || !docking.batchQueue.isEmpty ? .done : .ready
        case .bindingSite:
            if docking.gridInitialized || docking.selectedPocket != nil { return .done }
            return hasProtein ? .ready : .blocked
        case .dock:
            if docking.isDocking || docking.isBatchDocking { return .running }
            if hasDockingOutput { return .done }
            return hasProtein && molecules.ligand != nil ? .ready : .blocked
        case .results:
            return hasDockingOutput ? .done : .blocked
        case .leadOptimization:
            if !viewModel.leadOpt.analogs.isEmpty { return .done }
            return !docking.dockingResults.isEmpty || !docking.screeningHits.isEmpty ? .ready : .blocked
        case .sequence:
            return nil
        }
    }

    /// One-line, data-driven summary shown under the step header.
    @MainActor
    func summary(in viewModel: AppViewModel) -> String? {
        let molecules = viewModel.molecules
        let docking = viewModel.docking
        switch self {
        case .search:
            guard let protein = molecules.protein else { return nil }
            return "\(protein.name) · \(protein.chains.count) chain\(protein.chains.count == 1 ? "" : "s") · \(protein.atomCount.formatted()) atoms"
        case .preparation:
            guard let protein = molecules.protein else { return nil }
            return molecules.proteinPrepared
                ? "Prepared · \(protein.atomCount.formatted()) atoms"
                : "\(protein.name) · not prepared yet"
        case .ligands:
            let count = viewModel.ligandDB.count
            let library = "\(count) ligand\(count == 1 ? "" : "s") in library"
            if let ligand = molecules.ligand { return "\(ligand.name) · \(library)" }
            return count > 0 ? library : nil
        case .bindingSite:
            guard docking.gridInitialized || docking.selectedPocket != nil else { return nil }
            let size = docking.gridHalfSize * 2
            let box = String(format: "%.0f×%.0f×%.0f Å", size.x, size.y, size.z)
            if let pocket = docking.selectedPocket { return "\(pocket.displayName) · \(box)" }
            return "Custom box · \(box)"
        case .dock:
            // Live progress belongs to the canvas HUD and the toolbar capsule; the header keeps
            // describing the configured search so the three never repeat each other.
            return "\(docking.searchMethod.rawValue) · \(docking.scoringMethod.rawValue)"
        case .results:
            let results = docking.dockingResults
            guard let best = results.first else {
                return docking.batchResults.isEmpty ? nil : "\(docking.batchResults.count) ligands docked"
            }
            let method = docking.scoringMethod
            let score = String(format: "%.1f", best.displayScore(method: method))
            return "\(results.count) pose\(results.count == 1 ? "" : "s") · best \(score) \(method.unitLabel)"
        case .leadOptimization:
            let analogs = viewModel.leadOpt.analogs
            guard !analogs.isEmpty else { return nil }
            return "\(analogs.count) analogs · \(viewModel.leadOpt.dockedAnalogCount) docked"
        case .sequence:
            guard let protein = molecules.protein else { return nil }
            // Count amino acids only (as the Sequence overview and Inspector do), not waters or caps.
            let aminoAcids = protein.residues.filter(\.isStandard)
            let chains = Set(aminoAcids.map(\.chainID)).count
            return "\(chains) chain\(chains == 1 ? "" : "s") · \(aminoAcids.count.formatted()) residues"
        }
    }
}

/// The currently shown step, published to menu commands (Next / Previous Step).
extension FocusedValues {
    @Entry var selectedStep: Binding<SidebarTab>?
}


// MARK: - Navigator column

/// Left column of the main window: step header, then the step view (which brings its own
/// `DSStepScaffold` with an optional pinned footer). Steps are switched from the toolbar's
/// `PipelineStepBar`.
struct NavigatorColumn: View {
    @Environment(AppViewModel.self) private var viewModel
    let selection: SidebarTab

    var body: some View {
        VStack(spacing: 0) {
            DSStepHeader(tab: selection, summary: selection.summary(in: viewModel))
            stepView
                .id(selection)   // fresh scroll position per step
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
    }

    @ViewBuilder
    private var stepView: some View {
        switch selection {
        case .search:           SearchTabView()
        case .preparation:      PreparationTabView()
        case .ligands:          LigandDatabaseView()
        case .bindingSite:      BindingSiteTabView()
        case .dock:             DockingTabView()
        case .results:          ResultsTabView()
        case .leadOptimization: LeadOptimizationTabView()
        case .sequence:         SequenceView()
        }
    }
}

// MARK: - Pipeline step bar (toolbar, centred)

/// How much of each step the bar can afford to show, widest first.
private enum StepBarDensity: CaseIterable { case full, compact, iconOnly }

/// The single step switcher: pipeline steps 1…7, a divider, then the Sequence tool, in one
/// capsule. Each button carries its status badge and the `pipeline_<Step>` accessibility ID.
/// Shows number + icon + label when `availableWidth` allows, then number + icon, then icons.
/// Every density is measured off-screen (hidden, inaccessible) so the bar always takes its
/// natural width — toolbar items must not be wider than their content — and the width used
/// is reported back so the shell can hand the rest of the toolbar to the activity capsule.
struct PipelineStepBar: View {
    @Environment(AppViewModel.self) private var viewModel
    let selection: SidebarTab
    let availableWidth: CGFloat
    @Binding var usedWidth: CGFloat
    let select: (SidebarTab) -> Void
    @State private var naturalWidths: [StepBarDensity: CGFloat] = [:]

    private var density: StepBarDensity {
        StepBarDensity.allCases.first { (naturalWidths[$0] ?? .infinity) <= availableWidth } ?? .iconOnly
    }

    var body: some View {
        bar(density)
            .background {
                ZStack {
                    ForEach(StepBarDensity.allCases, id: \.self) { candidate in
                        bar(candidate)
                            .onGeometryChange(for: CGFloat.self, of: \.size.width) { naturalWidths[candidate] = $0 }
                    }
                }
                .hidden()
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            }
            .onChange(of: naturalWidths[density], initial: true) { _, width in usedWidth = width ?? 0 }
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Pipeline steps")
    }

    private func bar(_ density: StepBarDensity) -> some View {
        HStack(spacing: density == .iconOnly ? DS.Space.xxs : DS.Space.xs) {
            ForEach(SidebarTab.allCases.filter(\.isPipelineStep), id: \.self) { button(for: $0, density) }
            Divider()
                .frame(height: DS.Size.stepBarButton - DS.Space.m)
                .padding(.horizontal, DS.Space.xs)
            ForEach(SidebarTab.allCases.filter { !$0.isPipelineStep }, id: \.self) { button(for: $0, density) }
        }
        .padding(DS.Space.xxs + 1)
        .fixedSize()
        .glassEffect(.regular, in: Capsule())
    }

    private func button(for tab: SidebarTab, _ density: StepBarDensity) -> some View {
        PipelineStepButton(tab: tab, status: tab.status(in: viewModel), isSelected: selection == tab, density: density) {
            select(tab)
        }
    }
}

private struct PipelineStepButton: View {
    let tab: SidebarTab
    let status: DSStepStatus?
    let isSelected: Bool
    let density: StepBarDensity
    let action: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: DS.Space.xs + 1) {
                if density != .iconOnly { leadingMark }
                Image(systemName: tab.icon)
                    .font(DS.Font.body.weight(isSelected ? .semibold : .regular))
                    .symbolRenderingMode(.hierarchical)
                    .frame(minWidth: 16)
                    .overlay(alignment: .bottomTrailing) {
                        if density == .iconOnly { badge.offset(x: 4, y: 3) }
                    }
                if density == .full {
                    Text(tab.shortTitle)
                        .font(DS.Font.label.weight(isSelected ? .semibold : .medium))
                        .lineLimit(1)
                        .fixedSize()
                }
            }
            .foregroundStyle(foreground)
            .padding(.horizontal, density == .iconOnly ? DS.Space.s : DS.Space.m)
            .frame(minWidth: DS.Size.stepBarButton, minHeight: DS.Size.stepBarButton)
            .background(Capsule().fill(fill))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help(helpText)
        .accessibilityLabel(tab.rawValue)
        .accessibilityValue(statusText)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .accessibilityIdentifier(AccessibilityID.pipelineTab(tab.rawValue))
    }

    /// Step number in a small disc (status badge on its corner); tools get the badge alone.
    @ViewBuilder
    private var leadingMark: some View {
        if let number = tab.stepNumber {
            Text("\(number)")
                .font(DS.Font.caption.weight(.bold))
                .monospacedDigit()
                .frame(width: 17, height: 17)
                .background(Circle().fill(isSelected ? AnyShapeStyle(DS.Color.accentOnFill.opacity(0.22)) : AnyShapeStyle(DS.Color.controlFill)))
                .overlay(alignment: .bottomTrailing) { badge.offset(x: 4, y: 3) }
        }
    }

    @ViewBuilder
    private var badge: some View {
        if let status, status != .ready || !isSelected {
            DSStatusBadge(status, size: 6)
                .padding(1)
                .background(Circle().fill(.background))
        }
    }

    private var foreground: AnyShapeStyle {
        if isSelected { return AnyShapeStyle(DS.Color.accentOnFill) }
        return status == .blocked ? AnyShapeStyle(.secondary) : AnyShapeStyle(.primary)
    }

    private var fill: AnyShapeStyle {
        if isSelected { return AnyShapeStyle(DS.Color.accent) }
        return isHovering ? AnyShapeStyle(DS.Color.controlFill) : AnyShapeStyle(.clear)
    }

    private var helpText: String {
        let prefix = tab.stepNumber.map { "\($0). " } ?? ""
        return "\(prefix)\(tab.rawValue) — \(tab.subtitle) (⌘\(tab.shortcutNumber))"
    }

    private var statusText: String {
        switch status {
        case .done:      "Done"
        case .ready:     "Ready"
        case .blocked:   "Not available yet"
        case .running:   "Running"
        case .attention: "Needs attention"
        case nil:        "Tool"
        }
    }
}
