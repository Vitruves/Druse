// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Sidebar tabs ordered by pipeline processing flow:
/// Search → Prepare → Sequence → Ligands → Docking → Results → Lead Opt
enum SidebarTab: String, CaseIterable {
    case search           = "Search"
    case preparation      = "Preparation"
    case sequence         = "Sequence"
    case ligands          = "Ligands"
    case dock             = "Docking"
    case results          = "Results"
    case leadOptimization = "Lead Opt"

    var icon: String {
        switch self {
        case .search:           "magnifyingglass"
        case .preparation:      "wand.and.stars"
        case .sequence:         "textformat.abc"
        case .ligands:          "tray.full"
        case .dock:             "arrow.triangle.merge"
        case .results:          "chart.bar.xaxis"
        case .leadOptimization: "arrow.triangle.branch"
        }
    }

    var subtitle: String {
        switch self {
        case .search:           "Load structure"
        case .preparation:      "Clean & protonate"
        case .sequence:         "Inspect chains"
        case .ligands:          "Add compounds"
        case .dock:             "Run docking"
        case .results:          "Analyze poses"
        case .leadOptimization: "Optimize leads"
        }
    }

    /// Compact label for the toolbar pill
    var shortLabel: String {
        switch self {
        case .search:           "Search"
        case .preparation:      "Prep"
        case .sequence:         "Seq"
        case .ligands:          "Ligands"
        case .dock:             "Dock"
        case .results:          "Results"
        case .leadOptimization: "Lead Opt"
        }
    }
}

// MARK: - Pipeline Bar (horizontal, lives in toolbar)

struct PipelineBar: View {
    @Environment(AppViewModel.self) private var viewModel
    @Binding var selectedTab: SidebarTab
    @Binding var panelOpen: Bool

    var body: some View {
        HStack(spacing: 2) {
            ForEach(SidebarTab.allCases, id: \.self) { tab in
                pipelineButton(tab)
                if tab != SidebarTab.allCases.last {
                    Image(systemName: "chevron.right")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.quaternary)
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func pipelineButton(_ tab: SidebarTab) -> some View {
        let isSelected = selectedTab == tab && panelOpen
        let status = stepStatus(tab)

        Button(action: {
            withAnimation(.easeInOut(duration: 0.15)) {
                if selectedTab == tab && panelOpen {
                    panelOpen = false
                } else {
                    selectedTab = tab
                    panelOpen = true
                }
            }
        }) {
            HStack(spacing: 8) {
                // Status-aware icon
                ZStack {
                    Circle()
                        .fill(circleFill(status: status, isSelected: isSelected))
                        .frame(width: 26, height: 26)

                    if isSelected {
                        Circle()
                            .stroke(Color.accentColor, lineWidth: 1.5)
                            .frame(width: 26, height: 26)
                    }

                    circleContent(tab: tab, status: status, isSelected: isSelected)
                }

                VStack(alignment: .leading, spacing: 1) {
                    Text(tab.rawValue)
                        .font(isSelected ? .callout.weight(.bold) : .callout.weight(.semibold))
                        .foregroundStyle(isSelected ? .primary : .secondary)
                        .lineLimit(1)
                    Text(tab.subtitle)
                        .font(.caption)
                        .foregroundStyle(isSelected ? .secondary : .secondary)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
            )
            .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier(AccessibilityID.pipelineTab(tab.rawValue))
    }

    // MARK: - Step Status

    private enum StepStatus { case completed, available, upcoming }

    private func stepStatus(_ tab: SidebarTab) -> StepStatus {
        switch tab {
        case .search:
            return viewModel.molecules.protein != nil ? .completed : .upcoming
        case .preparation:
            if viewModel.molecules.proteinPrepared { return .completed }
            return viewModel.molecules.protein != nil ? .available : .upcoming
        case .sequence:
            return viewModel.molecules.protein != nil ? .completed : .upcoming
        case .ligands:
            return viewModel.molecules.ligand != nil ? .completed : .upcoming
        case .dock:
            if !viewModel.docking.dockingResults.isEmpty { return .completed }
            return (viewModel.molecules.ligand != nil && viewModel.molecules.protein != nil) ? .available : .upcoming
        case .results:
            return !viewModel.docking.dockingResults.isEmpty ? .completed : .upcoming
        case .leadOptimization:
            if !viewModel.leadOpt.analogs.isEmpty { return .completed }
            return !viewModel.docking.dockingResults.isEmpty || !viewModel.docking.screeningHits.isEmpty ? .available : .upcoming
        }
    }

    private func circleFill(status: StepStatus, isSelected: Bool) -> Color {
        if isSelected { return Color.accentColor.opacity(0.25) }
        switch status {
        case .completed: return Color.green.opacity(0.2)
        case .available: return Color.orange.opacity(0.15)
        case .upcoming:  return Color.primary.opacity(0.06)
        }
    }

    @ViewBuilder
    private func circleContent(tab: SidebarTab, status: StepStatus, isSelected: Bool) -> some View {
        if status == .completed && !isSelected {
            Image(systemName: "checkmark")
                .font(.caption.weight(.bold))
                .foregroundStyle(.green)
        } else if status == .available && !isSelected {
            Image(systemName: tab.icon)
                .font(.caption)
                .foregroundStyle(.orange)
        } else {
            Image(systemName: tab.icon)
                .font(.caption)
                .foregroundStyle(isSelected ? .primary : .secondary)
        }
    }
}

// MARK: - Pipeline Content Panel (overlay on render area)

struct PipelineContentPanel: View {
    @Environment(AppViewModel.self) private var viewModel
    @Binding var selectedTab: SidebarTab
    @Binding var panelOpen: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: selectedTab.icon)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                Text(selectedTab.rawValue)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.primary)
                Spacer()
                Button(action: { withAnimation(.easeInOut(duration: 0.15)) { panelOpen = false } }) {
                    Image(systemName: "xmark")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 20, height: 20)
                        .background(Color.primary.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
                .buttonStyle(.plain)
                .help("Close panel")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    Color.clear.frame(height: 0).id("__panel_top__")
                    switch selectedTab {
                    case .search:           SearchTabView()
                    case .preparation:      PreparationTabView()
                    case .sequence:         SequenceView()
                    case .ligands:          LigandDatabaseView()
                    case .dock:             DockingTabView()
                    case .results:          ResultsTabView()
                    case .leadOptimization: LeadOptimizationTabView()
                    }
                }
                .onChange(of: selectedTab) { _, _ in
                    proxy.scrollTo("__panel_top__", anchor: .top)
                }
                .onChange(of: panelOpen) { _, isOpen in
                    if isOpen { proxy.scrollTo("__panel_top__", anchor: .top) }
                }
            }
        }
        .frame(width: 300)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

// MARK: - Legacy wrapper (keep SidebarView compiling for any remaining references)

struct SidebarView: View {
    var body: some View {
        EmptyView()
    }
}
