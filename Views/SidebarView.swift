import SwiftUI

/// Sidebar tabs ordered by pipeline processing flow:
/// Search → Prepare → Sequence → Ligands → Docking → Results
enum SidebarTab: String, CaseIterable {
    case search       = "Search"
    case preparation  = "Preparation"
    case sequence     = "Sequence"
    case ligands      = "Ligands"
    case dock         = "Docking"
    case results      = "Results"

    var icon: String {
        switch self {
        case .search:      "magnifyingglass"
        case .preparation: "wand.and.stars"
        case .sequence:    "textformat.abc"
        case .ligands:     "tray.full"
        case .dock:        "arrow.triangle.merge"
        case .results:     "chart.bar.xaxis"
        }
    }

    var subtitle: String {
        switch self {
        case .search:      "Load structure"
        case .preparation: "Clean & protonate"
        case .sequence:    "Inspect chains"
        case .ligands:     "Add compounds"
        case .dock:        "Run docking"
        case .results:     "Analyze poses"
        }
    }
}

struct SidebarView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var selectedTab: SidebarTab = .search
    @State private var hoveredTab: SidebarTab?
    @State private var panelExpanded: Bool = true

    var body: some View {
        HStack(spacing: 0) {
            // Left: vertical step pipeline
            VStack(spacing: 0) {
                ForEach(SidebarTab.allCases, id: \.self) { tab in
                    stepRow(tab)
                }

                Spacer()

                // Expand button at bottom of stepper (only when collapsed)
                if !panelExpanded {
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.2)) { panelExpanded = true }
                    }) {
                        Image(systemName: "sidebar.right")
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.plain)
                    .help("Expand panel")
                }
            }
            .frame(width: panelExpanded ? 150 : 52)
            .padding(.top, 8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))

            // Panel content — only when expanded
            if panelExpanded {
                Divider()

                VStack(spacing: 0) {
                    // Panel header with collapse button
                    HStack {
                        Text(selectedTab.rawValue)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: {
                            withAnimation(.easeInOut(duration: 0.2)) { panelExpanded = false }
                        }) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(.secondary)
                                .frame(width: 22, height: 22)
                                .background(Color.primary.opacity(0.06))
                                .clipShape(RoundedRectangle(cornerRadius: 5))
                        }
                        .buttonStyle(.plain)
                        .help("Collapse panel")
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)

                    Divider()

                    ScrollView {
                        switch selectedTab {
                        case .search:      SearchTabView()
                        case .preparation: PreparationTabView()
                        case .sequence:    SequenceView()
                        case .ligands:     LigandDatabaseView()
                        case .dock:        DockingTabView()
                        case .results:     ResultsTabView()
                        }
                    }
                }
                .frame(width: 300)
            }
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }

    // MARK: - Step Row

    @ViewBuilder
    private func stepRow(_ tab: SidebarTab) -> some View {
        let isSelected = selectedTab == tab
        let isHovered = hoveredTab == tab
        let status = stepStatus(tab)

        Button(action: {
            withAnimation(.easeInOut(duration: 0.15)) {
                if selectedTab == tab && panelExpanded {
                    // Clicking the already-selected step collapses the panel
                    panelExpanded = false
                } else {
                    selectedTab = tab
                    panelExpanded = true
                }
            }
        }) {
            HStack(spacing: 0) {
                // Step indicator column
                VStack(spacing: 0) {
                    Rectangle()
                        .fill(tab == .search ? Color.clear : connectorColor(for: previousTab(tab)))
                        .frame(width: 2, height: 8)

                    ZStack {
                        Circle()
                            .fill(stepCircleFill(status: status, isSelected: isSelected))
                            .frame(width: 24, height: 24)

                        if isSelected {
                            Circle()
                                .stroke(Color.accentColor, lineWidth: 2)
                                .frame(width: 24, height: 24)
                        } else if status == .available {
                            Circle()
                                .stroke(Color.orange.opacity(0.5), lineWidth: 1.5)
                                .frame(width: 24, height: 24)
                        }

                        stepCircleContent(tab: tab, status: status, isSelected: isSelected)
                    }

                    Rectangle()
                        .fill(tab == .results ? Color.clear : connectorColor(for: tab))
                        .frame(width: 2, height: 8)
                }
                .frame(width: panelExpanded ? 40 : 52)

                // Label (only when panel expanded)
                if panelExpanded {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(tab.rawValue)
                            .font(.system(size: 11, weight: isSelected ? .semibold : .regular))
                            .foregroundStyle(isSelected ? .primary : .secondary)
                            .lineLimit(1)
                        Text(tab.subtitle)
                            .font(.system(size: 9))
                            .foregroundStyle(isSelected ? .secondary : .tertiary)
                            .lineLimit(1)
                    }
                    .padding(.trailing, 8)
                    Spacer(minLength: 0)
                }
            }
            .frame(height: 40)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : isHovered ? Color.primary.opacity(0.04) : .clear)
                .padding(.horizontal, 4)
        )
        .onHover { hovering in hoveredTab = hovering ? tab : nil }
    }

    // MARK: - Step Status

    /// Three states: completed (green), available (orange — can do but hasn't), upcoming (gray)
    private enum StepStatus {
        case completed, available, upcoming
    }

    private func stepStatus(_ tab: SidebarTab) -> StepStatus {
        switch tab {
        case .search:
            return viewModel.protein != nil ? .completed : .upcoming
        case .preparation:
            if viewModel.proteinPrepared { return .completed }
            return viewModel.protein != nil ? .available : .upcoming
        case .sequence:
            return viewModel.protein != nil ? .completed : .upcoming
        case .ligands:
            return viewModel.ligand != nil ? .completed : .upcoming
        case .dock:
            if !viewModel.dockingResults.isEmpty { return .completed }
            return (viewModel.ligand != nil && viewModel.protein != nil) ? .available : .upcoming
        case .results:
            return !viewModel.dockingResults.isEmpty ? .completed : .upcoming
        }
    }

    private func stepCircleFill(status: StepStatus, isSelected: Bool) -> Color {
        if isSelected { return Color.accentColor.opacity(0.2) }
        switch status {
        case .completed: return Color.green.opacity(0.15)
        case .available: return Color.orange.opacity(0.12)
        case .upcoming:  return Color.primary.opacity(0.05)
        }
    }

    @ViewBuilder
    private func stepCircleContent(tab: SidebarTab, status: StepStatus, isSelected: Bool) -> some View {
        if status == .completed && !isSelected {
            Image(systemName: "checkmark")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.green)
        } else if status == .available && !isSelected {
            Image(systemName: tab.icon)
                .font(.system(size: 10))
                .foregroundStyle(.orange)
        } else {
            Image(systemName: tab.icon)
                .font(.system(size: 10))
                .foregroundStyle(isSelected ? .primary : .tertiary)
        }
    }

    private func connectorColor(for tab: SidebarTab) -> Color {
        let s = stepStatus(tab)
        if s == .completed { return Color.green.opacity(0.4) }
        if s == .available { return Color.orange.opacity(0.3) }
        return Color.primary.opacity(0.08)
    }

    private func previousTab(_ tab: SidebarTab) -> SidebarTab {
        let all = SidebarTab.allCases
        guard let idx = all.firstIndex(of: tab), idx > 0 else { return .search }
        return all[idx - 1]
    }
}
