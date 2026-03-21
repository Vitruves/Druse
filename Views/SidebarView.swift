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
}

struct SidebarView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var selectedTab: SidebarTab = .search

    var body: some View {
        VStack(spacing: 0) {
            // Tab bar — icon-only with tooltips
            HStack(spacing: 0) {
                ForEach(SidebarTab.allCases, id: \.self) { tab in
                    Button(action: { selectedTab = tab }) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 14))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(selectedTab == tab ? .primary : .tertiary)
                    .background(selectedTab == tab ? Color.accentColor.opacity(0.12) : .clear)
                    .help(tab.rawValue)
                }
            }
            .frame(height: 36)
            .background(.ultraThinMaterial)

            Divider()

            // Tab content
            ScrollView {
                switch selectedTab {
                case .search:
                    SearchTabView()
                case .preparation:
                    PreparationTabView()
                case .sequence:
                    SequenceView()
                case .ligands:
                    LigandDatabaseView()
                case .dock:
                    DockingTabView()
                case .results:
                    ResultsTabView()
                }
            }
        }
        .frame(width: 280)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }
}
