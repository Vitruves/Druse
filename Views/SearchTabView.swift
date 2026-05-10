// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

struct SearchTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var pdbID: String = ""
    @State private var searchQuery: String = ""
    @State private var cachedEntries: [CachedPDBEntry] = []
    @State private var isLoadingCache: Bool = true
    /// Tracks which card initiated the in-flight load so the inline progress
    /// indicator only appears under the source card (not under every card).
    @State private var loadSource: LoadSource? = nil

    private enum LoadSource { case loadStructure, recentDownloads, searchResults }

    var body: some View {
        VStack(alignment: .leading, spacing: PanelStyle.cardSpacing) {
            loadStructureCard
            searchRCSBCard
            if isLoadingCache || !cachedEntries.isEmpty {
                recentDownloadsCard
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .task { await refreshCachedEntries() }
        .onChange(of: viewModel.workspace.isLoading) { _, loading in
            if !loading {
                loadSource = nil
                Task { await refreshCachedEntries() }
            }
        }
    }

    /// True when the global isLoading flag is set AND this card initiated
    /// (or is associated with) the load.
    private func isLoading(in source: LoadSource) -> Bool {
        viewModel.workspace.isLoading && loadSource == source
    }

    @ViewBuilder
    private func inlineLoadingIndicator() -> some View {
        HStack(spacing: 6) {
            ProgressView().controlSize(.small)
            Text(viewModel.workspace.loadingMessage)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Load structure card

    @ViewBuilder
    private var loadStructureCard: some View {
        PanelCard("Load Structure", icon: "tray.and.arrow.down") {
            VStack(alignment: .leading, spacing: 10) {
                if viewModel.molecules.protein != nil || viewModel.molecules.ligand != nil {
                    activeMoleculesSection
                }

                PanelSubheader(title: "From File")
                PanelChoiceGrid(columns: 2) {
                    PanelSecondaryButton(
                        title: "Protein", icon: "doc",
                        help: "Open .pdb / .cif protein file"
                    ) { viewModel.importFile() }
                    .accessibilityIdentifier(AccessibilityID.searchImportProtein)

                    PanelSecondaryButton(
                        title: "Ligand", icon: "doc",
                        help: "Open .sdf / .mol2 / .pdb ligand file"
                    ) { viewModel.importFile() }
                    .accessibilityIdentifier(AccessibilityID.searchImportLigand)
                }

                PanelLabeledDivider(title: "From PDB", icon: "arrow.down.doc")

                HStack(spacing: 6) {
                    TextField("e.g. 1HSG", text: $pdbID)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                        .font(PanelStyle.monoBody)
                        .onSubmit { fetchPDB() }
                        .accessibilityIdentifier(AccessibilityID.searchPDBField)

                    PanelSecondaryButton(
                        title: "Fetch", icon: "arrow.down.circle",
                        isDisabled: pdbID.trimmingCharacters(in: .whitespaces).count < 4
                            || viewModel.workspace.isLoading,
                        help: "Fetch this PDB entry from RCSB"
                    ) { fetchPDB() }
                    .accessibilityIdentifier(AccessibilityID.searchFetchPDB)
                    .frame(width: 84)
                }

                if isLoading(in: .loadStructure) {
                    inlineLoadingIndicator()
                }
            }
        }
    }

    @ViewBuilder
    private var activeMoleculesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let prot = viewModel.molecules.protein {
                PanelHighlightRow(color: .cyan) {
                    HStack(spacing: 6) {
                        Image(systemName: "cube.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.cyan)
                        Text(prot.name)
                            .font(PanelStyle.bodyFont.weight(.medium))
                            .lineLimit(1)
                        Spacer()
                        Text("\(prot.atomCount) atoms")
                            .font(PanelStyle.monoSmall)
                            .foregroundStyle(.secondary)
                        Button(action: {
                            viewModel.molecules.protein = nil
                            viewModel.pushToRenderer()
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Unload protein")
                        .plainButtonAccessibility(AccessibilityID.searchClearProtein)
                    }
                }
            }
            if let lig = viewModel.molecules.ligand {
                PanelHighlightRow(color: .green) {
                    HStack(spacing: 6) {
                        Image(systemName: "hexagon.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.green)
                        Text(lig.name)
                            .font(PanelStyle.bodyFont.weight(.medium))
                            .lineLimit(1)
                        Spacer()
                        Text("\(lig.atomCount) atoms")
                            .font(PanelStyle.monoSmall)
                            .foregroundStyle(.secondary)
                        Button(action: { viewModel.removeLigandFromView() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Remove ligand")
                        .plainButtonAccessibility(AccessibilityID.searchClearLigand)
                    }
                }
            }
        }
    }

    // MARK: - Search RCSB card

    @ViewBuilder
    private var searchRCSBCard: some View {
        PanelCard(
            "Search RCSB",
            icon: "magnifyingglass",
            accessory: {
                if !viewModel.workspace.searchResults.isEmpty {
                    Text("\(viewModel.workspace.searchResults.count)")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.primary.opacity(0.08)))
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 6) {
                    TextField("e.g. HIV protease", text: $searchQuery)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                        .font(PanelStyle.bodyFont)
                        .onSubmit { viewModel.searchPDB(query: searchQuery) }
                        .accessibilityIdentifier(AccessibilityID.searchQueryField)

                    PanelSecondaryButton(
                        title: "Search", icon: "magnifyingglass",
                        isDisabled: searchQuery.isEmpty || viewModel.workspace.isSearching
                    ) { viewModel.searchPDB(query: searchQuery) }
                    .accessibilityIdentifier(AccessibilityID.searchRunSearch)
                    .frame(width: 84)
                }

                if viewModel.workspace.isSearching {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Searching…")
                            .font(PanelStyle.smallFont)
                            .foregroundStyle(.secondary)
                    }
                }

                if !viewModel.workspace.searchResults.isEmpty {
                    PanelLabeledDivider(title: "Results")
                    ForEach(viewModel.workspace.searchResults) { result in
                        resultRow(result)
                    }
                    if isLoading(in: .searchResults) {
                        inlineLoadingIndicator()
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func resultRow(_ result: PDBSearchResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(result.id)
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                Spacer()
                if let res = result.resolution {
                    HStack(spacing: 2) {
                        Text(String(format: "%.1f", res))
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        Text("Å")
                            .font(PanelStyle.smallFont)
                    }
                    .foregroundStyle(.secondary)
                }
            }
            Text(result.title)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 6) {
                if let method = result.experimentMethod {
                    Text(method)
                        .font(.system(size: 10, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                        .foregroundStyle(Color.accentColor)
                }
                if let date = result.releaseDate {
                    Text(date)
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                PanelSecondaryButton(
                    title: "Load", icon: "arrow.down.circle",
                    isDisabled: viewModel.workspace.isLoading
                ) {
                    loadSource = .searchResults
                    viewModel.loadFromPDB(id: result.id)
                }
                .accessibilityIdentifier(AccessibilityID.searchLoadStructure)
                .frame(width: 80)
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.primary.opacity(0.04))
        )
    }

    // MARK: - Recent downloads card

    @ViewBuilder
    private var recentDownloadsCard: some View {
        PanelCard(
            "Recent Downloads",
            icon: "clock.arrow.circlepath",
            accessory: {
                if isLoadingCache {
                    ProgressView().controlSize(.small)
                } else if !cachedEntries.isEmpty {
                    Text("\(cachedEntries.count)")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 6) {
                if isLoadingCache && cachedEntries.isEmpty {
                    Text("Scanning cache…")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 4) {
                        ForEach(cachedEntries) { entry in
                            cachedEntryRow(entry)
                        }
                    }
                }
                if isLoading(in: .recentDownloads) {
                    inlineLoadingIndicator()
                }
            }
        }
    }

    @ViewBuilder
    private func cachedEntryRow(_ entry: CachedPDBEntry) -> some View {
        HStack(spacing: 6) {
            Button(action: {
                loadSource = .recentDownloads
                viewModel.loadFromPDB(id: entry.id)
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "doc")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Text(entry.id)
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    Text(entry.fetchedAt.formatted(.relative(presentation: .named)))
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer()
                    Text("\(entry.sizeBytes / 1024) KB")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(viewModel.workspace.isLoading)

            Button(action: {
                Task {
                    await PDBService.shared.removeCachedEntry(id: entry.id)
                    await refreshCachedEntries()
                }
            }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Remove from cache")
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(Color.primary.opacity(0.04))
        )
    }

    // MARK: - Actions

    private func refreshCachedEntries() async {
        isLoadingCache = true
        let entries = await PDBService.shared.cachedEntries()
        cachedEntries = entries
        isLoadingCache = false
    }

    private func fetchPDB() {
        let id = pdbID.trimmingCharacters(in: .whitespaces)
        guard id.count >= 4 else { return }
        loadSource = .loadStructure
        viewModel.loadFromPDB(id: id)
    }
}
