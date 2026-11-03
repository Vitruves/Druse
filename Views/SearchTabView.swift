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
        Group {
            if viewModel.molecules.protein != nil {
                DSStepScaffold {
                    cards
                } footer: {
                    HStack {
                        Spacer(minLength: 0)
                        DSNextStepButton(SidebarTab.search.suggestedNextStep(in: viewModel) ?? .preparation)
                            .fixedSize()
                    }
                }
            } else {
                DSStepScaffold { cards }
            }
        }
        .task { await refreshCachedEntries() }
        .onChange(of: viewModel.workspace.isLoading) { _, loading in
            if !loading {
                loadSource = nil
                Task { await refreshCachedEntries() }
            }
        }
    }

    @ViewBuilder
    private var cards: some View {
        if viewModel.molecules.protein != nil || viewModel.molecules.ligand != nil {
            loadedCard
        }
        loadStructureCard
        searchRCSBCard
        if isLoadingCache || !cachedEntries.isEmpty {
            recentDownloadsCard
        }
    }

    /// True when the global isLoading flag is set AND this card initiated
    /// (or is associated with) the load.
    private func isLoading(in source: LoadSource) -> Bool {
        viewModel.workspace.isLoading && loadSource == source
    }

    private func inlineLoadingIndicator() -> some View {
        HStack(spacing: DS.Space.s) {
            ProgressView().controlSize(.small)
            Text(viewModel.workspace.loadingMessage)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    // MARK: - Loaded card

    private var loadedCard: some View {
        PanelCard("Loaded", icon: "checkmark.circle") {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                if let prot = viewModel.molecules.protein {
                    loadedRow(
                        color: DS.Color.protein,
                        name: prot.name,
                        detail: "\(prot.chains.count) chain\(prot.chains.count == 1 ? "" : "s") · \(prot.atomCount.formatted()) atoms",
                        clearHelp: "Unload protein",
                        clearID: AccessibilityID.searchClearProtein
                    ) {
                        viewModel.molecules.protein = nil
                        viewModel.pushToRenderer()
                    }
                }
                if let lig = viewModel.molecules.ligand {
                    loadedRow(
                        color: DS.Color.ligand,
                        name: lig.name,
                        detail: "\(lig.heavyAtomCount) heavy atoms",
                        clearHelp: "Remove ligand",
                        clearID: AccessibilityID.searchClearLigand
                    ) {
                        viewModel.removeLigandFromView()
                    }
                }
            }
        }
    }

    private func loadedRow(
        color: Color,
        name: String,
        detail: String,
        clearHelp: String,
        clearID: String,
        onClear: @escaping () -> Void
    ) -> some View {
        PanelHighlightRow(color: color) {
            HStack(spacing: DS.Space.m) {
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 1) {
                    Text(name)
                        .font(DS.Font.label.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(detail)
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: DS.Space.xs)
                Button(action: onClear) {
                    Image(systemName: "xmark.circle.fill")
                        .font(DS.Font.label)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help(clearHelp)
                .accessibilityLabel(clearHelp)
                .accessibilityIdentifier(clearID)
            }
        }
    }

    // MARK: - Load structure card

    private var loadStructureCard: some View {
        PanelCard("Load Structure", icon: "square.and.arrow.down.on.square") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                PanelLabeledDivider(title: "From a file")
                PanelChoiceGrid(columns: 2) {
                    PanelSecondaryButton(
                        title: "Protein…", icon: "cube",
                        help: "Open a .pdb / .cif protein file (⌘O)"
                    ) { viewModel.importFile() }
                    .accessibilityIdentifier(AccessibilityID.searchImportProtein)

                    PanelSecondaryButton(
                        title: "Ligand…", icon: "hexagon",
                        help: "Open a .sdf / .mol2 / .pdb ligand file"
                    ) { viewModel.importFile() }
                    .accessibilityIdentifier(AccessibilityID.searchImportLigand)
                }

                PanelLabeledDivider(title: "From the PDB")

                HStack(spacing: DS.Space.s) {
                    TextField("PDB ID, e.g. 1HSG", text: $pdbID)
                        .textFieldStyle(.roundedBorder)
                        .font(DS.Font.mono)
                        .onSubmit { fetchPDB() }
                        .accessibilityIdentifier(AccessibilityID.searchPDBField)

                    Button(action: fetchPDB) {
                        Label("Fetch", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                    .disabled(pdbID.trimmingCharacters(in: .whitespaces).count < 4
                              || viewModel.workspace.isLoading)
                    .help("Fetch this entry from RCSB")
                    .accessibilityIdentifier(AccessibilityID.searchFetchPDB)
                }

                if isLoading(in: .loadStructure) {
                    inlineLoadingIndicator()
                }

                PanelHint(
                    text: "Or drop .pdb, .cif, .sdf or .mol2 files onto the viewport.",
                    icon: "arrow.down.doc"
                )
            }
        }
    }

    // MARK: - Search RCSB card

    private var searchRCSBCard: some View {
        PanelCard(
            "Search RCSB",
            icon: "magnifyingglass",
            accessory: {
                if !viewModel.workspace.searchResults.isEmpty {
                    DSTag("\(viewModel.workspace.searchResults.count)", monospaced: true)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                HStack(spacing: DS.Space.s) {
                    TextField("e.g. HIV protease", text: $searchQuery)
                        .textFieldStyle(.roundedBorder)
                        .font(DS.Font.label)
                        .onSubmit { runSearch() }
                        .accessibilityIdentifier(AccessibilityID.searchQueryField)

                    Button(action: runSearch) {
                        Label("Search", systemImage: "magnifyingglass")
                    }
                    .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                    .disabled(searchQuery.isEmpty || viewModel.workspace.isSearching)
                    .help("Search the RCSB Protein Data Bank")
                    .accessibilityIdentifier(AccessibilityID.searchRunSearch)
                }

                if viewModel.workspace.isSearching {
                    HStack(spacing: DS.Space.s) {
                        ProgressView().controlSize(.small)
                        Text("Searching…")
                            .font(DS.Font.secondary)
                            .foregroundStyle(.secondary)
                    }
                }

                if !viewModel.workspace.searchResults.isEmpty {
                    VStack(spacing: DS.Space.s) {
                        ForEach(viewModel.workspace.searchResults) { result in
                            resultRow(result)
                        }
                    }
                    if isLoading(in: .searchResults) {
                        inlineLoadingIndicator()
                    }
                }
            }
        }
    }

    private func resultRow(_ result: PDBSearchResult) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xs) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                Text(result.id)
                    .font(DS.Font.mono.weight(.semibold))
                    .textSelection(.enabled)
                if let res = result.resolution {
                    DSTag(String(format: "%.2f Å", res), tone: resolutionTone(res), monospaced: true)
                        .help("Resolution")
                }
                Spacer(minLength: DS.Space.xs)
                Button {
                    loadSource = .searchResults
                    viewModel.loadFromPDB(id: result.id)
                } label: {
                    Label("Load", systemImage: "arrow.down.circle")
                }
                .buttonStyle(DSSecondaryButtonStyle(tint: DS.Color.accent, fullWidth: false))
                .disabled(viewModel.workspace.isLoading)
                .help("Load \(result.id)")
                .accessibilityIdentifier(AccessibilityID.searchLoadStructure)
            }
            Text(result.title)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .help(result.title)
            if result.experimentMethod != nil || result.releaseDate != nil {
                HStack(spacing: DS.Space.s) {
                    if let method = result.experimentMethod {
                        DSTag(method.capitalized)
                    }
                    if let date = result.releaseDate {
                        Text(date)
                            .font(DS.Font.caption)
                            .monospacedDigit()
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(DS.Space.m)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(DS.Color.controlFill)
        )
    }

    /// Crystallographic quality at a glance: ≤ 2 Å good, ≤ 3 Å fair, above that weak.
    private func resolutionTone(_ resolution: Float) -> DSTone {
        resolution <= 2.0 ? .success : resolution <= 3.0 ? .neutral : .warning
    }

    // MARK: - Recent downloads card

    private var recentDownloadsCard: some View {
        PanelCard(
            "Recent Downloads",
            icon: "clock.arrow.circlepath",
            accessory: {
                if isLoadingCache {
                    ProgressView().controlSize(.small)
                } else if !cachedEntries.isEmpty {
                    DSTag("\(cachedEntries.count)", monospaced: true)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                if isLoadingCache && cachedEntries.isEmpty {
                    Text("Scanning cache…")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(cachedEntries) { entry in
                        CachedEntryRow(
                            entry: entry,
                            isDisabled: viewModel.workspace.isLoading,
                            onLoad: {
                                loadSource = .recentDownloads
                                viewModel.loadFromPDB(id: entry.id)
                            },
                            onRemove: {
                                Task {
                                    await PDBService.shared.removeCachedEntry(id: entry.id)
                                    await refreshCachedEntries()
                                }
                            }
                        )
                    }
                }
                if isLoading(in: .recentDownloads) {
                    inlineLoadingIndicator()
                }
            }
        }
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

    private func runSearch() {
        guard !searchQuery.isEmpty else { return }
        viewModel.searchPDB(query: searchQuery)
    }
}

// MARK: - Recent download row (hover-highlighted list row)

private struct CachedEntryRow: View {
    let entry: CachedPDBEntry
    let isDisabled: Bool
    let onLoad: () -> Void
    let onRemove: () -> Void
    @State private var isHovering = false

    var body: some View {
        HStack(spacing: DS.Space.s) {
            Button(action: onLoad) {
                HStack(spacing: DS.Space.m) {
                    Image(systemName: "cube")
                        .font(DS.Font.secondary)
                        .foregroundStyle(DS.Color.protein)
                    Text(entry.id)
                        .font(DS.Font.mono.weight(.semibold))
                    Text(entry.fetchedAt.formatted(.relative(presentation: .named)))
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer(minLength: DS.Space.xs)
                    Text(ByteCountFormatter.string(fromByteCount: Int64(entry.sizeBytes), countStyle: .file))
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.tertiary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(isDisabled)
            .help("Load \(entry.id)")

            // Always present (reachable by keyboard focus and VoiceOver); quiet until hovered.
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(DS.Font.label)
                    .foregroundStyle(isHovering ? AnyShapeStyle(.secondary) : AnyShapeStyle(.quaternary))
            }
            .buttonStyle(.borderless)
            .help("Remove \(entry.id) from the download cache")
            .accessibilityLabel("Remove \(entry.id) from cache")
        }
        .padding(.horizontal, DS.Space.s)
        .frame(height: DS.Size.rowHeight + DS.Space.xs)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(isHovering ? DS.Color.controlFill : Color.clear)
        )
        .onHover { isHovering = $0 }
    }
}
