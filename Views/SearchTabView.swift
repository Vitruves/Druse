import SwiftUI

struct SearchTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var pdbID: String = ""
    @State private var searchQuery: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Open from file
            VStack(alignment: .leading, spacing: 8) {
                Label("Open from File", systemImage: "folder")
                    .font(.callout.weight(.semibold))

                HStack(spacing: 4) {
                    Button(action: { viewModel.importFile() }) {
                        Label("Protein (.pdb)", systemImage: "doc")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .accessibilityIdentifier(AccessibilityID.searchImportProtein)

                    Button(action: { viewModel.importFile() }) {
                        Label("Ligand (.sdf)", systemImage: "doc")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .accessibilityIdentifier(AccessibilityID.searchImportLigand)
                }

                // Loaded molecules
                if let prot = viewModel.molecules.protein {
                    HStack(spacing: 8) {
                        Circle().fill(.cyan).frame(width: 8, height: 8)
                        Text(prot.name)
                            .font(.subheadline.weight(.medium))
                        Text("\(prot.atomCount) atoms")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.molecules.protein = nil; viewModel.pushToRenderer() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.body)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .contentShape(Rectangle().size(width: 24, height: 24))
                        .plainButtonAccessibility(AccessibilityID.searchClearProtein)
                    }
                    .padding(8)
                    .background(Color.cyan.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                if let lig = viewModel.molecules.ligand {
                    HStack(spacing: 8) {
                        Circle().fill(.green).frame(width: 8, height: 8)
                        Text(lig.name)
                            .font(.subheadline.weight(.medium))
                        Text("\(lig.atomCount) atoms")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.removeLigandFromView() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.body)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .contentShape(Rectangle().size(width: 24, height: 24))
                        .plainButtonAccessibility(AccessibilityID.searchClearLigand)
                    }
                    .padding(8)
                    .background(Color.green.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }

            Divider()

            // Quick fetch by PDB ID
            VStack(alignment: .leading, spacing: 8) {
                Label("Fetch by PDB ID", systemImage: "arrow.down.doc")
                    .font(.callout.weight(.semibold))

                HStack(spacing: 8) {
                    TextField("e.g. 1HSG", text: $pdbID)
                        .textFieldStyle(.roundedBorder)
                        .font(.callout.monospaced())
                        .onSubmit { fetchPDB() }
                        .accessibilityIdentifier(AccessibilityID.searchPDBField)

                    Button("Fetch") { fetchPDB() }
                        .controlSize(.small)
                        .disabled(pdbID.trimmingCharacters(in: .whitespaces).count < 4 || viewModel.workspace.isLoading)
                        .accessibilityIdentifier(AccessibilityID.searchFetchPDB)
                }
            }

            Divider()

            // Keyword search
            VStack(alignment: .leading, spacing: 8) {
                Label("Search RCSB", systemImage: "magnifyingglass")
                    .font(.callout.weight(.semibold))

                HStack(spacing: 8) {
                    TextField("e.g. HIV protease", text: $searchQuery)
                        .textFieldStyle(.roundedBorder)
                        .font(.callout)
                        .onSubmit { viewModel.searchPDB(query: searchQuery) }
                        .accessibilityIdentifier(AccessibilityID.searchQueryField)

                    Button("Search") { viewModel.searchPDB(query: searchQuery) }
                        .controlSize(.small)
                        .disabled(searchQuery.isEmpty || viewModel.workspace.isSearching)
                        .accessibilityIdentifier(AccessibilityID.searchRunSearch)
                }
            }

            // Loading indicators
            if viewModel.workspace.isLoading {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(viewModel.workspace.loadingMessage)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            if viewModel.workspace.isSearching {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Searching...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            // Search results as cards
            if !viewModel.workspace.searchResults.isEmpty {
                Divider()

                Text("\(viewModel.workspace.searchResults.count) results")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)

                ForEach(viewModel.workspace.searchResults) { result in
                    resultCard(result)
                }
            }
        }
        .padding(12)
    }

    private func fetchPDB() {
        let id = pdbID.trimmingCharacters(in: .whitespaces)
        guard id.count >= 4 else { return }
        viewModel.loadFromPDB(id: id)
    }

    @ViewBuilder
    private func resultCard(_ result: PDBSearchResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header: PDB ID + resolution
            HStack(alignment: .firstTextBaseline) {
                Text(result.id)
                    .font(.title3.bold().monospaced())

                Spacer()

                if let res = result.resolution {
                    HStack(spacing: 2) {
                        Text(String(format: "%.1f", res))
                            .font(.callout.weight(.semibold).monospaced())
                        Text("\u{00C5}")
                            .font(.footnote)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            // Title
            Text(result.title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(3)

            // Method + date
            HStack(spacing: 8) {
                if let method = result.experimentMethod {
                    Text(method)
                        .font(.footnote.weight(.medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.accentColor.opacity(0.12))
                        .clipShape(Capsule())
                }

                if let date = result.releaseDate {
                    Text(date)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            // Load button
            Button(action: { viewModel.loadFromPDB(id: result.id) }) {
                Label("Load Structure", systemImage: "arrow.down.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(viewModel.workspace.isLoading)
            .accessibilityIdentifier(AccessibilityID.searchLoadStructure)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 0.5)
        )
    }
}
