import SwiftUI

struct SearchTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var pdbID: String = ""
    @State private var searchQuery: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Open from file
            VStack(alignment: .leading, spacing: 6) {
                Label("Open from File", systemImage: "folder")
                    .font(.system(size: 12, weight: .semibold))

                HStack(spacing: 4) {
                    Button(action: { viewModel.importFile() }) {
                        Label("Protein (.pdb)", systemImage: "doc")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button(action: { viewModel.importFile() }) {
                        Label("Ligand (.sdf)", systemImage: "doc")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                // Loaded molecules
                if let prot = viewModel.molecules.protein {
                    HStack(spacing: 6) {
                        Circle().fill(.cyan).frame(width: 8, height: 8)
                        Text(prot.name)
                            .font(.system(size: 11, weight: .medium))
                        Text("\(prot.atomCount) atoms")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.molecules.protein = nil; viewModel.pushToRenderer() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 14))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .contentShape(Rectangle().size(width: 24, height: 24))
                    }
                    .padding(6)
                    .background(Color.cyan.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                if let lig = viewModel.molecules.ligand {
                    HStack(spacing: 6) {
                        Circle().fill(.green).frame(width: 8, height: 8)
                        Text(lig.name)
                            .font(.system(size: 11, weight: .medium))
                        Text("\(lig.atomCount) atoms")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.removeLigandFromView() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 14))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .contentShape(Rectangle().size(width: 24, height: 24))
                    }
                    .padding(6)
                    .background(Color.green.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }

            Divider()

            // Quick fetch by PDB ID
            VStack(alignment: .leading, spacing: 6) {
                Label("Fetch by PDB ID", systemImage: "arrow.down.doc")
                    .font(.system(size: 12, weight: .semibold))

                HStack(spacing: 6) {
                    TextField("e.g. 1HSG", text: $pdbID)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, design: .monospaced))
                        .onSubmit { fetchPDB() }

                    Button("Fetch") { fetchPDB() }
                        .controlSize(.small)
                        .disabled(pdbID.trimmingCharacters(in: .whitespaces).count < 4 || viewModel.workspace.isLoading)
                }
            }

            Divider()

            // Keyword search
            VStack(alignment: .leading, spacing: 6) {
                Label("Search RCSB", systemImage: "magnifyingglass")
                    .font(.system(size: 12, weight: .semibold))

                HStack(spacing: 6) {
                    TextField("e.g. HIV protease", text: $searchQuery)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))
                        .onSubmit { viewModel.searchPDB(query: searchQuery) }

                    Button("Search") { viewModel.searchPDB(query: searchQuery) }
                        .controlSize(.small)
                        .disabled(searchQuery.isEmpty || viewModel.workspace.isSearching)
                }
            }

            // Loading indicators
            if viewModel.workspace.isLoading {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(viewModel.workspace.loadingMessage)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            if viewModel.workspace.isSearching {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Searching...")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            // Search results as cards
            if !viewModel.workspace.searchResults.isEmpty {
                Divider()

                Text("\(viewModel.workspace.searchResults.count) results")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)

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
        VStack(alignment: .leading, spacing: 6) {
            // Header: PDB ID + resolution
            HStack(alignment: .firstTextBaseline) {
                Text(result.id)
                    .font(.system(size: 16, weight: .bold, design: .monospaced))

                Spacer()

                if let res = result.resolution {
                    HStack(spacing: 2) {
                        Text(String(format: "%.1f", res))
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        Text("\u{00C5}")
                            .font(.system(size: 10))
                    }
                    .foregroundStyle(.secondary)
                }
            }

            // Title
            Text(result.title)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .lineLimit(3)

            // Method + date
            HStack(spacing: 8) {
                if let method = result.experimentMethod {
                    Text(method)
                        .font(.system(size: 10, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.12))
                        .clipShape(Capsule())
                }

                if let date = result.releaseDate {
                    Text(date)
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
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
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
        )
    }
}
