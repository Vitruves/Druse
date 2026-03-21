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
                if let prot = viewModel.protein {
                    HStack {
                        Circle().fill(.cyan).frame(width: 6, height: 6)
                        Text(prot.name)
                            .font(.system(size: 10, weight: .medium))
                        Text("\(prot.atomCount) atoms")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.protein = nil; viewModel.pushToRenderer() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                if let lig = viewModel.ligand {
                    HStack {
                        Circle().fill(.green).frame(width: 6, height: 6)
                        Text(lig.name)
                            .font(.system(size: 10, weight: .medium))
                        Text("\(lig.atomCount) atoms")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.clearLigand() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
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
                        .disabled(pdbID.trimmingCharacters(in: .whitespaces).count < 4 || viewModel.isLoading)
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
                        .disabled(searchQuery.isEmpty || viewModel.isSearching)
                }
            }

            // Loading indicators
            if viewModel.isLoading {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(viewModel.loadingMessage)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            if viewModel.isSearching {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Searching...")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            // Search results as cards
            if !viewModel.searchResults.isEmpty {
                Divider()

                Text("\(viewModel.searchResults.count) results")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)

                ForEach(viewModel.searchResults) { result in
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
                        .font(.system(size: 9, weight: .medium))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.12))
                        .clipShape(Capsule())
                }

                if let date = result.releaseDate {
                    Text(date)
                        .font(.system(size: 9))
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
            .disabled(viewModel.isLoading)
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
