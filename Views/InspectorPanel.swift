import SwiftUI

struct InspectorPanel: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Atom Inspector
                if let atom = viewModel.selectedAtom {
                    atomInspector(atom)
                } else {
                    ContentUnavailableView {
                        Label("No Selection", systemImage: "atom")
                    } description: {
                        Text("Click an atom to inspect it")
                    }
                    .frame(height: 120)
                }

                Divider()

                // Chain Visibility
                chainVisibilitySection

                Divider()

                // Residue Subsets (MOE-style)
                residueSubsetsSection

                Divider()

                // Molecule Statistics
                statisticsSection
            }
            .padding(12)
        }
        .frame(width: 260)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }

    // MARK: - Atom Inspector

    @ViewBuilder
    private func atomInspector(_ atom: Atom) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Atom Inspector", icon: "atom")

            HStack(spacing: 10) {
                // Element badge
                ZStack {
                    Circle()
                        .fill(Color(
                            red: Double(atom.element.color.x),
                            green: Double(atom.element.color.y),
                            blue: Double(atom.element.color.z)
                        ))
                        .frame(width: 36, height: 36)
                    Text(atom.element.symbol)
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)
                        .shadow(radius: 1)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(atom.element.name)
                        .font(.system(size: 13, weight: .semibold))
                    Text(atom.name)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            infoRow("Residue", "\(atom.residueName) \(atom.residueSeq)")
            infoRow("Chain", atom.chainID)
            infoRow("Charge", String(format: "%.3f", atom.charge))
            infoRow("VdW Radius", String(format: "%.2f \u{00C5}", atom.element.vdwRadius))

            // Coordinates
            VStack(alignment: .leading, spacing: 3) {
                Text("Position")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    coordLabel("X", atom.position.x)
                    coordLabel("Y", atom.position.y)
                    coordLabel("Z", atom.position.z)
                }
            }
        }
    }

    // MARK: - Chain Visibility

    @ViewBuilder
    private var chainVisibilitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Chains", icon: "link")

            if viewModel.allChains.isEmpty && viewModel.ligand == nil {
                Text("No chains loaded")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(viewModel.allChains) { chain in
                    let atomCount = chainAtomCount(chain)
                    HStack(spacing: 6) {
                        Circle()
                            .fill(chain.displayColor)
                            .frame(width: 8, height: 8)

                        VStack(alignment: .leading, spacing: 1) {
                            Text("Chain \(chain.id)")
                                .font(.system(size: 12, weight: .medium))
                            HStack(spacing: 4) {
                                Text(chain.type == .protein ? "Protein" : chain.type == .ligand ? "Ligand" : "Other")
                                    .font(.system(size: 9))
                                    .foregroundStyle(.secondary)
                                Text("\(atomCount)")
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.tertiary)
                            }
                        }

                        Spacer()

                        // Show Only button (hides all other chains)
                        Button(action: { showOnlyChain(chain.id) }) {
                            Image(systemName: "eye")
                                .font(.system(size: 9))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.tertiary)
                        .help("Show only this chain")

                        Toggle("", isOn: Binding(
                            get: { !viewModel.hiddenChainIDs.contains(chain.id) },
                            set: { _ in viewModel.toggleChainVisibility(chain.id) }
                        ))
                        .toggleStyle(.switch)
                        .controlSize(.mini)
                    }
                }

                if !viewModel.hiddenChainIDs.isEmpty {
                    Text("Hidden chains excluded from pocket detection")
                        .font(.system(size: 9))
                        .foregroundStyle(.orange.opacity(0.7))
                }

                // Ligand visibility toggle (independent of chain toggles)
                if viewModel.ligand != nil {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.orange)
                            .frame(width: 8, height: 8)

                        Text("Ligand")
                            .font(.system(size: 12, weight: .medium))

                        Spacer()

                        Text(viewModel.ligand?.name ?? "")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)

                        Toggle("", isOn: Binding(
                            get: { viewModel.showLigand },
                            set: { newValue in
                                viewModel.showLigand = newValue
                                viewModel.pushToRenderer()
                            }
                        ))
                        .toggleStyle(.switch)
                        .controlSize(.mini)
                    }
                }
            }
        }
    }

    // MARK: - Residue Subsets

    @ViewBuilder
    private var residueSubsetsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionHeader("Subsets", icon: "square.stack.3d.up")
                Spacer()
                Button(action: { viewModel.createSubsetFromSelection() }) {
                    Image(systemName: "plus.circle")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain)
                .disabled(viewModel.selectedResidueIndices.isEmpty)
                .help("Create subset from selected residues")
            }

            if viewModel.residueSubsets.isEmpty {
                Text("Select residues and click + to create a subset")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(viewModel.residueSubsets) { subset in
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color(
                                red: Double(subset.color.x),
                                green: Double(subset.color.y),
                                blue: Double(subset.color.z)
                            ))
                            .frame(width: 8, height: 8)

                        Text(subset.name)
                            .font(.system(size: 11, weight: .medium))
                            .lineLimit(1)

                        Text("\(subset.residueIndices.count)")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)

                        Spacer()

                        // Select button
                        Button(action: { viewModel.selectSubset(id: subset.id) }) {
                            Image(systemName: "selection.pin.in.out")
                                .font(.system(size: 10))
                        }
                        .buttonStyle(.plain)
                        .help("Select all residues in this subset")

                        // Define pocket
                        Button(action: { viewModel.definePocketFromSubset(id: subset.id) }) {
                            Image(systemName: "cube.transparent")
                                .font(.system(size: 10))
                        }
                        .buttonStyle(.plain)
                        .help("Define docking pocket from this subset")

                        // Toggle visibility
                        Toggle("", isOn: Binding(
                            get: { subset.isVisible },
                            set: { _ in viewModel.toggleSubsetVisibility(id: subset.id) }
                        ))
                        .toggleStyle(.switch)
                        .controlSize(.mini)

                        // Delete
                        Button(action: { viewModel.deleteSubset(id: subset.id) }) {
                            Image(systemName: "trash")
                                .font(.system(size: 10))
                                .foregroundStyle(.red)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: - Statistics

    @ViewBuilder
    private var statisticsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Statistics", icon: "chart.bar")

            if let prot = viewModel.protein {
                Text("Protein: \(prot.name)")
                    .font(.system(size: 11, weight: .medium))
                infoRow("Atoms", "\(prot.atomCount)")
                infoRow("Heavy", "\(prot.heavyAtomCount)")
                infoRow("Bonds", "\(prot.bondCount)")
                infoRow("Residues", "\(prot.residues.count)")
                infoRow("Chains", "\(prot.chains.count)")
                infoRow("MW", String(format: "%.1f Da", prot.molecularWeight))
            }

            if let lig = viewModel.ligand {
                if viewModel.protein != nil { Divider() }
                Text("Ligand: \(lig.name)")
                    .font(.system(size: 11, weight: .medium))
                infoRow("Atoms", "\(lig.atomCount)")
                infoRow("Heavy", "\(lig.heavyAtomCount)")
                infoRow("Bonds", "\(lig.bondCount)")
                infoRow("MW", String(format: "%.1f Da", lig.molecularWeight))
            }

            if viewModel.protein == nil && viewModel.ligand == nil {
                Text("No molecules loaded")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Helpers

    private func sectionHeader(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(.primary)
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .leading)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.primary)
            Spacer()
        }
    }

    private func coordLabel(_ axis: String, _ value: Float) -> some View {
        HStack(spacing: 2) {
            Text(axis)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.tertiary)
            Text(String(format: "%7.3f", value))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.primary)
        }
    }

    private func chainAtomCount(_ chain: Chain) -> Int {
        guard let mol = viewModel.protein else { return 0 }
        return chain.residueIndices.reduce(0) { sum, resIdx in
            guard resIdx < mol.residues.count else { return sum }
            return sum + mol.residues[resIdx].atomIndices.count
        }
    }

    private func showOnlyChain(_ chainID: String) {
        let allIDs = Set(viewModel.allChains.map(\.id))
        viewModel.hiddenChainIDs = allIDs.subtracting([chainID])
        viewModel.pushToRenderer()
    }
}
