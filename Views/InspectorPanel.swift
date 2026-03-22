import SwiftUI

struct InspectorPanel: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var selectionMode: SelectionMode = .atom

    private enum SelectionMode: String, CaseIterable {
        case atom = "Atom"
        case residue = "Residue"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Selection mode toggle + inspector
                selectionSection

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

    // MARK: - Selection Section

    @ViewBuilder
    private var selectionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Mode toggle
            HStack {
                sectionHeader("Selection", icon: "cursorarrow.click.2")
                Spacer()
                Picker("", selection: $selectionMode) {
                    ForEach(SelectionMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 130)
            }

            // Content based on selection state
            if selectionMode == .atom {
                atomSelectionContent
            } else {
                residueSelectionContent
            }
        }
    }

    // MARK: - Atom Selection Content

    @ViewBuilder
    private var atomSelectionContent: some View {
        let selectedCount = viewModel.selectedAtomIndices.count

        if selectedCount == 0 {
            ContentUnavailableView {
                Label("No Selection", systemImage: "atom")
            } description: {
                Text("Click an atom to inspect it")
            }
            .frame(height: 100)
        } else if selectedCount == 1, let atom = viewModel.selectedAtom {
            // Single atom detail
            atomInspector(atom)
        } else {
            // Multi-atom summary
            multiAtomSummary(count: selectedCount)
        }
    }

    // MARK: - Residue Selection Content

    @ViewBuilder
    private var residueSelectionContent: some View {
        let selectedCount = viewModel.selectedResidueIndices.count

        if selectedCount == 0 {
            ContentUnavailableView {
                Label("No Selection", systemImage: "rectangle.stack")
            } description: {
                Text("Click a residue to inspect it")
            }
            .frame(height: 100)
        } else if selectedCount == 1, let prot = viewModel.protein {
            // Single residue detail
            let resIdx = viewModel.selectedResidueIndices.first!
            if resIdx < prot.residues.count {
                singleResidueInspector(prot.residues[resIdx], in: prot)
            }
        } else {
            // Multi-residue summary
            multiResidueSummary(count: selectedCount)
        }
    }

    // MARK: - Single Atom Inspector

    @ViewBuilder
    private func atomInspector(_ atom: Atom) -> some View {
        HStack(spacing: 10) {
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

    // MARK: - Multi-Atom Summary

    @ViewBuilder
    private func multiAtomSummary(count: Int) -> some View {
        let allAtoms = (viewModel.protein?.atoms ?? []) + (viewModel.ligand?.atoms ?? [])
        let selected = viewModel.selectedAtomIndices.compactMap { idx in
            idx < allAtoms.count ? allAtoms[idx] : nil
        }

        // Element distribution
        let elementCounts = Dictionary(grouping: selected, by: \.element.symbol)
            .mapValues(\.count)
            .sorted { $0.value > $1.value }

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "atom")
                    .font(.system(size: 14))
                    .foregroundStyle(.cyan)
                Text("\(count) atoms selected")
                    .font(.system(size: 13, weight: .semibold))
            }

            // Element breakdown
            HStack(spacing: 4) {
                ForEach(elementCounts.prefix(6), id: \.key) { symbol, cnt in
                    Text("\(symbol):\(cnt)")
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.primary.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
            }

            // Unique residues & chains
            let residues = Set(selected.map { "\($0.residueName)\($0.residueSeq)" })
            let chains = Set(selected.map(\.chainID))
            infoRow("Residues", "\(residues.count)")
            infoRow("Chains", chains.sorted().joined(separator: ", "))

            // Average charge
            if !selected.isEmpty {
                let avgCharge = selected.map(\.charge).reduce(0, +) / Float(selected.count)
                let totalCharge = selected.map(\.charge).reduce(0, +)
                infoRow("Avg charge", String(format: "%.3f", avgCharge))
                infoRow("Total charge", String(format: "%.2f", totalCharge))
            }

            // Clear selection
            Button(action: {
                viewModel.selectedAtomIndices.removeAll()
                viewModel.selectedAtomIndex = nil
                viewModel.pushToRenderer()
            }) {
                Label("Clear Selection", systemImage: "xmark.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    // MARK: - Single Residue Inspector

    @ViewBuilder
    private func singleResidueInspector(_ residue: Residue, in prot: Molecule) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(residue.name)
                    .font(.system(size: 16, weight: .bold, design: .monospaced))
                Text("#\(residue.sequenceNumber)")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("Chain \(residue.chainID)")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            infoRow("Atoms", "\(residue.atomIndices.count)")
            let heavyCount = residue.atomIndices.filter { idx in
                idx < prot.atoms.count && prot.atoms[idx].element != .H
            }.count
            infoRow("Heavy atoms", "\(heavyCount)")

            // Total charge of residue
            let totalCharge = residue.atomIndices.compactMap { idx in
                idx < prot.atoms.count ? prot.atoms[idx].charge : nil
            }.reduce(Float(0), +)
            infoRow("Net charge", String(format: "%.2f", totalCharge))
        }
    }

    // MARK: - Multi-Residue Summary

    @ViewBuilder
    private func multiResidueSummary(count: Int) -> some View {
        if let prot = viewModel.protein {
            let residues = viewModel.selectedResidueIndices.compactMap { idx in
                idx < prot.residues.count ? prot.residues[idx] : nil
            }

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "rectangle.stack")
                    .font(.system(size: 14))
                    .foregroundStyle(.cyan)
                Text("\(count) residues selected")
                    .font(.system(size: 13, weight: .semibold))
            }

            // Residue type breakdown
            let typeCounts = Dictionary(grouping: residues, by: \.name)
                .mapValues(\.count)
                .sorted { $0.value > $1.value }

            HStack(spacing: 4) {
                ForEach(typeCounts.prefix(8), id: \.key) { name, cnt in
                    Text("\(name):\(cnt)")
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.primary.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
            }

            // Total atoms
            let totalAtoms = residues.reduce(0) { $0 + $1.atomIndices.count }
            let chains = Set(residues.map(\.chainID))
            infoRow("Total atoms", "\(totalAtoms)")
            infoRow("Chains", chains.sorted().joined(separator: ", "))

            // Sequence range
            let seqNums = residues.map(\.sequenceNumber).sorted()
            if let first = seqNums.first, let last = seqNums.last {
                infoRow("Range", "\(first)\u{2013}\(last)")
            }

            // Actions
            HStack(spacing: 6) {
                Button(action: { viewModel.createSubsetFromSelection() }) {
                    Label("Create Subset", systemImage: "plus.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(action: {
                    viewModel.selectedResidueIndices.removeAll()
                    viewModel.selectedAtomIndices.removeAll()
                    viewModel.selectedAtomIndex = nil
                    viewModel.pushToRenderer()
                }) {
                    Label("Clear", systemImage: "xmark.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
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
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(chain.displayColor)
                                .frame(width: 8, height: 8)

                            VStack(alignment: .leading, spacing: 1) {
                                Text("Chain \(chain.id)")
                                    .font(.system(size: 12, weight: .medium))
                                HStack(spacing: 4) {
                                    Text(chain.type == .protein ? "Protein" : chain.type == .ligand ? "Ligand" : "Other")
                                        .font(.system(size: 10))
                                        .foregroundStyle(.secondary)
                                    Text("\(atomCount) atoms")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.tertiary)
                                }
                            }

                            Spacer()

                            // Convert non-protein chains to active ligand
                            if chain.type != .protein {
                                Button(action: { viewModel.extractChainAsLigand(chainID: chain.id) }) {
                                    Image(systemName: "arrow.right.circle")
                                        .font(.system(size: 11))
                                        .padding(3)
                                        .background(Color.orange.opacity(0.1))
                                        .clipShape(RoundedRectangle(cornerRadius: 4))
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.orange)
                                .help("Extract as ligand (adds to database)")
                            }

                            Button(action: { viewModel.removeChain(chainID: chain.id) }) {
                                Image(systemName: "trash")
                                    .font(.system(size: 10))
                                    .padding(3)
                                    .background(Color.red.opacity(0.05))
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.red.opacity(0.6))
                            .help("Remove chain from structure")

                            Button(action: { showOnlyChain(chain.id) }) {
                                Image(systemName: "eye")
                                    .font(.system(size: 11))
                                    .padding(3)
                                    .background(Color.primary.opacity(0.05))
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .help("Show only this chain")

                            Toggle("", isOn: Binding(
                                get: { !viewModel.hiddenChainIDs.contains(chain.id) },
                                set: { _ in viewModel.toggleChainVisibility(chain.id) }
                            ))
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                        }
                    }
                }

                if !viewModel.hiddenChainIDs.isEmpty {
                    Text("Hidden chains excluded from pocket detection")
                        .font(.system(size: 10))
                        .foregroundStyle(.orange.opacity(0.8))
                        .padding(.top, 2)
                }

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

                        Button(action: { viewModel.definePocketFromLigand() }) {
                            Image(systemName: "scope")
                                .font(.system(size: 11))
                                .padding(3)
                                .background(Color.green.opacity(0.1))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.green)
                        .help("Define docking pocket from ligand position")

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
                        .font(.system(size: 13))
                        .padding(3)
                        .background(Color.accentColor.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 5))
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
                        Button(action: { viewModel.selectSubset(id: subset.id) }) {
                            Image(systemName: "selection.pin.in.out")
                                .font(.system(size: 11))
                                .padding(3)
                                .background(Color.primary.opacity(0.05))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        .buttonStyle(.plain)
                        .help("Select all residues in this subset")
                        Button(action: { viewModel.definePocketFromSubset(id: subset.id) }) {
                            Image(systemName: "cube.transparent")
                                .font(.system(size: 11))
                                .padding(3)
                                .background(Color.primary.opacity(0.05))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        .buttonStyle(.plain)
                        .help("Define docking pocket from this subset")
                        Toggle("", isOn: Binding(
                            get: { subset.isVisible },
                            set: { _ in viewModel.toggleSubsetVisibility(id: subset.id) }
                        ))
                        .toggleStyle(.switch)
                        .controlSize(.mini)
                        Button(action: { viewModel.deleteSubset(id: subset.id) }) {
                            Image(systemName: "trash")
                                .font(.system(size: 11))
                                .foregroundStyle(.red)
                                .padding(3)
                                .background(Color.red.opacity(0.06))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
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
