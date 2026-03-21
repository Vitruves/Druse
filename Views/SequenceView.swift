import SwiftUI
import AppKit

/// One-letter amino acid code lookup.
private let threeToOne: [String: String] = [
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
]

/// Secondary structure color mapping.
private func ssColor(_ ss: SecondaryStructure) -> Color {
    switch ss {
    case .helix: .red.opacity(0.3)
    case .sheet: .yellow.opacity(0.3)
    case .coil:  .gray.opacity(0.15)
    case .turn:  .cyan.opacity(0.2)
    }
}

/// SS legend label text.
private func ssLabel(_ ss: SecondaryStructure) -> String {
    switch ss {
    case .helix: "Helix"
    case .sheet: "Sheet"
    case .coil:  "Coil"
    case .turn:  "Turn"
    }
}

// MARK: - Sequence Entry

/// Represents a single cell in the sequence display: either a residue or a gap marker.
private enum SequenceCell: Identifiable {
    case residue(ResidueEntry)
    case gap(id: String, missingCount: Int, afterSeqNum: Int, beforeSeqNum: Int)

    var id: String {
        switch self {
        case .residue(let r): "res_\(r.resIdx)"
        case .gap(let id, _, _, _): id
        }
    }
}

private struct ResidueEntry {
    let resIdx: Int
    let name: String
    let oneLetterCode: String
    let seqNum: Int
    let chainID: String
    let ss: SecondaryStructure
}

// MARK: - SequenceView

struct SequenceView: View {
    @Environment(AppViewModel.self) private var viewModel
    @State private var lastClickedResIdx: Int?
    @State private var hoveredResIdx: Int?
    @State private var collapsedChains: Set<String> = []

    // Rename chain sheet
    @State private var showRenameSheet = false
    @State private var renameChainID: String = ""
    @State private var renameNewID: String = ""

    // Merge chain sheet
    @State private var showMergeSheet = false
    @State private var mergeSourceID: String = ""
    @State private var mergeTargetID: String = ""

    // Delete confirmation
    @State private var showDeleteConfirm = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if viewModel.protein == nil {
                ContentUnavailableView {
                    Label("No Protein", systemImage: "textformat.abc")
                } description: {
                    Text("Load a protein to view its sequence")
                }
            } else {
                sequenceContent
            }
        }
        .padding(12)
        .sheet(isPresented: $showRenameSheet) { renameSheet }
        .sheet(isPresented: $showMergeSheet) { mergeSheet }
        .alert("Delete \(viewModel.selectedResidueIndices.count) residues?",
               isPresented: $showDeleteConfirm) {
            Button("Delete", role: .destructive) { viewModel.deleteSelectedResidues() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will remove the selected residues and their atoms from the structure. This cannot be undone.")
        }
    }

    // MARK: - Main Content

    @ViewBuilder
    private var sequenceContent: some View {
        if let prot = viewModel.protein {
            // Header
            HStack {
                Label("Sequence", systemImage: "textformat.abc")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                HStack(spacing: 6) {
                    ssLegendDot(.helix)
                    ssLegendDot(.sheet)
                    ssLegendDot(.coil)
                }
            }

            // Toolbar
            toolbar(prot)

            // Selection info
            if !viewModel.selectedResidueIndices.isEmpty {
                HStack(spacing: 4) {
                    Image(systemName: "scope")
                        .font(.system(size: 9))
                        .foregroundStyle(.cyan)
                    Text("\(viewModel.selectedResidueIndices.count) selected")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.cyan)
                    Spacer()
                    Button {
                        viewModel.selectedResidueIndices.removeAll()
                        viewModel.selectedAtomIndices.removeAll()
                        viewModel.selectedAtomIndex = nil
                        viewModel.pushToRenderer()
                    } label: {
                        Text("Clear")
                            .font(.system(size: 9))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
            }

            // Hover info
            if let hIdx = hoveredResIdx, let prot = viewModel.protein, hIdx < prot.residues.count {
                let res = prot.residues[hIdx]
                let ss = secondaryStructure(for: res, in: prot)
                HStack(spacing: 4) {
                    Text(res.name)
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                    Text("#\(res.sequenceNumber)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text("Chain \(res.chainID)")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                    Text(ssLabel(ss))
                        .font(.system(size: 9))
                        .foregroundStyle(ssColor(ss))
                }
                .transition(.opacity)
            }

            // Chain sections
            let proteinChains = prot.chains.filter { $0.type == .protein }
            if proteinChains.isEmpty {
                Text("No protein chains found")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            } else {
                ScrollView(.vertical, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(proteinChains, id: \.id) { chain in
                            chainSection(chain, prot: prot)
                        }
                    }
                }
                .frame(maxHeight: .infinity)
            }

            Divider()
            selectedResidueInfo(prot)
        }
    }

    // MARK: - Toolbar

    @ViewBuilder
    private func toolbar(_ prot: Molecule) -> some View {
        HStack(spacing: 4) {
            // Copy sequence
            Button {
                viewModel.copySequenceToClipboard()
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy sequence to clipboard")
            .controlSize(.small)

            // Select by SS menu
            Menu {
                Button("All Helices") { viewModel.selectBySecondaryStructure(.helix) }
                Button("All Sheets") { viewModel.selectBySecondaryStructure(.sheet) }
                Button("All Coils") { viewModel.selectBySecondaryStructure(.coil) }
                Button("All Turns") { viewModel.selectBySecondaryStructure(.turn) }
            } label: {
                Image(systemName: "line.3.horizontal.decrease.circle")
            }
            .menuStyle(.borderlessButton)
            .help("Select by secondary structure")
            .controlSize(.small)
            .frame(width: 20)

            Spacer()

            Text("\(prot.residues.filter(\.isStandard).count) residues")
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Chain Section

    @ViewBuilder
    private func chainSection(_ chain: Chain, prot: Molecule) -> some View {
        let isCollapsed = collapsedChains.contains(chain.id)
        let residueEntries = chainResidues(chain, prot: prot)
        let cells = buildCells(from: residueEntries, chainID: chain.id)

        VStack(alignment: .leading, spacing: 3) {
            // Chain header
            HStack(spacing: 4) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        if isCollapsed {
                            collapsedChains.remove(chain.id)
                        } else {
                            collapsedChains.insert(chain.id)
                        }
                    }
                } label: {
                    Image(systemName: isCollapsed ? "chevron.right" : "chevron.down")
                        .font(.system(size: 8))
                        .frame(width: 10)
                }
                .buttonStyle(.plain)

                Text("Chain \(chain.id)")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(chain.displayColor)

                Text("\(residueEntries.count) res")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)

                Spacer()

                // Chain context menu
                Menu {
                    Button("Select Chain") {
                        viewModel.selectChain(chain.id)
                    }
                    Button("Copy Sequence") {
                        viewModel.copySequenceToClipboard(chainID: chain.id)
                    }
                    Divider()
                    Button("Rename Chain...") {
                        renameChainID = chain.id
                        renameNewID = chain.id
                        showRenameSheet = true
                    }
                    if prot.chains.filter({ $0.type == .protein }).count > 1 {
                        Menu("Merge Into...") {
                            ForEach(prot.chains.filter({ $0.type == .protein && $0.id != chain.id }), id: \.id) { target in
                                Button("Chain \(target.id)") {
                                    mergeSourceID = chain.id
                                    mergeTargetID = target.id
                                    showMergeSheet = true
                                }
                            }
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .menuStyle(.borderlessButton)
                .frame(width: 16)
            }
            .padding(.horizontal, 2)

            if !isCollapsed {
                // Sequence grid with gaps
                FlowLayout(spacing: 1) {
                    ForEach(cells) { cell in
                        switch cell {
                        case .residue(let entry):
                            residueCell(entry)
                                .contextMenu { residueContextMenu(entry, chainID: chain.id, prot: prot) }
                        case .gap(_, let count, let after, let before):
                            gapCell(count: count, after: after, before: before)
                        }
                    }
                }
            }
        }
        .padding(.vertical, 2)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(.quaternary.opacity(0.3))
        )
    }

    // MARK: - Gap Detection

    /// Build sequence cells for a chain, inserting gap markers where residue numbers are discontinuous.
    private func buildCells(from entries: [ResidueEntry], chainID: String) -> [SequenceCell] {
        guard !entries.isEmpty else { return [] }
        var cells: [SequenceCell] = []

        for (i, entry) in entries.enumerated() {
            if i > 0 {
                let prev = entries[i - 1]
                let expectedNext = prev.seqNum + 1
                if entry.seqNum > expectedNext {
                    let missingCount = entry.seqNum - expectedNext
                    cells.append(.gap(
                        id: "gap_\(chainID)_\(prev.seqNum)_\(entry.seqNum)",
                        missingCount: missingCount,
                        afterSeqNum: prev.seqNum,
                        beforeSeqNum: entry.seqNum
                    ))
                }
            }
            cells.append(.residue(entry))
        }
        return cells
    }

    /// Build ResidueEntry list for a chain (standard residues only, sorted by sequence number).
    private func chainResidues(_ chain: Chain, prot: Molecule) -> [ResidueEntry] {
        chain.residueIndices.compactMap { idx -> ResidueEntry? in
            guard idx < prot.residues.count else { return nil }
            let res = prot.residues[idx]
            guard res.isStandard else { return nil }
            let code = threeToOne[res.name] ?? "?"
            let ss = secondaryStructure(for: res, in: prot)
            return ResidueEntry(
                resIdx: idx, name: res.name, oneLetterCode: code,
                seqNum: res.sequenceNumber, chainID: res.chainID, ss: ss
            )
        }
        .sorted { $0.seqNum < $1.seqNum }
    }

    private func secondaryStructure(for res: Residue, in prot: Molecule) -> SecondaryStructure {
        for ssa in prot.secondaryStructureAssignments {
            if ssa.chain == res.chainID && res.sequenceNumber >= ssa.start && res.sequenceNumber <= ssa.end {
                return ssa.type
            }
        }
        return .coil
    }

    // MARK: - Gap Cell

    @ViewBuilder
    private func gapCell(count: Int, after: Int, before: Int) -> some View {
        HStack(spacing: 0) {
            Text("···\(count)···")
                .font(.system(size: 8, weight: .medium, design: .monospaced))
                .foregroundStyle(.orange.opacity(0.8))
        }
        .padding(.horizontal, 2)
        .frame(height: 16)
        .background(
            RoundedRectangle(cornerRadius: 2)
                .fill(.orange.opacity(0.1))
                .strokeBorder(.orange.opacity(0.3), lineWidth: 0.5)
        )
        .help("Gap: \(count) missing residues (\(after + 1)–\(before - 1))")
    }

    // MARK: - Residue Cell

    @ViewBuilder
    private func residueCell(_ entry: ResidueEntry) -> some View {
        let isSelected = viewModel.selectedResidueIndices.contains(entry.resIdx)
        let isHovered = hoveredResIdx == entry.resIdx

        // Show residue number every 10 residues
        let showNumber = entry.seqNum % 10 == 0

        VStack(spacing: 0) {
            if showNumber {
                Text("\(entry.seqNum)")
                    .font(.system(size: 6, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .frame(height: 8)
            }

            Text(entry.oneLetterCode)
                .font(.system(size: 10, weight: isSelected ? .bold : .medium, design: .monospaced))
                .frame(width: 14, height: 16)
                .background(cellBackground(isSelected: isSelected, isHovered: isHovered, ss: entry.ss))
                .border(cellBorder(isSelected: isSelected, isHovered: isHovered), width: 1)
                .foregroundStyle(isSelected ? .white : .secondary)
        }
        .help("\(entry.name) \(entry.seqNum) (Chain \(entry.chainID))")
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.1)) {
                hoveredResIdx = hovering ? entry.resIdx : nil
            }
        }
        .onTapGesture {
            handleResidueClick(entry)
        }
    }

    private func cellBackground(isSelected: Bool, isHovered: Bool, ss: SecondaryStructure) -> Color {
        if isSelected { return .cyan.opacity(0.6) }
        if isHovered { return ssColor(ss).opacity(0.5) }
        return ssColor(ss)
    }

    private func cellBorder(isSelected: Bool, isHovered: Bool) -> Color {
        if isSelected { return .cyan }
        if isHovered { return .white.opacity(0.3) }
        return .clear
    }

    // MARK: - Context Menu

    @ViewBuilder
    private func residueContextMenu(_ entry: ResidueEntry, chainID: String, prot: Molecule) -> some View {
        let isSelected = viewModel.selectedResidueIndices.contains(entry.resIdx)
        let selCount = viewModel.selectedResidueIndices.count

        if !isSelected {
            Button("Select Residue \(entry.name) \(entry.seqNum)") {
                viewModel.selectedResidueIndices = [entry.resIdx]
                syncAtomSelectionFromResidues()
                viewModel.pushToRenderer()
            }
        }

        Button("Select Chain \(chainID)") {
            viewModel.selectChain(chainID)
        }

        Divider()

        if selCount > 0 {
            Button("Delete \(selCount) Selected Residue\(selCount > 1 ? "s" : "")") {
                showDeleteConfirm = true
            }

            Button("Create Subset from Selection") {
                viewModel.createSubsetFromSelection()
            }

            Button("Define Pocket from Selection") {
                viewModel.definePocketFromSelection()
            }

            Divider()
        }

        Button("Copy Sequence") {
            viewModel.copySequenceToClipboard(chainID: chainID)
        }

        Menu("Select by Structure") {
            Button("Helices in Chain \(chainID)") {
                viewModel.selectBySecondaryStructure(.helix, chainID: chainID)
            }
            Button("Sheets in Chain \(chainID)") {
                viewModel.selectBySecondaryStructure(.sheet, chainID: chainID)
            }
            Button("Coils in Chain \(chainID)") {
                viewModel.selectBySecondaryStructure(.coil, chainID: chainID)
            }
        }

        Divider()

        Button("Rename Chain \(chainID)...") {
            renameChainID = chainID
            renameNewID = chainID
            showRenameSheet = true
        }

        if prot.chains.filter({ $0.type == .protein }).count > 1 {
            Menu("Merge Chain \(chainID) Into...") {
                ForEach(prot.chains.filter({ $0.type == .protein && $0.id != chainID }), id: \.id) { target in
                    Button("Chain \(target.id)") {
                        mergeSourceID = chainID
                        mergeTargetID = target.id
                        showMergeSheet = true
                    }
                }
            }
        }
    }

    // MARK: - Click Handling

    private func handleResidueClick(_ entry: ResidueEntry) {
        let modifiers = NSEvent.modifierFlags
        let isCmd = modifiers.contains(.command)
        let isShift = modifiers.contains(.shift)

        if isShift, let last = lastClickedResIdx {
            // Range select between last clicked and current
            guard let prot = viewModel.protein else { return }
            // Find all residues in the same chain between the two indices
            let lo = min(last, entry.resIdx)
            let hi = max(last, entry.resIdx)
            if !isCmd {
                // Shift without Cmd: replace selection with range
                viewModel.selectedResidueIndices.removeAll()
            }
            for i in lo...hi {
                guard i < prot.residues.count else { continue }
                viewModel.selectedResidueIndices.insert(i)
            }
        } else if isCmd {
            // Toggle individual residue
            if viewModel.selectedResidueIndices.contains(entry.resIdx) {
                viewModel.selectedResidueIndices.remove(entry.resIdx)
            } else {
                viewModel.selectedResidueIndices.insert(entry.resIdx)
            }
        } else {
            // Single select
            viewModel.selectedResidueIndices = [entry.resIdx]
        }

        lastClickedResIdx = entry.resIdx
        syncAtomSelectionFromResidues()
        viewModel.pushToRenderer()
    }

    /// Update selectedAtomIndices to match the current residue selection.
    private func syncAtomSelectionFromResidues() {
        guard let prot = viewModel.protein else { return }
        var atomIndices = Set<Int>()
        for resIdx in viewModel.selectedResidueIndices {
            guard resIdx < prot.residues.count else { continue }
            atomIndices.formUnion(prot.residues[resIdx].atomIndices)
        }
        viewModel.selectedAtomIndices = atomIndices
        viewModel.selectedAtomIndex = atomIndices.first
    }

    // MARK: - SS Legend Dot

    private func ssLegendDot(_ ss: SecondaryStructure) -> some View {
        HStack(spacing: 2) {
            Circle()
                .fill(ssColor(ss))
                .frame(width: 6, height: 6)
            Text(ssLabel(ss))
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Selected Residue Info

    @ViewBuilder
    private func selectedResidueInfo(_ prot: Molecule) -> some View {
        let sortedSelected = viewModel.selectedResidueIndices.sorted()
        if sortedSelected.isEmpty {
            EmptyView()
        } else if sortedSelected.count == 1, let first = sortedSelected.first, first < prot.residues.count {
            let res = prot.residues[first]
            VStack(alignment: .leading, spacing: 4) {
                Label("Selected", systemImage: "scope")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                HStack {
                    Text("\(res.name) \(res.sequenceNumber)")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                    Text("Chain \(res.chainID)")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                Text("\(res.atomIndices.count) atoms")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
        } else {
            VStack(alignment: .leading, spacing: 4) {
                Label("Selected (\(sortedSelected.count) residues)", systemImage: "scope")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                let ranges = condenseRanges(sortedSelected, prot: prot)
                Text(ranges)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(3)

                let totalAtoms = sortedSelected.reduce(0) { sum, idx in
                    idx < prot.residues.count ? sum + prot.residues[idx].atomIndices.count : sum
                }
                Text("\(totalAtoms) atoms total")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private func condenseRanges(_ indices: [Int], prot: Molecule) -> String {
        guard !indices.isEmpty else { return "" }
        var parts: [String] = []
        var rangeStart = indices[0]
        var rangePrev = indices[0]
        for i in 1..<indices.count {
            if indices[i] == rangePrev + 1 {
                rangePrev = indices[i]
            } else {
                parts.append(rangeString(rangeStart, rangePrev, prot: prot))
                rangeStart = indices[i]
                rangePrev = indices[i]
            }
        }
        parts.append(rangeString(rangeStart, rangePrev, prot: prot))
        return parts.joined(separator: ", ")
    }

    private func rangeString(_ start: Int, _ end: Int, prot: Molecule) -> String {
        let startSeq = start < prot.residues.count ? prot.residues[start].sequenceNumber : start
        let endSeq = end < prot.residues.count ? prot.residues[end].sequenceNumber : end
        return startSeq == endSeq ? "\(startSeq)" : "\(startSeq)-\(endSeq)"
    }

    // MARK: - Rename Sheet

    private var renameSheet: some View {
        VStack(spacing: 12) {
            Text("Rename Chain \(renameChainID)")
                .font(.headline)

            TextField("New chain ID", text: $renameNewID)
                .textFieldStyle(.roundedBorder)
                .frame(width: 120)

            HStack {
                Button("Cancel") { showRenameSheet = false }
                    .keyboardShortcut(.cancelAction)
                Button("Rename") {
                    viewModel.renameChain(from: renameChainID, to: renameNewID)
                    showRenameSheet = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(renameNewID.isEmpty || renameNewID == renameChainID)
            }
        }
        .padding(20)
        .frame(width: 250)
    }

    // MARK: - Merge Sheet

    private var mergeSheet: some View {
        VStack(spacing: 12) {
            Text("Merge Chain \(mergeSourceID) into Chain \(mergeTargetID)?")
                .font(.headline)
                .multilineTextAlignment(.center)

            Text("All residues from chain \(mergeSourceID) will be reassigned to chain \(mergeTargetID). This cannot be undone.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            HStack {
                Button("Cancel") { showMergeSheet = false }
                    .keyboardShortcut(.cancelAction)
                Button("Merge") {
                    viewModel.mergeChains(from: mergeSourceID, into: mergeTargetID)
                    showMergeSheet = false
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 300)
    }
}

// MARK: - Flow Layout (wrapping grid)

struct FlowLayout: Layout {
    var spacing: CGFloat = 2

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = layout(in: proposal.width ?? 250, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = layout(in: bounds.width, subviews: subviews)
        for (index, pos) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + pos.x, y: bounds.minY + pos.y),
                                   proposal: .unspecified)
        }
    }

    private func layout(in width: CGFloat, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > width && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
        }

        return (CGSize(width: width, height: y + rowHeight), positions)
    }
}
