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

    /// Cached secondary structure lookup: (chainID, seqNum) → SecondaryStructure.
    /// Rebuilt when protein changes (via .id on the outer view).
    @State private var ssCache: [String: SecondaryStructure] = [:]
    /// Protein identity token to detect when cache must be rebuilt.
    @State private var cachedProteinID: UUID?

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
            if viewModel.molecules.protein == nil {
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
        .onAppear {
            if let prot = viewModel.molecules.protein, cachedProteinID != prot.id {
                rebuildSSCache(prot: prot)
            }
        }
        .onChange(of: viewModel.molecules.protein?.id) { _, newID in
            if let prot = viewModel.molecules.protein, let newID, cachedProteinID != newID {
                rebuildSSCache(prot: prot)
            }
        }
        .sheet(isPresented: $showRenameSheet) { renameSheet }
        .sheet(isPresented: $showMergeSheet) { mergeSheet }
        .alert("Delete \(viewModel.workspace.selectedResidueIndices.count) residues?",
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
        if let prot = viewModel.molecules.protein {
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
            if !viewModel.workspace.selectedResidueIndices.isEmpty {
                HStack(spacing: 4) {
                    Image(systemName: "scope")
                        .font(.system(size: 9))
                        .foregroundStyle(.cyan)
                    Text("\(viewModel.workspace.selectedResidueIndices.count) selected")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.cyan)
                    Spacer()
                    Button {
                        viewModel.workspace.selectedResidueIndices.removeAll()
                        viewModel.workspace.selectedAtomIndices.removeAll()
                        viewModel.workspace.selectedAtomIndex = nil
                        viewModel.pushToRenderer()
                    } label: {
                        Text("Clear")
                            .font(.system(size: 9))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
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
                        .font(.system(size: 10))
                        .frame(width: 12)
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
                // Canvas-based sequence rendering: draws all residues directly
                // instead of creating 1000+ individual SwiftUI views
                SequenceCanvas(
                    cells: cells,
                    selectedResidueIndices: viewModel.workspace.selectedResidueIndices,
                    hoveredResIdx: hoveredResIdx,
                    ssCache: ssCache,
                    onTapCell: { entry in handleResidueClick(entry) },
                    onHoverCell: { resIdx in hoveredResIdx = resIdx },
                    contextMenuBuilder: { entry in residueContextMenu(entry, chainID: chain.id, prot: prot) }
                )
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
            let ss = cachedSS(chainID: res.chainID, seqNum: res.sequenceNumber)
            return ResidueEntry(
                resIdx: idx, name: res.name, oneLetterCode: code,
                seqNum: res.sequenceNumber, chainID: res.chainID, ss: ss
            )
        }
        .sorted { $0.seqNum < $1.seqNum }
    }

    private func cachedSS(chainID: String, seqNum: Int) -> SecondaryStructure {
        ssCache["\(chainID)_\(seqNum)"] ?? .coil
    }

    /// Rebuild the secondary structure cache from scratch.
    private func rebuildSSCache(prot: Molecule) {
        var cache: [String: SecondaryStructure] = [:]
        for ssa in prot.secondaryStructureAssignments {
            for seq in ssa.start...ssa.end {
                cache["\(ssa.chain)_\(seq)"] = ssa.type
            }
        }
        ssCache = cache
        cachedProteinID = prot.id
    }

    // Residue cells are now rendered by SequenceCanvas (Canvas-based).

    // MARK: - Context Menu

    @ViewBuilder
    private func residueContextMenu(_ entry: ResidueEntry, chainID: String, prot: Molecule) -> some View {
        let isSelected = viewModel.workspace.selectedResidueIndices.contains(entry.resIdx)
        let selCount = viewModel.workspace.selectedResidueIndices.count

        if !isSelected {
            Button("Select Residue \(entry.name) \(entry.seqNum)") {
                viewModel.workspace.selectedResidueIndices = [entry.resIdx]
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

            Button("Add Docking Constraint\u{2026}") {
                viewModel.showConstraintSheetForResidue(entry.resIdx)
            }

            let isFlexible = viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.contains(entry.resIdx)
            Button(isFlexible ? "Remove Flexible Residue" : "Make Flexible (Induced Fit)") {
                if isFlexible {
                    viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll { $0 == entry.resIdx }
                } else if viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.count < FlexibleResidueConfig.maxFlexibleResidues {
                    if let prot = viewModel.molecules.protein,
                       let atom = prot.atoms.first(where: { $0.residueSeq == entry.resIdx }),
                       RotamerLibrary.rotamers(for: atom.residueName) != nil {
                        viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.append(entry.resIdx)
                    }
                }
            }
            .disabled(!isFlexible && viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.count >= FlexibleResidueConfig.maxFlexibleResidues)

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
            guard let prot = viewModel.molecules.protein else { return }
            // Find all residues in the same chain between the two indices
            let lo = min(last, entry.resIdx)
            let hi = max(last, entry.resIdx)
            if !isCmd {
                // Shift without Cmd: replace selection with range
                viewModel.workspace.selectedResidueIndices.removeAll()
            }
            for i in lo...hi {
                guard i < prot.residues.count else { continue }
                viewModel.workspace.selectedResidueIndices.insert(i)
            }
        } else if isCmd {
            // Toggle individual residue
            if viewModel.workspace.selectedResidueIndices.contains(entry.resIdx) {
                viewModel.workspace.selectedResidueIndices.remove(entry.resIdx)
            } else {
                viewModel.workspace.selectedResidueIndices.insert(entry.resIdx)
            }
        } else {
            // Single select
            viewModel.workspace.selectedResidueIndices = [entry.resIdx]
        }

        lastClickedResIdx = entry.resIdx
        syncAtomSelectionFromResidues()
        viewModel.pushToRenderer()
    }

    /// Update selectedAtomIndices to match the current residue selection.
    private func syncAtomSelectionFromResidues() {
        guard let prot = viewModel.molecules.protein else { return }
        var atomIndices = Set<Int>()
        for resIdx in viewModel.workspace.selectedResidueIndices {
            guard resIdx < prot.residues.count else { continue }
            atomIndices.formUnion(prot.residues[resIdx].atomIndices)
        }
        viewModel.workspace.selectedAtomIndices = atomIndices
        viewModel.workspace.selectedAtomIndex = atomIndices.first
    }

    // MARK: - SS Legend Dot

    private func ssLegendDot(_ ss: SecondaryStructure) -> some View {
        HStack(spacing: 3) {
            Circle()
                .fill(ssColor(ss))
                .frame(width: 7, height: 7)
            Text(ssLabel(ss))
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Selected Residue Info

    @ViewBuilder
    private func selectedResidueInfo(_ prot: Molecule) -> some View {
        let sortedSelected = viewModel.workspace.selectedResidueIndices.sorted()
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

// MARK: - SequenceCanvas (high-performance Canvas-based sequence rendering)

/// Renders the entire residue sequence into a single Canvas view instead of
/// creating 1000+ individual SwiftUI Text views. This eliminates the layout
/// and diffing overhead that caused tab-switching lag for large proteins.
private struct SequenceCanvas<MenuContent: View>: View {
    let cells: [SequenceCell]
    let selectedResidueIndices: Set<Int>
    let hoveredResIdx: Int?
    let ssCache: [String: SecondaryStructure]
    let onTapCell: (ResidueEntry) -> Void
    let onHoverCell: (Int?) -> Void
    @ViewBuilder let contextMenuBuilder: (ResidueEntry) -> MenuContent

    // Layout constants
    private let cellW: CGFloat = 14
    private let cellH: CGFloat = 16
    private let numberH: CGFloat = 8
    private let gapW: CGFloat = 40
    private let spacing: CGFloat = 1

    // Precomputed layout: (cellIndex, origin) pairs
    @State private var cellRects: [CGRect] = []
    @State private var canvasHeight: CGFloat = 20
    @State private var containerWidth: CGFloat = 250

    // Track which residue the cursor is over for the context menu
    @State private var contextHitEntry: ResidueEntry?
    // Debounce hover: only update when the hit cell index changes
    @State private var lastHitCellIndex: Int? = nil

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            Canvas { context, size in
                drawCells(context: context, size: size, containerWidth: width)
            }
            .frame(height: canvasHeight)
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    let hitIdx = hitTest(location)
                    // Only update state when the hovered cell actually changes
                    guard hitIdx != lastHitCellIndex else { return }
                    lastHitCellIndex = hitIdx
                    if let hitIdx, case .residue(let entry) = cells[hitIdx] {
                        onHoverCell(entry.resIdx)
                        contextHitEntry = entry
                    } else {
                        onHoverCell(nil)
                    }
                case .ended:
                    lastHitCellIndex = nil
                    onHoverCell(nil)
                }
            }
            .onTapGesture { location in
                if let hitIdx = hitTest(location), case .residue(let entry) = cells[hitIdx] {
                    onTapCell(entry)
                }
            }
            .contextMenu {
                if let entry = contextHitEntry {
                    contextMenuBuilder(entry)
                }
            }
            .onChange(of: width) { _, newWidth in
                if abs(newWidth - containerWidth) > 1 {
                    containerWidth = newWidth
                    recomputeLayout(width: newWidth)
                }
            }
            .onAppear {
                containerWidth = width
                recomputeLayout(width: width)
            }
            .onChange(of: cells.count) { _, _ in
                recomputeLayout(width: containerWidth)
            }
        }
        .frame(height: canvasHeight)
    }

    // MARK: - Layout

    private func recomputeLayout(width: CGFloat) {
        var rects: [CGRect] = []
        rects.reserveCapacity(cells.count)
        var x: CGFloat = 0
        var y: CGFloat = 0
        let rowH = cellH + numberH

        for cell in cells {
            let w: CGFloat
            switch cell {
            case .residue: w = cellW
            case .gap: w = gapW
            }

            if x + w > width && x > 0 {
                x = 0
                y += rowH + spacing
            }
            rects.append(CGRect(x: x, y: y, width: w, height: rowH))
            x += w + spacing
        }

        cellRects = rects
        canvasHeight = max(y + rowH + 2, 20)
    }

    // MARK: - Hit Testing (row-based for O(cols) instead of O(N))

    private func hitTest(_ point: CGPoint) -> Int? {
        guard !cellRects.isEmpty else { return nil }
        let rowH = cellH + numberH + spacing
        // Find which row the point is in
        let rowIdx = Int(point.y / rowH)
        let rowMinY = CGFloat(rowIdx) * rowH
        let rowMaxY = rowMinY + rowH
        // Only scan cells in this row
        for (i, rect) in cellRects.enumerated() {
            if rect.minY >= rowMinY && rect.minY < rowMaxY && rect.contains(point) {
                return i
            }
            // Past this row — stop early
            if rect.minY >= rowMaxY { break }
        }
        return nil
    }

    // MARK: - Drawing

    private func drawCells(context: GraphicsContext, size: CGSize, containerWidth: CGFloat) {
        // Recompute layout inline if rects are stale
        let rects: [CGRect]
        if cellRects.count == cells.count {
            rects = cellRects
        } else {
            // Fallback: compute synchronously
            var computed: [CGRect] = []
            var x: CGFloat = 0
            var y: CGFloat = 0
            let rowH = cellH + numberH
            for cell in cells {
                let w: CGFloat = (cell.isResidue ? cellW : gapW)
                if x + w > containerWidth && x > 0 { x = 0; y += rowH + spacing }
                computed.append(CGRect(x: x, y: y, width: w, height: rowH))
                x += w + spacing
            }
            rects = computed
        }

        let residueFont = Font.system(size: 10, weight: .medium, design: .monospaced)
        let residueFontBold = Font.system(size: 10, weight: .bold, design: .monospaced)
        let numberFont = Font.system(size: 6, design: .monospaced)
        let gapFont = Font.system(size: 9, weight: .medium, design: .monospaced)

        for (i, cell) in cells.enumerated() {
            guard i < rects.count else { break }
            let rect = rects[i]

            switch cell {
            case .residue(let entry):
                let isSelected = selectedResidueIndices.contains(entry.resIdx)
                let isHovered = hoveredResIdx == entry.resIdx

                // Background
                let bgColor: Color
                if isSelected {
                    bgColor = .cyan.opacity(0.6)
                } else if isHovered {
                    bgColor = ssColor(entry.ss).opacity(0.5)
                } else {
                    bgColor = ssColor(entry.ss)
                }

                let cellRect = CGRect(x: rect.minX, y: rect.minY + numberH, width: cellW, height: cellH)
                context.fill(Path(cellRect), with: .color(bgColor))

                // Selection/hover border
                if isSelected {
                    context.stroke(Path(cellRect), with: .color(.cyan), lineWidth: 1)
                } else if isHovered {
                    context.stroke(Path(cellRect), with: .color(.white.opacity(0.3)), lineWidth: 1)
                }

                // Residue number every 10
                if entry.seqNum % 10 == 0 {
                    let numText = Text("\(entry.seqNum)")
                        .font(numberFont)
                        .foregroundColor(.gray.opacity(0.5))
                    context.draw(context.resolve(numText),
                                 at: CGPoint(x: rect.minX + cellW / 2, y: rect.minY + numberH / 2),
                                 anchor: .center)
                }

                // One-letter code
                let letterText = Text(entry.oneLetterCode)
                    .font(isSelected ? residueFontBold : residueFont)
                    .foregroundColor(isSelected ? .white : .secondary)
                context.draw(context.resolve(letterText),
                             at: CGPoint(x: cellRect.midX, y: cellRect.midY),
                             anchor: .center)

            case .gap(_, let count, _, _):
                let gapRect = CGRect(x: rect.minX, y: rect.minY + numberH, width: gapW, height: cellH)
                let gapPath = RoundedRectangle(cornerRadius: 2).path(in: gapRect)
                // Gap background
                context.fill(
                    gapPath,
                    with: .color(.orange.opacity(0.1))
                )
                context.stroke(
                    gapPath,
                    with: .color(.orange.opacity(0.3)),
                    lineWidth: 0.5
                )
                // Gap label
                let gapText = Text("···\(count)···")
                    .font(gapFont)
                    .foregroundColor(.orange.opacity(0.8))
                context.draw(context.resolve(gapText),
                             at: CGPoint(x: gapRect.midX, y: gapRect.midY),
                             anchor: .center)
            }
        }
    }
}

// Helper for rounded rect path in Canvas
private struct RoundedRect: Shape {
    let rect: CGRect
    let cornerSize: CGSize
    func path(in bounds: CGRect) -> Path {
        Path(roundedRect: rect, cornerSize: cornerSize)
    }
}

private extension SequenceCell {
    var isResidue: Bool {
        if case .residue = self { return true }
        return false
    }
}
