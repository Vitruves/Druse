// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

/// One-letter amino acid code lookup.
private let threeToOne: [String: String] = [
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
]

/// One-letter nucleotide code lookup (DNA and RNA residue names).
private let nucleotideToOne: [String: String] = [
    // DNA
    "DA": "A", "DT": "T", "DG": "G", "DC": "C", "DU": "U", "DI": "I",
    // RNA (single-letter residue names in PDB)
    "A": "A", "U": "U", "G": "G", "C": "C", "I": "I",
    // Longer PDB names
    "ADE": "A", "THY": "T", "GUA": "G", "CYT": "C", "URA": "U",
]

// MARK: - Secondary-structure presentation (shared with InspectorPanel)

extension SecondaryStructure {
    /// Adaptive data colours for chips, legends and sequence cells. Dedicated
    /// hues (magenta helix, gold sheet, green turn — the familiar ribbon
    /// convention) so structure never reads as an error, warning or water.
    var ssTint: Color {
        switch self {
        case .helix: DS.dynamic(light: 0xB8327F, dark: 0xF07AC0)
        case .sheet: DS.dynamic(light: 0xA77B00, dark: 0xE8C255)
        case .turn:  DS.dynamic(light: 0x2B8A5E, dark: 0x5FD39B)
        case .coil:  Color.secondary
        }
    }

    var ssDisplayName: String {
        switch self {
        case .helix: "Alpha Helix"
        case .sheet: "Beta Sheet"
        case .turn:  "Turn"
        case .coil:  "Coil"
        }
    }

    var ssShortName: String {
        switch self {
        case .helix: "Helix"
        case .sheet: "Sheet"
        case .turn:  "Turn"
        case .coil:  "Coil"
        }
    }

    var ssSymbol: String {
        switch self {
        case .helix: "arrow.trianglehead.turn.up.right.diamond"
        case .sheet: "arrow.right.to.line"
        case .turn, .coil: "scribble"
        }
    }

    /// Cell background in the sequence canvas.
    fileprivate var ssFill: Color {
        switch self {
        case .helix: ssTint.opacity(0.22)
        case .sheet: ssTint.opacity(0.26)
        case .turn:  ssTint.opacity(0.22)
        case .coil:  Color.primary.opacity(0.06)
        }
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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
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
        DSStepScaffold {
            if let prot = viewModel.molecules.protein {
                sequenceContent(prot)
            } else {
                DSEmptyState(
                    "No Protein Loaded",
                    systemImage: "textformat.abc",
                    description: "Load a structure to view its sequence."
                )
                .padding(.top, DS.Space.xxl)
            }
        }
        .onAppear { viewModel.ensureSecondaryStructure() }
        .onChange(of: viewModel.molecules.protein?.id) { _, _ in
            viewModel.ensureSecondaryStructure()
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
    private func sequenceContent(_ prot: Molecule) -> some View {
        let proteinChains = prot.chains.filter { $0.type == .protein }
        let naChains = prot.chains.filter { $0.type == .nucleicAcid }
        let ssLookup = secondaryStructureLookup(prot)

        overviewCard(prot)

        if proteinChains.isEmpty && naChains.isEmpty {
            PanelHint(text: "No protein or nucleic acid chains found.", icon: "info.circle")
        } else {
            ForEach(proteinChains, id: \.id) { chain in
                chainSection(chain, prot: prot, ssLookup: ssLookup)
            }
            if !naChains.isEmpty {
                PanelLabeledDivider(title: "Nucleic Acid", icon: "helix")
                ForEach(naChains, id: \.id) { chain in
                    naChainSection(chain, prot: prot)
                }
            }
        }

        if !viewModel.workspace.selectedResidueIndices.isEmpty {
            selectedResidueCard(prot)
        }
    }

    // MARK: - Overview card (actions + structure legend)

    private func overviewCard(_ prot: Molecule) -> some View {
        let aaCount = prot.residues.filter(\.isStandard).count
        let naCount = prot.chains.filter { $0.type == .nucleicAcid }
            .flatMap(\.residueIndices)
            .filter { $0 < prot.residues.count && nucleotideToOne[prot.residues[$0].name] != nil }
            .count
        let countText = naCount > 0 ? "\(aaCount) aa + \(naCount) nt" : "\(aaCount) aa"
        let selectedCount = viewModel.workspace.selectedResidueIndices.count

        return PanelCard(
            "Overview",
            icon: "list.bullet.rectangle",
            accessory: { DSTag(countText, monospaced: true) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                PanelChoiceGrid(columns: 2) {
                    PanelSecondaryButton(
                        title: "Copy", icon: "doc.on.doc",
                        help: "Copy the selected residues (or the whole sequence) as one-letter codes"
                    ) { viewModel.copySequenceToClipboard() }
                    .accessibilityIdentifier(AccessibilityID.seqCopySequence)

                    PanelSecondaryButton(
                        title: "Deselect All", icon: "xmark.circle",
                        isDisabled: selectedCount == 0,
                        help: "Deselect all residues"
                    ) { clearSelection() }
                    .accessibilityIdentifier(AccessibilityID.seqClearSelection)
                }

                PanelSubheader(title: "Select by structure")
                PanelChoiceGrid(columns: 4) {
                    ssButton(.helix, plural: "helices", id: AccessibilityID.seqSelectHelices)
                    ssButton(.sheet, plural: "sheets", id: AccessibilityID.seqSelectSheets)
                    ssButton(.coil, plural: "coils", id: AccessibilityID.seqSelectCoils)
                    ssButton(.turn, plural: "turns", id: AccessibilityID.seqSelectTurns)
                }

                HStack(spacing: DS.Space.xs) {
                    Image(systemName: selectedCount > 0 ? "scope" : "cursorarrow.click")
                        .font(DS.Font.caption)
                        .foregroundStyle(selectedCount > 0 ? DS.Color.accent : Color.secondary)
                    Text(selectedCount > 0
                         ? "\(selectedCount) residue\(selectedCount == 1 ? "" : "s") selected"
                         : "Click a residue · ⇧ range · ⌘ toggle")
                        .font(DS.Font.secondary)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
    }

    /// Structure-selection button whose swatch doubles as the cell-color legend.
    private func ssButton(_ ss: SecondaryStructure, plural: String, id: String) -> some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        return Button {
            viewModel.selectBySecondaryStructure(ss)
        } label: {
            HStack(spacing: DS.Space.xs) {
                RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                    .fill(ss.ssFill)
                    .overlay(
                        RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                            .strokeBorder(ss.ssTint.opacity(0.6), lineWidth: 0.5)
                    )
                    .frame(width: 10, height: 10)
                Text(ss.ssShortName)
                    .font(DS.Font.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, DS.Space.xs)
            .frame(maxWidth: .infinity)
            .frame(height: DS.Size.buttonHeight)
            .background(shape.fill(DS.Color.controlFill))
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .help("Select all \(plural)")
        .accessibilityLabel("Select all \(plural)")
        .accessibilityIdentifier(id)
    }

    private func clearSelection() {
        viewModel.workspace.selectedResidueIndices.removeAll()
        viewModel.workspace.selectedAtomIndices.removeAll()
        viewModel.workspace.selectedAtomIndex = nil
        viewModel.pushToRenderer()
    }

    // MARK: - Chain Section

    private func toggleCollapsed(_ chainID: String) {
        withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) {
            if collapsedChains.contains(chainID) {
                collapsedChains.remove(chainID)
            } else {
                collapsedChains.insert(chainID)
            }
        }
    }

    private func chainHeaderLeading(_ chain: Chain, isCollapsed: Bool) -> some View {
        HStack(spacing: DS.Space.s) {
            Button { toggleCollapsed(chain.id) } label: {
                Image(systemName: "chevron.right")
                    .font(DS.Font.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees(isCollapsed ? 0 : 90))
                    .frame(width: 14, height: 14)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(isCollapsed ? "Expand chain \(chain.id)" : "Collapse chain \(chain.id)")
            .accessibilityLabel(isCollapsed ? "Expand chain \(chain.id)" : "Collapse chain \(chain.id)")

            Circle()
                .fill(chain.displayColor)
                .frame(width: 8, height: 8)
            Text("Chain \(chain.id)")
                .font(DS.Font.label.weight(.semibold).monospaced())
        }
    }

    @ViewBuilder
    private func chainSection(_ chain: Chain, prot: Molecule, ssLookup: [String: SecondaryStructure]) -> some View {
        let isCollapsed = collapsedChains.contains(chain.id)
        let residueEntries = chainResidues(chain, prot: prot, ssLookup: ssLookup)
        let cells = buildCells(from: residueEntries, chainID: chain.id)
        let gapCount = cells.filter { !$0.isResidue }.count

        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(spacing: DS.Space.s) {
                chainHeaderLeading(chain, isCollapsed: isCollapsed)
                DSTag("\(residueEntries.count) res", monospaced: true)
                if gapCount > 0 {
                    DSTag("\(gapCount) gap\(gapCount == 1 ? "" : "s")", tone: .warning, monospaced: true)
                        .help("Missing residues detected from numbering discontinuities")
                }
                Spacer(minLength: DS.Space.xs)
                chainMenu(chain, prot: prot)
            }

            if !isCollapsed {
                // Canvas-based sequence rendering: draws all residues directly
                // instead of creating 1000+ individual SwiftUI views
                SequenceCanvas(
                    cells: cells,
                    selectedResidueIndices: viewModel.workspace.selectedResidueIndices,
                    hoveredResIdx: hoveredResIdx,
                    onTapCell: { entry in handleResidueClick(entry, chainOrder: residueEntries) },
                    onHoverCell: { resIdx in hoveredResIdx = resIdx },
                    contextMenuBuilder: { entry in residueContextMenu(entry, chainID: chain.id, prot: prot) }
                )
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: DS.Space.m)
    }

    private func chainMenu(_ chain: Chain, prot: Molecule) -> some View {
        Menu {
            Button("Select Chain") {
                viewModel.selectChain(chain.id)
            }
            Button("Copy Sequence") {
                viewModel.copySequenceToClipboard(chainID: chain.id)
            }
            Divider()
            Button("Rename Chain…") {
                renameChainID = chain.id
                renameNewID = chain.id
                showRenameSheet = true
            }
            if prot.chains.filter({ $0.type == .protein }).count > 1 {
                Menu("Merge Into…") {
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
                .font(DS.Font.label)
                .foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Chain \(chain.id) actions")
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
    private func chainResidues(_ chain: Chain, prot: Molecule, ssLookup: [String: SecondaryStructure]) -> [ResidueEntry] {
        chain.residueIndices.compactMap { idx -> ResidueEntry? in
            guard idx < prot.residues.count else { return nil }
            let res = prot.residues[idx]
            guard res.isStandard else { return nil }
            let code = threeToOne[res.name] ?? "?"
            let ss = ssLookup["\(res.chainID)_\(res.sequenceNumber)"] ?? .coil
            return ResidueEntry(
                resIdx: idx, name: res.name, oneLetterCode: code,
                seqNum: res.sequenceNumber, chainID: res.chainID, ss: ss
            )
        }
        .sorted { $0.seqNum < $1.seqNum }
    }

    /// Build ResidueEntry list for a nucleic acid chain.
    private func naChainResidues(_ chain: Chain, prot: Molecule) -> [ResidueEntry] {
        chain.residueIndices.compactMap { idx -> ResidueEntry? in
            guard idx < prot.residues.count else { return nil }
            let res = prot.residues[idx]
            guard let code = nucleotideToOne[res.name] else { return nil }  // skip non-nucleotide residues
            return ResidueEntry(
                resIdx: idx, name: res.name, oneLetterCode: code,
                seqNum: res.sequenceNumber, chainID: res.chainID, ss: .coil
            )
        }
        .sorted { $0.seqNum < $1.seqNum }
    }

    // MARK: - Nucleic Acid Chain Section

    @ViewBuilder
    private func naChainSection(_ chain: Chain, prot: Molecule) -> some View {
        let isCollapsed = collapsedChains.contains(chain.id)
        let residueEntries = naChainResidues(chain, prot: prot)
        let isDNA = residueEntries.contains { $0.name.hasPrefix("D") }

        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(spacing: DS.Space.s) {
                chainHeaderLeading(chain, isCollapsed: isCollapsed)
                DSTag(isDNA ? "DNA" : "RNA", tone: .accent)
                DSTag("\(residueEntries.count) nt", monospaced: true)
                Spacer(minLength: 0)
            }

            if !isCollapsed && !residueEntries.isEmpty {
                // Text-based sequence for nucleic acids, in blocks of 10 with a position ruler.
                let seqString = residueEntries.map(\.oneLetterCode).joined()
                let chunks = stride(from: 0, to: seqString.count, by: 10).map { start in
                    let s = seqString.index(seqString.startIndex, offsetBy: start)
                    let e = seqString.index(s, offsetBy: min(10, seqString.count - start))
                    return String(seqString[s..<e])
                }

                FlowLayout(spacing: DS.Space.m) {
                    ForEach(Array(chunks.enumerated()), id: \.offset) { ci, chunk in
                        VStack(alignment: .leading, spacing: 0) {
                            Text("\(ci * 10 + 1)")
                                .font(DS.Font.caption)
                                .monospacedDigit()
                                .foregroundStyle(.tertiary)
                            Text(chunk)
                                .font(DS.Font.monoSmall.weight(.medium))
                                .foregroundStyle(.primary)
                        }
                    }
                }
                .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: DS.Space.m)
    }

    /// (chainID, seqNum) → secondary structure, read from the molecule so it follows
    /// assignments made after the view appeared (e.g. backbone-derived ones).
    private func secondaryStructureLookup(_ prot: Molecule) -> [String: SecondaryStructure] {
        var lookup: [String: SecondaryStructure] = [:]
        for ssa in prot.secondaryStructureAssignments where ssa.start <= ssa.end {
            for seq in ssa.start...ssa.end {
                lookup["\(ssa.chain)_\(seq)"] = ssa.type
            }
        }
        return lookup
    }

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
            .accessibilityIdentifier(AccessibilityID.seqDeleteSelected)

            Button("Create Subset from Selection") {
                viewModel.createSubsetFromSelection()
            }

            Button("Define Pocket from Selection") {
                viewModel.definePocketFromSelection()
            }

            Button("Add Docking Constraint\u{2026}") {
                viewModel.showConstraintSheetForResidue(entry.resIdx)
            }

            // FlexibleResidueConfig stores residue sequence numbers (what FlexDockingEngine
            // matches against atom.residueSeq), not indices into `residues`.
            let flexConfig = viewModel.docking.flexibleResidueConfig
            let isFlexible = flexConfig.flexibleResidueIndices.contains(entry.seqNum)
            let hasRotamers = RotamerLibrary.rotamers(for: entry.name) != nil
            Button(isFlexible ? "Remove Flexible Residue" : "Make Flexible (Induced Fit)") {
                if isFlexible {
                    viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll { $0 == entry.seqNum }
                } else {
                    viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.append(entry.seqNum)
                }
            }
            .disabled(!isFlexible && (!hasRotamers || flexConfig.flexibleResidueIndices.count >= FlexibleResidueConfig.maxFlexibleResidues))

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
            Button("Turns in Chain \(chainID)") {
                viewModel.selectBySecondaryStructure(.turn, chainID: chainID)
            }
        }

        Divider()

        Button("Rename Chain \(chainID)…") {
            renameChainID = chainID
            renameNewID = chainID
            showRenameSheet = true
        }

        if prot.chains.filter({ $0.type == .protein }).count > 1 {
            Menu("Merge Chain \(chainID) Into…") {
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

    /// `chainOrder` is the chain's displayed residue order: a shift-range spans the cells
    /// between the two clicks, never the waters, ligands or other chains whose indices
    /// happen to fall between them in `residues`.
    private func handleResidueClick(_ entry: ResidueEntry, chainOrder: [ResidueEntry]) {
        let modifiers = NSEvent.modifierFlags
        let isCmd = modifiers.contains(.command)
        let isShift = modifiers.contains(.shift)

        if isShift, let last = lastClickedResIdx,
           let anchor = chainOrder.firstIndex(where: { $0.resIdx == last }),
           let current = chainOrder.firstIndex(where: { $0.resIdx == entry.resIdx }) {
            if !isCmd {
                // Shift without Cmd: replace selection with range
                viewModel.workspace.selectedResidueIndices.removeAll()
            }
            for cell in chainOrder[min(anchor, current)...max(anchor, current)] {
                viewModel.workspace.selectedResidueIndices.insert(cell.resIdx)
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

    // MARK: - Selected Residue Card

    private func selectedResidueCard(_ prot: Molecule) -> some View {
        let sortedSelected = viewModel.workspace.selectedResidueIndices.sorted()
        let totalAtoms = sortedSelected.reduce(0) { sum, idx in
            idx < prot.residues.count ? sum + prot.residues[idx].atomIndices.count : sum
        }

        return PanelCard(
            "Selected",
            icon: "scope",
            accessory: { DSTag("\(sortedSelected.count)", tone: .accent, monospaced: true) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                if sortedSelected.count == 1, let first = sortedSelected.first, first < prot.residues.count {
                    let res = prot.residues[first]
                    HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                        Text("\(res.name) \(res.sequenceNumber)")
                            .font(DS.Font.mono.weight(.medium))
                        Text("Chain \(res.chainID)")
                            .font(DS.Font.secondary)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text(condenseRanges(sortedSelected, prot: prot))
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .textSelection(.enabled)
                }
                Text("\(totalAtoms) atoms")
                    .font(DS.Font.secondary)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
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
        return startSeq == endSeq ? "\(startSeq)" : "\(startSeq)–\(endSeq)"
    }

    // MARK: - Rename Sheet

    private var renameSheet: some View {
        DSSheetScaffold(
            title: "Rename Chain \(renameChainID)",
            subtitle: "Chain identifiers are usually a single letter or digit.",
            systemImage: "character.cursor.ibeam"
        ) {
            Form {
                TextField("New chain ID", text: $renameNewID)
                    .font(DS.Font.mono)
            }
            .formStyle(.grouped)
        } footer: {
            Button("Cancel") { showRenameSheet = false }
                .keyboardShortcut(.cancelAction)
            Button("Rename") {
                viewModel.renameChain(from: renameChainID, to: renameNewID)
                showRenameSheet = false
            }
            .keyboardShortcut(.defaultAction)
            .disabled(renameNewID.isEmpty || renameNewID == renameChainID)
        }
        .frame(minWidth: 360, idealWidth: 380, minHeight: 230, idealHeight: 240)
    }

    // MARK: - Merge Sheet

    private var mergeSheet: some View {
        DSSheetScaffold(
            title: "Merge Chains",
            subtitle: "All residues from chain \(mergeSourceID) will be reassigned to chain \(mergeTargetID). This cannot be undone.",
            systemImage: "arrow.triangle.merge"
        ) {
            Form {
                LabeledContent("From") {
                    Text("Chain \(mergeSourceID)").font(DS.Font.mono)
                }
                LabeledContent("Into") {
                    Text("Chain \(mergeTargetID)").font(DS.Font.mono)
                }
            }
            .formStyle(.grouped)
        } footer: {
            Button("Cancel") { showMergeSheet = false }
                .keyboardShortcut(.cancelAction)
            Button("Merge") {
                viewModel.mergeChains(from: mergeSourceID, into: mergeTargetID)
                showMergeSheet = false
            }
            .keyboardShortcut(.defaultAction)
        }
        .frame(minWidth: 380, idealWidth: 400, minHeight: 260, idealHeight: 280)
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
    let onTapCell: (ResidueEntry) -> Void
    let onHoverCell: (Int?) -> Void
    @ViewBuilder let contextMenuBuilder: (ResidueEntry) -> MenuContent

    // Layout constants
    private let cellW: CGFloat = 14
    private let cellH: CGFloat = 16
    private let numberH: CGFloat = 11
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

        let residueFont = DS.Font.monoSmall.weight(.medium)
        let residueFontBold = DS.Font.monoSmall.weight(.bold)
        let numberFont = DS.Font.canvasLabel.monospacedDigit()
        let gapFont = DS.Font.monoSmall.weight(.medium)

        for (i, cell) in cells.enumerated() {
            guard i < rects.count else { break }
            let rect = rects[i]

            switch cell {
            case .residue(let entry):
                let isSelected = selectedResidueIndices.contains(entry.resIdx)
                let isHovered = hoveredResIdx == entry.resIdx

                // Background
                let cellRect = CGRect(x: rect.minX, y: rect.minY + numberH, width: cellW, height: cellH)
                context.fill(Path(cellRect), with: .color(isSelected ? DS.Color.accent.opacity(0.35) : entry.ss.ssFill))
                if isHovered && !isSelected {
                    context.fill(Path(cellRect), with: .color(.primary.opacity(0.10)))
                }

                // Selection/hover border
                if isSelected {
                    context.stroke(Path(cellRect), with: .color(DS.Color.accent), lineWidth: 1)
                } else if isHovered {
                    context.stroke(Path(cellRect), with: .color(.primary.opacity(0.3)), lineWidth: 1)
                }

                // Residue number every 10
                if entry.seqNum % 10 == 0 {
                    let numText = Text("\(entry.seqNum)")
                        .font(numberFont)
                        .foregroundColor(.secondary)
                    context.draw(context.resolve(numText),
                                 at: CGPoint(x: rect.minX + cellW / 2, y: rect.minY + numberH / 2),
                                 anchor: .center)
                }

                // One-letter code
                let letterText = Text(entry.oneLetterCode)
                    .font(isSelected ? residueFontBold : residueFont)
                    .foregroundColor(isSelected ? .primary : .primary.opacity(0.8))
                context.draw(context.resolve(letterText),
                             at: CGPoint(x: cellRect.midX, y: cellRect.midY),
                             anchor: .center)

            case .gap(_, let count, _, _):
                let gapRect = CGRect(x: rect.minX, y: rect.minY + numberH, width: gapW, height: cellH)
                let gapPath = RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous).path(in: gapRect)
                // Gap background
                context.fill(
                    gapPath,
                    with: .color(DS.Color.warning.opacity(0.12))
                )
                context.stroke(
                    gapPath,
                    with: .color(DS.Color.warning.opacity(0.4)),
                    lineWidth: 0.5
                )
                // Gap label
                let gapText = Text("···\(count)···")
                    .font(gapFont)
                    .foregroundColor(DS.Color.warning)
                context.draw(context.resolve(gapText),
                             at: CGPoint(x: gapRect.midX, y: gapRect.midY),
                             anchor: .center)
            }
        }
    }
}

private extension SequenceCell {
    var isResidue: Bool {
        if case .residue = self { return true }
        return false
    }
}
