// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Swatch palette (molecule coloring data palette, not UI chrome)

private let swatchPresetColors: [Color] = [
    Color(red: 0.2, green: 0.5, blue: 1.0),   // blue
    Color(red: 0.0, green: 0.75, blue: 0.95),  // cyan
    Color(red: 0.0, green: 0.85, blue: 0.7),   // teal
    Color(red: 0.3, green: 0.7, blue: 0.5),    // sea green
    Color(red: 0.3, green: 0.85, blue: 0.3),   // green
    Color(red: 0.6, green: 0.85, blue: 0.2),   // lime
    Color(red: 0.95, green: 0.85, blue: 0.2),  // yellow
    Color(red: 1.0, green: 0.65, blue: 0.0),   // orange
    Color(red: 1.0, green: 0.4, blue: 0.3),    // red-orange
    Color(red: 0.95, green: 0.25, blue: 0.3),  // red
    Color(red: 0.9, green: 0.3, blue: 0.6),    // pink
    Color(red: 0.75, green: 0.35, blue: 0.85), // purple
    Color(red: 0.85, green: 0.85, blue: 0.85), // light gray
    Color(red: 0.6, green: 0.6, blue: 0.65),   // gray
    Color(red: 0.55, green: 0.45, blue: 0.35), // brown
    Color(red: 0.4, green: 0.4, blue: 0.5),    // slate
]

private func swatchColorMatches(_ a: Color, _ b: Color) -> Bool {
    let c1 = NSColor(a).usingColorSpace(.deviceRGB)
    let c2 = NSColor(b).usingColorSpace(.deviceRGB)
    guard let c1, let c2 else { return false }
    return abs(c1.redComponent - c2.redComponent) < 0.05
        && abs(c1.greenComponent - c2.greenComponent) < 0.05
        && abs(c1.blueComponent - c2.blueComponent) < 0.05
}

private func swatchColor(_ v: SIMD3<Float>) -> Color {
    Color(red: Double(v.x), green: Double(v.y), blue: Double(v.z))
}

private func swatchColor(_ v: SIMD4<Float>) -> Color {
    swatchColor(SIMD3(v.x, v.y, v.z))
}

private func rgbComponents(_ color: Color) -> SIMD3<Float>? {
    guard let comps = NSColor(color).usingColorSpace(.deviceRGB) else { return nil }
    return SIMD3<Float>(Float(comps.redComponent), Float(comps.greenComponent), Float(comps.blueComponent))
}

/// A swatch: rounded square with a hairline so light colors stay visible on light backgrounds.
private struct Swatch: View {
    let color: Color
    var size: CGFloat = 16
    var isSelected: Bool = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
        shape
            .fill(color)
            .frame(width: size, height: size)
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
            .padding(isSelected ? 2 : 0)
            .overlay(
                RoundedRectangle(cornerRadius: DS.Radius.xs + 2, style: .continuous)
                    .strokeBorder(isSelected ? DS.Color.accent : Color.clear, lineWidth: 1.5)
            )
    }
}

/// The colour button of a chain or ligand row: a round dot with a darker ring, so a grey colour
/// never reads as an unchecked checkbox next to the row's switch.
private struct ColorDot<Fill: View>: View {
    @ViewBuilder let fill: () -> Fill

    var body: some View {
        fill()
            .frame(width: 14, height: 14)
            .clipShape(Circle())
            .overlay(Circle().strokeBorder(Color.primary.opacity(0.25), lineWidth: 1))
            .frame(width: 18, height: 18)
    }
}

/// 4×4 grid of preset swatches used by both color popovers.
private struct SwatchGrid: View {
    let isSelected: (Color) -> Bool
    let onPick: (Color) -> Void

    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.fixed(24), spacing: DS.Space.xs), count: 4), spacing: DS.Space.xs) {
            ForEach(Array(swatchPresetColors.enumerated()), id: \.offset) { _, preset in
                Button { onPick(preset) } label: {
                    Swatch(color: preset, size: 18, isSelected: isSelected(preset))
                        .frame(width: 24, height: 24)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }
}

/// Chain color picker: CPK / Chain / Custom modes
private struct ChainColorPicker: View {
    var chainID: String
    var paletteColor: SIMD3<Float>
    @Binding var mode: WorkspaceState.ChainColorMode
    @Binding var customColor: SIMD3<Float>?
    @Binding var scope: WorkspaceState.ChainColorScope
    var onChange: () -> Void

    @State private var showPopover = false

    var body: some View {
        Button(action: { showPopover.toggle() }) {
            ColorDot { swatchIcon }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Chain \(chainID) color")
        .accessibilityLabel("Chain \(chainID) color")
        .popover(isPresented: $showPopover, arrowEdge: .leading) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                modeButton("CPK", icon: "atom", isActive: mode == .cpk) {
                    mode = .cpk
                    onChange()
                    showPopover = false
                }
                modeButton("Chain", icon: "link", isActive: mode == .chainDefault, swatchColor: paletteColor) {
                    mode = .chainDefault
                    onChange()
                    showPopover = false
                }

                Divider()

                PanelSubheader(title: "Custom")

                SwatchGrid(
                    isSelected: { preset in
                        guard mode == .custom, let customColor else { return false }
                        return swatchColorMatches(swatchColor(customColor), preset)
                    },
                    onPick: { preset in
                        if let rgb = rgbComponents(preset) { customColor = rgb }
                        mode = .custom
                        onChange()
                        showPopover = false
                    }
                )

                Divider()

                Toggle(isOn: Binding(
                    get: { scope == .carbonsOnly },
                    set: { scope = $0 ? .carbonsOnly : .allAtoms; onChange() }
                )) {
                    Text("Carbons only").font(DS.Font.secondary)
                }
                .toggleStyle(.checkbox)
                .controlSize(.small)
                .disabled(mode == .cpk)
                .help("Apply color only to carbon atoms; heteroatoms keep CPK colors")
            }
            .padding(DS.Space.l)
            .frame(width: 136)
        }
    }

    @ViewBuilder
    private var swatchIcon: some View {
        switch mode {
        case .cpk:
            // Multi-color CPK indicator (C gray, N blue, O red)
            ZStack {
                Color(red: 0.5, green: 0.5, blue: 0.5)
                Circle().fill(Color(red: 0.0, green: 0.4, blue: 1.0)).frame(width: 7, height: 7).offset(x: -3, y: -3)
                Circle().fill(Color(red: 1.0, green: 0.2, blue: 0.2)).frame(width: 6, height: 6).offset(x: 3, y: 3)
            }
        case .chainDefault:
            swatchColor(paletteColor)
        case .custom:
            swatchColor(customColor ?? paletteColor)
        }
    }

    private func modeButton(_ title: String, icon: String, isActive: Bool, swatchColor sc: SIMD3<Float>? = nil, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: DS.Space.s) {
                if let sc {
                    Swatch(color: swatchColor(sc), size: 10)
                        .frame(width: 14)
                } else {
                    Image(systemName: icon)
                        .font(DS.Font.secondary)
                        .frame(width: 14)
                }
                Text(title)
                    .font(DS.Font.label.weight(isActive ? .semibold : .regular))
                Spacer()
                if isActive {
                    Image(systemName: "checkmark")
                        .font(DS.Font.caption.weight(.bold))
                        .foregroundStyle(DS.Color.accent)
                }
            }
            .padding(.horizontal, DS.Space.xs)
            .padding(.vertical, DS.Space.xs)
            .background(
                RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                    .fill(isActive ? DS.Color.accentSoft : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }
}

/// Simple color swatch picker (for ligand carbon color)
private struct InlineColorPicker: View {
    @Binding var color: Color
    var label: String = ""
    var carbonsOnly: Binding<Bool>? = nil
    var onReset: (() -> Void)? = nil

    @State private var showPopover = false

    var body: some View {
        Button(action: { showPopover.toggle() }) {
            ColorDot { color }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(label)
        .accessibilityLabel(label)
        .popover(isPresented: $showPopover, arrowEdge: .leading) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                SwatchGrid(
                    isSelected: { swatchColorMatches(color, $0) },
                    onPick: { preset in
                        color = preset
                        showPopover = false
                    }
                )
                if let carbonsOnly {
                    Divider()
                    Toggle(isOn: carbonsOnly) {
                        Text("Carbons only").font(DS.Font.secondary)
                    }
                    .toggleStyle(.checkbox)
                    .controlSize(.small)
                    .help("Apply color only to carbon atoms; heteroatoms keep CPK colors")
                }
                if let onReset {
                    Divider()
                    Button {
                        onReset()
                        showPopover = false
                    } label: {
                        Label("Default", systemImage: "arrow.counterclockwise")
                            .font(DS.Font.secondary)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderless)
                    .help("Reset to the color scheme's default")
                }
            }
            .padding(DS.Space.l)
            .frame(width: 136)
        }
    }
}

// MARK: - Inspector

struct InspectorPanel: View {
    @Environment(AppViewModel.self) private var viewModel

    private var selectionMode: WorkspaceState.SelectionMode {
        viewModel.workspace.selectionMode
    }

    private var hasMolecule: Bool {
        viewModel.molecules.protein != nil || viewModel.molecules.ligand != nil
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if hasMolecule {
                ScrollView {
                    VStack(alignment: .leading, spacing: DS.Space.m) {
                        selectionCard
                        chainsCard
                        subsetsCard
                        statisticsCard
                    }
                    .padding(DS.Space.l)
                }
                .scrollIndicators(.automatic)
            } else {
                emptyInspector
            }
        }
    }

    /// Sized for a 280 pt column: ContentUnavailableView's title2 headline reads as a
    /// page-level message here.
    private var emptyInspector: some View {
        VStack(spacing: DS.Space.s) {
            Image(systemName: "sidebar.squares.trailing")
                .font(.system(size: 26, weight: .light))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(.tertiary)
                .padding(.bottom, DS.Space.xxs)
            Text("No Structure Loaded")
                .font(DS.Font.sectionTitle)
                .foregroundStyle(.secondary)
            Text("Chains, selection details and statistics appear here once a protein or ligand is open.")
                .font(DS.Font.secondary)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, DS.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Title row of the pane. It has no close button: the toolbar's Inspector toggle (⌥⌘0)
    /// shows and hides it, as for the navigator, so the pane reads as part of the window.
    private var header: some View {
        HStack(spacing: DS.Space.s) {
            Text("Inspector")
                .font(DS.Font.sectionTitle)
            Spacer(minLength: DS.Space.xs)
            if hasMolecule {
                SelectionShortcutsButton()
            }
        }
        .padding(.horizontal, DS.Space.l)
        .frame(height: 36)
    }

    // MARK: - Selection card

    private var selectionCard: some View {
        PanelCard(
            "Selection",
            icon: "cursorarrow.click.2",
            accessory: {
                Picker("Selection mode", selection: Binding(
                    get: { viewModel.workspace.selectionMode },
                    set: { viewModel.workspace.selectionMode = $0 }
                )) {
                    ForEach([WorkspaceState.SelectionMode.residue, .atom], id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .fixedSize()
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if selectionMode == .atom {
                    atomSelectionContent
                } else {
                    residueSelectionContent
                }
            }
        }
    }

    // MARK: - Empty selection placeholder

    /// One line pointing at the canvas; the full mouse/keyboard list is in the header's (?) popover.
    private var emptySelectionContent: some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            Text("No selection")
                .font(DS.Font.label.weight(.medium))
                .foregroundStyle(.secondary)
            Text(selectionMode == .atom
                 ? "Click an atom in the viewport; ⌥-drag to box-select."
                 : "Click a residue in the viewport or the Sequence tool; ⌥-drag to box-select.")
                .font(DS.Font.secondary)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Atom Selection Content

    @ViewBuilder
    private var atomSelectionContent: some View {
        let selectedCount = viewModel.workspace.selectedAtomIndices.count

        if selectedCount == 0 {
            emptySelectionContent
        } else if selectedCount == 1, let atom = viewModel.selectedAtom {
            atomInspector(atom)
        } else {
            multiAtomSummary(count: selectedCount)
        }
    }

    // MARK: - Residue Selection Content

    @ViewBuilder
    private var residueSelectionContent: some View {
        let selectedCount = viewModel.workspace.selectedResidueIndices.count

        if selectedCount == 0 {
            emptySelectionContent
        } else if selectedCount == 1, let prot = viewModel.molecules.protein,
                  let resIdx = viewModel.workspace.selectedResidueIndices.first {
            if resIdx < prot.residues.count {
                singleResidueInspector(prot.residues[resIdx], in: prot)
            }
        } else {
            multiResidueSummary(count: selectedCount)
        }
    }

    // MARK: - Single Atom Inspector

    @ViewBuilder
    private func atomInspector(_ atom: Atom) -> some View {
        let elementColor = swatchColor(atom.element.color)
        let tile = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        HStack(spacing: DS.Space.m) {
            Text(atom.element.symbol)
                .font(DS.Font.sectionTitle.monospaced())
                .foregroundStyle(.primary)
                .frame(width: 36, height: 36)
                .background(tile.fill(elementColor.opacity(0.22)))
                .overlay(tile.strokeBorder(elementColor, lineWidth: 1.5))

            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                Text(atom.element.name)
                    .font(DS.Font.sectionTitle)
                Text(atom.name)
                    .font(DS.Font.mono)
                    .foregroundStyle(.secondary)
            }
        }

        VStack(spacing: DS.Space.xs) {
            infoRow("Residue", "\(atom.residueName) \(atom.residueSeq)")
            infoRow("Chain", atom.chainID)
            if atom.formalCharge != 0 {
                infoRow("Formal charge", "\(atom.formalCharge > 0 ? "+" : "")\(atom.formalCharge)")
            }
            infoRow("Partial charge", String(format: "%.3f", atom.charge))
            infoRow("vdW radius", String(format: "%.2f \u{00C5}", atom.element.vdwRadius))
            if atom.tempFactor > 0 {
                infoRow("B-factor", String(format: "%.1f", atom.tempFactor))
            }
            if atom.occupancy < 1.0 {
                infoRow("Occupancy", String(format: "%.2f", atom.occupancy))
            }
        }

        let bonds = connectedBonds(for: atom)
        if !bonds.isEmpty {
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                PanelSubheader(title: "Bonds (\(bonds.count))")
                ForEach(bonds, id: \.id) { bond in
                    let allAtoms = (viewModel.molecules.protein?.atoms ?? []) + (viewModel.molecules.ligand?.atoms ?? [])
                    let otherIdx = bond.atomIndex1 == atom.id ? bond.atomIndex2 : bond.atomIndex1
                    if otherIdx < allAtoms.count {
                        let other = allAtoms[otherIdx]
                        HStack(spacing: DS.Space.xs) {
                            Circle()
                                .fill(swatchColor(other.element.color))
                                .overlay(Circle().strokeBorder(DS.Color.hairline, lineWidth: 0.5))
                                .frame(width: 7, height: 7)
                            Text("\(other.element.symbol) \(other.name)")
                                .font(DS.Font.monoSmall)
                            Spacer()
                            Text(bond.order == .aromatic ? "arom" : "\(bond.order.rawValue)\u{00D7}")
                                .font(DS.Font.monoSmall)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }

        VStack(alignment: .leading, spacing: DS.Space.xs) {
            PanelSubheader(title: "Position")
            HStack(spacing: DS.Space.m) {
                coordLabel("X", atom.position.x)
                coordLabel("Y", atom.position.y)
                coordLabel("Z", atom.position.z)
            }
            .textSelection(.enabled)
        }
    }

    private func connectedBonds(for atom: Atom) -> [Bond] {
        let allBonds = (viewModel.molecules.protein?.bonds ?? []) + (viewModel.molecules.ligand?.bonds ?? [])
        return allBonds.filter { $0.atomIndex1 == atom.id || $0.atomIndex2 == atom.id }
    }

    // MARK: - Multi-Atom Summary

    @ViewBuilder
    private func multiAtomSummary(count: Int) -> some View {
        let allAtoms = (viewModel.molecules.protein?.atoms ?? []) + (viewModel.molecules.ligand?.atoms ?? [])
        let selected = viewModel.workspace.selectedAtomIndices.compactMap { idx in
            idx < allAtoms.count ? allAtoms[idx] : nil
        }
        let elementCounts = Dictionary(grouping: selected, by: \.element.symbol)
            .mapValues(\.count)
            .sorted { $0.value > $1.value }

        selectionTitle("\(count) atoms selected", icon: "atom")

        FlowLayout(spacing: DS.Space.xs) {
            ForEach(elementCounts.prefix(8), id: \.key) { symbol, cnt in
                countChip(label: symbol, count: cnt)
            }
        }

        let residues = Set(selected.map { "\($0.residueName)\($0.residueSeq)" })
        let chains = Set(selected.map(\.chainID))
        VStack(spacing: DS.Space.xs) {
            infoRow("Residues", "\(residues.count)")
            infoRow("Chains", chains.sorted().joined(separator: ", "))
            if !selected.isEmpty {
                let totalCharge = selected.map(\.charge).reduce(0, +)
                infoRow("Avg charge", String(format: "%.3f", totalCharge / Float(selected.count)))
                infoRow("Total charge", String(format: "%.2f", totalCharge))
            }
        }

        selectionActions
    }

    // MARK: - Single Residue Inspector

    @ViewBuilder
    private func singleResidueInspector(_ residue: Residue, in prot: Molecule) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.m) {
            Text(residue.name)
                .font(DS.Font.stepTitle.monospaced())
            Text("#\(residue.sequenceNumber)")
                .font(DS.Font.mono)
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Spacer()
            DSTag("Chain \(residue.chainID)", monospaced: true)
        }

        if let ss = secondaryStructure(for: residue, in: prot) {
            DSTag(ss.ssDisplayName, tint: ss.ssTint, systemImage: ss.ssSymbol)
        }

        let heavyCount = residue.atomIndices.filter { idx in
            idx < prot.atoms.count && prot.atoms[idx].element != .H
        }.count
        let totalCharge = residue.atomIndices.compactMap { idx in
            idx < prot.atoms.count ? prot.atoms[idx].charge : nil
        }.reduce(Float(0), +)
        let bFactors = residue.atomIndices.compactMap { idx in
            idx < prot.atoms.count ? prot.atoms[idx].tempFactor : nil
        }
        VStack(spacing: DS.Space.xs) {
            infoRow("Atoms", "\(residue.atomIndices.count)")
            infoRow("Heavy atoms", "\(heavyCount)")
            infoRow("Net charge", String(format: "%.2f", totalCharge))
            if let avgB = bFactors.isEmpty ? nil : bFactors.reduce(0, +) / Float(bFactors.count), avgB > 0 {
                infoRow("Avg B-factor", String(format: "%.1f", avgB))
            }
        }

        let elements = Dictionary(grouping: residue.atomIndices.compactMap { idx in
            idx < prot.atoms.count ? prot.atoms[idx].element : nil
        }, by: \.symbol).mapValues(\.count).sorted { $0.value > $1.value }
        FlowLayout(spacing: DS.Space.xs) {
            ForEach(elements, id: \.key) { symbol, cnt in
                countChip(label: symbol, count: cnt)
            }
        }

        selectionActions
    }

    private func secondaryStructure(for residue: Residue, in prot: Molecule) -> SecondaryStructure? {
        for ssa in prot.secondaryStructureAssignments {
            if ssa.chain == residue.chainID &&
               residue.sequenceNumber >= ssa.start &&
               residue.sequenceNumber <= ssa.end {
                return ssa.type
            }
        }
        return nil
    }

    // MARK: - Multi-Residue Summary

    @ViewBuilder
    private func multiResidueSummary(count: Int) -> some View {
        if let prot = viewModel.molecules.protein {
            let residues = viewModel.workspace.selectedResidueIndices.compactMap { idx in
                idx < prot.residues.count ? prot.residues[idx] : nil
            }
            let typeCounts = Dictionary(grouping: residues, by: \.name)
                .mapValues(\.count)
                .sorted { $0.value > $1.value }

            selectionTitle("\(count) residues selected", icon: "rectangle.stack")

            FlowLayout(spacing: DS.Space.xs) {
                ForEach(typeCounts.prefix(12), id: \.key) { name, cnt in
                    countChip(label: name, count: cnt)
                }
            }

            let totalAtoms = residues.reduce(0) { $0 + $1.atomIndices.count }
            let chains = Set(residues.map(\.chainID))
            let seqNums = residues.map(\.sequenceNumber).sorted()
            VStack(spacing: DS.Space.xs) {
                infoRow("Total atoms", "\(totalAtoms)")
                infoRow("Chains", chains.sorted().joined(separator: ", "))
                if let first = seqNums.first, let last = seqNums.last {
                    infoRow("Range", "\(first)\u{2013}\(last)")
                }
            }

            selectionActions
        }
    }

    private func selectionTitle(_ text: String, icon: String) -> some View {
        Label {
            Text(text)
                .font(DS.Font.label.weight(.semibold))
                .monospacedDigit()
        } icon: {
            Image(systemName: icon)
                .foregroundStyle(DS.Color.accent)
        }
    }

    // MARK: - Selection Actions (shared between atom & residue summaries)

    private var selectionActions: some View {
        let hasResidues = !viewModel.workspace.selectedResidueIndices.isEmpty
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)

        return VStack(spacing: DS.Space.s) {
            Divider()
            PanelChoiceGrid(columns: 2) {
                Menu {
                    ForEach([5, 6, 8, 10] as [Float], id: \.self) { d in
                        Button(String(format: "Within %.0f Å", d)) {
                            viewModel.selectResiduesWithinDistance(d)
                        }
                    }
                } label: {
                    HStack(spacing: DS.Space.xs) {
                        Image(systemName: "circle.dashed")
                            .font(DS.Font.secondary.weight(.medium))
                        Text("Nearby")
                            .font(DS.Font.label)
                            .lineLimit(1)
                        Image(systemName: "chevron.down")
                            .font(DS.Font.caption.weight(.semibold))
                            .foregroundStyle(.tertiary)
                    }
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity)
                    .frame(height: DS.Size.buttonHeight)
                    .background(shape.fill(DS.Color.controlFill))
                    .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
                    .contentShape(shape)
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .help("Extend selection to residues within a distance threshold")

                PanelSecondaryButton(
                    title: "+1 Residue", icon: "arrow.left.and.right",
                    isDisabled: !hasResidues,
                    help: "Extend the selection by one residue on each side"
                ) { viewModel.extendSelectionByOneResidue() }

                PanelSecondaryButton(
                    title: "Invert", icon: "arrow.triangle.2.circlepath",
                    help: "Invert the selection"
                ) { viewModel.invertSelection() }

                PanelSecondaryButton(
                    title: "Subset", icon: "plus.circle",
                    isDisabled: !hasResidues,
                    help: "Create a named subset from the selected residues"
                ) { viewModel.createSubsetFromSelection() }

                PanelSecondaryButton(
                    title: "Pocket", icon: "scope",
                    tint: DS.Color.pocket,
                    isDisabled: !hasResidues,
                    help: "Define the docking pocket from the selected residues"
                ) { viewModel.definePocketFromSelection() }

                PanelSecondaryButton(
                    title: "Clear", icon: "xmark.circle",
                    tint: DS.Color.danger,
                    help: "Deselect everything"
                ) { viewModel.deselectAll() }
            }
        }
    }

    // MARK: - Chains & Display

    /// Chains of the receptor. The active ligand is always normalized to chain "L"
    /// (`setLigandForDocking`) and has its own row below; listing it here as well would
    /// duplicate it with protein-only actions (hide/remove/extract work on the receptor).
    private var proteinChains: [Chain] {
        viewModel.molecules.protein?.chains ?? []
    }

    private var chainsCard: some View {
        PanelCard(
            "Chains & Display",
            icon: "link",
            accessory: {
                if !proteinChains.isEmpty {
                    DSTag("\(proteinChains.count)", monospaced: true)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                if proteinChains.isEmpty && viewModel.molecules.ligand == nil {
                    Text("No chains loaded")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(proteinChains.enumerated()), id: \.element.id) { chainIndex, chain in
                        chainRow(chain, index: chainIndex)
                    }

                    if !viewModel.workspace.hiddenChainIDs.isEmpty {
                        PanelHint(
                            text: "Hidden chains are excluded from pocket detection.",
                            icon: "eye.slash",
                            color: DS.Color.warning
                        )
                    }

                    if let lig = viewModel.molecules.ligand {
                        if !proteinChains.isEmpty { Divider() }
                        ligandDisplaySection(lig)
                    }
                }
            }
        }
    }

    private func chainRow(_ chain: Chain, index chainIndex: Int) -> some View {
        let palette = WorkspaceState.MoleculeColorScheme.chainPalette
        let paletteColor = palette[chainIndex % palette.count]
        // Default mode mirrors AppViewModel.pushToRenderer: only the
        // .chainColored scheme implies palette colors at startup;
        // every other scheme renders chains as CPK by default.
        let defaultMode: WorkspaceState.ChainColorMode =
            viewModel.workspace.colorScheme == .chainColored ? .chainDefault : .cpk
        let isVisible = !viewModel.workspace.hiddenChainIDs.contains(chain.id)

        return HStack(spacing: DS.Space.s) {
            ChainColorPicker(
                chainID: chain.id,
                paletteColor: paletteColor,
                mode: Binding(
                    get: { viewModel.workspace.chainColorModes[chain.id] ?? defaultMode },
                    set: { viewModel.workspace.chainColorModes[chain.id] = $0 }
                ),
                customColor: Binding(
                    get: { viewModel.workspace.chainColorOverrides[chain.id] },
                    set: { viewModel.workspace.chainColorOverrides[chain.id] = $0 }
                ),
                scope: Binding(
                    get: { viewModel.workspace.chainColorScopes[chain.id] ?? .carbonsOnly },
                    set: { viewModel.workspace.chainColorScopes[chain.id] = $0 }
                ),
                onChange: { viewModel.pushToRenderer() }
            )

            VStack(alignment: .leading, spacing: 0) {
                Text("Chain \(chain.id)")
                    .font(DS.Font.label.weight(.semibold))
                Text("\(chain.type.label) · \(chainAtomCount(chain)) atoms")
                    .font(DS.Font.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .opacity(isVisible ? 1 : 0.5)

            Spacer(minLength: DS.Space.xxs)

            if chain.type != .protein {
                rowIconButton("arrow.right.circle", tint: DS.Color.ligand, help: "Extract chain \(chain.id) as ligand") {
                    viewModel.extractChainAsLigand(chainID: chain.id)
                }
            }
            rowIconButton("eye", tint: .secondary, help: "Show only chain \(chain.id)") {
                showOnlyChain(chain.id)
            }
            rowIconButton("trash", tint: .secondary, help: "Remove chain \(chain.id)") {
                viewModel.removeChain(chainID: chain.id)
            }

            Toggle("Show chain \(chain.id)", isOn: Binding(
                get: { isVisible },
                set: { _ in viewModel.toggleChainVisibility(chain.id) }
            ))
            .toggleStyle(.switch)
            .controlSize(.mini)
            .labelsHidden()
            .help(isVisible ? "Hide chain \(chain.id)" : "Show chain \(chain.id)")
        }
    }

    private func ligandDisplaySection(_ lig: Molecule) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(spacing: DS.Space.s) {
                InlineColorPicker(
                    color: Binding(
                        get: {
                            // Effective rendered color: user override, else the scheme's
                            // ligand-carbon color, else the CPK carbon gray.
                            swatchColor(viewModel.workspace.ligandCarbonColor
                                ?? viewModel.workspace.colorScheme.ligandCarbon
                                ?? SIMD3<Float>(0.56, 0.56, 0.56))
                        },
                        set: { newColor in
                            if let rgb = rgbComponents(newColor) {
                                viewModel.workspace.ligandCarbonColor = rgb
                                viewModel.pushToRenderer()
                            }
                        }
                    ),
                    label: "Set ligand carbon color",
                    carbonsOnly: Binding(
                        get: { viewModel.workspace.ligandColorCarbonsOnly },
                        set: {
                            viewModel.workspace.ligandColorCarbonsOnly = $0
                            viewModel.pushToRenderer()
                        }
                    ),
                    onReset: {
                        viewModel.workspace.ligandCarbonColor = nil
                        viewModel.pushToRenderer()
                    }
                )

                VStack(alignment: .leading, spacing: 0) {
                    Text("Ligand")
                        .font(DS.Font.label.weight(.semibold))
                    Text(lig.name)
                        .font(DS.Font.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .opacity(viewModel.workspace.showLigand ? 1 : 0.5)

                Spacer(minLength: DS.Space.xxs)

                rowIconButton("scope", tint: .secondary, help: "Focus camera on ligand") {
                    viewModel.fitToLigand()
                }
                rowIconButton("trash", tint: .secondary, help: "Remove ligand") {
                    viewModel.removeLigandFromView()
                }

                Toggle("Show ligand", isOn: Binding(
                    get: { viewModel.workspace.showLigand },
                    set: { newValue in
                        viewModel.workspace.showLigand = newValue
                        viewModel.pushToRenderer()
                    }
                ))
                .toggleStyle(.switch)
                .controlSize(.mini)
                .labelsHidden()
                .help(viewModel.workspace.showLigand ? "Hide ligand" : "Show ligand")
            }

            Picker("Ligand style", selection: Binding(
                get: { viewModel.workspace.effectiveLigandRenderMode },
                set: { newMode in viewModel.setLigandRenderMode(newMode) }
            )) {
                Text("Stick").tag(RenderMode.wireframe)
                Text("Ball & Stick").tag(RenderMode.ballAndStick)
                Text("CPK").tag(RenderMode.spaceFilling)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .help("Ligand render style")
        }
    }

    // MARK: - Residue Subsets

    private var subsetsCard: some View {
        PanelCard(
            "Subsets",
            icon: "square.stack.3d.up",
            accessory: {
                Button(action: { viewModel.createSubsetFromSelection() }) {
                    Label("New Subset", systemImage: "plus")
                        .labelStyle(.iconOnly)
                        .font(DS.Font.secondary.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.workspace.selectedResidueIndices.isEmpty)
                .help(viewModel.workspace.selectedResidueIndices.isEmpty
                      ? "Select residues first, then save them as a subset"
                      : "Create subset from selected residues")
                .accessibilityLabel("Create subset from selected residues")
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                if viewModel.workspace.residueSubsets.isEmpty {
                    Text("Select residues, then press + to save them as a named subset.")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    ForEach(viewModel.workspace.residueSubsets) { subset in
                        subsetRow(subset)
                    }
                }
            }
        }
    }

    private func subsetRow(_ subset: ResidueSubset) -> some View {
        HStack(spacing: DS.Space.s) {
            Swatch(color: swatchColor(subset.color), size: 10)
            Text(subset.name)
                .font(DS.Font.label.weight(.medium))
                .lineLimit(1)
            Text("\(subset.residueIndices.count)")
                .font(DS.Font.monoSmall)
                .foregroundStyle(.secondary)
            Spacer(minLength: DS.Space.xxs)
            rowIconButton("selection.pin.in.out", tint: .secondary, help: "Select all residues in this subset") {
                viewModel.selectSubset(id: subset.id)
            }
            rowIconButton("scope", tint: DS.Color.pocket, help: "Define docking pocket from this subset") {
                viewModel.definePocketFromSubset(id: subset.id)
            }
            rowIconButton("trash", tint: .secondary, help: "Delete this subset") {
                viewModel.deleteSubset(id: subset.id)
            }
            Toggle("Show subset \(subset.name)", isOn: Binding(
                get: { subset.isVisible },
                set: { _ in viewModel.toggleSubsetVisibility(id: subset.id) }
            ))
            .toggleStyle(.switch)
            .controlSize(.mini)
            .labelsHidden()
            .help(subset.isVisible ? "Hide subset" : "Show subset")
        }
    }

    // MARK: - Statistics (single home for protein / ligand statistics)

    private var statisticsCard: some View {
        PanelCard("Statistics", icon: "chart.bar") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if let prot = viewModel.molecules.protein {
                    let composition = ResidueComposition(prot)
                    statsHeader("Protein", name: prot.name, color: DS.Color.protein)
                    DSMetricGrid(columns: 2) {
                        DSMetric("Atoms", value: prot.atomCount.formatted())
                        DSMetric("Heavy atoms", value: prot.heavyAtomCount.formatted())
                        DSMetric("Residues", value: composition.aminoAcids.formatted())
                        DSMetric("Chains", value: "\(composition.polymerChains)")
                        DSMetric("Bonds", value: prot.bondCount.formatted())
                        if let mw = composition.sequenceMass {
                            DSMetric("MW", value: String(format: "%.1f", mw / 1000), unit: "kDa")
                        }
                    }
                    if composition.waters > 0 || !composition.hetNames.isEmpty {
                        VStack(spacing: DS.Space.xs) {
                            if composition.waters > 0 {
                                infoRow("Waters", "\(composition.waters)")
                            }
                            if !composition.hetNames.isEmpty {
                                infoRow("HET groups", composition.hetNames.joined(separator: ", "))
                            }
                        }
                    }
                }

                if let lig = viewModel.molecules.ligand {
                    if viewModel.molecules.protein != nil { Divider() }
                    statsHeader("Ligand", name: lig.name, color: DS.Color.ligand)
                    DSMetricGrid(columns: 2) {
                        DSMetric("Atoms", value: "\(lig.atomCount)")
                        DSMetric("Heavy atoms", value: "\(lig.heavyAtomCount)")
                        DSMetric("Bonds", value: "\(lig.bondCount)")
                        DSMetric("MW", value: String(format: "%.1f", RDKitBridge.parentMolecularWeight(of: lig)), unit: "Da")
                    }
                    conformerStepper
                }
            }
        }
    }

    private func statsHeader(_ kind: String, name: String, color: Color) -> some View {
        HStack(spacing: DS.Space.s) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(kind)
                .font(DS.Font.label.weight(.semibold))
            Text(name)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    @ViewBuilder
    private var conformerStepper: some View {
        let confs = viewModel.workspace.ligandConformers
        if confs.count > 1 {
            let activeIndex = viewModel.workspace.activeConformerIndex
            VStack(spacing: DS.Space.xs) {
                HStack(spacing: DS.Space.s) {
                    Text("Conformer")
                        .font(DS.Font.label)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        viewModel.switchConformer(to: max(activeIndex - 1, 0))
                    } label: {
                        Image(systemName: "chevron.left")
                    }
                    .buttonStyle(.borderless)
                    .disabled(activeIndex == 0)
                    .help("Previous conformer")

                    Text("\(activeIndex + 1) / \(confs.count)")
                        .font(DS.Font.monoSmall.weight(.medium))
                        .monospacedDigit()

                    Button {
                        viewModel.switchConformer(to: min(activeIndex + 1, confs.count - 1))
                    } label: {
                        Image(systemName: "chevron.right")
                    }
                    .buttonStyle(.borderless)
                    .disabled(activeIndex >= confs.count - 1)
                    .help("Next conformer")
                }
                if activeIndex < confs.count {
                    infoRow("Energy", String(format: "%.2f kcal/mol", confs[activeIndex].energy))
                }
            }
        }
    }

    // MARK: - Helpers

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.m) {
            Text(label)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer(minLength: DS.Space.xs)
            Text(value)
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)
        }
    }

    /// Compact icon-only row action.
    private func rowIconButton(_ systemImage: String, tint: Color, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(DS.Font.secondary)
                .foregroundStyle(tint)
                .frame(width: 20, height: 20)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(help)
        .accessibilityLabel(help)
    }

    /// Single-line "LABEL N" chip with a fixed intrinsic width so labels never wrap mid-token.
    private func countChip(label: String, count: Int) -> some View {
        HStack(spacing: DS.Space.xxs) {
            Text(label)
                .font(DS.Font.monoSmall.weight(.semibold))
                .foregroundStyle(.primary)
            Text("\(count)")
                .font(DS.Font.monoSmall)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, DS.Space.s)
        .padding(.vertical, DS.Space.xxs)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                .fill(DS.Color.controlFill)
        )
        .fixedSize()
    }

    private func coordLabel(_ axis: String, _ value: Float) -> some View {
        HStack(spacing: DS.Space.xxs) {
            Text(axis)
                .font(DS.Font.caption.weight(.bold))
                .foregroundStyle(.tertiary)
            Text(String(format: "%.3f", value))
                .font(DS.Font.monoSmall)
                .monospacedDigit()
        }
    }

    private func chainAtomCount(_ chain: Chain) -> Int {
        guard let mol = viewModel.molecules.protein else { return 0 }
        return chain.residueIndices.reduce(0) { sum, resIdx in
            guard resIdx < mol.residues.count else { return sum }
            return sum + mol.residues[resIdx].atomIndices.count
        }
    }

    private func showOnlyChain(_ chainID: String) {
        let allIDs = Set(proteinChains.map(\.id))
        viewModel.workspace.hiddenChainIDs = allIDs.subtracting([chainID])
        viewModel.pushToRenderer()
    }
}

// MARK: - Selection shortcuts

/// (?) button in the Inspector header: a popover listing the viewport's selection gestures.
private struct SelectionShortcutsButton: View {
    @State private var isShown = false

    private static let hints: [(keys: [String], description: String)] = [
        (["Click"], "select an item"),
        (["⌥", "Click"], "add to selection"),
        (["⌥", "Drag"], "box-select"),
        (["Double-click"], "select chain"),
        (["Esc"], "clear the selection"),
    ]

    var body: some View {
        Button { isShown.toggle() } label: {
            Label("Selection Shortcuts", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .font(DS.Font.body)
                .foregroundStyle(.secondary)
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help("Selection shortcuts")
        .popover(isPresented: $isShown, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                Text("Selecting in the viewport")
                    .font(DS.Font.sectionTitle)
                Grid(alignment: .leading, horizontalSpacing: DS.Space.m, verticalSpacing: DS.Space.s) {
                    ForEach(Self.hints, id: \.description) { hint in
                        GridRow {
                            HStack(spacing: DS.Space.xxs) {
                                ForEach(Array(hint.keys.enumerated()), id: \.offset) { index, key in
                                    if index > 0 {
                                        Text("+").font(DS.Font.caption).foregroundStyle(.tertiary)
                                    }
                                    keycap(key)
                                }
                            }
                            Text(hint.description)
                                .font(DS.Font.secondary)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(DS.Space.l)
        }
    }

    private func keycap(_ text: String) -> some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
        return Text(text)
            .font(DS.Font.monoSmall.weight(.semibold))
            .padding(.horizontal, DS.Space.xs)
            .padding(.vertical, 1)
            .background(shape.fill(DS.Color.controlFill))
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
            .fixedSize()
    }
}

// MARK: - Residue composition

/// Polymer vs solvent vs HET breakdown of a receptor. `Molecule.residues` holds every
/// residue record (waters, caps, ions), so its raw count overstates the protein length.
@MainActor
private struct ResidueComposition {
    /// Average residue masses (Da) of the 20 standard amino acids, anhydrous (ExPASy ProtParam).
    private static let residueMass: [String: Double] = [
        "ALA": 71.0788, "ARG": 156.1875, "ASN": 114.1038, "ASP": 115.0886, "CYS": 103.1388,
        "GLN": 128.1307, "GLU": 129.1155, "GLY": 57.0519, "HIS": 137.1411, "ILE": 113.1594,
        "LEU": 113.1594, "LYS": 128.1741, "MET": 131.1926, "PHE": 147.1766, "PRO": 97.1167,
        "SER": 87.0782, "THR": 101.1051, "TRP": 186.2132, "TYR": 163.1760, "VAL": 99.1326,
    ]
    private static let waterMass = 18.01524

    let aminoAcids: Int
    let polymerChains: Int
    let waters: Int
    let hetNames: [String]
    /// Sequence mass of the modelled protein (residue masses + one water per chain), the
    /// ProtParam convention: independent of hydrogens, waters and capping groups present.
    let sequenceMass: Float?

    init(_ prot: Molecule) {
        let standard = prot.residues.filter(\.isStandard)
        let proteinChains = Set(standard.map(\.chainID))
        let nucleicIndices = Set(prot.chains.filter { $0.type == .nucleicAcid }.flatMap(\.residueIndices))
        aminoAcids = standard.count
        polymerChains = proteinChains.count + prot.chains.filter { $0.type == .nucleicAcid }.count
        waters = prot.residues.filter(\.isWater).count
        hetNames = Array(Set(prot.residues.enumerated().filter { idx, res in
            !res.isStandard && !res.isWater && !nucleicIndices.contains(idx)
        }.map(\.element.name))).sorted()
        let residueSum = standard.reduce(0.0) { $0 + (Self.residueMass[$1.name] ?? 0) }
        sequenceMass = standard.isEmpty ? nil : Float(residueSum + Self.waterMass * Double(proteinChains.count))
    }
}
