import SwiftUI

// MARK: - Inline Color Swatch Picker

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

/// Chain color picker: CPK / Chain / Custom modes
private struct ChainColorPicker: View {
    var chainID: String
    var paletteColor: SIMD3<Float>
    @Binding var mode: WorkspaceState.ChainColorMode
    @Binding var customColor: SIMD3<Float>?
    var onChange: () -> Void

    @State private var showPopover = false

    var body: some View {
        Button(action: { showPopover.toggle() }) {
            swatchIcon
                .frame(width: 18, height: 18)
        }
        .buttonStyle(.plain)
        .help("Chain color mode")
        .popover(isPresented: $showPopover, arrowEdge: .leading) {
            VStack(alignment: .leading, spacing: 8) {
                // Mode buttons
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

                // Custom color grid
                Text("Custom")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)

                LazyVGrid(columns: Array(repeating: GridItem(.fixed(22), spacing: 4), count: 4), spacing: 4) {
                    ForEach(Array(swatchPresetColors.enumerated()), id: \.offset) { _, preset in
                        let isSelected = mode == .custom && customColor != nil && swatchColorMatches(
                            Color(red: Double(customColor!.x), green: Double(customColor!.y), blue: Double(customColor!.z)),
                            preset
                        )
                        Button(action: {
                            if let comps = NSColor(preset).usingColorSpace(.deviceRGB) {
                                customColor = SIMD3<Float>(Float(comps.redComponent), Float(comps.greenComponent), Float(comps.blueComponent))
                            }
                            mode = .custom
                            onChange()
                            showPopover = false
                        }) {
                            Circle()
                                .fill(preset)
                                .frame(width: 20, height: 20)
                                .overlay(
                                    Circle()
                                        .strokeBorder(Color.primary.opacity(isSelected ? 0.9 : 0), lineWidth: 2)
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(8)
            .frame(width: 120)
        }
    }

    @ViewBuilder
    private var swatchIcon: some View {
        switch mode {
        case .cpk:
            // Multi-color CPK indicator
            ZStack {
                Circle().fill(Color(red: 0.5, green: 0.5, blue: 0.5)).frame(width: 16, height: 16) // C gray
                Circle().fill(Color(red: 0.0, green: 0.4, blue: 1.0)).frame(width: 8, height: 8).offset(x: -3, y: -3) // N blue
                Circle().fill(Color(red: 1.0, green: 0.2, blue: 0.2)).frame(width: 6, height: 6).offset(x: 3, y: 3) // O red
            }
            .clipShape(Circle())
            .overlay(Circle().strokeBorder(Color.primary.opacity(0.2), lineWidth: 0.5))
        case .chainDefault:
            Circle()
                .fill(Color(red: Double(paletteColor.x), green: Double(paletteColor.y), blue: Double(paletteColor.z)))
                .frame(width: 16, height: 16)
                .overlay(Circle().strokeBorder(Color.primary.opacity(0.2), lineWidth: 0.5))
        case .custom:
            let c = customColor ?? paletteColor
            Circle()
                .fill(Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z)))
                .frame(width: 16, height: 16)
                .overlay(Circle().strokeBorder(Color.primary.opacity(0.2), lineWidth: 0.5))
        }
    }

    private func modeButton(_ title: String, icon: String, isActive: Bool, swatchColor: SIMD3<Float>? = nil, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let sc = swatchColor {
                    Circle()
                        .fill(Color(red: Double(sc.x), green: Double(sc.y), blue: Double(sc.z)))
                        .frame(width: 10, height: 10)
                } else {
                    Image(systemName: icon)
                        .font(.footnote)
                        .frame(width: 10)
                }
                Text(title)
                    .font(.footnote.weight(isActive ? .semibold : .regular))
                Spacer()
                if isActive {
                    Image(systemName: "checkmark")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 4)
            .background(isActive ? Color.accentColor.opacity(0.1) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
        .buttonStyle(.plain)
    }
}

/// Simple color swatch picker (for ligand carbon color)
private struct InlineColorPicker: View {
    @Binding var color: Color
    var label: String = ""
    var size: CGFloat = 16
    var onReset: (() -> Void)? = nil

    @State private var showPopover = false

    var body: some View {
        Button(action: { showPopover.toggle() }) {
            Circle()
                .fill(color)
                .frame(width: size, height: size)
                .overlay(
                    Circle()
                        .strokeBorder(Color.primary.opacity(0.2), lineWidth: 0.5)
                )
        }
        .buttonStyle(.plain)
        .help(label)
        .popover(isPresented: $showPopover, arrowEdge: .leading) {
            VStack(spacing: 8) {
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(22), spacing: 4), count: 4), spacing: 4) {
                    ForEach(Array(swatchPresetColors.enumerated()), id: \.offset) { _, preset in
                        Button(action: {
                            color = preset
                            showPopover = false
                        }) {
                            Circle()
                                .fill(preset)
                                .frame(width: 20, height: 20)
                                .overlay(
                                    Circle()
                                        .strokeBorder(Color.primary.opacity(swatchColorMatches(color, preset) ? 0.9 : 0), lineWidth: 2)
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
                if let onReset {
                    Divider()
                    Button(action: {
                        onReset()
                        showPopover = false
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.counterclockwise")
                                .font(.footnote)
                            Text("Default")
                                .font(.footnote)
                        }
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 2)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
        }
    }
}

struct InspectorPanel: View {
    @Environment(AppViewModel.self) private var viewModel
    @Binding var showInspector: Bool

    private var selectionMode: WorkspaceState.SelectionMode {
        get { viewModel.workspace.selectionMode }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header with close button
            HStack {
                Image(systemName: "sidebar.right")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Text("Inspector")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Button(action: { withAnimation(.easeInOut(duration: 0.15)) { showInspector = false } }) {
                    Image(systemName: "xmark")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 18, height: 18)
                        .background(Color.primary.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
                .buttonStyle(.plain)
                .help("Hide inspector")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    selectionSection

                    Divider()

                    chainVisibilitySection

                    Divider()

                    residueSubsetsSection

                    Divider()

                    statisticsSection
                }
                .padding(12)
            }
        }
        .frame(width: 260)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Selection Section

    @ViewBuilder
    private var selectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Mode toggle
            HStack {
                sectionHeader("Selection", icon: "cursorarrow.click.2")
                Spacer()
                Picker("", selection: Binding(
                    get: { viewModel.workspace.selectionMode },
                    set: { viewModel.workspace.selectionMode = $0 }
                )) {
                    ForEach([WorkspaceState.SelectionMode.residue, .atom], id: \.self) { mode in
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
        let selectedCount = viewModel.workspace.selectedAtomIndices.count

        if selectedCount == 0 {
            ContentUnavailableView {
                Label("No Selection", systemImage: "atom")
            } description: {
                Text("Click to select \u{2022} \u{2325}Click to multi-select \u{2022} \u{2325}Drag to box-select \u{2022} Double-click to select chain")
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
        let selectedCount = viewModel.workspace.selectedResidueIndices.count

        if selectedCount == 0 {
            ContentUnavailableView {
                Label("No Selection", systemImage: "rectangle.stack")
            } description: {
                Text("Click to select \u{2022} \u{2325}Click to multi-select \u{2022} \u{2325}Drag to box-select \u{2022} Double-click to select chain")
            }
            .frame(height: 100)
        } else if selectedCount == 1, let prot = viewModel.molecules.protein {
            // Single residue detail
            let resIdx = viewModel.workspace.selectedResidueIndices.first!
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
                    .font(.body.monospaced().weight(.bold))
                    .foregroundStyle(.white)
                    .shadow(radius: 1)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(atom.element.name)
                    .font(.headline)
                Text(atom.name)
                    .font(.subheadline.monospaced())
                    .foregroundStyle(.secondary)
            }
        }

        infoRow("Residue", "\(atom.residueName) \(atom.residueSeq)")
        infoRow("Chain", atom.chainID)
        if atom.formalCharge != 0 {
            infoRow("Formal charge", "\(atom.formalCharge > 0 ? "+" : "")\(atom.formalCharge)")
        }
        infoRow("Partial charge", String(format: "%.3f", atom.charge))
        infoRow("VdW Radius", String(format: "%.2f \u{00C5}", atom.element.vdwRadius))
        if atom.tempFactor > 0 {
            infoRow("B-factor", String(format: "%.1f", atom.tempFactor))
        }
        if atom.occupancy < 1.0 {
            infoRow("Occupancy", String(format: "%.2f", atom.occupancy))
        }

        // Connected bonds
        let bonds = connectedBonds(for: atom)
        if !bonds.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                Text("Bonds (\(bonds.count))")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                ForEach(bonds, id: \.id) { bond in
                    let allAtoms = (viewModel.molecules.protein?.atoms ?? []) + (viewModel.molecules.ligand?.atoms ?? [])
                    let otherIdx = bond.atomIndex1 == atom.id ? bond.atomIndex2 : bond.atomIndex1
                    if otherIdx < allAtoms.count {
                        let other = allAtoms[otherIdx]
                        HStack(spacing: 4) {
                            Circle()
                                .fill(Color(red: Double(other.element.color.x),
                                            green: Double(other.element.color.y),
                                            blue: Double(other.element.color.z)))
                                .frame(width: 6, height: 6)
                            Text("\(other.element.symbol) \(other.name)")
                                .font(.footnote.monospaced())
                            Spacer()
                            Text(bond.order == .aromatic ? "arom" : "\(bond.order.rawValue)\u{00D7}")
                                .font(.footnote.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }

        VStack(alignment: .leading, spacing: 3) {
            Text("Position")
                .font(.footnote.weight(.medium))
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                coordLabel("X", atom.position.x)
                coordLabel("Y", atom.position.y)
                coordLabel("Z", atom.position.z)
            }
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

        // Element distribution
        let elementCounts = Dictionary(grouping: selected, by: \.element.symbol)
            .mapValues(\.count)
            .sorted { $0.value > $1.value }

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "atom")
                    .font(.body)
                    .foregroundStyle(.cyan)
                Text("\(count) atoms selected")
                    .font(.headline)
            }

            // Element breakdown
            HStack(spacing: 4) {
                ForEach(elementCounts.prefix(6), id: \.key) { symbol, cnt in
                    Text("\(symbol):\(cnt)")
                        .font(.footnote.monospaced())
                        .padding(.horizontal, 4)
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

            // Selection actions
            selectionActions
        }
    }

    // MARK: - Single Residue Inspector

    @ViewBuilder
    private func singleResidueInspector(_ residue: Residue, in prot: Molecule) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(residue.name)
                    .font(.title3.monospaced().weight(.bold))
                Text("#\(residue.sequenceNumber)")
                    .font(.callout.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("Chain \(residue.chainID)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            // Secondary structure
            let ss = secondaryStructure(for: residue, in: prot)
            if let ss {
                HStack(spacing: 4) {
                    Image(systemName: ss == .helix ? "arrow.trianglehead.turn.up.right.diamond"
                          : ss == .sheet ? "arrow.right.to.line" : "scribble")
                        .font(.footnote)
                        .foregroundStyle(ss == .helix ? .red : ss == .sheet ? .cyan : .secondary)
                    Text(ss == .helix ? "Alpha Helix" : ss == .sheet ? "Beta Sheet"
                         : ss == .turn ? "Turn" : "Coil")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(ss == .helix ? .red : ss == .sheet ? .cyan : .secondary)
                }
            }

            infoRow("Atoms", "\(residue.atomIndices.count)")
            let heavyCount = residue.atomIndices.filter { idx in
                idx < prot.atoms.count && prot.atoms[idx].element != .H
            }.count
            infoRow("Heavy atoms", "\(heavyCount)")

            // Element composition
            let elements = Dictionary(grouping: residue.atomIndices.compactMap { idx in
                idx < prot.atoms.count ? prot.atoms[idx].element : nil
            }, by: \.symbol).mapValues(\.count).sorted { $0.value > $1.value }
            HStack(spacing: 4) {
                ForEach(elements, id: \.key) { symbol, cnt in
                    Text("\(symbol):\(cnt)")
                        .font(.footnote.monospaced())
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.primary.opacity(0.05))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
            }

            // Total charge of residue
            let totalCharge = residue.atomIndices.compactMap { idx in
                idx < prot.atoms.count ? prot.atoms[idx].charge : nil
            }.reduce(Float(0), +)
            infoRow("Net charge", String(format: "%.2f", totalCharge))

            // Average B-factor
            let bFactors = residue.atomIndices.compactMap { idx in
                idx < prot.atoms.count ? prot.atoms[idx].tempFactor : nil
            }
            if let avgB = bFactors.isEmpty ? nil : bFactors.reduce(0, +) / Float(bFactors.count), avgB > 0 {
                infoRow("Avg B-factor", String(format: "%.1f", avgB))
            }

            selectionActions
        }
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

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "rectangle.stack")
                    .font(.body)
                    .foregroundStyle(.cyan)
                Text("\(count) residues selected")
                    .font(.headline)
            }

            // Residue type breakdown
            let typeCounts = Dictionary(grouping: residues, by: \.name)
                .mapValues(\.count)
                .sorted { $0.value > $1.value }

            HStack(spacing: 4) {
                ForEach(typeCounts.prefix(8), id: \.key) { name, cnt in
                    Text("\(name):\(cnt)")
                        .font(.footnote.monospaced())
                        .padding(.horizontal, 4)
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

            // Selection actions
            selectionActions
        }
        }
    }

    // MARK: - Selection Actions (shared between atom & residue summaries)

    @ViewBuilder
    private var selectionActions: some View {
        let hasResidues = !viewModel.workspace.selectedResidueIndices.isEmpty

        VStack(spacing: 6) {
            // Extend nearby
            HStack(spacing: 6) {
                Menu {
                    ForEach([5, 6, 8, 10] as [Float], id: \.self) { d in
                        Button(String(format: "%.0f \u{00C5}", d)) {
                            viewModel.selectResiduesWithinDistance(d)
                        }
                    }
                } label: {
                    Label("Extend Nearby", systemImage: "circle.dashed")
                        .frame(maxWidth: .infinity)
                }
                .menuStyle(.borderlessButton)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
                .background(Color.primary.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .controlSize(.small)

                Button(action: { viewModel.extendSelectionByOneResidue() }) {
                    Label("+1 Res", systemImage: "arrow.left.and.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!hasResidues)
            }

            HStack(spacing: 6) {
                Button(action: { viewModel.invertSelection() }) {
                    Label("Invert", systemImage: "arrow.triangle.2.circlepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(action: { viewModel.createSubsetFromSelection() }) {
                    Label("Subset", systemImage: "plus.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!hasResidues)
            }

            HStack(spacing: 6) {
                Button(action: { viewModel.definePocketFromSelection() }) {
                    Label("Define Pocket", systemImage: "scope")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!hasResidues)

                Button(action: { viewModel.deselectAll() }) {
                    Label("Clear", systemImage: "xmark.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Chain Visibility

    @ViewBuilder
    private var chainVisibilitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Chains & Display", icon: "link")

            if viewModel.allChains.isEmpty && viewModel.molecules.ligand == nil {
                Text("No chains loaded")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(viewModel.allChains.enumerated()), id: \.element.id) { chainIndex, chain in
                    let atomCount = chainAtomCount(chain)
                    let palette = WorkspaceState.MoleculeColorScheme.chainPalette
                    let paletteColor = palette[chainIndex % palette.count]
                    HStack(spacing: 6) {
                        ChainColorPicker(
                            chainID: chain.id,
                            paletteColor: paletteColor,
                            mode: Binding(
                                get: { viewModel.workspace.chainColorModes[chain.id] ?? .chainDefault },
                                set: { viewModel.workspace.chainColorModes[chain.id] = $0 }
                            ),
                            customColor: Binding(
                                get: { viewModel.workspace.chainColorOverrides[chain.id] },
                                set: { viewModel.workspace.chainColorOverrides[chain.id] = $0 }
                            ),
                            onChange: { viewModel.pushToRenderer() }
                        )

                        Text("\(chain.id)")
                            .font(.callout.weight(.medium))
                            .fixedSize()

                        Text("\(chain.type.label) \(atomCount)")
                            .font(.footnote.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)

                        Spacer(minLength: 2)

                        if chain.type != .protein {
                            Button(action: { viewModel.extractChainAsLigand(chainID: chain.id) }) {
                                Image(systemName: "arrow.right.circle")
                                    .font(.footnote)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.orange)
                            .help("Extract as ligand")
                        }

                        Button(action: { viewModel.removeChain(chainID: chain.id) }) {
                            Image(systemName: "trash")
                                .font(.footnote)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.red.opacity(0.6))
                        .help("Remove chain")

                        Button(action: { showOnlyChain(chain.id) }) {
                            Image(systemName: "eye")
                                .font(.footnote)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Show only this chain")

                        Toggle("", isOn: Binding(
                            get: { !viewModel.workspace.hiddenChainIDs.contains(chain.id) },
                            set: { _ in viewModel.toggleChainVisibility(chain.id) }
                        ))
                        .toggleStyle(.switch)
                        .controlSize(.mini)
                        .fixedSize()
                    }
                }

                if !viewModel.workspace.hiddenChainIDs.isEmpty {
                    Text("Hidden chains excluded from pocket detection")
                        .font(.footnote)
                        .foregroundStyle(.orange.opacity(0.8))
                        .padding(.top, 2)
                }

                if let lig = viewModel.molecules.ligand {
                    Divider().padding(.vertical, 2)
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 8) {
                            // Ligand carbon color picker (with CPK reset)
                            InlineColorPicker(
                                color: Binding(
                                    get: {
                                        if let lc = viewModel.workspace.ligandCarbonColor {
                                            return Color(red: Double(lc.x), green: Double(lc.y), blue: Double(lc.z))
                                        }
                                        return Color.orange
                                    },
                                    set: { newColor in
                                        if let comps = NSColor(newColor).usingColorSpace(.deviceRGB) {
                                            viewModel.workspace.ligandCarbonColor = SIMD3<Float>(
                                                Float(comps.redComponent),
                                                Float(comps.greenComponent),
                                                Float(comps.blueComponent)
                                            )
                                            viewModel.pushToRenderer()
                                        }
                                    }
                                ),
                                label: "Set ligand carbon color",
                                onReset: {
                                    viewModel.workspace.ligandCarbonColor = nil
                                    viewModel.pushToRenderer()
                                }
                            )

                            Text("Ligand")
                                .font(.callout.weight(.medium))
                            Spacer()
                            Text(lig.name)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }

                        // Ligand render mode picker
                        HStack(spacing: 4) {
                            Text("Render")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Picker("", selection: Binding(
                                get: { viewModel.workspace.effectiveLigandRenderMode },
                                set: { newMode in
                                    viewModel.setLigandRenderMode(newMode)
                                }
                            )) {
                                Text("Stick").tag(RenderMode.wireframe)
                                Text("B&S").tag(RenderMode.ballAndStick)
                                Text("CPK").tag(RenderMode.spaceFilling)
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 150)
                        }

                        HStack(spacing: 6) {
                            if lig.atoms.first?.isHetAtom == true && lig.smiles == nil {
                                Button(action: { viewModel.definePocketFromLigand() }) {
                                    Image(systemName: "scope")
                                        .font(.footnote)
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.green)
                                .help("Define pocket from ligand")
                            }

                            Spacer()

                            Button(action: { viewModel.removeLigandFromView() }) {
                                Image(systemName: "trash")
                                    .font(.footnote)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.red.opacity(0.6))
                            .help("Remove ligand")

                            Toggle("", isOn: Binding(
                                get: { viewModel.workspace.showLigand },
                                set: { newValue in
                                    viewModel.workspace.showLigand = newValue
                                    viewModel.pushToRenderer()
                                }
                            ))
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                            .fixedSize()
                        }
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
                        .font(.body)
                        .padding(4)
                        .background(Color.accentColor.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .disabled(viewModel.workspace.selectedResidueIndices.isEmpty)
                .help("Create subset from selected residues")
            }

            if viewModel.workspace.residueSubsets.isEmpty {
                Text("Select residues and click + to create a subset")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.workspace.residueSubsets) { subset in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color(
                                red: Double(subset.color.x),
                                green: Double(subset.color.y),
                                blue: Double(subset.color.z)
                            ))
                            .frame(width: 8, height: 8)
                        Text(subset.name)
                            .font(.subheadline.weight(.medium))
                            .lineLimit(1)
                        Text("\(subset.residueIndices.count)")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: { viewModel.selectSubset(id: subset.id) }) {
                            Image(systemName: "selection.pin.in.out")
                                .font(.subheadline)
                                .padding(4)
                                .background(Color.primary.opacity(0.05))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        .buttonStyle(.plain)
                        .help("Select all residues in this subset")
                        Button(action: { viewModel.definePocketFromSubset(id: subset.id) }) {
                            Image(systemName: "cube.transparent")
                                .font(.subheadline)
                                .padding(4)
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
                                .font(.subheadline)
                                .foregroundStyle(.red)
                                .padding(4)
                                .background(Color.red.opacity(0.06))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        .buttonStyle(.plain)
                        .help("Delete this subset")
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

            if let prot = viewModel.molecules.protein {
                Text("Protein: \(prot.name)")
                    .font(.subheadline.weight(.medium))
                infoRow("Atoms", "\(prot.atomCount)")
                infoRow("Heavy", "\(prot.heavyAtomCount)")
                infoRow("Bonds", "\(prot.bondCount)")
                infoRow("Residues", "\(prot.residues.count)")
                infoRow("Chains", "\(prot.chains.count)")
                infoRow("MW", String(format: "%.1f Da", prot.molecularWeight))
            }

            if let lig = viewModel.molecules.ligand {
                if viewModel.molecules.protein != nil { Divider() }
                Text("Ligand: \(lig.name)")
                    .font(.subheadline.weight(.medium))
                infoRow("Atoms", "\(lig.atomCount)")
                infoRow("Heavy", "\(lig.heavyAtomCount)")
                infoRow("Bonds", "\(lig.bondCount)")
                infoRow("MW", String(format: "%.1f Da", lig.molecularWeight))

                // Conformer picker
                let confs = viewModel.workspace.ligandConformers
                if confs.count > 1 {
                    Divider()
                    HStack(spacing: 8) {
                        Text("Conformer")
                            .font(.footnote.weight(.medium))
                        Spacer()
                        Button(action: {
                            let prev = max(viewModel.workspace.activeConformerIndex - 1, 0)
                            viewModel.switchConformer(to: prev)
                        }) {
                            Image(systemName: "chevron.left")
                                .font(.footnote)
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.workspace.activeConformerIndex == 0)
                        .help("Previous conformer")

                        Text("\(viewModel.workspace.activeConformerIndex + 1) / \(confs.count)")
                            .font(.footnote.monospaced().weight(.medium))

                        Button(action: {
                            let next = min(viewModel.workspace.activeConformerIndex + 1, confs.count - 1)
                            viewModel.switchConformer(to: next)
                        }) {
                            Image(systemName: "chevron.right")
                                .font(.footnote)
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.workspace.activeConformerIndex >= confs.count - 1)
                        .help("Next conformer")
                    }

                    let active = confs[viewModel.workspace.activeConformerIndex]
                    infoRow("Energy", String(format: "%.2f kcal/mol", active.energy))
                }
            }

            if viewModel.molecules.protein == nil && viewModel.molecules.ligand == nil {
                Text("No molecules loaded")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Helpers

    private func sectionHeader(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.callout.weight(.semibold))
            .foregroundStyle(.primary)
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .leading)
            Text(value)
                .font(.subheadline.monospaced())
                .foregroundStyle(.primary)
            Spacer()
        }
    }

    private func coordLabel(_ axis: String, _ value: Float) -> some View {
        HStack(spacing: 2) {
            Text(axis)
                .font(.footnote.weight(.bold))
                .foregroundStyle(.secondary)
            Text(String(format: "%7.3f", value))
                .font(.footnote.monospaced())
                .foregroundStyle(.primary)
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
        let allIDs = Set(viewModel.allChains.map(\.id))
        viewModel.workspace.hiddenChainIDs = allIDs.subtracting([chainID])
        viewModel.pushToRenderer()
    }
}
