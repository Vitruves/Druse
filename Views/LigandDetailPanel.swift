import SwiftUI
import AppKit
import MetalKit

// MARK: - LigandDatabaseWindow Detail Panel Extension
// Bottom detail area: 2D/3D structure preview, properties/prepare/variants/conformers tabs,
// conformer carousel, variant generation, conformer generation, and helper actions.

extension LigandDatabaseWindow {

    // MARK: - Bottom Detail Area (HSplitView: 2D/3D preview | tabbed options)

    @ViewBuilder
    func bottomDetailArea(_ entry: LigandEntry) -> some View {
        HSplitView {
            // LEFT: Structure preview (2D or 3D)
            structurePreviewPanel(entry)
                .frame(minWidth: 300, idealWidth: 450)

            // RIGHT: Tabbed options panel
            VStack(spacing: 0) {
                Picker("", selection: $bottomTab) {
                    ForEach(BottomTab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)

                Divider()

                ScrollView {
                    switch bottomTab {
                    case .properties:
                        propertiesTab(entry)
                    case .prepare:
                        prepareTab(entry)
                    case .variants:
                        variantsTab(entry)
                    case .conformers:
                        conformersTab(entry)
                    }
                }
            }
            .frame(minWidth: 320, idealWidth: 400)
        }
    }

    // MARK: - Structure Preview Panel (2D/3D Toggle)

    @ViewBuilder
    private func structurePreviewPanel(_ entry: LigandEntry) -> some View {
        VStack(spacing: 0) {
            // Header: name + prepared badge + 2D/3D toggle
            HStack(spacing: 6) {
                Text(entry.name)
                    .font(.system(size: 12, weight: .semibold))
                if entry.isPrepared {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.system(size: 11))
                }
                Spacer()

                // 2D / 3D mode picker
                Picker("", selection: $show3DPreview) {
                    Image(systemName: "square.grid.2x2").tag(false)
                    Image(systemName: "cube").tag(true)
                }
                .pickerStyle(.segmented)
                .frame(width: 70)
                .controlSize(.mini)
                .help("Toggle 2D depiction / 3D ball-and-stick")

                Text(entry.smiles.prefix(30) + (entry.smiles.count > 30 ? "..." : ""))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            // Structure visualization
            if show3DPreview {
                // 3D Metal viewport — full rotate/pan/zoom
                if let renderer = miniRenderer {
                    MetalView(renderer: renderer)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if entry.atoms.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "cube")
                            .font(.system(size: 28))
                            .foregroundStyle(.tertiary)
                        Text("Prepare ligand for 3D view")
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ProgressView("Initializing 3D...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .onAppear { initMiniRenderer(atoms: entry.atoms, bonds: entry.bonds) }
                }
            } else {
                // 2D RDKit depiction
                if isComputing2D {
                    ProgressView("Computing 2D layout...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let coords = ligand2DCoords {
                    Canvas { context, size in
                        draw2DStructure(context: context, size: size, coords: coords)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: .controlBackgroundColor))
                } else {
                    VStack(spacing: 6) {
                        Image(systemName: "hexagon")
                            .font(.system(size: 28))
                            .foregroundStyle(.tertiary)
                        Text("No 2D structure")
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }

            // Hint: 2D can't show conformer differences
            if !conformers.isEmpty && !show3DPreview {
                Text("Switch to 3D to see conformer spatial differences")
                    .font(.system(size: 8))
                    .foregroundStyle(.quaternary)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 2)
            }

            // Conformer carousel (when conformers exist)
            if !conformers.isEmpty {
                Divider()
                HStack(spacing: 8) {
                    Button(action: {
                        selectedConformerIndex = max(0, selectedConformerIndex - 1)
                    }) {
                        Image(systemName: "chevron.left")
                    }
                    .buttonStyle(.plain)
                    .disabled(selectedConformerIndex == 0)

                    Text("Conformer \(selectedConformerIndex + 1) / \(conformers.count)")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))

                    Button(action: {
                        selectedConformerIndex = min(conformers.count - 1, selectedConformerIndex + 1)
                    }) {
                        Image(systemName: "chevron.right")
                    }
                    .buttonStyle(.plain)
                    .disabled(selectedConformerIndex >= conformers.count - 1)

                    Spacer()

                    if selectedConformerIndex < conformers.count {
                        Text(String(format: "%.1f kcal/mol", conformers[selectedConformerIndex].energy))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .onChange(of: selectedConformerIndex) { _, newIdx in
                    guard newIdx >= 0, newIdx < conformers.count else { return }
                    let conf = conformers[newIdx]
                    // Update inspected entry atoms/bonds for downstream consumers
                    if var updated = inspectedEntry {
                        updated.atoms = conf.molecule.atoms
                        updated.bonds = conf.molecule.bonds
                        inspectedEntry = updated
                    }
                    // Update 3D viewport with this conformer's geometry
                    if show3DPreview {
                        updateMiniRenderer(atoms: conf.molecule.atoms, bonds: conf.molecule.bonds)
                    }
                }
            }
        }
        .onChange(of: show3DPreview) { _, is3D in
            if is3D, let entry = inspectedEntry, !entry.atoms.isEmpty {
                if miniRenderer == nil {
                    initMiniRenderer(atoms: entry.atoms, bonds: entry.bonds)
                } else {
                    updateMiniRenderer(atoms: entry.atoms, bonds: entry.bonds)
                }
            }
        }
    }

    // MARK: - Properties Tab (descriptors only)

    @ViewBuilder
    func propertiesTab(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let desc = entry.descriptors {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    GridRow {
                        Text("MW").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.1f Da", desc.molecularWeight)).font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("cLogP").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.2f", desc.logP)).font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("TPSA").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.1f \u{00C5}\u{00B2}", desc.tpsa)).font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("HBD / HBA").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text("\(desc.hbd) / \(desc.hba)").font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("RotBonds").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text("\(desc.rotatableBonds)").font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("Rings").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text("\(desc.rings) (\(desc.aromaticRings) arom)").font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("Fsp3").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.2f", desc.fractionCSP3)).font(.system(size: 11, design: .monospaced))
                    }
                    GridRow {
                        Text("Heavy atoms").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                        Text("\(desc.heavyAtomCount)").font(.system(size: 11, design: .monospaced))
                    }
                }

                Divider()

                HStack(spacing: 8) {
                    ruleBadge("Lipinski", passed: desc.lipinski)
                    ruleBadge("Veber", passed: desc.veber)
                }
            } else {
                Text("Prepare this ligand to compute properties")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }

            // Binding affinity (if available)
            if entry.ki != nil || entry.pKi != nil || entry.ic50 != nil {
                Divider()
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    if let ki = entry.ki {
                        GridRow {
                            Text("Ki").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                            Text(String(format: "%.2f nM", ki)).font(.system(size: 11, design: .monospaced))
                        }
                    }
                    if let pKi = entry.pKi {
                        GridRow {
                            Text("pKi").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                            Text(String(format: "%.2f", pKi)).font(.system(size: 11, design: .monospaced))
                        }
                    }
                    if let ic50 = entry.ic50 {
                        GridRow {
                            Text("IC50").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                            Text(String(format: "%.2f nM", ic50)).font(.system(size: 11, design: .monospaced))
                        }
                    }
                }
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private func ruleBadge(_ name: String, passed: Bool) -> some View {
        Text(name).font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Capsule().fill(passed ? Color.green.opacity(0.2) : Color.red.opacity(0.15)))
            .foregroundStyle(passed ? .green : .red)
            .strikethrough(!passed)
    }

    // MARK: - Prepare Tab

    @ViewBuilder
    func prepareTab(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // Quick single-structure preparation
            Label("Quick Prepare", systemImage: "wand.and.stars")
                .font(.system(size: 11, weight: .semibold))

            Toggle("Add hydrogens", isOn: $prepAddHydrogens)
                .toggleStyle(.switch).controlSize(.small)
            Toggle("Energy minimize (MMFF94)", isOn: $prepMinimize)
                .toggleStyle(.switch).controlSize(.small)
            Toggle("Compute Gasteiger charges", isOn: $prepComputeCharges)
                .toggleStyle(.switch).controlSize(.small)

            Button(action: {
                if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                    db.prepareEntry(at: idx, addH: prepAddHydrogens, minimize: prepMinimize, charges: prepComputeCharges)
                    inspectedEntry = db.entries[idx]
                    compute2DPreview(smiles: entry.smiles)
                    if show3DPreview, let updated = inspectedEntry {
                        updateMiniRenderer(atoms: updated.atoms, bonds: updated.bonds)
                    }
                }
            }) {
                Label("Prepare", systemImage: "wand.and.stars")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            if entry.isPrepared {
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                    Text("Prepared").font(.system(size: 10, weight: .medium)).foregroundStyle(.green)
                    if let date = entry.preparationDate {
                        Text(date.formatted(date: .abbreviated, time: .shortened))
                            .font(.system(size: 9)).foregroundStyle(.tertiary)
                    }
                }
            }

            Divider()

            // Full ensemble preparation (the main workflow)
            Label("Full Ensemble Preparation", systemImage: "sparkles")
                .font(.system(size: 11, weight: .semibold))

            Text("Generates all chemically probable forms at target pH: protomers, tautomers, and conformers. Each form is fully prepared (polar H, MMFF94 minimized, Gasteiger charges). Boltzmann population weights are assigned based on MMFF94 energy.")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                Text("Target pH").font(.system(size: 10))
                Slider(value: $variantPH, in: 1...14, step: 0.1).controlSize(.mini)
                Text(String(format: "%.1f", variantPH)).font(.system(size: 10, design: .monospaced)).frame(width: 30)
            }
            HStack {
                Text("Conformers/form").font(.system(size: 10))
                Spacer()
                Stepper("\(conformerBudgetPerVariant)", value: $conformerBudgetPerVariant, in: 1...50, step: 1)
                    .controlSize(.small)
            }
            HStack {
                Text("Energy cutoff").font(.system(size: 10))
                Slider(value: $variantEnergyCutoff, in: 5...50, step: 1).controlSize(.mini)
                Text(String(format: "%.0f kcal", variantEnergyCutoff)).font(.system(size: 10, design: .monospaced)).frame(width: 50)
            }

            Button(action: { runEnsemblePreparation(entry) }) {
                Label(isGeneratingVariants ? "Preparing ensemble..." : "Prepare Full Ensemble",
                      systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(entry.smiles.isEmpty || isGeneratingVariants || entry.parentID != nil)

            if isGeneratingVariants {
                ProgressView("Running protomer x tautomer x conformer pipeline...")
                    .controlSize(.small)
            }

            // Show ensemble summary if children exist
            let children = db.children(of: entry.id)
            if !children.isEmpty {
                Divider()
                let nForms = Set(children.map(\.smiles)).count
                let nTotal = children.count
                HStack(spacing: 8) {
                    Label("\(nForms) forms, \(nTotal) entries", systemImage: "checkmark.circle.fill")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.green)
                }
                Text("Browse in Variants tab. Click entries to inspect in 2D/3D.")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
    }

    // MARK: - Variants Tab (ensemble browser with energy + population + manual curation)

    @ViewBuilder
    func variantsTab(_ entry: LigandEntry) -> some View {
        // When inspecting a child variant, show siblings (children of parent)
        let effectiveParentID = entry.parentID ?? entry.id
        let isChildEntry = entry.parentID != nil
        let children = db.children(of: effectiveParentID)

        VStack(alignment: .leading, spacing: 10) {
            Label("Tautomers & Protomers", systemImage: "arrow.triangle.2.circlepath")
                .font(.system(size: 11, weight: .semibold))

            if isChildEntry {
                Text("Viewing variant of \(db.parent(of: entry)?.name ?? "parent"). Generate from the parent entry.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text("pH").font(.system(size: 10))
                Slider(value: $variantPH, in: 1...14, step: 0.1).controlSize(.mini)
                Text(String(format: "%.1f", variantPH)).font(.system(size: 10, design: .monospaced)).frame(width: 30)
            }
            HStack {
                Text("Max tautomers").font(.system(size: 10))
                Spacer()
                Stepper("\(variantMaxTautomers)", value: $variantMaxTautomers, in: 1...50).controlSize(.small)
            }
            HStack {
                Text("Max protomers").font(.system(size: 10))
                Spacer()
                Stepper("\(variantMaxProtomers)", value: $variantMaxProtomers, in: 1...20).controlSize(.small)
            }
            HStack {
                Text("Energy cutoff").font(.system(size: 10))
                Slider(value: $variantEnergyCutoff, in: 1...50, step: 1).controlSize(.mini)
                Text(String(format: "%.0f kcal", variantEnergyCutoff)).font(.system(size: 10, design: .monospaced)).frame(width: 50)
            }

            HStack(spacing: 6) {
                Button(action: { generateTautomers(entry) }) {
                    Label("Generate Tautomers", systemImage: "arrow.triangle.2.circlepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(entry.smiles.isEmpty || isGeneratingVariants || isChildEntry)

                Button(action: { generateProtomers(entry) }) {
                    Label("Generate Protomers", systemImage: "drop.triangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(entry.smiles.isEmpty || isGeneratingVariants || isChildEntry)
            }

            if isGeneratingVariants {
                ProgressView("Generating variants...").controlSize(.small)
            }

            // Variant list (clickable to inspect)
            if !children.isEmpty {
                Divider()
                Text("\(children.count) variants")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                ForEach(children) { child in
                    variantRow(child)
                }
            } else if !isChildEntry {
                Text("Generate tautomers or protomers to see variants here")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private func variantRow(_ child: LigandEntry) -> some View {
        let isSelected = selectedVariantID == child.id
        let allChildren = db.children(of: child.parentID ?? child.id)
        let minE = allChildren.compactMap(\.relativeEnergy).min() ?? 0

        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 5) {
                // Kind badge
                Text(child.variantKind == .tautomer ? "T" : "P")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 14, height: 14)
                    .background(Circle().fill(child.variantKind == .tautomer ? Color.purple : Color.cyan))

                // Label
                Text(child.variantLineage ?? child.name)
                    .font(.system(size: 10, weight: isSelected ? .semibold : .medium))
                    .lineLimit(1)

                Spacer()

                // Energy (absolute)
                if let energy = child.relativeEnergy {
                    let dE = energy - minE
                    Text(String(format: "%.1f kcal/mol", energy))
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(dE < 1 ? .green : dE < 5 ? .yellow : dE < 15 ? .orange : .red)
                }

                // Action buttons
                Button(action: { useVariantForDocking(child) }) {
                    Image(systemName: "arrow.right.circle").font(.system(size: 10))
                }
                .buttonStyle(.plain).foregroundStyle(.tint)
                .help("Use for docking")
                .disabled(child.atoms.isEmpty)

                Button(action: { deleteVariant(child.id) }) {
                    Image(systemName: "xmark.circle").font(.system(size: 10))
                }
                .buttonStyle(.plain).foregroundStyle(.red.opacity(0.7))
                .help("Remove from ensemble (chemically improbable or unwanted)")
            }

            // SMILES preview
            Text(child.smiles)
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 4)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture { inspectVariantChild(child) }
    }

    // MARK: - Conformers Tab

    @ViewBuilder
    func conformersTab(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Conformer Browser", systemImage: "cube.transparent")
                .font(.system(size: 11, weight: .semibold))

            HStack {
                Text("Max conformers")
                    .font(.system(size: 10))
                Spacer()
                Stepper("\(conformerBudgetPerVariant)", value: $conformerBudgetPerVariant, in: 5...200, step: 5)
                    .controlSize(.small)
            }

            Button(action: { generateConformersForEntry(entry) }) {
                Label(isGeneratingConformers ? "Generating..." : "Generate Conformers",
                      systemImage: "cube.transparent")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.small)
            .disabled(entry.smiles.isEmpty || isGeneratingConformers)

            if !conformers.isEmpty {
                Divider()
                Text("\(conformers.count) conformers (sorted by MMFF94 energy)")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                ForEach(conformers) { conf in
                    let isActive = conf.id == selectedConformerIndex
                    HStack(spacing: 6) {
                        Text("#\(conf.id + 1)")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundStyle(isActive ? .primary : .secondary)
                            .frame(width: 28, alignment: .trailing)
                        Text(String(format: "%.2f kcal/mol", conf.energy))
                            .font(.system(size: 10, design: .monospaced))
                        Spacer()
                        let minE = conformers.map(\.energy).min() ?? 0
                        let maxE = conformers.map(\.energy).max() ?? 1
                        let range = max(maxE - minE, 0.1)
                        let frac = (conf.energy - minE) / range
                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 2)
                                .fill(frac < 0.3 ? Color.green : frac < 0.7 ? .yellow : .orange)
                                .frame(width: max(4, geo.size.width * (1 - frac)))
                        }
                        .frame(width: 60, height: 8)
                    }
                    .padding(.vertical, 1)
                    .background(isActive ? Color.accentColor.opacity(0.08) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 3))
                    .contentShape(Rectangle())
                    .onTapGesture { selectedConformerIndex = conf.id }
                }
            }
        }
        .padding(12)
        .onAppear {
            if conformers.isEmpty, !entry.conformers.isEmpty {
                conformers = entry.conformers.map { c in
                    ConformerEntry(id: c.id, molecule: MoleculeData(name: entry.name, title: entry.smiles,
                                                                     atoms: c.atoms, bonds: c.bonds), energy: c.energy)
                }
            }
        }
    }

    // MARK: - 2D Structure Drawing

    func compute2DPreview(smiles: String) {
        guard !smiles.isEmpty else { ligand2DCoords = nil; return }
        isComputing2D = true
        Task {
            let result = await Task.detached { RDKitBridge.compute2DCoords(smiles: smiles) }.value
            ligand2DCoords = result
            isComputing2D = false
        }
    }

    func draw2DStructure(context: GraphicsContext, size: CGSize, coords: RDKitBridge.Coords2D) {
        let positions = coords.positions
        guard !positions.isEmpty else { return }

        let xs = positions.map(\.x), ys = positions.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!, minY = ys.min()!, maxY = ys.max()!
        let rangeX = max(maxX - minX, 1), rangeY = max(maxY - minY, 1)
        let padding: CGFloat = 30
        let scale = min((size.width - padding * 2) / rangeX, (size.height - padding * 2) / rangeY) * 0.85
        let cx = size.width / 2, cy = size.height / 2
        let midX = (minX + maxX) / 2, midY = (minY + maxY) / 2

        func project(_ p: CGPoint) -> CGPoint {
            CGPoint(x: cx + (p.x - midX) * scale, y: cy - (p.y - midY) * scale)
        }

        // Bonds
        for (a1, a2, order) in coords.bonds {
            guard a1 < positions.count, a2 < positions.count else { continue }
            let p1 = project(positions[a1]), p2 = project(positions[a2])
            let bondColor = Color.primary.opacity(0.7)
            if order == 2 {
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = sqrt(dx*dx + dy*dy)
                guard len > 0 else { continue }
                let nx = -dy/len * 2, ny = dx/len * 2
                for sign in [-1.0, 1.0] {
                    var path = Path()
                    path.move(to: CGPoint(x: p1.x + nx*sign, y: p1.y + ny*sign))
                    path.addLine(to: CGPoint(x: p2.x + nx*sign, y: p2.y + ny*sign))
                    context.stroke(path, with: .color(bondColor), lineWidth: 1.5)
                }
            } else {
                var path = Path()
                path.move(to: p1)
                path.addLine(to: p2)
                context.stroke(path, with: .color(bondColor), lineWidth: order == 4 ? 2 : 1.8)
                if order == 4 {
                    let dx = p2.x - p1.x, dy = p2.y - p1.y
                    let len = sqrt(dx*dx + dy*dy)
                    if len > 0 {
                        let nx = -dy/len * 3, ny = dx/len * 3
                        var dashed = Path()
                        dashed.move(to: CGPoint(x: p1.x + nx, y: p1.y + ny))
                        dashed.addLine(to: CGPoint(x: p2.x + nx, y: p2.y + ny))
                        context.stroke(dashed, with: .color(bondColor.opacity(0.4)),
                                       style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                    }
                }
            }
        }

        // Atoms
        for (i, pos) in positions.enumerated() {
            let p = project(pos)
            let z = coords.atomicNums[i]
            if z == 6 {
                let r: CGFloat = 3
                context.fill(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                             with: .color(.primary.opacity(0.5)))
            } else {
                let r: CGFloat = 10
                let (sym, col) = atomSymbolColor(z)
                context.fill(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                             with: .color(Color(nsColor: .controlBackgroundColor)))
                context.stroke(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                               with: .color(col), lineWidth: 1.5)
                let text = Text(sym).font(.system(size: 11, weight: .bold, design: .monospaced)).foregroundColor(col)
                context.draw(context.resolve(text), at: p, anchor: .center)
            }
        }
    }

    func atomSymbolColor(_ atomicNum: Int) -> (String, Color) {
        switch atomicNum {
        case 7:  return ("N", .blue)
        case 8:  return ("O", .red)
        case 9:  return ("F", .green)
        case 15: return ("P", .orange)
        case 16: return ("S", .yellow)
        case 17: return ("Cl", .green)
        case 35: return ("Br", Color(red: 0.6, green: 0.2, blue: 0.2))
        case 53: return ("I", .purple)
        default: return (Element(rawValue: atomicNum)?.symbol ?? "?", .gray)
        }
    }

    // MARK: - Actions

    func useVariantForDocking(_ child: LigandEntry) {
        guard !child.atoms.isEmpty else {
            viewModel.log.warn("\(child.name) has no 3D coordinates", category: .molecule)
            return
        }
        let mol = Molecule(name: child.name, atoms: child.atoms, bonds: child.bonds,
                           title: child.smiles, smiles: child.smiles)
        viewModel.setLigandForDocking(mol, entryID: child.id)
    }

    /// Inspect a variant child: update detail panel, 2D/3D preview, conformers
    func inspectVariantChild(_ child: LigandEntry) {
        inspectedEntry = child
        selectedVariantID = child.id
        compute2DPreview(smiles: child.smiles)
        conformers = child.conformers.map { c in
            ConformerEntry(id: c.id, molecule: MoleculeData(name: child.name, title: child.smiles,
                                                             atoms: c.atoms, bonds: c.bonds), energy: c.energy)
        }
        selectedConformerIndex = 0
        if show3DPreview && !child.atoms.isEmpty {
            updateMiniRenderer(atoms: child.atoms, bonds: child.bonds)
        }
    }

    func deleteVariant(_ id: UUID) {
        let parentID = db.entries.first(where: { $0.id == id })?.parentID
        db.removeWithChildren(id: id)
        // If we deleted the inspected entry, switch back to parent
        if inspectedEntry?.id == id {
            if let pid = parentID {
                inspectedEntry = db.entries.first { $0.id == pid }
                compute2DPreview(smiles: inspectedEntry?.smiles ?? "")
            } else {
                inspectedEntry = nil
            }
            conformers = []
            selectedConformerIndex = 0
        }
        if selectedVariantID == id { selectedVariantID = nil }
    }

    // MARK: - Full Ensemble Preparation (protomer x tautomer x conformer)

    func runEnsemblePreparation(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isGeneratingVariants = true

        Task {
            let smi = entry.smiles, nm = entry.name
            let pH = variantPH, pkaT = variantPkaThreshold
            let maxT = variantMaxTautomers, maxP = variantMaxProtomers
            let cutoff = variantEnergyCutoff
            let nConf = conformerBudgetPerVariant

            let result = await Task.detached { @Sendable in
                RDKitBridge.prepareEnsemble(
                    smiles: smi, name: nm, pH: pH, pkaThreshold: pkaT,
                    maxTautomers: maxT, maxProtomers: maxP,
                    energyCutoff: cutoff, conformersPerForm: nConf
                )
            }.value

            guard result.success else {
                viewModel.log.error("Ensemble preparation failed: \(result.errorMessage)", category: .molecule)
                isGeneratingVariants = false
                return
            }

            guard !result.members.isEmpty else {
                viewModel.log.info("Ensemble produced no valid members for \(nm)", category: .molecule)
                isGeneratingVariants = false
                return
            }

            // Batch all DB mutations into a single entries didSet to avoid save storm
            let entryID = entry.id
            let members = result.members
            db.batchMutate { entries in
                // Remove existing children
                let childIDs = Set(entries.filter { $0.parentID == entryID }.map(\.id))
                if !childIDs.isEmpty {
                    entries.removeAll { childIDs.contains($0.id) }
                }

                // Add each ensemble member as a child entry
                // C++ kind: 0=parent, 1=tautomer, 2=protomer, 3=taut+prot
                let parentName = entries.first(where: { $0.id == entryID })?.name ?? nm
                for member in members {
                    guard !member.molecule.atoms.isEmpty else { continue }
                    let kind: VariantKind = (member.kind >= 2) ? .protomer : .tautomer
                    let child = LigandEntry(
                        name: "\(parentName)_\(member.label)",
                        smiles: member.smiles,
                        atoms: member.molecule.atoms,
                        bonds: member.molecule.bonds,
                        isPrepared: true,
                        conformerCount: 1,
                        variantLineage: "\(member.label) of \(parentName)",
                        parentID: entryID,
                        variantKind: kind,
                        relativeEnergy: member.mmffEnergy
                    )
                    entries.append(child)
                }

                // Mark parent as prepared
                if let idx = entries.firstIndex(where: { $0.id == entryID }) {
                    if let bestMember = members.first {
                        entries[idx].atoms = bestMember.molecule.atoms
                        entries[idx].bonds = bestMember.molecule.bonds
                    }
                    entries[idx].isPrepared = true
                    entries[idx].preparationDate = Date()
                    entries[idx].conformerCount = members.count
                }
            }

            expandedEntryIDs.insert(entry.id)

            // Refresh inspected entry
            if inspectedEntry?.id == entry.id {
                inspectedEntry = db.entries.first { $0.id == entry.id }
                compute2DPreview(smiles: entry.smiles)
            }

            // Load all ensemble members into the conformer carousel for browsing
            conformers = result.members.enumerated().map { idx, m in
                ConformerEntry(id: idx, molecule: m.molecule, energy: m.mmffEnergy)
            }
            selectedConformerIndex = 0
            if show3DPreview, let best = result.members.first {
                updateMiniRenderer(atoms: best.molecule.atoms, bonds: best.molecule.bonds)
            }

            isGeneratingVariants = false
            viewModel.log.success(
                "Ensemble: \(result.members.count) entries (\(result.numForms) forms x conformers) for \(nm)",
                category: .molecule)
        }
    }

    // MARK: - Variant Generation (single-entry, async)

    func generateTautomers(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isGeneratingVariants = true

        Task {
            let smi = entry.smiles, nm = entry.name
            let maxT = variantMaxTautomers, cutoff = variantEnergyCutoff

            let results = await Task.detached { @Sendable in
                RDKitBridge.enumerateTautomers(smiles: smi, name: nm,
                                                maxTautomers: maxT, energyCutoff: cutoff)
            }.value

            // Replace existing tautomer children (batch to avoid save storm)
            let entryID = entry.id
            db.batchMutate { entries in
                let existingIDs = Set(entries.filter { $0.parentID == entryID && $0.variantKind == .tautomer }.map(\.id))
                if !existingIDs.isEmpty {
                    entries.removeAll { existingIDs.contains($0.id) }
                }
                let parentName = entries.first(where: { $0.id == entryID })?.name ?? nm
                for r in results {
                    entries.append(LigandEntry(
                        name: "\(parentName)_\(r.label)", smiles: r.smiles,
                        atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                        isPrepared: !r.molecule.atoms.isEmpty, conformerCount: 0,
                        variantLineage: "\(r.label) of \(parentName)",
                        parentID: entryID, variantKind: .tautomer,
                        relativeEnergy: r.score
                    ))
                }
            }
            expandedEntryIDs.insert(entry.id)
            isGeneratingVariants = false
            viewModel.log.success("Generated \(results.count) tautomers for \(nm)", category: .molecule)
        }
    }

    func generateProtomers(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isGeneratingVariants = true

        Task {
            let smi = entry.smiles, nm = entry.name
            let maxP = variantMaxProtomers, pH = variantPH, pkaT = variantPkaThreshold

            let results = await Task.detached { @Sendable in
                RDKitBridge.enumerateProtomers(smiles: smi, name: nm,
                                                maxProtomers: maxP, pH: pH, pkaThreshold: pkaT)
            }.value

            let entryID = entry.id
            db.batchMutate { entries in
                let existingIDs = Set(entries.filter { $0.parentID == entryID && $0.variantKind == .protomer }.map(\.id))
                if !existingIDs.isEmpty {
                    entries.removeAll { existingIDs.contains($0.id) }
                }
                let parentName = entries.first(where: { $0.id == entryID })?.name ?? nm
                for r in results {
                    entries.append(LigandEntry(
                        name: "\(parentName)_\(r.label)", smiles: r.smiles,
                        atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                        isPrepared: !r.molecule.atoms.isEmpty, conformerCount: 0,
                        variantLineage: "\(r.label) of \(parentName)",
                        parentID: entryID, variantKind: .protomer,
                        relativeEnergy: r.score
                    ))
                }
            }
            expandedEntryIDs.insert(entry.id)
            isGeneratingVariants = false
            viewModel.log.success("Generated \(results.count) protomers for \(nm) at pH \(String(format: "%.1f", pH))", category: .molecule)
        }
    }

    // MARK: - Conformer Generation (single-entry, async)

    func generateConformersForEntry(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isGeneratingConformers = true

        Task {
            let smi = entry.smiles, nm = entry.name
            let nc = conformerBudgetPerVariant, mn = prepMinimize

            let results = await Task.detached { @Sendable in
                RDKitBridge.generateConformers(smiles: smi, name: nm, count: nc, minimize: mn)
            }.value

            // Update database entry with best conformer + store all conformers
            if let best = results.first {
                var updated = entry
                updated.atoms = best.molecule.atoms
                updated.bonds = best.molecule.bonds
                updated.conformerCount = results.count
                updated.isPrepared = true
                updated.conformers = results.enumerated().map { idx, r in
                    LigandConformer(id: idx, atoms: r.molecule.atoms, bonds: r.molecule.bonds, energy: r.energy)
                }
                db.update(updated)

                // Refresh inspected entry
                if inspectedEntry?.id == entry.id {
                    inspectedEntry = db.entries.first { $0.id == entry.id }
                }
            }

            // Update local conformer carousel
            conformers = results.enumerated().map { idx, r in
                ConformerEntry(id: idx, molecule: r.molecule, energy: r.energy)
            }
            selectedConformerIndex = 0

            isGeneratingConformers = false
            viewModel.log.success("Generated \(results.count) conformers for \(nm)", category: .molecule)

            // Update 3D preview if active
            if show3DPreview, let best = results.first {
                updateMiniRenderer(atoms: best.molecule.atoms, bonds: best.molecule.bonds)
            }
        }
    }

    // MARK: - Mini 3D Renderer

    func initMiniRenderer(atoms: [Atom], bonds: [Bond]) {
        guard miniRenderer == nil else {
            updateMiniRenderer(atoms: atoms, bonds: bonds)
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let tempView = MTKView(frame: CGRect(x: 0, y: 0, width: 400, height: 300), device: device)
        tempView.colorPixelFormat = .bgra8Unorm
        tempView.depthStencilPixelFormat = .depth32Float
        tempView.sampleCount = 4
        guard let renderer = Renderer(mtkView: tempView) else { return }
        renderer.updateMoleculeData(atoms: atoms, bonds: bonds)
        renderer.fitToContent()
        miniRenderer = renderer
    }

    func updateMiniRenderer(atoms: [Atom], bonds: [Bond]) {
        guard let renderer = miniRenderer else {
            initMiniRenderer(atoms: atoms, bonds: bonds)
            return
        }
        renderer.updateMoleculeData(atoms: atoms, bonds: bonds)
        renderer.fitToContent()
    }
}
