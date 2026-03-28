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
                    case .populateAndPrepare:
                        populateAndPrepareTab(entry)
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
            HStack(spacing: 8) {
                Text(entry.name)
                    .font(.headline)
                if entry.isPrepared {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.subheadline)
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
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            // Structure visualization
            if show3DPreview {
                // 3D Metal viewport — full rotate/pan/zoom
                if let renderer = miniRenderer {
                    MetalView(renderer: renderer)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if entry.atoms.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "cube")
                            .font(.title)
                            .foregroundStyle(.secondary)
                        Text("Prepare ligand for 3D view")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ProgressView("Initializing 3D...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .onAppear { initMiniRenderer(atoms: entry.atoms, bonds: entry.bonds) }
                }
            } else {
                // 2D RDKit depiction (prefer SVG for publication-quality rendering)
                if isComputing2D {
                    ProgressView("Computing 2D layout...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let svg = ligand2DSVG,
                          let svgData = svg.data(using: .utf8),
                          let nsImage = NSImage(data: svgData) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                } else if let coords = ligand2DCoords {
                    // Fallback to Canvas drawing if SVG generation failed
                    Canvas { context, size in
                        draw2DStructure(context: context, size: size, coords: coords)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: .textBackgroundColor))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "hexagon")
                            .font(.title)
                            .foregroundStyle(.secondary)
                        Text("No 2D structure")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }

            // Enumeration info (when this entry came from enumeration)
            if let entry = inspectedEntry, entry.isEnumerated {
                Divider()
                HStack(spacing: 8) {
                    if let kind = entry.formKind {
                        let kindColor: Color = switch kind {
                        case .parent: .green
                        case .tautomer: .cyan
                        case .protomer: .orange
                        case .tautomerProtomer: .purple
                        }
                        Text(kind.symbol)
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(.white)
                            .frame(width: 16, height: 14)
                            .background(RoundedRectangle(cornerRadius: 4).fill(kindColor))
                        Text(kind.label)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    if let pn = entry.parentName {
                        Text("from \(pn)")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let pop = entry.populationWeight {
                        Text(String(format: "%.1f%%", pop * 100))
                            .font(.footnote.monospaced())
                            .foregroundStyle(pop > 0.3 ? .green : pop > 0.1 ? .yellow : .secondary)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(Color.secondary.opacity(0.05))
            }

            // Hint: 2D can't show conformer differences
            if !conformers.isEmpty && !show3DPreview {
                Text("Switch to 3D to see conformer spatial differences")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
                        .font(.footnote.monospaced().weight(.medium))

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
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
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
        VStack(alignment: .leading, spacing: 12) {
            if let desc = entry.descriptors {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    GridRow {
                        Text("MW").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.1f Da", desc.molecularWeight)).font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("cLogP").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.2f", desc.logP)).font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("TPSA").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.1f \u{00C5}\u{00B2}", desc.tpsa)).font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("HBD / HBA").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text("\(desc.hbd) / \(desc.hba)").font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("RotBonds").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text("\(desc.rotatableBonds)").font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("Rings").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text("\(desc.rings) (\(desc.aromaticRings) arom)").font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("Fsp3").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text(String(format: "%.2f", desc.fractionCSP3)).font(.subheadline.monospaced())
                    }
                    GridRow {
                        Text("Heavy atoms").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                        Text("\(desc.heavyAtomCount)").font(.subheadline.monospaced())
                    }
                }

                Divider()

                HStack(spacing: 8) {
                    ruleBadge("Lipinski", passed: desc.lipinski)
                    ruleBadge("Veber", passed: desc.veber)
                }
            } else {
                Text("Prepare this ligand to compute properties")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            // Binding affinity (if available)
            if entry.ki != nil || entry.pKi != nil || entry.ic50 != nil {
                Divider()
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    if let ki = entry.ki {
                        GridRow {
                            Text("Ki").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                            Text(String(format: "%.2f nM", ki)).font(.subheadline.monospaced())
                        }
                    }
                    if let pKi = entry.pKi {
                        GridRow {
                            Text("pKi").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                            Text(String(format: "%.2f", pKi)).font(.subheadline.monospaced())
                        }
                    }
                    if let ic50 = entry.ic50 {
                        GridRow {
                            Text("IC50").font(.footnote.weight(.medium)).foregroundStyle(.secondary)
                            Text(String(format: "%.2f nM", ic50)).font(.subheadline.monospaced())
                        }
                    }
                }
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private func ruleBadge(_ name: String, passed: Bool) -> some View {
        Text(name).font(.footnote.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 2)
            .background(Capsule().fill(passed ? Color.green.opacity(0.2) : Color.red.opacity(0.15)))
            .foregroundStyle(passed ? .green : .red)
            .strikethrough(!passed)
    }

    // MARK: - Prepare Tab

    @ViewBuilder
    func prepareTab(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Quick single-structure preparation
            Label("Quick Prepare", systemImage: "wand.and.stars")
                .font(.subheadline.weight(.semibold))

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
                    Text("Prepared").font(.footnote.weight(.medium)).foregroundStyle(.green)
                    if let date = entry.preparationDate {
                        Text(date.formatted(date: .abbreviated, time: .shortened))
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }

            Divider()

            // Full ensemble preparation (the main workflow)
            Label("Full Ensemble Preparation", systemImage: "sparkles")
                .font(.subheadline.weight(.semibold))

            Text("Generates all chemically probable forms at target pH: protomers, tautomers, and conformers. Each form is fully prepared (polar H, MMFF94 minimized, Gasteiger charges). Boltzmann population weights are assigned based on MMFF94 energy.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                Text("Target pH").font(.footnote)
                Slider(value: $variantPH, in: 1...14, step: 0.1).controlSize(.mini)
                Text(String(format: "%.1f", variantPH)).font(.footnote.monospaced()).frame(width: 30)
            }
            HStack {
                Text("Conformers/form").font(.footnote)
                Spacer()
                Stepper("\(prepNumConformers)", value: $prepNumConformers, in: 1...50, step: 1)
                    .controlSize(.small)
            }
            HStack {
                Text("Energy cutoff").font(.footnote)
                Slider(value: $variantEnergyCutoff, in: 5...50, step: 1).controlSize(.mini)
                Text(String(format: "%.0f kcal", variantEnergyCutoff)).font(.footnote.monospaced()).frame(width: 50)
            }

            Button(action: { runEnsemblePreparation(entry) }) {
                Label(isGeneratingVariants ? "Preparing ensemble..." : "Prepare Full Ensemble",
                      systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(entry.smiles.isEmpty || isGeneratingVariants || entry.isEnumerated)

            if isGeneratingVariants {
                ProgressView("Running protomer x tautomer x conformer pipeline...")
                    .controlSize(.small)
            }

            // Show ensemble summary if siblings exist
            let siblings = db.siblings(of: entry)
            if !siblings.isEmpty {
                Divider()
                let nForms = Set(siblings.map(\.smiles)).count
                let nTotal = siblings.count
                HStack(spacing: 8) {
                    Label("\(nForms) forms, \(nTotal) entries", systemImage: "checkmark.circle.fill")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.green)
                }
                Text("Browse in Variants tab. Click entries to inspect in 2D/3D.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
    }

    // MARK: - Variants Tab (ensemble browser with energy + population + manual curation)

    @ViewBuilder
    func variantsTab(_ entry: LigandEntry) -> some View {
        // Show sibling forms (flat model — siblings share the same parentName)
        let isChildEntry = entry.isEnumerated
        let siblings = db.siblings(of: entry)

        VStack(alignment: .leading, spacing: 12) {
            Label("Tautomers & Protomers", systemImage: "arrow.triangle.2.circlepath")
                .font(.subheadline.weight(.semibold))

            if isChildEntry {
                Text("Viewing variant of parent entry. Generate from the parent entry.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text("pH").font(.footnote)
                Slider(value: $variantPH, in: 1...14, step: 0.1).controlSize(.mini)
                Text(String(format: "%.1f", variantPH)).font(.footnote.monospaced()).frame(width: 30)
            }
            HStack {
                Text("Max tautomers").font(.footnote)
                Spacer()
                Stepper("\(variantMaxTautomers)", value: $variantMaxTautomers, in: 1...50).controlSize(.small)
            }
            HStack {
                Text("Max protomers").font(.footnote)
                Spacer()
                Stepper("\(variantMaxProtomers)", value: $variantMaxProtomers, in: 1...20).controlSize(.small)
            }
            HStack {
                Text("Energy cutoff").font(.footnote)
                Slider(value: $variantEnergyCutoff, in: 1...50, step: 1).controlSize(.mini)
                Text(String(format: "%.0f kcal", variantEnergyCutoff)).font(.footnote.monospaced()).frame(width: 50)
            }

            HStack(spacing: 8) {
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
            if !siblings.isEmpty {
                Divider()
                Text("\(siblings.count) variants")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)

                ForEach(siblings) { child in
                    variantRow(child)
                }
            } else if !isChildEntry {
                Text("Generate tautomers or protomers to see variants here")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private func variantRow(_ child: LigandEntry) -> some View {
        let isSelected = inspectedEntry?.id == child.id
        let allSiblings = db.siblings(of: child)
        let minE = allSiblings.compactMap(\.relativeEnergy).min() ?? 0

        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                // Kind badge
                Text(child.formKind == .tautomer ? "T" : "P")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(width: 14, height: 14)
                    .background(Circle().fill(child.formKind == .tautomer ? Color.purple : Color.cyan))

                // Label
                Text(child.name)
                    .font(.footnote.weight(isSelected ? .semibold : .medium))
                    .lineLimit(1)

                Spacer()

                // Energy (absolute)
                if let energy = child.relativeEnergy {
                    let dE = energy - minE
                    Text(String(format: "%.1f kcal/mol", energy))
                        .font(.caption.monospaced())
                        .foregroundStyle(dE < 1 ? .green : dE < 5 ? .yellow : dE < 15 ? .orange : .red)
                }

                // Action buttons
                Button(action: { useVariantForDocking(child) }) {
                    Image(systemName: "arrow.right.circle").font(.footnote)
                }
                .buttonStyle(.plain).foregroundStyle(.tint)
                .help("Use for docking")
                .disabled(child.atoms.isEmpty)

                Button(action: { deleteVariant(child.id) }) {
                    Image(systemName: "xmark.circle").font(.footnote)
                }
                .buttonStyle(.plain).foregroundStyle(.red.opacity(0.7))
                .help("Remove from ensemble (chemically improbable or unwanted)")
            }

            // SMILES preview
            Text(child.smiles)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 4)
        .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture { inspectVariantChild(child) }
    }

    // (Ensemble tab removed — forms are now shown inline in the table as expandable child rows)

    // (removeForm removed — flat model has no forms within entries)

    // MARK: - Conformers Tab

    @ViewBuilder
    func conformersTab(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Conformer Browser", systemImage: "cube.transparent")
                .font(.subheadline.weight(.semibold))

            HStack {
                Text("Max conformers")
                    .font(.footnote)
                Spacer()
                Stepper("\(prepNumConformers)", value: $prepNumConformers, in: 5...200, step: 5)
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
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)

                ForEach(conformers) { conf in
                    let isActive = conf.id == selectedConformerIndex
                    HStack(spacing: 8) {
                        Text("#\(conf.id + 1)")
                            .font(.footnote.monospaced().weight(.bold))
                            .foregroundStyle(isActive ? .primary : .secondary)
                            .frame(width: 28, alignment: .trailing)
                        Text(String(format: "%.2f kcal/mol", conf.energy))
                            .font(.footnote.monospaced())
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
                    .clipShape(RoundedRectangle(cornerRadius: 4))
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

    // MARK: - Populate & Prepare Tab (unified pipeline)

    @ViewBuilder
    func populateAndPrepareTab(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            Label("Populate & Prepare", systemImage: "wand.and.stars")
                .font(.headline)

            Text("Automated pipeline: add polar H → MMFF94 minimize → Gasteiger charges → enumerate tautomers & protomers at target pH → generate conformers → filter by Boltzmann population. Output replaces the raw molecule with docking-ready ligand(s).")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            // Configuration
            Label("Configuration", systemImage: "gearshape")
                .font(.subheadline.weight(.medium))

            Grid(alignment: .leading, horizontalSpacing: 8, verticalSpacing: 8) {
                GridRow {
                    Text("Target pH").font(.footnote).frame(width: 100, alignment: .leading)
                    Slider(value: $variantPH, in: 1...14, step: 0.1).controlSize(.mini)
                    Text(String(format: "%.1f", variantPH))
                        .font(.footnote.monospaced()).frame(width: 30)
                }
                GridRow {
                    Text("pKa method").font(.footnote).frame(width: 100, alignment: .leading)
                    Picker("", selection: $pkaMethod) {
                        ForEach(PKaMethod.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    .help(pkaMethod.description)
                    Spacer()
                }
                GridRow {
                    Text("pKa threshold").font(.footnote).frame(width: 100, alignment: .leading)
                    Slider(value: $variantPkaThreshold, in: 0.5...5.0, step: 0.5).controlSize(.mini)
                    Text(String(format: "%.1f", variantPkaThreshold))
                        .font(.footnote.monospaced()).frame(width: 30)
                }
                GridRow {
                    Text("Conformers/form").font(.footnote).frame(width: 100, alignment: .leading)
                    Stepper("\(prepNumConformers)", value: $prepNumConformers, in: 1...100, step: 5)
                        .controlSize(.small)
                    Spacer()
                }
                GridRow {
                    Text("Max tautomers").font(.footnote).frame(width: 100, alignment: .leading)
                    Stepper("\(variantMaxTautomers)", value: $variantMaxTautomers, in: 1...50)
                        .controlSize(.small)
                    Spacer()
                }
                GridRow {
                    Text("Max protomers").font(.footnote).frame(width: 100, alignment: .leading)
                    Stepper("\(variantMaxProtomers)", value: $variantMaxProtomers, in: 1...20)
                        .controlSize(.small)
                    Spacer()
                }
                GridRow {
                    Text("Energy cutoff").font(.footnote).frame(width: 100, alignment: .leading)
                    Slider(value: $variantEnergyCutoff, in: 5...50, step: 1).controlSize(.mini)
                    Text(String(format: "%.0f kcal", variantEnergyCutoff))
                        .font(.footnote.monospaced()).frame(width: 50)
                }
                GridRow {
                    Text("Min population").font(.footnote).frame(width: 100, alignment: .leading)
                    Slider(value: $variantMinPopulation, in: 0...20, step: 0.5).controlSize(.mini)
                    Text(String(format: "%.1f%%", variantMinPopulation))
                        .font(.footnote.monospaced()).frame(width: 50)
                }
            }

            Text("Forms with Boltzmann population below \(String(format: "%.1f%%", variantMinPopulation)) will be discarded. Each molecule produces a different number of forms depending on its chemistry.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            // Run button
            Button(action: {
                runPopulateAndPrepare(entries: [entry])
            }) {
                Label(isProcessing ? "Processing..." : "Run Populate & Prepare",
                      systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(entry.smiles.isEmpty || isProcessing || entry.isEnumerated)

            if isProcessing {
                VStack(spacing: 4) {
                    ProgressView(processingMessage)
                        .controlSize(.small)
                    if batchProgress.total > 0 {
                        ProgressView(value: Double(batchProgress.current), total: Double(batchProgress.total))
                            .controlSize(.small)
                    }
                }
            }

            // Results summary
            let siblings = db.siblings(of: entry)
            if entry.isPrepared || !siblings.isEmpty {
                Divider()
                Label("Output", systemImage: "checkmark.circle.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.green)

                if entry.isPrepared {
                    HStack(spacing: 8) {
                        Image(systemName: "molecule").foregroundStyle(.green)
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Input molecule → prepared ligand")
                                .font(.footnote.weight(.medium))
                            Text("\(entry.atoms.count) atoms, \(entry.conformerCount) conformers")
                                .font(.footnote).foregroundStyle(.secondary)
                            if let date = entry.preparationDate {
                                Text(date.formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        Button("Use for Docking") {
                            useEntryForDocking(entry)
                        }
                        .controlSize(.small)
                        .disabled(entry.atoms.isEmpty)
                    }
                    .padding(8)
                    .background(Color.green.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                if !siblings.isEmpty {
                    let nForms = Set(siblings.map(\.smiles)).count
                    Text("\(nForms) chemical forms, \(siblings.count) variant entries")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    ForEach(siblings) { child in
                        variantRow(child)
                    }
                }
            }

            // Quick prepare (simple preparation without ensemble)
            if !entry.isPrepared {
                Divider()
                DisclosureGroup("Quick Prepare (single conformer, no variants)") {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Add hydrogens", isOn: $prepAddHydrogens)
                            .toggleStyle(.switch).controlSize(.small)
                        Toggle("Energy minimize (MMFF94)", isOn: $prepMinimize)
                            .toggleStyle(.switch).controlSize(.small)
                        Toggle("Compute Gasteiger charges", isOn: $prepComputeCharges)
                            .toggleStyle(.switch).controlSize(.small)
                        Button("Quick Prepare") {
                            prepareSingleEntry(entry)
                        }
                        .controlSize(.small)
                        .disabled(entry.smiles.isEmpty)
                    }
                    .padding(.top, 4)
                }
                .font(.footnote)
                .foregroundStyle(.secondary)
            }
        }
        .padding(12)
    }

    // MARK: - 2D Structure Drawing

    /// Debounced 2D preview — cancels previous computation when a new row is selected.
    private static var preview2DTask: Task<Void, Never>?

    func compute2DPreview(smiles: String) {
        guard !smiles.isEmpty else { ligand2DCoords = nil; ligand2DSVG = nil; return }

        // Cancel any in-flight computation
        Self.preview2DTask?.cancel()
        isComputing2D = true

        Self.preview2DTask = Task {
            // Small delay to debounce rapid clicks
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled else { return }

            let smi = smiles
            let (coords, svg) = await Task.detached { @Sendable in
                (RDKitBridge.compute2DCoords(smiles: smi),
                 RDKitBridge.moleculeToSVG(smiles: smi, width: 500, height: 400))
            }.value
            guard !Task.isCancelled else { return }
            ligand2DCoords = coords
            ligand2DSVG = svg
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
                             with: .color(Color(nsColor: .textBackgroundColor)))
                context.stroke(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                               with: .color(col), lineWidth: 1.5)
                let text = Text(sym).font(.subheadline.monospaced().weight(.bold)).foregroundColor(col)
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
        let deletedEntry = db.entries.first(where: { $0.id == id })
        db.removeWithChildren(id: id)
        // If we deleted the inspected entry, switch to a sibling or clear
        if inspectedEntry?.id == id {
            if let pn = deletedEntry?.parentName,
               let sibling = db.entries.first(where: { $0.parentName == pn }) {
                inspectedEntry = sibling
                compute2DPreview(smiles: sibling.smiles)
            } else {
                inspectedEntry = nil
            }
            conformers = []
            selectedConformerIndex = 0
        }
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
            let nConf = prepNumConformers

            let useGNN = pkaMethod == .gnn
            let result = await Task.detached { @Sendable in
                let predictions = useGNN ? PKaGNNPredictor.predict(smiles: smi) : []
                return RDKitBridge.prepareEnsembleWithSites(
                    smiles: smi, name: nm, pH: pH, pkaThreshold: pkaT,
                    maxTautomers: maxT, maxProtomers: maxP,
                    energyCutoff: cutoff, conformersPerForm: nConf,
                    sites: predictions
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
                // Remove existing siblings (forms sharing the same parentName)
                let pName = entries.first(where: { $0.id == entryID })?.name ?? nm
                let siblingIDs = Set(entries.filter { $0.parentName == pName }.map(\.id))
                if !siblingIDs.isEmpty {
                    entries.removeAll { siblingIDs.contains($0.id) }
                }

                // Add each ensemble member as a sibling entry
                // C++ kind: 0=parent, 1=tautomer, 2=protomer, 3=taut+prot
                for member in members {
                    guard !member.molecule.atoms.isEmpty else { continue }
                    let kind: ChemicalFormKind = (member.kind >= 2) ? .protomer : .tautomer
                    var child = LigandEntry(
                        name: "\(pName)_\(member.label)",
                        smiles: member.smiles,
                        atoms: member.molecule.atoms,
                        bonds: member.molecule.bonds,
                        isPrepared: true,
                        conformerCount: 1
                    )
                    child.parentName = pName
                    child.formKind = kind
                    child.relativeEnergy = member.mmffEnergy
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

            // Replace existing tautomer siblings (batch to avoid save storm)
            let entryID = entry.id
            db.batchMutate { entries in
                let pName = entries.first(where: { $0.id == entryID })?.name ?? nm
                let existingIDs = Set(entries.filter { $0.parentName == pName && $0.formKind == .tautomer }.map(\.id))
                if !existingIDs.isEmpty {
                    entries.removeAll { existingIDs.contains($0.id) }
                }
                for r in results {
                    var child = LigandEntry(
                        name: "\(pName)_\(r.label)", smiles: r.smiles,
                        atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                        isPrepared: !r.molecule.atoms.isEmpty, conformerCount: 0
                    )
                    child.parentName = pName
                    child.formKind = .tautomer
                    child.relativeEnergy = r.score
                    entries.append(child)
                }
            }
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
                let pName = entries.first(where: { $0.id == entryID })?.name ?? nm
                let existingIDs = Set(entries.filter { $0.parentName == pName && $0.formKind == .protomer }.map(\.id))
                if !existingIDs.isEmpty {
                    entries.removeAll { existingIDs.contains($0.id) }
                }
                for r in results {
                    var child = LigandEntry(
                        name: "\(pName)_\(r.label)", smiles: r.smiles,
                        atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                        isPrepared: !r.molecule.atoms.isEmpty, conformerCount: 0
                    )
                    child.parentName = pName
                    child.formKind = .protomer
                    child.relativeEnergy = r.score
                    entries.append(child)
                }
            }
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
            let nc = prepNumConformers, mn = prepMinimize

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
