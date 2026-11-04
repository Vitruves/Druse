// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Pharmacophore Editor

/// Pharmacophore query editor with interactive 2D structure view.
///
/// Layout: large Canvas structure view (left) + control panel (right).
/// Features are pharmacophoric points using short labels (Don, Acc, Hyd, Aro).
///
/// Key interactions:
/// - Click atom → create feature (auto-detects aromatic rings as groups)
/// - Click featured atom → remove feature (toggle)
/// - Auto-Detect → grouped features (one per pharmacophore point)
/// - Essential / Optional per feature → hard / soft GPU constraints
struct PharmacophoreEditorView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.dismiss) private var dismiss

    // Source
    @State private var sourceMode: SourceMode = .singleLigand
    @State private var selectedEntryID: UUID?
    @State private var smilesInput: String = ""

    // MCS
    @State private var mcsSmarts: String?
    @State private var mcsMatchedAtoms: Set<Int> = []
    @State private var isFindingMCS: Bool = false
    @State private var selectedForMCS: Set<UUID> = []
    @State private var showOnlyScaffold: Bool = true

    // 2D structure
    @State private var coords2D: RDKitBridge.Coords2D?
    @State private var activeSmiles: String = ""
    @State private var errorMessage: String?

    // Features
    @State private var features: [PharmaFeature] = []
    @State private var selectedFeatureID: UUID? = nil
    @State private var isDetecting: Bool = false
    @State private var featureOverlays: [FeatureOverlay] = []

    // Draw mode
    @State private var drawnSmarts: String = ""
    @State private var drawValidated: Bool = false

    // Multi-select
    @State private var pendingMultiSelect: [Int] = []

    // View state
    @State private var hoveredAtom: Int? = nil

    enum SourceMode: String, CaseIterable {
        case singleLigand = "Single Ligand"
        case commonScaffold = "Common Scaffold"
        case smiles = "SMILES"
        case draw = "Draw"
    }

    struct PharmaFeature: Identifiable {
        let id = UUID()
        var name: String
        var expression: String
        var type: ConstraintInteractionType
        var atomIndices: [Int]
        var isEssential: Bool = false
        var isIgnored: Bool = false
        var radius: Float = 1.5
    }

    struct FeatureOverlay {
        let type: PharmacophoreFeatureType
        let atomIndices: [Int]
        let centroid: CGPoint
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HSplitView {
                structureView
                    .frame(minWidth: 440)
                rightPanel
                    .frame(minWidth: DS.Size.navigatorMin, idealWidth: DS.Size.navigatorIdeal, maxWidth: DS.Size.navigatorMax)
            }
            Divider()
            footer
        }
        .frame(minWidth: 820, idealWidth: 960, minHeight: 580, idealHeight: 680)
        .onAppear {
            if !viewModel.ligandDB.entries.isEmpty {
                sourceMode = .singleLigand
                selectedForMCS = Set(viewModel.ligandDB.entries.map(\.id))
                // Start from the ligand being docked rather than an empty "Select…" canvas.
                if selectedEntryID == nil, let active = viewModel.activeLigandEntryID,
                   let entry = viewModel.ligandDB.entries.first(where: { $0.id == active }) {
                    selectedEntryID = active
                    loadStructure(smiles: entry.smiles)
                }
            }
        }
    }

    // =========================================================================
    // MARK: - Header
    // =========================================================================

    private var header: some View {
        HStack(spacing: DS.Space.l) {
            Image(systemName: "circle.hexagongrid.circle")
                .font(.title3)
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(DS.Color.accent)
            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                Text("Pharmacophore Editor")
                    .font(DS.Font.stepTitle)
                Text("Pick pharmacophoric features to turn into docking constraints.")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: DS.Space.l)

            Picker("Source", selection: $sourceMode) {
                ForEach(SourceMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .onChange(of: sourceMode) { _, _ in resetState() }

            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Close")
            .accessibilityLabel("Close")
        }
        .padding(.horizontal, DS.Space.xl)
        .padding(.vertical, DS.Space.l)
    }

    // =========================================================================
    // MARK: - Structure View (Canvas-based)
    // =========================================================================

    @ViewBuilder
    private var structureView: some View {
        ZStack {
            DS.Color.diagramPaper

            if sourceMode == .draw && !drawValidated {
                ChemDrawerView(
                    initialSmiles: activeSmiles.isEmpty ? nil : activeSmiles,
                    mode: .scaffold
                ) { smiles, smarts in
                    if !smiles.isEmpty {
                        activeSmiles = smiles
                        drawnSmarts = smarts
                    }
                }
            } else if let coords = coords2D {
                GeometryReader { geo in
                    let tx = StructureTransform(positions: coords.positions, viewSize: geo.size, padding: 44)
                    ZStack {
                        // Canvas: bonds, atoms, feature spheres
                        Canvas { ctx, size in
                            let t = StructureTransform(positions: coords.positions, viewSize: size, padding: 44)
                            drawFeatureOverlayHalos(ctx: ctx, tx: t)
                            drawBonds(ctx: ctx, tx: t)
                            drawAtoms(ctx: ctx, tx: t)
                            drawFeatureSpheres(ctx: ctx, tx: t)
                        }

                        // Hit targets
                        ForEach(0..<coords.positions.count, id: \.self) { idx in
                            let pos = tx.toView(coords.positions[idx])
                            let dimmed = !mcsMatchedAtoms.isEmpty && showOnlyScaffold && !mcsMatchedAtoms.contains(idx)
                            if !dimmed {
                                Circle()
                                    .fill(Color.clear)
                                    .frame(width: 34, height: 34)
                                    .contentShape(Circle())
                                    .onTapGesture {
                                        let shiftHeld = NSEvent.modifierFlags.contains(.shift)
                                        if shiftHeld {
                                            handleShiftClick(idx)
                                        } else {
                                            if !pendingMultiSelect.isEmpty {
                                                commitMultiSelect()
                                            }
                                            handleAtomClick(idx)
                                        }
                                    }
                                    .onHover { h in hoveredAtom = h ? idx : nil }
                                    .position(x: pos.x, y: pos.y)
                            }
                        }

                        // Multi-select highlight
                        ForEach(pendingMultiSelect, id: \.self) { idx in
                            if idx < coords.positions.count {
                                let pos = tx.toView(coords.positions[idx])
                                Circle()
                                    .stroke(DS.Color.warning, lineWidth: 2.5)
                                    .frame(width: 30, height: 30)
                                    .position(x: pos.x, y: pos.y)
                                    .allowsHitTesting(false)
                            }
                        }

                        // Hover highlight
                        if let idx = hoveredAtom, idx < coords.positions.count {
                            let pos = tx.toView(coords.positions[idx])
                            let hasFeature = featureForAtom(idx) != nil
                            let inMultiSelect = pendingMultiSelect.contains(idx)
                            Circle()
                                .stroke(hasFeature ? DS.Color.danger.opacity(0.7)
                                        : inMultiSelect ? DS.Color.warning.opacity(0.85)
                                        : DS.Color.accent.opacity(0.7), lineWidth: 2)
                                .frame(width: 28, height: 28)
                                .position(x: pos.x, y: pos.y)
                                .allowsHitTesting(false)
                        }
                    }
                    .overlay(alignment: .bottom) { statusBar }
                }
            } else if isFindingMCS || isDetecting {
                VStack(spacing: DS.Space.m) {
                    ProgressView()
                    Text(isFindingMCS ? "Computing common scaffold…" : "Detecting features…")
                        .font(DS.Font.label)
                        .foregroundStyle(.secondary)
                }
            } else if let error = errorMessage {
                DSEmptyState("Structure Unavailable", systemImage: "exclamationmark.triangle", description: error)
            } else {
                DSEmptyState(
                    "No Structure",
                    systemImage: "hexagon",
                    description: "Choose a ligand, a common scaffold, a SMILES string or draw a scaffold to begin."
                )
            }
        }
    }

    // MARK: Status Bar

    @ViewBuilder
    private var statusBar: some View {
        if !pendingMultiSelect.isEmpty {
            HStack(spacing: DS.Space.m) {
                Image(systemName: "rectangle.dashed.badge.record")
                    .foregroundStyle(DS.Color.warning)
                Text("\(pendingMultiSelect.count) atom\(pendingMultiSelect.count == 1 ? "" : "s") selected")
                    .fontWeight(.medium)
                    .monospacedDigit()
                Text("Shift-click to add more, click to confirm group")
                    .foregroundStyle(.secondary)
            }
            .modifier(StructureStatusChrome())
        } else if let idx = hoveredAtom, let coords = coords2D, idx < coords.atomicNums.count {
            let elem = Element(rawValue: coords.atomicNums[idx]) ?? .C
            let existing = featureForAtom(idx)
            let isScaffold = mcsMatchedAtoms.isEmpty || mcsMatchedAtoms.contains(idx)

            HStack(spacing: DS.Space.m) {
                HStack(spacing: DS.Space.xs) {
                    Text("Atom \(idx)").fontWeight(.semibold).monospacedDigit()
                    Text(elem.symbol).fontWeight(.bold).foregroundStyle(elementColor(coords.atomicNums[idx]))
                }
                if atomHasAromaticBond(idx) {
                    Text("aromatic").foregroundStyle(DS.Color.pocket)
                }
                if !mcsMatchedAtoms.isEmpty {
                    Text(isScaffold ? "scaffold" : "decoration")
                        .foregroundStyle(isScaffold ? DS.Color.success : Color.secondary)
                }
                if let f = existing {
                    HStack(spacing: DS.Space.xs) {
                        Image(systemName: f.type.icon).foregroundStyle(f.type.dsColor)
                        Text(f.name).fontWeight(.medium)
                        Text("(click to remove)").foregroundStyle(DS.Color.danger)
                    }
                } else {
                    Text("Click to add • Shift-click for group").foregroundStyle(.secondary)
                }
            }
            .modifier(StructureStatusChrome())
        }
    }

    // =========================================================================
    // MARK: - Right Panel
    // =========================================================================

    private var rightPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                sourceCard
                featuresCard
            }
            .padding(DS.Space.l)
        }
        .scrollIndicators(.automatic)
    }

    private var sourceCard: some View {
        let (title, icon): (String, String) = {
            switch sourceMode {
            case .singleLigand:   return ("Ligand", "hexagon")
            case .commonScaffold: return ("Common Scaffold", "square.on.square.intersection.dashed")
            case .smiles:         return ("SMILES", "character.cursor.ibeam")
            case .draw:           return (drawValidated ? "Drawn Structure" : "Draw Scaffold", "pencil.and.outline")
            }
        }()
        return PanelCard(title, icon: icon) {
            switch sourceMode {
            case .singleLigand:   singleLigandPanel
            case .commonScaffold: commonScaffoldPanel
            case .smiles:         smilesPanel
            case .draw:           drawPanel
            }
        }
    }

    @ViewBuilder
    private var singleLigandPanel: some View {
        let entries = viewModel.ligandDB.entries
        if entries.isEmpty {
            PanelHint(text: "No ligands in the database. Import ligands or use SMILES.", icon: "info.circle")
        } else {
            Picker("Ligand", selection: $selectedEntryID) {
                Text("Select…").tag(nil as UUID?)
                ForEach(entries) { Text($0.name).tag($0.id as UUID?) }
            }
            .labelsHidden()
            .controlSize(.small)
            .onChange(of: selectedEntryID) { _, id in
                if let id, let e = entries.first(where: { $0.id == id }) {
                    loadStructure(smiles: e.smiles)
                }
            }
        }
    }

    @ViewBuilder
    private var commonScaffoldPanel: some View {
        let entries = viewModel.ligandDB.entries
        if entries.count < 2 {
            PanelHint(text: "Need at least 2 ligands in the database.", icon: "info.circle")
        } else {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                HStack {
                    Text("\(selectedForMCS.count)/\(entries.count) selected")
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button(selectedForMCS.count == entries.count ? "None" : "All") {
                        selectedForMCS = selectedForMCS.count == entries.count ? [] : Set(entries.map(\.id))
                    }
                    .buttonStyle(.borderless)
                    .font(DS.Font.secondary.weight(.medium))
                }

                ScrollView(.vertical) {
                    VStack(spacing: 0) {
                        ForEach(entries) { entry in
                            mcsEntryRow(entry: entry)
                        }
                    }
                    .padding(DS.Space.xs)
                }
                .frame(maxHeight: 120)
                .background(
                    RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                        .fill(DS.Color.controlFill)
                )

                Button {
                    findMCS()
                } label: {
                    HStack(spacing: DS.Space.s) {
                        if isFindingMCS { ProgressView().controlSize(.mini) }
                        Text("Find Common Scaffold")
                    }
                }
                .buttonStyle(.dsPrimary)
                .disabled(selectedForMCS.count < 2 || isFindingMCS)

                if let smarts = mcsSmarts {
                    mcsResultView(smarts: smarts)
                }
            }
        }
    }

    private func mcsEntryRow(entry: LigandEntry) -> some View {
        Toggle(isOn: Binding(
            get: { selectedForMCS.contains(entry.id) },
            set: { on in
                if on { selectedForMCS.insert(entry.id) }
                else { selectedForMCS.remove(entry.id) }
            }
        )) {
            Text(entry.name)
                .font(DS.Font.secondary)
                .lineLimit(1)
        }
        .toggleStyle(.checkbox)
        .controlSize(.small)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, DS.Space.xxs)
        .padding(.horizontal, DS.Space.xs)
    }

    private func mcsResultView(smarts: String) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xs) {
            Label("\(mcsMatchedAtoms.count) shared atoms", systemImage: "checkmark.circle.fill")
                .font(DS.Font.secondary.weight(.medium))
                .foregroundStyle(DS.Color.success)
            smilesText(smarts)
            Toggle("Highlight scaffold only", isOn: $showOnlyScaffold)
                .font(DS.Font.secondary)
                .toggleStyle(.checkbox)
        }
    }

    private var smilesPanel: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            TextField("SMILES", text: $smilesInput)
                .textFieldStyle(.roundedBorder)
                .font(DS.Font.monoSmall)
                .onSubmit { loadStructure(smiles: smilesInput) }
            HStack(spacing: DS.Space.s) {
                Button("Load") { loadStructure(smiles: smilesInput) }
                    .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                    .disabled(smilesInput.isEmpty)
                Button {
                    if let s = NSPasteboard.general.string(forType: .string) {
                        smilesInput = s.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                } label: { Image(systemName: "doc.on.clipboard") }
                .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                .help("Paste SMILES from the clipboard")
                .accessibilityLabel("Paste SMILES")
                if let s = viewModel.molecules.ligand?.smiles, !s.isEmpty {
                    Button("Current Ligand") { smilesInput = s; loadStructure(smiles: s) }
                        .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                }
            }
        }
    }

    @ViewBuilder
    private var drawPanel: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            if drawValidated {
                Label("Structure validated", systemImage: "checkmark.circle.fill")
                    .font(DS.Font.secondary.weight(.medium))
                    .foregroundStyle(DS.Color.success)
                smilesText(activeSmiles)
                if !drawnSmarts.isEmpty {
                    VStack(alignment: .leading, spacing: DS.Space.xxs) {
                        Text("SMARTS")
                            .font(DS.Font.eyebrow)
                            .tracking(0.6)
                            .foregroundStyle(.tertiary)
                        smilesText(drawnSmarts)
                    }
                }
                PanelSecondaryButton(title: "Edit in Drawer", icon: "pencil") {
                    drawValidated = false
                    features = []; featureOverlays = []; selectedFeatureID = nil
                    pendingMultiSelect = []
                }
            } else {
                PanelHint(text: "Draw a chemical scaffold in the editor. It is used as a SMARTS query for pharmacophore enforcement.",
                          icon: "info.circle")
                if !activeSmiles.isEmpty {
                    Label("Structure loaded", systemImage: "checkmark.circle.fill")
                        .font(DS.Font.secondary.weight(.medium))
                        .foregroundStyle(DS.Color.success)
                    smilesText(activeSmiles)
                }
                Button {
                    validateDrawnStructure()
                } label: {
                    Label("Validate Structure", systemImage: "checkmark.shield")
                }
                .buttonStyle(.dsPrimary)
                .disabled(activeSmiles.isEmpty)
            }
        }
    }

    private func smilesText(_ text: String) -> some View {
        Text(text)
            .font(DS.Font.monoSmall)
            .foregroundStyle(.secondary)
            .lineLimit(3)
            .truncationMode(.middle)
            .textSelection(.enabled)
    }

    // MARK: Features

    private var featuresCard: some View {
        PanelCard(
            "Features",
            icon: "list.number",
            accessory: { DSTag("\(features.count)", tone: features.isEmpty ? .neutral : .accent, monospaced: true) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if !pendingMultiSelect.isEmpty {
                    HStack(spacing: DS.Space.s) {
                        Button {
                            commitMultiSelect()
                        } label: {
                            Label("Confirm Group (\(pendingMultiSelect.count))", systemImage: "checkmark.circle.fill")
                        }
                        .buttonStyle(DSPrimaryButtonStyle(tint: DS.Color.warning))
                        Button("Cancel") { pendingMultiSelect = [] }
                            .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                    }
                }

                HStack(spacing: DS.Space.s) {
                    Button {
                        autoDetectFeatures()
                    } label: {
                        HStack(spacing: DS.Space.xs) {
                            if isDetecting { ProgressView().controlSize(.mini) }
                            else { Image(systemName: "wand.and.stars") }
                            Text("Auto-Detect")
                        }
                    }
                    .buttonStyle(.dsSecondary)
                    .disabled(coords2D == nil || isDetecting)
                    .help("Detect donors, acceptors, aromatic rings and hydrophobic groups")

                    PanelSecondaryButton(
                        title: "Clear All", icon: "trash",
                        isDisabled: features.isEmpty && pendingMultiSelect.isEmpty
                    ) {
                        features.removeAll(); featureOverlays = []; selectedFeatureID = nil
                        pendingMultiSelect = []
                    }
                }

                if !featureOverlays.isEmpty { featureLegend }

                if features.isEmpty {
                    PanelHint(text: "Click atoms on the structure or use Auto-Detect.", icon: "hand.point.up.left")
                } else {
                    HStack(spacing: DS.Space.s) {
                        Button("All Essential") {
                            for i in features.indices { features[i].isEssential = true }
                        }
                        Button("All Optional") {
                            for i in features.indices { features[i].isEssential = false }
                        }
                        Spacer()
                    }
                    .buttonStyle(.borderless)
                    .font(DS.Font.secondary.weight(.medium))

                    VStack(spacing: DS.Space.s) {
                        ForEach(features.indices, id: \.self) { idx in
                            featureCard(idx: idx)
                        }
                    }
                }
            }
        }
    }

    private var featureLegend: some View {
        let typeCounts = Dictionary(grouping: featureOverlays, by: \.type)
        return FlowLayout(spacing: DS.Space.s) {
            ForEach(Array(typeCounts.keys.sorted(by: { $0.rawValue < $1.rawValue })), id: \.self) { type in
                HStack(spacing: DS.Space.xs) {
                    Circle().fill(type.dsColor).frame(width: 7, height: 7)
                    Text("\(typeCounts[type]?.count ?? 0) \(shortTypeName(type))")
                        .monospacedDigit()
                }
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: Feature Card (split into sub-views for type-checker)

    private func featureCard(idx: Int) -> some View {
        let f = features[idx]
        let isSelected = selectedFeatureID == f.id
        let typeColor = f.type.dsColor
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)

        return VStack(alignment: .leading, spacing: DS.Space.s) {
            featureCardHeader(idx: idx, f: f, typeColor: typeColor)
            featureCardControls(idx: idx)
            featureCardRadius(idx: idx, f: f)
        }
        .padding(DS.Space.m)
        .background(shape.fill(isSelected ? typeColor.opacity(0.08) : DS.Color.cardFill))
        .overlay(shape.strokeBorder(isSelected ? typeColor.opacity(0.45) : DS.Color.hairline,
                                    lineWidth: isSelected ? 1 : 0.5))
        .contentShape(shape)
        .onTapGesture { selectedFeatureID = f.id }
        .opacity(f.isIgnored ? 0.5 : 1.0)
    }

    private func featureCardHeader(idx: Int, f: PharmaFeature, typeColor: Color) -> some View {
        HStack(spacing: DS.Space.s) {
            Image(systemName: f.type.icon)
                .font(DS.Font.label)
                .foregroundStyle(f.isIgnored ? Color.secondary : typeColor)
                .frame(width: 16)
            Text(f.name)
                .font(DS.Font.mono.weight(.bold))
            Text(f.expression)
                .font(DS.Font.monoSmall)
                .foregroundStyle(.secondary)
            if f.atomIndices.count > 1 {
                DSTag("\(f.atomIndices.count) atoms", monospaced: true)
            }
            Spacer(minLength: DS.Space.xs)
            Button { features[idx].isIgnored.toggle() } label: {
                Image(systemName: f.isIgnored ? "eye.slash" : "eye")
                    .font(DS.Font.secondary)
                    .foregroundStyle(f.isIgnored ? DS.Color.danger : Color.secondary)
            }
            .buttonStyle(.borderless)
            .help(f.isIgnored ? "Include this feature" : "Ignore this feature")
            Button { features.remove(at: idx); selectedFeatureID = nil } label: {
                Image(systemName: "xmark")
                    .font(DS.Font.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.borderless)
            .help("Remove feature")
        }
    }

    private func featureCardControls(idx: Int) -> some View {
        HStack(spacing: DS.Space.s) {
            Picker("Requirement", selection: $features[idx].isEssential) {
                Text("Optional").tag(false)
                Text("Essential").tag(true)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
            .help("Essential features become hard constraints; optional ones are soft")

            Spacer(minLength: DS.Space.xs)

            Picker("Type", selection: Binding(
                get: { features[idx].type },
                set: { features[idx].type = $0; features[idx].expression = expressionLabel($0) }
            )) {
                ForEach(ConstraintInteractionType.allCases, id: \.self) {
                    Text(expressionLabel($0)).tag($0)
                }
            }
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
        }
    }

    private func featureCardRadius(idx: Int, f: PharmaFeature) -> some View {
        HStack(spacing: DS.Space.s) {
            Text("Radius")
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
            Slider(value: snappedSliderBinding($features[idx].radius, range: 0.5...4.0, step: 0.25), in: 0.5...4.0)
                .controlSize(.small)
            Text(String(format: "%.1f Å", f.radius))
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .fixedSize()
        }
    }

    // =========================================================================
    // MARK: - Footer
    // =========================================================================

    private var footer: some View {
        let active = features.filter { !$0.isIgnored }
        let essential = active.filter(\.isEssential).count
        let optional = active.count - essential
        return HStack(spacing: DS.Space.m) {
            if essential > 0 { DSTag("\(essential) essential", tone: .danger, monospaced: true) }
            if optional > 0 { DSTag("\(optional) optional", monospaced: true) }
            Spacer()
            Button("Cancel", role: .cancel) { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Apply to Docking") { applyFeatures(); dismiss() }
                .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                .keyboardShortcut(.defaultAction)
                .disabled(active.isEmpty)
        }
        .padding(DS.Space.l)
    }

    // =========================================================================
    // MARK: - Canvas Drawing
    // =========================================================================

    private func drawFeatureOverlayHalos(ctx: GraphicsContext, tx: StructureTransform) {
        for overlay in featureOverlays {
            let pt = tx.toView(overlay.centroid)
            let color = overlay.type.dsColor
            let r: CGFloat = 20
            let rect = CGRect(x: pt.x - r, y: pt.y - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color.opacity(0.1)))
            ctx.stroke(Path(ellipseIn: rect), with: .color(color.opacity(0.3)),
                       style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        }
    }

    private func drawBonds(ctx: GraphicsContext, tx: StructureTransform) {
        guard let coords = coords2D else { return }
        let scaffoldDim = !mcsMatchedAtoms.isEmpty && showOnlyScaffold

        for (a1, a2, order) in coords.bonds {
            guard a1 < coords.positions.count, a2 < coords.positions.count else { continue }
            let p1 = tx.toView(coords.positions[a1])
            let p2 = tx.toView(coords.positions[a2])
            let inScaffold = !scaffoldDim || (mcsMatchedAtoms.contains(a1) && mcsMatchedAtoms.contains(a2))
            let color = DS.Color.diagramInk.opacity(inScaffold ? 0.9 : 0.12)

            if order == 2 {
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = max(sqrt(dx * dx + dy * dy), 0.01)
                let nx = -dy / len * 2.0, ny = dx / len * 2.0
                for s in [-1.0, 1.0] {
                    var p = Path()
                    p.move(to: CGPoint(x: p1.x + nx * s, y: p1.y + ny * s))
                    p.addLine(to: CGPoint(x: p2.x + nx * s, y: p2.y + ny * s))
                    ctx.stroke(p, with: .color(color), lineWidth: 1.5)
                }
            } else if order == 3 {
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = max(sqrt(dx * dx + dy * dy), 0.01)
                let nx = -dy / len * 3.0, ny = dx / len * 3.0
                for o in [-1.0, 0.0, 1.0] {
                    var p = Path()
                    p.move(to: CGPoint(x: p1.x + nx * o, y: p1.y + ny * o))
                    p.addLine(to: CGPoint(x: p2.x + nx * o, y: p2.y + ny * o))
                    ctx.stroke(p, with: .color(color), lineWidth: 1.2)
                }
            } else {
                var p = Path()
                p.move(to: p1); p.addLine(to: p2)
                ctx.stroke(p, with: .color(color), lineWidth: 2.0)
            }
        }
    }

    /// Skeletal-formula style: heteroatoms (and featured carbons) get their symbol on a
    /// paper-coloured disc that masks the bonds; plain carbons are small ink dots.
    private func drawAtoms(ctx: GraphicsContext, tx: StructureTransform) {
        guard let coords = coords2D else { return }
        let scaffoldDim = !mcsMatchedAtoms.isEmpty && showOnlyScaffold

        for (idx, pos) in coords.positions.enumerated() {
            guard idx < coords.atomicNums.count else { continue }
            let vp = tx.toView(pos)
            let an = coords.atomicNums[idx]
            let inScaffold = !scaffoldDim || mcsMatchedAtoms.contains(idx)
            let hasFeature = featureForAtom(idx) != nil
            let opacity: Double = inScaffold ? 1.0 : 0.12

            if an != 6 || hasFeature {
                let r: CGFloat = hasFeature ? 11 : 9
                let rect = CGRect(x: vp.x - r, y: vp.y - r, width: r * 2, height: r * 2)
                ctx.fill(Path(ellipseIn: rect), with: .color(DS.Color.diagramPaper))
                let symbol = (Element(rawValue: an) ?? .C).symbol
                let label = Text(symbol)
                    .font(DS.Font.label.weight(.bold))
                    .foregroundColor(elementColor(an).opacity(opacity))
                ctx.draw(ctx.resolve(label), at: vp, anchor: .center)
            } else {
                let r: CGFloat = 2.5
                let rect = CGRect(x: vp.x - r, y: vp.y - r, width: r * 2, height: r * 2)
                ctx.fill(Path(ellipseIn: rect), with: .color(DS.Color.diagramInk.opacity(0.45 * opacity)))
            }
        }
    }

    private func drawFeatureSpheres(ctx: GraphicsContext, tx: StructureTransform) {
        guard let coords = coords2D else { return }
        // Track label stacks for features sharing the same atom set (e.g. an
        // aromatic ring assigned both Aro and Hyd) so labels don't superpose.
        var stackIndexByKey: [String: Int] = [:]
        for f in features where !f.isIgnored {
            var cx: CGFloat = 0, cy: CGFloat = 0, count = 0
            for idx in f.atomIndices where idx < coords.positions.count {
                cx += coords.positions[idx].x; cy += coords.positions[idx].y; count += 1
            }
            guard count > 0 else { continue }
            cx /= CGFloat(count); cy /= CGFloat(count)
            let vp = tx.toView(CGPoint(x: cx, y: cy))

            let color = f.type.dsColor
            let ringR: CGFloat = 16 + CGFloat(f.radius) * 3
            let rect = CGRect(x: vp.x - ringR, y: vp.y - ringR, width: ringR * 2, height: ringR * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color.opacity(f.isEssential ? 0.15 : 0.07)))

            if f.isEssential {
                ctx.stroke(Path(ellipseIn: rect), with: .color(color.opacity(0.85)), lineWidth: 2.5)
            } else {
                ctx.stroke(Path(ellipseIn: rect), with: .color(color.opacity(0.6)),
                           style: StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
            }

            let stackKey = f.atomIndices.sorted().map(String.init).joined(separator: ",")
            let stackIdx = stackIndexByKey[stackKey, default: 0]
            stackIndexByKey[stackKey] = stackIdx + 1
            let labelOffset: CGFloat = 5 + CGFloat(stackIdx) * 12

            let label = Text("\(f.name) \(f.expression)")
                .font(DS.Font.canvasLabel.monospaced().weight(.bold))
                .foregroundColor(color)
            ctx.draw(ctx.resolve(label), at: CGPoint(x: vp.x, y: vp.y - ringR - labelOffset), anchor: .bottom)
        }
    }

    // =========================================================================
    // MARK: - Ring Detection
    // =========================================================================

    private func findAromaticRing(containing atomIdx: Int) -> [Int]? {
        guard let coords = coords2D else { return nil }

        // Build adjacency from bonds between aromatic atoms
        var adj: [Int: Set<Int>] = [:]
        for (a1, a2, _) in coords.bonds {
            guard a1 < coords.isAromatic.count, a2 < coords.isAromatic.count,
                  coords.isAromatic[a1], coords.isAromatic[a2] else { continue }
            adj[a1, default: []].insert(a2)
            adj[a2, default: []].insert(a1)
        }
        guard let neighbors = adj[atomIdx], !neighbors.isEmpty else { return nil }

        var bestRing: [Int]? = nil
        for startNeighbor in neighbors {
            var queue: [(node: Int, path: [Int])] = [(startNeighbor, [atomIdx, startNeighbor])]
            var visited: Set<Int> = [atomIdx, startNeighbor]

            while !queue.isEmpty {
                let (current, path) = queue.removeFirst()
                for next in adj[current] ?? [] {
                    if next == atomIdx && path.count >= 3 {
                        if bestRing == nil || path.count < bestRing!.count { bestRing = path }
                        continue
                    }
                    if visited.contains(next) || path.count > 8 { continue }
                    visited.insert(next)
                    queue.append((next, path + [next]))
                }
            }
        }
        return bestRing
    }

    private func atomHasAromaticBond(_ idx: Int) -> Bool {
        guard let coords = coords2D, idx < coords.isAromatic.count else { return false }
        return coords.isAromatic[idx]
    }

    // =========================================================================
    // MARK: - Actions
    // =========================================================================

    private func resetState() {
        coords2D = nil; features = []; featureOverlays = []
        mcsSmarts = nil; mcsMatchedAtoms = []; errorMessage = nil
        hoveredAtom = nil; selectedFeatureID = nil
        drawValidated = false; pendingMultiSelect = []
    }

    private func validateDrawnStructure() {
        guard !activeSmiles.isEmpty else { return }
        guard let c = RDKitBridge.compute2DCoords(smiles: activeSmiles) else {
            errorMessage = "Invalid structure — could not compute 2D layout"
            return
        }
        coords2D = c
        features = []; featureOverlays = []; selectedFeatureID = nil
        pendingMultiSelect = []
        drawValidated = true
    }

    private func loadStructure(smiles: String) {
        errorMessage = nil; features = []; featureOverlays = []
        mcsMatchedAtoms = []; activeSmiles = smiles; selectedFeatureID = nil
        guard let c = RDKitBridge.compute2DCoords(smiles: smiles) else {
            errorMessage = "Failed to compute 2D structure"; coords2D = nil; return
        }
        coords2D = c
    }

    private func handleAtomClick(_ idx: Int) {
        // Toggle off: click featured atom removes its feature
        if let existingIdx = features.firstIndex(where: { $0.atomIndices.contains(idx) }) {
            features.remove(at: existingIdx)
            selectedFeatureID = nil
            return
        }

        // Aromatic ring: group all ring atoms into one feature
        if atomHasAromaticBond(idx), let ring = findAromaticRing(containing: idx) {
            let ringSet = Set(ring)
            if let existing = features.firstIndex(where: { !Set($0.atomIndices).intersection(ringSet).isEmpty }) {
                selectedFeatureID = features[existing].id
                return
            }
            let name = "F\(nextFeatureNumber())"
            features.append(PharmaFeature(name: name, expression: "Aro", type: .piStacking, atomIndices: ring))
            selectedFeatureID = features.last?.id
            return
        }

        // Single atom
        let type = suggestedType(for: idx)
        let name = "F\(nextFeatureNumber())"
        features.append(PharmaFeature(name: name, expression: expressionLabel(type), type: type, atomIndices: [idx]))
        selectedFeatureID = features.last?.id
    }

    private func handleShiftClick(_ idx: Int) {
        // Toggle atom in/out of pending multi-select group
        if let existing = pendingMultiSelect.firstIndex(of: idx) {
            pendingMultiSelect.remove(at: existing)
        } else {
            // Don't add atoms that already belong to a feature
            guard !features.contains(where: { $0.atomIndices.contains(idx) }) else { return }
            pendingMultiSelect.append(idx)
        }
    }

    private func commitMultiSelect() {
        guard pendingMultiSelect.count >= 1 else { return }
        // Determine the best type for the group
        let type: ConstraintInteractionType
        if pendingMultiSelect.count == 1 {
            type = suggestedType(for: pendingMultiSelect[0])
        } else {
            // Multi-atom: default to hydrophobic (common for multi-select use case)
            type = .hydrophobic
        }
        let name = "F\(nextFeatureNumber())"
        features.append(PharmaFeature(name: name, expression: expressionLabel(type), type: type,
                                       atomIndices: pendingMultiSelect))
        selectedFeatureID = features.last?.id
        pendingMultiSelect = []
    }

    private func findMCS() {
        let entries = viewModel.ligandDB.entries
        let selected = entries.filter { selectedForMCS.contains($0.id) }
        let smiles = selected.map(\.smiles)
        guard smiles.count >= 2 else { return }

        isFindingMCS = true; errorMessage = nil; mcsSmarts = nil; mcsMatchedAtoms = []
        let firstSmiles = smiles[0]

        Task.detached {
            let result = RDKitBridge.findMCS(smilesArray: smiles, timeoutSeconds: 30)
            var matchedSet = Set<Int>()
            if let mcs = result, !mcs.smartsPattern.isEmpty {
                if let m = RDKitBridge.matchScaffold(smiles: firstSmiles, scaffoldSMARTS: mcs.smartsPattern),
                   m.hasMatch {
                    matchedSet = Set(m.matchedAtomIndices.map { Int($0) })
                }
            }
            let capturedMatchedSet = matchedSet
            await MainActor.run {
                isFindingMCS = false
                guard let mcs = result, !mcs.smartsPattern.isEmpty, mcs.numAtoms >= 2 else {
                    errorMessage = "No meaningful common substructure found"; return
                }
                mcsSmarts = mcs.smartsPattern
                loadStructure(smiles: firstSmiles)
                mcsMatchedAtoms = capturedMatchedSet
            }
        }
    }

    private func autoDetectFeatures() {
        guard !activeSmiles.isEmpty else { return }
        isDetecting = true
        let smiles = activeSmiles

        Task.detached {
            let detected = RDKitBridge.detectPharmacophoreFeatures(smiles: smiles)
            await MainActor.run {
                isDetecting = false
                guard let detected, !detected.isEmpty, let coords = coords2D else { return }

                var overlays: [FeatureOverlay] = []
                for f in detected {
                    let indices = f.atomIndices.map { Int($0) }.filter { $0 < coords.positions.count }
                    let filteredIndices: [Int]
                    if !mcsMatchedAtoms.isEmpty && showOnlyScaffold {
                        filteredIndices = indices.filter { mcsMatchedAtoms.contains($0) }
                    } else {
                        filteredIndices = indices
                    }
                    guard !filteredIndices.isEmpty else { continue }

                    var cx: CGFloat = 0, cy: CGFloat = 0
                    for i in filteredIndices { cx += coords.positions[i].x; cy += coords.positions[i].y }
                    cx /= CGFloat(filteredIndices.count); cy /= CGFloat(filteredIndices.count)
                    overlays.append(FeatureOverlay(type: f.type, atomIndices: filteredIndices,
                                                   centroid: CGPoint(x: cx, y: cy)))

                    // One grouped feature per pharmacophore point
                    let indexSet = Set(filteredIndices)
                    if features.contains(where: { !Set($0.atomIndices).intersection(indexSet).isEmpty }) { continue }

                    let type = f.type.constraintInteractionType
                    let name = "F\(nextFeatureNumber())"
                    features.append(PharmaFeature(
                        name: name, expression: expressionLabel(type),
                        type: type, atomIndices: filteredIndices
                    ))

                    // Aromatic rings also satisfy hydrophobic contacts — pair them.
                    if f.type == .aromatic {
                        let hydName = "F\(nextFeatureNumber())"
                        features.append(PharmaFeature(
                            name: hydName, expression: expressionLabel(.hydrophobic),
                            type: .hydrophobic, atomIndices: filteredIndices
                        ))
                        overlays.append(FeatureOverlay(type: .hydrophobic, atomIndices: filteredIndices,
                                                       centroid: CGPoint(x: cx, y: cy)))
                    }
                }
                featureOverlays = overlays
            }
        }
    }

    private func applyFeatures() {
        guard coords2D != nil else { return }
        let activeFeatures = features.filter { !$0.isIgnored }
        guard !activeFeatures.isEmpty else { return }

        let pos3D: [SIMD3<Float>]?
        if let id = selectedEntryID,
           let e = viewModel.ligandDB.entries.first(where: { $0.id == id }), !e.atoms.isEmpty {
            pos3D = e.atoms.filter { $0.element != .H }.map(\.position)
        } else if let lig = viewModel.molecules.ligand, lig.smiles == activeSmiles {
            pos3D = lig.atoms.filter { $0.element != .H }.map(\.position)
        } else {
            pos3D = RDKitBridge.smilesToMolecule(smiles: activeSmiles, numConformers: 10)
                .molecule?.atoms.filter { $0.element != .H }.map(\.position)
        }
        guard let positions = pos3D, !positions.isEmpty else { return }

        let baseGroupID = viewModel.docking.pharmacophoreConstraints.count
        var groupID = baseGroupID

        for f in activeFeatures {
            var centroid = SIMD3<Float>.zero; var count: Float = 0
            for idx in f.atomIndices where idx < positions.count {
                centroid += positions[idx]; count += 1
            }
            guard count > 0 else { continue }
            centroid /= count

            let strength: ConstraintStrength = f.isEssential
                ? .hard : .soft(kcalPerAngstromSq: 10.0)

            var def = PharmacophoreConstraintDef(
                targetScope: .atom, interactionType: f.type,
                strength: strength, distanceThreshold: f.radius, sourceType: .receptor
            )
            def.targetPositions = [centroid]
            def.groupID = groupID
            def.residueName = f.expression
            def.atomName = f.name

            viewModel.docking.pharmacophoreConstraints.append(def)
            groupID += 1
        }
        viewModel.pushToRenderer()
    }

    // =========================================================================
    // MARK: - Helpers
    // =========================================================================

    private func featureForAtom(_ idx: Int) -> PharmaFeature? {
        features.first { $0.atomIndices.contains(idx) && !$0.isIgnored }
    }

    private func nextFeatureNumber() -> Int {
        let nums = features.compactMap { f -> Int? in
            guard f.name.hasPrefix("F"), let n = Int(f.name.dropFirst()) else { return nil }
            return n
        }
        return (nums.max() ?? 0) + 1
    }

    private func suggestedType(for idx: Int) -> ConstraintInteractionType {
        guard let c = coords2D, idx < c.atomicNums.count else { return .hydrophobic }
        switch c.atomicNums[idx] {
        case 7: return .hbondDonor
        case 8: return .hbondAcceptor
        case 16: return .hbondAcceptor
        case 9, 17, 35, 53: return .halogen
        default: return .hydrophobic
        }
    }

    private func expressionLabel(_ type: ConstraintInteractionType) -> String {
        switch type {
        case .hbondDonor: return "Don"
        case .hbondAcceptor: return "Acc"
        case .saltBridge: return "Cat|Ani"
        case .piStacking: return "Aro"
        case .halogen: return "Hal"
        case .metalCoordination: return "Met"
        case .hydrophobic: return "Hyd"
        }
    }

    private func shortTypeName(_ type: PharmacophoreFeatureType) -> String {
        switch type {
        case .donor: return "Don"
        case .acceptor: return "Acc"
        case .hydrophobic: return "Hyd"
        case .aromatic: return "Aro"
        case .positiveIonizable: return "+Ion"
        case .negativeIonizable: return "-Ion"
        }
    }

    /// Element ink for the 2D diagram (light/dark aware, legible on `diagramPaper`).
    private func elementColor(_ n: Int) -> Color {
        switch n {
        case 6:      return DS.Color.diagramInk
        case 7:      return ElementInk.nitrogen
        case 8:      return ElementInk.oxygen
        case 16:     return ElementInk.sulfur
        case 9, 17:  return ElementInk.lightHalogen
        case 35, 53: return ElementInk.heavyHalogen
        case 15:     return ElementInk.phosphorus
        default:     return ElementInk.other
        }
    }
}

// MARK: - Status chip chrome (floats over the structure canvas)

private struct StructureStatusChrome: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(DS.Font.caption)
            .padding(.horizontal, DS.Space.l)
            .padding(.vertical, DS.Space.s)
            .dsFloating(cornerRadius: DS.Radius.m)
            .padding(.bottom, DS.Space.m)
    }
}

// MARK: - Diagram palettes

/// Element colours for the 2D structure (data palette, tuned per appearance).
private enum ElementInk {
    static let nitrogen     = DS.dynamic(light: 0x2F5BD1, dark: 0x7AA2FF)
    static let oxygen       = DS.dynamic(light: 0xCF222E, dark: 0xFF7B72)
    static let sulfur       = DS.dynamic(light: 0x9A6700, dark: 0xE3B341)
    static let lightHalogen = DS.dynamic(light: 0x1A7F37, dark: 0x3FB950)
    static let heavyHalogen = DS.dynamic(light: 0x9A2B1B, dark: 0xE0826B)
    static let phosphorus   = DS.dynamic(light: 0xBC4C00, dark: 0xF0883E)
    static let other        = DS.dynamic(light: 0x6E7781, dark: 0x8C959F)
}

extension ConstraintInteractionType {
    /// UI colour for this constraint type (hues follow the 3D constraint markers).
    var dsColor: Color {
        switch self {
        case .hbondDonor:        DS.Color.ligand
        case .hbondAcceptor:     DS.Color.water
        case .saltBridge:        DS.Color.interactionPiCation
        case .piStacking:        DS.Color.pocket
        case .halogen:           DS.Color.interactionPiStack
        case .metalCoordination: DS.Color.interactionMetal
        case .hydrophobic:       DS.Color.interactionHydrophobic
        }
    }
}

extension PharmacophoreFeatureType {
    /// UI colour for this detected feature type (matches `ConstraintInteractionType.dsColor`).
    var dsColor: Color {
        switch self {
        case .donor:             DS.Color.ligand
        case .acceptor:          DS.Color.water
        case .hydrophobic:       DS.Color.interactionHydrophobic
        case .aromatic:          DS.Color.pocket
        case .positiveIonizable: DS.Color.interactionHBond
        case .negativeIonizable: DS.Color.interactionSaltBridge
        }
    }
}

// MARK: - Coordinate Transform

struct StructureTransform {
    let scale: CGFloat
    let offsetX: CGFloat
    let offsetY: CGFloat

    init(positions: [CGPoint], viewSize: CGSize, padding: CGFloat) {
        guard !positions.isEmpty, viewSize.width > 0, viewSize.height > 0 else {
            self.scale = 1; self.offsetX = 0; self.offsetY = 0; return
        }
        let xs = positions.map(\.x), ys = positions.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!
        let minY = ys.min()!, maxY = ys.max()!
        let spanX = max(maxX - minX, 1), spanY = max(maxY - minY, 1)
        let s = min((viewSize.width - padding * 2) / spanX, (viewSize.height - padding * 2) / spanY)
        self.scale = s
        self.offsetX = (viewSize.width - spanX * s) / 2 - minX * s
        self.offsetY = (viewSize.height - spanY * s) / 2 - minY * s
    }

    func toView(_ p: CGPoint) -> CGPoint {
        CGPoint(x: p.x * scale + offsetX, y: p.y * scale + offsetY)
    }
}
