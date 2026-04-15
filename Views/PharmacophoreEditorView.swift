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
    @State private var showDrawer: Bool = false
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
            toolbar
            Divider()
            HSplitView {
                structureView
                    .frame(minWidth: 440)
                rightPanel
                    .frame(width: 300)
            }
            Divider()
            footer
        }
        .frame(minWidth: 820, minHeight: 580)
        .onAppear {
            if !viewModel.ligandDB.entries.isEmpty {
                sourceMode = .singleLigand
                selectedForMCS = Set(viewModel.ligandDB.entries.map(\.id))
            }
        }
    }

    // =========================================================================
    // MARK: - Toolbar
    // =========================================================================

    @ViewBuilder
    private var toolbar: some View {
        HStack(spacing: 10) {
            Image(systemName: "circle.hexagongrid.circle")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.purple)
            Text("Pharmacophore Editor")
                .font(.headline)

            Spacer()

            Picker("", selection: $sourceMode) {
                ForEach(SourceMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 320)
            .onChange(of: sourceMode) { _, _ in resetState() }

            Spacer()

            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    // =========================================================================
    // MARK: - Structure View (Canvas-based)
    // =========================================================================

    @ViewBuilder
    private var structureView: some View {
        ZStack {
            Color(nsColor: .controlBackgroundColor)

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
                                    .stroke(Color.orange, lineWidth: 2.5)
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
                                .stroke(hasFeature ? Color.red.opacity(0.6)
                                        : inMultiSelect ? Color.orange.opacity(0.8)
                                        : Color.white.opacity(0.5), lineWidth: 2)
                                .frame(width: 28, height: 28)
                                .position(x: pos.x, y: pos.y)
                                .allowsHitTesting(false)
                        }
                    }
                    .overlay(alignment: .bottom) { statusBar }
                }
            } else if isFindingMCS || isDetecting {
                VStack(spacing: 8) {
                    ProgressView()
                    Text(isFindingMCS ? "Computing common scaffold..." : "Detecting features...")
                        .font(.callout).foregroundStyle(.secondary)
                }
            } else if let error = errorMessage {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle).foregroundStyle(.red.opacity(0.5))
                    Text(error)
                        .font(.callout).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center).padding(.horizontal)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "hexagon")
                        .font(.system(size: 36)).foregroundStyle(.quaternary)
                    Text("Load a structure to begin")
                        .font(.callout).foregroundStyle(.tertiary)
                }
            }
        }
    }

    // MARK: Status Bar

    @ViewBuilder
    private var statusBar: some View {
        if !pendingMultiSelect.isEmpty {
            HStack(spacing: 10) {
                Image(systemName: "rectangle.dashed.badge.record").foregroundStyle(.orange)
                Text("\(pendingMultiSelect.count) atom\(pendingMultiSelect.count == 1 ? "" : "s") selected")
                    .fontWeight(.medium).foregroundStyle(.orange)
                Text("shift+click to add more, click to confirm group")
                    .foregroundStyle(.secondary)
            }
            .font(.system(size: 10))
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .padding(.bottom, 8)
        } else if let idx = hoveredAtom, let coords = coords2D, idx < coords.atomicNums.count {
            let elem = Element(rawValue: coords.atomicNums[idx]) ?? .C
            let existing = featureForAtom(idx)
            let isScaffold = mcsMatchedAtoms.isEmpty || mcsMatchedAtoms.contains(idx)
            let isAro = atomHasAromaticBond(idx)

            HStack(spacing: 10) {
                HStack(spacing: 3) {
                    Text("Atom \(idx)").fontWeight(.semibold)
                    Text(elem.symbol).foregroundStyle(elementColor(coords.atomicNums[idx])).fontWeight(.bold)
                }
                if isAro { Text("aromatic").foregroundStyle(.purple) }
                if !mcsMatchedAtoms.isEmpty {
                    Text(isScaffold ? "scaffold" : "decoration")
                        .foregroundStyle(isScaffold ? Color.green : Color.secondary)
                }
                if let f = existing {
                    HStack(spacing: 3) {
                        Image(systemName: f.type.icon).foregroundStyle(swiftColor(f.type.color))
                        Text(f.name).fontWeight(.medium)
                        Text("(click to remove)").foregroundStyle(.red)
                    }
                } else {
                    Text("click to add \u{2022} shift+click for group").foregroundStyle(.secondary)
                }
            }
            .font(.system(size: 10))
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .padding(.bottom, 8)
        }
    }

    // =========================================================================
    // MARK: - Right Panel
    // =========================================================================

    @ViewBuilder
    private var rightPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                sourcePanel
                Divider()
                toolsPanel
                Divider()
                featureListPanel
            }
            .padding(10)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    @ViewBuilder
    private var sourcePanel: some View {
        switch sourceMode {
        case .singleLigand: singleLigandPanel
        case .commonScaffold: commonScaffoldPanel
        case .smiles: smilesPanel
        case .draw: drawPanel
        }
    }

    @ViewBuilder
    private var singleLigandPanel: some View {
        let entries = viewModel.ligandDB.entries
        VStack(alignment: .leading, spacing: 4) {
            Label("Ligand", systemImage: "list.bullet")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            if entries.isEmpty {
                Text("No ligands in database.\nImport ligands or use SMILES.")
                    .font(.caption2).foregroundStyle(.tertiary)
            } else {
                Picker("", selection: $selectedEntryID) {
                    Text("Select...").tag(nil as UUID?)
                    ForEach(entries) { Text($0.name).tag($0.id as UUID?) }
                }
                .labelsHidden().controlSize(.small)
                .onChange(of: selectedEntryID) { _, id in
                    if let id, let e = entries.first(where: { $0.id == id }) {
                        loadStructure(smiles: e.smiles)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var commonScaffoldPanel: some View {
        let entries = viewModel.ligandDB.entries
        VStack(alignment: .leading, spacing: 4) {
            Label("Common Scaffold", systemImage: "square.on.square.intersection.dashed")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)

            if entries.count < 2 {
                Text("Need \u{2265}2 ligands.").font(.caption2).foregroundStyle(.tertiary)
            } else {
                HStack {
                    Text("\(selectedForMCS.count)/\(entries.count)").font(.caption.monospaced())
                    Spacer()
                    Button(selectedForMCS.count == entries.count ? "None" : "All") {
                        selectedForMCS = selectedForMCS.count == entries.count ? [] : Set(entries.map(\.id))
                    }
                    .font(.caption2).buttonStyle(.plain).foregroundStyle(.blue)
                }

                ScrollView(.vertical) {
                    VStack(spacing: 0) {
                        ForEach(entries) { entry in
                            mcsEntryRow(entry: entry)
                        }
                    }
                }
                .frame(maxHeight: 120)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))

                Button {
                    findMCS()
                } label: {
                    HStack {
                        if isFindingMCS { ProgressView().controlSize(.mini) }
                        Text("Find Common Scaffold").frame(maxWidth: .infinity)
                    }
                    .padding(.vertical, 2)
                }
                .buttonStyle(.borderedProminent).controlSize(.small)
                .disabled(selectedForMCS.count < 2 || isFindingMCS)

                if let smarts = mcsSmarts {
                    mcsResultView(smarts: smarts)
                }
            }
        }
    }

    @ViewBuilder
    private func mcsEntryRow(entry: LigandEntry) -> some View {
        HStack(spacing: 4) {
            Toggle("", isOn: Binding(
                get: { selectedForMCS.contains(entry.id) },
                set: { on in
                    if on { selectedForMCS.insert(entry.id) }
                    else { selectedForMCS.remove(entry.id) }
                }
            ))
            .toggleStyle(.checkbox).labelsHidden().controlSize(.small)
            Text(entry.name).font(.system(size: 10)).lineLimit(1)
            Spacer()
        }
        .padding(.vertical, 1).padding(.horizontal, 4)
    }

    @ViewBuilder
    private func mcsResultView(smarts: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 3) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
                Text("\(mcsMatchedAtoms.count) shared atoms").font(.caption.weight(.medium))
            }
            Text(smarts)
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(.tertiary).lineLimit(2).textSelection(.enabled)
            Toggle("Highlight scaffold only", isOn: $showOnlyScaffold)
                .font(.caption2).toggleStyle(.checkbox)
        }
    }

    @ViewBuilder
    private var smilesPanel: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("SMILES", systemImage: "character.cursor.ibeam")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            TextField("SMILES", text: $smilesInput)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 10, design: .monospaced))
                .onSubmit { loadStructure(smiles: smilesInput) }
            HStack(spacing: 4) {
                Button("Load") { loadStructure(smiles: smilesInput) }
                    .buttonStyle(.borderedProminent).controlSize(.small)
                    .disabled(smilesInput.isEmpty)
                Button {
                    if let s = NSPasteboard.general.string(forType: .string) {
                        smilesInput = s.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                } label: { Image(systemName: "doc.on.clipboard") }
                .buttonStyle(.bordered).controlSize(.small)
                if let s = viewModel.molecules.ligand?.smiles, !s.isEmpty {
                    Button("Current Ligand") { smilesInput = s; loadStructure(smiles: s) }
                        .buttonStyle(.bordered).controlSize(.small)
                }
            }
        }
    }

    @ViewBuilder
    private var drawPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            if drawValidated {
                // After validation — show structure info and allow editing back
                Label("Drawn Structure", systemImage: "pencil.and.outline")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 3) {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
                        Text("Structure validated").font(.caption.weight(.medium))
                    }
                    Text(activeSmiles)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .textSelection(.enabled)
                    if !drawnSmarts.isEmpty {
                        HStack(spacing: 3) {
                            Text("SMARTS:").font(.system(size: 9, weight: .semibold)).foregroundStyle(.purple)
                            Text(drawnSmarts)
                                .font(.system(size: 8, design: .monospaced))
                                .foregroundStyle(.tertiary)
                                .lineLimit(2)
                                .textSelection(.enabled)
                        }
                    }
                }

                Button {
                    drawValidated = false
                    features = []; featureOverlays = []; selectedFeatureID = nil
                    pendingMultiSelect = []
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "pencil")
                        Text("Edit in Drawer")
                    }
                    .font(.system(size: 10)).frame(maxWidth: .infinity).padding(.vertical, 2)
                }
                .buttonStyle(.bordered).controlSize(.small)
            } else {
                // Before validation — drawing mode
                Label("Draw Scaffold", systemImage: "pencil.and.outline")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)

                Text("Draw a chemical scaffold in the editor.\nThe structure will be used as a SMARTS query for pharmacophore enforcement.")
                    .font(.caption2).foregroundStyle(.tertiary)

                if !activeSmiles.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 3) {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
                            Text("Structure loaded").font(.caption.weight(.medium))
                        }
                        Text(activeSmiles)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .textSelection(.enabled)
                    }
                }

                Button {
                    validateDrawnStructure()
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "checkmark.shield")
                        Text("Validate Structure")
                    }
                    .font(.system(size: 10)).frame(maxWidth: .infinity).padding(.vertical, 2)
                }
                .buttonStyle(.borderedProminent).controlSize(.small)
                .disabled(activeSmiles.isEmpty)
            }
        }
    }

    // MARK: Tools

    @ViewBuilder
    private var toolsPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Multi-select confirmation bar
            if !pendingMultiSelect.isEmpty {
                HStack(spacing: 4) {
                    Button {
                        commitMultiSelect()
                    } label: {
                        HStack(spacing: 3) {
                            Image(systemName: "checkmark.circle.fill")
                            Text("Confirm Group (\(pendingMultiSelect.count))")
                        }
                        .font(.system(size: 10)).frame(maxWidth: .infinity).padding(.vertical, 2)
                    }
                    .buttonStyle(.borderedProminent).controlSize(.small)
                    .tint(.orange)

                    Button {
                        pendingMultiSelect = []
                    } label: {
                        HStack(spacing: 3) {
                            Image(systemName: "xmark")
                            Text("Cancel")
                        }
                        .font(.system(size: 10)).padding(.vertical, 2)
                    }
                    .buttonStyle(.bordered).controlSize(.small)
                }
            }

            HStack(spacing: 4) {
                Button {
                    autoDetectFeatures()
                } label: {
                    HStack(spacing: 3) {
                        if isDetecting { ProgressView().controlSize(.mini) }
                        else { Image(systemName: "wand.and.stars") }
                        Text("Auto-Detect")
                    }
                    .font(.system(size: 10)).frame(maxWidth: .infinity).padding(.vertical, 2)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(coords2D == nil || isDetecting)

                Button {
                    features.removeAll(); featureOverlays = []; selectedFeatureID = nil
                    pendingMultiSelect = []
                } label: {
                    HStack(spacing: 3) { Image(systemName: "trash"); Text("Clear All") }
                    .font(.system(size: 10)).frame(maxWidth: .infinity).padding(.vertical, 2)
                }
                .buttonStyle(.bordered).controlSize(.small).disabled(features.isEmpty && pendingMultiSelect.isEmpty)
            }

            if !featureOverlays.isEmpty { featureLegend }
        }
    }

    @ViewBuilder
    private var featureLegend: some View {
        let typeCounts = Dictionary(grouping: featureOverlays, by: \.type)
        HStack(spacing: 8) {
            ForEach(Array(typeCounts.keys.sorted(by: { $0.rawValue < $1.rawValue })), id: \.self) { type in
                HStack(spacing: 2) {
                    Circle().fill(swiftColor(type.color).opacity(0.6)).frame(width: 7, height: 7)
                    Text("\(typeCounts[type]!.count) \(shortTypeName(type))")
                        .font(.system(size: 9)).foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: Feature List

    @ViewBuilder
    private var featureListPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("Features (\(features.count))", systemImage: "list.number")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
            }

            if !features.isEmpty {
                HStack(spacing: 4) {
                    Button("All Essential") {
                        for i in features.indices { features[i].isEssential = true }
                    }
                    .font(.system(size: 9)).buttonStyle(.bordered).controlSize(.mini)
                    Button("All Optional") {
                        for i in features.indices { features[i].isEssential = false }
                    }
                    .font(.system(size: 9)).buttonStyle(.bordered).controlSize(.mini)
                    Spacer()
                }
            }

            if features.isEmpty {
                Text("Click atoms on the structure\nor use Auto-Detect.")
                    .font(.caption2).foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center).padding(.vertical, 8)
            } else {
                ForEach(features.indices, id: \.self) { idx in
                    featureCard(idx: idx)
                }
            }
        }
    }

    // MARK: Feature Card (split into sub-views for type-checker)

    @ViewBuilder
    private func featureCard(idx: Int) -> some View {
        let f = features[idx]
        let isSelected = selectedFeatureID == f.id
        let typeColor = swiftColor(f.type.color)

        VStack(spacing: 5) {
            featureCardHeader(idx: idx, f: f, typeColor: typeColor)
            featureCardControls(idx: idx, f: f)
            featureCardRadius(idx: idx, f: f)
        }
        .padding(8)
        .background(isSelected ? typeColor.opacity(0.06) : Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? typeColor.opacity(0.4) : Color.gray.opacity(0.08), lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture { selectedFeatureID = f.id }
        .opacity(f.isIgnored ? 0.5 : 1.0)
    }

    @ViewBuilder
    private func featureCardHeader(idx: Int, f: PharmaFeature, typeColor: Color) -> some View {
        HStack(spacing: 5) {
            Image(systemName: f.type.icon)
                .font(.system(size: 12))
                .foregroundStyle(f.isIgnored ? Color.gray : typeColor)
                .frame(width: 16)
            Text(f.name).font(.system(size: 11, weight: .bold, design: .monospaced))
            Text(f.expression).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            if f.atomIndices.count > 1 {
                Text("(\(f.atomIndices.count))").font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary)
            }
            Spacer()
            Button { features[idx].isIgnored.toggle() } label: {
                Image(systemName: features[idx].isIgnored ? "eye.slash" : "eye")
                    .font(.system(size: 10))
                    .foregroundStyle(features[idx].isIgnored ? Color.red.opacity(0.7) : Color.secondary)
            }
            .buttonStyle(.plain)
            Button { features.remove(at: idx); selectedFeatureID = nil } label: {
                Image(systemName: "xmark").font(.system(size: 9)).foregroundStyle(Color.secondary.opacity(0.4))
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder
    private func featureCardControls(idx: Int, f: PharmaFeature) -> some View {
        HStack(spacing: 4) {
            HStack(spacing: 0) {
                Button { features[idx].isEssential = false } label: {
                    Text("Optional")
                        .font(.system(size: 9, weight: .semibold))
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(!f.isEssential ? Color.accentColor.opacity(0.15) : Color.clear)
                        .foregroundStyle(!f.isEssential ? Color.primary : Color.secondary.opacity(0.5))
                }
                .buttonStyle(.plain)
                Button { features[idx].isEssential = true } label: {
                    Text("Essential")
                        .font(.system(size: 9, weight: .semibold))
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(f.isEssential ? Color.red.opacity(0.15) : Color.clear)
                        .foregroundStyle(f.isEssential ? Color.red : Color.secondary.opacity(0.5))
                }
                .buttonStyle(.plain)
            }
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.gray.opacity(0.2), lineWidth: 0.5))

            Spacer()

            Picker("", selection: Binding(
                get: { features[idx].type },
                set: { features[idx].type = $0; features[idx].expression = expressionLabel($0) }
            )) {
                ForEach(ConstraintInteractionType.allCases, id: \.self) {
                    Text(expressionLabel($0)).tag($0)
                }
            }
            .controlSize(.mini).frame(width: 80)
        }
    }

    @ViewBuilder
    private func featureCardRadius(idx: Int, f: PharmaFeature) -> some View {
        HStack(spacing: 4) {
            Text("Radius").font(.system(size: 9)).foregroundStyle(.secondary).frame(width: 36, alignment: .leading)
            Slider(value: $features[idx].radius, in: 0.5...4.0, step: 0.25)
            Text("\(String(format: "%.1f", f.radius)) \u{00C5}")
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary).frame(width: 32)
        }
    }

    // =========================================================================
    // MARK: - Footer
    // =========================================================================

    @ViewBuilder
    private var footer: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Spacer()
            if !features.isEmpty {
                let active = features.filter { !$0.isIgnored }
                let essential = active.filter(\.isEssential).count
                let optional = active.count - essential
                HStack(spacing: 8) {
                    if essential > 0 { Text("\(essential) essential").font(.footnote.weight(.medium)).foregroundStyle(.red) }
                    if optional > 0 { Text("\(optional) optional").font(.footnote).foregroundStyle(.secondary) }
                }
            }
            Button("Apply to Docking") { applyFeatures(); dismiss() }
                .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
                .disabled(features.filter({ !$0.isIgnored }).isEmpty)
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
    }

    // =========================================================================
    // MARK: - Canvas Drawing
    // =========================================================================

    private func drawFeatureOverlayHalos(ctx: GraphicsContext, tx: StructureTransform) {
        for overlay in featureOverlays {
            let pt = tx.toView(overlay.centroid)
            let color = swiftColor(overlay.type.color)
            let r: CGFloat = 20
            let rect = CGRect(x: pt.x - r, y: pt.y - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color.opacity(0.1)))
            ctx.stroke(Path(ellipseIn: rect), with: .color(color.opacity(0.25)),
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
            let color = Color(nsColor: .labelColor).opacity(inScaffold ? 1.0 : 0.12)

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

    private func drawAtoms(ctx: GraphicsContext, tx: StructureTransform) {
        guard let coords = coords2D else { return }
        let scaffoldDim = !mcsMatchedAtoms.isEmpty && showOnlyScaffold

        for (idx, pos) in coords.positions.enumerated() {
            guard idx < coords.atomicNums.count else { continue }
            let vp = tx.toView(pos)
            let an = coords.atomicNums[idx]
            let elem = Element(rawValue: an) ?? .C
            let inScaffold = !scaffoldDim || mcsMatchedAtoms.contains(idx)
            let hasFeature = featureForAtom(idx) != nil

            let r: CGFloat = hasFeature ? 13 : 10
            let opacity: Double = inScaffold ? 1.0 : 0.12
            let rect = CGRect(x: vp.x - r, y: vp.y - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(elementColor(an).opacity(opacity)))

            if an != 6 || hasFeature {
                let label = Text(elem.symbol)
                    .font(.system(size: hasFeature ? 11 : 10, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                ctx.draw(ctx.resolve(label), at: vp, anchor: .center)
            }
        }
    }

    private func drawFeatureSpheres(ctx: GraphicsContext, tx: StructureTransform) {
        guard let coords = coords2D else { return }
        for f in features where !f.isIgnored {
            var cx: CGFloat = 0, cy: CGFloat = 0, count = 0
            for idx in f.atomIndices where idx < coords.positions.count {
                cx += coords.positions[idx].x; cy += coords.positions[idx].y; count += 1
            }
            guard count > 0 else { continue }
            cx /= CGFloat(count); cy /= CGFloat(count)
            let vp = tx.toView(CGPoint(x: cx, y: cy))

            let color = swiftColor(f.type.color)
            let ringR: CGFloat = 16 + CGFloat(f.radius) * 3
            let rect = CGRect(x: vp.x - ringR, y: vp.y - ringR, width: ringR * 2, height: ringR * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color.opacity(f.isEssential ? 0.15 : 0.07)))

            if f.isEssential {
                ctx.stroke(Path(ellipseIn: rect), with: .color(color.opacity(0.8)), lineWidth: 2.5)
            } else {
                ctx.stroke(Path(ellipseIn: rect), with: .color(color.opacity(0.5)),
                           style: StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
            }

            let label = Text("\(f.name) \(f.expression)")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(color)
            ctx.draw(ctx.resolve(label), at: CGPoint(x: vp.x, y: vp.y - ringR - 5), anchor: .bottom)
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

    private func swiftColor(_ c: SIMD4<Float>) -> Color {
        Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
    }

    private func elementColor(_ n: Int) -> Color {
        switch n {
        case 6:  return Color(red: 0.35, green: 0.35, blue: 0.35)
        case 7:  return Color(red: 0.2, green: 0.3, blue: 0.85)
        case 8:  return Color(red: 0.85, green: 0.15, blue: 0.15)
        case 16: return Color(red: 0.8, green: 0.7, blue: 0.1)
        case 9:  return Color(red: 0.1, green: 0.7, blue: 0.3)
        case 17: return Color(red: 0.1, green: 0.6, blue: 0.2)
        case 35: return Color(red: 0.6, green: 0.1, blue: 0.1)
        case 15: return Color(red: 0.7, green: 0.4, blue: 0.0)
        default: return Color(red: 0.5, green: 0.5, blue: 0.5)
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
