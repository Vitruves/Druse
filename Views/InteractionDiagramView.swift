// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import UniformTypeIdentifiers

/// 2D Protein–Ligand Interaction Diagram (ProLig2D) with a radial schematic
/// layout. Renders a flat skeletal depiction with the ligand centered and
/// interacting residues arranged radially. Supports zoom/pan, hover tooltips,
/// and PNG/PDF export. All layout and drawing live in
/// `InteractionDiagramRenderer.swift`; this view is the SwiftUI shell.
struct InteractionDiagramView: View {
    let interactions: [MolecularInteraction]
    let ligandName: String
    /// Heavy atoms in the order interaction `ligandAtomIndex` values refer to.
    let ligandAtoms: [Atom]
    /// Bonds between `ligandAtoms`, indexed into that array.
    let ligandBonds: [Bond]
    let proteinAtoms: [Atom]
    /// Recorded SMILES; nil for SDF/PDB ligands, whose SMILES is derived from the atom graph.
    let ligandSmiles: String?
    let poseEnergy: Float
    let poseIndex: Int
    var scoringMethod: ScoringMethod = .vina

    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var scene: DiagramScene?
    @State private var isLoading = true
    @State private var showLegend = true

    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var lastPanOffset: CGSize = .zero
    @State private var lastZoomScale: CGFloat = 1.0

    @State private var exportSize: CGSize = .zero

    @State private var hoveredResidue: String? = nil
    @State private var hoverPoint: CGPoint = .zero
    @State private var residueScreenCenters: [(label: String, center: CGPoint,
                                               interactions: [MolecularInteraction])] = []

    private enum ExportFormat { case png, pdf }

    var body: some View {
        VStack(spacing: 0) {
            if isLoading {
                ProgressView("Generating 2D layout…")
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if scene != nil {
                diagramCanvas
            } else {
                contactListFallback
            }
            if showLegend {
                legend
            }
        }
        .frame(minWidth: 650, minHeight: 550)
        // The toolbar picker already names the pose; the subtitle adds the ligand and exact score.
        .navigationSubtitle(String(format: "%@ · %@ %.2f %@", ligandName, scoringMethod.shortLabel, poseEnergy, scoringMethod.unitLabel))
        .toolbar { toolbarContent }
        .task {
            let recorded = ligandSmiles
            let atoms = ligandAtoms
            let bonds = ligandBonds
            let coords = await Task.detached { () -> RDKitBridge.Coords2D? in
                guard let smiles = recorded.flatMap({ $0.isEmpty ? nil : $0 })
                        ?? RDKitBridge.atomsBondsToSMILES(atoms: atoms, bonds: bonds),
                      let depiction = RDKitBridge.compute2DCoords(smiles: smiles) else { return nil }
                // Fall back to SMILES order only if the graphs cannot be matched (as before).
                return alignDepiction(depiction, toHeavyAtoms: atoms, bonds: bonds) ?? depiction
            }.value
            if let coords {
                scene = buildDiagramScene(
                    coords: coords,
                    interactions: interactions,
                    ligandAtoms: ligandAtoms,
                    proteinAtoms: proteinAtoms)
            }
            isLoading = false
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem {
            ControlGroup {
                Button { setZoom(max(0.3, zoomScale - 0.2)) } label: {
                    Label("Zoom Out", systemImage: "minus.magnifyingglass")
                }
                .help("Zoom out")
                Button { resetView() } label: {
                    Text(String(format: "%.0f%%", zoomScale * 100))
                        .monospacedDigit()
                        .frame(minWidth: 40)
                }
                .help("Reset zoom and pan (or double-click the diagram)")
                Button { setZoom(min(5.0, zoomScale + 0.2)) } label: {
                    Label("Zoom In", systemImage: "plus.magnifyingglass")
                }
                .help("Zoom in")
            }
            .disabled(scene == nil)
        }
        ToolbarItem {
            Toggle(isOn: $showLegend) {
                Label("Legend", systemImage: "list.bullet.rectangle")
            }
            .help(showLegend ? "Hide legend" : "Show legend")
        }
        ToolbarItem {
            Menu {
                Button("Export as PNG…") { export(.png) }
                Button("Export as PDF…") { export(.pdf) }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(scene == nil)
            .help("Export the diagram (white background)")
        }
    }

    private func setZoom(_ value: CGFloat) {
        withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) {
            zoomScale = value
            lastZoomScale = value
        }
    }

    private func resetView() {
        withAnimation(DS.Motion.respecting(reduceMotion)) {
            zoomScale = 1.0
            panOffset = .zero
            lastZoomScale = 1.0
            lastPanOffset = .zero
        }
    }

    // MARK: - Diagram canvas

    @ViewBuilder
    private var diagramCanvas: some View {
        let palette = DiagramPalette.resolved(dark: colorScheme == .dark)
        GeometryReader { geo in
            Canvas { context, size in
                guard let s = scene else { return }
                var ctx = context
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                ctx.translateBy(x: center.x + panOffset.width, y: center.y + panOffset.height)
                ctx.scaleBy(x: zoomScale, y: zoomScale)
                ctx.translateBy(x: -center.x, y: -center.y)
                let centers = drawScene(s, in: ctx, size: size, palette: palette)
                DispatchQueue.main.async { self.residueScreenCenters = centers }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(palette.paper)
            .gesture(
                MagnifyGesture()
                    .onChanged { value in
                        zoomScale = max(0.3, min(5.0, lastZoomScale * value.magnification))
                    }
                    .onEnded { _ in lastZoomScale = zoomScale }
            )
            .simultaneousGesture(
                DragGesture()
                    .onChanged { value in
                        panOffset = CGSize(
                            width: lastPanOffset.width + value.translation.width,
                            height: lastPanOffset.height + value.translation.height)
                    }
                    .onEnded { _ in lastPanOffset = panOffset }
            )
            .onTapGesture(count: 2) { resetView() }
            .overlay {
                Color.clear
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            hoverPoint = location
                            hoveredResidue = findResidueAtPoint(location, in: geo.size)
                        case .ended:
                            hoveredResidue = nil
                        }
                    }
            }
            .overlay(alignment: .topLeading) {
                if let res = hoveredResidue {
                    residueTooltip(res)
                        .position(x: hoverPoint.x + 70, y: hoverPoint.y - 24)
                        .allowsHitTesting(false)
                }
            }
            .onChange(of: geo.size) { _, newSize in exportSize = newSize }
            .onAppear { exportSize = geo.size }
        }
    }

    private func residueTooltip(_ residue: String) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            Text(residue).font(DS.Font.sectionTitle)
            ForEach(residueInteractions(for: residue), id: \.id) { inter in
                HStack(spacing: DS.Space.xs) {
                    Circle()
                        .fill(inter.type.themeColor)
                        .frame(width: 6, height: 6)
                    Text(inter.type.label).font(DS.Font.secondary)
                    Text(String(format: "%.1f Å", inter.distance))
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(DS.Space.m)
        .dsFloating(cornerRadius: DS.Radius.m)
    }

    // MARK: - Contact list (no 2D layout)

    /// Contacts grouped by residue, closest first: what the diagram would have shown.
    private var residueContacts: [(key: ResidueKey, types: [MolecularInteraction.InteractionType], closest: Float)] {
        var byResidue: [ResidueKey: [MolecularInteraction]] = [:]
        for ixn in interactions where proteinAtoms.indices.contains(ixn.proteinAtomIndex) {
            let atom = proteinAtoms[ixn.proteinAtomIndex]
            byResidue[ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID), default: []].append(ixn)
        }
        return byResidue.map { key, contacts in
            let present = Set(contacts.map(\.type))
            return (key: key,
                    types: MolecularInteraction.InteractionType.allCases.filter { present.contains($0) },
                    closest: contacts.map(\.distance).min() ?? 0)
        }
        .sorted { $0.closest < $1.closest }
    }

    /// Shown when RDKit cannot lay the ligand out in 2D: a compact notice and the
    /// contacts as a residue table, so the window still answers "what touches what".
    private var contactListFallback: some View {
        let rows = residueContacts
        return VStack(alignment: .leading, spacing: DS.Space.l) {
            HStack(alignment: .top, spacing: DS.Space.m) {
                Image(systemName: "exclamationmark.triangle")
                    .font(DS.Font.body)
                    .foregroundStyle(DS.Color.warning)
                VStack(alignment: .leading, spacing: DS.Space.xxs) {
                    Text("No 2D layout for this ligand")
                        .font(DS.Font.sectionTitle)
                    Text("RDKit could not derive a 2D depiction from its atoms. Load the ligand from SMILES or an SDF with bond orders to get the diagram; its contacts are listed below.")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if rows.isEmpty {
                Text("This pose makes no contacts with the protein.")
                    .font(DS.Font.label)
                    .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        HStack(spacing: DS.Space.m) {
                            Text("Residue").frame(width: 110, alignment: .leading)
                            Text("Interactions").frame(maxWidth: .infinity, alignment: .leading)
                            Text("Closest").frame(width: 64, alignment: .trailing)
                        }
                        .font(DS.Font.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, DS.Space.l)
                        .frame(height: DS.Size.rowHeight + DS.Space.xs)
                        Divider()
                        ForEach(Array(rows.enumerated()), id: \.element.key) { position, row in
                            if position > 0 { Divider().padding(.leading, DS.Space.l) }
                            HStack(spacing: DS.Space.m) {
                                Text(verbatim: "\(row.key.name) \(row.key.seq)\(row.key.chain.isEmpty ? "" : " · \(row.key.chain)")")
                                    .font(DS.Font.mono)
                                    .frame(width: 110, alignment: .leading)
                                FlowLayout(spacing: DS.Space.xs, lineSpacing: DS.Space.xs) {
                                    ForEach(row.types, id: \.rawValue) { type in
                                        DSTag(type.label, tint: type.themeColor)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                Text(String(format: "%.1f Å", row.closest))
                                    .font(DS.Font.monoSmall)
                                    .monospacedDigit()
                                    .foregroundStyle(.secondary)
                                    .frame(width: 64, alignment: .trailing)
                            }
                            .padding(.horizontal, DS.Space.l)
                            .padding(.vertical, DS.Space.s)
                            .accessibilityElement(children: .combine)
                        }
                    }
                    .dsCard(padding: 0)
                }
            }
        }
        .padding(DS.Space.xxl)
        .frame(maxWidth: 720, maxHeight: .infinity, alignment: .topLeading)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Legend

    /// Distinct residues in contact — atom-pair counts alone overstate hydrophobic contacts.
    private var contactResidueCount: Int {
        Set(interactions.compactMap { ixn -> ResidueKey? in
            guard proteinAtoms.indices.contains(ixn.proteinAtomIndex) else { return nil }
            let atom = proteinAtoms[ixn.proteinAtomIndex]
            return ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
        }).count
    }

    private var legend: some View {
        let presentTypes = Set(interactions.map(\.type))
        let types = MolecularInteraction.InteractionType.allCases.filter { presentTypes.contains($0) }
        let presentHalos = halosPresent(in: interactions)
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            Divider()
            VStack(alignment: .leading, spacing: DS.Space.s) {
                legendRow("Interactions") {
                    ForEach(types, id: \.rawValue) { type in
                        let count = interactions.filter { $0.type == type }.count
                        HStack(spacing: DS.Space.xs) {
                            interactionSymbol(type).frame(width: 24, height: 12)
                            Text(type.label)
                            Text("\(count)").monospacedDigit().foregroundStyle(.tertiary)
                        }
                    }
                }
                if !presentHalos.isEmpty {
                    legendRow("Ligand atoms") {
                        ForEach(presentHalos, id: \.label) { halo in
                            HStack(spacing: DS.Space.xs) {
                                ZStack {
                                    Circle().fill(halo.color.opacity(0.18))
                                    Circle().strokeBorder(halo.color.opacity(0.7), lineWidth: 1.5)
                                }
                                .frame(width: 12, height: 12)
                                Text(halo.label)
                            }
                        }
                    }
                }
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    legendRow("Residues") {
                        ForEach(DiagramResidueClass.allCases, id: \.self) { cls in
                            HStack(spacing: DS.Space.xs) {
                                RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                                    .fill(cls.themeColor.opacity(0.18))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                                            .strokeBorder(cls.themeColor, lineWidth: 1)
                                    )
                                    .frame(width: 12, height: 12)
                                Text(cls.label)
                            }
                        }
                    }
                    Spacer(minLength: DS.Space.m)
                    Text("\(interactions.count) contacts · \(contactResidueCount) residues")
                        .font(DS.Font.secondary)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, DS.Space.xl)
            .padding(.bottom, DS.Space.m)
        }
        .background(.bar)
    }

    private func legendRow<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.l) {
            Text(title)
                .font(DS.Font.eyebrow)
                .textCase(.uppercase)
                .tracking(0.6)
                .foregroundStyle(.tertiary)
                .frame(width: DS.Size.labelColumn, alignment: .leading)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: DS.Space.l) { content() }
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .fixedSize()
            }
        }
    }

    /// Halo entries surfaced in the legend; mirrors `pharmacophoreHalo` in the renderer.
    private func halosPresent(in interactions: [MolecularInteraction]) -> [(label: String, color: Color)] {
        var entries: [(label: String, color: Color)] = []
        let types = Set(interactions.map(\.type))
        if types.contains(.hbond) || types.contains(.halogen) {
            entries.append((label: "H-bond donor/acceptor", color: DS.Color.interactionHBond))
        }
        if types.contains(.saltBridge) {
            entries.append((label: "salt bridge", color: DS.Color.interactionSaltBridge))
        }
        if types.contains(.metalCoord) {
            entries.append((label: "metal coord", color: DS.Color.interactionMetal))
        }
        if types.contains(.chalcogen) {
            entries.append((label: "chalcogen", color: MolecularInteraction.InteractionType.chalcogen.themeColor))
        }
        return entries
    }

    @ViewBuilder
    private func interactionSymbol(_ type: MolecularInteraction.InteractionType) -> some View {
        let c = type.themeColor
        Canvas { ctx, size in
            let y = size.height / 2
            switch type {
            case .hbond:
                var p = Path()
                p.move(to: CGPoint(x: 0, y: y))
                p.addLine(to: CGPoint(x: size.width, y: y))
                ctx.stroke(p, with: .color(c), style: StrokeStyle(lineWidth: 2, dash: [3, 2]))
            case .hydrophobic:
                var p = Path()
                p.move(to: CGPoint(x: 2, y: size.height))
                p.addQuadCurve(to: CGPoint(x: size.width - 2, y: size.height),
                               control: CGPoint(x: size.width / 2, y: -2))
                ctx.stroke(p, with: .color(c), lineWidth: 2)
            case .saltBridge:
                for dy in [-2.0, 2.0] as [CGFloat] {
                    var p = Path()
                    p.move(to: CGPoint(x: 0, y: y + dy))
                    p.addLine(to: CGPoint(x: size.width, y: y + dy))
                    ctx.stroke(p, with: .color(c), style: StrokeStyle(lineWidth: 1.5, dash: [2, 1.4]))
                }
            default:
                var p = Path()
                p.move(to: CGPoint(x: 0, y: y))
                p.addLine(to: CGPoint(x: size.width, y: y))
                ctx.stroke(p, with: .color(c), lineWidth: 2)
            }
        }
    }

    // MARK: - Export

    /// Exports always use the light palette on white paper, whatever the app appearance.
    private func export(_ format: ExportFormat) {
        guard let s = scene else { return }
        let size = CGSize(width: max(exportSize.width, 900), height: max(exportSize.height, 700))
        let palette = DiagramPalette.resolved(dark: false)
        let renderer = ImageRenderer(content:
            Canvas { context, canvasSize in
                context.fill(Path(CGRect(origin: .zero, size: canvasSize)), with: .color(palette.paper))
                drawScene(s, in: context, size: canvasSize, palette: palette)
            }
            .frame(width: size.width, height: size.height)
            .environment(\.colorScheme, .light)
        )

        let panel = NSSavePanel()
        let baseName = "interaction_diagram_pose\(poseIndex + 1)"
        switch format {
        case .png:
            panel.allowedContentTypes = [.png]
            panel.nameFieldStringValue = "\(baseName).png"
        case .pdf:
            panel.allowedContentTypes = [.pdf]
            panel.nameFieldStringValue = "\(baseName).pdf"
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }

        switch format {
        case .png:
            renderer.scale = 2.0
            if let image = renderer.nsImage,
               let tiff = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiff),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        case .pdf:
            renderer.render { renderSize, draw in
                var box = CGRect(origin: .zero, size: renderSize)
                guard let pdf = CGContext(url as CFURL, mediaBox: &box, nil) else { return }
                pdf.beginPDFPage(nil)
                draw(pdf)
                pdf.endPDFPage()
                pdf.closePDF()
            }
        }
    }

    // MARK: - Hover hit-test

    private func findResidueAtPoint(_ point: CGPoint, in size: CGSize) -> String? {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let px = (point.x - center.x - panOffset.width) / zoomScale + center.x
        let py = (point.y - center.y - panOffset.height) / zoomScale + center.y
        let transformed = CGPoint(x: px, y: py)
        let hitRadius: CGFloat = 42.0
        for entry in residueScreenCenters {
            let dx = transformed.x - entry.center.x
            let dy = transformed.y - entry.center.y
            if dx * dx + dy * dy <= hitRadius * hitRadius { return entry.label }
        }
        return nil
    }

    private func residueInteractions(for residueLabel: String) -> [MolecularInteraction] {
        interactions.filter { ixn in
            let pIdx = ixn.proteinAtomIndex
            guard pIdx < proteinAtoms.count else { return false }
            let atom = proteinAtoms[pIdx]
            return "\(atom.residueName)\(atom.residueSeq)" == residueLabel
        }
    }
}
