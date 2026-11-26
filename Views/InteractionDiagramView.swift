// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import UniformTypeIdentifiers

/// 2D Protein–Ligand Interaction Diagram with a LigPlot/PoseView-style
/// layout: the ligand's RDKit depiction in the middle, interacting residues
/// placed outward from their contacts with overlap and crossing resolution.
/// Supports zoom/pan, hover (tooltip + emphasis), click-to-select in 3D, and
/// PNG/PDF/SVG export. Layout and drawing live in `InteractionDiagramLayout.swift`
/// and `InteractionDiagramRenderer.swift`; this view is the SwiftUI shell.
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
    /// Called when a residue is clicked (selects it in the 3D view).
    var onSelectResidue: ((ResidueKey) -> Void)?

    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var scene: DiagramScene?
    @State private var isLoading = true
    @State private var showLegend = true

    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var lastPanOffset: CGSize = .zero
    @State private var lastZoomScale: CGFloat = 1.0

    @State private var hoveredResidue: Int?
    @State private var hoverPoint: CGPoint = .zero

    private enum ExportFormat { case png, pdf, svg }

    var body: some View {
        VStack(spacing: 0) {
            if isLoading {
                ProgressView("Generating 2D layout…")
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let scene {
                diagramCanvas(scene)
                if showLegend {
                    legend(scene)
                }
            } else {
                contactListFallback
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
            let contacts = interactions
            let protein = proteinAtoms
            scene = await Task.detached { () -> DiagramScene? in
                guard let smiles = recorded.flatMap({ $0.isEmpty ? nil : $0 })
                        ?? RDKitBridge.atomsBondsToSMILES(atoms: atoms, bonds: bonds),
                      let depiction = RDKitBridge.compute2DCoords(smiles: smiles) else { return nil }
                // Fall back to SMILES order only if the graphs cannot be matched.
                let aligned = alignDepiction(depiction, toHeavyAtoms: atoms, bonds: bonds)
                return buildDiagramScene(coords: aligned ?? depiction, aligned: aligned != nil,
                                         interactions: contacts, ligandAtoms: atoms, proteinAtoms: protein)
            }.value
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
                Button("Export as SVG…") { export(.svg) }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(scene == nil)
            .help("Export the diagram with its legend (white background)")
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

    /// Diagram space → view space: fit to the canvas, then the user's zoom/pan.
    private func viewTransform(_ scene: DiagramScene, size: CGSize) -> CGAffineTransform {
        let c = CGPoint(x: size.width / 2, y: size.height / 2)
        return diagramFitTransform(bounds: scene.bounds, in: size)
            .concatenating(CGAffineTransform(translationX: c.x + panOffset.width, y: c.y + panOffset.height)
                .scaledBy(x: zoomScale, y: zoomScale)
                .translatedBy(x: -c.x, y: -c.y))
    }

    private func diagramCanvas(_ scene: DiagramScene) -> some View {
        let palette = DiagramPalette.resolved(dark: colorScheme == .dark)
        return GeometryReader { geo in
            let transform = viewTransform(scene, size: geo.size)
            Canvas { context, _ in
                let surface = CanvasDiagramSurface(context)
                surface.concatenate(transform)
                drawScene(scene, on: surface, palette: palette, emphasis: hoveredResidue)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(palette.paper)
            .contentShape(Rectangle())
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
            .onTapGesture {
                if let index = hoveredResidue { onSelectResidue?(scene.residues[index].key) }
            }
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    hoverPoint = location
                    let p = location.applying(transform.inverted())
                    hoveredResidue = scene.residues.lastIndex { $0.contains(p) }
                case .ended:
                    hoveredResidue = nil
                }
            }
            .overlay(alignment: .topLeading) {
                if let index = hoveredResidue {
                    residueTooltip(scene.residues[index])
                        .fixedSize()
                        .offset(x: min(hoverPoint.x + 16, max(geo.size.width - 240, 0)),
                                y: max(hoverPoint.y - 12, 0))
                        .allowsHitTesting(false)
                }
            }
        }
    }

    private func residueTooltip(_ residue: DiagramScene.Residue) -> some View {
        let contacts = residue.interactions.sorted {
            ($0.type.rawValue, $0.distance) < ($1.type.rawValue, $1.distance)
        }
        return VStack(alignment: .leading, spacing: DS.Space.xxs) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.xs) {
                Text(verbatim: "\(residue.key.name) \(residue.key.seq)").font(DS.Font.sectionTitle)
                if !residue.key.chain.isEmpty {
                    Text(verbatim: "chain \(residue.key.chain)")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                }
            }
            ForEach(contacts.prefix(10), id: \.id) { inter in
                HStack(spacing: DS.Space.xs) {
                    Circle()
                        .fill(inter.type.themeColor)
                        .frame(width: 6, height: 6)
                    Text(inter.type.label).font(DS.Font.secondary)
                    Text(verbatim: proteinAtoms.indices.contains(inter.proteinAtomIndex)
                         ? proteinAtoms[inter.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces) : "")
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.1f Å", inter.distance))
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
            if contacts.count > 10 {
                Text("+\(contacts.count - 10) more")
                    .font(DS.Font.caption)
                    .foregroundStyle(.tertiary)
            }
            if onSelectResidue != nil {
                Text("Click to select in 3D")
                    .font(DS.Font.caption)
                    .foregroundStyle(.tertiary)
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

    /// Legend of what the diagram actually draws (not every detected atom pair).
    private func legend(_ scene: DiagramScene) -> some View {
        let palette = DiagramPalette.resolved(dark: colorScheme == .dark)
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            Divider()
            VStack(alignment: .leading, spacing: DS.Space.s) {
                if !scene.drawnTypes.isEmpty {
                    legendRow("Interactions") {
                        ForEach(scene.drawnTypes, id: \.rawValue) { type in
                            HStack(spacing: DS.Space.xs) {
                                legendSample(.interaction(type), palette: palette)
                                Text(type.label)
                                Text("\(scene.drawnCount(type))").monospacedDigit().foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    legendRow("Residues") {
                        ForEach(scene.residueClasses, id: \.self) { cls in
                            HStack(spacing: DS.Space.xs) {
                                legendSample(.residueClass(cls), palette: palette)
                                Text(cls.label)
                            }
                        }
                    }
                    Spacer(minLength: DS.Space.m)
                    Text("\(scene.residues.count) residues · distances in Å")
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

    private func legendSample(_ item: DiagramLegendItem, palette: DiagramPalette) -> some View {
        Canvas { context, size in
            DiagramFigureLayout.drawSample(item, at: CGPoint(x: 0, y: size.height / 2),
                                           on: CanvasDiagramSurface(context), palette: palette)
        }
        .frame(width: DiagramFigureLayout.sampleSize.width, height: DiagramFigureLayout.sampleSize.height)
        .accessibilityHidden(true)
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

    // MARK: - Export

    /// Exports the figure (diagram at 1:1 plus a legend) with the light palette on
    /// white paper, whatever the app appearance. PDF and SVG are vector.
    private func export(_ format: ExportFormat) {
        guard let s = scene else { return }
        let palette = DiagramPalette.resolved(dark: false)
        let layout = DiagramFigureLayout(scene: s)

        let panel = NSSavePanel()
        let baseName = "interaction_diagram_pose\(poseIndex + 1)"
        switch format {
        case .png:
            panel.allowedContentTypes = [.png]
            panel.nameFieldStringValue = "\(baseName).png"
        case .pdf:
            panel.allowedContentTypes = [.pdf]
            panel.nameFieldStringValue = "\(baseName).pdf"
        case .svg:
            panel.allowedContentTypes = [.svg]
            panel.nameFieldStringValue = "\(baseName).svg"
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }

        if format == .svg {
            let surface = SVGDiagramSurface(size: layout.size)
            drawFigure(s, layout: layout, on: surface, palette: palette)
            try? surface.document.write(to: url, atomically: true, encoding: .utf8)
            return
        }

        let renderer = ImageRenderer(content:
            Canvas { context, _ in
                drawFigure(s, layout: layout, on: CanvasDiagramSurface(context), palette: palette)
            }
            .frame(width: layout.size.width, height: layout.size.height)
            .environment(\.colorScheme, .light)
        )
        if format == .png {
            renderer.scale = 3.0
            if let cgImage = renderer.cgImage,
               let pngData = NSBitmapImageRep(cgImage: cgImage).representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        } else {
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
}
