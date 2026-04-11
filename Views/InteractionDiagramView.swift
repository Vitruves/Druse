import SwiftUI
import UniformTypeIdentifiers

/// 2D Protein–Ligand Interaction Diagram (ProLig2D) with a radial schematic
/// layout. Renders a flat skeletal depiction with the ligand centered and
/// interacting residues arranged radially. Supports zoom/pan, hover tooltips,
/// and PNG export. All layout and drawing live in
/// `InteractionDiagramRenderer.swift`; this view is the SwiftUI shell.
struct InteractionDiagramView: View {
    let interactions: [MolecularInteraction]
    let ligandAtoms: [Atom]
    let ligandBonds: [Bond]
    let proteinAtoms: [Atom]
    let ligandSmiles: String?
    let poseEnergy: Float
    let poseIndex: Int
    var scoringMethod: ScoringMethod = .vina

    @State private var scene: DiagramScene?
    @State private var isLoading = true

    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var lastPanOffset: CGSize = .zero
    @State private var lastZoomScale: CGFloat = 1.0

    @State private var exportSize: CGSize = .zero

    @State private var hoveredResidue: String? = nil
    @State private var hoverPoint: CGPoint = .zero
    @State private var residueScreenCenters: [(label: String, center: CGPoint,
                                               interactions: [MolecularInteraction])] = []

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if isLoading {
                ProgressView("Generating 2D layout...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if scene != nil {
                diagramCanvas
            } else {
                fallbackDiagram
            }
            Divider()
            legend
        }
        .frame(minWidth: 650, minHeight: 550)
        .task {
            if let smiles = ligandSmiles, !smiles.isEmpty,
               let coords = await Task.detached(operation: {
                   RDKitBridge.compute2DCoords(smiles: smiles)
               }).value
            {
                scene = buildDiagramScene(
                    coords: coords,
                    interactions: interactions,
                    ligandAtoms: ligandAtoms,
                    proteinAtoms: proteinAtoms)
            }
            isLoading = false
        }
    }

    // MARK: - Diagram canvas

    @ViewBuilder
    private var diagramCanvas: some View {
        GeometryReader { geo in
            Canvas { context, size in
                guard let s = scene else { return }
                var ctx = context
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                ctx.translateBy(x: center.x + panOffset.width, y: center.y + panOffset.height)
                ctx.scaleBy(x: zoomScale, y: zoomScale)
                ctx.translateBy(x: -center.x, y: -center.y)
                let centers = drawScene(s, in: ctx, size: size)
                DispatchQueue.main.async { self.residueScreenCenters = centers }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .controlBackgroundColor))
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
            .onTapGesture(count: 2) {
                withAnimation(.easeInOut(duration: 0.3)) {
                    zoomScale = 1.0
                    panOffset = .zero
                    lastZoomScale = 1.0
                    lastPanOffset = .zero
                }
            }
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
                    let resIxns = residueInteractions(for: res)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(res).font(.subheadline.weight(.bold))
                        ForEach(resIxns, id: \.id) { inter in
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(interactionColor(inter.type))
                                    .frame(width: 6, height: 6)
                                Text(inter.type.label).font(.footnote)
                                Text(String(format: "%.1f \u{00C5}", inter.distance))
                                    .font(.footnote.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(.ultraThickMaterial))
                    .shadow(radius: 4)
                    .position(x: hoverPoint.x + 60, y: hoverPoint.y - 20)
                    .allowsHitTesting(false)
                }
            }
            .onChange(of: geo.size) { _, newSize in exportSize = newSize }
            .onAppear { exportSize = geo.size }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Label("Interaction Diagram", systemImage: "circle.hexagongrid")
                .font(.body.weight(.semibold))
            Spacer()
            Text("Pose #\(poseIndex + 1)")
                .font(.callout.monospaced().weight(.medium))
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f %@", poseEnergy, scoringMethod.unitLabel))
                .font(.callout.monospaced().weight(.semibold))
                .foregroundStyle(poseEnergy < -6 ? .green : poseEnergy < 0 ? .orange : .red)

            HStack(spacing: 4) {
                Button {
                    withAnimation { zoomScale = max(0.3, zoomScale - 0.2); lastZoomScale = zoomScale }
                } label: { Image(systemName: "minus.magnifyingglass").font(.callout) }
                .buttonStyle(.plain).help("Zoom out")
                Text(String(format: "%.0f%%", zoomScale * 100))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(width: 36)
                Button {
                    withAnimation { zoomScale = min(5.0, zoomScale + 0.2); lastZoomScale = zoomScale }
                } label: { Image(systemName: "plus.magnifyingglass").font(.callout) }
                .buttonStyle(.plain).help("Zoom in")
                Button {
                    withAnimation { zoomScale = 1.0; panOffset = .zero
                                    lastZoomScale = 1.0; lastPanOffset = .zero }
                } label: { Image(systemName: "arrow.counterclockwise").font(.subheadline) }
                .buttonStyle(.plain).help("Reset zoom (or double-click)")
            }
            .padding(.horizontal, 8)

            Button { exportPNG() } label: {
                Image(systemName: "square.and.arrow.down").font(.callout)
            }
            .buttonStyle(.plain).help("Save as PNG")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Legend

    private var legend: some View {
        let presentTypes = Set(interactions.map(\.type))
        let types = MolecularInteraction.InteractionType.allCases.filter { presentTypes.contains($0) }
        return VStack(spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(types, id: \.rawValue) { type in
                        let count = interactions.filter { $0.type == type }.count
                        HStack(spacing: 4) {
                            interactionSymbol(type).frame(width: 24, height: 14)
                            Text("\(type.label) (\(count))")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .fixedSize()
                        }
                    }
                }
            }
            HStack(spacing: 0) {
                HStack(spacing: 12) {
                    residueLegendDot("polar", Color(red: 0.2, green: 0.6, blue: 0.7))
                    residueLegendDot("acidic", Color(red: 0.8, green: 0.2, blue: 0.2))
                    residueLegendDot("basic", Color(red: 0.3, green: 0.3, blue: 0.9))
                    residueLegendDot("greasy", Color(red: 0.4, green: 0.6, blue: 0.3))
                }
                Spacer()
                Text("\(interactions.count) interactions")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func residueLegendDot(_ label: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 4)
                .fill(color.opacity(0.3))
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(color, lineWidth: 1))
                .frame(width: 14, height: 14)
            Text(label).font(.footnote.weight(.medium)).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func interactionSymbol(_ type: MolecularInteraction.InteractionType) -> some View {
        let c = interactionColor(type)
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

    // MARK: - Fallback

    private var fallbackDiagram: some View {
        VStack(spacing: 8) {
            Image(systemName: "atom").font(.largeTitle).foregroundStyle(.secondary)
            Text("Could not generate 2D layout").font(.body).foregroundStyle(.secondary)
            Text("SMILES required for 2D depiction")
                .font(.subheadline).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - PNG export

    private func exportPNG() {
        guard let s = scene else { return }
        let scale: CGFloat = 2.0
        let width = max(exportSize.width, 900) * scale
        let height = max(exportSize.height, 700) * scale

        let renderer = ImageRenderer(content:
            Canvas { context, size in
                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.white))
                _ = drawScene(s, in: context, size: size)
            }
            .frame(width: width / scale, height: height / scale)
        )
        renderer.scale = scale
        guard let image = renderer.nsImage else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "interaction_diagram_pose\(poseIndex + 1).png"
        if panel.runModal() == .OK, let url = panel.url,
           let tiff = image.tiffRepresentation,
           let bitmap = NSBitmapImageRep(data: tiff),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            try? pngData.write(to: url)
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
