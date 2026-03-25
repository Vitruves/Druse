import SwiftUI
import UniformTypeIdentifiers

/// 2D Protein-Ligand Interaction Diagram (ProLig2D) — MOE-style.
/// Renders a flat schematic with the ligand centered and interacting residues
/// arranged radially. Supports zoom/pan for exploration, and export to PNG.
/// Residues colored by chemical property (acidic/basic/polar/greasy).
struct InteractionDiagramView: View {
    let interactions: [MolecularInteraction]
    let ligandAtoms: [Atom]
    let ligandBonds: [Bond]
    let proteinAtoms: [Atom]
    let ligandSmiles: String?
    let poseEnergy: Float
    let poseIndex: Int

    @Environment(\.dismiss) private var dismiss
    @State private var coords2D: RDKitBridge.Coords2D?
    @State private var isLoading = true

    // Zoom & pan state
    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var lastPanOffset: CGSize = .zero
    @State private var lastZoomScale: CGFloat = 1.0

    // Export state
    @State private var exportSize: CGSize = .zero

    // Hover state
    @State private var hoveredResidue: String? = nil
    @State private var hoverPoint: CGPoint = .zero
    @State private var residuePositions: [(name: String, center: CGPoint, interactions: [MolecularInteraction])] = []

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if isLoading {
                ProgressView("Generating 2D layout...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let coords = coords2D {
                diagramCanvas(coords: coords)
            } else {
                fallbackDiagram
            }
            Divider()
            legend
        }
        .frame(minWidth: 1000, minHeight: 750)
        .task {
            if let smiles = ligandSmiles, !smiles.isEmpty {
                let result = await Task.detached {
                    RDKitBridge.compute2DCoords(smiles: smiles)
                }.value
                coords2D = result
            }
            isLoading = false
        }
    }

    // MARK: - Zoomable Diagram Canvas

    @ViewBuilder
    private func diagramCanvas(coords: RDKitBridge.Coords2D) -> some View {
        GeometryReader { geo in
            Canvas { context, size in
                // Apply zoom + pan transform
                var ctx = context
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                ctx.translateBy(x: center.x + panOffset.width, y: center.y + panOffset.height)
                ctx.scaleBy(x: zoomScale, y: zoomScale)
                ctx.translateBy(x: -center.x, y: -center.y)
                drawDiagram(context: ctx, size: size, coords: coords)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .controlBackgroundColor))
            .gesture(
                MagnifyGesture()
                    .onChanged { value in
                        zoomScale = max(0.3, min(5.0, lastZoomScale * value.magnification))
                    }
                    .onEnded { value in
                        lastZoomScale = zoomScale
                    }
            )
            .simultaneousGesture(
                DragGesture()
                    .onChanged { value in
                        panOffset = CGSize(
                            width: lastPanOffset.width + value.translation.width,
                            height: lastPanOffset.height + value.translation.height
                        )
                    }
                    .onEnded { value in
                        lastPanOffset = panOffset
                    }
            )
            .onTapGesture(count: 2) {
                // Double-tap to reset zoom
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
                    let resInteractions = residueInteractions(for: res)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(res).font(.system(size: 11, weight: .bold))
                        ForEach(resInteractions, id: \.id) { inter in
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(interactionColor(inter.type))
                                    .frame(width: 6, height: 6)
                                Text(inter.type.label)
                                    .font(.system(size: 9))
                                Text(String(format: "%.1f \u{00C5}", inter.distance))
                                    .font(.system(size: 9, design: .monospaced))
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
            .onChange(of: geo.size) { _, newSize in
                exportSize = newSize
            }
            .onAppear {
                exportSize = geo.size
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Label("Interaction Diagram", systemImage: "circle.hexagongrid")
                .font(.system(size: 14, weight: .semibold))
            Spacer()
            Text("Pose #\(poseIndex + 1)")
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f kcal/mol", poseEnergy))
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(poseEnergy < -6 ? .green : poseEnergy < 0 ? .orange : .red)

            // Zoom controls
            HStack(spacing: 4) {
                Button(action: { withAnimation { zoomScale = max(0.3, zoomScale - 0.2); lastZoomScale = zoomScale } }) {
                    Image(systemName: "minus.magnifyingglass").font(.system(size: 12))
                }
                .buttonStyle(.plain)
                Text(String(format: "%.0f%%", zoomScale * 100))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 36)
                Button(action: { withAnimation { zoomScale = min(5.0, zoomScale + 0.2); lastZoomScale = zoomScale } }) {
                    Image(systemName: "plus.magnifyingglass").font(.system(size: 12))
                }
                .buttonStyle(.plain)
                Button(action: {
                    withAnimation { zoomScale = 1.0; panOffset = .zero; lastZoomScale = 1.0; lastPanOffset = .zero }
                }) {
                    Image(systemName: "arrow.counterclockwise").font(.system(size: 11))
                }
                .buttonStyle(.plain)
                .help("Reset zoom (or double-click)")
            }
            .padding(.horizontal, 8)

            Button(action: { exportPNG() }) {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 12))
            }
            .buttonStyle(.plain)
            .help("Save as PNG")

            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Close diagram")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Legend

    private var legend: some View {
        let presentTypes = Set(interactions.map(\.type))
        let types = MolecularInteraction.InteractionType.allCases.filter { presentTypes.contains($0) }

        return VStack(spacing: 6) {
            // Top row: Interaction type legend (wraps if needed)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(types, id: \.rawValue) { type in
                        let count = interactions.filter { $0.type == type }.count
                        HStack(spacing: 4) {
                            interactionSymbol(type)
                                .frame(width: 24, height: 14)
                            Text("\(type.label) (\(count))")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                                .fixedSize()
                        }
                    }
                }
            }

            // Bottom row: Residue types + total count
            HStack(spacing: 0) {
                HStack(spacing: 10) {
                    residueLegendDot("polar", Color(red: 0.2, green: 0.6, blue: 0.7))
                    residueLegendDot("acidic", Color(red: 0.8, green: 0.2, blue: 0.2))
                    residueLegendDot("basic", Color(red: 0.3, green: 0.3, blue: 0.9))
                    residueLegendDot("greasy", Color(red: 0.4, green: 0.6, blue: 0.3))
                }

                Spacer()

                Text("\(interactions.count) interactions")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func residueLegendDot(_ label: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 3)
                .fill(color.opacity(0.3))
                .overlay(RoundedRectangle(cornerRadius: 3).stroke(color, lineWidth: 1))
                .frame(width: 14, height: 14)
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func interactionSymbol(_ type: MolecularInteraction.InteractionType) -> some View {
        let c = interactionColor(type)
        Canvas { ctx, size in
            let y = size.height / 2
            switch type {
            case .hbond:
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y), to: CGPoint(x: size.width, y: y),
                               color: c, dashLen: 3, lineWidth: 2)
            case .hydrophobic:
                var path = Path()
                path.move(to: CGPoint(x: 2, y: size.height))
                path.addQuadCurve(to: CGPoint(x: size.width - 2, y: size.height),
                                  control: CGPoint(x: size.width / 2, y: -2))
                ctx.stroke(path, with: .color(c), lineWidth: 2)
            case .saltBridge:
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y - 2), to: CGPoint(x: size.width, y: y - 2),
                               color: c, dashLen: 2, lineWidth: 1.5)
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y + 2), to: CGPoint(x: size.width, y: y + 2),
                               color: c, dashLen: 2, lineWidth: 1.5)
            default:
                var path = Path()
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                ctx.stroke(path, with: .color(c), lineWidth: 2)
            }
        }
    }

    // MARK: - Fallback

    private var fallbackDiagram: some View {
        VStack(spacing: 8) {
            Image(systemName: "atom")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            Text("Could not generate 2D layout")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            Text("SMILES required for 2D depiction")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Main Drawing (scaled up ~50% for legibility)

    private func drawDiagram(context: GraphicsContext, size: CGSize, coords: RDKitBridge.Coords2D) {
        let padding: CGFloat = 160  // generous space for residue bubbles

        let positions = coords.positions
        guard !positions.isEmpty else { return }

        let xs = positions.map(\.x)
        let ys = positions.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!
        let minY = ys.min()!, maxY = ys.max()!
        let rangeX = max(maxX - minX, 1)
        let rangeY = max(maxY - minY, 1)

        let ligandAreaW = size.width - padding * 2
        let ligandAreaH = size.height - padding * 2
        let scale = min(ligandAreaW / rangeX, ligandAreaH / rangeY) * 0.75
        let cx = size.width / 2
        let cy = size.height / 2
        let midX = (minX + maxX) / 2
        let midY = (minY + maxY) / 2

        func project(_ p: CGPoint) -> CGPoint {
            CGPoint(x: cx + (p.x - midX) * scale,
                    y: cy - (p.y - midY) * scale)
        }

        // 1. Draw ligand bonds (thicker)
        for (a1, a2, order) in coords.bonds {
            guard a1 < positions.count, a2 < positions.count else { continue }
            let p1 = project(positions[a1])
            let p2 = project(positions[a2])
            drawBond(context: context, from: p1, to: p2, order: order)
        }

        // 2. Draw ligand atoms (larger)
        for (i, pos) in positions.enumerated() {
            let p = project(pos)
            let atomicNum = coords.atomicNums[i]
            drawLigandAtom(context: context, at: p, atomicNum: atomicNum, index: i)
        }

        // 3. Group interactions by residue
        let residueGroups = groupInteractionsByResidue()
        guard !residueGroups.isEmpty else { return }

        let ligandCenter = CGPoint(x: cx, y: cy)
        var residuePlacements: [(key: ResidueKey, center: CGPoint, interactions: [MolecularInteraction])] = []

        for (key, ixns) in residueGroups {
            var sumX: CGFloat = 0, sumY: CGFloat = 0
            var count = 0
            for ixn in ixns {
                let ligIdx = ixn.ligandAtomIndex
                if ligIdx < positions.count {
                    let p = project(positions[ligIdx])
                    sumX += p.x
                    sumY += p.y
                    count += 1
                }
            }
            guard count > 0 else { continue }
            let avgLigandPos = CGPoint(x: sumX / CGFloat(count), y: sumY / CGFloat(count))

            let dx = avgLigandPos.x - ligandCenter.x
            let dy = avgLigandPos.y - ligandCenter.y
            let dist = max(sqrt(dx * dx + dy * dy), 1)
            let pushDist: CGFloat = min(size.width, size.height) * 0.44
            let residueCenter = CGPoint(
                x: ligandCenter.x + dx / dist * pushDist,
                y: ligandCenter.y + dy / dist * pushDist
            )
            residuePlacements.append((key, residueCenter, ixns))
        }

        // Resolve overlaps with tighter angular constraints
        resolveOverlaps(&residuePlacements, around: ligandCenter,
                        radius: min(size.width, size.height) * 0.44)

        // 4. Store positions for hover detection (in pre-transform coordinates)
        DispatchQueue.main.async {
            self.residuePositions = residuePlacements.map { (key, center, ixns) in
                (name: key.label, center: center, interactions: ixns)
            }
        }

        // 5. Draw interaction lines and residue bubbles
        for (key, residueCenter, ixns) in residuePlacements {
            // Separate hydrophobic from directional interactions
            let hydrophobic = ixns.filter { $0.type == .hydrophobic }
            let directional = ixns.filter { $0.type != .hydrophobic }

            // Draw hydrophobic contacts as a single proximity arc between the ligand
            // region and the residue bubble
            if !hydrophobic.isEmpty {
                // Compute average ligand contact position for this hydrophobic residue
                var sumX: CGFloat = 0, sumY: CGFloat = 0, cnt: CGFloat = 0
                for ixn in hydrophobic {
                    let ligIdx = ixn.ligandAtomIndex
                    guard ligIdx < positions.count else { continue }
                    let p = project(positions[ligIdx])
                    sumX += p.x; sumY += p.y; cnt += 1
                }
                if cnt > 0 {
                    let avgLig = CGPoint(x: sumX / cnt, y: sumY / cnt)
                    let dx = residueCenter.x - avgLig.x
                    let dy = residueCenter.y - avgLig.y
                    let len = sqrt(dx * dx + dy * dy)
                    let nx = -dy / max(len, 1) * 18
                    let ny = dx / max(len, 1) * 18

                    // Draw proximity arc (curved line from ligand zone to residue)
                    let fromPt = CGPoint(x: avgLig.x, y: avgLig.y)
                    let toPt = shorten(from: fromPt, to: residueCenter, fromInset: 8, toInset: 45)
                    let mid = CGPoint(x: (toPt.0.x + toPt.1.x) / 2 + nx,
                                      y: (toPt.0.y + toPt.1.y) / 2 + ny)
                    var arcPath = Path()
                    arcPath.move(to: toPt.0)
                    arcPath.addQuadCurve(to: toPt.1, control: mid)
                    let arcColor = interactionColor(.hydrophobic)
                    context.stroke(arcPath, with: .color(arcColor),
                                   style: StrokeStyle(lineWidth: 1.5, lineCap: .round))

                    // Small "hydrophobic zone" arc near the ligand contact point
                    let arcR: CGFloat = 10
                    let contactAngle = atan2(residueCenter.y - avgLig.y, residueCenter.x - avgLig.x)
                    var zonePath = Path()
                    zonePath.addArc(center: avgLig, radius: arcR,
                                    startAngle: .radians(contactAngle - 0.6),
                                    endAngle: .radians(contactAngle + 0.6),
                                    clockwise: false)
                    context.stroke(zonePath, with: .color(arcColor.opacity(0.5)),
                                   style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                }
            }

            // Draw directional interactions as individual lines
            for ixn in directional {
                let ligIdx = ixn.ligandAtomIndex
                guard ligIdx < positions.count else { continue }
                let ligPos = project(positions[ligIdx])
                let toResidue = shorten(from: ligPos, to: residueCenter, fromInset: 8, toInset: 45)
                drawInteractionLine(context: context, from: toResidue.0, to: toResidue.1,
                                    type: ixn.type, distance: ixn.distance)
            }

            drawResidueBubble(context: context, at: residueCenter, key: key, interactions: ixns)
        }
    }

    // MARK: - PNG Export

    private func exportPNG() {
        let scale: CGFloat = 2.0
        let width = max(exportSize.width, 900) * scale
        let height = max(exportSize.height, 700) * scale

        guard let coords = coords2D else { return }

        let renderer = ImageRenderer(content:
            Canvas { context, size in
                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.white))
                self.drawDiagram(context: context, size: size, coords: coords)
            }
            .frame(width: width / scale, height: height / scale)
        )
        renderer.scale = scale

        guard let image = renderer.nsImage else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "interaction_diagram_pose\(poseIndex + 1).png"
        if panel.runModal() == .OK, let url = panel.url {
            if let tiff = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiff),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        }
    }

    // MARK: - Hover Detection

    private func findResidueAtPoint(_ point: CGPoint, in size: CGSize) -> String? {
        // Transform the point from view space back to diagram space
        // accounting for zoom and pan
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        // Reverse the transform: translate, scale, translate
        let px = (point.x - center.x - panOffset.width) / zoomScale + center.x
        let py = (point.y - center.y - panOffset.height) / zoomScale + center.y
        let transformed = CGPoint(x: px, y: py)

        let hitRadius: CGFloat = 42.0  // bubble is ~72 wide x 40 tall, generous hit area
        for entry in residuePositions {
            let dx = transformed.x - entry.center.x
            let dy = transformed.y - entry.center.y
            if dx * dx + dy * dy <= hitRadius * hitRadius {
                return entry.name
            }
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

    // MARK: - Residue Grouping

    private struct ResidueKey: Hashable {
        let name: String
        let seq: Int
        let chain: String
        var label: String { "\(name)\(seq)" }
    }

    private func groupInteractionsByResidue() -> [ResidueKey: [MolecularInteraction]] {
        var groups: [ResidueKey: [MolecularInteraction]] = [:]
        for ixn in interactions {
            let pIdx = ixn.proteinAtomIndex
            guard pIdx < proteinAtoms.count else { continue }
            let atom = proteinAtoms[pIdx]
            let key = ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
            groups[key, default: []].append(ixn)
        }
        return groups
    }

    // MARK: - Drawing Primitives (scaled up for legibility)

    private func drawBond(context: GraphicsContext, from p1: CGPoint, to p2: CGPoint, order: Int) {
        let bondColor = Color.primary.opacity(0.7)
        let lineW: CGFloat = 2.0  // thicker than before

        if order == 1 || order == 4 {
            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(bondColor), lineWidth: order == 4 ? lineW + 0.5 : lineW)

            if order == 4 {
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = sqrt(dx * dx + dy * dy)
                guard len > 0 else { return }
                let nx = -dy / len * 3.5, ny = dx / len * 3.5
                drawDashedLine(ctx: context,
                               from: CGPoint(x: p1.x + nx, y: p1.y + ny),
                               to: CGPoint(x: p2.x + nx, y: p2.y + ny),
                               color: bondColor.opacity(0.5), dashLen: 3, lineWidth: 1.0)
            }
        } else if order == 2 {
            let dx = p2.x - p1.x, dy = p2.y - p1.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2, ny = dx / len * 2
            for sign in [-1.0, 1.0] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * sign, y: p1.y + ny * sign))
                path.addLine(to: CGPoint(x: p2.x + nx * sign, y: p2.y + ny * sign))
                context.stroke(path, with: .color(bondColor), lineWidth: lineW - 0.3)
            }
        } else if order == 3 {
            let dx = p2.x - p1.x, dy = p2.y - p1.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2.5, ny = dx / len * 2.5
            for offset in [-1.0, 0.0, 1.0] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * offset, y: p1.y + ny * offset))
                path.addLine(to: CGPoint(x: p2.x + nx * offset, y: p2.y + ny * offset))
                context.stroke(path, with: .color(bondColor), lineWidth: lineW - 0.5)
            }
        }
    }

    private func drawLigandAtom(context: GraphicsContext, at point: CGPoint, atomicNum: Int, index: Int) {
        let pharmaColor = pharmacophoreColor(forLigandAtomIndex: index)

        if atomicNum == 6 {
            // Carbon: slightly larger dot with optional halo
            let r: CGFloat = 3.5
            if let haloColor = pharmaColor {
                let hr: CGFloat = 10
                context.fill(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                             with: .color(haloColor.opacity(0.2)))
                context.stroke(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                               with: .color(haloColor.opacity(0.6)), lineWidth: 1.5)
            }
            context.fill(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                         with: .color(.primary.opacity(0.6)))
            return
        }

        let (symbol, color) = atomDisplay(atomicNum)
        let r: CGFloat = 13  // larger circle for heteroatoms

        if let haloColor = pharmaColor {
            let hr: CGFloat = 18
            context.fill(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                         with: .color(haloColor.opacity(0.15)))
            context.stroke(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                           with: .color(haloColor.opacity(0.5)), lineWidth: 2)
        }

        context.fill(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                     with: .color(Color(nsColor: .controlBackgroundColor)))
        context.stroke(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                       with: .color(color), lineWidth: 2)
        let text = Text(symbol).font(.system(size: 13, weight: .bold, design: .monospaced)).foregroundColor(color)
        context.draw(context.resolve(text), at: point, anchor: .center)
    }

    private func pharmacophoreColor(forLigandAtomIndex index: Int) -> Color? {
        let types = interactions.filter { $0.ligandAtomIndex == index }.map(\.type)
        guard !types.isEmpty else { return nil }

        if types.contains(.hbond) || types.contains(.halogen) {
            return Color(red: 0.2, green: 0.6, blue: 1.0)
        }
        if types.contains(.saltBridge) {
            return Color(red: 1.0, green: 0.5, blue: 0.1)
        }
        if types.contains(.piStack) || types.contains(.piCation) || types.contains(.amideStack) {
            return Color(red: 0.6, green: 0.3, blue: 0.9)
        }
        if types.contains(.metalCoord) {
            return Color(red: 1.0, green: 0.85, blue: 0.0)
        }
        if types.contains(.chalcogen) {
            return Color(red: 0.5, green: 0.8, blue: 0.2)
        }
        if types.contains(.hydrophobic) || types.contains(.chPi) {
            return Color(red: 0.9, green: 0.8, blue: 0.2)
        }
        return Color.gray.opacity(0.5)
    }

    private func drawResidueBubble(context: GraphicsContext, at center: CGPoint,
                                    key: ResidueKey, interactions: [MolecularInteraction]) {
        // MOE-style: residue colored by chemical property
        let dominantType = dominantInteractionType(interactions)
        let (fillColor, borderColor) = residuePropertyColors(key.name)

        // Find the interacting atom name(s) on the protein side
        let atomNames = Set(interactions.compactMap { ixn -> String? in
            guard ixn.proteinAtomIndex < proteinAtoms.count else { return nil }
            return proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
        })
        let atomLabel = atomNames.prefix(2).joined(separator: ",")

        let labelStr = key.label
        let bubbleW: CGFloat = max(80, CGFloat(labelStr.count) * 9 + 28)
        let bubbleH: CGFloat = atomLabel.isEmpty ? 40 : 50

        let rect = CGRect(x: center.x - bubbleW / 2, y: center.y - bubbleH / 2,
                          width: bubbleW, height: bubbleH)

        // Drop shadow
        let shadowRect = rect.offsetBy(dx: 2, dy: 2)
        context.fill(Path(roundedRect: shadowRect, cornerRadius: 8), with: .color(.black.opacity(0.15)))

        // Main bubble
        context.fill(Path(roundedRect: rect, cornerRadius: 8), with: .color(fillColor))
        context.stroke(Path(roundedRect: rect, cornerRadius: 8), with: .color(borderColor), lineWidth: 2)

        // Residue name
        let yOffset: CGFloat = atomLabel.isEmpty ? -4 : -9
        let resLabel = Text(key.label)
            .font(.system(size: 13, weight: .bold))
            .foregroundColor(.primary)
        context.draw(context.resolve(resLabel), at: CGPoint(x: center.x, y: center.y + yOffset), anchor: .center)

        // Interaction type + interacting atom name
        let typeStr = atomLabel.isEmpty ? dominantType.label : "\(dominantType.label) (\(atomLabel))"
        let typeLabel = Text(typeStr)
            .font(.system(size: 8, weight: .medium))
            .foregroundColor(borderColor)
        context.draw(context.resolve(typeLabel), at: CGPoint(x: center.x, y: center.y + yOffset + 16), anchor: .center)
    }

    /// MOE-style: color by amino acid chemical property.
    private func residuePropertyColors(_ residueName: String) -> (fill: Color, border: Color) {
        let acidic = Set(["ASP", "GLU"])
        let basic = Set(["ARG", "LYS", "HIS"])
        let polar = Set(["SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP"])

        if acidic.contains(residueName) {
            return (Color(red: 0.95, green: 0.85, blue: 0.85), Color(red: 0.8, green: 0.2, blue: 0.2))
        }
        if basic.contains(residueName) {
            return (Color(red: 0.85, green: 0.85, blue: 0.95), Color(red: 0.3, green: 0.3, blue: 0.9))
        }
        if polar.contains(residueName) {
            return (Color(red: 0.85, green: 0.92, blue: 0.95), Color(red: 0.2, green: 0.6, blue: 0.7))
        }
        return (Color(red: 0.88, green: 0.92, blue: 0.85), Color(red: 0.4, green: 0.6, blue: 0.3))
    }

    private func dominantInteractionType(_ interactions: [MolecularInteraction]) -> MolecularInteraction.InteractionType {
        let types = Set(interactions.map(\.type))
        if types.contains(.hbond) { return .hbond }
        if types.contains(.saltBridge) { return .saltBridge }
        if types.contains(.metalCoord) { return .metalCoord }
        if types.contains(.piStack) { return .piStack }
        if types.contains(.piCation) { return .piCation }
        if types.contains(.amideStack) { return .amideStack }
        if types.contains(.halogen) { return .halogen }
        if types.contains(.chalcogen) { return .chalcogen }
        if types.contains(.chPi) { return .chPi }
        return .hydrophobic
    }

    private func drawInteractionLine(context: GraphicsContext, from: CGPoint, to: CGPoint,
                                      type: MolecularInteraction.InteractionType, distance: Float) {
        let color = interactionColor(type)
        let lineW: CGFloat = 2.0

        switch type {
        case .hbond:
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 6, lineWidth: lineW)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 7)
        case .hydrophobic, .chPi:
            let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
            let dx = to.x - from.x, dy = to.y - from.y
            let len = sqrt(dx * dx + dy * dy)
            let nx = -dy / max(len, 1) * 15
            let ny = dx / max(len, 1) * 15
            let ctrl = CGPoint(x: mid.x + nx, y: mid.y + ny)
            var path = Path()
            path.move(to: from)
            path.addQuadCurve(to: to, control: ctrl)
            context.stroke(path, with: .color(color), lineWidth: 1.5)
        case .saltBridge:
            let dx = to.x - from.x, dy = to.y - from.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2.5, ny = dx / len * 2.5
            for sign in [-1.0, 1.0] {
                let f = CGPoint(x: from.x + nx * sign, y: from.y + ny * sign)
                let t = CGPoint(x: to.x + nx * sign, y: to.y + ny * sign)
                drawDashedLine(ctx: context, from: f, to: t, color: color, dashLen: 5, lineWidth: 1.5)
            }
        case .piStack:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: lineW)
        case .piCation:
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 5, lineWidth: lineW)
        case .halogen:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: lineW)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 6)
        case .metalCoord:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: 2.5)
        case .amideStack:
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 6, lineWidth: lineW)
        case .chalcogen:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: lineW)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 6)
        }

        // Distance label only for directional interactions (not hydrophobic)
        if type != .hydrophobic && type != .chPi {
            let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
            let distText = Text(String(format: "%.1f", distance))
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundColor(.secondary)
            context.draw(context.resolve(distText), at: CGPoint(x: mid.x, y: mid.y - 8), anchor: .center)
        }
    }

    // MARK: - Drawing Utilities

    private func drawDashedLine(ctx: GraphicsContext, from: CGPoint, to: CGPoint,
                                 color: Color, dashLen: CGFloat, lineWidth: CGFloat) {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        ctx.stroke(path, with: .color(color),
                   style: StrokeStyle(lineWidth: lineWidth, dash: [dashLen, dashLen * 0.7]))
    }

    private func drawArrowhead(context: GraphicsContext, at tip: CGPoint, from: CGPoint,
                                color: Color, size: CGFloat) {
        let dx = tip.x - from.x
        let dy = tip.y - from.y
        let len = sqrt(dx * dx + dy * dy)
        guard len > 0 else { return }
        let ux = dx / len, uy = dy / len
        let px = -uy, py = ux

        var arrow = Path()
        arrow.move(to: tip)
        arrow.addLine(to: CGPoint(x: tip.x - ux * size + px * size * 0.5,
                                  y: tip.y - uy * size + py * size * 0.5))
        arrow.addLine(to: CGPoint(x: tip.x - ux * size - px * size * 0.5,
                                  y: tip.y - uy * size - py * size * 0.5))
        arrow.closeSubpath()
        context.fill(arrow, with: .color(color))
    }

    private func interactionColor(_ type: MolecularInteraction.InteractionType) -> Color {
        let c = type.color
        return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
    }

    private func atomDisplay(_ atomicNum: Int) -> (String, Color) {
        switch atomicNum {
        case 7:  return ("N", .blue)
        case 8:  return ("O", .red)
        case 9:  return ("F", .green)
        case 15: return ("P", .orange)
        case 16: return ("S", .yellow)
        case 17: return ("Cl", .green)
        case 35: return ("Br", Color(red: 0.6, green: 0.2, blue: 0.2))
        case 53: return ("I", .purple)
        default:
            let elem = Element(rawValue: atomicNum)
            return (elem?.symbol ?? "?", .gray)
        }
    }

    private func shorten(from: CGPoint, to: CGPoint,
                         fromInset: CGFloat, toInset: CGFloat) -> (CGPoint, CGPoint) {
        let dx = to.x - from.x
        let dy = to.y - from.y
        let len = sqrt(dx * dx + dy * dy)
        guard len > fromInset + toInset else { return (from, to) }
        let ux = dx / len, uy = dy / len
        return (CGPoint(x: from.x + ux * fromInset, y: from.y + uy * fromInset),
                CGPoint(x: to.x - ux * toInset, y: to.y - uy * toInset))
    }

    /// Resolve overlapping residue bubbles by angular redistribution.
    private func resolveOverlaps(
        _ placements: inout [(key: ResidueKey, center: CGPoint, interactions: [MolecularInteraction])],
        around origin: CGPoint,
        radius: CGFloat
    ) {
        guard placements.count > 1 else { return }

        struct PolarEntry {
            var angle: CGFloat
            var originalIndex: Int
        }

        var polar: [PolarEntry] = placements.enumerated().map { i, p in
            let dx = p.center.x - origin.x
            let dy = p.center.y - origin.y
            return PolarEntry(angle: atan2(dy, dx), originalIndex: i)
        }
        polar.sort { $0.angle < $1.angle }

        let minSpacing: CGFloat = 0.50  // ~29 degrees between bubbles

        // 5-pass relaxation for better distribution
        for _ in 0..<5 {
            for i in 1..<polar.count {
                let gap = polar[i].angle - polar[i - 1].angle
                if gap < minSpacing {
                    polar[i].angle = polar[i - 1].angle + minSpacing
                }
            }
            if polar.count > 1 {
                let wrapGap = (polar[0].angle + 2 * .pi) - polar[polar.count - 1].angle
                if wrapGap < minSpacing {
                    polar[polar.count - 1].angle = (polar[0].angle + 2 * .pi) - minSpacing
                }
            }
        }

        // Scale radius outward when many residues
        let totalSpan = polar.last!.angle - polar.first!.angle
        let effectiveRadius: CGFloat
        if placements.count > 12 {
            effectiveRadius = radius * 1.25
        } else if totalSpan > 1.8 * .pi {
            effectiveRadius = radius * 1.15
        } else {
            effectiveRadius = radius
        }

        for entry in polar {
            let idx = entry.originalIndex
            placements[idx].center = CGPoint(
                x: origin.x + cos(entry.angle) * effectiveRadius,
                y: origin.y + sin(entry.angle) * effectiveRadius
            )
        }
    }
}
