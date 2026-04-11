import SwiftUI

// MARK: - Residue Key

struct ResidueKey: Hashable {
    let name: String
    let seq: Int
    let chain: String
    var label: String { "\(name)\(seq)" }
}

// MARK: - DiagramScene
//
// All geometry needed to draw the interaction diagram, computed once from
// the input coordinates and interactions. Drawing is a pure function of this
// scene plus a target canvas size (the auto-fit transform is applied at draw
// time so the scene is canvas-size independent).

struct DiagramScene {
    // --- Ligand ---
    let ligandPositions: [CGPoint]                            // 2D, centered at origin
    let ligandBonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)]
    let ligandAtomicNums: [Int]
    let ligandFormalCharges: [Int]
    let ligandRings: [[Int]]                                  // aromatic rings (atom indices)
    let ligandAABB: CGRect
    let ligandCenter: CGPoint                                 // geometric center for outward normals
    let pharmacophoreColors: [Int: Color]                     // ligand atom index → halo color

    // --- Residues ---
    let residues: [ResidueDraw]
    struct ResidueDraw {
        let key: ResidueKey
        let center: CGPoint
        let template: SideChainTemplate?
        let scPositions: [CGPoint]
        let interactingAtomNames: Set<String>
        let propertyFill: Color
        let propertyBorder: Color
        let dominantType: MolecularInteraction.InteractionType
        let interactions: [MolecularInteraction]
        let donorHydrogens: [(pos: CGPoint, atomPos: CGPoint)]   // explicit H atoms for H-bond donors
    }

    // --- Connectors ---
    let directionalLines: [DirectionalLine]
    struct DirectionalLine {
        let from: CGPoint                                     // ligand contact
        let to: CGPoint                                       // residue side-chain atom or ring centroid
        let routedVia: CGPoint?                               // bezier control if routed around ligand
        let type: MolecularInteraction.InteractionType
        let distance: Float
    }
    let hydrophobicMarks: [HydrophobicMark]
    struct HydrophobicMark {
        let anchor: CGPoint                                   // ligand atom (or ring centroid) position
        let outwardNormal: CGPoint                            // unit vector from ligand center → anchor
        let connectors: [CGPoint]                             // residue bubble centers to connect to
    }
}

// MARK: - Constants

private let kBaseScale: CGFloat = 30
private let kPiTypes: Set<MolecularInteraction.InteractionType> = [.piStack, .piCation, .chPi, .amideStack]
private let kDiagramHydroExcluded: Set<String> = ["ASP", "GLU", "ASN", "GLN", "SER", "THR", "CYS"]
private let kDiagramHydroAllowed: [String: Set<String>] = [
    "ARG": ["CB", "CG"],
    "LYS": ["CB", "CG", "CD"],
]

// MARK: - Scene Builder

/// Compute a complete diagram scene from raw input data. Pure function.
func buildDiagramScene(
    coords: RDKitBridge.Coords2D,
    interactions: [MolecularInteraction],
    ligandAtoms: [Atom],
    proteinAtoms: [Atom]
) -> DiagramScene? {
    let positions = coords.positions
    guard !positions.isEmpty else { return nil }

    // --- Project ligand to abstract space centered at (0,0) ---
    let xs = positions.map(\.x), ys = positions.map(\.y)
    let minX = xs.min()!, maxX = xs.max()!
    let minY = ys.min()!, maxY = ys.max()!
    let midX = (minX + maxX) / 2, midY = (minY + maxY) / 2

    let projected = positions.map {
        CGPoint(x: ($0.x - midX) * kBaseScale, y: -($0.y - midY) * kBaseScale)
    }
    let ligandCenter = CGPoint.zero

    let projXs = projected.map(\.x), projYs = projected.map(\.y)
    let ligandAABB = CGRect(
        x: (projXs.min() ?? 0) - 20, y: (projYs.min() ?? 0) - 20,
        width: ((projXs.max() ?? 0) - (projXs.min() ?? 0)) + 40,
        height: ((projYs.max() ?? 0) - (projYs.min() ?? 0)) + 40
    )

    // --- Aromatic rings + ring centroid lookup ---
    let rings = detectAromaticRings(bonds: coords.bonds,
                                     atomCount: projected.count,
                                     isAromatic: coords.isAromatic)
    var ringCentroidForAtom: [Int: CGPoint] = [:]
    for ring in rings {
        let pts = ring.compactMap { $0 < projected.count ? projected[$0] : nil }
        guard pts.count == ring.count, !pts.isEmpty else { continue }
        let cx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
        let cy = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
        for idx in ring { ringCentroidForAtom[idx] = CGPoint(x: cx, y: cy) }
    }

    func ligandContactPoint(for ixn: MolecularInteraction) -> CGPoint? {
        let idx = ixn.ligandAtomIndex
        guard idx < projected.count else { return nil }
        if kPiTypes.contains(ixn.type) {
            if let centroid = ringCentroidForAtom[idx] { return centroid }
            // Fallback: nearest aromatic ring centroid
            let atomPos = projected[idx]
            var bestDist: CGFloat = .greatestFiniteMagnitude
            var bestCentroid: CGPoint?
            for ring in rings {
                let pts = ring.compactMap { $0 < projected.count ? projected[$0] : nil }
                guard pts.count == ring.count, !pts.isEmpty else { continue }
                let cx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
                let cy = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
                let d = hypot(atomPos.x - cx, atomPos.y - cy)
                if d < bestDist { bestDist = d; bestCentroid = CGPoint(x: cx, y: cy) }
            }
            if let centroid = bestCentroid, bestDist < 40 { return centroid }
        }
        return projected[idx]
    }

    // --- Group interactions by residue (with hydrophobic display filter) ---
    var residueGroups: [ResidueKey: [MolecularInteraction]] = [:]
    for ixn in interactions {
        let pIdx = ixn.proteinAtomIndex
        guard pIdx < proteinAtoms.count else { continue }
        let atom = proteinAtoms[pIdx]
        if ixn.type == .hydrophobic {
            let resName = atom.residueName
            let atomName = atom.name.trimmingCharacters(in: .whitespaces)
            if kDiagramHydroExcluded.contains(resName) { continue }
            if let allowed = kDiagramHydroAllowed[resName], !allowed.contains(atomName) { continue }
        }
        let key = ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
        residueGroups[key, default: []].append(ixn)
    }
    guard !residueGroups.isEmpty else { return nil }

    // --- Place residues on an orbit around the ligand ---
    let ligandHalfDiag = hypot(ligandAABB.width, ligandAABB.height) / 2
    let orbitRadius = ligandHalfDiag + 130

    var placements: [(key: ResidueKey, angle: CGFloat,
                      center: CGPoint, ixns: [MolecularInteraction])] = []
    for (key, ixns) in residueGroups {
        // Weight contact points: closer-to-ligand-center contacts get more weight
        var sumX: CGFloat = 0, sumY: CGFloat = 0, wTotal: CGFloat = 0
        for ixn in ixns {
            guard let p = ligandContactPoint(for: ixn) else { continue }
            // Inverse-distance weighting biases toward the strongest direction
            // when contacts span > 90°.
            let w: CGFloat = 1.0 / (1.0 + CGFloat(ixn.distance))
            sumX += p.x * w; sumY += p.y * w; wTotal += w
        }
        guard wTotal > 0 else { continue }
        let avgPos = CGPoint(x: sumX / wTotal, y: sumY / wTotal)
        let angle = atan2(avgPos.y - ligandCenter.y, avgPos.x - ligandCenter.x)
        let center = CGPoint(x: ligandCenter.x + cos(angle) * orbitRadius,
                             y: ligandCenter.y + sin(angle) * orbitRadius)
        placements.append((key, angle, center, ixns))
    }

    // Sort by angle then resolve overlaps via angular relaxation (preserves order)
    placements.sort { $0.angle < $1.angle }
    relaxAnglesAndPlace(&placements, origin: ligandCenter, radius: orbitRadius)

    // --- Build per-residue draw data with side chain transforms ---
    var residues: [DiagramScene.ResidueDraw] = []
    for placement in placements {
        let (key, _, center, ixns) = placement

        // Average ligand contact direction (for side-chain orientation)
        var sX: CGFloat = 0, sY: CGFloat = 0, n: CGFloat = 0
        for ixn in ixns {
            if let p = ligandContactPoint(for: ixn) {
                sX += p.x; sY += p.y; n += 1
            }
        }
        let avgContact = n > 0 ? CGPoint(x: sX / n, y: sY / n) : ligandCenter

        let template = sideChainTemplates[key.name]
        var scPositions: [CGPoint] = []

        if let tmpl = template, !tmpl.atoms.isEmpty {
            // Determine the pivot: prefer ring centroid for π-interactions,
            // else the named interacting atom, else the outermost atom.
            let piIxn = ixns.first { kPiTypes.contains($0.type) }
            let nonHydroIxn = ixns.first { $0.type != .hydrophobic }

            if let pi = piIxn, !tmpl.aromaticRings.isEmpty,
               pi.proteinAtomIndex < proteinAtoms.count {
                // Pick the ring whose atom names match the protein atom name
                let resName = proteinAtoms[pi.proteinAtomIndex].residueName
                let ringNames = sideChainRingAtomNames(for: resName)
                if let chosenRing = pickRing(template: tmpl, ringNames: ringNames) {
                    scPositions = transformSideChainRingPivot(
                        template: tmpl, bubbleCenter: center,
                        ringAtomIndices: chosenRing, towardLigand: avgContact)
                }
            }
            if scPositions.isEmpty, let ix = nonHydroIxn,
               ix.proteinAtomIndex < proteinAtoms.count {
                let aName = proteinAtoms[ix.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
                if tmpl.atoms.contains(where: { $0.name == aName }) {
                    scPositions = transformSideChain(template: tmpl, bubbleCenter: center,
                                                      interactingAtomName: aName,
                                                      towardLigand: avgContact)
                }
            }
            if scPositions.isEmpty {
                // Default: pivot on the outermost atom
                let lastName = tmpl.atoms.last!.name
                scPositions = transformSideChain(template: tmpl, bubbleCenter: center,
                                                  interactingAtomName: lastName,
                                                  towardLigand: avgContact)
            }
        }

        let interactingAtomNames = Set(ixns.compactMap { ixn -> String? in
            guard ixn.proteinAtomIndex < proteinAtoms.count else { return nil }
            return proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
        })
        let (fill, border) = residuePropertyColors(key.name)
        let dominant = dominantInteractionType(ixns)

        // Pre-compute donor hydrogens (for H-bond donors that need an explicit H)
        var donorHs: [(pos: CGPoint, atomPos: CGPoint)] = []
        if let tmpl = template, scPositions.count == tmpl.atoms.count {
            for (i, atom) in tmpl.atoms.enumerated() {
                guard interactingAtomNames.contains(atom.name) else { continue }
                let isDonor: Bool = {
                    switch (key.name, atom.name) {
                    case ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
                         ("ASN", "ND2"), ("GLN", "NE2"), ("LYS", "NZ"),
                         ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
                         ("TRP", "NE1"), ("HIS", "ND1"), ("HIS", "NE2"):
                        return true
                    default: return false
                    }
                }()
                let hbondIxn = ixns.first { ixn in
                    ixn.type == .hbond &&
                    ixn.proteinAtomIndex < proteinAtoms.count &&
                    proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces) == atom.name
                }
                guard isDonor, let hbond = hbondIxn,
                      let lp = ligandContactPoint(for: hbond) else { continue }
                let pos = scPositions[i]
                let dx = lp.x - pos.x, dy = lp.y - pos.y
                let len = max(hypot(dx, dy), 1)
                let hPos = CGPoint(x: pos.x + dx / len * 12, y: pos.y + dy / len * 12)
                donorHs.append((pos: hPos, atomPos: pos))
            }
        }

        residues.append(DiagramScene.ResidueDraw(
            key: key, center: center, template: template,
            scPositions: scPositions,
            interactingAtomNames: interactingAtomNames,
            propertyFill: fill, propertyBorder: border,
            dominantType: dominant, interactions: ixns,
            donorHydrogens: donorHs))
    }

    // --- Build directional lines + hydrophobic marks ---
    var directionalLines: [DiagramScene.DirectionalLine] = []
    var hydroByAnchor: [Int: (anchor: CGPoint, normal: CGPoint, connectors: [CGPoint])] = [:]

    for residue in residues {
        for ixn in residue.interactions {
            guard let ligPos = ligandContactPoint(for: ixn) else { continue }

            if ixn.type == .hydrophobic {
                // Anchor: prefer ring centroid if the ligand atom is in an
                // aromatic ring, else the atom position.
                let anchor = ligPos
                // Outward normal from ligand center
                let dx = anchor.x - ligandCenter.x
                let dy = anchor.y - ligandCenter.y
                let len = max(hypot(dx, dy), 1)
                let normal = CGPoint(x: dx / len, y: dy / len)
                // Use ligand atom index as anchor key (or ring centroid pseudo-key)
                let key = ixn.ligandAtomIndex
                if var entry = hydroByAnchor[key] {
                    entry.connectors.append(residue.center)
                    hydroByAnchor[key] = entry
                } else {
                    hydroByAnchor[key] = (anchor: anchor, normal: normal,
                                          connectors: [residue.center])
                }
                continue
            }

            // Directional: target = side-chain atom or ring centroid
            var targetPos = residue.center
            if let tmpl = residue.template, !residue.scPositions.isEmpty,
               ixn.proteinAtomIndex < proteinAtoms.count {
                if kPiTypes.contains(ixn.type) {
                    // Target the side-chain ring centroid
                    let resName = proteinAtoms[ixn.proteinAtomIndex].residueName
                    let ringNames = sideChainRingAtomNames(for: resName)
                    if !ringNames.isEmpty,
                       let centroid = pickRingCentroid(template: tmpl,
                                                       scPositions: residue.scPositions,
                                                       ringNames: ringNames) {
                        targetPos = centroid
                    }
                } else {
                    let aName = proteinAtoms[ixn.proteinAtomIndex].name
                        .trimmingCharacters(in: .whitespaces)
                    if let idx = tmpl.atoms.firstIndex(where: { $0.name == aName }),
                       idx < residue.scPositions.count {
                        targetPos = residue.scPositions[idx]
                    }
                }
            }

            // Inset endpoints so the line stops short of atoms/labels
            let toInset: CGFloat = (targetPos == residue.center) ? 40 : 8
            let shortened = shortenSegment(from: ligPos, to: targetPos,
                                            fromInset: 8, toInset: toInset)

            // Route around the ligand AABB if the line crosses through it
            var routed: CGPoint? = nil
            if lineCrossesRect(from: shortened.0, to: shortened.1, rect: ligandAABB),
               !ligandAABB.contains(shortened.1) {
                let corners = [
                    CGPoint(x: ligandAABB.minX - 12, y: ligandAABB.minY - 12),
                    CGPoint(x: ligandAABB.maxX + 12, y: ligandAABB.minY - 12),
                    CGPoint(x: ligandAABB.maxX + 12, y: ligandAABB.maxY + 12),
                    CGPoint(x: ligandAABB.minX - 12, y: ligandAABB.maxY + 12),
                ]
                let mid = CGPoint(x: (shortened.0.x + shortened.1.x) / 2,
                                  y: (shortened.0.y + shortened.1.y) / 2)
                routed = corners.min(by: {
                    hypot($0.x - mid.x, $0.y - mid.y) < hypot($1.x - mid.x, $1.y - mid.y)
                })
            }

            directionalLines.append(DiagramScene.DirectionalLine(
                from: shortened.0, to: shortened.1, routedVia: routed,
                type: ixn.type, distance: ixn.distance))
        }
    }

    let hydrophobicMarks = hydroByAnchor.values.map {
        DiagramScene.HydrophobicMark(anchor: $0.anchor,
                                     outwardNormal: $0.normal,
                                     connectors: $0.connectors)
    }

    // --- Pharmacophore halo colors per ligand atom ---
    var pharmaColors: [Int: Color] = [:]
    for i in 0..<projected.count {
        if let c = pharmacophoreColor(forLigandAtomIndex: i,
                                       interactions: interactions,
                                       rings: rings) {
            pharmaColors[i] = c
        }
    }

    // Formal charges
    var charges: [Int] = Array(repeating: 0, count: projected.count)
    for i in 0..<min(projected.count, ligandAtoms.count) {
        charges[i] = ligandAtoms[i].formalCharge
    }

    let typedBonds = coords.bonds.map { (a, b, order) -> (a: Int, b: Int, order: Int, isAromatic: Bool) in
        let aro = a < coords.isAromatic.count && b < coords.isAromatic.count
            && coords.isAromatic[a] && coords.isAromatic[b]
        return (a: a, b: b, order: order, isAromatic: aro)
    }

    return DiagramScene(
        ligandPositions: projected,
        ligandBonds: typedBonds,
        ligandAtomicNums: coords.atomicNums,
        ligandFormalCharges: charges,
        ligandRings: rings,
        ligandAABB: ligandAABB,
        ligandCenter: ligandCenter,
        pharmacophoreColors: pharmaColors,
        residues: residues,
        directionalLines: directionalLines,
        hydrophobicMarks: hydrophobicMarks)
}

// MARK: - Layout helpers

private func relaxAnglesAndPlace(
    _ placements: inout [(key: ResidueKey, angle: CGFloat,
                          center: CGPoint, ixns: [MolecularInteraction])],
    origin: CGPoint,
    radius: CGFloat
) {
    guard placements.count > 1 else { return }
    var angles = placements.map(\.angle)
    let minSpacing: CGFloat = 0.56  // ~32°

    for _ in 0..<12 {
        for i in 1..<angles.count {
            let gap = angles[i] - angles[i - 1]
            if gap < minSpacing { angles[i] = angles[i - 1] + minSpacing }
        }
        for i in stride(from: angles.count - 2, through: 0, by: -1) {
            let gap = angles[i + 1] - angles[i]
            if gap < minSpacing { angles[i] = angles[i + 1] - minSpacing }
        }
        if angles.count > 1 {
            let wrap = (angles[0] + 2 * .pi) - angles[angles.count - 1]
            if wrap < minSpacing {
                let deficit = minSpacing - wrap
                angles[angles.count - 1] -= deficit / 2
                angles[0] += deficit / 2
            }
        }
    }

    let totalSpan = angles.last! - angles.first!
    let effRadius: CGFloat
    if placements.count > 14 {
        effRadius = radius * 1.3
    } else if placements.count > 10 {
        effRadius = radius * 1.2
    } else if totalSpan > 1.8 * .pi {
        effRadius = radius * 1.15
    } else {
        effRadius = radius
    }

    for i in 0..<placements.count {
        placements[i].angle = angles[i]
        placements[i].center = CGPoint(x: origin.x + cos(angles[i]) * effRadius,
                                       y: origin.y + sin(angles[i]) * effRadius)
    }
}

private func sideChainRingAtomNames(for residueName: String) -> Set<String> {
    switch residueName {
    case "PHE": return ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    case "TYR": return ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    case "TRP": return ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
    case "HIS": return ["CG", "ND1", "CD2", "CE1", "NE2"]
    default:    return []
    }
}

private func pickRing(template: SideChainTemplate, ringNames: Set<String>) -> [Int]? {
    template.aromaticRings.first { ring in
        ring.allSatisfy { idx in
            idx < template.atoms.count && ringNames.contains(template.atoms[idx].name)
        }
    } ?? template.aromaticRings.first
}

private func pickRingCentroid(template: SideChainTemplate,
                               scPositions: [CGPoint],
                               ringNames: Set<String>) -> CGPoint? {
    guard let ring = pickRing(template: template, ringNames: ringNames) else { return nil }
    var rx: CGFloat = 0, ry: CGFloat = 0, rn: CGFloat = 0
    for idx in ring where idx < scPositions.count {
        rx += scPositions[idx].x; ry += scPositions[idx].y; rn += 1
    }
    guard rn > 0 else { return nil }
    return CGPoint(x: rx / rn, y: ry / rn)
}

// MARK: - Skeletal Renderer
//
// Single shared function used for both ligand and side chains. Carbons are
// bare bond vertices (no marker). Heteroatoms render as element labels with
// a canvas-color background pill so the bond appears to terminate at the
// letter. Aromatic rings get an inscribed circle. Double bonds render as
// parallel lines (inner line shortened, offset toward ring centroid for ring
// bonds).

struct SkeletalStyle {
    let bondWidth: CGFloat
    let bondColor: Color
    let heteroatomFontSize: CGFloat
    let backgroundColor: Color
    let aromaticCircleScale: CGFloat        // 0.55 of average vertex radius
    let heteroatomLabelInset: CGFloat       // bond inset around heteroatom labels
}

extension SkeletalStyle {
    static let ligand = SkeletalStyle(
        bondWidth: 2.2, bondColor: Color.primary.opacity(0.8),
        heteroatomFontSize: 13,
        backgroundColor: Color(nsColor: .controlBackgroundColor),
        aromaticCircleScale: 0.55, heteroatomLabelInset: 9)

    static let sideChain = SkeletalStyle(
        bondWidth: 1.9, bondColor: Color.primary.opacity(0.7),
        heteroatomFontSize: 11,
        backgroundColor: Color(nsColor: .controlBackgroundColor),
        aromaticCircleScale: 0.55, heteroatomLabelInset: 7.5)
}

func drawSkeletal(
    context: GraphicsContext,
    positions: [CGPoint],
    bonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)],
    atomicNums: [Int],
    rings: [[Int]],
    formalCharges: [Int],
    style: SkeletalStyle,
    halos: [Int: Color] = [:]
) {
    // Build a per-atom set of ring memberships for orientation of double-bond inner lines
    var ringCentroids: [(ring: Set<Int>, centroid: CGPoint)] = []
    for ring in rings {
        var cx: CGFloat = 0, cy: CGFloat = 0
        var n: CGFloat = 0
        for idx in ring where idx < positions.count {
            cx += positions[idx].x; cy += positions[idx].y; n += 1
        }
        guard n > 0 else { continue }
        ringCentroids.append((ring: Set(ring), centroid: CGPoint(x: cx / n, y: cy / n)))
    }

    // Helper: shrink a bond's endpoints if either atom is a heteroatom (so
    // the bond visually stops at the letter, not behind it)
    func bondEndpoints(_ a: Int, _ b: Int) -> (CGPoint, CGPoint) {
        let p1 = positions[a], p2 = positions[b]
        let isHetA = a < atomicNums.count && atomicNums[a] != 6
        let isHetB = b < atomicNums.count && atomicNums[b] != 6
        guard isHetA || isHetB else { return (p1, p2) }
        let dx = p2.x - p1.x, dy = p2.y - p1.y
        let len = hypot(dx, dy)
        guard len > 0 else { return (p1, p2) }
        let ux = dx / len, uy = dy / len
        let inset = style.heteroatomLabelInset
        let q1 = isHetA ? CGPoint(x: p1.x + ux * inset, y: p1.y + uy * inset) : p1
        let q2 = isHetB ? CGPoint(x: p2.x - ux * inset, y: p2.y - uy * inset) : p2
        return (q1, q2)
    }

    // --- Halos (drawn first, behind everything) ---
    for (idx, color) in halos where idx < positions.count {
        let p = positions[idx]
        let isHet = idx < atomicNums.count && atomicNums[idx] != 6
        let r: CGFloat = isHet ? 18 : 10
        context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r,
                                             width: r * 2, height: r * 2)),
                     with: .color(color.opacity(0.18)))
        context.stroke(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r,
                                               width: r * 2, height: r * 2)),
                       with: .color(color.opacity(0.55)), lineWidth: 1.8)
    }

    // --- Bonds ---
    for bond in bonds {
        guard bond.a < positions.count, bond.b < positions.count else { continue }
        let (p1, p2) = bondEndpoints(bond.a, bond.b)
        let dx = p2.x - p1.x, dy = p2.y - p1.y
        let len = hypot(dx, dy)
        guard len > 0 else { continue }

        if bond.order == 2 && !bond.isAromatic {
            // Double bond: outer line full length, inner line 60% length
            // Inner offset direction: toward ring centroid if both atoms in
            // a ring, else perpendicular toward +y.
            var nx = -dy / len, ny = dx / len
            if let entry = ringCentroids.first(where: { $0.ring.contains(bond.a) && $0.ring.contains(bond.b) }) {
                let mid = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
                let toC = CGPoint(x: entry.centroid.x - mid.x, y: entry.centroid.y - mid.y)
                if nx * toC.x + ny * toC.y < 0 { nx = -nx; ny = -ny }
            }
            // Outer line on the original bond axis
            var outer = Path(); outer.move(to: p1); outer.addLine(to: p2)
            context.stroke(outer, with: .color(style.bondColor), lineWidth: style.bondWidth)
            // Inner line offset by ~3 px
            let off: CGFloat = 3.5
            let shrink: CGFloat = 0.18
            let i1 = CGPoint(x: p1.x + dx * shrink + nx * off,
                             y: p1.y + dy * shrink + ny * off)
            let i2 = CGPoint(x: p2.x - dx * shrink + nx * off,
                             y: p2.y - dy * shrink + ny * off)
            var inner = Path(); inner.move(to: i1); inner.addLine(to: i2)
            context.stroke(inner, with: .color(style.bondColor),
                           lineWidth: style.bondWidth * 0.85)
        } else if bond.order == 3 {
            // Triple bond: three parallel lines
            let nx = -dy / len, ny = dx / len
            let off: CGFloat = 2.6
            for sign in [-1.0, 0.0, 1.0] as [CGFloat] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * off * sign, y: p1.y + ny * off * sign))
                path.addLine(to: CGPoint(x: p2.x + nx * off * sign, y: p2.y + ny * off * sign))
                context.stroke(path, with: .color(style.bondColor),
                               lineWidth: style.bondWidth * 0.85)
            }
        } else {
            // Single bond (or aromatic rendered as single line + ring circle)
            var path = Path(); path.move(to: p1); path.addLine(to: p2)
            context.stroke(path, with: .color(style.bondColor), lineWidth: style.bondWidth)
        }
    }

    // --- Aromatic ring inscribed circles ---
    for entry in ringCentroids {
        let pts = entry.ring.compactMap { idx in idx < positions.count ? positions[idx] : nil }
        guard pts.count >= 3 else { continue }
        let avgR = pts.map { hypot($0.x - entry.centroid.x, $0.y - entry.centroid.y) }
            .reduce(0, +) / CGFloat(pts.count)
        let r = avgR * style.aromaticCircleScale
        context.stroke(Path(ellipseIn: CGRect(x: entry.centroid.x - r,
                                               y: entry.centroid.y - r,
                                               width: r * 2, height: r * 2)),
                       with: .color(style.bondColor.opacity(0.6)),
                       lineWidth: style.bondWidth * 0.7)
    }

    // --- Atoms ---
    for (i, pos) in positions.enumerated() {
        let z = i < atomicNums.count ? atomicNums[i] : 6
        // Carbon: bare vertex (skeletal style)
        if z == 6 { continue }
        let (symbol, color) = atomDisplay(z)
        // Background pill so bonds visually stop at the letter
        let labelW: CGFloat = symbol.count > 1 ? 18 : 13
        let labelH: CGFloat = 14
        context.fill(Path(ellipseIn: CGRect(x: pos.x - labelW / 2,
                                             y: pos.y - labelH / 2,
                                             width: labelW, height: labelH)),
                     with: .color(style.backgroundColor))
        let text = Text(symbol)
            .font(.system(size: style.heteroatomFontSize, weight: .bold, design: .monospaced))
            .foregroundColor(color)
        context.draw(context.resolve(text), at: pos, anchor: .center)

        // Formal charge
        if i < formalCharges.count, formalCharges[i] != 0 {
            let label = formalCharges[i] > 0 ? "+" : "−"
            let chargeColor: Color = formalCharges[i] > 0
                ? Color(red: 0.3, green: 0.5, blue: 1.0)
                : Color(red: 1.0, green: 0.3, blue: 0.2)
            context.draw(context.resolve(
                Text(label).font(.system(size: 12, weight: .heavy, design: .rounded))
                    .foregroundColor(chargeColor)),
                at: CGPoint(x: pos.x + 9, y: pos.y - 9), anchor: .center)
        }
    }
}

// MARK: - Scene Drawer

/// Draw a complete diagram scene with auto-fit. Returns the screen-space
/// residue centers (after the auto-fit transform) for hover hit-testing.
@discardableResult
func drawScene(_ scene: DiagramScene, in context: GraphicsContext, size: CGSize)
    -> [(label: String, center: CGPoint, interactions: [MolecularInteraction])]
{
    // --- Auto-fit ---
    var allPoints = scene.ligandPositions
    for r in scene.residues {
        allPoints.append(CGPoint(x: r.center.x - 50, y: r.center.y - 24))
        allPoints.append(CGPoint(x: r.center.x + 50, y: r.center.y + 24))
        for p in r.scPositions {
            allPoints.append(CGPoint(x: p.x - 14, y: p.y - 14))
            allPoints.append(CGPoint(x: p.x + 14, y: p.y + 14))
        }
    }
    for mark in scene.hydrophobicMarks {
        let tipX = mark.anchor.x + mark.outwardNormal.x * 18
        let tipY = mark.anchor.y + mark.outwardNormal.y * 18
        allPoints.append(CGPoint(x: tipX - 6, y: tipY - 6))
        allPoints.append(CGPoint(x: tipX + 6, y: tipY + 6))
    }

    let xs = allPoints.map(\.x), ys = allPoints.map(\.y)
    let minX = xs.min() ?? 0, maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0, maxY = ys.max() ?? 0
    let w = max(maxX - minX, 1), h = max(maxY - minY, 1)
    let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2

    let margin: CGFloat = 24
    let fitScale = min((size.width - margin * 2) / w,
                       (size.height - margin * 2) / h, 1.3)
    let fitOffsetX = size.width / 2 - cx * fitScale
    let fitOffsetY = size.height / 2 - cy * fitScale

    var ctx = context
    ctx.translateBy(x: fitOffsetX, y: fitOffsetY)
    ctx.scaleBy(x: fitScale, y: fitScale)

    // --- Layer 1: ligand skeleton ---
    drawSkeletal(context: ctx,
                 positions: scene.ligandPositions,
                 bonds: scene.ligandBonds,
                 atomicNums: scene.ligandAtomicNums,
                 rings: scene.ligandRings,
                 formalCharges: scene.ligandFormalCharges,
                 style: .ligand,
                 halos: scene.pharmacophoreColors)

    // --- Layer 2: hydrophobic eyelashes (LigPlot+ style) ---
    let hydroColor = interactionColor(.hydrophobic)
    for mark in scene.hydrophobicMarks {
        drawHydrophobicEyelash(context: ctx, mark: mark, color: hydroColor)
    }

    // --- Layer 3: directional interaction lines ---
    for line in scene.directionalLines {
        drawDirectionalLine(context: ctx, line: line)
    }

    // --- Layer 4: residue side chains + bubbles (on top of lines) ---
    for residue in scene.residues {
        drawResidue(context: ctx, residue: residue)
    }

    // --- Compute screen-space residue centers for hover ---
    return scene.residues.map { r in
        (label: r.key.label,
         center: CGPoint(x: r.center.x * fitScale + fitOffsetX,
                         y: r.center.y * fitScale + fitOffsetY),
         interactions: r.interactions)
    }
}

// MARK: - Hydrophobic eyelash

private func drawHydrophobicEyelash(
    context: GraphicsContext, mark: DiagramScene.HydrophobicMark, color: Color
) {
    // Place eyelash anchor 14 px outside the ligand atom along the outward normal
    let baseDist: CGFloat = 14
    let baseX = mark.anchor.x + mark.outwardNormal.x * baseDist
    let baseY = mark.anchor.y + mark.outwardNormal.y * baseDist
    let base = CGPoint(x: baseX, y: baseY)

    // Tangent direction (perpendicular to normal)
    let tx = -mark.outwardNormal.y
    let ty = mark.outwardNormal.x

    // Draw 3 short eyelash arcs across a 16 px tangent span
    let span: CGFloat = 16
    let arcCount = 3
    for i in 0..<arcCount {
        let t = arcCount == 1 ? 0.5 : CGFloat(i) / CGFloat(arcCount - 1) - 0.5
        let cx = base.x + tx * span * t
        let cy = base.y + ty * span * t
        let inner = CGPoint(x: cx, y: cy)
        let outer = CGPoint(x: cx + mark.outwardNormal.x * 6,
                            y: cy + mark.outwardNormal.y * 6)
        var p = Path(); p.move(to: inner); p.addLine(to: outer)
        context.stroke(p, with: .color(color),
                       style: StrokeStyle(lineWidth: 2.2, lineCap: .round))
    }
    // Arc connecting the eyelash bases
    var arc = Path()
    let arcLeft = CGPoint(x: base.x - tx * span / 2, y: base.y - ty * span / 2)
    let arcRight = CGPoint(x: base.x + tx * span / 2, y: base.y + ty * span / 2)
    let ctrl = CGPoint(x: base.x + mark.outwardNormal.x * 4,
                       y: base.y + mark.outwardNormal.y * 4)
    arc.move(to: arcLeft)
    arc.addQuadCurve(to: arcRight, control: ctrl)
    context.stroke(arc, with: .color(color),
                   style: StrokeStyle(lineWidth: 2.2, lineCap: .round))

    // Dashed connector to each residue bubble
    for connector in mark.connectors {
        let tip = CGPoint(x: base.x + mark.outwardNormal.x * 8,
                          y: base.y + mark.outwardNormal.y * 8)
        let shortened = shortenSegment(from: tip, to: connector, fromInset: 0, toInset: 42)
        var path = Path(); path.move(to: shortened.0); path.addLine(to: shortened.1)
        context.stroke(path, with: .color(color.opacity(0.55)),
                       style: StrokeStyle(lineWidth: 1.4, dash: [4, 3]))
    }
}

// MARK: - Directional line

private func drawDirectionalLine(context: GraphicsContext, line: DiagramScene.DirectionalLine) {
    let color = interactionColor(line.type)
    let lw: CGFloat = 2.0

    var path = Path()
    path.move(to: line.from)
    if let ctrl = line.routedVia {
        path.addQuadCurve(to: line.to, control: ctrl)
    } else {
        path.addLine(to: line.to)
    }

    let stroke: StrokeStyle
    switch line.type {
    case .hbond, .saltBridge, .piCation, .amideStack:
        stroke = StrokeStyle(lineWidth: lw, dash: [6, 4.2])
    case .metalCoord:
        stroke = StrokeStyle(lineWidth: 2.5)
    default:
        stroke = StrokeStyle(lineWidth: lw)
    }
    context.stroke(path, with: .color(color), style: stroke)

    // Distance label at the midpoint of the line (or the bezier midpoint)
    let labelPt: CGPoint
    if let ctrl = line.routedVia {
        labelPt = CGPoint(x: 0.25 * line.from.x + 0.5 * ctrl.x + 0.25 * line.to.x,
                          y: 0.25 * line.from.y + 0.5 * ctrl.y + 0.25 * line.to.y)
    } else {
        labelPt = CGPoint(x: (line.from.x + line.to.x) / 2,
                          y: (line.from.y + line.to.y) / 2)
    }
    let distStr = String(format: "%.1f", line.distance)
    let sz = NSAttributedString(string: distStr,
        attributes: [.font: NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)]).size()
    let pill = CGRect(x: labelPt.x - sz.width / 2 - 4,
                      y: labelPt.y - sz.height / 2 - 2,
                      width: sz.width + 8, height: sz.height + 4)
    context.fill(Path(roundedRect: pill, cornerRadius: 4),
                 with: .color(Color(nsColor: .controlBackgroundColor).opacity(0.9)))
    context.draw(context.resolve(
        Text(distStr).font(.footnote.monospaced().weight(.medium))
            .foregroundColor(.secondary)), at: labelPt, anchor: .center)
}

// MARK: - Residue (side chain + bubble)

private func drawResidue(context: GraphicsContext, residue: DiagramScene.ResidueDraw) {
    // Side chain via the shared skeletal renderer
    if let tmpl = residue.template, !residue.scPositions.isEmpty,
       residue.scPositions.count == tmpl.atoms.count {
        let bonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)] =
            tmpl.bonds.map { (a: $0.from, b: $0.to, order: $0.order, isAromatic: $0.isAromatic) }
        let atomicNums = tmpl.atoms.map { $0.element.rawValue }
        let formalCharges: [Int] = tmpl.atoms.map { atom in
            switch (residue.key.name, atom.name) {
            case ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"):
                return -1
            case ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"):
                return +1
            default: return 0
            }
        }

        drawSkeletal(context: context,
                     positions: residue.scPositions,
                     bonds: bonds,
                     atomicNums: atomicNums,
                     rings: tmpl.aromaticRings,
                     formalCharges: formalCharges,
                     style: .sideChain)

        // Highlight interacting atoms with a colored ring
        let dominantColor = interactionColor(residue.dominantType)
        for (i, atom) in tmpl.atoms.enumerated() where i < residue.scPositions.count {
            guard residue.interactingAtomNames.contains(atom.name) else { continue }
            let pos = residue.scPositions[i]
            let r: CGFloat = atom.element == .C ? 7 : 11
            context.stroke(
                Path(ellipseIn: CGRect(x: pos.x - r, y: pos.y - r, width: r * 2, height: r * 2)),
                with: .color(dominantColor), lineWidth: 1.8)
        }

        // Donor hydrogens
        for (hPos, atomPos) in residue.donorHydrogens {
            var p = Path(); p.move(to: atomPos); p.addLine(to: hPos)
            context.stroke(p, with: .color(residue.propertyBorder.opacity(0.5)), lineWidth: 1.0)
            let hr: CGFloat = 5
            context.fill(Path(ellipseIn: CGRect(x: hPos.x - hr, y: hPos.y - hr,
                                                 width: hr * 2, height: hr * 2)),
                         with: .color(Color(nsColor: .controlBackgroundColor)))
            context.draw(context.resolve(
                Text("H").font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundColor(.gray)), at: hPos, anchor: .center)
        }
    }

    // Core bubble (residue label) drawn on top of side chain
    let nameSize = NSAttributedString(string: residue.key.label,
        attributes: [.font: NSFont.systemFont(ofSize: 12, weight: .bold)]).size()
    let bubbleW: CGFloat = max(nameSize.width + 24, 72)
    let bubbleH: CGFloat = 38
    let rect = CGRect(x: residue.center.x - bubbleW / 2,
                      y: residue.center.y - bubbleH / 2,
                      width: bubbleW, height: bubbleH)

    context.fill(Path(roundedRect: rect.offsetBy(dx: 1.5, dy: 1.5), cornerRadius: 8),
                 with: .color(Color(nsColor: .shadowColor).opacity(0.12)))
    context.fill(Path(roundedRect: rect, cornerRadius: 8), with: .color(residue.propertyFill))
    context.stroke(Path(roundedRect: rect, cornerRadius: 8),
                   with: .color(residue.propertyBorder), lineWidth: 2)

    context.draw(context.resolve(
        Text(residue.key.label).font(.callout.weight(.bold)).foregroundColor(.primary)),
        at: CGPoint(x: residue.center.x, y: residue.center.y - 4), anchor: .center)
    context.draw(context.resolve(
        Text(residue.dominantType.label).font(.caption2.weight(.medium))
            .foregroundColor(residue.propertyBorder)),
        at: CGPoint(x: residue.center.x, y: residue.center.y + 11), anchor: .center)
}

// MARK: - Pharmacophore halo color

private func pharmacophoreColor(forLigandAtomIndex index: Int,
                                interactions: [MolecularInteraction],
                                rings: [[Int]]) -> Color? {
    var types = interactions.filter { $0.ligandAtomIndex == index }.map(\.type)
    if types.isEmpty {
        for ring in rings where ring.contains(index) {
            for ixn in interactions where kPiTypes.contains(ixn.type) {
                if ring.contains(ixn.ligandAtomIndex) { types.append(ixn.type) }
            }
        }
    }
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

// MARK: - Shared helpers

func interactionColor(_ type: MolecularInteraction.InteractionType) -> Color {
    let c = type.color
    return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
}

func atomDisplay(_ atomicNum: Int) -> (String, Color) {
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

func residuePropertyColors(_ residueName: String) -> (fill: Color, border: Color) {
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

func dominantInteractionType(_ interactions: [MolecularInteraction]) -> MolecularInteraction.InteractionType {
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

func shortenSegment(from p1: CGPoint, to p2: CGPoint,
                    fromInset: CGFloat, toInset: CGFloat) -> (CGPoint, CGPoint) {
    let dx = p2.x - p1.x, dy = p2.y - p1.y
    let len = sqrt(dx * dx + dy * dy)
    guard len > fromInset + toInset else { return (p1, p2) }
    let ux = dx / len, uy = dy / len
    return (CGPoint(x: p1.x + ux * fromInset, y: p1.y + uy * fromInset),
            CGPoint(x: p2.x - ux * toInset, y: p2.y - uy * toInset))
}

func lineCrossesRect(from p1: CGPoint, to p2: CGPoint, rect: CGRect) -> Bool {
    if rect.contains(p1) && rect.contains(p2) { return false }
    if rect.contains(p1) || rect.contains(p2) { return false }
    let edges: [(CGPoint, CGPoint)] = [
        (CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY)),
        (CGPoint(x: rect.maxX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.maxY)),
        (CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.maxY)),
        (CGPoint(x: rect.minX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.minY)),
    ]
    var crossCount = 0
    for (e1, e2) in edges where segmentsIntersect(p1, p2, e1, e2) { crossCount += 1 }
    return crossCount >= 2
}

private func segmentsIntersect(_ a1: CGPoint, _ a2: CGPoint,
                                _ b1: CGPoint, _ b2: CGPoint) -> Bool {
    func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    let d1 = cross(b1, b2, a1), d2 = cross(b1, b2, a2)
    let d3 = cross(a1, a2, b1), d4 = cross(a1, a2, b2)
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
           ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
}
