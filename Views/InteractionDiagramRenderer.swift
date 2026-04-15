// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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
    let pharmacophoreColors: [Int: Color]                     // ligand atom index → halo color (non-π)

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
        let anchor: CGPoint                                   // group centroid (or ring centroid)
        let baseline: CGPoint                                 // eyelash arc center, outside the ligand
        let outwardNormal: CGPoint                            // unit vector anchor → baseline
        let span: CGFloat                                     // tangent span across the eyelash
        let residueCenter: CGPoint                            // single residue this mark connects to
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
    let orbitRadius = ligandHalfDiag + 180   // CB at edge-50, accommodates long chains

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
            // Centroid-pivot transform: every branch of the side chain stays
            // visible (no atoms hidden behind the bubble).
            scPositions = transformSideChain(template: tmpl,
                                              bubbleCenter: center,
                                              towardLigand: avgContact)
        }

        // Per-atom highlights are for directional interactions only —
        // hydrophobic contacts are visualized as eyelashes on the ligand side,
        // not as colored dots on side chain carbons.
        let interactingAtomNames = Set(ixns.compactMap { ixn -> String? in
            guard ixn.type != .hydrophobic else { return nil }
            guard ixn.proteinAtomIndex < proteinAtoms.count else { return nil }
            return proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
        })
        let (fill, border) = residuePropertyColors(key.name)
        let dominant = dominantInteractionType(ixns)

        residues.append(DiagramScene.ResidueDraw(
            key: key, center: center, template: template,
            scPositions: scPositions,
            interactingAtomNames: interactingAtomNames,
            propertyFill: fill, propertyBorder: border,
            dominantType: dominant, interactions: ixns))
    }

    // --- Build directional (non-hydrophobic) lines ---
    var directionalLines: [DiagramScene.DirectionalLine] = []
    for residue in residues {
        for ixn in residue.interactions where ixn.type != .hydrophobic {
            guard let ligPos = ligandContactPoint(for: ixn) else { continue }

            // Directional: target = side-chain atom or ring centroid
            var targetPos = residue.center
            if let tmpl = residue.template, !residue.scPositions.isEmpty,
               ixn.proteinAtomIndex < proteinAtoms.count {
                if kPiTypes.contains(ixn.type) {
                    let aName = proteinAtoms[ixn.proteinAtomIndex].name
                        .trimmingCharacters(in: .whitespaces)
                    if let centroid = sideChainRingCentroid(template: tmpl,
                                                            scPositions: residue.scPositions,
                                                            matchingAtomName: aName) {
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

            let toInset: CGFloat = (targetPos == residue.center) ? 40 : 8
            let shortened = shortenSegment(from: ligPos, to: targetPos,
                                            fromInset: 8, toInset: toInset)

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

    // --- Hydrophobic clustering ---
    //
    // A residue's hydrophobic contacts are grouped by ligand connectivity:
    // contiguous contact atoms become one cluster (so an entire phenyl ring or
    // an alkyl segment becomes ONE eyelash, not one per atom). When every
    // cluster atom belongs to the same aromatic ring, the anchor is the ring
    // centroid; otherwise the cluster centroid. Each (residue, cluster) pair
    // produces its own eyelash, oriented toward that residue and pushed past
    // the ligand AABB so it never overlaps the molecule.
    var ligandAdj: [[Int]] = Array(repeating: [], count: projected.count)
    for bond in coords.bonds where bond.0 < projected.count && bond.1 < projected.count {
        ligandAdj[bond.0].append(bond.1)
        ligandAdj[bond.1].append(bond.0)
    }
    var ringIdxForAtom: [Int: Int] = [:]
    for (rIdx, ring) in rings.enumerated() {
        for a in ring { ringIdxForAtom[a] = rIdx }
    }

    var hydrophobicMarks: [DiagramScene.HydrophobicMark] = []
    for residue in residues {
        let contactAtoms = Set(residue.interactions
            .filter { $0.type == .hydrophobic }
            .map(\.ligandAtomIndex)
            .filter { $0 < projected.count })
        guard !contactAtoms.isEmpty else { continue }

        var visited = Set<Int>()
        for start in contactAtoms where !visited.contains(start) {
            var component: [Int] = []
            var queue = [start]
            visited.insert(start)
            while !queue.isEmpty {
                let n = queue.removeFirst()
                component.append(n)
                for nb in ligandAdj[n]
                where contactAtoms.contains(nb) && !visited.contains(nb) {
                    visited.insert(nb)
                    queue.append(nb)
                }
            }

            // Anchor: full ring centroid if all atoms share one aromatic ring,
            // otherwise centroid of the contact atoms themselves.
            let anchor: CGPoint
            let radius: CGFloat
            let ringIdxs = Set(component.compactMap { ringIdxForAtom[$0] })
            if ringIdxs.count == 1, let rIdx = ringIdxs.first {
                let ringPts = rings[rIdx].compactMap {
                    $0 < projected.count ? projected[$0] : nil
                }
                let cx = ringPts.map(\.x).reduce(0, +) / CGFloat(ringPts.count)
                let cy = ringPts.map(\.y).reduce(0, +) / CGFloat(ringPts.count)
                anchor = CGPoint(x: cx, y: cy)
                radius = ringPts.map { hypot($0.x - cx, $0.y - cy) }.max() ?? 0
            } else {
                let pts = component.map { projected[$0] }
                let cx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
                let cy = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
                anchor = CGPoint(x: cx, y: cy)
                radius = pts.map { hypot($0.x - cx, $0.y - cy) }.max() ?? 0
            }

            // Direction toward the residue (not the ligand center) — guarantees
            // the eyelash sits on the residue side of the cluster.
            let rdx = residue.center.x - anchor.x
            let rdy = residue.center.y - anchor.y
            let rlen = max(hypot(rdx, rdy), 1)
            let normal = CGPoint(x: rdx / rlen, y: rdy / rlen)

            // Push the baseline past whichever is further: cluster radius or
            // the ligand AABB exit point along the normal.
            let aabbExit = rayBoxExitDistance(origin: anchor,
                                               direction: normal, box: ligandAABB)
            // Just outside the cluster (radius + 14) OR just past the AABB
            // exit (+ 10), whichever is larger. Capped to prevent runaway
            // distances when the residue is very far from the ligand.
            let baseDist = min(max(radius + 14, aabbExit + 10), radius + 30)
            let baseline = CGPoint(x: anchor.x + normal.x * baseDist,
                                   y: anchor.y + normal.y * baseDist)
            let span = max(18, radius * 1.4)

            hydrophobicMarks.append(DiagramScene.HydrophobicMark(
                anchor: anchor, baseline: baseline,
                outwardNormal: normal, span: span,
                residueCenter: residue.center))
        }
    }

    // --- Pharmacophore halo colors per ligand atom (excluding π types) ---
    // π interactions are now visualized as a translucent fill on the whole
    // aromatic ring, so we don't add per-atom halos for them.
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

/// Centroid of a side-chain aromatic ring in screen space. For TRP (two
/// fused rings) picks whichever ring contains the named protein atom; falls
/// back to the first ring otherwise.
private func sideChainRingCentroid(template: SideChainTemplate,
                                   scPositions: [CGPoint],
                                   matchingAtomName: String) -> CGPoint? {
    guard !template.aromaticRings.isEmpty else { return nil }
    let ring = template.aromaticRings.first { ring in
        ring.contains(where: { idx in
            idx < template.atoms.count && template.atoms[idx].name == matchingAtomName
        })
    } ?? template.aromaticRings[0]
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
    /// Unified RDKit-style skeletal rendering used for both the ligand and the
    /// side chains so they look visually consistent.
    static let classic = SkeletalStyle(
        bondWidth: 2.2, bondColor: Color.primary.opacity(0.85),
        heteroatomFontSize: 13,
        backgroundColor: Color(nsColor: .controlBackgroundColor),
        aromaticCircleScale: 0.55, heteroatomLabelInset: 9)
}

func drawSkeletal(
    context: GraphicsContext,
    positions: [CGPoint],
    bonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)],
    atomicNums: [Int],
    rings: [[Int]],
    formalCharges: [Int],
    style: SkeletalStyle,
    halos: [Int: Color] = [:],
    labelOverrides: [Int: String] = [:]    // explicit labels (e.g. "NH", "OH")
) {
    // Per-ring info, kept 1:1 with `rings` so callers can index by position.
    struct RingInfo {
        let atoms: Set<Int>
        let centroid: CGPoint
        let avgRadius: CGFloat
    }
    let ringInfos: [RingInfo?] = rings.map { ring in
        var cx: CGFloat = 0, cy: CGFloat = 0, n: CGFloat = 0
        for idx in ring where idx < positions.count {
            cx += positions[idx].x; cy += positions[idx].y; n += 1
        }
        guard n > 0 else { return nil }
        let centroid = CGPoint(x: cx / n, y: cy / n)
        let avgR = ring
            .compactMap { $0 < positions.count ? positions[$0] : nil }
            .map { hypot($0.x - centroid.x, $0.y - centroid.y) }
            .reduce(0, +) / n
        return RingInfo(atoms: Set(ring), centroid: centroid, avgRadius: avgR)
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
            if let entry = ringInfos.compactMap({ $0 }).first(where: { $0.atoms.contains(bond.a) && $0.atoms.contains(bond.b) }) {
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
    for info in ringInfos {
        guard let info else { continue }
        let r = info.avgRadius * style.aromaticCircleScale
        let rect = CGRect(x: info.centroid.x - r, y: info.centroid.y - r,
                          width: r * 2, height: r * 2)
        context.stroke(Path(ellipseIn: rect),
                       with: .color(style.bondColor.opacity(0.6)),
                       lineWidth: style.bondWidth * 0.7)
    }

    // --- Atoms ---
    for (i, pos) in positions.enumerated() {
        let z = i < atomicNums.count ? atomicNums[i] : 6
        // Carbon: bare vertex (skeletal style)
        if z == 6 { continue }
        let (defaultSymbol, color) = atomDisplay(z)
        let symbol = labelOverrides[i] ?? defaultSymbol
        // Background pill sized to the actual rendered text
        let nsFont = NSFont.monospacedSystemFont(ofSize: style.heteroatomFontSize, weight: .bold)
        let textSize = NSAttributedString(string: symbol, attributes: [.font: nsFont]).size()
        let labelW = textSize.width + 6
        let labelH = textSize.height + 2
        context.fill(Path(roundedRect: CGRect(x: pos.x - labelW / 2,
                                              y: pos.y - labelH / 2,
                                              width: labelW, height: labelH),
                                       cornerRadius: labelH / 2),
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
        let outDist: CGFloat = mark.span * 0.12 + 12      // bulge + tooth
        let tipX = mark.baseline.x + mark.outwardNormal.x * outDist
        let tipY = mark.baseline.y + mark.outwardNormal.y * outDist
        let tx = -mark.outwardNormal.y * mark.span / 2
        let ty = mark.outwardNormal.x * mark.span / 2
        allPoints.append(CGPoint(x: tipX + tx, y: tipY + ty))
        allPoints.append(CGPoint(x: tipX - tx, y: tipY - ty))
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
                 style: .classic,
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
    // Tangent direction (perpendicular to outward normal toward the residue)
    let tx = -mark.outwardNormal.y
    let ty = mark.outwardNormal.x

    let span = mark.span
    let teethCount = max(3, min(7, Int(span / 6)))
    let toothLen: CGFloat = 6
    let bulge: CGFloat = max(4, span * 0.12)

    // Quad-curve baseline cradling the cluster, opening toward the residue
    let arcLeft = CGPoint(x: mark.baseline.x - tx * span / 2,
                          y: mark.baseline.y - ty * span / 2)
    let arcRight = CGPoint(x: mark.baseline.x + tx * span / 2,
                           y: mark.baseline.y + ty * span / 2)
    let ctrl = CGPoint(x: mark.baseline.x + mark.outwardNormal.x * bulge,
                       y: mark.baseline.y + mark.outwardNormal.y * bulge)
    var arc = Path()
    arc.move(to: arcLeft)
    arc.addQuadCurve(to: arcRight, control: ctrl)
    context.stroke(arc, with: .color(color),
                   style: StrokeStyle(lineWidth: 2.2, lineCap: .round))

    // Teeth pointing outward from the baseline along the normal
    for i in 0..<teethCount {
        let t = teethCount == 1 ? CGFloat(0.5) : CGFloat(i) / CGFloat(teethCount - 1)
        let mt = 1 - t
        let bx = mt * mt * arcLeft.x + 2 * mt * t * ctrl.x + t * t * arcRight.x
        let by = mt * mt * arcLeft.y + 2 * mt * t * ctrl.y + t * t * arcRight.y
        let outer = CGPoint(x: bx + mark.outwardNormal.x * toothLen,
                            y: by + mark.outwardNormal.y * toothLen)
        var p = Path(); p.move(to: CGPoint(x: bx, y: by)); p.addLine(to: outer)
        context.stroke(p, with: .color(color),
                       style: StrokeStyle(lineWidth: 1.8, lineCap: .round))
    }

    // Dashed connector from the outer tip of the eyelash bulge to the residue
    let tip = CGPoint(x: mark.baseline.x + mark.outwardNormal.x * (bulge + toothLen),
                      y: mark.baseline.y + mark.outwardNormal.y * (bulge + toothLen))
    let shortened = shortenSegment(from: tip, to: mark.residueCenter,
                                    fromInset: 0, toInset: 42)
    var path = Path(); path.move(to: shortened.0); path.addLine(to: shortened.1)
    context.stroke(path, with: .color(color.opacity(0.55)),
                   style: StrokeStyle(lineWidth: 1.4, dash: [4, 3]))
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
        // Carboxylate: only the *deprotonated* oxygen carries the formal
        // charge. The double-bonded oxygen (OD1 / OE1 in our kekulized
        // template) stays neutral. This matches Druse's protonation model in
        // [Chemistry/Protonation.swift](Chemistry/Protonation.swift) which
        // assigns -1 to a single carboxylate oxygen at pH 7.4.
        let formalCharges: [Int] = tmpl.atoms.map { atom in
            switch (residue.key.name, atom.name) {
            case ("ASP", "OD2"), ("GLU", "OE2"):
                return -1
            case ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"):
                return +1
            default: return 0
            }
        }

        // Explicit hydrogen-count labels for heteroatoms (RDKit convention).
        // We list every donor/charged-N/S-H/O-H atom by name; everything else
        // falls back to the bare element symbol.
        var labelOverrides: [Int: String] = [:]
        for (i, atom) in tmpl.atoms.enumerated() {
            let label: String?
            switch (residue.key.name, atom.name) {
            case ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"):    label = "OH"
            case ("CYS", "SG"):                                    label = "SH"
            case ("LYS", "NZ"):                                    label = "NH\u{2083}"      // NH3 (charged)
            case ("ARG", "NH1"), ("ARG", "NH2"):                   label = "NH\u{2082}"      // NH2 (charged)
            case ("ARG", "NE"):                                    label = "NH"
            case ("ASN", "ND2"), ("GLN", "NE2"):                   label = "NH\u{2082}"      // amide
            case ("HIS", "ND1"), ("HIS", "NE2"):                   label = "NH"              // tautomer
            case ("TRP", "NE1"):                                   label = "NH"
            default:                                                label = nil
            }
            if let label { labelOverrides[i] = label }
        }

        // Atom-index set covered by any aromatic ring (used to skip per-atom
        // highlight rings on aromatic carbons — the π line already targets
        // the ring centroid, no need for a per-atom dot).
        let aromaticAtomIdx: Set<Int> = Set(tmpl.aromaticRings.flatMap { $0 })

        drawSkeletal(context: context,
                     positions: residue.scPositions,
                     bonds: bonds,
                     atomicNums: atomicNums,
                     rings: tmpl.aromaticRings,
                     formalCharges: formalCharges,
                     style: .classic,
                     labelOverrides: labelOverrides)

        // Per-atom highlight ring for interacting atoms — but skip aromatic
        // ring carbons (the π line shows the ring interaction by itself).
        let dominantColor = interactionColor(residue.dominantType)
        for (i, atom) in tmpl.atoms.enumerated() where i < residue.scPositions.count {
            guard residue.interactingAtomNames.contains(atom.name) else { continue }
            if aromaticAtomIdx.contains(i) { continue }
            let pos = residue.scPositions[i]
            let r: CGFloat = atom.element == .C ? 7 : 11
            context.stroke(
                Path(ellipseIn: CGRect(x: pos.x - r, y: pos.y - r, width: r * 2, height: r * 2)),
                with: .color(dominantColor), lineWidth: 1.8)
        }

        // (Donor hydrogens are now part of the heteroatom label itself —
        // the "OH"/"NH"/"NH2" text already shows the explicit H atoms.)

        // Connector line: bubble edge → CB. The bubble itself is drawn next
        // and will cover the inner end of this line.
        let cb = residue.scPositions[0]
        let cdx = cb.x - residue.center.x
        let cdy = cb.y - residue.center.y
        let clen = max(hypot(cdx, cdy), 1)
        var connector = Path()
        connector.move(to: residue.center)
        connector.addLine(to: CGPoint(x: residue.center.x + cdx / clen * (clen - 2),
                                      y: residue.center.y + cdy / clen * (clen - 2)))
        context.stroke(connector, with: .color(Color.primary.opacity(0.85)), lineWidth: 2.2)
    }

    // Core bubble (residue label) drawn on top of side chain + connector
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
    // Per-atom halos for directional interactions only.
    // - π types are visualized via the line targeting the ring centroid.
    // - hydrophobic is visualized via the eyelash on the ligand side.
    let excluded = kPiTypes.union([.hydrophobic, .chPi])
    let types = interactions
        .filter { $0.ligandAtomIndex == index && !excluded.contains($0.type) }
        .map(\.type)
    guard !types.isEmpty else { return nil }
    if types.contains(.hbond) || types.contains(.halogen) {
        return Color(red: 0.2, green: 0.6, blue: 1.0)
    }
    if types.contains(.saltBridge) {
        return Color(red: 1.0, green: 0.5, blue: 0.1)
    }
    if types.contains(.metalCoord) {
        return Color(red: 1.0, green: 0.85, blue: 0.0)
    }
    if types.contains(.chalcogen) {
        return Color(red: 0.5, green: 0.8, blue: 0.2)
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

/// Distance from `origin` along unit `direction` at which the ray first exits
/// `box`. If origin is inside the box, this is the distance to the closest
/// boundary along the direction. Returns 0 if the direction is degenerate.
func rayBoxExitDistance(origin: CGPoint, direction: CGPoint, box: CGRect) -> CGFloat {
    var tx = CGFloat.greatestFiniteMagnitude
    var ty = CGFloat.greatestFiniteMagnitude
    if direction.x > 1e-6 {
        tx = (box.maxX - origin.x) / direction.x
    } else if direction.x < -1e-6 {
        tx = (box.minX - origin.x) / direction.x
    }
    if direction.y > 1e-6 {
        ty = (box.maxY - origin.y) / direction.y
    } else if direction.y < -1e-6 {
        ty = (box.minY - origin.y) / direction.y
    }
    return max(0, min(tx, ty))
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
