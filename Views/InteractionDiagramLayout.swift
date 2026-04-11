import SwiftUI

// MARK: - Side Chain Templates
//
// Each residue is drawn as: a colored core box (residue name + seq number)
// with a side chain rendered in classical skeletal style next to it. The
// interaction line connects to the specific interacting atom in the side
// chain (or the ring centroid for π-interactions), not the box center.
//
// Template coordinates are in a local frame: (0,0) = Cα attachment point,
// chain extends rightward. Rotated at render time so the interacting atom
// points toward the ligand.

struct SideChainAtom {
    let name: String
    let offset: CGPoint     // local coords relative to Cα
    let element: Element
}

struct SideChainBond {
    let from: Int
    let to: Int
    let order: Int          // 1=single, 2=double (kekulized)
    let isAromatic: Bool    // true for ring bonds in PHE/TYR/HIS/TRP
}

struct SideChainTemplate {
    let atoms: [SideChainAtom]
    let bonds: [SideChainBond]
    let aromaticRings: [[Int]]   // atom-index sets per aromatic ring
}

/// Side chain templates for the 20 standard amino acids.
/// Only functional group atoms are included (not Cα/backbone).
let sideChainTemplates: [String: SideChainTemplate] = {
    typealias A = SideChainAtom
    typealias B = SideChainBond
    let s: CGFloat = 18  // bond length in local coords (matches ligand baseScale)
    let h: CGFloat = s * 0.866  // sin(60°) for hexagon geometry

    var t: [String: SideChainTemplate] = [:]

    // No side chain
    t["GLY"] = SideChainTemplate(atoms: [], bonds: [], aromaticRings: [])

    // Simple
    t["ALA"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C)],
        bonds: [],
        aromaticRings: [])

    // Branching aliphatics
    t["VAL"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s * 2, y: -s * 0.6), element: .C),
                A(name: "CG2", offset: CGPoint(x: s * 2, y: s * 0.6), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["LEU"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 3, y: -s * 0.6), element: .C),
                A(name: "CD2", offset: CGPoint(x: s * 3, y: s * 0.6), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ILE"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s * 2, y: -s * 0.5), element: .C),
                A(name: "CG2", offset: CGPoint(x: s, y: s), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 3, y: -s * 0.5), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["PRO"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 1.7, y: s * 0.7), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 0.7, y: s * 1.2), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Hydroxyl / thiol
    t["SER"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG", offset: CGPoint(x: s * 2, y: 0), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["THR"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG1", offset: CGPoint(x: s * 2, y: -s * 0.6), element: .O),
                A(name: "CG2", offset: CGPoint(x: s * 2, y: s * 0.6), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["CYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "SG", offset: CGPoint(x: s * 2, y: 0), element: .S)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["MET"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "SD", offset: CGPoint(x: s * 3, y: 0), element: .S),
                A(name: "CE", offset: CGPoint(x: s * 4, y: 0), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Carboxylate / amide
    t["ASP"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s * 3, y: -s * 0.6), element: .O),
                A(name: "OD2", offset: CGPoint(x: s * 3, y: s * 0.6), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 2, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["GLU"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 3, y: 0), element: .C),
                A(name: "OE1", offset: CGPoint(x: s * 4, y: -s * 0.6), element: .O),
                A(name: "OE2", offset: CGPoint(x: s * 4, y: s * 0.6), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 2, isAromatic: false),
                B(from: 2, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ASN"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s * 3, y: -s * 0.6), element: .O),
                A(name: "ND2", offset: CGPoint(x: s * 3, y: s * 0.6), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 2, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["GLN"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 3, y: 0), element: .C),
                A(name: "OE1", offset: CGPoint(x: s * 4, y: -s * 0.6), element: .O),
                A(name: "NE2", offset: CGPoint(x: s * 4, y: s * 0.6), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 2, isAromatic: false),
                B(from: 2, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Charged
    t["LYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 3, y: 0), element: .C),
                A(name: "CE", offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "NZ", offset: CGPoint(x: s * 5, y: 0), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false),
                B(from: 3, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ARG"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 3, y: 0), element: .C),
                A(name: "NE", offset: CGPoint(x: s * 4, y: 0), element: .N),
                A(name: "CZ", offset: CGPoint(x: s * 5, y: 0), element: .C),
                A(name: "NH1", offset: CGPoint(x: s * 6, y: -s * 0.6), element: .N),
                A(name: "NH2", offset: CGPoint(x: s * 6, y: s * 0.6), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false),
                B(from: 3, to: 4, order: 1, isAromatic: false),
                B(from: 4, to: 5, order: 2, isAromatic: false),
                B(from: 4, to: 6, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Aromatic — proper hexagonal geometry, bond length s
    t["PHE"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CE1", offset: CGPoint(x: s * 3.5, y: -h), element: .C),
                A(name: "CZ",  offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "CE2", offset: CGPoint(x: s * 3.5, y: h), element: .C),
                A(name: "CD2", offset: CGPoint(x: s * 2.5, y: h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: true),
                B(from: 2, to: 3, order: 1, isAromatic: true),
                B(from: 3, to: 4, order: 1, isAromatic: true),
                B(from: 4, to: 5, order: 1, isAromatic: true),
                B(from: 5, to: 6, order: 1, isAromatic: true),
                B(from: 6, to: 1, order: 1, isAromatic: true)],
        aromaticRings: [[1, 2, 3, 4, 5, 6]])

    t["TYR"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CE1", offset: CGPoint(x: s * 3.5, y: -h), element: .C),
                A(name: "CZ",  offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "CE2", offset: CGPoint(x: s * 3.5, y: h), element: .C),
                A(name: "CD2", offset: CGPoint(x: s * 2.5, y: h), element: .C),
                A(name: "OH",  offset: CGPoint(x: s * 5, y: 0), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: true),
                B(from: 2, to: 3, order: 1, isAromatic: true),
                B(from: 3, to: 4, order: 1, isAromatic: true),
                B(from: 4, to: 5, order: 1, isAromatic: true),
                B(from: 5, to: 6, order: 1, isAromatic: true),
                B(from: 6, to: 1, order: 1, isAromatic: true),
                B(from: 4, to: 7, order: 1, isAromatic: false)],
        aromaticRings: [[1, 2, 3, 4, 5, 6]])

    // HIS — 5-membered imidazole, regular pentagon (bond length s)
    // Pentagon center at (2.85s, 0), circumradius r = s / (2*sin(36°)) ≈ 0.851s
    t["HIS"] = {
        let r = s / (2 * sin(.pi / 5))           // ≈ 0.851 s
        let cx: CGFloat = s * 2 + r              // pentagon center
        // vertices at 180°, 108°, 36°, -36°, -108° (CW from CG, using screen y-down)
        func v(_ deg: CGFloat) -> CGPoint {
            CGPoint(x: cx + r * cos(deg * .pi / 180),
                    y: -r * sin(deg * .pi / 180))   // screen y inverted
        }
        return SideChainTemplate(
            atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                    A(name: "CG",  offset: v(180), element: .C),
                    A(name: "ND1", offset: v(108), element: .N),
                    A(name: "CE1", offset: v(36),  element: .C),
                    A(name: "NE2", offset: v(-36), element: .N),
                    A(name: "CD2", offset: v(-108), element: .C)],
            bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                    B(from: 1, to: 2, order: 1, isAromatic: true),
                    B(from: 2, to: 3, order: 1, isAromatic: true),
                    B(from: 3, to: 4, order: 1, isAromatic: true),
                    B(from: 4, to: 5, order: 1, isAromatic: true),
                    B(from: 5, to: 1, order: 1, isAromatic: true)],
            aromaticRings: [[1, 2, 3, 4, 5]])
    }()

    // TRP — indole: pyrrole (5-ring) fused to benzene (6-ring) sharing the
    // CD2-CE2 edge. Lay out the benzene as a regular hexagon and attach the
    // pentagon vertices CG/CD1/NE1 on the left side of the shared edge.
    t["TRP"] = {
        // Benzene center at (cx_b, 0). Benzene vertices (bond length s):
        //   CD2 (top-left)  ─ CE2 (top-right)
        //   CE3            CZ2
        //   CZ3            CH2
        // Place CD2 and CE2 horizontally so their edge points "up" (negative y).
        // Choose: CD2 = (cx_b - 0.5s, -h), CE2 = (cx_b + 0.5s, -h),
        //         CE3 = (cx_b - s, 0), CZ3 = (cx_b - 0.5s, h),
        //         CH2 = (cx_b + 0.5s, h), CZ2 = (cx_b + s, 0)
        // Then attach pyrrole on the upper-left side of the CD2-CE2 edge so the
        // pyrrole ring extends upward.
        let cxB: CGFloat = s * 4
        let CD2 = CGPoint(x: cxB - 0.5 * s, y: h)
        let CE2 = CGPoint(x: cxB + 0.5 * s, y: h)
        let CE3 = CGPoint(x: cxB - s,       y: 0)
        let CZ3 = CGPoint(x: cxB - 0.5 * s, y: -h)
        let CH2 = CGPoint(x: cxB + 0.5 * s, y: -h)
        let CZ2 = CGPoint(x: cxB + s,       y: 0)

        // Pyrrole on the lower-left side of the CD2-CE2 edge (extends downward
        // in screen space). Pentagon shares the CE2-CD2 edge with benzene.
        // Pentagon circumradius r5 = s / (2*sin(36°))
        let r5 = s / (2 * sin(.pi / 5))
        // Midpoint of shared edge
        let midX = (CD2.x + CE2.x) / 2
        let midY = (CD2.y + CE2.y) / 2
        // Pentagon center is offset perpendicular to CD2->CE2 by apothem
        let apothem = r5 * cos(.pi / 5)
        // CD2->CE2 unit vector = (1, 0); perpendicular toward +y (downward) = (0, 1)
        let pcx = midX + 0 * apothem
        let pcy = midY + 1 * apothem
        // Pentagon vertices: shared edge endpoints (CD2 at angle 180°-36°=144°,
        // CE2 at 36°), then NE1 (-36°), CD1 (-108°), CG (-180°)
        func pv(_ deg: CGFloat) -> CGPoint {
            CGPoint(x: pcx + r5 * cos(deg * .pi / 180),
                    y: pcy + r5 * sin(deg * .pi / 180))
        }
        let NE1 = pv(-36 + 360)   // angle below shared edge, right side
        let CD1 = pv(-108 + 360)
        let CG  = pv(180)

        return SideChainTemplate(
            atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                    A(name: "CG",  offset: CG, element: .C),
                    A(name: "CD1", offset: CD1, element: .C),
                    A(name: "NE1", offset: NE1, element: .N),
                    A(name: "CE2", offset: CE2, element: .C),
                    A(name: "CD2", offset: CD2, element: .C),
                    A(name: "CE3", offset: CE3, element: .C),
                    A(name: "CZ3", offset: CZ3, element: .C),
                    A(name: "CH2", offset: CH2, element: .C),
                    A(name: "CZ2", offset: CZ2, element: .C)],
            bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),  // CB-CG
                    // Pyrrole
                    B(from: 1, to: 2, order: 1, isAromatic: true),   // CG-CD1
                    B(from: 2, to: 3, order: 1, isAromatic: true),   // CD1-NE1
                    B(from: 3, to: 4, order: 1, isAromatic: true),   // NE1-CE2
                    B(from: 4, to: 5, order: 1, isAromatic: true),   // CE2-CD2 (fused)
                    B(from: 5, to: 1, order: 1, isAromatic: true),   // CD2-CG
                    // Benzene
                    B(from: 5, to: 6, order: 1, isAromatic: true),   // CD2-CE3
                    B(from: 6, to: 7, order: 1, isAromatic: true),   // CE3-CZ3
                    B(from: 7, to: 8, order: 1, isAromatic: true),   // CZ3-CH2
                    B(from: 8, to: 9, order: 1, isAromatic: true),   // CH2-CZ2
                    B(from: 9, to: 4, order: 1, isAromatic: true)],  // CZ2-CE2
            aromaticRings: [[1, 2, 3, 4, 5],          // pyrrole
                            [4, 5, 6, 7, 8, 9]])      // benzene
    }()

    return t
}()

// MARK: - Aromatic Ring Detection (ligand)

/// Find all aromatic 5- and 6-membered rings in the kekulized bond graph.
/// A ring is "aromatic" iff every atom in it has `isAromatic[i] == true`
/// (per-atom aromaticity is captured before kekulization in the C++ side).
func detectAromaticRings(bonds: [(Int, Int, Int)],
                          atomCount: Int,
                          isAromatic: [Bool]) -> [[Int]] {
    var adj: [[Int]] = Array(repeating: [], count: atomCount)
    for (a1, a2, _) in bonds {
        guard a1 < atomCount, a2 < atomCount else { continue }
        adj[a1].append(a2); adj[a2].append(a1)
    }
    var allRings: [[Int]] = []
    var globalSeen = Set<Int>()
    for start in 0..<atomCount where !adj[start].isEmpty && !globalSeen.contains(start) {
        var comp: [Int] = []; var queue = [start]; var seen = Set([start])
        while !queue.isEmpty {
            let n = queue.removeFirst(); comp.append(n)
            for nb in adj[n] where !seen.contains(nb) { seen.insert(nb); queue.append(nb) }
        }
        globalSeen.formUnion(seen)
        let compSet = Set(comp)
        for ringSize in [5, 6] {
            for s in comp {
                var path = [s]
                findRingDFS(cur: s, start: s, depth: 1, maxDepth: ringSize,
                            path: &path, adj: adj, nodes: compSet, rings: &allRings)
            }
        }
    }
    var unique = Set<[Int]>()
    return allRings
        .filter { ring in ring.allSatisfy { idx in idx < isAromatic.count && isAromatic[idx] } }
        .filter { unique.insert(canonRing($0)).inserted }
}

private func findRingDFS(cur: Int, start: Int, depth: Int, maxDepth: Int,
                          path: inout [Int], adj: [[Int]], nodes: Set<Int>, rings: inout [[Int]]) {
    if depth == maxDepth {
        if adj[cur].contains(start) { rings.append(Array(path)) }
        return
    }
    for next in adj[cur] {
        guard nodes.contains(next), !path.contains(next) else { continue }
        if next == start && depth > 2 { continue }
        path.append(next)
        findRingDFS(cur: next, start: start, depth: depth + 1, maxDepth: maxDepth,
                    path: &path, adj: adj, nodes: nodes, rings: &rings)
        path.removeLast()
    }
}

private func canonRing(_ ring: [Int]) -> [Int] {
    guard let minVal = ring.min(), let minIdx = ring.firstIndex(of: minVal) else { return ring }
    let rot = Array(ring[minIdx...]) + Array(ring[..<minIdx])
    let rev = [rot[0]] + rot[1...].reversed()
    return rot.lexicographicallyPrecedes(rev) ? rot : rev
}

// MARK: - Side Chain Transform

/// Transform a side chain template so the interacting atom (or ring centroid)
/// faces toward the ligand. Returns screen-space positions for each template
/// atom.
func transformSideChain(template: SideChainTemplate, bubbleCenter: CGPoint,
                         pivotPoint: CGPoint, towardLigand: CGPoint) -> [CGPoint] {
    guard !template.atoms.isEmpty else { return [] }

    // Direction from bubble center toward ligand contact
    let dx = towardLigand.x - bubbleCenter.x
    let dy = towardLigand.y - bubbleCenter.y
    let angle = atan2(dy, dx)
    let cosA = cos(angle), sinA = sin(angle)

    let dist = max(hypot(dx, dy), 1)
    let ux = dx / dist, uy = dy / dist
    // Place pivot atom outside the bubble (half-width ~40px + clearance)
    let anchorDist: CGFloat = 70

    return template.atoms.map { atom in
        let lx = atom.offset.x - pivotPoint.x
        let ly = atom.offset.y - pivotPoint.y
        let rx = lx * cosA - ly * sinA
        let ry = lx * sinA + ly * cosA
        return CGPoint(x: bubbleCenter.x + ux * anchorDist + rx,
                       y: bubbleCenter.y + uy * anchorDist + ry)
    }
}

/// Convenience: pivot on a named atom in the template.
func transformSideChain(template: SideChainTemplate, bubbleCenter: CGPoint,
                         interactingAtomName: String, towardLigand: CGPoint) -> [CGPoint] {
    let targetIdx = template.atoms.firstIndex(where: { $0.name == interactingAtomName })
        ?? template.atoms.count - 1
    return transformSideChain(template: template, bubbleCenter: bubbleCenter,
                              pivotPoint: template.atoms[targetIdx].offset,
                              towardLigand: towardLigand)
}

/// Convenience: pivot on the centroid of a ring (atom indices into the
/// template). Used for π-interactions where the line should connect to the
/// ring center, not a single atom.
func transformSideChainRingPivot(template: SideChainTemplate, bubbleCenter: CGPoint,
                                  ringAtomIndices: [Int], towardLigand: CGPoint) -> [CGPoint] {
    var px: CGFloat = 0, py: CGFloat = 0
    var n: CGFloat = 0
    for idx in ringAtomIndices where idx < template.atoms.count {
        px += template.atoms[idx].offset.x
        py += template.atoms[idx].offset.y
        n += 1
    }
    guard n > 0 else { return transformSideChain(template: template, bubbleCenter: bubbleCenter,
                                                 interactingAtomName: "", towardLigand: towardLigand) }
    return transformSideChain(template: template, bubbleCenter: bubbleCenter,
                              pivotPoint: CGPoint(x: px / n, y: py / n),
                              towardLigand: towardLigand)
}
