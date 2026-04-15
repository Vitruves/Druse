// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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
    let s: CGFloat = 26  // bond length in local coords — close to ligand baseScale (30)
    let h: CGFloat = s * 0.866  // sin(60°) — used for both hexagons and zigzag y-offsets

    var t: [String: SideChainTemplate] = [:]

    // No side chain
    t["GLY"] = SideChainTemplate(atoms: [], bonds: [], aromaticRings: [])

    // Simple
    t["ALA"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C)],
        bonds: [],
        aromaticRings: [])

    // Branching aliphatics — proper Y at CB with 120° bond angles.
    t["VAL"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CG2", offset: CGPoint(x: s * 1.5, y:  h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["LEU"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CD2", offset: CGPoint(x: s, y: -h * 2), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    // ILE — Y at CB (CG1 upper, CG2 lower); CG1-CD1 zigzag right.
    t["ILE"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CG2", offset: CGPoint(x: s * 1.5, y:  h), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C)],
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

    // Hydroxyl / thiol — slight zigzag for visual variety.
    t["SER"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG", offset: CGPoint(x: s * 1.5, y: -h), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["THR"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG1", offset: CGPoint(x: s * 1.5, y: -h), element: .O),
                A(name: "CG2", offset: CGPoint(x: s * 1.5, y:  h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["CYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "SG", offset: CGPoint(x: s * 1.5, y: -h), element: .S)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false)],
        aromaticRings: [])

    // MET — proper zigzag CB-CG-SD-CE with 120° angles at every vertex.
    t["MET"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "SD", offset: CGPoint(x: s * 2.5, y: -h), element: .S),
                A(name: "CE", offset: CGPoint(x: s * 3, y: 0), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Carboxylate / amide — sp2 Y at CG (or CD) with both heteroatoms at 120°.
    t["ASP"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s * 2.5, y: -h), element: .O),
                A(name: "OD2", offset: CGPoint(x: s * 2.5, y:  h), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 2, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["GLU"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD",  offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "OE1", offset: CGPoint(x: s * 3, y: 0), element: .O),
                A(name: "OE2", offset: CGPoint(x: s * 3, y: -h * 2), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 2, isAromatic: false),
                B(from: 2, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ASN"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s * 2.5, y: -h), element: .O),
                A(name: "ND2", offset: CGPoint(x: s * 2.5, y:  h), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 2, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["GLN"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD",  offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "OE1", offset: CGPoint(x: s * 3, y: 0), element: .O),
                A(name: "NE2", offset: CGPoint(x: s * 3, y: -h * 2), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 2, isAromatic: false),
                B(from: 2, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Charged — long zigzag chains.
    t["LYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CE", offset: CGPoint(x: s * 3, y: 0), element: .C),
                A(name: "NZ", offset: CGPoint(x: s * 4, y: 0), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false),
                B(from: 3, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ARG"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD",  offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "NE",  offset: CGPoint(x: s * 3, y: 0), element: .N),
                A(name: "CZ",  offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "NH1", offset: CGPoint(x: s * 4.5, y: -h), element: .N),
                A(name: "NH2", offset: CGPoint(x: s * 4.5, y:  h), element: .N)],
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

    // TRP — proper indole: pyrrole (5-ring) fused to benzene (6-ring) along
    // the CD2-CE2 edge. CG attaches to CB on the left of the pyrrole; the
    // benzene extends to the lower right. All bonds length s, all angles 108°
    // (pentagon) or 120° (hexagon).
    t["TRP"] = {
        // Pyrrole pentagon: CG at (2s, 0), pentagon vertices derived from the
        // regular pentagon centered at (2s + r5, 0) where r5 = s/(2 sin 36°).
        // Coefficients precomputed for clarity.
        let CG  = CGPoint(x: s * 2, y: 0)
        let CD1 = CGPoint(x: s * 2.588, y: -s * 0.809)
        let NE1 = CGPoint(x: s * 3.539, y: -s * 0.500)
        let CE2 = CGPoint(x: s * 3.539, y:  s * 0.500)
        let CD2 = CGPoint(x: s * 2.588, y:  s * 0.809)

        // Benzene hexagon: shares the CD2-CE2 edge. Hexagon center is offset
        // from the edge midpoint by hexagon apothem (s*sqrt(3)/2) toward +y
        // (away from the pentagon center which is on -y from the edge).
        // Vertices in cycle order: CD2 → CE3 → CZ3 → CH2 → CZ2 → CE2 → CD2.
        let CE3 = CGPoint(x: s * 2.380, y: s * 1.785)
        let CZ3 = CGPoint(x: s * 3.123, y: s * 2.455)
        let CH2 = CGPoint(x: s * 4.073, y: s * 2.146)
        let CZ2 = CGPoint(x: s * 4.281, y: s * 1.169)

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
                    // Benzene (CD2 → CE3 → CZ3 → CH2 → CZ2 → CE2)
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

/// Distance from the residue bubble center at which CB (the side chain
/// attachment point) sits — places CB just outside the bubble edge.
let kCBAnchorDistance: CGFloat = 50

/// Extra rotation applied AFTER the centroid alignment so that CB→CG is
/// never parallel to the bubble→ligand axis. Without this, Y-branch residues
/// (TYR/PHE/ASP/LEU/VAL/THR/...) end up with a horizontal CB→CG bond,
/// because for any sp2/sp3 atom with three 120°-spaced substituents the
/// centroid of those substituents lies exactly on the CB→CG line.
/// −30° (math, CCW) tilts the chain so CB→CG sits at +30° above the chain
/// axis, restoring the conventional zigzag appearance.
let kSideChainZigzagTilt: CGFloat = -.pi / 6

/// Transform a side chain template so:
///   - CB (atoms[0]) sits at `kCBAnchorDistance` from the bubble center
///   - the chain extends outward toward the ligand contact
///   - the rotation aligns the CB→(centroid of all other atoms) vector with
///     the bubble→ligand vector. Using the centroid (not a single atom) keeps
///     every branch of the side chain visible — Y-branches like THR/VAL/ILE
///     no longer hide a CH3 behind the bubble.
///
/// Returns screen-space positions for each template atom.
func transformSideChain(template: SideChainTemplate,
                         bubbleCenter: CGPoint,
                         towardLigand: CGPoint) -> [CGPoint] {
    guard !template.atoms.isEmpty else { return [] }

    let cb = template.atoms[0].offset

    // Reference direction = from CB to centroid of all other side-chain atoms
    var refX: CGFloat = 0, refY: CGFloat = 0, n: CGFloat = 0
    for atom in template.atoms.dropFirst() {
        refX += atom.offset.x - cb.x
        refY += atom.offset.y - cb.y
        n += 1
    }
    if n > 0 { refX /= n; refY /= n }
    if hypot(refX, refY) < 1e-3 { refX = 1; refY = 0 }
    let templateAngle = atan2(refY, refX)

    // Direction from bubble center toward ligand contact
    let dx = towardLigand.x - bubbleCenter.x
    let dy = towardLigand.y - bubbleCenter.y
    let len = max(hypot(dx, dy), 1)
    let ux = dx / len, uy = dy / len
    let screenAngle = atan2(uy, ux)

    let rotateBy = screenAngle - templateAngle + kSideChainZigzagTilt
    let cosA = cos(rotateBy), sinA = sin(rotateBy)

    // CB position in screen space (just outside the bubble along bubble→ligand)
    let cbX = bubbleCenter.x + ux * kCBAnchorDistance
    let cbY = bubbleCenter.y + uy * kCBAnchorDistance

    return template.atoms.map { atom in
        let lx = atom.offset.x - cb.x
        let ly = atom.offset.y - cb.y
        let rx = lx * cosA - ly * sinA
        let ry = lx * sinA + ly * cosA
        return CGPoint(x: cbX + rx, y: cbY + ry)
    }
}
