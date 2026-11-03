// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PoseValidator.swift — PoseBusters-style validity checks for docking poses.
//
// Implements a focused subset of PoseBusters tests usable from the topology
// available in Druse (no RDKit detour required). Tests:
//
//   1. Bond lengths   — each bond's length is within ±25% of the covalent
//                        reference (sum of covalent radii).
//   2. Bond angles    — each angle (a—b—c) is within [70°, 175°]
//                        (upper bound 180° at sp centres: triple or
//                        cumulated double bonds, e.g. nitriles, alkynes).
//   3. Internal clash — non-bonded heavy atom pairs (separation ≥ 4 bonds)
//                        must not overlap below 0.7 × (vdw_i + vdw_j).
//   4. Protein clash  — minimum ligand–protein heavy atom distance ≥ 2.0 Å
//                        (severe clash threshold; mild clashes are common
//                        and surface as bumpScore in the docking score).
//   5. Connectivity   — all heavy atoms reachable via bonds (graph connected).
//   6. Extent         — ligand bounding box ≤ 30 Å in any dimension.
//
// Designed to be O(N²) on ligand atoms (N is small: ≤ ~100) and O(N×M) on
// protein heavy atoms (M ≤ ~10k), bounded by a 5 Å pre-filter on z-axis.
// ============================================================================

import Foundation
import simd

enum PoseValidator {

    /// Pose-independent ligand topology, built once per ligand and shared by every pose
    /// of a batch (adjacency, bond-separation exclusions and connectivity only depend on bonds).
    private struct Topology {
        let atomCount: Int
        let adjacency: [[Int]]
        /// sp centres (incident triple bond, or two cumulated double bonds) whose
        /// equilibrium bond angle is 180°, so the 175° upper bound does not apply.
        let isLinearCenter: [Bool]
        /// Pairs (i < j) separated by more than 3 bonds — the internal-clash candidates.
        let nonBondedPairs: [(Int, Int)]
        let isConnected: Bool

        init(atomCount n: Int, bonds: [Bond]) {
            atomCount = n
            var adj: [[Int]] = Array(repeating: [], count: n)
            var tripleBond = Array(repeating: false, count: n)
            var doubleBondCount = Array(repeating: 0, count: n)
            for b in bonds {
                let a1 = b.atomIndex1, a2 = b.atomIndex2
                guard a1 >= 0, a2 >= 0, a1 < n, a2 < n, a1 != a2, !adj[a1].contains(a2) else { continue }
                adj[a1].append(a2)
                adj[a2].append(a1)
                switch b.order {
                case .triple:
                    tripleBond[a1] = true; tripleBond[a2] = true
                case .double:
                    doubleBondCount[a1] += 1; doubleBondCount[a2] += 1
                default:
                    break
                }
            }
            adjacency = adj
            // Two-coordinate only: sulfonyl S / phosphate P also carry two double bonds but are tetrahedral.
            isLinearCenter = (0..<n).map { adj[$0].count == 2 && (tripleBond[$0] || doubleBondCount[$0] >= 2) }

            // Bond-graph distance up to 3 hops (1-2, 1-3, 1-4 are excluded from the clash check).
            var pairs: [(Int, Int)] = []
            var depth = Array(repeating: -1, count: n)
            for i in 0..<n {
                var touched = [i]
                depth[i] = 0
                var frontier = [i]
                for d in 1...3 {
                    var next: [Int] = []
                    for c in frontier {
                        for nb in adj[c] where depth[nb] < 0 {
                            depth[nb] = d
                            next.append(nb)
                            touched.append(nb)
                        }
                    }
                    frontier = next
                }
                for j in (i + 1)..<n where depth[j] < 0 { pairs.append((i, j)) }
                for t in touched { depth[t] = -1 }
            }
            nonBondedPairs = pairs

            // Connectivity (BFS from atom 0)
            if n == 0 {
                isConnected = true
            } else {
                var visited = Array(repeating: false, count: n)
                var stack = [0]
                visited[0] = true
                var visitedCount = 1
                while let cur = stack.popLast() {
                    for nb in adj[cur] where !visited[nb] {
                        visited[nb] = true
                        visitedCount += 1
                        stack.append(nb)
                    }
                }
                isConnected = visitedCount == n
            }
        }
    }

    static func validate(
        ligandHeavyAtoms: [Atom],
        ligandHeavyBonds: [Bond],
        ligandPositions: [SIMD3<Float>],
        proteinHeavyAtoms: [Atom]
    ) -> PoseValidity {
        validate(
            topology: Topology(atomCount: ligandHeavyAtoms.count, bonds: ligandHeavyBonds),
            ligandHeavyAtoms: ligandHeavyAtoms,
            ligandHeavyBonds: ligandHeavyBonds,
            ligandPositions: ligandPositions,
            proteinHeavyAtoms: proteinHeavyAtoms
        )
    }

    private static func validate(
        topology: Topology,
        ligandHeavyAtoms: [Atom],
        ligandHeavyBonds: [Bond],
        ligandPositions: [SIMD3<Float>],
        proteinHeavyAtoms: [Atom]
    ) -> PoseValidity {
        var v = PoseValidity()
        let n = ligandHeavyAtoms.count
        guard n > 0, ligandPositions.count == n, topology.atomCount == n else { return v }

        // -- 1. Bond lengths --
        var worstRatio: Float = 1.0
        for b in ligandHeavyBonds {
            guard b.atomIndex1 < n, b.atomIndex2 < n else { continue }
            let a1 = ligandHeavyAtoms[b.atomIndex1]
            let a2 = ligandHeavyAtoms[b.atomIndex2]
            let p1 = ligandPositions[b.atomIndex1]
            let p2 = ligandPositions[b.atomIndex2]
            let actual = simd_distance(p1, p2)
            let reference = a1.element.covalentRadius + a2.element.covalentRadius
            guard reference > 0 else { continue }
            let ratio = actual / reference
            let dev = abs(ratio - 1.0)
            if dev > abs(worstRatio - 1.0) { worstRatio = ratio }
            if ratio < 0.75 || ratio > 1.25 {
                v.bondLengthsOK = false
            }
        }
        v.worstBondRatio = worstRatio
        if !v.bondLengthsOK { v.failures.append("bond length") }

        let adj = topology.adjacency

        // -- 2. Bond angles --
        for j in 0..<n where adj[j].count >= 2 {
            let maxAngle: Float = topology.isLinearCenter[j] ? 180.0 : 175.0
            for ai in 0..<adj[j].count {
                let i = adj[j][ai]
                for bi in (ai + 1)..<adj[j].count {
                    let k = adj[j][bi]
                    let v1 = ligandPositions[i] - ligandPositions[j]
                    let v2 = ligandPositions[k] - ligandPositions[j]
                    let len1 = simd_length(v1)
                    let len2 = simd_length(v2)
                    if len1 < 1e-4 || len2 < 1e-4 { continue }
                    let cos = simd_dot(v1, v2) / (len1 * len2)
                    let cosClamped = max(min(cos, 1.0), -1.0)
                    let angDeg = acosf(cosClamped) * 180.0 / .pi
                    if angDeg < 70.0 || angDeg > maxAngle {
                        v.bondAnglesOK = false
                    }
                }
            }
        }
        if !v.bondAnglesOK { v.failures.append("bond angle") }

        // -- 5. Connectivity (pose-independent, from the bond graph) --
        v.connectedOK = topology.isConnected
        if !v.connectedOK { v.failures.append("connectivity") }

        // -- 3. Internal clash (pairs separated by ≥ 4 bonds) --
        var minInternal: Float = .greatestFiniteMagnitude
        for (i, j) in topology.nonBondedPairs {
            let d = simd_distance(ligandPositions[i], ligandPositions[j])
            if d < minInternal { minInternal = d }
            let threshold = 0.7 * (ligandHeavyAtoms[i].element.vdwRadius
                                   + ligandHeavyAtoms[j].element.vdwRadius)
            if d < threshold { v.internalClashOK = false }
        }
        v.minInternalDistance = minInternal
        if !v.internalClashOK { v.failures.append("internal clash") }

        // -- 4. Protein clash --
        // Pre-filter protein atoms by ligand bounding box + 5 Å margin.
        var lo = ligandPositions[0]
        var hi = ligandPositions[0]
        for p in ligandPositions {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        let pad: Float = 5.0
        lo -= SIMD3<Float>(repeating: pad)
        hi += SIMD3<Float>(repeating: pad)
        var minProt: Float = .greatestFiniteMagnitude
        for pa in proteinHeavyAtoms {
            let pp = pa.position
            if pp.x < lo.x || pp.x > hi.x ||
               pp.y < lo.y || pp.y > hi.y ||
               pp.z < lo.z || pp.z > hi.z { continue }
            for lp in ligandPositions {
                let d = simd_distance(pp, lp)
                if d < minProt { minProt = d }
            }
        }
        v.minProteinDistance = minProt
        if minProt < 2.0 { v.proteinClashOK = false; v.failures.append("protein clash") }

        // -- 6. Extent --
        let extent = hi - lo - SIMD3<Float>(repeating: 2 * pad)
        if max(extent.x, max(extent.y, extent.z)) > 30.0 {
            v.extentOK = false
            v.failures.append("extent")
        }

        return v
    }

    /// Validate a batch of poses against the same protein.
    static func validateBatch(
        results: [DockingResult],
        ligandHeavyAtoms: [Atom],
        ligandHeavyBonds: [Bond],
        proteinHeavyAtoms: [Atom]
    ) -> [DockingResult] {
        let topology = Topology(atomCount: ligandHeavyAtoms.count, bonds: ligandHeavyBonds)
        return results.map { r in
            var out = r
            guard !r.transformedAtomPositions.isEmpty else { return out }
            out.validity = validate(
                topology: topology,
                ligandHeavyAtoms: ligandHeavyAtoms,
                ligandHeavyBonds: ligandHeavyBonds,
                ligandPositions: r.transformedAtomPositions,
                proteinHeavyAtoms: proteinHeavyAtoms
            )
            return out
        }
    }
}
