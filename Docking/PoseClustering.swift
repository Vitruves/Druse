// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

/// Complete-linkage agglomerative hierarchical clustering of docking poses by RMSD.
enum PoseClustering {

    /// Cluster docking poses using complete-linkage agglomerative clustering.
    ///
    /// Complete linkage defines the distance between two clusters as the *maximum* RMSD
    /// between any pair of poses across the two clusters. Two clusters are merged only when
    /// every member of one cluster is within `rmsdCutoff` of every member of the other.
    ///
    /// - Parameters:
    ///   - poses: Array of docking results with `transformedAtomPositions` (heavy atoms only).
    ///   - rmsdCutoff: Maximum RMSD (Angstroms) to merge two clusters. Default 2.0 A.
    /// - Returns: Array of clusters, each an array of indices into the input `poses` array.
    ///            Clusters are sorted by the best (lowest) energy within each cluster.
    static func clusterPoses(poses: [DockingResult], rmsdCutoff: Float = 2.0) -> [[Int]] {
        let n = poses.count
        guard n > 0 else { return [] }
        if n == 1 { return [[0]] }

        // Build condensed distance matrix (RMSD between all pairs, heavy atoms only)
        // Store as flat upper-triangular: index for (i,j) with i<j is i*n - i*(i+1)/2 + j - i - 1
        let condensedSize = n * (n - 1) / 2
        var dist = [Float](repeating: 0, count: condensedSize)

        for i in 0..<n {
            let posI = poses[i].transformedAtomPositions
            for j in (i + 1)..<n {
                let posJ = poses[j].transformedAtomPositions
                let r = rmsd(posI, posJ)
                let idx = i * n - i * (i + 1) / 2 + j - i - 1
                dist[idx] = r
            }
        }

        // Each element starts in its own cluster
        // clusterOf[i] tracks which cluster index element i belongs to
        var clusterMembers: [[Int]] = (0..<n).map { [$0] }
        var active = Set(0..<n)  // active cluster indices

        // Agglomerative loop: merge closest pair under complete linkage until no merge possible
        while active.count > 1 {
            var bestI = -1
            var bestJ = -1
            var bestDist: Float = .infinity

            let activeList = Array(active).sorted()

            // Find the pair of active clusters with minimum complete-linkage distance
            for ai in 0..<activeList.count {
                let ci = activeList[ai]
                for aj in (ai + 1)..<activeList.count {
                    let cj = activeList[aj]

                    // Complete linkage: max RMSD between all pairs across clusters
                    var maxDist: Float = 0
                    var exceeds = false
                    for mi in clusterMembers[ci] {
                        for mj in clusterMembers[cj] {
                            let (lo, hi) = mi < mj ? (mi, mj) : (mj, mi)
                            let idx = lo * n - lo * (lo + 1) / 2 + hi - lo - 1
                            let d = dist[idx]
                            if d > rmsdCutoff {
                                exceeds = true
                                break
                            }
                            if d > maxDist { maxDist = d }
                        }
                        if exceeds { break }
                    }

                    if !exceeds && maxDist < bestDist {
                        bestDist = maxDist
                        bestI = ci
                        bestJ = cj
                    }
                }
            }

            // No pair within cutoff -- stop merging
            if bestI < 0 { break }

            // Merge bestJ into bestI
            clusterMembers[bestI].append(contentsOf: clusterMembers[bestJ])
            clusterMembers[bestJ].removeAll()
            active.remove(bestJ)
        }

        // Collect active clusters
        var clusters = active.sorted().map { clusterMembers[$0] }

        // Sort clusters by best (lowest) energy within each cluster
        clusters.sort { clusterA, clusterB in
            let bestA = clusterA.map { poses[$0].energy }.min() ?? .infinity
            let bestB = clusterB.map { poses[$0].energy }.min() ?? .infinity
            return bestA < bestB
        }

        // Within each cluster, sort members by energy
        clusters = clusters.map { cluster in
            cluster.sorted { poses[$0].energy < poses[$1].energy }
        }

        return clusters
    }

    // MARK: - RMSD (heavy atoms only)

    /// Compute RMSD between two sets of 3D positions.
    /// Positions correspond to heavy atoms from the ligand.
    private static func rmsd(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var sum: Float = 0
        for i in 0..<a.count {
            sum += simd_distance_squared(a[i], b[i])
        }
        return sqrt(sum / Float(a.count))
    }
}
