import simd
import Foundation

/// Geometric pocket detection using alpha-sphere probes and DBSCAN clustering.
/// Implements the algorithm from PLAN.md Phase 2.3:
/// 1. Grid of sphere probes
/// 2. Filter by distance to nearest protein atom (2.0-5.5 Å)
/// 3. Buriedness scoring (26-direction ray casting)
/// 4. DBSCAN clustering
/// 5. Druggability ranking (volume × buriedness × polarity)
enum PocketDetection {

    struct Pocket: Identifiable {
        let id: Int
        var center: SIMD3<Float>
        var probePoints: [SIMD3<Float>]
        var volume: Float           // Å³
        var avgBuriedness: Float    // 0-1
        var residueIndices: [Int]   // protein residues within contact distance
        var druggability: Float     // composite score
    }

    struct DetectionParams {
        var gridSpacing: Float = 1.0       // Å between probe points
        var minDistance: Float = 2.0       // min distance from protein surface
        var maxDistance: Float = 5.5       // max distance from protein surface
        var buriednessThreshold: Float = 0.3 // min buriedness to keep probe
        var clusterEps: Float = 3.5       // DBSCAN epsilon (Å)
        var clusterMinPts: Int = 8        // DBSCAN min points
        var contactDistance: Float = 4.5  // distance to associate residues
    }

    // MARK: - Main Detection

    @MainActor
    static func detectPockets(
        in molecule: Molecule,
        params: DetectionParams = DetectionParams()
    ) -> [Pocket] {
        let proteinAtoms = molecule.atoms.filter { !$0.isHetAtom }
        guard proteinAtoms.count >= 10 else { return [] }

        let positions = proteinAtoms.map(\.position)

        // Step 1: Compute bounding box with padding
        let (bbMin, bbMax) = boundingBox(positions)
        let padding: Float = params.maxDistance + 2.0
        let gridMin = bbMin - SIMD3<Float>(repeating: padding)
        let gridMax = bbMax + SIMD3<Float>(repeating: padding)

        // Step 2: Build spatial hash for protein atoms
        let cellSize: Float = params.maxDistance + 1.0
        var spatialHash: [SIMD3<Int32>: [Int]] = [:]
        for (i, pos) in positions.enumerated() {
            let cell = SIMD3<Int32>(
                Int32(floorf(pos.x / cellSize)),
                Int32(floorf(pos.y / cellSize)),
                Int32(floorf(pos.z / cellSize))
            )
            spatialHash[cell, default: []].append(i)
        }

        // Step 3: Generate probe grid and filter by distance
        var probes: [SIMD3<Float>] = []
        var probeDistances: [Float] = []

        var x = gridMin.x
        while x <= gridMax.x {
            var y = gridMin.y
            while y <= gridMax.y {
                var z = gridMin.z
                while z <= gridMax.z {
                    let probe = SIMD3<Float>(x, y, z)
                    let minDist = nearestAtomDistance(probe, positions: positions,
                                                      spatialHash: spatialHash, cellSize: cellSize)

                    if minDist >= params.minDistance && minDist <= params.maxDistance {
                        probes.append(probe)
                        probeDistances.append(minDist)
                    }
                    z += params.gridSpacing
                }
                y += params.gridSpacing
            }
            x += params.gridSpacing
        }

        guard !probes.isEmpty else { return [] }

        // Step 4: Compute buriedness for each probe (26-direction ray casting)
        var buriedness: [Float] = []
        buriedness.reserveCapacity(probes.count)

        for probe in probes {
            let b = computeBuriedness(probe, positions: positions,
                                       spatialHash: spatialHash, cellSize: cellSize)
            buriedness.append(b)
        }

        // Step 5: Filter by buriedness
        var filteredProbes: [SIMD3<Float>] = []
        for (i, probe) in probes.enumerated() {
            if buriedness[i] >= params.buriednessThreshold {
                filteredProbes.append(probe)
            }
        }

        guard !filteredProbes.isEmpty else { return [] }

        // Step 6: DBSCAN clustering
        let clusters = dbscan(points: filteredProbes, eps: params.clusterEps, minPts: params.clusterMinPts)

        // Step 7: Build pockets from clusters
        var pockets: [Pocket] = []
        for (clusterId, clusterPoints) in clusters.enumerated() {
            guard clusterPoints.count >= params.clusterMinPts else { continue }

            let center = centroid(clusterPoints)
            let volume = Float(clusterPoints.count) * powf(params.gridSpacing, 3)

            // Average buriedness of cluster probes
            var totalBuriedness: Float = 0
            for pt in clusterPoints {
                totalBuriedness += computeBuriedness(pt, positions: positions,
                                                      spatialHash: spatialHash, cellSize: cellSize)
            }
            let avgB = totalBuriedness / Float(clusterPoints.count)

            // Find nearby residues
            let resIndices = findNearbyResidues(center: center, radius: params.contactDistance + volume.squareRoot() * 0.3,
                                                 molecule: molecule)

            let druggability = volume * avgB * Float(resIndices.count) / 100.0

            pockets.append(Pocket(
                id: clusterId,
                center: center,
                probePoints: clusterPoints,
                volume: volume,
                avgBuriedness: avgB,
                residueIndices: resIndices,
                druggability: druggability
            ))
        }

        // Sort by druggability
        pockets.sort { $0.druggability > $1.druggability }

        return pockets
    }

    // MARK: - Ligand-Guided Pocket Detection

    @MainActor
    static func detectPocketFromLigand(
        protein: Molecule,
        ligand: Molecule,
        contactDistance: Float = 4.5
    ) -> Pocket {
        let ligandCenter = ligand.center
        var nearbyResidues: Set<Int> = []

        for (resIdx, residue) in protein.residues.enumerated() {
            for atomIdx in residue.atomIndices {
                guard atomIdx < protein.atoms.count else { continue }
                let atomPos = protein.atoms[atomIdx].position
                for ligAtom in ligand.atoms {
                    if simd_length(atomPos - ligAtom.position) < contactDistance {
                        nearbyResidues.insert(resIdx)
                        break
                    }
                }
                if nearbyResidues.contains(resIdx) { break }
            }
        }

        return Pocket(
            id: 0,
            center: ligandCenter,
            probePoints: ligand.atoms.map(\.position),
            volume: 4.0 / 3.0 * .pi * powf(ligand.radius, 3),
            avgBuriedness: 0.8,
            residueIndices: Array(nearbyResidues),
            druggability: 1.0
        )
    }

    // MARK: - Helper: Nearest Atom Distance

    private static func nearestAtomDistance(
        _ point: SIMD3<Float>,
        positions: [SIMD3<Float>],
        spatialHash: [SIMD3<Int32>: [Int]],
        cellSize: Float
    ) -> Float {
        let cell = SIMD3<Int32>(
            Int32(floorf(point.x / cellSize)),
            Int32(floorf(point.y / cellSize)),
            Int32(floorf(point.z / cellSize))
        )

        var minDist: Float = .infinity
        for dx: Int32 in -1...1 {
            for dy: Int32 in -1...1 {
                for dz: Int32 in -1...1 {
                    let neighbor = SIMD3<Int32>(cell.x + dx, cell.y + dy, cell.z + dz)
                    guard let indices = spatialHash[neighbor] else { continue }
                    for i in indices {
                        let d = simd_length(point - positions[i])
                        if d < minDist { minDist = d }
                    }
                }
            }
        }
        return minDist
    }

    // MARK: - Buriedness (26-direction ray casting)

    private static let rayDirections: [SIMD3<Float>] = {
        var dirs: [SIMD3<Float>] = []
        for dx in [-1, 0, 1] {
            for dy in [-1, 0, 1] {
                for dz in [-1, 0, 1] {
                    if dx == 0 && dy == 0 && dz == 0 { continue }
                    dirs.append(simd_normalize(SIMD3<Float>(Float(dx), Float(dy), Float(dz))))
                }
            }
        }
        return dirs
    }()

    private static func computeBuriedness(
        _ point: SIMD3<Float>,
        positions: [SIMD3<Float>],
        spatialHash: [SIMD3<Int32>: [Int]],
        cellSize: Float,
        maxRayDist: Float = 10.0
    ) -> Float {
        var blocked = 0

        for dir in rayDirections {
            // Cast ray from point in direction, check if it hits any atom within maxRayDist
            var hit = false

            // Simple: check if any atom is within 2.0 Å of the ray line
            let steps = Int(maxRayDist / 2.0)
            for step in 1...steps {
                let testPoint = point + dir * Float(step) * 2.0
                let nearDist = nearestAtomDistance(testPoint, positions: positions,
                                                    spatialHash: spatialHash, cellSize: cellSize)
                if nearDist < 2.0 {
                    hit = true
                    break
                }
            }

            if hit { blocked += 1 }
        }

        return Float(blocked) / Float(rayDirections.count)
    }

    // MARK: - DBSCAN Clustering

    private static func dbscan(points: [SIMD3<Float>], eps: Float, minPts: Int) -> [[SIMD3<Float>]] {
        let n = points.count
        var labels = [Int](repeating: -1, count: n) // -1 = unvisited
        var clusterId = 0

        for i in 0..<n {
            guard labels[i] == -1 else { continue }

            let neighbors = regionQuery(points: points, pointIdx: i, eps: eps)
            if neighbors.count < minPts {
                labels[i] = -2 // noise
                continue
            }

            // Expand cluster
            labels[i] = clusterId
            var seeds = neighbors
            var seedIdx = 0

            while seedIdx < seeds.count {
                let j = seeds[seedIdx]
                if labels[j] == -2 { labels[j] = clusterId } // noise → border
                if labels[j] != -1 { seedIdx += 1; continue } // already processed

                labels[j] = clusterId
                let jNeighbors = regionQuery(points: points, pointIdx: j, eps: eps)
                if jNeighbors.count >= minPts {
                    seeds.append(contentsOf: jNeighbors)
                }
                seedIdx += 1
            }

            clusterId += 1
        }

        // Group points by cluster
        var clusters: [[SIMD3<Float>]] = Array(repeating: [], count: clusterId)
        for (i, label) in labels.enumerated() {
            if label >= 0 && label < clusterId {
                clusters[label].append(points[i])
            }
        }

        return clusters
    }

    private static func regionQuery(points: [SIMD3<Float>], pointIdx: Int, eps: Float) -> [Int] {
        let p = points[pointIdx]
        let eps2 = eps * eps
        var neighbors: [Int] = []
        for (j, q) in points.enumerated() {
            let diff = p - q
            if simd_dot(diff, diff) <= eps2 {
                neighbors.append(j)
            }
        }
        return neighbors
    }

    // MARK: - Find Nearby Residues

    @MainActor
    private static func findNearbyResidues(center: SIMD3<Float>, radius: Float, molecule: Molecule) -> [Int] {
        var result: [Int] = []
        let r2 = radius * radius
        for (resIdx, residue) in molecule.residues.enumerated() {
            guard residue.isStandard else { continue }
            for atomIdx in residue.atomIndices {
                guard atomIdx < molecule.atoms.count else { continue }
                let diff = molecule.atoms[atomIdx].position - center
                if simd_dot(diff, diff) <= r2 {
                    result.append(resIdx)
                    break
                }
            }
        }
        return result
    }
}
