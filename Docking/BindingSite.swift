import Foundation
import Metal
import simd

// MARK: - Data Model

/// Represents a detected binding pocket on a protein surface.
struct BindingPocket: Identifiable, Sendable {
    let id: Int
    var center: SIMD3<Float>
    var size: SIMD3<Float>               // half-widths of search box in each dimension
    var volume: Float                     // cubic angstroms
    var buriedness: Float                 // 0-1, fraction of enclosure
    var polarity: Float                   // 0-1, fraction of nearby polar atoms (N, O)
    var druggability: Float               // composite score: volume * buriedness * (1 + polarity)
    var residueIndices: [Int]             // indices into protein.residues
    var probePositions: [SIMD3<Float>]    // retained for visualization
}

// MARK: - Sendable snapshot for off-main-actor computation

/// A lightweight, Sendable snapshot of the protein data needed for pocket detection.
/// Extracted on the main actor from the @MainActor Molecule, then freely used in
/// background computation without holding any reference to the original.
private struct ProteinSnapshot: Sendable {
    let positions: [SIMD3<Float>]         // all heavy atom positions
    let elements: [Element]               // parallel to positions
    let atomResidueIndex: [Int]           // which residue each snapshot atom belongs to (-1 if none)
    let residueAtomIndices: [[Int]]       // residue index -> snapshot atom indices
    let residueIsStandard: [Bool]
}

// MARK: - Spatial Hash

/// Uniform-grid spatial hash for O(1) neighbor lookups in 3D.
private struct SpatialHash: Sendable {
    let cellSize: Float
    private let cells: [CellKey: [Int]]

    /// Hashable 3D cell coordinate.
    struct CellKey: Hashable, Sendable {
        let x: Int32
        let y: Int32
        let z: Int32
    }

    init(positions: [SIMD3<Float>], cellSize: Float) {
        self.cellSize = cellSize
        var map: [CellKey: [Int]] = [:]
        map.reserveCapacity(positions.count / 4)
        for (i, pos) in positions.enumerated() {
            let key = Self.cellKeyFor(pos, cellSize: cellSize)
            map[key, default: []].append(i)
        }
        self.cells = map
    }

    static func cellKeyFor(_ point: SIMD3<Float>, cellSize: Float) -> CellKey {
        CellKey(
            x: Int32(floorf(point.x / cellSize)),
            y: Int32(floorf(point.y / cellSize)),
            z: Int32(floorf(point.z / cellSize))
        )
    }

    /// Returns (squared distance, index) to the nearest position. Index is -1 if none found.
    func nearestDistanceSquared(
        to point: SIMD3<Float>,
        positions: [SIMD3<Float>]
    ) -> (distSq: Float, index: Int) {
        let cell = Self.cellKeyFor(point, cellSize: cellSize)
        var bestDistSq: Float = .infinity
        var bestIdx = -1

        for dx: Int32 in -1 ... 1 {
            for dy: Int32 in -1 ... 1 {
                for dz: Int32 in -1 ... 1 {
                    let neighbor = CellKey(x: cell.x &+ dx, y: cell.y &+ dy, z: cell.z &+ dz)
                    guard let indices = cells[neighbor] else { continue }
                    for i in indices {
                        let diff = point - positions[i]
                        let d2 = simd_dot(diff, diff)
                        if d2 < bestDistSq {
                            bestDistSq = d2
                            bestIdx = i
                        }
                    }
                }
            }
        }
        return (bestDistSq, bestIdx)
    }

    /// Returns all indices within `radius` of `point`.
    func indicesWithin(
        radius: Float,
        of point: SIMD3<Float>,
        positions: [SIMD3<Float>]
    ) -> [Int] {
        let r2 = radius * radius
        let cellsToCheck = Int32(ceilf(radius / cellSize))
        let cell = Self.cellKeyFor(point, cellSize: cellSize)
        var result: [Int] = []

        for dx in -cellsToCheck ... cellsToCheck {
            for dy in -cellsToCheck ... cellsToCheck {
                for dz in -cellsToCheck ... cellsToCheck {
                    let neighbor = CellKey(x: cell.x &+ dx, y: cell.y &+ dy, z: cell.z &+ dz)
                    guard let indices = cells[neighbor] else { continue }
                    for i in indices {
                        let diff = point - positions[i]
                        if simd_dot(diff, diff) <= r2 {
                            result.append(i)
                        }
                    }
                }
            }
        }
        return result
    }
}

// MARK: - Precomputed ray directions

/// 26 directions for buriedness scoring: all non-zero {-1,0,1}^3 combinations, normalized.
/// These cover the 6 face normals, 12 edge midpoints, and 8 cube vertices.
private let buriednessRayDirections: [SIMD3<Float>] = {
    var dirs: [SIMD3<Float>] = []
    dirs.reserveCapacity(26)
    for dx in [-1, 0, 1] {
        for dy in [-1, 0, 1] {
            for dz in [-1, 0, 1] {
                guard dx != 0 || dy != 0 || dz != 0 else { continue }
                let v = SIMD3<Float>(Float(dx), Float(dy), Float(dz))
                dirs.append(simd_normalize(v))
            }
        }
    }
    return dirs
}()

private struct BuriedProbe: Sendable {
    var position: SIMD3<Float>
    var buriedness: Float
}

private final class PocketDetectionMetalAccelerator {
    nonisolated(unsafe) static let shared: PocketDetectionMetalAccelerator? = {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let distanceFunction = library.makeFunction(name: "computePocketDistanceGrid"),
              let buriednessFunction = library.makeFunction(name: "scorePocketBuriedness")
        else { return nil }

        do {
            let distancePipeline = try device.makeComputePipelineState(function: distanceFunction)
            let buriednessPipeline = try device.makeComputePipelineState(function: buriednessFunction)
            return PocketDetectionMetalAccelerator(
                device: device,
                commandQueue: commandQueue,
                distancePipeline: distancePipeline,
                buriednessPipeline: buriednessPipeline
            )
        } catch {
            return nil
        }
    }()

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let distancePipeline: MTLComputePipelineState
    private let buriednessPipeline: MTLComputePipelineState

    private init(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        distancePipeline: MTLComputePipelineState,
        buriednessPipeline: MTLComputePipelineState
    ) {
        self.device = device
        self.commandQueue = commandQueue
        self.distancePipeline = distancePipeline
        self.buriednessPipeline = buriednessPipeline
    }

    func findBuriedProbes(
        positions: [SIMD3<Float>],
        radii: [Float],
        gridMin: SIMD3<Float>,
        spacing: Float,
        dims: SIMD3<UInt32>,
        minProbeDist: Float,
        maxProbeDist: Float,
        buriednessThreshold: Float,
        rayMaxDist: Float
    ) -> [BuriedProbe]? {
        guard !positions.isEmpty, positions.count == radii.count else { return [] }

        let nx = Int(dims.x)
        let ny = Int(dims.y)
        let totalPoints = Int(dims.x * dims.y * dims.z)
        guard totalPoints > 0 else { return [] }

        var gpuAtoms = zip(positions, radii).map { PocketAtomGPU(position: $0.0, vdwRadius: $0.1) }
        var params = PocketGridParams(
            origin: gridMin,
            spacing: spacing,
            dims: dims,
            totalPoints: UInt32(totalPoints),
            minProbeDist: minProbeDist,
            maxProbeDist: maxProbeDist,
            numAtoms: UInt32(gpuAtoms.count),
            rayStep: max(1.0, min(2.0, spacing * 1.25)),
            rayMaxDist: rayMaxDist,
            probeCount: 0,
            _pad0: 0
        )

        guard let atomBuffer = device.makeBuffer(
                bytes: &gpuAtoms,
                length: gpuAtoms.count * MemoryLayout<PocketAtomGPU>.stride,
                options: .storageModeShared
              ),
              let paramsBuffer = device.makeBuffer(
                bytes: &params,
                length: MemoryLayout<PocketGridParams>.stride,
                options: .storageModeShared
              ),
              let distanceBuffer = device.makeBuffer(
                length: totalPoints * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let candidateMaskBuffer = device.makeBuffer(
                length: totalPoints * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
              )
        else { return nil }

        let threadsPerThreadgroup = MTLSize(width: 128, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (totalPoints + 127) / 128, height: 1, depth: 1)

        guard let distanceCommandBuffer = commandQueue.makeCommandBuffer(),
              let distanceEncoder = distanceCommandBuffer.makeComputeCommandEncoder()
        else { return nil }

        distanceEncoder.setComputePipelineState(distancePipeline)
        distanceEncoder.setBuffer(distanceBuffer, offset: 0, index: 0)
        distanceEncoder.setBuffer(candidateMaskBuffer, offset: 0, index: 1)
        distanceEncoder.setBuffer(atomBuffer, offset: 0, index: 2)
        distanceEncoder.setBuffer(paramsBuffer, offset: 0, index: 3)
        distanceEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        distanceEncoder.endEncoding()
        distanceCommandBuffer.commit()
        distanceCommandBuffer.waitUntilCompleted()

        let candidateMask = candidateMaskBuffer.contents().bindMemory(to: UInt32.self, capacity: totalPoints)
        var probes: [PocketProbe] = []
        probes.reserveCapacity(max(totalPoints / 20, 256))

        for index in 0..<totalPoints where candidateMask[index] != 0 {
            probes.append(PocketProbe(
                position: Self.probePosition(for: index, origin: gridMin, spacing: spacing, nx: nx, ny: ny),
                buriedness: 0
            ))
        }

        guard !probes.isEmpty else { return [] }

        params.probeCount = UInt32(probes.count)
        paramsBuffer.contents().copyMemory(from: &params, byteCount: MemoryLayout<PocketGridParams>.stride)

        guard let probeBuffer = device.makeBuffer(
            bytes: &probes,
            length: probes.count * MemoryLayout<PocketProbe>.stride,
            options: .storageModeShared
        ),
        let buriednessCommandBuffer = commandQueue.makeCommandBuffer(),
        let buriednessEncoder = buriednessCommandBuffer.makeComputeCommandEncoder()
        else { return nil }

        let probeThreadgroups = MTLSize(width: (probes.count + 127) / 128, height: 1, depth: 1)
        buriednessEncoder.setComputePipelineState(buriednessPipeline)
        buriednessEncoder.setBuffer(probeBuffer, offset: 0, index: 0)
        buriednessEncoder.setBuffer(distanceBuffer, offset: 0, index: 1)
        buriednessEncoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        buriednessEncoder.dispatchThreadgroups(probeThreadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        buriednessEncoder.endEncoding()
        buriednessCommandBuffer.commit()
        buriednessCommandBuffer.waitUntilCompleted()

        let scoredProbes = probeBuffer.contents().bindMemory(to: PocketProbe.self, capacity: probes.count)
        var buried: [BuriedProbe] = []
        buried.reserveCapacity(probes.count / 3)

        for index in 0..<probes.count {
            let probe = scoredProbes[index]
            if probe.buriedness >= buriednessThreshold {
                buried.append(BuriedProbe(position: probe.position, buriedness: probe.buriedness))
            }
        }

        return buried
    }

    private static func probePosition(
        for linearIndex: Int,
        origin: SIMD3<Float>,
        spacing: Float,
        nx: Int,
        ny: Int
    ) -> SIMD3<Float> {
        let iz = linearIndex / (nx * ny)
        let iy = (linearIndex - iz * nx * ny) / nx
        let ix = linearIndex - iz * nx * ny - iy * nx
        return origin + SIMD3<Float>(Float(ix), Float(iy), Float(iz)) * spacing
    }
}

// MARK: - BindingSiteDetector

/// Geometric pocket detection using alpha-sphere probes + DBSCAN clustering.
///
/// ## Algorithm
/// 1. Grid probing at ~1 angstrom spacing within bounding box + 6 angstrom padding
/// 2. Spatial hash of protein atoms for O(1) neighbor queries
/// 3. Filter probes: keep those 2.0-5.5 angstroms from nearest atom surface
/// 4. Buriedness scoring: 26-direction ray casting, keep probes with buriedness > 0.4
/// 5. DBSCAN clustering (eps=3.0, minPts=5) of buried probes
/// 6. Pocket ranking by druggability = volume * buriedness * (1 + polarity)
/// 7. Search box from max probe extent + 4 angstrom padding
///
/// The public methods are `@MainActor` so they can read from `Molecule` directly.
/// All heavy computation uses a `ProteinSnapshot` value type and is nonisolated.
enum BindingSiteDetector {

    // MARK: - Public API

    /// Detect pockets geometrically using alpha-sphere probes and DBSCAN.
    @MainActor
    static func detectPockets(
        protein: Molecule,
        gridSpacing: Float = 1.5,
        excludedChainIDs: Set<String> = []
    ) -> [BindingPocket] {
        let snapshot = extractSnapshot(from: protein, excludedChainIDs: excludedChainIDs)
        return computePockets(
            snapshot: snapshot,
            gridSpacing: gridSpacing,
            accelerator: PocketDetectionMetalAccelerator.shared
        )
    }

    /// Find the pocket defined by a co-crystallized or docked ligand.
    /// Returns the pocket formed by protein residues within `distance` of any ligand atom.
    @MainActor
    static func ligandGuidedPocket(
        protein: Molecule,
        ligand: Molecule,
        distance: Float = 6.0,
        excludedChainIDs: Set<String> = []
    ) -> BindingPocket? {
        let snapshot = extractSnapshot(from: protein, excludedChainIDs: excludedChainIDs)
        let ligandPositions = ligand.atoms.map(\.position)
        guard !ligandPositions.isEmpty else { return nil }
        return computeLigandGuidedPocket(
            snapshot: snapshot,
            ligandPositions: ligandPositions,
            distance: distance
        )
    }

    /// Build a pocket from a manual residue selection.
    @MainActor
    static func pocketFromResidues(
        protein: Molecule,
        residueIndices: [Int],
        padding: Float = 4.0
    ) -> BindingPocket {
        let snapshot = extractSnapshot(from: protein)
        return computePocketFromResidues(
            snapshot: snapshot,
            residueIndices: residueIndices,
            padding: padding
        )
    }

    // MARK: - Snapshot Extraction (@MainActor)

    @MainActor
    private static func extractSnapshot(from protein: Molecule, excludedChainIDs: Set<String> = []) -> ProteinSnapshot {
        var positions: [SIMD3<Float>] = []
        var elements: [Element] = []
        var atomResidueIndex: [Int] = []
        var residueAtomIndices: [[Int]] = Array(repeating: [], count: protein.residues.count)
        let residueIsStandard: [Bool] = protein.residues.map(\.isStandard)

        positions.reserveCapacity(protein.atoms.count)
        elements.reserveCapacity(protein.atoms.count)
        atomResidueIndex.reserveCapacity(protein.atoms.count)

        for (resIdx, residue) in protein.residues.enumerated() {
            for atomIdx in residue.atomIndices {
                guard atomIdx < protein.atoms.count else { continue }
                let atom = protein.atoms[atomIdx]
                // Skip hydrogens
                guard atom.element != .H else { continue }
                // Skip excluded chains
                if !excludedChainIDs.isEmpty && excludedChainIDs.contains(atom.chainID) { continue }
                let snapshotIdx = positions.count
                positions.append(atom.position)
                elements.append(atom.element)
                atomResidueIndex.append(resIdx)
                residueAtomIndices[resIdx].append(snapshotIdx)
            }
        }

        return ProteinSnapshot(
            positions: positions,
            elements: elements,
            atomResidueIndex: atomResidueIndex,
            residueAtomIndices: residueAtomIndices,
            residueIsStandard: residueIsStandard
        )
    }

    // MARK: - Core Geometric Detection (nonisolated)

    private static func computePockets(
        snapshot snap: ProteinSnapshot,
        gridSpacing: Float,
        accelerator: PocketDetectionMetalAccelerator?
    ) -> [BindingPocket] {
        let positions = snap.positions
        guard positions.count >= 10 else { return [] }

        // ---- Step 1: Bounding box + 6 angstrom padding ----
        let (bbMin, bbMax) = boundingBoxOf(positions)
        let padding: Float = 6.0
        let gridMin = bbMin - SIMD3<Float>(repeating: padding)
        let gridMax = bbMax + SIMD3<Float>(repeating: padding)

        // ---- Step 2: Spatial hash (cell size = max probe distance) ----
        let maxProbeDist: Float = 5.5
        let minProbeDist: Float = 2.0
        let spatialHash = SpatialHash(positions: positions, cellSize: maxProbeDist)

        let gridExtent = gridMax - gridMin
        let nx = Int(ceilf(gridExtent.x / gridSpacing)) + 1
        let ny = Int(ceilf(gridExtent.y / gridSpacing)) + 1
        let nz = Int(ceilf(gridExtent.z / gridSpacing)) + 1
        let dims = SIMD3<UInt32>(UInt32(nx), UInt32(ny), UInt32(nz))
        let radii = snap.elements.map(\.vdwRadius)

        let buriednessThreshold: Float = 0.4
        let rayMaxDist: Float = 10.0
        let gpuBuriedProbes = accelerator?.findBuriedProbes(
            positions: positions,
            radii: radii,
            gridMin: gridMin,
            spacing: gridSpacing,
            dims: dims,
            minProbeDist: minProbeDist,
            maxProbeDist: maxProbeDist,
            buriednessThreshold: buriednessThreshold,
            rayMaxDist: rayMaxDist
        )
        let buriedProbes = (gpuBuriedProbes?.isEmpty == false ? gpuBuriedProbes : nil)
            ?? computeBuriedProbesCPU(
                snapshot: snap,
                gridMin: gridMin,
                gridMax: gridMax,
                gridSpacing: gridSpacing,
                spatialHash: spatialHash,
                minProbeDist: minProbeDist,
                maxProbeDist: maxProbeDist,
                buriednessThreshold: buriednessThreshold,
                rayMaxDist: rayMaxDist
            )

        guard !buriedProbes.isEmpty else { return [] }

        // ---- Step 5: DBSCAN clustering ----
        let epsilon: Float = 3.0
        let minPoints = 5
        let probePositions = buriedProbes.map(\.position)
        let probeBuriedness = buriedProbes.map(\.buriedness)

        let clusterLabels = dbscanCluster(
            points: probePositions,
            eps: epsilon,
            minPts: minPoints
        )

        // Group probes by cluster label (label >= 0 means assigned to a cluster)
        var clusterMap: [Int: [Int]] = [:]
        for (i, label) in clusterLabels.enumerated() {
            guard label >= 0 else { continue }
            clusterMap[label, default: []].append(i)
        }

        // ---- Step 6: Build and rank pockets ----
        let contactDistance: Float = 4.5
        var pockets: [BindingPocket] = []
        pockets.reserveCapacity(clusterMap.count)

        for (_, memberIndices) in clusterMap {
            guard memberIndices.count >= minPoints else { continue }

            let clusterPositions = memberIndices.map { probePositions[$0] }
            let clusterBuriedness = memberIndices.map { probeBuriedness[$0] }

            // Centroid
            let center = centroidOf(clusterPositions)

            // Volume = count * gridSpacing^3
            let volume = Float(clusterPositions.count) * powf(gridSpacing, 3)

            // Filter micro-pockets: minimum 100 Å³ (small fragment) to 200 Å³ (drug-like)
            guard volume >= 100.0 else { continue }

            // Mean buriedness
            let avgBuriedness = clusterBuriedness.reduce(0, +) / Float(clusterBuriedness.count)

            // Find protein atoms within contactDistance of any cluster probe
            let pocketExtent = halfExtentsOf(clusterPositions, center: center)
            let searchRadius = max(pocketExtent.x, max(pocketExtent.y, pocketExtent.z)) + contactDistance
            let candidateAtoms = spatialHash.indicesWithin(
                radius: searchRadius, of: center, positions: positions
            )

            // Refine: only keep atoms actually within contactDistance of a cluster probe
            var pocketAtomSet = Set<Int>()
            let contactDistSq = contactDistance * contactDistance
            for atomIdx in candidateAtoms {
                let atomPos = positions[atomIdx]
                for probePos in clusterPositions {
                    if simd_length_squared(atomPos - probePos) <= contactDistSq {
                        pocketAtomSet.insert(atomIdx)
                        break
                    }
                }
            }

            // Polarity and hydrophobicity analysis
            var polarCount = 0
            var hydrophobicCount = 0
            for atomIdx in pocketAtomSet {
                let elem = snap.elements[atomIdx]
                if elem == .N || elem == .O { polarCount += 1 }
                if elem == .C || elem == .S { hydrophobicCount += 1 }
            }
            let totalPocketAtoms = max(Float(pocketAtomSet.count), 1)
            let polarity = Float(polarCount) / totalPocketAtoms
            let hydrophobicity = Float(hydrophobicCount) / totalPocketAtoms

            // Druggability score (fpocket-inspired multi-factor):
            //   - Normalized volume (log scale, caps extreme sizes)
            //   - Buriedness (strongly enclosing pockets rank higher)
            //   - Hydrophobic-polar balance (ideal ~60-70% hydrophobic for drug binding)
            //   - Penalty for very small or very large pockets
            let logVolume = log(max(volume, 1.0))  // normalize extreme volumes
            let hpBalance = 1.0 - abs(hydrophobicity - 0.65) * 2.0  // peak at 65% hydrophobic
            let sizeScore: Float = volume >= 300 && volume <= 2000 ? 1.0 : 0.5  // sweet spot
            let druggability = logVolume * avgBuriedness * (1.0 + Float(hpBalance)) * sizeScore

            // Residue indices: unique standard residues contributing atoms to the pocket
            var residueSet = Set<Int>()
            for atomIdx in pocketAtomSet {
                let resIdx = snap.atomResidueIndex[atomIdx]
                if resIdx >= 0 && resIdx < snap.residueIsStandard.count
                    && snap.residueIsStandard[resIdx]
                {
                    residueSet.insert(resIdx)
                }
            }

            // ---- Step 7: Search box = max extent + 4 angstrom padding ----
            let boxPadding: Float = 4.0
            let halfSize = pocketExtent + SIMD3<Float>(repeating: boxPadding)

            pockets.append(BindingPocket(
                id: 0, // temporary, reassigned after sorting
                center: center,
                size: halfSize,
                volume: volume,
                buriedness: avgBuriedness,
                polarity: polarity,
                druggability: druggability,
                residueIndices: Array(residueSet).sorted(),
                probePositions: clusterPositions
            ))
        }

        // Sort by druggability descending and assign sequential IDs
        pockets.sort { $0.druggability > $1.druggability }
        for i in pockets.indices {
            pockets[i] = BindingPocket(
                id: i,
                center: pockets[i].center,
                size: pockets[i].size,
                volume: pockets[i].volume,
                buriedness: pockets[i].buriedness,
                polarity: pockets[i].polarity,
                druggability: pockets[i].druggability,
                residueIndices: pockets[i].residueIndices,
                probePositions: pockets[i].probePositions
            )
        }

        return pockets
    }

    private static func computeBuriedProbesCPU(
        snapshot snap: ProteinSnapshot,
        gridMin: SIMD3<Float>,
        gridMax: SIMD3<Float>,
        gridSpacing: Float,
        spatialHash: SpatialHash,
        minProbeDist: Float,
        maxProbeDist: Float,
        buriednessThreshold: Float,
        rayMaxDist: Float
    ) -> [BuriedProbe] {
        let positions = snap.positions
        let gridExtent = gridMax - gridMin
        let nx = Int(ceilf(gridExtent.x / gridSpacing)) + 1
        let ny = Int(ceilf(gridExtent.y / gridSpacing)) + 1
        let nz = Int(ceilf(gridExtent.z / gridSpacing)) + 1

        let sliceLock = NSLock()
        nonisolated(unsafe) var allFilteredPositions: [SIMD3<Float>] = []
        allFilteredPositions.reserveCapacity(max(nx * ny * nz / 20, 256))

        DispatchQueue.concurrentPerform(iterations: nx) { ix in
            let gx = gridMin.x + Float(ix) * gridSpacing
            var sliceProbes: [SIMD3<Float>] = []
            sliceProbes.reserveCapacity(ny * nz / 10)

            for iy in 0..<ny {
                let gy = gridMin.y + Float(iy) * gridSpacing
                for iz in 0..<nz {
                    let gz = gridMin.z + Float(iz) * gridSpacing
                    let probe = SIMD3<Float>(gx, gy, gz)
                    let (nearDistSq, nearIdx) = spatialHash.nearestDistanceSquared(
                        to: probe, positions: positions
                    )
                    if nearIdx >= 0 {
                        let centerDist = sqrtf(nearDistSq)
                        let vdw = snap.elements[nearIdx].vdwRadius
                        let surfDist = centerDist - vdw
                        if surfDist >= minProbeDist && surfDist <= maxProbeDist {
                            sliceProbes.append(probe)
                        }
                    }
                }
            }

            if !sliceProbes.isEmpty {
                sliceLock.lock()
                allFilteredPositions.append(contentsOf: sliceProbes)
                sliceLock.unlock()
            }
        }

        guard !allFilteredPositions.isEmpty else { return [] }

        let probeCount = allFilteredPositions.count
        let buriedLock = NSLock()
        nonisolated(unsafe) var buriedProbes: [BuriedProbe] = []
        buriedProbes.reserveCapacity(probeCount / 3)

        DispatchQueue.concurrentPerform(iterations: probeCount) { i in
            let b = buriednessAt(
                point: allFilteredPositions[i],
                positions: positions,
                elements: snap.elements,
                spatialHash: spatialHash,
                maxRayDist: rayMaxDist
            )
            if b >= buriednessThreshold {
                let probe = BuriedProbe(position: allFilteredPositions[i], buriedness: b)
                buriedLock.lock()
                buriedProbes.append(probe)
                buriedLock.unlock()
            }
        }

        return buriedProbes
    }

    // MARK: - Ligand-Guided Pocket

    private static func computeLigandGuidedPocket(
        snapshot: ProteinSnapshot,
        ligandPositions: [SIMD3<Float>],
        distance: Float
    ) -> BindingPocket? {
        let positions = snapshot.positions
        guard !positions.isEmpty, !ligandPositions.isEmpty else { return nil }

        let spatialHash = SpatialHash(positions: positions, cellSize: distance)

        // Find all protein atoms within distance of any ligand atom
        var pocketAtomSet = Set<Int>()
        for ligPos in ligandPositions {
            let nearby = spatialHash.indicesWithin(
                radius: distance, of: ligPos, positions: positions
            )
            for idx in nearby { pocketAtomSet.insert(idx) }
        }

        guard !pocketAtomSet.isEmpty else { return nil }

        // Identify residues
        var residueSet = Set<Int>()
        for atomIdx in pocketAtomSet {
            let resIdx = snapshot.atomResidueIndex[atomIdx]
            if resIdx >= 0 && resIdx < snapshot.residueIsStandard.count
                && snapshot.residueIsStandard[resIdx]
            {
                residueSet.insert(resIdx)
            }
        }

        let center = centroidOf(ligandPositions)
        let localPocketPositions = pocketAtomSet.compactMap { idx -> SIMD3<Float>? in
            guard idx < positions.count else { return nil }
            return positions[idx]
        }
        let extentSource = (localPocketPositions.isEmpty ? ligandPositions : localPocketPositions) + ligandPositions
        let extent = halfExtentsOf(extentSource, center: center)
        let boxPadding: Float = 4.0
        // Clamp half-extents: allow large pockets (up to 25 Å half-width = 50 Å across)
        // The docking engine applies its own adaptive grid sizing on top of this.
        let clampedExtent = simd_min(extent, SIMD3<Float>(repeating: 25.0))
        let halfSize = clampedExtent + SIMD3<Float>(repeating: boxPadding)

        // Buriedness at the ligand centroid
        let buriedness = buriednessAt(
            point: center,
            positions: positions,
            elements: snapshot.elements,
            spatialHash: spatialHash,
            maxRayDist: 10.0
        )

        // Polarity
        var polarCount = 0
        for atomIdx in pocketAtomSet {
            let elem = snapshot.elements[atomIdx]
            if elem == .N || elem == .O { polarCount += 1 }
        }
        let polarity = Float(polarCount) / Float(pocketAtomSet.count)

        let volume = 8.0 * halfSize.x * halfSize.y * halfSize.z
        let druggability = volume * buriedness * (1.0 + polarity)

        return BindingPocket(
            id: 0,
            center: center,
            size: halfSize,
            volume: volume,
            buriedness: buriedness,
            polarity: polarity,
            druggability: druggability,
            residueIndices: Array(residueSet).sorted(),
            probePositions: ligandPositions
        )
    }

    // MARK: - Manual Residue Selection

    private static func computePocketFromResidues(
        snapshot: ProteinSnapshot,
        residueIndices: [Int],
        padding: Float
    ) -> BindingPocket {
        let positions = snapshot.positions

        // Gather all atom positions for selected residues
        var residuePositions: [SIMD3<Float>] = []
        for resIdx in residueIndices {
            guard resIdx < snapshot.residueAtomIndices.count else { continue }
            for atomIdx in snapshot.residueAtomIndices[resIdx] {
                guard atomIdx < positions.count else { continue }
                residuePositions.append(positions[atomIdx])
            }
        }

        guard !residuePositions.isEmpty else {
            return BindingPocket(
                id: 0, center: .zero, size: SIMD3<Float>(repeating: padding),
                volume: 0, buriedness: 0, polarity: 0, druggability: 0,
                residueIndices: residueIndices, probePositions: []
            )
        }

        let center = centroidOf(residuePositions)
        let extent = halfExtentsOf(residuePositions, center: center)
        let halfSize = extent + SIMD3<Float>(repeating: padding)

        // Polarity from selected residue atoms
        var polarCount = 0
        var totalCount = 0
        for resIdx in residueIndices {
            guard resIdx < snapshot.residueAtomIndices.count else { continue }
            for atomIdx in snapshot.residueAtomIndices[resIdx] {
                guard atomIdx < positions.count else { continue }
                totalCount += 1
                let elem = snapshot.elements[atomIdx]
                if elem == .N || elem == .O { polarCount += 1 }
            }
        }
        let polarity: Float = totalCount > 0 ? Float(polarCount) / Float(totalCount) : 0

        // Buriedness at center
        let spatialHash = SpatialHash(positions: positions, cellSize: 5.5)
        let buriedness = buriednessAt(
            point: center,
            positions: positions,
            elements: snapshot.elements,
            spatialHash: spatialHash,
            maxRayDist: 10.0
        )

        let volume = 8.0 * halfSize.x * halfSize.y * halfSize.z
        let druggability = volume * buriedness * (1.0 + polarity)

        return BindingPocket(
            id: 0,
            center: center,
            size: halfSize,
            volume: volume,
            buriedness: buriedness,
            polarity: polarity,
            druggability: druggability,
            residueIndices: residueIndices,
            probePositions: residuePositions
        )
    }

    // MARK: - Buriedness (26-direction ray casting)

    /// Cast 26 rays from `point`. A ray "hits" if it encounters the VdW sphere of a
    /// protein atom within `maxRayDist`. Buriedness = fraction of rays that hit.
    private static func buriednessAt(
        point: SIMD3<Float>,
        positions: [SIMD3<Float>],
        elements: [Element],
        spatialHash: SpatialHash,
        maxRayDist: Float
    ) -> Float {
        var blockedCount = 0

        for dir in buriednessRayDirections {
            if rayCastsHit(
                origin: point,
                direction: dir,
                maxDist: maxRayDist,
                positions: positions,
                elements: elements,
                spatialHash: spatialHash
            ) {
                blockedCount += 1
            }
        }

        return Float(blockedCount) / Float(buriednessRayDirections.count)
    }

    /// March along a ray in 2 angstrom steps. At each step, check if the nearest atom's
    /// VdW sphere contains the sample point (i.e., the ray passes through the atom).
    private static func rayCastsHit(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        maxDist: Float,
        positions: [SIMD3<Float>],
        elements: [Element],
        spatialHash: SpatialHash
    ) -> Bool {
        let stepSize: Float = 2.0
        let steps = Int(ceilf(maxDist / stepSize))

        for step in 0 ... steps {
            let t = min(Float(step) * stepSize, maxDist)
            let samplePoint = origin + direction * t

            let (nearDistSq, nearIdx) = spatialHash.nearestDistanceSquared(
                to: samplePoint, positions: positions
            )
            if nearIdx >= 0 {
                let vdw = elements[nearIdx].vdwRadius
                if nearDistSq <= vdw * vdw {
                    return true
                }
            }
        }

        return false
    }

    // MARK: - DBSCAN Clustering

    /// Standard DBSCAN. Returns cluster labels parallel to `points`:
    /// label >= 0 is a cluster ID, label == -1 is noise.
    private static func dbscanCluster(
        points: [SIMD3<Float>],
        eps: Float,
        minPts: Int
    ) -> [Int] {
        let n = points.count
        guard n > 0 else { return [] }

        // Build a spatial hash of probe points for fast neighbor queries during clustering
        let probeHash = SpatialHash(positions: points, cellSize: eps)

        // Precompute neighborhoods to avoid redundant work in seed expansion
        var neighborhoods: [[Int]] = Array(repeating: [], count: n)
        for i in 0 ..< n {
            neighborhoods[i] = probeHash.indicesWithin(radius: eps, of: points[i], positions: points)
        }

        var labels = [Int](repeating: -2, count: n) // -2 = unvisited, -1 = noise
        var clusterId = 0

        for i in 0 ..< n {
            guard labels[i] == -2 else { continue }

            let neighbors = neighborhoods[i]
            if neighbors.count < minPts {
                labels[i] = -1 // noise
                continue
            }

            // Expand new cluster
            let currentCluster = clusterId
            clusterId += 1
            labels[i] = currentCluster

            var queue = neighbors
            var queueIdx = 0

            while queueIdx < queue.count {
                let j = queue[queueIdx]
                queueIdx += 1

                if labels[j] == -1 {
                    labels[j] = currentCluster // noise -> border point
                }
                guard labels[j] == -2 else { continue } // already processed

                labels[j] = currentCluster

                let jNeighbors = neighborhoods[j]
                if jNeighbors.count >= minPts {
                    queue.append(contentsOf: jNeighbors)
                }
            }
        }

        return labels
    }

    // MARK: - Geometry Helpers

    private static func boundingBoxOf(
        _ positions: [SIMD3<Float>]
    ) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard let first = positions.first else { return (.zero, .zero) }
        var lo = first
        var hi = first
        for p in positions {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        return (lo, hi)
    }

    private static func centroidOf(_ points: [SIMD3<Float>]) -> SIMD3<Float> {
        guard !points.isEmpty else { return .zero }
        var sum = SIMD3<Float>.zero
        for p in points { sum += p }
        return sum / Float(points.count)
    }

    /// Half-extents: maximum absolute distance from center in each axis direction.
    private static func halfExtentsOf(
        _ points: [SIMD3<Float>],
        center: SIMD3<Float>
    ) -> SIMD3<Float> {
        var maxExt = SIMD3<Float>.zero
        for p in points {
            let diff = simd_abs(p - center)
            maxExt = simd_max(maxExt, diff)
        }
        return maxExt
    }
}
