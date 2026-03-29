import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - DockingEngine Analysis (result extraction, clustering, diagnostics, interactions)

@MainActor
extension DockingEngine {

    // MARK: - Pose Data Extraction Helpers

    func torsions(from pose: DockPose) -> [Float] {
        let count = max(0, min(Int(pose.numTorsions), 32))
        guard count > 0 else { return [] }
        return withUnsafePointer(to: pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                Array(UnsafeBufferPointer(start: buffer, count: count))
            }
        }
    }

    func chiAngles(from pose: DockPose) -> [Float] {
        let count = max(0, min(Int(pose.numChiAngles), 24))
        guard count > 0 else { return [] }
        return withUnsafePointer(to: pose.chiAngles) {
            $0.withMemoryRebound(to: Float.self, capacity: 24) { buffer in
                Array(UnsafeBufferPointer(start: buffer, count: count))
            }
        }
    }

    // MARK: - Result Extraction

    /// Apply rigid-body + torsion transform to recover docked atom positions.
    /// Uses the exact same formulas as the GPU kernels for bit-exact consistency:
    ///   - Rigid body: quatRotate() formula from DockingCompute.metal line 75-79
    ///   - Torsions: Rodrigues rotation from DockingCompute.metal line 114-124
    func applyPoseTransform(_ pose: DockPose, ligandAtoms: [Atom], centroid: SIMD3<Float>) -> [SIMD3<Float>] {
        let q = SIMD4<Float>(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)
        let u = SIMD3<Float>(q.x, q.y, q.z)
        let s = q.w
        let trans = SIMD3<Float>(pose.translation.x, pose.translation.y, pose.translation.z)

        var positions = ligandAtoms.map { atom -> SIMD3<Float> in
            let v = atom.position - centroid
            let rotated = 2.0 * simd_dot(u, v) * u + (s * s - simd_dot(u, u)) * v + 2.0 * s * simd_cross(u, v)
            return rotated + trans
        }

        if let edgeBuf = torsionEdgeBuffer, let idxBuf = movingIndicesBuffer {
            let edges = edgeBuf.contents().bindMemory(to: TorsionEdge.self, capacity: Int(pose.numTorsions))
            let moving = idxBuf.contents().bindMemory(to: Int32.self, capacity: idxBuf.length / MemoryLayout<Int32>.stride)

            for t in 0..<Int(pose.numTorsions) {
                let angle = withUnsafePointer(to: pose.torsions) {
                    $0.withMemoryRebound(to: Float.self, capacity: 32) { $0[t] }
                }
                if abs(angle) < 1e-6 { continue }

                let edge = edges[t]
                let pivotIdx = Int(edge.atom1)
                let axisIdx = Int(edge.atom2)
                guard pivotIdx < positions.count, axisIdx < positions.count else { continue }

                let pivot = positions[pivotIdx]
                let axis = simd_normalize(positions[axisIdx] - pivot)
                let cosA = cos(angle)
                let sinA = sin(angle)

                for i in 0..<Int(edge.movingCount) {
                    let atomIdx = Int(moving[Int(edge.movingStart) + i])
                    guard atomIdx >= 0, atomIdx < positions.count else { continue }
                    let v = positions[atomIdx] - pivot
                    let rotated = v * cosA + simd_cross(axis, v) * sinA + axis * simd_dot(axis, v) * (1.0 - cosA)
                    positions[atomIdx] = pivot + rotated
                }
            }
        }

        return positions
    }

    func extractBestPose(from buffer: MTLBuffer? = nil, ligandAtoms: [Atom], centroid: SIMD3<Float>, maxPoses: Int? = nil, scoringMethod: ScoringMethod = .vina) -> DockingResult? {
        guard let buffer = buffer ?? populationBuffer else { return nil }
        let poseCount = min(maxPoses ?? Int.max, buffer.length / MemoryLayout<DockPose>.stride)
        guard poseCount > 0 else { return nil }
        let poses = buffer.contents().bindMemory(to: DockPose.self, capacity: poseCount)

        var bestIdx = -1
        var bestE: Float = .infinity
        for i in 0..<poseCount {
            let e = poses[i].energy
            guard e.isFinite, e < 1e9 else { continue }
            if e < bestE { bestE = e; bestIdx = i }
        }
        guard bestIdx >= 0 else { return nil }

        let p = poses[bestIdx]
        let quat = simd_quatf(ix: p.rotation.x, iy: p.rotation.y, iz: p.rotation.z, r: p.rotation.w)
        let trans = SIMD3<Float>(p.translation.x, p.translation.y, p.translation.z)

        let transformed = applyPoseTransform(p, ligandAtoms: ligandAtoms, centroid: centroid)

        var result = DockingResult(
            id: bestIdx,
            pose: DockPoseSwift(translation: trans, rotation: quat, torsions: torsions(from: p), chiAngles: chiAngles(from: p)),
            energy: p.energy,
            stericEnergy: p.stericEnergy,
            hydrophobicEnergy: p.hydrophobicEnergy,
            hbondEnergy: p.hbondEnergy,
            torsionPenalty: p.torsionPenalty,
            generation: Int(p.generation),
            transformedAtomPositions: transformed
        )
        result.drusinaCorrection = p.drusinaCorrection
        result.constraintPenalty = p.constraintPenalty

        return result
    }

    func extractAllResults(
        from buffer: MTLBuffer? = nil,
        ligandAtoms: [Atom],
        centroid: SIMD3<Float>,
        idOffset: Int = 0,
        sortByEnergy: Bool = true,
        maxPoses: Int? = nil,
        scoringMethod: ScoringMethod = .vina
    ) -> [DockingResult] {
        guard let buffer = buffer ?? populationBuffer else { return [] }
        let poseCount = min(maxPoses ?? Int.max, buffer.length / MemoryLayout<DockPose>.stride)
        guard poseCount > 0 else { return [] }
        let poses = buffer.contents().bindMemory(to: DockPose.self, capacity: poseCount)

        var results: [DockingResult] = []
        results.reserveCapacity(poseCount)

        for i in 0..<poseCount {
            let p = poses[i]
            guard p.energy.isFinite, p.energy < 1e9 else { continue }

            let quat = simd_quatf(ix: p.rotation.x, iy: p.rotation.y, iz: p.rotation.z, r: p.rotation.w)
            let trans = SIMD3<Float>(p.translation.x, p.translation.y, p.translation.z)
            let transformed = applyPoseTransform(p, ligandAtoms: ligandAtoms, centroid: centroid)

            guard transformed.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { continue }

            var r = DockingResult(
                id: results.count + idOffset,
                pose: DockPoseSwift(translation: trans, rotation: quat, torsions: torsions(from: p), chiAngles: chiAngles(from: p)),
                energy: p.energy,
                stericEnergy: p.stericEnergy,
                hydrophobicEnergy: p.hydrophobicEnergy,
                hbondEnergy: p.hbondEnergy,
                torsionPenalty: p.torsionPenalty,
                generation: Int(p.generation),
                transformedAtomPositions: transformed
            )
            r.drusinaCorrection = p.drusinaCorrection
            r.constraintPenalty = p.constraintPenalty

            results.append(r)
        }
        return sortByEnergy ? results.sorted { $0.energy < $1.energy } : results
    }

    // MARK: - RMSD Clustering

    func clusterPoses(_ results: [DockingResult]) async -> [DockingResult] {
        guard !results.isEmpty else { return [] }
        let threshold: Float = 2.0
        var out = results.sorted { $0.energy < $1.energy }
        let n = out.count

        for i in 0..<n {
            out[i].clusterID = -1
            out[i].clusterRank = 0
        }

        let rmsdMatrix = await computeRMSDMatrixGPU(out) ?? computeRMSDMatrixCPU(out)

        var clusterID = 0
        for i in 0..<n {
            guard out[i].clusterID == -1 else { continue }
            out[i].clusterID = clusterID
            out[i].clusterRank = 0
            var rank = 1
            for j in (i+1)..<n {
                guard out[j].clusterID == -1 else { continue }
                let idx = i * n - i * (i + 1) / 2 + j - i - 1
                if rmsdMatrix[idx] < threshold {
                    out[j].clusterID = clusterID
                    out[j].clusterRank = rank
                    rank += 1
                }
            }
            clusterID += 1
        }
        return out
    }

    private func rmsd(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        let s = zip(a, b).reduce(Float(0)) { $0 + simd_distance_squared($1.0, $1.1) }
        return sqrt(s / Float(a.count))
    }

    private func computeRMSDMatrixGPU(_ results: [DockingResult]) async -> [Float]? {
        guard let pipeline = pairwiseRMSDPipeline,
              let first = results.first,
              !first.transformedAtomPositions.isEmpty else { return nil }

        let n = results.count
        let numAtoms = first.transformedAtomPositions.count
        guard n > 1 else { return [] }

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(n * numAtoms)
        for r in results {
            guard r.transformedAtomPositions.count == numAtoms else { return nil }
            positions.append(contentsOf: r.transformedAtomPositions)
        }

        let matrixSize = n * (n - 1) / 2
        var params = RMSDParams(numPoses: UInt32(n), numAtoms: UInt32(numAtoms), _pad0: 0, _pad1: 0)

        guard let posBuffer = device.makeBuffer(
                    bytes: &positions,
                    length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
                    options: .storageModeShared),
              let matrixBuffer = device.makeBuffer(
                    length: matrixSize * MemoryLayout<Float>.stride,
                    options: .storageModeShared),
              let paramsBuffer = device.makeBuffer(
                    bytes: &params,
                    length: MemoryLayout<RMSDParams>.stride,
                    options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else { return nil }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(posBuffer, offset: 0, index: 0)
        enc.setBuffer(matrixBuffer, offset: 0, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)

        let totalPairs = n * (n - 1) / 2
        let threadWidth = 256
        let tgSize = MTLSize(width: threadWidth, height: 1, depth: 1)
        let tgCount = MTLSize(width: (totalPairs + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        await withCheckedContinuation { continuation in
            cmdBuf.addCompletedHandler { _ in
                continuation.resume()
            }
            cmdBuf.commit()
        }

        let ptr = matrixBuffer.contents().bindMemory(to: Float.self, capacity: matrixSize)
        return Array(UnsafeBufferPointer(start: ptr, count: matrixSize))
    }

    /// CPU fallback for pairwise RMSD matrix (parallelized with concurrentPerform).
    private func computeRMSDMatrixCPU(_ results: [DockingResult]) -> [Float] {
        let n = results.count
        let matrixSize = n * (n - 1) / 2
        var matrix = [Float](repeating: 0, count: matrixSize)
        // Parallel computation for large result sets
        if n > 50 {
            // Parallel RMSD computation using raw pointer to avoid inout capture issues
            let storage = UnsafeMutablePointer<Float>.allocate(capacity: matrixSize)
            storage.initialize(repeating: 0, count: matrixSize)
            nonisolated(unsafe) let base = storage
            DispatchQueue.concurrentPerform(iterations: n) { i in
                for j in (i+1)..<n {
                    let idx = i * n - i * (i + 1) / 2 + j - i - 1
                    let posA = results[i].transformedAtomPositions
                    let posB = results[j].transformedAtomPositions
                    guard posA.count == posB.count, !posA.isEmpty else {
                        base[idx] = .infinity
                        continue
                    }
                    let s = zip(posA, posB).reduce(Float(0)) { $0 + simd_distance_squared($1.0, $1.1) }
                    base[idx] = sqrt(s / Float(posA.count))
                }
            }
            for i in 0..<matrixSize { matrix[i] = storage[i] }
            storage.deallocate()
        } else {
            for i in 0..<n {
                for j in (i+1)..<n {
                    let idx = i * n - i * (i + 1) / 2 + j - i - 1
                    matrix[idx] = rmsd(results[i].transformedAtomPositions, results[j].transformedAtomPositions)
                }
            }
        }
        return matrix
    }

    // MARK: - Docking Diagnostics

    func computeDiagnostics(
        results: [DockingResult],
        ligandAtoms: [Atom],
        heavyBonds: [Bond]
    ) -> DockingDiagnostics {
        let gridOrigin = SIMD3<Float>(gridParams.origin.x, gridParams.origin.y, gridParams.origin.z)
        let gridEnd = gridOrigin + SIMD3<Float>(Float(gridParams.dims.x), Float(gridParams.dims.y), Float(gridParams.dims.z)) * gridParams.spacing
        let gridCenter = (gridOrigin + gridEnd) * 0.5

        let validResults = results.filter { $0.energy.isFinite && $0.energy < 1e9 }
        let energies = validResults.map(\.energy)

        let minE = energies.min() ?? .infinity
        let maxE = energies.max() ?? -.infinity
        let meanE = energies.isEmpty ? 0 : energies.reduce(0, +) / Float(energies.count)
        let variance = energies.isEmpty ? 0 : energies.map { ($0 - meanE) * ($0 - meanE) }.reduce(0, +) / Float(energies.count)
        let stddevE = sqrt(variance)

        var insideGrid = 0
        var outsideGrid = 0
        var centroidDistances: [Float] = []
        var minProteinDistances: [Float] = []

        for r in validResults {
            let positions = r.transformedAtomPositions
            let allInside = positions.allSatisfy { p in
                p.x >= gridOrigin.x && p.x <= gridEnd.x &&
                p.y >= gridOrigin.y && p.y <= gridEnd.y &&
                p.z >= gridOrigin.z && p.z <= gridEnd.z
            }
            if allInside { insideGrid += 1 } else { outsideGrid += 1 }

            if !positions.isEmpty {
                let centroid = positions.reduce(.zero, +) / Float(positions.count)
                centroidDistances.append(simd_distance(centroid, gridCenter))

                if !proteinAtoms.isEmpty {
                    var minDist: Float = .infinity
                    for lp in positions {
                        for pa in proteinAtoms {
                            let d = simd_distance(lp, pa.position)
                            if d < minDist { minDist = d }
                        }
                    }
                    minProteinDistances.append(minDist)
                }
            }
        }

        let meanCentroidDist = centroidDistances.isEmpty ? 0 : centroidDistances.reduce(0, +) / Float(centroidDistances.count)
        let meanProteinDist = minProteinDistances.isEmpty ? 0 : minProteinDistances.reduce(0, +) / Float(minProteinDistances.count)

        let contactPoses = minProteinDistances.filter { $0 < 4.0 }.count

        var bondDeviations: [Float] = []
        for r in validResults.prefix(10) {
            let positions = r.transformedAtomPositions
            for b in heavyBonds {
                guard b.atomIndex1 < ligandAtoms.count, b.atomIndex2 < ligandAtoms.count,
                      b.atomIndex1 < positions.count, b.atomIndex2 < positions.count else { continue }
                let orig = simd_distance(ligandAtoms[b.atomIndex1].position, ligandAtoms[b.atomIndex2].position)
                guard orig > 0.01 else { continue }
                let docked = simd_distance(positions[b.atomIndex1], positions[b.atomIndex2])
                guard docked.isFinite else { continue }
                bondDeviations.append(abs(docked - orig))
            }
        }
        let meanBondDev = bondDeviations.isEmpty ? 0 : bondDeviations.reduce(0, +) / Float(bondDeviations.count)
        let maxBondDev = bondDeviations.max() ?? 0

        let clusterCount = Set(validResults.map(\.clusterID)).count

        let centroidSpread: Float
        if centroidDistances.count >= 2 {
            let meanCD = centroidDistances.reduce(0, +) / Float(centroidDistances.count)
            let cdVar = centroidDistances.map { ($0 - meanCD) * ($0 - meanCD) }.reduce(0, +) / Float(centroidDistances.count)
            centroidSpread = sqrt(cdVar)
        } else {
            centroidSpread = 0
        }

        return DockingDiagnostics(
            totalPopulation: results.count,
            validPoses: validResults.count,
            invalidPoses: results.count - validResults.count,
            posesInsideGrid: insideGrid,
            posesOutsideGrid: outsideGrid,
            posesWithProteinContact: contactPoses,
            meanLigandProteinDistance: meanProteinDist,
            meanCentroidToGridCenter: meanCentroidDist,
            centroidSpread: centroidSpread,
            minEnergy: minE,
            maxEnergy: maxE,
            meanEnergy: meanE,
            energyStdDev: stddevE,
            clusterCount: clusterCount,
            meanBondLengthDeviation: meanBondDev,
            maxBondLengthDeviation: maxBondDev,
            gridDimensions: SIMD3(Float(gridParams.dims.x), Float(gridParams.dims.y), Float(gridParams.dims.z)),
            gridSpacing: gridParams.spacing,
            gridBoxSize: gridEnd - gridOrigin
        )
    }
}

// MARK: - Interaction Detection

enum InteractionDetector {

    // MARK: - Aromatic Ring Detection

    struct AromaticRing: Sendable {
        let centroid: SIMD3<Float>
        let normal: SIMD3<Float>
        let atomIndices: [Int]
    }

    private static let aromaticResidueAtoms: [String: Set<String>] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    ]

    static func detectAromaticRings(
        atoms: [Atom],
        positions: [SIMD3<Float>]? = nil,
        bonds: [Bond]? = nil
    ) -> [AromaticRing] {
        var rings: [AromaticRing] = []

        var residueAtoms: [String: [(Int, Atom)]] = [:]
        for (i, atom) in atoms.enumerated() {
            let key = "\(atom.chainID)_\(atom.residueSeq)_\(atom.residueName)"
            residueAtoms[key, default: []].append((i, atom))
        }

        for (_, atomList) in residueAtoms {
            guard let resName = atomList.first?.1.residueName,
                  let targetNames = aromaticResidueAtoms[resName] else { continue }

            let ringAtoms = atomList.filter { targetNames.contains($0.1.name.trimmingCharacters(in: .whitespaces)) }
            guard ringAtoms.count >= 5 else { continue }

            if let ring = buildRing(indices: ringAtoms.map(\.0), atoms: atoms, positions: positions) {
                rings.append(ring)
            }
        }

        if let bonds = bonds {
            let graphRings = detectRingsFromBonds(atoms: atoms, bonds: bonds, positions: positions)
            rings.append(contentsOf: graphRings)
        }

        return rings
    }

    private static func buildRing(
        indices: [Int], atoms: [Atom], positions: [SIMD3<Float>]?
    ) -> AromaticRing? {
        let ringPositions = indices.map { positions?[$0] ?? atoms[$0].position }
        let centroid = ringPositions.reduce(.zero, +) / Float(ringPositions.count)

        var normal = SIMD3<Float>(0, 1, 0)
        if ringPositions.count >= 3 {
            let v1 = ringPositions[1] - ringPositions[0]
            let v2 = ringPositions[2] - ringPositions[0]
            let n = simd_cross(v1, v2)
            let len = simd_length(n)
            if len > 1e-6 { normal = n / len }
        }
        return AromaticRing(centroid: centroid, normal: normal, atomIndices: indices)
    }

    private static func detectRingsFromBonds(
        atoms: [Atom], bonds: [Bond], positions: [SIMD3<Float>]?
    ) -> [AromaticRing] {
        let n = atoms.count
        guard n > 4 else { return [] }

        let aromaticElements: Set<Element> = [.C, .N, .O, .S]
        var adj: [[Int]] = Array(repeating: [], count: n)
        for bond in bonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < n, b < n else { continue }
            guard aromaticElements.contains(atoms[a].element),
                  aromaticElements.contains(atoms[b].element) else { continue }
            adj[a].append(b)
            adj[b].append(a)
        }

        var foundRings: Set<[Int]> = []

        for start in 0..<n {
            guard aromaticElements.contains(atoms[start].element) else { continue }
            findCycles(start: start, adj: adj, maxLen: 6, found: &foundRings)
        }

        var result: [AromaticRing] = []
        for ringIndices in foundRings {
            let ringPos = ringIndices.map { positions?[$0] ?? atoms[$0].position }

            guard ringPos.count >= 5 else { continue }
            let v1 = ringPos[1] - ringPos[0]
            let v2 = ringPos[2] - ringPos[0]
            let normal = simd_cross(v1, v2)
            let normalLen = simd_length(normal)
            guard normalLen > 1e-6 else { continue }
            let n = normal / normalLen
            let centroid = ringPos.reduce(.zero, +) / Float(ringPos.count)

            var planar = true
            for p in ringPos {
                let dist = abs(simd_dot(p - centroid, n))
                if dist > 0.5 { planar = false; break }
            }
            guard planar else { continue }

            let cnCount = ringIndices.filter { atoms[$0].element == .C || atoms[$0].element == .N }.count
            guard cnCount >= ringIndices.count - 1 else { continue }

            result.append(AromaticRing(centroid: centroid, normal: n, atomIndices: ringIndices))
        }
        return result
    }

    private static func findCycles(
        start: Int, adj: [[Int]], maxLen: Int, found: inout Set<[Int]>
    ) {
        var stack: [(node: Int, path: [Int])] = [(start, [start])]

        while !stack.isEmpty {
            let (node, path) = stack.removeLast()
            guard path.count <= maxLen else { continue }

            for neighbor in adj[node] {
                if neighbor == start && (path.count == 5 || path.count == 6) {
                    let ring = path.sorted()
                    if ring[0] == start {
                        found.insert(ring)
                    }
                } else if neighbor > start && !path.contains(neighbor) && path.count < maxLen {
                    stack.append((neighbor, path + [neighbor]))
                }
            }
        }
    }

    // MARK: - Full Interaction Detection

    static func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        ligandBonds: [Bond] = []
    ) -> [MolecularInteraction] {
        var result: [MolecularInteraction] = []
        var idCounter = 0

        let positiveResAtoms: Set<String> = ["NZ", "NH1", "NH2", "NE"]
        let negativeResAtoms: Set<String> = ["OD1", "OD2", "OE1", "OE2"]
        let metals: Set<Element> = [.Fe, .Zn, .Ca, .Mg, .Mn, .Cu]

        if let gpu = InteractionDetectorGPU.shared {
            let gpuResults = gpu.detect(
                ligandAtoms: ligandAtoms,
                ligandPositions: ligandPositions,
                proteinAtoms: proteinAtoms,
                positiveResAtoms: positiveResAtoms,
                negativeResAtoms: negativeResAtoms,
                metals: metals
            )
            for gi in gpuResults {
                result.append(MolecularInteraction(
                    id: idCounter,
                    ligandAtomIndex: Int(gi.ligandAtomIndex),
                    proteinAtomIndex: Int(gi.proteinAtomIndex),
                    type: MolecularInteraction.InteractionType(rawValue: Int(gi.type)) ?? .hbond,
                    distance: gi.distance,
                    ligandPosition: gi.ligandPosition,
                    proteinPosition: gi.proteinPosition
                ))
                idCounter += 1
            }
        } else {
            // CPU fallback with spatial grid
            var ligandHasStrongInteraction: Set<Int> = []
            let cellSize: Float = 6.0
            let invCell: Float = 1.0 / cellSize
            struct CellKey: Hashable { let x, y, z: Int }
            var proteinGrid: [CellKey: [Int]] = [:]
            proteinGrid.reserveCapacity(proteinAtoms.count / 3)
            for (pi, protAtom) in proteinAtoms.enumerated() {
                let ck = CellKey(x: Int(floor(protAtom.position.x * invCell)),
                                 y: Int(floor(protAtom.position.y * invCell)),
                                 z: Int(floor(protAtom.position.z * invCell)))
                proteinGrid[ck, default: []].append(pi)
            }

            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count else { continue }
                let lp = ligandPositions[li]
                let lcx = Int(floor(lp.x * invCell))
                let lcy = Int(floor(lp.y * invCell))
                let lcz = Int(floor(lp.z * invCell))

                for ndx in -1...1 { for ndy in -1...1 { for ndz in -1...1 {
                let nkey = CellKey(x: lcx + ndx, y: lcy + ndy, z: lcz + ndz)
                guard let cellIndices = proteinGrid[nkey] else { continue }
                for pi in cellIndices {
                    let protAtom = proteinAtoms[pi]
                    let d = simd_distance(lp, protAtom.position)
                    guard d < 6.0 else { continue }
                    let protName = protAtom.name.trimmingCharacters(in: .whitespaces)

                    if d < 2.8 {
                        let ligCoord = ligAtom.element == .N || ligAtom.element == .O || ligAtom.element == .S
                        let protMetal = metals.contains(protAtom.element)
                        let ligMetal = metals.contains(ligAtom.element)
                        let protCoord = protAtom.element == .N || protAtom.element == .O || protAtom.element == .S
                        if (protMetal && ligCoord) || (ligMetal && protCoord) {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .metalCoord, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li); continue
                        }
                    }
                    if d < 4.0 {
                        let protPositive = positiveResAtoms.contains(protName)
                        let protNegative = negativeResAtoms.contains(protName)
                        if (protPositive && ligAtom.formalCharge < 0) || (protNegative && ligAtom.formalCharge > 0) {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .saltBridge, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li); continue
                        }
                    }
                    if d >= 2.2 && d <= 3.5 {
                        let ligDA = ligAtom.element == .N || ligAtom.element == .O
                        let proDA = protAtom.element == .N || protAtom.element == .O
                        if ligDA && proDA {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .hbond, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li); continue
                        }
                    }
                    if d >= 2.5 && d <= 3.5 {
                        let halogen = ligAtom.element == .F || ligAtom.element == .Cl || ligAtom.element == .Br
                        let acceptor = protAtom.element == .N || protAtom.element == .O
                        if halogen && acceptor {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .halogen, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li)
                        }
                    }
                }
                }}} // spatial grid
            }

            // CPU hydrophobic contacts
            let invCellH = invCell
            var hydroCount: [Int: Int] = [:]
            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count, !ligandHasStrongInteraction.contains(li) else { continue }
                guard ligAtom.element == .C || ligAtom.element == .S else { continue }
                let lp = ligandPositions[li]
                let cx = Int(floor(lp.x * invCellH)), cy = Int(floor(lp.y * invCellH)), cz = Int(floor(lp.z * invCellH))
                for ndx in -1...1 { for ndy in -1...1 { for ndz in -1...1 {
                guard (hydroCount[li, default: 0]) < 2 else { continue }
                let hk = CellKey(x: cx + ndx, y: cy + ndy, z: cz + ndz)
                guard let hci = proteinGrid[hk] else { continue }
                for pi in hci {
                    let pa = proteinAtoms[pi]
                    guard pa.element == .C || pa.element == .S else { continue }
                    let d = simd_distance(lp, pa.position)
                    guard d >= 3.4 && d <= 4.0, (hydroCount[li, default: 0]) < 2 else { continue }
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                        type: .hydrophobic, distance: d,
                        ligandPosition: lp, proteinPosition: pa.position))
                    idCounter += 1; hydroCount[li, default: 0] += 1
                }
                }}}
            }
        }

        // π-π stacking (always CPU — few rings)
        let proteinRings = detectAromaticRings(atoms: proteinAtoms)
        let ligandRings = detectAromaticRings(
            atoms: ligandAtoms, positions: ligandPositions,
            bonds: ligandBonds.isEmpty ? nil : ligandBonds
        )

        for ligRing in ligandRings {
            for protRing in proteinRings {
                let d = simd_distance(ligRing.centroid, protRing.centroid)
                guard d >= 3.3 && d <= 5.5 else { continue }
                let dotN = abs(simd_dot(ligRing.normal, protRing.normal))
                let isFaceToFace = dotN > 0.85 && d < 4.2
                let isEdgeToFace = dotN < 0.5 && d >= 4.0 && d <= 5.5
                if isFaceToFace || isEdgeToFace {
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: ligRing.atomIndices.first ?? 0,
                        proteinAtomIndex: protRing.atomIndices.first ?? 0,
                        type: .piStack, distance: d,
                        ligandPosition: ligRing.centroid, proteinPosition: protRing.centroid))
                    idCounter += 1
                }
            }
        }

        // π-cation (always CPU — few rings)
        for ligRing in ligandRings {
            for (pi, protAtom) in proteinAtoms.enumerated() {
                let protName = protAtom.name.trimmingCharacters(in: .whitespaces)
                guard positiveResAtoms.contains(protName) || protAtom.formalCharge > 0 else { continue }
                let d = simd_distance(ligRing.centroid, protAtom.position)
                guard d < 5.0 else { continue }
                let toAtom = simd_normalize(protAtom.position - ligRing.centroid)
                if abs(simd_dot(toAtom, ligRing.normal)) > 0.6 {
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: ligRing.atomIndices.first ?? 0,
                        proteinAtomIndex: pi, type: .piCation, distance: d,
                        ligandPosition: ligRing.centroid, proteinPosition: protAtom.position))
                    idCounter += 1
                }
            }
        }
        for protRing in proteinRings {
            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count else { continue }
                guard ligAtom.formalCharge > 0 else { continue }
                let lp = ligandPositions[li]
                let d = simd_distance(protRing.centroid, lp)
                guard d < 5.0 else { continue }
                let toAtom = simd_normalize(lp - protRing.centroid)
                if abs(simd_dot(toAtom, protRing.normal)) > 0.6 {
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: li,
                        proteinAtomIndex: protRing.atomIndices.first ?? 0,
                        type: .piCation, distance: d,
                        ligandPosition: lp, proteinPosition: protRing.centroid))
                    idCounter += 1
                }
            }
        }

        return result
    }
}

// MARK: - GPU Interaction Detection Accelerator

/// Metal-accelerated interaction detection. One thread per ligand atom,
/// each checking all protein atoms with atomic append to output buffer.
final class InteractionDetectorGPU {
    nonisolated(unsafe) static let shared: InteractionDetectorGPU? = {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "detectInteractions")
        else { return nil }

        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            return InteractionDetectorGPU(device: device, commandQueue: commandQueue, pipeline: pipeline)
        } catch {
            return nil
        }
    }()

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let maxInteractions = 2048

    private init(device: MTLDevice, commandQueue: MTLCommandQueue, pipeline: MTLComputePipelineState) {
        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pipeline
    }

    private func atomFlags(for atom: Atom, positiveResAtoms: Set<String>, negativeResAtoms: Set<String>, metals: Set<Element>) -> UInt32 {
        var flags: UInt32 = 0
        switch atom.element {
        case .N:  flags |= UInt32(IDET_FLAG_N)
        case .O:  flags |= UInt32(IDET_FLAG_O)
        case .S:  flags |= UInt32(IDET_FLAG_S)
        case .C:  flags |= UInt32(IDET_FLAG_C)
        case .F:  flags |= UInt32(IDET_FLAG_F) | UInt32(IDET_FLAG_HALOGEN)
        case .Cl: flags |= UInt32(IDET_FLAG_CL) | UInt32(IDET_FLAG_HALOGEN)
        case .Br: flags |= UInt32(IDET_FLAG_BR) | UInt32(IDET_FLAG_HALOGEN)
        default:  break
        }
        if metals.contains(atom.element) { flags |= UInt32(IDET_FLAG_METAL) }
        let name = atom.name.trimmingCharacters(in: .whitespaces)
        if positiveResAtoms.contains(name) { flags |= UInt32(IDET_FLAG_POS_RES) }
        if negativeResAtoms.contains(name) { flags |= UInt32(IDET_FLAG_NEG_RES) }
        return flags
    }

    func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        positiveResAtoms: Set<String>,
        negativeResAtoms: Set<String>,
        metals: Set<Element>
    ) -> [GPUInteraction] {
        let nLig = ligandAtoms.count
        let nProt = proteinAtoms.count
        guard nLig > 0, nProt > 0, ligandPositions.count >= nLig else { return [] }

        var protGPU = proteinAtoms.map { atom -> InteractionAtomGPU in
            InteractionAtomGPU(
                position: atom.position,
                flags: atomFlags(for: atom, positiveResAtoms: positiveResAtoms, negativeResAtoms: negativeResAtoms, metals: metals),
                formalCharge: Int32(atom.formalCharge),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        var ligGPU = ligandAtoms.map { atom -> InteractionAtomGPU in
            InteractionAtomGPU(
                position: atom.position,
                flags: atomFlags(for: atom, positiveResAtoms: positiveResAtoms, negativeResAtoms: negativeResAtoms, metals: metals),
                formalCharge: Int32(atom.formalCharge),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        var positions = Array(ligandPositions.prefix(nLig))
        var counter: UInt32 = 0
        var params = InteractionDetectParams(
            numLigandAtoms: UInt32(nLig),
            numProteinAtoms: UInt32(nProt),
            maxInteractions: UInt32(maxInteractions),
            _pad0: 0
        )

        guard let protBuffer = device.makeBuffer(bytes: &protGPU, length: nProt * MemoryLayout<InteractionAtomGPU>.stride, options: .storageModeShared),
              let ligBuffer = device.makeBuffer(bytes: &ligGPU, length: nLig * MemoryLayout<InteractionAtomGPU>.stride, options: .storageModeShared),
              let posBuffer = device.makeBuffer(bytes: &positions, length: nLig * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let outBuffer = device.makeBuffer(length: maxInteractions * MemoryLayout<GPUInteraction>.stride, options: .storageModeShared),
              let ctrBuffer = device.makeBuffer(bytes: &counter, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let paramBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<InteractionDetectParams>.stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else { return [] }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(protBuffer, offset: 0, index: 0)
        enc.setBuffer(ligBuffer, offset: 0, index: 1)
        enc.setBuffer(posBuffer, offset: 0, index: 2)
        enc.setBuffer(outBuffer, offset: 0, index: 3)
        enc.setBuffer(ctrBuffer, offset: 0, index: 4)
        enc.setBuffer(paramBuffer, offset: 0, index: 5)

        let tgSize = MTLSize(width: min(nLig, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (nLig + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let count = min(Int(ctrBuffer.contents().load(as: UInt32.self)), maxInteractions)
        guard count > 0 else { return [] }

        let ptr = outBuffer.contents().bindMemory(to: GPUInteraction.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
