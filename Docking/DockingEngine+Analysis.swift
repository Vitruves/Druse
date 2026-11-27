// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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
            let edgeCount = edgeBuf.length / MemoryLayout<TorsionEdge>.stride
            let movingCount = idxBuf.length / MemoryLayout<Int32>.stride
            let edges = UnsafeBufferPointer(
                start: edgeBuf.contents().bindMemory(to: TorsionEdge.self, capacity: edgeCount),
                count: edgeCount)
            let moving = UnsafeBufferPointer(
                start: idxBuf.contents().bindMemory(to: Int32.self, capacity: movingCount),
                count: movingCount)
            applyTorsionRotations(to: &positions, angles: torsions(from: pose), edges: edges, moving: moving)
        }

        return positions
    }

    func extractBestPose(from buffer: MTLBuffer? = nil, ligandAtoms: [Atom], centroid: SIMD3<Float>, maxPoses: Int) -> DockingResult? {
        guard let buffer = buffer ?? populationBuffer else { return nil }
        let poseCount = min(maxPoses, buffer.length / MemoryLayout<DockPose>.stride)
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
        maxPoses: Int
    ) -> [DockingResult] {
        guard let buffer = buffer ?? populationBuffer else { return [] }
        let poseCount = min(maxPoses, buffer.length / MemoryLayout<DockPose>.stride)
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
        return results
    }

    // MARK: - Output Selection & RMSD Clustering

    /// AutoDock Vina's `energy_range` default: poses scoring more than this above the best
    /// are not reported. Most GA walkers of a multi-run search never reach the binding
    /// basin, so the raw final populations are dominated by unconverged poses.
    nonisolated static let outputEnergyRangeKcal: Float = 3.0

    /// AutoDock Vina's `min_rmsd` for reported poses: within a cluster, a pose closer than
    /// this to a better pose already kept is a duplicate of that minimum.
    nonisolated static let outputDuplicateRMSD: Float = 1.0

    /// Leader-clustering radius for binding-mode clusters.
    nonisolated static let clusterRMSD: Float = 2.0

    /// Vina's `num_modes` analogue: only the best-scoring binding modes are reported. Vina's
    /// default is 9; Druse keeps more so the explicit rerank of the top
    /// `DockingConfig.explicitRerankTopClusters` clusters always has candidates to reorder.
    nonisolated static let outputMaxClusters = 20

    /// `outputEnergyRangeKcal` in the units of `DockingResult.energy` for `scoringMethod`:
    /// kcal/mol for Vina, Drusina and PIGNet2; −pKd for DruseAF, where ΔG = −RT·ln10·pKd.
    static func outputEnergyWindow(for scoringMethod: ScoringMethod) -> Float {
        let rtLn10Kcal: Float = 0.0019872 * 298.15 * Float(M_LN10)
        return scoringMethod.isAffinityScore ? outputEnergyRangeKcal / rtLn10Kcal : outputEnergyRangeKcal
    }

    /// Poses whose score lies within the output energy window of the best one.
    static func withinOutputEnergyWindow(_ results: [DockingResult], scoringMethod: ScoringMethod) -> [DockingResult] {
        guard let best = results.lazy.map(\.energy).min() else { return [] }
        let cutoff = best + outputEnergyWindow(for: scoringMethod)
        return results.filter { $0.energy <= cutoff }
    }

    /// Vina's conformation-independent rotor weight (`wRotEntropy` in DockingCommon.h).
    nonisolated static let vinaRotEntropyWeight: Float = 0.05846

    /// AutoDock Vina reports each pose against the intramolecular energy of the best pose
    /// (Vina 1.1/1.2 `model::eval_adjusted`: E_i = conf_independent(inter_i + intra_i − intra_best)),
    /// so the top pose scores its intermolecular energy alone. The GPU kernels subtract the
    /// input conformer's intra energy instead, which makes absolute scores depend on the
    /// input geometry (a strained input inflates every score). The two references differ
    /// by a per-run constant, so re-referencing is a uniform shift: ranking, the energy
    /// window and clustering are unchanged. Valid only for energies of the form
    /// (inter + intra − inputIntra)·norm + penalties (plain Vina scoring, rigid receptor).
    func rereferencedToBestPoseIntra(
        _ results: [DockingResult],
        ligandAtoms: [DockLigandAtom],
        pairList: [UInt32],
        inputReferenceIntra: Float,
        numTorsions: Int
    ) -> [DockingResult] {
        guard let best = results.min(by: { $0.energy < $1.energy }),
              let shift = bestPoseReferenceShift(
                  best, ligandAtoms: ligandAtoms, pairList: pairList,
                  inputReferenceIntra: inputReferenceIntra, numTorsions: numTorsions
              ) else { return results }
        ActivityLog.shared.info(
            "[Engine] Vina intra reference: input conformer \(String(format: "%.3f", inputReferenceIntra)), " +
            "best pose \(String(format: "%.3f", shift.bestIntra)) → score shift \(String(format: "%+.3f", shift.energy)) kcal/mol",
            category: .dock
        )
        return results.map { shift.applied(to: $0) }
    }

    /// Score shift that moves a Vina result from the input-conformer intra reference to
    /// `best`'s own intra energy. Also used for the live "best so far" pose, so the readout
    /// matches the final ranking. Nil when the pose has no per-atom positions.
    func bestPoseReferenceShift(
        _ best: DockingResult,
        ligandAtoms: [DockLigandAtom],
        pairList: [UInt32],
        inputReferenceIntra: Float,
        numTorsions: Int
    ) -> VinaReferenceShift? {
        guard best.transformedAtomPositions.count == ligandAtoms.count else { return nil }
        var posed = ligandAtoms
        for i in posed.indices { posed[i].position = best.transformedAtomPositions[i] }
        let bestIntra = intramolecularReferenceEnergy(ligandAtoms: posed, pairList: pairList)
        let norm = 1 / (1 + Self.vinaRotEntropyWeight * Float(numTorsions))
        let delta = inputReferenceIntra - bestIntra
        // torsionPenalty is normalized(inter + intraDelta) − inter − intraDelta; moving the
        // reference changes intraDelta by `delta`, so the normalization term moves by
        // delta·(norm − 1) and the breakdown keeps summing to the shifted total.
        return VinaReferenceShift(energy: delta * norm, torsion: delta * (norm - 1), bestIntra: bestIntra)
    }

    struct VinaReferenceShift {
        let energy: Float
        let torsion: Float
        let bestIntra: Float

        func applied(to result: DockingResult) -> DockingResult {
            var r = result
            r.energy += energy
            r.torsionPenalty += torsion
            return r
        }
    }

    /// Greedy leader clustering in energy order (Vina-style): the lowest-energy unassigned
    /// pose becomes a cluster leader and absorbs every unassigned pose within
    /// `clusterRMSD`. Absorbed poses closer than `outputDuplicateRMSD` to a better pose of
    /// the same cluster are dropped as duplicates (many walkers converge to the same minimum),
    /// and clustering stops after `outputMaxClusters` modes.
    /// Only leader-vs-unassigned RMSDs are evaluated (with early exit once the bound is
    /// exceeded), instead of materialising the full n(n-1)/2 matrix.
    /// The RMSD work runs off the main actor; the result is identical to running it inline.
    func clusterPoses(_ results: [DockingResult]) async -> [DockingResult] {
        guard !results.isEmpty else { return [] }
        return await Task.detached(priority: .userInitiated) {
            DockingEngine.leaderClusters(results)
        }.value
    }

    nonisolated static func leaderClusters(_ results: [DockingResult]) -> [DockingResult] {
        let sorted = results.sorted { $0.energy < $1.energy }
        let positions = sorted.map(\.transformedAtomPositions)

        var out: [DockingResult] = []
        out.reserveCapacity(sorted.count)
        var unassigned = Array(0..<sorted.count)
        var clusterID = 0

        while clusterID < outputMaxClusters, let leader = unassigned.first {
            var leaderResult = sorted[leader]
            leaderResult.clusterID = clusterID
            leaderResult.clusterRank = 0
            out.append(leaderResult)

            let candidates = Array(unassigned.dropFirst())
            let clusterLimit = clusterRMSD * clusterRMSD * Float(positions[leader].count)
            let duplicateLimit = outputDuplicateRMSD * outputDuplicateRMSD * Float(positions[leader].count)
            let distances = leaderSquaredDistances(
                leader: positions[leader], candidates: candidates,
                positions: positions, limit: clusterLimit
            )

            var kept = [leader]
            var remaining: [Int] = []
            remaining.reserveCapacity(candidates.count)
            for (c, j) in candidates.enumerated() {
                guard distances[c] < clusterLimit else {
                    remaining.append(j)
                    continue
                }
                if distances[c] < duplicateLimit || kept.dropFirst().contains(where: {
                    squaredDistanceSum(positions[$0], positions[j], limit: duplicateLimit) < duplicateLimit
                }) {
                    continue
                }
                var member = sorted[j]
                member.clusterID = clusterID
                member.clusterRank = kept.count
                out.append(member)
                kept.append(j)
            }
            unassigned = remaining
            clusterID += 1
        }
        return out.sorted { $0.energy < $1.energy }
    }

    /// Per-candidate squared-distance sums, exact below the cluster limit.
    /// Parallelised across worker threads for large candidate sets.
    private nonisolated static func leaderSquaredDistances(
        leader: [SIMD3<Float>],
        candidates: [Int],
        positions: [[SIMD3<Float>]],
        limit: Float
    ) -> [Float] {
        var distances = [Float](repeating: .infinity, count: candidates.count)
        if candidates.count >= 2048 {
            distances.withUnsafeMutableBufferPointer { buf in
                nonisolated(unsafe) let out = buf
                CPUBudget.concurrentPerform(iterations: candidates.count) { c in
                    out[c] = squaredDistanceSum(leader, positions[candidates[c]], limit: limit)
                }
            }
        } else {
            for c in candidates.indices {
                distances[c] = squaredDistanceSum(leader, positions[candidates[c]], limit: limit)
            }
        }
        return distances
    }

    /// Accumulates Σ|a_i - b_i|² until the limit is reached.
    /// Poses with mismatched or empty atom lists never cluster together.
    private nonisolated static func squaredDistanceSum(
        _ a: [SIMD3<Float>], _ b: [SIMD3<Float>], limit: Float
    ) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var sum: Float = 0
        for i in 0..<a.count {
            sum += simd_distance_squared(a[i], b[i])
            if sum >= limit { break }
        }
        return sum
    }

    // MARK: - Docking Diagnostics

    func computeDiagnostics(
        results: [DockingResult],
        ligandAtoms: [Atom],
        heavyBonds: [Bond]
    ) -> DockingDiagnostics {
        let gridOrigin = SIMD3<Float>(gridParams.origin.x, gridParams.origin.y, gridParams.origin.z)
        // The last grid node sits at origin + (dims - 1) * spacing (gridUpperBound in DockingCommon.h).
        let nodeSpans = SIMD3<Float>(
            Float(max(gridParams.dims.x, 1) - 1), Float(max(gridParams.dims.y, 1) - 1), Float(max(gridParams.dims.z, 1) - 1)
        )
        let gridEnd = gridOrigin + nodeSpans * gridParams.spacing
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

        // Bin protein atoms into contactCutoff-sized cells: any atom closer than the cutoff
        // lies in the 27-cell neighbourhood, so in-contact poses (the common case) get an
        // exact minimum without scanning the whole receptor. Poses with no atom within the
        // cutoff fall back to the exhaustive scan so the reported mean stays exact.
        let contactCutoff: Float = 4.0
        struct Cell: Hashable { let x, y, z: Int32 }
        func cell(_ p: SIMD3<Float>) -> Cell? {
            let q = (p / contactCutoff).rounded(.down)
            let limit = Float(Int32.max / 2)
            guard q.x.isFinite, q.y.isFinite, q.z.isFinite,
                  abs(q.x) < limit, abs(q.y) < limit, abs(q.z) < limit else { return nil }
            return Cell(x: Int32(q.x), y: Int32(q.y), z: Int32(q.z))
        }
        var proteinCells: [Cell: [SIMD3<Float>]] = [:]
        for pa in proteinAtoms {
            if let c = cell(pa.position) { proteinCells[c, default: []].append(pa.position) }
        }

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
                    var minDistSq: Float = .infinity
                    for lp in positions {
                        guard let c = cell(lp) else { continue }
                        for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                            guard let bucket = proteinCells[Cell(x: c.x + dx, y: c.y + dy, z: c.z + dz)] else { continue }
                            for pp in bucket { minDistSq = min(minDistSq, simd_distance_squared(lp, pp)) }
                        }}}
                    }
                    if minDistSq >= contactCutoff * contactCutoff {
                        for lp in positions {
                            for pa in proteinAtoms { minDistSq = min(minDistSq, simd_distance_squared(lp, pa.position)) }
                        }
                    }
                    minProteinDistances.append(sqrt(minDistSq))
                }
            }
        }

        let meanCentroidDist = centroidDistances.isEmpty ? 0 : centroidDistances.reduce(0, +) / Float(centroidDistances.count)
        let meanProteinDist = minProteinDistances.isEmpty ? 0 : minProteinDistances.reduce(0, +) / Float(minProteinDistances.count)

        let contactPoses = minProteinDistances.filter { $0 < contactCutoff }.count

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

// MARK: - Torsion Application

/// Rotate the moving atoms of each torsion edge about its bond axis, in edge order.
/// CPU mirror of `applyTorsions()` in DockingCommon.h so CPU-reconstructed poses match
/// what the GPU scored. Edge and moving-index lookups are bounds-checked.
func applyTorsionRotations(
    to positions: inout [SIMD3<Float>],
    angles: [Float],
    edges: UnsafeBufferPointer<TorsionEdge>,
    moving: UnsafeBufferPointer<Int32>
) {
    for t in 0..<min(angles.count, edges.count) {
        let angle = angles[t]
        if abs(angle) < 1e-6 { continue }

        let edge = edges[t]
        let pivotIdx = Int(edge.atom1)
        let axisIdx = Int(edge.atom2)
        guard pivotIdx >= 0, axisIdx >= 0, pivotIdx < positions.count, axisIdx < positions.count else { continue }

        let pivot = positions[pivotIdx]
        let axis = simd_normalize(positions[axisIdx] - pivot)
        let cosA = cos(angle)
        let sinA = sin(angle)

        let start = Int(edge.movingStart)
        let end = min(start + Int(edge.movingCount), moving.count)
        guard start >= 0, start < end else { continue }
        for m in start..<end {
            let atomIdx = Int(moving[m])
            guard atomIdx >= 0, atomIdx < positions.count else { continue }
            let v = positions[atomIdx] - pivot
            let rotated = v * cosA + simd_cross(axis, v) * sinA + axis * simd_dot(axis, v) * (1.0 - cosA)
            positions[atomIdx] = pivot + rotated
        }
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

    // MARK: - H-bond Donor Detection
    //
    // An atom can DONATE in an H-bond only if it actually has a hydrogen
    // attached. The interaction detector previously checked elements only
    // (any N or O within 2.2-3.5 Å of another N or O = "H-bond"), which
    // produced false positives for pure acceptors like a neutral tertiary
    // amine (3 bonds, 0 H) sitting next to a deprotonated carboxylate
    // (1 bond, charge -1, 0 H). Both have lone pairs but no H to share.
    //
    // For ligand atoms we compute implicit-H counts from the bond graph and
    // formal charge. For protein atoms we use a residue+atom-name lookup —
    // explicit bonds are not always present on protein Atoms, but the
    // protonation pipeline (Chemistry/Protonation.swift) has already set
    // formal charges so the donor list below is unambiguous.

    // H-bond donor/acceptor classification is in Chemistry/HBondClassifier.swift
    // so it can be shared with ML/Inference.swift without crossing into
    // @MainActor DockingEngine. These thin wrappers keep existing call sites
    // unchanged (e.g. `Self.computeLigandHBondDonorFlags(...)`).

    static func computeLigandHBondDonorFlags(atoms: [Atom], bonds: [Bond]) -> [Bool] {
        HBondClassifier.computeLigandHBondDonorFlags(atoms: atoms, bonds: bonds)
    }

    static func computeLigandHBondAcceptorFlags(atoms: [Atom], bonds: [Bond]) -> [Bool] {
        HBondClassifier.computeLigandHBondAcceptorFlags(atoms: atoms, bonds: bonds)
    }

    static func proteinHBondDonorFlag(_ atom: Atom) -> Bool {
        HBondClassifier.proteinHBondDonorFlag(atom)
    }

    static func proteinHBondAcceptorFlag(_ atom: Atom) -> Bool {
        HBondClassifier.proteinHBondAcceptorFlag(atom)
    }

    // MARK: - Protein Feature Geometry
    //
    // The planes, σ-hole axes and cation sites the directional interactions
    // need. Built to mirror `DockingEngine.prepareDrusinaBuffers` so that the
    // detector and the Drusina kernel see the same features.

    /// A protein amide plane — a peptide link C(i)=O(i)···N(i+1), or an
    /// Asn/Gln sidechain carboxamide — for amide-π stacking.
    struct AmidePlane: Sendable {
        let centroid: SIMD3<Float>
        let normal: SIMD3<Float>
        let atomIndex: Int      // representative atom (the amide carbon)
    }

    /// A protein sulfur σ-hole (Met SD, Cys SG) for chalcogen bonding.
    /// One record per S–C bond: the σ-hole lies opposite the bond.
    struct SulfurSigmaHole: Sendable {
        let position: SIMD3<Float>
        let bondedCDir: SIMD3<Float>   // normalize(C - S)
        let atomIndex: Int
    }

    private static func amidePlane(
        c: SIMD3<Float>, o: SIMD3<Float>, n: SIMD3<Float>, atomIndex: Int
    ) -> AmidePlane? {
        let normal = simd_cross(o - c, n - c)
        let len = simd_length(normal)
        guard len > 1e-6 else { return nil }
        return AmidePlane(centroid: (c + o + n) / 3.0, normal: normal / len, atomIndex: atomIndex)
    }

    /// Amide planes of the protein: one per peptide link plus Asn/Gln sidechains.
    static func detectAmidePlanes(atoms: [Atom]) -> [AmidePlane] {
        struct Residue { var seq: Int; var name: String; var atoms: [(Int, Atom)] }
        var byResidue: [String: Residue] = [:]
        for (i, atom) in atoms.enumerated() where !atom.isHetAtom {
            let key = "\(atom.chainID)_\(atom.residueSeq)"
            if byResidue[key] == nil {
                byResidue[key] = Residue(
                    seq: atom.residueSeq,
                    name: atom.residueName.trimmingCharacters(in: .whitespaces),
                    atoms: [])
            }
            byResidue[key]?.atoms.append((i, atom))
        }

        func named(_ residue: Residue, _ name: String) -> (Int, Atom)? {
            residue.atoms.first { $0.1.name.trimmingCharacters(in: .whitespaces) == name }
        }

        var planes: [AmidePlane] = []

        // Peptide link: C and O of residue i with N of residue i+1. The N, C
        // and O of a single residue are N–CA–C=O and are not an amide group.
        var chains: [String: [Residue]] = [:]
        for (key, residue) in byResidue {
            let chain = String(key.prefix(while: { $0 != "_" }))
            chains[chain, default: []].append(residue)
        }
        for (_, residues) in chains {
            let ordered = residues.sorted { $0.seq < $1.seq }
            for i in 0..<ordered.count where i + 1 < ordered.count {
                guard ordered[i + 1].seq == ordered[i].seq + 1,
                      let c = named(ordered[i], "C"),
                      let o = named(ordered[i], "O"),
                      let n = named(ordered[i + 1], "N") else { continue }
                if let p = amidePlane(c: c.1.position, o: o.1.position,
                                      n: n.1.position, atomIndex: c.0) {
                    planes.append(p)
                }
            }
        }

        // Sidechain carboxamides
        for (_, residue) in byResidue {
            let names: (String, String, String)
            switch residue.name {
            case "ASN": names = ("CG", "OD1", "ND2")
            case "GLN": names = ("CD", "OE1", "NE2")
            default: continue
            }
            guard let c = named(residue, names.0), let o = named(residue, names.1),
                  let n = named(residue, names.2) else { continue }
            if let p = amidePlane(c: c.1.position, o: o.1.position,
                                  n: n.1.position, atomIndex: c.0) {
                planes.append(p)
            }
        }
        return planes
    }

    /// Protein sulfur σ-holes (Met SD → CG/CE, Cys SG → CB).
    static func detectSulfurSigmaHoles(atoms: [Atom]) -> [SulfurSigmaHole] {
        var byResidue: [String: [Atom]] = [:]
        for atom in atoms {
            byResidue["\(atom.chainID)_\(atom.residueSeq)", default: []].append(atom)
        }
        var holes: [SulfurSigmaHole] = []
        for (i, atom) in atoms.enumerated() {
            let name = atom.name.trimmingCharacters(in: .whitespaces)
            let resName = atom.residueName.trimmingCharacters(in: .whitespaces)
            let carbonNames: [String]
            if resName == "MET" && name == "SD" { carbonNames = ["CG", "CE"] }
            else if resName == "CYS" && name == "SG" { carbonNames = ["CB"] }
            else { continue }
            let sameRes = byResidue["\(atom.chainID)_\(atom.residueSeq)"] ?? []
            for cName in carbonNames {
                guard let c = sameRes.first(where: {
                    $0.name.trimmingCharacters(in: .whitespaces) == cName
                }) else { continue }
                let dir = c.position - atom.position
                let len = simd_length(dir)
                guard len > 1e-6 else { continue }
                holes.append(SulfurSigmaHole(position: atom.position,
                                             bondedCDir: dir / len, atomIndex: i))
            }
        }
        return holes
    }

    /// Cation sites, matching `DockingEngine.prepareDrusinaBuffers`: Lys NZ,
    /// Arg NH1/NH2, anything with a positive formal charge, and metal ions.
    static let cationAtomNames: Set<String> = ["NZ", "NH1", "NH2"]
    static let cationMetals: Set<Element> = [.Zn, .Fe, .Mg, .Ca, .Mn, .Cu, .Co, .Ni]

    static func isCationSite(_ atom: Atom) -> Bool {
        cationAtomNames.contains(atom.name.trimmingCharacters(in: .whitespaces))
            || atom.formalCharge > 0
            || cationMetals.contains(atom.element)
    }

    /// Halogen element code matching `HalogenBondInfo.elementType`
    /// (0 = F, 1 = Cl, 2 = Br, 3 = I); nil for every other element.
    static func halogenElementType(_ element: Element) -> Int? {
        switch element {
        case .F:  return 0
        case .Cl: return 1
        case .Br: return 2
        case .I:  return 3
        default:  return nil
        }
    }

    /// Element-specific halogen bond optimum (Scholfield 2013 CSD statistics).
    static func halogenOptimalDistance(_ elementType: Int) -> Float {
        switch elementType {
        case 2:  return kICHalogenDistOptBr
        case 3:  return kICHalogenDistOptI
        default: return kICHalogenDistOptCl
        }
    }

    // MARK: - Spatial Index

    /// Uniform-grid neighbour index. Shared by hydrophobic-carbon typing, the
    /// CPU contact fallback and the directional interaction search, so the
    /// protein is bucketed once per pose instead of once per interaction type.
    struct NeighborGrid {
        private struct Key: Hashable { let x, y, z: Int }
        private let cellSize: Float
        private let cells: [Key: [Int]]

        init(positions: [SIMD3<Float>], cellSize: Float) {
            self.cellSize = cellSize
            let inv = 1.0 / cellSize
            var cells: [Key: [Int]] = [:]
            cells.reserveCapacity(positions.count / 3 + 1)
            for (i, p) in positions.enumerated() {
                let key = Key(x: Int(floor(p.x * inv)), y: Int(floor(p.y * inv)),
                              z: Int(floor(p.z * inv)))
                cells[key, default: []].append(i)
            }
            self.cells = cells
        }

        /// Indices in the 27 cells around `point`. Complete for radii <= cellSize.
        func neighbors(of point: SIMD3<Float>) -> [Int] {
            let inv = 1.0 / cellSize
            let cx = Int(floor(point.x * inv)), cy = Int(floor(point.y * inv))
            let cz = Int(floor(point.z * inv))
            var out: [Int] = []
            for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
                if let bucket = cells[Key(x: cx + dx, y: cy + dy, z: cz + dz)] {
                    out.append(contentsOf: bucket)
                }
            }}}
            return out
        }
    }

    /// Vina types a carbon bonded to N or O as C_P (polar) and every other
    /// carbon as C_H, and only C_H carries the hydrophobic term — so only C_H
    /// may be drawn as a hydrophobic contact. Sulfur is typed S_P and is not
    /// hydrophobic under Vina either.
    private static let covalentCNOCutoff: Float = 1.75

    static func hydrophobicCarbonFlags(
        atoms: [Atom], positions: [SIMD3<Float>], grid: NeighborGrid
    ) -> [Bool] {
        atoms.indices.map { i in
            guard atoms[i].element == .C else { return false }
            let p = positions[i]
            for j in grid.neighbors(of: p) where j != i {
                let e = atoms[j].element
                guard e == .N || e == .O else { continue }
                if simd_distance(p, positions[j]) < covalentCNOCutoff { return false }
            }
            return true
        }
    }

    static func hydrophobicCarbonFlags(atoms: [Atom], bonds: [Bond]) -> [Bool] {
        var polar = [Bool](repeating: false, count: atoms.count)
        for bond in bonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < atoms.count, b < atoms.count else { continue }
            if atoms[a].element == .C, atoms[b].element == .N || atoms[b].element == .O {
                polar[a] = true
            }
            if atoms[b].element == .C, atoms[a].element == .N || atoms[a].element == .O {
                polar[b] = true
            }
        }
        return atoms.indices.map { atoms[$0].element == .C && !polar[$0] }
    }

    /// Carbons covalently bonded to `index`, from the bond graph when the
    /// caller has one and by distance otherwise.
    private static func bondedCarbons(
        of index: Int, atoms: [Atom], bonds: [Bond], positions: [SIMD3<Float>]
    ) -> [Int] {
        if !bonds.isEmpty {
            var out: [Int] = []
            for bond in bonds {
                let partner: Int?
                if bond.atomIndex1 == index { partner = bond.atomIndex2 }
                else if bond.atomIndex2 == index { partner = bond.atomIndex1 }
                else { partner = nil }
                if let p = partner, p < atoms.count, atoms[p].element == .C { out.append(p) }
            }
            return out
        }
        let p = positions[index]
        return atoms.indices.filter {
            $0 != index && atoms[$0].element == .C
                && simd_distance(p, positions[$0]) < 2.2   // covers C-I at ~2.1 Å
        }
    }

    // MARK: - Full Interaction Detection

    static func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        ligandBonds: [Bond] = [],
        scoringMethod: ScoringMethod
    ) -> [MolecularInteraction] {
        let allowedTypes = scoringMethod.accountedInteractionTypes
        if allowedTypes.isEmpty { return [] }
        guard !ligandAtoms.isEmpty, !proteinAtoms.isEmpty,
              ligandPositions.count >= ligandAtoms.count else { return [] }
        var result: [MolecularInteraction] = []
        var idCounter = 0

        let positiveResAtoms: Set<String> = ["NZ", "NH1", "NH2", "NE"]
        let negativeResAtoms: Set<String> = ["OD1", "OD2", "OE1", "OE2"]
        let metals: Set<Element> = [.Fe, .Zn, .Ca, .Mg, .Mn, .Cu]
        let maxHydrophobicPerAtom = Int(kICHydrophobicMaxPerAtom)

        // Donor flags: an atom can DONATE to an H-bond only if it actually has
        // a hydrogen attached. For ligand atoms we derive this from the bond
        // graph (bond order sum + formal charge → implicit H count). For
        // protein atoms we use a residue-name lookup, since explicit bonds are
        // not always available on the Atom struct.
        let ligandHasH: [Bool] = computeLigandHBondDonorFlags(
            atoms: ligandAtoms, bonds: ligandBonds)
        let proteinHasH: [Bool] = proteinAtoms.map(proteinHBondDonorFlag)
        let proteinIsAcceptor: [Bool] = proteinAtoms.map(proteinHBondAcceptorFlag)
        let ligandIsAcceptor: [Bool] = computeLigandHBondAcceptorFlags(
            atoms: ligandAtoms, bonds: ligandBonds)

        let ligPositions = Array(ligandPositions.prefix(ligandAtoms.count))
        let proteinPositions = proteinAtoms.map(\.position)
        let grid = NeighborGrid(positions: proteinPositions, cellSize: kICMaxInteractionReach)
        let proteinHydrophobicC = hydrophobicCarbonFlags(
            atoms: proteinAtoms, positions: proteinPositions, grid: grid)
        let ligandHydrophobicC = ligandBonds.isEmpty
            ? hydrophobicCarbonFlags(
                atoms: ligandAtoms, positions: ligPositions,
                grid: NeighborGrid(positions: ligPositions, cellSize: kICMaxInteractionReach))
            : hydrophobicCarbonFlags(atoms: ligandAtoms, bonds: ligandBonds)

        /// Vina's hydrophobic term covers non-polar carbon and the halogens
        /// (xsIsHydrophobic: C_H, F_H, Cl_H, Br_H, I_H), and nothing else.
        func isVinaHydrophobic(_ index: Int, hydrophobicCarbon: [Bool], atoms: [Atom]) -> Bool {
            hydrophobicCarbon[index] || halogenElementType(atoms[index].element) != nil
        }

        if let gpu = InteractionDetectorGPU.shared,
           let gpuResults = gpu.detect(
                ligandAtoms: ligandAtoms,
                ligandPositions: ligandPositions,
                proteinAtoms: proteinAtoms,
                ligandHasH: ligandHasH,
                proteinHasH: proteinHasH,
                ligandHydrophobic: ligandAtoms.indices.map {
                    isVinaHydrophobic($0, hydrophobicCarbon: ligandHydrophobicC, atoms: ligandAtoms)
                },
                proteinHydrophobic: proteinAtoms.indices.map {
                    isVinaHydrophobic($0, hydrophobicCarbon: proteinHydrophobicC, atoms: proteinAtoms)
                },
                positiveResAtoms: positiveResAtoms,
                negativeResAtoms: negativeResAtoms,
                metals: metals
           ) {
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
            // CPU fallback (also used when the GPU path fails). Same criteria
            // and same precedence as Metal/InteractionCompute.metal.
            var hydroCount = [Int](repeating: 0, count: ligandAtoms.count)

            for (li, ligAtom) in ligandAtoms.enumerated() {
                let lp = ligPositions[li]
                let ligCoord = ligAtom.element == .N || ligAtom.element == .O || ligAtom.element == .S
                let ligMetal = metals.contains(ligAtom.element)
                let ligDA = ligAtom.element == .N || ligAtom.element == .O
                let ligHydro = isVinaHydrophobic(li, hydrophobicCarbon: ligandHydrophobicC,
                                                 atoms: ligandAtoms)

                for pi in grid.neighbors(of: lp) {
                    let protAtom = proteinAtoms[pi]
                    let d = simd_distance(lp, protAtom.position)
                    guard d < kICMaxInteractionReach else { continue }
                    let protName = protAtom.name.trimmingCharacters(in: .whitespaces)
                    var type: MolecularInteraction.InteractionType?

                    if d >= kICMetalDistMin && d <= kICMetalDistMax {
                        let protMetal = metals.contains(protAtom.element)
                        let protCoord = protAtom.element == .N || protAtom.element == .O
                            || protAtom.element == .S
                        if (protMetal && ligCoord) || (ligMetal && protCoord) {
                            type = .metalCoord
                        }
                    }
                    if type == nil && d <= kICSaltBridgeDistMax {
                        let protPositive = positiveResAtoms.contains(protName)
                        let protNegative = negativeResAtoms.contains(protName)
                        if (protPositive && ligAtom.formalCharge < 0)
                            || (protNegative && ligAtom.formalCharge > 0) {
                            type = .saltBridge
                        }
                    }
                    if type == nil && d >= kICHBondDistMin && d <= kICHBondDistMax {
                        let proDA = protAtom.element == .N || protAtom.element == .O
                        if ligDA && proDA && (ligandHasH[li] || proteinHasH[pi]) {
                            type = .hbond
                        }
                    }
                    if type == nil && d >= kICHydrophobicDistMin && d <= kICHydrophobicDistMax
                        && hydroCount[li] < maxHydrophobicPerAtom
                        && ligHydro
                        && isVinaHydrophobic(pi, hydrophobicCarbon: proteinHydrophobicC,
                                             atoms: proteinAtoms) {
                        type = .hydrophobic
                        hydroCount[li] += 1
                    }

                    guard let t = type else { continue }
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                        type: t, distance: d,
                        ligandPosition: lp, proteinPosition: protAtom.position))
                    idCounter += 1
                }
            }
        }

        // ------------------------------------------------------------------
        // Directional interactions — always CPU.
        //
        // These need a ring plane, an amide plane or a σ-hole axis, none of
        // which the GPU detector's flat atom buffer carries. All criteria come
        // from Metal/InteractionCriteria.h, so they are identical to the ones
        // Metal/DockingScoring.metal rewards.
        //
        // Where the geometry physically admits a single partner — a σ-hole is
        // one lobe, a ring face takes one stacking partner — only the best
        // partner is emitted, matching the scoring kernel's per-site cap.
        // ------------------------------------------------------------------
        let proteinRings = detectAromaticRings(atoms: proteinAtoms)
        let ligandRings = detectAromaticRings(
            atoms: ligandAtoms, positions: ligandPositions,
            bonds: ligandBonds.isEmpty ? nil : ligandBonds
        )
        let ligandRingMembers = Set(ligandRings.flatMap(\.atomIndices))

        func emit(_ type: MolecularInteraction.InteractionType,
                  ligandAtomIndex: Int, proteinAtomIndex: Int, distance: Float,
                  ligandPosition: SIMD3<Float>, proteinPosition: SIMD3<Float>) {
            result.append(MolecularInteraction(
                id: idCounter, ligandAtomIndex: ligandAtomIndex,
                proteinAtomIndex: proteinAtomIndex, type: type, distance: distance,
                ligandPosition: ligandPosition, proteinPosition: proteinPosition))
            idCounter += 1
        }

        /// Lateral displacement of `other` from the axis through `origin` along
        /// `normal` — the stacking offset (Salentin 2015). Mirrors
        /// `drusinaLateralOffset`.
        func lateralOffset(_ origin: SIMD3<Float>, _ normal: SIMD3<Float>,
                           _ other: SIMD3<Float>) -> Float {
            let delta = other - origin
            let axial = simd_dot(delta, normal)
            return max(simd_length_squared(delta) - axial * axial, 0).squareRoot()
        }

        /// Verdonk 2003 ramp, mirroring `drusinaRamp`. Used here only to rank
        /// candidate partners for a site that can take just one.
        func ramp(_ x: Float, _ inner: Float, _ outer: Float) -> Float {
            let ax = abs(x)
            if ax <= inner { return 1 }
            if ax >= outer { return 0 }
            return (outer - ax) / (outer - inner)
        }

        /// Angular gate: 0 at the criterion threshold, 1 at perfect alignment.
        /// Linear where the kernel's `drusinaAngleGate` is cubic — it is used
        /// only to rank candidate partners, and both are monotone in the same
        /// variable, so they pick the same winner.
        func angleGate(_ value: Float, _ threshold: Float, _ ideal: Float) -> Float {
            min(max((value - threshold) / (ideal - threshold), 0), 1)
        }

        /// One slot per ring face (true = along +normal, false = along -normal).
        struct BestPartner {
            var score: Float = 0
            var ligandAtomIndex = 0
            var proteinAtomIndex = 0
            var ligandPosition = SIMD3<Float>.zero
            var proteinPosition = SIMD3<Float>.zero
            var distance: Float = 0

            mutating func offer(_ score: Float, ligandAtomIndex: Int, proteinAtomIndex: Int,
                                ligandPosition: SIMD3<Float>, proteinPosition: SIMD3<Float>,
                                distance: Float) {
                guard score > self.score else { return }
                self = BestPartner(score: score, ligandAtomIndex: ligandAtomIndex,
                                   proteinAtomIndex: proteinAtomIndex,
                                   ligandPosition: ligandPosition,
                                   proteinPosition: proteinPosition, distance: distance)
            }
        }

        func emitBest(_ type: MolecularInteraction.InteractionType, _ slots: [BestPartner]) {
            for slot in slots where slot.score > 0 {
                emit(type, ligandAtomIndex: slot.ligandAtomIndex,
                     proteinAtomIndex: slot.proteinAtomIndex, distance: slot.distance,
                     ligandPosition: slot.ligandPosition, proteinPosition: slot.proteinPosition)
            }
        }

        // ---- π-π stacking: best face-to-face per ring face + best T-shaped ----
        // Bissantz 2010, McGaughey 1998; offset cap from Salentin 2015.
        for ligRing in ligandRings {
            var faceUp = BestPartner(), faceDown = BestPartner(), edge = BestPartner()
            let ligAtom = ligRing.atomIndices.first ?? 0
            for protRing in proteinRings {
                let d = simd_distance(ligRing.centroid, protRing.centroid)
                guard d >= kICPiStackDistMin, d <= kICPiStackEdgeDistMax else { continue }
                let dotN = abs(simd_dot(ligRing.normal, protRing.normal))
                let protAtom = protRing.atomIndices.first ?? 0

                if dotN >= kICPiStackFaceCosMin, d <= kICPiStackFaceDistMax,
                   lateralOffset(ligRing.centroid, ligRing.normal, protRing.centroid)
                        <= kICPiStackOffsetMax {
                    let score = angleGate(dotN, kICPiStackFaceCosMin, 1)
                        * ramp(d - kICPiStackFaceDistOpt,
                               kICPiStackFaceRampInner, kICPiStackFaceRampOuter)
                    let above = simd_dot(protRing.centroid - ligRing.centroid, ligRing.normal) >= 0
                    if above {
                        faceUp.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: protAtom,
                                     ligandPosition: ligRing.centroid,
                                     proteinPosition: protRing.centroid, distance: d)
                    } else {
                        faceDown.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: protAtom,
                                       ligandPosition: ligRing.centroid,
                                       proteinPosition: protRing.centroid, distance: d)
                    }
                } else if dotN <= kICPiStackEdgeCosMax {
                    let score = angleGate(dotN, kICPiStackEdgeCosMax, 0)
                        * ramp(d - kICPiStackEdgeDistOpt,
                               kICPiStackEdgeRampInner, kICPiStackEdgeRampOuter)
                    edge.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: protAtom,
                               ligandPosition: ligRing.centroid,
                               proteinPosition: protRing.centroid, distance: d)
                }
            }
            emitBest(.piStack, [faceUp, faceDown, edge])
        }

        // ---- π-cation: ligand ring ↔ protein cation (best per ring face) ----
        // Gallivan 1999, Bissantz 2010.
        for ligRing in ligandRings {
            var up = BestPartner(), down = BestPartner()
            let ligAtom = ligRing.atomIndices.first ?? 0
            for (pi, protAtom) in proteinAtoms.enumerated() where isCationSite(protAtom) {
                let d = simd_distance(ligRing.centroid, protAtom.position)
                guard d <= kICPiCationDistMax,
                      lateralOffset(ligRing.centroid, ligRing.normal, protAtom.position)
                        <= kICPiCationOffsetMax else { continue }
                let cosA = abs(simd_dot(simd_normalize(protAtom.position - ligRing.centroid),
                                        ligRing.normal))
                guard cosA >= kICPiCationCosMin else { continue }
                let score = angleGate(cosA, kICPiCationCosMin, 1)
                    * ramp(d - kICPiCationDistOpt, kICPiCationRampInner, kICPiCationRampOuter)
                let above = simd_dot(protAtom.position - ligRing.centroid, ligRing.normal) >= 0
                if above {
                    up.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: pi,
                             ligandPosition: ligRing.centroid,
                             proteinPosition: protAtom.position, distance: d)
                } else {
                    down.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: pi,
                               ligandPosition: ligRing.centroid,
                               proteinPosition: protAtom.position, distance: d)
                }
            }
            emitBest(.piCation, [up, down])
        }

        // ---- π-cation: protein ring ↔ ligand cation (best per ring) ----
        for protRing in proteinRings {
            var best = BestPartner()
            let protAtom = protRing.atomIndices.first ?? 0
            for (li, ligAtom) in ligandAtoms.enumerated() where ligAtom.formalCharge > 0 {
                let lp = ligPositions[li]
                let d = simd_distance(protRing.centroid, lp)
                guard d <= kICPiCationDistMax,
                      lateralOffset(protRing.centroid, protRing.normal, lp)
                        <= kICPiCationOffsetMax else { continue }
                let cosA = abs(simd_dot(simd_normalize(lp - protRing.centroid), protRing.normal))
                guard cosA >= kICPiCationCosMin else { continue }
                let score = angleGate(cosA, kICPiCationCosMin, 1)
                    * ramp(d - kICPiCationDistOpt, kICPiCationRampInner, kICPiCationRampOuter)
                best.offer(score, ligandAtomIndex: li, proteinAtomIndex: protAtom,
                           ligandPosition: lp, proteinPosition: protRing.centroid, distance: d)
            }
            emitBest(.piCation, [best])
        }

        // ---- CH-π: ligand aliphatic C-H ↔ protein aromatic ring ----
        // Nishio 2011, Brandl 2001. The heavy-atom carbon stands in for the
        // C-H vector; one CH per ring face, as in the kernel.
        for protRing in proteinRings {
            var up = BestPartner(), down = BestPartner()
            let protAtom = protRing.atomIndices.first ?? 0
            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard ligAtom.element == .C, ligandHydrophobicC[li],
                      !ligandRingMembers.contains(li) else { continue }
                let lp = ligPositions[li]
                let d = simd_distance(lp, protRing.centroid)
                guard d >= kICCHPiDistMin, d <= kICCHPiDistMax,
                      lateralOffset(protRing.centroid, protRing.normal, lp)
                        <= kICCHPiOffsetMax else { continue }
                let cosA = abs(simd_dot(simd_normalize(lp - protRing.centroid), protRing.normal))
                guard cosA >= kICCHPiCosMin else { continue }
                let score = angleGate(cosA, kICCHPiCosMin, 1)
                    * ramp(d - kICCHPiDistOpt, kICCHPiRampInner, kICCHPiRampOuter)
                let above = simd_dot(lp - protRing.centroid, protRing.normal) >= 0
                if above {
                    up.offer(score, ligandAtomIndex: li, proteinAtomIndex: protAtom,
                             ligandPosition: lp, proteinPosition: protRing.centroid, distance: d)
                } else {
                    down.offer(score, ligandAtomIndex: li, proteinAtomIndex: protAtom,
                               ligandPosition: lp, proteinPosition: protRing.centroid, distance: d)
                }
            }
            emitBest(.chPi, [up, down])
        }

        // ---- Amide-π: protein amide plane ↔ ligand aromatic ring ----
        // Harder 2013, Bissantz 2010. Parallel stacking only; one amide per
        // ring face, as in the kernel.
        if !ligandRings.isEmpty {
            let amides = detectAmidePlanes(atoms: proteinAtoms)
            for ligRing in ligandRings {
                var up = BestPartner(), down = BestPartner()
                let ligAtom = ligRing.atomIndices.first ?? 0
                for amide in amides {
                    let d = simd_distance(ligRing.centroid, amide.centroid)
                    guard d >= kICAmidePiDistMin, d <= kICAmidePiDistMax else { continue }
                    let dotN = abs(simd_dot(ligRing.normal, amide.normal))
                    guard dotN >= kICAmidePiCosMin,
                          lateralOffset(ligRing.centroid, ligRing.normal, amide.centroid)
                            <= kICAmidePiOffsetMax else { continue }
                    let score = angleGate(dotN, kICAmidePiCosMin, 1)
                        * ramp(d - kICAmidePiDistOpt, kICAmidePiRampInner, kICAmidePiRampOuter)
                    let above = simd_dot(amide.centroid - ligRing.centroid, ligRing.normal) >= 0
                    if above {
                        up.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: amide.atomIndex,
                                 ligandPosition: ligRing.centroid,
                                 proteinPosition: amide.centroid, distance: d)
                    } else {
                        down.offer(score, ligandAtomIndex: ligAtom, proteinAtomIndex: amide.atomIndex,
                                   ligandPosition: ligRing.centroid,
                                   proteinPosition: amide.centroid, distance: d)
                    }
                }
                emitBest(.amideStack, [up, down])
            }
        }

        // ---- Halogen bonds: ligand C-X ... protein O/N σ-hole ----
        // Auffinger 2004, Scholfield 2013. Fluorine has no usable σ-hole and is
        // excluded, matching the kernel (which skips elementType 0). One
        // acceptor per halogen: the σ-hole is a single lobe along the C-X axis.
        for (li, ligAtom) in ligandAtoms.enumerated() {
            guard let elementType = halogenElementType(ligAtom.element), elementType != 0,
                  let ci = bondedCarbons(of: li, atoms: ligandAtoms, bonds: ligandBonds,
                                         positions: ligPositions).first else { continue }
            let xPos = ligPositions[li]
            let xToC = simd_normalize(ligPositions[ci] - xPos)
            let optDist = halogenOptimalDistance(elementType)
            var best = BestPartner()
            for pi in grid.neighbors(of: xPos) where proteinIsAcceptor[pi] {
                let pPos = proteinPositions[pi]
                let d = simd_distance(xPos, pPos)
                guard d >= kICHalogenDistMin, d <= kICHalogenDistMax else { continue }
                let cosTheta = simd_dot(xToC, simd_normalize(pPos - xPos))
                guard cosTheta <= kICHalogenCosMax else { continue }
                let score = angleGate(cosTheta, kICHalogenCosMax, -1)
                    * ramp(d - optDist, kICHalogenRampInner, kICHalogenRampOuter)
                best.offer(score, ligandAtomIndex: li, proteinAtomIndex: pi,
                           ligandPosition: xPos, proteinPosition: pPos, distance: d)
            }
            emitBest(.halogen, [best])
        }

        // ---- Chalcogen bonds: ligand C-S ... protein O/N σ-hole ----
        // Beno 2015, Pascoe 2017. One acceptor per S-C σ-hole; a thioether has
        // two C-S bonds and therefore two independent σ-holes.
        for (li, ligAtom) in ligandAtoms.enumerated() where ligAtom.element == .S {
            let sPos = ligPositions[li]
            for ci in bondedCarbons(of: li, atoms: ligandAtoms, bonds: ligandBonds,
                                    positions: ligPositions) {
                let sToC = simd_normalize(ligPositions[ci] - sPos)
                var best = BestPartner()
                for pi in grid.neighbors(of: sPos) where proteinIsAcceptor[pi] {
                    let pPos = proteinPositions[pi]
                    let d = simd_distance(sPos, pPos)
                    guard d >= kICChalcogenDistMin, d <= kICChalcogenDistMax else { continue }
                    let cosTheta = simd_dot(sToC, simd_normalize(pPos - sPos))
                    guard cosTheta <= kICChalcogenCosMax else { continue }
                    let score = angleGate(cosTheta, kICChalcogenCosMax, -1)
                        * ramp(d - kICChalcogenDistOpt,
                               kICChalcogenRampInner, kICChalcogenRampOuter)
                    best.offer(score, ligandAtomIndex: li, proteinAtomIndex: pi,
                               ligandPosition: sPos, proteinPosition: pPos, distance: d)
                }
                emitBest(.chalcogen, [best])
            }
        }

        // ---- Chalcogen bonds: protein Met/Cys S ... ligand O/N σ-hole ----
        for hole in detectSulfurSigmaHoles(atoms: proteinAtoms) {
            var best = BestPartner()
            for li in ligandAtoms.indices where ligandIsAcceptor[li] {
                let lp = ligPositions[li]
                let d = simd_distance(lp, hole.position)
                guard d >= kICChalcogenDistMin, d <= kICChalcogenDistMax else { continue }
                let cosTheta = simd_dot(hole.bondedCDir, simd_normalize(lp - hole.position))
                guard cosTheta <= kICChalcogenCosMax else { continue }
                let score = angleGate(cosTheta, kICChalcogenCosMax, -1)
                    * ramp(d - kICChalcogenDistOpt,
                           kICChalcogenRampInner, kICChalcogenRampOuter)
                best.offer(score, ligandAtomIndex: li, proteinAtomIndex: hole.atomIndex,
                           ligandPosition: lp, proteinPosition: hole.position, distance: d)
            }
            emitBest(.chalcogen, [best])
        }

        if allowedTypes.count < MolecularInteraction.InteractionType.allCases.count {
            result.removeAll { !allowedTypes.contains($0.type) }
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

    /// `IDET_FLAG_C` means "hydrophobic under Vina" (`xsIsHydrophobic`: C_H and
    /// the halogens), not "element is carbon". The kernel's hydrophobic test
    /// reads that one flag, so a polar carbon (bonded to N or O, Vina type
    /// C_P) and sulfur (S_P) must not carry it — Vina scores no hydrophobic
    /// term for either, and the detector may not draw one.
    private func atomFlags(for atom: Atom,
                           hasHBondDonor: Bool,
                           isHydrophobic: Bool,
                           positiveResAtoms: Set<String>,
                           negativeResAtoms: Set<String>,
                           metals: Set<Element>) -> UInt32 {
        var flags: UInt32 = 0
        switch atom.element {
        case .N:  flags |= UInt32(IDET_FLAG_N)
        case .O:  flags |= UInt32(IDET_FLAG_O)
        case .S:  flags |= UInt32(IDET_FLAG_S)
        default:  break
        }
        if isHydrophobic { flags |= UInt32(IDET_FLAG_C) }
        if metals.contains(atom.element) { flags |= UInt32(IDET_FLAG_METAL) }
        let name = atom.name.trimmingCharacters(in: .whitespaces)
        if positiveResAtoms.contains(name) { flags |= UInt32(IDET_FLAG_POS_RES) }
        if negativeResAtoms.contains(name) { flags |= UInt32(IDET_FLAG_NEG_RES) }
        if hasHBondDonor { flags |= UInt32(IDET_FLAG_HB_DONOR) }
        return flags
    }

    func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        ligandHasH: [Bool],
        proteinHasH: [Bool],
        ligandHydrophobic: [Bool],
        proteinHydrophobic: [Bool],
        positiveResAtoms: Set<String>,
        negativeResAtoms: Set<String>,
        metals: Set<Element>
    ) -> [GPUInteraction]? {
        let nLig = ligandAtoms.count
        let nProt = proteinAtoms.count
        guard nLig > 0, nProt > 0, ligandPositions.count >= nLig else { return [] }

        var protGPU = proteinAtoms.enumerated().map { (i, atom) -> InteractionAtomGPU in
            InteractionAtomGPU(
                position: atom.position,
                flags: atomFlags(for: atom,
                                 hasHBondDonor: i < proteinHasH.count && proteinHasH[i],
                                 isHydrophobic: i < proteinHydrophobic.count && proteinHydrophobic[i],
                                 positiveResAtoms: positiveResAtoms,
                                 negativeResAtoms: negativeResAtoms, metals: metals),
                formalCharge: Int32(atom.formalCharge),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        var ligGPU = ligandAtoms.enumerated().map { (i, atom) -> InteractionAtomGPU in
            InteractionAtomGPU(
                position: atom.position,
                flags: atomFlags(for: atom,
                                 hasHBondDonor: i < ligandHasH.count && ligandHasH[i],
                                 isHydrophobic: i < ligandHydrophobic.count && ligandHydrophobic[i],
                                 positiveResAtoms: positiveResAtoms,
                                 negativeResAtoms: negativeResAtoms, metals: metals),
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
        else { return nil }

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
        // A faulted command buffer leaves the counter/output undefined: let the caller
        // fall back to the CPU detector instead of reporting no interactions.
        guard cmdBuf.status == .completed else { return nil }

        let count = min(Int(ctrBuffer.contents().load(as: UInt32.self)), maxInteractions)
        guard count > 0 else { return [] }

        let ptr = outBuffer.contents().bindMemory(to: GPUInteraction.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
