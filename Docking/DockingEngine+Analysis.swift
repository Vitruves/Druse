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
        guard let best = results.map(\.energy).min() else { return [] }
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

    private nonisolated static func leaderClusters(_ results: [DockingResult]) -> [DockingResult] {
        let sorted = results.sorted { $0.energy < $1.energy }
        let positions = sorted.map(\.transformedAtomPositions)
        let duplicateSumSq = { (atomCount: Int) in outputDuplicateRMSD * outputDuplicateRMSD * Float(atomCount) }

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
            let within = withinRMSDFlags(
                leader: positions[leader], candidates: candidates,
                positions: positions, threshold: clusterRMSD
            )

            var kept = [leader]
            var remaining: [Int] = []
            remaining.reserveCapacity(candidates.count)
            for (c, j) in candidates.enumerated() {
                guard within[c] else {
                    remaining.append(j)
                    continue
                }
                let maxSumSq = duplicateSumSq(positions[j].count)
                if kept.contains(where: { isWithinRMSD(positions[$0], positions[j], maxSumSq: maxSumSq) }) {
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

    /// For each candidate pose, whether its RMSD to `leader` is below `threshold`.
    /// Parallelised across worker threads for large candidate sets.
    private nonisolated static func withinRMSDFlags(
        leader: [SIMD3<Float>],
        candidates: [Int],
        positions: [[SIMD3<Float>]],
        threshold: Float
    ) -> [Bool] {
        let maxSumSq = threshold * threshold * Float(leader.count)
        var flags = [Bool](repeating: false, count: candidates.count)
        if candidates.count >= 2048 {
            flags.withUnsafeMutableBufferPointer { buf in
                nonisolated(unsafe) let out = buf
                DispatchQueue.concurrentPerform(iterations: candidates.count) { c in
                    out[c] = isWithinRMSD(leader, positions[candidates[c]], maxSumSq: maxSumSq)
                }
            }
        } else {
            for c in candidates.indices {
                flags[c] = isWithinRMSD(leader, positions[candidates[c]], maxSumSq: maxSumSq)
            }
        }
        return flags
    }

    /// RMSD(a, b) < threshold, expressed as Σ|a_i - b_i|² < threshold² · N, with early exit.
    /// Poses with mismatched or empty atom lists never cluster together.
    private nonisolated static func isWithinRMSD(
        _ a: [SIMD3<Float>], _ b: [SIMD3<Float>], maxSumSq: Float
    ) -> Bool {
        guard a.count == b.count, !a.isEmpty else { return false }
        var sum: Float = 0
        for i in 0..<a.count {
            sum += simd_distance_squared(a[i], b[i])
            if sum >= maxSumSq { return false }
        }
        return true
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
        var result: [MolecularInteraction] = []
        var idCounter = 0

        let positiveResAtoms: Set<String> = ["NZ", "NH1", "NH2", "NE"]
        let negativeResAtoms: Set<String> = ["OD1", "OD2", "OE1", "OE2"]
        let metals: Set<Element> = [.Fe, .Zn, .Ca, .Mg, .Mn, .Cu]

        // Donor flags: an atom can DONATE to an H-bond only if it has at
        // least one attached hydrogen. For ligand atoms we derive this from
        // the bond graph (bond order sum + formal charge → implicit H count).
        // For protein atoms we use a residue-name lookup since explicit
        // bonds are not always available on the Atom struct.
        let ligandHasH: [Bool] = computeLigandHBondDonorFlags(
            atoms: ligandAtoms, bonds: ligandBonds)
        let proteinHasH: [Bool] = proteinAtoms.map(proteinHBondDonorFlag)

        if let gpu = InteractionDetectorGPU.shared,
           let gpuResults = gpu.detect(
                ligandAtoms: ligandAtoms,
                ligandPositions: ligandPositions,
                proteinAtoms: proteinAtoms,
                ligandHasH: ligandHasH,
                proteinHasH: proteinHasH,
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
            // CPU fallback with spatial grid (also used when the GPU path fails)
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
                        // At least one side must carry an explicit/implicit H
                        // to actually donate. Two pure acceptors (e.g. neutral
                        // tertiary amine + deprotonated carboxylate) cannot
                        // form an H-bond.
                        let donor = (li < ligandHasH.count && ligandHasH[li])
                                 || (pi < proteinHasH.count && proteinHasH[pi])
                        if ligDA && proDA && donor {
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

    private func atomFlags(for atom: Atom,
                           hasHBondDonor: Bool,
                           positiveResAtoms: Set<String>,
                           negativeResAtoms: Set<String>,
                           metals: Set<Element>) -> UInt32 {
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
        if hasHBondDonor { flags |= UInt32(IDET_FLAG_HB_DONOR) }
        return flags
    }

    func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        ligandHasH: [Bool],
        proteinHasH: [Bool],
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
