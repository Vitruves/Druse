// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - DockingEngine Execution (runDocking, GA loop, GPU dispatch, reranking)

@MainActor
extension DockingEngine {

    // MARK: - Rerank Helpers

    private struct RerankRNG {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed &+ 0x9E3779B97F4A7C15
        }

        mutating func nextUInt32() -> UInt32 {
            state = state &* 2862933555777941757 &+ 3037000493
            return UInt32(truncatingIfNeeded: state >> 16)
        }

        mutating func nextFloat() -> Float {
            Float(nextUInt32()) / Float(UInt32.max)
        }

        mutating func signed(amplitude: Float) -> Float {
            (nextFloat() * 2 - 1) * amplitude
        }

        mutating func vectorInUnitSphere(scale: Float) -> SIMD3<Float> {
            for _ in 0..<16 {
                let v = SIMD3<Float>(
                    signed(amplitude: 1),
                    signed(amplitude: 1),
                    signed(amplitude: 1)
                )
                let len2 = simd_length_squared(v)
                if len2 > 1e-4, len2 <= 1 {
                    return v * scale
                }
            }
            return SIMD3<Float>(scale, 0, 0)
        }
    }

    private func wrappedAngle(_ angle: Float) -> Float {
        var wrapped = angle
        while wrapped > .pi { wrapped -= 2 * .pi }
        while wrapped < -.pi { wrapped += 2 * .pi }
        return wrapped
    }

    private struct RerankProfile {
        var variantsPerCluster: Int
        var localSearchSteps: Int
        var translationJitter: Float
        var rotationJitter: Float
        var torsionJitter: Float

        var runsLocalOptimization: Bool { localSearchSteps > 0 }
    }

    private func rerankProfile(for scoringMethod: ScoringMethod) -> RerankProfile {
        switch scoringMethod {
        case .drusina:
            // Drusina rerank should stay close to sampled basin leaders. Its score is not
            // aligned with the Vina-style local optimizer, so aggressive second-pass
            // refinement tends to move poses off the Drusina-preferred basin.
            return RerankProfile(
                variantsPerCluster: 1,
                localSearchSteps: 0,
                translationJitter: 0,
                rotationJitter: 0,
                torsionJitter: 0
            )
        default:
            return RerankProfile(
                variantsPerCluster: max(config.explicitRerankVariantsPerCluster, 1),
                localSearchSteps: max(config.explicitRerankLocalSearchSteps, 1),
                translationJitter: 0.75,
                rotationJitter: 0.18,
                torsionJitter: 0.30
            )
        }
    }

    func makeDockPose(from result: DockingResult) -> DockPose {
        var pose = DockPose()
        pose.translation = result.pose.translation
        pose.energy = result.energy
        pose.rotation = SIMD4<Float>(
            result.pose.rotation.imag.x,
            result.pose.rotation.imag.y,
            result.pose.rotation.imag.z,
            result.pose.rotation.real
        )
        let torsions = result.pose.torsions
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for i in 0..<min(torsions.count, 32) {
                    buffer[i] = torsions[i]
                }
            }
        }
        pose.numTorsions = Int32(min(torsions.count, 32))
        pose.generation = Int32(result.generation)
        pose.stericEnergy = result.stericEnergy
        pose.hydrophobicEnergy = result.hydrophobicEnergy
        pose.hbondEnergy = result.hbondEnergy
        pose.torsionPenalty = result.torsionPenalty
        pose.clashPenalty = 0
        pose.drusinaCorrection = 0
        pose.constraintPenalty = 0
        return pose
    }

    private func makeRerankSeedPose(
        from result: DockingResult,
        variantIndex: Int,
        profile: RerankProfile
    ) -> DockPose {
        var pose = makeDockPose(from: result)
        guard variantIndex > 0 else { return pose }

        let seed = UInt64(bitPattern: Int64(result.id &* 1_315_423_911
            ^ result.clusterID &* 374_761_393
            ^ variantIndex &* 668_265_263))
        var rng = RerankRNG(seed: seed)

        pose.translation += rng.vectorInUnitSphere(scale: profile.translationJitter)

        let axis = simd_normalize(rng.vectorInUnitSphere(scale: 1))
        let deltaRotation = simd_quatf(angle: rng.signed(amplitude: profile.rotationJitter), axis: axis)
        let currentRotation = simd_quatf(ix: pose.rotation.x, iy: pose.rotation.y, iz: pose.rotation.z, r: pose.rotation.w)
        let updatedRotation = deltaRotation * currentRotation
        pose.rotation = SIMD4<Float>(
            updatedRotation.imag.x,
            updatedRotation.imag.y,
            updatedRotation.imag.z,
            updatedRotation.real
        )

        let torsionCount = max(0, min(Int(pose.numTorsions), 32))
        guard torsionCount > 0 else { return pose }
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for index in 0..<torsionCount {
                    buffer[index] = wrappedAngle(buffer[index] + rng.signed(amplitude: profile.torsionJitter))
                }
            }
        }
        return pose
    }

    // MARK: - Explicit Reranking

    func scorePopulationExplicit(
        buffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        let simdWidth = explicitScorePipeline.threadExecutionWidth
        let totalThreads = max(populationSize, 1) * simdWidth
        let tgSize = MTLSize(width: simdWidth, height: 1, depth: 1)
        let tgCount = MTLSize(width: (totalThreads + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligandAtomBuffer!, 1), (proteinAtomBuffer!, 2),
            (gridParamsBuffer, 3), (gaParamsBuffer, 4),
            (torsionEdgeBuffer!, 5), (movingIndicesBuffer!, 6),
            (intraPairsBuffer!, 7)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: explicitScorePipeline, buffers: buffers,
                        threadGroups: tgCount, threadGroupSize: tgSize)
    }

    /// Decompose Vina intermolecular energy into steric/hydrophobic/hbond via explicit
    /// pair evaluation.  Only updates the three decomposition fields; total energy is
    /// left untouched so rankings are not affected.
    func decomposeVinaEnergy(
        buffer: MTLBuffer,
        populationSize: Int
    ) {
        guard let ligBuf = ligandAtomBuffer,
              let paBuf = proteinAtomBuffer,
              let gpBuf = gridParamsBuffer,
              let gaBuf = gaParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer else { return }

        let simdWidth = decomposeVinaEnergyPipeline.threadExecutionWidth
        let totalThreads = max(populationSize, 1) * simdWidth
        let tgSize = MTLSize(width: simdWidth, height: 1, depth: 1)
        let tgCount = MTLSize(width: (totalThreads + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        let buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligBuf, 1), (paBuf, 2),
            (gpBuf, 3), (gaBuf, 4),
            (teBuf, 5), (miBuf, 6)
        ]
        dispatchCompute(pipeline: decomposeVinaEnergyPipeline, buffers: buffers,
                        threadGroups: tgCount, threadGroupSize: tgSize)
    }

    func localOptimizeGrid(
        buffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        let tgSize: MTLSize
        let tgCount: MTLSize
        if localSearchIsSIMD {
            tgSize = MTLSize(width: 32, height: 1, depth: 1)
            tgCount = MTLSize(width: max(populationSize, 1), height: 1, depth: 1)
        } else {
            tgSize = MTLSize(width: min(max(populationSize, 1), 64), height: 1, depth: 1)
            tgCount = MTLSize(width: (max(populationSize, 1) + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        }
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer, 4), (gaParamsBuffer, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (intraPairsBuffer!, 8)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: activeLocalSearchPipeline, buffers: buffers,
                        threadGroups: tgCount, threadGroupSize: tgSize)
    }

    func rerankClusterRepresentativesExplicit(
        _ results: [DockingResult],
        scoringMethod: ScoringMethod,
        ligandAtoms: [Atom],
        centroid: SIMD3<Float>
    ) -> [DockingResult] {
        guard config.explicitRerankTopClusters > 0,
              proteinAtomBuffer != nil,
              let gridParamsBuffer,
              let gaParamsBuffer,
              !results.isEmpty else {
            return results
        }

        let grouped = Dictionary(grouping: results, by: \.clusterID)
        let leaders = results
            .filter { $0.clusterRank == 0 }
            .sorted { $0.energy < $1.energy }
        guard !leaders.isEmpty else { return results }

        let rerankCount = min(config.explicitRerankTopClusters, leaders.count)
        let rerankProfile = rerankProfile(for: scoringMethod)
        let variantsPerCluster = rerankProfile.variantsPerCluster
        let rerankLeaders = Array(leaders.prefix(rerankCount))
        var representativePoses: [DockPose] = []
        var variantClusterIDs: [Int] = []
        representativePoses.reserveCapacity(rerankLeaders.count * variantsPerCluster)
        variantClusterIDs.reserveCapacity(rerankLeaders.count * variantsPerCluster)
        for leader in rerankLeaders {
            for variantIndex in 0..<variantsPerCluster {
                representativePoses.append(makeRerankSeedPose(
                    from: leader,
                    variantIndex: variantIndex,
                    profile: rerankProfile
                ))
                variantClusterIDs.append(leader.clusterID)
            }
        }

        let repBuffer = device.makeBuffer(
            bytes: &representativePoses,
            length: representativePoses.count * MemoryLayout<DockPose>.stride,
            options: .storageModeShared
        )
        guard let repBuffer else { return results }

        let currentGA = gaParamsBuffer.contents().bindMemory(to: GAParams.self, capacity: 1).pointee
        var rerankGA = currentGA
        rerankGA.populationSize = UInt32(representativePoses.count)
        rerankGA.localSearchSteps = UInt32(rerankProfile.localSearchSteps)
        let rerankGABuffer = device.makeBuffer(
            bytes: &rerankGA,
            length: MemoryLayout<GAParams>.stride,
            options: .storageModeShared
        )
        guard let rerankGABuffer else { return results }

        if rerankProfile.runsLocalOptimization {
            localOptimizeGrid(
                buffer: repBuffer,
                gridParamsBuffer: gridParamsBuffer,
                gaParamsBuffer: rerankGABuffer,
                populationSize: representativePoses.count
            )
        }
        if scoringMethod == .drusina {
            let tgSize = MTLSize(width: min(representativePoses.count, 256), height: 1, depth: 1)
            let tgCount = MTLSize(width: (representativePoses.count + 255) / 256, height: 1, depth: 1)
            scoreDrusina(
                buffer: repBuffer,
                gaParamsBuffer: rerankGABuffer,
                tg: tgCount,
                tgs: tgSize
            )
        } else if scoringMethod == .pignet2 {
            let tgSize = MTLSize(width: 1, height: 1, depth: 1)
            let tgCount = MTLSize(width: representativePoses.count, height: 1, depth: 1)
            scorePIGNet2(
                buffer: repBuffer,
                gaParamsBuffer: rerankGABuffer,
                tg: tgCount,
                tgs: tgSize
            )
        } else {
            scorePopulationExplicit(
                buffer: repBuffer,
                gridParamsBuffer: gridParamsBuffer,
                gaParamsBuffer: rerankGABuffer,
                populationSize: representativePoses.count
            )
        }

        let rescoredLeaders = extractAllResults(
            from: repBuffer,
            ligandAtoms: ligandAtoms,
            centroid: centroid,
            idOffset: 0,
            sortByEnergy: false
        )
        guard rescoredLeaders.count == representativePoses.count else { return results }

        var representativeByCluster: [Int: DockingResult] = [:]
        for (index, rescored) in rescoredLeaders.enumerated() {
            let sourceClusterID = variantClusterIDs[index]
            var updated = rescored
            updated.clusterID = sourceClusterID
            updated.clusterRank = 0
            if updated.energy < (representativeByCluster[sourceClusterID]?.energy ?? .infinity) {
                representativeByCluster[sourceClusterID] = updated
            }
        }

        let sortedClusterIDs = leaders
            .map(\.clusterID)
            .sorted {
                let lhs = representativeByCluster[$0]?.energy ?? grouped[$0]?.first?.energy ?? .infinity
                let rhs = representativeByCluster[$1]?.energy ?? grouped[$1]?.first?.energy ?? .infinity
                return lhs < rhs
            }

        var reranked: [DockingResult] = []
        reranked.reserveCapacity(results.count)

        for (newClusterID, oldClusterID) in sortedClusterIDs.enumerated() {
            guard let members = grouped[oldClusterID] else { continue }
            let originalLeader = members.first { $0.clusterRank == 0 }
            var leader = representativeByCluster[oldClusterID] ?? originalLeader ?? members[0]
            leader.clusterID = newClusterID
            leader.clusterRank = 0
            reranked.append(leader)

            var rank = 1
            for member in members.sorted(by: { $0.energy < $1.energy }) where member.clusterRank != 0 {
                var updated = member
                updated.clusterID = newClusterID
                updated.clusterRank = rank
                rank += 1
                reranked.append(updated)
            }
        }

        return reranked
    }

    // MARK: - Run Docking

    func runDocking(
        ligand: Molecule, pocket: BindingPocket, config: DockingConfig = DockingConfig(),
        scoringMethod: ScoringMethod = .vina
    ) async -> [DockingResult] {
        guard !isRunning else { return [] }
        self.config = config
        isRunning = true
        currentGeneration = 0
        bestEnergy = .infinity

        // Diffusion-guided search needs an attention-producing learned scorer; Vina/Drusina
        // grids don't expose per-atom gradients. Promote scoring to DruseAF so the user
        // doesn't have to remember to flip both knobs.
        var scoringMethod = scoringMethod
        if config.searchMethod == .diffusionGuided && scoringMethod != .druseAffinity {
            ActivityLog.shared.info(
                "[Engine] Diffusion search requires DruseAF scoring — promoting from \(scoringMethod.rawValue) to Druse Affinity",
                category: .dock
            )
            scoringMethod = .druseAffinity
        }

        let preparedLigand = prepareLigandGeometry(ligand)
        let heavyAtoms = preparedLigand.heavyAtoms
        let heavyBonds = preparedLigand.heavyBonds
        let centroid = preparedLigand.centroid
        var gpuLigAtoms = preparedLigand.gpuAtoms

        ActivityLog.shared.info(
            "[Engine] Ligand geometry: \(heavyAtoms.count) heavy atoms, \(heavyBonds.count) heavy bonds, " +
            "\(gpuLigAtoms.count) GPU atoms, centroid=(\(String(format: "%.2f, %.2f, %.2f", centroid.x, centroid.y, centroid.z)))",
            category: .dock
        )

        guard gpuLigAtoms.count > 0 else {
            ActivityLog.shared.error("[Engine] Ligand has no valid heavy atoms — cannot dock", category: .dock)
            isRunning = false
            return []
        }
        guard gpuLigAtoms.count <= 128 else {
            ActivityLog.shared.error("[Engine] Ligand has \(gpuLigAtoms.count) heavy atoms, exceeding the 128-atom GPU limit", category: .dock)
            isRunning = false
            return []
        }

        var ligMin = SIMD3<Float>(repeating: .infinity)
        var ligMax = SIMD3<Float>(repeating: -.infinity)
        var ligandRadiusSquared: Float = 0
        for a in gpuLigAtoms {
            ligMin = simd_min(ligMin, a.position)
            ligMax = simd_max(ligMax, a.position)
            ligandRadiusSquared = max(ligandRadiusSquared, simd_length_squared(a.position))
        }
        let ligandHalfExtent = (ligMax - ligMin) * 0.5
        let ligandRadius = max(sqrt(ligandRadiusSquared / Float(max(gpuLigAtoms.count, 1))), 1.0)

        let requiredVinaTypes = Array(Set(gpuLigAtoms.map(\.vinaType).filter { $0 >= 0 && $0 <= maxSupportedVinaType })).sorted()
        if requiredVinaTypes.isEmpty {
            ActivityLog.shared.warn("[Engine] Ligand has no recognized Vina atom types (\(gpuLigAtoms.count) atoms, types: \(Set(gpuLigAtoms.map(\.vinaType)).sorted())) — docking may produce zero scores", category: .dock)
        }
        if let protein = proteinStructure ?? (!proteinAtoms.isEmpty ? Molecule(name: "cached", atoms: proteinAtoms, bonds: [], title: "") : nil) {
            computeGridMaps(
                protein: protein,
                pocket: pocket,
                spacing: config.gridSpacing,
                ligandExtent: ligandHalfExtent,
                requiredVinaTypes: requiredVinaTypes
            )
        }

        // Reuse ligand atom buffer if large enough, otherwise reallocate
        let ligAtomSize = gpuLigAtoms.count * MemoryLayout<DockLigandAtom>.stride
        if ligAtomSize > lastLigandAtomBufferCapacity {
            ligandAtomBuffer = device.makeBuffer(bytes: &gpuLigAtoms, length: ligAtomSize, options: .storageModeShared)
            lastLigandAtomBufferCapacity = ligAtomSize
        } else {
            ligandAtomBuffer?.contents().copyMemory(from: &gpuLigAtoms, byteCount: ligAtomSize)
        }

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1),
                atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count),
                movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map { Int32($0) })
        }
        let numTorsions = min(torsionEdges.count, 32)

        if torsionEdges.isEmpty {
            torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0))
        }
        if movingIndices.isEmpty {
            movingIndices.append(0)
        }
        let torsionSize = torsionEdges.count * MemoryLayout<TorsionEdge>.stride
        if torsionSize > lastTorsionEdgeBufferCapacity {
            torsionEdgeBuffer = device.makeBuffer(bytes: &torsionEdges, length: torsionSize, options: .storageModeShared)
            lastTorsionEdgeBufferCapacity = torsionSize
        } else {
            torsionEdgeBuffer?.contents().copyMemory(from: &torsionEdges, byteCount: torsionSize)
        }
        let movingSize = movingIndices.count * MemoryLayout<Int32>.stride
        if movingSize > lastMovingIndicesBufferCapacity {
            movingIndicesBuffer = device.makeBuffer(bytes: &movingIndices, length: movingSize, options: .storageModeShared)
            lastMovingIndicesBufferCapacity = movingSize
        } else {
            movingIndicesBuffer?.contents().copyMemory(from: &movingIndices, byteCount: movingSize)
        }

        // Build flat pair list for intramolecular energy
        var excluded = Set<UInt32>()
        var adj: [[Int]] = Array(repeating: [], count: gpuLigAtoms.count)
        for bond in heavyBonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < gpuLigAtoms.count, b < gpuLigAtoms.count else { continue }
            adj[a].append(b)
            adj[b].append(a)
        }

        for i in 0..<gpuLigAtoms.count {
            for j in adj[i] where j > i {
                excluded.insert(UInt32(i) | (UInt32(j) << 16))
                for k in adj[j] where k > i && k != i {
                    let lo = min(i, k), hi = max(i, k)
                    excluded.insert(UInt32(lo) | (UInt32(hi) << 16))
                }
                for k in adj[i] where k != j {
                    let lo = min(j, k), hi = max(j, k)
                    excluded.insert(UInt32(lo) | (UInt32(hi) << 16))
                }
            }
        }

        var pairList = [UInt32]()
        pairList.reserveCapacity(gpuLigAtoms.count * (gpuLigAtoms.count - 1) / 2)
        for i in 0..<gpuLigAtoms.count {
            for j in (i + 1)..<gpuLigAtoms.count {
                let packed = UInt32(i) | (UInt32(j) << 16)
                if !excluded.contains(packed) {
                    pairList.append(packed)
                }
            }
        }

        let pairSize = max(pairList.count, 1) * MemoryLayout<UInt32>.stride
        if pairSize > lastIntraPairsBufferCapacity {
            intraPairsBuffer = device.makeBuffer(bytes: &pairList, length: pairSize, options: .storageModeShared)
            lastIntraPairsBufferCapacity = pairSize
        } else {
            intraPairsBuffer?.contents().copyMemory(from: &pairList, byteCount: pairSize)
        }
        let referenceIntraEnergy = intramolecularReferenceEnergy(
            ligandAtoms: gpuLigAtoms,
            pairList: pairList
        )

        ActivityLog.shared.info(
            "[Engine] Torsion tree: \(numTorsions) rotatable bonds, \(movingIndices.count) moving atom indices, refIntraE=\(String(format: "%.3f", referenceIntraEnergy))",
            category: .dock
        )

        // Prepare Drusina buffers (always — needed for per-term decomposition diagnostics
        // even when the GA uses a different scoring method)
        let useDrusina = scoringMethod == .drusina && drusinaScorePipeline != nil
        prepareDrusinaBuffers(
            ligandAtoms: heavyAtoms,
            ligandBonds: heavyBonds,
            gpuLigAtoms: &gpuLigAtoms,
            centroid: centroid)

        // Prepare DruseAF Metal ML scoring buffers
        let useDruseAF = scoringMethod == .druseAffinity
            && druseAFSetupPipeline != nil && druseAFEncodePipeline != nil
            && druseAFScorePipeline != nil && druseAFWeights != nil
        if useDruseAF {
            prepareDruseAFBuffers(
                ligandAtoms: heavyAtoms,
                ligandBonds: heavyBonds,
                gpuLigAtoms: gpuLigAtoms,
                pocket: pocket,
                popSize: config.populationSize)
        }

        // Prepare PIGNet2 physics-informed GNN scoring buffers.
        // PIGNet2 is a pose scorer, not a geometry-stable docking potential, so
        // docking search stays Vina-guided and PIGNet2 is applied as a reranker.
        let usePIGNet2Rescore = scoringMethod == .pignet2
            && pignet2SetupPipeline != nil && pignet2EncodePipeline != nil
            && pignet2ScorePipeline != nil && pignet2Weights != nil
        if usePIGNet2Rescore {
            preparePIGNet2Buffers(
                ligandAtoms: heavyAtoms,
                ligandBonds: heavyBonds,
                gpuLigAtoms: gpuLigAtoms,
                pocket: pocket,
                popSize: config.populationSize)
        } else if scoringMethod == .pignet2 {
            // User picked PIGNet2 but the kernel failed to JIT-compile (typically
            // XPC_ERROR_CONNECTION_INTERRUPTED on the score kernel). Make this
            // visible; silently falling back to Vina is misleading.
            var missing: [String] = []
            if pignet2SetupPipeline == nil  { missing.append("setup") }
            if pignet2EncodePipeline == nil { missing.append("encode") }
            if pignet2ScorePipeline == nil  { missing.append("score") }
            if pignet2Weights == nil        { missing.append("weights") }
            let detail = missing.isEmpty ? "unknown reason" : "missing \(missing.joined(separator: ", "))"
            ActivityLog.shared.error(
                "PIGNet2 unavailable (\(detail)) - docking with Vina instead. Check Xcode console for MTLCompiler XPC errors and try /Library/Caches/com.apple.metal cleanup.",
                category: .dock
            )
        }
        ActivityLog.shared.info("[Engine] Scoring method: \(scoringMethod.rawValue), Drusina: \(useDrusina), DruseAF: \(useDruseAF), PIGNet2 rerank: \(usePIGNet2Rescore)", category: .dock)

        let popSize = config.populationSize
        let poseSize = popSize * MemoryLayout<DockPose>.stride
        if poseSize > lastPopulationBufferCapacity {
            populationBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            offspringBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            bestPopulationBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            lastPopulationBufferCapacity = poseSize
        }

        if let bestBuf = bestPopulationBuffer {
            let totalPoseSlots = bestBuf.length / MemoryLayout<DockPose>.stride
            let ptr = bestBuf.contents().bindMemory(to: DockPose.self, capacity: totalPoseSlots)
            for i in 0..<totalPoseSlots {
                ptr[i].energy = .infinity
            }
        }

        var gaParams = GAParams(
            populationSize: UInt32(popSize),
            numLigandAtoms: UInt32(gpuLigAtoms.count),
            numTorsions: UInt32(numTorsions),
            generation: 0,
            localSearchSteps: UInt32(max(config.localSearchSteps, 1)),
            mutationRate: config.mutationRate,
            crossoverRate: config.crossoverRate,
            translationStep: config.translationStep,
            rotationStep: config.rotationStep,
            torsionStep: config.torsionStep,
            gridSpacing: config.gridSpacing,
            ligandRadius: ligandRadius,
            mcTemperature: config.mcTemperature,
            referenceIntraEnergy: referenceIntraEnergy,
            numIntraPairs: UInt32(pairList.count),
            runSeed: 0,
            torsionExactFraction: config.torsionExactFraction,
            torsionLocalFraction: config.torsionLocalFraction,
            torsionLocalAmplitude: config.torsionLocalAmplitude,
            torsionRandomResetProbability: config.torsionRandomResetProbability,
            torsionPerturbationScale: config.torsionPerturbationScale,
            _pad0: 0
        )
        gaParamsBuffer = device.makeBuffer(bytes: &gaParams, length: MemoryLayout<GAParams>.stride, options: .storageModeShared)

        gaParamsRing = (0..<3).compactMap { _ in
            device.makeBuffer(length: MemoryLayout<GAParams>.stride, options: .storageModeShared)
        }

        let tgSize = MTLSize(width: min(popSize, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (popSize + 255) / 256, height: 1, depth: 1)

        if !config.enableFlexibility {
            gaParams.numTorsions = 0
            gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)
        }

        ActivityLog.shared.info(
            "[Engine] GA: pop=\(popSize), torsions=\(gaParams.numTorsions), " +
            "ligAtoms=\(gaParams.numLigandAtoms), lsSteps=\(gaParams.localSearchSteps), " +
            "T=\(String(format: "%.3f", gaParams.mcTemperature)), flex=\(config.enableFlexibility)",
            category: .dock
        )
        ActivityLog.shared.info(
            "[Engine] GPU dispatch: tgSize=\(tgSize.width), tgCount=\(tgCount.width), poseStride=\(MemoryLayout<DockPose>.stride) bytes",
            category: .dock
        )
        ActivityLog.shared.info(
            "[Engine] Buffers: pop=\(populationBuffer != nil), offspring=\(offspringBuffer != nil), " +
            "best=\(bestPopulationBuffer != nil), ligAtom=\(ligandAtomBuffer != nil), " +
            "grid=\(vinaAffinityGridBuffer != nil), gridParams=\(gridParamsBuffer != nil), " +
            "torsion=\(torsionEdgeBuffer != nil), moving=\(movingIndicesBuffer != nil), " +
            "exclusion=\(intraPairsBuffer != nil), gaParams=\(gaParamsBuffer != nil)",
            category: .dock
        )
        if let flexEng = flexEngine, flexEng.isEnabled {
            ActivityLog.shared.info(
                "[Engine] Flex: atoms=\(flexEng.numFlexAtoms), torsions=\(flexEng.numFlexTorsions), " +
                "chiSlots=\(flexEng.numChiSlots), buffers=(atom=\(flexEng.flexAtomBuffer != nil), " +
                "edge=\(flexEng.flexTorsionEdgeBuffer != nil), moving=\(flexEng.flexMovingIndicesBuffer != nil), " +
                "params=\(flexEng.flexParamsBuffer != nil))",
                category: .dock
            )
        }

        // Resolve auto search method
        let resolvedSearchMethod: SearchMethod
        if config.searchMethod == .auto {
            if numTorsions > 10 {
                resolvedSearchMethod = .fragmentBased
            } else if numTorsions > 6 {
                resolvedSearchMethod = .parallelTempering
            } else {
                resolvedSearchMethod = .genetic
            }
            ActivityLog.shared.info("[Engine] Auto search method resolved to: \(resolvedSearchMethod.rawValue) (torsions=\(numTorsions))", category: .dock)
        } else {
            resolvedSearchMethod = config.searchMethod
        }
        ActivityLog.shared.info("[Engine] Search method: \(resolvedSearchMethod.rawValue)", category: .dock)

        let totalRuns = max(config.numRuns, 1)
        var aggregatedResults: [DockingResult] = []
        let localSearchFrequency = max(config.localSearchFrequency, 1)
        let liveUpdateFrequency = max(config.liveUpdateFrequency, 1)

        func emitLiveUpdate(generation: Int) {
            if let best = extractBestPose(from: bestPopulationBuffer, ligandAtoms: heavyAtoms, centroid: centroid) {
                bestEnergy = min(bestEnergy, best.energy)
                let interactions = InteractionDetector.detect(
                    ligandAtoms: heavyAtoms,
                    ligandPositions: best.transformedAtomPositions,
                    proteinAtoms: proteinAtoms,
                    ligandBonds: heavyBonds,
                    scoringMethod: scoringMethod
                )
                onPoseUpdate?(best, interactions)
            }
            onGenerationComplete?(generation, bestEnergy)
        }

        // Validate all critical buffers before entering the GA loop
        guard let popBuf = populationBuffer,
              let offBuf = offspringBuffer,
              let bestBuf = bestPopulationBuffer,
              let ligBuf = ligandAtomBuffer,
              let gpBuf = gridParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer,
              let emBuf = intraPairsBuffer,
              let gaBuf = gaParamsBuffer
        else {
            ActivityLog.shared.error("[Engine] Critical buffer nil — cannot start GA. pop=\(populationBuffer != nil) grid=\(gridParamsBuffer != nil) ligand=\(ligandAtomBuffer != nil) torsion=\(torsionEdgeBuffer != nil) moving=\(movingIndicesBuffer != nil) exclusion=\(intraPairsBuffer != nil) ga=\(gaParamsBuffer != nil)", category: .dock)
            isRunning = false
            return []
        }
        let affinityBuf = vinaAffinityGridBuffer
        let typeIdxBuf = vinaTypeIndexBuffer
        if !useDruseAF && (affinityBuf == nil || typeIdxBuf == nil) {
            ActivityLog.shared.error("[Engine] Vina affinity grids nil — cannot score (affinity=\(affinityBuf != nil) typeIdx=\(typeIdxBuf != nil))", category: .dock)
            isRunning = false
            return []
        }

        // =====================================================================
        // MARK: Parallel Tempering / Replica Exchange Monte Carlo
        // =====================================================================
        if resolvedSearchMethod == .parallelTempering,
           let perturbPipe = mcPerturbReplicaPipeline,
           let acceptPipe = metropolisAcceptReplicaPipeline,
           let swapPipe = replicaSwapPipeline,
           let afBuf = affinityBuf,
           let tiBuf = typeIdxBuf {

            let K = min(config.replicaExchange.numReplicas, Int(MAX_REPLICAS))
            let popPerReplica = max(popSize / K, 4)
            let totalPoses = K * popPerReplica
            let totalPoseSize = totalPoses * MemoryLayout<DockPose>.stride

            let replicaPopBuf = device.makeBuffer(length: totalPoseSize, options: .storageModeShared)!
            let replicaOffBuf = device.makeBuffer(length: totalPoseSize, options: .storageModeShared)!
            let replicaBestBuf = device.makeBuffer(length: totalPoseSize, options: .storageModeShared)!

            do {
                let ptr = replicaBestBuf.contents().bindMemory(to: DockPose.self, capacity: totalPoses)
                for i in 0..<totalPoses { ptr[i].energy = .infinity }
            }

            var repParams = ReplicaParams()
            repParams.numReplicas = UInt32(K)
            repParams.populationPerReplica = UInt32(popPerReplica)
            repParams.totalPoses = UInt32(totalPoses)
            repParams.swapGeneration = 0
            let tMin = config.replicaExchange.minTemperature
            let tMax = config.replicaExchange.maxTemperature
            withUnsafeMutablePointer(to: &repParams.temperatures) { ptr in
                ptr.withMemoryRebound(to: Float.self, capacity: Int(MAX_REPLICAS)) { temps in
                    for k in 0..<K {
                        let frac = K > 1 ? Float(k) / Float(K - 1) : 0
                        temps[k] = tMin * pow(tMax / max(tMin, 0.01), frac)
                    }
                    for k in K..<Int(MAX_REPLICAS) {
                        temps[k] = tMax
                    }
                }
            }
            replicaParamsBuffer = device.makeBuffer(bytes: &repParams, length: MemoryLayout<ReplicaParams>.stride, options: .storageModeShared)

            var repGAParams = gaParams
            repGAParams.populationSize = UInt32(totalPoses)
            let repGAParamsRing = (0..<3).compactMap { _ in
                device.makeBuffer(length: MemoryLayout<GAParams>.stride, options: .storageModeShared)
            }

            let repTgSize = MTLSize(width: min(totalPoses, 256), height: 1, depth: 1)
            let repTgCount = MTLSize(width: (totalPoses + 255) / 256, height: 1, depth: 1)

            repGAParams.populationSize = UInt32(totalPoses)
            let initGABuf = device.makeBuffer(bytes: &repGAParams, length: MemoryLayout<GAParams>.stride, options: .storageModeShared)!
            dispatchCompute(pipeline: initPopPipeline, buffers: [
                (replicaPopBuf, 0), (gpBuf, 1), (initGABuf, 2)
            ], threadGroups: repTgCount, threadGroupSize: repTgSize)

            let vinaScoreBuffersRep: [(MTLBuffer, Int)] = [
                (replicaPopBuf, 0), (ligBuf, 1),
                (afBuf, 2), (tiBuf, 3),
                (gpBuf, 4), (initGABuf, 5),
                (teBuf, 6), (miBuf, 7),
                (emBuf, 8)
            ]
            dispatchCompute(pipeline: scorePipeline, buffers: vinaScoreBuffersRep, threadGroups: repTgCount, threadGroupSize: repTgSize)
            copyPoseBuffer(from: replicaPopBuf, to: replicaBestBuf, poseCount: totalPoses)

            ActivityLog.shared.info(
                "[Engine] REMC: \(K) replicas × \(popPerReplica) poses = \(totalPoses) total, T=[\(String(format: "%.2f", tMin))–\(String(format: "%.2f", tMax))]",
                category: .dock
            )

            let swapInterval = max(config.replicaExchange.swapInterval, 1)
            let stepsPerReplica = config.replicaExchange.stepsPerReplica
            var lastCmdBuf: (any MTLCommandBuffer)?
            let explorationCutoff = Int(Float(stepsPerReplica) * config.explorationPhaseRatio)

            for step in 0..<stepsPerReplica {
                guard isRunning else { break }
                currentGeneration = step
                repGAParams.generation = UInt32(step)

                if step < explorationCutoff {
                    repGAParams.translationStep = config.explorationTranslationStep
                    repGAParams.rotationStep = config.explorationRotationStep
                    repGAParams.mutationRate = config.explorationMutationRate
                    repGAParams.mcTemperature = config.mcTemperature
                    repGAParams.localSearchSteps = UInt32(max(config.localSearchSteps / 2, 5))
                } else {
                    let exploitSteps = stepsPerReplica - explorationCutoff
                    let progress = Float(step - explorationCutoff) / Float(max(exploitSteps, 1))
                    let decay = 1.0 - progress
                    repGAParams.translationStep = config.translationStep + decay * (config.explorationTranslationStep - config.translationStep) * 0.3
                    repGAParams.rotationStep = config.rotationStep + decay * (config.explorationRotationStep - config.rotationStep) * 0.3
                    repGAParams.mutationRate = config.mutationRate + decay * (config.explorationMutationRate - config.mutationRate) * 0.5
                    repGAParams.mcTemperature = config.mcTemperature * (0.3 + 0.7 * decay)
                    repGAParams.localSearchSteps = UInt32(max(config.localSearchSteps, 1))
                }

                let effectiveLSFreq = step < explorationCutoff
                    ? max(config.explorationLocalSearchFrequency, 1)
                    : localSearchFrequency

                repParams.swapGeneration = UInt32(step)
                replicaParamsBuffer?.contents().copyMemory(from: &repParams, byteCount: MemoryLayout<ReplicaParams>.stride)

                let ringBuf = repGAParamsRing[step % 3]
                ringBuf.contents().copyMemory(from: &repGAParams, byteCount: MemoryLayout<GAParams>.stride)

                guard let repParamsBuf = replicaParamsBuffer else { break }

                let perturbBuffers: [(MTLBuffer, Int)] = [
                    (replicaOffBuf, 0), (replicaPopBuf, 1),
                    (ringBuf, 2), (gpBuf, 3), (repParamsBuf, 4)
                ]
                let scoreBuffers: [(MTLBuffer, Int)] = [
                    (replicaOffBuf, 0), (ligBuf, 1),
                    (afBuf, 2), (tiBuf, 3),
                    (gpBuf, 4), (ringBuf, 5),
                    (teBuf, 6), (miBuf, 7),
                    (emBuf, 8)
                ]
                let acceptBuffers: [(MTLBuffer, Int)] = [
                    (replicaPopBuf, 0), (replicaOffBuf, 1),
                    (replicaBestBuf, 2), (ringBuf, 3), (repParamsBuf, 4)
                ]

                let doLS = step % effectiveLSFreq == 0 && !useDruseAF

                var dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])] = [
                    (pipeline: perturbPipe, buffers: perturbBuffers),
                    (pipeline: scorePipeline, buffers: scoreBuffers)
                ]

                if doLS && !localSearchIsSIMD {
                    dispatches.insert((pipeline: activeLocalSearchPipeline, buffers: scoreBuffers), at: 1)
                }

                dispatches.append((pipeline: acceptPipe, buffers: acceptBuffers))

                if doLS && localSearchIsSIMD {
                    lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: repTgCount, threadGroupSize: repTgSize)
                    let loTempPoses = popPerReplica
                    let simdTgSize = MTLSize(width: 32, height: 1, depth: 1)
                    let simdTgCount = MTLSize(width: loTempPoses, height: 1, depth: 1)
                    let loScoreBuffers: [(MTLBuffer, Int)] = [
                        (replicaOffBuf, 0), (ligBuf, 1),
                        (afBuf, 2), (tiBuf, 3),
                        (gpBuf, 4), (ringBuf, 5),
                        (teBuf, 6), (miBuf, 7),
                        (emBuf, 8)
                    ]
                    lastCmdBuf = dispatchComputeAsync(pipeline: activeLocalSearchPipeline, buffers: loScoreBuffers,
                                                       threadGroups: simdTgCount, threadGroupSize: simdTgSize)
                } else {
                    lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: repTgCount, threadGroupSize: repTgSize)
                }

                if step % swapInterval == 0 && step > 0 {
                    if let buf = lastCmdBuf {
                        await buf.completed()
                        lastCmdBuf = nil
                    }
                    let swapTgCount = MTLSize(width: max(K - 1, 1), height: 1, depth: 1)
                    let swapTgSize = MTLSize(width: max(K - 1, 1), height: 1, depth: 1)
                    lastCmdBuf = dispatchComputeAsync(pipeline: swapPipe, buffers: [
                        (replicaPopBuf, 0), (replicaBestBuf, 1), (repParamsBuf, 2)
                    ], threadGroups: swapTgCount, threadGroupSize: swapTgSize)
                }

                if step % liveUpdateFrequency == 0 || step == stepsPerReplica - 1 {
                    if let buf = lastCmdBuf {
                        await buf.completed()
                        lastCmdBuf = nil
                    }
                    if let best = extractBestPose(from: replicaBestBuf, ligandAtoms: heavyAtoms, centroid: centroid, maxPoses: popPerReplica) {
                        bestEnergy = min(bestEnergy, best.energy)
                        let interactions = InteractionDetector.detect(
                            ligandAtoms: heavyAtoms,
                            ligandPositions: best.transformedAtomPositions,
                            proteinAtoms: proteinAtoms,
                            ligandBonds: heavyBonds,
                            scoringMethod: scoringMethod
                        )
                        onPoseUpdate?(best, interactions)
                    }
                    onGenerationComplete?(step, bestEnergy)
                }
                await Task.yield()
            }

            if let buf = lastCmdBuf {
                await buf.completed()
            }

            decomposeVinaEnergy(buffer: replicaBestBuf, populationSize: totalPoses)

            aggregatedResults = extractAllResults(
                from: replicaBestBuf,
                ligandAtoms: heavyAtoms,
                centroid: centroid,
                idOffset: 0,
                maxPoses: totalPoses,
                scoringMethod: scoringMethod
            )

            ActivityLog.shared.info("[Engine] REMC completed: \(aggregatedResults.count) poses extracted from \(K) replicas", category: .dock)

        } else if resolvedSearchMethod == .fragmentBased {
        // =====================================================================
        // MARK: Fragment-Based Incremental Construction
        // =====================================================================
        guard let fragEng = fragmentEngine,
              let snap = gridSnapshot(),
              let smiles = ligand.smiles, !smiles.isEmpty else {
            let reason: String
            if fragmentEngine == nil { reason = "fragment engine pipelines unavailable" }
            else if gridSnapshot() == nil { reason = "grid snapshot unavailable (run grid computation first)" }
            else { reason = "ligand SMILES not available — fragment decomposition requires SMILES" }
            ActivityLog.shared.error("[Engine] Fragment-based docking unavailable: \(reason)", category: .dock)
            isRunning = false
            return []
        }

        let fragmentPoses = await fragEng.runFragmentDocking(
            ligandSmiles: smiles,
            gpuLigAtoms: gpuLigAtoms,
            gridSnapshot: snap,
            config: config.fragment
        )

        if fragmentPoses.isEmpty {
            ActivityLog.shared.error("[Engine] Fragment-based docking produced no valid poses", category: .dock)
            isRunning = false
            return []
        }

        // Copy fragment-reconstructed poses into the population buffer (truncate or pad).
        let fragCount = min(fragmentPoses.count, popSize)
        let popPtr = popBuf.contents().bindMemory(to: DockPose.self, capacity: popSize)
        for i in 0..<fragCount { popPtr[i] = fragmentPoses[i] }
        for i in fragCount..<popSize { popPtr[i] = fragmentPoses[0] }

        ActivityLog.shared.info(
            "[Engine] Fragment search produced \(fragmentPoses.count) poses; refining \(fragCount) with Vina local search",
            category: .dock
        )

        // Local-search refinement + final scoring (matches the post-search refinement of REMC/GA).
        gaParams.localSearchSteps = UInt32(max(config.localSearchSteps, 20))
        gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)
        if !useDruseAF {
            localOptimize(buffer: popBuf, tg: tgCount, tgs: tgSize)
        }
        if useDruseAF {
            scoreDruseAF(buffer: popBuf, gaParamsBuffer: gaBuf, tg: tgCount, tgs: tgSize)
        } else if useDrusina {
            scoreDrusina(buffer: popBuf, gaParamsBuffer: gaBuf, tg: tgCount, tgs: tgSize)
        } else {
            scorePopulation(buffer: popBuf, tg: tgCount, tgs: tgSize)
        }
        copyPoseBuffer(from: popBuf, to: bestBuf, poseCount: popSize)

        if usePIGNet2Rescore {
            let pigTgSize = MTLSize(width: 1, height: 1, depth: 1)
            let pigTgCount = MTLSize(width: popSize, height: 1, depth: 1)
            scorePIGNet2(buffer: bestBuf, gaParamsBuffer: gaBuf, tg: pigTgCount, tgs: pigTgSize)
        } else {
            decomposeVinaEnergy(buffer: bestBuf, populationSize: popSize)
        }

        aggregatedResults = extractAllResults(
            from: bestBuf,
            ligandAtoms: heavyAtoms,
            centroid: centroid,
            scoringMethod: scoringMethod
        )
        ActivityLog.shared.info("[Engine] Fragment-based docking complete: \(aggregatedResults.count) poses extracted", category: .dock)

        } else if resolvedSearchMethod == .diffusionGuided {
        // =====================================================================
        // MARK: Diffusion-Guided (DruseAF attention-guided reverse diffusion)
        // =====================================================================
        guard useDruseAF else {
            ActivityLog.shared.error(
                "[Engine] Diffusion-guided docking requires DruseAF scoring. Set scoring method to 'Druse Affinity' and ensure DruseAF weights are loaded.",
                category: .dock
            )
            isRunning = false
            return []
        }
        ActivityLog.shared.info(
            "[Engine] Diffusion: \(popSize) poses, \(config.diffusion.numDenoisingSteps) denoising steps, schedule=\(config.diffusion.noiseSchedule.rawValue), backend=\(useAFv4 ? "v4 PGN" : "v3 cross-attn")",
            category: .dock
        )

        // Per-step progress: copy live poses into bestBuf and emit live update so the UI animates.
        let onDiffStep: (Int, Int) -> Void = { [weak self] step, _ in
            guard let self = self else { return }
            self.copyPoseBuffer(from: popBuf, to: bestBuf, poseCount: popSize)
            self.currentGeneration = step
            emitLiveUpdate(generation: step)
        }

        if useAFv4 {
            // v4 PGN attention-guided diffusion.
            guard let diffEng = diffusionEngine, diffEng.isAvailableV4,
                  let scoreGradPipe = afv4ScoreWithGradPipeline,
                  let encodePipe = druseAFEncodePipeline,
                  let v4Params = afv4ParamsBuffer,
                  let compatParams = afv4EncodeCompatParamsBuffer,
                  let dProtPos = druseAFProtPosBuffer,
                  let protPP = afv4ProtPairProjBuffer,
                  let ligPP = afv4LigPairProjBuffer,
                  let ligHidden = afv4LigHiddenBuffer,
                  let dWeights = druseAFWeights?.weightBuffer,
                  let dEntries = druseAFWeights?.entryBuffer,
                  let dInter = druseAFIntermediateBuffer else {
                var missing: [String] = []
                if diffusionEngine == nil || diffusionEngine?.isAvailableV4 == false { missing.append("diffusion init/denoise pipelines") }
                if afv4ScoreWithGradPipeline == nil { missing.append("v4 score-with-gradient pipeline") }
                if druseAFEncodePipeline == nil { missing.append("DruseAF encode pipeline") }
                if afv4ParamsBuffer == nil { missing.append("v4 params buffer") }
                if afv4EncodeCompatParamsBuffer == nil { missing.append("v4 encode-compat params") }
                if afv4ProtPairProjBuffer == nil || afv4LigPairProjBuffer == nil { missing.append("v4 pair projections") }
                if afv4LigHiddenBuffer == nil { missing.append("v4 lig hidden buffer") }
                if druseAFProtPosBuffer == nil { missing.append("protein positions") }
                if druseAFIntermediateBuffer == nil { missing.append("intermediate buffer") }
                if druseAFWeights == nil { missing.append("DruseAF weights") }
                ActivityLog.shared.error("[Engine] Diffusion (v4) unavailable: missing \(missing.joined(separator: ", "))", category: .dock)
                isRunning = false
                return []
            }

            _ = await diffEng.runDiffusionDockingV4(
                numPoses: popSize,
                numLigandAtoms: gpuLigAtoms.count,
                numTorsions: numTorsions,
                config: config.diffusion,
                gaParamsBuffer: gaBuf,
                gridParamsBuffer: gpBuf,
                ligandAtomBuffer: ligBuf,
                torsionEdgeBuffer: teBuf,
                movingIndicesBuffer: miBuf,
                afv4ParamsBuffer: v4Params,
                afv4EncodeCompatParamsBuffer: compatParams,
                afv4ProtPosBuffer: dProtPos,
                afv4ProtPairProjBuffer: protPP,
                afv4LigPairProjBuffer: ligPP,
                afv4LigHiddenBuffer: ligHidden,
                druseAFWeightBuffer: dWeights,
                druseAFEntryBuffer: dEntries,
                druseAFIntermediateBuffer: dInter,
                druseAFEncodePipeline: encodePipe,
                afv4ScoreWithGradPipeline: scoreGradPipe,
                populationBuffer: popBuf,
                onStep: onDiffStep
            )
        } else {
            // v3 cross-attention path (kept for the case v3 weights are bundled).
            guard let diffEng = diffusionEngine, diffEng.isAvailable,
                  let encodePipe = druseAFEncodePipeline,
                  let dParams = druseAFParamsBuffer,
                  let dProtPos = druseAFProtPosBuffer,
                  let dWeights = druseAFWeights?.weightBuffer,
                  let dEntries = druseAFWeights?.entryBuffer,
                  let dSetup = druseAFSetupBuffer,
                  let dInter = druseAFIntermediateBuffer else {
                var missing: [String] = []
                if diffusionEngine == nil || diffusionEngine?.isAvailable == false { missing.append("diffusion v3 pipelines") }
                if druseAFParamsBuffer == nil { missing.append("DruseAF v3 params buffer") }
                if druseAFSetupBuffer == nil { missing.append("DruseAF v3 setup buffer") }
                if druseAFEncodePipeline == nil { missing.append("DruseAF encode pipeline") }
                if druseAFProtPosBuffer == nil { missing.append("protein positions") }
                if druseAFIntermediateBuffer == nil { missing.append("intermediate buffer") }
                if druseAFWeights == nil { missing.append("DruseAF weights") }
                ActivityLog.shared.error("[Engine] Diffusion (v3) unavailable: missing \(missing.joined(separator: ", "))", category: .dock)
                isRunning = false
                return []
            }

            _ = await diffEng.runDiffusionDocking(
                numPoses: popSize,
                numLigandAtoms: gpuLigAtoms.count,
                numTorsions: numTorsions,
                config: config.diffusion,
                gaParamsBuffer: gaBuf,
                gridParamsBuffer: gpBuf,
                ligandAtomBuffer: ligBuf,
                torsionEdgeBuffer: teBuf,
                movingIndicesBuffer: miBuf,
                druseAFParamsBuffer: dParams,
                druseAFProtPosBuffer: dProtPos,
                druseAFWeightBuffer: dWeights,
                druseAFEntryBuffer: dEntries,
                druseAFSetupBuffer: dSetup,
                druseAFIntermediateBuffer: dInter,
                druseAFEncodePipeline: encodePipe,
                populationBuffer: popBuf,
                onStep: onDiffStep
            )
        }

        // Post-diffusion refinement: short Vina analytical local search to clean up the
        // poses (diffusion gives a coarse attractor; Vina gradients tighten contacts and
        // resolve clashes). Then final DruseAF rescore.
        let refSteps = max(config.diffusion.refinementSteps, 0)
        if refSteps > 0, vinaAffinityGridBuffer != nil, vinaTypeIndexBuffer != nil {
            var refGA = gaParams
            refGA.localSearchSteps = UInt32(refSteps)
            gaParamsBuffer?.contents().copyMemory(from: &refGA, byteCount: MemoryLayout<GAParams>.stride)
            localOptimize(buffer: popBuf, tg: tgCount, tgs: tgSize)
            ActivityLog.shared.info("[Engine] Diffusion refined with \(refSteps) Vina local-search steps", category: .dock)
            gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)
        }

        // Vina local search and the denoise kernel both write pose.translation freely;
        // the diffusion clamp only runs inside the denoise loop. Re-clamp here, and reject
        // poses that drifted far enough outside the box that re-clamping would distort
        // their geometry into the wall.
        do {
            let popPtr = popBuf.contents().bindMemory(to: DockPose.self, capacity: popSize)
            let center = gridParams.searchCenter
            let halfExt = gridParams.searchHalfExtent
            let cMin = center - halfExt
            let cMax = center + halfExt
            let maxDriftAng: Float = 1.5
            var rejected = 0
            for i in 0..<popSize {
                let t = popPtr[i].translation
                let outside = simd_max(simd_abs(t - center) - halfExt, .zero)
                let driftMag = outside.x + outside.y + outside.z
                if driftMag > maxDriftAng {
                    popPtr[i].energy = .infinity
                    rejected += 1
                } else {
                    popPtr[i].translation = simd_clamp(t, cMin, cMax)
                }
            }
            if rejected > 0 {
                ActivityLog.shared.info("[Engine] Diffusion: rejected \(rejected)/\(popSize) poses that drifted >\(String(format: "%.1f", maxDriftAng)) Å outside the search box", category: .dock)
            }
        }

        // The denoising kernel zeroes pose.energy at the end of every step (intended for
        // mid-loop "needs rescoring" marker). Score one more time so extraction sees real
        // energies. scoreDruseAF dispatches v3 or v4 internally based on useAFv4.
        scoreDruseAF(buffer: popBuf, gaParamsBuffer: gaBuf, tg: tgCount, tgs: tgSize)
        copyPoseBuffer(from: popBuf, to: bestBuf, poseCount: popSize)

        aggregatedResults = extractAllResults(
            from: bestBuf,
            ligandAtoms: heavyAtoms,
            centroid: centroid,
            scoringMethod: scoringMethod
        )
        ActivityLog.shared.info("[Engine] Diffusion-guided docking complete: \(aggregatedResults.count) poses extracted", category: .dock)

        } else {
        // =====================================================================
        // MARK: Standard GA / ILS Search
        // =====================================================================

        runLoop: for runIndex in 0..<totalRuns {
            guard isRunning else { break }

            // Per-run random seed ensures truly independent MC trajectories (Vina-aligned).
            gaParams.runSeed = UInt32(runIndex) &* 1000003 &+ UInt32.random(in: 0..<1_000_000)
            gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)

            dispatchCompute(pipeline: initPopPipeline, buffers: [
                (popBuf, 0), (gpBuf, 1), (gaBuf, 2)
            ], threadGroups: tgCount, threadGroupSize: tgSize)
            if !useDruseAF {
                localOptimize(buffer: popBuf, tg: tgCount, tgs: tgSize)
            }
            if useDruseAF {
                scoreDruseAF(buffer: popBuf, gaParamsBuffer: gaBuf, tg: tgCount, tgs: tgSize)
            } else if useDrusina {
                scoreDrusina(buffer: popBuf, gaParamsBuffer: gaBuf, tg: tgCount, tgs: tgSize)
            } else {
                scorePopulation(buffer: popBuf, tg: tgCount, tgs: tgSize)
            }
            copyPoseBuffer(from: popBuf, to: bestBuf, poseCount: popSize)

            let generationBase = runIndex * config.generationsPerRun

            let explorationCutoff = Int(Float(config.generationsPerRun) * config.explorationPhaseRatio)

            var lastCmdBuf: (any MTLCommandBuffer)?

            for step in 0..<config.generationsPerRun {
                guard isRunning else {
                    if let buf = lastCmdBuf {
                        await buf.completed()
                        lastCmdBuf = nil
                    }
                    decomposeVinaEnergy(buffer: bestBuf, populationSize: popSize)
                    aggregatedResults.append(contentsOf: extractAllResults(
                        from: bestBuf,
                        ligandAtoms: heavyAtoms,
                        centroid: centroid,
                        idOffset: aggregatedResults.count,
                        scoringMethod: scoringMethod
                    ))
                    break runLoop
                }
                let globalGeneration = generationBase + step
                currentGeneration = globalGeneration
                gaParams.generation = UInt32(globalGeneration)

                if step < explorationCutoff {
                    gaParams.translationStep = config.explorationTranslationStep
                    gaParams.rotationStep = config.explorationRotationStep
                    gaParams.mutationRate = config.explorationMutationRate
                    gaParams.mcTemperature = config.mcTemperature
                    gaParams.localSearchSteps = UInt32(max(config.localSearchSteps / 2, 5))
                } else {
                    // Gradual annealing: linearly decay mutation rate, step sizes, and temperature
                    // from exploration values to refinement values over the exploitation phase
                    let exploitSteps = config.generationsPerRun - explorationCutoff
                    let progress = Float(step - explorationCutoff) / Float(max(exploitSteps, 1))
                    let decay = 1.0 - progress  // 1.0 at start of exploitation, 0.0 at end
                    gaParams.translationStep = config.translationStep + decay * (config.explorationTranslationStep - config.translationStep) * 0.3
                    gaParams.rotationStep = config.rotationStep + decay * (config.explorationRotationStep - config.rotationStep) * 0.3
                    gaParams.mutationRate = config.mutationRate + decay * (config.explorationMutationRate - config.mutationRate) * 0.5
                    // Cool temperature from full to 30% for tighter Metropolis acceptance late in search
                    gaParams.mcTemperature = config.mcTemperature * (0.3 + 0.7 * decay)
                    gaParams.localSearchSteps = UInt32(max(config.localSearchSteps, 1))
                }

                let effectiveLSFreq = step < explorationCutoff
                    ? max(config.explorationLocalSearchFrequency, 1)
                    : localSearchFrequency

                let ringBuf = gaParamsRing[step % 3]
                ringBuf.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)

                let perturbBuffers: [(MTLBuffer, Int)] = [
                    (offBuf, 0), (popBuf, 1),
                    (ringBuf, 2), (gpBuf, 3)
                ]
                var vinaScoreBuffers: [(MTLBuffer, Int)] = []
                if !useDruseAF, let ab = affinityBuf, let tb = typeIdxBuf {
                    vinaScoreBuffers = [
                        (offBuf, 0), (ligBuf, 1),
                        (ab, 2), (tb, 3),
                        (gpBuf, 4), (ringBuf, 5),
                        (teBuf, 6), (miBuf, 7),
                        (emBuf, 8)
                    ]
                    if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
                        vinaScoreBuffers.append(contentsOf: [
                            (pcBuf, 15), (ppBuf, 16)
                        ])
                    }
                }
                let acceptBuffers: [(MTLBuffer, Int)] = [
                    (popBuf, 0), (offBuf, 1),
                    (bestBuf, 2), (ringBuf, 3)
                ]
                var dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])] = [
                    (pipeline: mcPerturbPipeline, buffers: perturbBuffers)
                ]
                let doLocalSearch = step % effectiveLSFreq == 0 && !useDruseAF
                if doLocalSearch && !localSearchIsSIMD {
                    dispatches.append((pipeline: activeLocalSearchPipeline, buffers: vinaScoreBuffers))
                }
                if useDruseAF, let encodePipe = druseAFEncodePipeline,
                   let afParamsBuf = druseAFParamsBuffer,
                   let afIntermed = druseAFIntermediateBuffer {
                    let encodeBuffers: [(MTLBuffer, Int)] = [
                        (offBuf, 0), (ligBuf, 1), (ringBuf, 2),
                        (teBuf, 3), (miBuf, 4),
                        (afParamsBuf, 5), (afIntermed, 6), (gpBuf, 7)
                    ]
                    dispatches.append((pipeline: encodePipe, buffers: encodeBuffers))
                } else if useDrusina, let drusinaPipe = drusinaScorePipeline,
                          let prb = proteinRingBuffer, let lrb = ligandRingBuffer,
                          let pcb = proteinCationBuffer, let dpb = drusinaParamsBuffer,
                          let pab = proteinAtomBuffer, let hib = halogenInfoBuffer,
                          let pamb = proteinAmideBuffer, let cib = chalcogenInfoBuffer,
                          let sbgb = saltBridgeGroupBuffer,
                          let elecBuf = electrostaticGridBuffer,
                          let pchBuf = proteinChalcogenBuffer,
                          let tsBuf = torsionStrainBuffer,
                          let lhbBuf = ligandHBondInfoBuffer,
                          let phbBuf = proteinHBondInfoBuffer {
                    var drusinaBuffers = vinaScoreBuffers
                    drusinaBuffers.append(contentsOf: [
                        (prb, 9), (lrb, 10),
                        (pcb, 11), (dpb, 12),
                        (pab, 13), (hib, 14),
                        (pamb, 15), (cib, 16),
                        (sbgb, 17),
                        (elecBuf, 18), (pchBuf, 19), (tsBuf, 20),
                        (lhbBuf, 21), (phbBuf, 22)
                    ])
                    dispatches.append((pipeline: drusinaPipe, buffers: drusinaBuffers))
                } else {
                    dispatches.append((pipeline: scorePipeline, buffers: vinaScoreBuffers))
                }
                if !useDruseAF, let fe = flexEngine, fe.isEnabled {
                    lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)
                    if doLocalSearch && localSearchIsSIMD {
                        let simdTgSize = MTLSize(width: 32, height: 1, depth: 1)
                        let simdTgCount = MTLSize(width: popSize, height: 1, depth: 1)
                        lastCmdBuf = dispatchComputeAsync(pipeline: activeLocalSearchPipeline, buffers: vinaScoreBuffers,
                                                           threadGroups: simdTgCount, threadGroupSize: simdTgSize)
                    }
                    fe.dispatchChiEvolution(
                        offspringBuffer: offBuf, populationBuffer: popBuf,
                        gaParamsBuffer: ringBuf, populationSize: popSize
                    )
                    fe.dispatchFlexScoring(
                        populationBuffer: offBuf, ligandAtomBuffer: ligBuf,
                        gaParamsBuffer: ringBuf, torsionEdgeBuffer: teBuf,
                        movingIndicesBuffer: miBuf, populationSize: popSize
                    )
                    if step % effectiveLSFreq == 0,
                       let ab = affinityBuf, let tb = typeIdxBuf {
                        fe.dispatchFlexLocalSearch(
                            populationBuffer: offBuf, ligandAtomBuffer: ligBuf,
                            gaParamsBuffer: ringBuf, torsionEdgeBuffer: teBuf,
                            movingIndicesBuffer: miBuf, affinityGridBuffer: ab,
                            typeIndexBuffer: tb, gridParamsBuffer: gpBuf,
                            intraPairsBuffer: emBuf, populationSize: popSize
                        )
                    }
                    lastCmdBuf = dispatchComputeAsync(pipeline: metropolisAcceptPipeline, buffers: acceptBuffers,
                                                       threadGroups: tgCount, threadGroupSize: tgSize)
                } else if useDruseAF, let scorePipe = druseAFScorePipeline,
                          let afWeights = druseAFWeights,
                          let afParamsBuf = druseAFParamsBuffer,
                          let afProtPos = druseAFProtPosBuffer,
                          let afSetup = druseAFSetupBuffer,
                          let afIntermed = druseAFIntermediateBuffer {
                    lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)
                    let afScoreBuffers: [(MTLBuffer, Int)] = [
                        (offBuf, 0), (afProtPos, 1),
                        (afWeights.weightBuffer, 2), (afWeights.entryBuffer, 3),
                        (afParamsBuf, 4), (afIntermed, 5), (afSetup, 6)
                    ]
                    let afTgSize = MTLSize(width: 32, height: 1, depth: 1)
                    let afTgCount = MTLSize(width: popSize, height: 1, depth: 1)
                    lastCmdBuf = dispatchComputeAsync(pipeline: scorePipe, buffers: afScoreBuffers,
                                                       threadGroups: afTgCount, threadGroupSize: afTgSize)
                    lastCmdBuf = dispatchComputeAsync(pipeline: metropolisAcceptPipeline, buffers: acceptBuffers,
                                                       threadGroups: tgCount, threadGroupSize: tgSize)
                } else {
                    if doLocalSearch && localSearchIsSIMD {
                        lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)
                        let simdTgSize = MTLSize(width: 32, height: 1, depth: 1)
                        let simdTgCount = MTLSize(width: popSize, height: 1, depth: 1)
                        lastCmdBuf = dispatchComputeAsync(pipeline: activeLocalSearchPipeline, buffers: vinaScoreBuffers,
                                                           threadGroups: simdTgCount, threadGroupSize: simdTgSize)
                        lastCmdBuf = dispatchComputeAsync(pipeline: metropolisAcceptPipeline, buffers: acceptBuffers,
                                                           threadGroups: tgCount, threadGroupSize: tgSize)
                    } else {
                        dispatches.append((pipeline: metropolisAcceptPipeline, buffers: acceptBuffers))
                        lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)
                    }
                }

                let updateFreq = step < explorationCutoff
                    ? max(liveUpdateFrequency * 2 / 3, 1)
                    : liveUpdateFrequency
                if step % updateFreq == 0 || step == config.generationsPerRun - 1 {
                    if let buf = lastCmdBuf {
                        await buf.completed()
                        lastCmdBuf = nil
                    }
                    emitLiveUpdate(generation: globalGeneration)
                }

                await Task.yield()
            }

            if let buf = lastCmdBuf {
                await buf.completed()
                lastCmdBuf = nil
            }

            if usePIGNet2Rescore {
                let pigTgSize = MTLSize(width: 1, height: 1, depth: 1)
                let pigTgCount = MTLSize(width: popSize, height: 1, depth: 1)
                scorePIGNet2(buffer: bestBuf, gaParamsBuffer: gaBuf, tg: pigTgCount, tgs: pigTgSize)
            } else {
                decomposeVinaEnergy(buffer: bestBuf, populationSize: popSize)
            }

            aggregatedResults.append(contentsOf: extractAllResults(
                from: bestPopulationBuffer,
                ligandAtoms: heavyAtoms,
                centroid: centroid,
                idOffset: aggregatedResults.count,
                scoringMethod: scoringMethod
            ))
        }

        } // end else (standard GA/ILS path)

        let clustered = await clusterPoses(aggregatedResults)
        var reranked = rerankClusterRepresentativesExplicit(
            clustered,
            scoringMethod: scoringMethod,
            ligandAtoms: heavyAtoms,
            centroid: centroid
        )

        // D4 dispersion single-point for top reranked poses
        if scoringMethod == .drusina, heavyAtoms.count >= 2 {
            let topN = min(reranked.count, config.explicitRerankTopClusters)
            let formalCharge = Int32(heavyAtoms.reduce(0) { $0 + $1.formalCharge })
            for i in 0..<topN {
                let positions = reranked[i].transformedAtomPositions
                guard positions.count == heavyAtoms.count else { continue }
                var posFlat = [Float](repeating: 0, count: positions.count * 3)
                var atomicNums = [Int32](repeating: 0, count: positions.count)
                for j in 0..<positions.count {
                    posFlat[j * 3]     = positions[j].x
                    posFlat[j * 3 + 1] = positions[j].y
                    posFlat[j * 3 + 2] = positions[j].z
                    atomicNums[j] = Int32(heavyAtoms[j].element.rawValue)
                }
                let solv = druse_xtb_solvent_none()
                if let result = druse_xtb_compute_energy(
                    &posFlat, &atomicNums, Int32(positions.count), formalCharge, 50, solv
                ) {
                    defer { druse_xtb_free_energy_result(result) }
                    if result.pointee.success {
                        reranked[i].gfn2DispersionEnergy = result.pointee.dispersionEnergy * 627.509
                    }
                }
            }
        }

        lastDiagnostics = computeDiagnostics(
            results: reranked,
            ligandAtoms: heavyAtoms,
            heavyBonds: heavyBonds
        )

        isRunning = false
        onDockingComplete?(reranked)
        return reranked
    }

    func stopDocking() { isRunning = false }

    // MARK: - Debug Scoring

    func debugScorePose(
        ligand: Molecule,
        translation: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        torsions: [Float] = []
    ) -> DockingResult? {
        guard vinaAffinityGridBuffer != nil,
              vinaTypeIndexBuffer != nil,
              gridParamsBuffer != nil else {
            return nil
        }

        let preparedLigand = prepareLigandGeometry(ligand)
        let heavyAtoms = preparedLigand.heavyAtoms
        let heavyBonds = preparedLigand.heavyBonds
        let centroid = preparedLigand.centroid
        var gpuLigAtoms = preparedLigand.gpuAtoms

        guard gpuLigAtoms.count > 0, gpuLigAtoms.count <= 128 else { return nil }

        ligandAtomBuffer = device.makeBuffer(
            bytes: &gpuLigAtoms,
            length: gpuLigAtoms.count * MemoryLayout<DockLigandAtom>.stride,
            options: .storageModeShared
        )

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1),
                atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count),
                movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map(Int32.init))
        }
        let numTorsions = min(torsionEdges.count, 32)

        if torsionEdges.isEmpty {
            torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0))
        }
        if movingIndices.isEmpty {
            movingIndices.append(0)
        }

        torsionEdgeBuffer = device.makeBuffer(
            bytes: &torsionEdges,
            length: torsionEdges.count * MemoryLayout<TorsionEdge>.stride,
            options: .storageModeShared
        )
        movingIndicesBuffer = device.makeBuffer(
            bytes: &movingIndices,
            length: movingIndices.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        )

        var excludedRR = Set<UInt32>()
        var adjacency = Array(repeating: [Int](), count: gpuLigAtoms.count)
        for bond in heavyBonds {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard a < gpuLigAtoms.count, b < gpuLigAtoms.count else { continue }
            adjacency[a].append(b)
            adjacency[b].append(a)
        }
        for i in 0..<gpuLigAtoms.count {
            for j in adjacency[i] where j > i {
                excludedRR.insert(UInt32(i) | (UInt32(j) << 16))
                for k in adjacency[j] where k > i && k != i {
                    let lo = min(i, k), hi = max(i, k)
                    excludedRR.insert(UInt32(lo) | (UInt32(hi) << 16))
                }
            }
        }
        var rrPairList = [UInt32]()
        for i in 0..<gpuLigAtoms.count {
            for j in (i + 1)..<gpuLigAtoms.count {
                let packed = UInt32(i) | (UInt32(j) << 16)
                if !excludedRR.contains(packed) { rrPairList.append(packed) }
            }
        }
        intraPairsBuffer = device.makeBuffer(
            bytes: &rrPairList,
            length: max(rrPairList.count, 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        var pose = DockPose()
        pose.translation = translation
        pose.energy = 1e10
        pose.rotation = SIMD4<Float>(rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for i in 0..<min(torsions.count, 32) {
                    buffer[i] = torsions[i]
                }
            }
        }
        pose.numTorsions = Int32(numTorsions)
        pose.generation = 0
        pose.stericEnergy = 0
        pose.hydrophobicEnergy = 0
        pose.hbondEnergy = 0
        pose.torsionPenalty = 0
        pose.clashPenalty = 0
        pose.drusinaCorrection = 0
        pose.constraintPenalty = 0

        populationBuffer = device.makeBuffer(
            bytes: &pose,
            length: MemoryLayout<DockPose>.stride,
            options: .storageModeShared
        )

        var gaParams = GAParams(
            populationSize: 1,
            numLigandAtoms: UInt32(gpuLigAtoms.count),
            numTorsions: UInt32(numTorsions),
            generation: 0,
            localSearchSteps: 1,
            mutationRate: 0,
            crossoverRate: 0,
            translationStep: config.translationStep,
            rotationStep: config.rotationStep,
            torsionStep: config.torsionStep,
            gridSpacing: gridParams.spacing,
            ligandRadius: max(sqrt(gpuLigAtoms.map { simd_length_squared($0.position) }.reduce(0, +) / Float(max(gpuLigAtoms.count, 1))), 1.0),
            mcTemperature: config.mcTemperature,
            referenceIntraEnergy: intramolecularReferenceEnergy(
                ligandAtoms: gpuLigAtoms,
                pairList: rrPairList
            ),
            numIntraPairs: UInt32(rrPairList.count),
            runSeed: 0,
            torsionExactFraction: config.torsionExactFraction,
            torsionLocalFraction: config.torsionLocalFraction,
            torsionLocalAmplitude: config.torsionLocalAmplitude,
            torsionRandomResetProbability: config.torsionRandomResetProbability,
            torsionPerturbationScale: config.torsionPerturbationScale,
            _pad0: 0
        )
        gaParamsBuffer = device.makeBuffer(
            bytes: &gaParams,
            length: MemoryLayout<GAParams>.stride,
            options: .storageModeShared
        )

        let tgSize = MTLSize(width: 1, height: 1, depth: 1)
        let tgCount = MTLSize(width: 1, height: 1, depth: 1)
        let wasRunning = isRunning
        isRunning = true
        scorePopulation(buffer: populationBuffer!, tg: tgCount, tgs: tgSize)
        decomposeVinaEnergy(buffer: populationBuffer!, populationSize: 1)
        isRunning = wasRunning

        return extractBestPose(ligandAtoms: heavyAtoms, centroid: centroid)
    }

    // MARK: - GPU Dispatch Helpers

    func localOptimize(buffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        guard let ligBuf = ligandAtomBuffer,
              let affBuf = vinaAffinityGridBuffer,
              let typBuf = vinaTypeIndexBuffer,
              let gpBuf = gridParamsBuffer,
              let gaBuf = gaParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer,
              let ipBuf = intraPairsBuffer else {
            ActivityLog.shared.error("[Engine] localOptimize: missing required GPU buffers — aborting docking", category: .dock)
            isRunning = false
            return
        }
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligBuf, 1),
            (affBuf, 2), (typBuf, 3),
            (gpBuf, 4), (gaBuf, 5),
            (teBuf, 6), (miBuf, 7),
            (ipBuf, 8)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        // SIMD local search kernel expects 32 threads per threadgroup (one pose per threadgroup)
        let effectiveTg: MTLSize
        let effectiveTgs: MTLSize
        if localSearchIsSIMD {
            let popSize = tg.width * tgs.width
            effectiveTgs = MTLSize(width: 32, height: 1, depth: 1)
            effectiveTg = MTLSize(width: popSize, height: 1, depth: 1)
        } else {
            effectiveTg = tg
            effectiveTgs = tgs
        }
        dispatchCompute(pipeline: activeLocalSearchPipeline, buffers: buffers,
                        threadGroups: effectiveTg, threadGroupSize: effectiveTgs)
    }

    func scorePopulation(buffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        guard let ligBuf = ligandAtomBuffer,
              let affBuf = vinaAffinityGridBuffer,
              let typBuf = vinaTypeIndexBuffer,
              let gpBuf = gridParamsBuffer,
              let gaBuf = gaParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer,
              let ipBuf = intraPairsBuffer else {
            ActivityLog.shared.error("[Engine] scorePopulation: missing required GPU buffers — aborting docking", category: .dock)
            isRunning = false
            return
        }
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligBuf, 1),
            (affBuf, 2), (typBuf, 3),
            (gpBuf, 4), (gaBuf, 5),
            (teBuf, 6), (miBuf, 7),
            (ipBuf, 8)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: scorePipeline, buffers: buffers,
                        threadGroups: tg, threadGroupSize: tgs)
    }

    func scoreDrusina(buffer: MTLBuffer, gaParamsBuffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        guard let pipe = drusinaScorePipeline,
              let prBuf = proteinRingBuffer, let lrBuf = ligandRingBuffer,
              let pcBuf = proteinCationBuffer, let dpBuf = drusinaParamsBuffer,
              let paBuf = proteinAtomBuffer, let hiBuf = halogenInfoBuffer,
              let amBuf = proteinAmideBuffer, let chBuf = chalcogenInfoBuffer,
              let sbBuf = saltBridgeGroupBuffer,
              let elecBuf = electrostaticGridBuffer,
              let pchBuf = proteinChalcogenBuffer,
              let tsBuf = torsionStrainBuffer,
              let lhbBuf = ligandHBondInfoBuffer,
              let phbBuf = proteinHBondInfoBuffer,
              let ligBuf = ligandAtomBuffer,
              let affBuf = vinaAffinityGridBuffer,
              let typBuf = vinaTypeIndexBuffer,
              let gpBuf = gridParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer,
              let ipBuf = intraPairsBuffer else {
            scorePopulation(buffer: buffer, tg: tg, tgs: tgs)
            return
        }
        dispatchCompute(pipeline: pipe, buffers: [
            (buffer, 0), (ligBuf, 1),
            (affBuf, 2), (typBuf, 3),
            (gpBuf, 4), (gaParamsBuffer, 5),
            (teBuf, 6), (miBuf, 7),
            (ipBuf, 8),
            (prBuf, 9), (lrBuf, 10), (pcBuf, 11), (dpBuf, 12), (paBuf, 13), (hiBuf, 14),
            (amBuf, 15), (chBuf, 16), (sbBuf, 17),
            (elecBuf, 18), (pchBuf, 19), (tsBuf, 20),
            (lhbBuf, 21), (phbBuf, 22)
        ], threadGroups: tg, threadGroupSize: tgs)
    }

    func scoreDruseAF(buffer: MTLBuffer, gaParamsBuffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        guard let encodePipe = druseAFEncodePipeline,
              let afWeights = druseAFWeights,
              let afProtPos = druseAFProtPosBuffer,
              let afIntermed = druseAFIntermediateBuffer,
              let ligBuf = ligandAtomBuffer,
              let gpBuf = gridParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer else {
            ActivityLog.shared.error("[DruseAF] scoreDruseAF: missing buffers", category: .dock)
            return
        }

        if useAFv4 {
            // === v4 PGN scoring ===
            guard let v4Score = afv4ScorePipeline,
                  let v4Params = afv4ParamsBuffer,
                  let compatParams = afv4EncodeCompatParamsBuffer,
                  let protPP = afv4ProtPairProjBuffer,
                  let ligPP = afv4LigPairProjBuffer,
                  let ligHidden = afv4LigHiddenBuffer else { return }

            // Transform ligand positions (reuse v3 druseAFEncode kernel with compat params)
            dispatchCompute(pipeline: encodePipe, buffers: [
                (buffer, 0), (ligBuf, 1), (gaParamsBuffer, 2),
                (teBuf, 3), (miBuf, 4),
                (compatParams, 5), (afIntermed, 6), (gpBuf, 7)
            ], threadGroups: tg, threadGroupSize: tgs)

            // v4 Score: 1 threadgroup per pose, 64 threads per group (one per ligand atom)
            let popSize = Int(tg.width * tgs.width)
            let scoreTgSize = MTLSize(width: 64, height: 1, depth: 1)
            let scoreTgCount = MTLSize(width: popSize, height: 1, depth: 1)
            dispatchCompute(pipeline: v4Score, buffers: [
                (buffer, 0), (afProtPos, 1),
                (protPP, 2), (ligPP, 3), (ligHidden, 4),
                (afWeights.weightBuffer, 5), (afWeights.entryBuffer, 6),
                (v4Params, 7), (afIntermed, 8), (gpBuf, 9)
            ], threadGroups: scoreTgCount, threadGroupSize: scoreTgSize)
        } else {
            // === v3 cross-attention scoring (legacy) ===
            guard let scorePipe = druseAFScorePipeline,
                  let afParams = druseAFParamsBuffer,
                  let afSetup = druseAFSetupBuffer else { return }

            dispatchCompute(pipeline: encodePipe, buffers: [
                (buffer, 0), (ligBuf, 1), (gaParamsBuffer, 2),
                (teBuf, 3), (miBuf, 4),
                (afParams, 5), (afIntermed, 6), (gpBuf, 7)
            ], threadGroups: tg, threadGroupSize: tgs)
            let popSize = Int(tg.width * tgs.width)
            let simdTgSize = MTLSize(width: 32, height: 1, depth: 1)
            let simdTgCount = MTLSize(width: popSize, height: 1, depth: 1)
            dispatchCompute(pipeline: scorePipe, buffers: [
                (buffer, 0), (afProtPos, 1),
                (afWeights.weightBuffer, 2), (afWeights.entryBuffer, 3),
                (afParams, 4), (afIntermed, 5), (afSetup, 6)
            ], threadGroups: simdTgCount, threadGroupSize: simdTgSize)
        }
    }

    func scorePIGNet2(buffer: MTLBuffer, gaParamsBuffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        guard let encodePipe = pignet2EncodePipeline,
              let scorePipe = pignet2ScorePipeline,
              let pig2Weights = pignet2Weights,
              let pigParams = pignet2ParamsBuffer,
              let pigProtPos = pignet2ProtPosBuffer,
              let pigSetup = pignet2SetupBuffer,
              let pigIntermed = pignet2IntermediateBuffer,
              let pigLigFeat = pignet2LigFeatBuffer,
              let pigProtAux = pignet2ProtAuxBuffer,
              let pigLigAux = pignet2LigAuxBuffer,
              let pigLigEdgeBuf = pignet2LigEdgeBuffer,
              let ligBuf = ligandAtomBuffer,
              let gpBuf = gridParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer else {
            ActivityLog.shared.error("[PIGNet2] scorePIGNet2: missing buffers", category: .dock)
            return
        }
        let ga = gaParamsBuffer.contents().bindMemory(to: GAParams.self, capacity: 1).pointee
        let pig = pigParams.contents().bindMemory(to: PIGNet2Params.self, capacity: 1).pointee
        let actualPopSize = max(Int(ga.populationSize), 1)
        let scratchBytes = actualPopSize * 2 * max(Int(pig.numProteinAtoms), 1) * Int(PIG_DIM) * 4
        if scratchBytes > pignet2ScoreScratchCapacity {
            pignet2ScoreScratchBuffer = device.makeBuffer(length: scratchBytes, options: .storageModePrivate)
            pignet2ScoreScratchCapacity = scratchBytes
        }
        guard let pigScoreScratch = pignet2ScoreScratchBuffer else {
            ActivityLog.shared.error("[PIGNet2] scorePIGNet2: missing score scratch buffer", category: .dock)
            return
        }

        // Encode: transform ligand positions
        dispatchCompute(pipeline: encodePipe, buffers: [
            (buffer, 0), (ligBuf, 1), (gaParamsBuffer, 2),
            (teBuf, 3), (miBuf, 4),
            (pigParams, 5), (pigIntermed, 6), (gpBuf, 7)
        ], threadGroups: tg, threadGroupSize: tgs)

        // Score: GNN + physics (1 thread per pose)
        let pigTgSize = MTLSize(width: 1, height: 1, depth: 1)
        let pigTgCount = MTLSize(width: actualPopSize, height: 1, depth: 1)
        dispatchCompute(pipeline: scorePipe, buffers: [
            (buffer, 0), (pigProtPos, 1),
            (pig2Weights.weightBuffer, 2), (pig2Weights.entryBuffer, 3),
            (pigParams, 4), (pigIntermed, 5), (pigSetup, 6),
            (pigLigFeat, 7), (pigProtAux, 8), (pigLigAux, 9),
            (pigLigEdgeBuf, 10), (gaParamsBuffer, 11), (pigScoreScratch, 12)
        ], threadGroups: pigTgCount, threadGroupSize: pigTgSize)
    }

    func dispatchCompute(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadGroups: MTLSize, threadGroupSize: MTLSize
    ) {
        guard isRunning else { return }
        guard threadGroups.width > 0, threadGroups.height > 0, threadGroups.depth > 0,
              threadGroupSize.width > 0, threadGroupSize.height > 0, threadGroupSize.depth > 0
        else {
            ActivityLog.shared.warn("[Engine] Skipped dispatch: zero threadGroups=\(threadGroups.width)×\(threadGroups.height)×\(threadGroups.depth) or threadGroupSize=\(threadGroupSize.width)×\(threadGroupSize.height)×\(threadGroupSize.depth)", category: .dock)
            return
        }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if cmdBuf.status == .error {
            let err = cmdBuf.error?.localizedDescription ?? "unknown"
            ActivityLog.shared.error("[Engine] GPU command buffer error: \(err)", category: .dock)
            isRunning = false
        }
    }

    @discardableResult
    func dispatchComputeAsync(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadGroups: MTLSize, threadGroupSize: MTLSize
    ) -> MTLCommandBuffer? {
        guard isRunning else { return nil }
        guard threadGroups.width > 0, threadGroups.height > 0, threadGroups.depth > 0,
              threadGroupSize.width > 0, threadGroupSize.height > 0, threadGroupSize.depth > 0
        else { return nil }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        return cmdBuf
    }

    func dispatchBatch(_ dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])],
                                threadGroups: MTLSize, threadGroupSize: MTLSize) {
        guard isRunning else { return }
        guard threadGroups.width > 0, threadGroups.height > 0, threadGroups.depth > 0,
              threadGroupSize.width > 0, threadGroupSize.height > 0, threadGroupSize.depth > 0
        else {
            ActivityLog.shared.warn("[Engine] Skipped batch dispatch: zero threadGroups=\(threadGroups.width)×\(threadGroups.height)×\(threadGroups.depth) or threadGroupSize=\(threadGroupSize.width)×\(threadGroupSize.height)×\(threadGroupSize.depth)", category: .dock)
            return
        }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        for d in dispatches {
            enc.setComputePipelineState(d.pipeline)
            for (buf, idx) in d.buffers { enc.setBuffer(buf, offset: 0, index: idx) }
            enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if cmdBuf.status == .error {
            let err = cmdBuf.error?.localizedDescription ?? "unknown"
            ActivityLog.shared.error("[Engine] GPU batch error: \(err)", category: .dock)
            isRunning = false
        }
    }

    @discardableResult
    func dispatchBatchAsync(_ dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])],
                                     threadGroups: MTLSize, threadGroupSize: MTLSize) -> MTLCommandBuffer? {
        guard isRunning else { return nil }
        guard threadGroups.width > 0, threadGroups.height > 0, threadGroups.depth > 0,
              threadGroupSize.width > 0, threadGroupSize.height > 0, threadGroupSize.depth > 0
        else {
            ActivityLog.shared.warn("[Engine] Skipped batch async dispatch: zero threadGroups=\(threadGroups.width)×\(threadGroups.height)×\(threadGroups.depth) or threadGroupSize=\(threadGroupSize.width)×\(threadGroupSize.height)×\(threadGroupSize.depth)", category: .dock)
            return nil
        }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return nil }
        for d in dispatches {
            enc.setComputePipelineState(d.pipeline)
            for (buf, idx) in d.buffers { enc.setBuffer(buf, offset: 0, index: idx) }
            enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }
        enc.endEncoding()
        cmdBuf.commit()
        return cmdBuf
    }

    func copyPoseBuffer(from source: MTLBuffer, to destination: MTLBuffer, poseCount: Int) {
        let byteCount = poseCount * MemoryLayout<DockPose>.stride
        destination.contents().copyMemory(from: source.contents(), byteCount: byteCount)
    }

    /// Compute per-term Drusina decomposition for poses in the given buffer.
    /// Returns one DrusinaDecomposition per pose, or nil if buffers are missing.
    /// Note: bypasses the isRunning guard since this is called after docking completes.
    func computeDrusinaDecomposition(poseBuffer: MTLBuffer, gaParamsBuffer: MTLBuffer, poseCount: Int) -> [DrusinaDecomposition]? {
        guard let pipe = drusinaDecompositionPipeline,
              let prBuf = proteinRingBuffer, let lrBuf = ligandRingBuffer,
              let pcBuf = proteinCationBuffer, let dpBuf = drusinaParamsBuffer,
              let paBuf = proteinAtomBuffer, let hiBuf = halogenInfoBuffer,
              let amBuf = proteinAmideBuffer, let chBuf = chalcogenInfoBuffer,
              let sbBuf = saltBridgeGroupBuffer,
              let elecBuf = electrostaticGridBuffer,
              let pchBuf = proteinChalcogenBuffer,
              let tsBuf = torsionStrainBuffer,
              let lhbBuf = ligandHBondInfoBuffer,
              let phbBuf = proteinHBondInfoBuffer,
              let ligBuf = ligandAtomBuffer,
              let gpBuf = gridParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer else { return nil }

        let decompSize = MemoryLayout<DrusinaDecomposition>.stride * poseCount
        guard let decompBuf = device.makeBuffer(length: max(decompSize, 48), options: .storageModeShared) else { return nil }

        let tgs = MTLSize(width: min(poseCount, 256), height: 1, depth: 1)
        let tg = MTLSize(width: (poseCount + tgs.width - 1) / tgs.width, height: 1, depth: 1)

        // Dispatch directly (not via dispatchCompute which requires isRunning)
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(pipe)
        let buffers: [(MTLBuffer, Int)] = [
            (poseBuffer, 0), (ligBuf, 1), (gaParamsBuffer, 2),
            (teBuf, 3), (miBuf, 4),
            (prBuf, 5), (lrBuf, 6), (pcBuf, 7), (dpBuf, 8), (paBuf, 9),
            (gpBuf, 10), (hiBuf, 11), (amBuf, 12), (chBuf, 13), (sbBuf, 14),
            (elecBuf, 15), (pchBuf, 16), (tsBuf, 17), (decompBuf, 18),
            (lhbBuf, 19), (phbBuf, 20)
        ]
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = decompBuf.contents().bindMemory(to: DrusinaDecomposition.self, capacity: poseCount)
        return Array(UnsafeBufferPointer(start: ptr, count: poseCount))
    }
}
