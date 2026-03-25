import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - Fragment Decomposition Result (Swift mirror of C++ DruseFragmentResult)

struct FragmentDecomposition: Sendable {
    let numFragments: Int
    let anchorFragmentIdx: Int
    let fragmentMembership: [Int32]     // per-heavy-atom fragment index
    let fragmentSizes: [Int32]          // atoms per fragment
    /// Ordered connections: [parentFrag, childFrag, atomA, atomB] × numConnections
    let connections: [(parent: Int, child: Int, atomA: Int, atomB: Int)]
    let centroids: [SIMD3<Float>]       // per-fragment centroid
}

struct ScaffoldMatchResult: Sendable {
    let hasMatch: Bool
    let matchedAtomIndices: [Int32]
    let tanimotoSimilarity: Float
}

// MARK: - RDKitBridge Extensions for Fragment Decomposition

extension RDKitBridge {

    static func decomposeFragments(smiles: String, scaffoldSMARTS: String? = nil) -> FragmentDecomposition? {
        guard let result = druse_decompose_fragments(smiles, scaffoldSMARTS) else { return nil }
        defer { druse_free_fragment_result(result) }

        guard result.pointee.success else { return nil }

        let nHeavy = Int(result.pointee.numHeavyAtoms)
        let nFrags = Int(result.pointee.numFragments)
        let nConn = Int(result.pointee.numConnections)

        let membership = Array(UnsafeBufferPointer(start: result.pointee.fragmentMembership, count: nHeavy))
        let sizes = Array(UnsafeBufferPointer(start: result.pointee.fragmentSizes, count: nFrags))

        var connections: [(parent: Int, child: Int, atomA: Int, atomB: Int)] = []
        if let connPtr = result.pointee.connections {
            for i in 0..<nConn {
                connections.append((
                    parent: Int(connPtr[i * 4]),
                    child: Int(connPtr[i * 4 + 1]),
                    atomA: Int(connPtr[i * 4 + 2]),
                    atomB: Int(connPtr[i * 4 + 3])
                ))
            }
        }

        var centroids: [SIMD3<Float>] = []
        if let centPtr = result.pointee.centroids {
            for f in 0..<nFrags {
                centroids.append(SIMD3<Float>(centPtr[f * 3], centPtr[f * 3 + 1], centPtr[f * 3 + 2]))
            }
        }

        return FragmentDecomposition(
            numFragments: nFrags,
            anchorFragmentIdx: Int(result.pointee.anchorFragmentIdx),
            fragmentMembership: membership,
            fragmentSizes: sizes,
            connections: connections,
            centroids: centroids
        )
    }

    static func matchScaffold(smiles: String, scaffoldSMARTS: String) -> ScaffoldMatchResult? {
        guard let result = druse_match_scaffold(smiles, scaffoldSMARTS) else { return nil }
        defer { druse_free_scaffold_match(result) }

        let indices: [Int32]
        if result.pointee.hasMatch, let ptr = result.pointee.matchedAtomIndices {
            indices = Array(UnsafeBufferPointer(start: ptr, count: Int(result.pointee.matchCount)))
        } else {
            indices = []
        }

        return ScaffoldMatchResult(
            hasMatch: result.pointee.hasMatch,
            matchedAtomIndices: indices,
            tanimotoSimilarity: result.pointee.tanimotoSimilarity
        )
    }

    static func tanimotoSimilarity(smiles1: String, smiles2: String) -> Float {
        return druse_tanimoto_similarity(smiles1, smiles2)
    }
}

// MARK: - Fragment Docking Engine

/// Orchestrates fragment-based incremental construction docking on Metal.
///
/// Algorithm:
/// 1. Decompose ligand into rigid fragments at rotatable bonds
/// 2. Place anchor fragment in binding site (GPU parallel sampling)
/// 3. Score anchor placements, prune to beam width
/// 4. Grow next fragment by sampling torsion angles, score, prune
/// 5. Repeat until full ligand reconstructed
/// 6. Final refinement with Vina analytical gradient local search
@MainActor
final class FragmentDockingEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private var initAnchorPipeline: MTLComputePipelineState?
    private var scoreFragmentPipeline: MTLComputePipelineState?
    private var pruneBeamPipeline: MTLComputePipelineState?
    private var growFragmentPipeline: MTLComputePipelineState?
    private var reconstructPosePipeline: MTLComputePipelineState?

    init?(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue

        guard let library = device.makeDefaultLibrary() else { return nil }

        do {
            if let f = library.makeFunction(name: "initAnchorPlacements") {
                initAnchorPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "scoreFragmentPlacements") {
                scoreFragmentPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "pruneFragmentBeam") {
                pruneBeamPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "growFragment") {
                growFragmentPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "reconstructFullPose") {
                reconstructPosePipeline = try device.makeComputePipelineState(function: f)
            }
        } catch {
            ActivityLog.shared.error("[FragmentDocking] Failed to create pipelines: \(error)", category: .dock)
            return nil
        }
    }

    /// Run fragment-based docking.
    ///
    /// - Parameters:
    ///   - ligandSmiles: SMILES of the ligand (needed for RDKit fragment decomposition)
    ///   - gpuLigAtoms: Prepared GPU ligand atoms
    ///   - gridSnapshot: Precomputed Vina affinity grids
    ///   - config: Fragment docking configuration
    ///   - onProgress: Called with (currentFragment, totalFragments)
    /// - Returns: Array of reconstructed DockPose candidates for final refinement
    func runFragmentDocking(
        ligandSmiles: String,
        gpuLigAtoms: [DockLigandAtom],
        gridSnapshot: DockingGridSnapshot,
        config: FragmentDockingConfig
    ) async -> [DockPose] {
        guard let initPipe = initAnchorPipeline,
              let scorePipe = scoreFragmentPipeline,
              let prunePipe = pruneBeamPipeline,
              let growPipe = growFragmentPipeline,
              let reconPipe = reconstructPosePipeline,
              let affinityBuf = gridSnapshot.vinaAffinityGridBuffer,
              let typeIdxBuf = gridSnapshot.vinaTypeIndexBuffer
        else {
            ActivityLog.shared.error("[FragmentDocking] Missing pipelines or grid buffers", category: .dock)
            return []
        }

        // Step 1: Decompose ligand into fragments
        let scaffoldSMARTS = config.scaffoldMode == .manual ? config.scaffoldSMARTS : nil
        guard let decomp = RDKitBridge.decomposeFragments(smiles: ligandSmiles, scaffoldSMARTS: scaffoldSMARTS) else {
            ActivityLog.shared.error("[FragmentDocking] Fragment decomposition failed", category: .dock)
            return []
        }

        ActivityLog.shared.info(
            "[FragmentDocking] Decomposed into \(decomp.numFragments) fragments, " +
            "anchor=\(decomp.anchorFragmentIdx), connections=\(decomp.connections.count)",
            category: .dock
        )

        // If only 1 fragment, fragment-based docking degenerates to simple rigid placement
        // Still works — just no growth phase

        // Step 2: Build GPU fragment data
        var gpuFragAtoms: [FragmentAtom] = []
        var gpuFragDefs: [FragmentDef] = []

        for fragIdx in 0..<decomp.numFragments {
            let atomStart = UInt32(gpuFragAtoms.count)
            var atomCount: UInt32 = 0

            for (heavyIdx, membership) in decomp.fragmentMembership.enumerated() {
                if Int(membership) == fragIdx && heavyIdx < gpuLigAtoms.count {
                    let ligAtom = gpuLigAtoms[heavyIdx]
                    gpuFragAtoms.append(FragmentAtom(
                        position: ligAtom.position,
                        vdwRadius: ligAtom.vdwRadius,
                        charge: ligAtom.charge,
                        vinaType: ligAtom.vinaType,
                        globalAtomIndex: Int32(heavyIdx),
                        _pad0: 0
                    ))
                    atomCount += 1
                }
            }

            let centroid = fragIdx < decomp.centroids.count ? decomp.centroids[fragIdx] : .zero
            let parentFrag: Int32 = {
                for conn in decomp.connections {
                    if conn.child == fragIdx { return Int32(conn.parent) }
                }
                return -1
            }()

            gpuFragDefs.append(FragmentDef(
                atomStart: atomStart,
                atomCount: atomCount,
                connectingTorsionIdx: 0,
                parentFragmentIdx: parentFrag,
                centroid: centroid,
                _pad0: 0
            ))
        }

        // Create GPU buffers
        guard !gpuFragAtoms.isEmpty else { return [] }
        var fragAtomsCopy = gpuFragAtoms
        var fragDefsCopy = gpuFragDefs
        guard let fragAtomBuf = device.makeBuffer(bytes: &fragAtomsCopy,
                length: fragAtomsCopy.count * MemoryLayout<FragmentAtom>.stride, options: .storageModeShared),
              let fragDefBuf = device.makeBuffer(bytes: &fragDefsCopy,
                length: fragDefsCopy.count * MemoryLayout<FragmentDef>.stride, options: .storageModeShared)
        else { return [] }

        var ligAtomsCopy = gpuLigAtoms
        guard let ligAtomBuf = device.makeBuffer(bytes: &ligAtomsCopy,
                length: ligAtomsCopy.count * MemoryLayout<DockLigandAtom>.stride, options: .storageModeShared)
        else { return [] }

        let gpBuf = gridSnapshot.gridParamsBuffer

        // Step 3: Initialize anchor placements
        let anchorSamples = config.anchorSamplingCount
        let placementBufSize = max(anchorSamples, config.beamWidth * config.torsionSamples) * MemoryLayout<FragmentPlacement>.stride
        guard let placementBuf = device.makeBuffer(length: placementBufSize, options: .storageModeShared) else { return [] }

        var searchParams = FragmentSearchParams(
            numFragments: UInt32(decomp.numFragments),
            beamWidth: UInt32(config.beamWidth),
            currentFragment: 0,
            numPlacements: UInt32(anchorSamples),
            numAnchorSamples: UInt32(anchorSamples),
            torsionSamples: UInt32(config.torsionSamples),
            pruneThreshold: config.growthPruneThreshold,
            numLigandAtoms: UInt32(gpuLigAtoms.count)
        )
        guard let searchParamsBuf = device.makeBuffer(bytes: &searchParams,
                length: MemoryLayout<FragmentSearchParams>.stride, options: .storageModeShared)
        else { return [] }

        // Initialize anchor placements
        let tgSize = MTLSize(width: min(anchorSamples, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (anchorSamples + 255) / 256, height: 1, depth: 1)

        dispatchSync(pipeline: initPipe, buffers: [
            (placementBuf, 0), (gpBuf, 1), (searchParamsBuf, 2)
        ], threadGroups: tgCount, threadGroupSize: tgSize)

        // Score anchor placements
        dispatchSync(pipeline: scorePipe, buffers: [
            (placementBuf, 0), (fragAtomBuf, 1), (fragDefBuf, 2),
            (affinityBuf, 3), (typeIdxBuf, 4), (gpBuf, 5), (searchParamsBuf, 6)
        ], threadGroups: tgCount, threadGroupSize: tgSize)

        // Prune to beam width
        dispatchSync(pipeline: prunePipe, buffers: [
            (placementBuf, 0), (searchParamsBuf, 1)
        ], threadGroups: MTLSize(width: 1, height: 1, depth: 1),
           threadGroupSize: MTLSize(width: 1, height: 1, depth: 1))

        ActivityLog.shared.info("[FragmentDocking] Anchor placements scored and pruned to beam width \(config.beamWidth)", category: .dock)

        // Step 4: Grow fragments in BFS order
        // Build fragment growth order (BFS from anchor)
        var growOrder: [Int] = []
        for conn in decomp.connections {
            growOrder.append(conn.child)
        }

        for (growStep, childFragIdx) in growOrder.enumerated() {
            let totalExpanded = config.beamWidth * config.torsionSamples
            let expandedBufSize = totalExpanded * MemoryLayout<FragmentPlacement>.stride
            guard let expandedBuf = device.makeBuffer(length: expandedBufSize, options: .storageModeShared) else { break }

            searchParams.currentFragment = UInt32(childFragIdx)
            searchParams.numPlacements = UInt32(config.beamWidth) // parent count
            searchParamsBuf.contents().copyMemory(from: &searchParams, byteCount: MemoryLayout<FragmentSearchParams>.stride)

            let growTgSize = MTLSize(width: min(totalExpanded, 256), height: 1, depth: 1)
            let growTgCount = MTLSize(width: (totalExpanded + 255) / 256, height: 1, depth: 1)

            dispatchSync(pipeline: growPipe, buffers: [
                (expandedBuf, 0), (placementBuf, 1), (fragAtomBuf, 2), (fragDefBuf, 3),
                (ligAtomBuf, 4), (affinityBuf, 5), (typeIdxBuf, 6), (gpBuf, 7), (searchParamsBuf, 8)
            ], threadGroups: growTgCount, threadGroupSize: growTgSize)

            // Prune expanded placements
            searchParams.numPlacements = UInt32(totalExpanded)
            searchParamsBuf.contents().copyMemory(from: &searchParams, byteCount: MemoryLayout<FragmentSearchParams>.stride)

            dispatchSync(pipeline: prunePipe, buffers: [
                (expandedBuf, 0), (searchParamsBuf, 1)
            ], threadGroups: MTLSize(width: 1, height: 1, depth: 1),
               threadGroupSize: MTLSize(width: 1, height: 1, depth: 1))

            // Copy surviving placements back to main placement buffer for next iteration
            placementBuf.contents().copyMemory(from: expandedBuf.contents(),
                byteCount: min(placementBufSize, expandedBufSize))

            ActivityLog.shared.info("[FragmentDocking] Grew fragment \(childFragIdx) (step \(growStep + 1)/\(growOrder.count))", category: .dock)
            await Task.yield()
        }

        // Step 5: Reconstruct full poses from final placements
        let finalCount = min(config.beamWidth, Int(searchParams.numPlacements))
        let poseBufSize = finalCount * MemoryLayout<DockPose>.stride
        guard let poseBuf = device.makeBuffer(length: poseBufSize, options: .storageModeShared) else { return [] }

        searchParams.numPlacements = UInt32(finalCount)
        searchParamsBuf.contents().copyMemory(from: &searchParams, byteCount: MemoryLayout<FragmentSearchParams>.stride)

        let reconTgSize = MTLSize(width: min(finalCount, 256), height: 1, depth: 1)
        let reconTgCount = MTLSize(width: (finalCount + 255) / 256, height: 1, depth: 1)

        dispatchSync(pipeline: reconPipe, buffers: [
            (poseBuf, 0), (placementBuf, 1), (fragDefBuf, 2), (searchParamsBuf, 3), (gpBuf, 4)
        ], threadGroups: reconTgCount, threadGroupSize: reconTgSize)

        // Extract poses
        let posePtr = poseBuf.contents().bindMemory(to: DockPose.self, capacity: finalCount)
        var poses: [DockPose] = []
        for i in 0..<finalCount {
            let p = posePtr[i]
            if p.energy.isFinite && p.energy < 1e9 {
                poses.append(p)
            }
        }

        poses.sort { $0.energy < $1.energy }
        ActivityLog.shared.info("[FragmentDocking] Reconstructed \(poses.count) valid poses", category: .dock)
        return poses
    }

    // MARK: - GPU Dispatch Helper

    private func dispatchSync(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadGroups: MTLSize,
        threadGroupSize: MTLSize
    ) {
        guard threadGroups.width > 0, threadGroupSize.width > 0 else { return }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }
}
