// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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

        return FragmentDecomposition(
            numFragments: nFrags,
            anchorFragmentIdx: Int(result.pointee.anchorFragmentIdx),
            fragmentMembership: membership,
            fragmentSizes: sizes,
            connections: connections
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

    /// Must match MAX_BEAM_WIDTH in ShaderTypes.h (size of the prune kernel's selection array).
    private static let maxBeamWidth = 256

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
        } catch {
            ActivityLog.shared.error("[FragmentDocking] Failed to create pipelines: \(error)", category: .dock)
            return nil
        }
    }

    /// Run fragment-based docking.
    ///
    /// - Parameters:
    ///   - ligandSmiles: SMILES of the ligand (needed for RDKit fragment decomposition)
    ///   - gpuLigAtoms: Prepared GPU ligand atoms (centroid-subtracted input conformer)
    ///   - gridSnapshot: Precomputed Vina affinity grids
    ///   - config: Fragment docking configuration
    /// - Returns: Rigid-body DockPose candidates (torsions at the input conformer) for final refinement
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
              let affinityBuf = gridSnapshot.vinaAffinityGridBuffer,
              let typeIdxBuf = gridSnapshot.vinaTypeIndexBuffer
        else {
            ActivityLog.shared.error("[FragmentDocking] Missing pipelines or grid buffers", category: .dock)
            return []
        }

        let beamWidth = min(max(config.beamWidth, 1), Self.maxBeamWidth)
        let anchorSamples = max(config.anchorSamplingCount, 1)
        let torsionSamples = max(config.torsionSamples, 1)

        // Step 1: Decompose ligand into fragments
        let scaffoldSMARTS = config.scaffoldMode == .manual ? config.scaffoldSMARTS : nil
        guard let decomp = RDKitBridge.decomposeFragments(smiles: ligandSmiles, scaffoldSMARTS: scaffoldSMARTS),
              decomp.numFragments > 0,
              (0..<decomp.numFragments).contains(decomp.anchorFragmentIdx)
        else {
            ActivityLog.shared.error("[FragmentDocking] Fragment decomposition failed", category: .dock)
            return []
        }

        ActivityLog.shared.info(
            "[FragmentDocking] Decomposed into \(decomp.numFragments) fragments, " +
            "anchor=\(decomp.anchorFragmentIdx), connections=\(decomp.connections.count)",
            category: .dock
        )
        if decomp.fragmentMembership.count != gpuLigAtoms.count {
            ActivityLog.shared.warn(
                "[FragmentDocking] SMILES has \(decomp.fragmentMembership.count) heavy atoms but the prepared ligand has " +
                "\(gpuLigAtoms.count); fragment membership may not map onto the docked atoms",
                category: .dock
            )
        }

        // Step 2: Build GPU fragment data.
        // The kernels hard-code the anchor as fragment 0 (initAnchorPlacements writes fragmentIdx = 0),
        // so fragments are renumbered in growth (BFS) order with the anchor first. Fragments not
        // reachable from the anchor (e.g. counter-ions) are appended and never grown.
        let growOrderOld = decomp.connections.map(\.child)
        var fragmentOrder = [decomp.anchorFragmentIdx]
        var seen = Set(fragmentOrder)
        for f in growOrderOld + Array(0..<decomp.numFragments) {
            if seen.insert(f).inserted { fragmentOrder.append(f) }
        }
        var newIndex = [Int](repeating: -1, count: decomp.numFragments)
        for (newIdx, oldIdx) in fragmentOrder.enumerated() { newIndex[oldIdx] = newIdx }

        var membersByFragment = [[Int]](repeating: [], count: decomp.numFragments)
        for (heavyIdx, membership) in decomp.fragmentMembership.enumerated()
        where heavyIdx < gpuLigAtoms.count && (0..<decomp.numFragments).contains(Int(membership)) {
            membersByFragment[Int(membership)].append(heavyIdx)
        }
        // Connections are oriented by the decomposition: atomA lies in the parent fragment,
        // atomB in the child. Keep the bond so growth rotates about the real connecting bond.
        var connectionOf = [(parent: Int, child: Int, atomA: Int, atomB: Int)?](repeating: nil, count: decomp.numFragments)
        for conn in decomp.connections { connectionOf[conn.child] = conn }

        var gpuFragAtoms: [FragmentAtom] = []
        var gpuFragDefs: [FragmentDef] = []
        for oldIdx in fragmentOrder {
            let members = membersByFragment[oldIdx]
            // Centroid in the same (centroid-subtracted) frame as the atom positions. The C++
            // decomposition reports centroids of a fresh ETKDG embedding, an unrelated frame.
            let centroid = members.isEmpty ? SIMD3<Float>.zero
                : members.reduce(SIMD3<Float>.zero) { $0 + gpuLigAtoms[$1].position } / Float(members.count)
            let atomStart = UInt32(gpuFragAtoms.count)
            for heavyIdx in members {
                let ligAtom = gpuLigAtoms[heavyIdx]
                gpuFragAtoms.append(FragmentAtom(
                    position: ligAtom.position,
                    vdwRadius: ligAtom.vdwRadius,
                    charge: ligAtom.charge,
                    vinaType: ligAtom.vinaType,
                    globalAtomIndex: Int32(heavyIdx),
                    _pad0: 0
                ))
            }
            // Bond atom positions in the docked (centroid-subtracted) frame. If an index does not
            // map onto the prepared ligand, fall back to the fragment centroids as the axis.
            var parentIdx: Int32 = -1
            var bondParent = centroid
            var bondChild = centroid
            if let conn = connectionOf[oldIdx] {
                parentIdx = Int32(newIndex[conn.parent])
                if gpuLigAtoms.indices.contains(conn.atomA), gpuLigAtoms.indices.contains(conn.atomB) {
                    bondParent = gpuLigAtoms[conn.atomA].position
                    bondChild = gpuLigAtoms[conn.atomB].position
                } else {
                    let parentMembers = membersByFragment[conn.parent]
                    if !parentMembers.isEmpty {
                        bondParent = parentMembers.reduce(SIMD3<Float>.zero) { $0 + gpuLigAtoms[$1].position }
                            / Float(parentMembers.count)
                    }
                }
            }
            gpuFragDefs.append(FragmentDef(
                atomStart: atomStart,
                atomCount: UInt32(members.count),
                parentFragmentIdx: parentIdx,
                _pad0: 0,
                centroid: centroid,
                bondParentAtom: bondParent,
                bondChildAtom: bondChild
            ))
        }
        // Fragment k (k ≥ 1) is grown at step k − 1, so beam level k holds fragment k's placements.
        let growOrder = growOrderOld.map { newIndex[$0] }

        // Create GPU buffers
        guard !gpuFragAtoms.isEmpty, gpuFragDefs[0].atomCount > 0 else { return [] }
        guard let fragAtomBuf = device.makeBuffer(bytes: gpuFragAtoms,
                length: gpuFragAtoms.count * MemoryLayout<FragmentAtom>.stride, options: .storageModeShared),
              let fragDefBuf = device.makeBuffer(bytes: gpuFragDefs,
                length: gpuFragDefs.count * MemoryLayout<FragmentDef>.stride, options: .storageModeShared)
        else { return [] }

        let gpBuf = gridSnapshot.gridParamsBuffer

        // Step 3: Initialize anchor placements
        let totalExpanded = beamWidth * torsionSamples
        let placementStride = MemoryLayout<FragmentPlacement>.stride
        guard let placementBuf = device.makeBuffer(length: max(anchorSamples, beamWidth) * placementStride,
                                                   options: .storageModeShared),
              let expandedBuf = device.makeBuffer(length: totalExpanded * placementStride,
                                                  options: .storageModeShared),
              // Per-level pruned beams, read by growFragment to locate each child's real parent placement.
              let historyBuf = device.makeBuffer(length: (growOrder.count + 1) * beamWidth * placementStride,
                                                 options: .storageModeShared)
        else { return [] }

        var searchParams = FragmentSearchParams(
            numFragments: UInt32(gpuFragDefs.count),
            beamWidth: UInt32(beamWidth),
            currentFragment: 0,
            numPlacements: UInt32(anchorSamples),
            numAnchorSamples: UInt32(anchorSamples),
            torsionSamples: UInt32(torsionSamples),
            pruneThreshold: config.growthPruneThreshold,
            numLigandAtoms: UInt32(gpuLigAtoms.count)
        )
        guard let searchParamsBuf = device.makeBuffer(bytes: &searchParams,
                length: MemoryLayout<FragmentSearchParams>.stride, options: .storageModeShared)
        else { return [] }

        let tgSize = MTLSize(width: min(anchorSamples, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (anchorSamples + 255) / 256, height: 1, depth: 1)
        let singleThread = MTLSize(width: 1, height: 1, depth: 1)

        dispatchSync(pipeline: initPipe, buffers: [
            (placementBuf, 0), (gpBuf, 1), (searchParamsBuf, 2)
        ], threadGroups: tgCount, threadGroupSize: tgSize)

        dispatchSync(pipeline: scorePipe, buffers: [
            (placementBuf, 0), (fragAtomBuf, 1), (fragDefBuf, 2),
            (affinityBuf, 3), (typeIdxBuf, 4), (gpBuf, 5), (searchParamsBuf, 6)
        ], threadGroups: tgCount, threadGroupSize: tgSize)

        dispatchSync(pipeline: prunePipe, buffers: [
            (placementBuf, 0), (searchParamsBuf, 1)
        ], threadGroups: singleThread, threadGroupSize: singleThread)

        // The prune kernel compacts survivors into slots 0..<min(beamWidth, N); slots beyond
        // N were never written, so the beam size must not exceed the number of placements.
        var beamCount = min(beamWidth, anchorSamples)

        // Each pruned beam is kept on the host so final placements can be traced back to their
        // anchor (child.parentPlacementIdx indexes the previous level's compacted beam).
        func recordBeam(level: Int, count: Int) -> [FragmentPlacement] {
            historyBuf.contents().advanced(by: level * beamWidth * placementStride)
                .copyMemory(from: placementBuf.contents(), byteCount: count * placementStride)
            let ptr = placementBuf.contents().bindMemory(to: FragmentPlacement.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
        var beamHistory: [[FragmentPlacement]] = [recordBeam(level: 0, count: beamCount)]

        ActivityLog.shared.info("[FragmentDocking] Anchor placements scored and pruned to beam width \(beamCount)", category: .dock)

        // Step 4: Grow fragments in BFS order
        for (growStep, childFragIdx) in growOrder.enumerated() {
            let expandedCount = beamCount * torsionSamples

            searchParams.currentFragment = UInt32(childFragIdx)
            searchParams.numPlacements = UInt32(beamCount) // parent count
            searchParamsBuf.contents().copyMemory(from: &searchParams, byteCount: MemoryLayout<FragmentSearchParams>.stride)

            let growTgSize = MTLSize(width: min(expandedCount, 256), height: 1, depth: 1)
            let growTgCount = MTLSize(width: (expandedCount + 255) / 256, height: 1, depth: 1)

            dispatchSync(pipeline: growPipe, buffers: [
                (expandedBuf, 0), (placementBuf, 1), (fragAtomBuf, 2), (fragDefBuf, 3),
                (affinityBuf, 5), (typeIdxBuf, 6), (gpBuf, 7), (searchParamsBuf, 8),
                (historyBuf, 9)
            ], threadGroups: growTgCount, threadGroupSize: growTgSize)

            // Prune expanded placements
            searchParams.numPlacements = UInt32(expandedCount)
            searchParamsBuf.contents().copyMemory(from: &searchParams, byteCount: MemoryLayout<FragmentSearchParams>.stride)

            dispatchSync(pipeline: prunePipe, buffers: [
                (expandedBuf, 0), (searchParamsBuf, 1)
            ], threadGroups: singleThread, threadGroupSize: singleThread)

            // Survivors are compacted into the front of the expanded buffer; only the beam is carried forward.
            beamCount = min(beamWidth, expandedCount)
            placementBuf.contents().copyMemory(from: expandedBuf.contents(), byteCount: beamCount * placementStride)
            beamHistory.append(recordBeam(level: growStep + 1, count: beamCount))

            ActivityLog.shared.info("[FragmentDocking] Grew fragment \(childFragIdx) (step \(growStep + 1)/\(growOrder.count))", category: .dock)
            await Task.yield()
        }

        // Step 5: Convert each surviving chain into a rigid-body DockPose.
        // The anchor placement scores atoms at R·(p − c_anchor) + T, which equals the DockPose
        // transform R·p + t with t = T − R·c_anchor (p are centroid-subtracted ligand coordinates).
        // Torsions stay at the input conformer: fragment connecting torsions are not expressed in
        // the ligand's torsion-tree indexing, so they are left to the subsequent local search.
        let anchorCentroid = gpuFragDefs[0].centroid
        var poses: [DockPose] = []
        for leaf in beamHistory[beamHistory.count - 1] where leaf.valid != 0 && leaf.energy.isFinite && leaf.energy < 1e9 {
            var idx = Int(leaf.parentPlacementIdx)
            var anchor = leaf
            var traced = true
            for level in stride(from: beamHistory.count - 2, through: 0, by: -1) {
                guard beamHistory[level].indices.contains(idx) else { traced = false; break }
                anchor = beamHistory[level][idx]
                idx = Int(anchor.parentPlacementIdx)
            }
            guard traced else { continue }

            var pose = DockPose()
            pose.rotation = anchor.rotation
            pose.translation = anchor.translation - Self.quatRotate(anchor.rotation, anchorCentroid)
            pose.energy = leaf.energy
            poses.append(pose)
        }

        poses.sort { $0.energy < $1.energy }
        ActivityLog.shared.info("[FragmentDocking] Reconstructed \(poses.count) valid poses", category: .dock)
        return poses
    }

    /// Same quaternion rotation as `quatRotate` in FragmentDocking.metal.
    private static func quatRotate(_ q: SIMD4<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
        let u = SIMD3<Float>(q.x, q.y, q.z)
        let s = q.w
        return 2 * simd_dot(u, v) * u + (s * s - simd_dot(u, u)) * v + 2 * s * simd_cross(u, v)
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
