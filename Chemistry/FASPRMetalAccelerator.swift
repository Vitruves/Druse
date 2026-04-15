// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// FASPRMetalAccelerator.swift — GPU-accelerated FASPR sidechain packing energy
//
// Metal compute dispatch wrapper for FASPR sidechain packing energy kernels.
// Evaluates VDW self-energy (rotamer vs backbone) and pair-energy (rotamer
// vs rotamer) on GPU for rapid dead-end elimination scoring.
//
// Architecture:
//   Swift caller → Metal compute dispatch (self / pair VDW)
//   → CPU readback of per-rotamer or per-pair energies
//
// All computation in float32 / kcal/mol / Angstrom.
// ============================================================================

import Foundation
import Metal
import simd

final class FASPRMetalAccelerator: @unchecked Sendable {

    let device: MTLDevice
    let queue: MTLCommandQueue

    private let selfEnergyPipeline: MTLComputePipelineState
    private let pairEnergyPipeline: MTLComputePipelineState

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary()
        else { return nil }

        self.device = device
        self.queue = queue

        do {
            selfEnergyPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "faspr_self_energy")!)
            pairEnergyPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "faspr_pair_energy")!)
        } catch {
            return nil
        }
    }

    // MARK: - Buffer Helpers

    private func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        data.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: ptr.count * MemoryLayout<T>.stride,
                              options: .storageModeShared)
        }
    }

    private func makeBuffer<T>(count: Int, type: T.Type = T.self) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<T>.stride, options: .storageModeShared)
    }

    private func threadgroupConfig(pipeline: MTLComputePipelineState, count: Int) -> (MTLSize, MTLSize) {
        let threadWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        return (MTLSize(width: count, height: 1, depth: 1),
                MTLSize(width: threadWidth, height: 1, depth: 1))
    }

    // MARK: - Self-Energy

    /// Compute self-energy: VDW between each rotamer's sidechain and backbone atoms.
    /// Returns energy per rotamer.
    func computeSelfVDW(
        rotamerAtoms: [FASPRGPUAtom],
        backboneAtoms: [FASPRGPUAtom],
        rotamerOffsets: [UInt32],
        rotamerCount: Int
    ) -> [Float] {
        // Compute max atoms per rotamer from offsets
        var maxAtomsPerRotamer: UInt32 = 0
        for i in 0..<rotamerCount {
            let count = rotamerOffsets[i + 1] - rotamerOffsets[i]
            maxAtomsPerRotamer = max(maxAtomsPerRotamer, count)
        }

        guard let rotamerAtomBuf = makeBuffer(rotamerAtoms),
              let backboneBuf = makeBuffer(backboneAtoms),
              let offsetBuf = makeBuffer(rotamerOffsets),
              let energyBuf = makeBuffer(count: rotamerCount, type: Float.self)
        else { return Array(repeating: Float.infinity, count: rotamerCount) }

        var params = FASPRSelfParams(
            rotamerCount: UInt32(rotamerCount),
            backboneAtomCount: UInt32(backboneAtoms.count),
            maxAtomsPerRotamer: maxAtomsPerRotamer,
            vdwRepCut: 10.0,
            dstarMinCut: 0.015,
            dstarMaxCut: 1.90
        )

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return Array(repeating: Float.infinity, count: rotamerCount)
        }

        encoder.setComputePipelineState(selfEnergyPipeline)
        encoder.setBuffer(rotamerAtomBuf, offset: 0, index: 0)
        encoder.setBuffer(backboneBuf, offset: 0, index: 1)
        encoder.setBuffer(offsetBuf, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<FASPRSelfParams>.stride, index: 3)
        encoder.setBuffer(energyBuf, offset: 0, index: 4)

        let (grid, tg) = threadgroupConfig(pipeline: selfEnergyPipeline, count: rotamerCount)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read results
        let ptr = energyBuf.contents().bindMemory(to: Float.self, capacity: rotamerCount)
        return Array(UnsafeBufferPointer(start: ptr, count: rotamerCount))
    }

    // MARK: - Pair-Energy

    /// Compute pair-energy: VDW between rotamer pairs from two sites.
    /// Returns flat array [rot1 * rot2Count + rot2].
    func computePairVDW(
        site1Atoms: [FASPRGPUAtom],
        site2Atoms: [FASPRGPUAtom],
        offsets1: [UInt32],
        offsets2: [UInt32],
        rot1Count: Int,
        rot2Count: Int
    ) -> [Float] {
        let totalPairs = rot1Count * rot2Count

        // Compute max atoms per rotamer for each site
        var maxAtoms1: UInt32 = 0
        for i in 0..<rot1Count {
            let count = offsets1[i + 1] - offsets1[i]
            maxAtoms1 = max(maxAtoms1, count)
        }
        var maxAtoms2: UInt32 = 0
        for i in 0..<rot2Count {
            let count = offsets2[i + 1] - offsets2[i]
            maxAtoms2 = max(maxAtoms2, count)
        }

        guard let site1Buf = makeBuffer(site1Atoms),
              let site2Buf = makeBuffer(site2Atoms),
              let offset1Buf = makeBuffer(offsets1),
              let offset2Buf = makeBuffer(offsets2),
              let energyBuf = makeBuffer(count: totalPairs, type: Float.self)
        else { return Array(repeating: Float.infinity, count: totalPairs) }

        var params = FASPRPairParams(
            rot1Count: UInt32(rot1Count),
            rot2Count: UInt32(rot2Count),
            maxAtoms1: maxAtoms1,
            maxAtoms2: maxAtoms2,
            vdwRepCut: 10.0,
            dstarMinCut: 0.015,
            dstarMaxCut: 1.90,
            _pad0: 0
        )

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return Array(repeating: Float.infinity, count: totalPairs)
        }

        encoder.setComputePipelineState(pairEnergyPipeline)
        encoder.setBuffer(site1Buf, offset: 0, index: 0)
        encoder.setBuffer(site2Buf, offset: 0, index: 1)
        encoder.setBuffer(offset1Buf, offset: 0, index: 2)
        encoder.setBuffer(offset2Buf, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<FASPRPairParams>.stride, index: 4)
        encoder.setBuffer(energyBuf, offset: 0, index: 5)

        let (grid, tg) = threadgroupConfig(pipeline: pairEnergyPipeline, count: totalPairs)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read results
        let ptr = energyBuf.contents().bindMemory(to: Float.self, capacity: totalPairs)
        return Array(UnsafeBufferPointer(start: ptr, count: totalPairs))
    }
}
