// ============================================================================
// LoopMetalAccelerator.swift — GPU-accelerated loop refinement
//
// Manages Metal compute pipeline states for loop energy/gradient evaluation.
// Drives a steepest descent minimizer on CPU with GPU force evaluation.
//
// Architecture:
//   Swift minimizer loop → Metal compute dispatch (energy + gradient)
//   → CPU line search + position update → repeat until converged
//
// All computation in float32 / kcal/mol / Angstrom.
// ============================================================================

import Foundation
import Metal
import simd

final class LoopMetalAccelerator: @unchecked Sendable {

    let device: MTLDevice
    let queue: MTLCommandQueue

    private let bondPipeline: MTLComputePipelineState
    private let anglePipeline: MTLComputePipelineState
    private let torsionPipeline: MTLComputePipelineState
    private let stericPipeline: MTLComputePipelineState

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary()
        else { return nil }

        self.device = device
        self.queue = queue

        do {
            bondPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "loop_bond_energy")!)
            anglePipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "loop_angle_energy")!)
            torsionPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "loop_torsion_energy")!)
            stericPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "loop_steric_restraint")!)
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

    // MARK: - Minimization

    struct LoopRefineInput {
        let atoms: [Atom]
        let bonds: [Bond]
        let isLoopAtom: [Bool]
        // Backbone angles: (atom1, atom2_central, atom3, angleDeg)
        let angles: [(Int, Int, Int, Float)]
        // Omega torsions: (atom1, atom2, atom3, atom4)
        let torsions: [(Int, Int, Int, Int)]
    }

    struct LoopRefineOutput {
        let positions: [SIMD3<Float>]
        let finalEnergy: Float
        let iterations: Int
    }

    /// Run GPU-accelerated loop refinement.
    /// Steepest descent with backtracking line search, GPU energy+gradient evaluation.
    func refine(input: LoopRefineInput, maxIterations: Int = 500, tolerance: Float = 0.1) -> LoopRefineOutput {
        let natom = input.atoms.count

        // Build GPU data structures
        var gpuAtoms = buildGPUAtoms(input: input)
        let refAtoms = gpuAtoms  // reference positions for restraints
        let gpuBonds = buildGPUBonds(input: input)
        let gpuAngles = buildGPUAngles(input: input)
        let gpuTorsions = buildGPUTorsions(input: input)

        // Create persistent buffers
        guard let bondBuf = makeBuffer(gpuBonds),
              let angleBuf = makeBuffer(gpuAngles),
              let torsionBuf = makeBuffer(gpuTorsions),
              let refBuf = makeBuffer(refAtoms),
              let energyBuf = makeBuffer(count: natom, type: Float.self),
              let gradBuf = makeBuffer(count: natom, type: SIMD3<Float>.self)
        else {
            return LoopRefineOutput(positions: input.atoms.map(\.position), finalEnergy: .infinity, iterations: 0)
        }

        var bestEnergy: Float = .infinity
        var iteration = 0
        let stepSize: Float = 0.02  // Angstrom, conservative for loop building

        for iter in 0..<maxIterations {
            iteration = iter + 1

            // Zero energy and gradient buffers
            zeroBuffer(energyBuf, count: natom, type: Float.self)
            zeroBuffer(gradBuf, count: natom, type: SIMD3<Float>.self)

            // Upload current positions
            guard let atomBuf = makeBuffer(gpuAtoms) else { break }

            var params = LoopRefineParams(
                atomCount: UInt32(natom),
                bondCount: UInt32(gpuBonds.count),
                angleCount: UInt32(gpuAngles.count),
                torsionCount: UInt32(gpuTorsions.count),
                restraintK: 500.0,    // kcal/mol/A^2 for non-loop atoms
                stericCutoff: 8.0,    // Angstrom
                computeGrad: 1,
                _pad0: 0
            )

            // Dispatch all kernels sequentially
            dispatchBondKernel(atomBuf: atomBuf, bondBuf: bondBuf, params: &params,
                               energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)

            if !gpuAngles.isEmpty {
                dispatchAngleKernel(atomBuf: atomBuf, angleBuf: angleBuf, params: &params,
                                    energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)
            }

            if !gpuTorsions.isEmpty {
                dispatchTorsionKernel(atomBuf: atomBuf, torsionBuf: torsionBuf, params: &params,
                                      energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)
            }

            dispatchStericKernel(atomBuf: atomBuf, refBuf: refBuf, params: &params,
                                 energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)

            // Read results
            let ePtr = energyBuf.contents().bindMemory(to: Float.self, capacity: natom)
            let gPtr = gradBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: natom)

            var totalEnergy: Float = 0
            for i in 0..<natom { totalEnergy += ePtr[i] }

            // Check convergence
            var maxGrad: Float = 0
            for i in 0..<natom where input.isLoopAtom[i] {
                maxGrad = max(maxGrad, simd_length(gPtr[i]))
            }

            bestEnergy = totalEnergy
            if maxGrad < tolerance { break }

            // Steepest descent step: only move loop atoms
            for i in 0..<natom where input.isLoopAtom[i] {
                let grad = gPtr[i]
                let gradLen = simd_length(grad)
                guard gradLen > 1e-6 else { continue }

                // Clamp step to avoid explosions
                let scale = min(stepSize, 0.5 / gradLen)
                gpuAtoms[i].position -= grad * scale
            }
        }

        // Extract final positions
        var positions = input.atoms.map(\.position)
        for i in 0..<natom where input.isLoopAtom[i] {
            positions[i] = gpuAtoms[i].position
        }

        return LoopRefineOutput(positions: positions, finalEnergy: bestEnergy, iterations: iteration)
    }

    // MARK: - GPU Dispatch

    private func dispatchBondKernel(
        atomBuf: MTLBuffer, bondBuf: MTLBuffer,
        params: inout LoopRefineParams,
        energyBuf: MTLBuffer, gradBuf: MTLBuffer, natom: Int
    ) {
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(bondPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(bondBuf, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LoopRefineParams>.stride, index: 2)
        encoder.setBuffer(energyBuf, offset: 0, index: 3)
        encoder.setBuffer(gradBuf, offset: 0, index: 4)

        let (grid, tg) = threadgroupConfig(pipeline: bondPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private func dispatchAngleKernel(
        atomBuf: MTLBuffer, angleBuf: MTLBuffer,
        params: inout LoopRefineParams,
        energyBuf: MTLBuffer, gradBuf: MTLBuffer, natom: Int
    ) {
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(anglePipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(angleBuf, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LoopRefineParams>.stride, index: 2)
        encoder.setBuffer(energyBuf, offset: 0, index: 3)
        encoder.setBuffer(gradBuf, offset: 0, index: 4)

        let (grid, tg) = threadgroupConfig(pipeline: anglePipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private func dispatchTorsionKernel(
        atomBuf: MTLBuffer, torsionBuf: MTLBuffer,
        params: inout LoopRefineParams,
        energyBuf: MTLBuffer, gradBuf: MTLBuffer, natom: Int
    ) {
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(torsionPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(torsionBuf, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LoopRefineParams>.stride, index: 2)
        encoder.setBuffer(energyBuf, offset: 0, index: 3)
        encoder.setBuffer(gradBuf, offset: 0, index: 4)

        let (grid, tg) = threadgroupConfig(pipeline: torsionPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private func dispatchStericKernel(
        atomBuf: MTLBuffer, refBuf: MTLBuffer,
        params: inout LoopRefineParams,
        energyBuf: MTLBuffer, gradBuf: MTLBuffer, natom: Int
    ) {
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(stericPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(refBuf, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LoopRefineParams>.stride, index: 2)
        encoder.setBuffer(energyBuf, offset: 0, index: 3)
        encoder.setBuffer(gradBuf, offset: 0, index: 4)

        let (grid, tg) = threadgroupConfig(pipeline: stericPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Data Conversion

    private func buildGPUAtoms(input: LoopRefineInput) -> [LoopRefineAtom] {
        input.atoms.enumerated().map { (i, atom) in
            LoopRefineAtom(
                position: atom.position,
                mass: atom.element.mass,
                sigma: atom.element.vdwRadius * 0.8909,  // approximate LJ sigma
                epsilon: 0.086,                           // generic kcal/mol
                isLoop: input.isLoopAtom[i] ? 1 : 0,
                _pad0: 0
            )
        }
    }

    private func buildGPUBonds(input: LoopRefineInput) -> [LoopRefineBond] {
        input.bonds.compactMap { bond in
            let a1 = bond.atomIndex1
            let a2 = bond.atomIndex2
            guard a1 >= 0, a1 < input.atoms.count, a2 >= 0, a2 < input.atoms.count else { return nil }
            // Only include bonds involving at least one loop atom
            guard input.isLoopAtom[a1] || input.isLoopAtom[a2] else { return nil }
            let dist = simd_distance(input.atoms[a1].position, input.atoms[a2].position)
            return LoopRefineBond(
                atom1: UInt32(a1),
                atom2: UInt32(a2),
                length: dist,
                k: estimateBondK(z1: input.atoms[a1].element.rawValue, z2: input.atoms[a2].element.rawValue)
            )
        }
    }

    private func buildGPUAngles(input: LoopRefineInput) -> [LoopRefineAngle] {
        input.angles.map { (a1, a2, a3, angleDeg) in
            LoopRefineAngle(
                atom1: UInt32(a1),
                atom2: UInt32(a2),
                atom3: UInt32(a3),
                angle: angleDeg * .pi / 180.0,
                k: 80.0,  // kcal/mol/rad^2 (typical backbone)
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }
    }

    private func buildGPUTorsions(input: LoopRefineInput) -> [LoopRefineTorsion] {
        input.torsions.map { (a1, a2, a3, a4) in
            LoopRefineTorsion(
                atom1: UInt32(a1),
                atom2: UInt32(a2),
                atom3: UInt32(a3),
                atom4: UInt32(a4),
                phase: .pi,       // 180 degrees (trans peptide bond)
                k: 10.0,          // kcal/mol
                periodicity: 2,
                _pad0: 0
            )
        }
    }

    private func estimateBondK(z1: Int, z2: Int) -> Float {
        let minZ = min(z1, z2)
        let maxZ = max(z1, z2)
        if minZ == 6 && maxZ == 6  { return 310.0 }  // C-C
        if minZ == 6 && maxZ == 7  { return 337.0 }  // C-N
        if minZ == 6 && maxZ == 8  { return 360.0 }  // C-O
        if minZ == 6 && maxZ == 16 { return 237.0 }  // C-S
        if minZ == 7 && maxZ == 7  { return 337.0 }  // N-N
        return 300.0  // fallback
    }

    // MARK: - Utility

    private func zeroBuffer(_ buffer: MTLBuffer, count: Int, type: Any.Type) {
        memset(buffer.contents(), 0, buffer.length)
    }
}
