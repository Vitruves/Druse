// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// LoopMetalAccelerator.swift — GPU-accelerated loop refinement
//
// Manages Metal compute pipeline states for loop energy/gradient evaluation.
// Drives a steepest descent minimizer on CPU with GPU force evaluation.
//
// Architecture:
//   Swift minimizer loop → Metal compute dispatch (energy + gradient)
//   → CPU steepest-descent update (fixed step factor, per-atom displacement
//     clamped; no line search) → repeat until converged
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
              let library = device.makeDefaultLibrary(),
              let bondFn = library.makeFunction(name: "loop_bond_energy"),
              let angleFn = library.makeFunction(name: "loop_angle_energy"),
              let torsionFn = library.makeFunction(name: "loop_torsion_energy"),
              let stericFn = library.makeFunction(name: "loop_steric_restraint")
        else { return nil }

        self.device = device
        self.queue = queue

        do {
            bondPipeline = try device.makeComputePipelineState(function: bondFn)
            anglePipeline = try device.makeComputePipelineState(function: angleFn)
            torsionPipeline = try device.makeComputePipelineState(function: torsionFn)
            stericPipeline = try device.makeComputePipelineState(function: stericFn)
        } catch {
            return nil
        }
    }

    // MARK: - Buffer Helpers

    /// Shared buffer holding `data`. Empty arrays get a one-element placeholder
    /// (Metal rejects zero-length buffers; kernels never read it since the count is 0).
    private func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        data.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress, ptr.count > 0 else {
                return device.makeBuffer(length: MemoryLayout<T>.stride, options: .storageModeShared)
            }
            return device.makeBuffer(bytes: base,
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
    /// Steepest descent with a fixed step factor and per-atom displacement clamped to
    /// 0.5 Å (no line search); GPU energy+gradient evaluation.
    func refine(input: LoopRefineInput, maxIterations: Int = 500, tolerance: Float = 0.1) -> LoopRefineOutput {
        let natom = input.atoms.count

        let initialPositions = input.atoms.map(\.position)
        guard natom > 0 else {
            return LoopRefineOutput(positions: initialPositions, finalEnergy: 0, iterations: 0)
        }

        // Build GPU data structures
        let gpuAtoms = buildGPUAtoms(input: input)
        let gpuBonds = buildGPUBonds(input: input)
        let gpuAngles = buildGPUAngles(input: input)
        let gpuTorsions = buildGPUTorsions(input: input)

        // Persistent buffers. The atom buffer is updated in place each step;
        // the reference copy keeps the starting positions for restraints.
        guard let atomBuf = makeBuffer(gpuAtoms),
              let refBuf = makeBuffer(gpuAtoms),
              let bondBuf = makeBuffer(gpuBonds),
              let angleBuf = makeBuffer(gpuAngles),
              let torsionBuf = makeBuffer(gpuTorsions),
              let energyBuf = makeBuffer(count: natom, type: Float.self),
              let gradBuf = makeBuffer(count: natom, type: SIMD3<Float>.self)
        else {
            return LoopRefineOutput(positions: initialPositions, finalEnergy: .infinity, iterations: 0)
        }
        let atomPtr = atomBuf.contents().bindMemory(to: LoopRefineAtom.self, capacity: natom)

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

        var bestEnergy: Float = .infinity
        var iteration = 0
        let stepSize: Float = 0.02  // Angstrom, conservative for loop building

        for iter in 0..<maxIterations {
            iteration = iter + 1

            // Zero energy and gradient buffers
            memset(energyBuf.contents(), 0, energyBuf.length)
            memset(gradBuf.contents(), 0, gradBuf.length)

            // All terms in one command buffer: a serial compute encoder runs the
            // dispatches in order (bond kernel assigns, the others accumulate).
            guard let cmdBuf = queue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else { break }
            encode(encoder, pipeline: bondPipeline, atomBuf: atomBuf, termBuf: bondBuf,
                   params: &params, energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)
            if !gpuAngles.isEmpty {
                encode(encoder, pipeline: anglePipeline, atomBuf: atomBuf, termBuf: angleBuf,
                       params: &params, energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)
            }
            if !gpuTorsions.isEmpty {
                encode(encoder, pipeline: torsionPipeline, atomBuf: atomBuf, termBuf: torsionBuf,
                       params: &params, energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)
            }
            encode(encoder, pipeline: stericPipeline, atomBuf: atomBuf, termBuf: refBuf,
                   params: &params, energyBuf: energyBuf, gradBuf: gradBuf, natom: natom)
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            guard cmdBuf.status == .completed else { break }

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
                atomPtr[i].position -= grad * scale
            }
        }

        // Extract final positions
        var positions = initialPositions
        for i in 0..<natom where input.isLoopAtom[i] {
            positions[i] = atomPtr[i].position
        }

        return LoopRefineOutput(positions: positions, finalEnergy: bestEnergy, iterations: iteration)
    }

    // MARK: - GPU Dispatch

    /// Encode one loop-refinement kernel. All kernels share the binding layout
    /// (atoms, per-term data, params, energies, gradients) and run one thread per atom.
    private func encode(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        atomBuf: MTLBuffer, termBuf: MTLBuffer,
        params: inout LoopRefineParams,
        energyBuf: MTLBuffer, gradBuf: MTLBuffer, natom: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(termBuf, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LoopRefineParams>.stride, index: 2)
        encoder.setBuffer(energyBuf, offset: 0, index: 3)
        encoder.setBuffer(gradBuf, offset: 0, index: 4)

        let (grid, tg) = threadgroupConfig(pipeline: pipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
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
            // Backbone bonds of a freshly built loop start far from ideal, so their
            // target must be the ideal length (keeping the starting distance would
            // freeze the bad geometry). Other bonds come from ideal templates.
            let length = Self.idealBackboneBondLength(input.atoms[a1], input.atoms[a2])
                ?? simd_distance(input.atoms[a1].position, input.atoms[a2].position)
            return LoopRefineBond(
                atom1: UInt32(a1),
                atom2: UInt32(a2),
                length: length,
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

    /// Ideal backbone bond length (Engh & Huber) for N–CA, CA–C, C=O, CA–CB within
    /// a residue and the C–N peptide bond between residues; nil for other bonds.
    /// Also used for the OpenMM loop refinement bond targets.
    static func idealBackboneBondLength(_ a: Atom, _ b: Atom) -> Float? {
        let names: Set<String> = [a.name.trimmingCharacters(in: .whitespaces),
                                  b.name.trimmingCharacters(in: .whitespaces)]
        let sameResidue = a.chainID == b.chainID && a.residueSeq == b.residueSeq
            && a.insertionCode.trimmingCharacters(in: .whitespaces)
                == b.insertionCode.trimmingCharacters(in: .whitespaces)
        if sameResidue {
            if names == ["N", "CA"] { return 1.458 }
            if names == ["CA", "C"] { return 1.525 }
            if names == ["C", "O"] { return 1.231 }
            if names == ["CA", "CB"] { return 1.530 }
        } else if names == ["C", "N"] {
            return 1.335
        }
        return nil
    }
}
