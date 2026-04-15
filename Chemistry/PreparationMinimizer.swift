// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PreparationMinimizer.swift — Post-preparation energy minimization
//
// Runs L-BFGS minimization with GPU-accelerated energy/gradient evaluation
// after sidechain packing and before hydrogen addition. Uses region-based
// restraints: backbone atoms are strongly restrained, existing sidechains
// moderately, and reconstructed/repacked atoms are free.
//
// Reuses bond/angle/torsion kernels from LoopRefineCompute.metal and adds
// a VDW+restraint kernel from PreparationMinimizeCompute.metal.
// ============================================================================

import Foundation
import Metal
import simd

final class PreparationMinimizer: @unchecked Sendable {

    let device: MTLDevice
    let queue: MTLCommandQueue

    private let bondPipeline: MTLComputePipelineState
    private let anglePipeline: MTLComputePipelineState
    private let torsionPipeline: MTLComputePipelineState
    private let vdwRestraintPipeline: MTLComputePipelineState

    // MARK: - Types

    enum AtomRegion: UInt32, Sendable {
        case backbone           = 0   // strong restraint
        case existingSidechain  = 1   // medium restraint
        case reconstructed      = 2   // free
        case hydrogen           = 3   // free
    }

    struct MinimizationInput {
        let atoms: [Atom]
        let bonds: [Bond]
        let atomRegions: [AtomRegion]
        let angles: [(Int, Int, Int, Float)]    // (atom1, atom2_central, atom3, angleDeg)
        let torsions: [(Int, Int, Int, Int)]     // omega torsions
    }

    struct MinimizationOutput {
        let positions: [SIMD3<Float>]
        let finalEnergy: Float
        let iterations: Int
        let converged: Bool
    }

    // MARK: - Init

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
            vdwRestraintPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "prep_vdw_restraint")!)
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

    // MARK: - Minimize

    func minimize(
        input: MinimizationInput,
        maxIterations: Int = 200,
        gradientTolerance: Float = 0.5,
        restraintK_backbone: Float = 500.0,
        restraintK_existing: Float = 50.0,
        restraintK_reconstructed: Float = 0.0,
        stericCutoff: Float = 8.0
    ) -> MinimizationOutput {

        let atomCount = input.atoms.count
        guard atomCount > 0 else {
            return MinimizationOutput(positions: [], finalEnergy: 0, iterations: 0, converged: true)
        }

        // Flatten positions to [Float] array (3N)
        let n = atomCount * 3
        var x = [Float](repeating: 0, count: n)
        for i in 0..<atomCount {
            x[i * 3] = input.atoms[i].position.x
            x[i * 3 + 1] = input.atoms[i].position.y
            x[i * 3 + 2] = input.atoms[i].position.z
        }

        // Reference positions for restraints
        let refPositions = input.atoms.map(\.position)

        // Prepare GPU data
        let regions = input.atomRegions.map(\.rawValue)

        // Build bond data for LoopRefine kernels
        let loopBonds = input.bonds.compactMap { bond -> LoopRefineBond? in
            let i1 = bond.atomIndex1
            let i2 = bond.atomIndex2
            guard i1 < atomCount, i2 < atomCount else { return nil }
            let dist = simd_distance(input.atoms[i1].position, input.atoms[i2].position)
            return LoopRefineBond(atom1: UInt32(i1), atom2: UInt32(i2),
                                  length: max(dist, 0.5), k: 300.0)  // kcal/mol/A^2
        }

        // Prepare VDW parameters from elements
        let sigma: [Float] = input.atoms.map { $0.element.vdwRadius * 0.8908987 }  // r_min -> sigma
        let epsilon: [Float] = input.atoms.map { elementEpsilon($0.element) }

        let params = PrepMinParams(
            atomCount: UInt32(atomCount),
            restraintK_backbone: restraintK_backbone,
            restraintK_existing: restraintK_existing,
            restraintK_reconstructed: restraintK_reconstructed,
            stericCutoff: stericCutoff,
            computeGrad: 1,
            _pad0: 0,
            _pad1: 0
        )

        // L-BFGS minimization
        let result = LBFGSOptimizer.minimize(
            params: LBFGSOptimizer.Parameters(
                m: 8,
                maxIterations: maxIterations,
                epsilon: gradientTolerance,
                ftol: 1e-4,
                wolfe: 0.9,
                maxLineSearch: 20
            ),
            n: n,
            x0: &x,
            evaluate: { [self] (positions: inout [Float], gradient: inout [Float]) -> Float in
                evaluateEnergyAndGradient(
                    positions: positions,
                    gradient: &gradient,
                    refPositions: refPositions,
                    regions: regions,
                    sigma: sigma,
                    epsilon: epsilon,
                    bonds: loopBonds,
                    params: params
                )
            }
        )

        // Convert back to SIMD3<Float>
        var finalPositions = [SIMD3<Float>](repeating: .zero, count: atomCount)
        for i in 0..<atomCount {
            finalPositions[i] = SIMD3<Float>(x[i*3], x[i*3+1], x[i*3+2])
        }

        return MinimizationOutput(
            positions: finalPositions,
            finalEnergy: result.energy,
            iterations: result.iterations,
            converged: result.converged
        )
    }

    // MARK: - Energy + Gradient Evaluation

    private func evaluateEnergyAndGradient(
        positions: [Float],
        gradient: inout [Float],
        refPositions: [SIMD3<Float>],
        regions: [UInt32],
        sigma: [Float],
        epsilon: [Float],
        bonds: [LoopRefineBond],
        params: PrepMinParams
    ) -> Float {

        let atomCount = Int(params.atomCount)

        // Build PrepMinAtom array from current positions
        var prepAtoms = [PrepMinAtom]()
        prepAtoms.reserveCapacity(atomCount)
        for i in 0..<atomCount {
            prepAtoms.append(PrepMinAtom(
                position: SIMD3<Float>(positions[i*3], positions[i*3+1], positions[i*3+2]),
                sigma: sigma[i],
                epsilon: epsilon[i],
                region: regions[i],
                _pad0: 0
            ))
        }

        // Reference atoms
        var refAtoms = [PrepMinAtom]()
        refAtoms.reserveCapacity(atomCount)
        for i in 0..<atomCount {
            refAtoms.append(PrepMinAtom(
                position: refPositions[i],
                sigma: sigma[i],
                epsilon: epsilon[i],
                region: regions[i],
                _pad0: 0
            ))
        }

        // GPU dispatch
        guard let atomBuf = makeBuffer(prepAtoms),
              let refBuf = makeBuffer(refAtoms),
              let paramBuf = makeBuffer([params]),
              let energyBuf = makeBuffer(count: atomCount, type: Float.self),
              let gradBuf = makeBuffer(count: atomCount, type: SIMD3<Float>.self),
              let cmdBuf = queue.makeCommandBuffer()
        else {
            return Float.greatestFiniteMagnitude
        }

        // VDW + Restraint kernel
        if let encoder = cmdBuf.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(vdwRestraintPipeline)
            encoder.setBuffer(atomBuf, offset: 0, index: 0)
            encoder.setBuffer(refBuf, offset: 0, index: 1)
            encoder.setBuffer(paramBuf, offset: 0, index: 2)
            encoder.setBuffer(energyBuf, offset: 0, index: 3)
            encoder.setBuffer(gradBuf, offset: 0, index: 4)
            let (grid, tg) = threadgroupConfig(pipeline: vdwRestraintPipeline, count: atomCount)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back results
        var totalEnergy: Float = 0
        let energyPtr = energyBuf.contents().bindMemory(to: Float.self, capacity: atomCount)
        for i in 0..<atomCount {
            totalEnergy += energyPtr[i]
        }

        let gradPtr = gradBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: atomCount)
        for i in 0..<atomCount {
            let g = gradPtr[i]
            gradient[i*3]     = g.x
            gradient[i*3 + 1] = g.y
            gradient[i*3 + 2] = g.z
        }

        // Add bond energy (CPU, simple harmonic)
        for bond in bonds {
            let i1 = Int(bond.atom1)
            let i2 = Int(bond.atom2)
            let p1 = SIMD3<Float>(positions[i1*3], positions[i1*3+1], positions[i1*3+2])
            let p2 = SIMD3<Float>(positions[i2*3], positions[i2*3+1], positions[i2*3+2])
            let diff = p1 - p2
            let r = simd_length(diff)
            guard r > 1e-6 else { continue }
            let dr = r - bond.length
            let k = bond.k
            totalEnergy += 0.5 * k * dr * dr
            let dEdr = k * dr / r
            let gContrib = dEdr * diff
            gradient[i1*3]     += gContrib.x
            gradient[i1*3 + 1] += gContrib.y
            gradient[i1*3 + 2] += gContrib.z
            gradient[i2*3]     -= gContrib.x
            gradient[i2*3 + 1] -= gContrib.y
            gradient[i2*3 + 2] -= gContrib.z
        }

        return totalEnergy
    }

    // MARK: - Element VDW Epsilon

    private func elementEpsilon(_ element: Element) -> Float {
        switch element {
        case .C:  return 0.086
        case .N:  return 0.17
        case .O:  return 0.21
        case .S:  return 0.25
        case .H:  return 0.015
        case .F:  return 0.061
        case .Cl: return 0.265
        case .Br: return 0.32
        case .P:  return 0.20
        default:  return 0.10
        }
    }
}

// MARK: - Metal-compatible structs (matching ShaderTypes.h layout)

private struct LoopRefineBond {
    var atom1: UInt32
    var atom2: UInt32
    var length: Float
    var k: Float
}

/// Must match PrepMinAtom in ShaderTypes.h exactly (32 bytes).
struct PrepMinAtom {
    var position: SIMD3<Float>
    var sigma: Float
    var epsilon: Float
    var region: UInt32
    var _pad0: Float
}

/// Must match PrepMinParams in ShaderTypes.h exactly (32 bytes).
struct PrepMinParams {
    var atomCount: UInt32
    var restraintK_backbone: Float
    var restraintK_existing: Float
    var restraintK_reconstructed: Float
    var stericCutoff: Float
    var computeGrad: UInt32
    var _pad0: UInt32
    var _pad1: UInt32
}
