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
// Pairwise VDW + positional restraints run on the GPU
// (PreparationMinimizeCompute.metal); harmonic bond terms are added on the CPU.
// Bonded pairs are excluded from the VDW term (1-2, 1-3 skipped; 1-4 scaled) via a
// host-built exclusion list, so covalent geometry is not distorted by LJ.
// ============================================================================

import Foundation
import Metal
import simd

final class PreparationMinimizer: @unchecked Sendable {

    let device: MTLDevice
    let queue: MTLCommandQueue

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
              let library = device.makeDefaultLibrary(),
              let vdwFunction = library.makeFunction(name: "prep_vdw_restraint"),
              let vdwPipeline = try? device.makeComputePipelineState(function: vdwFunction)
        else { return nil }

        self.device = device
        self.queue = queue
        self.vdwRestraintPipeline = vdwPipeline
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

        // Build bond data (harmonic, evaluated on the CPU)
        let loopBonds = input.bonds.compactMap { bond -> LoopRefineBond? in
            let i1 = bond.atomIndex1
            let i2 = bond.atomIndex2
            guard i1 < atomCount, i2 < atomCount else { return nil }
            let dist = simd_distance(input.atoms[i1].position, input.atoms[i2].position)
            return LoopRefineBond(atom1: UInt32(i1), atom2: UInt32(i2),
                                  length: max(dist, 0.5), k: 300.0)  // kcal/mol/A^2
        }

        // Reference atoms carry the restraint anchors plus the per-atom LJ parameters
        // (sigma, element epsilon) and region; they are constant for the run.
        // The kernel mixes sigma_ij = (sigma_i + sigma_j) / 2, so sigma_i must be the
        // like-pair value: the LJ minimum of a like pair is at the contact distance
        // 2 * R_vdW, hence sigma_i = 2 * R_vdW * 2^(-1/6). This puts the LJ minimum
        // of any pair at R_i + R_j (Bondi contact), not at their mean radius.
        let refAtoms: [PrepMinAtom] = input.atoms.indices.map { i in
            let atom = input.atoms[i]
            let region = i < input.atomRegions.count ? input.atomRegions[i] : .existingSidechain
            return PrepMinAtom(
                position: atom.position,
                sigma: 2 * atom.element.vdwRadius * 0.8908987,  // 2^(-1/6) * r_min
                epsilon: elementEpsilon(atom.element),
                region: region.rawValue,
                _pad0: 0
            )
        }

        var params = PrepMinParams(
            atomCount: UInt32(atomCount),
            restraintK_backbone: restraintK_backbone,
            restraintK_existing: restraintK_existing,
            restraintK_reconstructed: restraintK_reconstructed,
            stericCutoff: stericCutoff,
            computeGrad: 1,
            _pad0: 0,
            _pad1: 0
        )

        let exclusions = Self.buildExclusions(atomCount: atomCount, bonds: input.bonds)

        // Buffers are allocated once and reused by every energy/gradient evaluation;
        // only the current positions are rewritten in `atomBuf` per call.
        // Metal buffers cannot be empty, so the partner/scale arrays get a dummy entry.
        guard let atomBuf = makeBuffer(refAtoms),
              let refBuf = makeBuffer(refAtoms),
              let paramBuf = device.makeBuffer(bytes: &params,
                                               length: MemoryLayout<PrepMinParams>.stride,
                                               options: .storageModeShared),
              let energyBuf = makeBuffer(count: atomCount, type: Float.self),
              let gradBuf = makeBuffer(count: atomCount, type: SIMD3<Float>.self),
              let exclOffsetBuf = makeBuffer(exclusions.offsets),
              let exclPartnerBuf = makeBuffer(exclusions.partners.isEmpty ? [UInt32(0)] : exclusions.partners),
              let exclScaleBuf = makeBuffer(exclusions.scales.isEmpty ? [Float(0)] : exclusions.scales)
        else {
            return MinimizationOutput(positions: input.atoms.map(\.position), finalEnergy: 0,
                                      iterations: 0, converged: false)
        }
        let buffers = EvaluationBuffers(atoms: atomBuf, reference: refBuf, params: paramBuf,
                                        energies: energyBuf, gradients: gradBuf,
                                        exclusionOffsets: exclOffsetBuf,
                                        exclusionPartners: exclPartnerBuf,
                                        exclusionScales: exclScaleBuf)

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
                    atomCount: atomCount,
                    bonds: loopBonds,
                    buffers: buffers
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

    // MARK: - Bonded Exclusions

    /// LJ scale for 1-4 pairs. AMBER force fields divide 1-4 van der Waals by
    /// SCNB = 2.0 (Cornell et al., J. Am. Chem. Soc. 1995, 117, 5179).
    private static let oneFourVDWScale: Float = 0.5

    /// CSR exclusion list: for atom i, partners[offsets[i] ..< offsets[i+1]] in
    /// ascending order, with scale 0 for 1-2/1-3 pairs and `oneFourVDWScale` for 1-4.
    private static func buildExclusions(
        atomCount: Int,
        bonds: [Bond]
    ) -> (offsets: [UInt32], partners: [UInt32], scales: [Float]) {
        var neighbors = [[Int]](repeating: [], count: atomCount)
        for bond in bonds where bond.atomIndex1 < atomCount && bond.atomIndex2 < atomCount
            && bond.atomIndex1 != bond.atomIndex2 {
            neighbors[bond.atomIndex1].append(bond.atomIndex2)
            neighbors[bond.atomIndex2].append(bond.atomIndex1)
        }

        var offsets = [UInt32](repeating: 0, count: atomCount + 1)
        var partners: [UInt32] = []
        var scales: [Float] = []

        for i in 0..<atomCount {
            var pairScale: [Int: Float] = [:]
            for j in neighbors[i] {
                pairScale[j] = 0                                   // 1-2
                for k in neighbors[j] where k != i {
                    pairScale[k] = 0                               // 1-3
                }
            }
            for j in neighbors[i] {
                for k in neighbors[j] where k != i {
                    for l in neighbors[k] where l != i && l != j && pairScale[l] == nil {
                        pairScale[l] = oneFourVDWScale             // 1-4
                    }
                }
            }
            for partner in pairScale.keys.sorted() {
                partners.append(UInt32(partner))
                scales.append(pairScale[partner]!)
            }
            offsets[i + 1] = UInt32(partners.count)
        }

        return (offsets, partners, scales)
    }

    // MARK: - Energy + Gradient Evaluation

    private struct EvaluationBuffers {
        let atoms: MTLBuffer
        let reference: MTLBuffer
        let params: MTLBuffer
        let energies: MTLBuffer
        let gradients: MTLBuffer
        let exclusionOffsets: MTLBuffer
        let exclusionPartners: MTLBuffer
        let exclusionScales: MTLBuffer
    }

    private func evaluateEnergyAndGradient(
        positions: [Float],
        gradient: inout [Float],
        atomCount: Int,
        bonds: [LoopRefineBond],
        buffers: EvaluationBuffers
    ) -> Float {
        // Update current positions in place (sigma/epsilon/region stay as initialized)
        let atomPtr = buffers.atoms.contents().bindMemory(to: PrepMinAtom.self, capacity: atomCount)
        for i in 0..<atomCount {
            atomPtr[i].position = SIMD3<Float>(positions[i*3], positions[i*3+1], positions[i*3+2])
        }

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else {
            return Float.greatestFiniteMagnitude
        }

        // VDW + Restraint kernel
        encoder.setComputePipelineState(vdwRestraintPipeline)
        encoder.setBuffer(buffers.atoms, offset: 0, index: 0)
        encoder.setBuffer(buffers.reference, offset: 0, index: 1)
        encoder.setBuffer(buffers.params, offset: 0, index: 2)
        encoder.setBuffer(buffers.energies, offset: 0, index: 3)
        encoder.setBuffer(buffers.gradients, offset: 0, index: 4)
        encoder.setBuffer(buffers.exclusionOffsets, offset: 0, index: 5)
        encoder.setBuffer(buffers.exclusionPartners, offset: 0, index: 6)
        encoder.setBuffer(buffers.exclusionScales, offset: 0, index: 7)
        let (grid, tg) = threadgroupConfig(pipeline: vdwRestraintPipeline, count: atomCount)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back results
        var totalEnergy: Float = 0
        let energyPtr = buffers.energies.contents().bindMemory(to: Float.self, capacity: atomCount)
        for i in 0..<atomCount {
            totalEnergy += energyPtr[i]
        }

        let gradPtr = buffers.gradients.contents().bindMemory(to: SIMD3<Float>.self, capacity: atomCount)
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
