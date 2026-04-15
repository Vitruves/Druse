// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - Diffusion Variance Schedule

/// Reverse diffusion noise schedule for pose generation.
struct DiffusionSchedule: Sendable {
    let T: Int
    let betas: [Float]          // noise schedule β_t
    let alphas: [Float]         // α_t = 1 - β_t
    let alphaCum: [Float]       // ᾱ_t = ∏ α_i
    let sigmas: [Float]         // σ_t = sqrt(1 - ᾱ_t)

    static func cosine(steps: Int) -> DiffusionSchedule {
        let T = steps
        var betas = [Float](repeating: 0, count: T)
        var alphas = [Float](repeating: 0, count: T)
        var alphaCum = [Float](repeating: 0, count: T)
        var sigmas = [Float](repeating: 0, count: T)

        let s = Float(0.008) // small offset to prevent β_0 = 0
        for t in 0..<T {
            let t0 = Float(t) / Float(T)
            let t1 = Float(t + 1) / Float(T)
            let a0 = cos((t0 + s) / (1 + s) * .pi / 2)
            let a1 = cos((t1 + s) / (1 + s) * .pi / 2)
            let beta = min(1 - (a1 * a1) / (a0 * a0), 0.999)
            betas[t] = beta
            alphas[t] = 1 - beta
            alphaCum[t] = t == 0 ? alphas[0] : alphaCum[t - 1] * alphas[t]
            sigmas[t] = sqrt(1 - alphaCum[t])
        }

        return DiffusionSchedule(T: T, betas: betas, alphas: alphas, alphaCum: alphaCum, sigmas: sigmas)
    }

    static func linear(steps: Int, betaStart: Float = 0.0001, betaEnd: Float = 0.02) -> DiffusionSchedule {
        let T = steps
        var betas = [Float](repeating: 0, count: T)
        var alphas = [Float](repeating: 0, count: T)
        var alphaCum = [Float](repeating: 0, count: T)
        var sigmas = [Float](repeating: 0, count: T)

        for t in 0..<T {
            betas[t] = betaStart + (betaEnd - betaStart) * Float(t) / Float(T - 1)
            alphas[t] = 1 - betas[t]
            alphaCum[t] = t == 0 ? alphas[0] : alphaCum[t - 1] * alphas[t]
            sigmas[t] = sqrt(1 - alphaCum[t])
        }

        return DiffusionSchedule(T: T, betas: betas, alphas: alphas, alphaCum: alphaCum, sigmas: sigmas)
    }

    static func quadratic(steps: Int) -> DiffusionSchedule {
        let T = steps
        var betas = [Float](repeating: 0, count: T)
        var alphas = [Float](repeating: 0, count: T)
        var alphaCum = [Float](repeating: 0, count: T)
        var sigmas = [Float](repeating: 0, count: T)

        let betaStart: Float = 0.0001
        let betaEnd: Float = 0.02
        for t in 0..<T {
            let frac = Float(t) / Float(T - 1)
            betas[t] = betaStart + (betaEnd - betaStart) * frac * frac
            alphas[t] = 1 - betas[t]
            alphaCum[t] = t == 0 ? alphas[0] : alphaCum[t - 1] * alphas[t]
            sigmas[t] = sqrt(1 - alphaCum[t])
        }

        return DiffusionSchedule(T: T, betas: betas, alphas: alphas, alphaCum: alphaCum, sigmas: sigmas)
    }
}

// MARK: - Diffusion Docking Engine

/// Orchestrates DruseAF attention-guided reverse diffusion for pose generation.
///
/// Requires DruseAF weights to be loaded. Falls back to GA search if unavailable.
@MainActor
final class DiffusionDockingEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private var diffInitNoisePipeline: MTLComputePipelineState?
    private var diffDenoisingStepPipeline: MTLComputePipelineState?
    private var druseAFScoreWithGradPipeline: MTLComputePipelineState?

    // Attention gradient buffer (per-pose × per-atom)
    private var attnGradientBuffer: MTLBuffer?
    private var diffParamsBuffer: MTLBuffer?

    init?(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue

        guard let library = device.makeDefaultLibrary() else { return nil }

        do {
            if let f = library.makeFunction(name: "diffusionInitNoise") {
                diffInitNoisePipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "diffusionDenoisingStep") {
                diffDenoisingStepPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "druseAFScoreWithGradient") {
                druseAFScoreWithGradPipeline = try device.makeComputePipelineState(function: f)
            }
        } catch {
            ActivityLog.shared.error("[DiffusionDocking] Failed to create pipelines: \(error)", category: .dock)
            return nil
        }
    }

    var isAvailable: Bool {
        diffInitNoisePipeline != nil
            && diffDenoisingStepPipeline != nil
            && druseAFScoreWithGradPipeline != nil
    }

    /// Run diffusion-guided docking.
    ///
    /// - Parameters:
    ///   - dockingEngine: Parent engine (for DruseAF setup/encode and Vina local search)
    ///   - gpuLigAtoms: Prepared GPU ligand atoms
    ///   - numTorsions: Number of rotatable bonds
    ///   - config: Diffusion docking configuration
    ///   - druseAFBuffers: Pre-configured DruseAF buffers from DockingEngine
    ///   - gaParams: GA params (for atom count, torsion info)
    ///   - gridParams: Grid params buffer
    ///   - ligandAtomBuffer: Ligand atom buffer
    ///   - torsionEdgeBuffer: Torsion edge buffer
    ///   - movingIndicesBuffer: Moving indices buffer
    /// - Returns: DockPose array sorted by energy
    func runDiffusionDocking(
        numPoses: Int,
        numLigandAtoms: Int,
        numTorsions: Int,
        config: DiffusionDockingConfig,
        gaParamsBuffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        ligandAtomBuffer: MTLBuffer,
        torsionEdgeBuffer: MTLBuffer,
        movingIndicesBuffer: MTLBuffer,
        // DruseAF buffers
        druseAFParamsBuffer: MTLBuffer,
        druseAFProtPosBuffer: MTLBuffer,
        druseAFWeightBuffer: MTLBuffer,
        druseAFEntryBuffer: MTLBuffer,
        druseAFSetupBuffer: MTLBuffer,
        druseAFIntermediateBuffer: MTLBuffer,
        druseAFEncodePipeline: MTLComputePipelineState,
        // Population buffer (output)
        populationBuffer: MTLBuffer
    ) async -> Int {
        guard let initPipe = diffInitNoisePipeline,
              let denoisePipe = diffDenoisingStepPipeline,
              let scoreGradPipe = druseAFScoreWithGradPipeline
        else {
            ActivityLog.shared.error("[DiffusionDocking] Pipelines not available", category: .dock)
            return 0
        }

        let T = config.numDenoisingSteps
        let N = numPoses
        let L = numLigandAtoms

        // Build noise schedule
        let schedule: DiffusionSchedule
        switch config.noiseSchedule {
        case .cosine:    schedule = .cosine(steps: T)
        case .linear:    schedule = .linear(steps: T)
        case .quadratic: schedule = .quadratic(steps: T)
        }

        // Allocate attention gradient buffer
        let gradBufSize = N * L * MemoryLayout<AttentionGradient>.stride
        if attnGradientBuffer == nil || attnGradientBuffer!.length < gradBufSize {
            attnGradientBuffer = device.makeBuffer(length: gradBufSize, options: .storageModeShared)
        }
        guard let gradBuf = attnGradientBuffer else { return 0 }

        // Build diffusion params
        var diffParams = DiffusionParams(
            numPoses: UInt32(N),
            currentStep: UInt32(T),
            totalSteps: UInt32(T),
            noiseScale: schedule.sigmas.last ?? 1.0,
            guidanceScale: config.guidanceScale,
            numLigandAtoms: UInt32(L),
            numTorsions: UInt32(numTorsions),
            translationNoise: 1.0,   // will be updated per step
            rotationNoise: 0.3,
            torsionNoise: Float.pi,
            _pad0: 0, _pad1: 0
        )
        if diffParamsBuffer == nil {
            diffParamsBuffer = device.makeBuffer(length: MemoryLayout<DiffusionParams>.stride, options: .storageModeShared)
        }
        guard let diffBuf = diffParamsBuffer else { return 0 }

        let tgSize = MTLSize(width: min(N, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        let simdTgSize = MTLSize(width: 32, height: 1, depth: 1)
        let simdTgCount = MTLSize(width: N, height: 1, depth: 1)

        // Step 1: Initialize from noise
        diffParams.noiseScale = 1.0
        diffParams.translationNoise = 1.0
        diffParams.rotationNoise = 0.3
        diffParams.torsionNoise = .pi
        diffBuf.contents().copyMemory(from: &diffParams, byteCount: MemoryLayout<DiffusionParams>.stride)

        dispatchSync(pipeline: initPipe, buffers: [
            (populationBuffer, 0), (gridParamsBuffer, 1), (diffBuf, 2), (gaParamsBuffer, 3)
        ], threadGroups: tgCount, threadGroupSize: tgSize)

        ActivityLog.shared.info("[DiffusionDocking] Initialized \(N) noise poses, running \(T) denoising steps", category: .dock)

        // Step 2: Reverse diffusion loop
        for t in stride(from: T - 1, through: 0, by: -1) {
            diffParams.currentStep = UInt32(t)
            let sigma_t = schedule.sigmas[t]
            let sigma_prev = t > 0 ? schedule.sigmas[t - 1] : 0.0
            diffParams.noiseScale = sigma_prev
            diffParams.translationNoise = sigma_t
            diffParams.rotationNoise = sigma_t * 0.3
            diffParams.torsionNoise = sigma_t * .pi
            diffBuf.contents().copyMemory(from: &diffParams, byteCount: MemoryLayout<DiffusionParams>.stride)

            // 2a: DruseAF encode (transform ligand positions for this pose)
            let encodeBuffers: [(MTLBuffer, Int)] = [
                (populationBuffer, 0), (ligandAtomBuffer, 1), (gaParamsBuffer, 2),
                (torsionEdgeBuffer, 3), (movingIndicesBuffer, 4),
                (druseAFParamsBuffer, 5), (druseAFIntermediateBuffer, 6), (gridParamsBuffer, 7)
            ]
            dispatchSync(pipeline: druseAFEncodePipeline, buffers: encodeBuffers,
                         threadGroups: tgCount, threadGroupSize: tgSize)

            // 2b: DruseAF score with gradient
            let scoreGradBuffers: [(MTLBuffer, Int)] = [
                (populationBuffer, 0), (druseAFProtPosBuffer, 1),
                (druseAFWeightBuffer, 2), (druseAFEntryBuffer, 3),
                (druseAFParamsBuffer, 4), (druseAFIntermediateBuffer, 5),
                (druseAFSetupBuffer, 6), (gradBuf, 7)
            ]
            dispatchSync(pipeline: scoreGradPipe, buffers: scoreGradBuffers,
                         threadGroups: simdTgCount, threadGroupSize: simdTgSize)

            // 2c: Denoising step (use attention gradients to guide poses)
            let denoiseBuffers: [(MTLBuffer, Int)] = [
                (populationBuffer, 0), (gradBuf, 1), (diffBuf, 2),
                (gaParamsBuffer, 3), (gridParamsBuffer, 4), (ligandAtomBuffer, 5)
            ]
            dispatchSync(pipeline: denoisePipe, buffers: denoiseBuffers,
                         threadGroups: tgCount, threadGroupSize: tgSize)

            await Task.yield()
        }

        ActivityLog.shared.info("[DiffusionDocking] Diffusion completed, \(N) poses generated", category: .dock)
        return N
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
