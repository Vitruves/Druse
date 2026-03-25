// ============================================================================
// XTBMetalAccelerator.swift — GPU dispatch for GFN2-xTB pairwise kernels
//
// Manages Metal compute pipeline states and buffers for offloading O(N²)
// atom-pair computations from the C++ xTB implementation.
//
// Architecture:
//   C++ SCC loop → C function pointers (DruseXTBGPUContext) → this class
//   → Metal compute dispatch → results copied back to C++ (double arrays)
//
// The C++ code uses double precision; Metal uses float. Conversions happen
// at the boundary (here). For drug-sized molecules the float precision is
// adequate for pairwise energy/gradient terms.
// ============================================================================

import Foundation
import Metal
import simd

// MARK: - XTBMetalAccelerator

/// GPU accelerator for GFN2-xTB O(N²) pairwise computations.
/// Thread-safe: all dispatches are synchronous (waitUntilCompleted).
@MainActor
final class XTBMetalAccelerator {

    // Singleton — initialized once, used by C callbacks
    static var shared: XTBMetalAccelerator?

    let device: MTLDevice
    let queue: MTLCommandQueue

    // Pipeline states for each kernel
    private let cnPipeline: MTLComputePipelineState
    private let d4Pipeline: MTLComputePipelineState
    private let repPipeline: MTLComputePipelineState
    private let bornPipeline: MTLComputePipelineState
    private let gbPipeline: MTLComputePipelineState
    private let cnGradPipeline: MTLComputePipelineState

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary()
        else { return nil }

        self.device = device
        self.queue = queue

        do {
            cnPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "xtb_compute_cn")!)
            d4Pipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "xtb_d4_dispersion")!)
            repPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "xtb_repulsion")!)
            bornPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "xtb_born_radii")!)
            gbPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "xtb_gb_solvation")!)
            cnGradPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "xtb_cn_gradient")!)
        } catch {
            print("[XTBMetalAccelerator] Failed to create pipelines: \(error)")
            return nil
        }
    }

    // MARK: - Dispatch Helpers

    nonisolated private func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        data.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count * MemoryLayout<T>.stride,
                              options: .storageModeShared)
        }
    }

    nonisolated private func makeBuffer<T>(count: Int, type: T.Type = T.self) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<T>.stride, options: .storageModeShared)
    }

    nonisolated private func threadgroupConfig(pipeline: MTLComputePipelineState, count: Int) -> (MTLSize, MTLSize) {
        let threadWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let tgSize = MTLSize(width: threadWidth, height: 1, depth: 1)
        return (gridSize, tgSize)
    }

    // MARK: - Coordination Number

    /// Compute coordination numbers on GPU.
    /// Input: positions (Bohr, double), atomic numbers.
    /// Output: cn array (double).
    nonisolated func computeCN(positions: UnsafePointer<Double>, Z: UnsafePointer<Int32>,
                               natom: Int, cnOut: UnsafeMutablePointer<Double>) {
        // Build CN atom buffer
        var cnAtoms = [GFN2CNAtom]()
        cnAtoms.reserveCapacity(natom)
        for i in 0..<natom {
            let p = SIMD3<Float>(Float(positions[3*i]), Float(positions[3*i+1]), Float(positions[3*i+2]))
            let covRad = Float(getCovRadBohr(Int(Z[i])))
            cnAtoms.append(GFN2CNAtom(position: p, covRadius: covRad))
        }

        var params = GFN2CNParams(atomCount: UInt32(natom), _pad0: 0, _pad1: 0, _pad2: 0)

        guard let atomBuf = makeBuffer(cnAtoms),
              let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<GFN2CNParams>.stride, options: .storageModeShared),
              let outBuf = makeBuffer(count: natom, type: Float.self),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return }

        encoder.setComputePipelineState(cnPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.setBuffer(outBuf, offset: 0, index: 2)
        let (grid, tg) = threadgroupConfig(pipeline: cnPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: natom)
        for i in 0..<natom { cnOut[i] = Double(ptr[i]) }
    }

    // MARK: - D4 Dispersion

    /// Compute D4 dispersion energy and optional gradient on GPU.
    /// Returns total dispersion energy in Hartree.
    nonisolated func computeD4Dispersion(positions: UnsafePointer<Double>, Z: UnsafePointer<Int32>,
                                         natom: Int, cn: UnsafePointer<Double>,
                                         gradient: UnsafeMutablePointer<Double>?) -> Double {
        var dispAtoms = [GFN2DispAtom]()
        dispAtoms.reserveCapacity(natom)
        for i in 0..<natom {
            let p = SIMD3<Float>(Float(positions[3*i]), Float(positions[3*i+1]), Float(positions[3*i+2]))
            let zi = Int(Z[i])
            let c6ref = Float(getRefC6(zi))
            let cnRef = Float(getCNRef(zi))
            let qDip = Float(sqrt(getRefC6(zi)) * 2.5)
            dispAtoms.append(GFN2DispAtom(
                position: p, c6ref: c6ref, cn: Float(cn[i]), cnRef: cnRef,
                qDipole: qDip, _pad0: 0))
        }

        let hasGrad = gradient != nil
        var params = GFN2DispParams(
            atomCount: UInt32(natom), s6: 1.0, s8: 2.7, a1: 0.52, a2_bohr: 5.0,
            computeGrad: hasGrad ? 1 : 0, _pad0: 0, _pad1: 0)

        guard let atomBuf = makeBuffer(dispAtoms),
              let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<GFN2DispParams>.stride, options: .storageModeShared),
              let energyBuf = makeBuffer(count: natom, type: Float.self),
              let gradBuf = makeBuffer(count: natom, type: SIMD3<Float>.self),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return 0.0 }

        encoder.setComputePipelineState(d4Pipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.setBuffer(energyBuf, offset: 0, index: 2)
        encoder.setBuffer(gradBuf, offset: 0, index: 3)
        let (grid, tg) = threadgroupConfig(pipeline: d4Pipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Sum per-atom partial energies
        let ePtr = energyBuf.contents().bindMemory(to: Float.self, capacity: natom)
        var Edisp: Double = 0.0
        for i in 0..<natom { Edisp += Double(ePtr[i]) }

        // Copy gradient (the C++ caller pre-zeroes with memset, so = is correct here;
        // the GPU kernel already accumulated all pair contributions per atom)
        if hasGrad, let grad = gradient {
            let gPtr = gradBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: natom)
            for i in 0..<natom {
                grad[3*i]   += Double(gPtr[i].x)
                grad[3*i+1] += Double(gPtr[i].y)
                grad[3*i+2] += Double(gPtr[i].z)
            }
        }

        return Edisp
    }

    // MARK: - Repulsion

    /// Compute repulsion energy and optional gradient on GPU.
    nonisolated func computeRepulsion(positions: UnsafePointer<Double>, Z: UnsafePointer<Int32>,
                                      natom: Int, gradient: UnsafeMutablePointer<Double>?) -> Double {
        var repAtoms = [GFN2RepAtom]()
        repAtoms.reserveCapacity(natom)
        for i in 0..<natom {
            let p = SIMD3<Float>(Float(positions[3*i]), Float(positions[3*i+1]), Float(positions[3*i+2]))
            let zi = Int(Z[i])
            let (zeff, arep) = getRepParams(zi)
            repAtoms.append(GFN2RepAtom(
                position: p, zeff: Float(zeff), arep: Float(arep),
                _pad0: 0, _pad1: 0, _pad2: 0))
        }

        let hasGrad = gradient != nil
        var params = GFN2RepParams(
            atomCount: UInt32(natom), kexp: 1.5, computeGrad: hasGrad ? 1 : 0, _pad0: 0)

        guard let atomBuf = makeBuffer(repAtoms),
              let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<GFN2RepParams>.stride, options: .storageModeShared),
              let energyBuf = makeBuffer(count: natom, type: Float.self),
              let gradBuf = makeBuffer(count: natom, type: SIMD3<Float>.self),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return 0.0 }

        encoder.setComputePipelineState(repPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.setBuffer(energyBuf, offset: 0, index: 2)
        encoder.setBuffer(gradBuf, offset: 0, index: 3)
        let (grid, tg) = threadgroupConfig(pipeline: repPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ePtr = energyBuf.contents().bindMemory(to: Float.self, capacity: natom)
        var Erep: Double = 0.0
        for i in 0..<natom { Erep += Double(ePtr[i]) }

        if hasGrad, let grad = gradient {
            let gPtr = gradBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: natom)
            for i in 0..<natom {
                grad[3*i]   = Double(gPtr[i].x)
                grad[3*i+1] = Double(gPtr[i].y)
                grad[3*i+2] = Double(gPtr[i].z)
            }
        }

        return Erep
    }

    // MARK: - Born Radii + SASA

    /// Compute OBC-II Born radii and Gaussian SASA on GPU.
    nonisolated func computeBornRadii(positions: UnsafePointer<Double>, Z: UnsafePointer<Int32>,
                                      natom: Int, probeRadBohr: Float, offsetBohr: Float, bornScale: Float,
                                      bradOut: UnsafeMutablePointer<Double>,
                                      sasaOut: UnsafeMutablePointer<Double>) {
        var bornAtoms = [GFN2BornAtom]()
        bornAtoms.reserveCapacity(natom)
        for i in 0..<natom {
            let p = SIMD3<Float>(Float(positions[3*i]), Float(positions[3*i+1]), Float(positions[3*i+2]))
            bornAtoms.append(GFN2BornAtom(position: p, vdwRadius: Float(getVdwRadBohr(Int(Z[i])))))
        }

        var params = GFN2BornParams(
            atomCount: UInt32(natom), probeRadius: probeRadBohr,
            bornOffset: offsetBohr, bornScale: bornScale)

        guard let atomBuf = makeBuffer(bornAtoms),
              let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<GFN2BornParams>.stride, options: .storageModeShared),
              let bradBuf = makeBuffer(count: natom, type: Float.self),
              let sasaBuf = makeBuffer(count: natom, type: Float.self),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return }

        encoder.setComputePipelineState(bornPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.setBuffer(bradBuf, offset: 0, index: 2)
        encoder.setBuffer(sasaBuf, offset: 0, index: 3)
        let (grid, tg) = threadgroupConfig(pipeline: bornPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let bPtr = bradBuf.contents().bindMemory(to: Float.self, capacity: natom)
        let sPtr = sasaBuf.contents().bindMemory(to: Float.self, capacity: natom)
        for i in 0..<natom {
            bradOut[i] = Double(bPtr[i])
            sasaOut[i] = Double(sPtr[i])
        }
    }

    // MARK: - GB Solvation Energy

    /// Compute Generalized Born + SASA solvation energy on GPU.
    nonisolated func computeGBSolvation(positions: UnsafePointer<Double>,
                                        atomCharges: UnsafePointer<Double>,
                                        natom: Int, brad: UnsafePointer<Double>,
                                        sasa: UnsafePointer<Double>,
                                        keps: Float, gammaAU: Float) -> Double {
        var gbAtoms = [GFN2GBAtom]()
        gbAtoms.reserveCapacity(natom)
        for i in 0..<natom {
            let p = SIMD3<Float>(Float(positions[3*i]), Float(positions[3*i+1]), Float(positions[3*i+2]))
            gbAtoms.append(GFN2GBAtom(
                position: p, charge: -Float(atomCharges[i]),
                bornRadius: Float(brad[i]), sasa: Float(sasa[i]),
                _pad0: 0, _pad1: 0))
        }

        var params = GFN2GBParams(
            atomCount: UInt32(natom), keps: keps, gamma_au: gammaAU, computeGrad: 0)

        guard let atomBuf = makeBuffer(gbAtoms),
              let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<GFN2GBParams>.stride, options: .storageModeShared),
              let energyBuf = makeBuffer(count: natom, type: Float.self),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return 0.0 }

        encoder.setComputePipelineState(gbPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.setBuffer(energyBuf, offset: 0, index: 2)
        let (grid, tg) = threadgroupConfig(pipeline: gbPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ePtr = energyBuf.contents().bindMemory(to: Float.self, capacity: natom)
        var Esolv: Double = 0.0
        for i in 0..<natom { Esolv += Double(ePtr[i]) }
        return Esolv
    }

    // MARK: - CN Gradient

    /// Propagate dE/dCN gradient through CN chain rule on GPU.
    nonisolated func computeCNGradient(positions: UnsafePointer<Double>, Z: UnsafePointer<Int32>,
                                       natom: Int, dEdCN: UnsafePointer<Double>,
                                       gradientOut: UnsafeMutablePointer<Double>) {
        var cnGradAtoms = [GFN2CNGradAtom]()
        cnGradAtoms.reserveCapacity(natom)
        for i in 0..<natom {
            let p = SIMD3<Float>(Float(positions[3*i]), Float(positions[3*i+1]), Float(positions[3*i+2]))
            cnGradAtoms.append(GFN2CNGradAtom(
                position: p, covRadius: Float(getCovRadBohr(Int(Z[i]))),
                dEdCN: Float(dEdCN[i]), _pad0: 0, _pad1: 0, _pad2: 0))
        }

        var params = GFN2CNParams(atomCount: UInt32(natom), _pad0: 0, _pad1: 0, _pad2: 0)

        guard let atomBuf = makeBuffer(cnGradAtoms),
              let paramBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<GFN2CNParams>.stride, options: .storageModeShared),
              let gradBuf = makeBuffer(count: natom, type: SIMD3<Float>.self),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return }

        encoder.setComputePipelineState(cnGradPipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.setBuffer(gradBuf, offset: 0, index: 2)
        let (grid, tg) = threadgroupConfig(pipeline: cnGradPipeline, count: natom)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let gPtr = gradBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: natom)
        for i in 0..<natom {
            gradientOut[3*i]   += Double(gPtr[i].x)
            gradientOut[3*i+1] += Double(gPtr[i].y)
            gradientOut[3*i+2] += Double(gPtr[i].z)
        }
    }

    // MARK: - C-Callable Bridge

    /// Install GPU context into the C++ xTB implementation.
    /// Call this once at app startup after Metal is available.
    func installGPUContext() {
        let ctx = UnsafeMutablePointer<DruseXTBGPUContext>.allocate(capacity: 1)
        ctx.pointee.context = Unmanaged.passUnretained(self as AnyObject).toOpaque()

        ctx.pointee.gpu_compute_cn = { (ctxPtr, pos, Z, natom, cnOut) in
            guard let ctxPtr, let acc = Unmanaged<AnyObject>.fromOpaque(ctxPtr).takeUnretainedValue() as? XTBMetalAccelerator
            else { return }
            acc.computeCN(positions: pos!, Z: Z!, natom: Int(natom), cnOut: cnOut!)
        }

        ctx.pointee.gpu_compute_d4 = { (ctxPtr, pos, Z, natom, cn, gradient) -> Double in
            guard let ctxPtr, let acc = Unmanaged<AnyObject>.fromOpaque(ctxPtr).takeUnretainedValue() as? XTBMetalAccelerator
            else { return 0.0 }
            return acc.computeD4Dispersion(positions: pos!, Z: Z!, natom: Int(natom),
                                            cn: cn!, gradient: gradient)
        }

        ctx.pointee.gpu_compute_repulsion = { (ctxPtr, pos, Z, natom, gradient) -> Double in
            guard let ctxPtr, let acc = Unmanaged<AnyObject>.fromOpaque(ctxPtr).takeUnretainedValue() as? XTBMetalAccelerator
            else { return 0.0 }
            return acc.computeRepulsion(positions: pos!, Z: Z!, natom: Int(natom), gradient: gradient)
        }

        ctx.pointee.gpu_compute_born = { (ctxPtr, pos, Z, natom, probe, offset, scale, brad, sasa) in
            guard let ctxPtr, let acc = Unmanaged<AnyObject>.fromOpaque(ctxPtr).takeUnretainedValue() as? XTBMetalAccelerator
            else { return }
            acc.computeBornRadii(positions: pos!, Z: Z!, natom: Int(natom),
                                 probeRadBohr: probe, offsetBohr: offset, bornScale: scale,
                                 bradOut: brad!, sasaOut: sasa!)
        }

        ctx.pointee.gpu_compute_cn_gradient = { (ctxPtr, pos, Z, natom, dEdCN, gradient) in
            guard let ctxPtr, let acc = Unmanaged<AnyObject>.fromOpaque(ctxPtr).takeUnretainedValue() as? XTBMetalAccelerator
            else { return }
            acc.computeCNGradient(positions: pos!, Z: Z!, natom: Int(natom),
                                  dEdCN: dEdCN!, gradientOut: gradient!)
        }

        druse_xtb_set_gpu_context(ctx)
    }
}

// MARK: - Parameter Lookup Tables (mirroring C++ gfn2Params)

/// Covalent radii in Bohr for CN computation (D3-type).
private func getCovRadBohr(_ Z: Int) -> Double {
    // Pyykko covalent radii (Angstrom → Bohr)
    let covRadAng: [Int: Double] = [
        1: 0.32, 2: 0.46, 3: 1.33, 4: 1.02, 5: 0.85, 6: 0.75, 7: 0.71,
        8: 0.63, 9: 0.64, 10: 0.67, 11: 1.55, 12: 1.39, 13: 1.26, 14: 1.16,
        15: 1.11, 16: 1.03, 17: 0.99, 18: 0.96, 19: 1.96, 20: 1.71,
        25: 1.19, 26: 1.16, 27: 1.11, 28: 1.10, 29: 1.12, 30: 1.18,
        35: 1.14, 53: 1.33
    ]
    let ANG_TO_BOHR = 1.0 / 0.529177210903
    return (covRadAng[Z] ?? 1.2) * ANG_TO_BOHR
}

/// Reference C6 coefficients (Hartree·Bohr⁶) for D4 dispersion.
private func getRefC6(_ Z: Int) -> Double {
    let d4RefC6: [Int: Double] = [
        1: 3.61, 2: 1.46, 3: 1380.0, 4: 214.0, 5: 99.5, 6: 46.6,
        7: 24.2, 8: 15.6, 9: 9.52, 10: 6.38, 11: 1470.0, 12: 626.0,
        13: 528.0, 14: 305.0, 15: 185.0, 16: 134.0, 17: 94.6, 18: 64.3,
        19: 3880.0, 20: 2180.0, 25: 552.0, 26: 482.0, 27: 408.0,
        28: 373.0, 29: 253.0, 30: 284.0, 35: 162.0
    ]
    if let c6 = d4RefC6[Z] { return c6 }
    return 25.0 * pow(Double(Z) / 6.0, 1.5)
}

/// Reference CN for D4 weighting (simplified single-reference).
private func getCNRef(_ Z: Int) -> Double {
    if Z <= 2 { return 1.0 }
    if Z <= 10 { return 3.0 }
    return 4.0
}

/// Van der Waals radii in Bohr (Bondi radii) for Born model.
private func getVdwRadBohr(_ Z: Int) -> Double {
    let vdw: [Int: Double] = [
        1: 2.268, 2: 2.646, 3: 3.438, 4: 2.873, 5: 3.627, 6: 3.213,
        7: 2.929, 8: 2.873, 9: 2.797, 10: 2.910, 11: 4.290, 12: 3.250,
        13: 3.495, 14: 3.968, 15: 3.402, 16: 3.402, 17: 3.307, 18: 3.553,
        19: 5.197, 20: 4.214, 25: 3.78, 26: 3.78, 27: 3.78, 28: 3.08,
        29: 2.65, 30: 2.59, 35: 3.495
    ]
    return vdw[Z] ?? 3.4
}

/// Repulsion parameters (zeff, arep) from GFN2 parameter set.
private func getRepParams(_ Z: Int) -> (zeff: Double, arep: Double) {
    // Extracted from gfn2Params in druse_xtb.cpp
    let repParams: [Int: (Double, Double)] = [
        1:  (1.105388, 2.213717),
        2:  (1.094283, 3.604670),
        3:  (1.289367, 0.475307),
        4:  (4.221216, 0.939696),
        5:  (7.192431, 1.373856),
        6:  (4.231078, 1.247655),
        7:  (5.493710, 1.670786),
        8:  (6.667385, 1.559498),
        9:  (7.670655, 2.363646),
        14: (6.476760, 1.048000),
        15: (5.800540, 1.300000),
        16: (6.271840, 1.450000),
        17: (7.145500, 1.890000),
        35: (8.250000, 1.950000),
        53: (9.100000, 2.100000),
    ]
    if let p = repParams[Z] { return p }
    // Fallback for unsupported elements
    return (Double(Z) * 0.5, 1.5)
}
