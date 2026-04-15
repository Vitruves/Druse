// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Accelerate
import Metal
import simd

// MARK: - QEq Charge Calculator

/// Charge Equilibration method (Rappe & Goddard, J. Phys. Chem., 1991, 95, 3358).
///
/// Uses shielded Coulomb interactions J_ij = 1 / sqrt(r_ij^2 + sigma_ij^2)
/// with per-element parameters (no hybridization dependence).
final class QEqChargeCalculator: ChargeCalculator, @unchecked Sendable {

    private let device: MTLDevice?
    private let pipeline: MTLComputePipelineState?

    // MARK: - Rappe-Goddard Parameters (eV)

    /// (chi: electronegativity, J: idempotential / self-Coulomb) per element.
    static let parameters: [Element: (chi: Float, J: Float)] = [
        .H:  (chi: 4.528,  J: 13.890),
        .C:  (chi: 5.343,  J: 10.126),
        .N:  (chi: 6.899,  J: 11.760),
        .O:  (chi: 8.741,  J: 13.364),
        .F:  (chi: 10.874, J: 14.948),
        .P:  (chi: 5.463,  J: 7.733),
        .S:  (chi: 6.928,  J: 8.972),
        .Cl: (chi: 8.564,  J: 9.892),
        .Br: (chi: 7.790,  J: 8.850),
        .Zn: (chi: 5.106,  J: 10.59),
        .Fe: (chi: 4.670,  J: 8.860),
        .Mg: (chi: 3.951,  J: 7.335),
        .Ca: (chi: 3.231,  J: 5.710),
        .Mn: (chi: 4.330,  J: 8.440),
        // Note: Iodine (I) not in Element enum (only goes to Kr=36). Uses Br fallback.
    ]

    /// Coulomb constant in eV*Angstrom (e^2 / 4*pi*eps0 in eV*A).
    static let coulombConstant: Float = 14.3996

    // MARK: - Init

    init(device: MTLDevice? = nil) {
        self.device = device
        if let dev = device,
           let lib = dev.makeDefaultLibrary(),
           let fn = lib.makeFunction(name: "computeQEqMatrix") {
            self.pipeline = try? dev.makeComputePipelineState(function: fn)
        } else {
            self.pipeline = nil
        }
    }

    // MARK: - ChargeCalculator

    func computeCharges(atoms: [Atom], bonds: [Bond], totalCharge: Int) async throws -> [Float] {
        let n = atoms.count
        guard n >= 2 else { throw ChargeCalculationError.tooFewAtoms }

        // Look up per-atom parameters
        let (chi, J0) = lookupParameters(atoms: atoms)

        // Route to GPU or CPU based on system size
        if n > 256, let device = device, let pipeline = pipeline {
            return try await solveGPU(
                atoms: atoms, chi: chi, J0: J0,
                totalCharge: totalCharge, device: device, pipeline: pipeline
            )
        } else {
            return try solveCPU(atoms: atoms, chi: chi, J0: J0, totalCharge: totalCharge)
        }
    }

    // MARK: - CPU Solve (LAPACK)

    private func solveCPU(
        atoms: [Atom], chi: [Float], J0: [Float], totalCharge: Int
    ) throws -> [Float] {
        let n = atoms.count
        let dim = n + 1

        // Build (N+1)x(N+1) matrix with shielded Coulomb off-diagonals
        var A = [Float](repeating: 0.0, count: dim * dim)
        var b = [Float](repeating: 0.0, count: dim)

        for i in 0..<n {
            // Diagonal: self-Coulomb (idempotential J_ii)
            A[i * dim + i] = J0[i]
            // RHS: electronegativity (positive chi → atom attracts electrons → negative charge)
            b[i] = chi[i]

            // Off-diagonal: shielded Coulomb interaction
            for j in (i + 1)..<n {
                let r2 = simd_distance_squared(atoms[i].position, atoms[j].position)
                // Shielding parameter: sigma_ij = 1/(2*J_i) + 1/(2*J_j)
                let sigma = 0.5 / J0[i] + 0.5 / J0[j]
                let sigma2 = sigma * sigma
                // J_ij = e^2/(4*pi*eps0) / sqrt(r^2 + sigma^2)
                let jij = Self.coulombConstant / sqrt(r2 + sigma2)
                A[i * dim + j] = jij
                A[j * dim + i] = jij
            }

            // Lagrange multiplier row/column for charge conservation
            A[i * dim + n] = -1.0
            A[n * dim + i] = -1.0
        }
        A[n * dim + n] = 0.0
        b[n] = Float(totalCharge)

        // Solve Ax = b using LAPACK sgesv
        var n_lapack = Int32(dim)
        var nrhs = Int32(1)
        var lda = Int32(dim)
        var ipiv = [Int32](repeating: 0, count: dim)
        var ldb = Int32(dim)
        var info = Int32(0)

        sgesv_(&n_lapack, &nrhs, &A, &lda, &ipiv, &b, &ldb, &info)

        guard info == 0 else {
            throw ChargeCalculationError.singularMatrix
        }

        // b[0..<n] = charges, b[n] = equalized electrochemical potential
        return Array(b[0..<n])
    }

    // MARK: - GPU Solve

    private func solveGPU(
        atoms: [Atom], chi: [Float], J0: [Float],
        totalCharge: Int, device: MTLDevice, pipeline: MTLComputePipelineState
    ) async throws -> [Float] {
        let n = atoms.count
        let dim = n + 1

        var positions = atoms.map { SIMD3<Float>($0.position) }

        guard let posBuffer = device.makeBuffer(bytes: &positions, length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let j0Buffer = device.makeBuffer(bytes: J0, length: n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let chiBuffer = device.makeBuffer(bytes: chi, length: n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let matBuffer = device.makeBuffer(length: dim * dim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let rhsBuffer = device.makeBuffer(length: dim * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            return try solveCPU(atoms: atoms, chi: chi, J0: J0, totalCharge: totalCharge)
        }

        var atomCount = UInt32(n)
        var totalChargeF = Float(totalCharge)

        guard let queue = device.makeCommandQueue(),
              let cmdBuffer = queue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder()
        else {
            return try solveCPU(atoms: atoms, chi: chi, J0: J0, totalCharge: totalCharge)
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(posBuffer, offset: 0, index: 0)
        encoder.setBuffer(j0Buffer, offset: 0, index: 1)
        encoder.setBuffer(chiBuffer, offset: 0, index: 2)
        encoder.setBuffer(matBuffer, offset: 0, index: 3)
        encoder.setBuffer(rhsBuffer, offset: 0, index: 4)
        encoder.setBytes(&atomCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&totalChargeF, length: MemoryLayout<Float>.stride, index: 6)

        let gridSize = MTLSize(width: dim, height: dim, depth: 1)
        let threadGroupSize = MTLSize(
            width: min(16, pipeline.maxTotalThreadsPerThreadgroup),
            height: min(16, pipeline.maxTotalThreadsPerThreadgroup / 16),
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        cmdBuffer.commit()
        await cmdBuffer.completed()

        // Read back and solve with LAPACK
        var A = [Float](repeating: 0.0, count: dim * dim)
        var b = [Float](repeating: 0.0, count: dim)
        memcpy(&A, matBuffer.contents(), dim * dim * MemoryLayout<Float>.stride)
        memcpy(&b, rhsBuffer.contents(), dim * MemoryLayout<Float>.stride)

        var n_lapack = Int32(dim)
        var nrhs = Int32(1)
        var lda = Int32(dim)
        var ipiv = [Int32](repeating: 0, count: dim)
        var ldb = Int32(dim)
        var info = Int32(0)

        sgesv_(&n_lapack, &nrhs, &A, &lda, &ipiv, &b, &ldb, &info)

        guard info == 0 else {
            throw ChargeCalculationError.singularMatrix
        }

        return Array(b[0..<n])
    }

    // MARK: - Parameter Lookup

    /// Look up chi and J0 for each atom. Falls back to carbon-like values for unknown elements.
    func lookupParameters(atoms: [Atom]) -> (chi: [Float], J0: [Float]) {
        var chi = [Float](repeating: 0.0, count: atoms.count)
        var J0 = [Float](repeating: 0.0, count: atoms.count)

        for i in 0..<atoms.count {
            if let params = Self.parameters[atoms[i].element] {
                chi[i] = params.chi
                J0[i] = params.J
            } else {
                // Fallback: use rough electronegativity estimates
                let fallback = Self.fallbackParameters(for: atoms[i].element)
                chi[i] = fallback.chi
                J0[i] = fallback.J
            }
        }

        return (chi, J0)
    }

    /// Fallback parameters for elements not in the Rappe-Goddard table.
    private static func fallbackParameters(for element: Element) -> (chi: Float, J: Float) {
        switch element {
        case .Li: return (chi: 3.006, J: 4.772)
        case .Be: return (chi: 4.900, J: 6.840)
        case .B:  return (chi: 4.290, J: 8.298)
        case .Na: return (chi: 2.843, J: 4.592)
        case .Al: return (chi: 3.230, J: 5.383)
        case .Si: return (chi: 4.168, J: 6.768)
        case .K:  return (chi: 2.421, J: 3.840)
        case .Sc: return (chi: 3.395, J: 6.206)
        case .Ti: return (chi: 3.470, J: 6.820)
        case .V:  return (chi: 3.650, J: 6.740)
        case .Cr: return (chi: 3.720, J: 6.766)
        case .Co: return (chi: 4.105, J: 7.860)
        case .Ni: return (chi: 4.015, J: 7.635)
        case .Cu: return (chi: 4.480, J: 7.726)
        case .Ga: return (chi: 3.200, J: 5.670)
        case .Ge: return (chi: 4.600, J: 7.252)
        case .As: return (chi: 5.300, J: 8.298)
        case .Se: return (chi: 5.890, J: 7.680)
        case .Kr: return (chi: 7.600, J: 13.99)
        case .He: return (chi: 12.30, J: 24.59)
        case .Ne: return (chi: 10.80, J: 21.56)
        case .Ar: return (chi: 7.700, J: 15.76)
        default:
            return (chi: 5.0, J: 10.0)
        }
    }
}
