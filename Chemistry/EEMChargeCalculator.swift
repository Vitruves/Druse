// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Accelerate
import Metal
import simd

// MARK: - EEM Atom Type

/// Atom types for EEM parameter lookup, based on element + hybridization.
enum EEMAtomType: Hashable, Sendable {
    // Carbon
    case C_sp3, C_sp2, C_ar, C_sp
    // Hydrogen
    case H
    // Nitrogen
    case N_sp3, N_sp2, N_ar, N_sp
    // Oxygen
    case O_sp3, O_sp2, O_ar
    // Halogens
    case F, Cl, Br
    // Phosphorus
    case P_sp3
    // Sulfur
    case S_sp3, S_sp2, S_ar
    // Metals
    case Zn, Fe, Mg, Ca, Mn
    // Fallback
    case unknown(Element)
}

// MARK: - EEM Charge Calculator

/// Electronegativity Equalization Method (Bultinck et al., J. Phys. Chem. A, 2002, 106, 7887).
///
/// Solves an (N+1)x(N+1) linear system to determine partial charges that equalize
/// electronegativity across all atoms subject to a total charge constraint.
final class EEMChargeCalculator: ChargeCalculator, @unchecked Sendable {

    private let device: MTLDevice?
    private let pipeline: MTLComputePipelineState?

    // MARK: - Bultinck STO-3G Parameters

    /// (electronegativity chi, hardness eta) for each EEM atom type.
    static let parameters: [EEMAtomType: (chi: Float, eta: Float)] = [
        // Carbon types
        .C_sp3:  (chi: 6.0413, eta: 9.9426),
        .C_sp2:  (chi: 7.2810, eta: 9.8890),
        .C_ar:   (chi: 7.1703, eta: 9.5862),
        .C_sp:   (chi: 7.7352, eta: 10.196),
        // Hydrogen
        .H:      (chi: 4.5280, eta: 13.8904),
        // Nitrogen types
        .N_sp3:  (chi: 11.543, eta: 12.008),
        .N_sp2:  (chi: 11.760, eta: 13.048),
        .N_ar:   (chi: 12.154, eta: 11.696),
        .N_sp:   (chi: 12.178, eta: 14.614),
        // Oxygen types
        .O_sp3:  (chi: 14.243, eta: 12.594),
        .O_sp2:  (chi: 15.075, eta: 12.251),
        .O_ar:   (chi: 13.892, eta: 13.120),
        // Fluorine
        .F:      (chi: 13.364, eta: 14.964),
        // Phosphorus
        .P_sp3:  (chi: 6.5427, eta: 7.7229),
        // Sulfur
        .S_sp3:  (chi: 8.0728, eta: 7.9910),
        .S_sp2:  (chi: 8.5458, eta: 8.8672),
        .S_ar:   (chi: 8.3978, eta: 8.1450),
        // Chlorine
        .Cl:     (chi: 11.308, eta: 9.8620),
        // Bromine
        .Br:     (chi: 10.175, eta: 8.4122),
        // Note: Iodine (I, Z=53) not in Element enum (max Kr=36). Uses Br fallback.
        // Metals (fallback values from extended EEM)
        .Zn:     (chi: 5.106,  eta: 10.59),
        .Fe:     (chi: 4.670,  eta: 8.860),
        .Mg:     (chi: 3.951,  eta: 7.335),
        .Ca:     (chi: 3.231,  eta: 5.710),
        .Mn:     (chi: 4.330,  eta: 8.440),
    ]

    /// Coulomb scaling factor (Bohr to Angstrom conversion).
    static let kappa: Float = 0.529177

    // MARK: - Init

    init(device: MTLDevice? = nil) {
        self.device = device
        // Try to load Metal kernel for GPU-accelerated matrix construction.
        if let dev = device,
           let lib = dev.makeDefaultLibrary(),
           let fn = lib.makeFunction(name: "computeEEMMatrix") {
            self.pipeline = try? dev.makeComputePipelineState(function: fn)
        } else {
            self.pipeline = nil
        }
    }

    // MARK: - ChargeCalculator

    func computeCharges(atoms: [Atom], bonds: [Bond], totalCharge: Int) async throws -> [Float] {
        let n = atoms.count
        guard n >= 2 else { throw ChargeCalculationError.tooFewAtoms }

        // 1. Classify atom types based on element + hybridization from bond orders
        let types = classifyAtomTypes(atoms: atoms, bonds: bonds)

        // 2. Look up parameters
        let (chi, eta) = lookupParameters(types: types, atoms: atoms)

        // 3. Build and solve the linear system
        if n > 256, let device = device, let pipeline = pipeline {
            return try await solveGPU(
                atoms: atoms, chi: chi, eta: eta,
                totalCharge: totalCharge, device: device, pipeline: pipeline
            )
        } else {
            return try solveCPU(atoms: atoms, chi: chi, eta: eta, totalCharge: totalCharge)
        }
    }

    // MARK: - CPU Solve (LAPACK)

    private func solveCPU(
        atoms: [Atom], chi: [Float], eta: [Float], totalCharge: Int
    ) throws -> [Float] {
        let n = atoms.count
        let dim = n + 1

        // Build (N+1)x(N+1) matrix A in column-major order (LAPACK convention)
        var A = [Float](repeating: 0.0, count: dim * dim)
        var b = [Float](repeating: 0.0, count: dim)

        for i in 0..<n {
            // Diagonal: hardness
            A[i * dim + i] = eta[i]
            // RHS: -electronegativity
            b[i] = -chi[i]

            // Off-diagonal: Coulomb interactions
            for j in (i + 1)..<n {
                let r = simd_distance(atoms[i].position, atoms[j].position)
                let jij = Self.kappa / max(r, 0.1)
                A[i * dim + j] = jij
                A[j * dim + i] = jij
            }

            // Lagrange multiplier column/row for charge constraint
            A[i * dim + n] = -1.0
            A[n * dim + i] = -1.0
        }
        A[n * dim + n] = 0.0
        b[n] = -Float(totalCharge)

        // Solve Ax = b using LAPACK sgesv (general linear solve with partial pivoting)
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

        // b[0..<n] contains charges, b[n] is the equalized chemical potential (mu)
        return Array(b[0..<n])
    }

    // MARK: - GPU Solve

    private func solveGPU(
        atoms: [Atom], chi: [Float], eta: [Float],
        totalCharge: Int, device: MTLDevice, pipeline: MTLComputePipelineState
    ) async throws -> [Float] {
        let n = atoms.count
        let dim = n + 1

        // Prepare position data
        var positions = atoms.map { SIMD3<Float>($0.position) }

        // Create Metal buffers
        guard let posBuffer = device.makeBuffer(bytes: &positions, length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let etaBuffer = device.makeBuffer(bytes: eta, length: n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let chiBuffer = device.makeBuffer(bytes: chi, length: n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let matBuffer = device.makeBuffer(length: dim * dim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let rhsBuffer = device.makeBuffer(length: dim * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            // Fall back to CPU if buffer allocation fails
            return try solveCPU(atoms: atoms, chi: chi, eta: eta, totalCharge: totalCharge)
        }

        var atomCount = UInt32(n)
        var kappaVal = Self.kappa
        var totalChargeF = Float(totalCharge)

        guard let queue = device.makeCommandQueue(),
              let cmdBuffer = queue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder()
        else {
            return try solveCPU(atoms: atoms, chi: chi, eta: eta, totalCharge: totalCharge)
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(posBuffer, offset: 0, index: 0)
        encoder.setBuffer(etaBuffer, offset: 0, index: 1)
        encoder.setBuffer(chiBuffer, offset: 0, index: 2)
        encoder.setBuffer(matBuffer, offset: 0, index: 3)
        encoder.setBuffer(rhsBuffer, offset: 0, index: 4)
        encoder.setBytes(&atomCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&kappaVal, length: MemoryLayout<Float>.stride, index: 6)
        encoder.setBytes(&totalChargeF, length: MemoryLayout<Float>.stride, index: 7)

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

        // Copy matrix and RHS back for LAPACK solve
        var A = [Float](repeating: 0.0, count: dim * dim)
        var b = [Float](repeating: 0.0, count: dim)
        memcpy(&A, matBuffer.contents(), dim * dim * MemoryLayout<Float>.stride)
        memcpy(&b, rhsBuffer.contents(), dim * MemoryLayout<Float>.stride)

        // Solve with LAPACK
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

    // MARK: - Atom Type Classification

    /// Classify each atom into an EEM atom type based on element and hybridization
    /// inferred from bond orders.
    func classifyAtomTypes(atoms: [Atom], bonds: [Bond]) -> [EEMAtomType] {
        let n = atoms.count

        // Build per-atom bond order summary
        var maxBondOrder = [Int](repeating: 0, count: n)
        var totalBondOrder = [Int](repeating: 0, count: n)
        var hasAromatic = [Bool](repeating: false, count: n)
        var neighborCount = [Int](repeating: 0, count: n)

        for bond in bonds {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard a < n, b < n else { continue }

            let order = bond.order.rawValue
            totalBondOrder[a] += order
            totalBondOrder[b] += order
            maxBondOrder[a] = max(maxBondOrder[a], order)
            maxBondOrder[b] = max(maxBondOrder[b], order)
            neighborCount[a] += 1
            neighborCount[b] += 1

            if bond.order == .aromatic {
                hasAromatic[a] = true
                hasAromatic[b] = true
            }
        }

        return atoms.enumerated().map { (i, atom) in
            classifySingleAtom(
                element: atom.element,
                maxBondOrder: maxBondOrder[i],
                totalBondOrder: totalBondOrder[i],
                hasAromatic: hasAromatic[i],
                neighborCount: neighborCount[i]
            )
        }
    }

    /// Classify a single atom based on element and bonding environment.
    private func classifySingleAtom(
        element: Element,
        maxBondOrder: Int,
        totalBondOrder: Int,
        hasAromatic: Bool,
        neighborCount: Int
    ) -> EEMAtomType {
        switch element {
        case .C:
            if hasAromatic { return .C_ar }
            if maxBondOrder >= 3 { return .C_sp }
            if maxBondOrder == 2 { return .C_sp2 }
            return .C_sp3

        case .H:
            return .H

        case .N:
            if hasAromatic { return .N_ar }
            if maxBondOrder >= 3 { return .N_sp }
            if maxBondOrder == 2 { return .N_sp2 }
            // Planar sp2 nitrogen (amide-like): 3 bonds, total order > 3
            if neighborCount == 3 && totalBondOrder > 3 { return .N_sp2 }
            return .N_sp3

        case .O:
            if hasAromatic { return .O_ar }
            if maxBondOrder >= 2 { return .O_sp2 }
            return .O_sp3

        case .F:  return .F
        case .Cl: return .Cl
        case .Br: return .Br

        case .P:  return .P_sp3

        case .S:
            if hasAromatic { return .S_ar }
            if maxBondOrder >= 2 { return .S_sp2 }
            return .S_sp3

        case .Zn: return .Zn
        case .Fe: return .Fe
        case .Mg: return .Mg
        case .Ca: return .Ca
        case .Mn: return .Mn

        default:
            return .unknown(element)
        }
    }

    // MARK: - Parameter Lookup

    /// Look up chi (electronegativity) and eta (hardness) for each atom.
    /// Falls back to Sanderson-type estimates for unknown types.
    func lookupParameters(types: [EEMAtomType], atoms: [Atom]) -> (chi: [Float], eta: [Float]) {
        var chi = [Float](repeating: 0.0, count: types.count)
        var eta = [Float](repeating: 0.0, count: types.count)

        for i in 0..<types.count {
            if let params = Self.parameters[types[i]] {
                chi[i] = params.chi
                eta[i] = params.eta
            } else {
                // Fallback: use Mulliken-type estimate from ionization potential / electron affinity
                let fallback = Self.fallbackParameters(for: atoms[i].element)
                chi[i] = fallback.chi
                eta[i] = fallback.eta
            }
        }

        return (chi, eta)
    }

    /// Provide rough fallback parameters for elements not in the Bultinck table.
    /// Uses Mulliken electronegativity = (IP + EA) / 2 and hardness = IP - EA.
    private static func fallbackParameters(for element: Element) -> (chi: Float, eta: Float) {
        // Generic fallbacks based on electronegativity trends
        switch element {
        case .Li: return (chi: 3.006, eta: 4.772)
        case .Be: return (chi: 4.900, eta: 6.840)
        case .B:  return (chi: 4.290, eta: 8.298)
        case .Na: return (chi: 2.843, eta: 4.592)
        case .Al: return (chi: 3.230, eta: 5.383)
        case .Si: return (chi: 4.168, eta: 6.768)
        case .K:  return (chi: 2.421, eta: 3.840)
        case .Sc: return (chi: 3.395, eta: 6.206)
        case .Ti: return (chi: 3.470, eta: 6.820)
        case .V:  return (chi: 3.650, eta: 6.740)
        case .Cr: return (chi: 3.720, eta: 6.766)
        case .Co: return (chi: 4.105, eta: 7.860)
        case .Ni: return (chi: 4.015, eta: 7.635)
        case .Cu: return (chi: 4.480, eta: 7.726)
        case .Ga: return (chi: 3.200, eta: 5.670)
        case .Ge: return (chi: 4.600, eta: 7.252)
        case .As: return (chi: 5.300, eta: 8.298)
        case .Se: return (chi: 5.890, eta: 7.680)
        case .Kr: return (chi: 7.600, eta: 13.99)
        case .He: return (chi: 12.30, eta: 24.59)
        case .Ne: return (chi: 10.80, eta: 21.56)
        case .Ar: return (chi: 7.700, eta: 15.76)
        default:
            // Very rough fallback: use Allred-Rochow scale approximation
            return (chi: 5.0, eta: 10.0)
        }
    }
}
