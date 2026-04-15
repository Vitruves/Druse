// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - GFN2-xTB Refinement Configuration

/// Controls how GFN2-xTB post-docking refinement is applied.
struct GFN2RefinementConfig: Sendable {
    /// Enable GFN2-xTB geometry optimization of docked poses.
    var enabled: Bool = false

    /// Solvation model for scoring (gas phase is faster but less accurate).
    var solvation: GFN2SolvationMode = .water

    /// Optimization convergence level.
    var optLevel: GFN2OptLevel = .normal

    /// Maximum optimization steps (0 = auto based on optLevel and atom count).
    var maxSteps: Int32 = 0

    /// Number of top poses to refine (higher = slower but more thorough).
    var topPosesToRefine: Int = 20

    /// Whether to freeze protein atoms during optimization (ligand-only opt).
    var freezeProtein: Bool = true

    /// Weight for blending GFN2 energy into final ranking score.
    /// finalScore = (1 - weight) * vinaScore + weight * gfn2Score_kcal
    var blendWeight: Float = 0.3

    /// If true, update the docked ligand coordinates with optimized geometry.
    var updateCoordinates: Bool = true

    /// Harmonic position restraint strength (Hartree/Å²) to keep ligand near docked pose.
    /// 0.005 allows ~0.3 Å local relaxation per atom while preventing drift.
    /// 0 = no restraints (not recommended for post-dock refinement).
    var restraintStrength: Float = 0.005

    /// Maximum heavy-atom RMSD (Å) allowed between docked and optimized pose.
    /// Poses exceeding this are rejected (original docked coordinates kept).
    var maxRMSD: Float = 2.0
}

/// Solvation mode for GFN2 calculations.
enum GFN2SolvationMode: String, CaseIterable, Sendable {
    case none = "Gas Phase"
    case water = "Water (ALPB)"
    case gbsa = "Water (GBSA)"

    var cConfig: DruseXTBSolvationConfig {
        switch self {
        case .none:  return druse_xtb_solvent_none()
        case .water: return druse_xtb_solvent_water()
        case .gbsa:
            var c = druse_xtb_solvent_water()
            c.model = DRUSE_XTB_SOLV_GBSA
            return c
        }
    }
}

/// Optimization convergence level mapped to C API.
enum GFN2OptLevel: String, CaseIterable, Sendable {
    case crude = "Crude"
    case normal = "Normal"
    case tight = "Tight"

    var cLevel: DruseXTBOptLevel {
        switch self {
        case .crude:  return DRUSE_XTB_OPT_CRUDE
        case .normal: return DRUSE_XTB_OPT_NORMAL
        case .tight:  return DRUSE_XTB_OPT_TIGHT
        }
    }
}

// MARK: - GFN2 Refinement Result

struct GFN2RefinementResult: Sendable {
    let totalEnergy: Float        // Hartree
    let electronicEnergy: Float
    let repulsionEnergy: Float
    let dispersionEnergy: Float
    let solvationEnergy: Float
    let charges: [Float]
    let optimizedPositions: [SIMD3<Float>]?  // nil if not optimized
    let converged: Bool
    let steps: Int
    let gradientNorm: Float

    /// Total GFN2-xTB energy in kcal/mol (from Hartree).
    var totalEnergy_kcal: Float { totalEnergy * 627.509 }
}

// MARK: - GFN2 Refiner

/// Orchestrates GFN2-xTB calculations for post-docking refinement.
///
/// Three modes of operation:
/// 1. **Energy scoring**: Single-point GFN2 energy for re-ranking poses.
/// 2. **Gradient**: Energy + analytical gradients (for external minimizers).
/// 3. **Geometry optimization**: Full L-BFGS relaxation of docked ligand.
enum GFN2Refiner {

    // MARK: - Single-Point Energy

    /// Compute GFN2-xTB total energy (with D4 + solvation) for a molecule.
    static func computeEnergy(
        atoms: [Atom],
        totalCharge: Int = 0,
        solvation: GFN2SolvationMode = .water
    ) async throws -> GFN2RefinementResult {
        let n = atoms.count
        guard n >= 2 else { throw GFN2Error.tooFewAtoms }

        var positions = packPositions(atoms)
        var atomicNumbers = packAtomicNumbers(atoms)
        let solv = solvation.cConfig

        return try await Task.detached(priority: .userInitiated) {
            guard let result = druse_xtb_compute_energy(
                &positions, &atomicNumbers,
                Int32(n), Int32(totalCharge), 50, solv
            ) else {
                throw GFN2Error.calculationFailed("energy computation returned nil")
            }
            defer { druse_xtb_free_energy_result(result) }

            guard result.pointee.success else {
                let msg = extractErrorMessage(result.pointee.errorMessage)
                throw GFN2Error.calculationFailed(msg)
            }

            let charges = Array(UnsafeBufferPointer(
                start: result.pointee.charges, count: n))

            return GFN2RefinementResult(
                totalEnergy: result.pointee.totalEnergy,
                electronicEnergy: result.pointee.electronicEnergy,
                repulsionEnergy: result.pointee.repulsionEnergy,
                dispersionEnergy: result.pointee.dispersionEnergy,
                solvationEnergy: result.pointee.solvationEnergy,
                charges: charges,
                optimizedPositions: nil,
                converged: result.pointee.converged,
                steps: 0,
                gradientNorm: 0
            )
        }.value
    }

    // MARK: - Gradient

    /// Compute GFN2-xTB energy and analytical nuclear gradients.
    static func computeGradient(
        atoms: [Atom],
        totalCharge: Int = 0,
        solvation: GFN2SolvationMode = .water
    ) async throws -> (energy: GFN2RefinementResult, gradient: [SIMD3<Float>]) {
        let n = atoms.count
        guard n >= 2 else { throw GFN2Error.tooFewAtoms }

        var positions = packPositions(atoms)
        var atomicNumbers = packAtomicNumbers(atoms)
        let solv = solvation.cConfig

        return try await Task.detached(priority: .userInitiated) {
            guard let result = druse_xtb_compute_gradient(
                &positions, &atomicNumbers,
                Int32(n), Int32(totalCharge), 50, solv
            ) else {
                throw GFN2Error.calculationFailed("gradient computation returned nil")
            }
            defer { druse_xtb_free_gradient_result(result) }

            guard result.pointee.success else {
                let msg = extractErrorMessage(result.pointee.errorMessage)
                throw GFN2Error.calculationFailed(msg)
            }

            let charges = Array(UnsafeBufferPointer(
                start: result.pointee.charges, count: n))
            let gradFlat = Array(UnsafeBufferPointer(
                start: result.pointee.gradient, count: 3 * n))

            // Convert flat gradient to SIMD3 array
            var grad = [SIMD3<Float>]()
            grad.reserveCapacity(n)
            for i in 0..<n {
                grad.append(SIMD3(gradFlat[3*i], gradFlat[3*i+1], gradFlat[3*i+2]))
            }

            let res = GFN2RefinementResult(
                totalEnergy: result.pointee.totalEnergy,
                electronicEnergy: result.pointee.electronicEnergy,
                repulsionEnergy: result.pointee.repulsionEnergy,
                dispersionEnergy: result.pointee.dispersionEnergy,
                solvationEnergy: result.pointee.solvationEnergy,
                charges: charges,
                optimizedPositions: nil,
                converged: result.pointee.converged,
                steps: 0,
                gradientNorm: result.pointee.gradientNorm
            )
            return (res, grad)
        }.value
    }

    // MARK: - Geometry Optimization

    /// Optimize molecular geometry using GFN2-xTB L-BFGS.
    ///
    /// - Parameters:
    ///   - atoms: Input molecular coordinates.
    ///   - totalCharge: Net molecular charge.
    ///   - solvation: Implicit solvation model.
    ///   - optLevel: Convergence tightness.
    ///   - maxSteps: Max optimization steps (0 = auto).
    ///   - freezeMask: Per-atom freeze flags (true = don't move). nil = optimize all.
    ///   - referencePositions: Optional reference positions for harmonic restraints.
    ///   - restraintStrength: Spring constant in Hartree/Å² (e.g. 0.005). Only used with referencePositions.
    static func optimizeGeometry(
        atoms: [Atom],
        totalCharge: Int = 0,
        solvation: GFN2SolvationMode = .water,
        optLevel: GFN2OptLevel = .normal,
        maxSteps: Int32 = 0,
        freezeMask: [Bool]? = nil,
        referencePositions: [SIMD3<Float>]? = nil,
        restraintStrength: Float = 0.0
    ) async throws -> GFN2RefinementResult {
        let n = atoms.count
        guard n >= 2 else { throw GFN2Error.tooFewAtoms }

        var positions = packPositions(atoms)
        var atomicNumbers = packAtomicNumbers(atoms)
        let solv = solvation.cConfig
        let cLevel = optLevel.cLevel

        // Prepare freeze mask
        var mask: [Bool]?
        if let freezeMask {
            mask = freezeMask
        }

        // Prepare reference positions for restraints
        var refPos: [Float]?
        if let referencePositions, referencePositions.count == n {
            refPos = [Float](repeating: 0, count: n * 3)
            for i in 0..<n {
                refPos![i * 3]     = referencePositions[i].x
                refPos![i * 3 + 1] = referencePositions[i].y
                refPos![i * 3 + 2] = referencePositions[i].z
            }
        }

        return try await Task.detached(priority: .userInitiated) {
            let resultPtr: UnsafeMutablePointer<DruseXTBOptResult>?

            if var freezeArray = mask {
                if var refArray = refPos {
                    resultPtr = druse_xtb_optimize_geometry(
                        &positions, &atomicNumbers,
                        Int32(n), Int32(totalCharge),
                        solv, cLevel, maxSteps, &freezeArray,
                        &refArray, restraintStrength
                    )
                } else {
                    resultPtr = druse_xtb_optimize_geometry(
                        &positions, &atomicNumbers,
                        Int32(n), Int32(totalCharge),
                        solv, cLevel, maxSteps, &freezeArray,
                        nil, 0.0
                    )
                }
            } else {
                if var refArray = refPos {
                    resultPtr = druse_xtb_optimize_geometry(
                        &positions, &atomicNumbers,
                        Int32(n), Int32(totalCharge),
                        solv, cLevel, maxSteps, nil,
                        &refArray, restraintStrength
                    )
                } else {
                    resultPtr = druse_xtb_optimize_geometry(
                        &positions, &atomicNumbers,
                        Int32(n), Int32(totalCharge),
                        solv, cLevel, maxSteps, nil,
                        nil, 0.0
                    )
                }
            }

            guard let result = resultPtr else {
                throw GFN2Error.calculationFailed("optimization returned nil")
            }
            defer { druse_xtb_free_opt_result(result) }

            guard result.pointee.success else {
                let msg = extractErrorMessage(result.pointee.errorMessage)
                throw GFN2Error.calculationFailed(msg)
            }

            let charges = result.pointee.charges != nil
                ? Array(UnsafeBufferPointer(start: result.pointee.charges, count: n))
                : [Float](repeating: 0, count: n)

            // Unpack optimized positions
            var optPositions: [SIMD3<Float>]?
            if let optPtr = result.pointee.optimizedPositions {
                let flat = Array(UnsafeBufferPointer(start: optPtr, count: 3 * n))
                var pos = [SIMD3<Float>]()
                pos.reserveCapacity(n)
                for i in 0..<n {
                    pos.append(SIMD3(flat[3*i], flat[3*i+1], flat[3*i+2]))
                }
                optPositions = pos
            }

            return GFN2RefinementResult(
                totalEnergy: result.pointee.totalEnergy,
                electronicEnergy: result.pointee.electronicEnergy,
                repulsionEnergy: result.pointee.repulsionEnergy,
                dispersionEnergy: result.pointee.dispersionEnergy,
                solvationEnergy: result.pointee.solvationEnergy,
                charges: charges,
                optimizedPositions: optPositions,
                converged: result.pointee.converged,
                steps: Int(result.pointee.optimizationSteps),
                gradientNorm: result.pointee.finalGradientNorm
            )
        }.value
    }

    // MARK: - Helpers

    private static func packPositions(_ atoms: [Atom]) -> [Float] {
        var positions = [Float](repeating: 0, count: atoms.count * 3)
        for i in 0..<atoms.count {
            positions[i * 3]     = atoms[i].position.x
            positions[i * 3 + 1] = atoms[i].position.y
            positions[i * 3 + 2] = atoms[i].position.z
        }
        return positions
    }

    private static func packAtomicNumbers(_ atoms: [Atom]) -> [Int32] {
        atoms.map { Int32($0.element.rawValue) }
    }

    private static func extractErrorMessage(
        _ buf: (CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
                CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar)
    ) -> String {
        withUnsafePointer(to: buf) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 512) {
                String(cString: $0)
            }
        }
    }
}

// MARK: - Errors

enum GFN2Error: Error, LocalizedError {
    case tooFewAtoms
    case calculationFailed(String)
    case unsupportedElement(Int)

    var errorDescription: String? {
        switch self {
        case .tooFewAtoms:
            return "GFN2-xTB requires at least 2 atoms"
        case .calculationFailed(let msg):
            return "GFN2-xTB calculation failed: \(msg)"
        case .unsupportedElement(let z):
            return "GFN2-xTB does not support element Z=\(z)"
        }
    }
}
