import Foundation
import Metal
import simd

// MARK: - Charge Calculator Protocol

/// Protocol for computing partial atomic charges on a molecular system.
protocol ChargeCalculator: Sendable {
    /// Compute partial atomic charges. Returns one charge per atom.
    func computeCharges(
        atoms: [Atom],
        bonds: [Bond],
        totalCharge: Int
    ) async throws -> [Float]
}

// MARK: - Errors

enum ChargeCalculationError: Error, LocalizedError {
    case tooFewAtoms
    case singularMatrix
    case convergenceFailed(iterations: Int)
    case methodUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .tooFewAtoms:
            return "Too few atoms for charge calculation"
        case .singularMatrix:
            return "Singular matrix in charge solver"
        case .convergenceFailed(let n):
            return "Charge calculation did not converge after \(n) iterations"
        case .methodUnavailable(let m):
            return "\(m) is not available"
        }
    }
}

// MARK: - Factory

/// Factory for creating charge calculators.
@MainActor
enum ChargeCalculatorFactory {
    static func calculator(for method: ChargeMethod, device: MTLDevice? = nil) -> ChargeCalculator {
        switch method {
        case .gasteiger: return GasteigerChargeCalculator()
        case .eem:       return EEMChargeCalculator(device: device)
        case .qeq:       return QEqChargeCalculator(device: device)
        case .xtb:       return XTBChargeCalculator()
        }
    }
}

// MARK: - Gasteiger (RDKit wrapper)

/// Wrapper around existing RDKit Gasteiger implementation.
/// Atoms are assumed to already carry Gasteiger charges from RDKitBridge.
struct GasteigerChargeCalculator: ChargeCalculator {
    func computeCharges(atoms: [Atom], bonds: [Bond], totalCharge: Int) async throws -> [Float] {
        guard atoms.count >= 2 else { throw ChargeCalculationError.tooFewAtoms }
        // Gasteiger charges are pre-computed by the C++ core via RDKit.
        // Return the charges already stored on the atoms.
        return atoms.map { $0.charge }
    }
}

// MARK: - xTB (GFN2-xTB via C bridge)

/// GFN2-xTB semi-empirical charge calculator.
/// Calls through to the C API in druse_xtb.h for Mulliken partial charges.
struct XTBChargeCalculator: ChargeCalculator {
    func computeCharges(atoms: [Atom], bonds: [Bond], totalCharge: Int) async throws -> [Float] {
        guard druse_xtb_available() else {
            throw ChargeCalculationError.methodUnavailable("GFN2-xTB")
        }

        let n = atoms.count
        guard n >= 2 else { throw ChargeCalculationError.tooFewAtoms }

        // Prepare position array (x0,y0,z0,x1,y1,z1,...)
        var positions = [Float](repeating: 0, count: n * 3)
        var atomicNumbers = [Int32](repeating: 0, count: n)
        for i in 0..<n {
            positions[i * 3]     = atoms[i].position.x
            positions[i * 3 + 1] = atoms[i].position.y
            positions[i * 3 + 2] = atoms[i].position.z
            atomicNumbers[i] = Int32(atoms[i].element.rawValue)
        }

        // Run on background thread (xTB can take ~100ms for large molecules)
        return try await Task.detached(priority: .userInitiated) {
            guard let result = druse_xtb_compute_charges(
                &positions, &atomicNumbers, Int32(n), Int32(totalCharge), 50
            ) else {
                throw ChargeCalculationError.convergenceFailed(iterations: 50)
            }
            defer { druse_xtb_free_result(result) }

            guard result.pointee.success else {
                let msg = withUnsafePointer(to: result.pointee.errorMessage) { ptr in
                    ptr.withMemoryRebound(to: CChar.self, capacity: 512) { String(cString: $0) }
                }
                throw ChargeCalculationError.methodUnavailable(msg)
            }

            let charges = Array(UnsafeBufferPointer(start: result.pointee.charges, count: n))
            return charges
        }.value
    }
}
