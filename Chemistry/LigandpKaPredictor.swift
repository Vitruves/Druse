import Foundation
import simd

/// GFN2-xTB based pKa prediction for ligand ionizable sites.
///
/// For each ionizable site detected by SMARTS patterns:
/// 1. Generate protonated and deprotonated 3D structures (via RDKit C++)
/// 2. Compute GFN2-xTB single-point energies with GBSA water solvation (Metal GPU)
/// 3. Derive aqueous pKa from the free energy difference:
///    ΔG = E(deprotonated) - E(protonated)  [in kcal/mol]
///    pKa = ΔG / (2.303 × RT)              [RT = 0.592 kcal/mol at 298.15K]
///
/// This replaces the hardcoded pKa lookup table with molecule-specific predictions
/// that account for inductive effects, resonance, and solvation.
enum LigandpKaPredictor {

    /// Result of pKa prediction for a single ionizable site.
    struct SitePKa: Sendable {
        let atomIdx: Int
        let groupName: String
        let isAcid: Bool
        let defaultPKa: Double       // from the lookup table
        let computedPKa: Double      // from GFN2-xTB
        let deltaG_kcal: Double      // ΔG = E(deprot) - E(prot) in kcal/mol
        let converged: Bool          // true if both GFN2 calculations converged
    }

    /// Predict pKa for all ionizable sites in a molecule using GFN2-xTB.
    ///
    /// - Parameters:
    ///   - smiles: Input SMILES string
    ///   - temperature: Temperature in Kelvin (default 298.15)
    /// - Returns: Array of per-site pKa predictions. Empty if no ionizable sites found.
    static func predictpKa(
        smiles: String,
        temperature: Double = 298.15
    ) async -> [SitePKa] {
        // Step 1: Detect ionizable sites
        let sites = RDKitBridge.detectIonizableSites(smiles: smiles)
        guard !sites.isEmpty else { return [] }

        let RT = 0.001987204 * temperature  // kB * T in kcal/mol

        // Step 2: For each site, compute GFN2 pKa
        var predictions: [SitePKa] = []

        for site in sites {
            // Generate protonated + deprotonated 3D structures
            guard let pair = RDKitBridge.generateSiteProtomers(
                smiles: smiles, atomIdx: site.atomIdx, isAcid: site.isAcid
            ) else {
                // Fallback to default pKa if structure generation fails
                predictions.append(SitePKa(
                    atomIdx: site.atomIdx, groupName: site.groupName,
                    isAcid: site.isAcid, defaultPKa: site.defaultPKa,
                    computedPKa: site.defaultPKa, deltaG_kcal: 0,
                    converged: false
                ))
                continue
            }

            // Run GFN2-xTB single-point on both forms (with water solvation)
            do {
                let eProt = try await GFN2Refiner.computeEnergy(
                    atoms: pair.protonatedAtoms,
                    totalCharge: pair.protonatedCharge,
                    solvation: .water
                )
                let eDeprot = try await GFN2Refiner.computeEnergy(
                    atoms: pair.deprotonatedAtoms,
                    totalCharge: pair.deprotonatedCharge,
                    solvation: .water
                )

                guard eProt.converged, eDeprot.converged else {
                    // SCC didn't converge — fall back to default
                    predictions.append(SitePKa(
                        atomIdx: site.atomIdx, groupName: site.groupName,
                        isAcid: site.isAcid, defaultPKa: site.defaultPKa,
                        computedPKa: site.defaultPKa, deltaG_kcal: 0,
                        converged: false
                    ))
                    continue
                }

                // ΔG = E(deprotonated) - E(protonated) in kcal/mol
                // For acids: positive ΔG → harder to deprotonate → higher pKa
                // For bases: compute proton affinity (PA = E(neutral) - E(protonated))
                let deltaG: Double
                if site.isAcid {
                    // Acid: A-H → A⁻ + H⁺
                    // ΔG = E(A⁻) - E(A-H) > 0 means deprotonation is unfavorable
                    deltaG = Double(eDeprot.totalEnergy_kcal - eProt.totalEnergy_kcal)
                } else {
                    // Base: B + H⁺ → B-H⁺
                    // ΔG = E(B) - E(B-H⁺) > 0 means protonation is favorable → high pKa
                    deltaG = Double(eProt.totalEnergy_kcal - eDeprot.totalEnergy_kcal)
                }

                // pKa = ΔG / (2.303 × RT)
                // Note: This is a gas-phase + GBSA solvation calculation.
                // Empirical correction: shift to match aqueous reference.
                // The GBSA solvation partially accounts for aqueous effects,
                // but a linear correction improves accuracy:
                //   pKa_aq ≈ a × pKa_raw + b
                // Using established GFN2-xTB/GBSA calibration (a≈0.73, b≈1.8)
                let pKaRaw = deltaG / (2.303 * RT)
                let pKaCalibrated = 0.73 * pKaRaw + 1.8

                predictions.append(SitePKa(
                    atomIdx: site.atomIdx, groupName: site.groupName,
                    isAcid: site.isAcid, defaultPKa: site.defaultPKa,
                    computedPKa: pKaCalibrated, deltaG_kcal: deltaG,
                    converged: true
                ))

            } catch {
                // GFN2 computation failed — fall back to default
                predictions.append(SitePKa(
                    atomIdx: site.atomIdx, groupName: site.groupName,
                    isAcid: site.isAcid, defaultPKa: site.defaultPKa,
                    computedPKa: site.defaultPKa, deltaG_kcal: 0,
                    converged: false
                ))
            }
        }

        return predictions
    }

    /// Extract just the computed pKa values as an array (for passing to ensemble generation).
    static func pKaArray(from predictions: [SitePKa]) -> [Double] {
        predictions.map(\.computedPKa)
    }
}
