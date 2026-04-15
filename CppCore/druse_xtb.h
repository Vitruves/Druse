// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// druse_xtb.h — GFN2-xTB semi-empirical quantum chemistry (C API)
//
// Self-contained implementation of the GFN2-xTB tight-binding method:
//   • Mulliken partial charges via SCC iteration
//   • Analytical nuclear gradients for all energy terms
//   • D4 dispersion correction with BJ rational damping
//   • GBSA/ALPB implicit solvation (Born + SASA)
//   • L-BFGS geometry optimization
//
// No Fortran dependencies — uses Apple Accelerate LAPACK/BLAS.
//
// References:
//   Bannwarth, Ehlert, Grimme, JCTC 2019, 15, 1652-1671  (GFN2-xTB)
//   Caldeweyher et al., JCP 2019, 150, 154122             (D4 dispersion)
//   Ehlert et al., JCTC 2021, 17, 4250-4261               (ALPB solvation)
// ============================================================================

#ifndef DRUSE_XTB_H
#define DRUSE_XTB_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - Solvation Model Configuration
// ============================================================================

/// Solvation model selector.
typedef enum {
    DRUSE_XTB_SOLV_NONE  = 0,  // gas phase (no solvation)
    DRUSE_XTB_SOLV_GBSA  = 1,  // Generalized Born + SASA (Still kernel)
    DRUSE_XTB_SOLV_ALPB  = 2,  // Analytical Linearized Poisson-Boltzmann
} DruseXTBSolvationModel;

/// Solvent descriptor for implicit solvation.
typedef struct {
    DruseXTBSolvationModel model;
    float dielectricConstant;     // ε (80.2 for water, 36.6 for DMF, etc.)
    float probeRadius;            // solvent probe radius in Angstrom (1.4 for water)
    float surfaceTension;         // γ for SASA term (dyn/cm, ~3.6 for water)
    float bornOffset;             // offset for Born radii (0.09 Å typical)
    float bornScale;              // OBC descreening parameter (0.8 typical)
} DruseXTBSolvationConfig;

/// Convenience: water solvation preset (ALPB, ε=80.2).
static inline DruseXTBSolvationConfig druse_xtb_solvent_water(void) {
    DruseXTBSolvationConfig c;
    c.model = DRUSE_XTB_SOLV_ALPB;
    c.dielectricConstant = 80.2f;
    c.probeRadius = 1.4f;
    c.surfaceTension = 3.6f;
    c.bornOffset = 0.09f;
    c.bornScale = 0.8f;
    return c;
}

/// Convenience: no solvation (gas phase).
static inline DruseXTBSolvationConfig druse_xtb_solvent_none(void) {
    DruseXTBSolvationConfig c;
    c.model = DRUSE_XTB_SOLV_NONE;
    c.dielectricConstant = 1.0f;
    c.probeRadius = 0.0f;
    c.surfaceTension = 0.0f;
    c.bornOffset = 0.0f;
    c.bornScale = 0.0f;
    return c;
}

// ============================================================================
// MARK: - Optimization Convergence Level
// ============================================================================

typedef enum {
    DRUSE_XTB_OPT_CRUDE    = 0,  // ethr=5e-4, gthr=1e-2
    DRUSE_XTB_OPT_NORMAL   = 1,  // ethr=5e-6, gthr=1e-3
    DRUSE_XTB_OPT_TIGHT    = 2,  // ethr=1e-6, gthr=8e-4
    DRUSE_XTB_OPT_EXTREME  = 3,  // ethr=5e-8, gthr=5e-5
} DruseXTBOptLevel;

// ============================================================================
// MARK: - GFN2-xTB Charge Result (original API, preserved)
// ============================================================================

typedef struct {
    float *charges;           // Mulliken partial charges per atom (elementary charge)
    float totalEnergy;        // total GFN2-xTB energy (Hartree)
    float electronicEnergy;   // electronic energy (Hartree)
    float repulsionEnergy;    // repulsion energy (Hartree)
    int32_t atomCount;
    int32_t scfIterations;    // how many SCC iterations were needed
    bool converged;           // true if SCC converged within maxIterations
    bool success;             // true if calculation completed without errors
    char errorMessage[512];
} DruseXTBChargeResult;

// ============================================================================
// MARK: - Full Energy Result (with D4 + solvation decomposition)
// ============================================================================

typedef struct {
    // Energy decomposition (all in Hartree)
    float totalEnergy;        // sum of all terms below
    float electronicEnergy;   // band structure + Coulomb (2nd + 3rd order)
    float repulsionEnergy;    // nuclear repulsion
    float dispersionEnergy;   // D4 dispersion correction
    float solvationEnergy;    // GBSA/ALPB contribution (0 if gas phase)

    // Mulliken charges
    float *charges;           // Mulliken partial charges per atom
    int32_t atomCount;

    // SCC info
    int32_t scfIterations;
    bool converged;
    bool success;
    char errorMessage[512];
} DruseXTBEnergyResult;

// ============================================================================
// MARK: - Gradient Result (energy + analytical nuclear gradients)
// ============================================================================

typedef struct {
    // Energy (same decomposition as DruseXTBEnergyResult)
    float totalEnergy;
    float electronicEnergy;
    float repulsionEnergy;
    float dispersionEnergy;
    float solvationEnergy;

    // Gradient: dE/dR per atom, Nx3 array in Hartree/Bohr (row-major)
    // Negate to get force: F = -gradient
    float *gradient;          // Nx3 in Hartree/Bohr

    // Charges
    float *charges;
    int32_t atomCount;

    // Gradient norm (Hartree/Bohr)
    float gradientNorm;       // sqrt(sum(g_i^2)) / sqrt(N)

    int32_t scfIterations;
    bool converged;
    bool success;
    char errorMessage[512];
} DruseXTBGradientResult;

// ============================================================================
// MARK: - Geometry Optimization Result
// ============================================================================

typedef struct {
    // Optimized positions in Angstrom (Nx3, row-major)
    float *optimizedPositions;

    // Final energy decomposition (Hartree)
    float totalEnergy;
    float electronicEnergy;
    float repulsionEnergy;
    float dispersionEnergy;
    float solvationEnergy;

    // Final charges
    float *charges;
    int32_t atomCount;

    // Optimization stats
    int32_t optimizationSteps;
    float finalGradientNorm;  // Hartree/Bohr, RMS per atom
    float energyChange;       // last step ΔE (Hartree)

    bool converged;           // optimization converged
    bool success;
    char errorMessage[512];
} DruseXTBOptResult;

// ============================================================================
// MARK: - Original API (preserved for backward compatibility)
// ============================================================================

/// Compute GFN2-xTB Mulliken charges for a molecule.
///
/// @param positions     Nx3 array in Angstrom (row-major: x0,y0,z0,x1,y1,z1,...)
/// @param atomicNumbers N atomic numbers (1=H, 6=C, 7=N, 8=O, ...)
/// @param atomCount     Number of atoms
/// @param totalCharge   Net molecular charge (0 for neutral)
/// @param maxIterations Max SCC iterations (50 typical)
/// @return Heap-allocated result — caller must free with druse_xtb_free_result()
DruseXTBChargeResult* druse_xtb_compute_charges(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    int32_t maxIterations
);

/// Free a result from druse_xtb_compute_charges.
void druse_xtb_free_result(DruseXTBChargeResult *result);

/// Check if xTB is available (always true for compiled-in implementation).
bool druse_xtb_available(void);

// ============================================================================
// MARK: - Full Energy API (with D4 dispersion + solvation)
// ============================================================================

/// Compute GFN2-xTB total energy with all corrections.
///
/// @param positions     Nx3 in Angstrom
/// @param atomicNumbers N atomic numbers
/// @param atomCount     Number of atoms
/// @param totalCharge   Net molecular charge
/// @param maxIterations SCC iteration limit
/// @param solvation     Solvation config (use druse_xtb_solvent_none() for gas phase)
/// @return Heap-allocated — free with druse_xtb_free_energy_result()
DruseXTBEnergyResult* druse_xtb_compute_energy(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    int32_t maxIterations,
    DruseXTBSolvationConfig solvation
);

void druse_xtb_free_energy_result(DruseXTBEnergyResult *result);

// ============================================================================
// MARK: - Gradient API (analytical nuclear gradients)
// ============================================================================

/// Compute GFN2-xTB energy and analytical gradients.
///
/// Gradient includes: repulsion, electronic (Pulay + Hellmann-Feynman),
/// Coulomb, D4 dispersion, and solvation contributions.
///
/// @param positions     Nx3 in Angstrom
/// @param atomicNumbers N atomic numbers
/// @param atomCount     Number of atoms
/// @param totalCharge   Net molecular charge
/// @param maxIterations SCC iteration limit
/// @param solvation     Solvation config
/// @return Heap-allocated — free with druse_xtb_free_gradient_result()
DruseXTBGradientResult* druse_xtb_compute_gradient(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    int32_t maxIterations,
    DruseXTBSolvationConfig solvation
);

void druse_xtb_free_gradient_result(DruseXTBGradientResult *result);

// ============================================================================
// MARK: - Geometry Optimization API (L-BFGS)
// ============================================================================

/// Optimize molecular geometry using GFN2-xTB gradients and L-BFGS.
///
/// @param positions          Nx3 initial positions in Angstrom (NOT modified)
/// @param atomicNumbers      N atomic numbers
/// @param atomCount          Number of atoms
/// @param totalCharge        Net molecular charge
/// @param solvation          Solvation config
/// @param optLevel           Convergence tightness
/// @param maxSteps           Maximum optimization steps (0 = auto based on optLevel)
/// @param freezeMask         Optional bool[N] array: true = freeze that atom. NULL = optimize all.
/// @param referencePositions Optional Nx3 reference positions for harmonic restraints (Angstrom).
///                           NULL = no restraints. When set, atoms are restrained to these positions.
/// @param restraintStrength  Harmonic spring constant in Hartree/Angstrom^2 (e.g. 0.005).
///                           Only used when referencePositions is non-NULL.
/// @return Heap-allocated — free with druse_xtb_free_opt_result()
DruseXTBOptResult* druse_xtb_optimize_geometry(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    DruseXTBSolvationConfig solvation,
    DruseXTBOptLevel optLevel,
    int32_t maxSteps,
    const bool *freezeMask,
    const float *referencePositions,
    float restraintStrength
);

void druse_xtb_free_opt_result(DruseXTBOptResult *result);

// ============================================================================
// MARK: - GPU Acceleration Context
// ============================================================================

/// GPU dispatch function pointers. Set by Swift Metal layer at init.
/// When set, eligible O(N²) pairwise computations are offloaded to GPU.
/// All functions use double precision at the interface; float conversion
/// happens inside the GPU accelerator.
typedef struct DruseXTBGPUContext {
    void *context;  // opaque pointer to XTBMetalAccelerator (retained)

    /// Compute coordination numbers. cn_out must be pre-allocated [natom].
    void (*gpu_compute_cn)(void *ctx, const double *pos_bohr, const int32_t *Z,
                           int32_t natom, double *cn_out);

    /// Compute D4 dispersion energy. gradient is optional (NULL = energy only).
    /// gradient must be pre-zeroed [3*natom] if non-NULL.
    /// Returns total dispersion energy (Hartree).
    double (*gpu_compute_d4)(void *ctx, const double *pos_bohr, const int32_t *Z,
                             int32_t natom, const double *cn, double *gradient);

    /// Compute repulsion energy. gradient optional [3*natom].
    /// Returns total repulsion energy (Hartree).
    double (*gpu_compute_repulsion)(void *ctx, const double *pos_bohr, const int32_t *Z,
                                    int32_t natom, double *gradient);

    /// Compute Born radii (OBC-II) and Gaussian SASA.
    /// brad_out, sasa_out must be pre-allocated [natom].
    void (*gpu_compute_born)(void *ctx, const double *pos_bohr, const int32_t *Z,
                             int32_t natom, float probe_rad_bohr, float offset_bohr,
                             float born_scale, double *brad_out, double *sasa_out);

    /// Propagate dE/dCN gradient via chain rule. Accumulates into gradient [3*natom].
    void (*gpu_compute_cn_gradient)(void *ctx, const double *pos_bohr, const int32_t *Z,
                                    int32_t natom, const double *dEdCN, double *gradient);
} DruseXTBGPUContext;

/// Set the global GPU context. Pass NULL to revert to CPU-only.
/// The context is NOT owned by the xTB module; caller manages lifetime.
void druse_xtb_set_gpu_context(DruseXTBGPUContext *ctx);

#ifdef __cplusplus
}
#endif

#endif // DRUSE_XTB_H
