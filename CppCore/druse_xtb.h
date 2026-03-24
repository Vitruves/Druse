// ============================================================================
// druse_xtb.h — GFN2-xTB Mulliken charge calculator (C API)
//
// Self-contained implementation of the GFN2-xTB tight-binding method for
// computing Mulliken partial charges. No Fortran dependencies — uses Apple
// Accelerate LAPACK for eigenvalue solves and BLAS for matrix operations.
//
// Reference: C. Bannwarth, S. Ehlert and S. Grimme,
//            J. Chem. Theory Comput. 2019, 15, 1652-1671
// ============================================================================

#ifndef DRUSE_XTB_H
#define DRUSE_XTB_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - GFN2-xTB Charge Result
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
// MARK: - API Functions
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

#ifdef __cplusplus
}
#endif

#endif // DRUSE_XTB_H
