#ifndef DRUSE_OPENMM_H
#define DRUSE_OPENMM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - OpenMM Pocket Refinement (C API)
// ============================================================================

/// A single atom for OpenMM refinement (positions in Ångström).
typedef struct {
    float x, y, z;           // position (Å)
    float charge;             // partial charge (e)
    float sigmaNm;            // LJ sigma (nm)
    float epsilonKJ;          // LJ epsilon (kJ/mol)
    float mass;               // atomic mass (Da)
    int32_t atomicNum;        // atomic number (for bond estimation)
    bool isPocket;            // true if within pocket cutoff
} DruseOpenMMAtom;

/// A bond in the protein for geometry constraints during refinement.
typedef struct {
    int32_t atom1;            // index into protein atom array
    int32_t atom2;            // index into protein atom array
    float lengthNm;           // equilibrium bond length (nm)
} DruseOpenMMBond;

/// A ligand interaction site (fixed during refinement — acts as external field).
typedef struct {
    float x, y, z;            // position (nm — already converted)
    float charge;             // partial charge (e)
    float sigmaNm;            // LJ sigma (nm)
    float epsilonKJ;          // LJ epsilon (kJ/mol)
} DruseOpenMMLigandSite;

/// Result of pocket refinement.
typedef struct {
    bool success;
    float interactionEnergyKcal;   // protein-ligand interaction (kcal/mol)
    float *refinedPositionsX;      // refined protein heavy atom positions (Å)
    float *refinedPositionsY;
    float *refinedPositionsZ;
    int32_t atomCount;
    char errorMessage[256];
} DruseOpenMMResult;

/// Run OpenMM pocket refinement.
/// Protein atoms and bonds define the receptor.
/// Ligand sites define fixed interaction points (positions in nm).
/// Restraint force constants: pocketK and backboneK (kJ/mol/nm²).
/// Returns heap-allocated result — caller must free with druse_free_openmm_result().
DruseOpenMMResult* druse_openmm_refine(
    const DruseOpenMMAtom *proteinAtoms,
    int32_t proteinAtomCount,
    const DruseOpenMMBond *proteinBonds,
    int32_t proteinBondCount,
    const DruseOpenMMLigandSite *ligandSites,
    int32_t ligandSiteCount,
    float pocketK,           // restraint for pocket atoms (kJ/mol/nm²), default 600
    float backboneK,         // restraint for non-pocket atoms (kJ/mol/nm²), default 8000
    int32_t maxIterations    // minimization iterations, default 250
);

/// Free a result from druse_openmm_refine.
void druse_free_openmm_result(DruseOpenMMResult *result);

/// Check if OpenMM is available (compiled in).
bool druse_openmm_available(void);

// ============================================================================
// MARK: - OpenMM Loop Refinement (C API)
// ============================================================================

/// An angle restraint for backbone geometry.
typedef struct {
    int32_t atom1;
    int32_t atom2;            // central atom
    int32_t atom3;
    float angleDegrees;       // equilibrium angle (degrees)
    float forceConstant;      // kJ/mol/rad^2
} DruseOpenMMAngle;

/// A torsion (dihedral) restraint for backbone geometry.
typedef struct {
    int32_t atom1;
    int32_t atom2;
    int32_t atom3;
    int32_t atom4;
    int32_t periodicity;
    float phaseDegrees;       // equilibrium phase (degrees)
    float forceConstant;      // kJ/mol
} DruseOpenMMTorsion;

/// Result of loop refinement.
typedef struct {
    bool success;
    float finalEnergyKcal;
    float *refinedPositionsX;     // all atom positions (Å)
    float *refinedPositionsY;
    float *refinedPositionsZ;
    int32_t atomCount;
    char errorMessage[256];
} DruseOpenMMLoopResult;

/// Run OpenMM loop refinement.
/// Strong restraints on non-loop atoms, free movement for loop atoms.
/// Includes angle/torsion forces for proper backbone geometry and gap closure.
DruseOpenMMLoopResult* druse_openmm_refine_loop(
    const DruseOpenMMAtom *atoms,
    int32_t atomCount,
    const DruseOpenMMBond *bonds,
    int32_t bondCount,
    const DruseOpenMMAngle *angles,
    int32_t angleCount,
    const DruseOpenMMTorsion *torsions,
    int32_t torsionCount,
    const bool *isLoopAtom,            // per-atom flag: true for loop atoms
    int32_t maxIterations              // minimization iterations (default 1000)
);

/// Free a result from druse_openmm_refine_loop.
void druse_free_openmm_loop_result(DruseOpenMMLoopResult *result);

#ifdef __cplusplus
}
#endif

#endif // DRUSE_OPENMM_H
