#ifndef DRUSE_CORE_H
#define DRUSE_CORE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MARK: - Atom Data (returned from C++ to Swift)
// ============================================================================

typedef struct {
    float x, y, z;
    int32_t atomicNum;
    float charge;       // Gasteiger partial charge
    int32_t formalCharge;
    char symbol[4];
    char name[8];
    char residueName[8];
    int32_t residueSeq;
    char chainID[4];
    float occupancy;
    float tempFactor;
    char altLoc[4];
    bool isHetAtom;
} DruseAtom;

typedef struct {
    int32_t atom1;
    int32_t atom2;
    int32_t order;      // 1=single, 2=double, 3=triple, 4=aromatic
} DruseBond;

typedef struct {
    DruseAtom *atoms;
    int32_t atomCount;
    DruseBond *bonds;
    int32_t bondCount;
    char name[256];
    char smiles[1024];
    float molecularWeight;
    float logP;
    float tpsa;
    int32_t hbd;
    int32_t hba;
    int32_t rotatableBonds;
    int32_t numConformers;
    bool success;
    char errorMessage[512];
} DruseMoleculeResult;

// ============================================================================
// MARK: - SMILES → 3D Molecule
// ============================================================================

/// Parse SMILES and generate 3D coordinates with MMFF94 minimization.
/// Caller must free the result with druse_free_molecule_result().
DruseMoleculeResult* druse_smiles_to_3d(const char *smiles, const char *name);

/// Parse SMILES, generate multiple conformers, return the lowest-energy one.
/// numConformers: number of conformers to generate (typically 50-200).
DruseMoleculeResult* druse_smiles_to_3d_conformers(
    const char *smiles,
    const char *name,
    int32_t numConformers,
    bool mmffMinimize
);

// ============================================================================
// MARK: - Molecule Preparation
// ============================================================================

/// Add hydrogens to a molecule (from SMILES).
/// Returns a new result with hydrogens added.
DruseMoleculeResult* druse_add_hydrogens(const char *smiles, const char *name);

/// Compute Gasteiger charges. Returns result with charges populated.
DruseMoleculeResult* druse_compute_gasteiger_charges(const char *smiles, const char *name);

/// Full preparation pipeline: sanitize → addHs → 3D → MMFF minimize → Gasteiger charges
DruseMoleculeResult* druse_prepare_ligand(
    const char *smiles,
    const char *name,
    int32_t numConformers,
    bool addHydrogens,
    bool minimize,
    bool computeCharges
);

// ============================================================================
// MARK: - Conformer Generation (returns all conformers, not just best)
// ============================================================================

typedef struct {
    DruseMoleculeResult **conformers;
    int32_t count;
    double *energies;       // MMFF94 energy per conformer
} DruseConformerSet;

/// Generate N conformers with ETKDGv3, MMFF minimize each, return all sorted by energy.
DruseConformerSet* druse_generate_conformers(
    const char *smiles,
    const char *name,
    int32_t numConformers,
    bool minimize
);

void druse_free_conformer_set(DruseConformerSet *set);

// ============================================================================
// MARK: - Protein Preparation (fragment-based)
// ============================================================================

/// Add hydrogens to each standard residue fragment parsed from PDB content.
/// Returns atom count including new H atoms with 3D coordinates.
/// pdbContent: raw PDB file text.
DruseMoleculeResult* druse_add_hydrogens_pdb(const char *pdbContent);

/// Compute Gasteiger charges on each residue fragment from PDB.
/// Returns result with per-atom charges populated.
DruseMoleculeResult* druse_compute_charges_pdb(const char *pdbContent);

/// Compute Gasteiger charges for a molecule provided as an MDL mol block.
/// Preserves atom ordering so charges can be merged back onto an existing ligand.
DruseMoleculeResult* druse_compute_charges_molblock(const char *molBlock);

/// Compute upstream Vina XS atom types for a molecule provided as an MDL mol block.
/// Writes one type per atom into `outTypes` and returns the atom count, or -1 on error.
int32_t druse_compute_vina_types_molblock(const char *molBlock, int32_t *outTypes, int32_t maxAtoms);

// ============================================================================
// MARK: - Descriptors
// ============================================================================

typedef struct {
    float molecularWeight;
    float exactMW;
    float logP;         // Wildman-Crippen LogP
    float tpsa;         // Topological Polar Surface Area
    int32_t hbd;        // H-bond donors
    int32_t hba;        // H-bond acceptors
    int32_t rotatableBonds;
    int32_t rings;
    int32_t aromaticRings;
    int32_t heavyAtomCount;
    float fractionCSP3;
    bool lipinski;      // passes Lipinski's rule of 5
    bool veber;         // passes Veber's rules
} DruseDescriptors;

/// Compute molecular descriptors from SMILES.
DruseDescriptors druse_compute_descriptors(const char *smiles);

// ============================================================================
// MARK: - Batch Processing
// ============================================================================

/// Process a batch of SMILES strings.
/// Returns array of results (caller must free each with druse_free_molecule_result).
/// count is set to number of results.
DruseMoleculeResult** druse_batch_process(
    const char **smiles_array,
    const char **name_array,
    int32_t count,
    bool addHydrogens,
    bool minimize,
    bool computeCharges
);

// ============================================================================
// MARK: - Memory Management
// ============================================================================

void druse_free_molecule_result(DruseMoleculeResult *result);
void druse_free_batch_results(DruseMoleculeResult **results, int32_t count);

// ============================================================================
// MARK: - Version Info
// ============================================================================

const char* druse_rdkit_version(void);

// ============================================================================
// MARK: - Torsion Tree
// ============================================================================

typedef struct {
    int32_t atom1;           // pivot atom (stationary side)
    int32_t atom2;           // axis atom (rotating side)
    int32_t movingStart;     // index into movingAtomIndices array
    int32_t movingCount;     // number of atoms that rotate
} DruseTorsionEdge;

typedef struct {
    DruseTorsionEdge *edges;
    int32_t edgeCount;
    int32_t *movingAtomIndices;  // flat array of atom indices
    int32_t totalMovingAtoms;
} DruseTorsionTree;

/// Build torsion tree from SMILES. Identifies rotatable bonds,
/// BFS to find movable atom groups, orders root->leaf.
/// Caller must free with druse_free_torsion_tree().
DruseTorsionTree* druse_build_torsion_tree(const char *smiles);

/// Build torsion tree from an MDL mol block while preserving the atom order
/// encoded in that mol block.
DruseTorsionTree* druse_build_torsion_tree_molblock(const char *molBlock);

void druse_free_torsion_tree(DruseTorsionTree *tree);

// ============================================================================
// MARK: - Spatial Queries (nanoflann KD-tree)
// ============================================================================

typedef void* DruseKDTree;

/// Build KD-tree from 3D positions. Returns opaque handle.
DruseKDTree druse_build_kdtree(const float *positions, int32_t numPoints);

/// Query KD-tree for all points within radius of query point.
/// Returns count of found points, writes indices to outIndices (must be pre-allocated).
int32_t druse_kdtree_radius_search(
    DruseKDTree tree,
    const float *queryPoint,
    float radius,
    int32_t *outIndices,
    int32_t maxResults
);

/// Free KD-tree.
void druse_free_kdtree(DruseKDTree tree);

// ============================================================================
// MARK: - Linear Algebra (Eigen)
// ============================================================================

/// Kabsch superposition: align mobile onto reference.
/// Both arrays are Nx3 floats (row-major). Returns RMSD.
/// rotation_out: 3x3 matrix (row-major), translation_out: 3 floats.
float druse_kabsch_superpose(
    const float *mobile, const float *reference, int32_t n,
    float *rotation_out, float *translation_out
);

/// Compute RMSD between two coordinate sets (no alignment).
float druse_compute_rmsd(const float *a, const float *b, int32_t n);

// ============================================================================
// MARK: - Energy Minimization
// ============================================================================

/// Compatibility wrapper for a thorough MMFF94 minimization of a small molecule from SMILES.
/// The API name remains for compatibility; current implementation does not use LBFGS++ yet.
DruseMoleculeResult* druse_minimize_lbfgs(const char *smiles, const char *name, int32_t maxIters);

// ============================================================================
// MARK: - mmCIF Parser
// ============================================================================

/// Parse mmCIF format content. Returns same structure as PDB parsing.
DruseMoleculeResult* druse_parse_mmcif(const char *content);

// ============================================================================
// MARK: - Electrostatic Potential
// ============================================================================

/// Compute ESP at surface points from atom positions and charges.
/// atomPositions: Nx3, charges: N, surfacePoints: Mx3, outESP: M.
void druse_compute_esp(
    const float *atomPositions, const float *charges, int32_t nAtoms,
    const float *surfacePoints, int32_t nSurface,
    float *outESP
);

// ============================================================================
// MARK: - Parallel Batch Processing (TBB)
// ============================================================================

/// TBB-parallel batch processing of SMILES strings.
DruseMoleculeResult** druse_batch_process_parallel(
    const char **smiles_array,
    const char **name_array,
    int32_t count,
    bool addHydrogens,
    bool minimize,
    bool computeCharges
);

// ============================================================================
// MARK: - Fingerprints
// ============================================================================

typedef struct {
    float *bits;        // array of 0.0/1.0 values
    int32_t numBits;
} DruseFingerprint;

/// Compute Morgan (circular) fingerprint from SMILES.
/// radius: typically 2 for ECFP4, nBits: typically 2048.
DruseFingerprint* druse_morgan_fingerprint(const char *smiles, int32_t radius, int32_t nBits);

void druse_free_fingerprint(DruseFingerprint *fp);

// ============================================================================
// MARK: - 2D Coordinate Generation
// ============================================================================

typedef struct {
    float *coords;          // flat array: [x0, y0, x1, y1, ...] (heavy atoms only)
    int32_t *atomicNums;    // atomic number per heavy atom
    int32_t atomCount;
    DruseBond *bonds;       // bonds between heavy atoms
    int32_t bondCount;
} Druse2DResult;

/// Compute 2D depiction coordinates for a molecule from SMILES.
/// Returns heavy-atom positions suitable for 2D interaction diagrams.
/// Caller must free with druse_free_2d_result().
Druse2DResult* druse_compute_2d_coords(const char *smiles);

void druse_free_2d_result(Druse2DResult *result);

#ifdef __cplusplus
}
#endif

#endif /* DRUSE_CORE_H */
