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
// MARK: - Tautomer & Protomer Enumeration
// ============================================================================

typedef struct {
    int32_t kind;       // 0 = tautomer, 1 = protomer
    char label[128];    // human-readable description
} DruseVariantInfo;

typedef struct {
    DruseMoleculeResult **variants;
    int32_t count;
    double *scores;             // MMFF energy or tautomer score per variant
    DruseVariantInfo *infos;    // per-variant metadata (kind + label)
} DruseVariantSet;

/// Enumerate tautomers using RDKit MolStandardize::TautomerEnumerator.
/// Each tautomer gets 3D coordinates (ETKDGv3 + MMFF94 minimize).
/// maxTautomers: maximum number of tautomers to return.
/// energyCutoff: discard tautomers with MMFF energy > best + cutoff kcal/mol (0 = no cutoff).
DruseVariantSet* druse_enumerate_tautomers(
    const char *smiles,
    const char *name,
    int32_t maxTautomers,
    double energyCutoff
);

/// Enumerate protomers (protonation state variants) at a target pH.
/// Uses SMARTS-based ionizable group detection with Henderson-Hasselbalch.
/// maxProtomers: maximum total protomers to return.
/// pH: target pH (typically 7.4).
/// pkaThreshold: generate both states for groups where |pH - pKa| < threshold.
DruseVariantSet* druse_enumerate_protomers(
    const char *smiles,
    const char *name,
    int32_t maxProtomers,
    double pH,
    double pkaThreshold
);

void druse_free_variant_set(DruseVariantSet *set);

// ============================================================================
// MARK: - Unified Ligand Ensemble Preparation
// ============================================================================

/// Per-member data in the prepared ensemble.
typedef struct {
    DruseMoleculeResult *molecule;   // 3D structure with H, charges, minimized
    double mmffEnergy;               // MMFF94 energy (kcal/mol)
    double boltzmannWeight;          // population fraction (0-1, sums to 1 across ensemble)
    int32_t kind;                    // 0=parent, 1=tautomer, 2=protomer, 3=tautomer+protomer
    char label[256];                 // e.g. "Taut2_ProtAmineH_Conf3"
    char smiles[512];                // canonical SMILES for this form
    int32_t conformerIndex;          // conformer rank within its chemical form
    int32_t formIndex;               // index of the chemical form (protomer/tautomer combo)
} DruseEnsembleMember;

/// Full ensemble result from unified preparation.
typedef struct {
    DruseEnsembleMember *members;    // array of all prepared forms
    int32_t count;                   // total members
    int32_t numForms;                // number of distinct chemical forms (before conformer expansion)
    int32_t numConformersPerForm;    // conformers generated per form
    bool success;
    char errorMessage[512];
} DruseEnsembleResult;

/// Unified ligand preparation: protomer × tautomer × conformer pipeline.
///
/// Pipeline:
/// 1. Enumerate protomers at target pH (Henderson-Hasselbalch, ambiguous sites)
/// 2. For each protomer, enumerate tautomers
/// 3. Deduplicate across the cross-product by canonical SMILES
/// 4. For each unique form: generate conformers (ETKDGv3 + MMFF94)
/// 5. Full preparation: add polar H, Gasteiger charges
/// 6. Compute Boltzmann weights: w_i = exp(-E_i/kT) / Σ exp(-E_j/kT)
///
/// @param smiles         Input SMILES
/// @param name           Molecule name
/// @param pH             Target pH (e.g. 7.4)
/// @param pkaThreshold   Henderson-Hasselbalch ambiguity window (e.g. 2.0)
/// @param maxTautomers   Max tautomers per protomer form
/// @param maxProtomers   Max protomers from parent
/// @param energyCutoff   Discard forms with E > E_best + cutoff (kcal/mol, 0=no cutoff)
/// @param conformersPerForm  Conformers to generate per unique chemical form
/// @param temperature    Boltzmann temperature in Kelvin (298.15 = room temp)
/// @return Heap-allocated result — free with druse_free_ensemble_result()
DruseEnsembleResult* druse_prepare_ligand_ensemble(
    const char *smiles,
    const char *name,
    double pH,
    double pkaThreshold,
    int32_t maxTautomers,
    int32_t maxProtomers,
    double energyCutoff,
    int32_t conformersPerForm,
    double temperature
);

void druse_free_ensemble_result(DruseEnsembleResult *result);

// ============================================================================
// MARK: - Ionizable Site Detection & Per-Site Protomer Generation
// ============================================================================

/// An ionizable site detected in a molecule.
typedef struct {
    int32_t atomIdx;          // index of the ionizable atom
    bool isAcid;              // true = deprotonates (loses H), false = protonates (gains H)
    double defaultPKa;        // pKa from the built-in lookup table
    char groupName[64];       // human-readable group name (e.g. "Piperazine N")
} DruseIonSite;

/// Result of ionizable site detection.
typedef struct {
    DruseIonSite *sites;
    int32_t count;
} DruseIonSiteResult;

/// Detect all ionizable sites in a molecule.
/// Unlike protomer enumeration, this does NOT filter by pH — it returns ALL sites
/// so the caller can compute pKa for each and decide which are ambiguous.
DruseIonSiteResult* druse_detect_ionizable_sites(const char *smiles);
void druse_free_ion_sites(DruseIonSiteResult *result);

/// A pair of 3D structures: protonated and deprotonated forms of a single ionizable site.
typedef struct {
    DruseMoleculeResult *protonated;      // 3D structure (H added, minimized)
    DruseMoleculeResult *deprotonated;    // 3D structure (H removed, minimized)
    int32_t protonatedCharge;             // total formal charge of protonated form
    int32_t deprotonatedCharge;           // total formal charge of deprotonated form
    bool success;
    char errorMessage[256];
} DruseSiteProtomerPair;

/// Generate protonated and deprotonated 3D structures for a specific ionizable site.
/// Both forms are MMFF94-minimized and have Gasteiger charges.
/// @param smiles     Input SMILES
/// @param atomIdx    Atom index of the ionizable site
/// @param isAcid     If true, toggle deprotonation (remove H); if false, toggle protonation (add H)
DruseSiteProtomerPair* druse_generate_site_protomers(
    const char *smiles, int32_t atomIdx, bool isAcid);
void druse_free_site_protomer_pair(DruseSiteProtomerPair *result);

/// Ionizable site definition for GNN-driven ensemble preparation.
typedef struct {
    int32_t atomIdx;    // heavy-atom index in the canonical SMILES molecule
    double pKa;         // predicted pKa
    bool isAcid;        // true = deprotonates (loses H), false = protonates (gains H)
} DruseIonSiteDef;

/// Unified ligand ensemble preparation.
/// If sites/nSites are provided, uses those directly (GNN path — recommended).
/// If sites is NULL or nSites <= 0, falls back to SMARTS detection + lookup table.
/// @param sites    Array of ionizable sites with pKa (from GNN or other source), or NULL
/// @param nSites   Number of sites (0 = fall back to SMARTS table)
DruseEnsembleResult* druse_prepare_ligand_ensemble_ex(
    const char *smiles, const char *name,
    double pH, double pkaThreshold,
    int32_t maxTautomers, int32_t maxProtomers,
    double energyCutoff, int32_t conformersPerForm,
    double temperature,
    const DruseIonSiteDef *sites, int32_t nSites
);

// ============================================================================
// MARK: - pKa GNN Featurization
// ============================================================================

#define PKA_MAX_ATOMS  128
#define PKA_MAX_EDGES  512
#define PKA_ATOM_FEAT   25
#define PKA_BOND_FEAT   10

/// Molecular graph featurized for the pKa GNN Metal shader.
/// Arrays are heap-allocated to avoid Swift C-import tuple size limits.
typedef struct {
    float *atomFeatures;         // [numAtoms * 25] row-major (caller must not free)
    float *bondFeatures;         // [numEdges * 10] row-major
    int32_t *edgeSrc;            // [numEdges] source atom indices
    int32_t *edgeDst;            // [numEdges] dest atom indices
    int32_t numAtoms;
    int32_t numEdges;
    bool success;
    char errorMessage[256];
} DrusePKaGraphResult;

/// Featurize a SMILES string into a molecular graph for pKa GNN inference.
/// Features match the Python training script (train_pka.py) exactly.
/// Caller must free with druse_free_pka_graph().
DrusePKaGraphResult* druse_featurize_pka_graph(const char *smiles);
void druse_free_pka_graph(DrusePKaGraphResult *result);

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

/// Convert an MDL mol block to canonical SMILES.
/// Uses RDKit's native MolBlockToMol parser, which handles both 2D and 3D coordinates correctly.
/// Returns a result with the SMILES string populated (atoms/bonds are NOT populated).
const char* druse_molblock_to_smiles(const char *molBlock);

/// Free the string returned by druse_molblock_to_smiles.
void druse_free_string(const char *str);

/// Convert atoms+bonds (with 3D coordinates) to SMILES.
/// Builds an RWMol from the provided atom/bond arrays, sanitizes, and returns
/// a result with the canonical SMILES string populated.
/// Useful for co-crystallized ligands extracted from PDB.
DruseMoleculeResult* druse_atoms_bonds_to_smiles(
    const DruseAtom *atoms,
    int32_t atomCount,
    const DruseBond *bonds,
    int32_t bondCount,
    const char *name
);

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

/// Compute MMFF94 strain energy for a ligand with specified heavy atom coordinates.
/// Builds molecule from SMILES, places heavy atoms at given positions, optimizes H positions,
/// then evaluates MMFF94 energy. Returns energy in kcal/mol, or NaN on failure.
/// heavyPositions: interleaved [x0,y0,z0, x1,y1,z1, ...] for numHeavy atoms.
double druse_mmff_strain_energy(const char *smiles, const float *heavyPositions, int32_t numHeavy);

/// Compute MMFF94 energy of a molecule's current conformation (from SMILES with 3D embedding).
/// Returns energy in kcal/mol after MMFF minimization, or NaN on failure.
/// Use as reference energy; strain = druse_mmff_strain_energy() - reference.
double druse_mmff_reference_energy(const char *smiles);

// ============================================================================
// MARK: - mmCIF Parser
// ============================================================================

/// Parse macromolecular structure content (PDB/mmCIF/mmJSON auto-detected by gemmi).
DruseMoleculeResult* druse_parse_structure(const char *content);

/// Parse mmCIF format content. Returns same structure as PDB parsing.
DruseMoleculeResult* druse_parse_mmcif(const char *content);

// ============================================================================
// MARK: - Residue Topology (gemmi chemcomp)
// ============================================================================

typedef struct {
    char atomName[8];
    int32_t atomicNum;
    int32_t formalCharge;
    bool isHydrogen;
} DruseResidueTopologyAtom;

typedef struct {
    char atom1[8];
    char atom2[8];
    int32_t order;          // 1=single, 2=double, 3=triple, 4=aromatic/delocalized
    float idealLength;      // derived from ideal CCD coordinates when available
} DruseResidueTopologyBond;

typedef struct {
    char atom1[8];
    char atom2[8];          // central atom
    char atom3[8];
    float idealAngleDegrees; // derived from ideal CCD coordinates when available
} DruseResidueTopologyAngle;

typedef struct {
    DruseResidueTopologyAtom *atoms;
    int32_t atomCount;
    DruseResidueTopologyBond *bonds;
    int32_t bondCount;
    DruseResidueTopologyAngle *angles;
    int32_t angleCount;
    char residueName[16];
    bool success;
    char errorMessage[512];
} DruseResidueTopologyResult;

/// Parse a CCD/chemcomp CIF block and expose atom/bond/angle template data.
DruseResidueTopologyResult* druse_parse_chemcomp_cif(const char *content);
void druse_free_residue_topology_result(DruseResidueTopologyResult *result);

// ============================================================================
// MARK: - Protein Spatial Queries (gemmi neighbor search)
// ============================================================================

/// Query atom indices within `radius` of `queryPoint` in structure content.
/// Returns the number of indices written, or -1 on parse/query failure.
int32_t druse_find_structure_neighbors(
    const char *content,
    const float *queryPoint,
    float radius,
    bool includeHydrogens,
    int32_t *outIndices,
    int32_t maxResults
);

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
/// If cancel_flag is non-null, the loop checks it each iteration and skips
/// remaining work once *cancel_flag becomes true.
/// The cancel_flag is read atomically; the caller should use
/// druse_atomic_cancel_store() to set it.
DruseMoleculeResult** druse_batch_process_parallel(
    const char **smiles_array,
    const char **name_array,
    int32_t count,
    bool addHydrogens,
    bool minimize,
    bool computeCharges,
    const volatile int32_t *cancel_flag
);

/// Atomically store a value into a cancel flag (use value=1 to cancel, 0 to reset).
void druse_atomic_cancel_store(volatile int32_t *flag, int32_t value);

/// Atomically load the current value of a cancel flag.
int32_t druse_atomic_cancel_load(const volatile int32_t *flag);

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
// MARK: - Fragment Decomposition (for Fragment-Based Docking)
// ============================================================================

/// Result of decomposing a ligand into rigid fragments at rotatable bonds.
typedef struct {
    int32_t *fragmentMembership;    // length = numHeavyAtoms: fragment index per heavy atom
    int32_t numHeavyAtoms;
    int32_t numFragments;
    int32_t *fragmentSizes;         // length = numFragments: atom count per fragment
    int32_t anchorFragmentIdx;      // index of largest rigid fragment (default anchor)
    /// Fragment connectivity: flat [parentFrag, childFrag, torsionBondAtom1, torsionBondAtom2] × numConnections
    int32_t *connections;
    int32_t numConnections;
    /// Fragment centroids (flat xyz × numFragments)
    float *centroids;
    bool success;
    char errorMessage[512];
} DruseFragmentResult;

/// Decompose a molecule into rigid fragments at rotatable bonds.
/// Rings are kept intact. Terminal groups (single heavy atom leaves) form their own fragment.
/// If scaffoldSmarts is non-NULL, the anchor is the fragment containing the most scaffold atoms.
DruseFragmentResult* druse_decompose_fragments(
    const char *smiles,
    const char *scaffoldSmarts
);

void druse_free_fragment_result(DruseFragmentResult *result);

/// Scaffold matching: check if a molecule contains a substructure and compute Tanimoto similarity.
typedef struct {
    bool hasMatch;
    int32_t *matchedAtomIndices;    // heavy atom indices matching the scaffold
    int32_t matchCount;
    float tanimotoSimilarity;       // Morgan (radius=2, 2048 bits) Tanimoto coefficient
} DruseScaffoldMatch;

DruseScaffoldMatch* druse_match_scaffold(const char *smiles, const char *scaffoldSmarts);
void druse_free_scaffold_match(DruseScaffoldMatch *result);

/// Compute Tanimoto similarity between two molecules (Morgan radius=2, 2048 bits).
float druse_tanimoto_similarity(const char *smiles1, const char *smiles2);

// ============================================================================
// MARK: - 2D Coordinate Generation
// ============================================================================

typedef struct {
    float *coords;          // flat array: [x0, y0, x1, y1, ...] (heavy atoms only)
    int32_t *atomicNums;    // atomic number per heavy atom
    int32_t atomCount;
    DruseBond *bonds;       // bonds between heavy atoms (Kekulized: no order 4)
    int32_t bondCount;
    bool *isAromatic;       // per-atom aromaticity flag (set before Kekulization)
} Druse2DResult;

/// Compute 2D depiction coordinates for a molecule from SMILES.
/// Returns heavy-atom positions suitable for 2D interaction diagrams.
/// Caller must free with druse_free_2d_result().
Druse2DResult* druse_compute_2d_coords(const char *smiles);

void druse_free_2d_result(Druse2DResult *result);

/// Generate an SVG depiction of a molecule from SMILES using RDKit MolDraw2DSVG.
/// Returns a heap-allocated null-terminated C string with SVG content.
/// Includes wedge/dash stereo bonds, aromatic notation, and element coloring.
/// Caller must free with druse_free_string().
char* druse_mol_to_svg(const char *smiles, int32_t width, int32_t height);

// ============================================================================
// MARK: - Entity Sequence Extraction (SEQRES)
// ============================================================================

/// Per-chain full sequence from SEQRES / entity_poly_seq records.
typedef struct {
    char chainID[4];               // author chain ID
    char (*residueNames)[8];       // array of 3-letter residue codes
    int32_t residueCount;          // length of residueNames array
} DruseChainSequence;

/// Result of entity sequence extraction.
typedef struct {
    DruseChainSequence *chains;
    int32_t chainCount;
    bool success;
    char errorMessage[512];
} DruseEntitySequenceResult;

/// Extract per-chain SEQRES / entity_poly_seq sequences from structure content.
/// Uses gemmi's Entity::full_sequence populated by setup_entities().
/// Returns heap-allocated result — caller must free with druse_free_entity_sequence_result().
DruseEntitySequenceResult* druse_get_entity_sequences(const char *content);

/// Free result from druse_get_entity_sequences.
void druse_free_entity_sequence_result(DruseEntitySequenceResult *result);

// ============================================================================
// MARK: - Pharmacophore Feature Detection
// ============================================================================

/// Types of pharmacophore features detected from a molecule.
/// Maps to RDKit BaseFeatures.fdef families.
enum DrusePharmacophoreType {
    DRUSE_PHARMA_DONOR = 0,
    DRUSE_PHARMA_ACCEPTOR = 1,
    DRUSE_PHARMA_HYDROPHOBIC = 2,
    DRUSE_PHARMA_AROMATIC = 3,
    DRUSE_PHARMA_POS_IONIZABLE = 4,
    DRUSE_PHARMA_NEG_IONIZABLE = 5,
};

/// A single pharmacophore feature detected in a molecule.
typedef struct {
    float x, y, z;                      // 3D position (centroid of involved atoms)
    int32_t type;                       // DrusePharmacophoreType
    int32_t *atomIndices;               // heavy atom indices involved in this feature
    int32_t atomCount;                  // number of atoms in this feature
    char familyName[32];                // e.g. "Donor", "Acceptor", "Hydrophobe"
} DrusePharmacophoreFeature;

/// Result of pharmacophore feature detection.
typedef struct {
    DrusePharmacophoreFeature *features;
    int32_t featureCount;
    bool success;
    char errorMessage[512];
} DrusePharmacophoreFeatureResult;

/// Detect pharmacophore features from a SMILES string using RDKit's
/// MolChemicalFeatureFactory with BaseFeatures.fdef. The molecule is
/// embedded in 3D (ETKDGv3 + MMFF94) so feature positions are meaningful.
/// Caller must free with druse_free_pharmacophore_features().
DrusePharmacophoreFeatureResult* druse_detect_pharmacophore_features(const char *smiles);

void druse_free_pharmacophore_features(DrusePharmacophoreFeatureResult *result);

// ============================================================================
// MARK: - Maximum Common Substructure (MCS)
// ============================================================================

/// Result of MCS computation between multiple molecules.
typedef struct {
    char smartsPattern[2048];           // SMARTS of the MCS
    int32_t numAtoms;                   // number of atoms in MCS
    int32_t numBonds;                   // number of bonds in MCS
    bool completed;                     // false if timed out
    bool success;
    char errorMessage[512];
} DruseMCSResult;

/// Find the Maximum Common Substructure among a set of SMILES.
/// timeoutSeconds: max time in seconds (0 = no timeout).
/// Caller must free with druse_free_mcs_result().
DruseMCSResult* druse_find_mcs(
    const char **smilesArray,
    int32_t numMols,
    int32_t timeoutSeconds
);

void druse_free_mcs_result(DruseMCSResult *result);

/// Detect pharmacophore features from a SMILES string, using provided
/// 3D coordinates for heavy atoms instead of generating a new conformer.
/// heavyCoords: interleaved [x0,y0,z0, x1,y1,z1, ...] for numHeavy atoms.
/// Caller must free with druse_free_pharmacophore_features().
DrusePharmacophoreFeatureResult* druse_detect_pharmacophore_features_with_coords(
    const char *smiles,
    const float *heavyCoords,
    int32_t numHeavy
);

#ifdef __cplusplus
}
#endif

#endif /* DRUSE_CORE_H */
