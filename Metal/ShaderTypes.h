#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Buffer indices shared between Swift and Metal
enum BufferIndex {
    BufferIndexVertices  = 0,
    BufferIndexInstances = 1,
    BufferIndexUniforms  = 2,
};

// Texture indices
enum TextureIndex {
    TextureIndexColor = 0,
    TextureIndexDepth = 1,
};

// Uniforms passed to all shaders every frame
struct Uniforms {
    simd_float4x4 modelMatrix;
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 normalMatrix;
    simd_float3   cameraPosition;
    float         _pad0;
    simd_float3   lightDirection;
    float         _pad1;
    simd_float3   lightColor;
    float         ambientIntensity;
    float         time;
    int32_t       selectedAtomIndex;
    float         atomRadiusScale;
    float         bondRadiusScale;
    int32_t       lightingMode;     // 0 = uniform, 1 = directional (Blinn-Phong)
    float         clipNearZ;        // view-space Z clip near (0 = disabled)
    float         clipFarZ;         // view-space Z clip far (0 = disabled)
    int32_t       enableClipping;   // 1 = slab clipping enabled
    int32_t       themeMode;        // 0 = dark, 1 = light
    float         backgroundOpacity; // 0 = plain white/black, 1 = full gradient
    float         surfaceOpacity;    // 0.1–1.0 for molecular surface alpha
    float         gridLineWidth;     // screen-space pixels for grid box lines
    float         _pad2;
};

// Per-atom instance data for impostor sphere rendering
struct AtomInstance {
    simd_float3 position;
    float       radius;
    simd_float4 color;
    int32_t     atomIndex;
    int32_t     flags;      // bit 0: atom selected, bit 1: residue highlighted
    int32_t     formalCharge; // integer formal charge (+1, -1, etc.) for visual indicator
    float       _pad1;
};

// Per-bond instance data for impostor cylinder rendering
struct BondInstance {
    simd_float3 positionA;
    float       radiusA;
    simd_float3 positionB;
    float       radiusB;
    simd_float4 colorA;
    simd_float4 colorB;
};

// Per-vertex data for ribbon/cartoon rendering
struct RibbonVertex {
    simd_float3 position;
    simd_float3 normal;
    simd_float4 color;
    simd_float2 texCoord;
    uint32_t    flags;       // bit 0: residue selected
};

// Per-vertex data for interaction dashed lines
struct InteractionLineVertex {
    simd_float3 positionA;
    simd_float3 positionB;
    simd_float4 color;
    float       dashLength;     // 0 = solid, >0 = dashed
    int32_t     interactionType; // 0=hbond, 1=hydrophobic, 2=ionic, 3=pi-stack, 4=halogen
    float       _pad0;
    float       _pad1;
};

// Per-vertex data for grid box wireframe rendering
struct GridBoxVertex {
    simd_float3 position;
    simd_float4 color;
};

// ============================================================================
// Docking compute types
// ============================================================================

// Grid map parameters passed to compute kernels
struct GridParams {
    simd_float3 origin;     // corner of grid box (min x,y,z)
    float       spacing;    // grid spacing in Angstroms (0.375)
    simd_uint3  dims;       // grid dimensions (nx, ny, nz)
    uint32_t    _pad0;
    uint32_t    totalPoints;
    uint32_t    numProteinAtoms;
    uint32_t    numAffinityTypes;
    uint32_t    _pad2;
    simd_float3 searchCenter;      // translation search center (usually pocket center)
    float       _pad3;
    simd_float3 searchHalfExtent;  // centroid search half-extent around the center
    float       _pad4;
};

// AutoDock Vina X-Score atom types (subset aligned with upstream ids).
enum VinaAtomType {
    VINA_C_H   = 0,
    VINA_C_P   = 1,
    VINA_N_P   = 2,
    VINA_N_D   = 3,
    VINA_N_A   = 4,
    VINA_N_DA  = 5,
    VINA_O_P   = 6,
    VINA_O_D   = 7,
    VINA_O_A   = 8,
    VINA_O_DA  = 9,
    VINA_S_P   = 10,
    VINA_P_P   = 11,
    VINA_F_H   = 12,
    VINA_Cl_H  = 13,
    VINA_Br_H  = 14,
    VINA_I_H   = 15,
    VINA_Si    = 16,
    VINA_At    = 17,
    VINA_MET_D = 18,
    VINA_OTHER = 31
};

// Flags for GridProteinAtom (salt bridge detection in Drusina scoring)
#define GRPROT_FLAG_POS_CHARGED  (1u << 0)  // Lys NZ, Arg NH1/NH2/NE/CZ, His+ NE2/ND1
#define GRPROT_FLAG_NEG_CHARGED  (1u << 1)  // Asp OD1/OD2, Glu OE1/OE2

// Protein atom for grid map computation (packed for GPU)
struct GridProteinAtom {
    simd_float3 position;
    float       vdwRadius;
    float       charge;       // partial charge
    int32_t     vinaType;     // VinaAtomType enum
    uint32_t    flags;        // GRPROT_FLAG_POS_CHARGED, GRPROT_FLAG_NEG_CHARGED
    float       _pad1;
};

// Ligand atom flags
#define LIGATOM_FLAG_AROMATIC 1u

// Ligand atom for scoring
struct DockLigandAtom {
    simd_float3 position;     // will be transformed by pose (centroid-subtracted)
    float       vdwRadius;
    float       charge;
    int32_t     vinaType;     // VinaAtomType enum
    int32_t     formalCharge;  // formal charge (for π-cation / salt bridge detection)
    float       _pad1;
    uint32_t    flags;        // bit 0 = LIGATOM_FLAG_AROMATIC (member of aromatic ring)
};

// A single docking pose (chromosome in GA / ILS state)
struct DockPose {
    simd_float3 translation;
    float       energy;        // total Vina score (kcal/mol)
    simd_float4 rotation;      // quaternion (x,y,z,w)
    float       torsions[32];  // rotatable bond angles
    int32_t     numTorsions;
    int32_t     generation;
    float       stericEnergy;  // intermolecular contribution (kept for compatibility)
    float       hydrophobicEnergy;
    float       hbondEnergy;
    float       torsionPenalty;  // N_rot entropy term
    float       clashPenalty;    // out-of-grid + intramolecular correction term
    float       drusinaCorrection; // Drusina extended scoring correction (π-π, π-cation, halogen, metal)
    float       constraintPenalty; // pharmacophore constraint penalty (0 if all satisfied)

    // Receptor flexibility: sidechain chi angles for 1-6 flexible residues
    float       chiAngles[24]; // up to 6 residues × 4 chi angles each (radians)
    int32_t     numChiAngles;  // total chi angles across all flexible residues (0 = rigid receptor)
    int32_t     _flexPad;
};

// Flexible sidechain atom (excluded from grid, scored explicitly)
struct FlexSidechainAtom {
    simd_float3 referencePosition; // position in the original (rigid) conformation
    float       charge;
    int32_t     vinaType;          // VinaAtomType enum
    int32_t     residueIndex;      // which flexible residue (0-5)
    float       _pad0;
    float       _pad1;
};

// Flexible sidechain torsion (chi angle definition)
struct FlexTorsionEdge {
    int32_t     pivotAtom;    // local index in flex atom buffer: rotation pivot
    int32_t     axisAtom;     // local index in flex atom buffer: defines axis with pivotAtom
    int32_t     movingStart;  // first index in flex moving-indices array
    int32_t     movingCount;  // count of atoms that rotate with this chi angle
    int32_t     chiSlot;      // index into DockPose.chiAngles[] for this torsion
    int32_t     _pad;
};

// Parameters for flexible residue docking
struct FlexParams {
    uint32_t    numFlexAtoms;      // total sidechain atoms across all flexible residues
    uint32_t    numFlexTorsions;   // total chi angle torsions
    uint32_t    numFlexResidues;   // 0 = rigid receptor (skip all flex scoring)
    float       flexWeight;        // weight for flex-ligand pairwise scoring (default 1.0)
    float       chiStep;           // chi angle mutation step in radians (full=0.6, soft=0.25)
    float       _pad0;
};

// Torsion tree edge (for applying torsion rotations)
struct TorsionEdge {
    int32_t     atom1;        // root atom of rotation
    int32_t     atom2;        // axis atom of rotation
    int32_t     movingStart;  // first atom index that moves
    int32_t     movingCount;  // number of atoms that move
};

// GA parameters
struct GAParams {
    uint32_t    populationSize;
    uint32_t    numLigandAtoms;
    uint32_t    numTorsions;
    uint32_t    generation;
    uint32_t    localSearchSteps;
    float       mutationRate;
    float       crossoverRate;
    float       translationStep;
    float       rotationStep;
    float       torsionStep;
    float       gridSpacing;
    float       ligandRadius;
    float       mcTemperature;
    float       referenceIntraEnergy;
    uint32_t    numIntraPairs;       // number of packed pairs in intraPairs buffer (replaces exclusion bitmask)
    uint32_t    _pad0;
};

struct BatchLigandInfo {
    uint32_t    atomStart;
    uint32_t    atomCount;
    uint32_t    poseStart;
    uint32_t    poseCount;
};

struct BatchScreenPose {
    simd_float3 translation;
    float       energy;
    simd_float4 rotation;
    uint32_t    ligandIndex;
    uint32_t    poseIndex;
    float       stericEnergy;
    float       hydrophobicEnergy;
    float       hbondEnergy;
    float       clashPenalty;
    float       _pad0;
};

struct BatchScreenParams {
    uint32_t    totalPoses;
    uint32_t    posesPerLigand;
    uint32_t    numLigands;
    uint32_t    seed;
};

// Per-ligand metadata for batched GA virtual screening
struct BatchedGALigandInfo {
    uint32_t    atomStart;            // offset into flattened DockLigandAtom array
    uint32_t    atomCount;            // number of atoms for this ligand
    uint32_t    torsionEdgeStart;     // offset into flattened TorsionEdge array
    uint32_t    torsionEdgeCount;     // number of torsion edges
    uint32_t    movingIndicesStart;   // offset into flattened moving indices array
    uint32_t    pairListStart;        // offset into flattened intra pair list
    uint32_t    numPairs;             // number of intramolecular pairs
    float       referenceIntraEnergy;
    float       ligandRadius;
    uint32_t    movingIndicesCount;   // total moving indices for this ligand
    uint32_t    _pad0;
    uint32_t    _pad1;
};

// Parameters for batched GA virtual screening (multiple ligands in parallel)
struct BatchedGAParams {
    uint32_t    numLigands;
    uint32_t    populationSizePerLigand;
    uint32_t    totalPoses;           // numLigands * populationSizePerLigand
    uint32_t    generation;
    uint32_t    localSearchSteps;
    float       mutationRate;
    float       crossoverRate;
    float       translationStep;
    float       rotationStep;
    float       torsionStep;
    float       mcTemperature;
    uint32_t    _pad0;
};

// ============================================================================
// Drusina extended scoring types
// ============================================================================

// Pre-computed protein aromatic ring descriptor (fixed during docking)
struct ProteinRingGPU {
    simd_float3 centroid;
    float       _pad0;
    simd_float3 normal;
    float       _pad1;
};

// Ligand aromatic ring (atom indices for dynamic centroid/normal from transformed positions)
struct LigandRingGPU {
    int32_t     atomIndices[6];  // indices into DockLigandAtom array
    int32_t     numAtoms;        // 5 or 6
    int32_t     _pad;
};

// Halogen bond info (maps ligand halogen to its bonded carbon for σ-hole angle check)
struct HalogenBondInfo {
    int32_t     halogenAtomIndex;
    int32_t     carbonAtomIndex;
    int32_t     elementType;  // 0=F, 1=Cl, 2=Br, 3=I (for element-specific distances)
    int32_t     _pad;
};

// Chalcogen bond info (maps ligand sulfur to its bonded carbon for σ-hole angle check)
struct ChalcogenBondInfo {
    int32_t     sulfurAtomIndex;
    int32_t     carbonAtomIndex;
};

// Protein backbone amide plane (for amide-π stacking with ligand aromatics)
struct ProteinAmideGPU {
    simd_float3 centroid;     // center of C-O-N triangle
    float       _pad0;
    simd_float3 normal;       // normal to the amide plane
    float       _pad1;
};

// Protein charged group for group-based salt bridge scoring (Donald 2011, Bissantz 2010)
// One entry per functional group: Arg guanidinium, Lys NH3+, His+ imidazolium, Asp/Glu carboxylate
struct SaltBridgeGroupGPU {
    simd_float3 centroid;       // geometric center of charged atoms
    int32_t     chargeSign;     // +1 or -1
    float       burialFactor;   // 0.0 (exposed) to 1.0 (buried), from neighbor count proxy
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

// Protein chalcogen for bidirectional chalcogen bond scoring (Met SD, Cys SG)
struct ProteinChalcogenGPU {
    simd_float3 position;     // S atom position
    float       _pad0;
    simd_float3 bondedCDir;   // normalize(C - S): direction from S to bonded C (σ-hole is opposite)
    float       _pad1;
};

// Torsion strain info for amide/conjugated planarity penalty
struct TorsionStrainInfo {
    int32_t     atom0, atom1, atom2, atom3; // dihedral quad (i-j-k-l)
    int32_t     strainType;    // 0=none, 1=amide, 2=conjugated_sp2
    float       forceConstant; // penalty stiffness (default 5.0 for amide)
    float       _pad0;
    float       _pad1;
};

// Drusina scoring parameters
struct DrusinaParams {
    uint32_t    numProteinRings;
    uint32_t    numLigandRings;
    uint32_t    numProteinCations;
    uint32_t    numHalogens;
    float       wPiPi;           // π-π stacking weight
    float       wPiCation;       // π-cation weight
    float       wHalogenBond;    // halogen bond weight
    float       wMetalCoord;     // metal coordination weight
    // Extended interactions (salt bridge, amide-π, chalcogen bond)
    uint32_t    numProteinAmides;
    uint32_t    numChalcogens;
    float       wSaltBridge;     // salt bridge weight (per charged-group pair, not per-atom)
    float       wAmideStack;     // amide-π stacking weight
    float       wChalcogenBond;  // chalcogen bond weight
    uint32_t    numSaltBridgeGroups; // number of protein charged groups
    // New scoring terms
    float       wCoulomb;        // screened Coulomb weight (ε=4r dielectric)
    float       wCHPi;           // CH-π interaction weight
    float       wCooperativity;  // cooperativity bonus per extra interaction
    float       wTorsionStrain;  // torsion planarity penalty (positive = penalty)
    uint32_t    numProteinChalcogens; // protein S atoms (Met/Cys) for bidirectional scoring
    uint32_t    numTorsionStrains;    // amide/conjugated torsion count
    uint32_t    _padDP0;
    uint32_t    _padDP1;
};

// ============================================================================
// DruseAF ML scoring types
// ============================================================================

// Parameters for the DruseAF neural network scoring kernel
struct DruseAFParams {
    uint32_t    numProteinAtoms;   // actual P (<=256)
    uint32_t    numLigandAtoms;    // actual L (<=64)
    uint32_t    hiddenDim;         // 256
    uint32_t    numHeads;          // 4
    uint32_t    headDim;           // 64 (hiddenDim / numHeads)
    uint32_t    rbfBins;           // 50
    float       rbfGamma;          // 10.0
    float       rbfSpacing;        // 10.0 / 49.0 (matches torch.linspace(0, 10, 50))
    uint32_t    numCrossAttnLayers; // 3
    uint32_t    numWeightTensors;  // 56
    uint32_t    _pad0;
    uint32_t    _pad1;
};

// Weight tensor offset entry (byte offset + element count into packed weight buffer)
struct DruseAFWeightEntry {
    uint32_t    offset;    // float offset into weight buffer
    uint32_t    count;     // number of float elements
};

// ============================================================================
// PIGNet2 — Physics-Informed GNN scoring
// ============================================================================

#define PIG_DIM            128u   // hidden dimension
#define PIG_FEAT_DIM        47u   // input feature dimension (atom_to_features)
#define PIG_MAX_PROT       256u   // max protein heavy atoms in pocket
#define PIG_MAX_LIG         64u   // max ligand heavy atoms
#define PIG_MAX_TOTAL      320u   // PIG_MAX_PROT + PIG_MAX_LIG
#define PIG_MAX_INTRA_PROT 2048u  // max intramolecular edges (protein, bidirectional)
#define PIG_MAX_INTRA_LIG   512u  // max intramolecular edges (ligand, bidirectional)
#define PIG_MAX_INTER      8192u  // max intermolecular edges per pose (bidirectional)
#define PIG_N_GNN            3u   // GatedGAT / InteractionNet layers
#define PIG_INTER_CUTOFF   5.0f   // intermolecular edge distance cutoff (Angstroms)
#define PIG_INTERACT_MIN   0.5f   // min distance for pairwise interactions
#define PIG_VDW_EPS_LO     0.0178f
#define PIG_VDW_EPS_HI     0.0356f
#define PIG_DEV_VDW_COEFF  0.2f
#define PIG_VDW_CLAMP     100.0f
#define PIG_VDW_WIDTH_LO    1.0f   // Morse width scale range
#define PIG_VDW_WIDTH_HI    2.0f
#define PIG_SHORT_RANGE_A   2.1f   // Morse short-range width
#define PIG_HBOND_C1      (-0.7f)
#define PIG_HBOND_C2        0.0f
#define PIG_METAL_C1      (-0.7f)
#define PIG_METAL_C2        0.0f
#define PIG_HYDRO_C1        0.5f
#define PIG_HYDRO_C2        1.5f

struct PIGNet2Params {
    uint32_t    numProteinAtoms;      // P (<=256)
    uint32_t    numLigandAtoms;       // L (<=64)
    uint32_t    numProtIntraEdges;    // protein covalent edge pairs (bidirectional)
    uint32_t    numLigIntraEdges;     // ligand covalent edge pairs (bidirectional)
    uint32_t    numRotatableBonds;    // for rotor penalty
    uint32_t    numWeightTensors;     // 61 (PIGNet2-Morse)
    uint32_t    _pad0;
    uint32_t    _pad1;
};

// Per-atom auxiliary data for PIGNet2 physics layer
struct PIGNet2AtomAux {
    float       vdwRadius;
    uint32_t    flags;         // bit0: is_metal, bit1: h_donor, bit2: h_acceptor, bit3: hydrophobic
    float       formalCharge;
    float       _pad;
};

// PIGNet2 edge in CSR-like format: packed (source, target) uint16 pairs
struct PIGNet2Edge {
    uint16_t    src;
    uint16_t    dst;
};

// PIGNet2 setup buffer layout offsets (in floats, relative to buffer start)
// After setup kernel: stores protein embeddings + cached InteractionNet projections
// PROT_EMBED:      [P × PIG_DIM]       — protein features after 3× intra-GatedGAT
// INTER_W2X[0-2]:  [P × PIG_DIM] × 3   — cached W2*x_prot for each InteractionNet layer
// Total: 4 × P × PIG_DIM floats
#define PIG_SETUP_PROT_EMBED    0u
// Offsets computed at runtime based on actual P

// Parameters for the DruseAF v4 Pairwise Geometric Network (PGN)
struct DruseAFv4Params {
    uint32_t    numProteinAtoms;   // actual P (<=256)
    uint32_t    numLigandAtoms;    // actual L (<=64)
    uint32_t    numWeightTensors;  // 56
    uint32_t    numPoses;          // population size (for score kernel dispatch)
};

// ============================================================================
// Parallel Tempering / Replica Exchange types
// ============================================================================

#define MAX_REPLICAS 16u

struct ReplicaParams {
    uint32_t    numReplicas;            // K temperature replicas (max 16)
    uint32_t    populationPerReplica;   // poses per replica
    uint32_t    totalPoses;             // numReplicas * populationPerReplica
    uint32_t    swapGeneration;         // current generation (for swap timing)
    float       temperatures[MAX_REPLICAS]; // T_1 < T_2 < ... < T_K (kcal/mol)
};

// ============================================================================
// Fragment-Based Docking types
// ============================================================================

#define MAX_FRAGMENT_ATOMS  64u
#define MAX_FRAGMENTS       16u
#define MAX_BEAM_WIDTH     256u

// One atom within a rigid fragment
struct FragmentAtom {
    simd_float3 position;       // position relative to fragment centroid
    float       vdwRadius;
    float       charge;
    int32_t     vinaType;       // VinaAtomType enum
    int32_t     globalAtomIndex; // index in full ligand DockLigandAtom array
    float       _pad0;
};

// Definition of one rigid fragment in the decomposition
struct FragmentDef {
    uint32_t    atomStart;          // offset into FragmentAtom array
    uint32_t    atomCount;          // number of atoms in this fragment
    uint32_t    connectingTorsionIdx; // torsion edge index connecting to parent (-1 for anchor)
    int32_t     parentFragmentIdx;  // -1 for anchor fragment
    simd_float3 centroid;           // fragment centroid in ligand frame
    float       _pad0;
};

// A candidate placement for a fragment (beam search node)
struct FragmentPlacement {
    simd_float3 translation;        // world-space translation
    float       energy;             // cumulative partial energy
    simd_float4 rotation;           // quaternion for anchor; identity for grown fragments
    float       connectingTorsion;  // torsion angle at connecting bond
    int32_t     parentPlacementIdx; // index of parent in previous beam (-1 for anchor)
    int32_t     fragmentIdx;        // which fragment this placement is for
    int32_t     valid;              // 1 if this placement is active, 0 if pruned
};

// Parameters for fragment docking GPU kernels
struct FragmentSearchParams {
    uint32_t    numFragments;
    uint32_t    beamWidth;          // max surviving placements per level
    uint32_t    currentFragment;    // which fragment we're placing (0 = anchor)
    uint32_t    numPlacements;      // current number of active placements
    uint32_t    numAnchorSamples;   // initial anchor samples
    uint32_t    torsionSamples;     // torsion angle subdivisions per growth step
    float       pruneThreshold;     // kcal/mol above best for pruning
    uint32_t    numLigandAtoms;     // total atoms in full ligand
};

// ============================================================================
// Diffusion-Guided Docking types
// ============================================================================

struct DiffusionParams {
    uint32_t    numPoses;           // number of parallel diffusion trajectories
    uint32_t    currentStep;        // reverse diffusion step (T → 0)
    uint32_t    totalSteps;         // total denoising steps
    float       noiseScale;         // sigma_t for current step
    float       guidanceScale;      // DruseAF attention gradient multiplier
    uint32_t    numLigandAtoms;
    uint32_t    numTorsions;
    float       translationNoise;   // noise magnitude for translation
    float       rotationNoise;      // noise magnitude for rotation
    float       torsionNoise;       // noise magnitude for torsions
    uint32_t    _pad0;
    uint32_t    _pad1;
};

// Per-ligand-atom attention gradient from DruseAF (output of druseAFScoreWithGradient)
struct AttentionGradient {
    simd_float3 pullDirection;      // attention-weighted protein centroid minus ligand atom pos
    float       pullMagnitude;      // norm of raw attention-weighted pull
};

// ============================================================================
// Pharmacophore constraint types
// ============================================================================

#define MAX_PHARMACOPHORE_CONSTRAINTS 16u

// Constraint interaction category (determines compatible ligand atom matching)
enum PharmacophoreInteractionType {
    PHARMA_HBOND_DONOR    = 0,   // receptor wants a donor ligand atom nearby
    PHARMA_HBOND_ACCEPTOR = 1,   // receptor wants an acceptor ligand atom nearby
    PHARMA_SALT_BRIDGE    = 2,   // charged interaction required
    PHARMA_PI_STACK       = 3,   // aromatic ring required
    PHARMA_HALOGEN        = 4,   // halogen bond required
    PHARMA_METAL_COORD    = 5,   // metal coordination (ligand O/N/S near metal)
    PHARMA_HYDROPHOBIC    = 6,   // hydrophobic contact required
};

// A single pharmacophore constraint point.
// If multiple atoms share a groupID, the constraint is satisfied if ANY
// atom in the group has a compatible ligand atom within threshold.
struct PharmacophoreConstraint {
    simd_float3 position;           // 3D target position (protein atom coords)
    float       distanceThreshold;  // satisfaction cutoff (default 3.5 Å)
    uint32_t    compatibleVinaTypes; // bitmask: bit N set if VinaAtomType(N) is compatible
    float       strength;           // penalty scale (kcal/mol/Å²). soft=5.0, hard=1000.0
    uint16_t    groupID;            // constraints sharing groupID form OR-group
    uint16_t    constraintType;     // PharmacophoreInteractionType enum value
    int32_t     ligandAtomIndex;    // ≥0: ligand-side (only this atom checked), -1: any compatible
    float       _pad0;              // pad to 48 bytes (16-byte aligned)
};

// Global pharmacophore parameters
struct PharmacophoreParams {
    uint32_t    numConstraints;     // number of active PharmacophoreConstraint entries
    uint32_t    numGroups;          // number of distinct groupIDs
    float       globalScale;        // overall multiplier (1.0 default, 0 to disable)
    uint32_t    _pad0;
};

// ============================================================================
// Pocket detection compute types
// ============================================================================

struct PocketAtomGPU {
    simd_float3 position;
    float       vdwRadius;
};

struct PocketGridParams {
    simd_float3 origin;
    float       spacing;
    simd_uint3  dims;
    uint32_t    totalPoints;
    float       minProbeDist;
    float       maxProbeDist;
    uint32_t    numAtoms;
    float       rayStep;
    float       rayMaxDist;
    uint32_t    probeCount;
    uint32_t    _pad0;
};

struct PocketProbe {
    simd_float3 position;
    float       buriedness;
};

// ============================================================================
// Interaction detection compute types
// ============================================================================

// Element/property flags for GPU interaction detection
#define IDET_FLAG_N         (1u << 0)
#define IDET_FLAG_O         (1u << 1)
#define IDET_FLAG_S         (1u << 2)
#define IDET_FLAG_C         (1u << 3)
#define IDET_FLAG_F         (1u << 4)
#define IDET_FLAG_CL        (1u << 5)
#define IDET_FLAG_BR        (1u << 6)
#define IDET_FLAG_METAL     (1u << 7)   // Fe, Zn, Ca, Mg, Mn, Cu
#define IDET_FLAG_POS_RES   (1u << 8)   // positive residue atoms (NZ, NH1, NH2, NE)
#define IDET_FLAG_NEG_RES   (1u << 9)   // negative residue atoms (OD1, OD2, OE1, OE2)
#define IDET_FLAG_HALOGEN   (1u << 10)  // F, Cl, or Br

// Packed atom for GPU interaction detection
struct InteractionAtomGPU {
    simd_float3 position;
    uint32_t    flags;          // element + property bit flags
    int32_t     formalCharge;
    uint32_t    _pad0;
    uint32_t    _pad1;
    uint32_t    _pad2;
};

// GPU-detected interaction result
struct GPUInteraction {
    uint32_t    ligandAtomIndex;
    uint32_t    proteinAtomIndex;
    uint32_t    type;           // MolecularInteraction.InteractionType raw value
    float       distance;
    simd_float3 ligandPosition;
    simd_float3 proteinPosition;
};

// Parameters for interaction detection kernel
struct InteractionDetectParams {
    uint32_t numLigandAtoms;
    uint32_t numProteinAtoms;
    uint32_t maxInteractions;   // output buffer capacity
    uint32_t _pad0;
};

// ============================================================================
// ML Feature compute types
// ============================================================================

struct RBFParams {
    uint32_t nProt;
    uint32_t nLig;
    uint32_t numBins;       // 50
    float    gamma;         // 10.0
    float    binSpacing;    // 10.0 / 49.0 (torch.linspace(0, 10, 50))
    uint32_t _pad0;
};

// Pairwise RMSD compute types
struct RMSDParams {
    uint32_t numPoses;
    uint32_t numAtoms;
    uint32_t _pad0;
    uint32_t _pad1;
};


// ============================================================================
// H-Bond Network Scoring types (Phase 4)
// ============================================================================

// An atom in the H-bond scoring environment (protein background atoms)
struct HBondEnvAtom {
    simd_float3 position;
    float       vdwRadius;
    uint32_t    flags;          // HBNET_FLAG_* below
    int32_t     formalCharge;
    uint32_t    _pad0;
    uint32_t    _pad1;
};

// H-bond network scoring flags
enum {
    HBNET_FLAG_DONOR    = (1u << 0),   // Can donate H-bond (N-H, O-H)
    HBNET_FLAG_ACCEPTOR = (1u << 1),   // Can accept H-bond (N, O lone pair)
    HBNET_FLAG_HYDROGEN = (1u << 2),   // This is a hydrogen atom
    HBNET_FLAG_CHARGED  = (1u << 3),   // Atom belongs to a charged group
};

// One candidate atom position within a group state
struct HBondCandidateAtom {
    simd_float3 position;
    float       vdwRadius;
    uint32_t    flags;          // HBNET_FLAG_*
    int32_t     formalCharge;
    uint32_t    _pad0;
    uint32_t    _pad1;
};

// Parameters for the scoring kernel
struct HBondScoringParams {
    uint32_t numCandidates;       // Total candidate atoms across all group-state combos
    uint32_t numEnvAtoms;         // Number of environment (background) atoms
    float    bumpWeight;          // 10.0 (Reduce default)
    float    hbondWeight;         // 4.0  (Reduce default)
    float    minRegHBGap;         // 0.6 Å
    float    minChargedHBGap;     // 0.8 Å
    float    badBumpGapCut;       // 0.4 Å
    float    gapScale;            // 0.25 Å
};

// Output: per-candidate-atom score
struct HBondAtomScore {
    float totalScore;
    float bumpScore;
    float hbondScore;
    float contactScore;
    uint32_t hasBadBump;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

// ============================================================================
// MARK: - GFN2-xTB GPU Types
// ============================================================================

/// Parameters for GPU-accelerated coordination number.
struct GFN2CNParams {
    uint32_t atomCount;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

/// Per-atom data for CN computation.
struct GFN2CNAtom {
    simd_float3 position;        // Bohr
    float       covRadius;       // covalent radius (Bohr)
};

/// Parameters for GPU-accelerated Born radii computation.
struct GFN2BornParams {
    uint32_t atomCount;
    float    probeRadius;        // Bohr
    float    bornOffset;         // Bohr
    float    bornScale;          // OBC descreening factor
};

/// Per-atom data for Born radii GPU kernel.
struct GFN2BornAtom {
    simd_float3 position;        // Bohr
    float       vdwRadius;       // Bohr (Bondi)
};

/// Parameters for GPU-accelerated D4 dispersion.
struct GFN2DispParams {
    uint32_t atomCount;
    float    s6;
    float    s8;
    float    a1;
    float    a2_bohr;           // a2 in Bohr
    uint32_t computeGrad;       // 0 = energy only, 1 = energy + gradient
    uint32_t _pad0;
    uint32_t _pad1;
};

/// Per-atom data for D4 dispersion GPU kernel.
struct GFN2DispAtom {
    simd_float3 position;       // Bohr
    float       c6ref;          // reference C6 (Hartree·Bohr⁶)
    float       cn;             // coordination number
    float       cnRef;          // reference CN for D4 weighting
    float       qDipole;        // sqrt(C6_ref) * 2.5, for C8 = 3*C6ij*sqrt(qi*qj)
    float       _pad0;
};

/// Parameters for GPU-accelerated repulsion energy.
struct GFN2RepParams {
    uint32_t atomCount;
    float    kexp;              // exponent (1.5 for GFN2)
    uint32_t computeGrad;       // 0 = energy only, 1 = energy + gradient
    uint32_t _pad0;
};

/// Per-atom data for repulsion GPU kernel.
struct GFN2RepAtom {
    simd_float3 position;       // Bohr
    float       zeff;           // effective nuclear charge
    float       arep;           // repulsion exponent
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

/// Parameters for GPU-accelerated GB solvation energy.
struct GFN2GBParams {
    uint32_t atomCount;
    float    keps;              // (1 - 1/epsilon)
    float    gamma_au;          // surface tension (Hartree/Bohr²)
    uint32_t computeGrad;
};

/// Per-atom data for GB solvation GPU kernel (after Born radii computed).
struct GFN2GBAtom {
    simd_float3 position;       // Bohr
    float       charge;         // negated Mulliken charge (real partial charge)
    float       bornRadius;     // pre-computed Born radius
    float       sasa;           // pre-computed SASA
    float       _pad0;
    float       _pad1;
};

/// Per-atom data for CN gradient propagation kernel.
struct GFN2CNGradAtom {
    simd_float3 position;       // Bohr
    float       covRadius;      // covalent radius (Bohr)
    float       dEdCN;          // dE/dCN_i chain rule factor
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

// ============================================================================
// MARK: - Loop Refinement GPU Types
// ============================================================================

/// Per-atom data for loop refinement kernels.
struct LoopRefineAtom {
    simd_float3 position;        // Angstrom
    float       mass;            // Da (0 = frozen)
    float       sigma;           // LJ sigma (Angstrom)
    float       epsilon;         // LJ epsilon (kcal/mol)
    uint32_t    isLoop;          // 1 = loop atom (free), 0 = fixed
    float       _pad0;
};

/// Bond definition for harmonic bond force.
struct LoopRefineBond {
    uint32_t    atom1;
    uint32_t    atom2;
    float       length;          // equilibrium length (Angstrom)
    float       k;               // force constant (kcal/mol/A^2)
};

/// Angle definition for harmonic angle force.
struct LoopRefineAngle {
    uint32_t    atom1;
    uint32_t    atom2;           // central atom
    uint32_t    atom3;
    float       angle;           // equilibrium angle (radians)
    float       k;               // force constant (kcal/mol/rad^2)
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

/// Torsion definition for periodic torsion force.
struct LoopRefineTorsion {
    uint32_t    atom1;
    uint32_t    atom2;
    uint32_t    atom3;
    uint32_t    atom4;
    float       phase;           // phase (radians)
    float       k;               // force constant (kcal/mol)
    uint32_t    periodicity;
    float       _pad0;
};

/// Parameters for loop refinement kernels.
struct LoopRefineParams {
    uint32_t    atomCount;
    uint32_t    bondCount;
    uint32_t    angleCount;
    uint32_t    torsionCount;
    float       restraintK;      // positional restraint (kcal/mol/A^2) for non-loop atoms
    float       stericCutoff;    // cutoff for steric evaluation (Angstrom)
    uint32_t    computeGrad;     // 0 = energy only, 1 = energy + gradient
    uint32_t    _pad0;
};

// ============================================================================
// MARK: - FASPR Sidechain Packing GPU Types
// ============================================================================

/// Atom for FASPR energy calculations (VDW parameters baked in).
struct FASPRGPUAtom {
    simd_float3 position;        // Angstrom
    float       radius;          // VDW radius (Angstrom)
    float       depth;           // VDW well depth (kcal/mol)
    uint32_t    atomTypeIdx;     // 1-based FASPR atom type index
    uint32_t    _pad0;
};

/// Parameters for FASPR self-energy kernel (rotamer vs backbone).
struct FASPRSelfParams {
    uint32_t    rotamerCount;         // number of rotamers for this site
    uint32_t    backboneAtomCount;    // total backbone/environment atoms
    uint32_t    maxAtomsPerRotamer;   // max sidechain atoms per rotamer (padded)
    float       vdwRepCut;            // 10.0 kcal/mol cap
    float       dstarMinCut;          // 0.015
    float       dstarMaxCut;          // 1.90
};

/// Parameters for FASPR pair-energy kernel (rotamer vs rotamer).
struct FASPRPairParams {
    uint32_t    rot1Count;
    uint32_t    rot2Count;
    uint32_t    maxAtoms1;            // max sidechain atoms per rotamer (site 1)
    uint32_t    maxAtoms2;            // max sidechain atoms per rotamer (site 2)
    float       vdwRepCut;            // 10.0 kcal/mol
    float       dstarMinCut;
    float       dstarMaxCut;
    uint32_t    _pad0;
};

// ============================================================================
// MARK: - Preparation Minimizer GPU Types
// ============================================================================

/// Atom for preparation minimization with region-based restraints.
struct PrepMinAtom {
    simd_float3 position;        // current position (Angstrom)
    float       sigma;           // LJ sigma (Angstrom), combined for mixing
    float       epsilon;         // LJ epsilon (kcal/mol), combined for mixing
    uint32_t    region;          // 0=backbone, 1=existingSidechain, 2=reconstructed, 3=hydrogen
    float       _pad0;
};

/// Parameters for preparation minimization kernel.
struct PrepMinParams {
    uint32_t    atomCount;
    float       restraintK_backbone;      // kcal/mol/A^2
    float       restraintK_existing;      // kcal/mol/A^2
    float       restraintK_reconstructed; // kcal/mol/A^2 (typically 0)
    float       stericCutoff;             // Angstrom
    uint32_t    computeGrad;              // 0 = energy only, 1 = energy + gradient
    uint32_t    _pad0;
    uint32_t    _pad1;
};

#endif /* ShaderTypes_h */
