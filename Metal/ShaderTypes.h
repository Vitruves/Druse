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
};

// Per-atom instance data for impostor sphere rendering
struct AtomInstance {
    simd_float3 position;
    float       radius;
    simd_float4 color;
    int32_t     atomIndex;
    int32_t     flags;      // bit 0: atom selected, bit 1: residue highlighted
    float       _pad0;
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

// Protein atom for grid map computation (packed for GPU)
struct GridProteinAtom {
    simd_float3 position;
    float       vdwRadius;
    float       charge;       // partial charge
    int32_t     vinaType;     // VinaAtomType enum
    int32_t     _pad0;
    float       _pad1;
};

// Ligand atom for scoring
struct DockLigandAtom {
    simd_float3 position;     // will be transformed by pose (centroid-subtracted)
    float       vdwRadius;
    float       charge;
    int32_t     vinaType;     // VinaAtomType enum
    int32_t     _pad0;
    float       _pad1;
    float       _pad2;
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

#endif /* ShaderTypes_h */
