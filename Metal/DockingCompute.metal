#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Vina Scoring Weights (Trott & Olson, 2010)
// ============================================================================

constant float wGauss1      = -0.035579f;
constant float wGauss2      = -0.005156f;
constant float wRepulsion   = 0.840245f;
constant float wHydrophobic = -0.035069f;
constant float wHBond       = -0.587439f;
constant float wRotEntropy  = 0.05846f;
constant float wPenalty     = 10.0f;
constant uint  kAtomTileSize = 128u;

// Probe radius: weighted average of common drug atom VdW radii (C≈1.7, N≈1.55, O≈1.52).
// Lowered from 1.8 to reduce artificial repulsion at normal contact distances.
constant float kProbeRadius = 1.5f;
constant uint  kMaxVinaXSLookup = 32u;

constant float kVinaXSRadii[19] = {
    1.9f, 1.9f, 1.8f, 1.8f, 1.8f, 1.8f, 1.7f, 1.7f, 1.7f, 1.7f,
    2.0f, 2.1f, 1.5f, 1.8f, 2.0f, 2.2f, 2.2f, 2.3f, 1.2f
};

// ============================================================================
// MARK: - Helper Functions
// ============================================================================

/// Convert thread ID to 3D grid position in Angstroms.
inline float3 gridPosition(uint tid, constant GridParams &params) {
    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint iz = tid / (nx * ny);
    uint iy = (tid - iz * nx * ny) / nx;
    uint ix = tid - iz * nx * ny - iy * nx;
    return params.origin + float3(float(ix), float(iy), float(iz)) * params.spacing;
}

/// Trilinear interpolation on a 3D grid map stored in half precision.
/// Reads as half, promotes to float for interpolation math, returns float.
/// Returns smooth quadratic out-of-grid penalty when the query point lies outside the grid bounds.
inline float trilinearInterpolate(
    device const half *gridMap,
    float3 pos, float3 origin, float spacing, uint3 dims)
{
    float3 gc = (pos - origin) / spacing;

    if (gc.x < 0 || gc.y < 0 || gc.z < 0 ||
        gc.x >= float(dims.x - 1) || gc.y >= float(dims.y - 1) || gc.z >= float(dims.z - 1)) {
        // Smooth quadratic out-of-grid penalty (gives GA gradient back)
        float dx = max(max(-gc.x, gc.x - float(dims.x - 1)), 0.0f);
        float dy = max(max(-gc.y, gc.y - float(dims.y - 1)), 0.0f);
        float dz = max(max(-gc.z, gc.z - float(dims.z - 1)), 0.0f);
        return 100.0f * (dx * dx + dy * dy + dz * dz);
    }

    uint ix = uint(gc.x);  uint iy = uint(gc.y);  uint iz = uint(gc.z);
    float fx = gc.x - float(ix);
    float fy = gc.y - float(iy);
    float fz = gc.z - float(iz);

    uint nx = dims.x;  uint ny = dims.y;

    // Read half-precision grid values and promote to float for interpolation math
    float c000 = float(gridMap[iz * nx * ny + iy * nx + ix]);
    float c100 = float(gridMap[iz * nx * ny + iy * nx + ix + 1]);
    float c010 = float(gridMap[iz * nx * ny + (iy+1) * nx + ix]);
    float c110 = float(gridMap[iz * nx * ny + (iy+1) * nx + ix + 1]);
    float c001 = float(gridMap[(iz+1) * nx * ny + iy * nx + ix]);
    float c101 = float(gridMap[(iz+1) * nx * ny + iy * nx + ix + 1]);
    float c011 = float(gridMap[(iz+1) * nx * ny + (iy+1) * nx + ix]);
    float c111 = float(gridMap[(iz+1) * nx * ny + (iy+1) * nx + ix + 1]);

    return mix(mix(mix(c000, c100, fx), mix(c010, c110, fx), fy),
               mix(mix(c001, c101, fx), mix(c011, c111, fx), fy), fz);
}

/// Rotate vector v by unit quaternion q = (xyz, w).
inline float3 quatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

/// Smooth quadratic penalty for atoms outside the grid (gives gradient back to GA).
/// Uses wPenalty scaling to match Vina's linear distance penalty behavior.
inline float outOfGridPenalty(float3 pos, float3 gridMin, float3 gridMax) {
    float penalty = 0.0f;
    for (int d = 0; d < 3; d++) {
        if (pos[d] < gridMin[d]) {
            float dist = gridMin[d] - pos[d];
            penalty += dist * dist;
        } else if (pos[d] > gridMax[d]) {
            float dist = pos[d] - gridMax[d];
            penalty += dist * dist;
        }
    }
    return penalty;
}

inline float slopeStep(float xBad, float xGood, float x) {
    if (xBad < xGood) {
        if (x <= xBad) return 0.0f;
        if (x >= xGood) return 1.0f;
    } else {
        if (x >= xBad) return 0.0f;
        if (x <= xGood) return 1.0f;
    }
    return (x - xBad) / (xGood - xBad);
}

/// Smooth sigmoid approximation of slopeStep for continuous gradients.
/// k=10 gives <1% deviation from piecewise linear at midpoint and endpoints.
inline float smoothSlopeStep(float xBad, float xGood, float x) {
    float k = 10.0f;
    float t = k * (x - xBad) / (xGood - xBad) - k * 0.5f;
    return 1.0f / (1.0f + exp(-t));
}

/// Derivative of smoothSlopeStep w.r.t. x.
inline float smoothSlopeStepDeriv(float xBad, float xGood, float x) {
    float k = 10.0f;
    float t = k * (x - xBad) / (xGood - xBad) - k * 0.5f;
    float s = 1.0f / (1.0f + exp(-t));
    return k / (xGood - xBad) * s * (1.0f - s);
}

inline bool xsTypeSupported(int xsType) {
    return xsType >= VINA_C_H && xsType <= VINA_MET_D;
}

inline float xsRadius(int xsType) {
    return xsTypeSupported(xsType) ? kVinaXSRadii[xsType] : 0.0f;
}

inline bool xsIsHydrophobic(int xsType) {
    return xsType == VINA_C_H || xsType == VINA_F_H || xsType == VINA_Cl_H ||
           xsType == VINA_Br_H || xsType == VINA_I_H;
}

inline bool xsIsAcceptor(int xsType) {
    return xsType == VINA_N_A || xsType == VINA_N_DA ||
           xsType == VINA_O_A || xsType == VINA_O_DA;
}

inline bool xsIsDonor(int xsType) {
    return xsType == VINA_N_D || xsType == VINA_N_DA ||
           xsType == VINA_O_D || xsType == VINA_O_DA ||
           xsType == VINA_MET_D;
}

inline bool xsHBondPossible(int t1, int t2) {
    return (xsIsDonor(t1) && xsIsAcceptor(t2)) || (xsIsDonor(t2) && xsIsAcceptor(t1));
}

inline float optimalDistanceXS(int t1, int t2) {
    return xsRadius(t1) + xsRadius(t2);
}

/// Decomposed Vina pair energy: steric (gauss1+gauss2+repulsion), hydrophobic, hbond.
struct VinaTerms {
    float steric;
    float hydrophobic;
    float hbond;
    float total;
};

inline VinaTerms vinaPairEnergyDecomposed(int probeType, int proteinType, float r) {
    VinaTerms t = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!xsTypeSupported(probeType) || !xsTypeSupported(proteinType) || r >= 8.0f) {
        return t;
    }

    float d = r - optimalDistanceXS(probeType, proteinType);

    float dOverHalf = d * 2.0f;
    t.steric += wGauss1 * exp(-(dOverHalf * dOverHalf));

    float dMinus3Over2 = (d - 3.0f) * 0.5f;
    t.steric += wGauss2 * exp(-(dMinus3Over2 * dMinus3Over2));

    if (d < 0.0f) {
        t.steric += wRepulsion * d * d;
    }
    if (xsIsHydrophobic(probeType) && xsIsHydrophobic(proteinType)) {
        t.hydrophobic = wHydrophobic * slopeStep(1.5f, 0.5f, d);
    }
    if (xsHBondPossible(probeType, proteinType)) {
        t.hbond = wHBond * slopeStep(0.0f, -0.7f, d);
    }
    t.total = t.steric + t.hydrophobic + t.hbond;
    return t;
}

/// Combined Vina pair energy (backward-compatible, used by grid map computation and intramolecular).
inline float vinaPairEnergy(int probeType, int proteinType, float r) {
    return vinaPairEnergyDecomposed(probeType, proteinType, r).total;
}

inline float sampleTypedAffinityMap(
    device const half *affinityMaps,
    device const int32_t *typeIndexLookup,
    int ligType,
    float3 pos,
    constant GridParams &gp)
{
    if (ligType < 0 || ligType >= int(kMaxVinaXSLookup)) {
        return 0.0f;
    }
    int mapIndex = typeIndexLookup[ligType];
    if (mapIndex < 0) {
        return 0.0f;
    }
    device const half *map = affinityMaps + uint(mapIndex) * gp.totalPoints;
    return trilinearInterpolate(map, pos, gp.origin, gp.spacing, gp.dims);
}

// ============================================================================
// MARK: - Analytical Gradient Helpers
// ============================================================================

/// Trilinear interpolation with analytical gradient w.r.t. position.
/// Reads half-precision grid, promotes to float for math.
/// Returns the interpolated value and writes gradient to `grad`.
inline float trilinearInterpolateWithGrad(
    device const half *gridMap,
    float3 pos, float3 origin, float spacing, uint3 dims,
    thread float3 &grad)
{
    float3 gc = (pos - origin) / spacing;

    if (gc.x < 0 || gc.y < 0 || gc.z < 0 ||
        gc.x >= float(dims.x - 1) || gc.y >= float(dims.y - 1) || gc.z >= float(dims.z - 1)) {
        // Out-of-grid: quadratic penalty with gradient
        float dx = gc.x < 0 ? -gc.x : max(gc.x - float(dims.x - 1), 0.0f);
        float dy = gc.y < 0 ? -gc.y : max(gc.y - float(dims.y - 1), 0.0f);
        float dz = gc.z < 0 ? -gc.z : max(gc.z - float(dims.z - 1), 0.0f);
        float sx = gc.x < 0 ? -1.0f : 1.0f;
        float sy = gc.y < 0 ? -1.0f : 1.0f;
        float sz = gc.z < 0 ? -1.0f : 1.0f;
        grad = float3(200.0f * dx * sx, 200.0f * dy * sy, 200.0f * dz * sz) / spacing;
        return 100.0f * (dx * dx + dy * dy + dz * dz);
    }

    uint ix = uint(gc.x); uint iy = uint(gc.y); uint iz = uint(gc.z);
    float fx = gc.x - float(ix);
    float fy = gc.y - float(iy);
    float fz = gc.z - float(iz);
    uint nx = dims.x; uint ny = dims.y;

    // Read half-precision grid values and promote to float for interpolation math
    float c000 = float(gridMap[iz * nx * ny + iy * nx + ix]);
    float c100 = float(gridMap[iz * nx * ny + iy * nx + ix + 1]);
    float c010 = float(gridMap[iz * nx * ny + (iy+1) * nx + ix]);
    float c110 = float(gridMap[iz * nx * ny + (iy+1) * nx + ix + 1]);
    float c001 = float(gridMap[(iz+1) * nx * ny + iy * nx + ix]);
    float c101 = float(gridMap[(iz+1) * nx * ny + iy * nx + ix + 1]);
    float c011 = float(gridMap[(iz+1) * nx * ny + (iy+1) * nx + ix]);
    float c111 = float(gridMap[(iz+1) * nx * ny + (iy+1) * nx + ix + 1]);

    // Value
    float val = mix(mix(mix(c000, c100, fx), mix(c010, c110, fx), fy),
                    mix(mix(c001, c101, fx), mix(c011, c111, fx), fy), fz);

    // Gradient in grid coordinates, then convert to world via /spacing
    float invS = 1.0f / spacing;
    grad.x = mix(mix(c100-c000, c110-c010, fy), mix(c101-c001, c111-c011, fy), fz) * invS;
    grad.y = mix(mix(c010-c000, c110-c100, fx), mix(c011-c001, c111-c101, fx), fz) * invS;
    grad.z = mix(mix(c001-c000, c101-c100, fx), mix(c011-c010, c111-c110, fx), fy) * invS;

    return val;
}

/// Sample typed affinity map with gradient.
inline float sampleTypedAffinityMapWithGrad(
    device const half *affinityMaps,
    device const int32_t *typeIndexLookup,
    int ligType,
    float3 pos,
    constant GridParams &gp,
    thread float3 &grad)
{
    grad = float3(0);
    if (ligType < 0 || ligType >= int(kMaxVinaXSLookup)) return 0.0f;
    int mapIndex = typeIndexLookup[ligType];
    if (mapIndex < 0) return 0.0f;
    device const half *map = affinityMaps + uint(mapIndex) * gp.totalPoints;
    return trilinearInterpolateWithGrad(map, pos, gp.origin, gp.spacing, gp.dims, grad);
}

/// Vina pair energy with analytical derivative dE/dr.
inline float vinaPairEnergyWithDeriv(int t1, int t2, float r, thread float &dEdr) {
    dEdr = 0.0f;
    if (!xsTypeSupported(t1) || !xsTypeSupported(t2) || r >= 8.0f) return 0.0f;

    float d = r - optimalDistanceXS(t1, t2);
    float E = 0.0f;

    // Gauss1: w1 * exp(-4d²)
    float g1 = exp(-4.0f * d * d);
    E += wGauss1 * g1;
    dEdr += wGauss1 * g1 * (-8.0f * d);

    // Gauss2: w2 * exp(-((d-3)/2)²)
    float dm3h = (d - 3.0f) * 0.5f;
    float g2 = exp(-dm3h * dm3h);
    E += wGauss2 * g2;
    dEdr += wGauss2 * g2 * (-dm3h);  // chain rule: d/dd of -(dm3h²) = -dm3h

    // Repulsion: w_rep * d² if d<0
    if (d < 0.0f) {
        E += wRepulsion * d * d;
        dEdr += 2.0f * wRepulsion * d;
    }

    // Hydrophobic (smooth for continuous gradient)
    if (xsIsHydrophobic(t1) && xsIsHydrophobic(t2)) {
        E += wHydrophobic * smoothSlopeStep(1.5f, 0.5f, d);
        dEdr += wHydrophobic * smoothSlopeStepDeriv(1.5f, 0.5f, d);
    }

    // H-bond (smooth for continuous gradient)
    if (xsHBondPossible(t1, t2)) {
        E += wHBond * smoothSlopeStep(0.0f, -0.7f, d);
        dEdr += wHBond * smoothSlopeStepDeriv(0.0f, -0.7f, d);
    }

    return E;
}

/// Compute intramolecular energy with gradient w.r.t. atom positions.
/// Accumulates dE/dpos for each atom into the gradients array.
/// Uses flat pair list (packed uint32_t: low 16 = atom A, high 16 = atom B).
inline float intramolecularWithGrad(
    thread float3         *positions,
    thread float3         *gradients,
    constant DockLigandAtom *ligandAtoms,
    uint                    nAtoms,
    constant uint32_t      *intraPairs,
    uint                    numPairs)
{
    float total = 0.0f;
    for (uint p = 0; p < numPairs; p++) {
        uint packed = intraPairs[p];
        uint i = packed & 0xFFFFu;
        uint j = packed >> 16u;
        if (i >= nAtoms || j >= nAtoms) continue;

        float3 diff = positions[i] - positions[j];
        float r = length(diff);
        if (r < 1e-6f) continue;

        float dEdr;
        total += vinaPairEnergyWithDeriv(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, r, dEdr);

        float3 gradContrib = dEdr * diff / r;
        gradients[i] += gradContrib;
        gradients[j] -= gradContrib;
    }
    return total;
}

/// Apply torsion rotations to atom positions (in-place on stack array).
/// For each torsion edge: rotate all downstream atoms around the bond axis.
inline void applyTorsions(
    thread float3              *positions,
    uint                        nAtoms,
    device const DockPose      &pose,
    constant TorsionEdge       *torsionEdges,
    constant int32_t           *movingIndices,
    uint                        numTorsions)
{
    for (uint t = 0; t < numTorsions; t++) {
        float angle = pose.torsions[t];
        if (abs(angle) < 1e-6f) continue;

        TorsionEdge edge = torsionEdges[t];
        float3 pivot = positions[edge.atom1];
        float3 axis  = normalize(positions[edge.atom2] - pivot);

        // Rodrigues rotation
        float cosA = cos(angle);
        float sinA = sin(angle);

        for (int i = 0; i < edge.movingCount; i++) {
            int atomIdx = movingIndices[edge.movingStart + i];
            if (atomIdx < 0 || uint(atomIdx) >= nAtoms) continue;

            float3 v = positions[atomIdx] - pivot;
            float3 rotated = v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
            positions[atomIdx] = pivot + rotated;
        }
    }
}

/// Compute Vina-style ligand intramolecular energy over pre-computed non-excluded pairs.
/// Each pair is packed as a uint32_t: low 16 bits = atom A, high 16 bits = atom B.
/// This replaces the O(N²) bitmask loop with a flat O(numPairs) loop.
inline float intramolecularLigandEnergy(
    thread float3              *positions,
    constant DockLigandAtom    *ligandAtoms,
    uint                        nAtoms,
    constant uint32_t          *intraPairs,
    uint                        numPairs)
{
    float total = 0.0f;
    for (uint p = 0; p < numPairs; p++) {
        uint packed = intraPairs[p];
        uint i = packed & 0xFFFFu;
        uint j = packed >> 16u;
        if (i >= nAtoms || j >= nAtoms) continue;
        float r = distance(positions[i], positions[j]);
        total += vinaPairEnergy(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, r);
    }
    return total;
}

/// Transform all ligand atoms by rigid body + torsions, writing to output array.
inline void transformAtoms(
    thread float3              *outPositions,
    constant DockLigandAtom    *ligandAtoms,
    uint                        nAtoms,
    device const DockPose      &pose,
    constant TorsionEdge       *torsionEdges,
    constant int32_t           *movingIndices,
    uint                        numTorsions)
{
    // Step 1: rigid-body transform (quaternion rotation + translation)
    for (uint a = 0; a < nAtoms; a++) {
        outPositions[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    // Step 2: apply torsion rotations (root -> leaf order)
    applyTorsions(outPositions, nAtoms, pose, torsionEdges, movingIndices, numTorsions);
}

// ============================================================================
// MARK: - Grid Map Computation (Vina-style)
// ============================================================================

/// Compute steric grid map (Vina gauss1 + gauss2 + repulsion terms).
/// For each grid point, sums over all protein atoms within 8 A.
/// d = distance - (R_protein + R_probe), where R_probe = 1.8 (average C radius).
/// Weighted sum: wGauss1 * exp(-(d/0.5)^2) + wGauss2 * exp(-((d-3)/2)^2) + wRepulsion * (d<0 ? d^2 : 0)
kernel void computeStericGrid(
    device half                *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float totalE = 0.0f;
    // SoA threadgroup layout eliminates bank conflicts vs AoS GridProteinAtom
    threadgroup float3 tilePositions[kAtomTileSize];
    threadgroup float  tileRadii[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            uint atomIdx = base + lid;
            tilePositions[lid] = proteinAtoms[atomIdx].position;
            tileRadii[lid] = proteinAtoms[atomIdx].vdwRadius;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            float r = distance(gridPos, tilePositions[i]);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);  // avoid singularity

            float d = r - (tileRadii[i] + kProbeRadius);

            // Gauss1: exp(-(d/0.5)^2)
            float d_over_half = d * 2.0f;  // d / 0.5 = d * 2
            float gauss1 = exp(-(d_over_half * d_over_half));

            // Gauss2: exp(-((d-3)/2)^2)
            float d_minus3_over2 = (d - 3.0f) * 0.5f;
            float gauss2 = exp(-(d_minus3_over2 * d_minus3_over2));

            // Repulsion: d^2 if d < 0, else 0
            float repulsion = (d < 0.0f) ? (d * d) : 0.0f;

            totalE += wGauss1 * gauss1 + wGauss2 * gauss2 + wRepulsion * repulsion;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gridMap[tid] = half(clamp(totalE, -100.0f, 100.0f));
}

/// Compute hydrophobic grid map (legacy approximate Vina hydrophobic contact term).
/// Only protein atoms with hydrophobic XS types contribute.
/// d = distance - (R_protein + R_probe).
/// If d < 0.5: value = 1.0; if d > 1.5: value = 0.0; else linear ramp.
/// Weighted by wHydrophobic.
kernel void computeHydrophobicGrid(
    device half                *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float totalE = 0.0f;
    // SoA threadgroup layout eliminates bank conflicts vs AoS GridProteinAtom
    threadgroup float3 tilePositions[kAtomTileSize];
    threadgroup float  tileRadii[kAtomTileSize];
    threadgroup int    tileVinaTypes[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            uint atomIdx = base + lid;
            tilePositions[lid] = proteinAtoms[atomIdx].position;
            tileRadii[lid] = proteinAtoms[atomIdx].vdwRadius;
            tileVinaTypes[lid] = proteinAtoms[atomIdx].vinaType;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            int vt = tileVinaTypes[i];
            if (!xsIsHydrophobic(vt)) continue;

            float r = distance(gridPos, tilePositions[i]);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);

            float d = r - (tileRadii[i] + kProbeRadius);

            // Piecewise linear: 1.0 if d < 0.5, 0.0 if d > 1.5, linear ramp between
            float value;
            if (d < 0.5f) {
                value = 1.0f;
            } else if (d > 1.5f) {
                value = 0.0f;
            } else {
                value = (1.5f - d) / (1.5f - 0.5f);  // linear interpolation
            }

            totalE += wHydrophobic * value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gridMap[tid] = half(clamp(totalE, -100.0f, 100.0f));
}

/// Compute hydrogen bond grid map (legacy approximate Vina H-bond term).
/// Only protein atoms that are donors or acceptors contribute.
/// d = distance - (R_protein + R_probe).
/// If d < -0.7: value = 1.0; if d > 0: value = 0.0; else linear ramp.
/// Weighted by wHBond.
kernel void computeHBondGrid(
    device half                *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float totalE = 0.0f;
    // SoA threadgroup layout eliminates bank conflicts vs AoS GridProteinAtom
    threadgroup float3 tilePositions[kAtomTileSize];
    threadgroup float  tileRadii[kAtomTileSize];
    threadgroup int    tileVinaTypes[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            uint atomIdx = base + lid;
            tilePositions[lid] = proteinAtoms[atomIdx].position;
            tileRadii[lid] = proteinAtoms[atomIdx].vdwRadius;
            tileVinaTypes[lid] = proteinAtoms[atomIdx].vinaType;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            int vt = tileVinaTypes[i];
            bool isHBondCapable = xsIsDonor(vt) || xsIsAcceptor(vt);
            if (!isHBondCapable) continue;

            float r = distance(gridPos, tilePositions[i]);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);

            float d = r - (tileRadii[i] + kProbeRadius);

            // Piecewise linear: 1.0 if d < -0.7, 0.0 if d > 0, linear ramp between
            float value;
            if (d < -0.7f) {
                value = 1.0f;
            } else if (d > 0.0f) {
                value = 0.0f;
            } else {
                value = -d / 0.7f;  // linear interpolation from 0 at d=0 to 1 at d=-0.7
            }

            totalE += wHBond * value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gridMap[tid] = half(clamp(totalE, -100.0f, 100.0f));
}

/// Compute exact AutoDock Vina affinity maps for the requested ligand XS types.
/// Each map stores the full upstream Vina pairwise potential summed over protein atoms.
kernel void computeVinaAffinityMaps(
    device half                *affinityMaps  [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms  [[buffer(1)]],
    constant GridParams        &params        [[buffer(2)]],
    constant int32_t           *affinityTypes [[buffer(3)]],
    uint                        tid           [[thread_position_in_grid]],
    uint                        lid           [[thread_index_in_threadgroup]])
{
    uint totalEntries = params.totalPoints * params.numAffinityTypes;
    if (tid >= totalEntries || params.numAffinityTypes == 0) return;

    uint pointIdx = tid % params.totalPoints;
    uint typeIdx = tid / params.totalPoints;
    int probeType = affinityTypes[typeIdx];
    float3 gridPos = gridPosition(pointIdx, params);
    float totalE = 0.0f;
    // SoA threadgroup layout eliminates bank conflicts vs AoS GridProteinAtom
    threadgroup float3 tilePositions[kAtomTileSize];
    threadgroup float  tileRadii[kAtomTileSize];
    threadgroup float  tileCharges[kAtomTileSize];
    threadgroup int    tileVinaTypes[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            uint atomIdx = base + lid;
            tilePositions[lid] = proteinAtoms[atomIdx].position;
            tileRadii[lid] = proteinAtoms[atomIdx].vdwRadius;
            tileCharges[lid] = proteinAtoms[atomIdx].charge;
            tileVinaTypes[lid] = proteinAtoms[atomIdx].vinaType;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            float r = distance(gridPos, tilePositions[i]);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);
            totalE += vinaPairEnergy(probeType, tileVinaTypes[i], r);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    affinityMaps[typeIdx * params.totalPoints + pointIdx] = half(clamp(totalE, -100.0f, 100.0f));
}

/// Compute electrostatic potential grid using screened Coulomb with distance-dependent dielectric ε=4r.
/// Stores Φ(r) = Σ_protein 332.0 * q_p / (4 * r² ) in half precision.
/// During scoring, E_elec = q_ligand * Φ(r_ligand) via trilinear interpolation.
kernel void computeElectrostaticGrid(
    device half                *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float phi = 0.0f;

    threadgroup float3 tilePositions[kAtomTileSize];
    threadgroup float  tileCharges[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            uint atomIdx = base + lid;
            tilePositions[lid] = proteinAtoms[atomIdx].position;
            tileCharges[lid] = proteinAtoms[atomIdx].charge;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            float r = distance(gridPos, tilePositions[i]);
            if (r > 12.0f) continue;        // screened Coulomb decays fast
            r = max(r, 0.8f);               // clamp singularity
            // E = 332 * q / (ε * r) with ε = 4r → E = 332 * q / (4 * r²)
            phi += 332.0f * tileCharges[i] / (4.0f * r * r);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gridMap[tid] = half(clamp(phi, -50.0f, 50.0f));
}

// ============================================================================
// MARK: - Pocket Detection
// ============================================================================

constant float3 pocketRayDirectionsRaw[26] = {
    float3(-1.0f, -1.0f, -1.0f), float3(-1.0f, -1.0f,  0.0f), float3(-1.0f, -1.0f,  1.0f),
    float3(-1.0f,  0.0f, -1.0f), float3(-1.0f,  0.0f,  0.0f), float3(-1.0f,  0.0f,  1.0f),
    float3(-1.0f,  1.0f, -1.0f), float3(-1.0f,  1.0f,  0.0f), float3(-1.0f,  1.0f,  1.0f),
    float3( 0.0f, -1.0f, -1.0f), float3( 0.0f, -1.0f,  0.0f), float3( 0.0f, -1.0f,  1.0f),
    float3( 0.0f,  0.0f, -1.0f),                              float3( 0.0f,  0.0f,  1.0f),
    float3( 0.0f,  1.0f, -1.0f), float3( 0.0f,  1.0f,  0.0f), float3( 0.0f,  1.0f,  1.0f),
    float3( 1.0f, -1.0f, -1.0f), float3( 1.0f, -1.0f,  0.0f), float3( 1.0f, -1.0f,  1.0f),
    float3( 1.0f,  0.0f, -1.0f), float3( 1.0f,  0.0f,  0.0f), float3( 1.0f,  0.0f,  1.0f),
    float3( 1.0f,  1.0f, -1.0f), float3( 1.0f,  1.0f,  0.0f), float3( 1.0f,  1.0f,  1.0f)
};

inline float samplePocketDistanceGrid(
    device const float         *distanceGrid,
    float3                      pos,
    constant PocketGridParams  &params)
{
    float3 gc = (pos - params.origin) / params.spacing;

    if (gc.x < 0.0f || gc.y < 0.0f || gc.z < 0.0f ||
        gc.x >= float(params.dims.x - 1) ||
        gc.y >= float(params.dims.y - 1) ||
        gc.z >= float(params.dims.z - 1)) {
        return 1e4f;
    }

    uint ix = uint(gc.x);  uint iy = uint(gc.y);  uint iz = uint(gc.z);
    float fx = gc.x - float(ix);
    float fy = gc.y - float(iy);
    float fz = gc.z - float(iz);

    uint nx = params.dims.x;  uint ny = params.dims.y;

    float c000 = distanceGrid[iz * nx * ny + iy * nx + ix];
    float c100 = distanceGrid[iz * nx * ny + iy * nx + ix + 1];
    float c010 = distanceGrid[iz * nx * ny + (iy + 1) * nx + ix];
    float c110 = distanceGrid[iz * nx * ny + (iy + 1) * nx + ix + 1];
    float c001 = distanceGrid[(iz + 1) * nx * ny + iy * nx + ix];
    float c101 = distanceGrid[(iz + 1) * nx * ny + iy * nx + ix + 1];
    float c011 = distanceGrid[(iz + 1) * nx * ny + (iy + 1) * nx + ix];
    float c111 = distanceGrid[(iz + 1) * nx * ny + (iy + 1) * nx + ix + 1];

    return mix(mix(mix(c000, c100, fx), mix(c010, c110, fx), fy),
               mix(mix(c001, c101, fx), mix(c011, c111, fx), fy), fz);
}

kernel void computePocketDistanceGrid(
    device float               *distanceGrid  [[buffer(0)]],
    device uint                *candidateMask [[buffer(1)]],
    constant PocketAtomGPU     *atoms         [[buffer(2)]],
    constant PocketGridParams  &params        [[buffer(3)]],
    uint                        tid           [[thread_position_in_grid]],
    uint                        lid           [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;

    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint iz = tid / (nx * ny);
    uint iy = (tid - iz * nx * ny) / nx;
    uint ix = tid - iz * nx * ny - iy * nx;
    float3 gridPos = params.origin + float3(float(ix), float(iy), float(iz)) * params.spacing;

    float minSurface = 1e10f;
    threadgroup PocketAtomGPU atomTile[kAtomTileSize];

    for (uint base = 0; base < params.numAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numAtoms - base);
        if (lid < tileCount) {
            atomTile[lid] = atoms[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            float3 diff = gridPos - atomTile[i].position;
            float dist = length(diff);
            float surfaceDist = dist - atomTile[i].vdwRadius;
            minSurface = min(minSurface, surfaceDist);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    distanceGrid[tid] = minSurface;
    candidateMask[tid] = (minSurface >= params.minProbeDist && minSurface <= params.maxProbeDist) ? 1u : 0u;
}

kernel void scorePocketBuriedness(
    device PocketProbe         *probes        [[buffer(0)]],
    device const float         *distanceGrid  [[buffer(1)]],
    constant PocketGridParams  &params        [[buffer(2)]],
    uint                        tid           [[thread_position_in_grid]])
{
    if (tid >= params.probeCount) return;

    float3 origin = probes[tid].position;
    uint blockedCount = 0;

    for (uint d = 0; d < 26; d++) {
        float3 dir = normalize(pocketRayDirectionsRaw[d]);
        bool blocked = false;

        for (float step = params.rayStep; step <= params.rayMaxDist; step += params.rayStep) {
            float3 samplePos = origin + dir * step;
            float sdf = samplePocketDistanceGrid(distanceGrid, samplePos, params);
            if (sdf > 9e3f) break;
            if (sdf <= 0.0f) {
                blocked = true;
                break;
            }
        }

        if (blocked) blockedCount += 1u;
    }

    probes[tid].buriedness = float(blockedCount) / 26.0f;
}

// ============================================================================
// MARK: - Pharmacophore Constraint Evaluation
// ============================================================================

/// Compute pharmacophore constraint penalty for a pose.
/// For each constraint group (shared groupID), finds the minimum distance from
/// any compatible ligand atom to any constraint target in that group, then applies
/// a smooth quadratic penalty if the distance exceeds the threshold.
/// Group OR-logic: a group is satisfied if ANY member constraint has a compatible
/// ligand atom within its distance threshold.
inline float evaluateConstraintPenalty(
    thread float3               *positions,
    constant DockLigandAtom     *ligandAtoms,
    uint                         nAtoms,
    constant PharmacophoreConstraint *constraints,
    constant PharmacophoreParams &params)
{
    if (params.numConstraints == 0 || params.globalScale < 1e-6f) return 0.0f;

    uint nC = min(params.numConstraints, MAX_PHARMACOPHORE_CONSTRAINTS);
    uint nG = min(params.numGroups, MAX_PHARMACOPHORE_CONSTRAINTS);

    // Per-group: track best (minimum) violation and associated strength
    float groupViolation[MAX_PHARMACOPHORE_CONSTRAINTS];
    float groupStrength[MAX_PHARMACOPHORE_CONSTRAINTS];
    for (uint i = 0; i < nG; i++) {
        groupViolation[i] = 1e6f;
        groupStrength[i] = 0.0f;
    }

    for (uint c = 0; c < nC; c++) {
        PharmacophoreConstraint con = constraints[c];
        uint gid = con.groupID;
        if (gid >= nG) continue;

        float bestDist = 1e6f;

        if (con.ligandAtomIndex >= 0 && uint(con.ligandAtomIndex) < nAtoms) {
            // Ligand-side constraint: only check the specified atom
            bestDist = distance(positions[con.ligandAtomIndex], con.position);
        } else {
            // Receptor-side: find closest compatible ligand atom
            for (uint a = 0; a < nAtoms; a++) {
                uint32_t typeBit = 1u << uint32_t(ligandAtoms[a].vinaType);
                if ((typeBit & con.compatibleVinaTypes) == 0) continue;
                float d = distance(positions[a], con.position);
                bestDist = min(bestDist, d);
            }
        }

        float violation = max(bestDist - con.distanceThreshold, 0.0f);

        // OR-logic within group: keep the minimum violation
        if (violation < groupViolation[gid]) {
            groupViolation[gid] = violation;
            groupStrength[gid] = con.strength;
        }
    }

    // Sum quadratic penalties across all groups
    float totalPenalty = 0.0f;
    for (uint g = 0; g < nG; g++) {
        float v = groupViolation[g];
        if (v > 0.0f && v < 1e5f) {
            totalPenalty += groupStrength[g] * v * v;
        }
    }

    return totalPenalty * params.globalScale;
}

/// Compute constraint penalty WITH per-atom force accumulation for analytical gradients.
/// Returns the total penalty and adds forces (negative gradient) to the forces array.
inline float evaluateConstraintPenaltyWithGrad(
    thread float3               *positions,
    thread float3               *forces,
    constant DockLigandAtom     *ligandAtoms,
    uint                         nAtoms,
    constant PharmacophoreConstraint *constraints,
    constant PharmacophoreParams &params)
{
    if (params.numConstraints == 0 || params.globalScale < 1e-6f) return 0.0f;

    uint nC = min(params.numConstraints, MAX_PHARMACOPHORE_CONSTRAINTS);
    uint nG = min(params.numGroups, MAX_PHARMACOPHORE_CONSTRAINTS);

    // Two-pass: first find best per-group, then compute gradient for the best
    float groupViolation[MAX_PHARMACOPHORE_CONSTRAINTS];
    float groupStrength[MAX_PHARMACOPHORE_CONSTRAINTS];
    uint  groupBestConstraint[MAX_PHARMACOPHORE_CONSTRAINTS];
    uint  groupBestAtom[MAX_PHARMACOPHORE_CONSTRAINTS];

    for (uint i = 0; i < nG; i++) {
        groupViolation[i] = 1e6f;
        groupStrength[i] = 0.0f;
        groupBestConstraint[i] = 0;
        groupBestAtom[i] = 0;
    }

    for (uint c = 0; c < nC; c++) {
        PharmacophoreConstraint con = constraints[c];
        uint gid = con.groupID;
        if (gid >= nG) continue;

        if (con.ligandAtomIndex >= 0 && uint(con.ligandAtomIndex) < nAtoms) {
            uint ai = uint(con.ligandAtomIndex);
            float d = distance(positions[ai], con.position);
            float violation = max(d - con.distanceThreshold, 0.0f);
            if (violation < groupViolation[gid]) {
                groupViolation[gid] = violation;
                groupStrength[gid] = con.strength;
                groupBestConstraint[gid] = c;
                groupBestAtom[gid] = ai;
            }
        } else {
            for (uint a = 0; a < nAtoms; a++) {
                uint32_t typeBit = 1u << uint32_t(ligandAtoms[a].vinaType);
                if ((typeBit & con.compatibleVinaTypes) == 0) continue;
                float d = distance(positions[a], con.position);
                float violation = max(d - con.distanceThreshold, 0.0f);
                if (violation < groupViolation[gid]) {
                    groupViolation[gid] = violation;
                    groupStrength[gid] = con.strength;
                    groupBestConstraint[gid] = c;
                    groupBestAtom[gid] = a;
                }
            }
        }
    }

    float totalPenalty = 0.0f;
    for (uint g = 0; g < nG; g++) {
        float v = groupViolation[g];
        if (v > 0.0f && v < 1e5f) {
            float pen = groupStrength[g] * v * v * params.globalScale;
            totalPenalty += pen;

            // Gradient: d(strength * v^2)/d(pos_a) = 2 * strength * v * (pos_a - con.position) / |...|
            uint ai = groupBestAtom[g];
            uint ci = groupBestConstraint[g];
            float3 diff = positions[ai] - constraints[ci].position;
            float dist = length(diff);
            if (dist > 1e-6f) {
                float3 grad = 2.0f * groupStrength[g] * v * params.globalScale * (diff / dist);
                forces[ai] -= grad;
            }
        }
    }

    return totalPenalty;
}

// ============================================================================
// MARK: - Pose Scoring
// ============================================================================

/// Score all poses using exact Vina XS-type affinity maps with torsional flexibility.
/// One thread per pose. Grid maps are stored in half precision.
kernel void scorePoses(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    device const half          *affinityMaps  [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams    [[buffer(4)]],
    constant GAParams          &gaParams      [[buffer(5)]],
    constant TorsionEdge       *torsionEdges  [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *intraPairs   [[buffer(8)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tid           [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);

    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gridParams
        );
    }

    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;

    // Pharmacophore constraint penalty
    float cPen = evaluateConstraintPenalty(
        positions, ligandAtoms, nA, pharmaConstraints, pharmaParams);

    // Upstream Vina applies the conf-independent torsion term as a smooth divisor:
    // e / (1 + w_rot * N_tors / 5)
    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    float normalizedE = totalIntermolecular * normFactor;

    // Store compatibility fields. The typed affinity map contains the full
    // intermolecular Vina energy before the conf-independent torsion scaling.
    pose.stericEnergy      = totalIntermolecular;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy       = 0.0f;
    pose.torsionPenalty    = normalizedE - totalIntermolecular;
    pose.clashPenalty      = wPenalty * penalty + intraDelta;
    pose.constraintPenalty = cPen;

    // Total Vina score = normalized intermolecular + boundary penalty + internal + constraint
    pose.energy = normalizedE + pose.clashPenalty + cPen;
}

/// Rescore poses against explicit receptor atoms instead of interpolated affinity maps.
/// This mirrors Vina's late non-cache pose rescoring more closely for top basin representatives.
/// Uses SIMD intrinsics: each SIMD group cooperatively scores one pose by distributing
/// the protein atom inner loop across SIMD lanes, then reducing with simd_sum.
/// Dispatch with (populationSize * simdWidth) threads so each SIMD group maps to one pose.
kernel void scorePosesExplicit(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    constant GridProteinAtom   *proteinAtoms  [[buffer(2)]],
    constant GridParams        &gridParams    [[buffer(3)]],
    constant GAParams          &gaParams      [[buffer(4)]],
    constant TorsionEdge       *torsionEdges  [[buffer(5)]],
    constant int32_t           *movingIndices [[buffer(6)]],
    constant uint32_t          *intraPairs    [[buffer(7)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tid           [[thread_position_in_grid]],
    uint                        simdLane      [[thread_index_in_simdgroup]],
    uint                        simdSize      [[threads_per_simdgroup]])
{
    // Each SIMD group handles one pose; lanes split the protein atom loop
    uint poseIdx = tid / simdSize;
    if (poseIdx >= gaParams.populationSize) return;

    device DockPose &pose = poses[poseIdx];
    uint nAtoms = min(gaParams.numLigandAtoms, 128u);
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    transformAtoms(positions, ligandAtoms, nAtoms, pose, torsionEdges, movingIndices, nTorsions);

    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    float totalSteric = 0.0f;
    float totalHydrophobic = 0.0f;
    float totalHBond = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }

        // SIMD-cooperative: distribute protein atoms across lanes, reduce with simd_sum
        float laneSteric = 0.0f;
        float laneHydrophobic = 0.0f;
        float laneHBond = 0.0f;
        for (uint p = simdLane; p < gridParams.numProteinAtoms; p += simdSize) {
            float dist = distance(r, proteinAtoms[p].position);
            if (dist < 8.0f) {
                VinaTerms terms = vinaPairEnergyDecomposed(ligandAtoms[a].vinaType, proteinAtoms[p].vinaType, dist);
                laneSteric += terms.steric;
                laneHydrophobic += terms.hydrophobic;
                laneHBond += terms.hbond;
            }
        }
        // Reduce partial sums across SIMD lanes
        totalSteric += simd_sum(laneSteric);
        totalHydrophobic += simd_sum(laneHydrophobic);
        totalHBond += simd_sum(laneHBond);
    }

    // Only lane 0 writes the final result (all lanes have identical reduced values)
    if (simdLane != 0) return;

    float totalIntermolecular = totalSteric + totalHydrophobic + totalHBond;
    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nAtoms, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;

    float cPen = evaluateConstraintPenalty(
        positions, ligandAtoms, nAtoms, pharmaConstraints, pharmaParams);

    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    float normalizedE = totalIntermolecular * normFactor;

    pose.stericEnergy      = totalSteric;
    pose.hydrophobicEnergy = totalHydrophobic;
    pose.hbondEnergy       = totalHBond;
    pose.torsionPenalty    = normalizedE - totalIntermolecular;
    pose.clashPenalty      = wPenalty * penalty + intraDelta;
    pose.constraintPenalty = cPen;
    pose.energy            = normalizedE + pose.clashPenalty + cPen;
}

// ============================================================================
// MARK: - Drusina Extended Scoring
// ============================================================================

/// Verdonk 2003 block/ramp function: linear interpolation between full score and zero.
/// Returns 1.0 when |x| <= inner, ramps to 0.0 at |x| >= outer, 0 beyond.
/// GPU-friendly (no transcendentals), smooth enough for GA search.
inline float drusinaRamp(float x, float inner, float outer) {
    float ax = abs(x);
    if (ax <= inner) return 1.0f;
    if (ax >= outer) return 0.0f;
    return (outer - ax) / (outer - inner);
}

/// Compute Drusina correction terms:
///   - π-π stacking, π-cation, halogen bond, metal coordination (original)
///   - salt bridge (group-based, Donald 2011), amide-π stacking, chalcogen bond (extended)
///   - screened Coulomb (ε=4r), CH-π, cooperativity, torsion strain (new)
/// Returns total correction energy (kcal/mol, negative = favorable).
///
/// Design principles (Verdonk 2003, Bissantz 2010, Donald 2011):
///   - Linear ramp functions instead of steep Gaussians for smoother GA landscape
///   - Group-based salt bridge scoring (1 contribution per charged-group pair)
///   - Burial-weighted salt bridges (exposed contribute little, Bissantz 2010)
///   - Corrected optimal distances from CSD/PDB statistics
inline float computeDrusinaCorrections(
    thread float3                *positions,
    constant DockLigandAtom      *ligandAtoms,
    uint                          nAtoms,
    constant ProteinRingGPU      *proteinRings,
    constant LigandRingGPU       *ligandRings,
    constant float4              *proteinCations,
    constant DrusinaParams       &params,
    constant GridProteinAtom     *proteinAtoms,
    uint                          numProteinAtoms,
    constant HalogenBondInfo     *halogenInfo,
    constant ProteinAmideGPU     *proteinAmides,
    constant ChalcogenBondInfo   *chalcogenInfo,
    constant SaltBridgeGroupGPU  *saltBridgeGroups,
    device const half            *elecGrid,
    constant GridParams          &gridParams,
    constant ProteinChalcogenGPU *proteinChalcogens,
    constant TorsionStrainInfo   *torsionStrain)
{
    float drusinaE = 0.0f;
    int interactionCount = 0;

    // Pre-compute ligand centroid for spatial gating of protein partners.
    // Protein features beyond ligandCutoff from centroid cannot interact with any
    // ligand atom (max ligand radius ~8Å + max interaction range ~6Å = 14Å).
    float3 ligCentroid = float3(0);
    for (uint a = 0; a < nAtoms; a++) ligCentroid += positions[a];
    ligCentroid /= float(max(nAtoms, 1u));
    const float ligandCutoff = 14.0f;
    const float ligandCutoffSq = ligandCutoff * ligandCutoff;

    // Pre-compute ligand ring centroids and normals for reuse (π-π + amide-π)
    const uint MAX_LIG_RINGS = 8u;
    float3 ligRingCentroid[MAX_LIG_RINGS];
    float3 ligRingNormal[MAX_LIG_RINGS];
    uint numValidLigRings = 0;

    for (uint lr = 0; lr < min(params.numLigandRings, MAX_LIG_RINGS); lr++) {
        LigandRingGPU ring = ligandRings[lr];
        int nR = min(ring.numAtoms, 6);
        if (nR < 3) continue;

        float3 centroid = float3(0);
        int valid = 0;
        for (int i = 0; i < nR; i++) {
            int idx = ring.atomIndices[i];
            if (idx >= 0 && uint(idx) < nAtoms) {
                centroid += positions[idx];
                valid++;
            }
        }
        if (valid < 3) continue;
        centroid /= float(valid);

        int i0 = ring.atomIndices[0], i1 = ring.atomIndices[1], i2 = ring.atomIndices[2];
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            uint(i0) >= nAtoms || uint(i1) >= nAtoms || uint(i2) >= nAtoms) continue;
        float3 v1 = positions[i1] - positions[i0];
        float3 v2 = positions[i2] - positions[i0];
        float3 norm = cross(v1, v2);
        float nLen = length(norm);
        if (nLen < 1e-6f) continue;
        norm /= nLen;

        ligRingCentroid[numValidLigRings] = centroid;
        ligRingNormal[numValidLigRings] = norm;
        numValidLigRings++;
    }

    // ---- π-π stacking (ligand rings vs protein rings) ----
    // Bissantz 2010: parallel displaced 3.4-3.6 Å, T-shaped 4.5-5.5 Å
    // Using ramp functions for smooth landscape (Verdonk 2003)
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];

        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            if (distance_squared(ligCentroid, proteinRings[pr].centroid) > ligandCutoffSq) continue;
            float d = distance(centroid, proteinRings[pr].centroid);
            if (d < 3.2f || d > 5.8f) continue;
            float dotN = abs(dot(norm, proteinRings[pr].normal));

            // Face-to-face stacking (parallel, |dotN| > 0.8)
            // Optimal 3.5 Å (Bissantz 2010 Table 2), ramp from 3.3-4.5 Å
            if (dotN > 0.8f) {
                float dd = d - 3.5f;
                drusinaE += params.wPiPi * dotN * drusinaRamp(dd, 0.3f, 1.0f);
                interactionCount++;
            }
            // Edge-to-face / T-shaped (perpendicular, |dotN| < 0.4)
            // Optimal 4.8 Å, ramp from 4.2-5.5 Å (Bissantz 2010)
            else if (dotN < 0.4f) {
                float dd = d - 4.8f;
                float perpFactor = 1.0f - dotN;  // stronger when more perpendicular
                drusinaE += params.wPiPi * 0.6f * perpFactor * drusinaRamp(dd, 0.4f, 1.0f);
                interactionCount++;
            }
        }
    }

    // ---- π-cation: ligand ring vs protein cations ----
    // Bissantz 2010: cation 3.4-4.0 Å above ring, perpendicular approach preferred
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];

        for (uint pc = 0; pc < params.numProteinCations; pc++) {
            float3 cPos = proteinCations[pc].xyz;
            if (distance_squared(ligCentroid, cPos) > ligandCutoffSq) continue;
            float d = distance(centroid, cPos);
            if (d > 5.5f) continue;
            float3 toAtom = normalize(cPos - centroid);
            float cosA = abs(dot(toAtom, norm));
            // Require above-plane approach (cosA > 0.5 = within 60° of normal)
            if (cosA > 0.5f) {
                float dd = d - 3.8f;
                drusinaE += params.wPiCation * cosA * drusinaRamp(dd, 0.4f, 1.2f);
                interactionCount++;
            }
        }
    }

    // ---- π-cation: protein rings vs ligand cations ----
    for (uint a = 0; a < nAtoms; a++) {
        if (ligandAtoms[a].formalCharge <= 0) continue;
        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            if (distance_squared(ligCentroid, proteinRings[pr].centroid) > ligandCutoffSq) continue;
            float d = distance(positions[a], proteinRings[pr].centroid);
            if (d > 5.5f) continue;
            float3 toAtom = normalize(positions[a] - proteinRings[pr].centroid);
            float cosA = abs(dot(toAtom, proteinRings[pr].normal));
            if (cosA > 0.5f) {
                float dd = d - 3.8f;
                drusinaE += params.wPiCation * cosA * drusinaRamp(dd, 0.4f, 1.2f);
                interactionCount++;
            }
        }
    }

    // ---- Salt bridge: group-based scoring (Donald 2011, Bissantz 2010) ----
    // One contribution per (ligand charged atom, protein charged group) pair.
    // Optimal N-O distance 2.8 Å (Bissantz 2010 Table 2, CSD median).
    // Ramp: full score at 2.5-3.1 Å, fading to zero at 4.0 Å.
    // Burial-weighted: exposed salt bridges contribute little (Bissantz 2010).
    for (uint a = 0; a < nAtoms; a++) {
        int ligCharge = ligandAtoms[a].formalCharge;
        if (ligCharge == 0) continue;

        for (uint g = 0; g < params.numSaltBridgeGroups; g++) {
            if (distance_squared(ligCentroid, saltBridgeGroups[g].centroid) > ligandCutoffSq) continue;
            // Require opposite charges
            bool isSB = (ligCharge > 0 && saltBridgeGroups[g].chargeSign < 0) ||
                        (ligCharge < 0 && saltBridgeGroups[g].chargeSign > 0);
            if (!isSB) continue;

            float d = distance(positions[a], saltBridgeGroups[g].centroid);
            if (d > 4.0f) continue;

            // Ramp centered at 2.8 Å (Bissantz 2010: CSD median N-O = 2.79 Å)
            float dd = d - 2.8f;
            float distScore = drusinaRamp(dd, 0.3f, 1.2f);

            // Burial dampening (Bissantz 2010: exposed salt bridges ≈ no free energy)
            float burial = saltBridgeGroups[g].burialFactor;

            drusinaE += params.wSaltBridge * distScore * burial;
            interactionCount++;
        }
    }

    // ---- Amide-π stacking: protein backbone amide ↔ ligand aromatic ring ----
    // Bissantz 2010: amide-aryl distance 3.2-3.7 Å between planes
    // Harder 2013: optimal d=3.4 Å interplanar, parallel (|dotN|>0.8)
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];

        for (uint am = 0; am < params.numProteinAmides; am++) {
            if (distance_squared(ligCentroid, proteinAmides[am].centroid) > ligandCutoffSq) continue;
            float d = distance(centroid, proteinAmides[am].centroid);
            if (d < 3.0f || d > 5.2f) continue;

            float dotN = abs(dot(norm, proteinAmides[am].normal));

            // Parallel stacking (Harder 2013: interplanar ≤30°, |dotN| > 0.8)
            if (dotN > 0.8f) {
                float dd = d - 3.6f;
                drusinaE += params.wAmideStack * dotN * drusinaRamp(dd, 0.3f, 0.9f);
                interactionCount++;
            }
            // Tilted/offset stacking (weaker, broader distance range)
            else if (dotN > 0.6f) {
                float dd = d - 4.0f;
                drusinaE += params.wAmideStack * 0.4f * dotN * drusinaRamp(dd, 0.3f, 1.0f);
                interactionCount++;
            }
        }
    }

    // ---- Halogen bonds: C-X...O/N with σ-hole angle check ----
    // Element-specific optimal distances (Scholfield 2013, Bissantz 2010):
    //   F: skip (too weak), Cl: 3.27 Å, Br: 3.24 Å, I: 3.17 Å
    // C-X...A angle preferred near 160-180° (sigma-hole alignment)
    for (uint h = 0; h < params.numHalogens; h++) {
        int hi = halogenInfo[h].halogenAtomIndex;
        int ci = halogenInfo[h].carbonAtomIndex;
        int elemType = halogenInfo[h].elementType;
        if (hi < 0 || ci < 0 || uint(hi) >= nAtoms || uint(ci) >= nAtoms) continue;
        if (elemType == 0) continue;  // F halogen bonds too weak — skip

        // Element-specific optimal distance (CSD statistics, Scholfield 2013)
        float optDist = 3.2f;
        if      (elemType == 1) optDist = 3.27f;  // Cl
        else if (elemType == 2) optDist = 3.24f;  // Br
        else if (elemType == 3) optDist = 3.17f;  // I

        float3 halPos = positions[hi];
        float3 carPos = positions[ci];

        for (uint p = 0; p < numProteinAtoms; p++) {
            int pType = proteinAtoms[p].vinaType;
            bool isAcceptor = (pType == VINA_N_A || pType == VINA_N_DA ||
                               pType == VINA_O_A || pType == VINA_O_DA);
            if (!isAcceptor) continue;

            float d = distance(halPos, proteinAtoms[p].position);
            if (d < 2.5f || d > 4.2f) continue;

            // σ-hole directionality: C-X...A angle should be > 140° (cosine < -0.766)
            float3 xToC = normalize(carPos - halPos);
            float3 xToA = normalize(proteinAtoms[p].position - halPos);
            float cosTheta = dot(xToC, xToA);
            if (cosTheta < -0.766f) {
                float dd = d - optDist;
                float angleFactor = (-cosTheta - 0.766f) / 0.234f;
                drusinaE += params.wHalogenBond * angleFactor * drusinaRamp(dd, 0.3f, 0.8f);
                interactionCount++;
            }
        }
    }

    // ---- Chalcogen bonds: C-S...O/N with σ-hole angle check ----
    // Relaxed from 150° to 140° (Beno 2015, Pascoe 2017: S σ-hole valid at 140-180°)
    // Distance 2.8-4.2 Å, optimal ~3.3 Å (Bissantz 2010). Dual σ-holes for thioethers.
    for (uint ch = 0; ch < params.numChalcogens; ch++) {
        int si = chalcogenInfo[ch].sulfurAtomIndex;
        int ci = chalcogenInfo[ch].carbonAtomIndex;
        if (si < 0 || ci < 0 || uint(si) >= nAtoms || uint(ci) >= nAtoms) continue;
        float3 sPos = positions[si];
        float3 cPos = positions[ci];

        for (uint p = 0; p < numProteinAtoms; p++) {
            int pType = proteinAtoms[p].vinaType;
            bool isAcceptor = (pType == VINA_N_A || pType == VINA_N_DA ||
                               pType == VINA_O_A || pType == VINA_O_DA);
            if (!isAcceptor) continue;

            float d = distance(sPos, proteinAtoms[p].position);
            if (d < 2.8f || d > 4.2f) continue;

            // σ-hole directionality: C-S-A angle > 140° (cos < -0.766)
            float3 sToC = normalize(cPos - sPos);
            float3 sToA = normalize(proteinAtoms[p].position - sPos);
            float cosTheta = dot(sToC, sToA);
            if (cosTheta < -0.766f) {
                float dd = d - 3.3f;
                float angleFactor = (-cosTheta - 0.766f) / 0.234f;
                drusinaE += params.wChalcogenBond * angleFactor * drusinaRamp(dd, 0.3f, 0.8f);
                interactionCount++;
            }
        }
    }

    // ---- Protein chalcogen bonds: protein S (Met/Cys) → ligand N/O ----
    // Bidirectional: protein sulfur σ-hole interacting with ligand acceptors
    for (uint pch = 0; pch < params.numProteinChalcogens; pch++) {
        float3 sPos = proteinChalcogens[pch].position;
        float3 bondedCDir = proteinChalcogens[pch].bondedCDir; // direction S→C

        for (uint a = 0; a < nAtoms; a++) {
            int lt = ligandAtoms[a].vinaType;
            bool isAcceptor = (lt == VINA_N_A || lt == VINA_N_DA ||
                               lt == VINA_O_A || lt == VINA_O_DA);
            if (!isAcceptor) continue;

            float d = distance(positions[a], sPos);
            if (d < 2.8f || d > 4.2f) continue;

            // σ-hole is opposite C-S bond: check S...A aligns with -bondedCDir
            float3 sToA = normalize(positions[a] - sPos);
            float cosTheta = dot(bondedCDir, sToA); // negative when aligned with σ-hole
            if (cosTheta < -0.766f) {
                float dd = d - 3.3f;
                float angleFactor = (-cosTheta - 0.766f) / 0.234f;
                drusinaE += params.wChalcogenBond * angleFactor * drusinaRamp(dd, 0.3f, 0.8f);
                interactionCount++;
            }
        }
    }

    // ---- Enhanced metal coordination ----
    // Verdonk 2003: optimal distance loosened from 2.2 to 2.6 Å for raw PDB structures.
    // We use 2.4 Å optimal with cutoff at 3.5 Å (widened for prepared structure tolerance).
    // Also accept N_D (donor-only nitrogen, e.g. imidazole) as coordinating atom.
    for (uint p = 0; p < numProteinAtoms; p++) {
        if (proteinAtoms[p].vinaType != VINA_MET_D) continue;
        for (uint a = 0; a < nAtoms; a++) {
            int lt = ligandAtoms[a].vinaType;
            bool isCoord = (lt == VINA_N_A || lt == VINA_N_DA || lt == VINA_N_D ||
                            lt == VINA_O_A || lt == VINA_O_DA || lt == VINA_S_P);
            if (!isCoord) continue;
            float d = distance(positions[a], proteinAtoms[p].position);
            if (d > 3.5f) continue;
            float dd = d - 2.4f;
            drusinaE += params.wMetalCoord * drusinaRamp(dd, 0.3f, 1.1f);
            interactionCount++;
        }
    }

    // ---- Screened Coulomb (ε=4r, grid-based) ----
    // Potential grid precomputed from protein charges: Φ(r) = Σ 332*q_p/(4*r²)
    // E_elec = Σ q_ligand * Φ(r_ligand) via trilinear interpolation
    float coulombE = 0.0f;
    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];
        float phi = trilinearInterpolate(elecGrid, r, gridParams.origin,
                                          gridParams.spacing, gridParams.dims);
        if (phi < 99.0f)  // skip out-of-grid penalty values
            coulombE += ligandAtoms[a].charge * phi;
    }
    drusinaE += params.wCoulomb * coulombE;

    // ---- CH-π interactions: ligand aliphatic C → protein aromatic rings ----
    // Weaker individually (~-1 kcal/mol) but very frequent (2-5× per complex).
    // Score C...ring centroid distance (3.5-5.0 Å) without explicit H.
    for (uint a = 0; a < nAtoms; a++) {
        if (ligandAtoms[a].vinaType != VINA_C_H) continue;
        if (ligandAtoms[a].flags & LIGATOM_FLAG_AROMATIC) continue;
        float3 cPos = positions[a];
        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            if (distance_squared(ligCentroid, proteinRings[pr].centroid) > ligandCutoffSq) continue;
            float d = distance(cPos, proteinRings[pr].centroid);
            if (d < 3.5f || d > 5.0f) continue;
            float3 toC = normalize(cPos - proteinRings[pr].centroid);
            float cosA = abs(dot(toC, proteinRings[pr].normal));
            if (cosA > 0.3f) {  // within ~70° of ring normal
                float dd = d - 4.0f;
                drusinaE += params.wCHPi * cosA * drusinaRamp(dd, 0.5f, 1.0f);
                interactionCount++;
            }
        }
    }

    // ---- Torsion strain: amide planarity penalty ----
    // Amide bonds should be planar (φ ≈ 0 or π). Penalize deviation.
    float strainE = 0.0f;
    for (uint t = 0; t < params.numTorsionStrains; t++) {
        TorsionStrainInfo ts = torsionStrain[t];
        if (ts.atom0 < 0 || uint(ts.atom3) >= nAtoms) continue;
        float3 b1 = positions[ts.atom1] - positions[ts.atom0];
        float3 b2 = positions[ts.atom2] - positions[ts.atom1];
        float3 b3 = positions[ts.atom3] - positions[ts.atom2];
        float3 n1 = cross(b1, b2);
        float3 n2 = cross(b2, b3);
        float ln1 = length(n1), ln2 = length(n2);
        if (ln1 < 1e-6f || ln2 < 1e-6f) continue;
        n1 /= ln1; n2 /= ln2;
        float cosPhi = clamp(dot(n1, n2), -1.0f, 1.0f);
        float phi = acos(cosPhi);
        // Amide: should be 0 or π. Minimum deviation from either.
        float dev = min(phi, M_PI_F - phi);
        strainE += ts.forceConstant * dev * dev;
    }
    drusinaE += params.wTorsionStrain * strainE;

    // ---- Cooperativity bonus: reward multiple simultaneous interactions ----
    // Simple count-based: bonus for 2+ scored interaction types
    if (interactionCount > 1) {
        drusinaE += params.wCooperativity * float(interactionCount - 1);
    }

    // Safety cap: Drusina corrections should complement Vina, not overwhelm it.
    // Typical Vina scores are -5 to -15 kcal/mol; cap Drusina at -10 kcal/mol
    // to allow strong metal coordination + salt bridge contributions.
    drusinaE = max(drusinaE, -10.0f);

    return drusinaE;
}

/// Score poses with Vina grid maps + Drusina extended interaction corrections.
/// Used during GA when Drusina scoring is selected.
kernel void scorePosesDrusina(
    device DockPose              *poses           [[buffer(0)]],
    constant DockLigandAtom      *ligandAtoms     [[buffer(1)]],
    device const half            *affinityMaps    [[buffer(2)]],
    device const int32_t         *typeIndexLookup [[buffer(3)]],
    constant GridParams          &gridParams      [[buffer(4)]],
    constant GAParams            &gaParams        [[buffer(5)]],
    constant TorsionEdge         *torsionEdges    [[buffer(6)]],
    constant int32_t             *movingIndices   [[buffer(7)]],
    constant uint32_t            *intraPairs      [[buffer(8)]],
    constant ProteinRingGPU      *proteinRings    [[buffer(9)]],
    constant LigandRingGPU       *ligandRings     [[buffer(10)]],
    constant float4              *proteinCations  [[buffer(11)]],
    constant DrusinaParams       &drusinaParams   [[buffer(12)]],
    constant GridProteinAtom     *proteinAtoms    [[buffer(13)]],
    constant HalogenBondInfo     *halogenInfo     [[buffer(14)]],
    constant ProteinAmideGPU     *proteinAmides   [[buffer(15)]],
    constant ChalcogenBondInfo   *chalcogenInfo   [[buffer(16)]],
    constant SaltBridgeGroupGPU  *saltBridgeGroups [[buffer(17)]],
    device const half            *elecGrid        [[buffer(18)]],
    constant ProteinChalcogenGPU *proteinChalcogens [[buffer(19)]],
    constant TorsionStrainInfo   *torsionStrain   [[buffer(20)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    // --- Base Vina scoring from grid maps ---
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;
    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];
        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) { penalty += oopP; continue; }
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gridParams);
    }

    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;
    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    float normalizedE = totalIntermolecular * normFactor;

    // --- Drusina corrections ---
    float drusinaE = computeDrusinaCorrections(
        positions, ligandAtoms, nA,
        proteinRings, ligandRings, proteinCations, drusinaParams,
        proteinAtoms, gridParams.numProteinAtoms, halogenInfo,
        proteinAmides, chalcogenInfo, saltBridgeGroups,
        elecGrid, gridParams, proteinChalcogens, torsionStrain);

    pose.stericEnergy      = totalIntermolecular;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy       = 0.0f;
    pose.torsionPenalty    = normalizedE - totalIntermolecular;
    pose.clashPenalty      = wPenalty * penalty + intraDelta;
    pose.drusinaCorrection = drusinaE;
    pose.energy = normalizedE + pose.clashPenalty + drusinaE;
}

/// Apply Drusina corrections to already-scored poses (e.g., after explicit Vina rescoring).
kernel void applyDrusinaCorrection(
    device DockPose              *poses           [[buffer(0)]],
    constant DockLigandAtom      *ligandAtoms     [[buffer(1)]],
    constant GAParams            &gaParams        [[buffer(2)]],
    constant TorsionEdge         *torsionEdges    [[buffer(3)]],
    constant int32_t             *movingIndices   [[buffer(4)]],
    constant ProteinRingGPU      *proteinRings    [[buffer(5)]],
    constant LigandRingGPU       *ligandRings     [[buffer(6)]],
    constant float4              *proteinCations  [[buffer(7)]],
    constant DrusinaParams       &drusinaParams   [[buffer(8)]],
    constant GridProteinAtom     *proteinAtoms    [[buffer(9)]],
    constant GridParams          &gridParams      [[buffer(10)]],
    constant HalogenBondInfo     *halogenInfo     [[buffer(11)]],
    constant ProteinAmideGPU     *proteinAmides   [[buffer(12)]],
    constant ChalcogenBondInfo   *chalcogenInfo   [[buffer(13)]],
    constant SaltBridgeGroupGPU  *saltBridgeGroups [[buffer(14)]],
    device const half            *elecGrid        [[buffer(15)]],
    constant ProteinChalcogenGPU *proteinChalcogens [[buffer(16)]],
    constant TorsionStrainInfo   *torsionStrain   [[buffer(17)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    float drusinaE = computeDrusinaCorrections(
        positions, ligandAtoms, nA,
        proteinRings, ligandRings, proteinCations, drusinaParams,
        proteinAtoms, gridParams.numProteinAtoms, halogenInfo,
        proteinAmides, chalcogenInfo, saltBridgeGroups,
        elecGrid, gridParams, proteinChalcogens, torsionStrain);

    pose.drusinaCorrection = drusinaE;
    pose.energy += drusinaE;
}

// ============================================================================
// MARK: - GA Operations
// ============================================================================

/// PCG-style hash for high-quality pseudorandom numbers on GPU.
inline float gpuRandom(uint seed, uint seq) {
    uint state = seed * 747796405u + seq * 2891336453u + 1u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) / float(0xFFFFFFFFu);
}

inline float3 randomInsideUnitSphere(uint seed, uint seqBase) {
    for (uint attempt = 0; attempt < 4; attempt++) {
        float3 p = float3(
            gpuRandom(seed, seqBase + attempt * 3 + 0) * 2.0f - 1.0f,
            gpuRandom(seed, seqBase + attempt * 3 + 1) * 2.0f - 1.0f,
            gpuRandom(seed, seqBase + attempt * 3 + 2) * 2.0f - 1.0f
        );
        float r2 = dot(p, p);
        if (r2 > 1e-6f && r2 <= 1.0f) {
            return p;
        }
    }

    float3 dir = float3(
        gpuRandom(seed, seqBase + 16) * 2.0f - 1.0f,
        gpuRandom(seed, seqBase + 17) * 2.0f - 1.0f,
        gpuRandom(seed, seqBase + 18) * 2.0f - 1.0f
    );
    float len = max(length(dir), 1e-6f);
    float radius = pow(gpuRandom(seed, seqBase + 19), 1.0f / 3.0f);
    return dir / len * radius;
}

kernel void initializeBatchScreenPoses(
    device BatchScreenPose      *poses        [[buffer(0)]],
    constant BatchLigandInfo    *ligands      [[buffer(1)]],
    constant GridParams         &gridParams   [[buffer(2)]],
    constant BatchScreenParams  &params       [[buffer(3)]],
    uint                         tid          [[thread_position_in_grid]])
{
    if (tid >= params.totalPoses) return;

    uint ligandIndex = tid / max(params.posesPerLigand, 1u);
    uint poseIndex = tid % max(params.posesPerLigand, 1u);
    if (ligandIndex >= params.numLigands) return;

    BatchLigandInfo ligand = ligands[ligandIndex];
    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;
    float3 searchSize = halfExtent * 2.0f;
    float minDim = min(searchSize.x, min(searchSize.y, searchSize.z));
    float focusSpread = min(minDim * 0.45f, 10.0f);
    float3 spread = (poseIndex < max(ligand.poseCount * 3u / 5u, 1u))
        ? float3(focusSpread)
        : halfExtent;

    uint seed = params.seed ^ (ligandIndex * 2246822519u) ^ (poseIndex * 3266489917u);
    device BatchScreenPose &pose = poses[tid];
    pose.translation = center + float3(
        (gpuRandom(seed, 0) * 2.0f - 1.0f) * spread.x,
        (gpuRandom(seed, 1) * 2.0f - 1.0f) * spread.y,
        (gpuRandom(seed, 2) * 2.0f - 1.0f) * spread.z
    );

    float u1 = gpuRandom(seed, 3);
    float u2 = gpuRandom(seed, 4) * 2.0f * M_PI_F;
    float u3 = gpuRandom(seed, 5) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    pose.rotation = float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));
    pose.ligandIndex = ligandIndex;
    pose.poseIndex = poseIndex;
    pose.energy = 1e10f;
    pose.stericEnergy = 0.0f;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy = 0.0f;
    pose.clashPenalty = 0.0f;
}

kernel void scoreBatchRigidPoses(
    device BatchScreenPose      *poses         [[buffer(0)]],
    constant BatchLigandInfo    *ligands       [[buffer(1)]],
    constant DockLigandAtom     *ligandAtoms   [[buffer(2)]],
    device const half           *stericGrid    [[buffer(3)]],
    device const half           *hydrophobGrid [[buffer(4)]],
    device const half           *hbondGrid     [[buffer(5)]],
    constant GridParams         &gridParams    [[buffer(6)]],
    constant BatchScreenParams  &params        [[buffer(7)]],
    uint                         tid           [[thread_position_in_grid]])
{
    if (tid >= params.totalPoses) return;

    device BatchScreenPose &pose = poses[tid];
    if (pose.ligandIndex >= params.numLigands) return;

    BatchLigandInfo ligand = ligands[pose.ligandIndex];
    uint nAtoms = min(ligand.atomCount, 128u);

    float totalSteric = 0.0f;
    float totalHydrophobic = 0.0f;
    float totalHBond = 0.0f;
    float penalty = 0.0f;
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    for (uint a = 0; a < nAtoms; a++) {
        DockLigandAtom atom = ligandAtoms[ligand.atomStart + a];
        float3 r = quatRotate(pose.rotation, atom.position) + pose.translation;

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }

        totalSteric += trilinearInterpolate(stericGrid, r, gridParams.origin,
                                            gridParams.spacing, gridParams.dims);

        int ligVinaType = atom.vinaType;
        if (xsIsHydrophobic(ligVinaType)) {
            totalHydrophobic += trilinearInterpolate(hydrophobGrid, r, gridParams.origin,
                                                     gridParams.spacing, gridParams.dims);
        }

        bool ligHBondCapable = xsIsDonor(ligVinaType) || xsIsAcceptor(ligVinaType);
        if (ligHBondCapable) {
            totalHBond += trilinearInterpolate(hbondGrid, r, gridParams.origin,
                                               gridParams.spacing, gridParams.dims);
        }
    }

    pose.stericEnergy = totalSteric;
    pose.hydrophobicEnergy = totalHydrophobic;
    pose.hbondEnergy = totalHBond;
    pose.clashPenalty = wPenalty * penalty;
    pose.energy = totalSteric + totalHydrophobic + totalHBond + pose.clashPenalty;
}

/// Initialize population with conformer-aware seeding.
/// Three tiers to balance exploitation of the RDKit conformer with exploration:
///   Tier 1 (30%): Reference conformer (torsions=0) with positional + rotational diversity
///                  — these poses START from the chemically reasonable RDKit geometry
///   Tier 2 (30%): Reference torsions + small Gaussian noise (±30°) with random placement
///                  — conformational variants near the reference
///   Tier 3 (40%): Fully random torsions for exploration of alternative binding modes
kernel void initializePopulation(
    device DockPose            *poses        [[buffer(0)]],
    constant GridParams        &gridParams   [[buffer(1)]],
    constant GAParams          &gaParams     [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint seed = tid * 747796405u + gaParams.generation * 2891336453u + 67890u;
    uint popSize = gaParams.populationSize;
    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;

    float3 searchSize = halfExtent * 2.0f;
    float minDim = min(searchSize.x, min(searchSize.y, searchSize.z));
    float focusSpread = min(minDim * 0.5f, 12.0f);

    // --- Translation: focused near pocket center for all tiers ---
    float3 spread;
    if (tid < popSize * 8 / 10) {
        spread = float3(focusSpread);
    } else {
        spread = halfExtent;
    }

    poses[tid].translation = center + float3(
        (gpuRandom(seed, 0) * 2.0f - 1.0f) * spread.x,
        (gpuRandom(seed, 1) * 2.0f - 1.0f) * spread.y,
        (gpuRandom(seed, 2) * 2.0f - 1.0f) * spread.z
    );

    // --- Rotation: uniform random quaternion (Shoemake method) for all tiers ---
    float u1 = gpuRandom(seed, 3);
    float u2 = gpuRandom(seed, 4) * 2.0f * M_PI_F;
    float u3 = gpuRandom(seed, 5) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    poses[tid].rotation = float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));

    // --- Torsions: three-tier seeding strategy ---
    uint tier1End = popSize * 3 / 10;   // 30% reference conformer (torsions=0)
    uint tier2End = popSize * 6 / 10;   // 30% near-reference (small noise)

    if (tid < tier1End) {
        // Tier 1: Exact RDKit reference conformer — torsions = 0
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            poses[tid].torsions[t] = 0.0f;
        }
    } else if (tid < tier2End) {
        // Tier 2: Reference torsions + Gaussian noise (σ ≈ 30° = 0.52 rad)
        // Box-Muller for approximate Gaussian from uniform random
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            float u = max(gpuRandom(seed, 40 + t * 2), 1e-6f);
            float v = gpuRandom(seed, 41 + t * 2) * 2.0f * M_PI_F;
            float noise = sqrt(-2.0f * log(u)) * cos(v) * 0.52f;
            poses[tid].torsions[t] = noise;
            if (poses[tid].torsions[t] > M_PI_F) poses[tid].torsions[t] -= 2.0f * M_PI_F;
            if (poses[tid].torsions[t] < -M_PI_F) poses[tid].torsions[t] += 2.0f * M_PI_F;
        }
    } else {
        // Tier 3: Fully random torsions for exploration
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            poses[tid].torsions[t] = gpuRandom(seed, 6 + t) * 2.0f * M_PI_F - M_PI_F;
        }
    }

    poses[tid].numTorsions = int(gaParams.numTorsions);
    poses[tid].generation = 0;
    poses[tid].energy = 1e10f;
    poses[tid].stericEnergy = 0.0f;
    poses[tid].hydrophobicEnergy = 0.0f;
    poses[tid].hbondEnergy = 0.0f;
    poses[tid].torsionPenalty = 0.0f;
    poses[tid].clashPenalty = 0.0f;
    poses[tid].drusinaCorrection = 0.0f;
    poses[tid].constraintPenalty = 0.0f;

    // Initialize chi angles to zero (rigid receptor baseline)
    for (uint c = 0; c < 24; c++) {
        poses[tid].chiAngles[c] = 0.0f;
    }
    poses[tid].numChiAngles = 0;
}

/// GA evolution: tournament selection + arithmetic/SLERP crossover + Gaussian mutation.
/// Includes diversity injection every 10 generations for bottom 5% of population.
kernel void gaEvolve(
    device DockPose            *offspring    [[buffer(0)]],
    device const DockPose      *population   [[buffer(1)]],
    constant GAParams          &gaParams     [[buffer(2)]],
    constant GridParams        &gridParams   [[buffer(3)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint seed = tid * 31337u + gaParams.generation * 99991u;
    uint popSize = gaParams.populationSize;

    // Tournament selection (k=3) for two parents
    uint a1 = uint(gpuRandom(seed, 0) * float(popSize)) % popSize;
    uint b1 = uint(gpuRandom(seed, 1) * float(popSize)) % popSize;
    uint c1 = uint(gpuRandom(seed, 2) * float(popSize)) % popSize;
    uint parentA = a1;
    if (population[b1].energy < population[parentA].energy) parentA = b1;
    if (population[c1].energy < population[parentA].energy) parentA = c1;

    uint a2 = uint(gpuRandom(seed, 3) * float(popSize)) % popSize;
    uint b2 = uint(gpuRandom(seed, 4) * float(popSize)) % popSize;
    uint c2 = uint(gpuRandom(seed, 5) * float(popSize)) % popSize;
    uint parentB = a2;
    if (population[b2].energy < population[parentB].energy) parentB = b2;
    if (population[c2].energy < population[parentB].energy) parentB = c2;

    device DockPose &child = offspring[tid];
    float alpha = gpuRandom(seed, 7);

    if (gpuRandom(seed, 6) < gaParams.crossoverRate) {
        // Arithmetic crossover for translation
        child.translation = mix(population[parentA].translation, population[parentB].translation, alpha);

        // SLERP for quaternion
        float4 qa = population[parentA].rotation;
        float4 qb = population[parentB].rotation;
        float dotQ = dot(qa, qb);
        if (dotQ < 0.0f) { qb = -qb; dotQ = -dotQ; }
        dotQ = clamp(dotQ, -1.0f, 1.0f);

        if (dotQ > 0.9995f) {
            child.rotation = normalize(mix(qa, qb, alpha));
        } else {
            float theta = acos(dotQ);  // keep precise acos for domain [-1,1] stability
            float st = sin(theta);
            child.rotation = normalize(sin((1.0f - alpha) * theta) / st * qa + sin(alpha * theta) / st * qb);
        }

        // Uniform crossover for torsions
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            child.torsions[t] = (gpuRandom(seed, 10 + t) < 0.5f)
                ? population[parentA].torsions[t] : population[parentB].torsions[t];
        }
    } else {
        child.translation = population[parentA].translation;
        child.rotation = population[parentA].rotation;
        for (uint t = 0; t < gaParams.numTorsions; t++)
            child.torsions[t] = population[parentA].torsions[t];
    }

    float3 searchCenter = gridParams.searchCenter;
    float3 searchHalfExtent = gridParams.searchHalfExtent;
    float3 searchMin = searchCenter - searchHalfExtent;
    float3 searchMax = searchCenter + searchHalfExtent;

    // Diversity injection: reinitialize bottom 10% with random poses every 8 generations.
    bool doInject = (gaParams.generation % 8 == 0) && (gaParams.generation > 0);
    uint injectThreshold = popSize - max(popSize / 10, 3u);  // bottom 10%
    if (doInject && tid >= injectThreshold) {
        child.translation = searchCenter + float3(
            (gpuRandom(seed, 80) * 2.0f - 1.0f) * searchHalfExtent.x,
            (gpuRandom(seed, 81) * 2.0f - 1.0f) * searchHalfExtent.y,
            (gpuRandom(seed, 82) * 2.0f - 1.0f) * searchHalfExtent.z
        );
        float u1 = gpuRandom(seed, 83);
        float u2 = gpuRandom(seed, 84) * 2.0f * M_PI_F;
        float u3 = gpuRandom(seed, 85) * 2.0f * M_PI_F;
        child.rotation = normalize(float4(sqrt(1-u1)*sin(u2), sqrt(1-u1)*cos(u2), sqrt(u1)*sin(u3), sqrt(u1)*cos(u3)));
        for (uint t = 0; t < gaParams.numTorsions; t++)
            child.torsions[t] = gpuRandom(seed, 90 + t) * 2.0f * M_PI_F - M_PI_F;
        child.numTorsions = int(gaParams.numTorsions);
        child.generation = int(gaParams.generation) + 1;
        child.energy = 1e10f;
        child.stericEnergy = 0.0f;
        child.hydrophobicEnergy = 0.0f;
        child.hbondEnergy = 0.0f;
        child.torsionPenalty = 0.0f;
        child.clashPenalty = 0.0f;
        return;
    }

    // Mutation -- translation (occasional large jumps for exploration)
    float mutRoll = gpuRandom(seed, 50);
    if (mutRoll < gaParams.mutationRate) {
        float step = gaParams.translationStep;
        // 20% of mutations are large jumps (5x step) for wider exploration
        if (mutRoll < gaParams.mutationRate * 0.2f) step *= 5.0f;
        child.translation += float3(
            (gpuRandom(seed, 51) * 2.0f - 1.0f) * step,
            (gpuRandom(seed, 52) * 2.0f - 1.0f) * step,
            (gpuRandom(seed, 53) * 2.0f - 1.0f) * step
        );
        child.translation = clamp(child.translation, searchMin, searchMax);
    }

    // Mutation -- rotation
    if (gpuRandom(seed, 54) < gaParams.mutationRate) {
        float angle = (gpuRandom(seed, 55) * 2.0f - 1.0f) * gaParams.rotationStep;
        float3 axis = normalize(float3(gpuRandom(seed, 56) - 0.5f, gpuRandom(seed, 57) - 0.5f, gpuRandom(seed, 58) - 0.5f));
        float ha = angle * 0.5f;
        float4 dq = float4(axis * sin(ha), cos(ha));
        float4 q = child.rotation;
        child.rotation = normalize(float4(
            dq.w*q.x + dq.x*q.w + dq.y*q.z - dq.z*q.y,
            dq.w*q.y - dq.x*q.z + dq.y*q.w + dq.z*q.x,
            dq.w*q.z + dq.x*q.y - dq.y*q.x + dq.z*q.w,
            dq.w*q.w - dq.x*q.x - dq.y*q.y - dq.z*q.z
        ));
    }

    // Mutation -- torsions
    for (uint t = 0; t < gaParams.numTorsions; t++) {
        if (gpuRandom(seed, 60 + t) < gaParams.mutationRate) {
            child.torsions[t] += (gpuRandom(seed, 100 + t) * 2.0f - 1.0f) * gaParams.torsionStep;
            if (child.torsions[t] > M_PI_F) child.torsions[t] -= 2.0f * M_PI_F;
            if (child.torsions[t] < -M_PI_F) child.torsions[t] += 2.0f * M_PI_F;
        }
    }

    child.numTorsions = int(gaParams.numTorsions);
    child.generation = int(gaParams.generation) + 1;
    child.energy = 1e10f;
    child.stericEnergy = 0.0f;
    child.hydrophobicEnergy = 0.0f;
    child.hbondEnergy = 0.0f;
    child.torsionPenalty = 0.0f;
    child.clashPenalty = 0.0f;
}

// ============================================================================
// MARK: - Local Search (Steepest Descent)
// ============================================================================

/// Helper: compute total Vina score for a set of transformed atom positions.
/// Applies torsion entropy normalization consistent with scorePoses and evaluatePose.
inline float vinaScorePositions(
    thread float3              *positions,
    constant DockLigandAtom    *ligandAtoms,
    uint                        nAtoms,
    uint                        nTorsions,
    device const half          *affinityMaps,
    device const int32_t       *typeIndexLookup,
    constant GridParams        &gp,
    float                       referenceIntraEnergy,
    constant uint32_t          *intraPairs = nullptr,
    uint                        numIntraPairs = 0)
{
    float3 gridMin = gp.origin;
    float3 gridMax = gp.origin + float3(gp.dims) * gp.spacing;

    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }

        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gp
        );
    }

    float intraDelta = 0.0f;
    if (intraPairs && numIntraPairs > 0) {
        intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nAtoms, intraPairs, numIntraPairs)
            - referenceIntraEnergy;
    }

    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    return totalIntermolecular * normFactor + wPenalty * penalty + intraDelta;
}

/// Steepest descent local search (Lamarckian refinement).
/// Numerically estimates gradient on translation and steps downhill.
// ============================================================================
// MARK: - Monte Carlo Perturbation + Local Optimization (ILS)
// ============================================================================

/// Score a pose fully (transform + evaluate against grids). Returns total Vina energy.
inline float evaluatePose(
    device const DockPose      &pose,
    constant DockLigandAtom    *ligandAtoms,
    uint                        nAtoms,
    uint                        nTorsions,
    constant TorsionEdge       *torsionEdges,
    constant int32_t           *movingIndices,
    device const half          *affinityMaps,
    device const int32_t       *typeIndexLookup,
    constant GridParams        &gp,
    float                       referenceIntraEnergy,
    constant uint32_t          *intraPairs = nullptr,
    uint                        numIntraPairs = 0)
{
    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    // Apply rigid-body + torsion transform
    for (uint a = 0; a < nA; a++) {
        positions[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    applyTorsions(positions, nA, pose, torsionEdges, movingIndices, nTorsions);

    float3 gridMin = gp.origin;
    float3 gridMax = gp.origin + float3(gp.dims) * gp.spacing;

    float totalIntermolecular = 0.0f;
    float boundaryPenalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            boundaryPenalty += oopP;
            continue;
        }

        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gp
        );
    }

    float intraDelta = 0.0f;
    if (intraPairs && numIntraPairs > 0) {
        intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, numIntraPairs)
            - referenceIntraEnergy;
    }

    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    return totalIntermolecular * normFactor + wPenalty * boundaryPenalty + intraDelta;
}

/// Constrained version of evaluatePose: includes pharmacophore penalty.
inline float evaluatePoseConstrained(
    device const DockPose      &pose,
    constant DockLigandAtom    *ligandAtoms,
    uint                        nAtoms,
    uint                        nTorsions,
    constant TorsionEdge       *torsionEdges,
    constant int32_t           *movingIndices,
    device const half          *affinityMaps,
    device const int32_t       *typeIndexLookup,
    constant GridParams        &gp,
    float                       referenceIntraEnergy,
    constant uint32_t          *intraPairs,
    uint                        numIntraPairs,
    constant PharmacophoreConstraint *constraints,
    constant PharmacophoreParams &pharmaParams)
{
    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    for (uint a = 0; a < nA; a++) {
        positions[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    applyTorsions(positions, nA, pose, torsionEdges, movingIndices, nTorsions);

    float3 gridMin = gp.origin;
    float3 gridMax = gp.origin + float3(gp.dims) * gp.spacing;

    float totalIntermolecular = 0.0f;
    float boundaryPenalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];
        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            boundaryPenalty += oopP;
            continue;
        }
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gp
        );
    }

    float intraDelta = 0.0f;
    if (intraPairs && numIntraPairs > 0) {
        intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, numIntraPairs)
            - referenceIntraEnergy;
    }

    float constraintPen = evaluateConstraintPenalty(
        positions, ligandAtoms, nA, constraints, pharmaParams);

    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    return totalIntermolecular * normFactor + wPenalty * boundaryPenalty + intraDelta + constraintPen;
}

/// Monte Carlo perturbation: randomly perturb translation, rotation, and torsions.
/// One thread per pose. Writes perturbed pose to offspring buffer.
kernel void mcPerturb(
    device DockPose            *perturbed   [[buffer(0)]],
    device const DockPose      *current     [[buffer(1)]],
    constant GAParams          &gaParams    [[buffer(2)]],
    constant GridParams        &gridParams  [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint seed = tid * 314159u + gaParams.generation * 271828u + 42u;
    device const DockPose &src = current[tid];
    device DockPose &dst = perturbed[tid];

    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    dst = src;

    uint mutableEntities = 2u + gaParams.numTorsions;
    uint which = mutableEntities > 0u
        ? uint(gpuRandom(seed, 0) * float(mutableEntities)) % mutableEntities
        : 0u;

    if (which == 0u) {
        float3 delta = gaParams.translationStep * randomInsideUnitSphere(seed, 10);
        dst.translation = clamp(src.translation + delta, gMin, gMax);
    } else if (which == 1u) {
        float radius = max(gaParams.ligandRadius, 1.0f);
        float rotSpread = max(gaParams.rotationStep, gaParams.translationStep / radius);
        float3 rotVec = rotSpread * randomInsideUnitSphere(seed, 20);
        float angle = length(rotVec);
        if (angle > 1e-6f) {
            float3 axis = rotVec / angle;
            float halfA = angle * 0.5f;
            float4 dq = float4(axis * sin(halfA), cos(halfA));
            float4 sq = src.rotation;
            dst.rotation = normalize(float4(
                dq.w*sq.x + dq.x*sq.w + dq.y*sq.z - dq.z*sq.y,
                dq.w*sq.y - dq.x*sq.z + dq.y*sq.w + dq.z*sq.x,
                dq.w*sq.z + dq.x*sq.y - dq.y*sq.x + dq.z*sq.w,
                dq.w*sq.w - dq.x*sq.x - dq.y*sq.y - dq.z*sq.z
            ));
        }
    } else {
        uint torsionIndex = which - 2u;
        if (torsionIndex < gaParams.numTorsions) {
            dst.torsions[torsionIndex] = gpuRandom(seed, 30 + torsionIndex) * 2.0f * M_PI_F - M_PI_F;
        }
    }

    dst.numTorsions = src.numTorsions;
    dst.generation = src.generation + 1;
    dst.energy = 1e10f;
    dst.stericEnergy = 0.0f;
    dst.hydrophobicEnergy = 0.0f;
    dst.hbondEnergy = 0.0f;
    dst.torsionPenalty = 0.0f;
    dst.clashPenalty = 0.0f;
}

kernel void metropolisAccept(
    device DockPose            *current     [[buffer(0)]],
    device const DockPose      *candidate   [[buffer(1)]],
    device DockPose            *best        [[buffer(2)]],
    constant GAParams          &gaParams    [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &cur = current[tid];
    device DockPose &bestPose = best[tid];
    DockPose cand = candidate[tid];

    bool candValid = isfinite(cand.energy) && cand.energy < 1e9f;
    bool curValid = isfinite(cur.energy) && cur.energy < 1e9f;
    bool accept = false;

    if (candValid) {
        if (!curValid || cand.energy < cur.energy) {
            accept = true;
        } else {
            float temperature = max(gaParams.mcTemperature, 1e-4f);
            float acceptance = exp((cur.energy - cand.energy) / temperature);
            uint seed = tid * 92837111u + gaParams.generation * 689287499u + 17u;
            accept = gpuRandom(seed, 0) < min(acceptance, 1.0f);
        }
    }

    if (accept) {
        cur = cand;
    }

    bool bestValid = isfinite(bestPose.energy) && bestPose.energy < 1e9f;
    bool nowValid = isfinite(cur.energy) && cur.energy < 1e9f;
    if (nowValid && (!bestValid || cur.energy < bestPose.energy)) {
        bestPose = cur;
    }
}

/// Full-DOF local optimization: optimizes translation, rotation, AND torsions.
/// Uses steepest descent with numerical gradients across all degrees of freedom.
/// One thread per pose. Step count comes from GAParams.localSearchSteps.
kernel void localSearch(
    device DockPose            *poses        [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms  [[buffer(1)]],
    device const half          *affinityMaps [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams   [[buffer(4)]],
    constant GAParams          &gaParams     [[buffer(5)]],
    constant TorsionEdge       *torsionEdges [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *intraPairs   [[buffer(8)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = min(gaParams.numLigandAtoms, 128u);
    uint nTor = min(gaParams.numTorsions, 32u);
    uint nPairs = gaParams.numIntraPairs;
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    float stepSize = 0.08f;
    float h = 0.03f;  // finite difference step

    int maxSteps = max(int(gaParams.localSearchSteps), 1);
    for (int step = 0; step < maxSteps; step++) {
        float baseE = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                    torsionEdges, movingIndices,
                                    affinityMaps, typeIndexLookup, gridParams,
                                    gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                    pharmaConstraints, pharmaParams);

        // ---- Translation gradient (3 DOF) ----
        float gradT[3];
        for (int dim = 0; dim < 3; dim++) {
            float3 origT = pose.translation;
            pose.translation[dim] = origT[dim] + h;
            float ePlus = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                        torsionEdges, movingIndices,
                                        affinityMaps, typeIndexLookup, gridParams,
                                        gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                        pharmaConstraints, pharmaParams);
            pose.translation[dim] = origT[dim] - h;
            float eMinus = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                         torsionEdges, movingIndices,
                                         affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                         pharmaConstraints, pharmaParams);
            pose.translation = origT;
            gradT[dim] = (ePlus - eMinus) / (2.0f * h);
        }

        // ---- Rotation gradient (3 DOF via axis-angle) ----
        float gradR[3];
        float hRot = 0.02f;
        float4 origRot = pose.rotation;
        for (int dim = 0; dim < 3; dim++) {
            float3 rotAxis = float3(0); rotAxis[dim] = 1.0f;
            float halfA = hRot * 0.5f;
            float4 dqPlus  = float4(rotAxis * sin(halfA), cos(halfA));
            float4 dqMinus = float4(rotAxis * -sin(halfA), cos(halfA));

            // Apply positive rotation
            pose.rotation = normalize(float4(
                dqPlus.w*origRot.x + dqPlus.x*origRot.w + dqPlus.y*origRot.z - dqPlus.z*origRot.y,
                dqPlus.w*origRot.y - dqPlus.x*origRot.z + dqPlus.y*origRot.w + dqPlus.z*origRot.x,
                dqPlus.w*origRot.z + dqPlus.x*origRot.y - dqPlus.y*origRot.x + dqPlus.z*origRot.w,
                dqPlus.w*origRot.w - dqPlus.x*origRot.x - dqPlus.y*origRot.y - dqPlus.z*origRot.z
            ));
            float ePlus = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                        torsionEdges, movingIndices,
                                        affinityMaps, typeIndexLookup, gridParams,
                                        gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                        pharmaConstraints, pharmaParams);

            // Apply negative rotation
            pose.rotation = normalize(float4(
                dqMinus.w*origRot.x + dqMinus.x*origRot.w + dqMinus.y*origRot.z - dqMinus.z*origRot.y,
                dqMinus.w*origRot.y - dqMinus.x*origRot.z + dqMinus.y*origRot.w + dqMinus.z*origRot.x,
                dqMinus.w*origRot.z + dqMinus.x*origRot.y - dqMinus.y*origRot.x + dqMinus.z*origRot.w,
                dqMinus.w*origRot.w - dqMinus.x*origRot.x - dqMinus.y*origRot.y - dqMinus.z*origRot.z
            ));
            float eMinus = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                         torsionEdges, movingIndices,
                                         affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                         pharmaConstraints, pharmaParams);

            pose.rotation = origRot;
            gradR[dim] = (ePlus - eMinus) / (2.0f * hRot);
        }

        // ---- Torsion gradients ----
        float gradTor[32];
        float hTor = 0.02f;
        for (uint t = 0; t < nTor; t++) {
            float origTor = pose.torsions[t];
            pose.torsions[t] = origTor + hTor;
            float ePlus = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                        torsionEdges, movingIndices,
                                        affinityMaps, typeIndexLookup, gridParams,
                                        gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                        pharmaConstraints, pharmaParams);
            pose.torsions[t] = origTor - hTor;
            float eMinus = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                         torsionEdges, movingIndices,
                                         affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                         pharmaConstraints, pharmaParams);
            pose.torsions[t] = origTor;
            gradTor[t] = (ePlus - eMinus) / (2.0f * hTor);
        }

        // ---- Compute total gradient magnitude ----
        float gradMag = 0;
        for (int i = 0; i < 3; i++) gradMag += gradT[i] * gradT[i];
        for (int i = 0; i < 3; i++) gradMag += gradR[i] * gradR[i];
        for (uint t = 0; t < nTor; t++) gradMag += gradTor[t] * gradTor[t];
        gradMag = sqrt(gradMag);
        if (gradMag < 1e-6f) break;  // converged

        // Normalize gradient to step in consistent direction
        float scale = stepSize / gradMag;

        // ---- Apply gradient step ----
        float3 newT = pose.translation;
        for (int i = 0; i < 3; i++) newT[i] -= scale * gradT[i];
        newT = clamp(newT, gMin, gMax);

        // Apply rotation step
        float3 rotStep = float3(-scale * gradR[0], -scale * gradR[1], -scale * gradR[2]);
        float rotAngle = length(rotStep);
        float4 newRot = pose.rotation;
        if (rotAngle > 1e-6f) {
            float3 rotAx = rotStep / rotAngle;
            float halfRot = rotAngle * 0.5f;
            float4 dq = float4(rotAx * sin(halfRot), cos(halfRot));
            newRot = normalize(float4(
                dq.w*pose.rotation.x + dq.x*pose.rotation.w + dq.y*pose.rotation.z - dq.z*pose.rotation.y,
                dq.w*pose.rotation.y - dq.x*pose.rotation.z + dq.y*pose.rotation.w + dq.z*pose.rotation.x,
                dq.w*pose.rotation.z + dq.x*pose.rotation.y - dq.y*pose.rotation.x + dq.z*pose.rotation.w,
                dq.w*pose.rotation.w - dq.x*pose.rotation.x - dq.y*pose.rotation.y - dq.z*pose.rotation.z
            ));
        }

        // Save current state for rollback
        float3 oldT = pose.translation;
        float4 oldRot = pose.rotation;
        float oldTorsions[32];
        for (uint t = 0; t < nTor; t++) oldTorsions[t] = pose.torsions[t];

        // Apply
        pose.translation = newT;
        pose.rotation = newRot;
        for (uint t = 0; t < nTor; t++) {
            pose.torsions[t] -= scale * gradTor[t];
            if (pose.torsions[t] > M_PI_F) pose.torsions[t] -= 2.0f * M_PI_F;
            if (pose.torsions[t] < -M_PI_F) pose.torsions[t] += 2.0f * M_PI_F;
        }

        float newE = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                   torsionEdges, movingIndices,
                                   affinityMaps, typeIndexLookup, gridParams,
                                   gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                   pharmaConstraints, pharmaParams);

        if (newE < baseE) {
            pose.energy = newE;
            stepSize = min(stepSize * 1.2f, 1.0f);
        } else {
            // Rollback
            pose.translation = oldT;
            pose.rotation = oldRot;
            for (uint t = 0; t < nTor; t++) pose.torsions[t] = oldTorsions[t];
            pose.energy = baseE;
            stepSize *= 0.5f;
        }
        if (stepSize < 0.0005f) break;
    }
}

// ============================================================================
// MARK: - Analytical Gradient Local Search
// ============================================================================

/// Evaluate pose energy and compute analytical gradients for all DOF in a single pass.
/// Uses force/torque accumulation (same algorithm as AutoDock Vina tree.h).
///
/// For each torsion, the derivative is: dE/dθ = torque · axis
/// where torque = Σ cross(atom_pos - pivot, force_on_atom) for all downstream atoms.
/// Forces propagate upward leaf→root: parent accumulates child forces and torques.
///
/// Returns total energy. Fills gradT[3], gradR[3], gradTor[32].
inline float evaluatePoseWithGradient(
    device const DockPose   &pose,
    constant DockLigandAtom *ligandAtoms,
    uint                     nAtoms,
    uint                     nTorsions,
    constant TorsionEdge    *torsionEdges,
    constant int32_t        *movingIndices,
    device const half       *affinityMaps,
    device const int32_t    *typeIndexLookup,
    constant GridParams     &gp,
    float                    referenceIntraEnergy,
    constant uint32_t       *intraPairs,
    uint                     numIntraPairs,
    thread float            *gradT,
    thread float            *gradR,
    thread float            *gradTor)
{
    uint nA = min(nAtoms, 128u);
    uint nTor = min(nTorsions, 32u);

    // --- Forward pass: transform atoms (identical to evaluatePose) ---
    float3 positions[128];
    for (uint a = 0; a < nA; a++) {
        positions[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    for (uint t = 0; t < nTor; t++) {
        float angle = pose.torsions[t];
        if (abs(angle) < 1e-6f) continue;
        TorsionEdge edge = torsionEdges[t];
        float3 pivot = positions[edge.atom1];
        float3 axisVec = positions[edge.atom2] - pivot;
        float axisLen = length(axisVec);
        if (axisLen < 1e-6f) continue;
        float3 axis = axisVec / axisLen;
        float cosA = cos(angle);
        float sinA = sin(angle);
        for (int i = 0; i < edge.movingCount; i++) {
            int ai = movingIndices[edge.movingStart + i];
            if (ai < 0 || uint(ai) >= nA) continue;
            float3 v = positions[ai] - pivot;
            positions[ai] = pivot + v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
        }
    }

    // --- Compute energy and per-atom forces (negative gradient) ---
    float3 forces[128];  // forces[a] = -dE/dpos[a]
    for (uint a = 0; a < nA; a++) forces[a] = float3(0);

    float nRotF = float(nTor);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);

    // Intermolecular: grid-based energy + gradient
    float totalIntermolecular = 0.0f;
    for (uint a = 0; a < nA; a++) {
        float3 grad;
        float e = sampleTypedAffinityMapWithGrad(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, positions[a], gp, grad
        );
        totalIntermolecular += e;
        forces[a] -= grad * normFactor;  // force = -gradient
    }

    // Intramolecular energy + gradient (intramolecularWithGrad accumulates +dE/dpos)
    float intraE = 0.0f;
    if (intraPairs && numIntraPairs > 0) {
        float3 intraGrad[128];
        for (uint a = 0; a < nA; a++) intraGrad[a] = float3(0);
        intraE = intramolecularWithGrad(positions, intraGrad, ligandAtoms, nA, intraPairs, numIntraPairs);
        for (uint a = 0; a < nA; a++) forces[a] -= intraGrad[a];  // force = -gradient
    }
    float intraDelta = intraE - referenceIntraEnergy;
    float totalEnergy = totalIntermolecular * normFactor + intraDelta;

    // --- Torsion derivatives via force/torque accumulation (Vina algorithm) ---
    // Process torsions leaf→root. For each torsion:
    //   1. Sum force and torque over the torsion's moving atoms
    //   2. dE/dθ = -(torque · axis)  (torque from forces, sign: force = -dE/dpos)
    //   3. Propagate: parent torsion receives child's force and torque contribution

    // We process in reverse order (leaf→root) since torsions are stored root→leaf.
    // Each torsion's derivative includes contributions from all downstream torsions.
    float3 torsionForce[32];   // accumulated force for each torsion subtree
    float3 torsionTorque[32];  // accumulated torque for each torsion subtree
    for (uint t = 0; t < nTor; t++) {
        torsionForce[t] = float3(0);
        torsionTorque[t] = float3(0);
    }

    // Step 1: Each torsion accumulates force/torque from its own moving atoms
    for (uint t = 0; t < nTor; t++) {
        TorsionEdge edge = torsionEdges[t];
        float3 pivot = positions[edge.atom1];
        for (int i = 0; i < edge.movingCount; i++) {
            int ai = movingIndices[edge.movingStart + i];
            if (ai < 0 || uint(ai) >= nA) continue;
            torsionForce[t] += forces[ai];
            torsionTorque[t] += cross(positions[ai] - pivot, forces[ai]);
        }
    }

    // Step 2: Propagate child torsion force/torque to parent (leaf→root)
    // A child torsion's force gets added to the parent's force,
    // and cross(child_origin - parent_origin, child_force) + child_torque gets added to parent's torque.
    // Since our torsion tree is flat (not explicitly hierarchical), we check containment:
    // torsion j is a child of torsion i if torsion i's moving atoms include torsion j's pivot atom.
    // For the flat representation, torsions are ordered root→leaf, so we process in reverse.
    // But a simpler approach: each atom's force only needs to appear in the innermost torsion
    // that moves it, and the flat torsion loop already handles this correctly because
    // inner torsions' atoms are a subset of outer torsions' atoms.
    // The above per-torsion accumulation already double-counts atoms that appear in multiple
    // torsion moving sets. We need to subtract child contributions and add them as torque.

    // Actually, the correct approach for a flat torsion list where moving sets may overlap:
    // Skip the hierarchical propagation and compute each torsion's gradient directly as
    // the torque about the axis from ALL atoms downstream of that torsion bond.
    // This is what the per-torsion accumulation above already does correctly, because
    // edge.movingCount includes all atoms downstream of the torsion (including those
    // moved by child torsions).

    for (uint t = 0; t < nTor; t++) {
        TorsionEdge edge = torsionEdges[t];
        float3 axisVec = positions[edge.atom2] - positions[edge.atom1];
        float axisLen = length(axisVec);
        if (axisLen < 1e-6f) { gradTor[t] = 0.0f; continue; }
        float3 axis = axisVec / axisLen;
        // dE/dθ = -(torque · axis)  because force = -dE/dpos
        gradTor[t] = -dot(torsionTorque[t], axis);
    }

    // --- Translation gradient = -total_force = sum of dE/dpos ---
    float3 totalForce = float3(0);
    for (uint a = 0; a < nA; a++) totalForce += forces[a];
    gradT[0] = -totalForce.x;  // gradient = -force
    gradT[1] = -totalForce.y;
    gradT[2] = -totalForce.z;

    // --- Rotation gradient via torque about ligand center ---
    // Total torque about the rigid body origin (pose.translation):
    // τ = Σ cross(pos_a - origin, force_a)
    // dE/dω_k = -τ_k  (axis-angle parameterization)
    float3 totalTorque = float3(0);
    for (uint a = 0; a < nA; a++) {
        totalTorque += cross(positions[a] - pose.translation, forces[a]);
    }
    gradR[0] = -totalTorque.x;
    gradR[1] = -totalTorque.y;
    gradR[2] = -totalTorque.z;

    return totalEnergy;
}

/// Analytical gradient local search: same structure as numerical `localSearch`,
/// but computes exact gradients in a single forward pass instead of finite differences.
/// ~28x fewer evaluations for a typical drug molecule.
kernel void localSearchAnalytical(
    device DockPose            *poses        [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms  [[buffer(1)]],
    device const half          *affinityMaps [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams   [[buffer(4)]],
    constant GAParams          &gaParams     [[buffer(5)]],
    constant TorsionEdge       *torsionEdges [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *intraPairs   [[buffer(8)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = min(gaParams.numLigandAtoms, 128u);
    uint nTor = min(gaParams.numTorsions, 32u);
    uint nPairs = gaParams.numIntraPairs;
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    float stepSize = 0.08f;
    int maxSteps = max(int(gaParams.localSearchSteps), 1);

    for (int step = 0; step < maxSteps; step++) {
        float gradT[3], gradR[3], gradTor[32];

        float baseE = evaluatePoseWithGradient(
            pose, ligandAtoms, nAtoms, nTor,
            torsionEdges, movingIndices,
            affinityMaps, typeIndexLookup, gridParams,
            gaParams.referenceIntraEnergy, intraPairs, nPairs,
            gradT, gradR, gradTor
        );

        // Add pharmacophore constraint penalty + gradient to the analytical gradient.
        // We need the transformed positions to evaluate constraints. Re-transform (cheap).
        if (pharmaParams.numConstraints > 0) {
            float3 cPositions[128];
            float3 cForces[128];
            for (uint a = 0; a < nAtoms; a++) {
                cPositions[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
                cForces[a] = float3(0);
            }
            applyTorsions(cPositions, nAtoms, pose, torsionEdges, movingIndices, nTor);

            float cPen = evaluateConstraintPenaltyWithGrad(
                cPositions, cForces, ligandAtoms, nAtoms, pharmaConstraints, pharmaParams);
            baseE += cPen;

            // Propagate atom-level forces back to translation and rotation gradients.
            // Translation gradient: sum of all forces (dE/d(tx) = sum_a dE/d(xa))
            for (uint a = 0; a < nAtoms; a++) {
                // Force is negative gradient already, so negate back
                gradT[0] -= cForces[a].x;
                gradT[1] -= cForces[a].y;
                gradT[2] -= cForces[a].z;
            }
            // Rotation gradient: torque approximation (simplified — project onto rotation axes)
            for (uint a = 0; a < nAtoms; a++) {
                float3 r = cPositions[a] - pose.translation;
                float3 f = -cForces[a];  // convert from force to gradient direction
                float3 torque = cross(r, f);
                gradR[0] += torque.x;
                gradR[1] += torque.y;
                gradR[2] += torque.z;
            }
        }

        // Compute total gradient magnitude
        float gradMag = 0;
        for (int i = 0; i < 3; i++) gradMag += gradT[i] * gradT[i];
        for (int i = 0; i < 3; i++) gradMag += gradR[i] * gradR[i];
        for (uint t = 0; t < nTor; t++) gradMag += gradTor[t] * gradTor[t];
        gradMag = sqrt(gradMag);
        if (gradMag < 1e-6f) break;

        // Gradient-normalized step: always step proportional to gradient direction,
        // scaled by stepSize. This avoids overshooting when gradient is small.
        float scale = stepSize / gradMag;

        // Apply gradient step
        float3 newT = pose.translation;
        for (int i = 0; i < 3; i++) newT[i] -= scale * gradT[i];
        newT = clamp(newT, gMin, gMax);

        // Apply rotation step
        float3 rotStep = float3(-scale * gradR[0], -scale * gradR[1], -scale * gradR[2]);
        float rotAngle = length(rotStep);
        float4 newRot = pose.rotation;
        if (rotAngle > 1e-6f) {
            float3 rotAx = rotStep / rotAngle;
            float halfRot = rotAngle * 0.5f;
            float4 dq = float4(rotAx * sin(halfRot), cos(halfRot));
            newRot = normalize(float4(
                dq.w*pose.rotation.x + dq.x*pose.rotation.w + dq.y*pose.rotation.z - dq.z*pose.rotation.y,
                dq.w*pose.rotation.y - dq.x*pose.rotation.z + dq.y*pose.rotation.w + dq.z*pose.rotation.x,
                dq.w*pose.rotation.z + dq.x*pose.rotation.y - dq.y*pose.rotation.x + dq.z*pose.rotation.w,
                dq.w*pose.rotation.w - dq.x*pose.rotation.x - dq.y*pose.rotation.y - dq.z*pose.rotation.z
            ));
        }

        // Save for rollback
        float3 oldT = pose.translation;
        float4 oldRot = pose.rotation;
        float oldTorsions[32];
        for (uint t = 0; t < nTor; t++) oldTorsions[t] = pose.torsions[t];

        // Apply
        pose.translation = newT;
        pose.rotation = newRot;
        for (uint t = 0; t < nTor; t++) {
            pose.torsions[t] -= scale * gradTor[t];
            if (pose.torsions[t] > M_PI_F) pose.torsions[t] -= 2.0f * M_PI_F;
            if (pose.torsions[t] < -M_PI_F) pose.torsions[t] += 2.0f * M_PI_F;
        }

        // Verify improvement
        float newE = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                   torsionEdges, movingIndices,
                                   affinityMaps, typeIndexLookup, gridParams,
                                   gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                   pharmaConstraints, pharmaParams);

        if (newE < baseE) {
            pose.energy = newE;
            stepSize = min(stepSize * 1.2f, 1.0f);
        } else {
            pose.translation = oldT;
            pose.rotation = oldRot;
            for (uint t = 0; t < nTor; t++) pose.torsions[t] = oldTorsions[t];
            pose.energy = baseE;
            stepSize *= 0.5f;
        }
        if (stepSize < 0.0005f) break;
    }
}

// ============================================================================
// MARK: - SIMD-Cooperative Analytical Gradient Local Search
// ============================================================================

/// SIMD-cooperative version of localSearchAnalytical.
/// Dispatch with populationSize threadgroups of 32 threads each.
/// Each threadgroup (= 1 SIMD group on Apple Silicon) handles one pose.
/// Rigid body transform and grid scoring distributed across SIMD lanes.
/// Verification uses inline SIMD scoring (same FP reduction order as gradient loop)
/// instead of sequential evaluatePoseConstrained — eliminates FP mismatch that
/// caused false step rejections and premature convergence.
kernel void localSearchAnalyticalSIMD(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    device const half          *affinityMaps  [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams    [[buffer(4)]],
    constant GAParams          &gaParams      [[buffer(5)]],
    constant TorsionEdge       *torsionEdges  [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *intraPairs    [[buffer(8)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tgIdx         [[threadgroup_position_in_grid]],
    uint                        lane          [[thread_index_in_threadgroup]])
{
    if (tgIdx >= gaParams.populationSize) return;

    device DockPose &pose = poses[tgIdx];
    uint nA = min(gaParams.numLigandAtoms, 128u);
    uint nTor = min(gaParams.numTorsions, 32u);
    uint nPairs = gaParams.numIntraPairs;
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    float nRotF = float(nTor);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);

    threadgroup float3 tg_pos[128];
    threadgroup float3 tg_forces[128];
    threadgroup float3 tg_oldT;
    threadgroup float4 tg_oldRot;
    threadgroup float  tg_oldTor[32];

    int maxSteps = max(int(gaParams.localSearchSteps), 1);

    // L-BFGS history (m=5): store position and gradient differences
    const uint LBFGS_M = 5;
    const uint NDIM = 6 + nTor;  // 3 translation + 3 rotation + nTorsions
    float lbfgs_s[LBFGS_M][38];  // s_k = x_{k+1} - x_k  (position diff)
    float lbfgs_y[LBFGS_M][38];  // y_k = g_{k+1} - g_k  (gradient diff)
    float lbfgs_rho[LBFGS_M];    // 1 / dot(y_k, s_k)
    uint lbfgs_count = 0;         // how many history pairs stored
    uint lbfgs_newest = 0;        // circular buffer index

    float prevGrad[38];           // previous gradient for L-BFGS update
    float prevX[38];              // previous position for L-BFGS update
    bool hasPrev = false;

    for (int step = 0; step < maxSteps; step++) {
        // === Transform atoms: rigid body parallel, torsions sequential ===
        for (uint a = lane; a < nA; a += 32)
            tg_pos[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            for (uint t = 0; t < nTor; t++) {
                float angle = pose.torsions[t];
                if (abs(angle) < 1e-6f) continue;
                TorsionEdge edge = torsionEdges[t];
                float3 pivot = tg_pos[edge.atom1];
                float3 axis = normalize(tg_pos[edge.atom2] - pivot);
                float cosA = cos(angle); float sinA = sin(angle);
                for (int i = 0; i < edge.movingCount; i++) {
                    int ai = movingIndices[edge.movingStart + i];
                    if (ai < 0 || uint(ai) >= nA) continue;
                    float3 v = tg_pos[ai] - pivot;
                    tg_pos[ai] = pivot + v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Zero forces (parallel) ===
        for (uint a = lane; a < nA; a += 32) tg_forces[a] = float3(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Grid scoring with gradient (parallel across lanes) ===
        float laneGridE = 0.0f;
        for (uint a = lane; a < nA; a += 32) {
            float3 grad;
            float e = sampleTypedAffinityMapWithGrad(
                affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, tg_pos[a], gridParams, grad);
            laneGridE += e;
            tg_forces[a] = -grad * normFactor;
        }
        float totalIntermolecular = simd_sum(laneGridE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Intramolecular with gradient (lane 0 — force writes need sequential) ===
        float intraE = 0.0f;
        if (lane == 0 && nPairs > 0) {
            for (uint p = 0; p < nPairs; p++) {
                uint packed = intraPairs[p];
                uint i = packed & 0xFFFFu; uint j = packed >> 16u;
                if (i >= nA || j >= nA) continue;
                float3 diff = tg_pos[i] - tg_pos[j];
                float r = length(diff);
                if (r < 1e-6f) continue;
                float dEdr;
                intraE += vinaPairEnergyWithDeriv(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, r, dEdr);
                float3 gc = dEdr * diff / r;
                tg_forces[i] -= gc; tg_forces[j] += gc;
            }
        }
        intraE = simd_broadcast_first(intraE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float totalEnergy = totalIntermolecular * normFactor + intraE - gaParams.referenceIntraEnergy;

        // === Torsion gradients (parallel per torsion's moving atoms) ===
        float gradTor[32];
        for (uint t = 0; t < nTor; t++) {
            TorsionEdge edge = torsionEdges[t];
            float3 pivot = tg_pos[edge.atom1];
            float3 axisVec = tg_pos[edge.atom2] - pivot;
            float axisLen = length(axisVec);
            if (axisLen < 1e-6f) { gradTor[t] = 0; continue; }
            float3 axis = axisVec / axisLen;
            float lt = 0.0f;
            for (int i = int(lane); i < edge.movingCount; i += 32) {
                int ai = movingIndices[edge.movingStart + i];
                if (ai >= 0 && uint(ai) < nA)
                    lt += dot(cross(tg_pos[ai] - pivot, tg_forces[ai]), axis);
            }
            gradTor[t] = -simd_sum(lt);
        }

        // === Translation/rotation gradients (parallel reduction) ===
        float3 lf(0), ltr(0);
        for (uint a = lane; a < nA; a += 32) {
            lf += tg_forces[a];
            ltr += cross(tg_pos[a] - pose.translation, tg_forces[a]);
        }
        float3 totalForce = simd_sum(lf);
        float3 totalTorqueRot = simd_sum(ltr);
        float gradT[3] = {-totalForce.x, -totalForce.y, -totalForce.z};
        float gradR[3] = {-totalTorqueRot.x, -totalTorqueRot.y, -totalTorqueRot.z};

        // === Pharmacophore constraints (lane 0) ===
        if (pharmaParams.numConstraints > 0 && lane == 0) {
            float3 cPos[128], cF[128];
            for (uint a = 0; a < nA; a++) { cPos[a] = tg_pos[a]; cF[a] = float3(0); }
            float cPen = evaluateConstraintPenaltyWithGrad(cPos, cF, ligandAtoms, nA, pharmaConstraints, pharmaParams);
            totalEnergy += cPen;
            for (uint a = 0; a < nA; a++) {
                gradT[0] -= cF[a].x; gradT[1] -= cF[a].y; gradT[2] -= cF[a].z;
                float3 torque = cross(tg_pos[a] - pose.translation, -cF[a]);
                gradR[0] += torque.x; gradR[1] += torque.y; gradR[2] += torque.z;
            }
        }
        totalEnergy = simd_broadcast_first(totalEnergy);
        for (int i = 0; i < 3; i++) { gradT[i] = simd_broadcast_first(gradT[i]); gradR[i] = simd_broadcast_first(gradR[i]); }

        // Pack current gradient and position into flat vectors
        float curGrad[38], curX[38];
        curGrad[0] = gradT[0]; curGrad[1] = gradT[1]; curGrad[2] = gradT[2];
        curGrad[3] = gradR[0]; curGrad[4] = gradR[1]; curGrad[5] = gradR[2];
        for (uint t = 0; t < nTor; t++) curGrad[6+t] = gradTor[t];

        curX[0] = pose.translation.x; curX[1] = pose.translation.y; curX[2] = pose.translation.z;
        // Pack rotation as axis-angle (for L-BFGS position diffs)
        curX[3] = pose.rotation.x; curX[4] = pose.rotation.y; curX[5] = pose.rotation.z;
        for (uint t = 0; t < nTor; t++) curX[6+t] = pose.torsions[t];

        float gradMag = 0;
        for (uint d = 0; d < NDIM; d++) gradMag += curGrad[d] * curGrad[d];
        gradMag = sqrt(gradMag);
        if (gradMag < 1e-6f) break;

        // === Update L-BFGS history from previous step ===
        if (hasPrev) {
            float s_vec[38], y_vec[38];
            float sy = 0.0f;
            for (uint d = 0; d < NDIM; d++) {
                s_vec[d] = curX[d] - prevX[d];
                y_vec[d] = curGrad[d] - prevGrad[d];
                sy += s_vec[d] * y_vec[d];
            }
            if (sy > 1e-10f) {
                uint idx = lbfgs_newest;
                for (uint d = 0; d < NDIM; d++) {
                    lbfgs_s[idx][d] = s_vec[d];
                    lbfgs_y[idx][d] = y_vec[d];
                }
                lbfgs_rho[idx] = 1.0f / sy;
                lbfgs_newest = (lbfgs_newest + 1) % LBFGS_M;
                if (lbfgs_count < LBFGS_M) lbfgs_count++;
            }
        }

        // Save current for next iteration
        for (uint d = 0; d < NDIM; d++) { prevGrad[d] = curGrad[d]; prevX[d] = curX[d]; }
        hasPrev = true;

        // === L-BFGS two-loop recursion to compute search direction ===
        float q[38];
        for (uint d = 0; d < NDIM; d++) q[d] = curGrad[d];

        float alpha[LBFGS_M];
        // First loop: newest to oldest
        for (uint k = 0; k < lbfgs_count; k++) {
            uint idx = (lbfgs_newest + LBFGS_M - 1 - k) % LBFGS_M;
            float dotSQ = 0.0f;
            for (uint d = 0; d < NDIM; d++) dotSQ += lbfgs_s[idx][d] * q[d];
            alpha[k] = lbfgs_rho[idx] * dotSQ;
            for (uint d = 0; d < NDIM; d++) q[d] -= alpha[k] * lbfgs_y[idx][d];
        }

        // Initial Hessian estimate: H0 = (s'y / y'y) * I
        float H0 = 1.0f;
        if (lbfgs_count > 0) {
            uint lastIdx = (lbfgs_newest + LBFGS_M - 1) % LBFGS_M;
            float yy = 0.0f, sy2 = 0.0f;
            for (uint d = 0; d < NDIM; d++) {
                yy += lbfgs_y[lastIdx][d] * lbfgs_y[lastIdx][d];
                sy2 += lbfgs_s[lastIdx][d] * lbfgs_y[lastIdx][d];
            }
            if (yy > 1e-10f) H0 = sy2 / yy;
        }

        float r_dir[38];
        for (uint d = 0; d < NDIM; d++) r_dir[d] = H0 * q[d];

        // Second loop: oldest to newest
        for (uint k = lbfgs_count; k > 0; k--) {
            uint idx = (lbfgs_newest + LBFGS_M - k) % LBFGS_M;
            float dotYR = 0.0f;
            for (uint d = 0; d < NDIM; d++) dotYR += lbfgs_y[idx][d] * r_dir[d];
            float beta = lbfgs_rho[idx] * dotYR;
            for (uint d = 0; d < NDIM; d++) r_dir[d] += (alpha[lbfgs_count - k] - beta) * lbfgs_s[idx][d];
        }

        // r_dir is now the L-BFGS search direction (descent: -H*g)
        // Apply with a fixed step size of 1.0 (L-BFGS direction is already scaled)
        // but cap to prevent overshooting
        float dirMag = 0.0f;
        for (uint d = 0; d < NDIM; d++) dirMag += r_dir[d] * r_dir[d];
        dirMag = sqrt(dirMag);
        float maxStep = 0.5f;  // max displacement per step
        float scale = (dirMag > maxStep) ? (maxStep / dirMag) : 1.0f;

        // Unpack search direction
        float dirT[3] = {r_dir[0], r_dir[1], r_dir[2]};
        float dirR[3] = {r_dir[3], r_dir[4], r_dir[5]};
        float dirTor[32];
        for (uint t = 0; t < nTor; t++) dirTor[t] = r_dir[6+t];

        // === Apply step (lane 0 saves old pose, applies trial pose) ===
        if (lane == 0) {
            tg_oldT = pose.translation;
            tg_oldRot = pose.rotation;
            for (uint t = 0; t < nTor; t++) tg_oldTor[t] = pose.torsions[t];

            float3 newT = pose.translation;
            for (int i = 0; i < 3; i++) newT[i] -= scale * dirT[i];
            newT = clamp(newT, gMin, gMax);

            float3 rs = float3(-scale * dirR[0], -scale * dirR[1], -scale * dirR[2]);
            float ra = length(rs);
            float4 newRot = pose.rotation;
            if (ra > 1e-6f) {
                float3 ax = rs / ra; float hr = ra * 0.5f;
                float4 dq = float4(ax * sin(hr), cos(hr));
                newRot = normalize(float4(
                    dq.w*pose.rotation.x + dq.x*pose.rotation.w + dq.y*pose.rotation.z - dq.z*pose.rotation.y,
                    dq.w*pose.rotation.y - dq.x*pose.rotation.z + dq.y*pose.rotation.w + dq.z*pose.rotation.x,
                    dq.w*pose.rotation.z + dq.x*pose.rotation.y - dq.y*pose.rotation.x + dq.z*pose.rotation.w,
                    dq.w*pose.rotation.w - dq.x*pose.rotation.x - dq.y*pose.rotation.y - dq.z*pose.rotation.z));
            }
            pose.translation = newT; pose.rotation = newRot;
            for (uint t = 0; t < nTor; t++) {
                pose.torsions[t] -= scale * dirTor[t];
                if (pose.torsions[t] > M_PI_F) pose.torsions[t] -= 2.0f * M_PI_F;
                if (pose.torsions[t] < -M_PI_F) pose.torsions[t] += 2.0f * M_PI_F;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === SIMD-consistent verification ===
        // Re-score trial pose using same lane distribution as gradient loop.
        // This eliminates FP mismatch between gradient energy and verification energy
        // that caused false step rejections with the old evaluatePoseConstrained path.
        for (uint a = lane; a < nA; a += 32)
            tg_pos[a] = quatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            for (uint t = 0; t < nTor; t++) {
                float angle = pose.torsions[t];
                if (abs(angle) < 1e-6f) continue;
                TorsionEdge edge = torsionEdges[t];
                float3 pivot = tg_pos[edge.atom1];
                float3 axis = normalize(tg_pos[edge.atom2] - pivot);
                float cosA = cos(angle); float sinA = sin(angle);
                for (int i = 0; i < edge.movingCount; i++) {
                    int ai = movingIndices[edge.movingStart + i];
                    if (ai < 0 || uint(ai) >= nA) continue;
                    float3 v = tg_pos[ai] - pivot;
                    tg_pos[ai] = pivot + v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Grid energy + boundary penalty (parallel — same lane distribution)
        float laneNewGridE = 0.0f;
        float lanePenalty = 0.0f;
        for (uint a = lane; a < nA; a += 32) {
            float oopP = outOfGridPenalty(tg_pos[a], gridMin, gridMax);
            if (oopP > 0.0f) {
                lanePenalty += oopP;
            } else {
                laneNewGridE += sampleTypedAffinityMap(
                    affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, tg_pos[a], gridParams);
            }
        }
        float newGridE = simd_sum(laneNewGridE);
        float newPenalty = simd_sum(lanePenalty);

        // Intramolecular energy (lane 0 — same order as gradient loop)
        float newIntraE = 0.0f;
        if (lane == 0 && nPairs > 0) {
            for (uint p = 0; p < nPairs; p++) {
                uint packed = intraPairs[p];
                uint i = packed & 0xFFFFu; uint j = packed >> 16u;
                if (i >= nA || j >= nA) continue;
                float r = distance(tg_pos[i], tg_pos[j]);
                newIntraE += vinaPairEnergy(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, r);
            }
        }
        newIntraE = simd_broadcast_first(newIntraE);

        float newE = newGridE * normFactor + wPenalty * newPenalty
                   + newIntraE - gaParams.referenceIntraEnergy;

        // Pharmacophore constraint penalty on trial pose
        if (pharmaParams.numConstraints > 0 && lane == 0) {
            float3 cPos[128];
            for (uint a = 0; a < nA; a++) cPos[a] = tg_pos[a];
            newE += evaluateConstraintPenalty(cPos, ligandAtoms, nA, pharmaConstraints, pharmaParams);
        }
        newE = simd_broadcast_first(newE);

        // === Accept/reject ===
        bool accepted = (newE < totalEnergy);
        if (lane == 0) {
            if (accepted) {
                pose.energy = newE;
            } else {
                pose.translation = tg_oldT; pose.rotation = tg_oldRot;
                for (uint t = 0; t < nTor; t++) pose.torsions[t] = tg_oldTor[t];
                pose.energy = totalEnergy;
            }
        }
        // On rejection, invalidate L-BFGS history (position didn't change,
        // so s_k would be zero). Reset and fall back to steepest descent next step.
        if (!accepted) {
            lbfgs_count = 0;
            lbfgs_newest = 0;
            hasPrev = false;
        }
    }
}

// ============================================================================
// MARK: - Batched GA Kernels for Virtual Screening
// ============================================================================

/// Batched scoring: multiple ligands' populations scored simultaneously.
kernel void scorePosesBatched(
    device DockPose              *poses          [[buffer(0)]],
    constant DockLigandAtom      *allAtoms       [[buffer(1)]],
    device const half            *affinityMaps   [[buffer(2)]],
    device const int32_t         *typeIndexLookup [[buffer(3)]],
    constant GridParams          &gridParams     [[buffer(4)]],
    constant BatchedGAParams     &batchParams    [[buffer(5)]],
    constant TorsionEdge         *allTorsionEdges [[buffer(6)]],
    constant int32_t             *allMovingIndices [[buffer(7)]],
    constant uint32_t            *allIntraPairs   [[buffer(8)]],
    constant BatchedGALigandInfo *ligandInfo      [[buffer(9)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= batchParams.totalPoses) return;
    uint ligIdx = tid / batchParams.populationSizePerLigand;
    if (ligIdx >= batchParams.numLigands) return;

    BatchedGALigandInfo info = ligandInfo[ligIdx];
    constant DockLigandAtom *myAtoms = allAtoms + info.atomStart;
    constant TorsionEdge *myEdges = allTorsionEdges + info.torsionEdgeStart;
    constant int32_t *myMoving = allMovingIndices + info.movingIndicesStart;
    constant uint32_t *myPairs = allIntraPairs + info.pairListStart;

    device DockPose &pose = poses[tid];
    uint nA = min(info.atomCount, 128u);
    uint nTor = min(info.torsionEdgeCount, 32u);

    float3 positions[128];
    for (uint a = 0; a < nA; a++)
        positions[a] = quatRotate(pose.rotation, myAtoms[a].position) + pose.translation;
    applyTorsions(positions, nA, pose, myEdges, myMoving, nTor);

    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;
    float totalE = 0.0f, penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float oopP = outOfGridPenalty(positions[a], gridMin, gridMax);
        if (oopP > 0.0f) { penalty += oopP; continue; }
        totalE += sampleTypedAffinityMap(affinityMaps, typeIndexLookup, myAtoms[a].vinaType, positions[a], gridParams);
    }

    float intra = intramolecularLigandEnergy(positions, myAtoms, nA, myPairs, info.numPairs) - info.referenceIntraEnergy;
    float norm = 1.0f / (1.0f + wRotEntropy * float(nTor) / 5.0f);
    pose.stericEnergy = totalE;
    pose.clashPenalty = wPenalty * penalty + intra;
    pose.energy = totalE * norm + pose.clashPenalty;
}

/// Batched MC perturbation for virtual screening.
kernel void mcPerturbBatched(
    device DockPose              *perturbed     [[buffer(0)]],
    device const DockPose        *current       [[buffer(1)]],
    constant BatchedGAParams     &batchParams   [[buffer(2)]],
    constant GridParams          &gridParams    [[buffer(3)]],
    constant BatchedGALigandInfo *ligandInfo    [[buffer(4)]],
    uint                          tid           [[thread_position_in_grid]])
{
    if (tid >= batchParams.totalPoses) return;
    uint ligIdx = tid / batchParams.populationSizePerLigand;
    if (ligIdx >= batchParams.numLigands) return;

    BatchedGALigandInfo info = ligandInfo[ligIdx];
    uint nTor = min(info.torsionEdgeCount, 32u);

    uint seed = tid * 314159u + batchParams.generation * 271828u + 42u;
    device const DockPose &src = current[tid];
    device DockPose &dst = perturbed[tid];
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    dst = src;
    uint which = (2u + nTor) > 0u ? uint(gpuRandom(seed, 0) * float(2u + nTor)) % (2u + nTor) : 0u;

    if (which == 0u) {
        dst.translation = clamp(src.translation + batchParams.translationStep * randomInsideUnitSphere(seed, 10), gMin, gMax);
    } else if (which == 1u) {
        float radius = max(info.ligandRadius, 1.0f);
        float3 rv = max(batchParams.rotationStep, batchParams.translationStep / radius) * randomInsideUnitSphere(seed, 20);
        float a = length(rv);
        if (a > 1e-6f) {
            float3 ax = rv / a; float ha = a * 0.5f;
            float4 dq = float4(ax * sin(ha), cos(ha)); float4 sq = src.rotation;
            dst.rotation = normalize(float4(
                dq.w*sq.x+dq.x*sq.w+dq.y*sq.z-dq.z*sq.y, dq.w*sq.y-dq.x*sq.z+dq.y*sq.w+dq.z*sq.x,
                dq.w*sq.z+dq.x*sq.y-dq.y*sq.x+dq.z*sq.w, dq.w*sq.w-dq.x*sq.x-dq.y*sq.y-dq.z*sq.z));
        }
    } else if (which - 2u < nTor) {
        dst.torsions[which - 2u] = gpuRandom(seed, 30 + which - 2u) * 2.0f * M_PI_F - M_PI_F;
    }
    dst.numTorsions = int(nTor); dst.generation = src.generation + 1; dst.energy = 1e10f;
}

/// Batched Metropolis acceptance for virtual screening.
kernel void metropolisAcceptBatched(
    device DockPose              *current       [[buffer(0)]],
    device const DockPose        *candidate     [[buffer(1)]],
    device DockPose              *best          [[buffer(2)]],
    constant BatchedGAParams     &batchParams   [[buffer(3)]],
    uint                          tid           [[thread_position_in_grid]])
{
    if (tid >= batchParams.totalPoses) return;
    device DockPose &cur = current[tid];
    device DockPose &bp = best[tid];
    DockPose cand = candidate[tid];
    bool cv = isfinite(cand.energy) && cand.energy < 1e9f;
    bool accept = false;
    if (cv) {
        if (!isfinite(cur.energy) || cur.energy >= 1e9f || cand.energy < cur.energy) {
            accept = true;
        } else {
            float T = max(batchParams.mcTemperature, 0.01f);
            uint seed = tid * 1103515245u + batchParams.generation * 12345u;
            accept = gpuRandom(seed, 0) < exp((cur.energy - cand.energy) / T);
        }
    }
    if (accept) { cur = cand; if (cand.energy < bp.energy || !isfinite(bp.energy)) bp = cand; }
}

// ============================================================================
// MARK: - Parallel Tempering / Replica Exchange Monte Carlo
// ============================================================================

/// Temperature-aware MC perturbation for replica exchange.
/// Each pose belongs to a replica (replicaIdx = tid / populationPerReplica).
/// Perturbation step sizes scale with sqrt(T/T_min) — higher T replicas explore wider.
kernel void mcPerturbReplica(
    device DockPose            *perturbed   [[buffer(0)]],
    device const DockPose      *current     [[buffer(1)]],
    constant GAParams          &gaParams    [[buffer(2)]],
    constant GridParams        &gridParams  [[buffer(3)]],
    constant ReplicaParams     &repParams   [[buffer(4)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= repParams.totalPoses) return;

    uint replicaIdx = tid / repParams.populationPerReplica;
    float T = repParams.temperatures[min(replicaIdx, MAX_REPLICAS - 1u)];
    float Tmin = repParams.temperatures[0];
    float tempScale = sqrt(max(T / max(Tmin, 0.01f), 1.0f));

    uint seed = tid * 314159u + repParams.swapGeneration * 271828u + replicaIdx * 999983u + 42u;
    device const DockPose &src = current[tid];
    device DockPose &dst = perturbed[tid];

    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    dst = src;

    uint mutableEntities = 2u + gaParams.numTorsions;
    uint which = mutableEntities > 0u
        ? uint(gpuRandom(seed, 0) * float(mutableEntities)) % mutableEntities
        : 0u;

    if (which == 0u) {
        // Translation perturbation — scaled by temperature
        float step = gaParams.translationStep * tempScale;
        float3 delta = step * randomInsideUnitSphere(seed, 10);
        dst.translation = clamp(src.translation + delta, gMin, gMax);
    } else if (which == 1u) {
        // Rotation perturbation — scaled by temperature
        float radius = max(gaParams.ligandRadius, 1.0f);
        float rotSpread = max(gaParams.rotationStep, gaParams.translationStep / radius) * tempScale;
        float3 rotVec = rotSpread * randomInsideUnitSphere(seed, 20);
        float angle = length(rotVec);
        if (angle > 1e-6f) {
            float3 axis = rotVec / angle;
            float halfA = angle * 0.5f;
            float4 dq = float4(axis * sin(halfA), cos(halfA));
            float4 sq = src.rotation;
            dst.rotation = normalize(float4(
                dq.w*sq.x + dq.x*sq.w + dq.y*sq.z - dq.z*sq.y,
                dq.w*sq.y - dq.x*sq.z + dq.y*sq.w + dq.z*sq.x,
                dq.w*sq.z + dq.x*sq.y - dq.y*sq.x + dq.z*sq.w,
                dq.w*sq.w - dq.x*sq.x - dq.y*sq.y - dq.z*sq.z
            ));
        }
    } else {
        // Torsion perturbation — full random resample at high T, small perturbation at low T
        uint torsionIndex = which - 2u;
        if (torsionIndex < gaParams.numTorsions) {
            if (tempScale > 1.5f) {
                // High temperature: random resample for broad exploration
                dst.torsions[torsionIndex] = gpuRandom(seed, 30 + torsionIndex) * 2.0f * M_PI_F - M_PI_F;
            } else {
                // Low temperature: Gaussian-like perturbation
                float perturbation = (gpuRandom(seed, 30 + torsionIndex) - 0.5f) * gaParams.torsionStep * tempScale;
                float newVal = src.torsions[torsionIndex] + perturbation;
                // Wrap to [-pi, pi]
                newVal = newVal - 2.0f * M_PI_F * floor((newVal + M_PI_F) / (2.0f * M_PI_F));
                dst.torsions[torsionIndex] = newVal;
            }
        }
    }

    dst.numTorsions = src.numTorsions;
    dst.generation = src.generation + 1;
    dst.energy = 1e10f;
    dst.stericEnergy = 0.0f;
    dst.hydrophobicEnergy = 0.0f;
    dst.hbondEnergy = 0.0f;
    dst.torsionPenalty = 0.0f;
    dst.clashPenalty = 0.0f;
}

/// Temperature-aware Metropolis acceptance for replica exchange.
/// Acceptance probability uses the replica's own temperature.
kernel void metropolisAcceptReplica(
    device DockPose            *current     [[buffer(0)]],
    device const DockPose      *candidate   [[buffer(1)]],
    device DockPose            *best        [[buffer(2)]],
    constant GAParams          &gaParams    [[buffer(3)]],
    constant ReplicaParams     &repParams   [[buffer(4)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= repParams.totalPoses) return;

    uint replicaIdx = tid / repParams.populationPerReplica;
    float T = max(repParams.temperatures[min(replicaIdx, MAX_REPLICAS - 1u)], 1e-4f);

    device DockPose &cur = current[tid];
    device DockPose &bestPose = best[tid];
    DockPose cand = candidate[tid];

    bool candValid = isfinite(cand.energy) && cand.energy < 1e9f;
    bool curValid = isfinite(cur.energy) && cur.energy < 1e9f;
    bool accept = false;

    if (candValid) {
        if (!curValid || cand.energy < cur.energy) {
            accept = true;
        } else {
            float acceptance = exp((cur.energy - cand.energy) / T);
            uint seed = tid * 92837111u + repParams.swapGeneration * 689287499u + replicaIdx * 456789u + 17u;
            accept = gpuRandom(seed, 0) < min(acceptance, 1.0f);
        }
    }

    if (accept) {
        cur = cand;
    }

    bool bestValid = isfinite(bestPose.energy) && bestPose.energy < 1e9f;
    bool nowValid = isfinite(cur.energy) && cur.energy < 1e9f;
    if (nowValid && (!bestValid || cur.energy < bestPose.energy)) {
        bestPose = cur;
    }
}

/// Replica exchange swap kernel.
/// Dispatched with one thread per adjacent replica pair.
/// Uses checkerboard pattern: even generations swap pairs (0,1),(2,3),...
/// odd generations swap pairs (1,2),(3,4),... to avoid race conditions.
kernel void replicaSwap(
    device DockPose            *population  [[buffer(0)]],
    device DockPose            *best        [[buffer(1)]],
    constant ReplicaParams     &repParams   [[buffer(2)]],
    uint                        tid         [[thread_position_in_grid]])
{
    uint numPairs = repParams.numReplicas - 1u;
    if (tid >= numPairs) return;

    // Checkerboard: even gen swaps even pairs, odd gen swaps odd pairs
    uint parity = repParams.swapGeneration & 1u;
    if ((tid & 1u) != parity) return;

    uint replicaI = tid;
    uint replicaJ = tid + 1u;

    // Find the best (lowest energy) pose in each replica
    uint startI = replicaI * repParams.populationPerReplica;
    uint startJ = replicaJ * repParams.populationPerReplica;
    uint ppReplica = repParams.populationPerReplica;

    float bestEI = 1e10f;
    uint bestIdxI = startI;
    for (uint k = 0; k < ppReplica; k++) {
        float e = population[startI + k].energy;
        if (isfinite(e) && e < bestEI) {
            bestEI = e;
            bestIdxI = startI + k;
        }
    }

    float bestEJ = 1e10f;
    uint bestIdxJ = startJ;
    for (uint k = 0; k < ppReplica; k++) {
        float e = population[startJ + k].energy;
        if (isfinite(e) && e < bestEJ) {
            bestEJ = e;
            bestIdxJ = startJ + k;
        }
    }

    // Skip if either replica has no valid poses
    if (!isfinite(bestEI) || !isfinite(bestEJ)) return;
    if (bestEI >= 1e9f || bestEJ >= 1e9f) return;

    float Ti = max(repParams.temperatures[replicaI], 1e-4f);
    float Tj = max(repParams.temperatures[replicaJ], 1e-4f);
    float betaI = 1.0f / Ti;
    float betaJ = 1.0f / Tj;

    // Metropolis swap criterion: P = min(1, exp((βi - βj)(Ei - Ej)))
    float delta = (betaI - betaJ) * (bestEI - bestEJ);
    float P = min(1.0f, exp(delta));

    uint seed = tid * 482711u + repParams.swapGeneration * 937373u + 777u;
    float r = gpuRandom(seed, 0);

    if (r < P) {
        // Swap entire replica populations by exchanging all poses
        for (uint k = 0; k < ppReplica; k++) {
            DockPose tmp = population[startI + k];
            population[startI + k] = population[startJ + k];
            population[startJ + k] = tmp;

            DockPose tmpBest = best[startI + k];
            best[startI + k] = best[startJ + k];
            best[startJ + k] = tmpBest;
        }
    }
}
