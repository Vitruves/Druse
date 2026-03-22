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

/// Trilinear interpolation on a 3D grid map. Returns smooth quadratic
/// out-of-grid penalty when the query point lies outside the grid bounds.
inline float trilinearInterpolate(
    device const float *gridMap,
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

    float c000 = gridMap[iz * nx * ny + iy * nx + ix];
    float c100 = gridMap[iz * nx * ny + iy * nx + ix + 1];
    float c010 = gridMap[iz * nx * ny + (iy+1) * nx + ix];
    float c110 = gridMap[iz * nx * ny + (iy+1) * nx + ix + 1];
    float c001 = gridMap[(iz+1) * nx * ny + iy * nx + ix];
    float c101 = gridMap[(iz+1) * nx * ny + iy * nx + ix + 1];
    float c011 = gridMap[(iz+1) * nx * ny + (iy+1) * nx + ix];
    float c111 = gridMap[(iz+1) * nx * ny + (iy+1) * nx + ix + 1];

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
    device const float *affinityMaps,
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
    device const float *map = affinityMaps + uint(mapIndex) * gp.totalPoints;
    return trilinearInterpolate(map, pos, gp.origin, gp.spacing, gp.dims);
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

/// Compute Vina-style ligand intramolecular energy over all non-excluded heavy-atom pairs.
/// A constant reference value is subtracted later so rigid, pose-invariant contributions do
/// not distort absolute docking energies while torsion-dependent strain still affects ranking.
inline float intramolecularLigandEnergy(
    thread float3              *positions,
    constant DockLigandAtom    *ligandAtoms,
    uint                        nAtoms,
    constant uint32_t          *exclusionMask,
    uint                        maxAtoms)
{
    float total = 0.0f;
    for (uint i = 0; i < nAtoms; i++) {
        for (uint j = i + 1; j < nAtoms; j++) {
            // Check exclusion bitmask (row-major upper triangle)
            uint pairIdx = i * maxAtoms + j;
            uint word = pairIdx / 32;
            uint bit  = pairIdx % 32;
            if (exclusionMask[word] & (1u << bit)) continue;

            float r = distance(positions[i], positions[j]);
            total += vinaPairEnergy(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, r);
        }
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
    device float               *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float totalE = 0.0f;
    threadgroup GridProteinAtom atomTile[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            atomTile[lid] = proteinAtoms[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            float r = distance(gridPos, atomTile[i].position);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);  // avoid singularity

            float d = r - (atomTile[i].vdwRadius + kProbeRadius);

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

    gridMap[tid] = clamp(totalE, -100.0f, 100.0f);
}

/// Compute hydrophobic grid map (legacy approximate Vina hydrophobic contact term).
/// Only protein atoms with hydrophobic XS types contribute.
/// d = distance - (R_protein + R_probe).
/// If d < 0.5: value = 1.0; if d > 1.5: value = 0.0; else linear ramp.
/// Weighted by wHydrophobic.
kernel void computeHydrophobicGrid(
    device float               *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float totalE = 0.0f;
    threadgroup GridProteinAtom atomTile[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            atomTile[lid] = proteinAtoms[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            int vt = atomTile[i].vinaType;
            if (!xsIsHydrophobic(vt)) continue;

            float r = distance(gridPos, atomTile[i].position);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);

            float d = r - (atomTile[i].vdwRadius + kProbeRadius);

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

    gridMap[tid] = clamp(totalE, -100.0f, 100.0f);
}

/// Compute hydrogen bond grid map (legacy approximate Vina H-bond term).
/// Only protein atoms that are donors or acceptors contribute.
/// d = distance - (R_protein + R_probe).
/// If d < -0.7: value = 1.0; if d > 0: value = 0.0; else linear ramp.
/// Weighted by wHBond.
kernel void computeHBondGrid(
    device float               *gridMap      [[buffer(0)]],
    constant GridProteinAtom   *proteinAtoms [[buffer(1)]],
    constant GridParams        &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]],
    uint                        lid          [[thread_index_in_threadgroup]])
{
    if (tid >= params.totalPoints) return;
    float3 gridPos = gridPosition(tid, params);
    float totalE = 0.0f;
    threadgroup GridProteinAtom atomTile[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            atomTile[lid] = proteinAtoms[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            int vt = atomTile[i].vinaType;
            bool isHBondCapable = xsIsDonor(vt) || xsIsAcceptor(vt);
            if (!isHBondCapable) continue;

            float r = distance(gridPos, atomTile[i].position);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);

            float d = r - (atomTile[i].vdwRadius + kProbeRadius);

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

    gridMap[tid] = clamp(totalE, -100.0f, 100.0f);
}

/// Compute exact AutoDock Vina affinity maps for the requested ligand XS types.
/// Each map stores the full upstream Vina pairwise potential summed over protein atoms.
kernel void computeVinaAffinityMaps(
    device float               *affinityMaps  [[buffer(0)]],
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
    threadgroup GridProteinAtom atomTile[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            atomTile[lid] = proteinAtoms[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < tileCount; i++) {
            float r = distance(gridPos, atomTile[i].position);
            if (r > 8.0f) continue;
            r = max(r, 0.1f);
            totalE += vinaPairEnergy(probeType, atomTile[i].vinaType, r);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    affinityMaps[typeIdx * params.totalPoints + pointIdx] = clamp(totalE, -100.0f, 100.0f);
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
// MARK: - Pose Scoring
// ============================================================================

/// Score all poses using exact Vina XS-type affinity maps with torsional flexibility.
/// One thread per pose.
kernel void scorePoses(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    device const float         *affinityMaps  [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams    [[buffer(4)]],
    constant GAParams          &gaParams      [[buffer(5)]],
    constant TorsionEdge       *torsionEdges  [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *exclusionMask [[buffer(8)]],
    uint                        tid           [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    // Transform atoms: rigid body + torsions (stack allocation, max 128 atoms)
    float3 positions[128];
    uint nA = min(nAtoms, 128u);

    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    // Score against grid maps
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];

        // Check if atom is outside grid
        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;  // skip grid lookup for out-of-bounds atoms
        }
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gridParams
        );
    }

    // Vina-style ligand internal energy, referenced to the input conformer.
    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, exclusionMask, 128)
        - gaParams.referenceIntraEnergy;

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

    // Total Vina score = normalized intermolecular + boundary penalty + referenced internal energy
    pose.energy = normalizedE + pose.clashPenalty;
}

/// Rescore poses against explicit receptor atoms instead of interpolated affinity maps.
/// This mirrors Vina's late non-cache pose rescoring more closely for top basin representatives.
kernel void scorePosesExplicit(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    constant GridProteinAtom   *proteinAtoms  [[buffer(2)]],
    constant GridParams        &gridParams    [[buffer(3)]],
    constant GAParams          &gaParams      [[buffer(4)]],
    constant TorsionEdge       *torsionEdges  [[buffer(5)]],
    constant int32_t           *movingIndices [[buffer(6)]],
    constant uint32_t          *exclusionMask [[buffer(7)]],
    uint                        tid           [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
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

        for (uint p = 0; p < gridParams.numProteinAtoms; p++) {
            float dist = distance(r, proteinAtoms[p].position);
            if (dist < 8.0f) {
                VinaTerms terms = vinaPairEnergyDecomposed(ligandAtoms[a].vinaType, proteinAtoms[p].vinaType, dist);
                totalSteric += terms.steric;
                totalHydrophobic += terms.hydrophobic;
                totalHBond += terms.hbond;
            }
        }
    }

    float totalIntermolecular = totalSteric + totalHydrophobic + totalHBond;
    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nAtoms, exclusionMask, 128)
        - gaParams.referenceIntraEnergy;
    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    float normalizedE = totalIntermolecular * normFactor;

    pose.stericEnergy      = totalSteric;
    pose.hydrophobicEnergy = totalHydrophobic;
    pose.hbondEnergy       = totalHBond;
    pose.torsionPenalty    = normalizedE - totalIntermolecular;
    pose.clashPenalty      = wPenalty * penalty + intraDelta;
    pose.energy            = normalizedE + pose.clashPenalty;
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
    device const float          *stericGrid    [[buffer(3)]],
    device const float          *hydrophobGrid [[buffer(4)]],
    device const float          *hbondGrid     [[buffer(5)]],
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

/// Initialize random poses within the pocket-centered search box.
/// 70% focused near the pocket center, 30% exploratory across the search box.
kernel void initializePopulation(
    device DockPose            *poses        [[buffer(0)]],
    constant GridParams        &gridParams   [[buffer(1)]],
    constant GAParams          &gaParams     [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint seed = tid * 747796405u + gaParams.generation * 2891336453u + 67890u;
    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;

    // Balanced exploration/exploitation:
    // - 50% focused near pocket center
    // - 50% spread across the full search box
    float3 spread;
    float3 searchSize = halfExtent * 2.0f;
    float minDim = min(searchSize.x, min(searchSize.y, searchSize.z));
    float focusSpread = min(minDim * 0.5f, 12.0f);

    if (tid < gaParams.populationSize * 7 / 10) {
        spread = float3(focusSpread);
    } else {
        spread = halfExtent;
    }

    poses[tid].translation = center + float3(
        (gpuRandom(seed, 0) * 2.0f - 1.0f) * spread.x,
        (gpuRandom(seed, 1) * 2.0f - 1.0f) * spread.y,
        (gpuRandom(seed, 2) * 2.0f - 1.0f) * spread.z
    );

    // Uniform random quaternion (Shoemake method)
    float u1 = gpuRandom(seed, 3);
    float u2 = gpuRandom(seed, 4) * 2.0f * M_PI_F;
    float u3 = gpuRandom(seed, 5) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    poses[tid].rotation = float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));

    for (uint t = 0; t < gaParams.numTorsions; t++) {
        poses[tid].torsions[t] = gpuRandom(seed, 6 + t) * 2.0f * M_PI_F - M_PI_F;
    }

    poses[tid].numTorsions = int(gaParams.numTorsions);
    poses[tid].generation = 0;
    poses[tid].energy = 1e10f;
    poses[tid].stericEnergy = 0.0f;
    poses[tid].hydrophobicEnergy = 0.0f;
    poses[tid].hbondEnergy = 0.0f;
    poses[tid].torsionPenalty = 0.0f;
    poses[tid].clashPenalty = 0.0f;
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
    device const float         *affinityMaps,
    device const int32_t       *typeIndexLookup,
    constant GridParams        &gp,
    float                       referenceIntraEnergy,
    constant uint32_t          *exclusionMask = nullptr)
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

    // Intramolecular clashes: prevent ligand from folding into impossible conformations
    float intraDelta = 0.0f;
    if (exclusionMask) {
        intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nAtoms, exclusionMask, 128)
            - referenceIntraEnergy;
    }

    // Apply the upstream Vina conf-independent torsion divisor.
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
    device const float         *affinityMaps,
    device const int32_t       *typeIndexLookup,
    constant GridParams        &gp,
    float                       referenceIntraEnergy,
    constant uint32_t          *exclusionMask = nullptr)
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
    if (exclusionMask) {
        intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, exclusionMask, 128)
            - referenceIntraEnergy;
    }

    float nRotF = float(nTorsions);
    float normFactor = 1.0f / (1.0f + wRotEntropy * nRotF / 5.0f);
    return totalIntermolecular * normFactor + wPenalty * boundaryPenalty + intraDelta;
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
    device const float         *affinityMaps [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams   [[buffer(4)]],
    constant GAParams          &gaParams     [[buffer(5)]],
    constant TorsionEdge       *torsionEdges [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *exclusionMask [[buffer(8)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = min(gaParams.numLigandAtoms, 128u);
    uint nTor = min(gaParams.numTorsions, 32u);
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    float stepSize = 0.08f;
    float h = 0.03f;  // finite difference step

    int maxSteps = max(int(gaParams.localSearchSteps), 1);
    for (int step = 0; step < maxSteps; step++) {
        // Current energy
        float baseE = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                    torsionEdges, movingIndices,
                                    affinityMaps, typeIndexLookup, gridParams,
                                    gaParams.referenceIntraEnergy, exclusionMask);

        // ---- Translation gradient (3 DOF) ----
        float gradT[3];
        for (int dim = 0; dim < 3; dim++) {
            float3 origT = pose.translation;
            pose.translation[dim] = origT[dim] + h;
            float ePlus = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                        torsionEdges, movingIndices,
                                        affinityMaps, typeIndexLookup, gridParams,
                                        gaParams.referenceIntraEnergy, exclusionMask);
            pose.translation[dim] = origT[dim] - h;
            float eMinus = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                         torsionEdges, movingIndices,
                                         affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, exclusionMask);
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
            float ePlus = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                        torsionEdges, movingIndices,
                                        affinityMaps, typeIndexLookup, gridParams,
                                        gaParams.referenceIntraEnergy, exclusionMask);

            // Apply negative rotation
            pose.rotation = normalize(float4(
                dqMinus.w*origRot.x + dqMinus.x*origRot.w + dqMinus.y*origRot.z - dqMinus.z*origRot.y,
                dqMinus.w*origRot.y - dqMinus.x*origRot.z + dqMinus.y*origRot.w + dqMinus.z*origRot.x,
                dqMinus.w*origRot.z + dqMinus.x*origRot.y - dqMinus.y*origRot.x + dqMinus.z*origRot.w,
                dqMinus.w*origRot.w - dqMinus.x*origRot.x - dqMinus.y*origRot.y - dqMinus.z*origRot.z
            ));
            float eMinus = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                         torsionEdges, movingIndices,
                                         affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, exclusionMask);

            pose.rotation = origRot;
            gradR[dim] = (ePlus - eMinus) / (2.0f * hRot);
        }

        // ---- Torsion gradients ----
        float gradTor[32];
        float hTor = 0.02f;
        for (uint t = 0; t < nTor; t++) {
            float origTor = pose.torsions[t];
            pose.torsions[t] = origTor + hTor;
            float ePlus = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                        torsionEdges, movingIndices,
                                        affinityMaps, typeIndexLookup, gridParams,
                                        gaParams.referenceIntraEnergy, exclusionMask);
            pose.torsions[t] = origTor - hTor;
            float eMinus = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                         torsionEdges, movingIndices,
                                         affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, exclusionMask);
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

        // Normalize and clamp
        float scale = min(stepSize / gradMag, stepSize);

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

        float newE = evaluatePose(pose, ligandAtoms, nAtoms, nTor,
                                   torsionEdges, movingIndices,
                                   affinityMaps, typeIndexLookup, gridParams,
                                   gaParams.referenceIntraEnergy, exclusionMask);

        if (newE < baseE) {
            pose.energy = newE;
            stepSize = min(stepSize * 1.2f, 0.5f);
        } else {
            // Rollback
            pose.translation = oldT;
            pose.rotation = oldRot;
            for (uint t = 0; t < nTor; t++) pose.torsions[t] = oldTorsions[t];
            pose.energy = baseE;
            stepSize *= 0.5f;
        }
        if (stepSize < 0.001f) break;
    }
}
