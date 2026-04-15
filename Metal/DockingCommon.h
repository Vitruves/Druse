// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#ifndef DOCKING_COMMON_H
#define DOCKING_COMMON_H

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

/// Soft Vina quality gate: attenuates Drusina only for badly clashing poses (vinaE > 0).
/// Much softer than the original gate (offset=1, slope=0.35): this only kicks in when
/// Vina energy is positive (steric clashes), allowing Drusina to contribute for all
/// reasonable poses. Prevents Drusina from creating false attractors in empty space.
inline float drusinaVinaQuality(float vinaE, constant DrusinaParams &params) {
    // Sigmoid centered at vinaE = +2 kcal/mol, gentle slope.
    // quality ≈ 1.0 for vinaE < 0 (any favorable pose)
    // quality ≈ 0.5 at vinaE = +2 (moderate clash)
    // quality → 0.0 for vinaE >> +2 (severe clash)
    return 1.0f / (1.0f + exp(0.5f * (vinaE - 2.0f)));
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

/// AutoDock-Vina conf-independent rotatable-bond normalization.
/// Vina's num_tors_div term evaluates to E / (1 + weight_rot * num_tors).
inline float vinaRotNormalization(float nTorsions) {
    return 1.0f / (1.0f + wRotEntropy * nTorsions);
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
// MARK: - Pose Evaluation Helpers
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
    float normFactor = vinaRotNormalization(nRotF);
    return (totalIntermolecular + intraDelta) * normFactor + wPenalty * penalty;
}

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
    float normFactor = vinaRotNormalization(nRotF);
    return (totalIntermolecular + intraDelta) * normFactor + wPenalty * boundaryPenalty;
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
    float normFactor = vinaRotNormalization(nRotF);
    return (totalIntermolecular + intraDelta) * normFactor + wPenalty * boundaryPenalty + constraintPen;
}

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
    float normFactor = vinaRotNormalization(nRotF);

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
        for (uint a = 0; a < nA; a++) forces[a] -= intraGrad[a] * normFactor;  // force = -gradient
    }
    float intraDelta = intraE - referenceIntraEnergy;
    float totalEnergy = (totalIntermolecular + intraDelta) * normFactor;

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

// ============================================================================
// MARK: - GPU Random Number Generation
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

#endif // DOCKING_COMMON_H
