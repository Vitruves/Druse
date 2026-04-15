// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"

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
