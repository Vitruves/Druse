// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"

// MARK: - Grid Map Computation (Vina-style)
// ============================================================================

kernel void computeReceptorGrids(
    device half *stericMap [[buffer(0)]],
    constant GridProteinAtom *proteinAtoms [[buffer(1)]],
    constant GridParams &params [[buffer(2)]],
    device half *hydrophobicMap [[buffer(3)]],
    device half *hbondMap [[buffer(4)]],
    device half *electrostaticMap [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    bool active = tid < params.totalPoints;
    float3 gridPos = gridPosition(active ? tid : 0u, params);
    float steric = 0.0f, hydrophobic = 0.0f, hbond = 0.0f, potential = 0.0f;
    threadgroup float3 positions[kAtomTileSize];
    threadgroup float radii[kAtomTileSize];
    threadgroup float charges[kAtomTileSize];
    threadgroup int types[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint count = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < count) {
            positions[lid] = proteinAtoms[base + lid].position;
            radii[lid] = proteinAtoms[base + lid].vdwRadius;
            charges[lid] = proteinAtoms[base + lid].charge;
            types[lid] = proteinAtoms[base + lid].vinaType;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = 0; index < count; index++) {
            float radius = distance(gridPos, positions[index]);
            if (radius > 12.0f) continue;
            float coulombRadius = max(radius, 0.8f);
            potential += 332.0f * charges[index] / (4.0f * coulombRadius * coulombRadius);
            if (radius > 8.0f) continue;
            float gap = max(radius, 0.1f) - (radii[index] + kProbeRadius);
            float nearGauss = gap * 2.0f;
            float farGauss = (gap - 3.0f) * 0.5f;
            float overlap = min(gap, 0.0f);
            steric += wGauss1 * exp(-(nearGauss * nearGauss))
                    + wGauss2 * exp(-(farGauss * farGauss)) + wRepulsion * (overlap * overlap);
            if (xsIsHydrophobic(types[index]))
                hydrophobic += wHydrophobic * clamp(1.5f - gap, 0.0f, 1.0f);
            if (xsIsDonor(types[index]) || xsIsAcceptor(types[index]))
                hbond += wHBond * clamp(-gap / 0.7f, 0.0f, 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        stericMap[tid] = half(clamp(steric, -100.0f, 100.0f));
        hydrophobicMap[tid] = half(clamp(hydrophobic, -100.0f, 100.0f));
        hbondMap[tid] = half(clamp(hbond, -100.0f, 100.0f));
        electrostaticMap[tid] = half(clamp(potential, -50.0f, 50.0f));
    }
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
    if (params.numAffinityTypes == 0 || params.totalPoints == 0) return;  // uniform across the dispatch
    uint totalEntries = params.totalPoints * params.numAffinityTypes;
    // Threads past the end must not return early: they still load their tile slot
    // and reach every threadgroup barrier (the last threadgroup is partially filled).
    bool active = tid < totalEntries;
    uint entry = active ? tid : 0u;

    uint pointIdx = entry % params.totalPoints;
    uint typeIdx = entry / params.totalPoints;
    int probeType = affinityTypes[typeIdx];
    float3 gridPos = gridPosition(pointIdx, params);
    float totalE = 0.0f;
    // SoA threadgroup layout eliminates bank conflicts vs AoS GridProteinAtom
    threadgroup float3 tilePositions[kAtomTileSize];
    threadgroup int    tileVinaTypes[kAtomTileSize];

    for (uint base = 0; base < params.numProteinAtoms; base += kAtomTileSize) {
        uint tileCount = min(kAtomTileSize, params.numProteinAtoms - base);
        if (lid < tileCount) {
            uint atomIdx = base + lid;
            tilePositions[lid] = proteinAtoms[atomIdx].position;
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

    if (active) affinityMaps[typeIdx * params.totalPoints + pointIdx] = half(clamp(totalE, -100.0f, 100.0f));
}
