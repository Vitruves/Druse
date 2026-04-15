// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"

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
