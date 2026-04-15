// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Flexible Residue Grid Proxy
//
// When sidechain atoms are excluded from the receptor for flexible docking,
// the affinity grid develops attractive "holes" at their former positions.
// This kernel injects soft repulsive proxies into ALL affinity maps at the
// reference positions of excluded sidechain atoms, preventing the ligand
// from drifting into the void.
//
// The repulsive profile matches the Vina repulsion term: d < 0 → wRepulsion * d^2,
// where d = distance - (flexAtomRadius + probeRadius).
//
// This kernel runs ONCE after grid map computation and BEFORE docking begins.
// It modifies the affinity maps in-place (additive).
// ============================================================================

constant float kFlexProbeRadius = 1.5f;     // average probe radius (Å)
constant float kFlexRepulsionWeight = 0.84f; // matches wRepulsion = 0.840245
constant float kFlexRepulsionCutoff = 4.0f;  // only modify grid points within 4 Å

/// Inject repulsive proxy energy into typed affinity maps at flex atom reference positions.
/// Each thread handles one (grid point, affinity type) pair.
/// Grid points far from all flex atoms are untouched.
kernel void injectFlexRepulsion(
    device half                 *affinityMaps   [[buffer(0)]],
    constant FlexSidechainAtom  *flexAtoms      [[buffer(1)]],
    constant GridParams         &params         [[buffer(2)]],
    constant FlexParams         &flexParams     [[buffer(3)]],
    uint                         tid            [[thread_position_in_grid]])
{
    uint totalEntries = params.totalPoints * params.numAffinityTypes;
    if (tid >= totalEntries || flexParams.numFlexAtoms == 0) return;

    uint pointIdx = tid % params.totalPoints;
    uint typeIdx = tid / params.totalPoints;

    // Compute grid position for this point
    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint iz = pointIdx / (nx * ny);
    uint iy = (pointIdx - iz * nx * ny) / nx;
    uint ix = pointIdx - iz * nx * ny - iy * nx;
    float3 gridPos = params.origin + float3(float(ix), float(iy), float(iz)) * params.spacing;

    float repulsion = 0.0f;

    for (uint fa = 0; fa < flexParams.numFlexAtoms; fa++) {
        // Skip backbone pivot pseudo-atoms (vinaType == -1)
        if (flexAtoms[fa].vinaType < 0) continue;

        float3 flexPos = float3(flexAtoms[fa].referencePosition);
        float r = distance(gridPos, flexPos);

        if (r > kFlexRepulsionCutoff) continue;
        r = max(r, 0.1f);

        // Use the flex atom's VdW radius approximation from its Vina type
        // (XS radii range from 1.5 to 2.0 Å; use 1.7 as default)
        float flexRadius = 1.7f;

        float d = r - (flexRadius + kFlexProbeRadius);

        if (d < 0.0f) {
            // Quadratic repulsion: same as Vina repulsion term
            repulsion += kFlexRepulsionWeight * d * d;
        }
    }

    if (repulsion > 0.0f) {
        uint mapIdx = typeIdx * params.totalPoints + pointIdx;
        float current = float(affinityMaps[mapIdx]);
        affinityMaps[mapIdx] = half(clamp(current + repulsion, -100.0f, 100.0f));
    }
}

/// Inject repulsive proxy into the three legacy grids (steric, hydrophobic, hbond).
/// Each thread handles one grid point. Adds repulsion to the steric grid only
/// (hydrophobic and hbond are type-specific and should not get generic repulsion).
kernel void injectFlexRepulsionLegacy(
    device half                 *stericGrid     [[buffer(0)]],
    constant FlexSidechainAtom  *flexAtoms      [[buffer(1)]],
    constant GridParams         &params         [[buffer(2)]],
    constant FlexParams         &flexParams     [[buffer(3)]],
    uint                         tid            [[thread_position_in_grid]])
{
    if (tid >= params.totalPoints || flexParams.numFlexAtoms == 0) return;

    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint iz = tid / (nx * ny);
    uint iy = (tid - iz * nx * ny) / nx;
    uint ix = tid - iz * nx * ny - iy * nx;
    float3 gridPos = params.origin + float3(float(ix), float(iy), float(iz)) * params.spacing;

    float repulsion = 0.0f;

    for (uint fa = 0; fa < flexParams.numFlexAtoms; fa++) {
        if (flexAtoms[fa].vinaType < 0) continue;

        float3 flexPos = float3(flexAtoms[fa].referencePosition);
        float r = distance(gridPos, flexPos);
        if (r > kFlexRepulsionCutoff) continue;
        r = max(r, 0.1f);

        float flexRadius = 1.7f;
        float d = r - (flexRadius + kFlexProbeRadius);

        if (d < 0.0f) {
            repulsion += kFlexRepulsionWeight * d * d;
        }
    }

    if (repulsion > 0.0f) {
        float current = float(stericGrid[tid]);
        stericGrid[tid] = half(clamp(current + repulsion, -100.0f, 100.0f));
    }
}
