// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Fragment-Based Incremental Construction Docking
//
// Algorithm:
// 1. Place anchor fragment at many positions/orientations in the pocket
// 2. Score each anchor placement against Vina affinity grids
// 3. Prune to top-K (beam width) placements
// 4. For each surviving placement, grow next fragment by sampling torsion angles
// 5. Score partial ligand, prune again
// 6. Repeat until all fragments placed
// ============================================================================

// Reuse Vina scoring helpers from DockingCompute.metal (linked in same Metal library)
// These are available because all .metal files in the target compile into one library:
//   vinaPairEnergy, trilinearInterpolate, sampleTypedAffinityMap, etc.

// ---- RNG (same as DockingCompute) ----
inline float fragRandom(thread uint &seed, uint offset) {
    seed = seed * 1103515245u + 12345u + offset;
    return float(seed & 0x7FFFFFFFu) / float(0x7FFFFFFF);
}

inline float3 fragRandomInsideUnitSphere(thread uint &seed, uint base) {
    float3 v;
    for (int attempt = 0; attempt < 8; attempt++) {
        v = float3(
            fragRandom(seed, base + attempt * 3u) * 2.0f - 1.0f,
            fragRandom(seed, base + attempt * 3u + 1u) * 2.0f - 1.0f,
            fragRandom(seed, base + attempt * 3u + 2u) * 2.0f - 1.0f
        );
        if (length_squared(v) <= 1.0f) return v;
    }
    return normalize(v) * 0.5f;
}

// Shoemake uniform random quaternion
inline float4 fragRandomQuaternion(thread uint &seed, uint base) {
    float u1 = fragRandom(seed, base);
    float u2 = fragRandom(seed, base + 1u) * 2.0f * M_PI_F;
    float u3 = fragRandom(seed, base + 2u) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    return float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));
}

// Quaternion rotation of a vector
inline float3 quatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

// ============================================================================
// MARK: - Anchor Placement Initialization
// ============================================================================

/// Generate random anchor fragment placements within the search box.
/// One thread per placement. Writes to FragmentPlacement buffer.
kernel void initAnchorPlacements(
    device FragmentPlacement      *placements     [[buffer(0)]],
    constant GridParams           &gridParams     [[buffer(1)]],
    constant FragmentSearchParams &searchParams   [[buffer(2)]],
    uint                           tid            [[thread_position_in_grid]])
{
    if (tid >= searchParams.numAnchorSamples) return;

    uint seed = tid * 271828u + 314159u;

    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;

    device FragmentPlacement &p = placements[tid];

    // Random position within pocket search box (70% near center, 30% full box)
    float spreadFactor = (fragRandom(seed, 0) < 0.7f) ? 0.5f : 1.0f;
    float3 offset = fragRandomInsideUnitSphere(seed, 10) * halfExtent * spreadFactor;
    p.translation = center + offset;

    // Random orientation (uniform over SO(3))
    p.rotation = fragRandomQuaternion(seed, 20);

    p.energy = 1e10f;
    p.connectingTorsion = 0.0f;
    p.parentPlacementIdx = -1;
    // The host sets currentFragment to the anchor fragment for the initial placement pass.
    p.fragmentIdx = int(searchParams.currentFragment < searchParams.numFragments ? searchParams.currentFragment : 0u);
    p.valid = 1;
}

// ============================================================================
// MARK: - Fragment Scoring Against Vina Grids
// ============================================================================

/// Score anchor fragment placements against precomputed Vina affinity maps.
/// Each placement is scored by transforming fragment atoms and interpolating grid maps.
kernel void scoreFragmentPlacements(
    device FragmentPlacement        *placements     [[buffer(0)]],
    constant FragmentAtom           *fragAtoms      [[buffer(1)]],
    constant FragmentDef            *fragDefs       [[buffer(2)]],
    device const half               *affinityMaps   [[buffer(3)]],
    device const int32_t            *typeIndexLookup [[buffer(4)]],
    constant GridParams             &gridParams     [[buffer(5)]],
    constant FragmentSearchParams   &searchParams   [[buffer(6)]],
    uint                             tid            [[thread_position_in_grid]])
{
    uint numPlacements = searchParams.numPlacements;
    if (tid >= numPlacements) return;

    device FragmentPlacement &pl = placements[tid];
    if (!pl.valid) { pl.energy = 1e10f; return; }

    uint fragIdx = pl.fragmentIdx;
    constant FragmentDef &frag = fragDefs[fragIdx];

    float3 trans = pl.translation;
    float4 quat = pl.rotation;

    float totalEnergy = 0.0f;
    float outOfGridPenalty = 0.0f;

    for (uint a = 0; a < frag.atomCount && a < MAX_FRAGMENT_ATOMS; a++) {
        constant FragmentAtom &atom = fragAtoms[frag.atomStart + a];

        // Transform atom position: rotate fragment-local coords, then translate
        float3 localPos = atom.position - frag.centroid;
        float3 worldPos = quatRotate(quat, localPos) + trans;

        // Check grid bounds
        float3 gc = (worldPos - gridParams.origin) / gridParams.spacing;
        float3 dims = float3(gridParams.dims);
        if (gc.x < 0 || gc.y < 0 || gc.z < 0 ||
            gc.x >= dims.x - 1 || gc.y >= dims.y - 1 || gc.z >= dims.z - 1) {
            float dx = gc.x < 0 ? -gc.x : max(gc.x - (dims.x - 1), 0.0f);
            float dy = gc.y < 0 ? -gc.y : max(gc.y - (dims.y - 1), 0.0f);
            float dz = gc.z < 0 ? -gc.z : max(gc.z - (dims.z - 1), 0.0f);
            outOfGridPenalty += (dx * dx + dy * dy + dz * dz);
            continue;
        }

        // Sample typed affinity map
        int ligType = atom.vinaType;
        if (ligType < 0 || ligType >= 32) continue;
        int mapIdx = typeIndexLookup[ligType];
        if (mapIdx < 0) continue;

        // Trilinear interpolation
        uint nx = gridParams.dims.x;
        uint ny = gridParams.dims.y;
        uint ix = uint(gc.x); uint iy = uint(gc.y); uint iz = uint(gc.z);
        float fx = gc.x - float(ix); float fy = gc.y - float(iy); float fz = gc.z - float(iz);

        uint base = uint(mapIdx) * gridParams.totalPoints;
        uint idx000 = base + iz * nx * ny + iy * nx + ix;
        uint idx100 = idx000 + 1;
        uint idx010 = idx000 + nx;
        uint idx110 = idx000 + nx + 1;
        uint idx001 = idx000 + nx * ny;
        uint idx101 = idx001 + 1;
        uint idx011 = idx001 + nx;
        uint idx111 = idx001 + nx + 1;

        float c000 = float(affinityMaps[idx000]);
        float c100 = float(affinityMaps[idx100]);
        float c010 = float(affinityMaps[idx010]);
        float c110 = float(affinityMaps[idx110]);
        float c001 = float(affinityMaps[idx001]);
        float c101 = float(affinityMaps[idx101]);
        float c011 = float(affinityMaps[idx011]);
        float c111 = float(affinityMaps[idx111]);

        float val = c000 * (1-fx) * (1-fy) * (1-fz)
                  + c100 * fx * (1-fy) * (1-fz)
                  + c010 * (1-fx) * fy * (1-fz)
                  + c110 * fx * fy * (1-fz)
                  + c001 * (1-fx) * (1-fy) * fz
                  + c101 * fx * (1-fy) * fz
                  + c011 * (1-fx) * fy * fz
                  + c111 * fx * fy * fz;

        totalEnergy += val;
    }

    pl.energy = totalEnergy + 10.0f * outOfGridPenalty;
}

// ============================================================================
// MARK: - Beam Pruning
// ============================================================================

/// Prune placements to keep only the top beamWidth candidates.
/// Simple selection: mark all placements beyond the energy threshold as invalid.
/// This kernel runs single-threaded (small N) — sorts placements by energy.
kernel void pruneFragmentBeam(
    device FragmentPlacement      *placements    [[buffer(0)]],
    constant FragmentSearchParams &searchParams  [[buffer(1)]],
    uint                           tid           [[thread_position_in_grid]])
{
    if (tid != 0) return;

    uint N = searchParams.numPlacements;
    uint beamWidth = searchParams.beamWidth;
    float threshold = searchParams.pruneThreshold;

    // Find best energy
    float bestE = 1e10f;
    for (uint i = 0; i < N; i++) {
        if (placements[i].valid && placements[i].energy < bestE) {
            bestE = placements[i].energy;
        }
    }

    // Mark invalid if above threshold
    float cutoff = bestE + threshold;
    for (uint i = 0; i < N; i++) {
        if (!placements[i].valid || placements[i].energy > cutoff) {
            placements[i].valid = 0;
        }
    }

    // Select the best surviving placements and compact them into slots 0..<K.
    // Growth/reconstruction read a dense beam from the front of the buffer.
    FragmentPlacement selected[MAX_BEAM_WIDTH];
    uint keepLimit = min(beamWidth, MAX_BEAM_WIDTH);
    uint kept = 0;
    for (uint k = 0; k < keepLimit; k++) {
        float minE = 1e10f;
        uint minIdx = N;
        for (uint i = 0; i < N; i++) {
            if (placements[i].valid == 1 && placements[i].energy < minE) {
                minE = placements[i].energy;
                minIdx = i;
            }
        }
        if (minIdx >= N) break;
        selected[kept] = placements[minIdx];
        selected[kept].valid = 1;
        placements[minIdx].valid = 0;
        kept++;
    }

    for (uint i = 0; i < N; i++) {
        placements[i].valid = 0;
    }
    for (uint i = 0; i < kept; i++) {
        placements[i] = selected[i];
        placements[i].valid = 1;
    }

    // Keep deterministic invalid entries for callers that copy a fixed beam size.
    for (uint i = kept; i < beamWidth && i < N; i++) {
        placements[i].energy = 1e10f;
        placements[i].valid = 0;
    }
}

// ============================================================================
// MARK: - Fragment Growth
// ============================================================================

/// Grow the next fragment from each surviving parent placement by sampling torsion angles.
/// One thread per (parent_placement × torsion_sample).
/// Writes expanded placements to the output buffer.
kernel void growFragment(
    device FragmentPlacement        *outputPlacements [[buffer(0)]],
    device const FragmentPlacement  *parentPlacements [[buffer(1)]],
    constant FragmentAtom           *fragAtoms        [[buffer(2)]],
    constant FragmentDef            *fragDefs         [[buffer(3)]],
    device const half               *affinityMaps     [[buffer(5)]],
    device const int32_t            *typeIndexLookup  [[buffer(6)]],
    constant GridParams             &gridParams       [[buffer(7)]],
    constant FragmentSearchParams   &searchParams     [[buffer(8)]],
    device const FragmentPlacement  *beamHistory      [[buffer(9)]],
    uint                             tid              [[thread_position_in_grid]])
{
    uint numParents = searchParams.numPlacements;
    uint torsionSamples = searchParams.torsionSamples;
    uint totalThreads = numParents * torsionSamples;
    if (tid >= totalThreads) return;

    uint parentIdx = tid / torsionSamples;
    uint torsionSample = tid % torsionSamples;

    device const FragmentPlacement &parent = parentPlacements[parentIdx];
    device FragmentPlacement &child = outputPlacements[tid];

    if (!parent.valid) {
        child.valid = 0;
        child.energy = 1e10f;
        return;
    }

    uint childFragIdx = searchParams.currentFragment;
    constant FragmentDef &childFrag = fragDefs[childFragIdx];

    // Torsion angle for the connecting bond: evenly spaced + small random jitter
    uint seed = tid * 482711u + searchParams.currentFragment * 937373u;
    float baseTorsion = -M_PI_F + 2.0f * M_PI_F * (float(torsionSample) + 0.5f) / float(torsionSamples);
    float jitter = (fragRandom(seed, 0) - 0.5f) * (2.0f * M_PI_F / float(torsionSamples));
    float torsionAngle = baseTorsion + jitter;

    // Fragments are numbered in growth order with the anchor at 0, so beam level k holds the
    // placements of fragment k (beamHistory[k * beamWidth + i]). The previous level is not
    // necessarily this fragment's tree parent (e.g. siblings on the anchor), so walk the
    // placement chain back to the level of the real parent fragment.
    int parentFragIdx = childFrag.parentFragmentIdx;
    FragmentPlacement frame = parent;
    bool traced = parentFragIdx >= 0 && uint(parentFragIdx) < childFragIdx;
    for (uint level = childFragIdx - 1u; traced && int(level) > parentFragIdx; level--) {
        int idx = frame.parentPlacementIdx;
        if (idx < 0 || uint(idx) >= searchParams.beamWidth) { traced = false; break; }
        frame = beamHistory[(level - 1u) * searchParams.beamWidth + uint(idx)];
    }
    if (!traced) {
        child.valid = 0;
        child.energy = 1e10f;
        return;
    }

    // Torsion about the real connecting bond. In the ligand frame the child atoms rotate by
    // Q(θ) about the axis u = (b_child − b_parent) through b_child, x' = b_child + Q(x − b_child).
    // The parent frame maps x ↦ R_p(x − c_p) + t_p, so the child frame x ↦ R_c(x − c_c) + t_c is
    //   R_c = R_p·Q = Q_w·R_p   (Q_w: rotation by θ about the world axis R_p·u)
    //   t_c = t_p + R_p(b_child − c_p) + Q_w·R_p(c_c − b_child).
    // At θ = 0 the child lands exactly where the rigid input conformer puts it.
    constant FragmentDef &parentFrag = fragDefs[parentFragIdx];
    float4 pq = frame.rotation;
    float3 bondVec = childFrag.bondChildAtom - childFrag.bondParentAtom;
    float bondLen = length(bondVec);
    float4 rq = pq;
    if (bondLen > 1e-4f) {
        float3 axis = quatRotate(pq, bondVec / bondLen);
        float halfAngle = torsionAngle * 0.5f;
        float4 tq = float4(axis * sin(halfAngle), cos(halfAngle));
        // Child orientation: world-axis torsion rotation composed with parent orientation
        rq = normalize(float4(
            tq.w*pq.x + tq.x*pq.w + tq.y*pq.z - tq.z*pq.y,
            tq.w*pq.y - tq.x*pq.z + tq.y*pq.w + tq.z*pq.x,
            tq.w*pq.z + tq.x*pq.y - tq.y*pq.x + tq.z*pq.w,
            tq.w*pq.w - tq.x*pq.x - tq.y*pq.y - tq.z*pq.z
        ));
    }
    // Degenerate bond (coincident atoms): no defined torsion axis, inherit the parent orientation.
    child.rotation = rq;
    child.translation = frame.translation
        + quatRotate(pq, childFrag.bondChildAtom - parentFrag.centroid)
        + quatRotate(rq, childFrag.centroid - childFrag.bondChildAtom);

    child.connectingTorsion = torsionAngle;
    child.parentPlacementIdx = int(parentIdx);
    child.fragmentIdx = int(childFragIdx);
    child.valid = 1;

    // Score the child fragment atoms against Vina grids
    float energy = parent.energy; // cumulative
    float outOfGridPenalty = 0.0f;

    for (uint a = 0; a < childFrag.atomCount && a < MAX_FRAGMENT_ATOMS; a++) {
        constant FragmentAtom &atom = fragAtoms[childFrag.atomStart + a];
        float3 localPos = atom.position - childFrag.centroid;
        float3 worldPos = quatRotate(child.rotation, localPos) + child.translation;

        float3 gc = (worldPos - gridParams.origin) / gridParams.spacing;
        float3 dims = float3(gridParams.dims);
        if (gc.x < 0 || gc.y < 0 || gc.z < 0 ||
            gc.x >= dims.x - 1 || gc.y >= dims.y - 1 || gc.z >= dims.z - 1) {
            float dx = gc.x < 0 ? -gc.x : max(gc.x - (dims.x - 1), 0.0f);
            float dy = gc.y < 0 ? -gc.y : max(gc.y - (dims.y - 1), 0.0f);
            float dz = gc.z < 0 ? -gc.z : max(gc.z - (dims.z - 1), 0.0f);
            outOfGridPenalty += (dx * dx + dy * dy + dz * dz);
            continue;
        }

        int ligType = atom.vinaType;
        if (ligType < 0 || ligType >= 32) continue;
        int mapIdx = typeIndexLookup[ligType];
        if (mapIdx < 0) continue;

        uint nx = gridParams.dims.x;
        uint ny = gridParams.dims.y;
        uint ix = uint(gc.x); uint iy = uint(gc.y); uint iz = uint(gc.z);
        float fx = gc.x - float(ix); float fy = gc.y - float(iy); float fz = gc.z - float(iz);

        uint base = uint(mapIdx) * gridParams.totalPoints;
        uint idx000 = base + iz * nx * ny + iy * nx + ix;

        float c000 = float(affinityMaps[idx000]);
        float c100 = float(affinityMaps[idx000 + 1]);
        float c010 = float(affinityMaps[idx000 + nx]);
        float c110 = float(affinityMaps[idx000 + nx + 1]);
        float c001 = float(affinityMaps[idx000 + nx * ny]);
        float c101 = float(affinityMaps[idx000 + nx * ny + 1]);
        float c011 = float(affinityMaps[idx000 + nx * ny + nx]);
        float c111 = float(affinityMaps[idx000 + nx * ny + nx + 1]);

        float val = c000*(1-fx)*(1-fy)*(1-fz) + c100*fx*(1-fy)*(1-fz)
                  + c010*(1-fx)*fy*(1-fz) + c110*fx*fy*(1-fz)
                  + c001*(1-fx)*(1-fy)*fz + c101*fx*(1-fy)*fz
                  + c011*(1-fx)*fy*fz + c111*fx*fy*fz;

        energy += val;
    }

    child.energy = energy + 10.0f * outOfGridPenalty;
}
