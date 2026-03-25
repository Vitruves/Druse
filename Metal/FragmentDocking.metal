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
// 7. Reconstruct full DockPose from fragment placement chain
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
    p.fragmentIdx = 0; // anchor
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
    uint validCount = 0;
    for (uint i = 0; i < N; i++) {
        if (!placements[i].valid || placements[i].energy > cutoff) {
            placements[i].valid = 0;
        } else {
            validCount++;
        }
    }

    // If still too many, do a simple selection sort to keep top beamWidth
    if (validCount > beamWidth) {
        // Bubble the worst valid ones to invalid
        // Simple approach: find the beamWidth-th best energy, invalidate above it
        // Use partial selection: find the k-th smallest energy among valid placements
        // Iterative approach for GPU simplicity
        uint kept = 0;
        float kthEnergy = bestE;
        for (uint pass = 0; pass < beamWidth && pass < N; pass++) {
            float nextBest = 1e10f;
            uint nextIdx = N;
            for (uint i = 0; i < N; i++) {
                if (placements[i].valid && placements[i].energy <= nextBest) {
                    // Skip if we already counted this energy level
                    if (placements[i].energy > kthEnergy || (placements[i].energy == kthEnergy && i > nextIdx)) {
                        // already processed
                    } else {
                        nextBest = placements[i].energy;
                        nextIdx = i;
                    }
                }
            }
            if (nextIdx < N) {
                kthEnergy = nextBest;
                kept++;
            }
        }
        // Actually: simpler approach — just mark invalid everything above beam width
        // Count valid from start, once we hit beam width, invalidate the rest
        // But this doesn't sort by energy. Let's do an O(N*K) selection:
        uint remaining = 0;
        for (uint k = 0; k < beamWidth; k++) {
            float minE = 1e10f;
            uint minIdx = N;
            for (uint i = 0; i < N; i++) {
                if (placements[i].valid && placements[i].energy < minE) {
                    minE = placements[i].energy;
                    minIdx = i;
                }
            }
            if (minIdx < N) {
                // Temporarily mark as "selected" by negating valid
                placements[minIdx].valid = 2; // selected
                remaining++;
            }
        }
        // Now mark everything still valid=1 as invalid, and restore selected (2→1)
        for (uint i = 0; i < N; i++) {
            if (placements[i].valid == 2) {
                placements[i].valid = 1;
            } else {
                placements[i].valid = 0;
            }
        }
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
    constant DockLigandAtom         *fullLigAtoms     [[buffer(4)]],
    device const half               *affinityMaps     [[buffer(5)]],
    device const int32_t            *typeIndexLookup  [[buffer(6)]],
    constant GridParams             &gridParams       [[buffer(7)]],
    constant FragmentSearchParams   &searchParams     [[buffer(8)]],
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

    // The child fragment is placed relative to the parent by rotating around the connecting bond axis.
    // For simplicity, we place the child fragment's centroid at:
    // parent_translation + rotation of (child_centroid_offset) by torsion angle around bond axis
    //
    // In the full implementation, the connecting bond atoms define the axis.
    // Here we use the fragment centroid direction from parent centroid as an approximation.

    constant FragmentDef &parentFrag = fragDefs[parent.fragmentIdx];
    float3 parentCentroid = parentFrag.centroid;
    float3 childCentroid = childFrag.centroid;

    // Direction from parent to child in molecular frame
    float3 growDir = childCentroid - parentCentroid;
    float growDist = length(growDir);
    if (growDist < 0.1f) growDist = 3.0f; // fallback typical bond length
    growDir = normalize(growDir);

    // Rotate grow direction by torsion angle around parent's orientation
    float3 axis = quatRotate(parent.rotation, growDir);
    // Apply torsion rotation around axis
    float halfAngle = torsionAngle * 0.5f;
    float4 torsionQuat = float4(axis * sin(halfAngle), cos(halfAngle));

    // Child position: parent position + rotated growth vector
    float3 growVec = axis * growDist;
    // Rotate growVec by torsion quaternion
    float3 rotatedGrow = quatRotate(torsionQuat, growVec);
    child.translation = parent.translation + rotatedGrow;

    // Child orientation: parent orientation composed with torsion rotation
    float4 pq = parent.rotation;
    child.rotation = normalize(float4(
        torsionQuat.w*pq.x + torsionQuat.x*pq.w + torsionQuat.y*pq.z - torsionQuat.z*pq.y,
        torsionQuat.w*pq.y - torsionQuat.x*pq.z + torsionQuat.y*pq.w + torsionQuat.z*pq.x,
        torsionQuat.w*pq.z + torsionQuat.x*pq.y - torsionQuat.y*pq.x + torsionQuat.z*pq.w,
        torsionQuat.w*pq.w - torsionQuat.x*pq.x - torsionQuat.y*pq.y - torsionQuat.z*pq.z
    ));

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

// ============================================================================
// MARK: - Reconstruct Full Pose from Fragment Placement Chain
// ============================================================================

/// Convert a chain of fragment placements back to a single DockPose.
/// One thread per final placement. Traces back through parentPlacementIdx chain
/// to collect all torsion angles and compute the anchor-relative full pose.
kernel void reconstructFullPose(
    device DockPose                 *outputPoses     [[buffer(0)]],
    device const FragmentPlacement  *finalPlacements [[buffer(1)]],
    constant FragmentDef            *fragDefs        [[buffer(2)]],
    constant FragmentSearchParams   &searchParams    [[buffer(3)]],
    constant GridParams             &gridParams      [[buffer(4)]],
    uint                             tid             [[thread_position_in_grid]])
{
    uint numFinal = searchParams.numPlacements;
    if (tid >= numFinal) return;

    device const FragmentPlacement &fp = finalPlacements[tid];
    if (!fp.valid) {
        outputPoses[tid].energy = 1e10f;
        return;
    }

    device DockPose &pose = outputPoses[tid];

    // Trace back to anchor to collect torsion angles
    // The anchor placement provides translation and rotation
    // Each intermediate placement provides a connecting torsion angle

    // Walk the chain backwards
    float torsions[32];
    int numTorsions = 0;
    int traceIdx = int(tid);

    // Collect torsion angles from leaf to root
    // We'll reverse them later
    float chainTorsions[16]; // max fragments
    int chainLength = 0;

    device const FragmentPlacement *current = &finalPlacements[tid];
    while (current->parentPlacementIdx >= 0 && chainLength < 16) {
        chainTorsions[chainLength++] = current->connectingTorsion;
        // Note: parent placement is in the PREVIOUS level's buffer
        // For reconstruction, we store torsions in order from anchor outward
        // Since we're tracing backward, we'll reverse
        break; // In the beam search, parent indices point into the same buffer
               // For full chain reconstruction, the Swift code handles multi-level tracing
    }

    // The anchor placement gives us the global translation and rotation
    // For the simplified single-level case:
    pose.translation = fp.translation;
    pose.rotation = fp.rotation;
    pose.energy = fp.energy;
    pose.numTorsions = 0; // Will be set by Swift after full reconstruction

    // Zero out torsions
    for (int i = 0; i < 32; i++) {
        pose.torsions[i] = 0.0f;
    }

    // Store the connecting torsion
    if (chainLength > 0) {
        pose.torsions[0] = chainTorsions[0];
        pose.numTorsions = int(chainLength);
    }

    pose.generation = 0;
    pose.stericEnergy = 0.0f;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy = 0.0f;
    pose.torsionPenalty = 0.0f;
    pose.clashPenalty = 0.0f;
    pose.drusinaCorrection = 0.0f;
    pose.constraintPenalty = 0.0f;
    pose.numChiAngles = 0;
}
