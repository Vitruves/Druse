// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"

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

        float newE = evaluatePoseConstrained(pose, ligandAtoms, nAtoms, nTor,
                                   torsionEdges, movingIndices,
                                   affinityMaps, typeIndexLookup, gridParams,
                                   gaParams.referenceIntraEnergy, intraPairs, nPairs,
                                   pharmaConstraints, pharmaParams);

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

// ============================================================================
// MARK: - SIMD-Cooperative L-BFGS Local Search
// ============================================================================
// One SIMD group (32 lanes, dispatched as a 32-wide threadgroup) refines one pose over the
// full pose vector: 3 translation + 3 rotation + N torsion degrees of freedom.
//
// The optimizer is limited-memory BFGS with an Armijo backtracking line search. Steepest
// descent is unusable here because the degrees of freedom carry different units and very
// different gradient magnitudes: a torsion gradient is a torque summed over every moving
// atom with its lever arm, so it exceeds a translation force by one to two orders of
// magnitude. Normalising one combined gradient therefore spends the entire step budget
// rotating bonds while a ligand placed beside the pocket never translates into it. L-BFGS
// recovers the per-DOF scaling from the (s, y) history; the per-block trust region keeps the
// first iterations, taken before any history exists, physically sized.

constant uint  kLSMaxDOF     = 38u;    // 3 translation + 3 rotation + 32 torsions
constant uint  kLSHistory    = 5u;     // L-BFGS correction pairs
constant float kLSArmijoC1   = 1e-4f;  // sufficient-decrease coefficient
constant uint  kLSMaxLineTry = 8u;     // backtracking trials per iteration
constant float kLSTrustTrans = 1.0f;   // Å, largest translation of one accepted step
constant float kLSTrustAngle = 0.35f;  // rad, largest rotation and per-torsion change

// Intramolecular forces scatter to both atoms of a pair, so the pair loop cannot simply be
// strided over lanes and written per atom. Metal has no threadgroup float atomic, so the
// scatter accumulates in fixed point. Pair derivatives are bounded by the Vina terms (|dE/dr|
// stays below ~10 kcal/mol/Å) and an atom has fewer than 128 partners, so a component stays
// under ~1280 against the ±262144 this scale represents, while the quantum of 1.2e-4 enters
// only the search direction — every energy that is compared or reported stays in float.
constant float kLSForceScale = 8192.0f;

/// Advance a unit quaternion by an axis-angle increment.
static inline float4 lsQuatIncrement(float4 q, float3 axisAngle) {
    float angle = length(axisAngle);
    if (angle < 1e-9f) return q;
    float3 axis = axisAngle / angle;
    float halfAngle = angle * 0.5f;
    float4 dq = float4(axis * sin(halfAngle), cos(halfAngle));
    return normalize(float4(
        dq.w * q.x + dq.x * q.w + dq.y * q.z - dq.z * q.y,
        dq.w * q.y - dq.x * q.z + dq.y * q.w + dq.z * q.x,
        dq.w * q.z + dq.x * q.y - dq.y * q.x + dq.z * q.w,
        dq.w * q.w - dq.x * q.x - dq.y * q.y - dq.z * q.z));
}

/// Rebuild `pos` from a pose state. The rigid transform spreads over lanes; torsions are
/// applied in order because a nested torsion's pivot moves with its parent, but each
/// torsion's moving atoms spread over lanes.
static inline void lsBuildPositions(
    threadgroup float3        *pos,
    float3                     translation,
    float4                     rotation,
    threadgroup const float   *torsions,
    constant DockLigandAtom   *atoms,
    uint                       nA,
    uint                       nTor,
    constant TorsionEdge      *edges,
    constant int32_t          *movingIndices,
    uint                       lane)
{
    for (uint a = lane; a < nA; a += 32)
        pos[a] = quatRotate(rotation, atoms[a].position) + translation;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < nTor; t++) {
        // Every lane reads the same angle and edge, so a skipped torsion skips its barrier
        // on all lanes together.
        float angle = torsions[t];
        TorsionEdge edge = edges[t];
        if (abs(angle) < 1e-6f || edge.movingCount <= 0) continue;
        if (edge.atom1 < 0 || uint(edge.atom1) >= nA || edge.atom2 < 0 || uint(edge.atom2) >= nA) continue;
        float3 pivot = pos[edge.atom1];
        float3 axisVec = pos[edge.atom2] - pivot;
        float axisLen = length(axisVec);
        if (axisLen < 1e-6f) continue;
        float3 axis = axisVec / axisLen;
        float cosA = cos(angle);
        float sinA = sin(angle);
        // A torsion never moves its own pivot atoms, so these reads cannot tear.
        for (int i = int(lane); i < edge.movingCount; i += 32) {
            int ai = movingIndices[edge.movingStart + i];
            if (ai < 0 || uint(ai) >= nA) continue;
            float3 v = pos[ai] - pivot;
            pos[ai] = pivot + v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Energy of the configuration in `pos`, in the same expression `scorePoses` uses:
///     (grid + intra - reference) * norm + wPenalty * outOfGrid + constraints
/// Returned on every lane. With `wantForces`, also writes -dE/dx per atom into `force`,
/// scaled so a gradient projected from it differentiates exactly this expression.
///
/// Both modes evaluate the energy through the same expressions, so a line-search probe and
/// the gradient evaluation at the same configuration agree bit for bit. They differ only in
/// whether forces are accumulated, which keeps the probes cheap.
static inline float lsEnergy(
    threadgroup float3        *pos,
    threadgroup float3        *force,
    threadgroup atomic_int    *pairForce,
    constant DockLigandAtom   *atoms,
    uint                       nA,
    constant uint32_t         *intraPairs,
    uint                       nPairs,
    device const half         *affinityMaps,
    device const int32_t      *typeIndexLookup,
    constant GridParams       &gp,
    float                      referenceIntra,
    float                      normFactor,
    constant PharmacophoreConstraint *pharmaConstraints,
    constant PharmacophoreParams     &pharmaParams,
    bool                       wantForces,
    uint                       lane)
{
    if (wantForces) {
        for (uint i = lane; i < nA * 3u; i += 32)
            atomic_store_explicit(&pairForce[i], 0, memory_order_relaxed);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Intramolecular pairs spread over lanes. Forces scatter to both partners, so they
    // accumulate through threadgroup atomics rather than being serialised onto one lane.
    float laneIntra = 0.0f;
    for (uint p = lane; p < nPairs; p += 32) {
        uint packed = intraPairs[p];
        uint i = packed & 0xFFFFu;
        uint j = packed >> 16u;
        if (i >= nA || j >= nA) continue;
        float3 diff = pos[i] - pos[j];
        float r = length(diff);
        if (r < 1e-6f) continue;
        float dEdr;
        laneIntra += vinaPairEnergyWithDeriv(atoms[i].vinaType, atoms[j].vinaType, r, dEdr);
        if (wantForces && dEdr != 0.0f) {
            float3 gc = normFactor * dEdr * diff / r;
            for (uint d = 0; d < 3; d++) {
                int quantum = int(rint(gc[d] * kLSForceScale));
                atomic_fetch_add_explicit(&pairForce[i * 3u + d], -quantum, memory_order_relaxed);
                atomic_fetch_add_explicit(&pairForce[j * 3u + d],  quantum, memory_order_relaxed);
            }
        }
    }
    float intra = simd_sum(laneIntra);
    if (wantForces) simdgroup_barrier(mem_flags::mem_threadgroup);

    // Grid term, and the per-atom force assembled by the lane that owns the atom.
    float3 boxMin = searchBoxMin(gp);
    float3 boxMax = searchBoxMax(gp);
    float laneGrid = 0.0f;
    float lanePenalty = 0.0f;
    for (uint a = lane; a < nA; a += 32) {
        // Confinement applies everywhere, so its gradient pushes a stray atom back towards
        // the box even where the affinity maps still have a value to contribute.
        float3 boxGrad;
        lanePenalty += outOfBoxPenaltyWithGrad(pos[a], boxMin, boxMax, boxGrad);
        float3 part = -wPenalty * boxGrad;
        if (insideGrid(pos[a], gp)) {
            float3 grad;
            laneGrid += sampleTypedAffinityMapWithGrad(
                affinityMaps, typeIndexLookup, atoms[a].vinaType, pos[a], gp, grad);
            part -= grad * normFactor;
        }
        if (wantForces) {
            float3 scattered = float3(float(atomic_load_explicit(&pairForce[a * 3u + 0], memory_order_relaxed)),
                                      float(atomic_load_explicit(&pairForce[a * 3u + 1], memory_order_relaxed)),
                                      float(atomic_load_explicit(&pairForce[a * 3u + 2], memory_order_relaxed)));
            force[a] = part + scattered / kLSForceScale;
        }
    }
    float total = (simd_sum(laneGrid) + intra - referenceIntra) * normFactor
                + wPenalty * simd_sum(lanePenalty);

    // Constraints are few and each contributes to a single atom, so one lane walks them.
    if (pharmaParams.numConstraints > 0) {
        if (wantForces) simdgroup_barrier(mem_flags::mem_threadgroup);
        float pen = 0.0f;
        if (lane == 0) {
            pen = wantForces
                ? evaluateConstraintPenaltyWithGrad(pos, force, atoms, nA, pharmaConstraints, pharmaParams)
                : evaluateConstraintPenalty(pos, atoms, nA, pharmaConstraints, pharmaParams);
        }
        total += simd_broadcast_first(pen);
        if (wantForces) simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    return total;
}

/// Project per-atom forces onto the pose degrees of freedom, writing dE/dq into `grad`
/// (translation, rotation about the global axes, then one entry per torsion).
static inline void lsProjectGradient(
    threadgroup float3      *pos,
    threadgroup float3      *force,
    threadgroup float       *grad,
    float3                   translation,
    uint                     nA,
    uint                     nTor,
    constant TorsionEdge    *edges,
    constant int32_t        *movingIndices,
    uint                     lane)
{
    float3 laneForce = float3(0.0f);
    float3 laneTorque = float3(0.0f);
    for (uint a = lane; a < nA; a += 32) {
        laneForce += force[a];
        laneTorque += cross(pos[a] - translation, force[a]);
    }
    float3 totalForce = simd_sum(laneForce);
    float3 totalTorque = simd_sum(laneTorque);

    for (uint t = 0; t < nTor; t++) {
        TorsionEdge edge = edges[t];
        float value = 0.0f;
        if (edge.movingCount > 0 && edge.atom1 >= 0 && uint(edge.atom1) < nA
            && edge.atom2 >= 0 && uint(edge.atom2) < nA) {
            float3 pivot = pos[edge.atom1];
            float3 axisVec = pos[edge.atom2] - pivot;
            float axisLen = length(axisVec);
            if (axisLen > 1e-6f) {
                float3 axis = axisVec / axisLen;
                float laneTwist = 0.0f;
                for (int i = int(lane); i < edge.movingCount; i += 32) {
                    int ai = movingIndices[edge.movingStart + i];
                    if (ai >= 0 && uint(ai) < nA)
                        laneTwist += dot(cross(pos[ai] - pivot, force[ai]), axis);
                }
                value = -simd_sum(laneTwist);
            }
        }
        if (lane == 0) grad[6u + t] = value;
    }
    if (lane == 0) {
        grad[0] = -totalForce.x; grad[1] = -totalForce.y; grad[2] = -totalForce.z;
        grad[3] = -totalTorque.x; grad[4] = -totalTorque.y; grad[5] = -totalTorque.z;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

/// Inner product of two DOF vectors, returned on every lane.
static inline float lsDot(threadgroup const float *a, threadgroup const float *b, uint n, uint lane) {
    float s = 0.0f;
    for (uint i = lane; i < n; i += 32) s += a[i] * b[i];
    return simd_sum(s);
}

/// Largest step length along `dir` that respects the per-block trust region. Computed
/// redundantly on every lane: the vector is at most 38 entries, which is cheaper than a
/// reduction plus a broadcast.
static inline float lsTrustLimit(threadgroup const float *dir, uint nTor) {
    float transNorm = length(float3(dir[0], dir[1], dir[2]));
    float rotNorm = length(float3(dir[3], dir[4], dir[5]));
    float torMax = 0.0f;
    for (uint t = 0; t < nTor; t++) torMax = max(torMax, abs(dir[6u + t]));

    float limit = 1.0f;
    if (transNorm > 1e-12f) limit = min(limit, kLSTrustTrans / transNorm);
    if (rotNorm > 1e-12f) limit = min(limit, kLSTrustAngle / rotNorm);
    if (torMax > 1e-12f) limit = min(limit, kLSTrustAngle / torMax);
    return limit;
}


/// Refine one pose per SIMD group. Dispatch `populationSize` threadgroups of 32 threads.
///
/// Pose state is replicated in registers rather than shared through threadgroup memory: every
/// lane derives it from the same uniform direction and step length, so the copies stay
/// identical by construction and no barrier is needed to publish them. Torsions live in
/// threadgroup memory because `lsBuildPositions` indexes them dynamically; lane t owns
/// torsion t, so each element has exactly one writer.
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
    uint nDOF = 6u + nTor;
    float3 searchMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 searchMax = gridParams.searchCenter + gridParams.searchHalfExtent;
    float normFactor = vinaRotNormalization(float(nTor));

    threadgroup float3 tg_pos[128];
    threadgroup float3 tg_force[128];
    threadgroup atomic_int tg_pairForce[128 * 3];
    threadgroup float tg_grad[kLSMaxDOF];
    threadgroup float tg_gradNext[kLSMaxDOF];
    threadgroup float tg_dir[kLSMaxDOF];
    threadgroup float tg_sHist[kLSHistory][kLSMaxDOF];
    threadgroup float tg_yHist[kLSHistory][kLSMaxDOF];
    threadgroup float tg_torsions[32];
    threadgroup float tg_trialTorsions[32];

    float3 translation = pose.translation;
    float4 rotation = pose.rotation;
    for (uint t = lane; t < nTor; t += 32) tg_torsions[t] = pose.torsions[t];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    lsBuildPositions(tg_pos, translation, rotation, tg_torsions,
                     ligandAtoms, nA, nTor, torsionEdges, movingIndices, lane);
    float energy = lsEnergy(tg_pos, tg_force, tg_pairForce, ligandAtoms, nA,
                            intraPairs, nPairs, affinityMaps, typeIndexLookup, gridParams,
                            gaParams.referenceIntraEnergy, normFactor,
                            pharmaConstraints, pharmaParams, true, lane);
    lsProjectGradient(tg_pos, tg_force, tg_grad, translation, nA, nTor,
                      torsionEdges, movingIndices, lane);

    // Every scalar below is uniform across the SIMD group, so the history scalars are
    // register copies rather than shared state.
    float rho[kLSHistory];
    float coef[kLSHistory];
    float gamma = 1.0f;
    uint stored = 0u;
    uint head = 0u;

    int maxSteps = max(int(gaParams.localSearchSteps), 1);
    for (int step = 0; step < maxSteps; step++) {
        // --- Direction: L-BFGS two-loop recursion over the stored corrections ---
        for (uint i = lane; i < nDOF; i += 32) tg_dir[i] = tg_grad[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint count = min(stored, kLSHistory);
        for (uint k = 0; k < count; k++) {
            uint slot = (head + kLSHistory - 1u - k) % kLSHistory;
            float a = rho[slot] * lsDot(tg_sHist[slot], tg_dir, nDOF, lane);
            coef[slot] = a;
            for (uint i = lane; i < nDOF; i += 32) tg_dir[i] -= a * tg_yHist[slot][i];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (count > 0u) {
            for (uint i = lane; i < nDOF; i += 32) tg_dir[i] *= gamma;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint k = count; k > 0u; k--) {
            uint slot = (head + kLSHistory - k) % kLSHistory;
            float adjust = coef[slot] - rho[slot] * lsDot(tg_yHist[slot], tg_dir, nDOF, lane);
            for (uint i = lane; i < nDOF; i += 32) tg_dir[i] += adjust * tg_sHist[slot][i];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = lane; i < nDOF; i += 32) tg_dir[i] = -tg_dir[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float slope = lsDot(tg_grad, tg_dir, nDOF, lane);
        if (slope >= 0.0f) {
            // The stored curvature no longer describes this basin: restart from steepest descent.
            for (uint i = lane; i < nDOF; i += 32) tg_dir[i] = -tg_grad[i];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            slope = -lsDot(tg_grad, tg_grad, nDOF, lane);
            stored = 0u;
            head = 0u;
            gamma = 1.0f;
        }
        if (slope > -1e-9f) break;  // gradient is numerically zero

        // --- Armijo backtracking, first trial sized by the trust region ---
        float alpha = lsTrustLimit(tg_dir, nTor);
        float3 trialTranslation = translation;
        float4 trialRotation = rotation;
        bool accepted = false;
        for (uint attempt = 0; attempt < kLSMaxLineTry; attempt++) {
            trialTranslation = clamp(translation + alpha * float3(tg_dir[0], tg_dir[1], tg_dir[2]),
                                     searchMin, searchMax);
            trialRotation = lsQuatIncrement(rotation, alpha * float3(tg_dir[3], tg_dir[4], tg_dir[5]));
            for (uint t = lane; t < nTor; t += 32) {
                float value = tg_torsions[t] + alpha * tg_dir[6u + t];
                tg_trialTorsions[t] = value - 2.0f * M_PI_F * floor((value + M_PI_F) / (2.0f * M_PI_F));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            lsBuildPositions(tg_pos, trialTranslation, trialRotation, tg_trialTorsions,
                             ligandAtoms, nA, nTor, torsionEdges, movingIndices, lane);
            float trialEnergy = lsEnergy(tg_pos, tg_force, tg_pairForce, ligandAtoms, nA,
                                         intraPairs, nPairs, affinityMaps, typeIndexLookup, gridParams,
                                         gaParams.referenceIntraEnergy, normFactor,
                                         pharmaConstraints, pharmaParams, false, lane);
            if (trialEnergy <= energy + kLSArmijoC1 * alpha * slope) {
                energy = trialEnergy;
                accepted = true;
                break;
            }
            alpha *= 0.5f;
        }
        if (!accepted) break;  // no sufficient decrease along this direction

        // --- Record the correction pair and commit. The box clamp and the torsion wrap mean
        // the applied change can differ from alpha * direction, and the curvature pair has to
        // describe what was actually applied. Lane t owns torsion t here too, so writing the
        // history and overwriting the torsion need no barrier between them. ---
        if (lane < 3u) tg_sHist[head][lane] = trialTranslation[lane] - translation[lane];
        else if (lane < 6u) tg_sHist[head][lane] = alpha * tg_dir[lane];
        for (uint t = lane; t < nTor; t += 32) {
            float delta = tg_trialTorsions[t] - tg_torsions[t];
            tg_sHist[head][6u + t] = delta - 2.0f * M_PI_F * floor((delta + M_PI_F) / (2.0f * M_PI_F));
            tg_torsions[t] = tg_trialTorsions[t];
        }
        translation = trialTranslation;
        rotation = trialRotation;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // `tg_pos` still holds the accepted configuration — the accepted probe was the last
        // one evaluated — so the gradient needs no rebuild.
        lsEnergy(tg_pos, tg_force, tg_pairForce, ligandAtoms, nA,
                 intraPairs, nPairs, affinityMaps, typeIndexLookup, gridParams,
                 gaParams.referenceIntraEnergy, normFactor,
                 pharmaConstraints, pharmaParams, true, lane);
        lsProjectGradient(tg_pos, tg_force, tg_gradNext, translation, nA, nTor,
                          torsionEdges, movingIndices, lane);

        for (uint i = lane; i < nDOF; i += 32) tg_yHist[head][i] = tg_gradNext[i] - tg_grad[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float curvature = lsDot(tg_sHist[head], tg_yHist[head], nDOF, lane);
        if (curvature > 1e-10f) {
            rho[head] = 1.0f / curvature;
            gamma = curvature / max(lsDot(tg_yHist[head], tg_yHist[head], nDOF, lane), 1e-20f);
            head = (head + 1u) % kLSHistory;
            stored = min(stored + 1u, kLSHistory);
        }

        for (uint i = lane; i < nDOF; i += 32) tg_grad[i] = tg_gradNext[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        pose.translation = translation;
        pose.rotation = rotation;
        pose.energy = energy;
    }
    for (uint t = lane; t < nTor; t += 32) pose.torsions[t] = tg_torsions[t];
}
