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

        // Normalize and clamp
        float scale = min(stepSize / gradMag, stepSize);

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
            stepSize = min(stepSize * 1.2f, 0.5f);
        } else {
            pose.translation = oldT;
            pose.rotation = oldRot;
            for (uint t = 0; t < nTor; t++) pose.torsions[t] = oldTorsions[t];
            pose.energy = baseE;
            stepSize *= 0.5f;
        }
        if (stepSize < 0.001f) break;
    }
}

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
    float normFactor = vinaRotNormalization(nRotF);

    threadgroup float3 tg_pos[128];
    threadgroup float3 tg_forces[128];
    threadgroup float3 tg_oldT;
    threadgroup float4 tg_oldRot;
    threadgroup float  tg_oldTor[32];

    float stepSize = 0.08f;
    int maxSteps = max(int(gaParams.localSearchSteps), 1);

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
                float3 gc = normFactor * dEdr * diff / r;
                tg_forces[i] -= gc; tg_forces[j] += gc;
            }
        }
        intraE = simd_broadcast_first(intraE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float totalEnergy = (totalIntermolecular + intraE - gaParams.referenceIntraEnergy) * normFactor;

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

        float gradMag = 0;
        for (int i = 0; i < 3; i++) gradMag += gradT[i] * gradT[i] + gradR[i] * gradR[i];
        for (uint t = 0; t < nTor; t++) gradMag += gradTor[t] * gradTor[t];
        gradMag = sqrt(gradMag);
        if (gradMag < 1e-6f) break;

        // Normalize and clamp
        float scale = min(stepSize / gradMag, stepSize);

        // === Apply step (lane 0 saves old pose, applies trial pose) ===
        if (lane == 0) {
            tg_oldT = pose.translation;
            tg_oldRot = pose.rotation;
            for (uint t = 0; t < nTor; t++) tg_oldTor[t] = pose.torsions[t];

            float3 newT = pose.translation;
            for (int i = 0; i < 3; i++) newT[i] -= scale * gradT[i];
            newT = clamp(newT, gMin, gMax);

            float3 rs = float3(-scale * gradR[0], -scale * gradR[1], -scale * gradR[2]);
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
                pose.torsions[t] -= scale * gradTor[t];
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

        float newE = (newGridE + newIntraE - gaParams.referenceIntraEnergy) * normFactor
                   + wPenalty * newPenalty;

        // Pharmacophore constraint penalty on trial pose
        if (pharmaParams.numConstraints > 0 && lane == 0) {
            float3 cPos[128];
            for (uint a = 0; a < nA; a++) cPos[a] = tg_pos[a];
            newE += evaluateConstraintPenalty(cPos, ligandAtoms, nA, pharmaConstraints, pharmaParams);
        }
        newE = simd_broadcast_first(newE);

        // === Accept/reject ===
        if (lane == 0) {
            if (newE < totalEnergy) {
                pose.energy = newE; stepSize = min(stepSize * 1.2f, 0.5f);
            } else {
                pose.translation = tg_oldT; pose.rotation = tg_oldRot;
                for (uint t = 0; t < nTor; t++) pose.torsions[t] = tg_oldTor[t];
                pose.energy = totalEnergy; stepSize *= 0.5f;
            }
        }
        stepSize = simd_broadcast_first(stepSize);
        if (stepSize < 0.001f) break;
    }
}
