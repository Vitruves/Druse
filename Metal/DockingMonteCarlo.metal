// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"

kernel void mcPerturb(
    device DockPose            *perturbed   [[buffer(0)]],
    device const DockPose      *current     [[buffer(1)]],
    constant GAParams          &gaParams    [[buffer(2)]],
    constant GridParams        &gridParams  [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint seed = tid * 314159u + gaParams.generation * 271828u + gaParams.runSeed * 999983u + 42u;
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
            uint seed = tid * 92837111u + gaParams.generation * 689287499u + gaParams.runSeed * 1299827u + 17u;
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

// MARK: - Batched GA Kernels for Virtual Screening

kernel void scorePosesBatched(
    device DockPose              *poses          [[buffer(0)]],
    constant DockLigandAtom      *allAtoms       [[buffer(1)]],
    device const half            *affinityMaps   [[buffer(2)]],
    device const int32_t         *typeIndexLookup [[buffer(3)]],
    constant GridParams          &gridParams     [[buffer(4)]],
    constant BatchedGAParams     &batchParams    [[buffer(5)]],
    constant TorsionEdge         *allTorsionEdges [[buffer(6)]],
    constant int32_t             *allMovingIndices [[buffer(7)]],
    constant uint32_t            *allIntraPairs   [[buffer(8)]],
    constant BatchedGALigandInfo *ligandInfo      [[buffer(9)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= batchParams.totalPoses) return;
    uint ligIdx = tid / batchParams.populationSizePerLigand;
    if (ligIdx >= batchParams.numLigands) return;

    BatchedGALigandInfo info = ligandInfo[ligIdx];
    constant DockLigandAtom *myAtoms = allAtoms + info.atomStart;
    constant TorsionEdge *myEdges = allTorsionEdges + info.torsionEdgeStart;
    constant int32_t *myMoving = allMovingIndices + info.movingIndicesStart;
    constant uint32_t *myPairs = allIntraPairs + info.pairListStart;

    device DockPose &pose = poses[tid];
    uint nA = min(info.atomCount, 128u);
    uint nTor = min(info.torsionEdgeCount, 32u);

    float3 positions[128];
    for (uint a = 0; a < nA; a++)
        positions[a] = quatRotate(pose.rotation, myAtoms[a].position) + pose.translation;
    applyTorsions(positions, nA, pose, myEdges, myMoving, nTor);

    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;
    float totalE = 0.0f, penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float oopP = outOfGridPenalty(positions[a], gridMin, gridMax);
        if (oopP > 0.0f) { penalty += oopP; continue; }
        totalE += sampleTypedAffinityMap(affinityMaps, typeIndexLookup, myAtoms[a].vinaType, positions[a], gridParams);
    }

    float intra = intramolecularLigandEnergy(positions, myAtoms, nA, myPairs, info.numPairs) - info.referenceIntraEnergy;
    float norm = vinaRotNormalization(float(nTor));
    pose.stericEnergy = totalE;
    pose.clashPenalty = wPenalty * penalty + intra;
    pose.energy = (totalE + intra) * norm + wPenalty * penalty;
}

/// Batched MC perturbation for virtual screening.
kernel void mcPerturbBatched(
    device DockPose              *perturbed     [[buffer(0)]],
    device const DockPose        *current       [[buffer(1)]],
    constant BatchedGAParams     &batchParams   [[buffer(2)]],
    constant GridParams          &gridParams    [[buffer(3)]],
    constant BatchedGALigandInfo *ligandInfo    [[buffer(4)]],
    uint                          tid           [[thread_position_in_grid]])
{
    if (tid >= batchParams.totalPoses) return;
    uint ligIdx = tid / batchParams.populationSizePerLigand;
    if (ligIdx >= batchParams.numLigands) return;

    BatchedGALigandInfo info = ligandInfo[ligIdx];
    uint nTor = min(info.torsionEdgeCount, 32u);

    uint seed = tid * 314159u + batchParams.generation * 271828u + 42u;
    device const DockPose &src = current[tid];
    device DockPose &dst = perturbed[tid];
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;

    dst = src;
    uint which = (2u + nTor) > 0u ? uint(gpuRandom(seed, 0) * float(2u + nTor)) % (2u + nTor) : 0u;

    if (which == 0u) {
        dst.translation = clamp(src.translation + batchParams.translationStep * randomInsideUnitSphere(seed, 10), gMin, gMax);
    } else if (which == 1u) {
        float radius = max(info.ligandRadius, 1.0f);
        float3 rv = max(batchParams.rotationStep, batchParams.translationStep / radius) * randomInsideUnitSphere(seed, 20);
        float a = length(rv);
        if (a > 1e-6f) {
            float3 ax = rv / a; float ha = a * 0.5f;
            float4 dq = float4(ax * sin(ha), cos(ha)); float4 sq = src.rotation;
            dst.rotation = normalize(float4(
                dq.w*sq.x+dq.x*sq.w+dq.y*sq.z-dq.z*sq.y, dq.w*sq.y-dq.x*sq.z+dq.y*sq.w+dq.z*sq.x,
                dq.w*sq.z+dq.x*sq.y-dq.y*sq.x+dq.z*sq.w, dq.w*sq.w-dq.x*sq.x-dq.y*sq.y-dq.z*sq.z));
        }
    } else if (which - 2u < nTor) {
        dst.torsions[which - 2u] = gpuRandom(seed, 30 + which - 2u) * 2.0f * M_PI_F - M_PI_F;
    }
    dst.numTorsions = int(nTor); dst.generation = src.generation + 1; dst.energy = 1e10f;
}

/// Batched Metropolis acceptance for virtual screening.
kernel void metropolisAcceptBatched(
    device DockPose              *current       [[buffer(0)]],
    device const DockPose        *candidate     [[buffer(1)]],
    device DockPose              *best          [[buffer(2)]],
    constant BatchedGAParams     &batchParams   [[buffer(3)]],
    uint                          tid           [[thread_position_in_grid]])
{
    if (tid >= batchParams.totalPoses) return;
    device DockPose &cur = current[tid];
    device DockPose &bp = best[tid];
    DockPose cand = candidate[tid];
    bool cv = isfinite(cand.energy) && cand.energy < 1e9f;
    bool accept = false;
    if (cv) {
        if (!isfinite(cur.energy) || cur.energy >= 1e9f || cand.energy < cur.energy) {
            accept = true;
        } else {
            float T = max(batchParams.mcTemperature, 0.01f);
            uint seed = tid * 1103515245u + batchParams.generation * 12345u;
            accept = gpuRandom(seed, 0) < exp((cur.energy - cand.energy) / T);
        }
    }
    if (accept) { cur = cand; if (cand.energy < bp.energy || !isfinite(bp.energy)) bp = cand; }
}

// MARK: - Parallel Tempering

kernel void mcPerturbReplica(
    device DockPose            *perturbed   [[buffer(0)]],
    device const DockPose      *current     [[buffer(1)]],
    constant GAParams          &gaParams    [[buffer(2)]],
    constant GridParams        &gridParams  [[buffer(3)]],
    constant ReplicaParams     &repParams   [[buffer(4)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= repParams.totalPoses) return;

    uint replicaIdx = tid / repParams.populationPerReplica;
    float T = repParams.temperatures[min(replicaIdx, MAX_REPLICAS - 1u)];
    float Tmin = repParams.temperatures[0];
    float tempScale = sqrt(max(T / max(Tmin, 0.01f), 1.0f));

    uint seed = tid * 314159u + repParams.swapGeneration * 271828u + replicaIdx * 999983u + 42u;
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
        // Translation perturbation — scaled by temperature
        float step = gaParams.translationStep * tempScale;
        float3 delta = step * randomInsideUnitSphere(seed, 10);
        dst.translation = clamp(src.translation + delta, gMin, gMax);
    } else if (which == 1u) {
        // Rotation perturbation — scaled by temperature
        float radius = max(gaParams.ligandRadius, 1.0f);
        float rotSpread = max(gaParams.rotationStep, gaParams.translationStep / radius) * tempScale;
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
        // Torsion perturbation — full random resample at high T, small perturbation at low T
        uint torsionIndex = which - 2u;
        if (torsionIndex < gaParams.numTorsions) {
            if (tempScale > 1.5f) {
                // High temperature: random resample for broad exploration
                dst.torsions[torsionIndex] = gpuRandom(seed, 30 + torsionIndex) * 2.0f * M_PI_F - M_PI_F;
            } else {
                // Low temperature: Gaussian-like perturbation
                float perturbation = (gpuRandom(seed, 30 + torsionIndex) - 0.5f) * gaParams.torsionStep * tempScale;
                float newVal = src.torsions[torsionIndex] + perturbation;
                // Wrap to [-pi, pi]
                newVal = newVal - 2.0f * M_PI_F * floor((newVal + M_PI_F) / (2.0f * M_PI_F));
                dst.torsions[torsionIndex] = newVal;
            }
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

/// Temperature-aware Metropolis acceptance for replica exchange.
/// Acceptance probability uses the replica's own temperature.
kernel void metropolisAcceptReplica(
    device DockPose            *current     [[buffer(0)]],
    device const DockPose      *candidate   [[buffer(1)]],
    device DockPose            *best        [[buffer(2)]],
    constant GAParams          &gaParams    [[buffer(3)]],
    constant ReplicaParams     &repParams   [[buffer(4)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= repParams.totalPoses) return;

    uint replicaIdx = tid / repParams.populationPerReplica;
    float T = max(repParams.temperatures[min(replicaIdx, MAX_REPLICAS - 1u)], 1e-4f);

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
            float acceptance = exp((cur.energy - cand.energy) / T);
            uint seed = tid * 92837111u + repParams.swapGeneration * 689287499u + replicaIdx * 456789u + 17u;
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

/// Replica exchange swap kernel.
/// Dispatched with one thread per adjacent replica pair.
/// Uses checkerboard pattern: even generations swap pairs (0,1),(2,3),...
/// odd generations swap pairs (1,2),(3,4),... to avoid race conditions.
kernel void replicaSwap(
    device DockPose            *population  [[buffer(0)]],
    device DockPose            *best        [[buffer(1)]],
    constant ReplicaParams     &repParams   [[buffer(2)]],
    uint                        tid         [[thread_position_in_grid]])
{
    uint numPairs = repParams.numReplicas - 1u;
    if (tid >= numPairs) return;

    // Checkerboard: even gen swaps even pairs, odd gen swaps odd pairs
    uint parity = repParams.swapGeneration & 1u;
    if ((tid & 1u) != parity) return;

    uint replicaI = tid;
    uint replicaJ = tid + 1u;

    // Find the best (lowest energy) pose in each replica
    uint startI = replicaI * repParams.populationPerReplica;
    uint startJ = replicaJ * repParams.populationPerReplica;
    uint ppReplica = repParams.populationPerReplica;

    float bestEI = 1e10f;
    uint bestIdxI = startI;
    for (uint k = 0; k < ppReplica; k++) {
        float e = population[startI + k].energy;
        if (isfinite(e) && e < bestEI) {
            bestEI = e;
            bestIdxI = startI + k;
        }
    }

    float bestEJ = 1e10f;
    uint bestIdxJ = startJ;
    for (uint k = 0; k < ppReplica; k++) {
        float e = population[startJ + k].energy;
        if (isfinite(e) && e < bestEJ) {
            bestEJ = e;
            bestIdxJ = startJ + k;
        }
    }

    // Skip if either replica has no valid poses
    if (!isfinite(bestEI) || !isfinite(bestEJ)) return;
    if (bestEI >= 1e9f || bestEJ >= 1e9f) return;

    float Ti = max(repParams.temperatures[replicaI], 1e-4f);
    float Tj = max(repParams.temperatures[replicaJ], 1e-4f);
    float betaI = 1.0f / Ti;
    float betaJ = 1.0f / Tj;

    // Metropolis swap criterion: P = min(1, exp((βi - βj)(Ei - Ej)))
    float delta = (betaI - betaJ) * (bestEI - bestEJ);
    float P = min(1.0f, exp(delta));

    uint seed = tid * 482711u + repParams.swapGeneration * 937373u + 777u;
    float r = gpuRandom(seed, 0);

    if (r < P) {
        // Swap entire replica populations by exchanging all poses
        for (uint k = 0; k < ppReplica; k++) {
            DockPose tmp = population[startI + k];
            population[startI + k] = population[startJ + k];
            population[startJ + k] = tmp;

            DockPose tmpBest = best[startI + k];
            best[startI + k] = best[startJ + k];
            best[startJ + k] = tmpBest;
        }
    }
}
