// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"

// MARK: - GA Operations

kernel void initializeBatchScreenPoses(
    device BatchScreenPose      *poses        [[buffer(0)]],
    constant BatchLigandInfo    *ligands      [[buffer(1)]],
    constant GridParams         &gridParams   [[buffer(2)]],
    constant BatchScreenParams  &params       [[buffer(3)]],
    uint                         tid          [[thread_position_in_grid]])
{
    if (tid >= params.totalPoses) return;

    uint ligandIndex = tid / max(params.posesPerLigand, 1u);
    uint poseIndex = tid % max(params.posesPerLigand, 1u);
    if (ligandIndex >= params.numLigands) return;

    BatchLigandInfo ligand = ligands[ligandIndex];
    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;
    float3 searchSize = halfExtent * 2.0f;
    float minDim = min(searchSize.x, min(searchSize.y, searchSize.z));
    float focusSpread = min(minDim * 0.45f, 10.0f);
    float3 spread = (poseIndex < max(ligand.poseCount * 3u / 5u, 1u))
        ? float3(focusSpread)
        : halfExtent;

    uint seed = params.seed ^ (ligandIndex * 2246822519u) ^ (poseIndex * 3266489917u);
    device BatchScreenPose &pose = poses[tid];
    pose.translation = center + float3(
        (gpuRandom(seed, 0) * 2.0f - 1.0f) * spread.x,
        (gpuRandom(seed, 1) * 2.0f - 1.0f) * spread.y,
        (gpuRandom(seed, 2) * 2.0f - 1.0f) * spread.z
    );

    float u1 = gpuRandom(seed, 3);
    float u2 = gpuRandom(seed, 4) * 2.0f * M_PI_F;
    float u3 = gpuRandom(seed, 5) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    pose.rotation = float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));
    pose.ligandIndex = ligandIndex;
    pose.poseIndex = poseIndex;
    pose.energy = 1e10f;
    pose.stericEnergy = 0.0f;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy = 0.0f;
    pose.clashPenalty = 0.0f;
}

kernel void scoreBatchRigidPoses(
    device BatchScreenPose      *poses         [[buffer(0)]],
    constant BatchLigandInfo    *ligands       [[buffer(1)]],
    constant DockLigandAtom     *ligandAtoms   [[buffer(2)]],
    device const half           *stericGrid    [[buffer(3)]],
    device const half           *hydrophobGrid [[buffer(4)]],
    device const half           *hbondGrid     [[buffer(5)]],
    constant GridParams         &gridParams    [[buffer(6)]],
    constant BatchScreenParams  &params        [[buffer(7)]],
    uint                         tid           [[thread_position_in_grid]])
{
    if (tid >= params.totalPoses) return;

    device BatchScreenPose &pose = poses[tid];
    if (pose.ligandIndex >= params.numLigands) return;

    BatchLigandInfo ligand = ligands[pose.ligandIndex];
    uint nAtoms = min(ligand.atomCount, 128u);

    float totalSteric = 0.0f;
    float totalHydrophobic = 0.0f;
    float totalHBond = 0.0f;
    float penalty = 0.0f;
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    for (uint a = 0; a < nAtoms; a++) {
        DockLigandAtom atom = ligandAtoms[ligand.atomStart + a];
        float3 r = quatRotate(pose.rotation, atom.position) + pose.translation;

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }

        totalSteric += trilinearInterpolate(stericGrid, r, gridParams.origin,
                                            gridParams.spacing, gridParams.dims);

        int ligVinaType = atom.vinaType;
        if (xsIsHydrophobic(ligVinaType)) {
            totalHydrophobic += trilinearInterpolate(hydrophobGrid, r, gridParams.origin,
                                                     gridParams.spacing, gridParams.dims);
        }

        bool ligHBondCapable = xsIsDonor(ligVinaType) || xsIsAcceptor(ligVinaType);
        if (ligHBondCapable) {
            totalHBond += trilinearInterpolate(hbondGrid, r, gridParams.origin,
                                               gridParams.spacing, gridParams.dims);
        }
    }

    pose.stericEnergy = totalSteric;
    pose.hydrophobicEnergy = totalHydrophobic;
    pose.hbondEnergy = totalHBond;
    pose.clashPenalty = wPenalty * penalty;
    pose.energy = totalSteric + totalHydrophobic + totalHBond + pose.clashPenalty;
}

/// Initialize population with conformer-aware seeding.
/// Three tiers to balance exploitation of the RDKit conformer with exploration:
///   Tier 1 (30%): Reference conformer (torsions=0) with positional + rotational diversity
///                  — these poses START from the chemically reasonable RDKit geometry
///   Tier 2 (30%): Reference torsions + small Gaussian noise (±30°) with random placement
///                  — conformational variants near the reference
///   Tier 3 (40%): Fully random torsions for exploration of alternative binding modes
kernel void initializePopulation(
    device DockPose            *poses        [[buffer(0)]],
    constant GridParams        &gridParams   [[buffer(1)]],
    constant GAParams          &gaParams     [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    // Per-run seed ensures truly independent MC trajectories.
    uint seed = tid * 747796405u + gaParams.runSeed * 2891336453u + 67890u;
    uint popSize = gaParams.populationSize;
    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;

    float3 searchSize = halfExtent * 2.0f;
    float minDim = min(searchSize.x, min(searchSize.y, searchSize.z));
    float focusSpread = min(minDim * 0.45f, 10.0f);

    // --- Translation: 60% focused near center, 40% full box ---
    // GPU parallelism (200 poses) compensates for less steps per trajectory vs Vina,
    // so we keep moderate focus but ensure broad coverage.
    float3 spread;
    if (tid < popSize * 6 / 10) {
        spread = float3(focusSpread);
    } else {
        spread = halfExtent;
    }

    poses[tid].translation = center + float3(
        (gpuRandom(seed, 0) * 2.0f - 1.0f) * spread.x,
        (gpuRandom(seed, 1) * 2.0f - 1.0f) * spread.y,
        (gpuRandom(seed, 2) * 2.0f - 1.0f) * spread.z
    );

    // --- Rotation: uniform random quaternion (Shoemake method) ---
    float u1 = gpuRandom(seed, 3);
    float u2 = gpuRandom(seed, 4) * 2.0f * M_PI_F;
    float u3 = gpuRandom(seed, 5) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    poses[tid].rotation = float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));

    // --- Torsions: two-tier strategy ---
    // Tier 1 (20%): reference conformer (torsions=0) for fast convergence near input geometry
    // Tier 2 (80%): fully random for broad exploration (Vina-aligned)
    uint tier1End = popSize / 5;   // 20% reference conformer

    if (tid < tier1End) {
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            poses[tid].torsions[t] = 0.0f;
        }
    } else {
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            poses[tid].torsions[t] = gpuRandom(seed, 6 + t) * 2.0f * M_PI_F - M_PI_F;
        }
    }

    poses[tid].numTorsions = int(gaParams.numTorsions);
    poses[tid].generation = 0;
    poses[tid].energy = 1e10f;
    poses[tid].stericEnergy = 0.0f;
    poses[tid].hydrophobicEnergy = 0.0f;
    poses[tid].hbondEnergy = 0.0f;
    poses[tid].torsionPenalty = 0.0f;
    poses[tid].clashPenalty = 0.0f;
    poses[tid].drusinaCorrection = 0.0f;
    poses[tid].constraintPenalty = 0.0f;

    // Initialize chi angles to zero (rigid receptor baseline)
    for (uint c = 0; c < 24; c++) {
        poses[tid].chiAngles[c] = 0.0f;
    }
    poses[tid].numChiAngles = 0;
}

/// GA evolution: tournament selection + arithmetic/SLERP crossover + Gaussian mutation.
/// Includes diversity injection every 10 generations for bottom 5% of population.
kernel void gaEvolve(
    device DockPose            *offspring    [[buffer(0)]],
    device const DockPose      *population   [[buffer(1)]],
    constant GAParams          &gaParams     [[buffer(2)]],
    constant GridParams        &gridParams   [[buffer(3)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint seed = tid * 31337u + gaParams.generation * 99991u + gaParams.runSeed * 777767u;
    uint popSize = gaParams.populationSize;

    // Tournament selection (k=3) for two parents
    uint a1 = uint(gpuRandom(seed, 0) * float(popSize)) % popSize;
    uint b1 = uint(gpuRandom(seed, 1) * float(popSize)) % popSize;
    uint c1 = uint(gpuRandom(seed, 2) * float(popSize)) % popSize;
    uint parentA = a1;
    if (population[b1].energy < population[parentA].energy) parentA = b1;
    if (population[c1].energy < population[parentA].energy) parentA = c1;

    uint a2 = uint(gpuRandom(seed, 3) * float(popSize)) % popSize;
    uint b2 = uint(gpuRandom(seed, 4) * float(popSize)) % popSize;
    uint c2 = uint(gpuRandom(seed, 5) * float(popSize)) % popSize;
    uint parentB = a2;
    if (population[b2].energy < population[parentB].energy) parentB = b2;
    if (population[c2].energy < population[parentB].energy) parentB = c2;

    device DockPose &child = offspring[tid];
    float alpha = gpuRandom(seed, 7);

    if (gpuRandom(seed, 6) < gaParams.crossoverRate) {
        // Arithmetic crossover for translation
        child.translation = mix(population[parentA].translation, population[parentB].translation, alpha);

        // SLERP for quaternion
        float4 qa = population[parentA].rotation;
        float4 qb = population[parentB].rotation;
        float dotQ = dot(qa, qb);
        if (dotQ < 0.0f) { qb = -qb; dotQ = -dotQ; }
        dotQ = clamp(dotQ, -1.0f, 1.0f);

        if (dotQ > 0.9995f) {
            child.rotation = normalize(mix(qa, qb, alpha));
        } else {
            float theta = acos(dotQ);  // keep precise acos for domain [-1,1] stability
            float st = sin(theta);
            child.rotation = normalize(sin((1.0f - alpha) * theta) / st * qa + sin(alpha * theta) / st * qb);
        }

        // Uniform crossover for torsions
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            child.torsions[t] = (gpuRandom(seed, 10 + t) < 0.5f)
                ? population[parentA].torsions[t] : population[parentB].torsions[t];
        }
    } else {
        child.translation = population[parentA].translation;
        child.rotation = population[parentA].rotation;
        for (uint t = 0; t < gaParams.numTorsions; t++)
            child.torsions[t] = population[parentA].torsions[t];
    }

    float3 searchCenter = gridParams.searchCenter;
    float3 searchHalfExtent = gridParams.searchHalfExtent;
    float3 searchMin = searchCenter - searchHalfExtent;
    float3 searchMax = searchCenter + searchHalfExtent;

    // Diversity injection: reinitialize bottom 20% with random poses every 5 generations.
    bool doInject = (gaParams.generation % 5 == 0) && (gaParams.generation > 0);
    uint injectThreshold = popSize - max(popSize / 5, 6u);  // bottom 20%
    if (doInject && tid >= injectThreshold) {
        child.translation = searchCenter + float3(
            (gpuRandom(seed, 80) * 2.0f - 1.0f) * searchHalfExtent.x,
            (gpuRandom(seed, 81) * 2.0f - 1.0f) * searchHalfExtent.y,
            (gpuRandom(seed, 82) * 2.0f - 1.0f) * searchHalfExtent.z
        );
        float u1 = gpuRandom(seed, 83);
        float u2 = gpuRandom(seed, 84) * 2.0f * M_PI_F;
        float u3 = gpuRandom(seed, 85) * 2.0f * M_PI_F;
        child.rotation = normalize(float4(sqrt(1-u1)*sin(u2), sqrt(1-u1)*cos(u2), sqrt(u1)*sin(u3), sqrt(u1)*cos(u3)));
        for (uint t = 0; t < gaParams.numTorsions; t++)
            child.torsions[t] = gpuRandom(seed, 90 + t) * 2.0f * M_PI_F - M_PI_F;
        child.numTorsions = int(gaParams.numTorsions);
        child.generation = int(gaParams.generation) + 1;
        child.energy = 1e10f;
        child.stericEnergy = 0.0f;
        child.hydrophobicEnergy = 0.0f;
        child.hbondEnergy = 0.0f;
        child.torsionPenalty = 0.0f;
        child.clashPenalty = 0.0f;
        return;
    }

    // Mutation -- translation (occasional large jumps for exploration)
    float mutRoll = gpuRandom(seed, 50);
    if (mutRoll < gaParams.mutationRate) {
        float step = gaParams.translationStep;
        // 20% of mutations are large jumps (5x step) for wider exploration
        if (mutRoll < gaParams.mutationRate * 0.2f) step *= 5.0f;
        child.translation += float3(
            (gpuRandom(seed, 51) * 2.0f - 1.0f) * step,
            (gpuRandom(seed, 52) * 2.0f - 1.0f) * step,
            (gpuRandom(seed, 53) * 2.0f - 1.0f) * step
        );
        child.translation = clamp(child.translation, searchMin, searchMax);
    }

    // Mutation -- rotation
    if (gpuRandom(seed, 54) < gaParams.mutationRate) {
        float angle = (gpuRandom(seed, 55) * 2.0f - 1.0f) * gaParams.rotationStep;
        float3 axis = normalize(float3(gpuRandom(seed, 56) - 0.5f, gpuRandom(seed, 57) - 0.5f, gpuRandom(seed, 58) - 0.5f));
        float ha = angle * 0.5f;
        float4 dq = float4(axis * sin(ha), cos(ha));
        float4 q = child.rotation;
        child.rotation = normalize(float4(
            dq.w*q.x + dq.x*q.w + dq.y*q.z - dq.z*q.y,
            dq.w*q.y - dq.x*q.z + dq.y*q.w + dq.z*q.x,
            dq.w*q.z + dq.x*q.y - dq.y*q.x + dq.z*q.w,
            dq.w*q.w - dq.x*q.x - dq.y*q.y - dq.z*q.z
        ));
    }

    // Mutation -- torsions
    for (uint t = 0; t < gaParams.numTorsions; t++) {
        if (gpuRandom(seed, 60 + t) < gaParams.mutationRate) {
            child.torsions[t] += (gpuRandom(seed, 100 + t) * 2.0f - 1.0f) * gaParams.torsionStep;
            if (child.torsions[t] > M_PI_F) child.torsions[t] -= 2.0f * M_PI_F;
            if (child.torsions[t] < -M_PI_F) child.torsions[t] += 2.0f * M_PI_F;
        }
    }

    child.numTorsions = int(gaParams.numTorsions);
    child.generation = int(gaParams.generation) + 1;
    child.energy = 1e10f;
    child.stericEnergy = 0.0f;
    child.hydrophobicEnergy = 0.0f;
    child.hbondEnergy = 0.0f;
    child.torsionPenalty = 0.0f;
    child.clashPenalty = 0.0f;
}
