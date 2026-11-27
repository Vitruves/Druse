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
    float3 boxMin = searchBoxMin(gridParams);
    float3 boxMax = searchBoxMax(gridParams);

    for (uint a = 0; a < nAtoms; a++) {
        DockLigandAtom atom = ligandAtoms[ligand.atomStart + a];
        float3 r = quatRotate(pose.rotation, atom.position) + pose.translation;

        penalty += outOfBoxPenalty(r, boxMin, boxMax);
        if (!insideGrid(r, gridParams)) continue;

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
/// Three tiers to balance exploitation of the input conformer with exploration:
///   Tier 1 (70%): Reference conformer (torsions=0)
///   Tier 2 (25%): Small local torsion variants near the reference
///   Tier 3 (5%): Fully random torsions for alternative binding modes
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

    // The signed random offset below uses a radius, so scale a half-extent.
    float minHalfExtent = min(halfExtent.x, min(halfExtent.y, halfExtent.z));
    float focusSpread = min(minHalfExtent * 0.65f, 14.0f);

    // --- Translation: 40% focused near center, 60% full box ---
    // Tuned for broader pocket coverage (see docking_tuning.md).
    float3 spread;
    if (tid < popSize * 4 / 10) {
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

    // --- Torsions: reference-biased strategy ---
    // Redocking starts from a crystallographic conformer. Keep most initial
    // torsions near that conformer and reserve a smaller tail for broad search.
    float exactFraction = clamp(gaParams.torsionExactFraction, 0.0f, 1.0f);
    float localFraction = clamp(gaParams.torsionLocalFraction, 0.0f, 1.0f - exactFraction);
    float localAmplitude = max(gaParams.torsionLocalAmplitude, 0.0f);
    uint tier1End = uint(float(popSize) * exactFraction);
    uint tier2End = uint(float(popSize) * (exactFraction + localFraction));

    if (tid < tier1End) {
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            poses[tid].torsions[t] = 0.0f;
        }
    } else if (tid < tier2End) {
        for (uint t = 0; t < gaParams.numTorsions; t++) {
            poses[tid].torsions[t] = (gpuRandom(seed, 6 + t) * 2.0f - 1.0f) * localAmplitude;
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

/// Energy used for population ranking: non-finite and sentinel (1e10) energies sort last
/// so an unscored or NaN pose can never be mistaken for an elite.
inline float gaRankEnergy(float e) {
    return (isfinite(e) && e < 1e9f) ? e : 1e30f;
}

/// Wrap a torsion angle into [-pi, pi].
inline float gaWrapAngle(float a) {
    return a - 2.0f * M_PI_F * floor((a + M_PI_F) / (2.0f * M_PI_F));
}

/// GA evolution: elitist tournament selection + arithmetic/SLERP/uniform crossover +
/// per-gene mutation, with random immigrants replacing the worst-ranked slots.
///
/// Buffer layout and pose conventions are identical to `mcPerturb`
/// (offspring[0], population[1], GAParams[2], GridParams[3]) so the generation loop can
/// swap one proposal kernel for the other and keep the same scoring / local-search /
/// Metropolis-acceptance stages. Energies are "lower is better".
///
/// Elitism: each thread computes its own energy rank in the population (O(popSize), cheap
/// next to one scoring pass). The top `popSize/50` slots pass through unchanged, so the
/// survivor-selection kernel sees ΔE = 0 for them and the population's best poses cannot be
/// destroyed by a crossover generation; the periodic local search still refines them,
/// which only lowers their energy.
///
/// Every pose field is written on every path: offspring is a persistent buffer and
/// extraction reads back the energy decomposition, the chi angles and the torsion array,
/// so inherited-but-unwritten fields would leak stale values from earlier generations.
kernel void gaEvolve(
    device DockPose            *offspring    [[buffer(0)]],
    device const DockPose      *population   [[buffer(1)]],
    constant GAParams          &gaParams     [[buffer(2)]],
    constant GridParams        &gridParams   [[buffer(3)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint popSize = gaParams.populationSize;
    uint nTor = min(gaParams.numTorsions, 32u);
    uint seed = tid * 31337u + gaParams.generation * 99991u + gaParams.runSeed * 777767u;

    device DockPose &child = offspring[tid];

    // --- Energy rank of this slot (ties broken by index so ranks are a permutation) ---
    float myE = gaRankEnergy(population[tid].energy);
    uint rank = 0;
    for (uint i = 0; i < popSize; i++) {
        float e = gaRankEnergy(population[i].energy);
        if (e < myE || (e == myE && i < tid)) rank++;
    }

    // --- Elite pass-through: the best slots survive the generation intact ---
    uint numElite = max(popSize / 50u, 1u);
    if (rank < numElite) {
        child = population[tid];
        child.generation = int(gaParams.generation) + 1;
        child.energy = 1e10f;
        child.stericEnergy = 0.0f;
        child.hydrophobicEnergy = 0.0f;
        child.hbondEnergy = 0.0f;
        child.torsionPenalty = 0.0f;
        child.clashPenalty = 0.0f;
        child.drusinaCorrection = 0.0f;
        child.constraintPenalty = 0.0f;
        return;
    }

    float3 searchCenter = gridParams.searchCenter;
    float3 searchHalfExtent = gridParams.searchHalfExtent;
    float3 searchMin = searchCenter - searchHalfExtent;
    float3 searchMax = searchCenter + searchHalfExtent;

    // --- Random immigrants: every 3rd generation the worst-ranked 20% of slots draw a
    // fresh pose instead of a child. They compete against the slot they land in, which is
    // by construction one of the poorest poses in the population, so survivor selection
    // gives them a real chance of taking hold. Receptor chi angles are inherited: the
    // immigrant explores ligand placement, not sidechain packing.
    uint immigrantCount = max(popSize / 5u, 1u);
    bool immigrate = gaParams.generation > 0u
        && (gaParams.generation % 3u) == 0u
        && rank + immigrantCount >= popSize;
    if (immigrate) {
        child.translation = searchCenter + float3(
            (gpuRandom(seed, 200) * 2.0f - 1.0f) * searchHalfExtent.x,
            (gpuRandom(seed, 201) * 2.0f - 1.0f) * searchHalfExtent.y,
            (gpuRandom(seed, 202) * 2.0f - 1.0f) * searchHalfExtent.z
        );
        float u1 = gpuRandom(seed, 203);
        float u2 = gpuRandom(seed, 204) * 2.0f * M_PI_F;
        float u3 = gpuRandom(seed, 205) * 2.0f * M_PI_F;
        float sq1 = sqrt(1.0f - u1);
        float sq2 = sqrt(u1);
        child.rotation = normalize(float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3)));
        for (uint t = 0; t < 32u; t++) {
            child.torsions[t] = (t < nTor)
                ? gpuRandom(seed, 210 + t) * 2.0f * M_PI_F - M_PI_F
                : population[tid].torsions[t];
        }
        for (uint c = 0; c < 24u; c++) child.chiAngles[c] = population[tid].chiAngles[c];
        child.numChiAngles = population[tid].numChiAngles;
        child.numTorsions = int(nTor);
        child.generation = int(gaParams.generation) + 1;
        child.energy = 1e10f;
        child.stericEnergy = 0.0f;
        child.hydrophobicEnergy = 0.0f;
        child.hbondEnergy = 0.0f;
        child.torsionPenalty = 0.0f;
        child.clashPenalty = 0.0f;
        child.drusinaCorrection = 0.0f;
        child.constraintPenalty = 0.0f;
        return;
    }

    // --- Tournament selection (k = 3) for two parents ---
    uint a1 = min(uint(gpuRandom(seed, 0) * float(popSize)), popSize - 1u);
    uint b1 = min(uint(gpuRandom(seed, 1) * float(popSize)), popSize - 1u);
    uint c1 = min(uint(gpuRandom(seed, 2) * float(popSize)), popSize - 1u);
    uint parentA = a1;
    if (gaRankEnergy(population[b1].energy) < gaRankEnergy(population[parentA].energy)) parentA = b1;
    if (gaRankEnergy(population[c1].energy) < gaRankEnergy(population[parentA].energy)) parentA = c1;

    uint a2 = min(uint(gpuRandom(seed, 3) * float(popSize)), popSize - 1u);
    uint b2 = min(uint(gpuRandom(seed, 4) * float(popSize)), popSize - 1u);
    uint c2 = min(uint(gpuRandom(seed, 5) * float(popSize)), popSize - 1u);
    uint parentB = a2;
    if (gaRankEnergy(population[b2].energy) < gaRankEnergy(population[parentB].energy)) parentB = b2;
    if (gaRankEnergy(population[c2].energy) < gaRankEnergy(population[parentB].energy)) parentB = c2;

    // --- Crossover ---
    float alpha = gpuRandom(seed, 7);
    if (gpuRandom(seed, 6) < gaParams.crossoverRate) {
        // Arithmetic crossover for translation (both parents are in-box, so is the blend)
        child.translation = mix(population[parentA].translation, population[parentB].translation, alpha);

        // SLERP for the orientation quaternion
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

        // Uniform crossover for ligand torsions and receptor chi angles. Inactive torsion
        // slots track parent A so the tail of the array never holds stale values.
        for (uint t = 0; t < 32u; t++) {
            uint src = (t < nTor && gpuRandom(seed, 8 + t) < 0.5f) ? parentB : parentA;
            child.torsions[t] = population[src].torsions[t];
        }
        for (uint c = 0; c < 24u; c++) {
            uint src = (gpuRandom(seed, 40 + c) < 0.5f) ? parentB : parentA;
            child.chiAngles[c] = population[src].chiAngles[c];
        }
    } else {
        child.translation = population[parentA].translation;
        child.rotation = population[parentA].rotation;
        for (uint t = 0; t < 32u; t++) child.torsions[t] = population[parentA].torsions[t];
        for (uint c = 0; c < 24u; c++) child.chiAngles[c] = population[parentA].chiAngles[c];
    }
    child.numChiAngles = population[parentA].numChiAngles;

    // --- Mutation: translation (isotropic step, same calibration as mcPerturb) ---
    if (gpuRandom(seed, 64) < gaParams.mutationRate) {
        float3 delta = gaParams.translationStep * randomInsideUnitSphere(seed, 65);
        child.translation = clamp(child.translation + delta, searchMin, searchMax);
    }

    // --- Mutation: rotation ---
    if (gpuRandom(seed, 86) < gaParams.mutationRate) {
        float3 rotVec = gaParams.rotationStep * randomInsideUnitSphere(seed, 87);
        float angle = length(rotVec);
        if (angle > 1e-6f) {
            float3 axis = rotVec / angle;
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
    }

    // --- Mutation: torsions. Honours the same torsion-search policy as mcPerturb, so the
    // redocking / exploratory presets steer the GA the way they steer the MC path.
    float resetProbability = clamp(gaParams.torsionRandomResetProbability, 0.0f, 1.0f);
    float perturbationScale = max(gaParams.torsionPerturbationScale, 0.0f);
    for (uint t = 0; t < nTor; t++) {
        if (gpuRandom(seed, 108 + t) >= gaParams.mutationRate) continue;
        if (gpuRandom(seed, 140 + t) < resetProbability) {
            child.torsions[t] = gpuRandom(seed, 172 + t) * 2.0f * M_PI_F - M_PI_F;
        } else {
            float delta = (gpuRandom(seed, 172 + t) * 2.0f - 1.0f) * gaParams.torsionStep * perturbationScale;
            child.torsions[t] = gaWrapAngle(child.torsions[t] + delta);
        }
    }

    child.numTorsions = int(nTor);
    child.generation = int(gaParams.generation) + 1;
    child.energy = 1e10f;
    child.stericEnergy = 0.0f;
    child.hydrophobicEnergy = 0.0f;
    child.hbondEnergy = 0.0f;
    child.torsionPenalty = 0.0f;
    child.clashPenalty = 0.0f;
    child.drusinaCorrection = 0.0f;
    child.constraintPenalty = 0.0f;
}
