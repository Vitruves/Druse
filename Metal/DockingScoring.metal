// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "DockingCommon.h"
#include "InteractionCriteria.h"

// ============================================================================
// MARK: - Pose Scoring
// ============================================================================

/// Score all poses using exact Vina XS-type affinity maps with torsional flexibility.
/// One thread per pose. Grid maps are stored in half precision.
kernel void scorePoses(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    device const half          *affinityMaps  [[buffer(2)]],
    device const int32_t       *typeIndexLookup [[buffer(3)]],
    constant GridParams        &gridParams    [[buffer(4)]],
    constant GAParams          &gaParams      [[buffer(5)]],
    constant TorsionEdge       *torsionEdges  [[buffer(6)]],
    constant int32_t           *movingIndices [[buffer(7)]],
    constant uint32_t          *intraPairs   [[buffer(8)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tid           [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);

    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    float3 boxMin = searchBoxMin(gridParams);
    float3 boxMax = searchBoxMax(gridParams);

    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];

        penalty += outOfBoxPenalty(r, boxMin, boxMax);
        if (!insideGrid(r, gridParams)) continue;
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gridParams
        );
    }

    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;

    // Pharmacophore constraint penalty
    float cPen = evaluateConstraintPenalty(
        positions, ligandAtoms, nA, pharmaConstraints, pharmaParams);

    // Vina conf-independent: total = (inter + intra_delta) / (1 + w_rot * N_tors)
    // Intramolecular delta is INSIDE the normalization scope (Vina: conf_independent(inter + intra - ref)).
    float nRotF = float(nTorsions);
    float normFactor = vinaRotNormalization(nRotF);
    float normalizedE = (totalIntermolecular + intraDelta) * normFactor;

    pose.stericEnergy      = totalIntermolecular;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy       = 0.0f;
    pose.torsionPenalty    = normalizedE - totalIntermolecular - intraDelta;
    pose.clashPenalty      = wPenalty * penalty;
    pose.constraintPenalty = cPen;

    // Total Vina score = normalized(inter + intra_delta) + boundary penalty + constraint
    pose.energy = normalizedE + wPenalty * penalty + cPen;
}

/// Rescore poses against explicit receptor atoms instead of interpolated affinity maps.
/// This mirrors Vina's late non-cache pose rescoring more closely for top basin representatives.
/// Uses SIMD intrinsics: each SIMD group cooperatively scores one pose by distributing
/// the protein atom inner loop across SIMD lanes, then reducing with simd_sum.
/// Dispatch with (populationSize * simdWidth) threads so each SIMD group maps to one pose.
kernel void scorePosesExplicit(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    constant GridProteinAtom   *proteinAtoms  [[buffer(2)]],
    constant GridParams        &gridParams    [[buffer(3)]],
    constant GAParams          &gaParams      [[buffer(4)]],
    constant TorsionEdge       *torsionEdges  [[buffer(5)]],
    constant int32_t           *movingIndices [[buffer(6)]],
    constant uint32_t          *intraPairs    [[buffer(7)]],
    constant PharmacophoreConstraint *pharmaConstraints [[buffer(15)]],
    constant PharmacophoreParams &pharmaParams [[buffer(16)]],
    uint                        tid           [[thread_position_in_grid]],
    uint                        simdLane      [[thread_index_in_simdgroup]],
    uint                        simdSize      [[threads_per_simdgroup]])
{
    // Each SIMD group handles one pose; lanes split the protein atom loop
    uint poseIdx = tid / simdSize;
    if (poseIdx >= gaParams.populationSize) return;

    device DockPose &pose = poses[poseIdx];
    uint nAtoms = min(gaParams.numLigandAtoms, 128u);
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    transformAtoms(positions, ligandAtoms, nAtoms, pose, torsionEdges, movingIndices, nTorsions);

    float3 boxMin = searchBoxMin(gridParams);
    float3 boxMax = searchBoxMax(gridParams);

    float totalSteric = 0.0f;
    float totalHydrophobic = 0.0f;
    float totalHBond = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];

        penalty += outOfBoxPenalty(r, boxMin, boxMax);

        // SIMD-cooperative: distribute protein atoms across lanes, reduce with simd_sum
        float laneSteric = 0.0f;
        float laneHydrophobic = 0.0f;
        float laneHBond = 0.0f;
        for (uint p = simdLane; p < gridParams.numProteinAtoms; p += simdSize) {
            float dist = distance(r, proteinAtoms[p].position);
            if (dist < 8.0f) {
                VinaTerms terms = vinaPairEnergyDecomposed(ligandAtoms[a].vinaType, proteinAtoms[p].vinaType, dist);
                laneSteric += terms.steric;
                laneHydrophobic += terms.hydrophobic;
                laneHBond += terms.hbond;
            }
        }
        // Reduce partial sums across SIMD lanes
        totalSteric += simd_sum(laneSteric);
        totalHydrophobic += simd_sum(laneHydrophobic);
        totalHBond += simd_sum(laneHBond);
    }

    // Only lane 0 writes the final result (all lanes have identical reduced values)
    if (simdLane != 0) return;

    float totalIntermolecular = totalSteric + totalHydrophobic + totalHBond;
    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nAtoms, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;

    float cPen = evaluateConstraintPenalty(
        positions, ligandAtoms, nAtoms, pharmaConstraints, pharmaParams);

    float nRotF = float(nTorsions);
    float normFactor = vinaRotNormalization(nRotF);
    float normalizedE = (totalIntermolecular + intraDelta) * normFactor;

    pose.stericEnergy      = totalSteric;
    pose.hydrophobicEnergy = totalHydrophobic;
    pose.hbondEnergy       = totalHBond;
    pose.torsionPenalty    = normalizedE - totalIntermolecular - intraDelta;
    pose.clashPenalty      = wPenalty * penalty;
    pose.constraintPenalty = cPen;
    pose.energy            = normalizedE + wPenalty * penalty + cPen;
}

/// Decompose the Vina intermolecular energy of scored poses into steric, hydrophobic,
/// and H-bond components via explicit pair evaluation.  Only writes the three
/// decomposition fields — total energy and all other fields are left untouched.
/// Intended to run once on the final population after grid-based scoring.
kernel void decomposeVinaEnergy(
    device DockPose            *poses         [[buffer(0)]],
    constant DockLigandAtom    *ligandAtoms   [[buffer(1)]],
    constant GridProteinAtom   *proteinAtoms  [[buffer(2)]],
    constant GridParams        &gridParams    [[buffer(3)]],
    constant GAParams          &gaParams      [[buffer(4)]],
    constant TorsionEdge       *torsionEdges  [[buffer(5)]],
    constant int32_t           *movingIndices [[buffer(6)]],
    uint                        tid           [[thread_position_in_grid]],
    uint                        simdLane      [[thread_index_in_simdgroup]])
{
    uint simdSize = 32u;
    uint poseIdx = tid / simdSize;
    if (poseIdx >= gaParams.populationSize) return;

    device DockPose &pose = poses[poseIdx];
    uint nAtoms = min(gaParams.numLigandAtoms, 128u);
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    transformAtoms(positions, ligandAtoms, nAtoms, pose, torsionEdges, movingIndices, nTorsions);

    float totalSteric = 0.0f;
    float totalHydrophobic = 0.0f;
    float totalHBond = 0.0f;

    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];
        float laneSteric = 0.0f;
        float laneHydrophobic = 0.0f;
        float laneHBond = 0.0f;
        for (uint p = simdLane; p < gridParams.numProteinAtoms; p += simdSize) {
            float dist = distance(r, proteinAtoms[p].position);
            if (dist < 8.0f) {
                VinaTerms terms = vinaPairEnergyDecomposed(ligandAtoms[a].vinaType, proteinAtoms[p].vinaType, dist);
                laneSteric += terms.steric;
                laneHydrophobic += terms.hydrophobic;
                laneHBond += terms.hbond;
            }
        }
        totalSteric += simd_sum(laneSteric);
        totalHydrophobic += simd_sum(laneHydrophobic);
        totalHBond += simd_sum(laneHBond);
    }

    if (simdLane != 0) return;

    pose.stericEnergy      = totalSteric;
    pose.hydrophobicEnergy = totalHydrophobic;
    pose.hbondEnergy       = totalHBond;
}

// ============================================================================
// MARK: - Drusina Extended Scoring
// ============================================================================

/// Verdonk 2003 block/ramp function: linear interpolation between full score and zero.
/// Returns 1.0 when |x| <= inner, ramps to 0.0 at |x| >= outer, 0 beyond.
/// GPU-friendly (no transcendentals), smooth enough for GA search.
inline float drusinaRamp(float x, float inner, float outer) {
    float ax = abs(x);
    if (ax <= inner) return 1.0f;
    if (ax >= outer) return 0.0f;
    return (outer - ax) / (outer - inner);
}

/// Smooth angular gate: 0 at the criterion threshold, 1 at perfect alignment.
/// Replaces hard `if (cos > threshold) score *= cos` tests, which put a finite
/// step into the energy exactly at the threshold — two near-identical poses
/// then differ by a fixed jump, which is pose-ranking noise, not physics.
///
/// Hand-rolled rather than `smoothstep`, which is documented as undefined for
/// edge0 >= edge1; several criteria here run from a negative cosine threshold
/// down to -1 (σ-hole alignment, H-bond donor linearity).
inline float drusinaAngleGate(float cosValue, float cosThreshold, float cosIdeal) {
    float t = clamp((cosValue - cosThreshold) / (cosIdeal - cosThreshold), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/// Lateral displacement of `other` from the axis through `origin` along `normal`.
/// Two coplanar rings sitting side by side have a small centroid separation but
/// a large offset, and are not stacked (Salentin 2015 caps the offset at 2 Å).
inline float drusinaLateralOffset(float3 origin, float3 normal, float3 other) {
    float3 delta = other - origin;
    float axial = dot(delta, normal);
    float lateralSq = max(dot(delta, delta) - axial * axial, 0.0f);
    return sqrt(lateralSq);
}

/// Compute Drusina correction terms:
///   - π-π stacking, π-cation, halogen bond, metal coordination
///   - salt bridge (group-based, Donald 2011), amide-π stacking, chalcogen bond
///   - screened Coulomb (ε=4r), CH-π, H-bond directionality, desolvation,
///     torsion strain, cooperativity
/// Returns total correction energy (kcal/mol, negative = favorable).
///
/// Every geometric criterion comes from Metal/InteractionCriteria.h, which the
/// interaction detectors share. Do not inline cutoffs here.
///
/// Design principles (Verdonk 2003, Bissantz 2010, Donald 2011):
///   - Linear ramps, not steep Gaussians, for a smooth GA landscape
///   - Smooth angular gates, so no term steps discontinuously at a threshold
///   - ONE contribution per interaction site. Every term below takes the best
///     partner per site (per protein ring, per amide-accepting ring face, per
///     halogen, per charged group, per donor...) instead of summing over all
///     partners in range. Summing made the extended terms scale with local
///     atom density — burial depth and backbone count — rather than with
///     binding mode, and that density signal is uncorrelated with nativeness
///     (Benchmark/reports/drusina_regression_analysis.md, RC1/RC3).
inline DrusinaDecomposition computeDrusinaCorrections(
    thread float3                *positions,
    constant DockLigandAtom      *ligandAtoms,
    uint                          nAtoms,
    constant ProteinRingGPU      *proteinRings,
    constant LigandRingGPU       *ligandRings,
    constant float4              *proteinCations,
    constant DrusinaParams       &params,
    constant GridProteinAtom     *proteinAtoms,
    uint                          numProteinAtoms,
    constant HalogenBondInfo     *halogenInfo,
    constant ProteinAmideGPU     *proteinAmides,
    constant ChalcogenBondInfo   *chalcogenInfo,
    constant SaltBridgeGroupGPU  *saltBridgeGroups,
    device const half            *elecGrid,
    constant GridParams          &gridParams,
    constant ProteinChalcogenGPU *proteinChalcogens,
    constant TorsionStrainInfo   *torsionStrain,
    constant LigandHBondInfoGPU  *ligandHBondInfo,
    constant ProteinHBondInfoGPU *proteinHBondInfo)
{
    DrusinaDecomposition decomp = {};
    float drusinaE = 0.0f;
    int interactionCount = 0;

    // Per-term accumulators (weighted)
    float piPiE = 0.0f, piCationE = 0.0f, saltBridgeE = 0.0f, amidePiE = 0.0f;
    float halogenE = 0.0f, chalcogenE = 0.0f, metalE = 0.0f, coulombTermE = 0.0f;
    float chPiE = 0.0f, strainTermE = 0.0f, cooperativityE = 0.0f;
    float hbondDirE = 0.0f, desolvPolarE = 0.0f, desolvHydrophobicE = 0.0f;

    // Ligand centroid and bounding radius. The cull radius has to grow with the
    // ligand: a fixed 10 Å centroid cutoff silently dropped valid partners for
    // elongated ligands, whose atoms sit 6-7 Å from the centroid, making the
    // extended terms depend on ligand size. Culling at radius + reach is exact.
    float3 ligCentroid = float3(0);
    for (uint a = 0; a < nAtoms; a++) ligCentroid += positions[a];
    ligCentroid /= float(max(nAtoms, 1u));
    float ligandRadiusSq = 0.0f;
    for (uint a = 0; a < nAtoms; a++) {
        ligandRadiusSq = max(ligandRadiusSq, distance_squared(positions[a], ligCentroid));
    }
    const float cullRadius = sqrt(ligandRadiusSq) + kICMaxInteractionReach;
    const float cullRadiusSq = cullRadius * cullRadius;

    // Per-ligand-atom protein contact counts, shared by the Coulomb burial
    // scale and both desolvation terms. The protein loop is outermost so the
    // cull test runs once per protein atom instead of once per atom pair.
    const bool needPolarDesolv = (params.wDesolvPolar != 0.0f);
    const bool needHydroDesolv = (params.wDesolvHydrophobic != 0.0f);
    const bool needContacts = (params.wCoulomb != 0.0f) || needPolarDesolv || needHydroDesolv;
    uchar contactCount[128];
    uchar closeContactCount[128];
    bool hbondSatisfied[128];
    for (uint a = 0; a < nAtoms; a++) {
        contactCount[a] = 0;
        closeContactCount[a] = 0;
        hbondSatisfied[a] = false;
    }
    if (needContacts) {
        for (uint p = 0; p < numProteinAtoms; p++) {
            float3 pPos = proteinAtoms[p].position;
            if (distance_squared(ligCentroid, pPos) > cullRadiusSq) continue;
            int pType = proteinAtoms[p].vinaType;
            for (uint a = 0; a < nAtoms; a++) {
                float d = distance(positions[a], pPos);
                if (d < kICBurialContactRadius && contactCount[a] < 255) contactCount[a]++;
                if (d < kICExposureContactRadius && closeContactCount[a] < 255) closeContactCount[a]++;
                if (needPolarDesolv && d <= kICHBondDistMax &&
                    xsHBondPossible(ligandAtoms[a].vinaType, pType)) {
                    hbondSatisfied[a] = true;
                }
            }
        }
    }

    // Pre-compute ligand ring centroids and normals for reuse (π-π, π-cation, amide-π)
    const uint MAX_LIG_RINGS = 8u;
    float3 ligRingCentroid[MAX_LIG_RINGS];
    float3 ligRingNormal[MAX_LIG_RINGS];
    uint numValidLigRings = 0;

    for (uint lr = 0; lr < min(params.numLigandRings, MAX_LIG_RINGS); lr++) {
        LigandRingGPU ring = ligandRings[lr];
        int nR = min(ring.numAtoms, 6);
        if (nR < 3) continue;

        float3 centroid = float3(0);
        int valid = 0;
        for (int i = 0; i < nR; i++) {
            int idx = ring.atomIndices[i];
            if (idx >= 0 && uint(idx) < nAtoms) {
                centroid += positions[idx];
                valid++;
            }
        }
        if (valid < 3) continue;
        centroid /= float(valid);

        int i0 = ring.atomIndices[0], i1 = ring.atomIndices[1], i2 = ring.atomIndices[2];
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            uint(i0) >= nAtoms || uint(i1) >= nAtoms || uint(i2) >= nAtoms) continue;
        float3 v1 = positions[i1] - positions[i0];
        float3 v2 = positions[i2] - positions[i0];
        float3 norm = cross(v1, v2);
        float nLen = length(norm);
        if (nLen < 1e-6f) continue;
        norm /= nLen;

        ligRingCentroid[numValidLigRings] = centroid;
        ligRingNormal[numValidLigRings] = norm;
        numValidLigRings++;
    }

    // ---- π-π stacking (ligand rings vs protein rings) ----
    // A ring has two faces and an edge girdle, so it takes at most three
    // partners: the best face-to-face stack on each side plus the best
    // T-shaped contact. Summing over every protein ring in range let a
    // ring buried among aromatics collect an arbitrary number of stacks.
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];
        float bestFaceUp = 0.0f, bestFaceDown = 0.0f, bestEdge = 0.0f;

        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            float3 prCentroid = proteinRings[pr].centroid;
            if (distance_squared(ligCentroid, prCentroid) > cullRadiusSq) continue;
            float d = distance(centroid, prCentroid);
            if (d < kICPiStackDistMin || d > kICPiStackEdgeDistMax) continue;
            float3 prNormal = proteinRings[pr].normal;
            float dotN = abs(dot(norm, prNormal));

            // Face-to-face / parallel displaced
            if (dotN >= kICPiStackFaceCosMin && d <= kICPiStackFaceDistMax) {
                float offset = drusinaLateralOffset(centroid, norm, prCentroid);
                if (offset <= kICPiStackOffsetMax) {
                    float gate = drusinaAngleGate(dotN, kICPiStackFaceCosMin, 1.0f);
                    float score = params.wPiPi * gate *
                        drusinaRamp(d - kICPiStackFaceDistOpt,
                                    kICPiStackFaceRampInner, kICPiStackFaceRampOuter);
                    if (dot(prCentroid - centroid, norm) >= 0.0f) bestFaceUp = min(bestFaceUp, score);
                    else                                          bestFaceDown = min(bestFaceDown, score);
                }
            }
            // Edge-to-face / T-shaped
            else if (dotN <= kICPiStackEdgeCosMax) {
                float gate = drusinaAngleGate(dotN, kICPiStackEdgeCosMax, 0.0f);
                float score = params.wPiPi * kICPiStackEdgeScale * gate *
                    drusinaRamp(d - kICPiStackEdgeDistOpt,
                                kICPiStackEdgeRampInner, kICPiStackEdgeRampOuter);
                bestEdge = min(bestEdge, score);
            }
        }
        if (bestFaceUp   < 0.0f) { piPiE += bestFaceUp;   interactionCount++; }
        if (bestFaceDown < 0.0f) { piPiE += bestFaceDown; interactionCount++; }
        if (bestEdge     < 0.0f) { piPiE += bestEdge;     interactionCount++; }
    }

    // ---- π-cation: ligand ring vs protein cations (best cation per ring face) ----
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];
        float bestUp = 0.0f, bestDown = 0.0f;

        for (uint pc = 0; pc < params.numProteinCations; pc++) {
            float3 cPos = proteinCations[pc].xyz;
            if (distance_squared(ligCentroid, cPos) > cullRadiusSq) continue;
            float d = distance(centroid, cPos);
            if (d > kICPiCationDistMax) continue;
            if (drusinaLateralOffset(centroid, norm, cPos) > kICPiCationOffsetMax) continue;
            float cosA = abs(dot(normalize(cPos - centroid), norm));
            if (cosA < kICPiCationCosMin) continue;
            float score = params.wPiCation * drusinaAngleGate(cosA, kICPiCationCosMin, 1.0f) *
                drusinaRamp(d - kICPiCationDistOpt, kICPiCationRampInner, kICPiCationRampOuter);
            if (dot(cPos - centroid, norm) >= 0.0f) bestUp = min(bestUp, score);
            else                                    bestDown = min(bestDown, score);
        }
        if (bestUp   < 0.0f) { piCationE += bestUp;   interactionCount++; }
        if (bestDown < 0.0f) { piCationE += bestDown; interactionCount++; }
    }

    // ---- π-cation: protein rings vs ligand cations (best ligand cation per ring) ----
    for (uint pr = 0; pr < params.numProteinRings; pr++) {
        float3 prCentroid = proteinRings[pr].centroid;
        if (distance_squared(ligCentroid, prCentroid) > cullRadiusSq) continue;
        float3 prNormal = proteinRings[pr].normal;
        float best = 0.0f;
        for (uint a = 0; a < nAtoms; a++) {
            if (ligandAtoms[a].formalCharge <= 0) continue;
            float d = distance(positions[a], prCentroid);
            if (d > kICPiCationDistMax) continue;
            if (drusinaLateralOffset(prCentroid, prNormal, positions[a]) > kICPiCationOffsetMax) continue;
            float cosA = abs(dot(normalize(positions[a] - prCentroid), prNormal));
            if (cosA < kICPiCationCosMin) continue;
            best = min(best, params.wPiCation * drusinaAngleGate(cosA, kICPiCationCosMin, 1.0f) *
                drusinaRamp(d - kICPiCationDistOpt, kICPiCationRampInner, kICPiCationRampOuter));
        }
        if (best < 0.0f) { piCationE += best; interactionCount++; }
    }

    // ---- Salt bridge: one contribution per protein charged group ----
    // Donald 2011 pairs groups, not atoms. A ligand carboxylate has two
    // oxygens, and both used to score against the same Arg guanidinium.
    // Burial-weighted: exposed salt bridges contribute little (Bissantz 2010).
    // Fallback: sign of a large partial charge when the formal charge is 0
    // (handles neutral SMILES where ionizable groups are not protonated).
    for (uint g = 0; g < params.numSaltBridgeGroups; g++) {
        float3 gCentroid = saltBridgeGroups[g].centroid;
        if (distance_squared(ligCentroid, gCentroid) > cullRadiusSq) continue;
        int groupSign = saltBridgeGroups[g].chargeSign;
        float best = 0.0f;

        for (uint a = 0; a < nAtoms; a++) {
            int ligCharge = ligandAtoms[a].formalCharge;
            float chargeScale = 1.0f;
            if (ligCharge == 0) {
                float q = ligandAtoms[a].charge;
                if (q > kICSaltBridgeChargeThreshold) ligCharge = 1;
                else if (q < -kICSaltBridgeChargeThreshold) ligCharge = -1;
                else continue;
                chargeScale = kICSaltBridgePartialScale;
            }
            if (ligCharge * groupSign >= 0) continue;  // need opposite signs

            float d = distance(positions[a], gCentroid);
            if (d > kICSaltBridgeDistMax) continue;
            float distScore = drusinaRamp(d - kICSaltBridgeDistOpt,
                                          kICSaltBridgeRampInner, kICSaltBridgeRampOuter);
            best = min(best, params.wSaltBridge * distScore *
                             saltBridgeGroups[g].burialFactor * chargeScale);
        }
        if (best < 0.0f) { saltBridgeE += best; interactionCount++; }
    }

    // ---- Amide-π stacking: protein amide plane ↔ ligand aromatic ring ----
    // One amide per ring face. A typical protein contributes 100-300 amide
    // planes, nearly all backbone, so summing over them turned this term into
    // a burial-depth readout (regression report RC3).
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];
        float bestUp = 0.0f, bestDown = 0.0f;

        for (uint am = 0; am < params.numProteinAmides; am++) {
            float3 amCentroid = proteinAmides[am].centroid;
            if (distance_squared(ligCentroid, amCentroid) > cullRadiusSq) continue;
            float d = distance(centroid, amCentroid);
            if (d < kICAmidePiDistMin || d > kICAmidePiDistMax) continue;
            float dotN = abs(dot(norm, proteinAmides[am].normal));
            if (dotN < kICAmidePiCosMin) continue;
            if (drusinaLateralOffset(centroid, norm, amCentroid) > kICAmidePiOffsetMax) continue;

            float score = params.wAmideStack * drusinaAngleGate(dotN, kICAmidePiCosMin, 1.0f) *
                drusinaRamp(d - kICAmidePiDistOpt, kICAmidePiRampInner, kICAmidePiRampOuter);
            if (dot(amCentroid - centroid, norm) >= 0.0f) bestUp = min(bestUp, score);
            else                                          bestDown = min(bestDown, score);
        }
        if (bestUp   < 0.0f) { amidePiE += bestUp;   interactionCount++; }
        if (bestDown < 0.0f) { amidePiE += bestDown; interactionCount++; }
    }

    // ---- Halogen bonds: C-X...O/N with σ-hole angle check ----
    // One acceptor per halogen: a C-X σ-hole is a single lobe along the bond
    // axis and cannot engage two acceptors at once, but the old loop summed
    // over every acceptor in range (e.g. both carboxylate oxygens).
    // Fluorine has no usable σ-hole (Auffinger 2004) and is skipped.
    for (uint h = 0; h < params.numHalogens; h++) {
        int hi = halogenInfo[h].halogenAtomIndex;
        int ci = halogenInfo[h].carbonAtomIndex;
        int elemType = halogenInfo[h].elementType;
        if (hi < 0 || ci < 0 || uint(hi) >= nAtoms || uint(ci) >= nAtoms) continue;
        if (elemType == 0) continue;  // F

        float optDist = kICHalogenDistOptCl;
        if      (elemType == 2) optDist = kICHalogenDistOptBr;
        else if (elemType == 3) optDist = kICHalogenDistOptI;

        float3 halPos = positions[hi];
        float3 xToC = normalize(positions[ci] - halPos);
        float best = 0.0f;

        for (uint p = 0; p < numProteinAtoms; p++) {
            float3 pPos = proteinAtoms[p].position;
            if (distance_squared(ligCentroid, pPos) > cullRadiusSq) continue;
            if (!xsIsAcceptor(proteinAtoms[p].vinaType)) continue;

            float d = distance(halPos, pPos);
            if (d < kICHalogenDistMin || d > kICHalogenDistMax) continue;

            float cosTheta = dot(xToC, normalize(pPos - halPos));
            if (cosTheta > kICHalogenCosMax) continue;
            float angleFactor = drusinaAngleGate(cosTheta, kICHalogenCosMax, -1.0f);
            best = min(best, params.wHalogenBond * angleFactor *
                             drusinaRamp(d - optDist, kICHalogenRampInner, kICHalogenRampOuter));
        }
        if (best < 0.0f) { halogenE += best; interactionCount++; }
    }

    // ---- Chalcogen bonds: ligand C-S...O/N with σ-hole angle check ----
    // One acceptor per σ-hole. A thioether has two C-S bonds and therefore two
    // σ-holes, which arrive as two ChalcogenBondInfo records for the same S.
    for (uint ch = 0; ch < params.numChalcogens; ch++) {
        int si = chalcogenInfo[ch].sulfurAtomIndex;
        int ci = chalcogenInfo[ch].carbonAtomIndex;
        if (si < 0 || ci < 0 || uint(si) >= nAtoms || uint(ci) >= nAtoms) continue;
        float3 sPos = positions[si];
        float3 sToC = normalize(positions[ci] - sPos);
        float best = 0.0f;

        for (uint p = 0; p < numProteinAtoms; p++) {
            float3 pPos = proteinAtoms[p].position;
            if (distance_squared(ligCentroid, pPos) > cullRadiusSq) continue;
            if (!xsIsAcceptor(proteinAtoms[p].vinaType)) continue;

            float d = distance(sPos, pPos);
            if (d < kICChalcogenDistMin || d > kICChalcogenDistMax) continue;

            float cosTheta = dot(sToC, normalize(pPos - sPos));
            if (cosTheta > kICChalcogenCosMax) continue;
            float angleFactor = drusinaAngleGate(cosTheta, kICChalcogenCosMax, -1.0f);
            best = min(best, params.wChalcogenBond * angleFactor *
                             drusinaRamp(d - kICChalcogenDistOpt,
                                         kICChalcogenRampInner, kICChalcogenRampOuter));
        }
        if (best < 0.0f) { chalcogenE += best; interactionCount++; }
    }

    // ---- Protein chalcogen bonds: protein S (Met/Cys) → ligand N/O ----
    for (uint pch = 0; pch < params.numProteinChalcogens; pch++) {
        float3 sPos = proteinChalcogens[pch].position;
        if (distance_squared(ligCentroid, sPos) > cullRadiusSq) continue;
        float3 sToC = proteinChalcogens[pch].bondedCDir;  // σ-hole lies opposite
        float best = 0.0f;

        for (uint a = 0; a < nAtoms; a++) {
            if (!xsIsAcceptor(ligandAtoms[a].vinaType)) continue;

            float d = distance(positions[a], sPos);
            if (d < kICChalcogenDistMin || d > kICChalcogenDistMax) continue;

            float cosTheta = dot(sToC, normalize(positions[a] - sPos));
            if (cosTheta > kICChalcogenCosMax) continue;
            float angleFactor = drusinaAngleGate(cosTheta, kICChalcogenCosMax, -1.0f);
            best = min(best, params.wChalcogenBond * angleFactor *
                             drusinaRamp(d - kICChalcogenDistOpt,
                                         kICChalcogenRampInner, kICChalcogenRampOuter));
        }
        if (best < 0.0f) { chalcogenE += best; interactionCount++; }
    }

    // ---- Metal coordination (best coordinating ligand atom per metal) ----
    // Verdonk 2003: the scoring funnel reaches further than the coordination
    // shell so the GA has a gradient to follow into the site.
    for (uint p = 0; p < numProteinAtoms; p++) {
        if (proteinAtoms[p].vinaType != VINA_MET_D) continue;
        float3 mPos = proteinAtoms[p].position;
        if (distance_squared(ligCentroid, mPos) > cullRadiusSq) continue;
        float bestMetalScore = 0.0f;
        for (uint a = 0; a < nAtoms; a++) {
            int lt = ligandAtoms[a].vinaType;
            bool isCoord = (lt == VINA_N_A || lt == VINA_N_DA || lt == VINA_N_D ||
                            lt == VINA_O_A || lt == VINA_O_DA || lt == VINA_S_P);
            if (!isCoord) continue;
            float d = distance(positions[a], mPos);
            if (d > kICMetalScoreCutoff) continue;
            float score = drusinaRamp(d - kICMetalDistOpt, kICMetalRampInner, kICMetalRampOuter);
            bestMetalScore = min(bestMetalScore, params.wMetalCoord * score);
        }
        if (bestMetalScore < 0.0f) {
            metalE += bestMetalScore;
            interactionCount++;
        }
    }

    // ---- Screened Coulomb (ε=4r, grid-based) ----
    // Φ(r) is precomputed from protein charges over the whole box, so unlike
    // every geometric term above it is long-ranged and rewards a pose for
    // sitting anywhere favourable in the box (regression report RC2). Scaling
    // each atom's contribution by its protein contact count keeps the term
    // pocket-local: a ligand atom with no protein neighbour is screened by
    // water, not by the protein interior (Bissantz 2010 on exposed ion pairs).
    if (params.wCoulomb != 0.0f) {
        float coulombE = 0.0f;
        float3 elecGridMin = gridParams.origin;
        float3 elecGridMax = gridUpperBound(gridParams);
        for (uint a = 0; a < nAtoms; a++) {
            float3 r = positions[a];
            if (any(r < elecGridMin) || any(r > elecGridMax)) continue;
            float burial = mix(kICCoulombBurialMinScale, 1.0f,
                               clamp(float(contactCount[a]) / float(kICBurialPolarMinContacts),
                                     0.0f, 1.0f));
            float phi = trilinearInterpolate(elecGrid, r, gridParams.origin,
                                              gridParams.spacing, gridParams.dims);
            coulombE += ligandAtoms[a].charge * phi * burial;
        }
        coulombTermE = params.wCoulomb * coulombE;
    }

    // ---- CH-π interactions: ligand aliphatic C → protein aromatic rings ----
    // One CH per ring: a ring face accepts a single CH-π contact, but the old
    // loop let every aliphatic carbon within range of the same Phe score, so a
    // cyclohexyl ring sitting on a Phe collected six contributions.
    for (uint pr = 0; pr < params.numProteinRings; pr++) {
        float3 prCentroid = proteinRings[pr].centroid;
        if (distance_squared(ligCentroid, prCentroid) > cullRadiusSq) continue;
        float3 prNormal = proteinRings[pr].normal;
        float bestUp = 0.0f, bestDown = 0.0f;

        for (uint a = 0; a < nAtoms; a++) {
            if (ligandAtoms[a].vinaType != VINA_C_H) continue;
            if (ligandAtoms[a].flags & LIGATOM_FLAG_AROMATIC) continue;
            float3 cPos = positions[a];
            float d = distance(cPos, prCentroid);
            if (d < kICCHPiDistMin || d > kICCHPiDistMax) continue;
            if (drusinaLateralOffset(prCentroid, prNormal, cPos) > kICCHPiOffsetMax) continue;
            float cosA = abs(dot(normalize(cPos - prCentroid), prNormal));
            if (cosA < kICCHPiCosMin) continue;
            float score = params.wCHPi * drusinaAngleGate(cosA, kICCHPiCosMin, 1.0f) *
                drusinaRamp(d - kICCHPiDistOpt, kICCHPiRampInner, kICCHPiRampOuter);
            if (dot(cPos - prCentroid, prNormal) >= 0.0f) bestUp = min(bestUp, score);
            else                                          bestDown = min(bestDown, score);
        }
        if (bestUp   < 0.0f) { chPiE += bestUp;   interactionCount++; }
        if (bestDown < 0.0f) { chPiE += bestDown; interactionCount++; }
    }

    // ---- Torsion strain: amide planarity penalty ----
    // Amide bonds should be planar (φ ≈ 0 or π). Penalize deviation.
    float strainE = 0.0f;
    for (uint t = 0; t < params.numTorsionStrains; t++) {
        TorsionStrainInfo ts = torsionStrain[t];
        if (ts.atom0 < 0 || ts.atom1 < 0 || ts.atom2 < 0 || ts.atom3 < 0 ||
            uint(ts.atom0) >= nAtoms || uint(ts.atom1) >= nAtoms ||
            uint(ts.atom2) >= nAtoms || uint(ts.atom3) >= nAtoms) continue;
        float3 b1 = positions[ts.atom1] - positions[ts.atom0];
        float3 b2 = positions[ts.atom2] - positions[ts.atom1];
        float3 b3 = positions[ts.atom3] - positions[ts.atom2];
        float3 n1 = cross(b1, b2);
        float3 n2 = cross(b2, b3);
        float ln1 = length(n1), ln2 = length(n2);
        if (ln1 < 1e-6f || ln2 < 1e-6f) continue;
        n1 /= ln1; n2 /= ln2;
        float cosPhi = clamp(dot(n1, n2), -1.0f, 1.0f);
        float phi = acos(cosPhi);
        // Amide: should be 0 or π. Minimum deviation from either.
        float dev = min(phi, M_PI_F - phi);
        strainE += ts.forceConstant * dev * dev;
    }
    strainTermE = params.wTorsionStrain * strainE;

    // ---- H-bond directionality correction ----
    // Bonus-only: reward well-directed H-bonds that Vina already scores by
    // distance. One partner per ligand donor/acceptor entry — the old loop
    // summed over every protein partner in range, so a ligand hydroxyl facing
    // a Ser (which contributes a donor AND an acceptor entry) plus a backbone
    // N and O collected four directionality bonuses for one H-bond.
    // Reference: Korb 2009 (PLANTS), Verdonk 2003 (GoldScore)
    for (uint lh = 0; lh < params.numLigandHBondAtoms; lh++) {
        int li = ligandHBondInfo[lh].atomIndex;
        int lAnt = ligandHBondInfo[lh].antecedentIndex;
        bool ligIsDonor = ligandHBondInfo[lh].isDonor != 0;
        if (li < 0 || lAnt < 0 || uint(li) >= nAtoms || uint(lAnt) >= nAtoms) continue;

        float3 ligPos = positions[li];
        float3 ligAntPos = positions[lAnt];
        float best = 0.0f;

        for (uint ph = 0; ph < params.numProteinHBondAtoms; ph++) {
            float3 protPos = proteinHBondInfo[ph].position;
            if (distance_squared(ligCentroid, protPos) > cullRadiusSq) continue;

            bool protIsDonor = proteinHBondInfo[ph].isDonor > 0.5f;
            if (ligIsDonor == protIsDonor) continue;

            float d = distance(ligPos, protPos);
            if (d < kICHBondDistMin || d > kICHBondDistMax) continue;

            float3 protAntPos = proteinHBondInfo[ph].antecedent;

            // Distance factor: strongest at the CSD optimum, fading at DistMax
            float distFactor = drusinaRamp(d - kICHBondDistOpt,
                                           kICHBondRampInner, kICHBondRampOuter);

            // Donor angle: X-D...A, optimal 180°
            float3 donorPos, donorAntPos, acceptorPos;
            if (ligIsDonor) {
                donorPos = ligPos; donorAntPos = ligAntPos; acceptorPos = protPos;
            } else {
                donorPos = protPos; donorAntPos = protAntPos; acceptorPos = ligPos;
            }
            float cosDonor = dot(normalize(donorAntPos - donorPos),
                                 normalize(acceptorPos - donorPos));
            float donorAngleScore = drusinaAngleGate(cosDonor, kICHBondCosDonorBad, kICHBondCosDonorGood);

            // Acceptor angle: D...A-Y, optimal ~100° (sp2 lone pair)
            float3 accPos, accAntPos;
            if (ligIsDonor) {
                accPos = protPos; accAntPos = protAntPos;
            } else {
                accPos = ligPos; accAntPos = ligAntPos;
            }
            float cosAcc = dot(normalize(donorPos - accPos), normalize(accAntPos - accPos));
            float accAngleScore = drusinaAngleGate(cosAcc, kICHBondCosAccBad, kICHBondCosAccGood);

            // Pure bonus: only reward good angles (no penalty)
            float angleFactor = donorAngleScore * accAngleScore;
            if (angleFactor <= kICHBondAngleFactorMin) continue;
            best = min(best, params.wHBondDir * angleFactor * distFactor);
        }
        if (best < 0.0f) { hbondDirE += best; interactionCount++; }
    }

    // ---- Polar desolvation penalty ----
    // Penalize DEEPLY BURIED ligand polar atoms that lack H-bond satisfaction.
    // Surface-facing polars are not penalized.
    // Reference: HYDE scoring (Schneider 2013)
    if (needPolarDesolv) {
        for (uint a = 0; a < nAtoms; a++) {
            int lt = ligandAtoms[a].vinaType;
            bool isPolar = (lt == VINA_N_A || lt == VINA_N_D || lt == VINA_N_DA ||
                            lt == VINA_O_A || lt == VINA_O_D || lt == VINA_O_DA);
            if (!isPolar || hbondSatisfied[a]) continue;
            int nearby = int(contactCount[a]);
            if (nearby < kICBurialPolarMinContacts) continue;
            float burial = clamp(float(nearby - kICBurialPolarMinContacts) /
                                 float(kICBurialPolarFullContacts - kICBurialPolarMinContacts),
                                 0.0f, 1.0f);
            desolvPolarE += params.wDesolvPolar * burial;
        }
    }

    // ---- Hydrophobic desolvation penalty ----
    // Penalize ligand hydrophobic atoms dangling into solvent rather than
    // buried in the pocket. Reference: Böhm 1994, ChemScore (Eldridge 1997)
    if (needHydroDesolv) {
        for (uint a = 0; a < nAtoms; a++) {
            if (ligandAtoms[a].vinaType != VINA_C_H) continue;
            int nearby = int(closeContactCount[a]);
            if (nearby >= kICExposureMinContacts) continue;
            float exposure = 1.0f - float(nearby) / float(kICExposureMinContacts);
            desolvHydrophobicE += params.wDesolvHydrophobic * exposure;
        }
    }

    // ---- Cooperativity bonus: reward multiple simultaneous interactions ----
    // Simple count-based: bonus for 2+ scored interaction types
    if (interactionCount > 1) {
        cooperativityE = params.wCooperativity * float(interactionCount - 1);
    }

    // Sum all terms
    drusinaE = piPiE + piCationE + saltBridgeE + amidePiE + halogenE
             + chalcogenE + metalE + coulombTermE + chPiE + strainTermE
             + cooperativityE + hbondDirE + desolvPolarE + desolvHydrophobicE;

    // Absolute safety cap. The Vina-relative bound that actually governs
    // ranking is applied by the callers, which know the pose's Vina energy
    // (see drusinaBoundedCorrection).
    drusinaE = max(drusinaE, params.minCorrection);

    // Fill decomposition struct
    decomp.piPi = piPiE;
    decomp.piCation = piCationE;
    decomp.saltBridge = saltBridgeE;
    decomp.amidePi = amidePiE;
    decomp.halogenBond = halogenE;
    decomp.chalcogenBond = chalcogenE;
    decomp.metalCoord = metalE;
    decomp.coulomb = coulombTermE;
    decomp.chPi = chPiE;
    decomp.torsionStrain = strainTermE;
    decomp.cooperativity = cooperativityE;
    decomp.hbondDir = hbondDirE;
    decomp.desolvPolar = desolvPolarE;
    decomp.desolvHydrophobic = desolvHydrophobicE;
    decomp.total = drusinaE;

    return decomp;
}

/// Bound the Drusina correction against the pose's own Vina energy.
///
/// Two separate guards, both needed (regression report RC1):
///   1. Steric-sanity gate — a pose Vina calls a clash may not collect
///      interaction bonuses at all. `vinaE` must include the boundary/clash
///      penalty, not just the normalized interaction sum.
///   2. Relative bound — a favourable correction may not exceed a fixed
///      fraction of the favourable Vina energy. An absolute floor cannot do
///      this: it lets the correction outvote Vina for weakly bound poses while
///      staying inert for strongly bound ones, which is exactly backwards.
///      A pose with vinaE >= 0 therefore gets no bonus at all.
inline float drusinaBoundedCorrection(float rawCorrection, float vinaE,
                                      constant DrusinaParams &params) {
    float gated = rawCorrection * drusinaVinaQuality(vinaE, params);
    float relativeFloor = -kICDrusinaMaxVinaFraction * max(-vinaE, 0.0f);
    return max(gated, relativeFloor);
}

/// Score poses with Vina grid maps + Drusina extended interaction corrections.
/// Used during GA when Drusina scoring is selected.
kernel void scorePosesDrusina(
    device DockPose              *poses           [[buffer(0)]],
    constant DockLigandAtom      *ligandAtoms     [[buffer(1)]],
    device const half            *affinityMaps    [[buffer(2)]],
    device const int32_t         *typeIndexLookup [[buffer(3)]],
    constant GridParams          &gridParams      [[buffer(4)]],
    constant GAParams            &gaParams        [[buffer(5)]],
    constant TorsionEdge         *torsionEdges    [[buffer(6)]],
    constant int32_t             *movingIndices   [[buffer(7)]],
    constant uint32_t            *intraPairs      [[buffer(8)]],
    constant ProteinRingGPU      *proteinRings    [[buffer(9)]],
    constant LigandRingGPU       *ligandRings     [[buffer(10)]],
    constant float4              *proteinCations  [[buffer(11)]],
    constant DrusinaParams       &drusinaParams   [[buffer(12)]],
    constant GridProteinAtom     *proteinAtoms    [[buffer(13)]],
    constant HalogenBondInfo     *halogenInfo     [[buffer(14)]],
    constant ProteinAmideGPU     *proteinAmides   [[buffer(15)]],
    constant ChalcogenBondInfo   *chalcogenInfo   [[buffer(16)]],
    constant SaltBridgeGroupGPU  *saltBridgeGroups [[buffer(17)]],
    device const half            *elecGrid        [[buffer(18)]],
    constant ProteinChalcogenGPU *proteinChalcogens [[buffer(19)]],
    constant TorsionStrainInfo   *torsionStrain   [[buffer(20)]],
    constant LigandHBondInfoGPU  *ligandHBondInfo [[buffer(21)]],
    constant ProteinHBondInfoGPU *proteinHBondInfo [[buffer(22)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    // --- Base Vina scoring from grid maps ---
    float3 boxMin = searchBoxMin(gridParams);
    float3 boxMax = searchBoxMax(gridParams);
    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];
        penalty += outOfBoxPenalty(r, boxMin, boxMax);
        if (!insideGrid(r, gridParams)) continue;
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gridParams);
    }

    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;
    float nRotF = float(nTorsions);
    float normFactor = vinaRotNormalization(nRotF);
    float normalizedE = (totalIntermolecular + intraDelta) * normFactor;

    // --- Drusina corrections, bounded against this pose's Vina energy ---
    DrusinaDecomposition decomp = computeDrusinaCorrections(
        positions, ligandAtoms, nA,
        proteinRings, ligandRings, proteinCations, drusinaParams,
        proteinAtoms, gridParams.numProteinAtoms, halogenInfo,
        proteinAmides, chalcogenInfo, saltBridgeGroups,
        elecGrid, gridParams, proteinChalcogens, torsionStrain,
        ligandHBondInfo, proteinHBondInfo);

    pose.stericEnergy      = totalIntermolecular;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy       = 0.0f;
    pose.torsionPenalty    = normalizedE - totalIntermolecular - intraDelta;
    pose.clashPenalty      = wPenalty * penalty;

    // The boundary penalty is part of the pose's steric sanity, so it belongs
    // in the Vina energy the gate and the relative bound see.
    float vinaE = normalizedE + wPenalty * penalty;
    float drusinaE = drusinaBoundedCorrection(decomp.total, vinaE, drusinaParams);

    pose.drusinaCorrection = drusinaE;
    pose.energy = vinaE + drusinaE;
}

/// Apply Drusina corrections to already-scored poses (e.g., after explicit Vina rescoring).
kernel void applyDrusinaCorrection(
    device DockPose              *poses           [[buffer(0)]],
    constant DockLigandAtom      *ligandAtoms     [[buffer(1)]],
    constant GAParams            &gaParams        [[buffer(2)]],
    constant TorsionEdge         *torsionEdges    [[buffer(3)]],
    constant int32_t             *movingIndices   [[buffer(4)]],
    constant ProteinRingGPU      *proteinRings    [[buffer(5)]],
    constant LigandRingGPU       *ligandRings     [[buffer(6)]],
    constant float4              *proteinCations  [[buffer(7)]],
    constant DrusinaParams       &drusinaParams   [[buffer(8)]],
    constant GridProteinAtom     *proteinAtoms    [[buffer(9)]],
    constant GridParams          &gridParams      [[buffer(10)]],
    constant HalogenBondInfo     *halogenInfo     [[buffer(11)]],
    constant ProteinAmideGPU     *proteinAmides   [[buffer(12)]],
    constant ChalcogenBondInfo   *chalcogenInfo   [[buffer(13)]],
    constant SaltBridgeGroupGPU  *saltBridgeGroups [[buffer(14)]],
    device const half            *elecGrid        [[buffer(15)]],
    constant ProteinChalcogenGPU *proteinChalcogens [[buffer(16)]],
    constant TorsionStrainInfo   *torsionStrain   [[buffer(17)]],
    constant LigandHBondInfoGPU  *ligandHBondInfo [[buffer(18)]],
    constant ProteinHBondInfoGPU *proteinHBondInfo [[buffer(19)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    DrusinaDecomposition decomp = computeDrusinaCorrections(
        positions, ligandAtoms, nA,
        proteinRings, ligandRings, proteinCations, drusinaParams,
        proteinAtoms, gridParams.numProteinAtoms, halogenInfo,
        proteinAmides, chalcogenInfo, saltBridgeGroups,
        elecGrid, gridParams, proteinChalcogens, torsionStrain,
        ligandHBondInfo, proteinHBondInfo);

    // pose.energy already carries the Vina total including the clash penalty.
    float drusinaE = drusinaBoundedCorrection(decomp.total, pose.energy, drusinaParams);

    pose.drusinaCorrection = drusinaE;
    pose.energy += drusinaE;
}

/// Compute per-term Drusina decomposition for benchmark diagnostics.
/// Scores poses and writes DrusinaDecomposition to a side buffer (no vinaQuality attenuation).
kernel void scorePosesDecomposition(
    device DockPose              *poses           [[buffer(0)]],
    constant DockLigandAtom      *ligandAtoms     [[buffer(1)]],
    constant GAParams            &gaParams        [[buffer(2)]],
    constant TorsionEdge         *torsionEdges    [[buffer(3)]],
    constant int32_t             *movingIndices   [[buffer(4)]],
    constant ProteinRingGPU      *proteinRings    [[buffer(5)]],
    constant LigandRingGPU       *ligandRings     [[buffer(6)]],
    constant float4              *proteinCations  [[buffer(7)]],
    constant DrusinaParams       &drusinaParams   [[buffer(8)]],
    constant GridProteinAtom     *proteinAtoms    [[buffer(9)]],
    constant GridParams          &gridParams      [[buffer(10)]],
    constant HalogenBondInfo     *halogenInfo     [[buffer(11)]],
    constant ProteinAmideGPU     *proteinAmides   [[buffer(12)]],
    constant ChalcogenBondInfo   *chalcogenInfo   [[buffer(13)]],
    constant SaltBridgeGroupGPU  *saltBridgeGroups [[buffer(14)]],
    device const half            *elecGrid        [[buffer(15)]],
    constant ProteinChalcogenGPU *proteinChalcogens [[buffer(16)]],
    constant TorsionStrainInfo   *torsionStrain   [[buffer(17)]],
    device DrusinaDecomposition  *decompositions  [[buffer(18)]],
    constant LigandHBondInfoGPU  *ligandHBondInfo [[buffer(19)]],
    constant ProteinHBondInfoGPU *proteinHBondInfo [[buffer(20)]],
    uint                          tid             [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = gaParams.numLigandAtoms;
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 positions[128];
    uint nA = min(nAtoms, 128u);
    transformAtoms(positions, ligandAtoms, nA, pose, torsionEdges, movingIndices, nTorsions);

    DrusinaDecomposition decomp = computeDrusinaCorrections(
        positions, ligandAtoms, nA,
        proteinRings, ligandRings, proteinCations, drusinaParams,
        proteinAtoms, gridParams.numProteinAtoms, halogenInfo,
        proteinAmides, chalcogenInfo, saltBridgeGroups,
        elecGrid, gridParams, proteinChalcogens, torsionStrain,
        ligandHBondInfo, proteinHBondInfo);

    decompositions[tid] = decomp;
}
