#include "DockingCommon.h"

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

    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }
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

    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;

    float totalSteric = 0.0f;
    float totalHydrophobic = 0.0f;
    float totalHBond = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];

        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) {
            penalty += oopP;
            continue;
        }

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

/// Compute Drusina correction terms:
///   - π-π stacking, π-cation, halogen bond, metal coordination (original)
///   - salt bridge (group-based, Donald 2011), amide-π stacking, chalcogen bond (extended)
///   - screened Coulomb (ε=4r), CH-π, cooperativity, torsion strain (new)
/// Returns total correction energy (kcal/mol, negative = favorable).
///
/// Design principles (Verdonk 2003, Bissantz 2010, Donald 2011):
///   - Linear ramp functions instead of steep Gaussians for smoother GA landscape
///   - Group-based salt bridge scoring (1 contribution per charged-group pair)
///   - Burial-weighted salt bridges (exposed contribute little, Bissantz 2010)
///   - Corrected optimal distances from CSD/PDB statistics
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

    // Pre-compute ligand centroid for spatial gating of protein partners.
    // Protein features beyond ligandCutoff from centroid cannot interact with any
    // ligand atom (max ligand radius ~6Å + max interaction range ~4Å = 10Å).
    float3 ligCentroid = float3(0);
    for (uint a = 0; a < nAtoms; a++) ligCentroid += positions[a];
    ligCentroid /= float(max(nAtoms, 1u));
    const float ligandCutoff = 10.0f;
    const float ligandCutoffSq = ligandCutoff * ligandCutoff;

    // Pre-compute ligand ring centroids and normals for reuse (π-π + amide-π)
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
    // Bissantz 2010: parallel displaced 3.4-3.6 Å, T-shaped 4.5-5.5 Å
    // Using ramp functions for smooth landscape (Verdonk 2003)
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];

        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            if (distance_squared(ligCentroid, proteinRings[pr].centroid) > ligandCutoffSq) continue;
            float d = distance(centroid, proteinRings[pr].centroid);
            if (d < 3.2f || d > 5.8f) continue;
            float dotN = abs(dot(norm, proteinRings[pr].normal));

            // Face-to-face stacking (parallel, |dotN| > 0.8)
            // Optimal 3.5 Å (Bissantz 2010 Table 2), ramp from 3.3-4.5 Å
            if (dotN > 0.8f) {
                float dd = d - 3.5f;
                piPiE += params.wPiPi * dotN * drusinaRamp(dd, 0.3f, 1.0f);
                interactionCount++;
            }
            // Edge-to-face / T-shaped (perpendicular, |dotN| < 0.4)
            // Optimal 4.8 Å, ramp from 4.2-5.5 Å (Bissantz 2010)
            else if (dotN < 0.4f) {
                float dd = d - 4.8f;
                float perpFactor = 1.0f - dotN;  // stronger when more perpendicular
                piPiE += params.wPiPi * 0.6f * perpFactor * drusinaRamp(dd, 0.4f, 1.0f);
                interactionCount++;
            }
        }
    }

    // ---- π-cation: ligand ring vs protein cations ----
    // Bissantz 2010: cation 3.4-4.0 Å above ring, perpendicular approach preferred
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];

        for (uint pc = 0; pc < params.numProteinCations; pc++) {
            float3 cPos = proteinCations[pc].xyz;
            if (distance_squared(ligCentroid, cPos) > ligandCutoffSq) continue;
            float d = distance(centroid, cPos);
            if (d > 5.5f) continue;
            float3 toAtom = normalize(cPos - centroid);
            float cosA = abs(dot(toAtom, norm));
            // Require above-plane approach (cosA > 0.5 = within 60° of normal)
            if (cosA > 0.5f) {
                float dd = d - 3.8f;
                piCationE += params.wPiCation * cosA * drusinaRamp(dd, 0.4f, 1.2f);
                interactionCount++;
            }
        }
    }

    // ---- π-cation: protein rings vs ligand cations ----
    for (uint a = 0; a < nAtoms; a++) {
        if (ligandAtoms[a].formalCharge <= 0) continue;
        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            if (distance_squared(ligCentroid, proteinRings[pr].centroid) > ligandCutoffSq) continue;
            float d = distance(positions[a], proteinRings[pr].centroid);
            if (d > 5.5f) continue;
            float3 toAtom = normalize(positions[a] - proteinRings[pr].centroid);
            float cosA = abs(dot(toAtom, proteinRings[pr].normal));
            if (cosA > 0.5f) {
                float dd = d - 3.8f;
                piCationE += params.wPiCation * cosA * drusinaRamp(dd, 0.4f, 1.2f);
                interactionCount++;
            }
        }
    }

    // ---- Salt bridge: group-based scoring (Donald 2011, Bissantz 2010) ----
    // One contribution per (ligand charged atom, protein charged group) pair.
    // Optimal N-O distance 2.8 Å (Bissantz 2010 Table 2, CSD median).
    // Ramp: full score at 2.5-3.1 Å, fading to zero at 4.0 Å.
    // Burial-weighted: exposed salt bridges contribute little (Bissantz 2010).
    // Fallback: use sign of large Gasteiger partial charges when formal charge is 0
    // (handles neutral SMILES where ionizable groups aren't protonated).
    // Threshold 0.35 to only catch clearly ionic groups (amines ~0.3-0.4, carboxylates ~-0.4-0.5).
    for (uint a = 0; a < nAtoms; a++) {
        int ligCharge = ligandAtoms[a].formalCharge;
        float chargeScale = 1.0f;  // full weight for formal charges
        if (ligCharge == 0) {
            float q = ligandAtoms[a].charge;
            if (q > 0.35f) ligCharge = 1;
            else if (q < -0.35f) ligCharge = -1;
            else continue;
            chargeScale = 0.5f;  // partial charge inference is less certain
        }

        for (uint g = 0; g < params.numSaltBridgeGroups; g++) {
            if (distance_squared(ligCentroid, saltBridgeGroups[g].centroid) > ligandCutoffSq) continue;
            // Require opposite charges
            bool isSB = (ligCharge > 0 && saltBridgeGroups[g].chargeSign < 0) ||
                        (ligCharge < 0 && saltBridgeGroups[g].chargeSign > 0);
            if (!isSB) continue;

            float d = distance(positions[a], saltBridgeGroups[g].centroid);
            if (d > 4.0f) continue;

            // Ramp centered at 2.8 Å (Bissantz 2010: CSD median N-O = 2.79 Å)
            float dd = d - 2.8f;
            float distScore = drusinaRamp(dd, 0.3f, 1.2f);

            // Burial dampening (Bissantz 2010: exposed salt bridges ≈ no free energy)
            float burial = saltBridgeGroups[g].burialFactor;

            saltBridgeE += params.wSaltBridge * distScore * burial * chargeScale;
            interactionCount++;
        }
    }

    // ---- Amide-π stacking: protein backbone amide ↔ ligand aromatic ring ----
    // Bissantz 2010: amide-aryl distance 3.2-3.7 Å between planes
    // Harder 2013: optimal d=3.4 Å interplanar, parallel (|dotN|>0.8)
    for (uint lr = 0; lr < numValidLigRings; lr++) {
        float3 centroid = ligRingCentroid[lr];
        float3 norm = ligRingNormal[lr];

        for (uint am = 0; am < params.numProteinAmides; am++) {
            if (distance_squared(ligCentroid, proteinAmides[am].centroid) > ligandCutoffSq) continue;
            float d = distance(centroid, proteinAmides[am].centroid);
            if (d < 3.0f || d > 5.2f) continue;

            float dotN = abs(dot(norm, proteinAmides[am].normal));

            // Parallel stacking (Harder 2013: interplanar ≤30°, |dotN| > 0.8)
            if (dotN > 0.8f) {
                float dd = d - 3.6f;
                amidePiE += params.wAmideStack * dotN * drusinaRamp(dd, 0.3f, 0.9f);
                interactionCount++;
            }
            // Tilted/offset stacking (weaker, broader distance range)
            else if (dotN > 0.6f) {
                float dd = d - 4.0f;
                amidePiE += params.wAmideStack * 0.4f * dotN * drusinaRamp(dd, 0.3f, 1.0f);
                interactionCount++;
            }
        }
    }

    // ---- Halogen bonds: C-X...O/N with σ-hole angle check ----
    // Element-specific optimal distances (Scholfield 2013, Bissantz 2010):
    //   F: skip (too weak), Cl: 3.27 Å, Br: 3.24 Å, I: 3.17 Å
    // C-X...A angle preferred near 160-180° (sigma-hole alignment)
    for (uint h = 0; h < params.numHalogens; h++) {
        int hi = halogenInfo[h].halogenAtomIndex;
        int ci = halogenInfo[h].carbonAtomIndex;
        int elemType = halogenInfo[h].elementType;
        if (hi < 0 || ci < 0 || uint(hi) >= nAtoms || uint(ci) >= nAtoms) continue;
        if (elemType == 0) continue;  // F halogen bonds too weak — skip

        // Element-specific optimal distance (CSD statistics, Scholfield 2013)
        float optDist = 3.2f;
        if      (elemType == 1) optDist = 3.27f;  // Cl
        else if (elemType == 2) optDist = 3.24f;  // Br
        else if (elemType == 3) optDist = 3.17f;  // I

        float3 halPos = positions[hi];
        float3 carPos = positions[ci];

        for (uint p = 0; p < numProteinAtoms; p++) {
            int pType = proteinAtoms[p].vinaType;
            bool isAcceptor = (pType == VINA_N_A || pType == VINA_N_DA ||
                               pType == VINA_O_A || pType == VINA_O_DA);
            if (!isAcceptor) continue;

            float d = distance(halPos, proteinAtoms[p].position);
            if (d < 2.5f || d > 4.2f) continue;

            // σ-hole directionality: C-X...A angle should be > 140° (cosine < -0.766)
            float3 xToC = normalize(carPos - halPos);
            float3 xToA = normalize(proteinAtoms[p].position - halPos);
            float cosTheta = dot(xToC, xToA);
            if (cosTheta < -0.766f) {
                float dd = d - optDist;
                float angleFactor = (-cosTheta - 0.766f) / 0.234f;
                halogenE += params.wHalogenBond * angleFactor * drusinaRamp(dd, 0.3f, 0.8f);
                interactionCount++;
            }
        }
    }

    // ---- Chalcogen bonds: C-S...O/N with σ-hole angle check ----
    // Relaxed from 150° to 140° (Beno 2015, Pascoe 2017: S σ-hole valid at 140-180°)
    // Distance 2.8-4.2 Å, optimal ~3.3 Å (Bissantz 2010). Dual σ-holes for thioethers.
    for (uint ch = 0; ch < params.numChalcogens; ch++) {
        int si = chalcogenInfo[ch].sulfurAtomIndex;
        int ci = chalcogenInfo[ch].carbonAtomIndex;
        if (si < 0 || ci < 0 || uint(si) >= nAtoms || uint(ci) >= nAtoms) continue;
        float3 sPos = positions[si];
        float3 cPos = positions[ci];

        for (uint p = 0; p < numProteinAtoms; p++) {
            int pType = proteinAtoms[p].vinaType;
            bool isAcceptor = (pType == VINA_N_A || pType == VINA_N_DA ||
                               pType == VINA_O_A || pType == VINA_O_DA);
            if (!isAcceptor) continue;

            float d = distance(sPos, proteinAtoms[p].position);
            if (d < 2.8f || d > 4.2f) continue;

            // σ-hole directionality: C-S-A angle > 140° (cos < -0.766)
            float3 sToC = normalize(cPos - sPos);
            float3 sToA = normalize(proteinAtoms[p].position - sPos);
            float cosTheta = dot(sToC, sToA);
            if (cosTheta < -0.766f) {
                float dd = d - 3.3f;
                float angleFactor = (-cosTheta - 0.766f) / 0.234f;
                chalcogenE += params.wChalcogenBond * angleFactor * drusinaRamp(dd, 0.3f, 0.8f);
                interactionCount++;
            }
        }
    }

    // ---- Protein chalcogen bonds: protein S (Met/Cys) → ligand N/O ----
    // Bidirectional: protein sulfur σ-hole interacting with ligand acceptors
    for (uint pch = 0; pch < params.numProteinChalcogens; pch++) {
        float3 sPos = proteinChalcogens[pch].position;
        float3 bondedCDir = proteinChalcogens[pch].bondedCDir; // direction S→C

        for (uint a = 0; a < nAtoms; a++) {
            int lt = ligandAtoms[a].vinaType;
            bool isAcceptor = (lt == VINA_N_A || lt == VINA_N_DA ||
                               lt == VINA_O_A || lt == VINA_O_DA);
            if (!isAcceptor) continue;

            float d = distance(positions[a], sPos);
            if (d < 2.8f || d > 4.2f) continue;

            // σ-hole is opposite C-S bond: check S...A aligns with -bondedCDir
            float3 sToA = normalize(positions[a] - sPos);
            float cosTheta = dot(bondedCDir, sToA); // negative when aligned with σ-hole
            if (cosTheta < -0.766f) {
                float dd = d - 3.3f;
                float angleFactor = (-cosTheta - 0.766f) / 0.234f;
                chalcogenE += params.wChalcogenBond * angleFactor * drusinaRamp(dd, 0.3f, 0.8f);
                interactionCount++;
            }
        }
    }

    // ---- Enhanced metal coordination ----
    // Verdonk 2003: optimal distance ~2.0-2.4 Å for N/O/S → metal.
    // Moderate funnel: cutoff 4.5 Å, ramp out to 1.6 Å from optimal (full score 2.0-2.8 Å,
    // fading to zero at 4.0 Å). Cap at one interaction per metal to avoid multi-counting.
    for (uint p = 0; p < numProteinAtoms; p++) {
        if (proteinAtoms[p].vinaType != VINA_MET_D) continue;
        float bestMetalScore = 0.0f;
        for (uint a = 0; a < nAtoms; a++) {
            int lt = ligandAtoms[a].vinaType;
            bool isCoord = (lt == VINA_N_A || lt == VINA_N_DA || lt == VINA_N_D ||
                            lt == VINA_O_A || lt == VINA_O_DA || lt == VINA_S_P);
            if (!isCoord) continue;
            float d = distance(positions[a], proteinAtoms[p].position);
            if (d > 4.5f) continue;
            float dd = d - 2.4f;
            float score = drusinaRamp(dd, 0.4f, 1.6f);
            bestMetalScore = min(bestMetalScore, params.wMetalCoord * score);
        }
        if (bestMetalScore < 0.0f) {
            metalE += bestMetalScore;
            interactionCount++;
        }
    }

    // ---- Screened Coulomb (ε=4r, grid-based) ----
    // Potential grid precomputed from protein charges: Φ(r) = Σ 332*q_p/(4*r²)
    // E_elec = Σ q_ligand * Φ(r_ligand) via trilinear interpolation
    float coulombE = 0.0f;
    for (uint a = 0; a < nAtoms; a++) {
        float3 r = positions[a];
        float phi = trilinearInterpolate(elecGrid, r, gridParams.origin,
                                          gridParams.spacing, gridParams.dims);
        if (phi < 99.0f)  // skip out-of-grid penalty values
            coulombE += ligandAtoms[a].charge * phi;
    }
    coulombTermE = params.wCoulomb * coulombE;

    // ---- CH-π interactions: ligand aliphatic C → protein aromatic rings ----
    // Weaker individually (~-1 kcal/mol) but very frequent (2-5× per complex).
    // Score C...ring centroid distance (3.5-5.0 Å) without explicit H.
    // Tightened geometry: require approach within ~55° of ring normal (cosA > 0.55)
    // and use cosA² to strongly favor perpendicular approach.
    for (uint a = 0; a < nAtoms; a++) {
        if (ligandAtoms[a].vinaType != VINA_C_H) continue;
        if (ligandAtoms[a].flags & LIGATOM_FLAG_AROMATIC) continue;
        float3 cPos = positions[a];
        for (uint pr = 0; pr < params.numProteinRings; pr++) {
            if (distance_squared(ligCentroid, proteinRings[pr].centroid) > ligandCutoffSq) continue;
            float d = distance(cPos, proteinRings[pr].centroid);
            if (d < 3.5f || d > 5.0f) continue;
            float3 toC = normalize(cPos - proteinRings[pr].centroid);
            float cosA = abs(dot(toC, proteinRings[pr].normal));
            if (cosA > 0.55f) {  // within ~55° of ring normal
                float dd = d - 4.0f;
                chPiE += params.wCHPi * cosA * cosA * drusinaRamp(dd, 0.5f, 1.0f);
                interactionCount++;
            }
        }
    }

    // ---- Torsion strain: amide planarity penalty ----
    // Amide bonds should be planar (φ ≈ 0 or π). Penalize deviation.
    float strainE = 0.0f;
    for (uint t = 0; t < params.numTorsionStrains; t++) {
        TorsionStrainInfo ts = torsionStrain[t];
        if (ts.atom0 < 0 || uint(ts.atom3) >= nAtoms) continue;
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
    // Bonus-only: reward well-directed H-bonds that Vina already scores by distance.
    // Only fires on close-contact pairs (d < 3.0Å = actual H-bond distance) to avoid
    // scoring non-H-bond contacts. Pure bonus (no penalty for bad angles) to avoid
    // destabilizing pose ranking.
    // Reference: Korb 2009 (PLANTS), Verdonk 2003 (GoldScore)
    for (uint lh = 0; lh < params.numLigandHBondAtoms; lh++) {
        int li = ligandHBondInfo[lh].atomIndex;
        int lAnt = ligandHBondInfo[lh].antecedentIndex;
        bool ligIsDonor = ligandHBondInfo[lh].isDonor != 0;
        if (li < 0 || lAnt < 0 || uint(li) >= nAtoms || uint(lAnt) >= nAtoms) continue;

        float3 ligPos = positions[li];
        float3 ligAntPos = positions[lAnt];

        for (uint ph = 0; ph < params.numProteinHBondAtoms; ph++) {
            // Spatial gating: skip protein atoms far from ligand centroid
            if (distance_squared(ligCentroid, proteinHBondInfo[ph].position) > ligandCutoffSq) continue;

            bool protIsDonor = proteinHBondInfo[ph].isDonor > 0.5f;
            if (ligIsDonor == protIsDonor) continue;

            float3 protPos = proteinHBondInfo[ph].position;
            float d = distance(ligPos, protPos);
            // Tight cutoff: only actual H-bonds (< 3.2Å heavy-atom distance)
            if (d < 1.5f || d > 3.2f) continue;

            float3 protAntPos = proteinHBondInfo[ph].antecedent;

            // Distance factor: strongest at 2.7Å (optimal N-O), fading at 3.2Å
            float distFactor = drusinaRamp(d - 2.7f, 0.2f, 0.5f);

            // Donor angle: X-D...A, optimal 180°
            float3 donorPos, donorAntPos, acceptorPos;
            if (ligIsDonor) {
                donorPos = ligPos; donorAntPos = ligAntPos; acceptorPos = protPos;
            } else {
                donorPos = protPos; donorAntPos = protAntPos; acceptorPos = ligPos;
            }
            float3 dToAnt = normalize(donorAntPos - donorPos);
            float3 dToA   = normalize(acceptorPos - donorPos);
            float cosDonor = dot(dToAnt, dToA);
            // Score: 0 when bent (cos > -0.5), 1 when linear (cos = -1)
            float donorAngleScore = smoothstep(-0.5f, -1.0f, cosDonor);

            // Acceptor angle: D...A-Y, optimal ~120° (sp2)
            float3 accPos, accAntPos;
            if (ligIsDonor) {
                accPos = protPos; accAntPos = protAntPos;
            } else {
                accPos = ligPos; accAntPos = ligAntPos;
            }
            float3 aToD   = normalize(donorPos - accPos);
            float3 aToAnt = normalize(accAntPos - accPos);
            float cosAcc = dot(aToD, aToAnt);
            float accAngleScore = smoothstep(0.34f, -0.17f, cosAcc);

            // Pure bonus: only reward good angles (no penalty)
            float angleFactor = donorAngleScore * accAngleScore;
            if (angleFactor > 0.25f) {
                hbondDirE += params.wHBondDir * angleFactor * distFactor;
                interactionCount++;
            }
        }
    }

    // ---- Polar desolvation penalty ----
    // Penalize DEEPLY BURIED ligand polar atoms that lack H-bond satisfaction.
    // Only triggers when a polar is surrounded by many protein atoms (truly buried)
    // but has no H-bond partner. Surface-facing polars are not penalized.
    // Reference: HYDE scoring (Schneider 2013)
    for (uint a = 0; a < nAtoms; a++) {
        int lt = ligandAtoms[a].vinaType;
        bool isPolar = (lt == VINA_N_A || lt == VINA_N_D || lt == VINA_N_DA ||
                        lt == VINA_O_A || lt == VINA_O_D || lt == VINA_O_DA);
        if (!isPolar) continue;

        float3 lPos = positions[a];

        // Count protein neighbors for burial estimate and check H-bond satisfaction
        int nearbyCount = 0;
        bool hbondSatisfied = false;
        for (uint p = 0; p < numProteinAtoms; p++) {
            float d = distance(lPos, proteinAtoms[p].position);
            if (d < 5.0f) nearbyCount++;
            // H-bond partner check: tight distance for actual H-bonds
            if (d < 3.2f) {
                int pt = proteinAtoms[p].vinaType;
                if (xsHBondPossible(lt, pt)) {
                    hbondSatisfied = true;
                }
            }
        }

        // Only penalize deeply buried unsatisfied polars (>= 12 neighbors within 5Å)
        if (!hbondSatisfied && nearbyCount >= 12) {
            // Burial ramp: 0 at 12 neighbors, 1.0 at 25+ neighbors
            float burial = clamp(float(nearbyCount - 12) / 13.0f, 0.0f, 1.0f);
            desolvPolarE += params.wDesolvPolar * burial;
        }
    }

    // ---- Hydrophobic desolvation penalty ----
    // Penalize ligand hydrophobic atoms that are fully exposed to solvent.
    // Only triggers when a hydrophobic C has very few protein contacts (< 3 within 4.5Å),
    // indicating the atom is dangling into solvent rather than buried in the pocket.
    // Reference: Böhm 1994, ChemScore (Eldridge 1997)
    for (uint a = 0; a < nAtoms; a++) {
        if (ligandAtoms[a].vinaType != VINA_C_H) continue;
        float3 lPos = positions[a];

        int nearbyCount = 0;
        for (uint p = 0; p < numProteinAtoms; p++) {
            float d = distance(lPos, proteinAtoms[p].position);
            if (d < 4.5f) {
                nearbyCount++;
                if (nearbyCount >= 3) break;  // no penalty, early exit
            }
        }

        // Only penalize truly exposed atoms (< 3 protein neighbors)
        if (nearbyCount < 3) {
            // 0 neighbors → full penalty, 2 neighbors → 1/3 penalty
            float exposure = 1.0f - float(nearbyCount) / 3.0f;
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

    // Safety cap: Drusina corrections should remain bounded, but the cap itself
    // is a tunable parameter because Drusina is intended to be more than a tiny
    // post-hoc Vina correction.
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
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;
    float totalIntermolecular = 0.0f;
    float penalty = 0.0f;

    for (uint a = 0; a < nA; a++) {
        float3 r = positions[a];
        float oopP = outOfGridPenalty(r, gridMin, gridMax);
        if (oopP > 0.0f) { penalty += oopP; continue; }
        totalIntermolecular += sampleTypedAffinityMap(
            affinityMaps, typeIndexLookup, ligandAtoms[a].vinaType, r, gridParams);
    }

    float intraDelta = intramolecularLigandEnergy(positions, ligandAtoms, nA, intraPairs, gaParams.numIntraPairs)
        - gaParams.referenceIntraEnergy;
    float nRotF = float(nTorsions);
    float normFactor = vinaRotNormalization(nRotF);
    float normalizedE = (totalIntermolecular + intraDelta) * normFactor;

    // --- Drusina corrections (soft gate: only attenuates for clashing poses) ---
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

    // Soft gate: full contribution for vinaE < 0, attenuated for clashing poses
    float vinaQuality = drusinaVinaQuality(normalizedE, drusinaParams);
    float drusinaE = decomp.total * vinaQuality;

    pose.drusinaCorrection = drusinaE;
    pose.energy = normalizedE + wPenalty * penalty + drusinaE;
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

    // Soft gate: only attenuate for clashing poses (vinaE > +2)
    float vinaE = pose.energy;
    float vinaQuality = drusinaVinaQuality(vinaE, drusinaParams);
    float drusinaE = decomp.total * vinaQuality;

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
