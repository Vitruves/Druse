#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Flexible Sidechain Docking (Receptor Flexibility)
//
// This shader provides GPU-accelerated induced-fit docking by simultaneously
// optimizing ligand pose AND selected protein sidechain chi angles.
//
// Design:
//   - Flexible sidechain atoms are EXCLUDED from the precomputed grid maps.
//   - Their interaction with the ligand is scored explicitly (pairwise Vina).
//   - Chi angles are part of the DockPose chromosome and mutated/crossed by the GA.
//   - Local search includes chi angle gradient descent.
//
// Buffer layout for flex-aware kernels:
//   0: poses (DockPose[])
//   1: ligandAtoms (DockLigandAtom[])
//   2: flexAtoms (FlexSidechainAtom[])
//   3: flexTorsionEdges (FlexTorsionEdge[])
//   4: flexMovingIndices (int32_t[])
//   5: flexParams (FlexParams)
//   6: gaParams (GAParams)
// ============================================================================

// ============================================================================
// MARK: - Ligand Transform Helpers (duplicated from DockingCompute for cross-file visibility)
// ============================================================================

inline float3 flexQuatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

inline void flexApplyTorsions(
    thread float3 *positions, uint nAtoms,
    device const DockPose &pose,
    constant TorsionEdge *torsionEdges,
    constant int32_t *movingIndices,
    uint numTorsions)
{
    for (uint t = 0; t < numTorsions; t++) {
        float angle = pose.torsions[t];
        if (abs(angle) < 1e-6f) continue;
        TorsionEdge edge = torsionEdges[t];
        float3 pivot = positions[edge.atom1];
        float3 axis = normalize(positions[edge.atom2] - pivot);
        float cosA = cos(angle);
        float sinA = sin(angle);
        for (int i = 0; i < edge.movingCount; i++) {
            int atomIdx = movingIndices[edge.movingStart + i];
            if (atomIdx < 0 || uint(atomIdx) >= nAtoms) continue;
            float3 v = positions[atomIdx] - pivot;
            float3 rotated = v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
            positions[atomIdx] = pivot + rotated;
        }
    }
}

inline void flexTransformAtoms(
    thread float3 *outPositions,
    constant DockLigandAtom *ligandAtoms, uint nAtoms,
    device const DockPose &pose,
    constant TorsionEdge *torsionEdges,
    constant int32_t *movingIndices,
    uint numTorsions)
{
    for (uint a = 0; a < nAtoms; a++) {
        outPositions[a] = flexQuatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    flexApplyTorsions(outPositions, nAtoms, pose, torsionEdges, movingIndices, numTorsions);
}

// Vina scoring weights (must match DockingCompute.metal)
constant float fwGauss1      = -0.035579f;
constant float fwGauss2      = -0.005156f;
constant float fwRepulsion   = 0.840245f;
constant float fwHydrophobic = -0.035069f;
constant float fwHBond       = -0.587439f;

// ============================================================================
// MARK: - Vina XS Atom Type Helpers (duplicated from DockingCompute for linkage)
// ============================================================================

constant float kXSRadii[17] = {
    1.9f, 1.8f, 1.7f, 1.7f, 2.0f,
    2.0f, 1.7f, 1.8f, 1.5f, 1.5f,
    1.5f, 0.0f, 1.7f, 1.7f, 1.7f,
    1.7f, 1.0f
};

inline bool flexXSTypeSupported(int t) { return t >= 0 && t < 17; }

inline float flexOptimalDist(int t1, int t2) {
    if (!flexXSTypeSupported(t1) || !flexXSTypeSupported(t2)) return 8.0f;
    return kXSRadii[t1] + kXSRadii[t2];
}

inline bool flexIsHydrophobic(int t) { return t == 0 || t == 4 || t == 5 || t == 12 || t == 13 || t == 14; }

inline bool flexIsHAcceptor(int t)  { return t == 6 || t == 7 || t == 8 || t == 9 || t == 10 || t == 11; }
inline bool flexIsHDonor(int t)     { return t == 3 || t == 7 || t == 10 || t == 16; }
inline bool flexHBondPossible(int a, int b) {
    return (flexIsHAcceptor(a) && flexIsHDonor(b)) || (flexIsHDonor(a) && flexIsHAcceptor(b));
}

inline float flexSlopeStep(float hi, float lo, float x) {
    if (x <= lo) return 1.0f;
    if (x >= hi) return 0.0f;
    return (hi - x) / (hi - lo);
}

/// Compute Vina pairwise energy between a flex sidechain atom and a ligand atom.
inline float flexVinaPairEnergy(int flexType, int ligType, float r) {
    if (!flexXSTypeSupported(flexType) || !flexXSTypeSupported(ligType) || r >= 8.0f)
        return 0.0f;

    float d = r - flexOptimalDist(flexType, ligType);

    float dOverHalf = d * 2.0f;
    float e = fwGauss1 * exp(-(dOverHalf * dOverHalf));

    float dMinus3Over2 = (d - 3.0f) * 0.5f;
    e += fwGauss2 * exp(-(dMinus3Over2 * dMinus3Over2));

    if (d < 0.0f) e += fwRepulsion * d * d;
    if (flexIsHydrophobic(flexType) && flexIsHydrophobic(ligType))
        e += fwHydrophobic * flexSlopeStep(1.5f, 0.5f, d);
    if (flexHBondPossible(flexType, ligType))
        e += fwHBond * flexSlopeStep(0.0f, -0.7f, d);

    return e;
}

// ============================================================================
// MARK: - Position Flex Sidechain Atoms
// ============================================================================

/// Apply chi angle rotations to flexible sidechain atoms, producing positioned coordinates.
/// Similar to `applyTorsions` for ligand atoms but for protein sidechains.
inline void positionFlexAtoms(
    thread float3               *outPositions,
    constant FlexSidechainAtom  *flexAtoms,
    uint                         numFlexAtoms,
    constant FlexTorsionEdge    *flexEdges,
    constant int32_t            *flexMovingIndices,
    device const DockPose       &pose,
    uint                         numFlexTorsions)
{
    // Start from reference (rigid) positions
    for (uint i = 0; i < numFlexAtoms; i++) {
        outPositions[i] = float3(flexAtoms[i].referencePosition);
    }

    // Apply each chi angle rotation (root → leaf order)
    for (uint t = 0; t < numFlexTorsions; t++) {
        int chiSlot = flexEdges[t].chiSlot;
        if (chiSlot < 0 || chiSlot >= 24) continue;

        float angle = pose.chiAngles[chiSlot];
        if (abs(angle) < 1e-6f) continue;  // no rotation needed

        float3 pivot = outPositions[flexEdges[t].pivotAtom];
        float3 axisEnd = outPositions[flexEdges[t].axisAtom];
        float3 axis = normalize(axisEnd - pivot);

        float cosA = cos(angle);
        float sinA = sin(angle);

        uint start = flexEdges[t].movingStart;
        uint count = flexEdges[t].movingCount;
        for (uint m = start; m < start + count; m++) {
            int atomIdx = flexMovingIndices[m];
            if (atomIdx < 0 || uint(atomIdx) >= numFlexAtoms) continue;

            float3 rel = outPositions[atomIdx] - pivot;
            // Rodrigues rotation
            float3 rotated = rel * cosA + cross(axis, rel) * sinA + axis * dot(axis, rel) * (1.0f - cosA);
            outPositions[atomIdx] = pivot + rotated;
        }
    }
}

// ============================================================================
// MARK: - Score Flex-Ligand Interactions
// ============================================================================

/// Compute the delta Vina energy from rotating flex sidechains:
///   delta = pairwise(ligand, flex_rotated) - pairwise(ligand, flex_reference)
/// When chi = 0 (no rotation), positions match reference and delta = 0.
/// The grid already includes flex atoms at reference positions, so this delta
/// captures exactly the energy change from sidechain rotation.
inline float scoreFlexLigandDelta(
    thread float3               *ligPositions,
    constant DockLigandAtom     *ligandAtoms,
    uint                         numLigAtoms,
    thread float3               *flexPositions,
    constant FlexSidechainAtom  *flexAtoms,
    uint                         numFlexAtoms)
{
    float rotatedE = 0.0f;
    float referenceE = 0.0f;
    for (uint fa = 0; fa < numFlexAtoms; fa++) {
        int flexType = flexAtoms[fa].vinaType;
        if (flexType < 0) continue; // skip backbone pivots
        float3 rotatedPos = flexPositions[fa];
        float3 refPos = float3(flexAtoms[fa].referencePosition);
        for (uint la = 0; la < numLigAtoms; la++) {
            int ligType = ligandAtoms[la].vinaType;
            rotatedE += flexVinaPairEnergy(flexType, ligType, distance(rotatedPos, ligPositions[la]));
            referenceE += flexVinaPairEnergy(flexType, ligType, distance(refPos, ligPositions[la]));
        }
    }
    return rotatedE - referenceE;
}

/// Same as above but also accumulates forces on ligand atoms (for gradient-based local search).
inline float scoreFlexLigandWithGrad(
    thread float3               *ligPositions,
    thread float3               *ligForces,
    constant DockLigandAtom     *ligandAtoms,
    uint                         numLigAtoms,
    thread float3               *flexPositions,
    constant FlexSidechainAtom  *flexAtoms,
    uint                         numFlexAtoms)
{
    float totalE = 0.0f;
    float h = 0.01f;  // finite difference step for flex gradient

    for (uint fa = 0; fa < numFlexAtoms; fa++) {
        int flexType = flexAtoms[fa].vinaType;
        float3 fp = flexPositions[fa];
        for (uint la = 0; la < numLigAtoms; la++) {
            float3 lp = ligPositions[la];
            float r = distance(fp, lp);
            float e = flexVinaPairEnergy(flexType, ligandAtoms[la].vinaType, r);
            totalE += e;

            // Gradient via finite difference on distance
            if (r > 0.01f && r < 8.0f) {
                float ePlus = flexVinaPairEnergy(flexType, ligandAtoms[la].vinaType, r + h);
                float dEdR = (ePlus - e) / h;
                float3 dir = (lp - fp) / r;
                ligForces[la] -= dir * dEdR;  // force on ligand atom (toward lower energy)
            }
        }
    }
    return totalE;
}

// ============================================================================
// MARK: - Flex-Aware Scoring Kernel
// ============================================================================

/// Adds flexible sidechain contribution to poses that have already been scored
/// against the grid. This kernel runs AFTER the main scorePoses kernel.
/// It positions the flex sidechains using chi angles, scores them against the
/// ligand, and adds the energy to the total.
kernel void scoreFlexSidechains(
    device DockPose             *poses          [[buffer(0)]],
    constant DockLigandAtom     *ligandAtoms    [[buffer(1)]],
    constant FlexSidechainAtom  *flexAtoms      [[buffer(2)]],
    constant FlexTorsionEdge    *flexEdges      [[buffer(3)]],
    constant int32_t            *flexMoving     [[buffer(4)]],
    constant FlexParams         &flexParams     [[buffer(5)]],
    constant GAParams           &gaParams       [[buffer(6)]],
    constant TorsionEdge        *torsionEdges   [[buffer(7)]],
    constant int32_t            *movingIndices  [[buffer(8)]],
    uint                         tid            [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;
    if (flexParams.numFlexAtoms == 0 || flexParams.numFlexResidues == 0) return;

    device DockPose &pose = poses[tid];
    uint nLig = min(gaParams.numLigandAtoms, 128u);
    uint nFlex = min(flexParams.numFlexAtoms, 64u);
    uint nFlexTor = min(flexParams.numFlexTorsions, 24u);
    uint nTorsions = min(gaParams.numTorsions, 32u);

    // Position ligand atoms
    float3 ligPositions[128];
    flexTransformAtoms(ligPositions, ligandAtoms, nLig, pose, torsionEdges, movingIndices, nTorsions);

    // Position flex sidechain atoms
    float3 flexPositions[64];
    positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);

    // Score flex-ligand interactions
    float flexE = scoreFlexLigandDelta(
        ligPositions, ligandAtoms, nLig,
        flexPositions, flexAtoms, nFlex
    );

    // Add to total energy (weighted)
    pose.energy += flexParams.flexWeight * flexE;
}

// ============================================================================
// MARK: - GA Evolution for Chi Angles
// ============================================================================

/// Mutate and crossover chi angles alongside ligand torsions.
/// This kernel runs AFTER the main gaEvolve/mcPerturb kernel and modifies
/// only the chi angle portion of the offspring poses.
kernel void evolveChiAngles(
    device DockPose         *offspring   [[buffer(0)]],
    device const DockPose   *population  [[buffer(1)]],
    constant GAParams       &gaParams    [[buffer(2)]],
    constant FlexParams     &flexParams  [[buffer(3)]],
    uint                     tid         [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;
    if (flexParams.numFlexResidues == 0) return;

    uint nChi = min(uint(flexParams.numFlexTorsions), 24u);

    device DockPose &child = offspring[tid];

    // Use a simple hash-based RNG seeded from pose state + generation
    uint seed = tid * 2654435761u + gaParams.generation * 1664525u + 1013904223u;

    // Per-chi-angle mutation (same rate as torsion angles)
    float chiStep = flexParams.chiStep > 0.0f ? flexParams.chiStep : 0.6f;
    for (uint c = 0; c < nChi; c++) {
        seed = seed * 1664525u + 1013904223u;
        float r = float(seed & 0xFFFF) / 65536.0f;
        if (r < gaParams.mutationRate) {
            seed = seed * 1664525u + 1013904223u;
            float delta = (float(seed & 0xFFFF) / 65536.0f * 2.0f - 1.0f) * chiStep;
            child.chiAngles[c] += delta;
            // Wrap to [-π, π]
            if (child.chiAngles[c] > M_PI_F) child.chiAngles[c] -= 2.0f * M_PI_F;
            if (child.chiAngles[c] < -M_PI_F) child.chiAngles[c] += 2.0f * M_PI_F;
        }
    }

    child.numChiAngles = int(nChi);
}

// ============================================================================
// MARK: - Flex-Aware Local Search
// ============================================================================

/// Local search with chi angle gradients. Runs AFTER the main local search
/// to refine sidechain orientations.
/// Uses numerical finite differences on chi angles.
kernel void localSearchFlex(
    device DockPose             *poses          [[buffer(0)]],
    constant DockLigandAtom     *ligandAtoms    [[buffer(1)]],
    constant FlexSidechainAtom  *flexAtoms      [[buffer(2)]],
    constant FlexTorsionEdge    *flexEdges      [[buffer(3)]],
    constant int32_t            *flexMoving     [[buffer(4)]],
    constant FlexParams         &flexParams     [[buffer(5)]],
    constant GAParams           &gaParams       [[buffer(6)]],
    constant TorsionEdge        *torsionEdges   [[buffer(7)]],
    constant int32_t            *movingIndices  [[buffer(8)]],
    device const half           *affinityMaps   [[buffer(9)]],
    device const int32_t        *typeIndexLookup [[buffer(10)]],
    constant GridParams         &gridParams     [[buffer(11)]],
    constant uint32_t           *exclusionMask  [[buffer(12)]],
    uint                         tid            [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;
    if (flexParams.numFlexResidues == 0) return;

    device DockPose &pose = poses[tid];
    uint nLig = min(gaParams.numLigandAtoms, 128u);
    uint nFlex = min(flexParams.numFlexAtoms, 64u);
    uint nFlexTor = min(flexParams.numFlexTorsions, 24u);
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 ligPositions[128];
    float3 flexPositions[64];

    float hChi = 0.03f;   // finite difference step (radians)
    float stepSize = 0.3f;  // initial step
    uint maxSteps = min(gaParams.localSearchSteps, 15u);  // fewer steps for chi refinement

    for (uint step = 0; step < maxSteps; step++) {
        // Current energy
        flexTransformAtoms(ligPositions, ligandAtoms, nLig, pose, torsionEdges, movingIndices, nTorsions);
        positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
        float baseE = scoreFlexLigandDelta(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);

        // Compute gradient for each chi angle
        float gradChi[24];
        bool anyGrad = false;
        for (uint c = 0; c < nFlexTor; c++) {
            int slot = flexEdges[c].chiSlot;
            if (slot < 0 || slot >= 24) { gradChi[c] = 0; continue; }

            float origChi = pose.chiAngles[slot];

            pose.chiAngles[slot] = origChi + hChi;
            positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
            float ePlus = scoreFlexLigandDelta(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);

            pose.chiAngles[slot] = origChi - hChi;
            positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
            float eMinus = scoreFlexLigandDelta(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);

            pose.chiAngles[slot] = origChi;
            gradChi[c] = (ePlus - eMinus) / (2.0f * hChi);
            if (abs(gradChi[c]) > 1e-6f) anyGrad = true;
        }

        if (!anyGrad) break;

        // Line search: step along negative gradient
        for (uint c = 0; c < nFlexTor; c++) {
            int slot = flexEdges[c].chiSlot;
            if (slot < 0 || slot >= 24) continue;
            pose.chiAngles[slot] -= stepSize * gradChi[c];
            // Wrap to [-π, π]
            if (pose.chiAngles[slot] > M_PI_F) pose.chiAngles[slot] -= 2.0f * M_PI_F;
            if (pose.chiAngles[slot] < -M_PI_F) pose.chiAngles[slot] += 2.0f * M_PI_F;
        }

        // Check improvement
        positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
        float newE = scoreFlexLigandDelta(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);

        if (newE < baseE) {
            stepSize *= 1.2f;
        } else {
            // Revert
            for (uint c = 0; c < nFlexTor; c++) {
                int slot = flexEdges[c].chiSlot;
                if (slot < 0 || slot >= 24) continue;
                pose.chiAngles[slot] += stepSize * gradChi[c];
                if (pose.chiAngles[slot] > M_PI_F) pose.chiAngles[slot] -= 2.0f * M_PI_F;
                if (pose.chiAngles[slot] < -M_PI_F) pose.chiAngles[slot] += 2.0f * M_PI_F;
            }
            stepSize *= 0.5f;
            if (stepSize < 0.001f) break;
        }
    }

    // Final energy update: re-score with optimized chi angles and add to pose
    flexTransformAtoms(ligPositions, ligandAtoms, nLig, pose, torsionEdges, movingIndices, nTorsions);
    positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
    float finalFlexE = scoreFlexLigandDelta(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);
    pose.energy += flexParams.flexWeight * finalFlexE;
}
