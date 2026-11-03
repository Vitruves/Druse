// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "DockingCommon.h"

// ============================================================================
// MARK: - Flexible Sidechain Docking (Receptor Flexibility)
//
// This shader provides GPU-accelerated induced-fit docking by simultaneously
// optimizing ligand pose AND selected protein sidechain chi angles.
//
// Design:
//   - The affinity grid maps contain the flexible sidechain atoms at their reference
//     (chi = 0) positions, so flex scoring adds the pairwise Vina delta
//     E(ligand, rotated flex) - E(ligand, reference flex).
//   - Pairwise terms use the same XS typing and Vina functional form as the grid maps
//     (DockingCommon.h), and are inside the Vina N_rot normalization like every other
//     intermolecular term.
//   - Chi angles are part of the DockPose chromosome and mutated by the GA/MC.
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

inline float flexWrapAngle(float v) {
    return v - 2.0f * M_PI_F * floor((v + M_PI_F) / (2.0f * M_PI_F));
}

/// Pairwise Vina energy between the ligand and the flex atoms at `flexPositions`.
inline float flexLigandPairEnergy(
    thread float3               *ligPositions,
    constant DockLigandAtom     *ligandAtoms,
    uint                         numLigAtoms,
    thread float3               *flexPositions,
    constant FlexSidechainAtom  *flexAtoms,
    uint                         numFlexAtoms)
{
    float total = 0.0f;
    for (uint fa = 0; fa < numFlexAtoms; fa++) {
        int flexType = flexAtoms[fa].vinaType;
        if (flexType < 0) continue; // backbone pivots are not scored
        float3 fp = flexPositions[fa];
        for (uint la = 0; la < numLigAtoms; la++) {
            total += vinaPairEnergy(flexType, ligandAtoms[la].vinaType, distance(fp, ligPositions[la]));
        }
    }
    return total;
}

/// Gradient dE/dpos of the flex-ligand pair energy on each flex atom (piecewise Vina
/// derivative, consistent with flexLigandPairEnergy used for accept/reject).
inline void flexLigandGradient(
    thread float3               *ligPositions,
    constant DockLigandAtom     *ligandAtoms,
    uint                         numLigAtoms,
    thread float3               *flexPositions,
    thread float3               *flexGrad,
    constant FlexSidechainAtom  *flexAtoms,
    uint                         numFlexAtoms)
{
    for (uint fa = 0; fa < numFlexAtoms; fa++) {
        flexGrad[fa] = float3(0);
        int flexType = flexAtoms[fa].vinaType;
        if (flexType < 0) continue;
        float3 fp = flexPositions[fa];

        for (uint la = 0; la < numLigAtoms; la++) {
            float3 diff = fp - ligPositions[la];
            float r = length(diff);
            if (r < 1e-6f || r >= 8.0f) continue;

            float dEdr;
            vinaPairEnergyWithDeriv(flexType, ligandAtoms[la].vinaType, r, dEdr);
            flexGrad[fa] += dEdr * diff / r;
        }
    }
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
        constant FlexTorsionEdge &edge = flexEdges[t];
        int chiSlot = edge.chiSlot;
        if (chiSlot < 0 || chiSlot >= 24) continue;
        if (edge.pivotAtom < 0 || uint(edge.pivotAtom) >= numFlexAtoms ||
            edge.axisAtom < 0 || uint(edge.axisAtom) >= numFlexAtoms) continue;

        float angle = pose.chiAngles[chiSlot];
        if (abs(angle) < 1e-6f) continue;  // no rotation needed

        float3 pivot = outPositions[edge.pivotAtom];
        float3 axisVec = outPositions[edge.axisAtom] - pivot;
        float axisLen = length(axisVec);
        if (axisLen < 1e-6f) continue;
        float3 axis = axisVec / axisLen;

        float cosA = cos(angle);
        float sinA = sin(angle);

        uint start = uint(max(edge.movingStart, 0));
        uint count = uint(max(edge.movingCount, 0));
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
    float delta = 0.0f;
    for (uint fa = 0; fa < numFlexAtoms; fa++) {
        int flexType = flexAtoms[fa].vinaType;
        if (flexType < 0) continue; // skip backbone pivots
        float3 rotatedPos = flexPositions[fa];
        float3 refPos = float3(flexAtoms[fa].referencePosition);
        for (uint la = 0; la < numLigAtoms; la++) {
            int ligType = ligandAtoms[la].vinaType;
            delta += vinaPairEnergy(flexType, ligType, distance(rotatedPos, ligPositions[la]))
                   - vinaPairEnergy(flexType, ligType, distance(refPos, ligPositions[la]));
        }
    }
    return delta;
}

// ============================================================================
// MARK: - Flex-Aware Scoring Kernel
// ============================================================================

/// Adds flexible sidechain contribution to poses that have already been scored
/// against the grid. This kernel runs AFTER the main scorePoses kernel.
/// It positions the flex sidechains using chi angles, scores them against the
/// ligand, and adds the (N_rot-normalized) energy delta to the total.
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
    transformAtoms(ligPositions, ligandAtoms, nLig, pose, torsionEdges, movingIndices, nTorsions);

    // Position flex sidechain atoms
    float3 flexPositions[64];
    positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);

    // Score flex-ligand interactions
    float flexE = scoreFlexLigandDelta(
        ligPositions, ligandAtoms, nLig,
        flexPositions, flexAtoms, nFlex
    );

    // Intermolecular term: same conf-independent normalization as the grid score.
    pose.energy += flexParams.flexWeight * vinaRotNormalization(float(nTorsions)) * flexE;
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

    // Per-chi-angle mutation (same rate as torsion angles).
    // Draws use the high 16 bits of the LCG state: the low bits of a power-of-two
    // modulus LCG have very short periods (bit k repeats every 2^(k+1) steps).
    float chiStep = flexParams.chiStep > 0.0f ? flexParams.chiStep : 0.6f;
    for (uint c = 0; c < nChi; c++) {
        seed = seed * 1664525u + 1013904223u;
        float r = float(seed >> 16) / 65536.0f;
        if (r < gaParams.mutationRate) {
            seed = seed * 1664525u + 1013904223u;
            float delta = (float(seed >> 16) / 65536.0f * 2.0f - 1.0f) * chiStep;
            child.chiAngles[c] = flexWrapAngle(child.chiAngles[c] + delta);
        }
    }

    child.numChiAngles = int(nChi);
}

// ============================================================================
// MARK: - Flex-Aware Local Search
// ============================================================================

/// Local search with analytical chi angle gradients. Runs AFTER the main local search
/// and AFTER scoreFlexSidechains, which has already added the flex delta for the entry
/// chi angles to pose.energy; this kernel adds only the change produced by the search.
/// Uses torque projection (gradient · cross(axis, arm)) instead of finite differences.
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
    constant uint32_t           *intraPairs     [[buffer(12)]],
    uint                         tid            [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;
    if (flexParams.numFlexAtoms == 0 || flexParams.numFlexResidues == 0) return;

    device DockPose &pose = poses[tid];
    uint nLig = min(gaParams.numLigandAtoms, 128u);
    uint nFlex = min(flexParams.numFlexAtoms, 64u);
    uint nFlexTor = min(flexParams.numFlexTorsions, 24u);
    uint nTorsions = min(gaParams.numTorsions, 32u);

    float3 ligPositions[128];
    float3 flexPositions[64];
    float3 flexGrad[64];

    float stepSize = 0.3f;
    uint maxSteps = min(gaParams.localSearchSteps, 15u);

    // Transform ligand once (doesn't change during chi optimization)
    transformAtoms(ligPositions, ligandAtoms, nLig, pose, torsionEdges, movingIndices, nTorsions);

    // The reference-position term of the delta is constant during the search and cancels
    // in (final - initial), so only the rotated pair energy is tracked.
    positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
    float initialE = flexLigandPairEnergy(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);
    float currentE = initialE;

    for (uint step = 0; step < maxSteps; step++) {
        flexLigandGradient(ligPositions, ligandAtoms, nLig, flexPositions, flexGrad, flexAtoms, nFlex);

        // Project gradients onto chi angle axes via torque
        float gradChi[24];
        bool anyGrad = false;
        for (uint c = 0; c < nFlexTor; c++) {
            constant FlexTorsionEdge &edge = flexEdges[c];
            gradChi[c] = 0.0f;
            if (edge.chiSlot < 0 || edge.chiSlot >= 24) continue;
            if (edge.pivotAtom < 0 || uint(edge.pivotAtom) >= nFlex ||
                edge.axisAtom < 0 || uint(edge.axisAtom) >= nFlex) continue;

            float3 pivot = flexPositions[edge.pivotAtom];
            float3 axisVec = flexPositions[edge.axisAtom] - pivot;
            float axisLen = length(axisVec);
            if (axisLen < 1e-6f) continue;
            float3 axis = axisVec / axisLen;

            // dE/dchi = sum over moving atoms of: dot(cross(arm, grad), axis)
            float torque = 0.0f;
            uint start = uint(max(edge.movingStart, 0));
            uint count = uint(max(edge.movingCount, 0));
            for (uint m = start; m < start + count; m++) {
                int atomIdx = flexMoving[m];
                if (atomIdx < 0 || uint(atomIdx) >= nFlex) continue;
                float3 arm = flexPositions[atomIdx] - pivot;
                torque += dot(cross(arm, flexGrad[atomIdx]), axis);
            }
            gradChi[c] = torque;
            if (abs(torque) > 1e-6f) anyGrad = true;
        }

        if (!anyGrad) break;

        // Save chi angles for rollback, then step along negative gradient
        float oldChi[24];
        for (uint c = 0; c < nFlexTor; c++) {
            int slot = flexEdges[c].chiSlot;
            if (slot < 0 || slot >= 24) continue;
            oldChi[c] = pose.chiAngles[slot];
            pose.chiAngles[slot] = flexWrapAngle(oldChi[c] - stepSize * gradChi[c]);
        }

        // Accept only on improvement of the exact (piecewise) pair energy
        positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
        float newE = flexLigandPairEnergy(ligPositions, ligandAtoms, nLig, flexPositions, flexAtoms, nFlex);

        if (newE < currentE) {
            currentE = newE;
            stepSize *= 1.2f;
        } else {
            for (uint c = 0; c < nFlexTor; c++) {
                int slot = flexEdges[c].chiSlot;
                if (slot >= 0 && slot < 24) pose.chiAngles[slot] = oldChi[c];
            }
            stepSize *= 0.5f;
            if (stepSize < 0.001f) break;
            positionFlexAtoms(flexPositions, flexAtoms, nFlex, flexEdges, flexMoving, pose, nFlexTor);
        }
    }

    pose.energy += flexParams.flexWeight * vinaRotNormalization(float(nTorsions)) * (currentE - initialE);
}
