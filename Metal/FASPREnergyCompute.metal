// ============================================================================
// FASPREnergyCompute.metal — GPU kernels for FASPR sidechain packing energy
//
// Two kernels for GPU-accelerated VDW energy evaluation:
//   faspr_self_energy: rotamer sidechain vs backbone atoms (one thread per rotamer)
//   faspr_pair_energy: rotamer vs rotamer atoms (one thread per pair)
//
// VDW model: CHARMM19 LJ 6-12 with linear repulsion cap at 10 kcal/mol.
// Same formula as FASPR SelfEnergy::VDWEnergyAtomAndAtom().
//
// Units: Angstrom, kcal/mol throughout.
// ============================================================================

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// ============================================================================
// MARK: - VDW Energy Helper
// ============================================================================

/// Compute VDW energy from normalized distance dstar = dist / rij.
/// LJ 6-12 with linear repulsion cap.
inline float faspr_vdw_energy(float dstar, float eij,
                               float vdwRepCut, float dstarMinCut, float dstarMaxCut) {
    if (dstar > dstarMaxCut) return 0.0f;
    if (dstar > 1.0f) {
        float inv = 1.0f / dstar;
        float inv2 = inv * inv;
        float inv6 = inv2 * inv2 * inv2;
        float inv12 = inv6 * inv6;
        return 4.0f * eij * (inv12 - inv6);
    }
    if (dstar > dstarMinCut) {
        return vdwRepCut * (dstar - 1.0f) / (dstarMinCut - 1.0f);
    }
    return vdwRepCut;
}

// ============================================================================
// MARK: - Self-Energy Kernel
// ============================================================================

/// Compute VDW energy between one rotamer's sidechain atoms and all backbone atoms.
/// One thread per rotamer. Inner loop over backbone atoms.
///
/// Buffer layout:
///   rotamerAtoms: flattened array of sidechain atoms for all rotamers.
///     Rotamer i has atoms at indices [offsets[i] .. offsets[i+1]-1].
///     offsets has rotamerCount+1 entries (last = total atom count).
///   backboneAtoms: all backbone/environment atoms in the contact shell.
///
/// Output: energies[tid] = total VDW energy for rotamer tid.
kernel void faspr_self_energy(
    device const FASPRGPUAtom  *rotamerAtoms   [[buffer(0)]],
    device const FASPRGPUAtom  *backboneAtoms  [[buffer(1)]],
    device const uint          *offsets        [[buffer(2)]],  // rotamerCount+1 entries
    constant FASPRSelfParams   &params         [[buffer(3)]],
    device float               *energies       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.rotamerCount) return;

    float totalE = 0.0f;

    uint scStart = offsets[tid];
    uint scEnd   = offsets[tid + 1];

    for (uint s = scStart; s < scEnd; s++) {
        float3 posS = rotamerAtoms[s].position;
        float  radS = rotamerAtoms[s].radius;
        float  depS = rotamerAtoms[s].depth;

        for (uint b = 0; b < params.backboneAtomCount; b++) {
            float3 posB = backboneAtoms[b].position;
            float  radB = backboneAtoms[b].radius;
            float  depB = backboneAtoms[b].depth;

            float3 diff = posS - posB;
            float dist = length(diff);
            if (dist < 1e-6f) continue;

            float rij = radS + radB;
            float eij = sqrt(depS * depB);
            float dstar = dist / rij;

            totalE += faspr_vdw_energy(dstar, eij,
                                        params.vdwRepCut, params.dstarMinCut, params.dstarMaxCut);
        }
    }

    energies[tid] = totalE;
}

// ============================================================================
// MARK: - Pair-Energy Kernel
// ============================================================================

/// Compute VDW energy between two rotamers' sidechain atoms.
/// One thread per (rot1, rot2) pair. Grid size = rot1Count * rot2Count.
///
/// Buffer layout same as self-energy but with two sets of rotamer atoms.
///
/// Output: energies[rot1 * rot2Count + rot2] = pairwise VDW energy.
kernel void faspr_pair_energy(
    device const FASPRGPUAtom  *site1Atoms     [[buffer(0)]],
    device const FASPRGPUAtom  *site2Atoms     [[buffer(1)]],
    device const uint          *offsets1       [[buffer(2)]],
    device const uint          *offsets2       [[buffer(3)]],
    constant FASPRPairParams   &params         [[buffer(4)]],
    device float               *energies       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint totalPairs = params.rot1Count * params.rot2Count;
    if (tid >= totalPairs) return;

    uint r1 = tid / params.rot2Count;
    uint r2 = tid % params.rot2Count;

    float totalE = 0.0f;

    uint s1Start = offsets1[r1];
    uint s1End   = offsets1[r1 + 1];
    uint s2Start = offsets2[r2];
    uint s2End   = offsets2[r2 + 1];

    for (uint i = s1Start; i < s1End; i++) {
        float3 posI = site1Atoms[i].position;
        float  radI = site1Atoms[i].radius;
        float  depI = site1Atoms[i].depth;

        for (uint j = s2Start; j < s2End; j++) {
            float3 posJ = site2Atoms[j].position;
            float  radJ = site2Atoms[j].radius;
            float  depJ = site2Atoms[j].depth;

            float3 diff = posI - posJ;
            float dist = length(diff);
            if (dist < 1e-6f) continue;

            float rij = radI + radJ;
            float eij = sqrt(depI * depJ);
            float dstar = dist / rij;

            totalE += faspr_vdw_energy(dstar, eij,
                                        params.vdwRepCut, params.dstarMinCut, params.dstarMaxCut);
        }
    }

    energies[tid] = totalE;
}
