// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PreparationMinimizeCompute.metal — GPU-accelerated preparation minimization
//
// Computes pairwise VDW energy (LJ 6-12) and region-based positional restraints
// for structure preparation minimization:
//   - Pairwise Lennard-Jones 6-12 with Lorentz-Berthelot mixing rules
//   - Harmonic positional restraints per region (backbone, sidechain, reconstructed, H)
//
// One thread per atom. Per-atom partials accumulated without atomics, summed on CPU.
//
// Units: Angstrom, kcal/mol throughout (no unit conversion at GPU boundary).
// ============================================================================

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// ============================================================================
// MARK: - VDW Energy + Positional Restraints
// ============================================================================

/// Compute pairwise LJ 6-12 energy and harmonic positional restraint per atom.
/// E_vdw = 4 * eps_ij * [(sig_ij/r)^12 - (sig_ij/r)^6]
/// E_restraint = 0.5 * K[region] * |pos - refPos|^2
kernel void prep_vdw_restraint(
    device const PrepMinAtom    *atoms     [[buffer(0)]],
    device const PrepMinAtom    *refAtoms  [[buffer(1)]],
    constant PrepMinParams      &params    [[buffer(2)]],
    device float                *energies  [[buffer(3)]],
    device float3               *gradients [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float ei = 0.0f;
    float3 gi = float3(0.0f);
    float3 pi = atoms[tid].position;
    float sigma_i = atoms[tid].sigma;
    float epsilon_i = atoms[tid].epsilon;

    float cutoff2 = params.stericCutoff * params.stericCutoff;

    // --- Pairwise LJ 6-12 ---
    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;

        float3 diff = pi - atoms[j].position;
        float r2 = dot(diff, diff);
        if (r2 >= cutoff2 || r2 < 1e-6f) continue;

        // Lorentz-Berthelot mixing
        float sigma_ij = 0.5f * (sigma_i + atoms[j].sigma);
        float epsilon_ij = sqrt(epsilon_i * atoms[j].epsilon);
        if (epsilon_ij < 1e-12f) continue;

        float inv_r2 = 1.0f / r2;
        float sigma2 = sigma_ij * sigma_ij;
        float ratio2 = sigma2 * inv_r2;         // (sigma/r)^2
        float ratio6 = ratio2 * ratio2 * ratio2; // (sigma/r)^6
        float ratio12 = ratio6 * ratio6;         // (sigma/r)^12

        // Energy: count each pair once (lower index owns it)
        if (tid < j) {
            ei += 4.0f * epsilon_ij * (ratio12 - ratio6);
        }

        // Gradient: both atoms get forces
        if (params.computeGrad) {
            // dE/dr * (1/r) * diff  =>  force direction
            // dE/dr = 4*eps * (-12*sig^12/r^13 + 6*sig^6/r^7)
            //       = 4*eps * inv_r2 * (-12*ratio12 + 6*ratio6)
            float dEdr_over_r = 4.0f * epsilon_ij * inv_r2 *
                                (-12.0f * ratio12 + 6.0f * ratio6);
            gi += dEdr_over_r * diff;
        }
    }

    // --- Positional restraint ---
    // Determine restraint constant from region
    uint region = atoms[tid].region;
    float k_restraint = 0.0f;
    if (region == 0) {
        k_restraint = params.restraintK_backbone;
    } else if (region == 1) {
        k_restraint = params.restraintK_existing;
    } else if (region == 2) {
        k_restraint = params.restraintK_reconstructed;
    }
    // region == 3 (hydrogen): k_restraint stays 0

    if (k_restraint > 0.0f) {
        float3 refPos = refAtoms[tid].position;
        float3 disp = pi - refPos;
        float dist2 = dot(disp, disp);

        ei += 0.5f * k_restraint * dist2;

        if (params.computeGrad) {
            gi += k_restraint * disp;
        }
    }

    // --- Write outputs ---
    energies[tid] = ei;
    if (params.computeGrad) {
        gradients[tid] = gi;
    }
}
