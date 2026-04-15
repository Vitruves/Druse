// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// LoopRefineCompute.metal — GPU-accelerated loop refinement kernels
//
// Offloads force field energy + gradient evaluation for missing residue loops:
//   - Harmonic bond energy + gradient
//   - Harmonic angle energy + gradient
//   - Periodic torsion energy + gradient
//   - Steric repulsion (loop vs. non-loop, LJ repulsive)
//   - Positional restraints (freeze non-loop atoms)
//
// Each kernel uses one thread per atom. Same pattern as XTBCompute.metal:
// per-atom partials accumulated without atomics, summed on CPU.
//
// Units: Angstrom, kcal/mol throughout (no unit conversion at GPU boundary).
// ============================================================================

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// ============================================================================
// MARK: - Bond Energy + Gradient
// ============================================================================

/// Compute harmonic bond energy and gradient contributions per atom.
/// E_bond = 0.5 * k * (r - r0)^2
/// Each thread processes all bonds where it is atom1 or atom2.
kernel void loop_bond_energy(
    device const LoopRefineAtom   *atoms    [[buffer(0)]],
    device const LoopRefineBond   *bonds    [[buffer(1)]],
    constant LoopRefineParams     &params   [[buffer(2)]],
    device float                  *energies [[buffer(3)]],
    device float3                 *gradients[[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float ei = 0.0f;
    float3 gi = float3(0.0f);
    float3 pi = atoms[tid].position;

    for (uint b = 0; b < params.bondCount; b++) {
        uint a1 = bonds[b].atom1;
        uint a2 = bonds[b].atom2;

        bool isA1 = (a1 == tid);
        bool isA2 = (a2 == tid);
        if (!isA1 && !isA2) continue;

        uint other = isA1 ? a2 : a1;
        float3 diff = pi - atoms[other].position;
        float r = length(diff);
        if (r < 1e-6f) continue;

        float dr = r - bonds[b].length;
        float k = bonds[b].k;

        // Energy: only count once per bond (atom with lower index)
        if (tid < other) {
            ei += 0.5f * k * dr * dr;
        }

        // Gradient: both atoms get equal and opposite force
        if (params.computeGrad) {
            float dEdr = k * dr / r;
            gi += dEdr * diff;
        }
    }

    energies[tid] = ei;
    if (params.computeGrad) {
        gradients[tid] = gi;
    }
}

// ============================================================================
// MARK: - Angle Energy + Gradient
// ============================================================================

/// Compute harmonic angle energy and gradient contributions.
/// E_angle = 0.5 * k * (theta - theta0)^2
kernel void loop_angle_energy(
    device const LoopRefineAtom    *atoms    [[buffer(0)]],
    device const LoopRefineAngle   *angles   [[buffer(1)]],
    constant LoopRefineParams      &params   [[buffer(2)]],
    device float                   *energies [[buffer(3)]],
    device float3                  *gradients[[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float ei = 0.0f;
    float3 gi = float3(0.0f);

    for (uint a = 0; a < params.angleCount; a++) {
        uint a1 = angles[a].atom1;
        uint a2 = angles[a].atom2;  // central
        uint a3 = angles[a].atom3;

        bool isA1 = (a1 == tid);
        bool isA2 = (a2 == tid);
        bool isA3 = (a3 == tid);
        if (!isA1 && !isA2 && !isA3) continue;

        float3 p1 = atoms[a1].position;
        float3 p2 = atoms[a2].position;
        float3 p3 = atoms[a3].position;

        float3 v21 = p1 - p2;
        float3 v23 = p3 - p2;
        float r21 = length(v21);
        float r23 = length(v23);
        if (r21 < 1e-6f || r23 < 1e-6f) continue;

        float cosTheta = clamp(dot(v21, v23) / (r21 * r23), -1.0f, 1.0f);
        float theta = acos(cosTheta);
        float dTheta = theta - angles[a].angle;
        float k = angles[a].k;

        // Energy: count once (central atom)
        if (isA2) {
            ei += 0.5f * k * dTheta * dTheta;
        }

        // Gradient
        if (params.computeGrad) {
            float sinTheta = sin(theta);
            if (sinTheta < 1e-6f) continue;

            float dEdTheta = k * dTheta;
            float3 n21 = v21 / r21;
            float3 n23 = v23 / r23;

            // dTheta/dp1 = (cosTheta * n21 - n23) / (r21 * sinTheta)
            // dTheta/dp3 = (cosTheta * n23 - n21) / (r23 * sinTheta)
            // dTheta/dp2 = -(dTheta/dp1 + dTheta/dp3)
            float3 dTdp1 = (cosTheta * n21 - n23) / (r21 * sinTheta);
            float3 dTdp3 = (cosTheta * n23 - n21) / (r23 * sinTheta);

            if (isA1) gi += dEdTheta * dTdp1;
            if (isA3) gi += dEdTheta * dTdp3;
            if (isA2) gi -= dEdTheta * (dTdp1 + dTdp3);
        }
    }

    energies[tid] += ei;
    if (params.computeGrad) {
        gradients[tid] += gi;
    }
}

// ============================================================================
// MARK: - Torsion Energy + Gradient
// ============================================================================

/// Compute periodic torsion energy and gradient.
/// E_torsion = k * (1 + cos(n*phi - phase))
kernel void loop_torsion_energy(
    device const LoopRefineAtom     *atoms    [[buffer(0)]],
    device const LoopRefineTorsion  *torsions [[buffer(1)]],
    constant LoopRefineParams       &params   [[buffer(2)]],
    device float                    *energies [[buffer(3)]],
    device float3                   *gradients[[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float ei = 0.0f;
    float3 gi = float3(0.0f);

    for (uint t = 0; t < params.torsionCount; t++) {
        uint i = torsions[t].atom1;
        uint j = torsions[t].atom2;
        uint k = torsions[t].atom3;
        uint l = torsions[t].atom4;

        bool isI = (i == tid);
        bool isJ = (j == tid);
        bool isK = (k == tid);
        bool isL = (l == tid);
        if (!isI && !isJ && !isK && !isL) continue;

        float3 p0 = atoms[i].position;
        float3 p1 = atoms[j].position;
        float3 p2 = atoms[k].position;
        float3 p3 = atoms[l].position;

        float3 b1 = p1 - p0;
        float3 b2 = p2 - p1;
        float3 b3 = p3 - p2;

        float3 n1 = cross(b1, b2);
        float3 n2 = cross(b2, b3);
        float ln1 = length(n1);
        float ln2 = length(n2);
        if (ln1 < 1e-8f || ln2 < 1e-8f) continue;

        n1 /= ln1;
        n2 /= ln2;

        float cosPhi = clamp(dot(n1, n2), -1.0f, 1.0f);
        float signPhi = sign(dot(cross(n1, n2), normalize(b2)));
        float phi = signPhi * acos(cosPhi);

        float n_f = float(torsions[t].periodicity);
        float phase = torsions[t].phase;
        float kTors = torsions[t].k;

        // Energy: count once (atom j)
        if (isJ) {
            ei += kTors * (1.0f + cos(n_f * phi - phase));
        }

        // Gradient via SHAKE-style numerical approximation
        if (params.computeGrad) {
            float dEdPhi = -kTors * n_f * sin(n_f * phi - phase);
            float lb2 = length(b2);
            if (lb2 < 1e-8f) continue;

            // Gradient on terminal atoms (simplified projection)
            float3 dPhidI = -n1 * lb2 / (ln1 * ln1 + 1e-12f);
            float3 dPhidL =  n2 * lb2 / (ln2 * ln2 + 1e-12f);

            // Middle atoms by force balance
            float r1 = dot(b1, normalize(b2)) / lb2;
            float r2 = dot(b3, normalize(b2)) / lb2;

            float3 dPhidJ = -dPhidI + r1 * dPhidI - r2 * dPhidL;
            float3 dPhidK = -dPhidL - r1 * dPhidI + r2 * dPhidL;

            if (isI) gi += dEdPhi * dPhidI;
            if (isJ) gi += dEdPhi * dPhidJ;
            if (isK) gi += dEdPhi * dPhidK;
            if (isL) gi += dEdPhi * dPhidL;
        }
    }

    energies[tid] += ei;
    if (params.computeGrad) {
        gradients[tid] += gi;
    }
}

// ============================================================================
// MARK: - Steric Repulsion + Positional Restraints
// ============================================================================

/// Compute steric repulsion (loop vs non-loop) and positional restraints.
/// Steric: E = epsilon * (sigma/r)^12 (purely repulsive LJ)
/// Restraint: E = 0.5 * k * |r - r0|^2 for non-loop atoms
kernel void loop_steric_restraint(
    device const LoopRefineAtom  *atoms      [[buffer(0)]],
    device const LoopRefineAtom  *refAtoms   [[buffer(1)]],  // reference positions for restraints
    constant LoopRefineParams    &params     [[buffer(2)]],
    device float                 *energies   [[buffer(3)]],
    device float3                *gradients  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float ei = 0.0f;
    float3 gi = float3(0.0f);
    float3 pi = atoms[tid].position;
    bool isLoop = (atoms[tid].isLoop != 0);

    // Positional restraint for non-loop atoms
    if (!isLoop && params.restraintK > 0.0f) {
        float3 ref = refAtoms[tid].position;
        float3 dr = pi - ref;
        float r2 = dot(dr, dr);
        ei += 0.5f * params.restraintK * r2;
        if (params.computeGrad) {
            gi += params.restraintK * dr;
        }
    }

    // Steric repulsion: loop atoms interact with non-loop atoms
    if (isLoop) {
        float si = atoms[tid].sigma;
        float epsi = atoms[tid].epsilon;
        float cutoff2 = params.stericCutoff * params.stericCutoff;

        for (uint j = 0; j < params.atomCount; j++) {
            if (j == tid) continue;
            if (atoms[j].isLoop != 0) continue;  // only loop vs non-loop

            float3 diff = pi - atoms[j].position;
            float r2 = dot(diff, diff);
            if (r2 > cutoff2 || r2 < 1e-4f) continue;

            float r = sqrt(r2);
            float sj = atoms[j].sigma;
            float epsj = atoms[j].epsilon;
            float sigma_ij = 0.5f * (si + sj);
            float eps_ij = sqrt(epsi * epsj);

            float sr = sigma_ij / r;
            float sr6 = sr * sr * sr * sr * sr * sr;
            float sr12 = sr6 * sr6;

            // Upper triangle for energy
            if (j > tid) {
                ei += eps_ij * sr12;
            }

            // Gradient: both atoms
            if (params.computeGrad) {
                float dEdr = -12.0f * eps_ij * sr12 / r;
                gi += (dEdr / r) * diff;
            }
        }
    }

    energies[tid] += ei;
    if (params.computeGrad) {
        gradients[tid] += gi;
    }
}
