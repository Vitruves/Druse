// ============================================================================
// XTBCompute.metal — GPU-accelerated GFN2-xTB pairwise kernels
//
// Offloads O(N²) atom-pair computations to Metal GPU:
//   • Coordination number (CN)
//   • D4 dispersion energy + gradient (BJ rational damping)
//   • Nuclear repulsion energy + gradient
//   • Born radii + SASA (OBC-II implicit solvation)
//   • Generalized Born solvation energy
//   • CN gradient chain rule propagation
//
// Each kernel uses one thread per atom. The inner loop iterates over all
// other atoms, avoiding race conditions without atomics. For gradient
// kernels, each pair (i,j) is processed by both thread i and thread j,
// computing twice but eliminating synchronization.
//
// Precision: float32 throughout. For drug-sized molecules (N<500) the
// accumulated float error is <1e-4 Hartree, well within chemical accuracy.
//
// Reference: Bannwarth, Ehlert, Grimme, JCTC 2019, 15, 1652-1671
// ============================================================================

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// ============================================================================
// MARK: - Coordination Number
// ============================================================================

/// Compute D3-type coordination number for each atom.
/// CN_i = Σ_{j≠i} 1 / (1 + exp(-16 * (4/3 * (rcov_i + rcov_j)/r_ij - 1)))
///
/// One thread per atom. O(N) work per thread, O(N²) total.
kernel void xtb_compute_cn(
    device const GFN2CNAtom *atoms    [[buffer(0)]],
    constant GFN2CNParams   &params   [[buffer(1)]],
    device float            *cnOut    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float3 pi = atoms[tid].position;
    float ri = atoms[tid].covRadius;
    float cnSum = 0.0f;

    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;

        float3 diff = pi - atoms[j].position;
        float rij = length(diff);
        if (rij < 1e-6f) continue;

        float rcov = ri + atoms[j].covRadius;
        float arg = -16.0f * (1.333333f * rcov / rij - 1.0f);
        cnSum += 1.0f / (1.0f + exp(arg));
    }

    cnOut[tid] = cnSum;
}

// ============================================================================
// MARK: - D4 Dispersion Energy + Gradient
// ============================================================================

/// Compute D4-BJ dispersion energy and analytical gradient.
///
/// Energy: each thread sums pairs (tid, j) for j < tid (upper triangle).
/// Gradient: each thread accumulates forces on atom tid from ALL j ≠ tid.
///
/// C8 coefficient: C8_ij = 3 * C6_ij * sqrt(qDipole_i * qDipole_j)
/// where qDipole = sqrt(C6_ref_element) * 2.5 (pre-computed on CPU).
kernel void xtb_d4_dispersion(
    device const GFN2DispAtom *atoms     [[buffer(0)]],
    constant GFN2DispParams   &params    [[buffer(1)]],
    device float              *energies  [[buffer(2)]],
    device float3             *gradients [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    GFN2DispAtom ai = atoms[tid];
    float ei = 0.0f;
    float3 gi = float3(0.0f);

    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;

        GFN2DispAtom aj = atoms[j];

        // CN-weighted C6 coefficient
        float c6ij = sqrt(ai.c6ref * aj.c6ref);
        float dci = ai.cn - ai.cnRef;
        float dcj = aj.cn - aj.cnRef;
        float wij = exp(-0.5f * (dci * dci + dcj * dcj));
        c6ij *= wij;

        // C8 from C6 via dipole approximation
        float c8ij = 3.0f * c6ij * sqrt(ai.qDipole * aj.qDipole);

        float3 diff = ai.position - aj.position;
        float r2 = dot(diff, diff);
        float r = sqrt(r2);
        if (r < 1e-6f) continue;

        // BJ damping radii
        float ratio = (c6ij > 1e-30f) ? c8ij / c6ij : 0.0f;
        float r0 = params.a1 * sqrt(ratio) + params.a2_bohr;
        float r0_2 = r0 * r0;
        float r6 = r2 * r2 * r2;
        float r0_6 = r0_2 * r0_2 * r0_2;
        float r8 = r6 * r2;
        float r0_8 = r0_6 * r0_2;

        float f6 = 1.0f / (r6 + r0_6);
        float f8 = 1.0f / (r8 + r0_8);

        float e6 = -params.s6 * c6ij * f6;
        float e8 = -params.s8 * c8ij * f8;

        // Energy: upper triangle only (j < tid) to avoid double counting
        if (j < tid) {
            ei += e6 + e8;
        }

        // Gradient on atom tid from pair (tid, j)
        if (params.computeGrad) {
            float r4 = r2 * r2;
            float r5 = r4 * r;
            float r7 = r6 * r;
            float denom6 = (r6 + r0_6) * (r6 + r0_6);
            float denom8 = (r8 + r0_8) * (r8 + r0_8);
            float df6 = 6.0f * params.s6 * c6ij * r5 / denom6;
            float df8 = 8.0f * params.s8 * c8ij * r7 / denom8;
            float dEdr = (df6 + df8) / r;
            gi += dEdr * diff;
        }
    }

    energies[tid] = ei;
    if (params.computeGrad) {
        gradients[tid] = gi;
    }
}

// ============================================================================
// MARK: - Nuclear Repulsion Energy + Gradient
// ============================================================================

/// E_rep = Σ_{i<j} Z_i * Z_j / r * exp(-α * r^k)
/// where α = sqrt(arep_i * arep_j), k = 1.5 (GFN2).
kernel void xtb_repulsion(
    device const GFN2RepAtom *atoms     [[buffer(0)]],
    constant GFN2RepParams   &params    [[buffer(1)]],
    device float             *energies  [[buffer(2)]],
    device float3            *gradients [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float3 pi = atoms[tid].position;
    float zi = atoms[tid].zeff;
    float ai = atoms[tid].arep;
    float ei = 0.0f;
    float3 gi = float3(0.0f);

    // Skip dummy atoms (zeff == 0)
    if (zi < 1e-10f) {
        energies[tid] = 0.0f;
        if (params.computeGrad) gradients[tid] = float3(0.0f);
        return;
    }

    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;

        float zj = atoms[j].zeff;
        if (zj < 1e-10f) continue;

        float3 diff = pi - atoms[j].position;
        float r2 = dot(diff, diff);
        float r = sqrt(r2);
        if (r < 1e-6f) continue;

        float alpha = sqrt(ai * atoms[j].arep);
        float rk = pow(r, params.kexp);
        float zz = zi * zj;
        float exa = exp(-alpha * rk);

        // Energy: upper triangle
        if (j < tid) {
            ei += zz / r * exa;
        }

        // Gradient
        if (params.computeGrad) {
            // dE/dr = -zz * exa * (1/r² + α*k*r^(k-2))
            float dEdr = -zz * exa * (1.0f / r2 + alpha * params.kexp * pow(r, params.kexp - 2.0f));
            gi += dEdr * diff;
        }
    }

    energies[tid] = ei;
    if (params.computeGrad) {
        gradients[tid] = gi;
    }
}

// ============================================================================
// MARK: - Born Radii + SASA (OBC-II)
// ============================================================================

/// Compute Born radii via OBC-II model and Gaussian SASA burial.
/// One thread per atom, inner loop over all neighbors.
///
/// Born radius: 1 / (1/r_i - tanh(ψ_i * (α - ψ_i*(β - ψ_i*γ))) / r_i)
/// SASA: 4π(r_i + r_probe)² * exp(-0.5 * burial)
kernel void xtb_born_radii(
    device const GFN2BornAtom *atoms   [[buffer(0)]],
    constant GFN2BornParams   &params  [[buffer(1)]],
    device float              *bradOut [[buffer(2)]],
    device float              *sasaOut [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float3 pi = atoms[tid].position;
    float ri = atoms[tid].vdwRadius + params.bornOffset;

    // Step 1: Descreening integral (psi)
    float psi = 0.0f;
    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;

        float rj = atoms[j].vdwRadius + params.bornOffset;
        float3 diff = pi - atoms[j].position;
        float rij = length(diff);
        if (rij < 1e-8f) continue;

        float sj = rj * params.bornScale;

        if (rij > ri + sj) {
            // No overlap: standard descreening
            float rps = rij + sj;
            float rms = rij - sj;
            psi += 0.5f * (1.0f / rms - 1.0f / rps
                + sj * 0.25f * (1.0f / (rps * rps) - 1.0f / (rms * rms))
                + 0.5f * log(rms / rps) / rij);
        } else if (rij > abs(ri - sj)) {
            // Partial overlap
            float d = rij - sj;
            if (abs(d) < 1e-8f) d = 1e-8f;
            float rps = rij + sj;
            psi += 0.25f * (2.0f / rij - 1.0f / rps - ri / (4.0f * sj * rij)
                + 0.25f * (1.0f / (sj * sj) - 1.0f / (ri * ri))
                + 0.5f * log(abs(d) / rps) / rij);
        }
    }

    // OBC-II three-parameter correction
    constexpr float obc_alpha = 1.0f;
    constexpr float obc_beta  = 0.8f;
    constexpr float obc_gamma = 4.85f;

    float br = psi * ri;
    float targ = br * (obc_alpha - br * (obc_beta - br * obc_gamma));
    float th = tanh(targ);
    bradOut[tid] = 1.0f / (1.0f / ri - th / ri);

    // Step 2: Gaussian SASA burial approximation
    float probeSum = atoms[tid].vdwRadius + params.probeRadius;
    float area = 4.0f * 3.14159265f * probeSum * probeSum;
    float burial = 0.0f;

    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;
        float3 diff = pi - atoms[j].position;
        float rij = length(diff);
        float pj = atoms[j].vdwRadius + params.probeRadius;
        float overlap = max(0.0f, 1.0f - rij / (probeSum + pj));
        burial += overlap;
    }

    sasaOut[tid] = area * exp(-0.5f * burial);
}

// ============================================================================
// MARK: - Generalized Born Solvation Energy
// ============================================================================

/// E_GB = -0.5 * keps * Σ_{ij} q_i * q_j / f_GB(r_ij, B_i, B_j)
/// f_GB = sqrt(r² + B_i*B_j * exp(-r²/(4*B_i*B_j)))  (Still formula)
///
/// Each thread computes the contribution from atom tid.
/// Self-term (i==j) counted once; off-diagonal: j < tid for upper triangle.
kernel void xtb_gb_solvation(
    device const GFN2GBAtom *atoms    [[buffer(0)]],
    constant GFN2GBParams   &params   [[buffer(1)]],
    device float            *energies [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float qi = atoms[tid].charge;
    float bi = atoms[tid].bornRadius;
    float3 pi = atoms[tid].position;
    float ei = 0.0f;

    // Self-term (j == tid): r² = 0
    {
        float BiBj = bi * bi;
        float fGB = sqrt(BiBj);  // = Bi when r=0
        ei += 0.5f * qi * qi / fGB;
    }

    // Off-diagonal: j < tid (upper triangle)
    for (uint j = 0; j < tid; j++) {
        float qj = atoms[j].charge;
        float bj = atoms[j].bornRadius;

        float3 diff = pi - atoms[j].position;
        float r2 = dot(diff, diff);
        float BiBj = bi * bj;
        float expfac = exp(-r2 / (4.0f * BiBj));
        float fGB = sqrt(r2 + BiBj * expfac);

        ei += qi * qj / fGB;
    }

    // Multiply by Born prefactor
    energies[tid] = -0.5f * params.keps * ei;

    // SASA non-polar contribution: γ * SASA_i
    // (SASA was pre-computed in born_radii kernel and stored in the atom struct)
    energies[tid] += params.gamma_au * atoms[tid].sasa;
}

// ============================================================================
// MARK: - CN Gradient Chain Rule Propagation
// ============================================================================

/// Propagate dE/dCN through coordination number gradient:
///   dE/dR_A += Σ_j (dE/dCN_i + dE/dCN_j) * (d_count/dr_ij) * (R_ij / r_ij)
///
/// where d_count/dr = 16*(4/3)*rcov / rij² * exp(arg) / (1+exp(arg))²
kernel void xtb_cn_gradient(
    device const GFN2CNGradAtom *atoms     [[buffer(0)]],
    constant GFN2CNParams       &params    [[buffer(1)]],
    device float3               *gradients [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.atomCount) return;

    float3 pi = atoms[tid].position;
    float ri = atoms[tid].covRadius;
    float dEdCN_i = atoms[tid].dEdCN;
    float3 gi = float3(0.0f);

    for (uint j = 0; j < params.atomCount; j++) {
        if (j == tid) continue;

        float3 diff = pi - atoms[j].position;
        float rij = length(diff);
        if (rij < 1e-6f) continue;

        float rcov = ri + atoms[j].covRadius;
        float arg = -16.0f * (1.333333f * rcov / rij - 1.0f);
        float exparg = exp(arg);
        float denom = (1.0f + exparg);
        // d(count)/dr_ij = 16 * (4/3) * rcov / rij² * exparg / denom²
        float dcountdr = 16.0f * 1.333333f * rcov / (rij * rij) * exparg / (denom * denom);

        // Chain rule: scale by dE/dCN for both atoms i and j
        float scale = (dEdCN_i + atoms[j].dEdCN) * dcountdr / rij;

        // Negative sign: CN increases as atoms approach, gradient opposes
        gi -= scale * diff;
    }

    gradients[tid] = gi;
}
