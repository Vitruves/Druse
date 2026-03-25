#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// MARK: - EEM Matrix Construction

/// Compute the (N+1)x(N+1) EEM matrix and RHS vector on GPU.
///
/// Matrix layout (row-major):
///   A[i,j] = kappa / r_ij          (i != j, both < N)
///   A[i,i] = eta[i]                (diagonal hardness)
///   A[i,N] = A[N,i] = -1           (Lagrange multiplier)
///   A[N,N] = 0
///   b[i]   = -chi[i]               (i < N)
///   b[N]   = -totalCharge
///
kernel void computeEEMMatrix(
    device const float3 *positions   [[buffer(0)]],
    device const float  *eta         [[buffer(1)]],   // diagonal hardness
    device const float  *chi         [[buffer(2)]],   // electronegativity
    device float        *matrix      [[buffer(3)]],   // (N+1)x(N+1) row-major
    device float        *rhs         [[buffer(4)]],   // (N+1) RHS vector
    constant uint       &atomCount   [[buffer(5)]],
    constant float      &kappa       [[buffer(6)]],
    constant float      &totalCharge [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x;
    uint j = tid.y;
    uint dim = atomCount + 1;

    if (i >= dim || j >= dim) return;

    uint idx = i * dim + j;

    if (i < atomCount && j < atomCount) {
        if (i == j) {
            // Diagonal: hardness parameter
            matrix[idx] = eta[i];
            // Also fill RHS (only one thread per row does this)
            rhs[i] = -chi[i];
        } else {
            // Off-diagonal: Coulomb interaction
            float3 pi = positions[i];
            float3 pj = positions[j];
            float r = max(distance(pi, pj), 0.1f);
            matrix[idx] = kappa / r;
        }
    } else if (i == atomCount && j < atomCount) {
        // Last row: Lagrange multiplier constraint
        matrix[idx] = -1.0f;
        // Fill last RHS entry (only first thread in this row)
        if (j == 0) {
            rhs[atomCount] = -totalCharge;
        }
    } else if (i < atomCount && j == atomCount) {
        // Last column: Lagrange multiplier constraint
        matrix[idx] = -1.0f;
    } else {
        // Bottom-right corner
        matrix[idx] = 0.0f;
    }
}

// MARK: - QEq Matrix Construction

/// Compute the (N+1)x(N+1) QEq matrix with shielded Coulomb interactions.
///
/// Off-diagonal: J_ij = e^2/(4*pi*eps0) / sqrt(r_ij^2 + sigma_ij^2)
///   where sigma_ij = 1/(2*J_i) + 1/(2*J_j)
/// Diagonal: J_ii = J0[i] (idempotential / self-Coulomb)
///
kernel void computeQEqMatrix(
    device const float3 *positions   [[buffer(0)]],
    device const float  *J0          [[buffer(1)]],   // idempotential per atom
    device const float  *chi         [[buffer(2)]],   // electronegativity
    device float        *matrix      [[buffer(3)]],   // (N+1)x(N+1)
    device float        *rhs         [[buffer(4)]],
    constant uint       &atomCount   [[buffer(5)]],
    constant float      &totalCharge [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x;
    uint j = tid.y;
    uint dim = atomCount + 1;

    if (i >= dim || j >= dim) return;

    uint idx = i * dim + j;

    if (i < atomCount && j < atomCount) {
        if (i == j) {
            // Diagonal: self-Coulomb = idempotential
            matrix[idx] = J0[i];
            rhs[i] = -chi[i];
        } else {
            // Off-diagonal: shielded Coulomb
            float3 pi = positions[i];
            float3 pj = positions[j];
            float r2 = distance_squared(pi, pj);

            // Shielding: sigma = 1/(2*J_i) + 1/(2*J_j)
            float sigma = 0.5f / J0[i] + 0.5f / J0[j];
            float sigma2 = sigma * sigma;

            // J_ij = e^2/(4*pi*eps0) / sqrt(r^2 + sigma^2) in eV*Angstrom units
            matrix[idx] = 14.3996f / sqrt(r2 + sigma2);
        }
    } else if (i == atomCount && j < atomCount) {
        // Lagrange multiplier row
        matrix[idx] = -1.0f;
        if (j == 0) {
            rhs[atomCount] = -totalCharge;
        }
    } else if (i < atomCount && j == atomCount) {
        // Lagrange multiplier column
        matrix[idx] = -1.0f;
    } else {
        // Corner
        matrix[idx] = 0.0f;
    }
}

// MARK: - GFN2-xTB Gamma Matrix

/// Compute the Coulomb gamma matrix for GFN2-xTB tight-binding.
///
/// Uses the exponential gamma functional from DFTB:
///   gamma_ij = 1/r - f_exp(r, tau_i, tau_j)
/// where tau = 3.2 * U_H (Hubbard parameter).
///
/// For on-site (same atom): gamma_ii = U_i (self-energy = hardness)
/// For intra-atomic (different shells, same atom): averaged hardness
///
kernel void computeXTBGammaMatrix(
    device const float3 *positions    [[buffer(0)]],   // Angstrom
    device const float  *hubbardU     [[buffer(1)]],   // per shell
    device float        *gamma        [[buffer(2)]],   // nshell x nshell
    constant uint       &nshell       [[buffer(3)]],
    device const uint   *shellToAtom  [[buffer(4)]],   // shell -> atom index
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x;
    uint j = tid.y;
    if (i >= nshell || j >= nshell) return;

    uint iat = shellToAtom[i];
    uint jat = shellToAtom[j];

    float ui = hubbardU[i];
    float uj = hubbardU[j];

    float gam;

    if (iat == jat) {
        // On-site interactions
        if (i == j) {
            // Self-energy: hardness parameter
            gam = ui;
        } else {
            // Intra-atomic: average of shell hardnesses
            gam = 0.5f * (ui + uj);
        }
    } else {
        // Inter-atomic: exponential gamma functional
        float3 ri = positions[iat];
        float3 rj = positions[jat];
        float r = distance(ri, rj);
        r = max(r, 0.01f);

        // Convert Angstrom to Bohr for gamma calculation
        float r_bohr = r / 0.529177f;

        // Decay parameters: tau = 3.2 * U (controls range of Coulomb interaction)
        float taui = 3.2f * ui;
        float tauj = 3.2f * uj;

        float exp_gam;

        if (abs(ui - uj) < 0.001f) {
            // Same hardness: symmetric Slater-type overlap formula
            float tau = 0.5f * (taui + tauj);
            float tr = tau * r_bohr;
            float exp_term = exp(-tr);
            // Polynomial damping from Elstner et al.
            exp_gam = exp_term * (48.0f / r_bohr
                     + 33.0f * tau
                     + 9.0f * r_bohr * tau * tau
                     + r_bohr * r_bohr * tau * tau * tau) / 48.0f;
        } else {
            // Different hardness: asymmetric two-center formula
            float t4i = taui * taui * taui * taui;
            float t4j = tauj * tauj * tauj * tauj;
            float di = taui * taui - tauj * tauj;
            float dj = tauj * tauj - taui * taui;

            // Two-center integral from Ohno-Klopman type expansion
            exp_gam = exp(-taui * r_bohr) * t4j * taui / (2.0f * di * di)
                     * (1.0f - (2.0f * tauj * tauj * taui) / (r_bohr * di))
                    + exp(-tauj * r_bohr) * t4i * tauj / (2.0f * dj * dj)
                     * (1.0f - (2.0f * taui * taui * tauj) / (r_bohr * dj));
        }

        // gamma = 1/r - f_exp(r) in Bohr, convert Hartree -> eV
        gam = (1.0f / r_bohr - exp_gam) * 27.2114f;
    }

    gamma[i * nshell + j] = gam;
}

// ============================================================================
// MARK: - GFN2-xTB Born Radii (GPU)
// ============================================================================

/// Compute Born radii pairwise descreening integral on GPU.
/// Each thread (i,j) computes atom j's contribution to atom i's psi integral.
/// Results are atomically accumulated and the Born radius is finalized in a
/// second pass kernel.
///
/// This is the dominant O(N²) cost in GBSA/ALPB solvation.
kernel void computeBornPsi(
    device const GFN2BornAtom *atoms     [[buffer(0)]],
    device float              *psi       [[buffer(1)]],    // N psi values (atomic accumulate)
    constant GFN2BornParams   &params    [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x;
    uint j = tid.y;
    uint N = params.atomCount;
    if (i >= N || j >= N || i == j) return;

    float3 pi = atoms[i].position;
    float3 pj = atoms[j].position;
    float rij = max(distance(pi, pj), 0.001f);

    float ri = atoms[i].vdwRadius + params.bornOffset;
    float sj = (atoms[j].vdwRadius + params.bornOffset) * params.bornScale;

    float contribution = 0.0f;

    if (rij > ri + sj) {
        // No overlap
        float rps = rij + sj;
        float rms = rij - sj;
        contribution = 0.5f * (1.0f/rms - 1.0f/rps
            + sj * 0.25f * (1.0f/(rps*rps) - 1.0f/(rms*rms))
            + 0.5f * log(rms/rps) / rij);
    } else if (rij > abs(ri - sj)) {
        // Partial overlap
        float rps = rij + sj;
        float d = max(rij - sj, 0.001f);
        contribution = 0.25f * (2.0f/rij - 1.0f/rps - ri/(4.0f*sj*rij)
            + 0.25f * (1.0f/(sj*sj) - 1.0f/(ri*ri))
            + 0.5f * log(abs(d)/rps) / rij);
    }

    // Atomic add to psi[i]
    // Metal supports atomic_fetch_add_explicit for device float on Apple Silicon
    atomic_fetch_add_explicit(
        (device atomic_uint *)&psi[i],
        as_type<uint>(contribution),
        memory_order_relaxed);
}

/// Finalize Born radii from psi values using OBC-II correction.
kernel void finalizeBornRadii(
    device const GFN2BornAtom *atoms     [[buffer(0)]],
    device const float        *psi       [[buffer(1)]],
    device float              *brad      [[buffer(2)]],    // output Born radii
    device float              *sasa      [[buffer(3)]],    // output SASA per atom
    constant GFN2BornParams   &params    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if (i >= params.atomCount) return;

    float ri = atoms[i].vdwRadius + params.bornOffset;
    float br = psi[i] * ri;

    // OBC-II three-parameter scaling
    float alpha = 1.0f;
    float beta  = 0.8f;
    float gamm  = 4.85f;
    float arg = br * (alpha - br * (beta - br * gamm));
    float th = tanh(arg);

    brad[i] = 1.0f / (1.0f/ri - th/ri);

    // Approximate SASA (Gaussian burial)
    float probe_sum = atoms[i].vdwRadius + params.probeRadius;
    sasa[i] = 4.0f * M_PI_F * probe_sum * probe_sum;
}

// ============================================================================
// MARK: - GFN2-xTB D4 Dispersion Pairwise (GPU)
// ============================================================================

/// Compute pairwise D4 dispersion energy contributions on GPU.
/// Each thread handles one (i, j) pair with i > j.
/// Energy contributions are written to a pairwise output buffer
/// that is summed on CPU (or via parallel reduction).
kernel void computeD4Dispersion(
    device const GFN2DispAtom *atoms     [[buffer(0)]],
    device float              *pairEnergy [[buffer(1)]],   // N*(N-1)/2 pair energies
    constant GFN2DispParams   &params    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint N = params.atomCount;
    uint npairs = N * (N - 1) / 2;
    if (tid >= npairs) return;

    // Decode pair index: tid → (i, j) with i > j
    // Using inverse triangular number formula
    uint i = (uint)(0.5f + sqrt(0.25f + 2.0f * (float)tid));
    uint j = tid - i * (i - 1) / 2;
    if (i >= N) { i = N - 1; j = 0; }
    if (j >= i) return;

    float3 pi = atoms[i].position;
    float3 pj = atoms[j].position;
    float r2 = distance_squared(pi, pj);
    float r = sqrt(r2);
    if (r < 0.01f) { pairEnergy[tid] = 0.0f; return; }

    // CN-dependent C6 weighting
    float wi = exp(-4.0f * (atoms[i].cn - atoms[i].cnRef) * (atoms[i].cn - atoms[i].cnRef));
    float wj = exp(-4.0f * (atoms[j].cn - atoms[j].cnRef) * (atoms[j].cn - atoms[j].cnRef));
    float c6ij = sqrt(atoms[i].c6ref * atoms[j].c6ref) * wi * wj;

    // C8 approximation
    float c8ij = 3.0f * c6ij * sqrt(sqrt(atoms[i].c6ref) * 2.5f * sqrt(atoms[j].c6ref) * 2.5f);

    // BJ damping
    float r0 = params.a1 * sqrt(c6ij > 0 ? c8ij / c6ij : 0.0f) + params.a2_bohr;
    float r0_2 = r0 * r0;
    float r0_6 = r0_2 * r0_2 * r0_2;
    float r0_8 = r0_6 * r0_2;

    float r6 = r2 * r2 * r2;
    float r8 = r6 * r2;

    float e6 = -params.s6 * c6ij / (r6 + r0_6);
    float e8 = -params.s8 * c8ij / (r8 + r0_8);

    pairEnergy[tid] = e6 + e8;
}
