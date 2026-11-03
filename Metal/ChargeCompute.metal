// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

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
/// Off-diagonal (Klopman–Ohno shielded Coulomb; Klopman, JACS 1964, 86, 4550):
///   J_ij = k / sqrt(r_ij^2 + a_ij^2),  a_ij = k * (1/(2*J_i) + 1/(2*J_j))  [Å]
///   with k = e^2/(4*pi*eps0) = 14.3996 eV*Å. Must match QEqChargeCalculator.solveCPU.
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

            // Klopman–Ohno shielding length in Å (k/J turns eV^-1 into Å)
            const float k = 14.3996f;
            float a = k * (0.5f / J0[i] + 0.5f / J0[j]);
            matrix[idx] = k / sqrt(r2 + a * a);
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
