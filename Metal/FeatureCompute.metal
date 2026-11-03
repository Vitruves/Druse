// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - RBF Distance Encoding (ML Feature Extraction)
// ============================================================================

/// Compute pairwise distances and RBF-encoded features between protein and ligand atoms.
/// Each thread handles one (protein, ligand) atom pair.
/// Output: distOutput[nProt * nLig], rbfOutput[nProt * nLig * numBins]
kernel void computeRBFDistances(
    device const float3 *protPositions [[buffer(0)]],
    device const float3 *ligPositions  [[buffer(1)]],
    device float *rbfOutput            [[buffer(2)]],
    device float *distOutput           [[buffer(3)]],
    constant RBFParams &params         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint nPairs = params.nProt * params.nLig;
    if (tid >= nPairs) return;

    uint i = tid / params.nLig;
    uint j = tid % params.nLig;

    float d = distance(protPositions[i], ligPositions[j]);
    distOutput[tid] = d;

    uint rbfBase = tid * params.numBins;
    for (uint k = 0; k < params.numBins; k++) {
        float center = float(k) * params.binSpacing;
        float diff = d - center;
        rbfOutput[rbfBase + k] = exp(-params.gamma * diff * diff);
    }
}
