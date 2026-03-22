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

// ============================================================================
// MARK: - Pairwise RMSD (Pose Clustering)
// ============================================================================

/// Compute pairwise RMSD between all pose pairs for clustering.
/// 2D dispatch: tid.x = pose i, tid.y = pose j (upper triangle only).
/// posePositions is a flattened array: [numPoses * numAtoms] float3 values.
/// rmsdMatrix is a condensed upper-triangular matrix: [numPoses * (numPoses-1) / 2].
kernel void computePairwiseRMSD(
    device const float3 *posePositions [[buffer(0)]],
    device float *rmsdMatrix           [[buffer(1)]],
    constant RMSDParams &params        [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint j = tid.y;
    uint n = params.numPoses;

    // Only compute upper triangle (i < j)
    if (i >= n || j >= n || i >= j) return;

    // Condensed distance matrix index: i * n - i * (i + 1) / 2 + j - i - 1
    uint idx = i * n - i * (i + 1) / 2 + j - i - 1;

    float sumSq = 0.0f;
    uint baseI = i * params.numAtoms;
    uint baseJ = j * params.numAtoms;

    for (uint a = 0; a < params.numAtoms; a++) {
        float3 diff = posePositions[baseI + a] - posePositions[baseJ + a];
        sumSq += dot(diff, diff);
    }

    rmsdMatrix[idx] = sqrt(sumSq / float(params.numAtoms));
}
