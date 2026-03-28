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
/// 1D dispatch over upper-triangular pairs: tid maps to condensed index directly.
/// This avoids wasting ~55% of threads that a 2D NxN dispatch would discard.
/// posePositions is a flattened array: [numPoses * numAtoms] float3 values.
/// rmsdMatrix is a condensed upper-triangular matrix: [numPoses * (numPoses-1) / 2].
kernel void computePairwiseRMSD(
    device const float3 *posePositions [[buffer(0)]],
    device float *rmsdMatrix           [[buffer(1)]],
    constant RMSDParams &params        [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = params.numPoses;
    uint totalPairs = n * (n - 1) / 2;
    if (tid >= totalPairs) return;

    // Map linear tid → (i, j) upper-triangular pair.
    // Row i contains (n-1-i) pairs. Use inverse triangular formula:
    //   i = n - 1 - floor((sqrt(8*(totalPairs-1-tid)+1) - 1) / 2)
    //   j = tid - i*n + i*(i+1)/2 + i + 1
    uint k = totalPairs - 1 - tid;
    uint i = n - 1 - uint((sqrt(8.0f * float(k) + 1.0f) - 1.0f) * 0.5f);
    uint j = tid - i * n + i * (i + 1) / 2 + i + 1;

    float sumSq = 0.0f;
    uint baseI = i * params.numAtoms;
    uint baseJ = j * params.numAtoms;

    for (uint a = 0; a < params.numAtoms; a++) {
        float3 diff = posePositions[baseI + a] - posePositions[baseJ + a];
        sumSq += dot(diff, diff);
    }

    rmsdMatrix[tid] = sqrt(sumSq / float(params.numAtoms));
}

