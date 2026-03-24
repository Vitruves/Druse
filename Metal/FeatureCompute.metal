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

// ============================================================================
// MARK: - ML Pocket Detection: Surface Feature Computation
// ============================================================================

/// Each thread processes one grid point: finds nearest atom, computes 11-dim features.
/// Replaces O(gridPoints × atoms) CPU loop with GPU parallelism.
kernel void pocketSurfaceFeatures(
    device const float3 *gridPoints       [[buffer(0)]],
    device const PocketMLAtom *atoms      [[buffer(1)]],
    device PocketSurfacePoint *output     [[buffer(2)]],
    device atomic_uint *validCount        [[buffer(3)]],
    constant PocketDetectParams &params   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numGridPoints) return;

    float3 gp = gridPoints[tid];

    // Find nearest atom (brute force — fast on GPU with thousands of threads)
    float minDist = INFINITY;
    uint nearestIdx = 0;
    for (uint i = 0; i < params.numAtoms; i++) {
        float d = distance(gp, atoms[i].position);
        if (d < minDist) {
            minDist = d;
            nearestIdx = i;
        }
    }

    // Surface filter: only keep points near the molecular surface
    float surfaceDist = minDist - atoms[nearestIdx].vdwRadius;
    if (surfaceDist < -0.5f || surfaceDist > params.probeRadius + 0.5f) return;

    // Buriedness: count atoms within cutoff radius
    uint nearbyCount = 0;
    for (uint i = 0; i < params.numAtoms; i++) {
        if (distance(gp, atoms[i].position) <= params.buriednessCutoff) {
            nearbyCount++;
        }
    }

    // Compute features
    float3 normal = gp - atoms[nearestIdx].position;
    float normLen = length(normal);
    if (normLen > 0) normal /= normLen;

    PocketSurfacePoint pt;
    pt.position = gp;
    pt.nearestDist = normLen;
    pt.normal = normal;
    pt.hydrophobicity = atoms[nearestIdx].hydrophobicity;
    pt.charge = atoms[nearestIdx].charge;
    pt.aromatic = (atoms[nearestIdx].flags & 2) ? 1.0f : 0.0f;
    pt.donor = (atoms[nearestIdx].flags & 1) ? 1.0f : 0.0f;
    pt.acceptor = (atoms[nearestIdx].flags & 1) ? 1.0f : 0.0f;
    pt.buriedness = min(float(nearbyCount) / 20.0f, 1.0f);
    pt.curvature = 0.5f;
    pt.nearestAtomIdx = nearestIdx;

    uint idx = atomic_fetch_add_explicit(validCount, 1, memory_order_relaxed);
    output[idx] = pt;
}

// ============================================================================
// MARK: - Spatial Hash Grid Construction for Accelerated KNN
// ============================================================================

/// Pass 1: Count how many points fall into each grid cell.
/// Each thread handles one point, atomically increments its cell's counter.
/// Output: cellCounts[totalCells] — number of points per cell.
kernel void buildSpatialHashCount(
    device const PocketSurfacePoint *points  [[buffer(0)]],
    device atomic_uint *cellCounts           [[buffer(1)]],
    constant SpatialHashParams &params       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numPoints) return;

    float3 pos = points[tid].position;
    float3 rel = (pos - params.gridOrigin) / params.cellSize;
    uint cx = min(uint(max(rel.x, 0.0f)), params.gridDims.x - 1);
    uint cy = min(uint(max(rel.y, 0.0f)), params.gridDims.y - 1);
    uint cz = min(uint(max(rel.z, 0.0f)), params.gridDims.z - 1);

    uint cellIdx = cx + cy * params.gridDims.x + cz * params.gridDims.x * params.gridDims.y;
    atomic_fetch_add_explicit(&cellCounts[cellIdx], 1, memory_order_relaxed);
}

/// Pass 2: Scatter points into a cell-sorted index buffer.
/// Uses cellOffsets (prefix-sum of cellCounts, computed on CPU) to write each
/// point's index into the correct position. After this pass:
///   sortedIndices[cellOffsets[cell]..cellOffsets[cell]+cellCounts[cell]] holds
///   the indices of all points in that cell.
/// cellCounts is re-used as an atomic write cursor (reset to 0 before dispatch).
kernel void buildSpatialHashScatter(
    device const PocketSurfacePoint *points  [[buffer(0)]],
    device uint *sortedIndices               [[buffer(1)]],
    device const uint *cellOffsets           [[buffer(2)]],
    device atomic_uint *cellCounts           [[buffer(3)]],
    constant SpatialHashParams &params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numPoints) return;

    float3 pos = points[tid].position;
    float3 rel = (pos - params.gridOrigin) / params.cellSize;
    uint cx = min(uint(max(rel.x, 0.0f)), params.gridDims.x - 1);
    uint cy = min(uint(max(rel.y, 0.0f)), params.gridDims.y - 1);
    uint cz = min(uint(max(rel.z, 0.0f)), params.gridDims.z - 1);

    uint cellIdx = cx + cy * params.gridDims.x + cz * params.gridDims.x * params.gridDims.y;
    uint slot = atomic_fetch_add_explicit(&cellCounts[cellIdx], 1, memory_order_relaxed);
    sortedIndices[cellOffsets[cellIdx] + slot] = tid;
}

// ============================================================================
// MARK: - ML Pocket Detection: KNN + Neighbor Feature Aggregation
// ============================================================================

/// Each thread handles one point: finds k nearest neighbors and averages their features.
/// Output: neighborFeatures[numPoints * featureSize] — pre-aggregated for CoreML input.
///
/// When useSpatialHash == 1, uses a spatial hash grid (buffers 3-5) to restrict
/// neighbor search to the 27 surrounding cells, giving O(n) instead of O(n^2).
/// Falls back to brute-force when useSpatialHash == 0.
kernel void pocketKNNAggregate(
    device const PocketSurfacePoint *points  [[buffer(0)]],
    device float *neighborFeatures           [[buffer(1)]],
    constant PocketKNNParams &params         [[buffer(2)]],
    device const uint *sortedIndices         [[buffer(3)]],
    device const uint *cellOffsets           [[buffer(4)]],
    constant SpatialHashParams *hashParams   [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numPoints) return;

    float3 queryPos = points[tid].position;

    // Maintain a sorted list of k nearest distances (insertion sort for small k)
    const uint MAX_K = 32;
    uint k = min(params.k, MAX_K);
    float kDists[MAX_K];
    uint kIndices[MAX_K];
    for (uint i = 0; i < k; i++) { kDists[i] = INFINITY; kIndices[i] = tid; }

    if (params.useSpatialHash != 0) {
        // ---- Spatial hash accelerated path ----
        SpatialHashParams hp = hashParams[0];
        float3 rel = (queryPos - hp.gridOrigin) / hp.cellSize;
        int cx = int(max(rel.x, 0.0f));
        int cy = int(max(rel.y, 0.0f));
        int cz = int(max(rel.z, 0.0f));
        cx = min(cx, int(hp.gridDims.x) - 1);
        cy = min(cy, int(hp.gridDims.y) - 1);
        cz = min(cz, int(hp.gridDims.z) - 1);

        // Iterate over 3x3x3 neighborhood of cells
        for (int dz = -1; dz <= 1; dz++) {
            int nz = cz + dz;
            if (nz < 0 || nz >= int(hp.gridDims.z)) continue;
            for (int dy = -1; dy <= 1; dy++) {
                int ny = cy + dy;
                if (ny < 0 || ny >= int(hp.gridDims.y)) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx;
                    if (nx < 0 || nx >= int(hp.gridDims.x)) continue;

                    uint cellIdx = uint(nx) + uint(ny) * hp.gridDims.x
                                 + uint(nz) * hp.gridDims.x * hp.gridDims.y;
                    uint start = cellOffsets[cellIdx];
                    uint end   = cellOffsets[cellIdx + 1]; // prefix-sum: end of cell

                    for (uint s = start; s < end; s++) {
                        uint i = sortedIndices[s];
                        if (i == tid) continue;
                        float d = distance_squared(queryPos, points[i].position);

                        if (d < kDists[k - 1]) {
                            kDists[k - 1] = d;
                            kIndices[k - 1] = i;
                            // Bubble up to maintain sorted order
                            for (int j = int(k) - 2; j >= 0; j--) {
                                if (kDists[j + 1] < kDists[j]) {
                                    float td2 = kDists[j]; kDists[j] = kDists[j + 1]; kDists[j + 1] = td2;
                                    uint ti = kIndices[j]; kIndices[j] = kIndices[j + 1]; kIndices[j + 1] = ti;
                                } else break;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // ---- Brute-force fallback path ----
        for (uint i = 0; i < params.numPoints; i++) {
            if (i == tid) continue;
            float d = distance_squared(queryPos, points[i].position);

            if (d < kDists[k - 1]) {
                kDists[k - 1] = d;
                kIndices[k - 1] = i;
                for (int j = int(k) - 2; j >= 0; j--) {
                    if (kDists[j + 1] < kDists[j]) {
                        float td2 = kDists[j]; kDists[j] = kDists[j + 1]; kDists[j + 1] = td2;
                        uint ti = kIndices[j]; kIndices[j] = kIndices[j + 1]; kIndices[j + 1] = ti;
                    } else break;
                }
            }
        }
    }

    // Average neighbor features (11-dim)
    float meanFeat[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint ni = 0; ni < k; ni++) {
        PocketSurfacePoint nb = points[kIndices[ni]];
        meanFeat[0] += nb.normal.x;
        meanFeat[1] += nb.normal.y;
        meanFeat[2] += nb.normal.z;
        meanFeat[3] += nb.nearestDist;
        meanFeat[4] += nb.hydrophobicity;
        meanFeat[5] += nb.charge;
        meanFeat[6] += nb.aromatic;
        meanFeat[7] += nb.donor;
        meanFeat[8] += nb.acceptor;
        meanFeat[9] += nb.buriedness;
        meanFeat[10] += nb.curvature;
    }

    float scale = 1.0f / float(k);
    uint base = tid * params.featureSize;
    for (uint j = 0; j < 11; j++) {
        neighborFeatures[base + j] = meanFeat[j] * scale;
    }
}
