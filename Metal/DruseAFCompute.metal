// DruseAFCompute.metal — DruseAF neural network scoring on Metal GPU
//
// Implements DruseScorePKi v2 forward pass as two compute kernels:
//   Kernel 1 (druseAFEncode): pose transform + MLP encoding → prot_h, lig_h per pose
//   Kernel 2 (druseAFScore):  cross-attention × 2 → gated pooling → affinity/confidence heads
//
// Architecture: atom_dim=20, hidden=128, heads=4, head_dim=32, rbf=50, 2 cross-attn layers
// Score output: docking_score = pKd × confidence (negated for GA minimization)

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// ============================================================================
// Weight tensor indices (must match export_druseaf_weights.py EXPORT_WEIGHT_ORDER)
// ============================================================================

// MLP encoders
#define W_PROT_ENC_0_W   0   // [128, 20]
#define W_PROT_ENC_0_B   1   // [128]
#define W_PROT_ENC_2_W   2   // [128, 128]
#define W_PROT_ENC_2_B   3   // [128]
#define W_LIG_ENC_0_W    4   // [128, 20]
#define W_LIG_ENC_0_B    5   // [128]
#define W_LIG_ENC_2_W    6   // [128, 128]
#define W_LIG_ENC_2_B    7   // [128]

// Cross-attention layer 0
#define W_CA0_Q_W        8   // [128, 128]
#define W_CA0_Q_B        9   // [128]
#define W_CA0_K_W       10   // [128, 128]
#define W_CA0_K_B       11   // [128]
#define W_CA0_V_W       12   // [128, 128]
#define W_CA0_V_B       13   // [128]
#define W_CA0_RBF_W     14   // [4, 50]
#define W_CA0_RBF_B     15   // [4]
#define W_CA0_OUT_W     16   // [128, 128]
#define W_CA0_OUT_B     17   // [128]
#define W_CA0_NORM_W    18   // [128]
#define W_CA0_NORM_B    19   // [128]

// Cross-attention layer 1
#define W_CA1_Q_W       20
#define W_CA1_Q_B       21
#define W_CA1_K_W       22
#define W_CA1_K_B       23
#define W_CA1_V_W       24
#define W_CA1_V_B       25
#define W_CA1_RBF_W     26
#define W_CA1_RBF_B     27
#define W_CA1_OUT_W     28
#define W_CA1_OUT_B     29
#define W_CA1_NORM_W    30
#define W_CA1_NORM_B    31

// Gated attention pooling
#define W_GATE_0_W      32   // [64, 128]
#define W_GATE_0_B      33   // [64]
#define W_GATE_2_W      34   // [1, 64]
#define W_GATE_2_B      35   // [1]

// Affinity head
#define W_AFF_0_W       36   // [128, 128]
#define W_AFF_0_B       37   // [128]
#define W_AFF_3_W       38   // [1, 128]
#define W_AFF_3_B       39   // [1]

// Confidence head
#define W_CONF_0_W      40   // [64, 128]
#define W_CONF_0_B      41   // [64]
#define W_CONF_3_W      42   // [1, 64]
#define W_CONF_3_B      43   // [1]

// Max dimensions
#define DAF_MAX_PROT 256
#define DAF_MAX_LIG   64
#define DAF_HIDDEN   128
#define DAF_HEADS      4
#define DAF_HEAD_DIM  32
#define DAF_RBF_BINS  50
#define DAF_ATOM_DIM  20

// ============================================================================
// Helpers
// ============================================================================

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/// Get pointer to weight tensor by index.
inline device const float* getWeight(
    device const float *weights,
    constant DruseAFWeightEntry *entries,
    uint index)
{
    return weights + entries[index].offset;
}

/// Linear layer: out[outDim] = W[outDim, inDim] @ in[inDim] + b[outDim]
/// W is stored row-major: W[row * inDim + col]
inline void linearLayer(
    thread float *out,
    thread const float *in_ptr,
    device const float *W,
    device const float *b,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = b[i];
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

/// Linear layer overload for constant-space input (pre-computed features).
inline void linearLayer(
    thread float *out,
    constant const float *in_ptr,
    device const float *W,
    device const float *b,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = b[i];
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

/// Linear layer overload for device-space input (intermediate buffers).
inline void linearLayer(
    thread float *out,
    device const float *in_ptr,
    device const float *W,
    device const float *b,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = b[i];
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

/// LayerNorm: out = gamma * (x - mean) / sqrt(var + eps) + beta
inline void layerNorm(
    thread float *x,
    device const float *gamma,
    device const float *beta,
    uint dim)
{
    float mean = 0.0f;
    for (uint i = 0; i < dim; i++) mean += x[i];
    mean /= float(dim);

    float var = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= float(dim);

    float inv_std = 1.0f / sqrt(var + 1e-5f);
    for (uint i = 0; i < dim; i++) {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

/// Rotate vector v by unit quaternion q = (x,y,z,w).
inline float3 dafQuatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

/// Apply torsion rotation around axis atom1→atom2 to selected atoms.
inline void dafApplyTorsions(
    thread float3 *pos, uint nAtoms,
    device const DockPose &pose,
    constant TorsionEdge *edges,
    constant int32_t *movingIdx,
    uint nTorsions)
{
    for (uint t = 0; t < nTorsions; t++) {
        constant TorsionEdge &edge = edges[t];
        float angle = pose.torsions[t];
        if (abs(angle) < 1e-6f) continue;

        float3 axisOrigin = pos[edge.atom1];
        float3 axisDir = normalize(pos[edge.atom2] - axisOrigin);
        float cosA = cos(angle), sinA = sin(angle);

        for (int32_t mi = 0; mi < edge.movingCount; mi++) {
            int32_t ai = movingIdx[edge.movingStart + mi];
            if (ai < 0 || uint(ai) >= nAtoms) continue;
            float3 rel = pos[ai] - axisOrigin;
            float d = dot(rel, axisDir);
            float3 proj = axisDir * d;
            float3 perp = rel - proj;
            float3 w = cross(axisDir, perp);
            pos[ai] = axisOrigin + proj + perp * cosA + w * sinA;
        }
    }
}

// ============================================================================
// Kernel 1: Feature Extraction + MLP Encoding
// ============================================================================

// Per-pose intermediate buffer layout:
//   prot_h:  [DAF_MAX_PROT * DAF_HIDDEN] floats
//   lig_h:   [DAF_MAX_LIG * DAF_HIDDEN] floats
//   lig_pos: [DAF_MAX_LIG * 3] floats
// Total per pose: 256*128 + 64*128 + 64*3 = 41,152 floats = ~161 KB
#define DAF_INTERMEDIATE_FLOATS_PER_POSE (DAF_MAX_PROT * DAF_HIDDEN + DAF_MAX_LIG * DAF_HIDDEN + DAF_MAX_LIG * 3)
#define DAF_PROT_H_OFFSET 0
#define DAF_LIG_H_OFFSET  (DAF_MAX_PROT * DAF_HIDDEN)
#define DAF_LIG_POS_OFFSET (DAF_MAX_PROT * DAF_HIDDEN + DAF_MAX_LIG * DAF_HIDDEN)

kernel void druseAFEncode(
    device DockPose                *poses          [[buffer(0)]],
    constant DockLigandAtom        *ligandAtoms    [[buffer(1)]],
    constant GAParams              &gaParams       [[buffer(2)]],
    constant TorsionEdge           *torsionEdges   [[buffer(3)]],
    constant int32_t               *movingIndices  [[buffer(4)]],
    constant float                 *protFeatures   [[buffer(5)]],   // [P, 20] pre-computed
    constant float                 *ligFeatures    [[buffer(6)]],   // [L, 20] pre-computed
    constant float3                *protPositions  [[buffer(7)]],   // [P] protein pocket positions
    device const float             *weights        [[buffer(8)]],
    constant DruseAFWeightEntry    *weightEntries  [[buffer(9)]],
    constant DruseAFParams         &afParams       [[buffer(10)]],
    device float                   *intermediates  [[buffer(11)]],  // output per-pose
    constant GridParams            &gridParams     [[buffer(12)]],
    uint                            tid            [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint P = afParams.numProteinAtoms;
    uint L = min(gaParams.numLigandAtoms, uint(DAF_MAX_LIG));
    uint nTorsions = min(gaParams.numTorsions, 32u);

    // Get per-pose output region
    device float *myIntermediate = intermediates + tid * DAF_INTERMEDIATE_FLOATS_PER_POSE;
    device float *prot_h = myIntermediate + DAF_PROT_H_OFFSET;
    device float *lig_h  = myIntermediate + DAF_LIG_H_OFFSET;
    device float *lig_pos_out = myIntermediate + DAF_LIG_POS_OFFSET;

    // Weight pointers for MLP encoders
    device const float *prot_w0 = getWeight(weights, weightEntries, W_PROT_ENC_0_W);
    device const float *prot_b0 = getWeight(weights, weightEntries, W_PROT_ENC_0_B);
    device const float *prot_w2 = getWeight(weights, weightEntries, W_PROT_ENC_2_W);
    device const float *prot_b2 = getWeight(weights, weightEntries, W_PROT_ENC_2_B);
    device const float *lig_w0  = getWeight(weights, weightEntries, W_LIG_ENC_0_W);
    device const float *lig_b0  = getWeight(weights, weightEntries, W_LIG_ENC_0_B);
    device const float *lig_w2  = getWeight(weights, weightEntries, W_LIG_ENC_2_W);
    device const float *lig_b2  = getWeight(weights, weightEntries, W_LIG_ENC_2_B);

    // 1. Transform ligand atoms (rigid body + torsions)
    float3 ligPositions[DAF_MAX_LIG];
    device const DockPose &pose = poses[tid];
    for (uint a = 0; a < L; a++) {
        ligPositions[a] = dafQuatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    dafApplyTorsions(ligPositions, L, pose, torsionEdges, movingIndices, nTorsions);

    // Write ligand positions to intermediate buffer (needed by Kernel 2 for distances)
    for (uint a = 0; a < L; a++) {
        lig_pos_out[a * 3 + 0] = ligPositions[a].x;
        lig_pos_out[a * 3 + 1] = ligPositions[a].y;
        lig_pos_out[a * 3 + 2] = ligPositions[a].z;
    }

    // Check out-of-grid penalty (keep poses in search volume)
    float3 gridMin = gridParams.origin;
    float3 gridMax = gridParams.origin + float3(gridParams.dims) * gridParams.spacing;
    float oopTotal = 0.0f;
    for (uint a = 0; a < L; a++) {
        float3 r = ligPositions[a];
        float3 belowMin = max(gridMin - r, float3(0.0f));
        float3 aboveMax = max(r - gridMax, float3(0.0f));
        float3 d = belowMin + aboveMax;
        oopTotal += dot(d, d);
    }
    // Store out-of-grid penalty in pose for use by Kernel 2
    poses[tid].clashPenalty = oopTotal * 10.0f;

    // 2. Encode protein atoms: Linear(20→128) → SiLU → Linear(128→128)
    for (uint p = 0; p < P; p++) {
        constant float *feat = protFeatures + p * DAF_ATOM_DIM;
        float h1[DAF_HIDDEN];
        linearLayer(h1, feat, prot_w0, prot_b0, DAF_HIDDEN, DAF_ATOM_DIM);
        for (uint i = 0; i < DAF_HIDDEN; i++) h1[i] = silu(h1[i]);
        float h2[DAF_HIDDEN];
        linearLayer(h2, h1, prot_w2, prot_b2, DAF_HIDDEN, DAF_HIDDEN);
        for (uint i = 0; i < DAF_HIDDEN; i++) prot_h[p * DAF_HIDDEN + i] = h2[i];
    }
    // Zero-pad remaining protein slots
    for (uint p = P; p < DAF_MAX_PROT; p++) {
        for (uint i = 0; i < DAF_HIDDEN; i++) prot_h[p * DAF_HIDDEN + i] = 0.0f;
    }

    // 3. Encode ligand atoms: Linear(20→128) → SiLU → Linear(128→128)
    for (uint l = 0; l < L; l++) {
        constant float *feat = ligFeatures + l * DAF_ATOM_DIM;
        float h1[DAF_HIDDEN];
        linearLayer(h1, feat, lig_w0, lig_b0, DAF_HIDDEN, DAF_ATOM_DIM);
        for (uint i = 0; i < DAF_HIDDEN; i++) h1[i] = silu(h1[i]);
        float h2[DAF_HIDDEN];
        linearLayer(h2, h1, lig_w2, lig_b2, DAF_HIDDEN, DAF_HIDDEN);
        for (uint i = 0; i < DAF_HIDDEN; i++) lig_h[l * DAF_HIDDEN + i] = h2[i];
    }
    // Zero-pad remaining ligand slots
    for (uint l = L; l < DAF_MAX_LIG; l++) {
        for (uint i = 0; i < DAF_HIDDEN; i++) lig_h[l * DAF_HIDDEN + i] = 0.0f;
    }
}


// ============================================================================
// Kernel 2: Cross-Attention + Gated Pooling + Heads
// ============================================================================
//
// Each threadgroup (32 SIMD lanes) processes one pose.
// Ligand atoms are distributed across lanes (64 atoms / 32 lanes = 2 per lane).
// Cross-attention is computed per-ligand-atom without materializing the full
// attention matrix — each lane handles its own ligand atoms sequentially.

kernel void druseAFScore(
    device DockPose                *poses          [[buffer(0)]],
    constant float3                *protPositions  [[buffer(1)]],
    device const float             *weights        [[buffer(2)]],
    constant DruseAFWeightEntry    *weightEntries  [[buffer(3)]],
    constant DruseAFParams         &afParams       [[buffer(4)]],
    device float                   *intermediates  [[buffer(5)]],
    uint                            gid            [[threadgroup_position_in_grid]],
    uint                            lid            [[thread_index_in_threadgroup]],
    uint                            simd_lane      [[thread_index_in_simdgroup]])
{
    uint poseIdx = gid;
    uint P = afParams.numProteinAtoms;
    uint L = afParams.numLigandAtoms;
    uint D = DAF_HIDDEN;

    // Per-pose intermediate data
    device float *myIntermediate = intermediates + poseIdx * DAF_INTERMEDIATE_FLOATS_PER_POSE;
    device float *prot_h = myIntermediate + DAF_PROT_H_OFFSET;
    device float *lig_h  = myIntermediate + DAF_LIG_H_OFFSET;
    device float *lig_pos_buf = myIntermediate + DAF_LIG_POS_OFFSET;

    // Read ligand positions for this pose
    float3 ligPos[DAF_MAX_LIG];
    for (uint a = 0; a < L; a++) {
        ligPos[a] = float3(lig_pos_buf[a*3], lig_pos_buf[a*3+1], lig_pos_buf[a*3+2]);
    }

    // ---- Run 2 cross-attention layers ----
    for (uint layer = 0; layer < 2; layer++) {
        uint base = (layer == 0) ? W_CA0_Q_W : W_CA1_Q_W;

        device const float *q_w    = getWeight(weights, weightEntries, base + 0);
        device const float *q_b    = getWeight(weights, weightEntries, base + 1);
        device const float *k_w    = getWeight(weights, weightEntries, base + 2);
        device const float *k_b    = getWeight(weights, weightEntries, base + 3);
        device const float *v_w    = getWeight(weights, weightEntries, base + 4);
        device const float *v_b    = getWeight(weights, weightEntries, base + 5);
        device const float *rbf_w  = getWeight(weights, weightEntries, base + 6);
        device const float *rbf_b  = getWeight(weights, weightEntries, base + 7);
        device const float *out_w  = getWeight(weights, weightEntries, base + 8);
        device const float *out_b  = getWeight(weights, weightEntries, base + 9);
        device const float *norm_w = getWeight(weights, weightEntries, base + 10);
        device const float *norm_b = getWeight(weights, weightEntries, base + 11);

        float invScale = 1.0f / sqrt(float(DAF_HEAD_DIM));

        // Each lane processes a subset of ligand atoms
        // With 32 lanes and up to 64 ligand atoms: 2 atoms per lane
        for (uint laneAtom = 0; laneAtom < 2; laneAtom++) {
            uint l = lid * 2 + laneAtom;
            if (l >= L) continue;

            // Q projection for this ligand atom: [D] -> [D]
            float Q[DAF_HIDDEN];
            device const float *lig_h_l = lig_h + l * D;
            for (uint i = 0; i < D; i++) {
                float sum = q_b[i];
                for (uint j = 0; j < D; j++) sum += q_w[i * D + j] * lig_h_l[j];
                Q[i] = sum;
            }

            // For each head, compute attention over all protein atoms
            float attended[DAF_HIDDEN]; // output of attention for this ligand atom
            for (uint i = 0; i < D; i++) attended[i] = 0.0f;

            for (uint h = 0; h < DAF_HEADS; h++) {
                // Compute attention scores for this (ligand atom, head) over all protein atoms
                float attn_scores[DAF_MAX_PROT];
                float max_score = -1e9f;

                float dist_lp = 0.0f; // will be set per protein atom

                for (uint p = 0; p < P; p++) {
                    // K projection for protein atom p, head h
                    float kh_dot = 0.0f;
                    for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                        uint ki = h * DAF_HEAD_DIM + d;
                        float k_val = k_b[ki];
                        for (uint j = 0; j < D; j++) k_val += k_w[ki * D + j] * prot_h[p * D + j];
                        kh_dot += Q[h * DAF_HEAD_DIM + d] * k_val;
                    }
                    kh_dot *= invScale;

                    // RBF distance bias (computed on-the-fly, no materialization)
                    dist_lp = distance(ligPos[l], protPositions[p]);
                    float rbf_bias = rbf_b[h];
                    for (uint r = 0; r < DAF_RBF_BINS; r++) {
                        float center = float(r) * afParams.rbfSpacing;
                        float diff = dist_lp - center;
                        float rbf_val = exp(-afParams.rbfGamma * diff * diff);
                        rbf_bias += rbf_w[h * DAF_RBF_BINS + r] * rbf_val;
                    }

                    attn_scores[p] = kh_dot + rbf_bias;
                    max_score = max(max_score, attn_scores[p]);
                }

                // Softmax over protein atoms
                float sum_exp = 0.0f;
                for (uint p = 0; p < P; p++) {
                    attn_scores[p] = exp(attn_scores[p] - max_score);
                    sum_exp += attn_scores[p];
                }
                float inv_sum = 1.0f / max(sum_exp, 1e-8f);
                for (uint p = 0; p < P; p++) {
                    attn_scores[p] *= inv_sum;
                }

                // Weighted sum of V projections
                for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                    uint vi = h * DAF_HEAD_DIM + d;
                    float val = 0.0f;
                    for (uint p = 0; p < P; p++) {
                        float v_val = v_b[vi];
                        for (uint j = 0; j < D; j++) v_val += v_w[vi * D + j] * prot_h[p * D + j];
                        val += attn_scores[p] * v_val;
                    }
                    attended[h * DAF_HEAD_DIM + d] = val;
                }
            }

            // out_proj + residual + LayerNorm
            float projected[DAF_HIDDEN];
            linearLayer(projected, attended, out_w, out_b, D, D);

            // Residual connection
            float updated[DAF_HIDDEN];
            for (uint i = 0; i < D; i++) updated[i] = lig_h_l[i] + projected[i];

            // LayerNorm
            layerNorm(updated, norm_w, norm_b, D);

            // Write back to lig_h for next layer
            for (uint i = 0; i < D; i++) lig_h[l * D + i] = updated[i];
        }

        // Sync all lanes before next cross-attention layer
        threadgroup_barrier(mem_flags::mem_device);
    }

    // ---- Gated Attention Pooling ----
    // Each lane computes gate logits for its ligand atoms, then we reduce

    device const float *gate_w0 = getWeight(weights, weightEntries, W_GATE_0_W);
    device const float *gate_b0 = getWeight(weights, weightEntries, W_GATE_0_B);
    device const float *gate_w2 = getWeight(weights, weightEntries, W_GATE_2_W);
    device const float *gate_b2 = getWeight(weights, weightEntries, W_GATE_2_B);

    // Compute gate logits for each ligand atom
    float gate_logits[2] = {-1e9f, -1e9f}; // for 2 atoms per lane
    float gate_max = -1e9f;

    for (uint laneAtom = 0; laneAtom < 2; laneAtom++) {
        uint l = lid * 2 + laneAtom;
        if (l >= L) continue;
        device const float *h = lig_h + l * D;

        // Linear(128→64) → Tanh → Linear(64→1)
        float g1[64];
        linearLayer(g1, h, gate_w0, gate_b0, 64, D);
        for (uint i = 0; i < 64; i++) g1[i] = tanh(g1[i]);
        float g2 = gate_b2[0];
        for (uint i = 0; i < 64; i++) g2 += gate_w2[i] * g1[i];

        gate_logits[laneAtom] = g2;
        gate_max = max(gate_max, g2);
    }

    // Find global max for stable softmax (SIMD reduction)
    gate_max = simd_max(gate_max);

    // Compute exp and partial sums
    float gate_exp[2] = {0.0f, 0.0f};
    float lane_sum = 0.0f;
    for (uint laneAtom = 0; laneAtom < 2; laneAtom++) {
        uint l = lid * 2 + laneAtom;
        if (l >= L) continue;
        gate_exp[laneAtom] = exp(gate_logits[laneAtom] - gate_max);
        lane_sum += gate_exp[laneAtom];
    }
    float total_exp = simd_sum(lane_sum);
    float inv_total = 1.0f / max(total_exp, 1e-8f);

    // Weighted sum: complex_repr = sum(gate_weight[l] * lig_h[l])
    float complex_repr[DAF_HIDDEN];
    for (uint i = 0; i < D; i++) complex_repr[i] = 0.0f;

    for (uint laneAtom = 0; laneAtom < 2; laneAtom++) {
        uint l = lid * 2 + laneAtom;
        if (l >= L) continue;
        float w = gate_exp[laneAtom] * inv_total;
        device const float *h = lig_h + l * D;
        for (uint i = 0; i < D; i++) complex_repr[i] += w * h[i];
    }

    // SIMD reduce complex_repr across all lanes
    for (uint i = 0; i < D; i++) {
        complex_repr[i] = simd_sum(complex_repr[i]);
    }

    // Only lane 0 computes the final heads and writes the score
    if (simd_lane != 0) return;

    // ---- Affinity Head: Linear(128→128) → SiLU → Linear(128→1) ----
    device const float *aff_w0 = getWeight(weights, weightEntries, W_AFF_0_W);
    device const float *aff_b0 = getWeight(weights, weightEntries, W_AFF_0_B);
    device const float *aff_w3 = getWeight(weights, weightEntries, W_AFF_3_W);
    device const float *aff_b3 = getWeight(weights, weightEntries, W_AFF_3_B);

    float aff_h[DAF_HIDDEN];
    linearLayer(aff_h, complex_repr, aff_w0, aff_b0, D, D);
    for (uint i = 0; i < D; i++) aff_h[i] = silu(aff_h[i]);
    float pkd = aff_b3[0];
    for (uint i = 0; i < D; i++) pkd += aff_w3[i] * aff_h[i];

    // ---- Confidence Head: Linear(128→64) → SiLU → Linear(64→1) → sigmoid ----
    device const float *conf_w0 = getWeight(weights, weightEntries, W_CONF_0_W);
    device const float *conf_b0 = getWeight(weights, weightEntries, W_CONF_0_B);
    device const float *conf_w3 = getWeight(weights, weightEntries, W_CONF_3_W);
    device const float *conf_b3 = getWeight(weights, weightEntries, W_CONF_3_B);

    float conf_h[64];
    linearLayer(conf_h, complex_repr, conf_w0, conf_b0, 64, D);
    for (uint i = 0; i < 64; i++) conf_h[i] = silu(conf_h[i]);
    float conf_logit = conf_b3[0];
    for (uint i = 0; i < 64; i++) conf_logit += conf_w3[i] * conf_h[i];
    float confidence = 1.0f / (1.0f + exp(-conf_logit));

    // ---- Final score ----
    float docking_score = pkd * confidence;

    // NaN guard
    if (isnan(docking_score) || isinf(docking_score)) {
        docking_score = 0.0f;
        pkd = 0.0f;
        confidence = 0.0f;
    }

    // Write to pose (negate: GA minimizes, but higher docking_score is better)
    float oopPenalty = poses[poseIdx].clashPenalty; // set by Kernel 1
    poses[poseIdx].energy = -docking_score + oopPenalty;
    poses[poseIdx].stericEnergy = pkd;
    poses[poseIdx].hydrophobicEnergy = confidence;
    poses[poseIdx].hbondEnergy = docking_score;
    poses[poseIdx].torsionPenalty = 0.0f;
    poses[poseIdx].drusinaCorrection = 0.0f;
    poses[poseIdx].constraintPenalty = 0.0f;
}
