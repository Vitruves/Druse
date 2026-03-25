// DruseAFCompute.metal — DruseAF neural network scoring on Metal GPU
//
// Three-kernel architecture:
//   Kernel 0 (druseAFSetup):  ONE-TIME per dock — MLP encoding + K/V projections
//   Kernel 1 (druseAFEncode): per-generation — ligand position transform only
//   Kernel 2 (druseAFScore):  per-generation — cross-attention + pooling + heads
//
// Pre-computing protein/ligand encoding and K/V projections in the setup kernel
// eliminates ~95% of redundant work from the per-generation kernels.
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
// Setup buffer layout (pre-computed once per dock, constant during GA)
// ============================================================================

#define DAF_SETUP_PROT_H  0                                                    // [P, 128]
#define DAF_SETUP_LIG_H   (DAF_MAX_PROT * DAF_HIDDEN)                        // [L, 128]
#define DAF_SETUP_K0      (DAF_SETUP_LIG_H + DAF_MAX_LIG * DAF_HIDDEN)       // [P, 128]
#define DAF_SETUP_V0      (DAF_SETUP_K0 + DAF_MAX_PROT * DAF_HIDDEN)          // [P, 128]
#define DAF_SETUP_K1      (DAF_SETUP_V0 + DAF_MAX_PROT * DAF_HIDDEN)          // [P, 128]
#define DAF_SETUP_V1      (DAF_SETUP_K1 + DAF_MAX_PROT * DAF_HIDDEN)          // [P, 128]
#define DAF_SETUP_FLOATS  (DAF_SETUP_V1 + DAF_MAX_PROT * DAF_HIDDEN)
// Total: 5*256*128 + 64*128 = 171,776 floats = 671 KB

// Per-pose intermediate: just transformed ligand positions
#define DAF_POSE_LIG_POS_FLOATS (DAF_MAX_LIG * 3)  // 192 floats per pose

// ============================================================================
// Helpers
// ============================================================================

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline device const float* getWeight(
    device const float *weights,
    constant DruseAFWeightEntry *entries,
    uint index)
{
    return weights + entries[index].offset;
}

/// Linear layer: out[outDim] = W[outDim, inDim] @ in[inDim] + b[outDim]
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

inline void linearLayerFromConst(
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

inline void linearLayerFromDevice(
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

/// Apply torsion rotations to ligand positions.
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
// Kernel 0: ONE-TIME Setup (pre-compute encodings + K/V projections)
// ============================================================================
//
// Dispatched once before the GA loop with (P + L) threads.
// Thread 0..P-1: encode protein atom, compute K0, V0, K1, V1
// Thread P..P+L-1: encode ligand atom

kernel void druseAFSetup(
    constant float                 *protFeatures   [[buffer(0)]],   // [P, 20]
    constant float                 *ligFeatures    [[buffer(1)]],   // [L, 20]
    device const float             *weights        [[buffer(2)]],
    constant DruseAFWeightEntry    *weightEntries  [[buffer(3)]],
    constant DruseAFParams         &afParams       [[buffer(4)]],
    device float                   *setupBuffer    [[buffer(5)]],   // output
    uint                            tid            [[thread_position_in_grid]])
{
    uint P = afParams.numProteinAtoms;
    uint L = afParams.numLigandAtoms;
    if (tid >= P + L) return;

    bool isProtein = tid < P;
    uint atomIdx = isProtein ? tid : (tid - P);

    if (isProtein) {
        // --- Protein atom: encode + K/V projections ---
        constant float *feat = protFeatures + atomIdx * DAF_ATOM_DIM;

        device const float *enc_w0 = getWeight(weights, weightEntries, W_PROT_ENC_0_W);
        device const float *enc_b0 = getWeight(weights, weightEntries, W_PROT_ENC_0_B);
        device const float *enc_w2 = getWeight(weights, weightEntries, W_PROT_ENC_2_W);
        device const float *enc_b2 = getWeight(weights, weightEntries, W_PROT_ENC_2_B);

        // MLP encode: Linear(20→128) → SiLU → Linear(128→128)
        float h1[DAF_HIDDEN];
        linearLayerFromConst(h1, feat, enc_w0, enc_b0, DAF_HIDDEN, DAF_ATOM_DIM);
        for (uint i = 0; i < DAF_HIDDEN; i++) h1[i] = silu(h1[i]);
        float h2[DAF_HIDDEN];
        linearLayer(h2, h1, enc_w2, enc_b2, DAF_HIDDEN, DAF_HIDDEN);

        // Store prot_h
        device float *prot_h_out = setupBuffer + DAF_SETUP_PROT_H + atomIdx * DAF_HIDDEN;
        for (uint i = 0; i < DAF_HIDDEN; i++) prot_h_out[i] = h2[i];

        // K/V projections for layer 0
        device const float *k0_w = getWeight(weights, weightEntries, W_CA0_K_W);
        device const float *k0_b = getWeight(weights, weightEntries, W_CA0_K_B);
        device const float *v0_w = getWeight(weights, weightEntries, W_CA0_V_W);
        device const float *v0_b = getWeight(weights, weightEntries, W_CA0_V_B);

        float k0[DAF_HIDDEN], v0[DAF_HIDDEN];
        linearLayer(k0, h2, k0_w, k0_b, DAF_HIDDEN, DAF_HIDDEN);
        linearLayer(v0, h2, v0_w, v0_b, DAF_HIDDEN, DAF_HIDDEN);

        device float *k0_out = setupBuffer + DAF_SETUP_K0 + atomIdx * DAF_HIDDEN;
        device float *v0_out = setupBuffer + DAF_SETUP_V0 + atomIdx * DAF_HIDDEN;
        for (uint i = 0; i < DAF_HIDDEN; i++) { k0_out[i] = k0[i]; v0_out[i] = v0[i]; }

        // K/V projections for layer 1
        device const float *k1_w = getWeight(weights, weightEntries, W_CA1_K_W);
        device const float *k1_b = getWeight(weights, weightEntries, W_CA1_K_B);
        device const float *v1_w = getWeight(weights, weightEntries, W_CA1_V_W);
        device const float *v1_b = getWeight(weights, weightEntries, W_CA1_V_B);

        float k1[DAF_HIDDEN], v1[DAF_HIDDEN];
        linearLayer(k1, h2, k1_w, k1_b, DAF_HIDDEN, DAF_HIDDEN);
        linearLayer(v1, h2, v1_w, v1_b, DAF_HIDDEN, DAF_HIDDEN);

        device float *k1_out = setupBuffer + DAF_SETUP_K1 + atomIdx * DAF_HIDDEN;
        device float *v1_out = setupBuffer + DAF_SETUP_V1 + atomIdx * DAF_HIDDEN;
        for (uint i = 0; i < DAF_HIDDEN; i++) { k1_out[i] = k1[i]; v1_out[i] = v1[i]; }

    } else {
        // --- Ligand atom: encode only ---
        constant float *feat = ligFeatures + atomIdx * DAF_ATOM_DIM;

        device const float *enc_w0 = getWeight(weights, weightEntries, W_LIG_ENC_0_W);
        device const float *enc_b0 = getWeight(weights, weightEntries, W_LIG_ENC_0_B);
        device const float *enc_w2 = getWeight(weights, weightEntries, W_LIG_ENC_2_W);
        device const float *enc_b2 = getWeight(weights, weightEntries, W_LIG_ENC_2_B);

        float h1[DAF_HIDDEN];
        linearLayerFromConst(h1, feat, enc_w0, enc_b0, DAF_HIDDEN, DAF_ATOM_DIM);
        for (uint i = 0; i < DAF_HIDDEN; i++) h1[i] = silu(h1[i]);
        float h2[DAF_HIDDEN];
        linearLayer(h2, h1, enc_w2, enc_b2, DAF_HIDDEN, DAF_HIDDEN);

        device float *lig_h_out = setupBuffer + DAF_SETUP_LIG_H + atomIdx * DAF_HIDDEN;
        for (uint i = 0; i < DAF_HIDDEN; i++) lig_h_out[i] = h2[i];
    }
}


// ============================================================================
// Kernel 1: Per-Pose Ligand Position Transform
// ============================================================================
//
// Dispatched per generation with 1 thread per pose (same as Vina perturbation).
// Transforms ligand positions and writes to per-pose intermediate buffer.

kernel void druseAFEncode(
    device DockPose                *poses          [[buffer(0)]],
    constant DockLigandAtom        *ligandAtoms    [[buffer(1)]],
    constant GAParams              &gaParams       [[buffer(2)]],
    constant TorsionEdge           *torsionEdges   [[buffer(3)]],
    constant int32_t               *movingIndices  [[buffer(4)]],
    constant DruseAFParams         &afParams       [[buffer(5)]],
    device float                   *intermediates  [[buffer(6)]],   // per-pose lig positions
    constant GridParams            &gridParams     [[buffer(7)]],
    uint                            tid            [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint L = min(gaParams.numLigandAtoms, uint(DAF_MAX_LIG));
    uint nTorsions = min(gaParams.numTorsions, 32u);

    // Transform ligand atoms (rigid body + torsions)
    float3 ligPositions[DAF_MAX_LIG];
    device const DockPose &pose = poses[tid];
    for (uint a = 0; a < L; a++) {
        ligPositions[a] = dafQuatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    dafApplyTorsions(ligPositions, L, pose, torsionEdges, movingIndices, nTorsions);

    // Write transformed positions to intermediate
    device float *myPos = intermediates + tid * DAF_POSE_LIG_POS_FLOATS;
    for (uint a = 0; a < L; a++) {
        myPos[a * 3 + 0] = ligPositions[a].x;
        myPos[a * 3 + 1] = ligPositions[a].y;
        myPos[a * 3 + 2] = ligPositions[a].z;
    }

    // Out-of-grid penalty
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
    poses[tid].clashPenalty = oopTotal * 10.0f;
}


// ============================================================================
// Kernel 2: Cross-Attention + Pooling + Heads (using pre-computed K/V)
// ============================================================================
//
// Each threadgroup (32 SIMD lanes) processes one pose.
// Pre-computed K/V from setup buffer eliminates ~90% of attention computation.
//
// Optimizations over naive implementation:
//   1. RBF pre-computation: compute rbf_bias[H] once per (l,p) pair, reused
//      across heads (eliminates 75% of exp() calls)
//   2. Online softmax: stream protein atoms without storing 256-float array
//      (eliminates register spilling / 1KB per thread)
//   3. Fused Q·K + RBF: single protein atom loop computes both

kernel void druseAFScore(
    device DockPose                *poses          [[buffer(0)]],
    constant float3                *protPositions  [[buffer(1)]],
    device const float             *weights        [[buffer(2)]],
    constant DruseAFWeightEntry    *weightEntries  [[buffer(3)]],
    constant DruseAFParams         &afParams       [[buffer(4)]],
    device float                   *intermediates  [[buffer(5)]],   // per-pose lig positions
    device const float             *setupBuffer    [[buffer(6)]],   // pre-computed data
    uint                            gid            [[threadgroup_position_in_grid]],
    uint                            lid            [[thread_index_in_threadgroup]],
    uint                            simd_lane      [[thread_index_in_simdgroup]])
{
    uint poseIdx = gid;
    uint P = afParams.numProteinAtoms;
    uint L = afParams.numLigandAtoms;
    uint D = DAF_HIDDEN;

    // Read transformed ligand positions for this pose
    device float *myPos = intermediates + poseIdx * DAF_POSE_LIG_POS_FLOATS;
    float3 ligPos[2];
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l < L) {
            ligPos[la] = float3(myPos[l*3], myPos[l*3+1], myPos[l*3+2]);
        }
    }

    // Pre-computed data pointers
    device const float *prot_h   = setupBuffer + DAF_SETUP_PROT_H;
    device const float *lig_h    = setupBuffer + DAF_SETUP_LIG_H;
    device const float *preK0    = setupBuffer + DAF_SETUP_K0;
    device const float *preV0    = setupBuffer + DAF_SETUP_V0;
    device const float *preK1    = setupBuffer + DAF_SETUP_K1;
    device const float *preV1    = setupBuffer + DAF_SETUP_V1;

    // Per-lane ligand hidden states (start from pre-computed lig_h, evolve through layers)
    float lig_state[2][DAF_HIDDEN];
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l < L) {
            for (uint i = 0; i < D; i++) lig_state[la][i] = lig_h[l * D + i];
        }
    }

    // ---- Cross-attention layers ----
    for (uint layer = 0; layer < 2; layer++) {
        uint base = (layer == 0) ? W_CA0_Q_W : W_CA1_Q_W;

        device const float *q_w    = getWeight(weights, weightEntries, base + 0);
        device const float *q_b    = getWeight(weights, weightEntries, base + 1);
        device const float *rbf_w  = getWeight(weights, weightEntries, base + 6);
        device const float *rbf_b  = getWeight(weights, weightEntries, base + 7);
        device const float *out_w  = getWeight(weights, weightEntries, base + 8);
        device const float *out_b  = getWeight(weights, weightEntries, base + 9);
        device const float *norm_w = getWeight(weights, weightEntries, base + 10);
        device const float *norm_b = getWeight(weights, weightEntries, base + 11);

        device const float *preK = (layer == 0) ? preK0 : preK1;
        device const float *preV = (layer == 0) ? preV0 : preV1;

        float invScale = 1.0f / sqrt(float(DAF_HEAD_DIM));

        for (uint la = 0; la < 2; la++) {
            uint l = lid * 2 + la;
            if (l >= L) continue;

            // Q projection from current lig_state
            float Q[DAF_HIDDEN];
            linearLayer(Q, lig_state[la], q_w, q_b, D, D);

            // === Online softmax attention (no 256-element score array) ===
            // Track per-head: running max, sum_exp, weighted V accumulator
            float hd_max[DAF_HEADS];
            float hd_sum[DAF_HEADS];
            float hd_V[DAF_HIDDEN]; // accumulated weighted V, indexed [h * HEAD_DIM + d]
            for (uint h = 0; h < DAF_HEADS; h++) {
                hd_max[h] = -1e9f;
                hd_sum[h] = 0.0f;
            }
            for (uint i = 0; i < D; i++) hd_V[i] = 0.0f;

            // Stream over protein atoms — single pass, O(1) register overhead
            for (uint p = 0; p < P; p++) {
                // Pre-compute RBF distance bias for ALL heads at once (amortized 50 exp())
                float dist_lp = distance(ligPos[la], protPositions[p]);
                float rbf_vals[DAF_RBF_BINS];
                for (uint r = 0; r < DAF_RBF_BINS; r++) {
                    float center = float(r) * afParams.rbfSpacing;
                    float diff = dist_lp - center;
                    rbf_vals[r] = exp(-afParams.rbfGamma * diff * diff);
                }

                // Per-head: Q·K + rbf_bias → online softmax update
                for (uint h = 0; h < DAF_HEADS; h++) {
                    // Q·K dot product
                    float qk = 0.0f;
                    uint hOffset = h * DAF_HEAD_DIM;
                    device const float *Kp = preK + p * D + hOffset;
                    for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                        qk += Q[hOffset + d] * Kp[d];
                    }
                    qk *= invScale;

                    // RBF bias (dot product of shared rbf_vals with per-head weights)
                    float rbf_bias = rbf_b[h];
                    device const float *rw = rbf_w + h * DAF_RBF_BINS;
                    for (uint r = 0; r < DAF_RBF_BINS; r++) {
                        rbf_bias += rw[r] * rbf_vals[r];
                    }

                    float score = qk + rbf_bias;

                    // Online softmax: update running max, rescale accumulated values
                    float old_max = hd_max[h];
                    float new_max = max(old_max, score);
                    float exp_score = exp(score - new_max);
                    float rescale = exp(old_max - new_max); // correction factor

                    // Rescale running accumulator and add new contribution
                    hd_sum[h] = hd_sum[h] * rescale + exp_score;

                    device const float *Vp = preV + p * D + hOffset;
                    for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                        hd_V[hOffset + d] = hd_V[hOffset + d] * rescale + exp_score * Vp[d];
                    }
                    hd_max[h] = new_max;
                }
            }

            // Normalize V accumulators by softmax denominators
            float attended[DAF_HIDDEN];
            for (uint h = 0; h < DAF_HEADS; h++) {
                float inv_sum = 1.0f / max(hd_sum[h], 1e-8f);
                uint hOffset = h * DAF_HEAD_DIM;
                for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                    attended[hOffset + d] = hd_V[hOffset + d] * inv_sum;
                }
            }

            // out_proj + residual + LayerNorm
            float projected[DAF_HIDDEN];
            linearLayer(projected, attended, out_w, out_b, D, D);

            float updated[DAF_HIDDEN];
            for (uint i = 0; i < D; i++) updated[i] = lig_state[la][i] + projected[i];
            layerNorm(updated, norm_w, norm_b, D);

            for (uint i = 0; i < D; i++) lig_state[la][i] = updated[i];
        }
    }

    // ---- Gated Attention Pooling ----
    device const float *gate_w0 = getWeight(weights, weightEntries, W_GATE_0_W);
    device const float *gate_b0 = getWeight(weights, weightEntries, W_GATE_0_B);
    device const float *gate_w2 = getWeight(weights, weightEntries, W_GATE_2_W);
    device const float *gate_b2 = getWeight(weights, weightEntries, W_GATE_2_B);

    float gate_logits[2] = {-1e9f, -1e9f};
    float gate_max = -1e9f;

    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;

        float g1[64];
        linearLayer(g1, lig_state[la], gate_w0, gate_b0, 64, D);
        for (uint i = 0; i < 64; i++) g1[i] = tanh(g1[i]);
        float g2 = gate_b2[0];
        for (uint i = 0; i < 64; i++) g2 += gate_w2[i] * g1[i];

        gate_logits[la] = g2;
        gate_max = max(gate_max, g2);
    }

    gate_max = simd_max(gate_max);

    float gate_exp[2] = {0.0f, 0.0f};
    float lane_sum = 0.0f;
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;
        gate_exp[la] = exp(gate_logits[la] - gate_max);
        lane_sum += gate_exp[la];
    }
    float total_exp = simd_sum(lane_sum);
    float inv_total = 1.0f / max(total_exp, 1e-8f);

    float complex_repr[DAF_HIDDEN];
    for (uint i = 0; i < D; i++) complex_repr[i] = 0.0f;

    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;
        float w = gate_exp[la] * inv_total;
        for (uint i = 0; i < D; i++) complex_repr[i] += w * lig_state[la][i];
    }

    for (uint i = 0; i < D; i++) {
        complex_repr[i] = simd_sum(complex_repr[i]);
    }

    // Only lane 0 computes final heads
    if (simd_lane != 0) return;

    // ---- Affinity Head ----
    device const float *aff_w0 = getWeight(weights, weightEntries, W_AFF_0_W);
    device const float *aff_b0 = getWeight(weights, weightEntries, W_AFF_0_B);
    device const float *aff_w3 = getWeight(weights, weightEntries, W_AFF_3_W);
    device const float *aff_b3 = getWeight(weights, weightEntries, W_AFF_3_B);

    float aff_h[DAF_HIDDEN];
    linearLayer(aff_h, complex_repr, aff_w0, aff_b0, D, D);
    for (uint i = 0; i < D; i++) aff_h[i] = silu(aff_h[i]);
    float pkd = aff_b3[0];
    for (uint i = 0; i < D; i++) pkd += aff_w3[i] * aff_h[i];

    // ---- Confidence Head ----
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

    if (isnan(docking_score) || isinf(docking_score)) {
        docking_score = 0.0f;
        pkd = 0.0f;
        confidence = 0.0f;
    }

    float oopPenalty = poses[poseIdx].clashPenalty;
    poses[poseIdx].energy = -docking_score + oopPenalty;
    poses[poseIdx].stericEnergy = pkd;
    poses[poseIdx].hydrophobicEnergy = confidence;
    poses[poseIdx].hbondEnergy = docking_score;
    poses[poseIdx].torsionPenalty = 0.0f;
    poses[poseIdx].drusinaCorrection = 0.0f;
    poses[poseIdx].constraintPenalty = 0.0f;
}

// ============================================================================
// MARK: - DruseAF Score with Attention Gradient Output
// ============================================================================
//
// Variant of druseAFScore that additionally writes per-ligand-atom attention
// gradients (attention-weighted protein centroids) for diffusion-guided docking.
// The attention gradient for each ligand atom is the attention-weighted average
// of protein atom positions across all heads and layers, minus the ligand atom's
// current position. This gives a "pull direction" toward high-affinity regions.

kernel void druseAFScoreWithGradient(
    device DockPose                *poses          [[buffer(0)]],
    constant float3                *protPositions  [[buffer(1)]],
    device const float             *weights        [[buffer(2)]],
    constant DruseAFWeightEntry    *weightEntries  [[buffer(3)]],
    constant DruseAFParams         &afParams       [[buffer(4)]],
    device float                   *intermediates  [[buffer(5)]],
    device const float             *setupBuffer    [[buffer(6)]],
    device AttentionGradient       *attnGradients  [[buffer(7)]],  // [numPoses × L]
    uint                            gid            [[threadgroup_position_in_grid]],
    uint                            lid            [[thread_index_in_threadgroup]],
    uint                            simd_lane      [[thread_index_in_simdgroup]])
{
    uint poseIdx = gid;
    uint P = afParams.numProteinAtoms;
    uint L = afParams.numLigandAtoms;
    uint D = DAF_HIDDEN;

    device float *myPos = intermediates + poseIdx * DAF_POSE_LIG_POS_FLOATS;
    float3 ligPos[2];
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l < L) {
            ligPos[la] = float3(myPos[l*3], myPos[l*3+1], myPos[l*3+2]);
        }
    }

    device const float *prot_h   = setupBuffer + DAF_SETUP_PROT_H;
    device const float *lig_h    = setupBuffer + DAF_SETUP_LIG_H;
    device const float *preK0    = setupBuffer + DAF_SETUP_K0;
    device const float *preV0    = setupBuffer + DAF_SETUP_V0;
    device const float *preK1    = setupBuffer + DAF_SETUP_K1;
    device const float *preV1    = setupBuffer + DAF_SETUP_V1;

    float lig_state[2][DAF_HIDDEN];
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l < L) {
            for (uint i = 0; i < D; i++) lig_state[la][i] = lig_h[l * D + i];
        }
    }

    // Accumulate attention-weighted protein centroids per ligand atom
    float3 attnWeightedPos[2] = {float3(0.0f), float3(0.0f)};
    float attnWeightSum[2] = {0.0f, 0.0f};

    // Cross-attention layers (same as druseAFScore + attention centroid accumulation)
    for (uint layer = 0; layer < 2; layer++) {
        uint base = (layer == 0) ? W_CA0_Q_W : W_CA1_Q_W;

        device const float *q_w    = getWeight(weights, weightEntries, base + 0);
        device const float *q_b    = getWeight(weights, weightEntries, base + 1);
        device const float *rbf_w  = getWeight(weights, weightEntries, base + 6);
        device const float *rbf_b  = getWeight(weights, weightEntries, base + 7);
        device const float *out_w  = getWeight(weights, weightEntries, base + 8);
        device const float *out_b  = getWeight(weights, weightEntries, base + 9);
        device const float *norm_w = getWeight(weights, weightEntries, base + 10);
        device const float *norm_b = getWeight(weights, weightEntries, base + 11);

        device const float *preK = (layer == 0) ? preK0 : preK1;
        device const float *preV = (layer == 0) ? preV0 : preV1;

        float invScale = 1.0f / sqrt(float(DAF_HEAD_DIM));

        for (uint la = 0; la < 2; la++) {
            uint l = lid * 2 + la;
            if (l >= L) continue;

            float Q[DAF_HIDDEN];
            linearLayer(Q, lig_state[la], q_w, q_b, D, D);

            // Online softmax + centroid accumulation
            float hd_max[DAF_HEADS];
            float hd_sum[DAF_HEADS];
            float hd_V[DAF_HIDDEN];
            // Per-head centroid accumulators (for gradient output)
            float3 hd_centroid[DAF_HEADS];
            float hd_centroid_sum[DAF_HEADS];
            for (uint h = 0; h < DAF_HEADS; h++) {
                hd_max[h] = -1e9f;
                hd_sum[h] = 0.0f;
                hd_centroid[h] = float3(0.0f);
                hd_centroid_sum[h] = 0.0f;
            }
            for (uint i = 0; i < D; i++) hd_V[i] = 0.0f;

            for (uint p = 0; p < P; p++) {
                // Pre-compute RBF once per protein atom
                float dist_lp = distance(ligPos[la], protPositions[p]);
                float rbf_vals[DAF_RBF_BINS];
                for (uint r = 0; r < DAF_RBF_BINS; r++) {
                    float center = float(r) * afParams.rbfSpacing;
                    float diff = dist_lp - center;
                    rbf_vals[r] = exp(-afParams.rbfGamma * diff * diff);
                }

                for (uint h = 0; h < DAF_HEADS; h++) {
                    float qk = 0.0f;
                    uint hOffset = h * DAF_HEAD_DIM;
                    device const float *Kp = preK + p * D + hOffset;
                    for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                        qk += Q[hOffset + d] * Kp[d];
                    }
                    qk *= invScale;

                    float rbf_bias = rbf_b[h];
                    device const float *rw = rbf_w + h * DAF_RBF_BINS;
                    for (uint r = 0; r < DAF_RBF_BINS; r++) {
                        rbf_bias += rw[r] * rbf_vals[r];
                    }

                    float score = qk + rbf_bias;
                    float old_max = hd_max[h];
                    float new_max = max(old_max, score);
                    float exp_score = exp(score - new_max);
                    float rescale = exp(old_max - new_max);

                    hd_sum[h] = hd_sum[h] * rescale + exp_score;

                    device const float *Vp = preV + p * D + hOffset;
                    for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                        hd_V[hOffset + d] = hd_V[hOffset + d] * rescale + exp_score * Vp[d];
                    }

                    // Rescale centroid accumulator and add new contribution
                    hd_centroid[h] = hd_centroid[h] * rescale + exp_score * protPositions[p];
                    hd_centroid_sum[h] = hd_centroid_sum[h] * rescale + exp_score;

                    hd_max[h] = new_max;
                }
            }

            // Normalize V and accumulate attention centroids
            float attended[DAF_HIDDEN];
            for (uint h = 0; h < DAF_HEADS; h++) {
                float inv_sum = 1.0f / max(hd_sum[h], 1e-8f);
                uint hOffset = h * DAF_HEAD_DIM;
                for (uint d = 0; d < DAF_HEAD_DIM; d++) {
                    attended[hOffset + d] = hd_V[hOffset + d] * inv_sum;
                }
                // Accumulate attention centroid across heads/layers
                if (hd_centroid_sum[h] > 1e-8f) {
                    attnWeightedPos[la] += hd_centroid[h] / hd_centroid_sum[h];
                    attnWeightSum[la] += 1.0f;
                }
            }

            float projected[DAF_HIDDEN];
            linearLayer(projected, attended, out_w, out_b, D, D);

            float updated[DAF_HIDDEN];
            for (uint i = 0; i < D; i++) updated[i] = lig_state[la][i] + projected[i];
            layerNorm(updated, norm_w, norm_b, D);

            for (uint i = 0; i < D; i++) lig_state[la][i] = updated[i];
        }
    }

    // Write attention gradients: pull direction = attention_centroid - ligand_atom_position
    uint gradBase = poseIdx * L;
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;

        float3 centroid = float3(0.0f);
        if (attnWeightSum[la] > 1e-8f) {
            centroid = attnWeightedPos[la] / attnWeightSum[la];
        }
        float3 pull = centroid - ligPos[la];
        float pullMag = length(pull);

        device AttentionGradient &ag = attnGradients[gradBase + l];
        ag.pullDirection = (pullMag > 1e-6f) ? (pull / pullMag) : float3(0.0f);
        ag.pullMagnitude = pullMag;
    }

    // ---- Gated Attention Pooling (same as druseAFScore) ----
    device const float *gate_w0 = getWeight(weights, weightEntries, W_GATE_0_W);
    device const float *gate_b0 = getWeight(weights, weightEntries, W_GATE_0_B);
    device const float *gate_w2 = getWeight(weights, weightEntries, W_GATE_2_W);
    device const float *gate_b2 = getWeight(weights, weightEntries, W_GATE_2_B);

    float gate_logits[2] = {-1e9f, -1e9f};
    float gate_max = -1e9f;
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;
        float g1[64];
        linearLayer(g1, lig_state[la], gate_w0, gate_b0, 64, D);
        for (uint i = 0; i < 64; i++) g1[i] = tanh(g1[i]);
        float g2 = gate_b2[0];
        for (uint i = 0; i < 64; i++) g2 += gate_w2[i] * g1[i];
        gate_logits[la] = g2;
        gate_max = max(gate_max, g2);
    }
    gate_max = simd_max(gate_max);

    float gate_exp[2] = {0.0f, 0.0f};
    float lane_sum = 0.0f;
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;
        gate_exp[la] = exp(gate_logits[la] - gate_max);
        lane_sum += gate_exp[la];
    }
    float total_exp = simd_sum(lane_sum);
    float inv_total = 1.0f / max(total_exp, 1e-8f);

    float complex_repr[DAF_HIDDEN];
    for (uint i = 0; i < D; i++) complex_repr[i] = 0.0f;
    for (uint la = 0; la < 2; la++) {
        uint l = lid * 2 + la;
        if (l >= L) continue;
        float w = gate_exp[la] * inv_total;
        for (uint i = 0; i < D; i++) complex_repr[i] += w * lig_state[la][i];
    }
    for (uint i = 0; i < D; i++) complex_repr[i] = simd_sum(complex_repr[i]);

    if (simd_lane != 0) return;

    // ---- Affinity + Confidence Heads ----
    device const float *aff_w0 = getWeight(weights, weightEntries, W_AFF_0_W);
    device const float *aff_b0 = getWeight(weights, weightEntries, W_AFF_0_B);
    device const float *aff_w3 = getWeight(weights, weightEntries, W_AFF_3_W);
    device const float *aff_b3 = getWeight(weights, weightEntries, W_AFF_3_B);

    float aff_h[DAF_HIDDEN];
    linearLayer(aff_h, complex_repr, aff_w0, aff_b0, D, D);
    for (uint i = 0; i < D; i++) aff_h[i] = silu(aff_h[i]);
    float pkd = aff_b3[0];
    for (uint i = 0; i < D; i++) pkd += aff_w3[i] * aff_h[i];

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

    float docking_score = pkd * confidence;
    if (isnan(docking_score) || isinf(docking_score)) {
        docking_score = 0.0f; pkd = 0.0f; confidence = 0.0f;
    }

    float oopPenalty = poses[poseIdx].clashPenalty;
    poses[poseIdx].energy = -docking_score + oopPenalty;
    poses[poseIdx].stericEnergy = pkd;
    poses[poseIdx].hydrophobicEnergy = confidence;
    poses[poseIdx].hbondEnergy = docking_score;
    poses[poseIdx].torsionPenalty = 0.0f;
    poses[poseIdx].drusinaCorrection = 0.0f;
    poses[poseIdx].constraintPenalty = 0.0f;
}
