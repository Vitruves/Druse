// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// PIGNet2Compute.metal — Physics-Informed Graph Neural Network scoring on Metal GPU
//
// Three-kernel architecture (mirrors DruseAF pattern):
//   Kernel 0 (pignet2Setup):  ONE-TIME per dock — embed protein + 3× GatedGAT intra-conv
//   Kernel 1 (pignet2Encode): per-generation — ligand position transform
//   Kernel 2 (pignet2Score):  per-generation — embed lig + lig GatedGAT + build inter edges
//                             + 3× InteractionNet + physics energy decomposition
//
// Architecture: feat_dim=47, hidden=128, 3× GatedGAT + 3× InteractionNet + physics head
// Score output: sum(E_vdw + E_hbond + E_metal + E_hydrophobic) / rotor_penalty
//
// Reference: PIGNet2 (Phys.-Informed GNN, Moon et al. 2022)

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// ============================================================================
// Weight tensor indices (must match export_pignet2_weights.py EXPORT_WEIGHT_ORDER)
// ============================================================================

// Scalar physics coefficients
#define PW_HBOND_COEFF        0   // [1]
#define PW_HYDROPHOBIC_COEFF  1   // [1]
#define PW_ROTOR_COEFF        2   // [1]
#define PW_METAL_COEFF        3   // [1]
#define PW_IONIC_COEFF        4   // [1] (unused)

// Embedding
#define PW_EMBED_W            5   // [128, 47]

// Intra GatedGAT layer 0
#define PW_GAT0_W2            6   // [128, 128]
#define PW_GAT0_W1_W          7   // [128, 128]
#define PW_GAT0_W1_B          8   // [128]
#define PW_GAT0_GATE_W        9   // [1, 256]
#define PW_GAT0_GATE_B       10   // [1]

// Intra GatedGAT layer 1
#define PW_GAT1_W2           11
#define PW_GAT1_W1_W         12
#define PW_GAT1_W1_B         13
#define PW_GAT1_GATE_W       14
#define PW_GAT1_GATE_B       15

// Intra GatedGAT layer 2
#define PW_GAT2_W2           16
#define PW_GAT2_W1_W         17
#define PW_GAT2_W1_B         18
#define PW_GAT2_GATE_W       19
#define PW_GAT2_GATE_B       20

// Inter InteractionNet layer 0
#define PW_INT0_W1_W         21   // [128, 128]
#define PW_INT0_W1_B         22   // [128]
#define PW_INT0_W2_W         23   // [128, 128]
#define PW_INT0_W2_B         24   // [128]
#define PW_INT0_RNN_WIH      25   // [384, 128]  GRU weight_ih
#define PW_INT0_RNN_WHH      26   // [384, 128]  GRU weight_hh
#define PW_INT0_RNN_BIH      27   // [384]       GRU bias_ih
#define PW_INT0_RNN_BHH      28   // [384]       GRU bias_hh

// Inter InteractionNet layer 1
#define PW_INT1_W1_W         29
#define PW_INT1_W1_B         30
#define PW_INT1_W2_W         31
#define PW_INT1_W2_B         32
#define PW_INT1_RNN_WIH      33
#define PW_INT1_RNN_WHH      34
#define PW_INT1_RNN_BIH      35
#define PW_INT1_RNN_BHH      36

// Inter InteractionNet layer 2
#define PW_INT2_W1_W         37
#define PW_INT2_W1_B         38
#define PW_INT2_W2_W         39
#define PW_INT2_W2_B         40
#define PW_INT2_RNN_WIH      41
#define PW_INT2_RNN_WHH      42
#define PW_INT2_RNN_BIH      43
#define PW_INT2_RNN_BHH      44

// nn_vdw_epsilon: Linear(256,128) → ReLU → Linear(128,1) → Sigmoid
#define PW_EPS_L0_W          45   // [128, 256]
#define PW_EPS_L0_B          46   // [128]
#define PW_EPS_L2_W          47   // [1, 128]
#define PW_EPS_L2_B          48   // [1]

// nn_dvdw: Linear(256,128) → ReLU → Linear(128,1) → Tanh (dev_vdw_radii_coeff=0.0 in Morse)
#define PW_DVDW_L0_W         49   // [128, 256]
#define PW_DVDW_L0_B         50   // [128]
#define PW_DVDW_L2_W         51   // [1, 128]
#define PW_DVDW_L2_B         52   // [1]

// nn_vdw_width (Morse): Linear(256,128) → ReLU → Linear(128,1) → Sigmoid → scale [1.0, 2.0]
#define PW_WIDTH_L0_W        53   // [128, 256]
#define PW_WIDTH_L0_B        54   // [128]
#define PW_WIDTH_L2_W        55   // [1, 128]
#define PW_WIDTH_L2_B        56   // [1]

// nn_vdw_radius (Morse): Linear(256,128) → ReLU → Linear(128,1) → ReLU
#define PW_RAD_L0_W          57   // [128, 256]
#define PW_RAD_L0_B          58   // [128]
#define PW_RAD_L2_W          59   // [1, 128]
#define PW_RAD_L2_B          60   // [1]

// Setup buffer layout: protein embeddings after 3x intra-GatedGAT.
// InteractionNet projections are recomputed in the score kernel because protein
// embeddings change after each interaction layer.
#define PIG_SETUP_SECTIONS 1u

// Per-pose intermediate: 64 ligand atoms × 3 floats = 192 floats
#define PIG_POSE_LIG_POS_FLOATS (PIG_MAX_LIG * 3u)

// Atom aux flag bits
#define PIG_FLAG_METAL       0x1u
#define PIG_FLAG_H_DONOR     0x2u
#define PIG_FLAG_H_ACCEPTOR  0x4u
#define PIG_FLAG_HYDROPHOBIC 0x8u

// ============================================================================
// Helper: weight access
// ============================================================================

inline device const float* pigGetWeight(
    device const float *weights,
    constant DruseAFWeightEntry *entries,
    uint index)
{
    return weights + entries[index].offset;
}

// ============================================================================
// Helper: Linear layer (thread → thread)
// ============================================================================

// Inlined: this is the hot inner loop for the InteractionNet phase. With D=128
// constexpr at the call site the compiler unrolls and vectorises the matmul;
// without inlining each call becomes a real function call with no FMA fusion.
inline void pigLinear(
    thread float *out,
    thread const float *in_ptr,
    device const float *W,
    device const float *b,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = b ? b[i] : 0.0f;
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

// Linear without bias - inlined for the same hot-loop reason as `pigLinear`.
inline void pigLinearNoBias(
    thread float *out,
    thread const float *in_ptr,
    device const float *W,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = 0.0f;
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

// Linear from device memory input - inlined (hot-loop helper).
inline void pigLinearFromDevice(
    thread float *out,
    device const float *in_ptr,
    device const float *W,
    device const float *b,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = b ? b[i] : 0.0f;
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

// Linear without bias from device memory
inline void pigLinearNoBiasFromDevice(
    thread float *out,
    device const float *in_ptr,
    device const float *W,
    uint outDim, uint inDim)
{
    for (uint i = 0; i < outDim; i++) {
        float sum = 0.0f;
        for (uint j = 0; j < inDim; j++) {
            sum += W[i * inDim + j] * in_ptr[j];
        }
        out[i] = sum;
    }
}

// ============================================================================
// Helper: GatedGAT layer for a single atom
// ============================================================================
//
// GatedGAT message passing on CSR-format edges.
//   x_updated[i] = z * x[i] + (1-z) * relu(weighted_sum)
//   where z = sigmoid(gate([x[i], x_prime[i]]))
//   x_prime[i] = sum_j( softmax(attn_score_ij) * Wx_j )
//   attn_score_ij = (Wx_i^T * W2 * Wx_j) + (Wx_j^T * W2 * Wx_i)  (symmetrized)
//
// We operate on one atom at a time. The caller must provide the full node
// embeddings array (in device mem) and the CSR-format neighbor list.

inline void pigGatedGATAtom(
    thread float *x_out,           // [PIG_DIM] output for this atom
    thread const float *x_self,    // [PIG_DIM] current embedding for this atom
    device const float *allNodes,  // [N, PIG_DIM] all node embeddings
    uint numNodes,
    constant PIGNet2Edge *edges,
    uint numEdges,
    uint atomIdx,
    device const float *W1_w,     // [128, 128]
    device const float *W1_b,     // [128]
    device const float *W2,       // [128, 128] (no bias, symmetric attention matrix)
    device const float *gate_w,   // [1, 256]
    device const float *gate_b)   // [1]
{
    constexpr uint D = PIG_DIM;

    // Compute Wx_i = W1 @ x_self
    float Wx_self[PIG_DIM];
    pigLinear(Wx_self, x_self, W1_w, W1_b, D, D);

    // W2 * Wx_i is reused for every neighbor in the symmetrized attention term.
    float W2_Wxi[PIG_DIM];
    for (uint d = 0; d < D; d++) {
        float sum = 0.0f;
        for (uint k = 0; k < D; k++) {
            sum += W2[d * D + k] * Wx_self[k];
        }
        W2_Wxi[d] = sum;
    }

    // Collect neighbors: iterate edges where src or dst == atomIdx
    // Due to self-loops + bidirectional edges, we scan all edges that touch atomIdx.
    // For efficiency, only process edges where dst == atomIdx (target convention from PyG).
    // Self-loop is added implicitly.

    // First pass: compute attention scores for all neighbors + self
    float attn_scores[64]; // max neighbors (realistically ~10-20 for covalent bonds + self)
    uint neighbor_indices[64];
    float Wx_neighbors[64][PIG_DIM]; // Wx for each neighbor — stored for reuse
    uint n_nbrs = 0;

    // Add self-loop
    neighbor_indices[n_nbrs] = atomIdx;
    for (uint d = 0; d < D; d++) Wx_neighbors[n_nbrs][d] = Wx_self[d];
    n_nbrs++;

    // Scan edges for neighbors of atomIdx (dst == atomIdx, meaning source sends message to us)
    for (uint e = 0; e < numEdges && n_nbrs < 63; e++) {
        if (edges[e].dst == atomIdx) {
            uint src = edges[e].src;
            neighbor_indices[n_nbrs] = src;
            // Compute Wx_j for this neighbor
            device const float *x_j = allNodes + src * D;
            for (uint d = 0; d < D; d++) {
                float sum = W1_b[d];
                for (uint k = 0; k < D; k++) {
                    sum += W1_w[d * D + k] * x_j[k];
                }
                Wx_neighbors[n_nbrs][d] = sum;
            }
            n_nbrs++;
        }
    }

    // Compute attention scores: E_ij = Wx_i^T * W2 * Wx_j (symmetrized)
    for (uint n = 0; n < n_nbrs; n++) {
        // Compute W2 * Wx_j
        float W2_Wxj[PIG_DIM];
        for (uint d = 0; d < D; d++) {
            float sum = 0.0f;
            for (uint k = 0; k < D; k++) {
                sum += W2[d * D + k] * Wx_neighbors[n][k];
            }
            W2_Wxj[d] = sum;
        }
        // E_ij = dot(Wx_i, W2*Wx_j)
        float e_ij = 0.0f;
        for (uint d = 0; d < D; d++) {
            e_ij += Wx_self[d] * W2_Wxj[d];
        }
        // Symmetrize: + dot(Wx_j, W2*Wx_i)
        float e_ji = 0.0f;
        for (uint d = 0; d < D; d++) {
            e_ji += Wx_neighbors[n][d] * W2_Wxi[d];
        }
        attn_scores[n] = e_ij + e_ji;
    }

    // Softmax over neighbor attention scores
    float max_score = -1e30f;
    for (uint n = 0; n < n_nbrs; n++) max_score = max(max_score, attn_scores[n]);
    float sum_exp = 0.0f;
    for (uint n = 0; n < n_nbrs; n++) {
        attn_scores[n] = exp(attn_scores[n] - max_score);
        sum_exp += attn_scores[n];
    }
    float inv_sum = 1.0f / max(sum_exp, 1e-8f);

    // Weighted sum of neighbor Wx
    float x_prime[PIG_DIM];
    for (uint d = 0; d < D; d++) x_prime[d] = 0.0f;
    for (uint n = 0; n < n_nbrs; n++) {
        float w = attn_scores[n] * inv_sum;
        for (uint d = 0; d < D; d++) {
            x_prime[d] += w * Wx_neighbors[n][d];
        }
    }

    // ReLU
    for (uint d = 0; d < D; d++) x_prime[d] = max(x_prime[d], 0.0f);

    // Gate: z = sigmoid(gate_w @ [x_self, x_prime] + gate_b)
    float gate_input[PIG_DIM * 2];
    for (uint d = 0; d < D; d++) gate_input[d] = x_self[d];
    for (uint d = 0; d < D; d++) gate_input[D + d] = x_prime[d];

    float z_val = gate_b[0];
    for (uint d = 0; d < D * 2; d++) {
        z_val += gate_w[d] * gate_input[d];
    }
    float z = 1.0f / (1.0f + exp(-z_val));

    // Output: z * x_self + (1-z) * x_prime
    for (uint d = 0; d < D; d++) {
        x_out[d] = z * x_self[d] + (1.0f - z) * x_prime[d];
    }
}

// ============================================================================
// Helper: GRU cell
// ============================================================================
//
// GRUCell(input_size=128, hidden_size=128):
//   r = sigmoid(W_ir @ input + b_ir + W_hr @ hidden + b_hr)
//   z = sigmoid(W_iz @ input + b_iz + W_hz @ hidden + b_hz)
//   n = tanh(W_in @ input + b_in + r * (W_hn @ hidden + b_hn))
//   h' = (1 - z) * n + z * hidden
//
// weight_ih is [3*hidden, input] = [384, 128]: rows 0..127 = W_ir, 128..255 = W_iz, 256..383 = W_in
// weight_hh is [3*hidden, hidden] = [384, 128]: same layout
// bias_ih is [384]: same layout
// bias_hh is [384]: same layout

// Inlined: called per-atom-per-InteractionNet-layer in the score kernel - keeping
// it out-of-line was costing seconds per pose due to function-call overhead and
// loss of constexpr unrolling on the 128-dim matvecs inside.
inline void pigGRUCell(
    thread float *h_out,          // [128] new hidden state
    thread const float *input,    // [128]
    thread const float *hidden,   // [128]
    device const float *W_ih,     // [384, 128]
    device const float *W_hh,     // [384, 128]
    device const float *b_ih,     // [384]
    device const float *b_hh)     // [384]
{
    constexpr uint D = PIG_DIM;

    // Compute all gates from W_ih @ input + b_ih and W_hh @ hidden + b_hh
    float ih[384], hh[384];
    for (uint i = 0; i < 384; i++) {
        float si = b_ih[i];
        float sh = b_hh[i];
        for (uint j = 0; j < D; j++) {
            si += W_ih[i * D + j] * input[j];
            sh += W_hh[i * D + j] * hidden[j];
        }
        ih[i] = si;
        hh[i] = sh;
    }

    // r = sigmoid(ih[0..127] + hh[0..127])
    // z = sigmoid(ih[128..255] + hh[128..255])
    // n = tanh(ih[256..383] + r * hh[256..383])
    for (uint d = 0; d < D; d++) {
        float r = 1.0f / (1.0f + exp(-(ih[d] + hh[d])));
        float z = 1.0f / (1.0f + exp(-(ih[D + d] + hh[D + d])));
        float n = tanh(ih[2 * D + d] + r * hh[2 * D + d]);
        h_out[d] = (1.0f - z) * n + z * hidden[d];
    }
}

// ============================================================================
// Helper: Morse potential (PIGNet2-Morse variant)
// ============================================================================
//
// E = epsilon * ((1 - exp(-A*(D-R)))^2 - 1)
// With short-range variant: if D <= R, use short_range_A instead of A.
// Clamped to PIG_VDW_CLAMP.

inline float pigMorsePotential(float D, float R, float epsilon, float A) {
    float use_A = (D > R) ? A : PIG_SHORT_RANGE_A;
    float x = 1.0f - exp(-use_A * (D - R));
    float energy = epsilon * (x * x - 1.0f);
    return min(energy, PIG_VDW_CLAMP);
}

// ============================================================================
// Helper: Linear potential (H-bond, metal, hydrophobic)
// ============================================================================

inline float pigLinearPotential(float D, float R, float minima, float c1, float c2) {
    float e = (D - R - c2) / (c1 - c2);
    e = clamp(e, 0.0f, 1.0f);
    return e * minima;
}

// ============================================================================
// Helper: Quaternion rotation (same as DruseAF)
// ============================================================================

inline float3 pigQuatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

// ============================================================================
// Helper: Apply torsions (same as DruseAF)
// ============================================================================

inline void pigApplyTorsions(
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
// Kernel 0: ONE-TIME Setup — embed protein + 3× GatedGAT intra-conv
// ============================================================================
//
// Dispatched once with P threads. Each thread processes one protein atom through
// embedding + 3 GatedGAT layers (with threadgroup barriers between layers).
//
// Output setup buffer layout (all in floats):
//   [0 .. P*128) : protein embeddings after 3x GatedGAT

kernel void pignet2Setup(
    constant float                 *protFeatures   [[buffer(0)]],   // [P, 47]
    device const float             *weights        [[buffer(1)]],
    constant DruseAFWeightEntry    *weightEntries  [[buffer(2)]],
    constant PIGNet2Params         &params         [[buffer(3)]],
    device float                   *setupBuffer    [[buffer(4)]],   // output
    constant PIGNet2Edge           *protEdges      [[buffer(5)]],   // CSR protein edges
    device float                   *scratchBuffer  [[buffer(6)]],   // [P, 128] scratch for GatedGAT
    uint                            tid            [[thread_position_in_grid]],
    uint                            tgSize         [[threads_per_threadgroup]])
{
    uint P = params.numProteinAtoms;
    if (tid >= P) return;

    constexpr uint D = PIG_DIM;
    uint numEdges = params.numProtIntraEdges;

    // Embed: Linear(47→128, no bias)
    device const float *embed_w = pigGetWeight(weights, weightEntries, PW_EMBED_W);
    float x[PIG_DIM];
    {
        constant float *feat = protFeatures + tid * PIG_FEAT_DIM;
        for (uint i = 0; i < D; i++) {
            float sum = 0.0f;
            for (uint j = 0; j < PIG_FEAT_DIM; j++) {
                sum += embed_w[i * PIG_FEAT_DIM + j] * feat[j];
            }
            x[i] = sum;
        }
    }

    // Store initial embedding in setupBuffer (section 0)
    device float *mySlot = setupBuffer + tid * D;
    for (uint d = 0; d < D; d++) mySlot[d] = x[d];

    // 3× GatedGAT layers using setupBuffer as shared node storage
    // We use a ping-pong between setupBuffer and scratchBuffer
    uint gatWeightIdx[5 * 3] = {
        PW_GAT0_W2, PW_GAT0_W1_W, PW_GAT0_W1_B, PW_GAT0_GATE_W, PW_GAT0_GATE_B,
        PW_GAT1_W2, PW_GAT1_W1_W, PW_GAT1_W1_B, PW_GAT1_GATE_W, PW_GAT1_GATE_B,
        PW_GAT2_W2, PW_GAT2_W1_W, PW_GAT2_W1_B, PW_GAT2_GATE_W, PW_GAT2_GATE_B,
    };

    for (uint layer = 0; layer < PIG_N_GNN; layer++) {
        // Barrier: wait for all atoms to finish writing current layer embeddings
        threadgroup_barrier(mem_flags::mem_device);

        // Read current embedding from setupBuffer
        device const float *curNodes = setupBuffer;
        device float *outNodes = scratchBuffer;

        device const float *W2_ptr = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 0]);
        device const float *W1_w   = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 1]);
        device const float *W1_b   = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 2]);
        device const float *gate_w = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 3]);
        device const float *gate_b = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 4]);

        // Load own embedding from device
        float x_self[PIG_DIM];
        device const float *myNode = curNodes + tid * D;
        for (uint d = 0; d < D; d++) x_self[d] = myNode[d];

        // GatedGAT for this atom
        float x_new[PIG_DIM];
        pigGatedGATAtom(x_new, x_self, curNodes, P, protEdges, numEdges,
                        tid, W1_w, W1_b, W2_ptr, gate_w, gate_b);

        // Write to scratch
        device float *myOut = outNodes + tid * D;
        for (uint d = 0; d < D; d++) myOut[d] = x_new[d];

        threadgroup_barrier(mem_flags::mem_device);

        // Copy scratch back to setupBuffer for next layer
        for (uint d = 0; d < D; d++) {
            setupBuffer[tid * D + d] = scratchBuffer[tid * D + d];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);
}


// ============================================================================
// Kernel 1: Per-Pose Ligand Position Transform
// ============================================================================

kernel void pignet2Encode(
    device DockPose                *poses          [[buffer(0)]],
    constant DockLigandAtom        *ligandAtoms    [[buffer(1)]],
    constant GAParams              &gaParams       [[buffer(2)]],
    constant TorsionEdge           *torsionEdges   [[buffer(3)]],
    constant int32_t               *movingIndices  [[buffer(4)]],
    constant PIGNet2Params         &pigParams      [[buffer(5)]],
    device float                   *intermediates  [[buffer(6)]],
    constant GridParams            &gridParams     [[buffer(7)]],
    uint                            tid            [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint L = min(gaParams.numLigandAtoms, uint(PIG_MAX_LIG));
    uint nTorsions = min(gaParams.numTorsions, 32u);

    // Transform ligand atoms (rigid body + torsions)
    float3 ligPositions[PIG_MAX_LIG];
    device const DockPose &pose = poses[tid];
    for (uint a = 0; a < L; a++) {
        ligPositions[a] = pigQuatRotate(pose.rotation, ligandAtoms[a].position) + pose.translation;
    }
    pigApplyTorsions(ligPositions, L, pose, torsionEdges, movingIndices, nTorsions);

    // Write transformed positions
    device float *myPos = intermediates + tid * PIG_POSE_LIG_POS_FLOATS;
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
// Kernel 2: Per-Pose PIGNet2 Scoring
// ============================================================================
//
// 1 thread per pose (no SIMD cooperation needed given small atom counts).
// Performs: embed ligand → 3× GatedGAT on ligand → build inter edges →
//           3× InteractionNet → physics energy decomposition → rotor penalty.

kernel void pignet2Score(
    device DockPose                *poses           [[buffer(0)]],
    constant float                 *protPositions   [[buffer(1)]],   // [P, 3] (float3 packed)
    device const float             *weights         [[buffer(2)]],
    constant DruseAFWeightEntry    *weightEntries   [[buffer(3)]],
    constant PIGNet2Params         &params          [[buffer(4)]],
    device const float             *intermediates   [[buffer(5)]],   // per-pose lig positions
    device const float             *setupBuffer     [[buffer(6)]],   // protein embeddings + caches
    constant float                 *ligFeatures     [[buffer(7)]],   // [L, 47]
    constant PIGNet2AtomAux        *protAux         [[buffer(8)]],   // [P] protein atom aux
    constant PIGNet2AtomAux        *ligAux          [[buffer(9)]],   // [L] ligand atom aux
    constant PIGNet2Edge           *ligEdges        [[buffer(10)]],  // ligand intra edges
    constant GAParams              &gaParams        [[buffer(11)]],
    // Per-pose scratch for InteractionNet protein embeddings.
    // Layout: [popSize, 2, numProteinAtoms, PIG_DIM] — section 0 = current (h), section 1 = next (new).
    // Each pose's slice is 2 * numProteinAtoms * PIG_DIM floats. Moved off-stack so the
    // Metal compiler doesn't choke on a 262 KB per-thread frame
    // (XPC_ERROR_CONNECTION_INTERRUPTED).
    device float                   *protScratch     [[buffer(12)]],
    uint                            tid             [[thread_position_in_grid]])
{
    if (tid >= gaParams.populationSize) return;

    uint P = params.numProteinAtoms;
    uint L = min(gaParams.numLigandAtoms, uint(PIG_MAX_LIG));
    uint numLigEdges = params.numLigIntraEdges;
    constexpr uint D = PIG_DIM;

    // -------------------------------------------------------------------
    // Phase A: Embed ligand atoms + 3× GatedGAT on ligand intramolecular edges
    // -------------------------------------------------------------------

    // We store ligand embeddings in a local array (max 64 × 128 = 8192 floats = 32KB)
    // This is large but within Apple Silicon thread limits for a single-thread kernel.
    float lig_h[PIG_MAX_LIG][PIG_DIM];
    float lig_scratch[PIG_MAX_LIG][PIG_DIM];

    // Embed: Linear(47→128, no bias)
    device const float *embed_w = pigGetWeight(weights, weightEntries, PW_EMBED_W);
    for (uint a = 0; a < L; a++) {
        constant float *feat = ligFeatures + a * PIG_FEAT_DIM;
        for (uint d = 0; d < D; d++) {
            float sum = 0.0f;
            for (uint j = 0; j < PIG_FEAT_DIM; j++) {
                sum += embed_w[d * PIG_FEAT_DIM + j] * feat[j];
            }
            lig_h[a][d] = sum;
        }
    }

    // 3× GatedGAT on ligand intramolecular edges
    // For ligand we keep embeddings in thread-local arrays.
    // Self-loops are included, scanning edges where dst == atomIdx.
    uint gatWeightIdx[5 * 3] = {
        PW_GAT0_W2, PW_GAT0_W1_W, PW_GAT0_W1_B, PW_GAT0_GATE_W, PW_GAT0_GATE_B,
        PW_GAT1_W2, PW_GAT1_W1_W, PW_GAT1_W1_B, PW_GAT1_GATE_W, PW_GAT1_GATE_B,
        PW_GAT2_W2, PW_GAT2_W1_W, PW_GAT2_W1_B, PW_GAT2_GATE_W, PW_GAT2_GATE_B,
    };

    for (uint layer = 0; layer < PIG_N_GNN; layer++) {
        device const float *W2_ptr = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 0]);
        device const float *W1_w   = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 1]);
        device const float *W1_b   = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 2]);
        device const float *gate_w = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 3]);
        device const float *gate_b = pigGetWeight(weights, weightEntries, gatWeightIdx[layer * 5 + 4]);

        for (uint a = 0; a < L; a++) {
            // Compute Wx_self
            float Wx_self[PIG_DIM];
            pigLinear(Wx_self, lig_h[a], W1_w, W1_b, D, D);

            float W2_Wxi[PIG_DIM];
            for (uint d = 0; d < D; d++) {
                float sum = 0.0f;
                for (uint k = 0; k < D; k++) sum += W2_ptr[d * D + k] * Wx_self[k];
                W2_Wxi[d] = sum;
            }

            // Collect neighbors (self-loop + edges where dst == a)
            float attn_scores[32];
            float Wx_nbrs[32][PIG_DIM];
            uint n_nbrs = 0;

            // Self-loop
            for (uint d = 0; d < D; d++) Wx_nbrs[n_nbrs][d] = Wx_self[d];
            n_nbrs++;

            for (uint e = 0; e < numLigEdges && n_nbrs < 31; e++) {
                if (ligEdges[e].dst == a) {
                    uint src = ligEdges[e].src;
                    for (uint d = 0; d < D; d++) {
                        float sum = W1_b[d];
                        for (uint k = 0; k < D; k++) sum += W1_w[d * D + k] * lig_h[src][k];
                        Wx_nbrs[n_nbrs][d] = sum;
                    }
                    n_nbrs++;
                }
            }

            // Attention: E_ij = Wx_i^T * W2 * Wx_j + Wx_j^T * W2 * Wx_i
            for (uint n = 0; n < n_nbrs; n++) {
                float e_ij = 0.0f;
                for (uint d = 0; d < D; d++) {
                    float w2_wxj = 0.0f;
                    for (uint k = 0; k < D; k++) {
                        w2_wxj += W2_ptr[d * D + k] * Wx_nbrs[n][k];
                    }
                    e_ij += Wx_self[d] * w2_wxj + Wx_nbrs[n][d] * W2_Wxi[d];
                }
                attn_scores[n] = e_ij;
            }

            // Softmax
            float max_s = -1e30f;
            for (uint n = 0; n < n_nbrs; n++) max_s = max(max_s, attn_scores[n]);
            float sum_exp = 0.0f;
            for (uint n = 0; n < n_nbrs; n++) {
                attn_scores[n] = exp(attn_scores[n] - max_s);
                sum_exp += attn_scores[n];
            }
            float inv_sum = 1.0f / max(sum_exp, 1e-8f);

            // Weighted sum
            float x_prime[PIG_DIM];
            for (uint d = 0; d < D; d++) x_prime[d] = 0.0f;
            for (uint n = 0; n < n_nbrs; n++) {
                float w = attn_scores[n] * inv_sum;
                for (uint d = 0; d < D; d++) x_prime[d] += w * Wx_nbrs[n][d];
            }

            // ReLU + gate
            for (uint d = 0; d < D; d++) x_prime[d] = max(x_prime[d], 0.0f);

            float z_val = gate_b[0];
            for (uint d = 0; d < D; d++) z_val += gate_w[d] * lig_h[a][d];
            for (uint d = 0; d < D; d++) z_val += gate_w[D + d] * x_prime[d];
            float z = 1.0f / (1.0f + exp(-z_val));

            for (uint d = 0; d < D; d++) {
                lig_scratch[a][d] = z * lig_h[a][d] + (1.0f - z) * x_prime[d];
            }
        }
        // Copy scratch back
        for (uint a = 0; a < L; a++) {
            for (uint d = 0; d < D; d++) lig_h[a][d] = lig_scratch[a][d];
        }
    }

    // -------------------------------------------------------------------
    // Phase B: Build intermolecular edges (ligand → protein within 5 Å)
    // -------------------------------------------------------------------

    device const float *myLigPos = intermediates + tid * PIG_POSE_LIG_POS_FLOATS;

    // Store inter edges as (lig_idx, prot_idx) pairs
    // Max edges: L × P could be 64×256 = 16384, but cutoff limits this heavily.
    // We use a fixed-size buffer; typical count is ~200-500.
    uint inter_lig[4096];
    uint inter_prot[4096];
    uint numInterEdges = 0;
    float proteinClashPenalty = 0.0f;

    for (uint la = 0; la < L; la++) {
        float3 lpos = float3(myLigPos[la * 3], myLigPos[la * 3 + 1], myLigPos[la * 3 + 2]);
        for (uint pa = 0; pa < P; pa++) {
            float3 ppos = float3(protPositions[pa * 3], protPositions[pa * 3 + 1], protPositions[pa * 3 + 2]);
            float3 delta = lpos - ppos;
            float dist2 = dot(delta, delta);

            // Match PoseValidator's severe protein-clash threshold. The learned
            // Morse term alone is too soft for GA search and sub-0.5 Å contacts
            // are excluded from PIGNet2's interaction edge list, so add a hard
            // positive bump before the normal interaction cutoff filter.
            float dist = 0.0f;
            if (dist2 < (2.0f * 2.0f)) {
                dist = sqrt(max(dist2, 1e-8f));
                float overlap = 2.0f - dist;
                proteinClashPenalty += 50.0f * overlap * overlap;
            }

            if (dist2 >= (PIG_INTERACT_MIN * PIG_INTERACT_MIN) &&
                dist2 <= (PIG_INTER_CUTOFF * PIG_INTER_CUTOFF) &&
                numInterEdges < 4096) {
                inter_lig[numInterEdges] = la;
                inter_prot[numInterEdges] = pa;
                numInterEdges++;
            }
        }
    }

    // -------------------------------------------------------------------
    // Phase C: 3× InteractionNet (mixed protein ↔ ligand)
    // -------------------------------------------------------------------
    //
    // InteractionNet message passing:
    //   message = W2(x_j), aggregation = MAX over neighbors
    //   x' = relu(W1*x + max_agg)
    //   h' = GRU(x', x)
    //
    // For protein atoms: messages come FROM ligand atoms via inter-edges.
    // For ligand atoms: messages come FROM protein atoms via inter-edges.
    // Protein embeddings are loaded from setupBuffer; ligand from lig_h.

    // Working copies of protein embeddings live in device memory (per-pose).
    // Moving these off-stack is what lets the kernel JIT-compile reliably.
    // Two sections per pose used as a ping-pong (no copy at end of each layer).
    uint protSlice = max(P, 1u) * D;
    device float *prot_curr = protScratch + tid * (2u * protSlice);
    device float *prot_next = prot_curr + protSlice;
    {
        device const float *protEmb = setupBuffer;
        for (uint pa = 0; pa < P; pa++) {
            for (uint d = 0; d < D; d++) {
                prot_curr[pa * D + d] = protEmb[pa * D + d];
            }
        }
    }

    for (uint layer = 0; layer < PIG_N_GNN; layer++) {
        uint base = PW_INT0_W1_W + layer * 8;
        device const float *W1_w   = pigGetWeight(weights, weightEntries, base + 0);
        device const float *W1_b   = pigGetWeight(weights, weightEntries, base + 1);
        device const float *W2_w   = pigGetWeight(weights, weightEntries, base + 2);
        device const float *W2_b   = pigGetWeight(weights, weightEntries, base + 3);
        device const float *rnn_wih = pigGetWeight(weights, weightEntries, base + 4);
        device const float *rnn_whh = pigGetWeight(weights, weightEntries, base + 5);
        device const float *rnn_bih = pigGetWeight(weights, weightEntries, base + 6);
        device const float *rnn_bhh = pigGetWeight(weights, weightEntries, base + 7);

        // Cache W2*x_lig once per ligand atom for this layer. Protein aggregation
        // reuses these vectors for every inter-edge instead of recomputing a 128x128
        // matvec for each ligand-protein pair.
        for (uint la = 0; la < L; la++) {
            pigLinear(lig_scratch[la], lig_h[la], W2_w, W2_b, D, D);
        }

        // Cache W2*x_prot for the current protein embeddings. This must be
        // recomputed each InteractionNet layer because protein nodes are updated
        // by the previous layer in the reference PIGNet2 implementation.
        for (uint pa = 0; pa < P; pa++) {
            float ph_tl[PIG_DIM];
            for (uint d = 0; d < D; d++) ph_tl[d] = prot_curr[pa * D + d];
            float w2x[PIG_DIM];
            pigLinear(w2x, ph_tl, W2_w, W2_b, D, D);
            for (uint d = 0; d < D; d++) prot_next[pa * D + d] = w2x[d];
        }

        // For ligand atoms: aggregate messages from protein neighbors (MAX)
        float lig_new[PIG_MAX_LIG][PIG_DIM];
        for (uint la = 0; la < L; la++) {
            // Compute W1 * x_lig
            float w1x[PIG_DIM];
            pigLinear(w1x, lig_h[la], W1_w, W1_b, D, D);

            // MAX aggregation of W2*x_prot over inter-edge neighbors
            float max_msg[PIG_DIM];
            bool has_nbr = false;
            for (uint d = 0; d < D; d++) max_msg[d] = -1e30f;

            for (uint e = 0; e < numInterEdges; e++) {
                if (inter_lig[e] == la) {
                    // Message from the current protein embedding: cached W2*x_prot.
                    device const float *w2x_cached = prot_next + inter_prot[e] * D;
                    for (uint d = 0; d < D; d++) {
                        max_msg[d] = max(max_msg[d], w2x_cached[d]);
                    }
                    has_nbr = true;
                }
            }

            if (!has_nbr) {
                for (uint d = 0; d < D; d++) max_msg[d] = 0.0f;
            }

            // x' = relu(W1*x + max_msg)
            float x_prime[PIG_DIM];
            for (uint d = 0; d < D; d++) {
                x_prime[d] = max(w1x[d] + max_msg[d], 0.0f);
            }

            // GRU update: h' = GRU(x', x)
            pigGRUCell(lig_new[la], x_prime, lig_h[la], rnn_wih, rnn_whh, rnn_bih, rnn_bhh);
        }

        // Protein atoms with no ligand neighbors still follow the reference
        // InteractionNet update with a zero aggregate.
        for (uint pa = 0; pa < P; pa++) {
            // Snapshot current hidden state into thread memory.
            float ph_tl[PIG_DIM];
            for (uint d = 0; d < D; d++) ph_tl[d] = prot_curr[pa * D + d];

            float w1x[PIG_DIM];
            pigLinear(w1x, ph_tl, W1_w, W1_b, D, D);

            float max_msg[PIG_DIM];
            bool has_nbr = false;
            for (uint d = 0; d < D; d++) max_msg[d] = -1e30f;

            for (uint e = 0; e < numInterEdges; e++) {
                if (inter_prot[e] == pa) {
                    // Message from ligand atom: cached W2 * lig_h[src].
                    uint src = inter_lig[e];
                    for (uint d = 0; d < D; d++) {
                        max_msg[d] = max(max_msg[d], lig_scratch[src][d]);
                    }
                    has_nbr = true;
                }
            }

            if (!has_nbr) {
                for (uint d = 0; d < D; d++) max_msg[d] = 0.0f;
            }

            float x_prime[PIG_DIM];
            for (uint d = 0; d < D; d++) {
                x_prime[d] = max(w1x[d] + max_msg[d], 0.0f);
            }
            float pn_tl[PIG_DIM];
            pigGRUCell(pn_tl, x_prime, ph_tl, rnn_wih, rnn_whh, rnn_bih, rnn_bhh);
            for (uint d = 0; d < D; d++) prot_next[pa * D + d] = pn_tl[d];
        }

        // Update embeddings - ping-pong the protein pointers instead of copying.
        for (uint la = 0; la < L; la++) {
            for (uint d = 0; d < D; d++) lig_h[la][d] = lig_new[la][d];
        }
        device float *tmp = prot_curr;
        prot_curr = prot_next;
        prot_next = tmp;
    }

    // After the layer loop, prot_curr holds the final protein embeddings.
    // Alias `prot_h` for the rest of the kernel to keep the existing Phase D code path.
    device float *prot_h = prot_curr;

    // -------------------------------------------------------------------
    // Phase D: Pairwise physics energy computation
    // -------------------------------------------------------------------

    // Load scalar physics coefficients
    device const float *hbond_coeff_ptr = pigGetWeight(weights, weightEntries, PW_HBOND_COEFF);
    device const float *hydro_coeff_ptr = pigGetWeight(weights, weightEntries, PW_HYDROPHOBIC_COEFF);
    device const float *rotor_coeff_ptr = pigGetWeight(weights, weightEntries, PW_ROTOR_COEFF);
    device const float *metal_coeff_ptr = pigGetWeight(weights, weightEntries, PW_METAL_COEFF);
    float hbond_coeff2 = hbond_coeff_ptr[0] * hbond_coeff_ptr[0];
    float hydro_coeff2 = hydro_coeff_ptr[0] * hydro_coeff_ptr[0];
    float rotor_coeff2 = rotor_coeff_ptr[0] * rotor_coeff_ptr[0];
    float metal_coeff2 = metal_coeff_ptr[0] * metal_coeff_ptr[0];

    // Energy head weight pointers
    device const float *eps_l0_w = pigGetWeight(weights, weightEntries, PW_EPS_L0_W);
    device const float *eps_l0_b = pigGetWeight(weights, weightEntries, PW_EPS_L0_B);
    device const float *eps_l2_w = pigGetWeight(weights, weightEntries, PW_EPS_L2_W);
    device const float *eps_l2_b = pigGetWeight(weights, weightEntries, PW_EPS_L2_B);
    device const float *dvdw_l0_w = pigGetWeight(weights, weightEntries, PW_DVDW_L0_W);
    device const float *dvdw_l0_b = pigGetWeight(weights, weightEntries, PW_DVDW_L0_B);
    device const float *dvdw_l2_w = pigGetWeight(weights, weightEntries, PW_DVDW_L2_W);
    device const float *dvdw_l2_b = pigGetWeight(weights, weightEntries, PW_DVDW_L2_B);
    // Morse-specific weight pointers
    device const float *width_l0_w = pigGetWeight(weights, weightEntries, PW_WIDTH_L0_W);
    device const float *width_l0_b = pigGetWeight(weights, weightEntries, PW_WIDTH_L0_B);
    device const float *width_l2_w = pigGetWeight(weights, weightEntries, PW_WIDTH_L2_W);
    device const float *width_l2_b = pigGetWeight(weights, weightEntries, PW_WIDTH_L2_B);

    // Accumulate energies
    float E_vdw = 0.0f;
    float E_hbond = 0.0f;
    float E_metal = 0.0f;
    float E_hydrophobic = 0.0f;

    // PIGNet2 uses two distance ranges: 0.5-5 Å for InteractionNet graph edges,
    // but 0.5-999 Å for the final physics energy head.
    for (uint la = 0; la < L; la++) {
        float3 lpos = float3(myLigPos[la * 3], myLigPos[la * 3 + 1], myLigPos[la * 3 + 2]);
        uint lig_flags = ligAux[la].flags;
        float lig_vdw = ligAux[la].vdwRadius;
        bool lig_metal = (lig_flags & PIG_FLAG_METAL) != 0;
        bool lig_hbd = (lig_flags & PIG_FLAG_H_DONOR) != 0;
        bool lig_hba = (lig_flags & PIG_FLAG_H_ACCEPTOR) != 0;
        bool lig_hydro = (lig_flags & PIG_FLAG_HYDROPHOBIC) != 0;

        for (uint pa = 0; pa < P; pa++) {
            float3 ppos = float3(protPositions[pa * 3], protPositions[pa * 3 + 1], protPositions[pa * 3 + 2]);
            float dist2 = dot(lpos - ppos, lpos - ppos);
            if (dist2 < (PIG_INTERACT_MIN * PIG_INTERACT_MIN)) continue;
            float dist = sqrt(dist2);

            uint prot_flags = protAux[pa].flags;
            float prot_vdw = protAux[pa].vdwRadius;
            bool prot_metal = (prot_flags & PIG_FLAG_METAL) != 0;
            bool prot_hbd = (prot_flags & PIG_FLAG_H_DONOR) != 0;
            bool prot_hba = (prot_flags & PIG_FLAG_H_ACCEPTOR) != 0;
            bool prot_hydro = (prot_flags & PIG_FLAG_HYDROPHOBIC) != 0;

            float concat[PIG_DIM * 2];
            for (uint d = 0; d < D; d++) concat[d] = lig_h[la][d];
            for (uint d = 0; d < D; d++) concat[D + d] = prot_h[pa * D + d];

            float dvdw_h[PIG_DIM];
            pigLinear(dvdw_h, concat, dvdw_l0_w, dvdw_l0_b, D, D * 2);
            for (uint d = 0; d < D; d++) dvdw_h[d] = max(dvdw_h[d], 0.0f);
            float dvdw = dvdw_l2_b[0];
            for (uint d = 0; d < D; d++) dvdw += dvdw_l2_w[d] * dvdw_h[d];
            dvdw = tanh(dvdw) * PIG_DEV_VDW_COEFF;

            float R = lig_vdw + prot_vdw + dvdw;

            float eps_h[PIG_DIM];
            pigLinear(eps_h, concat, eps_l0_w, eps_l0_b, D, D * 2);
            for (uint d = 0; d < D; d++) eps_h[d] = max(eps_h[d], 0.0f);
            float eps_raw = eps_l2_b[0];
            for (uint d = 0; d < D; d++) eps_raw += eps_l2_w[d] * eps_h[d];
            float epsilon = (1.0f / (1.0f + exp(-eps_raw))) * (PIG_VDW_EPS_HI - PIG_VDW_EPS_LO) + PIG_VDW_EPS_LO;

            float width_h[PIG_DIM];
            pigLinear(width_h, concat, width_l0_w, width_l0_b, D, D * 2);
            for (uint d = 0; d < D; d++) width_h[d] = max(width_h[d], 0.0f);
            float width_raw = width_l2_b[0];
            for (uint d = 0; d < D; d++) width_raw += width_l2_w[d] * width_h[d];
            float morseWidth = (1.0f / (1.0f + exp(-width_raw))) * (PIG_VDW_WIDTH_HI - PIG_VDW_WIDTH_LO) + PIG_VDW_WIDTH_LO;

            if (!lig_metal && !prot_metal) {
                E_vdw += pigMorsePotential(dist, R, epsilon, morseWidth);
            }

            bool hbond_mask = false;
            if ((lig_hbd && !lig_metal) && (prot_hba && !prot_metal)) hbond_mask = true;
            if ((prot_hbd && !prot_metal) && (lig_hba && !lig_metal)) hbond_mask = true;
            if (hbond_mask) {
                E_hbond += pigLinearPotential(dist, R, -hbond_coeff2, PIG_HBOND_C1, PIG_HBOND_C2);
            }

            bool metal_mask = false;
            if (lig_metal && (prot_hba && !prot_metal)) metal_mask = true;
            if (prot_metal && (lig_hba && !lig_metal)) metal_mask = true;
            if (metal_mask) {
                E_metal += pigLinearPotential(dist, R, -metal_coeff2, PIG_METAL_C1, PIG_METAL_C2);
            }

            if (lig_hydro && prot_hydro) {
                E_hydrophobic += pigLinearPotential(dist, R, -hydro_coeff2, PIG_HYDRO_C1, PIG_HYDRO_C2);
            }
        }
    }

    // -------------------------------------------------------------------
    // Phase E: Rotor penalty + final score
    // -------------------------------------------------------------------

    float penalty = 1.0f + rotor_coeff2 * float(params.numRotatableBonds);
    E_vdw /= penalty;
    E_hbond /= penalty;
    E_metal /= penalty;
    E_hydrophobic /= penalty;

    float total = E_vdw + E_hbond + E_metal + E_hydrophobic;
    float oopPenalty = poses[tid].clashPenalty;
    float clashPenalty = oopPenalty + proteinClashPenalty;

    // Store in pose (energy for GA minimization; decomposition in diagnostic fields)
    poses[tid].energy = total + clashPenalty;
    poses[tid].stericEnergy = E_vdw;
    poses[tid].hydrophobicEnergy = E_hydrophobic;
    poses[tid].hbondEnergy = E_hbond + E_metal;
    poses[tid].torsionPenalty = penalty;
    poses[tid].drusinaCorrection = E_metal;
    poses[tid].clashPenalty = clashPenalty;
}
