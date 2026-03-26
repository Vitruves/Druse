// DruseAFv4Compute.metal
//
// DruseAF v4 — Pairwise Geometric Network (PGN)
// ~134K params, ~100x faster per-pose scoring than v3 cross-attention.
//
// Kernel pipeline:
//   SETUP (once per docking target, ~2ms):
//     1. druseAFv4Encode      — MLP encode protein + ligand atoms
//     2. druseAFv4MsgTransform — GELU(Linear(h)) per protein atom (×3 layers)
//     3. druseAFv4MsgAggregate — distance-attention aggregation + LN (×3 layers)
//     4. druseAFv4PairPrep     — compute pair projections for scoring
//
//   PER GENERATION:
//     5. druseAFEncode         — transform ligand positions (reuses v3 kernel)
//     6. druseAFv4Score        — pairwise Hadamard scoring, ~0.02ms/pose
//
// Weight tensor layout (56 tensors, DRAF v2 format):
//   [0-3]   Protein encoder       [4-27]  3× message passing (8 each)
//   [28-31] Ligand encoder        [32-35] Pair projections
//   [36-37] RBF projection        [38-39] Pair energy head
//   [40-41] Context gate          [42-43] Context projection
//   [44-45] Context LayerNorm     [46-49] Affinity head
//   [50-53] Confidence head       [54-55] pair_scale, pair_bias

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ---------------------------------------------------------------------------
// Constants (must match trainDruseAFv4.py exactly)
// ---------------------------------------------------------------------------

constant uint H       = 128;   // HIDDEN_DIM
constant uint PD      = 64;    // PAIR_DIM
constant uint FEAT    = 20;    // NUM_ATOM_FEATURES
constant uint MSG_RBF = 16;    // MSG_RBF_BINS
constant uint CRS_RBF = 24;    // CROSS_RBF_BINS
constant float RBF_G  = 2.0f;  // RBF_GAMMA
constant float MSG_CUT = 8.0f; // MSG_CUTOFF
constant float CRS_CUT = 8.0f; // CROSS_CUTOFF

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline float linear_1d(device const float* w, device const float* b,
                        thread const float* input, uint in_dim, uint out_idx) {
    float val = b[out_idx];
    for (uint k = 0; k < in_dim; k++)
        val += input[k] * w[out_idx * in_dim + k];
    return val;
}

// ---------------------------------------------------------------------------
// Kernel 1: druseAFv4Encode
//   Encode protein (tid < P) and ligand (tid >= P) atoms via MLP.
//   Thread count: P + L
// ---------------------------------------------------------------------------

kernel void druseAFv4Encode(
    device const float*             protFeat    [[buffer(0)]],  // [P, 20]
    device const float*             ligFeat     [[buffer(1)]],  // [L, 20]
    device float*                   protHidden  [[buffer(2)]],  // [P, H] out
    device float*                   ligHidden   [[buffer(3)]],  // [L, H] out
    device const float*             weights     [[buffer(4)]],
    device const DruseAFWeightEntry* entries    [[buffer(5)]],
    constant DruseAFv4Params&       params      [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint P = params.numProteinAtoms;
    uint L = params.numLigandAtoms;

    bool is_protein = (tid < P);
    bool is_ligand  = (tid >= P && tid < P + L);
    if (!is_protein && !is_ligand) return;

    // Select encoder weights: protein [0-3] or ligand [28-31]
    uint enc_base = is_protein ? 0 : 28;
    device const float* w0 = weights + entries[enc_base + 0].offset;
    device const float* b0 = weights + entries[enc_base + 1].offset;
    device const float* w1 = weights + entries[enc_base + 2].offset;
    device const float* b1 = weights + entries[enc_base + 3].offset;

    // Load input features
    float feat[FEAT];
    if (is_protein) {
        for (uint d = 0; d < FEAT; d++) feat[d] = protFeat[tid * FEAT + d];
    } else {
        uint li = tid - P;
        for (uint d = 0; d < FEAT; d++) feat[d] = ligFeat[li * FEAT + d];
    }

    // Layer 0: Linear(20→128) + GELU
    float h0[H];
    for (uint d = 0; d < H; d++)
        h0[d] = gelu_tanh(linear_1d(w0, b0, feat, FEAT, d));

    // Layer 1: Linear(128→128)
    float h1[H];
    for (uint d = 0; d < H; d++)
        h1[d] = linear_1d(w1, b1, h0, H, d);

    // Store
    if (is_protein) {
        for (uint d = 0; d < H; d++) protHidden[tid * H + d] = h1[d];
    } else {
        uint li = tid - P;
        for (uint d = 0; d < H; d++) ligHidden[li * H + d] = h1[d];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: druseAFv4MsgTransform
//   Compute msg_mlp(h) = GELU(Linear(h)) for each protein atom.
//   Dispatched once per message passing layer (3 total).
//   Thread count: P
// ---------------------------------------------------------------------------

kernel void druseAFv4MsgTransform(
    device const float*             protHidden  [[buffer(0)]],  // [P, H] in
    device float*                   msgTemp     [[buffer(1)]],  // [P, H] out
    device const float*             weights     [[buffer(2)]],
    device const DruseAFWeightEntry* entries    [[buffer(3)]],
    constant DruseAFv4Params&       params      [[buffer(4)]],
    constant uint&                  layerIdx    [[buffer(5)]],  // 0, 1, or 2
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numProteinAtoms) return;

    // Weight indices: layer base = 4 + layerIdx * 8
    // msg_mlp.0.weight = base+0, msg_mlp.0.bias = base+1
    uint base = 4 + layerIdx * 8;
    device const float* w = weights + entries[base + 0].offset;
    device const float* b = weights + entries[base + 1].offset;

    float input[H];
    for (uint d = 0; d < H; d++)
        input[d] = protHidden[tid * H + d];

    for (uint d = 0; d < H; d++)
        msgTemp[tid * H + d] = gelu_tanh(linear_1d(w, b, input, H, d));
}

// ---------------------------------------------------------------------------
// Kernel 3: druseAFv4MsgAggregate
//   Distance-attention message aggregation + residual + LayerNorm.
//   Thread count: P
// ---------------------------------------------------------------------------

kernel void druseAFv4MsgAggregate(
    device float*                   protHidden  [[buffer(0)]],  // [P, H] in/out
    device const float*             msgTemp     [[buffer(1)]],  // [P, H] transformed
    device const float*             protPos     [[buffer(2)]],  // [P, 3]
    device const float*             weights     [[buffer(3)]],
    device const DruseAFWeightEntry* entries    [[buffer(4)]],
    constant DruseAFv4Params&       params      [[buffer(5)]],
    constant uint&                  layerIdx    [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numProteinAtoms) return;
    uint P = params.numProteinAtoms;

    // Weight indices: base = 4 + layerIdx * 8
    // attn_mlp.0: base+2 (w), base+3 (b)  — Linear(16→32)
    // attn_mlp.2: base+4 (w), base+5 (b)  — Linear(32→1)
    // norm:       base+6 (w), base+7 (b)
    uint base = 4 + layerIdx * 8;
    device const float* aw1 = weights + entries[base + 2].offset;
    device const float* ab1 = weights + entries[base + 3].offset;
    device const float* aw2 = weights + entries[base + 4].offset;
    device const float* ab2 = weights + entries[base + 5].offset;
    device const float* lnw = weights + entries[base + 6].offset;
    device const float* lnb = weights + entries[base + 7].offset;

    float3 my_pos = float3(protPos[tid*3], protPos[tid*3+1], protPos[tid*3+2]);

    // Compute attention-weighted message sum
    float msg[H];
    for (uint d = 0; d < H; d++) msg[d] = 0.0f;

    // Two-pass online softmax for numerical stability
    float max_attn = -1e9f;
    float attn_sum = 0.0f;

    // Pass 1: compute attention logits and find max
    // (We store logits in a small array since P ≤ 256)
    float attn_logits[256];
    for (uint j = 0; j < P; j++) {
        if (j == tid) { attn_logits[j] = -1e9f; continue; }
        float3 j_pos = float3(protPos[j*3], protPos[j*3+1], protPos[j*3+2]);
        float d = distance(my_pos, j_pos);
        if (d > MSG_CUT) { attn_logits[j] = -1e9f; continue; }

        // RBF → attn_mlp: Linear(16→32, GELU) → Linear(32→1)
        float rbf[MSG_RBF];
        float spacing = MSG_CUT / float(MSG_RBF - 1);
        for (uint b = 0; b < MSG_RBF; b++) {
            float diff = d - float(b) * spacing;
            rbf[b] = exp(-RBF_G * diff * diff);
        }
        // Layer 1: Linear(16→32) + GELU
        float h32[32];
        for (uint k = 0; k < 32; k++) {
            float val = ab1[k];
            for (uint b = 0; b < MSG_RBF; b++)
                val += rbf[b] * aw1[k * MSG_RBF + b];
            h32[k] = gelu_tanh(val);
        }
        // Layer 2: Linear(32→1)
        float logit = ab2[0];
        for (uint k = 0; k < 32; k++)
            logit += h32[k] * aw2[k];  // aw2 is [1, 32], stored row-major

        attn_logits[j] = logit;
        max_attn = max(max_attn, logit);
    }

    // Pass 2: softmax + weighted sum
    for (uint j = 0; j < P; j++) {
        float w = exp(attn_logits[j] - max_attn);
        attn_sum += w;
        for (uint dd = 0; dd < H; dd++)
            msg[dd] += w * msgTemp[j * H + dd];
    }

    // Normalize
    float inv_sum = 1.0f / (attn_sum + 1e-8f);
    for (uint dd = 0; dd < H; dd++)
        msg[dd] *= inv_sum;

    // Residual + LayerNorm
    float h_new[H];
    float mean_val = 0.0f;
    for (uint dd = 0; dd < H; dd++) {
        h_new[dd] = protHidden[tid * H + dd] + msg[dd];
        mean_val += h_new[dd];
    }
    mean_val /= float(H);

    float var_val = 0.0f;
    for (uint dd = 0; dd < H; dd++) {
        float diff = h_new[dd] - mean_val;
        var_val += diff * diff;
    }
    var_val /= float(H);
    float inv_std = rsqrt(var_val + 1e-5f);

    for (uint dd = 0; dd < H; dd++)
        protHidden[tid * H + dd] = (h_new[dd] - mean_val) * inv_std * lnw[dd] + lnb[dd];
}

// ---------------------------------------------------------------------------
// Kernel 4: druseAFv4PairPrep
//   Compute pair projections: Linear(128→64) for protein and ligand.
//   Thread count: P + L
// ---------------------------------------------------------------------------

kernel void druseAFv4PairPrep(
    device const float*             protHidden  [[buffer(0)]],  // [P, H]
    device const float*             ligHidden   [[buffer(1)]],  // [L, H]
    device float*                   protPairProj [[buffer(2)]],  // [P, PD] out
    device float*                   ligPairProj  [[buffer(3)]],  // [L, PD] out
    device const float*             weights     [[buffer(4)]],
    device const DruseAFWeightEntry* entries    [[buffer(5)]],
    constant DruseAFv4Params&       params      [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint P = params.numProteinAtoms;
    uint L = params.numLigandAtoms;

    bool is_prot = (tid < P);
    bool is_lig  = (tid >= P && tid < P + L);
    if (!is_prot && !is_lig) return;

    // Weight indices: prot_pair_proj [32-33], lig_pair_proj [34-35]
    uint widx = is_prot ? 32 : 34;
    device const float* w = weights + entries[widx].offset;
    device const float* b = weights + entries[widx + 1].offset;

    float input[H];
    device const float* src = is_prot ? protHidden : ligHidden;
    uint atom_idx = is_prot ? tid : (tid - P);
    for (uint d = 0; d < H; d++)
        input[d] = src[atom_idx * H + d];

    device float* dst = is_prot ? protPairProj : ligPairProj;
    for (uint d = 0; d < PD; d++)
        dst[atom_idx * PD + d] = linear_1d(w, b, input, H, d);
}

// ---------------------------------------------------------------------------
// Kernel 5: druseAFv4Score
//   Per-pose pairwise Hadamard scoring.
//   Dispatch: numPoses threadgroups × MAX_LIG_ATOMS threads per group.
//   Each thread handles one ligand atom, loops over protein atoms within cutoff.
// ---------------------------------------------------------------------------

kernel void druseAFv4Score(
    device DockPose*                poses        [[buffer(0)]],   // [numPoses]
    device const float*             protPos      [[buffer(1)]],   // [P, 3]
    device const float*             protPairProj [[buffer(2)]],   // [P, PD]
    device const float*             ligPairProj  [[buffer(3)]],   // [L, PD]
    device const float*             ligHidden    [[buffer(4)]],   // [L, H]
    device const float*             weights      [[buffer(5)]],
    device const DruseAFWeightEntry* entries     [[buffer(6)]],
    constant DruseAFv4Params&       params       [[buffer(7)]],
    device const float*             ligTransformed [[buffer(8)]],  // [numPoses, L, 3]
    constant GridParams&            gridParams   [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint pose_idx = gid;
    uint P = params.numProteinAtoms;
    uint L = params.numLigandAtoms;
    bool valid = (tid < L);

    // Weight offsets for scoring components
    device const float* rbf_w  = weights + entries[36].offset;  // [PD, CRS_RBF]
    device const float* rbf_b  = weights + entries[37].offset;  // [PD]
    device const float* pe_w   = weights + entries[38].offset;  // [1, PD]
    device const float* pe_b   = weights + entries[39].offset;  // [1]
    device const float* cg_w   = weights + entries[40].offset;  // [1, PD]
    device const float* cg_b   = weights + entries[41].offset;  // [1]
    device const float* cp_w   = weights + entries[42].offset;  // [H, PD]
    device const float* cp_b   = weights + entries[43].offset;  // [H]
    device const float* ln_w   = weights + entries[44].offset;  // [H]
    device const float* ln_b   = weights + entries[45].offset;  // [H]

    // Load ligand pair projection
    float lig_p[PD];
    if (valid) {
        for (uint d = 0; d < PD; d++)
            lig_p[d] = ligPairProj[tid * PD + d];
    }

    // Get transformed ligand position
    float3 my_pos = float3(0);
    if (valid) {
        uint base = pose_idx * L * 3 + tid * 3;
        my_pos = float3(ligTransformed[base], ligTransformed[base+1], ligTransformed[base+2]);
    }

    // === Pairwise scoring loop ===
    float energy_sum = 0.0f;
    float ctx[PD];
    float gate_sum = 0.0f;
    for (uint d = 0; d < PD; d++) ctx[d] = 0.0f;

    if (valid) {
        float rbf_spacing = CRS_CUT / float(CRS_RBF - 1);

        for (uint p = 0; p < P; p++) {
            float3 p_pos = float3(protPos[p*3], protPos[p*3+1], protPos[p*3+2]);
            float dist = distance(my_pos, p_pos);
            if (dist > CRS_CUT || dist < 0.01f) continue;

            // RBF encoding
            float rbf[CRS_RBF];
            for (uint b = 0; b < CRS_RBF; b++) {
                float diff = dist - float(b) * rbf_spacing;
                rbf[b] = exp(-RBF_G * diff * diff);
            }

            // Compute pair[PD] = lig_p ⊙ prot_p ⊙ GELU(rbf_proj(rbf))
            float pair[PD];
            for (uint d = 0; d < PD; d++) {
                // RBF projection + GELU
                float rbf_val = rbf_b[d];
                for (uint b = 0; b < CRS_RBF; b++)
                    rbf_val += rbf[b] * rbf_w[d * CRS_RBF + b];
                rbf_val = gelu_tanh(rbf_val);

                // Load protein pair proj
                float pp = protPairProj[p * PD + d];

                // Hadamard product
                pair[d] = lig_p[d] * pp * rbf_val;
            }

            // Pair energy: GELU(pair) → Linear(PD→1)
            float e = pe_b[0];
            for (uint d = 0; d < PD; d++)
                e += gelu_tanh(pair[d]) * pe_w[d];
            energy_sum += e;

            // Context gate: pair → Linear(PD→1) (no GELU)
            float g = cg_b[0];
            for (uint d = 0; d < PD; d++)
                g += pair[d] * cg_w[d];
            float w = exp(g);
            gate_sum += w;

            // Accumulate weighted protein projection for context
            for (uint d = 0; d < PD; d++)
                ctx[d] += w * protPairProj[p * PD + d];
        }
    }

    // === Context normalization + lig hidden update ===
    // Reload ligand hidden state (was not kept in registers during pair loop)
    float lig_h_ctx[H];
    if (valid) {
        float inv_gate = 1.0f / (gate_sum + 1e-8f);
        for (uint d = 0; d < PD; d++)
            ctx[d] *= inv_gate;

        // context_proj: Linear(PD→H) + residual
        for (uint d = 0; d < H; d++) {
            float proj = cp_b[d];
            for (uint d2 = 0; d2 < PD; d2++)
                proj += ctx[d2] * cp_w[d * PD + d2];
            lig_h_ctx[d] = ligHidden[tid * H + d] + proj;
        }

        // LayerNorm
        float mean_val = 0.0f;
        for (uint d = 0; d < H; d++) mean_val += lig_h_ctx[d];
        mean_val /= float(H);
        float var_val = 0.0f;
        for (uint d = 0; d < H; d++) {
            float diff = lig_h_ctx[d] - mean_val;
            var_val += diff * diff;
        }
        var_val /= float(H);
        float inv_std = rsqrt(var_val + 1e-5f);
        for (uint d = 0; d < H; d++)
            lig_h_ctx[d] = (lig_h_ctx[d] - mean_val) * inv_std * ln_w[d] + ln_b[d];
    }

    // === Threadgroup reduction ===
    threadgroup float tg_energy[64];  // MAX_LIG_ATOMS
    threadgroup float tg_scratch[64]; // for dimension-wise reduction
    threadgroup float tg_repr[128];   // final pooled representation

    tg_energy[tid] = valid ? energy_sum : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce energy (thread 0)
    float total_energy = 0.0f;
    if (tid == 0) {
        for (uint i = 0; i < L; i++)
            total_energy += tg_energy[i];
    }

    // Reduce context representation (dimension by dimension)
    for (uint d = 0; d < H; d++) {
        tg_scratch[tid] = valid ? lig_h_ctx[d] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float sum = 0.0f;
            for (uint i = 0; i < L; i++) sum += tg_scratch[i];
            tg_repr[d] = sum / float(L);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Final prediction (thread 0 only) ===
    if (tid == 0) {
        // Load learned scalars
        float pair_scale = weights[entries[54].offset];
        float pair_bias  = weights[entries[55].offset];

        // Affinity head: Linear(128→64) → GELU → Linear(64→1)
        device const float* ah0_w = weights + entries[46].offset;
        device const float* ah0_b = weights + entries[47].offset;
        device const float* ah1_w = weights + entries[48].offset;
        device const float* ah1_b = weights + entries[49].offset;

        float aff_h[64];
        for (uint d = 0; d < 64; d++) {
            float val = ah0_b[d];
            for (uint d2 = 0; d2 < H; d2++)
                val += tg_repr[d2] * ah0_w[d * H + d2];
            aff_h[d] = gelu_tanh(val);
        }
        float global_aff = ah1_b[0];
        for (uint d = 0; d < 64; d++)
            global_aff += aff_h[d] * ah1_w[d];

        float pKd = total_energy * pair_scale + global_aff + pair_bias;

        // Confidence head: Linear(128→64) → GELU → Linear(64→1) → sigmoid
        device const float* ch0_w = weights + entries[50].offset;
        device const float* ch0_b = weights + entries[51].offset;
        device const float* ch1_w = weights + entries[52].offset;
        device const float* ch1_b = weights + entries[53].offset;

        float conf_h[64];
        for (uint d = 0; d < 64; d++) {
            float val = ch0_b[d];
            for (uint d2 = 0; d2 < H; d2++)
                val += tg_repr[d2] * ch0_w[d * H + d2];
            conf_h[d] = gelu_tanh(val);
        }
        float conf_logit = ch1_b[0];
        for (uint d = 0; d < 64; d++)
            conf_logit += conf_h[d] * ch1_w[d];
        float confidence = 1.0f / (1.0f + exp(-conf_logit));

        float score = pKd * confidence;

        // Out-of-grid penalty: penalize if ligand centroid is far from grid center
        float3 lig_center = float3(0);
        for (uint i = 0; i < L; i++) {
            uint b = pose_idx * L * 3 + i * 3;
            lig_center += float3(ligTransformed[b], ligTransformed[b+1], ligTransformed[b+2]);
        }
        lig_center /= float(L);

        float3 grid_center = gridParams.searchCenter;
        float3 half_ext = gridParams.searchHalfExtent;
        float3 delta = abs(lig_center - grid_center) - half_ext;
        float oob_penalty = max(delta.x, 0.0f) + max(delta.y, 0.0f) + max(delta.z, 0.0f);
        oob_penalty *= 10.0f;  // kcal/mol per Angstrom outside

        // Store results (same fields as v3 for compatibility)
        poses[pose_idx].energy = -score + oob_penalty;
        poses[pose_idx].stericEnergy = pKd;              // reuse for pKd
        poses[pose_idx].hydrophobicEnergy = confidence;   // reuse for confidence
        poses[pose_idx].hbondEnergy = score;              // reuse for combined score
        poses[pose_idx].clashPenalty = oob_penalty;
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: druseAFv4Rescore
//   Single-pose rescoring (not inside GA loop).
//   Same computation as druseAFv4Score but for a pre-placed ligand.
//   Thread count: L threads, 1 threadgroup (single pose).
//   Ligand positions read directly from ligPosBuffer (no torsion transform).
// ---------------------------------------------------------------------------

kernel void druseAFv4Rescore(
    device float*                   output       [[buffer(0)]],   // [3]: pKd, confidence, score
    device const float*             protPos      [[buffer(1)]],   // [P, 3]
    device const float*             protPairProj [[buffer(2)]],   // [P, PD]
    device const float*             ligPairProj  [[buffer(3)]],   // [L, PD]
    device const float*             ligHidden    [[buffer(4)]],   // [L, H]
    device const float*             ligPos       [[buffer(5)]],   // [L, 3]
    device const float*             weights      [[buffer(6)]],
    device const DruseAFWeightEntry* entries     [[buffer(7)]],
    constant DruseAFv4Params&       params       [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint P = params.numProteinAtoms;
    uint L = params.numLigandAtoms;
    bool valid = (tid < L);

    // Weight pointers (same as Score kernel)
    device const float* rbf_w = weights + entries[36].offset;
    device const float* rbf_b = weights + entries[37].offset;
    device const float* pe_w  = weights + entries[38].offset;
    device const float* pe_b  = weights + entries[39].offset;
    device const float* cg_w  = weights + entries[40].offset;
    device const float* cg_b  = weights + entries[41].offset;
    device const float* cp_w  = weights + entries[42].offset;
    device const float* cp_b  = weights + entries[43].offset;
    device const float* ln_w  = weights + entries[44].offset;
    device const float* ln_b  = weights + entries[45].offset;

    float lig_p[PD];
    float3 my_pos = float3(0);
    if (valid) {
        for (uint d = 0; d < PD; d++)
            lig_p[d] = ligPairProj[tid * PD + d];
        my_pos = float3(ligPos[tid*3], ligPos[tid*3+1], ligPos[tid*3+2]);
    }

    // Pairwise scoring (identical to druseAFv4Score inner loop)
    float energy_sum = 0.0f;
    float ctx[PD];
    float gate_sum = 0.0f;
    for (uint d = 0; d < PD; d++) ctx[d] = 0.0f;

    if (valid) {
        float rbf_spacing = CRS_CUT / float(CRS_RBF - 1);
        for (uint p = 0; p < P; p++) {
            float3 p_pos = float3(protPos[p*3], protPos[p*3+1], protPos[p*3+2]);
            float dist = distance(my_pos, p_pos);
            if (dist > CRS_CUT || dist < 0.01f) continue;

            float rbf[CRS_RBF];
            for (uint b = 0; b < CRS_RBF; b++) {
                float diff = dist - float(b) * rbf_spacing;
                rbf[b] = exp(-RBF_G * diff * diff);
            }

            float pair[PD];
            for (uint d = 0; d < PD; d++) {
                float rv = rbf_b[d];
                for (uint b = 0; b < CRS_RBF; b++)
                    rv += rbf[b] * rbf_w[d * CRS_RBF + b];
                rv = gelu_tanh(rv);
                float pp = protPairProj[p * PD + d];
                pair[d] = lig_p[d] * pp * rv;
            }

            float e = pe_b[0];
            for (uint d = 0; d < PD; d++) e += gelu_tanh(pair[d]) * pe_w[d];
            energy_sum += e;

            float g = cg_b[0];
            for (uint d = 0; d < PD; d++) g += pair[d] * cg_w[d];
            float w = exp(g);
            gate_sum += w;
            for (uint d = 0; d < PD; d++) ctx[d] += w * protPairProj[p * PD + d];
        }
    }

    // Context update
    float lig_h_ctx[H];
    if (valid) {
        float inv_gate = 1.0f / (gate_sum + 1e-8f);
        for (uint d = 0; d < PD; d++) ctx[d] *= inv_gate;
        for (uint d = 0; d < H; d++) {
            float proj = cp_b[d];
            for (uint d2 = 0; d2 < PD; d2++) proj += ctx[d2] * cp_w[d * PD + d2];
            lig_h_ctx[d] = ligHidden[tid * H + d] + proj;
        }
        float m = 0; for (uint d = 0; d < H; d++) m += lig_h_ctx[d]; m /= float(H);
        float v = 0; for (uint d = 0; d < H; d++) { float x = lig_h_ctx[d]-m; v += x*x; }
        v /= float(H);
        float is = rsqrt(v + 1e-5f);
        for (uint d = 0; d < H; d++)
            lig_h_ctx[d] = (lig_h_ctx[d] - m) * is * ln_w[d] + ln_b[d];
    }

    // Reduction + prediction (same as Score kernel)
    threadgroup float tg_energy[64];
    threadgroup float tg_scratch[64];
    threadgroup float tg_repr[128];

    tg_energy[tid] = valid ? energy_sum : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_energy = 0;
    if (tid == 0) {
        for (uint i = 0; i < L; i++) total_energy += tg_energy[i];
    }

    for (uint d = 0; d < H; d++) {
        tg_scratch[tid] = valid ? lig_h_ctx[d] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0; for (uint i = 0; i < L; i++) s += tg_scratch[i];
            tg_repr[d] = s / float(L);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float ps = weights[entries[54].offset];
        float pb = weights[entries[55].offset];

        device const float* ah0_w = weights + entries[46].offset;
        device const float* ah0_b = weights + entries[47].offset;
        device const float* ah1_w = weights + entries[48].offset;
        device const float* ah1_b = weights + entries[49].offset;
        float ah[64];
        for (uint d = 0; d < 64; d++) {
            float v = ah0_b[d];
            for (uint d2 = 0; d2 < H; d2++) v += tg_repr[d2] * ah0_w[d * H + d2];
            ah[d] = gelu_tanh(v);
        }
        float ga = ah1_b[0];
        for (uint d = 0; d < 64; d++) ga += ah[d] * ah1_w[d];
        float pKd = total_energy * ps + ga + pb;

        device const float* ch0_w = weights + entries[50].offset;
        device const float* ch0_b = weights + entries[51].offset;
        device const float* ch1_w = weights + entries[52].offset;
        device const float* ch1_b = weights + entries[53].offset;
        float ch[64];
        for (uint d = 0; d < 64; d++) {
            float v = ch0_b[d];
            for (uint d2 = 0; d2 < H; d2++) v += tg_repr[d2] * ch0_w[d * H + d2];
            ch[d] = gelu_tanh(v);
        }
        float cl = ch1_b[0];
        for (uint d = 0; d < 64; d++) cl += ch[d] * ch1_w[d];
        float confidence = 1.0f / (1.0f + exp(-cl));

        output[0] = pKd;
        output[1] = confidence;
        output[2] = pKd * confidence;
    }
}
