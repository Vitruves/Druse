#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Diffusion-Guided Docking
//
// Uses DruseAF attention maps as proxy gradients to guide reverse diffusion
// from random noise toward high-affinity binding poses.
//
// Algorithm:
// 1. Initialize N poses from noise (scaled by sigma_T)
// 2. For each denoising step t = T..1:
//    a. Run DruseAF forward pass to get attention-weighted protein positions
//    b. Compute proxy gradient: direction from ligand atom to attention centroid
//    c. Update translation, rotation, torsions using scaled gradient
//    d. Add noise at level sigma_{t-1}
// 3. Final refinement with Vina analytical gradient local search
// ============================================================================

// ---- RNG ----
inline float diffRandom(thread uint &seed, uint offset) {
    seed = seed * 1103515245u + 12345u + offset;
    return float(seed & 0x7FFFFFFFu) / float(0x7FFFFFFF);
}

inline float3 diffGaussian3(thread uint &seed, uint base) {
    // Box-Muller transform for approximate Gaussian
    float u1 = max(diffRandom(seed, base), 1e-7f);
    float u2 = diffRandom(seed, base + 1u) * 2.0f * M_PI_F;
    float r = sqrt(-2.0f * log(u1));
    float u3 = max(diffRandom(seed, base + 2u), 1e-7f);
    float u4 = diffRandom(seed, base + 3u) * 2.0f * M_PI_F;
    float r2 = sqrt(-2.0f * log(u3));
    return float3(r * cos(u2), r * sin(u2), r2 * cos(u4));
}

// Quaternion rotation of a vector
inline float3 diffQuatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

// ============================================================================
// MARK: - Noise Initialization
// ============================================================================

/// Initialize diffusion poses from noise.
/// Translation: Gaussian centered on pocket with sigma = pocket_half_extent × noiseScale
/// Rotation: Random quaternion
/// Torsions: Uniform [-pi, pi]
kernel void diffusionInitNoise(
    device DockPose            *poses       [[buffer(0)]],
    constant GridParams        &gridParams  [[buffer(1)]],
    constant DiffusionParams   &diffParams  [[buffer(2)]],
    constant GAParams          &gaParams    [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    if (tid >= diffParams.numPoses) return;

    uint seed = tid * 314159u + 271828u + 42u;
    device DockPose &pose = poses[tid];

    float3 center = gridParams.searchCenter;
    float3 halfExtent = gridParams.searchHalfExtent;

    // Translation: Gaussian noise scaled by pocket size and noise schedule
    float sigma = diffParams.translationNoise;
    float3 noise = diffGaussian3(seed, 10);
    pose.translation = center + noise * halfExtent * sigma;

    // Clamp to grid bounds
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;
    pose.translation = clamp(pose.translation, gMin, gMax);

    // Rotation: random quaternion (Shoemake method)
    float u1 = diffRandom(seed, 20);
    float u2 = diffRandom(seed, 21) * 2.0f * M_PI_F;
    float u3 = diffRandom(seed, 22) * 2.0f * M_PI_F;
    float sq1 = sqrt(1.0f - u1);
    float sq2 = sqrt(u1);
    pose.rotation = float4(sq1 * sin(u2), sq1 * cos(u2), sq2 * sin(u3), sq2 * cos(u3));

    // Torsions: uniform random
    for (uint t = 0; t < gaParams.numTorsions && t < 32; t++) {
        pose.torsions[t] = diffRandom(seed, 30 + t) * 2.0f * M_PI_F - M_PI_F;
    }

    pose.numTorsions = int(gaParams.numTorsions);
    pose.energy = 1e10f;
    pose.generation = 0;
    pose.stericEnergy = 0.0f;
    pose.hydrophobicEnergy = 0.0f;
    pose.hbondEnergy = 0.0f;
    pose.torsionPenalty = 0.0f;
    pose.clashPenalty = 0.0f;
    pose.drusinaCorrection = 0.0f;
    pose.constraintPenalty = 0.0f;
    pose.numChiAngles = 0;
}

// ============================================================================
// MARK: - Attention-Guided Denoising Step
// ============================================================================

/// Apply one denoising step using DruseAF attention gradients.
/// For each pose:
///   1. Read per-atom attention gradient (pullDirection from DruseAF)
///   2. Compute net translation gradient (mean of all atom pull directions)
///   3. Compute net rotation gradient (torque from pull directions about center)
///   4. Update pose with scaled gradient step
///   5. Add noise at current noise level
kernel void diffusionDenoisingStep(
    device DockPose                *poses         [[buffer(0)]],
    device const AttentionGradient *attnGrads     [[buffer(1)]],  // [numPoses × numLigandAtoms]
    constant DiffusionParams       &diffParams    [[buffer(2)]],
    constant GAParams              &gaParams      [[buffer(3)]],
    constant GridParams            &gridParams    [[buffer(4)]],
    constant DockLigandAtom        *ligandAtoms   [[buffer(5)]],
    uint                            tid           [[thread_position_in_grid]])
{
    if (tid >= diffParams.numPoses) return;

    device DockPose &pose = poses[tid];
    uint nAtoms = diffParams.numLigandAtoms;
    uint nTorsions = diffParams.numTorsions;
    float guidanceScale = diffParams.guidanceScale;
    float noiseScale = diffParams.noiseScale;

    uint seed = tid * 482711u + diffParams.currentStep * 937373u;

    // Read attention gradients for this pose
    uint gradBase = tid * nAtoms;

    // Compute net translation gradient: mean pull direction across all ligand atoms
    float3 netTransGrad = float3(0.0f);
    float3 netTorque = float3(0.0f);
    float totalMagnitude = 0.0f;

    for (uint a = 0; a < nAtoms && a < 128; a++) {
        device const AttentionGradient &ag = attnGrads[gradBase + a];
        float3 pull = ag.pullDirection;
        float mag = ag.pullMagnitude;

        netTransGrad += pull * mag;
        totalMagnitude += mag;

        // Torque contribution for rotation gradient
        float3 atomPos = ligandAtoms[a].position; // centroid-subtracted
        float3 worldPos = diffQuatRotate(pose.rotation, atomPos);
        netTorque += cross(worldPos, pull * mag);
    }

    if (totalMagnitude > 1e-6f) {
        netTransGrad /= totalMagnitude;
        netTorque /= totalMagnitude;
    }

    // Step 1: Update translation
    float transStep = guidanceScale * 0.5f; // Angstroms per step
    pose.translation += netTransGrad * transStep;

    // Clamp to grid
    float3 gMin = gridParams.searchCenter - gridParams.searchHalfExtent;
    float3 gMax = gridParams.searchCenter + gridParams.searchHalfExtent;
    pose.translation = clamp(pose.translation, gMin, gMax);

    // Step 2: Update rotation via torque → axis-angle perturbation
    float torqueMag = length(netTorque);
    if (torqueMag > 1e-6f) {
        float3 axis = netTorque / torqueMag;
        float angle = min(torqueMag * guidanceScale * 0.1f, 0.3f); // cap rotation step
        float halfA = angle * 0.5f;
        float4 dq = float4(axis * sin(halfA), cos(halfA));
        float4 sq = pose.rotation;
        pose.rotation = normalize(float4(
            dq.w*sq.x + dq.x*sq.w + dq.y*sq.z - dq.z*sq.y,
            dq.w*sq.y - dq.x*sq.z + dq.y*sq.w + dq.z*sq.x,
            dq.w*sq.z + dq.x*sq.y - dq.y*sq.x + dq.z*sq.w,
            dq.w*sq.w - dq.x*sq.x - dq.y*sq.y - dq.z*sq.z
        ));
    }

    // Step 3: Add noise for the next diffusion step (unless final step)
    if (noiseScale > 1e-6f) {
        float3 transNoise = diffGaussian3(seed, 100) * noiseScale * diffParams.translationNoise;
        pose.translation += transNoise;
        pose.translation = clamp(pose.translation, gMin, gMax);

        // Rotation noise
        float rotNoiseMag = noiseScale * diffParams.rotationNoise;
        float3 rotNoiseVec = diffGaussian3(seed, 110) * rotNoiseMag;
        float rotAngle = length(rotNoiseVec);
        if (rotAngle > 1e-6f) {
            float3 rotAxis = rotNoiseVec / rotAngle;
            float halfRot = rotAngle * 0.5f;
            float4 nq = float4(rotAxis * sin(halfRot), cos(halfRot));
            float4 sq = pose.rotation;
            pose.rotation = normalize(float4(
                nq.w*sq.x + nq.x*sq.w + nq.y*sq.z - nq.z*sq.y,
                nq.w*sq.y - nq.x*sq.z + nq.y*sq.w + nq.z*sq.x,
                nq.w*sq.z + nq.x*sq.y - nq.y*sq.x + nq.z*sq.w,
                nq.w*sq.w - nq.x*sq.x - nq.y*sq.y - nq.z*sq.z
            ));
        }

        // Torsion noise
        float torNoiseMag = noiseScale * diffParams.torsionNoise;
        for (uint t = 0; t < nTorsions && t < 32; t++) {
            float torNoise = (diffRandom(seed, 120 + t) - 0.5f) * torNoiseMag * 2.0f;
            pose.torsions[t] += torNoise;
            // Wrap to [-pi, pi]
            float v = pose.torsions[t];
            pose.torsions[t] = v - 2.0f * M_PI_F * floor((v + M_PI_F) / (2.0f * M_PI_F));
        }
    }

    pose.energy = 1e10f; // will be re-scored
    pose.generation = int(diffParams.currentStep);
}
