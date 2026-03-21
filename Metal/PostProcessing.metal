#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Post-Processing Parameters
// ============================================================================

struct PostProcessParams {
    float  ssaoRadius;
    float  ssaoBias;
    int    ssaoNumSamples;
    float  ssaoStrength;
    float  edgeStrength;
    float  nearPlane;
    float  farPlane;
    uint   width;
    uint   height;
    float  dofFocalDistance;   // focus distance from camera
    float  dofFocalRange;     // range around focal distance that is sharp
    float  dofMaxBlur;        // maximum blur radius in pixels
};

// ============================================================================
// MARK: - SSAO Compute
// ============================================================================

// 4x4 noise values for kernel rotation (baked in to avoid texture lookup)
constant float2 noiseValues[16] = {
    float2(-0.7071f,  0.7071f), float2( 0.5000f,  0.8660f),
    float2(-0.9659f, -0.2588f), float2( 0.2588f, -0.9659f),
    float2( 0.8660f,  0.5000f), float2(-0.2588f,  0.9659f),
    float2( 0.9659f, -0.2588f), float2(-0.5000f, -0.8660f),
    float2( 0.7071f,  0.7071f), float2(-0.8660f,  0.5000f),
    float2( 0.2588f,  0.9659f), float2(-0.7071f, -0.7071f),
    float2( 0.5000f, -0.8660f), float2( 0.9659f,  0.2588f),
    float2(-0.5000f,  0.8660f), float2(-0.9659f,  0.2588f)
};

// Hemisphere sample kernel (16 samples, biased toward surface)
constant float3 ssaoKernel[16] = {
    float3( 0.5381f,  0.1856f, 0.4319f), float3( 0.1379f,  0.2486f, 0.4430f),
    float3( 0.3371f,  0.5679f, 0.0057f), float3(-0.6999f, -0.0451f, 0.0019f),
    float3( 0.0689f, -0.1598f, 0.8547f), float3( 0.0560f,  0.0069f, 0.1843f),
    float3(-0.0146f,  0.1402f, 0.0762f), float3( 0.0100f, -0.1924f, 0.0344f),
    float3(-0.3577f, -0.5301f, 0.4358f), float3(-0.3169f,  0.1063f, 0.0158f),
    float3( 0.0103f, -0.5869f, 0.0046f), float3(-0.0897f, -0.4940f, 0.3287f),
    float3( 0.7119f, -0.0154f, 0.0918f), float3(-0.0533f,  0.0596f, 0.5411f),
    float3( 0.0352f, -0.0631f, 0.5460f), float3(-0.4776f,  0.2847f, 0.0271f)
};

/// Linearize depth from depth buffer value using near/far planes.
inline float linearizeDepth(float depth, float near, float far) {
    return near * far / (far - depth * (far - near));
}

/// Reconstruct view-space normal from depth buffer using cross-product of neighbors.
inline float3 reconstructNormal(texture2d<float, access::read> depthTex,
                                uint2 coord, uint width, uint height,
                                float near, float far) {
    float d  = linearizeDepth(depthTex.read(coord).r, near, far);
    float dR = linearizeDepth(depthTex.read(uint2(min(coord.x + 1, width - 1), coord.y)).r, near, far);
    float dU = linearizeDepth(depthTex.read(uint2(coord.x, min(coord.y + 1, height - 1))).r, near, far);

    float3 dx = float3(1.0f / float(width), 0.0f, dR - d);
    float3 dy = float3(0.0f, 1.0f / float(height), dU - d);

    float3 n = normalize(cross(dx, dy));
    return n;
}

/// SSAO compute shader.
/// For each pixel, sample hemisphere around surface normal; compare depth to estimate occlusion.
kernel void ssaoCompute(
    texture2d<float, access::read>  depthTexture  [[texture(0)]],
    texture2d<float, access::write> ssaoTexture   [[texture(1)]],
    constant PostProcessParams     &params        [[buffer(0)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float depth = depthTexture.read(gid).r;

    // Skip background pixels (depth ~= 1.0)
    if (depth > 0.999f) {
        ssaoTexture.write(float4(1.0f), gid);
        return;
    }

    float linearDepth = linearizeDepth(depth, params.nearPlane, params.farPlane);
    float3 normal = reconstructNormal(depthTexture, gid, params.width, params.height,
                                       params.nearPlane, params.farPlane);

    // Get noise rotation vector from 4x4 tile
    uint noiseIdx = (gid.x % 4) + (gid.y % 4) * 4;
    float2 noise = noiseValues[noiseIdx];

    // Build TBN from noise and normal
    float3 tangent = normalize(float3(noise, 0.0f) - normal * dot(float3(noise, 0.0f), normal));
    float3 bitangent = cross(normal, tangent);

    float occlusion = 0.0f;
    float radius = params.ssaoRadius;
    float bias = params.ssaoBias;

    for (int i = 0; i < params.ssaoNumSamples; i++) {
        // Orient sample in tangent space
        float3 sampleVec = tangent * ssaoKernel[i].x +
                           bitangent * ssaoKernel[i].y +
                           normal * ssaoKernel[i].z;
        sampleVec = sampleVec * radius;

        // Project sample to screen space
        float2 offset = sampleVec.xy / linearDepth;
        int2 sampleCoord = int2(gid) + int2(offset * float2(float(params.width), float(params.height)));

        // Clamp to texture bounds
        sampleCoord = clamp(sampleCoord, int2(0), int2(params.width - 1, params.height - 1));

        float sampleDepthRaw = depthTexture.read(uint2(sampleCoord)).r;
        float sampleDepth = linearizeDepth(sampleDepthRaw, params.nearPlane, params.farPlane);

        // Range check to avoid over-darkening at depth discontinuities
        float rangeCheck = smoothstep(0.0f, 1.0f, radius / abs(linearDepth - sampleDepth));
        occlusion += (sampleDepth <= linearDepth - bias + sampleVec.z ? 1.0f : 0.0f) * rangeCheck;
    }

    occlusion = 1.0f - (occlusion / float(params.ssaoNumSamples));
    ssaoTexture.write(float4(occlusion, occlusion, occlusion, 1.0f), gid);
}

// ============================================================================
// MARK: - SSAO Bilateral Blur
// ============================================================================

/// Bilateral blur (4x4 kernel) for SSAO texture; preserves edges using depth comparison.
kernel void ssaoBlur(
    texture2d<float, access::read>  ssaoInput     [[texture(0)]],
    texture2d<float, access::read>  depthTexture  [[texture(1)]],
    texture2d<float, access::write> ssaoOutput    [[texture(2)]],
    constant PostProcessParams     &params        [[buffer(0)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float centerDepth = depthTexture.read(gid).r;
    float centerLinear = linearizeDepth(centerDepth, params.nearPlane, params.farPlane);

    float totalWeight = 0.0f;
    float totalAO = 0.0f;

    for (int dy = -2; dy <= 1; dy++) {
        for (int dx = -2; dx <= 1; dx++) {
            int2 coord = int2(gid) + int2(dx, dy);
            coord = clamp(coord, int2(0), int2(params.width - 1, params.height - 1));

            float sampleAO = ssaoInput.read(uint2(coord)).r;
            float sampleDepth = depthTexture.read(uint2(coord)).r;
            float sampleLinear = linearizeDepth(sampleDepth, params.nearPlane, params.farPlane);

            // Bilateral weight: closer depth => higher weight
            float depthDiff = abs(centerLinear - sampleLinear);
            float weight = exp(-depthDiff * 100.0f);

            totalAO += sampleAO * weight;
            totalWeight += weight;
        }
    }

    float result = totalWeight > 0.0f ? totalAO / totalWeight : 1.0f;
    ssaoOutput.write(float4(result, result, result, 1.0f), gid);
}

// ============================================================================
// MARK: - Edge Detection (Sobel on Depth)
// ============================================================================

/// Sobel edge detection on depth buffer.
kernel void edgeDetect(
    texture2d<float, access::read>  depthTexture  [[texture(0)]],
    texture2d<float, access::write> edgeTexture   [[texture(1)]],
    constant PostProcessParams     &params        [[buffer(0)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    // Read 3x3 neighborhood of linearized depth
    float samples[9];
    int idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 coord = clamp(int2(gid) + int2(dx, dy), int2(0),
                               int2(params.width - 1, params.height - 1));
            samples[idx] = linearizeDepth(depthTexture.read(uint2(coord)).r,
                                          params.nearPlane, params.farPlane);
            idx++;
        }
    }

    // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
    float gx = -samples[0] + samples[2]
             - 2.0f * samples[3] + 2.0f * samples[5]
             - samples[6] + samples[8];

    // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
    float gy = -samples[0] - 2.0f * samples[1] - samples[2]
             + samples[6] + 2.0f * samples[7] + samples[8];

    // Normalize by center depth to make edges scale-independent
    float centerDepth = max(samples[4], 0.01f);
    float edgeStrength = sqrt(gx * gx + gy * gy) / centerDepth;

    // Threshold and smooth
    edgeStrength = smoothstep(0.01f, 0.1f, edgeStrength);

    edgeTexture.write(float4(edgeStrength, edgeStrength, edgeStrength, 1.0f), gid);
}

// ============================================================================
// MARK: - Texture Fill Utility
// ============================================================================

/// Fill a single-channel (R32Float) texture with a constant value.
kernel void fillTexture(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant float                 &fillValue     [[buffer(0)]],
    constant PostProcessParams     &params        [[buffer(1)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;
    outputTexture.write(float4(fillValue, fillValue, fillValue, 1.0f), gid);
}

// ============================================================================
// MARK: - Depth of Field
// ============================================================================

// Poisson-disc sampling pattern for bokeh-style DOF blur (16 samples)
constant float2 poissonDisc[16] = {
    float2( 0.0000f,  0.0000f),
    float2( 0.5272f,  0.2112f),
    float2(-0.2399f,  0.5040f),
    float2(-0.4798f, -0.2112f),
    float2( 0.2399f, -0.5040f),
    float2( 0.8660f, -0.5000f),
    float2( 0.0000f,  1.0000f),
    float2(-0.8660f,  0.5000f),
    float2(-0.8660f, -0.5000f),
    float2( 0.0000f, -1.0000f),
    float2( 0.8660f,  0.5000f),
    float2( 0.3535f,  0.3535f),
    float2(-0.3535f,  0.3535f),
    float2(-0.3535f, -0.3535f),
    float2( 0.3535f, -0.3535f),
    float2( 0.7071f,  0.0000f)
};

/// Circle-of-confusion (CoC) based depth of field.
/// Uses focal distance and range to determine blur amount, then applies
/// a variable-radius Gaussian blur weighted by the CoC at each pixel.
kernel void depthOfField(
    texture2d<float, access::read>  colorTexture  [[texture(0)]],
    texture2d<float, access::read>  depthTexture  [[texture(1)]],
    texture2d<float, access::write> dofTexture    [[texture(2)]],
    constant PostProcessParams     &params        [[buffer(0)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float depth = depthTexture.read(gid).r;
    float linearDepth = linearizeDepth(depth, params.nearPlane, params.farPlane);

    // Circle of confusion: how far from focal plane
    float focalDist = params.dofFocalDistance;
    float focalRange = params.dofFocalRange;
    float maxBlur = params.dofMaxBlur;

    // CoC: 0 at focal distance, grows linearly outside focal range
    float distFromFocus = abs(linearDepth - focalDist);
    float coc = clamp((distFromFocus - focalRange * 0.5f) / max(focalRange, 0.1f), 0.0f, 1.0f);
    float blurRadius = coc * maxBlur;

    // Skip blur for sharp pixels
    if (blurRadius < 0.5f) {
        dofTexture.write(colorTexture.read(gid), gid);
        return;
    }

    // Variable-radius Gaussian blur using Poisson-disc sampling
    float4 colorSum = float4(0.0f);
    float weightSum = 0.0f;

    for (int i = 0; i < 16; i++) {
        float2 offset = poissonDisc[i] * blurRadius;
        int2 sampleCoord = int2(gid) + int2(offset);
        sampleCoord = clamp(sampleCoord, int2(0), int2(params.width - 1, params.height - 1));

        // Read sample depth and compute its CoC to avoid bleeding sharp objects into blur
        float sampleDepthRaw = depthTexture.read(uint2(sampleCoord)).r;
        float sampleLinear = linearizeDepth(sampleDepthRaw, params.nearPlane, params.farPlane);
        float sampleDistFromFocus = abs(sampleLinear - focalDist);
        float sampleCoC = clamp((sampleDistFromFocus - focalRange * 0.5f) / max(focalRange, 0.1f), 0.0f, 1.0f);

        // Weight: Gaussian falloff * only accept blurry samples or center
        float dist = length(offset);
        float gaussWeight = exp(-dist * dist / max(blurRadius * blurRadius * 0.5f, 0.01f));

        // Prevent sharp foreground from leaking into bokeh background
        float cocWeight = (sampleCoC >= coc * 0.5f || i == 0) ? 1.0f : 0.2f;
        float weight = gaussWeight * cocWeight;

        colorSum += colorTexture.read(uint2(sampleCoord)) * weight;
        weightSum += weight;
    }

    float4 blurredColor = weightSum > 0.0f ? colorSum / weightSum : colorTexture.read(gid);
    dofTexture.write(blurredColor, gid);
}

// ============================================================================
// MARK: - Composite Post-Processing
// ============================================================================

/// Composite final color from color, SSAO, edge, and DOF textures.
/// Applies tone mapping (ACES filmic) for HDR-like output.
kernel void composite(
    texture2d<float, access::read>  colorTexture  [[texture(0)]],
    texture2d<float, access::read>  ssaoTexture   [[texture(1)]],
    texture2d<float, access::read>  edgeTexture   [[texture(2)]],
    texture2d<float, access::read>  dofTexture    [[texture(3)]],
    texture2d<float, access::write> outputTexture [[texture(4)]],
    constant PostProcessParams     &params        [[buffer(0)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    // Use DOF-blurred color if available, otherwise use raw color
    float4 color = dofTexture.read(gid);
    // Fallback: if DOF texture is black (unused), use raw color
    if (color.r + color.g + color.b < 1e-6f && color.a < 1e-6f) {
        color = colorTexture.read(gid);
    }

    float ssao = ssaoTexture.read(gid).r;
    float edge = edgeTexture.read(gid).r;

    // Apply SSAO darkening
    float aoFactor = 1.0f - params.ssaoStrength * (1.0f - ssao);

    // Apply edge darkening for silhouette effect
    float edgeFactor = 1.0f - params.edgeStrength * edge;

    float3 finalColor = color.rgb * aoFactor * edgeFactor;

    // Edge silhouette: darken toward near-black
    finalColor = mix(finalColor, float3(0.02f, 0.02f, 0.04f), edge * params.edgeStrength * 0.5f);

    // ACES filmic tone mapping (approximation by Krzysztof Narkowicz)
    // Ensures HDR-like rolloff without clipping highlights
    float3 x = finalColor;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    finalColor = clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);

    outputTexture.write(float4(finalColor, color.a), gid);
}

/// Legacy composite without DOF (backward compatibility).
kernel void compositePostProcess(
    texture2d<float, access::read>  colorTexture  [[texture(0)]],
    texture2d<float, access::read>  ssaoTexture   [[texture(1)]],
    texture2d<float, access::read>  edgeTexture   [[texture(2)]],
    texture2d<float, access::write> outputTexture [[texture(3)]],
    constant PostProcessParams     &params        [[buffer(0)]],
    uint2                           gid           [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 color = colorTexture.read(gid);
    float  ssao  = ssaoTexture.read(gid).r;
    float  edge  = edgeTexture.read(gid).r;

    float aoFactor = 1.0f - params.ssaoStrength * (1.0f - ssao);
    float edgeFactor = 1.0f - params.edgeStrength * edge;

    float3 finalColor = color.rgb * aoFactor * edgeFactor;
    finalColor = mix(finalColor, float3(0.02f, 0.02f, 0.04f), edge * params.edgeStrength * 0.5f);

    outputTexture.write(float4(finalColor, color.a), gid);
}
