#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;

// ============================================================================
// MARK: - Background Gradient
// ============================================================================

struct BackgroundVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex BackgroundVertexOut backgroundVertex(uint vertexID [[vertex_id]]) {
    // Fullscreen triangle strip: 4 vertices covering clip space
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    float2 texCoords[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0)
    };

    BackgroundVertexOut out;
    out.position = float4(positions[vertexID], 0.9999, 1.0);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment float4 backgroundFragment(BackgroundVertexOut in [[stage_in]]) {
    // Dark gradient: bottom dark → top slightly lighter
    float3 bottomColor = float3(0.06, 0.07, 0.10);
    float3 topColor    = float3(0.14, 0.16, 0.22);
    float t = in.texCoord.y;
    // Slight radial vignette
    float2 center = in.texCoord - float2(0.5);
    float vignette = 1.0 - 0.3 * dot(center, center);
    float3 color = mix(bottomColor, topColor, t) * vignette;
    return float4(color, 1.0);
}

// ============================================================================
// MARK: - Atom Impostor Sphere
// ============================================================================

struct AtomVertexOut {
    float4 position [[position]];
    float3 viewCenter;      // sphere center in view space
    float  radius;
    float4 color;
    float2 quadCoord;       // [-1,1] billboard coordinates
    int    atomIndex;
    int    flags;
    float  expandedRadius;
};

struct AtomFragmentOut {
    float4 color [[color(0)]];
    float  depth [[depth(any)]];
};

vertex AtomVertexOut atomVertex(
    uint vertexID   [[vertex_id]],
    uint instanceID [[instance_id]],
    constant AtomInstance *instances [[buffer(BufferIndexInstances)]],
    constant Uniforms     &uniforms [[buffer(BufferIndexUniforms)]]
) {
    AtomInstance inst = instances[instanceID];

    // Billboard quad corners (triangle strip: 4 vertices)
    float2 corners[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    float2 corner = corners[vertexID];

    // Transform sphere center to view space
    float4 worldPos = uniforms.modelMatrix * float4(inst.position, 1.0);
    float4 viewPos  = uniforms.viewMatrix * worldPos;
    float3 viewCenter = viewPos.xyz;

    // Expand billboard slightly beyond radius for edge coverage
    float displayRadius = inst.radius;
    float expand = displayRadius * 1.35;

    // Offset in view space (camera-aligned billboard)
    float4 billboardPos = viewPos;
    billboardPos.xy += corner * expand;

    AtomVertexOut out;
    out.position       = uniforms.projectionMatrix * billboardPos;
    out.viewCenter     = viewCenter;
    out.radius         = displayRadius;
    out.color          = inst.color;
    out.quadCoord      = corner;
    out.atomIndex      = inst.atomIndex;
    out.flags          = inst.flags;
    out.expandedRadius = expand;
    return out;
}

fragment AtomFragmentOut atomFragment(
    AtomVertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    // Ray-sphere intersection in view space
    // Ray: origin = point on billboard, direction = (0, 0, -1)
    float3 rayOrigin = float3(in.viewCenter.xy + in.quadCoord * in.expandedRadius, in.viewCenter.z);

    float3 oc = rayOrigin - in.viewCenter;
    float r = in.radius;

    // Quadratic: t^2 + 2*b*t + c = 0  (a = 1, ray dir = (0,0,-1))
    // oc.z is the z component, ray dir is -z
    float b = -oc.z;  // dot(oc, (0,0,-1)) = -oc.z
    float c = dot(oc, oc) - r * r;
    float disc = b * b - c;

    if (disc < 0.0) {
        discard_fragment();
    }

    float t = b - sqrt(disc);  // nearest intersection

    // Hit point in view space
    float3 hitPoint = rayOrigin + float3(0.0, 0.0, -1.0) * t;
    float3 normal = normalize(hitPoint - in.viewCenter);

    // Z-slab clipping: discard fragments outside the slab range
    if (uniforms.enableClipping) {
        float viewZ = -hitPoint.z;  // view space: -z is forward (into screen)
        if (viewZ < uniforms.clipNearZ || viewZ > uniforms.clipFarZ)
            discard_fragment();
    }

    // Compute correct depth
    float4 clipPos = uniforms.projectionMatrix * float4(hitPoint, 1.0);
    float depth = clipPos.z / clipPos.w;

    float3 viewDir = float3(0.0, 0.0, 1.0); // camera looks along -z
    float3 baseColor = in.color.rgb;

    // Selection highlighting
    if (in.flags & 2) {
        baseColor = mix(baseColor, float3(0.0, 0.9, 0.9), 0.35);
    }
    if (in.flags & 1) {
        float pulse = 0.3 + 0.15 * sin(uniforms.time * 5.0);
        baseColor = mix(baseColor, float3(1.0, 0.95, 0.3), pulse);
    }

    float3 litColor;
    if (uniforms.lightingMode == 0) {
        // Uniform lighting: high ambient + soft view-dependent shading
        float NdotV = max(dot(normal, viewDir), 0.0);
        litColor = baseColor * (0.60 + 0.40 * NdotV);
    } else {
        // Directional Blinn-Phong
        float3 lightDir = normalize((uniforms.viewMatrix * float4(uniforms.lightDirection, 0.0)).xyz);
        float NdotL = max(dot(normal, lightDir), 0.0);
        float3 halfVec = normalize(lightDir + viewDir);
        float NdotH = max(dot(normal, halfVec), 0.0);
        float specular = pow(NdotH, 64.0) * 0.6;
        float NdotV = max(dot(normal, viewDir), 0.0);
        float rim = pow(1.0 - NdotV, 3.0) * 0.15;
        litColor = baseColor * (uniforms.ambientIntensity + NdotL * uniforms.lightColor)
                 + specular * uniforms.lightColor
                 + rim * float3(0.4, 0.5, 0.7);
    }

    AtomFragmentOut out;
    out.color = float4(litColor, in.color.a);  // pass through instance alpha (1.0 opaque, <1.0 ghost)
    out.depth = depth;
    return out;
}

// ============================================================================
// MARK: - GPU Object-ID Picking (Atom)
// ============================================================================

/// Fragment output for the pick pass: writes atom ID to a R32Uint texture.
struct AtomPickFragmentOut {
    uint   objectID [[color(0)]];
    float  depth    [[depth(any)]];
};

/// Pick-pass fragment shader: same ray-sphere test as atomFragment,
/// but outputs the atom index instead of a shaded color.
/// 0xFFFFFFFF = background (no hit).
fragment AtomPickFragmentOut atomPickFragment(
    AtomVertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    float3 rayOrigin = float3(in.viewCenter.xy + in.quadCoord * in.expandedRadius, in.viewCenter.z);
    float3 oc = rayOrigin - in.viewCenter;
    float r = in.radius;
    float b = -oc.z;
    float c = dot(oc, oc) - r * r;
    float disc = b * b - c;

    if (disc < 0.0) {
        discard_fragment();
    }

    float t = b - sqrt(disc);
    float3 hitPoint = rayOrigin + float3(0.0, 0.0, -1.0) * t;

    if (uniforms.enableClipping) {
        float viewZ = -hitPoint.z;
        if (viewZ < uniforms.clipNearZ || viewZ > uniforms.clipFarZ)
            discard_fragment();
    }

    float4 clipPos = uniforms.projectionMatrix * float4(hitPoint, 1.0);

    AtomPickFragmentOut out;
    out.objectID = uint(in.atomIndex);
    out.depth = clipPos.z / clipPos.w;
    return out;
}

// ============================================================================
// MARK: - Bond Impostor Cylinder
// ============================================================================

struct BondVertexOut {
    float4 position [[position]];
    float3 viewPosA;
    float3 viewPosB;
    float3 viewHitPos;
    float4 colorA;
    float4 colorB;
    float  radiusA;
    float  radiusB;
};

struct BondFragmentOut {
    float4 color [[color(0)]];
    float  depth [[depth(any)]];
};

vertex BondVertexOut bondVertex(
    uint vertexID   [[vertex_id]],
    uint instanceID [[instance_id]],
    constant BondInstance *instances [[buffer(BufferIndexInstances)]],
    constant Uniforms     &uniforms [[buffer(BufferIndexUniforms)]]
) {
    BondInstance inst = instances[instanceID];

    // Transform endpoints to view space
    float4 wA = uniforms.modelMatrix * float4(inst.positionA, 1.0);
    float4 wB = uniforms.modelMatrix * float4(inst.positionB, 1.0);
    float4 vA = uniforms.viewMatrix * wA;
    float4 vB = uniforms.viewMatrix * wB;

    float3 viewA = vA.xyz;
    float3 viewB = vB.xyz;

    // Build a frame around the bond axis in view space
    float3 axis = viewB - viewA;
    float bondLength = length(axis);
    float3 axisDir = axis / max(bondLength, 0.0001);

    // Find perpendicular directions
    float3 up = abs(axisDir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(axisDir, up));
    float3 upDir = cross(right, axisDir);

    float bondRadius = max(inst.radiusA, inst.radiusB);
    float expand = bondRadius * 1.5;

    // 8 vertices forming a box around the cylinder
    // vertexID 0-3: cap A, vertexID 4-7: cap B
    float3 basePos;
    if (vertexID < 4) {
        basePos = viewA - axisDir * expand;
    } else {
        basePos = viewB + axisDir * expand;
    }

    int cornerIdx = vertexID % 4;
    float2 offsets[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    float2 off = offsets[cornerIdx];
    float3 vertexPos = basePos + right * off.x * expand + upDir * off.y * expand;

    BondVertexOut out;
    out.position   = uniforms.projectionMatrix * float4(vertexPos, 1.0);
    out.viewPosA   = viewA;
    out.viewPosB   = viewB;
    out.viewHitPos = vertexPos;
    out.colorA     = inst.colorA;
    out.colorB     = inst.colorB;
    out.radiusA    = inst.radiusA;
    out.radiusB    = inst.radiusB;
    return out;
}

fragment BondFragmentOut bondFragment(
    BondVertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    // Ray-cylinder intersection in view space
    float3 ro = in.viewHitPos;
    float3 rd = float3(0.0, 0.0, -1.0);

    float3 pa = in.viewPosA;
    float3 pb = in.viewPosB;
    float3 ba = pb - pa;
    float baba = dot(ba, ba);
    float3 oc = ro - pa;

    float radius = max(in.radiusA, in.radiusB);

    float bard = dot(ba, rd);
    float baoc = dot(ba, oc);

    float k2 = baba - bard * bard;
    float k1 = baba * dot(oc, rd) - baoc * bard;
    float k0 = baba * dot(oc, oc) - baoc * baoc - radius * radius * baba;

    float disc = k1 * k1 - k2 * k0;
    if (disc < 0.0) {
        discard_fragment();
    }

    float t = (-k1 - sqrt(disc)) / k2;
    float y = baoc + t * bard;

    // Check if hit is within cylinder length
    if (y < 0.0 || y > baba) {
        // Try hemisphere caps
        // Cap at A
        float3 capNorm = normalize(-ba);
        float denom = dot(rd, capNorm);
        if (abs(denom) > 0.0001) {
            float tc = dot(pa - ro, capNorm) / denom;
            float3 hitC = ro + rd * tc;
            if (length(hitC - pa) <= radius && tc > 0.0) {
                float4 clipPos = uniforms.projectionMatrix * float4(hitC, 1.0);
                BondFragmentOut out;
                out.color = in.colorA;
                out.depth = clipPos.z / clipPos.w;

                // Simple lighting on cap
                float3 normal = capNorm;
                float3 lightDir = normalize((uniforms.viewMatrix * float4(uniforms.lightDirection, 0.0)).xyz);
                float NdotL = max(dot(normal, lightDir), 0.0);
                out.color = float4(in.colorA.rgb * (uniforms.ambientIntensity + NdotL * uniforms.lightColor), 1.0);
                return out;
            }
        }
        // Cap at B
        capNorm = normalize(ba);
        denom = dot(rd, capNorm);
        if (abs(denom) > 0.0001) {
            float tc = dot(pb - ro, capNorm) / denom;
            float3 hitC = ro + rd * tc;
            if (length(hitC - pb) <= radius && tc > 0.0) {
                float4 clipPos = uniforms.projectionMatrix * float4(hitC, 1.0);
                BondFragmentOut out;
                float3 normal = capNorm;
                float3 lightDir = normalize((uniforms.viewMatrix * float4(uniforms.lightDirection, 0.0)).xyz);
                float NdotL = max(dot(normal, lightDir), 0.0);
                out.color = float4(in.colorB.rgb * (uniforms.ambientIntensity + NdotL * uniforms.lightColor), 1.0);
                out.depth = clipPos.z / clipPos.w;
                return out;
            }
        }
        discard_fragment();
    }

    // Hit point on cylinder body
    float3 hitPoint = ro + rd * t;

    // Z-slab clipping
    if (uniforms.enableClipping) {
        float viewZ = -hitPoint.z;
        if (viewZ < uniforms.clipNearZ || viewZ > uniforms.clipFarZ)
            discard_fragment();
    }

    float3 normal = normalize(oc + t * rd - ba * y / baba);

    // Color gradient from A to B at midpoint
    float param = y / baba;
    float4 baseColor = param < 0.5 ? in.colorA : in.colorB;

    // Lighting
    float3 viewDir = float3(0.0, 0.0, 1.0);
    float3 litColor;

    if (uniforms.lightingMode == 0) {
        float NdotV = max(dot(normal, viewDir), 0.0);
        litColor = baseColor.rgb * (0.60 + 0.40 * NdotV);
    } else {
        float3 lightDir = normalize((uniforms.viewMatrix * float4(uniforms.lightDirection, 0.0)).xyz);
        float NdotL = max(dot(normal, lightDir), 0.0);
        float3 halfVec = normalize(lightDir + viewDir);
        float NdotH = max(dot(normal, halfVec), 0.0);
        float specular = pow(NdotH, 32.0) * 0.4;
        litColor = baseColor.rgb * (uniforms.ambientIntensity + NdotL * uniforms.lightColor)
                 + specular * uniforms.lightColor;
    }

    float4 clipPos = uniforms.projectionMatrix * float4(hitPoint, 1.0);

    BondFragmentOut out;
    out.color = float4(litColor, baseColor.a);  // pass through instance alpha (1.0 opaque, <1.0 ghost)
    out.depth = clipPos.z / clipPos.w;
    return out;
}

// ============================================================================
// MARK: - Ribbon Rendering
// ============================================================================

struct RibbonVertexOut {
    float4 position [[position]];
    float3 normal;
    float4 color;
    float3 viewPos;
    float2 texCoord;
};

vertex RibbonVertexOut ribbonVertex(
    uint vertexID [[vertex_id]],
    constant RibbonVertex *vertices [[buffer(BufferIndexVertices)]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    RibbonVertexOut out;
    RibbonVertex v = vertices[vertexID];

    float4x4 mv = uniforms.viewMatrix * uniforms.modelMatrix;
    float4 viewPos = mv * float4(v.position, 1.0);

    out.position = uniforms.projectionMatrix * viewPos;
    out.normal = normalize((uniforms.normalMatrix * float4(v.normal, 0.0)).xyz);
    out.color = v.color;
    out.viewPos = viewPos.xyz;
    out.texCoord = v.texCoord;

    return out;
}

fragment float4 ribbonFragment(
    RibbonVertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    // Z-slab clipping
    if (uniforms.enableClipping) {
        float viewZ = -in.viewPos.z;
        if (viewZ < uniforms.clipNearZ || viewZ > uniforms.clipFarZ)
            discard_fragment();
    }

    float3 normal = normalize(in.normal);
    float3 viewDir = normalize(-in.viewPos);

    float3 color;

    if (uniforms.lightingMode == 0) {
        // Uniform lighting
        float NdotV = max(dot(normal, viewDir), 0.0);
        float backNdotV = max(dot(-normal, viewDir), 0.0) * 0.4;
        color = in.color.rgb * (0.55 + 0.45 * (NdotV + backNdotV));
    } else {
        // Directional Blinn-Phong with two-sided lighting
        float3 lightDir = normalize(uniforms.lightDirection);
        float3 halfVec = normalize(lightDir + viewDir);

        float ambient = uniforms.ambientIntensity;
        float diffuse = max(dot(normal, lightDir), 0.0);
        float specular = pow(max(dot(normal, halfVec), 0.0), 48.0) * 0.5;
        float backDiffuse = max(dot(-normal, lightDir), 0.0) * 0.4;

        color = in.color.rgb * (ambient + (diffuse + backDiffuse) * uniforms.lightColor)
              + specular * uniforms.lightColor;
    }

    return float4(color, in.color.a);
}

// ============================================================================
// MARK: - Interaction Line Rendering (dashed lines)
// ============================================================================

struct InteractionVertexOut {
    float4 position [[position]];
    float4 color;
    float  dashParam; // 0..1 along the line, for dash pattern
};

/// Renders interaction lines as screen-space billboard quads (triangle strip,
/// 4 vertices per instance) so they have visible thickness regardless of zoom.
vertex InteractionVertexOut interactionLineVertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    constant InteractionLineVertex *lines [[buffer(BufferIndexInstances)]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    InteractionVertexOut out;
    InteractionLineVertex line = lines[instanceID];

    float4x4 mvp = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix;

    // Project both endpoints to clip space
    float4 clipA = mvp * float4(line.positionA, 1.0);
    float4 clipB = mvp * float4(line.positionB, 1.0);

    // NDC positions
    float2 ndcA = clipA.xy / clipA.w;
    float2 ndcB = clipB.xy / clipB.w;

    // Screen-space perpendicular for billboard expansion
    float2 dir = ndcB - ndcA;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : float2(1.0, 0.0);
    float2 perp = float2(-dir.y, dir.x);

    // Line half-width in NDC (roughly 2 pixels on a 1000px viewport)
    float halfWidth = 0.004;

    // Triangle strip: 0=A-perp, 1=A+perp, 2=B-perp, 3=B+perp
    bool isB = (vertexID >= 2);
    float side = (vertexID % 2 == 0) ? -1.0 : 1.0;

    float4 clipPos = isB ? clipB : clipA;
    clipPos.xy += side * perp * halfWidth * clipPos.w;

    out.position = clipPos;
    out.color = line.color;
    out.dashParam = isB ? 1.0 : 0.0;

    return out;
}

fragment float4 interactionLineFragment(
    InteractionVertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    // Dashed pattern: discard every other segment
    float dashFreq = 8.0; // dashes per line
    float phase = fract(in.dashParam * dashFreq + uniforms.time * 0.5);
    if (phase > 0.5) discard_fragment();

    return in.color;
}

// ============================================================================
// MARK: - Grid Box Wireframe
// ============================================================================

struct GridBoxVertexOut {
    float4 position [[position]];
    float4 color;
};

vertex GridBoxVertexOut gridBoxVertex(
    uint vertexID [[vertex_id]],
    constant GridBoxVertex *vertices [[buffer(BufferIndexVertices)]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    GridBoxVertexOut out;
    float4x4 mvp = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix;
    out.position = mvp * float4(vertices[vertexID].position, 1.0);
    out.color = vertices[vertexID].color;
    return out;
}

fragment float4 gridBoxFragment(GridBoxVertexOut in [[stage_in]]) {
    return in.color;
}

// ============================================================================
// MARK: - Molecular Surface Rendering
// ============================================================================

// Must match GPUSurfaceVertex in SurfaceGenerator.swift
struct SurfaceVertexIn {
    packed_float3 position;
    packed_float3 normal;
    float4 color;
};

struct SurfaceVertexOut {
    float4 position [[position]];
    float3 normal;
    float4 color;
    float3 viewPos;
};

vertex SurfaceVertexOut surfaceVertex(
    uint vertexID [[vertex_id]],
    constant SurfaceVertexIn *vertices [[buffer(BufferIndexVertices)]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    SurfaceVertexOut out;
    SurfaceVertexIn v = vertices[vertexID];

    float3 pos = float3(v.position);
    float3 norm = float3(v.normal);

    float4x4 mv = uniforms.viewMatrix * uniforms.modelMatrix;
    float4 viewPos = mv * float4(pos, 1.0);

    out.position = uniforms.projectionMatrix * viewPos;
    out.normal = normalize((uniforms.normalMatrix * float4(norm, 0.0)).xyz);
    out.color = v.color;
    out.viewPos = viewPos.xyz;

    return out;
}

fragment float4 surfaceFragment(
    SurfaceVertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]]
) {
    // Z-slab clipping
    if (uniforms.enableClipping) {
        float viewZ = -in.viewPos.z;
        if (viewZ < uniforms.clipNearZ || viewZ > uniforms.clipFarZ)
            discard_fragment();
    }

    float3 normal = normalize(in.normal);
    float3 viewDir = normalize(-in.viewPos);

    float3 color;

    if (uniforms.lightingMode == 0) {
        // Uniform lighting with soft shading
        float NdotV = max(dot(normal, viewDir), 0.0);
        float backNdotV = max(dot(-normal, viewDir), 0.0) * 0.3;
        color = in.color.rgb * (0.50 + 0.50 * (NdotV + backNdotV));
    } else {
        // Directional Blinn-Phong with two-sided lighting
        float3 lightDir = normalize(uniforms.lightDirection);
        float3 halfVec = normalize(lightDir + viewDir);

        float ambient = uniforms.ambientIntensity;
        float diffuse = max(dot(normal, lightDir), 0.0);
        float specular = pow(max(dot(normal, halfVec), 0.0), 32.0) * 0.4;
        float backDiffuse = max(dot(-normal, lightDir), 0.0) * 0.3;

        color = in.color.rgb * (ambient + (diffuse + backDiffuse) * uniforms.lightColor)
              + specular * uniforms.lightColor;
    }

    return float4(color, in.color.a);
}
