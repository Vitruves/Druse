// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import simd
import Foundation

// MARK: - Quaternion

struct Quat {
    var w: Float
    var x: Float
    var y: Float
    var z: Float

    static let identity = Quat(w: 1, x: 0, y: 0, z: 0)

    var length: Float {
        sqrtf(w * w + x * x + y * y + z * z)
    }

    var normalized: Quat {
        let len = length
        guard len > 1e-10 else { return .identity }
        return Quat(w: w / len, x: x / len, y: y / len, z: z / len)
    }

    var conjugate: Quat {
        Quat(w: w, x: -x, y: -y, z: -z)
    }

    static func fromAxisAngle(_ axis: SIMD3<Float>, angle: Float) -> Quat {
        let halfAngle = angle * 0.5
        let s = sinf(halfAngle)
        let a = simd_normalize(axis)
        return Quat(w: cosf(halfAngle), x: a.x * s, y: a.y * s, z: a.z * s)
    }

    static func * (lhs: Quat, rhs: Quat) -> Quat {
        Quat(
            w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
            x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
            z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w
        )
    }

    func rotate(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let qv = SIMD3<Float>(x, y, z)
        let uv = simd_cross(qv, v)
        let uuv = simd_cross(qv, uv)
        return v + 2.0 * (w * uv + uuv)
    }

    var toMatrix: float4x4 {
        let q = self.normalized
        let xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z
        let xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z
        let wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z
        return float4x4(columns: (
            SIMD4<Float>(1 - 2*(yy+zz), 2*(xy+wz),     2*(xz-wy),     0),
            SIMD4<Float>(2*(xy-wz),     1 - 2*(xx+zz), 2*(yz+wx),     0),
            SIMD4<Float>(2*(xz+wy),     2*(yz-wx),     1 - 2*(xx+yy), 0),
            SIMD4<Float>(0,             0,             0,             1)
        ))
    }

    static func slerp(from q0: Quat, to q1: Quat, t: Float) -> Quat {
        var dot = q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z
        var q1 = q1
        if dot < 0 {
            q1 = Quat(w: -q1.w, x: -q1.x, y: -q1.y, z: -q1.z)
            dot = -dot
        }
        if dot > 0.9995 {
            return Quat(
                w: q0.w + t * (q1.w - q0.w),
                x: q0.x + t * (q1.x - q0.x),
                y: q0.y + t * (q1.y - q0.y),
                z: q0.z + t * (q1.z - q0.z)
            ).normalized
        }
        let theta0 = acosf(dot)
        let theta = theta0 * t
        let sinTheta = sinf(theta)
        let sinTheta0 = sinf(theta0)
        let s0 = cosf(theta) - dot * sinTheta / sinTheta0
        let s1 = sinTheta / sinTheta0
        return Quat(
            w: s0 * q0.w + s1 * q1.w,
            x: s0 * q0.x + s1 * q1.x,
            y: s0 * q0.y + s1 * q1.y,
            z: s0 * q0.z + s1 * q1.z
        ).normalized
    }
}

// MARK: - Matrix Utilities

enum Mat4 {
    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
        let yScale = 1.0 / tanf(fovY * 0.5)
        let xScale = yScale / aspect
        let zRange = far - near
        return float4x4(columns: (
            SIMD4<Float>(xScale, 0,      0,                          0),
            SIMD4<Float>(0,      yScale, 0,                          0),
            SIMD4<Float>(0,      0,      -(far + near) / zRange,    -1),
            SIMD4<Float>(0,      0,      -2 * far * near / zRange,   0)
        ))
    }

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
        let f = simd_normalize(center - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return float4x4(columns: (
            SIMD4<Float>(s.x,            u.x,            -f.x,           0),
            SIMD4<Float>(s.y,            u.y,            -f.y,           0),
            SIMD4<Float>(s.z,            u.z,            -f.z,           0),
            SIMD4<Float>(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)
        ))
    }

    static func translation(_ t: SIMD3<Float>) -> float4x4 {
        float4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        ))
    }

    static func scale(_ s: SIMD3<Float>) -> float4x4 {
        float4x4(columns: (
            SIMD4<Float>(s.x, 0,   0,   0),
            SIMD4<Float>(0,   s.y, 0,   0),
            SIMD4<Float>(0,   0,   s.z, 0),
            SIMD4<Float>(0,   0,   0,   1)
        ))
    }

    static let identity = float4x4(diagonal: SIMD4<Float>(repeating: 1))

    static func upperLeft3x3(_ m: float4x4) -> float3x3 {
        float3x3(columns: (
            SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z),
            SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z),
            SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z)
        ))
    }

    static func normalMatrix(_ modelView: float4x4) -> float4x4 {
        let upper = upperLeft3x3(modelView)
        let inv = upper.inverse.transpose
        return float4x4(columns: (
            SIMD4<Float>(inv.columns.0, 0),
            SIMD4<Float>(inv.columns.1, 0),
            SIMD4<Float>(inv.columns.2, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }
}

// MARK: - Geometry Utilities

func centroid(_ points: [SIMD3<Float>]) -> SIMD3<Float> {
    guard !points.isEmpty else { return .zero }
    var sum = SIMD3<Float>.zero
    for p in points { sum += p }
    return sum / Float(points.count)
}

func boundingBox(_ points: [SIMD3<Float>]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
    guard let first = points.first else {
        return (.zero, .zero)
    }
    var lo = first, hi = first
    for p in points {
        lo = simd_min(lo, p)
        hi = simd_max(hi, p)
    }
    return (lo, hi)
}

func boundingRadius(positions: [SIMD3<Float>], center: SIMD3<Float>) -> Float {
    var maxR: Float = 0
    for p in positions {
        let r = simd_length(p - center)
        if r > maxR { maxR = r }
    }
    return maxR
}

func raySphereIntersect(
    rayOrigin: SIMD3<Float>,
    rayDir: SIMD3<Float>,
    sphereCenter: SIMD3<Float>,
    sphereRadius: Float
) -> Float? {
    let oc = rayOrigin - sphereCenter
    let a = simd_dot(rayDir, rayDir)
    let b = 2.0 * simd_dot(oc, rayDir)
    let c = simd_dot(oc, oc) - sphereRadius * sphereRadius
    let disc = b * b - 4 * a * c
    guard disc >= 0 else { return nil }
    let t = (-b - sqrtf(disc)) / (2.0 * a)
    return t > 0 ? t : nil
}
