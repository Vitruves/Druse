// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import simd
import Foundation

final class Camera {
    // Arcball state
    var orientation: Quat = .identity
    var target: SIMD3<Float> = .zero
    var distance: Float = 15.0

    // Projection
    var fovY: Float = Float.pi / 4.0   // 45 degrees
    var aspectRatio: Float = 1.5
    var nearPlane: Float = 0.1
    var farPlane: Float = 500.0

    // Interaction state
    private var dragStartOrientation: Quat = .identity
    private var dragStartTarget: SIMD3<Float> = .zero
    private var dragStartPoint: SIMD2<Float> = .zero
    private var isDragging = false
    private var isPanning = false

    // Smooth animation
    private var targetOrientation: Quat = .identity
    private var targetTarget: SIMD3<Float> = .zero
    private var targetDistance: Float = 15.0
    private(set) var isAnimating = false
    private let smoothFactor: Float = 0.15

    // MARK: - Computed Matrices

    var eyePosition: SIMD3<Float> {
        let forward = orientation.rotate(SIMD3<Float>(0, 0, 1))
        return target + forward * distance
    }

    var viewMatrix: float4x4 {
        let eye = eyePosition
        let up = orientation.rotate(SIMD3<Float>(0, 1, 0))
        return Mat4.lookAt(eye: eye, center: target, up: up)
    }

    var projectionMatrix: float4x4 {
        Mat4.perspective(fovY: fovY, aspect: aspectRatio, near: nearPlane, far: farPlane)
    }

    var rightVector: SIMD3<Float> {
        orientation.rotate(SIMD3<Float>(1, 0, 0))
    }

    var upVector: SIMD3<Float> {
        orientation.rotate(SIMD3<Float>(0, 1, 0))
    }

    // MARK: - Rotation (Arcball)

    /// Whether the current drag is a Z-roll (Ctrl+drag) vs normal XY orbit.
    private var isZRoll = false

    func beginRotation(at screenPoint: SIMD2<Float>, zRoll: Bool = false) {
        isDragging = true
        isZRoll = zRoll
        dragStartOrientation = orientation
        dragStartPoint = screenPoint
    }

    func updateRotation(to screenPoint: SIMD2<Float>, viewportSize: SIMD2<Float>) {
        guard isDragging else { return }

        if isZRoll {
            // Ctrl+drag: rotate around the view axis (Z-roll / spin).
            // Angle is determined by the angular change of the cursor position
            // relative to the viewport center.
            let center = viewportSize * 0.5
            let startVec = dragStartPoint - center
            let currVec = screenPoint - center

            let startAngle = atan2(startVec.y, startVec.x)
            let currAngle = atan2(currVec.y, currVec.x)
            let rollAngle = currAngle - startAngle

            let viewForward = dragStartOrientation.rotate(SIMD3<Float>(0, 0, 1))
            let rotRoll = Quat.fromAxisAngle(viewForward, angle: rollAngle)
            orientation = (rotRoll * dragStartOrientation).normalized
        } else {
            // Normal drag: turntable-style orbit
            let delta = screenPoint - dragStartPoint
            let sensitivity: Float = 5.0 / max(viewportSize.x, viewportSize.y)

            let angleX = delta.x * sensitivity
            let angleY = delta.y * sensitivity

            // - Horizontal drag: orbit around global Y (keeps "up" consistent)
            // - Vertical drag: orbit around camera's local right axis (prevents gimbal lock)
            let rotYaw = Quat.fromAxisAngle(SIMD3<Float>(0, 1, 0), angle: -angleX)
            let cameraRight = dragStartOrientation.rotate(SIMD3<Float>(1, 0, 0))
            let rotPitch = Quat.fromAxisAngle(cameraRight, angle: -angleY)

            orientation = (rotYaw * rotPitch * dragStartOrientation).normalized
        }
    }

    func endRotation() {
        isDragging = false
        isZRoll = false
    }

    // MARK: - Pan

    func beginPan(at screenPoint: SIMD2<Float>) {
        isPanning = true
        dragStartTarget = target
        dragStartPoint = screenPoint
    }

    func updatePan(to screenPoint: SIMD2<Float>, viewportSize: SIMD2<Float>) {
        guard isPanning else { return }
        let delta = screenPoint - dragStartPoint
        let sensitivity = distance * tanf(fovY * 0.5) * 2.0 / viewportSize.y

        let right = rightVector
        let up = upVector
        let offset = right * (-delta.x * sensitivity) + up * (delta.y * sensitivity)
        target = dragStartTarget + offset
    }

    func endPan() {
        isPanning = false
    }

    // MARK: - Zoom

    func zoom(by delta: Float) {
        distance = max(0.5, distance * (1.0 - delta * 0.1))
    }

    func pinchZoom(magnification: Float) {
        distance = max(0.5, distance / max(magnification, 0.1))
    }

    // MARK: - Focus

    func fitToSphere(center: SIMD3<Float>, radius: Float) {
        target = center
        let fov = fovY * 0.5
        distance = max(radius / sinf(fov) * 1.2, 2.0)
        orientation = .identity
    }

    func focusOnPoint(_ point: SIMD3<Float>, animated: Bool = true) {
        if animated {
            targetTarget = point
            isAnimating = true
        } else {
            target = point
        }
    }

    func reset() {
        orientation = .identity
        target = .zero
        distance = 15.0
    }

    // MARK: - Animation Update

    func update() {
        guard isAnimating else { return }
        target = target + (targetTarget - target) * smoothFactor
        if simd_length(targetTarget - target) < 0.001 {
            target = targetTarget
            isAnimating = false
        }
    }

    // MARK: - Ray Casting (for picking)

    func screenToWorldRay(screenPoint: SIMD2<Float>, viewportSize: SIMD2<Float>) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        // Convert screen coordinates to NDC [-1, 1]
        let ndc = SIMD2<Float>(
            (2.0 * screenPoint.x / viewportSize.x) - 1.0,
            1.0 - (2.0 * screenPoint.y / viewportSize.y) // flip Y
        )

        // Unproject near and far points
        let invProj = projectionMatrix.inverse
        let invView = viewMatrix.inverse

        let nearNDC = SIMD4<Float>(ndc.x, ndc.y, -1.0, 1.0)
        let farNDC  = SIMD4<Float>(ndc.x, ndc.y,  1.0, 1.0)

        var nearView = invProj * nearNDC
        nearView /= nearView.w
        var farView = invProj * farNDC
        farView /= farView.w

        let nearWorld = invView * nearView
        let farWorld  = invView * farView

        let origin = SIMD3<Float>(nearWorld.x, nearWorld.y, nearWorld.z)
        let far3   = SIMD3<Float>(farWorld.x, farWorld.y, farWorld.z)
        let direction = simd_normalize(far3 - origin)

        return (origin, direction)
    }
}
