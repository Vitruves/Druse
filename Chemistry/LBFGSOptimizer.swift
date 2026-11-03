// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// LBFGSOptimizer.swift — Pure Swift L-BFGS optimizer
//
// Limited-memory BFGS with a bracketing line search (weak Wolfe conditions).
// No external dependencies (no Accelerate, no Eigen).
//
// Units: caller-defined (typically Angstrom / kcal·mol⁻¹ for molecular systems).
// ============================================================================

import simd

struct LBFGSOptimizer {

    struct Parameters {
        var m: Int = 8               // history size
        var maxIterations: Int = 200
        var epsilon: Float = 1e-5    // RMS gradient tolerance for convergence
        var ftol: Float = 1e-4       // Armijo condition parameter
        var wolfe: Float = 0.9       // Wolfe condition parameter
        var maxLineSearch: Int = 20
    }

    // MARK: - Public API

    /// Minimize a differentiable function using L-BFGS.
    /// - Parameters:
    ///   - params: Optimizer hyperparameters.
    ///   - n: Dimension of the problem (3N for N atoms).
    ///   - x0: Initial positions, modified in place to the optimized result.
    ///   - evaluate: Closure that computes f(x) and fills grad; returns energy.
    /// - Returns: Final energy, iteration count, and whether convergence was reached.
    static func minimize(
        params: Parameters = Parameters(),
        n: Int,
        x0: inout [Float],
        evaluate: (_ x: inout [Float], _ grad: inout [Float]) -> Float
    ) -> (energy: Float, iterations: Int, converged: Bool) {
        precondition(x0.count == n, "x0.count (\(x0.count)) must equal n (\(n))")

        var grad = [Float](repeating: 0, count: n)
        var energy = evaluate(&x0, &grad)

        // History buffers (cyclic)
        var sHistory = [[Float]]()  // s_k = x_{k+1} - x_k
        var yHistory = [[Float]]()  // y_k = g_{k+1} - g_k
        var rhoHistory = [Float]()  // 1 / dot(y_k, s_k)
        sHistory.reserveCapacity(params.m)
        yHistory.reserveCapacity(params.m)
        rhoHistory.reserveCapacity(params.m)

        var direction = [Float](repeating: 0, count: n)
        var xNew = [Float](repeating: 0, count: n)
        var gradNew = [Float](repeating: 0, count: n)
        var alphas = [Float](repeating: 0, count: params.m)

        var xCandidate = [Float](repeating: 0, count: n)
        var gradCandidate = [Float](repeating: 0, count: n)

        for iter in 0..<params.maxIterations {

            // Convergence on the RMS gradient, which is independent of the coordinate
            // origin and of the system size (unlike ||g|| / ||x||).
            let gradRMS = sqrtf(dot(grad, grad, n: n) / Float(max(n, 1)))
            if gradRMS < params.epsilon {
                return (energy, iter, true)
            }

            // --- Two-loop recursion to compute direction = -H * grad ---
            twoLoopRecursion(
                grad: grad,
                sHistory: sHistory,
                yHistory: yHistory,
                rhoHistory: rhoHistory,
                alphas: &alphas,
                direction: &direction,
                n: n
            )

            // Negate to get descent direction
            negate(&direction, n: n)

            // Directional derivative
            let dg = dot(direction, grad, n: n)
            if dg >= 0 {
                // Not a descent direction; reset to steepest descent
                for i in 0..<n { direction[i] = -grad[i] }
                sHistory.removeAll(keepingCapacity: true)
                yHistory.removeAll(keepingCapacity: true)
                rhoHistory.removeAll(keepingCapacity: true)
            }

            let dgForLS = dot(direction, grad, n: n)

            // --- Bracketing line search for the weak Wolfe conditions ---
            // Without curvature history the direction is the raw gradient, so the
            // trial step is scaled to move at most one length unit (liblbfgs: 1/||d||).
            var step: Float = 1.0
            if sHistory.isEmpty {
                let dnorm = sqrtf(dot(direction, direction, n: n))
                if dnorm > 1 { step = 1 / dnorm }
            }
            var lower: Float = 0
            var upper = Float.infinity
            var candidateEnergy: Float?

            for _ in 0..<params.maxLineSearch {
                // x_new = x + step * d
                for i in 0..<n {
                    xNew[i] = x0[i] + step * direction[i]
                }

                let fNew = evaluate(&xNew, &gradNew)

                // Armijo (sufficient decrease); false for NaN/Inf energies.
                let armijo = fNew <= energy + params.ftol * step * dgForLS
                if !armijo {
                    upper = step
                } else {
                    // Curvature (weak Wolfe): the slope must have flattened enough.
                    let curvature = dot(direction, gradNew, n: n) >= params.wolfe * dgForLS
                    // Keep the best Armijo point; it is the fallback if curvature never holds.
                    if curvature || candidateEnergy.map({ fNew < $0 }) ?? true {
                        candidateEnergy = fNew
                        swap(&xCandidate, &xNew)
                        swap(&gradCandidate, &gradNew)
                    }
                    if curvature { break }
                    lower = step
                }

                // Too-long steps shrink by bisection; too-short steps (Armijo holds but the
                // function is still decreasing steeply) expand until bracketed.
                step = upper.isFinite ? 0.5 * (lower + upper) : 2 * lower
            }

            guard let fAccepted = candidateEnergy else {
                // No step satisfied sufficient decrease; return current best
                return (energy, iter + 1, false)
            }

            let s = subtract(xCandidate, x0, n: n)
            let y = subtract(gradCandidate, grad, n: n)
            let ys = dot(y, s, n: n)

            if ys > 1e-10 {
                // Update history (cyclic buffer)
                if sHistory.count == params.m {
                    sHistory.removeFirst()
                    yHistory.removeFirst()
                    rhoHistory.removeFirst()
                }
                sHistory.append(s)
                yHistory.append(y)
                rhoHistory.append(1.0 / ys)
            }

            // Advance
            for i in 0..<n {
                x0[i] = xCandidate[i]
                grad[i] = gradCandidate[i]
            }
            energy = fAccepted
        }

        return (energy, params.maxIterations, false)
    }

    // MARK: - Two-loop recursion

    /// Compute H * grad via L-BFGS two-loop recursion.
    /// Result is placed in `direction` (caller negates for descent).
    private static func twoLoopRecursion(
        grad: [Float],
        sHistory: [[Float]],
        yHistory: [[Float]],
        rhoHistory: [Float],
        alphas: inout [Float],
        direction: inout [Float],
        n: Int
    ) {
        let k = sHistory.count

        // q = grad (copy)
        for i in 0..<n { direction[i] = grad[i] }

        if k == 0 {
            // No history: direction = grad (will be negated by caller)
            return
        }

        // Forward loop (newest to oldest)
        for i in stride(from: k - 1, through: 0, by: -1) {
            let alpha = rhoHistory[i] * dot(sHistory[i], direction, n: n)
            alphas[i] = alpha
            // q -= alpha * y[i]
            axpy(-alpha, yHistory[i], &direction, n: n)
        }

        // Initial Hessian scaling: gamma = dot(s_k, y_k) / dot(y_k, y_k)
        let newest = k - 1
        let ys = dot(sHistory[newest], yHistory[newest], n: n)
        let yy = dot(yHistory[newest], yHistory[newest], n: n)
        let gamma = yy > 1e-20 ? ys / yy : 1.0

        // r = gamma * q
        for i in 0..<n { direction[i] *= gamma }

        // Backward loop (oldest to newest)
        for i in 0..<k {
            let beta = rhoHistory[i] * dot(yHistory[i], direction, n: n)
            // r += (alpha[i] - beta) * s[i]
            axpy(alphas[i] - beta, sHistory[i], &direction, n: n)
        }
    }

    // MARK: - Vector operations (flat Float arrays)

    @inline(__always)
    private static func dot(_ a: [Float], _ b: [Float], n: Int) -> Float {
        var sum: Float = 0
        // Process in chunks of 4 using SIMD
        let n4 = n & ~3
        var i = 0
        while i < n4 {
            let va = simd_float4(a[i], a[i+1], a[i+2], a[i+3])
            let vb = simd_float4(b[i], b[i+1], b[i+2], b[i+3])
            sum += simd_dot(va, vb)
            i += 4
        }
        while i < n {
            sum += a[i] * b[i]
            i += 1
        }
        return sum
    }

    /// y += alpha * x
    @inline(__always)
    private static func axpy(_ alpha: Float, _ x: [Float], _ y: inout [Float], n: Int) {
        let n4 = n & ~3
        let va = simd_float4(repeating: alpha)
        var i = 0
        while i < n4 {
            let vx = simd_float4(x[i], x[i+1], x[i+2], x[i+3])
            let vy = simd_float4(y[i], y[i+1], y[i+2], y[i+3])
            let vr = vy + va * vx
            y[i]   = vr.x
            y[i+1] = vr.y
            y[i+2] = vr.z
            y[i+3] = vr.w
            i += 4
        }
        while i < n {
            y[i] += alpha * x[i]
            i += 1
        }
    }

    /// Return a - b
    @inline(__always)
    private static func subtract(_ a: [Float], _ b: [Float], n: Int) -> [Float] {
        var result = [Float](repeating: 0, count: n)
        for i in 0..<n {
            result[i] = a[i] - b[i]
        }
        return result
    }

    /// Negate in place
    @inline(__always)
    private static func negate(_ a: inout [Float], n: Int) {
        for i in 0..<n {
            a[i] = -a[i]
        }
    }
}
