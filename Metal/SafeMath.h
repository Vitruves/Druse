// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// SafeMath.h
//
// Overflow-safe transcendentals for the neural-network compute kernels.
//
// The target builds with MTL_FAST_MATH: YES (project.yml). Under fast math the
// Metal standard library evaluates tanh() through exp(2x), which overflows
// float32 and silently produces garbage long before tanh() itself would
// saturate. Measured on Apple Silicon with -ffast-math:
//
//     tanh(30) -> 1.0        (correct)
//     tanh(44) -> 0.0        (wrong: exp(88) overflows, inf/inf collapses to 0)
//     tanh(45) -> nan        (wrong)
//
// The tanh-approximated GELU is worse, because its argument is a cubic in x and
// reaches the overflow threshold at a plausible pre-activation magnitude:
//
//     gelu(10.00) -> 10.00   (correct)
//     gelu(10.05) ->  5.025  (wrong: silently halved)
//     gelu(10.07) -> nan     (wrong)
//
// A single NaN is fatal rather than local: the attention softmax that follows
// mixes every atom, so one poisoned activation turns an entire pose score into
// NaN. Clamping the argument is exact, not an approximation — tanh(20) differs
// from 1.0f by ~1e-17, far below float32 resolution — so these helpers are
// numerically identical to the unclamped forms everywhere the unclamped forms
// are correct, and correct where they are not.

#ifndef DRUSE_SAFE_MATH_H
#define DRUSE_SAFE_MATH_H

#include <metal_stdlib>

// Beyond this magnitude tanh is ±1 to within float32 resolution.
#define DRUSE_TANH_ARG_LIMIT 20.0f

/// tanh, saturating instead of overflowing. Use in place of tanh() in any
/// kernel whose argument is an unbounded learned quantity.
inline float safeTanh(float x) {
    return metal::tanh(metal::clamp(x, -DRUSE_TANH_ARG_LIMIT, DRUSE_TANH_ARG_LIMIT));
}

/// GELU, tanh approximation — matches PyTorch's `approximate="tanh"`.
inline float safeGeluTanh(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x); // sqrt(2/pi)
    return 0.5f * x * (1.0f + safeTanh(inner));
}

#endif // DRUSE_SAFE_MATH_H
