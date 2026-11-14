//! The f32 activation kernels: `tanh`, the logistic sigmoid, softplus and ELU over a slice,
//! in place, on the SIMD kernel of the current level, and SiLU and GELU's tanh form built
//! on the first two.
//!
//! Both are built from one exponential, `e^y` for `y <= 0`: `y` is split as `n · ln 2 + r`
//! with `|r| <= ln 2 / 2`, `e^r` comes from a degree-7 polynomial (the coefficients of the
//! Cephes library's `expf`), and the result is scaled by `2^n` in two steps, so results
//! below the smallest normal f32 round once into the subnormal range instead of flushing to
//! zero.
//!
//! - `tanh(x)` is `a + a·s·Q(s)` with `a = |x|`, `s = a²` and `Q` a degree-4 polynomial
//!   for `a < 0.55`, and `(1 - e) / (1 + e)` with `e = e^(-2a)` above; the sign of `x` is
//!   copied onto the result. The polynomial avoids the cancellation in `1 - e` near zero.
//! - `sigmoid(x)` is `1 / (1 + e)` for `x >= 0` and `e / (1 + e)` below, with
//!   `e = e^(-|x|)`, so nothing overflows.
//!
//! Every lane computes both branches and keeps one, so the time does not depend on the
//! values. NaN stays NaN (the input's NaN is returned), `tanh(±∞) = ±1`, `tanh(-0) = -0`,
//! `sigmoid(+∞) = 1` and `sigmoid(-∞) = 0`.
//!
//! **Accuracy.** Against the correctly rounded result (from f64), checked for every finite
//! f32, the reference loop and the SSE4.2 kernel are within 2 ulp for both functions. The
//! bound stated and tested at every level is 3 ulp, since the AVX2 and NEON kernels fuse
//! multiply-adds and round differently from the reference (they are tested on a sweep of
//! one f32 in 1,009 and the edge values). The SSE4.2 kernel performs the reference's
//! operations in the same order and matches it bit for bit.
//!
//! Softplus and ELU use the same exponential. Softplus is `max(x, 0) + ln(1 + e^(−|x|))`,
//! the logarithm a `log1p` on `(0, 1]`: `1 + t` rounded, its exact rounding error `c`, the
//! Cephes `logf` polynomial on the mantissa and `+ c/(1 + t)`. ELU's `e^x − 1` for `x <= 0`
//! is `q·2^n + (2^n − 1)` with `q = e^r − 1` from the exponential's polynomial, so it keeps
//! its relative accuracy near 0. Checked for every finite f32 at the reference and SSE4.2
//! levels, softplus is within 2 ulp and `e^x − 1` within 1 ulp. The AVX2 and NEON kernels
//! (Accelerate's are NEON's) fuse multiply-adds, so their stated bounds are one ulp more
//! (3 and 2), tested on the sweep.
//!
//! Log-sigmoid (`−softplus(−x)`), softplus at any `beta` and threshold, Mish
//! (`x · tanh(softplus(x))`, a block of 64 values at a time through the softplus and tanh
//! kernels), SELU and CELU are built on these two; each function's docs give its bound.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::kernels::activation::{sigmoid_f32, tanh_f32};
//! let mut t = [0.0f32, 0.5, -3.0, f32::INFINITY];
//! tanh_f32(&mut t);
//! assert_eq!(t[0], 0.0);
//! assert!((t[1] - 0.5f64.tanh() as f32).abs() <= 2.0 * f32::EPSILON);
//! assert!((t[2] - (-3.0f64).tanh() as f32).abs() <= 2.0 * f32::EPSILON);
//! assert_eq!(t[3], 1.0);
//! let mut s = [0.0f32, -200.0, 200.0];
//! sigmoid_f32(&mut s);
//! assert_eq!(s, [0.5, 0.0, 1.0]);
//! ```

use super::{Level, supported};

/// The input below which the exponential's argument is clamped: `e^-104` is below half the
/// smallest subnormal f32, so it rounds to zero, and `2^n` for the `n` it gives stays
/// within two normal scale factors.
pub(super) const EXP_MIN: f32 = -104.0;
/// `log2(e)`, to find `n = floor(y · log2(e) + 1/2)`.
pub(super) const LOG2E: f32 = std::f32::consts::LOG2_E;
/// The high part of `ln 2`, 0.693359375 exactly (the f32 this literal names), with trailing
/// zero bits, so `n · LN2_HI` is exact.
pub(super) const LN2_HI: f32 = 0.693_359_4;
/// The low part of `ln 2`: `LN2_HI + LN2_LO` is `ln 2` to about 1e-11.
pub(super) const LN2_LO: f32 = -2.121_944_4e-4;
/// The coefficients of `e^r ≈ 1 + r + r²·P(r)`, highest degree first (Cephes `expf`).
pub(super) const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    1.666_666_5e-1,
    0.5,
];
/// Below this `|x|`, `tanh` uses its polynomial instead of the exponential.
pub(super) const TANH_SMALL: f32 = 0.55;
/// The coefficients of `Q` in `tanh(a) ≈ a + a·s·Q(s)`, `s = a²`, highest degree first:
/// a fit minimising the largest relative error on `a < 0.55`, which is about 1e-9 before
/// rounding to f32.
pub(super) const TANH_Q: [f32; 5] = [
    -6.274_173_3e-3,
    2.107_163_5e-2,
    -5.385_23e-2,
    1.333_258_6e-1,
    -3.333_332e-1,
];

/// The sign bit of an f32.
pub(super) const SIGN: u32 = 0x8000_0000;
/// Below this `y`, `e^y − 1` rounds to −1 in f32 (`e^y < 2⁻²⁵`).
pub(super) const EXPM1_MIN: f32 = -17.5;
/// Below this magnitude, `e^y − 1` correctly rounded to f32 is `y` itself (`y²/2` is under
/// half an ulp of `y`), so ELU is `alpha · x` there. Taking that branch keeps the sign of
/// `−0`, which the exponential's polynomial loses (`−0 + 0` is `+0`).
pub(super) const EXPM1_TINY: f32 = 2.980_232_2e-8; // 2⁻²⁵
/// The coefficients of `P` in `ln f ≈ x − x²/2 + x³·P(x)`, `x = f − 1`, `f ∈ [√½, √2)`,
/// highest degree first (Cephes `logf`).
pub(super) const LOG_P: [f32; 9] = [
    7.037_683_6e-2,
    -1.151_461e-1,
    1.167_699_9e-1,
    -1.242_014_1e-1,
    1.424_932_3e-1,
    -1.666_805_8e-1,
    2.000_071_5e-1,
    -2.499_999_4e-1,
    3.333_333e-1,
];

/// `2^n` for `-126 <= n <= 127`, built from its exponent bits.
fn pow2(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

/// `e^y` for `y <= 0` (larger `y` gives wrong results; NaN gives a meaningless value), as
/// the kernels compute it, one rounding per operation and no fused multiply-add.
pub(super) fn exp_neg(y: f32) -> f32 {
    let y = if y < EXP_MIN { EXP_MIN } else { y };
    let n = (y * LOG2E + 0.5).floor();
    let r = y - n * LN2_HI;
    let r = r - n * LN2_LO;
    let mut p = EXP_P[0];
    for &c in &EXP_P[1..] {
        p = p * r + c;
    }
    let p = p * r * r + r + 1.0;
    // `n` is a whole number in [-150, 0]; each half is in [-75, 0], a normal power of two.
    let n = n as i32;
    let n1 = n >> 1;
    p * pow2(n1) * pow2(n - n1)
}

/// `e^y − 1` for `y <= 0` (NaN for NaN), as the kernels compute it, one rounding per
/// operation: with `y = n·ln 2 + r`, `e^r − 1 = q = r + r²·P(r)`, which keeps its relative
/// accuracy near 0, and `e^y − 1 = q·2^n + (2^n − 1)`. Within 1 ulp of the correctly rounded
/// result for every `y <= 0` (measured over every such f32).
pub(super) fn expm1_neg(y: f32) -> f32 {
    if y.is_nan() {
        return y;
    }
    if y < EXPM1_MIN {
        return -1.0;
    }
    let n = (y * LOG2E + 0.5).floor();
    let r = y - n * LN2_HI;
    let r = r - n * LN2_LO;
    let mut p = EXP_P[0];
    for &c in &EXP_P[1..] {
        p = p * r + c;
    }
    let q = p * r * r + r;
    // `n` is a whole number in [-25, 0], so `2^n` is a normal power of two.
    let s = pow2(n as i32);
    q * s + (s - 1.0)
}

/// `ln(1 + t)` for `t` in `(0, 1]`, as the kernels compute it, one rounding per operation:
/// `m = 1 + t` rounded, its rounding error `c = t − (m − 1)` (exact), `m = 2^k·f` with
/// `f ∈ [√½, √2)` and `k` 0 or 1, `ln f` from Cephes' polynomial in `x = f − 1` (exact),
/// and `ln(1 + t) = k·ln 2 + ln f + c/m`. Within 1 ulp of the correctly rounded result for
/// every such `t` (measured over every such f32).
pub(super) fn log1p_unit(t: f32) -> f32 {
    let m = 1.0 + t;
    let c = t - (m - 1.0);
    let (f, k) = if m > std::f32::consts::SQRT_2 {
        (m * 0.5, 1.0f32)
    } else {
        (m, 0.0)
    };
    let x = f - 1.0;
    let z = x * x;
    let mut p = LOG_P[0];
    for &cc in &LOG_P[1..] {
        p = p * x + cc;
    }
    let y = x * z * p;
    let y = y + k * LN2_LO;
    let y = y - 0.5 * z;
    let r = x + y;
    let r = r + k * LN2_HI;
    r + c / m
}

/// `ln(1 + e^x)` as the kernels compute it, one element at a time: `max(x, 0)` plus
/// [`log1p_unit`] of [`exp_neg`]`(−|x|)`, NaN for NaN. The reference twin of the SIMD
/// softplus.
pub(super) fn softplus_one(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let l = log1p_unit(exp_neg(-x.abs()));
    let m = if x > 0.0 { x } else { 0.0 };
    m + l
}

/// ELU with `alpha` as the kernels compute it, one element at a time: `x` above 0, `alpha`
/// times [`expm1_neg`]`(x)` at or below (`alpha · x` below [`EXPM1_TINY`] in size, which
/// keeps `−0`), NaN for NaN. The reference twin of the SIMD ELU.
pub(super) fn elu_one(alpha: f32, x: f32) -> f32 {
    if x.is_nan() || x > 0.0 {
        x
    } else if x.abs() < EXPM1_TINY {
        alpha * x
    } else {
        alpha * expm1_neg(x)
    }
}

/// `tanh(x)` as the kernels compute it, one element at a time: the reference twin of the
/// SIMD kernels, and the [`Level::Reference`] kernel.
pub(super) fn tanh_one(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    let s = a * a;
    let mut q = TANH_Q[0];
    for &c in &TANH_Q[1..] {
        q = q * s + c;
    }
    let small = a + a * (s * q);
    let e = exp_neg(-2.0 * a);
    let big = (1.0 - e) / (1.0 + e);
    let r = if a < TANH_SMALL { small } else { big };
    f32::from_bits(r.to_bits() | (x.to_bits() & SIGN))
}

/// The logistic sigmoid of `x` as the kernels compute it, one element at a time: the
/// reference twin of the SIMD kernels, and the [`Level::Reference`] kernel.
pub(super) fn sigmoid_one(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let e = exp_neg(-x.abs());
    let d = 1.0 + e;
    if x >= 0.0 { 1.0 / d } else { e / d }
}

/// Which function a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Func {
    /// The hyperbolic tangent.
    Tanh,
    /// The logistic sigmoid.
    Sigmoid,
}

/// Replaces every value with its hyperbolic tangent, with the kernel for the current
/// [`level`](super::level). Each result is within 3 ulp of the correctly rounded `tanh`
/// (see the [module docs](self)); a value's result depends only on the value and the level,
/// not on its place in the slice. It runs on the calling thread and allocates nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::tanh_f32;
/// let mut v = [-0.0f32, 20.0, f32::NAN];
/// tanh_f32(&mut v);
/// assert!(v[0] == 0.0 && v[0].is_sign_negative());
/// assert_eq!(v[1], 1.0);
/// assert!(v[2].is_nan());
/// ```
pub fn tanh_f32(values: &mut [f32]) {
    run(super::level(), Func::Tanh, values);
}

/// [`tanh_f32`] with the kernel level given explicitly, instead of the process-wide
/// setting. [`Level::Reference`] runs [`reference::tanh_f32`](super::reference::tanh_f32),
/// [`Level::Sse42`] gives the same bits as it, and [`Level::Avx2Fma`], [`Level::Neon`] and
/// [`Level::Accelerate`] (whose activations are NEON's) fuse multiply-adds.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::tanh_f32_with;
/// use cavitation_core::kernels::{Level, detected};
/// let (mut fast, mut slow) = ([0.25f32; 9], [0.25f32; 9]);
/// tanh_f32_with(detected(), &mut fast);
/// tanh_f32_with(Level::Reference, &mut slow);
/// for (f, s) in fast.iter().zip(&slow) {
///     assert!((f - s).abs() <= 4.0 * f32::EPSILON * s.abs());
/// }
/// ```
pub fn tanh_f32_with(level: Level, values: &mut [f32]) {
    run(level, Func::Tanh, values);
}

/// Replaces every value with its logistic sigmoid `1 / (1 + e^-x)`, with the kernel for
/// the current [`level`](super::level). Each result is within 3 ulp of the correctly
/// rounded sigmoid (see the [module docs](self)); a value's result depends only on the
/// value and the level. It runs on the calling thread and allocates nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::sigmoid_f32;
/// let mut v = [0.0f32, f32::NEG_INFINITY, f32::INFINITY];
/// sigmoid_f32(&mut v);
/// assert_eq!(v, [0.5, 0.0, 1.0]);
/// ```
pub fn sigmoid_f32(values: &mut [f32]) {
    run(super::level(), Func::Sigmoid, values);
}

/// [`sigmoid_f32`] with the kernel level given explicitly, instead of the process-wide
/// setting; the levels differ as for [`tanh_f32_with`].
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::sigmoid_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [2.0f32];
/// sigmoid_f32_with(Level::Reference, &mut v);
/// assert!((v[0] - 0.880_797_1).abs() <= 3.0 * f32::EPSILON);
/// ```
pub fn sigmoid_f32_with(level: Level, values: &mut [f32]) {
    run(level, Func::Sigmoid, values);
}

/// Replaces every value `x` with softplus, `ln(1 + e^x)`, computed as `max(x, 0)` plus
/// `ln(1 + e^(−|x|))` so nothing overflows, with the kernel for the current
/// [`level`](super::level). The exponential is the one `tanh` and the sigmoid use and the
/// logarithm a `log1p` on `(0, 1]` within 1 ulp; the result is within 2 ulp of the
/// correctly rounded softplus for every finite f32 (measured over all of them). NaN stays
/// NaN; a value's result depends only on the value and the level.
///
/// The SSE4.2 kernel gives the reference loop's bits. The AVX2 and NEON kernels (Accelerate
/// runs NEON's) fuse multiply-adds and round differently: their stated bound is one ulp
/// more, tested on a sweep of one f32 in 1,009.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::softplus_f32;
/// let mut v = [0.0f32, 30.0, -30.0];
/// softplus_f32(&mut v);
/// assert!((v[0] - std::f32::consts::LN_2).abs() <= f32::EPSILON);
/// assert_eq!(v[1], 30.0);
/// assert!((v[2] - 9.357_623e-14).abs() <= 1e-19);
/// ```
pub fn softplus_f32(values: &mut [f32]) {
    softplus_f32_with(super::level(), values);
}

/// [`softplus_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::softplus_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [1.0f32];
/// softplus_f32_with(Level::Reference, &mut v);
/// assert!((v[0] - 1.313_261_7).abs() <= 2.0 * f32::EPSILON);
/// ```
pub fn softplus_f32_with(level: Level, values: &mut [f32]) {
    assert!(
        supported(level),
        "activation kernel: {}",
        super::cannot_run(level)
    );
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports AVX2 and FMA (the guard).
        Level::Avx2Fma if supported(Level::Avx2Fma) => unsafe {
            super::x86::softplus_avx2_slice(values)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: this CPU supports NEON (the guard; Accelerate's support check includes
        // NEON).
        Level::Neon | Level::Accelerate if supported(level) => unsafe {
            super::aarch64::softplus_neon_slice(values)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports SSE4.2 (the guard).
        Level::Sse42 if supported(Level::Sse42) => unsafe {
            super::x86::softplus_sse_slice(values)
        },
        _ => {
            for v in values {
                *v = softplus_one(*v);
            }
        }
    }
}

/// Replaces every value `x` with ELU, `x` above 0 and `alpha · (e^x − 1)` at or below, with
/// the kernel for the current [`level`](super::level). `e^x − 1` is computed so that it
/// keeps its relative accuracy near 0, within 1 ulp of the correctly rounded result for
/// every finite f32 (measured over all of them), and is then multiplied by `alpha`,
/// rounded once. NaN stays NaN; a value's result depends only on the value, `alpha` and
/// the level.
///
/// The SSE4.2 kernel gives the reference loop's bits. The AVX2 and NEON kernels (Accelerate
/// runs NEON's) fuse multiply-adds and round differently: their stated bound is one ulp
/// more, tested on a sweep of one f32 in 1,009.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::elu_f32;
/// let mut v = [2.0f32, -1e-7, -50.0];
/// elu_f32(1.0, &mut v);
/// assert_eq!(v[0], 2.0);
/// assert!((v[1] + 1e-7).abs() <= 1e-14);
/// assert_eq!(v[2], -1.0);
/// ```
pub fn elu_f32(alpha: f32, values: &mut [f32]) {
    elu_f32_with(super::level(), alpha, values);
}

/// [`elu_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::elu_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [-1.0f32];
/// elu_f32_with(Level::Reference, 1.0, &mut v);
/// assert!((v[0] + 0.632_120_56).abs() <= f32::EPSILON);
/// ```
pub fn elu_f32_with(level: Level, alpha: f32, values: &mut [f32]) {
    assert!(
        supported(level),
        "activation kernel: {}",
        super::cannot_run(level)
    );
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports AVX2 and FMA (the guard).
        Level::Avx2Fma if supported(Level::Avx2Fma) => unsafe {
            super::x86::elu_avx2_slice(alpha, values)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: this CPU supports NEON (the guard; Accelerate's support check includes
        // NEON).
        Level::Neon | Level::Accelerate if supported(level) => unsafe {
            super::aarch64::elu_neon_slice(alpha, values)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports SSE4.2 (the guard).
        Level::Sse42 if supported(Level::Sse42) => unsafe {
            super::x86::elu_sse_slice(alpha, values)
        },
        _ => {
            for v in values {
                *v = elu_one(alpha, *v);
            }
        }
    }
}

/// Values per block of [`silu_f32_with`] and [`gelu_tanh_f32_with`]: the sigmoid or tanh of
/// a block goes into a buffer on the stack, so the in-place forms allocate nothing.
const BLOCK: usize = 64;

/// Replaces every value `x` with SiLU, `x · σ(x)`, its sigmoid from [`sigmoid_f32`] with the
/// kernel for the current [`level`](super::level). Each result is `x` times a sigmoid within
/// 3 ulp, rounded once. The sigmoid's 3 ulp are up to `3 · 2⁻²³` relative and an ulp of the
/// product can be as small as `2⁻²⁴` relative, so the bound is 7 ulp of the correctly
/// rounded SiLU (6 from the sigmoid, one more for the product's rounding) while `σ(x)` is a normal
/// `f32` (`x` above about −87.3). Below that the sigmoid is subnormal and its 3 ulp are
/// absolute, so the result is within `3 · 2⁻¹⁴⁹ · |x|` plus 1 ulp. A value's result depends
/// only on the value and the level.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::silu_f32;
/// let mut v = [0.0f32, 2.0, f32::NEG_INFINITY];
/// silu_f32(&mut v);
/// assert_eq!(v[0], 0.0);
/// assert!((v[1] - 1.761_594_2).abs() <= 4.0 * f32::EPSILON * 2.0);
/// assert!(v[2].is_nan()); // -∞ · 0
/// ```
pub fn silu_f32(values: &mut [f32]) {
    silu_f32_with(super::level(), values);
}

/// [`silu_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::silu_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [1.0f32];
/// silu_f32_with(Level::Reference, &mut v);
/// assert!((v[0] - 0.731_058_6).abs() <= 4.0 * f32::EPSILON);
/// ```
pub fn silu_f32_with(level: Level, values: &mut [f32]) {
    let mut s = [0.0f32; BLOCK];
    for block in values.chunks_mut(BLOCK) {
        let s = &mut s[..block.len()];
        s.copy_from_slice(block);
        sigmoid_f32_with(level, s);
        for (x, &s) in block.iter_mut().zip(s.iter()) {
            *x *= s;
        }
    }
}

/// Replaces every value `x` with the tanh form of GELU, `½x · (1 + tanh(u))` with
/// `u = √(2/π) · (x + 0.044715 · x³)`, its tanh from [`tanh_f32`] with the kernel for the
/// current [`level`](super::level). Each result is within `|x| · 2⁻²¹` plus 2 ulp of the
/// value from `f64` arithmetic: the tanh's error of at most 3 ulp, scaled by `½|x|`, is
/// absolute, so for large negative `x`, where `1 + tanh(u)` cancels, the result has few
/// correct digits, as the scalar `f32` formula has. A value's result depends only on the
/// value and the level.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::gelu_tanh_f32;
/// let mut v = [0.0f32, 1.0, 10.0];
/// gelu_tanh_f32(&mut v);
/// assert_eq!(v[0], 0.0);
/// assert!((v[1] - 0.841_192).abs() <= 1e-6);
/// assert_eq!(v[2], 10.0);
/// ```
pub fn gelu_tanh_f32(values: &mut [f32]) {
    gelu_tanh_f32_with(super::level(), values);
}

/// [`gelu_tanh_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::gelu_tanh_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [-1.0f32];
/// gelu_tanh_f32_with(Level::Reference, &mut v);
/// assert!((v[0] + 0.158_808).abs() <= 1e-6);
/// ```
pub fn gelu_tanh_f32_with(level: Level, values: &mut [f32]) {
    // The operations of the scalar formula, in its order, with `f32` constants.
    let (scale, cubic) = (
        crate::ops::SQRT_2_OVER_PI as f32,
        crate::ops::GELU_CUBIC as f32,
    );
    let mut t = [0.0f32; BLOCK];
    for block in values.chunks_mut(BLOCK) {
        let t = &mut t[..block.len()];
        for (u, &x) in t.iter_mut().zip(block.iter()) {
            *u = scale * (x + cubic * x * x * x);
        }
        tanh_f32_with(level, t);
        for (x, &t) in block.iter_mut().zip(t.iter()) {
            *x = 0.5 * *x * (1.0 + t);
        }
    }
}

/// Replaces every value `x` with softplus at the given `beta` and `threshold`, as
/// [`Tape::softplus`](crate::Tape::softplus) defines it: `x` itself where `z = beta · x`
/// is above `threshold`, and `ln(1 + e^z) / beta` at or below, with `ln(1 + e^z)` from
/// [`softplus_f32`] on the kernel for the current [`level`](super::level). For `beta` 1,
/// `z` is `x` and the division is exact, so the result is within [`softplus_f32`]'s bound;
/// for other `beta` it is within that bound of `ln(1 + e^z) / beta` for the rounded `z`,
/// plus the division's rounding. NaN stays NaN; a value's result depends only on the value,
/// `beta`, `threshold` and the level. It allocates nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::softplus_beta_f32;
/// let mut v = [30.0f32, 0.0];
/// softplus_beta_f32(1.0, 20.0, &mut v);
/// assert_eq!(v[0], 30.0); // above the threshold: x itself
/// assert!((v[1] - std::f32::consts::LN_2).abs() <= f32::EPSILON);
/// ```
pub fn softplus_beta_f32(beta: f32, threshold: f32, values: &mut [f32]) {
    softplus_beta_f32_with(super::level(), beta, threshold, values);
}

/// [`softplus_beta_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::softplus_beta_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [1.0f32];
/// softplus_beta_f32_with(Level::Reference, 2.0, 20.0, &mut v);
/// assert!((v[0] - 1.063_464_3).abs() <= 4.0 * f32::EPSILON);
/// ```
pub fn softplus_beta_f32_with(level: Level, beta: f32, threshold: f32, values: &mut [f32]) {
    let (mut z, mut sp) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
    for block in values.chunks_mut(BLOCK) {
        let (z, sp) = (&mut z[..block.len()], &mut sp[..block.len()]);
        for (z, &x) in z.iter_mut().zip(block.iter()) {
            *z = beta * x;
        }
        sp.copy_from_slice(z);
        softplus_f32_with(level, sp);
        for ((x, &z), &s) in block.iter_mut().zip(z.iter()).zip(sp.iter()) {
            *x = if z > threshold { *x } else { s / beta };
        }
    }
}

/// Replaces every value `x` with log-sigmoid, `ln σ(x)`, computed as `−softplus(−x)` with
/// [`softplus_f32`] on the kernel for the current [`level`](super::level): the same
/// operations as `min(x, 0) − ln(1 + e^(−|x|))`, so nothing overflows and the result is
/// within [`softplus_f32`]'s bound. NaN stays NaN; a value's result depends only on the
/// value and the level. It allocates nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::log_sigmoid_f32;
/// let mut v = [-1000.0f32, 1000.0, 0.0];
/// log_sigmoid_f32(&mut v);
/// assert_eq!(&v[..2], &[-1000.0, 0.0]);
/// assert!(v[1].is_sign_positive());
/// assert!((v[2] + std::f32::consts::LN_2).abs() <= f32::EPSILON);
/// ```
pub fn log_sigmoid_f32(values: &mut [f32]) {
    log_sigmoid_f32_with(super::level(), values);
}

/// [`log_sigmoid_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::log_sigmoid_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [1.0f32];
/// log_sigmoid_f32_with(Level::Reference, &mut v);
/// assert!((v[0] + 0.313_261_7).abs() <= 2.0 * f32::EPSILON);
/// ```
pub fn log_sigmoid_f32_with(level: Level, values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = -*v;
    }
    softplus_f32_with(level, values);
    // `0 − s` rather than `−s`: where `e^(−|x|)` underflows, softplus is `+0` and the
    // scalar formula `min(x, 0) − 0` gives `+0`, not `−0`.
    for v in values.iter_mut() {
        *v = 0.0 - *v;
    }
}

/// Replaces every value `x` with Mish, `x · tanh(softplus(x))`, with softplus at the
/// tape's threshold of 20 ([`softplus_beta_f32`] with `beta` 1) and the tanh of
/// [`tanh_f32`], both on the kernel for the current [`level`](super::level). Each kernel's
/// relative error passes through the next one undiminished at worst (the tanh's relative
/// condition number is at most 1), so the result is within the sum of the two bounds in
/// relative terms plus the product's rounding: 13 ulp of the correctly rounded Mish while
/// softplus is a normal `f32` (`x` above about −87.3), and within `6 · 2⁻¹⁴⁹ · |x|` plus
/// 1 ulp below. The worst over a sweep of one f32 in 1,009 is 4 ulp at every level. A value's result depends only on the value and the level. It allocates
/// nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::mish_f32;
/// let mut v = [0.0f32, 100.0, 1.0];
/// mish_f32(&mut v);
/// assert_eq!(&v[..2], &[0.0, 100.0]);
/// assert!((v[2] - 0.865_098_4).abs() <= 16.0 * f32::EPSILON);
/// ```
pub fn mish_f32(values: &mut [f32]) {
    mish_f32_with(super::level(), values);
}

/// [`mish_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::mish_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [-1.0f32];
/// mish_f32_with(Level::Reference, &mut v);
/// assert!((v[0] + 0.303_401_4).abs() <= 16.0 * f32::EPSILON);
/// ```
pub fn mish_f32_with(level: Level, values: &mut [f32]) {
    let mut t = [0.0f32; BLOCK];
    for block in values.chunks_mut(BLOCK) {
        let t = &mut t[..block.len()];
        t.copy_from_slice(block);
        softplus_beta_f32_with(level, 1.0, MISH_THRESHOLD, t);
        tanh_f32_with(level, t);
        for (x, &t) in block.iter_mut().zip(t.iter()) {
            *x *= t;
        }
    }
}

/// The softplus threshold inside Mish, as the tape's Mish uses it.
const MISH_THRESHOLD: f32 = 20.0;

/// Replaces every value `x` with SELU, `scale · elu(x, alpha)` with PyTorch's constants,
/// the ELU from [`elu_f32`] on the kernel for the current [`level`](super::level) and then
/// multiplied by `scale`, rounded once: within 3 ulp of the correctly rounded SELU at the
/// reference and SSE4.2 levels and 4 at the fused levels (the worst over a sweep of one f32
/// in 1,009 is 2 at every level). NaN stays NaN; a value's result depends only on the value
/// and the level.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::selu_f32;
/// let mut v = [0.0f32, 1.0, -100.0];
/// selu_f32(&mut v);
/// assert_eq!(v[0], 0.0);
/// assert_eq!(v[1], 1.050_701);
/// assert!((v[2] + 1.758_099_3).abs() <= 2.0 * f32::EPSILON);
/// ```
pub fn selu_f32(values: &mut [f32]) {
    selu_f32_with(super::level(), values);
}

/// [`selu_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::selu_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [-1.0f32];
/// selu_f32_with(Level::Reference, &mut v);
/// assert!((v[0] + 1.111_330_7).abs() <= 4.0 * f32::EPSILON);
/// ```
pub fn selu_f32_with(level: Level, values: &mut [f32]) {
    let (scale, alpha) = (crate::ops::SELU_SCALE as f32, crate::ops::SELU_ALPHA as f32);
    elu_f32_with(level, alpha, values);
    for v in values.iter_mut() {
        *v *= scale;
    }
}

/// Replaces every value `x` with CELU, `x` above 0 and `alpha · (e^(x/alpha) − 1)` at or
/// below, with `e^(x/alpha) − 1` from [`elu_f32`]'s `e^y − 1` on the kernel for the current
/// [`level`](super::level). For a positive `alpha`, `x / alpha` is rounded once and
/// `e^y − 1` has a relative condition number of at most 1 for `y ≤ 0`, so the result is
/// within 3 ulp of the correctly rounded CELU at the reference and SSE4.2 levels and 4 at
/// the fused levels while `x / alpha` is a normal `f32` (the worst over a sweep of one f32
/// in 1,009 is 2); where the quotient is subnormal, its rounding is absolute and CELU,
/// about `alpha · x / alpha`, carries it, as the scalar formula does. A negative `alpha` puts `x / alpha` above 0,
/// outside the kernel's `e^y − 1`, so it takes std's `exp_m1` one value at a time instead.
/// NaN stays NaN; a value's result depends only on the value, `alpha` and the level.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::celu_f32;
/// let mut v = [2.0f32, -1.0];
/// celu_f32(2.0, &mut v);
/// assert_eq!(v[0], 2.0);
/// assert!((v[1] + 0.786_938_7).abs() <= 4.0 * f32::EPSILON);
/// ```
pub fn celu_f32(alpha: f32, values: &mut [f32]) {
    celu_f32_with(super::level(), alpha, values);
}

/// [`celu_f32`] with the kernel level given explicitly.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::celu_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [-1.0f32];
/// celu_f32_with(Level::Reference, 1.0, &mut v);
/// assert!((v[0] + 0.632_120_56).abs() <= 2.0 * f32::EPSILON);
/// ```
pub fn celu_f32_with(level: Level, alpha: f32, values: &mut [f32]) {
    if alpha.is_nan() || alpha <= 0.0 {
        assert!(
            supported(level),
            "activation kernel: {}",
            super::cannot_run(level)
        );
        for v in values.iter_mut() {
            if *v <= 0.0 || v.is_nan() {
                *v = alpha * (*v / alpha).exp_m1();
            }
        }
        return;
    }
    // At or below 0, `x / alpha` is at or below 0 and the kernel gives
    // `alpha · (e^(x/alpha) − 1)`; above 0 it would give `x / alpha`, so `x` is kept there.
    let mut y = [0.0f32; BLOCK];
    for block in values.chunks_mut(BLOCK) {
        let y = &mut y[..block.len()];
        for (y, &x) in y.iter_mut().zip(block.iter()) {
            *y = x / alpha;
        }
        elu_f32_with(level, alpha, y);
        for (x, &e) in block.iter_mut().zip(y.iter()) {
            if *x <= 0.0 || x.is_nan() {
                *x = e;
            }
        }
    }
}

/// Runs `func` over `values` with the kernel for `level`.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`.
fn run(level: Level, func: Func, values: &mut [f32]) {
    assert!(
        supported(level),
        "activation kernel: {}",
        super::cannot_run(level)
    );
    // Each arm runs a kernel only after checking that this CPU supports it.
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports AVX2 and FMA (the guard).
        Level::Avx2Fma if supported(Level::Avx2Fma) => unsafe {
            super::x86::activation_avx2(func, values)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports SSE4.2 (the guard).
        Level::Sse42 if supported(Level::Sse42) => unsafe {
            super::x86::activation_sse(func, values)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: this CPU supports NEON (the guard; Accelerate's support check includes
        // NEON).
        Level::Neon | Level::Accelerate if supported(level) => unsafe {
            super::aarch64::activation_neon(func, values)
        },
        _ => match func {
            Func::Tanh => super::reference::tanh_f32(values),
            Func::Sigmoid => super::reference::sigmoid_f32(values),
        },
    }
}

/// Splits `values` for a `W`-lane kernel: the whole vectors, and the last
/// `values.len() mod W` values copied into a zero-padded vector, with their count. The
/// kernel runs the padded vector like any other and copies the count back, so a value's
/// result does not depend on whether it falls in the tail.
pub(super) fn split<const W: usize>(values: &mut [f32]) -> (&mut [[f32; W]], &mut [f32], [f32; W]) {
    let (chunks, tail) = values.as_chunks_mut::<W>();
    let mut pad = [0.0f32; W];
    pad[..tail.len()].copy_from_slice(tail);
    (chunks, tail, pad)
}

/// Tests for the activation kernels: each level against the reference twin, and the twin
/// and each level against the correctly rounded result.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Level;

    /// The levels this CPU can run.
    fn levels() -> Vec<Level> {
        Level::ALL
            .iter()
            .copied()
            .filter(|&l| supported(l))
            .collect()
    }

    /// The distance between two f32 in units in the last place, counting across zero (so
    /// -0 and +0 are 0 apart). Both must be non-NaN.
    fn ulps(a: f32, b: f32) -> u64 {
        let key = |v: f32| {
            let bits = v.to_bits() as i64;
            if bits & SIGN as i64 != 0 {
                -(bits & 0x7fff_ffff)
            } else {
                bits
            }
        };
        (key(a) - key(b)).unsigned_abs()
    }

    /// `tanh` correctly rounded to f32 (f64 `tanh` has far more precision than needed).
    fn tanh_exact(x: f32) -> f32 {
        (x as f64).tanh() as f32
    }

    /// The sigmoid correctly rounded to f32, from f64.
    fn sigmoid_exact(x: f32) -> f32 {
        let x = x as f64;
        let s = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.exp() / (1.0 + x.exp())
        };
        s as f32
    }

    /// One f32 in every `stride` bit patterns, both signs, finite only.
    fn sweep(stride: u32) -> Vec<f32> {
        (0..0x7f80_0000u32)
            .step_by(stride as usize)
            .flat_map(|b| [f32::from_bits(b), f32::from_bits(b | SIGN)])
            .collect()
    }

    /// Values where the formulas change or saturate, and the specials.
    fn edges() -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            1e-45,
            1e-30,
            1e-4,
            TANH_SMALL,
            TANH_SMALL.next_down(),
            TANH_SMALL.next_up(),
            1.0,
            8.0,
            9.01,
            10.0,
            51.0,
            52.0,
            87.0,
            88.8,
            103.0,
            104.0,
            150.0,
            f32::MAX,
            f32::INFINITY,
        ];
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        v.extend(neg);
        v
    }

    /// Runs `func` on `xs` at `level`, returning the results.
    fn apply(level: Level, func: Func, xs: &[f32]) -> Vec<f32> {
        let mut v = xs.to_vec();
        run(level, func, &mut v);
        v
    }

    /// The largest ulp error of `func` at `level` against `exact` over `xs`, with the input
    /// that gives it.
    fn worst(level: Level, func: Func, xs: &[f32], exact: fn(f32) -> f32) -> (u64, f32) {
        let got = apply(level, func, xs);
        xs.iter()
            .zip(&got)
            .map(|(&x, &g)| (ulps(g, exact(x)), x))
            .max_by_key(|&(u, _)| u)
            .unwrap_or((0, 0.0))
    }

    /// `tanh` and `sigmoid` are within 3 ulp of the correctly rounded result,
    /// at every level, over a sweep of the finite f32 and the edge values.
    #[test]
    fn within_the_stated_ulp_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        for level in levels() {
            let (t, tx) = worst(level, Func::Tanh, &xs, tanh_exact);
            assert!(t <= 3, "{level}: tanh is {t} ulp off at {tx:e}");
            let (s, sx) = worst(level, Func::Sigmoid, &xs, sigmoid_exact);
            assert!(s <= 3, "{level}: sigmoid is {s} ulp off at {sx:e}");
        }
    }

    /// Softplus from f64, rounded to f32.
    fn softplus_exact(x: f32) -> f32 {
        let x = x as f64;
        (x.max(0.0) + (-x.abs()).exp().ln_1p()) as f32
    }

    /// ELU with alpha 1 from f64, rounded to f32.
    fn elu_exact(x: f32) -> f32 {
        let w = x as f64;
        (if w > 0.0 { w } else { w.exp_m1() }) as f32
    }

    /// The worst ulp error of `f` over `xs` against `exact`, NaN only where `exact` is.
    fn worst_of(xs: &[f32], f: impl Fn(&mut [f32]), exact: fn(f32) -> f32) -> (u64, f32) {
        let mut v = xs.to_vec();
        f(&mut v);
        let mut worst = (0, 0.0);
        for (&x, &g) in xs.iter().zip(&v) {
            let w = exact(x);
            assert_eq!(g.is_nan(), w.is_nan(), "{x:e}: {g:e} against {w:e}");
            if !w.is_nan() && ulps(g, w) > worst.0 {
                worst = (ulps(g, w), x);
            }
        }
        worst
    }

    /// Softplus within 2 ulp and ELU within 1 ulp of the correctly rounded result at the
    /// levels that round as the reference does, one ulp more at the fused levels,
    /// over the sweep and the edge values; at SSE4.2 both give the reference's bits.
    #[test]
    fn softplus_and_elu_within_their_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        xs.push(f32::NAN);
        for level in levels() {
            let fused = u64::from(level != Level::Reference && level != Level::Sse42);
            let (s, sx) = worst_of(&xs, |v| softplus_f32_with(level, v), softplus_exact);
            assert!(s <= 2 + fused, "{level}: softplus is {s} ulp off at {sx:e}");
            let (e, ex) = worst_of(&xs, |v| elu_f32_with(level, 1.0, v), elu_exact);
            assert!(e <= 1 + fused, "{level}: elu is {e} ulp off at {ex:e}");
        }
        if supported(Level::Sse42) {
            for f in [softplus_f32_with, |l, v: &mut [f32]| {
                elu_f32_with(l, 1.5, v)
            }] {
                let (mut a, mut b) = (xs.clone(), xs.clone());
                f(Level::Reference, &mut a);
                f(Level::Sse42, &mut b);
                for ((x, a), b) in xs.iter().zip(&a).zip(&b) {
                    assert!(
                        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                        "{x:e}"
                    );
                }
            }
        }
    }

    /// Every finite f32 through softplus and ELU at every level, within the stated bounds.
    /// Slow, so run by hand:
    /// `cargo test --release -p cavitation-core --lib every_f32_softplus -- --ignored`.
    #[test]
    #[ignore = "checks all 4 billion f32 values; minutes in release"]
    fn every_f32_softplus_and_elu_within_their_bounds() {
        const BLOCK: u32 = 1 << 20;
        for level in levels() {
            let (mut s_max, mut e_max) = ((0, 0.0f32), (0, 0.0f32));
            let mut start = 0u32;
            while start < 0x7f80_0000 {
                let xs: Vec<f32> = (start..start.saturating_add(BLOCK).min(0x7f80_0000))
                    .flat_map(|b| [f32::from_bits(b), f32::from_bits(b | SIGN)])
                    .collect();
                let s = worst_of(&xs, |v| softplus_f32_with(level, v), softplus_exact);
                if s.0 > s_max.0 {
                    s_max = s;
                }
                let e = worst_of(&xs, |v| elu_f32_with(level, 1.0, v), elu_exact);
                if e.0 > e_max.0 {
                    e_max = e;
                }
                start = start.saturating_add(BLOCK);
            }
            println!(
                "{level}: softplus {} ulp at {:e}, elu {} ulp at {:e}",
                s_max.0, s_max.1, e_max.0, e_max.1
            );
            let fused = u64::from(level != Level::Reference && level != Level::Sse42);
            assert!(s_max.0 <= 2 + fused && e_max.0 <= 1 + fused, "{level}");
        }
    }

    /// `tanh`, the sigmoid, softplus and ELU give a value the same bits whatever the
    /// slice's length and the value's place in it, across the SSE4.2 kernels' blocks of 16
    /// values, their single vectors and their padded tails.
    #[test]
    fn kernels_do_not_depend_on_position() {
        let xs: Vec<f32> = (0..53).map(|i| -9.0 + 0.37 * i as f32).collect();
        let fs: [fn(Level, &mut [f32]); 4] = [
            tanh_f32_with,
            sigmoid_f32_with,
            softplus_f32_with,
            |l, v| elu_f32_with(l, 1.5, v),
        ];
        for level in levels() {
            for f in fs {
                for len in 0..=xs.len() {
                    let mut all = xs[..len].to_vec();
                    f(level, &mut all);
                    for (i, &x) in xs[..len].iter().enumerate() {
                        let mut one = [x];
                        f(level, &mut one);
                        assert_eq!(one[0].to_bits(), all[i].to_bits(), "{level}: {len} {i}");
                    }
                }
            }
        }
    }

    /// ELU keeps the sign of a zero at every level (`alpha · (e^±0 − 1)` is `alpha · ±0`),
    /// and is `alpha · x` for an input below `EXPM1_TINY` in size; SELU and CELU follow.
    #[test]
    fn elu_family_keeps_signed_zero() {
        for level in levels() {
            let mut v = [-0.0f32, 0.0, -1e-30, -1e-40];
            elu_f32_with(level, 2.0, &mut v);
            assert!(
                v[0] == 0.0 && v[0].is_sign_negative(),
                "{level}: elu(-0) is {}",
                v[0]
            );
            assert!(
                v[1] == 0.0 && v[1].is_sign_positive(),
                "{level}: elu(+0) is {}",
                v[1]
            );
            assert_eq!(v[2], -2e-30, "{level}");
            assert_eq!(v[3], 2.0 * -1e-40, "{level}");
            let mut v = [-0.0f32, 0.0];
            selu_f32_with(level, &mut v);
            assert!(
                v[0].is_sign_negative() && v[1].is_sign_positive(),
                "{level}: selu"
            );
            let mut v = [-0.0f32, 0.0];
            celu_f32_with(level, 1.5, &mut v);
            assert!(
                v[0].is_sign_negative() && v[1].is_sign_positive(),
                "{level}: celu"
            );
            let mut v = [1000.0f32, f32::INFINITY];
            log_sigmoid_f32_with(level, &mut v);
            assert!(
                v.iter().all(|v| *v == 0.0 && v.is_sign_positive()),
                "{level}: log_sigmoid {v:?}"
            );
        }
    }

    /// Log-sigmoid, softplus at another `beta`, Mish, SELU and CELU against `f64` over the
    /// sweep and the edge values at every level, within the bounds their docs state; NaN
    /// exactly where the `f64` formula gives NaN.
    #[test]
    fn composite_activations_within_their_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        xs.push(f32::NAN);
        for level in levels() {
            let fused = u64::from(level != Level::Reference && level != Level::Sse42);
            let (l, lx) = worst_of(
                &xs,
                |v| log_sigmoid_f32_with(level, v),
                |x| -softplus_exact(-x),
            );
            assert!(
                l <= 2 + fused,
                "{level}: log_sigmoid is {l} ulp off at {lx:e}"
            );
            // `beta` 2: against the exact softplus of the rounded `z = 2x` (exact here), over 2.
            let (s, sx) = worst_of(
                &xs,
                |v| softplus_beta_f32_with(level, 2.0, 20.0, v),
                |x| {
                    let z = 2.0 * x;
                    if z > 20.0 {
                        x
                    } else {
                        (softplus_exact64(z) / 2.0) as f32
                    }
                },
            );
            assert!(
                s <= 2 + fused,
                "{level}: softplus(beta 2) is {s} ulp off at {sx:e}"
            );
            let (e, ex) = worst_of(
                &xs,
                |v| selu_f32_with(level, v),
                |x| {
                    let (scale, alpha) = (crate::ops::SELU_SCALE, crate::ops::SELU_ALPHA);
                    let w = f64::from(x);
                    (if w > 0.0 {
                        scale * w
                    } else {
                        scale * alpha * w.exp_m1()
                    }) as f32
                },
            );
            assert!(e <= 3 + fused, "{level}: selu is {e} ulp off at {ex:e}");
            let (c, cx) = worst_of(
                &xs,
                |v| celu_f32_with(level, 2.5, v),
                |x| {
                    let w = f64::from(x);
                    (if w > 0.0 { w } else { 2.5 * (w / 2.5).exp_m1() }) as f32
                },
            );
            assert!(c <= 3 + fused, "{level}: celu is {c} ulp off at {cx:e}");
            let mut mish = xs.clone();
            mish_f32_with(level, &mut mish);
            let mut m_worst = (0, 0.0f32);
            for (&x, &got) in xs.iter().zip(&mish) {
                let w = f64::from(x);
                let sp = if w > 20.0 {
                    w
                } else {
                    w.max(0.0) + (-w.abs()).exp().ln_1p()
                };
                let want = w * sp.tanh();
                assert_eq!(
                    got.is_nan(),
                    want.is_nan(),
                    "{level}: mish({x:e}) is {got:e}"
                );
                if want.is_nan() {
                    continue;
                }
                if sp < f64::from(f32::MIN_POSITIVE) {
                    let ulp = (want as f32).abs().next_up() - (want as f32).abs();
                    let limit = 6.0 * 2f64.powi(-149) * w.abs() + f64::from(ulp);
                    let err = (f64::from(got) - want).abs();
                    assert!(
                        err <= limit,
                        "{level}: mish({x:e}) is {got:e}, want {want:e}"
                    );
                } else {
                    let u = ulps(got, want as f32);
                    if u > m_worst.0 {
                        m_worst = (u, x);
                    }
                }
            }
            println!(
                "{level}: log_sigmoid {l} at {lx:e}, softplus2 {s} at {sx:e}, selu {e} at {ex:e}, celu {c} at {cx:e}, mish {} at {:e}",
                m_worst.0, m_worst.1
            );
            assert!(
                m_worst.0 <= 13,
                "{level}: mish is {} ulp off at {:e}",
                m_worst.0,
                m_worst.1
            );
        }
    }

    /// Softplus in f64 for an f32 input, before rounding.
    fn softplus_exact64(x: f32) -> f64 {
        let x = f64::from(x);
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }

    /// CELU with a negative `alpha` takes the scalar formula, and every composite gives a
    /// value the same bits wherever it falls in the slice (150 values: two blocks and a part).
    #[test]
    fn composite_activations_do_not_depend_on_the_block() {
        let xs: Vec<f32> = (0..150).map(|i| -30.0 + 0.4 * i as f32).collect();
        let fs: [fn(Level, &mut [f32]); 6] = [
            log_sigmoid_f32_with,
            mish_f32_with,
            selu_f32_with,
            |l, v| softplus_beta_f32_with(l, 0.5, 3.0, v),
            |l, v| celu_f32_with(l, 1.5, v),
            |l, v| celu_f32_with(l, -1.5, v),
        ];
        for level in levels() {
            for f in fs {
                let mut all = xs.clone();
                f(level, &mut all);
                for (i, &x) in xs.iter().enumerate() {
                    let mut one = [x];
                    f(level, &mut one);
                    assert_eq!(one[0].to_bits(), all[i].to_bits(), "{level}: {x}");
                }
            }
        }
        let mut v = [-1.0f32, 2.0];
        celu_f32_with(Level::Reference, -1.5, &mut v);
        assert_eq!(v, [-1.5 * (-1.0f32 / -1.5).exp_m1(), 2.0]);
    }

    /// SiLU is within 7 ulp of the correctly rounded result (where the sigmoid is subnormal,
    /// within `3 · 2⁻¹⁴⁹ · |x|` plus 1 ulp), and the tanh form of GELU
    /// within `|x| · 2⁻²¹` plus 2 ulp of the `f64` result, at every level, over the sweep
    /// and the edge values; both give NaN exactly where the `f64` formula does.
    #[test]
    fn silu_and_gelu_tanh_within_their_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        let ulp = |v: f32| v.abs().next_up() - v.abs();
        for level in levels() {
            let mut silu = xs.clone();
            silu_f32_with(level, &mut silu);
            let mut gelu = xs.clone();
            gelu_tanh_f32_with(level, &mut gelu);
            for (i, &x) in xs.iter().enumerate() {
                let w = x as f64;
                let s = if w >= 0.0 {
                    1.0 / (1.0 + (-w).exp())
                } else {
                    w.exp() / (1.0 + w.exp())
                };
                let want = (w * s) as f32;
                let got = silu[i];
                assert_eq!(
                    got.is_nan(),
                    want.is_nan(),
                    "{level}: silu({x:e}) is {got:e}"
                );
                if !want.is_nan() && s < f64::from(f32::MIN_POSITIVE) {
                    let limit = 3.0 * 2f64.powi(-149) * w.abs() + f64::from(ulp(want));
                    let err = (f64::from(got) - w * s).abs();
                    assert!(err <= limit, "{level}: silu({x:e}) is {got:e}");
                } else if !want.is_nan() {
                    let u = ulps(got, want);
                    assert!(u <= 7, "{level}: silu({x:e}) is {u} ulp off: {got:e}");
                }
                let u = (2.0f64 / std::f64::consts::PI).sqrt() * (w + 0.044715 * w * w * w);
                let want = 0.5 * w * (1.0 + u.tanh());
                let got = gelu[i];
                assert_eq!(
                    got.is_nan(),
                    want.is_nan(),
                    "{level}: gelu({x:e}) is {got:e}"
                );
                if !want.is_nan() && want.is_finite() {
                    let limit = w.abs() * 2f64.powi(-21) + 2.0 * f64::from(ulp(want as f32));
                    let err = (f64::from(got) - want).abs();
                    assert!(
                        err <= limit,
                        "{level}: gelu({x:e}) is {got:e}, want {want:e}"
                    );
                }
            }
        }
    }

    /// A value's SiLU and GELU do not depend on where it falls in the slice: 150 values (two
    /// full blocks and a part) give the bits of one value at a time.
    #[test]
    fn silu_and_gelu_tanh_do_not_depend_on_the_block() {
        let xs: Vec<f32> = (0..150).map(|i| -6.0 + 0.08 * i as f32).collect();
        for level in levels() {
            for f in [silu_f32_with, gelu_tanh_f32_with] {
                let mut all = xs.clone();
                f(level, &mut all);
                for (i, &x) in xs.iter().enumerate() {
                    let mut one = [x];
                    f(level, &mut one);
                    assert_eq!(one[0].to_bits(), all[i].to_bits(), "{level}: {x}");
                }
            }
        }
    }

    /// Every finite f32, at every level, within the stated bounds. Slow (minutes), so run
    /// by hand: `cargo test --release -p cavitation-core --lib every_f32 -- --ignored`.
    #[test]
    #[ignore = "checks all 4 billion f32 values; minutes in release"]
    fn every_f32_within_the_stated_ulp_bounds() {
        const BLOCK: u32 = 1 << 20;
        for level in levels() {
            let (mut t_max, mut s_max) = ((0, 0.0f32), (0, 0.0f32));
            let mut start = 0u32;
            loop {
                let xs: Vec<f32> = (start..start.saturating_add(BLOCK).min(0x7f80_0000))
                    .flat_map(|b| [f32::from_bits(b), f32::from_bits(b | SIGN)])
                    .collect();
                if xs.is_empty() {
                    break;
                }
                let t = worst(level, Func::Tanh, &xs, tanh_exact);
                if t.0 > t_max.0 {
                    t_max = t;
                }
                let s = worst(level, Func::Sigmoid, &xs, sigmoid_exact);
                if s.0 > s_max.0 {
                    s_max = s;
                }
                start += BLOCK;
            }
            println!(
                "{level}: tanh {} ulp at {:e}, sigmoid {} ulp at {:e}",
                t_max.0, t_max.1, s_max.0, s_max.1
            );
            assert!(t_max.0 <= 3 && s_max.0 <= 3, "{level}");
        }
    }

    /// The reference twin itself is within 2 ulp for both, the bound measured for every
    /// finite f32 by `every_f32_within_the_stated_ulp_bounds`.
    #[test]
    fn the_reference_is_within_its_measured_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        let (t, tx) = worst(Level::Reference, Func::Tanh, &xs, tanh_exact);
        assert!(t <= 2, "tanh is {t} ulp off at {tx:e}");
        let (s, sx) = worst(Level::Reference, Func::Sigmoid, &xs, sigmoid_exact);
        assert!(s <= 2, "sigmoid is {s} ulp off at {sx:e}");
    }

    /// Every level is within 2 ulp of the reference twin, and SSE4.2 matches it bit for
    /// bit.
    #[test]
    fn levels_agree_with_the_reference() {
        let mut xs = sweep(4099);
        xs.extend(edges());
        for func in [Func::Tanh, Func::Sigmoid] {
            let want = apply(Level::Reference, func, &xs);
            for level in levels() {
                let got = apply(level, func, &xs);
                for ((&x, &g), &w) in xs.iter().zip(&got).zip(&want) {
                    if level == Level::Sse42 {
                        assert_eq!(g.to_bits(), w.to_bits(), "{level} {func:?} at {x:e}");
                    } else {
                        assert!(ulps(g, w) <= 2, "{level} {func:?} at {x:e}: {g:e} vs {w:e}");
                    }
                }
            }
        }
    }

    /// NaN stays NaN with its payload, the infinities and signed zeros give the documented
    /// values, at every level.
    #[test]
    fn specials() {
        let nan = f32::from_bits(0x7fc0_1234);
        let xs = [nan, -nan, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
        for level in levels() {
            let t = apply(level, Func::Tanh, &xs);
            assert_eq!(t[0].to_bits(), nan.to_bits(), "{level}");
            assert_eq!(t[1].to_bits(), (-nan).to_bits(), "{level}");
            assert_eq!(&t[2..4], &[1.0, -1.0], "{level}");
            assert_eq!((t[4].to_bits(), t[5].to_bits()), (0, SIGN), "{level}");
            let s = apply(level, Func::Sigmoid, &xs);
            assert_eq!(s[0].to_bits(), nan.to_bits(), "{level}");
            assert_eq!(&s[2..], &[1.0, 0.0, 0.5, 0.5], "{level}");
        }
    }

    /// `tanh` is odd, bit for bit, and results stay in [-1, 1] and [0, 1].
    #[test]
    fn symmetry_and_range() {
        let xs = sweep(65_537);
        for level in levels() {
            let t = apply(level, Func::Tanh, &xs);
            let s = apply(level, Func::Sigmoid, &xs);
            for (i, &x) in xs.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&t[i]),
                    "{level} tanh({x:e}) = {:e}",
                    t[i]
                );
                assert!(
                    (0.0..=1.0).contains(&s[i]),
                    "{level} sigmoid({x:e}) = {:e}",
                    s[i]
                );
                let neg = apply(level, Func::Tanh, &[-x])[0];
                assert_eq!(
                    neg.to_bits(),
                    (-t[i]).to_bits(),
                    "{level} tanh is odd at {x:e}"
                );
            }
        }
    }

    /// A value gives the same bits wherever it sits in the slice, tail included.
    #[test]
    fn results_do_not_depend_on_position() {
        let xs: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.37).collect();
        for level in levels() {
            for func in [Func::Tanh, Func::Sigmoid] {
                let whole = apply(level, func, &xs);
                for (i, &x) in xs.iter().enumerate() {
                    let one = apply(level, func, &[x])[0];
                    assert_eq!(one.to_bits(), whole[i].to_bits(), "{level} {func:?} at {i}");
                }
            }
        }
    }

    /// An empty slice is left alone.
    #[test]
    fn empty() {
        for level in levels() {
            assert!(apply(level, Func::Tanh, &[]).is_empty());
        }
    }
}
