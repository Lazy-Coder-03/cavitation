//! x86_64 microkernels: the SSE4.2 GEMM tile (6 by 8, multiply then add; SSE has no fused
//! multiply-add) with its packing routines, the AVX2 tile (6 by 16, fused multiply-add), and
//! the SSE4.2 and AVX2 inference matvec kernels, and the SSE4.2 and AVX2 `tanh` and sigmoid
//! kernels.

use std::arch::x86_64::{
    __m128, __m256, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_UNORD_Q, _mm_add_epi32,
    _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_blendv_ps, _mm_castsi128_ps, _mm_cmpge_ps,
    _mm_cmpgt_ps, _mm_cmple_ps, _mm_cmplt_ps, _mm_cmpunord_ps, _mm_cvttps_epi32, _mm_div_ps,
    _mm_floor_ps, _mm_load_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_movehl_ps, _mm_movelh_ps,
    _mm_mul_ps, _mm_or_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_shuffle_ps,
    _mm_slli_epi32, _mm_srai_epi32, _mm_storeu_ps, _mm_sub_epi32, _mm_sub_ps, _mm_unpackhi_ps,
    _mm_unpacklo_ps, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps,
    _mm256_blendv_ps, _mm256_broadcast_ss, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_cvttps_epi32,
    _mm256_div_ps, _mm256_floor_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps,
    _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_slli_epi32, _mm256_srai_epi32, _mm256_storeu_ps, _mm256_sub_epi32,
    _mm256_sub_ps,
};

use super::MatRef;
use super::activation::{
    EXP_MIN, EXP_P, EXPM1_MIN, EXPM1_TINY, Func, LN2_HI, LN2_LO, LOG_P, LOG2E, SIGN, TANH_Q,
    TANH_SMALL, split,
};
use super::gemm::{Dest, Kernel, pack_a, pack_b};
use super::matvec::Job;

/// The SSE4.2 GEMM kernel: 6 by 8 tiles held in 12 four-wide accumulators (with the two
/// `B` vectors, all 16 SSE registers in use, nothing spilled).
///
/// Packed `A` holds each value four times over (`A_REP = 4`), so the kernel loads it already
/// broadcast: on cores with one four-wide multiply per cycle, a shuffle or a register copy
/// in the inner loop costs as much as a multiply.
///
/// Block sizes `kc = 256`, `mc = 96`, `nc = 2048`, with one compute task per thread (set in
/// the driver), were chosen by running `cavitation-bench gemm --shapes mnist128` and
/// `--shapes digits128` with each candidate (`kc` 128 to 384, `mc` 48 to 144, 1 to 3 tasks
/// per thread), then re-checked on an idle 4-core Intel Celeron N5105 by timing whole
/// training steps of the 784 → 1024 → 256 → 10 network at batch 32 and 128 and the
/// 64 → 256 → 128 → 10 network at batch 128: `kc` 384 and 512, `mc` 48, 144 and 192, `nc`
/// 512 and 1024 and 2 or 3 tasks per thread were each as fast or slower on all three.
pub(super) struct Sse;

// SAFETY: `tile` reads `24·kc` values from `a` and `8·kc` from `b` (aligned loads at multiples
// of 4 floats from 16-byte aligned starts), and reads and writes C only at the
// `dest.rows` by `dest.cols` elements its contract names.
unsafe impl Kernel for Sse {
    const MR: usize = 6;
    const NR: usize = 8;
    const KC: usize = 256;
    const MC: usize = 96;
    const NC: usize = 2048;
    const A_REP: usize = 4;

    fn pack_a(a: &MatRef<f32>, rows: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        pack_a::<4>(a, rows, ks, 6, dst, pack_a_6);
    }

    fn pack_b(b: &MatRef<f32>, cols: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        if cols.1 < 8 || !pack_b_8(b, cols.0, ks, dst) {
            pack_b(b, cols, ks, 8, dst);
        }
    }

    unsafe fn tile(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
        // `tile_6x8` makes aligned loads from the A panel, which the driver aligns to 16
        // bytes only when `MR · A_REP` is a multiple of 4.
        const { assert!((Self::MR * Self::A_REP) % 4 == 0) };
        // SAFETY: the caller upholds this function's contract, which is `tile_6x8`'s, and
        // runs this kernel only on CPUs with SSE4.2.
        unsafe { tile_6x8(kc, a, b, alpha, beta, dest) }
    }
}

/// Loads 4 floats from the start of `v`.
///
/// # Panics
///
/// Panics if `v` has fewer than 4.
fn load4(v: &[f32]) -> __m128 {
    let v = &v[..4];
    // SAFETY: `v` holds 4 floats; the load is unaligned.
    unsafe { _mm_loadu_ps(v.as_ptr()) }
}

/// Stores `x` into the first 4 floats of `out`.
///
/// # Panics
///
/// Panics if `out` has fewer than 4.
fn store4(out: &mut [f32], x: __m128) {
    let out = &mut out[..4];
    // SAFETY: `out` holds 4 floats; the store is unaligned.
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), x) }
}

/// Writes the four lanes of `v`, each broadcast to 4 floats, at `out[0]`, `out[stride]`,
/// `out[2·stride]` and `out[3·stride]`.
fn spread(out: &mut [f32], stride: usize, v: __m128) {
    // SAFETY: SSE is part of the x86_64 baseline, so every x86_64 CPU runs these shuffles.
    let lanes = unsafe {
        [
            _mm_shuffle_ps::<0x00>(v, v),
            _mm_shuffle_ps::<0x55>(v, v),
            _mm_shuffle_ps::<0xAA>(v, v),
            _mm_shuffle_ps::<0xFF>(v, v),
        ]
    };
    for (q, lane) in lanes.into_iter().enumerate() {
        store4(&mut out[q * stride..], lane);
    }
}

/// `x` in all four lanes.
fn splat(x: f32) -> __m128 {
    // SAFETY: SSE is part of the x86_64 baseline.
    unsafe { _mm_set1_ps(x) }
}

/// Packs one full 6-row panel of `A` (rows `r0..r0 + 6`, columns `p0..p0 + kc`) into `out`
/// as `kc` groups of 6 values each broadcast to 4 floats, with vector loads and shuffles,
/// when `A`'s rows or columns are contiguous. Returns false, writing nothing, otherwise.
fn pack_a_6(a: &MatRef<f32>, r0: usize, (p0, kc): (usize, usize), out: &mut [f32]) -> bool {
    let (rs, cs) = (a.row_stride(), a.col_stride());
    let data = a.data;
    if cs == 1 {
        // Rows contiguous along k: 4 steps of k from each row at a time.
        let starts: [usize; 6] = std::array::from_fn(|r| (r0 + r) * rs + p0);
        let mut p = 0;
        while p + 4 <= kc {
            for (r, &s) in starts.iter().enumerate() {
                spread(&mut out[p * 24 + 4 * r..], 24, load4(&data[s + p..]));
            }
            p += 4;
        }
        for p in p..kc {
            for (r, &s) in starts.iter().enumerate() {
                store4(&mut out[p * 24 + 4 * r..], splat(data[s + p]));
            }
        }
        true
    } else if rs == 1 {
        // Columns contiguous along the rows: rows 0 to 3 in one load, 4 and 5 alone.
        for p in 0..kc {
            let s = (p0 + p) * cs + r0;
            let group = &mut out[p * 24..(p + 1) * 24];
            spread(group, 4, load4(&data[s..]));
            store4(&mut group[16..], splat(data[s + 4]));
            store4(&mut group[20..], splat(data[s + 5]));
        }
        true
    } else {
        false
    }
}

/// The transpose of the 4 by 4 matrix whose rows are `r`.
fn transpose4(r: [__m128; 4]) -> [__m128; 4] {
    // SAFETY: SSE is part of the x86_64 baseline.
    unsafe {
        let t0 = _mm_unpacklo_ps(r[0], r[1]);
        let t1 = _mm_unpacklo_ps(r[2], r[3]);
        let t2 = _mm_unpackhi_ps(r[0], r[1]);
        let t3 = _mm_unpackhi_ps(r[2], r[3]);
        [
            _mm_movelh_ps(t0, t1),
            _mm_movehl_ps(t1, t0),
            _mm_movelh_ps(t2, t3),
            _mm_movehl_ps(t3, t2),
        ]
    }
}

/// Packs one full 8-column panel of `B` (columns `j0..j0 + 8`, rows `p0..p0 + kc`) into
/// `out` as `kc` groups of 8 values, with vector loads, when `B`'s rows or columns are
/// contiguous. Returns false, writing nothing, otherwise.
fn pack_b_8(b: &MatRef<f32>, j0: usize, (p0, kc): (usize, usize), out: &mut [f32]) -> bool {
    let (rs, cs) = (b.row_stride(), b.col_stride());
    let data = b.data;
    if cs == 1 {
        // Rows contiguous along n: two vectors per step of k.
        for p in 0..kc {
            let s = (p0 + p) * rs + j0;
            store4(&mut out[8 * p..], load4(&data[s..]));
            store4(&mut out[8 * p + 4..], load4(&data[s + 4..]));
        }
        true
    } else if rs == 1 {
        // Columns contiguous along k: 4 steps of k from 4 columns, transposed in registers.
        let starts: [usize; 8] = std::array::from_fn(|j| (j0 + j) * cs + p0);
        let mut p = 0;
        while p + 4 <= kc {
            for half in 0..2 {
                let rows = std::array::from_fn(|q| load4(&data[starts[4 * half + q] + p..]));
                for (i, v) in transpose4(rows).into_iter().enumerate() {
                    store4(&mut out[8 * (p + i) + 4 * half..], v);
                }
            }
            p += 4;
        }
        for p in p..kc {
            for (j, &s) in starts.iter().enumerate() {
                out[8 * p + j] = data[s + p];
            }
        }
        true
    } else {
        false
    }
}

/// The 6 by 8 SSE tile; see [`Kernel::tile`] for the contract.
///
/// # Safety
///
/// As [`Kernel::tile`] with `MR = 6`, `NR = 8` and `A_REP = 4`, on a CPU with SSE4.2.
#[target_feature(enable = "sse4.2")]
unsafe fn tile_6x8(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
    let mut acc: [[__m128; 2]; 6] = [[_mm_setzero_ps(); 2]; 6];
    for p in 0..kc {
        // SAFETY: `p < kc`, so floats `8p..8p + 8` of the B panel are inside it, and `8p`
        // floats from a 16-byte aligned start is 16-byte aligned.
        let (b0, b1) = unsafe { (_mm_load_ps(b.add(8 * p)), _mm_load_ps(b.add(8 * p + 4))) };
        for (r, row) in acc.iter_mut().enumerate() {
            // SAFETY: `24p + 4r + 4 <= 24·kc`, inside the A panel (each value stored four
            // times), and a multiple of 4 floats from a 16-byte aligned start, so `pa` is
            // aligned for `__m128`. The value is loaded twice, once per multiply: volatile
            // keeps the two loads from being merged into one load and a register copy, and
            // the copy would take a slot from the multiplies.
            let (x0, x1) = unsafe {
                let pa = a.add(24 * p + 4 * r).cast::<__m128>();
                (std::ptr::read_volatile(pa), std::ptr::read_volatile(pa))
            };
            row[0] = _mm_add_ps(row[0], _mm_mul_ps(x0, b0));
            row[1] = _mm_add_ps(row[1], _mm_mul_ps(x1, b1));
        }
    }
    let va = _mm_set1_ps(alpha);
    for row in &mut acc {
        row[0] = _mm_mul_ps(va, row[0]);
        row[1] = _mm_mul_ps(va, row[1]);
    }
    if dest.rows == 6 && dest.cols == 8 && dest.cs == 1 {
        let vb = _mm_set1_ps(beta);
        for (r, row) in acc.iter().enumerate() {
            // SAFETY: row `r < 6 = dest.rows` and its 8 columns are elements of C, contiguous
            // since `cs = 1`; the caller gives this call sole access to them.
            unsafe {
                let p = dest.ptr.add(r * dest.rs);
                if beta == 0.0 {
                    _mm_storeu_ps(p, row[0]);
                    _mm_storeu_ps(p.add(4), row[1]);
                } else {
                    let c0 = _mm_mul_ps(vb, _mm_loadu_ps(p));
                    let c1 = _mm_mul_ps(vb, _mm_loadu_ps(p.add(4)));
                    _mm_storeu_ps(p, _mm_add_ps(row[0], c0));
                    _mm_storeu_ps(p.add(4), _mm_add_ps(row[1], c1));
                }
            }
        }
    } else {
        // An edge tile, or C with non-unit column stride: the same arithmetic, one element
        // at a time, storing only the valid part.
        let mut t = [0.0f32; 48];
        for (r, row) in acc.iter().enumerate() {
            // SAFETY: `t` has room for 8 floats at `8r` (r < 6).
            unsafe {
                _mm_storeu_ps(t.as_mut_ptr().add(8 * r), row[0]);
                _mm_storeu_ps(t.as_mut_ptr().add(8 * r + 4), row[1]);
            }
        }
        for r in 0..dest.rows {
            for j in 0..dest.cols {
                // SAFETY: `r < dest.rows` and `j < dest.cols`, so this is an element of C
                // the caller gives this call sole access to.
                unsafe {
                    let p = dest.ptr.add(r * dest.rs + j * dest.cs);
                    *p = if beta == 0.0 {
                        t[8 * r + j]
                    } else {
                        t[8 * r + j] + beta * *p
                    };
                }
            }
        }
    }
}

/// The AVX2 GEMM kernel: 6 by 16 tiles held in 12 eight-wide accumulators, updated with
/// fused multiply-add (with the two `B` vectors and one broadcast value of `A`, 15 of the
/// 16 AVX registers in use, nothing spilled).
///
/// Packed `A` holds each value once (`A_REP = 1`): AVX2's broadcast from memory
/// (`vbroadcastss`) loads one float into all eight lanes in a single load, without a
/// shuffle, so storing each value eight times over would only make the packed `A` panel
/// eight times larger. Both panels use the general packing routines.
///
/// Block sizes `kc = 256`, `mc = 96`, `nc = 2048` are estimates, not tuned: with `kc = 256`
/// one `A` panel (6 KB) and one `B` panel (16 KB) fit in a 32 KB L1 data cache, and one
/// task's packed rows of `A` (96 KB) in L2.
pub(super) struct Avx2;

// SAFETY: `tile` reads `6·kc` values from `a` and `16·kc` from `b`, with unaligned loads and
// one-float broadcasts (so it needs only the 4-byte alignment of f32), and reads and writes C
// only at the `dest.rows` by `dest.cols` elements its contract names.
unsafe impl Kernel for Avx2 {
    const MR: usize = 6;
    const NR: usize = 16;
    const KC: usize = 256;
    const MC: usize = 96;
    const NC: usize = 2048;
    const A_REP: usize = 1;

    fn pack_a(a: &MatRef<f32>, rows: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        pack_a::<1>(a, rows, ks, 6, dst, |_, _, _, _| false);
    }

    fn pack_b(b: &MatRef<f32>, cols: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        pack_b(b, cols, ks, 16, dst);
    }

    unsafe fn tile(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
        // SAFETY: the caller upholds this function's contract, which is `tile_6x16`'s, and
        // runs this kernel only on CPUs with AVX2 and FMA.
        unsafe { tile_6x16(kc, a, b, alpha, beta, dest) }
    }
}

/// The 6 by 16 AVX2 tile; see [`Kernel::tile`] for the contract. Each step of the sum is one
/// fused multiply-add (one rounding instead of two).
///
/// # Safety
///
/// As [`Kernel::tile`] with `MR = 6`, `NR = 16` and `A_REP = 1`, on a CPU with AVX2 and FMA;
/// `a` and `b` need only the alignment of f32.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_6x16(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
    let mut acc: [[__m256; 2]; 6] = [[_mm256_setzero_ps(); 2]; 6];
    for p in 0..kc {
        // SAFETY: `p < kc`, so floats `16p..16p + 16` of the B panel are inside it; the
        // loads are unaligned.
        let (b0, b1) = unsafe {
            (
                _mm256_loadu_ps(b.add(16 * p)),
                _mm256_loadu_ps(b.add(16 * p + 8)),
            )
        };
        for (r, row) in acc.iter_mut().enumerate() {
            // SAFETY: `6p + r < 6·kc`, inside the A panel; a one-float load.
            let x = unsafe { _mm256_broadcast_ss(&*a.add(6 * p + r)) };
            row[0] = _mm256_fmadd_ps(x, b0, row[0]);
            row[1] = _mm256_fmadd_ps(x, b1, row[1]);
        }
    }
    let va = _mm256_set1_ps(alpha);
    for row in &mut acc {
        row[0] = _mm256_mul_ps(va, row[0]);
        row[1] = _mm256_mul_ps(va, row[1]);
    }
    if dest.rows == 6 && dest.cols == 16 && dest.cs == 1 {
        let vb = _mm256_set1_ps(beta);
        for (r, row) in acc.iter().enumerate() {
            // SAFETY: row `r < 6 = dest.rows` and its 16 columns are elements of C, contiguous
            // since `cs = 1`; the caller gives this call sole access to them.
            unsafe {
                let p = dest.ptr.add(r * dest.rs);
                if beta == 0.0 {
                    _mm256_storeu_ps(p, row[0]);
                    _mm256_storeu_ps(p.add(8), row[1]);
                } else {
                    // A separate multiply and add, as the edge path below does, so a value
                    // does not depend on whether its tile lies on an edge.
                    let c0 = _mm256_mul_ps(vb, _mm256_loadu_ps(p));
                    let c1 = _mm256_mul_ps(vb, _mm256_loadu_ps(p.add(8)));
                    _mm256_storeu_ps(p, _mm256_add_ps(row[0], c0));
                    _mm256_storeu_ps(p.add(8), _mm256_add_ps(row[1], c1));
                }
            }
        }
    } else {
        // An edge tile, or C with non-unit column stride: the same arithmetic, one element
        // at a time, storing only the valid part.
        let mut t = [0.0f32; 96];
        for (r, row) in acc.iter().enumerate() {
            // SAFETY: `t` has room for 16 floats at `16r` (r < 6).
            unsafe {
                _mm256_storeu_ps(t.as_mut_ptr().add(16 * r), row[0]);
                _mm256_storeu_ps(t.as_mut_ptr().add(16 * r + 8), row[1]);
            }
        }
        for r in 0..dest.rows {
            for j in 0..dest.cols {
                // SAFETY: `r < dest.rows` and `j < dest.cols`, so this is an element of C
                // the caller gives this call sole access to.
                unsafe {
                    let p = dest.ptr.add(r * dest.rs + j * dest.cs);
                    *p = if beta == 0.0 {
                        t[16 * r + j]
                    } else {
                        t[16 * r + j] + beta * *p
                    };
                }
            }
        }
    }
}

/// The SSE4.2 inference matvec, streaming the weights in storage order: the input is taken
/// 8 rows at a time (then one at a time for the last `k mod 8`), and each group adds its rows,
/// in order, into every four-wide vector of outputs, keeping the partial sums in `out`
/// between groups. Each step of a sum is a multiply and then an add, as in the reference,
/// so every output equals the reference's bit for bit. The last `n mod 4` outputs are
/// computed one at a time, and the bias and ReLU are applied in a final pass.
///
/// # Safety
///
/// `job` must satisfy [`Job`]'s invariant for `out.len()` outputs, and the CPU must support
/// SSE4.2.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn matvec_sse(job: &Job, out: &mut [f32]) {
    let n = out.len();
    let nv = n / 4 * 4;
    let (k, vecs) = (job.x.len(), &mut out[..nv]);
    let mut i = 0;
    while i < k {
        let rows = if i + 8 <= k { 8 } else { 1 };
        // SAFETY: rows `i..i + rows` are below `k` and the columns `0..nv` are outputs, so
        // the weights read are inside `job.w` by the `Job` invariant this function's caller
        // upholds; `vecs` holds `nv` values, a multiple of the vector width; the CPU
        // supports SSE4.2.
        unsafe {
            match (rows, i == 0) {
                (8, true) => sse_rows::<8, true>(job, i, vecs),
                (8, false) => sse_rows::<8, false>(job, i, vecs),
                (_, true) => sse_rows::<1, true>(job, i, vecs),
                _ => sse_rows::<1, false>(job, i, vecs),
            }
        }
        i += rows;
    }
    if k == 0 {
        vecs.fill(0.0);
    }
    let zero = _mm_setzero_ps();
    for (j, o) in vecs.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let mut v = load4(o);
        if let Some(b) = job.bias {
            v = _mm_add_ps(load4(&b[4 * j..]), v);
        }
        if job.relu {
            // 0 where v <= 0 (so -0 becomes +0), v elsewhere; NaN fails the comparison and
            // stays.
            v = _mm_andnot_ps(_mm_cmple_ps(v, zero), v);
        }
        store4(o, v);
    }
    job.scalar_tail(nv, &mut out[nv..]);
}

/// Adds rows `i0..i0 + R` of the SSE4.2 matvec into every four-wide vector of `out`, in row
/// order: `out[j] = out[j] + x[i0]·w[i0][j] + ... + x[i0 + R - 1]·w[i0 + R - 1][j]`, each step
/// a multiply and then an add. With `FIRST` the sums start from zero instead of `out`.
///
/// # Safety
///
/// Weights `(i, j)` must lie inside `job.w` for `i0 <= i < i0 + R` and `j < out.len()` (a
/// multiple of 4), `i0 + R <= job.x.len()`, and the CPU must support SSE4.2.
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse_rows<const R: usize, const FIRST: bool>(job: &Job, i0: usize, out: &mut [f32]) {
    let xs: [__m128; R] = std::array::from_fn(|r| _mm_set1_ps(job.x[i0 + r]));
    let w = job.w.as_ptr();
    let ld = job.ld;
    for (j, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let mut a = if FIRST { _mm_setzero_ps() } else { load4(o) };
        for (r, &xv) in xs.iter().enumerate() {
            // SAFETY: row `i0 + r` and columns `4j..4j + 4` are inside `job.w` (this
            // function's contract), contiguous since a row's columns are adjacent; the load
            // is unaligned.
            let wv = unsafe { _mm_loadu_ps(w.add((i0 + r) * ld + 4 * j)) };
            a = _mm_add_ps(a, _mm_mul_ps(xv, wv));
        }
        store4(o, a);
    }
}

/// The AVX2 inference matvec, streaming the weights in storage order as the SSE4.2 one
/// does, with eight-wide vectors and one fused multiply-add per step. The last `n mod 8`
/// outputs are computed one at a time.
///
/// # Safety
///
/// `job` must satisfy [`Job`]'s invariant for `out.len()` outputs, and the CPU must support
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn matvec_avx2(job: &Job, out: &mut [f32]) {
    let n = out.len();
    let nv = n / 8 * 8;
    let (k, vecs) = (job.x.len(), &mut out[..nv]);
    let mut i = 0;
    while i < k {
        let rows = if i + 8 <= k { 8 } else { 1 };
        // SAFETY: rows `i..i + rows` are below `k` and the columns `0..nv` are outputs, so
        // the weights read are inside `job.w` by the `Job` invariant this function's caller
        // upholds; `vecs` holds `nv` values, a multiple of the vector width; the CPU
        // supports AVX2 and FMA.
        unsafe {
            match (rows, i == 0) {
                (8, true) => avx2_rows::<8, true>(job, i, vecs),
                (8, false) => avx2_rows::<8, false>(job, i, vecs),
                (_, true) => avx2_rows::<1, true>(job, i, vecs),
                _ => avx2_rows::<1, false>(job, i, vecs),
            }
        }
        i += rows;
    }
    if k == 0 {
        vecs.fill(0.0);
    }
    let zero = _mm256_setzero_ps();
    for (j, o) in vecs.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        // SAFETY: `o` holds 8 floats; the load is unaligned.
        let mut v = unsafe { _mm256_loadu_ps(o.as_ptr()) };
        if let Some(b) = job.bias {
            let b = &b[8 * j..8 * j + 8];
            // SAFETY: `b` holds 8 floats; the load is unaligned.
            v = _mm256_add_ps(unsafe { _mm256_loadu_ps(b.as_ptr()) }, v);
        }
        if job.relu {
            // 0 where v <= 0 (so -0 becomes +0), v elsewhere; NaN fails the ordered
            // comparison and stays.
            v = _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(v, zero), v);
        }
        // SAFETY: `o` holds 8 floats; the store is unaligned.
        unsafe { _mm256_storeu_ps(o.as_mut_ptr(), v) };
    }
    job.scalar_tail(nv, &mut out[nv..]);
}

/// Adds rows `i0..i0 + R` of the AVX2 matvec into every eight-wide vector of `out`, in row
/// order, one fused multiply-add per row. With `FIRST` the sums start from zero instead of
/// `out`.
///
/// # Safety
///
/// Weights `(i, j)` must lie inside `job.w` for `i0 <= i < i0 + R` and `j < out.len()` (a
/// multiple of 8), `i0 + R <= job.x.len()`, and the CPU must support AVX2 and FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_rows<const R: usize, const FIRST: bool>(job: &Job, i0: usize, out: &mut [f32]) {
    let xs: [__m256; R] = std::array::from_fn(|r| _mm256_set1_ps(job.x[i0 + r]));
    let w = job.w.as_ptr();
    let ld = job.ld;
    for (j, o) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut a = if FIRST {
            _mm256_setzero_ps()
        } else {
            // SAFETY: `o` holds 8 floats; the load is unaligned.
            unsafe { _mm256_loadu_ps(o.as_ptr()) }
        };
        for (r, &xv) in xs.iter().enumerate() {
            // SAFETY: row `i0 + r` and columns `8j..8j + 8` are inside `job.w` (this
            // function's contract), contiguous since a row's columns are adjacent; the load
            // is unaligned.
            let wv = unsafe { _mm256_loadu_ps(w.add((i0 + r) * ld + 8 * j)) };
            a = _mm256_fmadd_ps(xv, wv, a);
        }
        // SAFETY: `o` holds 8 floats; the store is unaligned.
        unsafe { _mm256_storeu_ps(o.as_mut_ptr(), a) };
    }
}

/// The SSE4.2 `tanh` or sigmoid kernel over `values`, in place, four lanes at a time (the
/// tail through a padded vector). It performs the reference twin's operations in the same
/// order, one rounding each, so its results equal the reference's bit for bit.
///
/// # Safety
///
/// The CPU must support SSE4.2.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn activation_sse(func: Func, values: &mut [f32]) {
    match func {
        Func::Tanh => sse_quads(values, |x| tanh_sse_k(x), |v| tanh_sse(v)),
        Func::Sigmoid => sse_quads(values, |x| sigmoid_sse_k(x), |v| sigmoid_sse(v)),
    }
}

/// Runs `lanes` over `values` four at a time, two vectors per step so that their long
/// dependency chains overlap, the tail through a zero-padded vector.
#[inline]
#[target_feature(enable = "sse4.2")]
fn sse_lanes(values: &mut [f32], lanes: impl Fn(__m128) -> __m128) {
    let (chunks, tail, mut pad) = split::<4>(values);
    let (pairs, single) = chunks.as_chunks_mut::<2>();
    for [c0, c1] in pairs {
        let (v0, v1) = (lanes(load4(c0)), lanes(load4(c1)));
        store4(c0, v0);
        store4(c1, v1);
    }
    for c in single {
        let v = lanes(load4(c));
        store4(c, v);
    }
    if !tail.is_empty() {
        let v = lanes(load4(&pad));
        store4(&mut pad, v);
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

/// `K` vectors at once: `f` on each.
#[inline]
#[target_feature(enable = "sse4.2")]
fn each<const K: usize>(a: [__m128; K], f: impl Fn(__m128) -> __m128) -> [__m128; K] {
    let mut out = a;
    for (o, &a) in out.iter_mut().zip(a.iter()) {
        *o = f(a);
    }
    out
}

/// `K` pairs of vectors at once: `f` on each pair.
#[inline]
#[target_feature(enable = "sse4.2")]
fn each2<const K: usize>(
    a: [__m128; K],
    b: [__m128; K],
    f: impl Fn(__m128, __m128) -> __m128,
) -> [__m128; K] {
    let mut out = a;
    for (o, (&a, &b)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = f(a, b);
    }
    out
}

/// `e^y` in each lane for `y <= 0`, as the reference's `exp_neg` computes it (NaN lanes
/// give a meaningless value, which the callers replace), on `K` vectors: each step is done
/// for all `K` before the next, so the `K` dependency chains sit side by side and a small
/// out-of-order core can overlap them. Every lane performs the reference's operations in
/// its order, whatever `K` is.
#[inline]
#[target_feature(enable = "sse4.2")]
fn exp_neg_sse_k<const K: usize>(y: [__m128; K]) -> [__m128; K] {
    let y = each(y, |y| _mm_max_ps(y, _mm_set1_ps(EXP_MIN)));
    let n = each(y, |y| {
        _mm_floor_ps(_mm_add_ps(
            _mm_mul_ps(y, _mm_set1_ps(LOG2E)),
            _mm_set1_ps(0.5),
        ))
    });
    let r = each2(y, n, |y, n| {
        _mm_sub_ps(y, _mm_mul_ps(n, _mm_set1_ps(LN2_HI)))
    });
    let r = each2(r, n, |r, n| {
        _mm_sub_ps(r, _mm_mul_ps(n, _mm_set1_ps(LN2_LO)))
    });
    let mut p = [_mm_set1_ps(EXP_P[0]); K];
    for &c in &EXP_P[1..] {
        p = each2(p, r, |p, r| _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(c)));
    }
    let p = each2(p, r, |p, r| {
        _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(_mm_mul_ps(p, r), r), r),
            _mm_set1_ps(1.0),
        )
    });
    each2(p, n, |p, n| {
        // `n` is a whole number in [-150, 0], so the truncating conversion is exact; each
        // half is in [-75, 0], a normal power of two.
        let n = _mm_cvttps_epi32(n);
        let n1 = _mm_srai_epi32::<1>(n);
        let n2 = _mm_sub_epi32(n, n1);
        let bias = _mm_set1_epi32(127);
        let s1 = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n1, bias)));
        let s2 = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n2, bias)));
        _mm_mul_ps(_mm_mul_ps(p, s1), s2)
    })
}

/// `e^y − 1` in each lane for `y <= 0`, as the reference's `expm1_neg` computes it (NaN
/// lanes give a meaningless value, which the callers replace), on `K` vectors side by side
/// as [`exp_neg_sse_k`] does.
#[inline]
#[target_feature(enable = "sse4.2")]
fn expm1_neg_sse_k<const K: usize>(y: [__m128; K]) -> [__m128; K] {
    let low = each(y, |y| _mm_cmplt_ps(y, _mm_set1_ps(EXPM1_MIN)));
    let y = each(y, |y| _mm_max_ps(y, _mm_set1_ps(EXPM1_MIN)));
    let n = each(y, |y| {
        _mm_floor_ps(_mm_add_ps(
            _mm_mul_ps(y, _mm_set1_ps(LOG2E)),
            _mm_set1_ps(0.5),
        ))
    });
    let r = each2(y, n, |y, n| {
        _mm_sub_ps(y, _mm_mul_ps(n, _mm_set1_ps(LN2_HI)))
    });
    let r = each2(r, n, |r, n| {
        _mm_sub_ps(r, _mm_mul_ps(n, _mm_set1_ps(LN2_LO)))
    });
    let mut p = [_mm_set1_ps(EXP_P[0]); K];
    for &c in &EXP_P[1..] {
        p = each2(p, r, |p, r| _mm_add_ps(_mm_mul_ps(p, r), _mm_set1_ps(c)));
    }
    let q = each2(p, r, |p, r| _mm_add_ps(_mm_mul_ps(_mm_mul_ps(p, r), r), r));
    let v = each2(q, n, |q, n| {
        // `n` is a whole number in [-25, 0], so the truncating conversion is exact and
        // `2^n` a normal power of two.
        let n = _mm_cvttps_epi32(n);
        let s = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n, _mm_set1_epi32(127))));
        _mm_add_ps(_mm_mul_ps(q, s), _mm_sub_ps(s, _mm_set1_ps(1.0)))
    });
    each2(v, low, |v, low| _mm_blendv_ps(v, _mm_set1_ps(-1.0), low))
}

/// `ln(1 + t)` in each lane for `t` in `[0, 1]`, as the reference's `log1p_unit` computes
/// it, on `K` vectors side by side as [`exp_neg_sse_k`] does.
#[inline]
#[target_feature(enable = "sse4.2")]
fn log1p_unit_sse_k<const K: usize>(t: [__m128; K]) -> [__m128; K] {
    let one = _mm_set1_ps(1.0);
    let m = each(t, |t| _mm_add_ps(one, t));
    let c = each2(t, m, |t, m| _mm_sub_ps(t, _mm_sub_ps(m, one)));
    let big = each(m, |m| {
        _mm_cmpgt_ps(m, _mm_set1_ps(std::f32::consts::SQRT_2))
    });
    let f = each2(m, big, |m, big| {
        _mm_blendv_ps(m, _mm_mul_ps(m, _mm_set1_ps(0.5)), big)
    });
    let k = each(big, |big| _mm_and_ps(big, one));
    let x = each(f, |f| _mm_sub_ps(f, one));
    let z = each(x, |x| _mm_mul_ps(x, x));
    let mut p = [_mm_set1_ps(LOG_P[0]); K];
    for &cc in &LOG_P[1..] {
        p = each2(p, x, |p, x| _mm_add_ps(_mm_mul_ps(p, x), _mm_set1_ps(cc)));
    }
    let xz = each2(x, z, |x, z| _mm_mul_ps(x, z));
    let y = each2(xz, p, |xz, p| _mm_mul_ps(xz, p));
    let y = each2(y, k, |y, k| {
        _mm_add_ps(y, _mm_mul_ps(k, _mm_set1_ps(LN2_LO)))
    });
    let y = each2(y, z, |y, z| _mm_sub_ps(y, _mm_mul_ps(_mm_set1_ps(0.5), z)));
    let r = each2(x, y, |x, y| _mm_add_ps(x, y));
    let r = each2(r, k, |r, k| {
        _mm_add_ps(r, _mm_mul_ps(k, _mm_set1_ps(LN2_HI)))
    });
    let q = each2(c, m, |c, m| _mm_div_ps(c, m));
    each2(r, q, |r, q| _mm_add_ps(r, q))
}

/// Softplus in each lane, as the reference's `softplus_one` computes it, on `K` vectors
/// side by side as [`exp_neg_sse_k`] does.
#[inline]
#[target_feature(enable = "sse4.2")]
fn softplus_sse_k<const K: usize>(x: [__m128; K]) -> [__m128; K] {
    let sign = _mm_castsi128_ps(_mm_set1_epi32(SIGN as i32));
    let l = log1p_unit_sse_k(exp_neg_sse_k(each(x, |x| _mm_or_ps(x, sign))));
    each2(x, l, |x, l| {
        let m = _mm_and_ps(x, _mm_cmpgt_ps(x, _mm_setzero_ps()));
        let r = _mm_add_ps(m, l);
        _mm_blendv_ps(r, x, _mm_cmpunord_ps(x, x))
    })
}

/// ELU with `alpha` in each lane, as the reference's `elu_one` computes it, on `K` vectors
/// side by side as [`exp_neg_sse_k`] does.
#[inline]
#[target_feature(enable = "sse4.2")]
fn elu_sse_k<const K: usize>(alpha: __m128, x: [__m128; K]) -> [__m128; K] {
    let e = expm1_neg_sse_k(each(x, |x| _mm_min_ps(x, _mm_setzero_ps())));
    each2(x, e, |x, e| {
        let neg = _mm_mul_ps(alpha, e);
        let sign = _mm_castsi128_ps(_mm_set1_epi32(SIGN as i32));
        let tiny = _mm_cmplt_ps(_mm_andnot_ps(sign, x), _mm_set1_ps(EXPM1_TINY));
        let neg = _mm_blendv_ps(neg, _mm_mul_ps(alpha, x), tiny);
        let r = _mm_blendv_ps(neg, x, _mm_cmpgt_ps(x, _mm_setzero_ps()));
        _mm_blendv_ps(r, x, _mm_cmpunord_ps(x, x))
    })
}

/// The SSE4.2 softplus over `values`, in place.
///
/// # Safety
///
/// The CPU must support SSE4.2.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn softplus_sse_slice(values: &mut [f32]) {
    // Four vectors at a time, side by side, so the core can overlap their long chains.
    // Measured with `cavitation-bench activation` on an Intel Celeron N5105: 9.09 ns per
    // value against 16.14 one vector at a time.
    sse_quads(values, |x| softplus_sse_k(x), |v| softplus_sse_k([v])[0]);
}

/// Runs `quad` over `values` sixteen at a time, as four vectors side by side, and `one`
/// over the rest through [`sse_lanes`]. `quad` must compute each vector as `one` does, so a
/// value's result does not depend on its place in the slice.
#[inline]
#[target_feature(enable = "sse4.2")]
fn sse_quads(
    values: &mut [f32],
    quad: impl Fn([__m128; 4]) -> [__m128; 4],
    one: impl Fn(__m128) -> __m128,
) {
    let (blocks, rest) = values.as_chunks_mut::<16>();
    for block in blocks {
        let (quads, _) = block.as_chunks_mut::<4>();
        let x = [0, 1, 2, 3].map(|i| load4(&quads[i]));
        for (q, y) in quads.iter_mut().zip(quad(x)) {
            store4(q, y);
        }
    }
    sse_lanes(rest, one);
}

/// The SSE4.2 ELU with `alpha` over `values`, in place.
///
/// # Safety
///
/// The CPU must support SSE4.2.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn elu_sse_slice(alpha: f32, values: &mut [f32]) {
    let a = _mm_set1_ps(alpha);
    sse_quads(values, |x| elu_sse_k(a, x), |v| elu_sse_k(a, [v])[0]);
}

/// `tanh` in each lane, as the reference's `tanh_one` computes it.
#[inline]
#[target_feature(enable = "sse4.2")]
fn tanh_sse(x: __m128) -> __m128 {
    tanh_sse_k([x])[0]
}

/// [`tanh_sse`] on `K` vectors, the exponential through [`exp_neg_sse_k`] so the `K` chains
/// sit side by side; every lane's operations are the one-vector form's.
#[inline]
#[target_feature(enable = "sse4.2")]
fn tanh_sse_k<const K: usize>(x: [__m128; K]) -> [__m128; K] {
    let sign = _mm_castsi128_ps(_mm_set1_epi32(SIGN as i32));
    let e = exp_neg_sse_k(each(x, |x| {
        _mm_mul_ps(_mm_set1_ps(-2.0), _mm_andnot_ps(sign, x))
    }));
    each2(x, e, |x, e| {
        let one = _mm_set1_ps(1.0);
        let a = _mm_andnot_ps(sign, x);
        let s = _mm_mul_ps(a, a);
        let mut q = _mm_set1_ps(TANH_Q[0]);
        for &c in &TANH_Q[1..] {
            q = _mm_add_ps(_mm_mul_ps(q, s), _mm_set1_ps(c));
        }
        let small = _mm_add_ps(a, _mm_mul_ps(a, _mm_mul_ps(s, q)));
        let big = _mm_div_ps(_mm_sub_ps(one, e), _mm_add_ps(one, e));
        let r = _mm_blendv_ps(big, small, _mm_cmplt_ps(a, _mm_set1_ps(TANH_SMALL)));
        let r = _mm_or_ps(r, _mm_and_ps(sign, x));
        _mm_blendv_ps(r, x, _mm_cmpunord_ps(x, x))
    })
}

/// The logistic sigmoid in each lane, as the reference's `sigmoid_one` computes it.
#[inline]
#[target_feature(enable = "sse4.2")]
fn sigmoid_sse(x: __m128) -> __m128 {
    sigmoid_sse_k([x])[0]
}

/// [`sigmoid_sse`] on `K` vectors, the exponential through [`exp_neg_sse_k`] so the `K`
/// chains sit side by side; every lane's operations are the one-vector form's.
#[inline]
#[target_feature(enable = "sse4.2")]
fn sigmoid_sse_k<const K: usize>(x: [__m128; K]) -> [__m128; K] {
    let sign = _mm_castsi128_ps(_mm_set1_epi32(SIGN as i32));
    // `-|x|`: `x` with its sign bit set.
    let e = exp_neg_sse_k(each(x, |x| _mm_or_ps(x, sign)));
    each2(x, e, |x, e| {
        let one = _mm_set1_ps(1.0);
        let d = _mm_add_ps(one, e);
        let r = _mm_blendv_ps(
            _mm_div_ps(e, d),
            _mm_div_ps(one, d),
            _mm_cmpge_ps(x, _mm_setzero_ps()),
        );
        _mm_blendv_ps(r, x, _mm_cmpunord_ps(x, x))
    })
}

/// The AVX2 `tanh` or sigmoid kernel over `values`, in place, eight lanes at a time (the
/// tail through a padded vector), the reference's algorithm with fused multiply-adds.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn activation_avx2(func: Func, values: &mut [f32]) {
    let lanes = |v: __m256| match func {
        Func::Tanh => tanh_avx2(v),
        Func::Sigmoid => sigmoid_avx2(v),
    };
    let (chunks, tail, mut pad) = split::<8>(values);
    for c in chunks {
        // SAFETY: `c` holds 8 floats; the load and store are unaligned.
        unsafe { _mm256_storeu_ps(c.as_mut_ptr(), lanes(_mm256_loadu_ps(c.as_ptr()))) };
    }
    if !tail.is_empty() {
        // SAFETY: `pad` holds 8 floats; the load and store are unaligned.
        unsafe { _mm256_storeu_ps(pad.as_mut_ptr(), lanes(_mm256_loadu_ps(pad.as_ptr()))) };
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

/// `e^y` in each lane for `y <= 0`, the reference's `exp_neg` with fused multiply-adds.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn exp_neg_avx2(y: __m256) -> __m256 {
    let y = _mm256_max_ps(y, _mm256_set1_ps(EXP_MIN));
    let n = _mm256_floor_ps(_mm256_fmadd_ps(
        y,
        _mm256_set1_ps(LOG2E),
        _mm256_set1_ps(0.5),
    ));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), y);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), r);
    let mut p = _mm256_set1_ps(EXP_P[0]);
    for &c in &EXP_P[1..] {
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
    }
    let p = _mm256_add_ps(
        _mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r),
        _mm256_set1_ps(1.0),
    );
    // As in the SSE4.2 kernel: `n` is whole and in [-150, 0].
    let n = _mm256_cvttps_epi32(n);
    let n1 = _mm256_srai_epi32::<1>(n);
    let n2 = _mm256_sub_epi32(n, n1);
    let bias = _mm256_set1_epi32(127);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias)));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias)));
    _mm256_mul_ps(_mm256_mul_ps(p, s1), s2)
}

/// `e^y − 1` in each lane for `y <= 0`, the reference's `expm1_neg` with fused
/// multiply-adds (NaN lanes give a meaningless value, which the callers replace).
#[inline]
#[target_feature(enable = "avx2,fma")]
fn expm1_neg_avx2(y: __m256) -> __m256 {
    let low = _mm256_cmp_ps::<_CMP_LT_OQ>(y, _mm256_set1_ps(EXPM1_MIN));
    let y = _mm256_max_ps(y, _mm256_set1_ps(EXPM1_MIN));
    let n = _mm256_floor_ps(_mm256_fmadd_ps(
        y,
        _mm256_set1_ps(LOG2E),
        _mm256_set1_ps(0.5),
    ));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), y);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), r);
    let mut p = _mm256_set1_ps(EXP_P[0]);
    for &c in &EXP_P[1..] {
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
    }
    let q = _mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r);
    // As in the SSE4.2 kernel: `n` is whole and in [-25, 0].
    let n = _mm256_cvttps_epi32(n);
    let s = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        n,
        _mm256_set1_epi32(127),
    )));
    let v = _mm256_fmadd_ps(q, s, _mm256_sub_ps(s, _mm256_set1_ps(1.0)));
    _mm256_blendv_ps(v, _mm256_set1_ps(-1.0), low)
}

/// `ln(1 + t)` in each lane for `t` in `[0, 1]`, the reference's `log1p_unit` with fused
/// multiply-adds in the polynomial; `1 + t` and its rounding error stay separate
/// operations, as the correction needs.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn log1p_unit_avx2(t: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let m = _mm256_add_ps(one, t);
    let c = _mm256_sub_ps(t, _mm256_sub_ps(m, one));
    let big = _mm256_cmp_ps::<_CMP_GT_OQ>(m, _mm256_set1_ps(std::f32::consts::SQRT_2));
    let f = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5)), big);
    let k = _mm256_and_ps(big, one);
    let x = _mm256_sub_ps(f, one);
    let z = _mm256_mul_ps(x, x);
    let mut p = _mm256_set1_ps(LOG_P[0]);
    for &cc in &LOG_P[1..] {
        p = _mm256_fmadd_ps(p, x, _mm256_set1_ps(cc));
    }
    let y = _mm256_mul_ps(_mm256_mul_ps(x, z), p);
    let y = _mm256_fmadd_ps(k, _mm256_set1_ps(LN2_LO), y);
    let y = _mm256_fnmadd_ps(_mm256_set1_ps(0.5), z, y);
    let r = _mm256_add_ps(x, y);
    let r = _mm256_fmadd_ps(k, _mm256_set1_ps(LN2_HI), r);
    _mm256_add_ps(r, _mm256_div_ps(c, m))
}

/// Softplus in each lane, the reference's `softplus_one` with fused multiply-adds.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn softplus_avx2(x: __m256) -> __m256 {
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(SIGN as i32));
    let l = log1p_unit_avx2(exp_neg_avx2(_mm256_or_ps(x, sign)));
    let m = _mm256_and_ps(x, _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps()));
    let r = _mm256_add_ps(m, l);
    _mm256_blendv_ps(r, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// ELU with `alpha` in each lane, the reference's `elu_one` with fused multiply-adds.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn elu_avx2(alpha: __m256, x: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let neg = _mm256_mul_ps(alpha, expm1_neg_avx2(_mm256_min_ps(x, zero)));
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(SIGN as i32));
    let tiny = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_andnot_ps(sign, x), _mm256_set1_ps(EXPM1_TINY));
    let neg = _mm256_blendv_ps(neg, _mm256_mul_ps(alpha, x), tiny);
    let r = _mm256_blendv_ps(neg, x, _mm256_cmp_ps::<_CMP_GT_OQ>(x, zero));
    _mm256_blendv_ps(r, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// Runs `lanes` over `values` eight at a time, the tail through a zero-padded vector.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn avx2_lanes(values: &mut [f32], lanes: impl Fn(__m256) -> __m256) {
    let (chunks, tail, mut pad) = split::<8>(values);
    for c in chunks {
        // SAFETY: `c` holds 8 floats; the load and store are unaligned.
        unsafe { _mm256_storeu_ps(c.as_mut_ptr(), lanes(_mm256_loadu_ps(c.as_ptr()))) };
    }
    if !tail.is_empty() {
        // SAFETY: `pad` holds 8 floats; the load and store are unaligned.
        unsafe { _mm256_storeu_ps(pad.as_mut_ptr(), lanes(_mm256_loadu_ps(pad.as_ptr()))) };
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

/// The AVX2 softplus over `values`, in place.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn softplus_avx2_slice(values: &mut [f32]) {
    avx2_lanes(values, |v| softplus_avx2(v));
}

/// The AVX2 ELU with `alpha` over `values`, in place.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn elu_avx2_slice(alpha: f32, values: &mut [f32]) {
    let a = _mm256_set1_ps(alpha);
    avx2_lanes(values, |v| elu_avx2(a, v));
}

/// `tanh` in each lane, the reference's `tanh_one` with fused multiply-adds.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn tanh_avx2(x: __m256) -> __m256 {
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(SIGN as i32));
    let one = _mm256_set1_ps(1.0);
    let a = _mm256_andnot_ps(sign, x);
    let s = _mm256_mul_ps(a, a);
    let mut q = _mm256_set1_ps(TANH_Q[0]);
    for &c in &TANH_Q[1..] {
        q = _mm256_fmadd_ps(q, s, _mm256_set1_ps(c));
    }
    let small = _mm256_fmadd_ps(a, _mm256_mul_ps(s, q), a);
    let e = exp_neg_avx2(_mm256_mul_ps(_mm256_set1_ps(-2.0), a));
    let big = _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
    let is_small = _mm256_cmp_ps::<_CMP_LT_OQ>(a, _mm256_set1_ps(TANH_SMALL));
    let r = _mm256_blendv_ps(big, small, is_small);
    let r = _mm256_or_ps(r, _mm256_and_ps(sign, x));
    _mm256_blendv_ps(r, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// The logistic sigmoid in each lane, the reference's `sigmoid_one` with fused
/// multiply-adds.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn sigmoid_avx2(x: __m256) -> __m256 {
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(SIGN as i32));
    let one = _mm256_set1_ps(1.0);
    // `-|x|`: `x` with its sign bit set.
    let e = exp_neg_avx2(_mm256_or_ps(x, sign));
    let d = _mm256_add_ps(one, e);
    let positive = _mm256_cmp_ps::<_CMP_GE_OQ>(x, _mm256_setzero_ps());
    let r = _mm256_blendv_ps(_mm256_div_ps(e, d), _mm256_div_ps(one, d), positive);
    _mm256_blendv_ps(r, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}
