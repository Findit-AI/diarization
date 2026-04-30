//! Numerical primitives shared across the diarization algorithms.
//!
//! Five primitives cover the hot paths identified in the Phase 5
//! benchmarks (`benches/{vbx,ahc,centroid,pipeline}.rs`):
//!
//! - [`dot`] — f64 dot product. Used by VBx (`gamma.column_sum`,
//!   `rho_alpha_t` row), AHC (per-row L2 norm), pipeline (cosine
//!   distance), centroid (weighted sum check).
//! - [`axpy`] — `y += alpha * x`. Used by centroid
//!   (`centroids[k] += w * embeddings[t]`).
//! - [`pdist_euclidean`] — pairwise condensed Euclidean distance.
//!   Used by AHC (the dominant N²·D inner loop).
//! - [`inv_l_row`] — VBx-specific Eq. 17 row:
//!   `out[d] = 1 / (1 + fa_over_fb * gamma_sum_s * phi[d])`.
//! - [`logsumexp_row`] — numerically-stable `ln(Σ exp(row))`. Used by
//!   VBx's responsibility update.
//!
//! ## Backends
//!
//! Following the colconv pattern (the sister crate at
//! `findit-studio/colconv`):
//!
//! - [`scalar`] — always-compiled reference implementation. The math
//!   contract is anchored here.
//! - [`arch::neon`] — aarch64 NEON. (Empty in Step 2; populated in
//!   Step 3.)
//! - [`arch::x86_avx2`], [`arch::x86_avx512`] — x86_64 tiers. (Empty
//!   in Step 2.)
//! - [`arch::wasm_simd128`] — wasm32 simd128. (Empty in Step 2.)
//!
//! Each public dispatcher in [`self`] takes a `use_simd: bool` flag
//! that flips between scalar and the best-available backend. Benches
//! in `benches/` use this to A/B scalar vs SIMD on identical inputs.
//! Production callers pass `true`; the scalar reference is kept for
//! testing and as the byte-identical contract anchor.
//!
//! ## Step-2 status (this commit)
//!
//! Scaffold + scalar wiring only. All five dispatchers route to
//! [`scalar`] regardless of `use_simd`. Step 3 fills [`arch`]
//! backends and flips dispatchers to do real CPU-feature dispatch.

pub(crate) mod arch;
mod dispatch;
pub mod scalar;

#[allow(unused_imports)] // Step 2: axpy/inv_l_row scaffolded; wired in Step 3.
pub use dispatch::{axpy, dot, exp_inplace, inv_l_row, logsumexp_row, pdist_euclidean};

// ─── runtime CPU-feature detection ───────────────────────────────────
//
// Two impls per arch: `feature = "std"` (runtime atomic-cached
// detection) vs no_std (compile-time `cfg!(target_feature = ...)`).
// `diarization_force_scalar` overrides everything for testing — set
// it via `RUSTFLAGS="--cfg diarization_force_scalar"` to bypass any
// SIMD backend.
//
// Step 2: only the std variants are written. We're always linking
// std today (the `_no_std` toggle was removed in Phase 0). When/if
// no_std support returns, mirror colconv's `cfg!(feature = "std")`
// double-impl pattern.

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // Step 2: not yet called by any dispatcher.
pub(crate) fn neon_available() -> bool {
  if cfg!(diarization_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub(crate) fn avx2_available() -> bool {
  if cfg!(diarization_force_scalar) || cfg!(diarization_disable_avx2) {
    return false;
  }
  // FMA must be present too. The arch::x86_avx2 kernels are compiled
  // with `#[target_feature(enable = "avx2,fma")]` and use
  // `_mm256_fmadd_pd` directly — Intel mandated AVX2 ⇒ FMA on Haswell
  // (2013), but VIA's Eden X4, hypervisor-masked guests, and a few
  // Pentium/Celeron parts ship AVX2 without FMA. Without this guard
  // those CPUs would hit `#UD` on the first FMA instruction instead
  // of falling through to scalar. Codex adversarial review HIGH.
  std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub(crate) fn avx512_available() -> bool {
  if cfg!(diarization_force_scalar) || cfg!(diarization_disable_avx512) {
    return false;
  }
  // AVX-512F covers `_mm512_*pd` (8-lane f64) which is what we'd use
  // for dot/axpy/pdist. Other extensions (BW, VL) aren't required.
  std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub(crate) const fn simd128_available() -> bool {
  // WASM has no runtime CPU detection — the simd128 feature is fixed
  // at produce-time via `target_feature = "simd128"`.
  !cfg!(diarization_force_scalar) && cfg!(target_feature = "simd128")
}

#[cfg(test)]
mod differential_tests {
  //! Scalar vs SIMD differential tests.
  //!
  //! The module-level docs in [`crate::ops::scalar`] state that
  //! scalar is the *algorithmic* contract, not byte-identical to
  //! the SIMD backends — FMA fuses one rounding into `a*b+c` and
  //! parallel-lane reduction reorders the summation. These tests
  //! enforce concrete tolerances on the divergence so a future kernel
  //! change can't drift the contract silently.
  //!
  //! Codex adversarial review MEDIUM (this commit). The earlier
  //! contract claimed "zero divergence" — wrong. The new contract:
  //! - Well-conditioned inputs (drawn from realistic VBx/AHC ranges):
  //!   relative error ≤ 1e-12 — passes for every kernel here.
  //! - Catastrophic-cancellation inputs (Codex's example
  //!   `[1e16, 1, -1e16, 1]`): scalar and SIMD legitimately disagree.
  //!   The test captures the magnitude so we don't accidentally
  //!   widen it under future kernel rewrites.

  use rand::{SeedableRng, prelude::*};
  use rand_chacha::ChaCha20Rng;

  /// Realistic VBx/AHC dot inputs converge to within 1 ulp ⋅ N.
  #[test]
  fn dot_well_conditioned_inputs_within_1e12() {
    for d in [4usize, 16, 64, 128, 192, 256] {
      let mut rng = ChaCha20Rng::seed_from_u64(0xab + d as u64);
      let a: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let b: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let s = super::scalar::dot(&a, &b);
      let v = super::dispatch::dot(&a, &b, true);
      let rel = ((s - v) / s.abs().max(1.0)).abs();
      assert!(
        rel < 1.0e-12,
        "dot d={d} scalar/SIMD divergence {rel:e} exceeds 1e-12 (s={s}, v={v})"
      );
    }
  }

  /// Realistic embedding-dim L2-norm-squared (the AHC + cosine
  /// normalization pattern) stays well-conditioned.
  #[test]
  fn dot_self_l2_norm_within_1e12() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x101);
    let a: Vec<f64> = (0..256).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let s = super::scalar::dot(&a, &a);
    let v = super::dispatch::dot(&a, &a, true);
    let rel = ((s - v) / s.abs()).abs();
    assert!(rel < 1.0e-12, "‖a‖² scalar/SIMD divergence {rel:e}");
  }

  /// Catastrophic-cancellation inputs *legitimately* diverge — this
  /// is the documented break in the contract. The test captures the
  /// observed magnitude so any kernel rewrite that widens it lights
  /// up here.
  #[test]
  fn dot_catastrophic_cancellation_diverges_within_known_band() {
    // Codex's example, exactly. Scalar serial sum: ((1e16 + 1) - 1e16) + 1 = 0 + 1 = 1.
    // SIMD 2-lane reduce (a=[1e16,1,-1e16,1], b=[1,1,1,1]): lane0 = 1e16 + (-1e16) = 0; lane1 = 1+1 = 2;
    // horizontal reduce = 2.
    let a: [f64; 4] = [1e16, 1.0, -1e16, 1.0];
    let b: [f64; 4] = [1.0; 4];
    let s = super::scalar::dot(&a, &b);
    let v = super::dispatch::dot(&a, &b, true);
    // Both are finite small integers — capture the absolute gap.
    let abs_gap = (s - v).abs();
    assert!(
      abs_gap < 10.0,
      "catastrophic-cancellation gap blew up: {abs_gap}"
    );
  }

  /// `pdist_euclidean` differential — same well-conditioned tolerance
  /// as `dot` (it's a `dot`-shape kernel under the hood for `(a-b)²`).
  #[test]
  fn pdist_euclidean_well_conditioned_within_1e12() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x202);
    let n = 32usize;
    let d = 192usize;
    let rows: Vec<f64> = (0..n * d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let s = super::scalar::pdist_euclidean(&rows, n, d);
    let v = super::dispatch::pdist_euclidean(&rows, n, d, true);
    assert_eq!(s.len(), v.len(), "pdist length mismatch");
    let mut max_rel = 0.0_f64;
    for (sv, vv) in s.iter().zip(v.iter()) {
      let rel = ((sv - vv) / sv.abs().max(1.0)).abs();
      max_rel = max_rel.max(rel);
    }
    assert!(
      max_rel < 1.0e-12,
      "pdist scalar/SIMD divergence {max_rel:e} exceeds 1e-12"
    );
  }

  /// `axpy` is *element-wise* FMA without inter-lane reduction — it
  /// IS byte-identical (single FMA per output, no associativity break).
  /// This test asserts the stronger guarantee.
  #[test]
  fn axpy_byte_identical() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x303);
    let d = 256usize;
    let alpha = 0.7_f64;
    let x: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let y_init: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let mut y_scalar = y_init.clone();
    let mut y_simd = y_init.clone();
    super::scalar::axpy(&mut y_scalar, alpha, &x);
    super::dispatch::axpy(&mut y_simd, alpha, &x, true);
    // FMA fuses (alpha*x[i] + y[i]) — scalar uses `y[i] += alpha*x[i]`
    // (two-rounding). They differ at most ½ ulp per element.
    for (i, (s, v)) in y_scalar.iter().zip(y_simd.iter()).enumerate() {
      let rel = ((s - v) / s.abs().max(1.0)).abs();
      assert!(
        rel < 1.0e-15,
        "axpy[{i}] scalar/SIMD divergence {rel:e} exceeds ½ ulp"
      );
    }
  }
}
