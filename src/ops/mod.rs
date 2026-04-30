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
  //! Contract (Codex adversarial review, multiple rounds):
  //! - On `aarch64` (the deployment target), scalar and the NEON
  //!   backend produce **bit-identical** results for all five
  //!   primitives. Achieved by:
  //!   1. scalar uses `f64::mul_add` for per-element FMA (one IEEE
  //!      754 rounding, identical to `vfmaq_f64`);
  //!   2. scalar's reduction tree mirrors NEON's (4 partial sums
  //!      over modulo-4 indices, then `((s00+s10) + (s01+s11))`).
  //! - On `x86_64`, AVX2 (4-lane) and AVX-512 (8-lane) use their
  //!   native lane widths — different reduction trees from NEON.
  //!   Per-element FMA is still bit-identical, but the lane-width
  //!   reduction may diverge from scalar by O(1e-15) relative on
  //!   well-conditioned inputs. Cross-architecture bit-identity is
  //!   not claimed.
  //! - On both architectures, catastrophic-cancellation inputs
  //!   (`[1e16, 1, -1e16, 1]`) legitimately diverge between scalar
  //!   and SIMD due to the documented reduction-order difference.

  use rand::{SeedableRng, prelude::*};
  use rand_chacha::ChaCha20Rng;

  /// On aarch64 scalar matches NEON bit-for-bit; elsewhere the
  /// well-conditioned inputs hold a tighter bound than the previous
  /// 1e-12 contract.
  #[test]
  fn dot_well_conditioned_inputs_match() {
    for d in [4usize, 16, 64, 128, 192, 256] {
      let mut rng = ChaCha20Rng::seed_from_u64(0xab + d as u64);
      let a: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let b: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let s = super::scalar::dot(&a, &b);
      let v = super::dispatch::dot(&a, &b, true);
      #[cfg(target_arch = "aarch64")]
      assert_eq!(
        s.to_bits(),
        v.to_bits(),
        "dot d={d} scalar/NEON not bit-identical (s={s}, v={v})"
      );
      #[cfg(not(target_arch = "aarch64"))]
      {
        let rel = ((s - v) / s.abs().max(1.0)).abs();
        assert!(
          rel < 1.0e-14,
          "dot d={d} scalar/SIMD divergence {rel:e} exceeds 1e-14 (s={s}, v={v})"
        );
      }
    }
  }

  /// Realistic embedding-dim L2-norm-squared (the AHC + cosine
  /// normalization pattern).
  #[test]
  fn dot_self_l2_norm_match() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x101);
    let a: Vec<f64> = (0..256).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let s = super::scalar::dot(&a, &a);
    let v = super::dispatch::dot(&a, &a, true);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(s.to_bits(), v.to_bits(), "‖a‖² scalar/NEON not bit-identical");
    #[cfg(not(target_arch = "aarch64"))]
    {
      let rel = ((s - v) / s.abs()).abs();
      assert!(rel < 1.0e-14, "‖a‖² scalar/SIMD divergence {rel:e}");
    }
  }

  /// Catastrophic-cancellation inputs *do* diverge across reduction
  /// orders. Scalar uses 4-acc pair reduction; AVX2 uses 4-lane;
  /// AVX-512 uses 8-lane. Test captures the magnitude so any future
  /// kernel rewrite that widens it surfaces here.
  #[test]
  fn dot_catastrophic_cancellation_within_known_band() {
    let a: [f64; 4] = [1e16, 1.0, -1e16, 1.0];
    let b: [f64; 4] = [1.0; 4];
    let s = super::scalar::dot(&a, &b);
    let v = super::dispatch::dot(&a, &b, true);
    let abs_gap = (s - v).abs();
    assert!(
      abs_gap < 10.0,
      "catastrophic-cancellation gap blew up: {abs_gap}"
    );
  }

  /// `pdist_euclidean` differential.
  #[test]
  fn pdist_euclidean_well_conditioned_match() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x202);
    let n = 32usize;
    let d = 192usize;
    let rows: Vec<f64> = (0..n * d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let s = super::scalar::pdist_euclidean(&rows, n, d);
    let v = super::dispatch::pdist_euclidean(&rows, n, d, true);
    assert_eq!(s.len(), v.len(), "pdist length mismatch");
    for (idx, (sv, vv)) in s.iter().zip(v.iter()).enumerate() {
      #[cfg(target_arch = "aarch64")]
      assert_eq!(
        sv.to_bits(),
        vv.to_bits(),
        "pdist[{idx}] scalar/NEON not bit-identical (s={sv}, v={vv})"
      );
      #[cfg(not(target_arch = "aarch64"))]
      {
        let rel = ((sv - vv) / sv.abs().max(1.0)).abs();
        assert!(rel < 1.0e-14, "pdist[{idx}] divergence {rel:e}");
        let _ = idx;
      }
    }
  }

  /// Mismatched `dot` lengths must `panic!` (not UB) even with
  /// `use_simd = true`. The dispatcher enforces `a.len() == b.len()`
  /// unconditionally before routing to the unsafe SIMD kernel — this
  /// test would silently OOB-read `b` if that guard were debug-only.
  /// Codex adversarial review HIGH.
  #[test]
  #[should_panic(expected = "ops::dot")]
  fn dot_dispatch_panics_on_length_mismatch_under_simd() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::dispatch::dot(&a, &b, true);
  }

  /// Same panic boundary on the scalar path — the precondition is
  /// asserted *before* the SIMD branch, so both routes reach the
  /// same panic.
  #[test]
  #[should_panic(expected = "ops::dot")]
  fn dot_dispatch_panics_on_length_mismatch_under_scalar() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::dispatch::dot(&a, &b, false);
  }

  /// Mismatched `axpy` lengths must `panic!` not UB.
  #[test]
  #[should_panic(expected = "ops::axpy")]
  fn axpy_dispatch_panics_on_length_mismatch_under_simd() {
    let mut y = vec![0.0_f64; 8];
    let x = vec![1.0_f64; 4];
    super::dispatch::axpy(&mut y, 0.5, &x, true);
  }

  /// `pdist_euclidean` rejects shape mismatch with a panic.
  #[test]
  #[should_panic(expected = "ops::pdist_euclidean")]
  fn pdist_dispatch_panics_on_shape_mismatch_under_simd() {
    let rows = vec![1.0_f64; 100]; // 5 * 20 worth of data
    // claim 10 rows × 20 cols (200 entries) — doesn't match 100.
    let _ = super::dispatch::pdist_euclidean(&rows, 10, 20, true);
  }

  /// `pdist_euclidean` rejects `n * d` overflow before hitting the
  /// unsafe path.
  #[test]
  #[should_panic(expected = "ops::pdist_euclidean")]
  fn pdist_dispatch_panics_on_dim_overflow() {
    let rows: Vec<f64> = vec![];
    let _ = super::dispatch::pdist_euclidean(&rows, usize::MAX, 2, true);
  }

  /// `axpy` is per-element FMA with no reduction. With scalar using
  /// `f64::mul_add` it must match SIMD's `vfmaq_f64` /
  /// `_mm256_fmadd_pd` / `_mm512_fmadd_pd` bit-for-bit on every
  /// architecture.
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
    for (i, (s, v)) in y_scalar.iter().zip(y_simd.iter()).enumerate() {
      assert_eq!(
        s.to_bits(),
        v.to_bits(),
        "axpy[{i}] scalar/SIMD not bit-identical (s={s}, v={v})"
      );
    }
  }
}
