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
pub use dispatch::{axpy, dot, inv_l_row, logsumexp_row, pdist_euclidean};

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
  std::arch::is_x86_feature_detected!("avx2")
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
