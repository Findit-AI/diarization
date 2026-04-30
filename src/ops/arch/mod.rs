//! Architecture-specific SIMD backends.
//!
//! Each submodule is gated on the target arch it targets. Backends
//! supply byte-identical f64 outputs to [`crate::ops::scalar`]; the
//! correctness contract is anchored by the scalar reference, and the
//! arch kernels are exercised end-to-end via the parity tests
//! (`reconstruct::rttm_parity_tests::*`) plus the parity tests under
//! `tests/`.
//!
//! Coverage in Step 3:
//! - NEON: `dot`, `axpy`, `pdist_euclidean` (f64×2 lanes, FMA).
//! - x86_avx2: same three primitives (f64×4 lanes, FMA).
//! - x86_avx512: same three primitives (f64×8 lanes, FMA).
//!
//! `inv_l_row` and `logsumexp_row` stay scalar in Step 3 — they're not
//! on the dominant hot path (per the bench analysis: AHC ≈ 53% of
//! pipeline cost is `pdist_euclidean`, VBx ≈ 32% is dominated by
//! `dot`/`axpy`-style work; `inv_l_row` and the `logsumexp` reduction
//! together are <5%). `logsumexp` would also need a vectorized `exp`
//! polynomial — punted to a later step if profiling justifies it.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx2;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx512;
