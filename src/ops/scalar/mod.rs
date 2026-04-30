//! Scalar reference implementations of the [`crate::ops`] primitives.
//!
//! Always compiled. The scalar path is the *algorithmic* contract —
//! same math, same input-validation behaviour — but it is **not**
//! byte-identical to the SIMD backends in [`crate::ops::arch`]:
//!
//! - **FMA fuses** `a * b + c` into one instruction with a single IEEE
//!   rounding step on `aarch64::vfmaq_f64`, `_mm256_fmadd_pd`, and
//!   `_mm512_fmadd_pd`. The scalar reference uses `acc += a * b` —
//!   two roundings (mul, then add). For exact-product inputs the two
//!   agree; otherwise FMA is closer to the infinite-precision result
//!   by ½ ulp.
//! - **Parallel-lane reduction** — the SIMD `dot` and `pdist`
//!   accumulate into 2 / 4 / 8 lanes (NEON / AVX2 / AVX-512) and
//!   horizontally reduce at the end, vs the scalar serial sum. Float
//!   addition is non-associative, so for inputs with catastrophic
//!   cancellation (e.g., `[1e16, 1, -1e16, 1]`) the two summation
//!   orders give different results.
//!
//! In practice, for diarization's well-conditioned inputs (PLDA
//! features in O(1), embeddings on the unit sphere, post-softmax
//! gamma in [0, 1]) the divergence stays under ~1e-12 relative —
//! see `crate::ops::tests` for the differential bound. AHC / VBx /
//! pipeline parity tests pass with SIMD on by default. Callers that
//! need *byte-identical* scalar output (regression diffs against a
//! reference implementation, golden files captured against the
//! scalar path) must pass `use_simd = false` at the dispatcher.
//!
//! Codex adversarial review MEDIUM (this commit).

mod axpy;
mod dot;
mod exp;
mod inv_l_row;
mod lse;
mod pdist_euclidean;

pub use axpy::axpy;
pub use dot::dot;
pub use exp::exp_inplace;
pub use inv_l_row::inv_l_row;
pub use lse::logsumexp_row;
pub use pdist_euclidean::pdist_euclidean;
