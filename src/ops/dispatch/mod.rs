//! Public dispatchers for [`crate::ops`] primitives.
//!
//! Each dispatcher takes a `use_simd: bool` flag. In Step 2 the
//! dispatchers route to [`crate::ops::scalar`] regardless. Step 3
//! adds `cfg_select!` arms that call into [`crate::ops::arch`]
//! backends after the matching `*_available()` check.
//!
//! Why the explicit flag instead of always-best-available: lets
//! benches measure scalar-vs-SIMD on identical inputs (criterion
//! prints adjacent rows for direct delta reading), and lets tests
//! force the scalar path under `diarization_force_scalar` without
//! recompiling.

mod axpy;
mod dot;
mod inv_l_row;
mod lse;
mod pdist_euclidean;

pub use axpy::axpy;
pub use dot::dot;
pub use inv_l_row::inv_l_row;
pub use lse::logsumexp_row;
pub use pdist_euclidean::pdist_euclidean;
