//! Public dispatchers for [`crate::ops`] primitives.
//!
//! Each dispatcher always selects the best-available SIMD backend
//! at runtime via `cfg_select!` arms guarded by `*_available()`
//! checks against [`crate::ops::arch`]. Callers needing scalar
//! output explicitly call [`crate::ops::scalar`]. Codex review
//! round 8.

mod axpy;
mod dot;
mod exp;
mod inv_l_row;
mod lse;
mod pdist_euclidean;

pub use axpy::{axpy, axpy_f32};
pub use dot::dot;
pub use exp::exp_inplace;
pub use inv_l_row::inv_l_row;
pub use lse::logsumexp_row;
pub use pdist_euclidean::pdist_euclidean;
