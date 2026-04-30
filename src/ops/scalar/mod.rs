//! Scalar reference implementations of the [`crate::ops`] primitives.
//!
//! These are always compiled and act as the byte-identical correctness
//! anchor. SIMD backends in [`crate::ops::arch`] must produce the same
//! `f64` outputs (modulo documented float-roundoff tolerance, which
//! for these primitives is zero — every backend uses the same f64
//! arithmetic, just packed into wider registers).

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
