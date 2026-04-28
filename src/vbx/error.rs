//! Error variants for `dia::vbx`.

use thiserror::Error;

/// Errors produced by `vbx_iterate`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
  /// Input shapes do not satisfy the contract.
  #[error("shape mismatch: {0}")]
  Shape(&'static str),

  /// A non-finite value (NaN / ±inf) appeared in an intermediate
  /// (rho, alpha, log_p_, ELBO, …). The algorithm has no recovery
  /// path; the caller should treat this as a hard failure.
  #[error("non-finite intermediate: {0}")]
  NonFinite(&'static str),

  /// `Phi` (the eigenvalue diagonal from `PldaTransform::phi()`) had
  /// a non-positive entry. The algorithm requires `Phi[d] > 0` for
  /// `sqrt(Phi)` and `1 + … * Phi` to be well-defined.
  #[error("Phi must be strictly positive; saw {0:.3e} at index {1}")]
  NonPositivePhi(f64, usize),
}
