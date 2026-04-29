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

  /// The ELBO decreased by more than the float-roundoff tolerance
  /// between two consecutive iterations. VB EM's monotonicity is a
  /// fundamental invariant — a regression beyond float noise
  /// indicates a bug, numerical instability, or an out-of-distribution
  /// input that should not be silently accepted. The returned `gamma`
  /// and `pi` from the failing iteration are NOT propagated; if the
  /// caller wants the last-known-good state, re-invoke with
  /// `max_iters` set to `iter` (the regression-triggering iteration
  /// index). Pyannote prints a `WARNING:` to stdout and keeps the
  /// regressed state; this is a deliberate fail-fast divergence
  /// (Codex review MEDIUM round 2 of Phase 2).
  #[error("ELBO regressed by {delta:.3e} at iteration {iter} (beyond float-roundoff tolerance)")]
  ElboRegression { iter: usize, delta: f64 },
}
