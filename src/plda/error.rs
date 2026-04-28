//! Error type for `dia::plda`.

use thiserror::Error;

/// Errors produced by `PldaTransform` construction.
///
/// The PLDA weights are embedded into the dia binary at compile time
/// via `include_bytes!`, so I/O / file-not-found / shape-mismatch
/// errors are eliminated. The only runtime construction failure is
/// the linear-algebra precondition: `W = inv(tr.T @ tr)` must be
/// positive-definite for the Cholesky reduction in
/// [`PldaTransform`](crate::plda::PldaTransform). If the embedded
/// weights are corrupted or the upstream `pyannote` PLDA matrices
/// drift such that `W` loses rank, this fires.
#[derive(Debug, Error)]
pub enum Error {
    /// The within-class covariance matrix `W = inv(tr.T @ tr)` is not
    /// symmetric positive-definite. Either the embedded `tr.bin` is
    /// corrupted, or pyannote's PLDA weights have changed in a way
    /// that breaks the generalized-eigh preconditions.
    #[error("PLDA: W matrix not positive-definite (corrupted weights or upstream drift)")]
    WNotPositiveDefinite,
}
