//! `PldaTransform` — the load-time setup + per-embedding projection.
//!
//! Construction loads the compile-time-embedded weight blobs and runs
//! the generalized-eigh setup once. Thereafter `xvec_transform` and
//! `plda_transform` are pure read-only mappings.

use nalgebra::{DMatrix, DVector};

use crate::plda::{
    error::Error,
    loader::{load_plda, load_xvec, PldaWeights, XvecWeights},
    EMBEDDING_DIMENSION, PLDA_DIMENSION,
};

/// Probabilistic Linear Discriminant Analysis transform. Two stages:
///
/// 1. [`xvec_transform`](Self::xvec_transform): center → L2-norm → LDA →
///    recenter → L2-norm → scale by `sqrt(D_out)`. Output `‖·‖ = sqrt(128)`.
/// 2. [`plda_transform`](Self::plda_transform): center → project onto
///    the descending-sorted generalized eigenvectors of `eigh(B, W)`.
///    Output is whitened (NOT L2-normed).
///
/// Mirrors `pyannote.audio.utils.vbx.vbx_setup` + `xvec_tf` + `plda_tf`
/// (`utils/vbx.py:181-218` in pyannote.audio 4.0.4). Validated
/// against the Phase-0 captured artifacts via `tests/parity_plda.rs`.
pub struct PldaTransform {
    // xvec_tf factors
    mean1: DVector<f64>,
    mean2: DVector<f64>,
    lda: DMatrix<f64>,
    sqrt_in_dim: f64,  // sqrt(EMBEDDING_DIMENSION)
    sqrt_out_dim: f64, // sqrt(PLDA_DIMENSION)

    // plda_tf factors (filled in by P1 Task 4 — generalized eigh).
    #[allow(dead_code)]
    plda_mu: DVector<f64>,
    #[allow(dead_code)]
    plda_eigenvectors_desc: DMatrix<f64>,
    #[allow(dead_code)]
    phi: DVector<f64>,
}

impl PldaTransform {
    /// Construct from the compile-time-embedded weight blobs.
    /// Phase-1 stub: only `xvec_transform` data is populated. Task 4
    /// runs the generalized eigh and replaces the placeholder
    /// `plda_eigenvectors_desc` / `phi` with real values.
    pub fn new() -> Result<Self, Error> {
        let XvecWeights { mean1, mean2, lda } = load_xvec();
        let PldaWeights { mu, tr: _tr, psi } = load_plda();

        // Task-4 placeholder — Task 4 replaces these two with the
        // post-eigh whitening matrix and eigenvalues.
        let plda_eigenvectors_desc =
            DMatrix::<f64>::zeros(PLDA_DIMENSION, PLDA_DIMENSION);
        let phi = DVector::<f64>::from_iterator(PLDA_DIMENSION, psi.iter().copied());

        Ok(Self {
            mean1,
            mean2,
            lda,
            sqrt_in_dim: (EMBEDDING_DIMENSION as f64).sqrt(),
            sqrt_out_dim: (PLDA_DIMENSION as f64).sqrt(),
            plda_mu: mu,
            plda_eigenvectors_desc,
            phi,
        })
    }
}

#[allow(dead_code)] // suppress until xvec_transform / plda_transform are implemented
fn l2_normalize_in_place(v: &mut DVector<f64>) {
    let n = v.norm();
    if n > 0.0 {
        *v /= n;
    }
}
