//! `PldaTransform` — the load-time setup + per-embedding projection.
//!
//! Construction loads the compile-time-embedded weight blobs and runs
//! the generalized-eigh setup once. Thereafter `xvec_transform` and
//! `plda_transform` are pure read-only mappings.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

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
    ///
    /// Runs the generalized symmetric eigenvalue solve `eigh(B, W)`
    /// once at construction time:
    ///
    /// ```text
    /// W = inv(tr.T @ tr)              # within-class precision
    /// B = inv((tr.T / psi) @ tr)      # between-class precision
    /// (eigenvalues, eigenvectors) = generalized_eigh(B, W)  # ascending
    /// → reverse to descending → store
    /// ```
    ///
    /// Mirrors `pyannote/audio/utils/vbx.py:201-208`.
    pub fn new() -> Result<Self, Error> {
        let XvecWeights { mean1, mean2, lda } = load_xvec();
        let PldaWeights { mu, tr, psi } = load_plda();

        // ── Build B and W matrices (clustering.py:201-203). ─────────
        //
        // `(tr.T / psi)` is a numpy broadcast. For shape `(M, M) /
        // (M,)`, numpy aligns dimensions from the right, so the (M,)
        // vector becomes a row applied per-column:
        //   `(tr.T / psi)[i, j] == tr.T[i, j] / psi[j]`
        // (NOT `/ psi[i]` — that would be per-row scaling). Verified
        // by hand against numpy's actual values.
        let tr_t = tr.transpose();
        let mut tr_t_scaled = tr_t.clone();
        for j in 0..PLDA_DIMENSION {
            let scale = 1.0 / psi[j];
            for i in 0..PLDA_DIMENSION {
                tr_t_scaled[(i, j)] *= scale;
            }
        }
        let w = (&tr_t * &tr)
            .try_inverse()
            .ok_or(Error::WNotPositiveDefinite)?;
        let b = (&tr_t_scaled * &tr)
            .try_inverse()
            .ok_or(Error::WNotPositiveDefinite)?;

        // ── Generalized eigh, sorted descending. ────────────────────
        let (eigenvalues_desc, eigenvectors_desc) =
            generalized_eigh_descending(&b, &w)?;

        // pyannote's `plda_tf` is `(x - mu) @ plda_tr.T` where
        // `plda_tr = wccn.T[::-1]`. Substituting:
        //   plda_tr.T = (wccn.T[::-1]).T = wccn[:, ::-1]
        // i.e. eigenvectors-as-columns in descending order. That's
        // exactly what `eigenvectors_desc` is. Storing it directly.
        let plda_eigenvectors_desc = eigenvectors_desc;
        let phi = eigenvalues_desc;

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

    /// First PLDA stage. Mirrors `xvec_tf` in
    /// `pyannote/audio/utils/vbx.py:211-213`:
    ///
    /// ```text
    /// xvec_tf(x) = sqrt(D_out) *
    ///     l2_norm( lda.T @ (sqrt(D_in) * l2_norm(x - mean1)) - mean2 )
    /// ```
    ///
    /// Output norm is `sqrt(PLDA_DIMENSION)` — i.e. `sqrt(128) ≈ 11.31`,
    /// **not** 1.0. The outer scale-by-`sqrt(D_out)` is load-bearing
    /// for the downstream PLDA whitening; downstream consumers MUST
    /// not re-normalize this output.
    pub fn xvec_transform(
        &self,
        input: &[f32; EMBEDDING_DIMENSION],
    ) -> [f64; PLDA_DIMENSION] {
        // 1. Promote f32 input to f64 and center: x = input - mean1.
        let mut x = DVector::<f64>::from_iterator(
            EMBEDDING_DIMENSION,
            input.iter().map(|v| *v as f64),
        );
        x -= &self.mean1;

        // 2. L2-normalize, then scale by sqrt(D_in).
        l2_normalize_in_place(&mut x);
        x *= self.sqrt_in_dim;

        // 3. lda.T @ x  →  (PLDA_DIMENSION,)-shaped vector.
        //    nalgebra's `tr_mul` is matmul-with-transposed-lhs; avoids
        //    an explicit transpose copy.
        let mut y = self.lda.tr_mul(&x);

        // 4. Recenter: y -= mean2.
        y -= &self.mean2;

        // 5. L2-normalize, then scale by sqrt(D_out).
        l2_normalize_in_place(&mut y);
        y *= self.sqrt_out_dim;

        let mut out = [0.0f64; PLDA_DIMENSION];
        for (slot, value) in out.iter_mut().zip(y.iter()) {
            *slot = *value;
        }
        out
    }

    /// Second PLDA stage. Mirrors `plda_tf` in
    /// `pyannote/audio/utils/vbx.py:215-217`:
    ///
    /// ```text
    /// plda_tf(x0) = (x0 - plda_mu) @ plda_tr.T
    /// ```
    ///
    /// where `plda_tr = wccn.T[::-1]` (eigenvectors of the generalized
    /// problem as ROWS, in descending eigenvalue order). So
    /// `plda_tr.T = wccn[:, ::-1]` — eigenvectors as columns, descending.
    /// We store that directly in `plda_eigenvectors_desc` and matmul.
    ///
    /// Output is whitened (NOT L2-normed). The Rust port uses
    /// `eigenvectors.tr_mul(centered_x)` to express the row-vector
    /// matmul in column-vector form — the resulting ordering matches
    /// pyannote's row-major numpy result.
    pub fn plda_transform(
        &self,
        post_xvec: &[f64; PLDA_DIMENSION],
    ) -> [f64; PLDA_DIMENSION] {
        // 1. Center: x = post_xvec - plda_mu.
        let mut x = DVector::<f64>::from_iterator(
            PLDA_DIMENSION,
            post_xvec.iter().copied(),
        );
        x -= &self.plda_mu;

        // 2. Project onto descending eigenvectors. pyannote does
        // `(x - mu) @ eigenvectors_desc` (row vector × matrix). In
        // column-vector terms that's `eigenvectors_desc.T @ (x - mu)`.
        // `tr_mul(&x)` computes `self.transpose() * x` without an
        // explicit transpose copy.
        let y = self.plda_eigenvectors_desc.tr_mul(&x);

        let mut out = [0.0f64; PLDA_DIMENSION];
        for (slot, value) in out.iter_mut().zip(y.iter()) {
            *slot = *value;
        }
        out
    }

    /// Convenience: chain `xvec_transform` → `plda_transform`.
    pub fn project(&self, input: &[f32; EMBEDDING_DIMENSION]) -> [f64; PLDA_DIMENSION] {
        let post_xvec = self.xvec_transform(input);
        self.plda_transform(&post_xvec)
    }

    /// Eigenvalue diagonal `phi` (descending) — `pyannote.audio.core.plda.PLDA.phi`.
    /// Consumed by VBx in Phase 2 as the across-class covariance diagonal.
    pub fn phi(&self) -> &[f64] {
        self.phi.as_slice()
    }
}

/// In-place L2 normalization. No-op for zero-norm input — pyannote's
/// `l2_norm` would produce NaN there, but real WeSpeaker outputs are
/// never exactly zero, so divergence on this edge case is intentional
/// (it prevents NaN propagation into the parity tests if the
/// embedding model ever degenerates).
fn l2_normalize_in_place(v: &mut DVector<f64>) {
    let n = v.norm();
    if n > 0.0 {
        *v /= n;
    }
}

/// Solve the generalized symmetric eigenvalue problem
/// `B v = λ W v`, returning eigenvalues and eigenvectors sorted
/// **descending** by `λ`. Matches scipy's `eigh(B, W)` followed by
/// `[::-1]` reversal, which is what pyannote does in `vbx_setup`
/// (`utils/vbx.py:206-208`).
///
/// `W` must be symmetric positive-definite (the algorithm
/// Cholesky-decomposes it).
///
/// # Algorithm
///
/// 1. Cholesky `W = L L^T`.
/// 2. Substitution `B' = L^{-1} B L^{-T}` (computed via two
///    triangular solves; never form `L^{-1}` explicitly).
/// 3. `B' = Y Λ Y^T` (ordinary symmetric eigh).
/// 4. Recover `V = L^{-T} Y` (one upper-triangular solve).
/// 5. Sort columns of `V` and entries of `Λ` by descending eigenvalue.
fn generalized_eigh_descending(
    b: &DMatrix<f64>,
    w: &DMatrix<f64>,
) -> Result<(DVector<f64>, DMatrix<f64>), Error> {
    let n = b.nrows();
    debug_assert_eq!(b.ncols(), n);
    debug_assert_eq!(w.shape(), (n, n));

    // Step 1: Cholesky. nalgebra's `cholesky()` returns Option<Cholesky>
    // and is `None` if the input isn't positive-definite.
    let chol = w
        .clone()
        .cholesky()
        .ok_or(Error::WNotPositiveDefinite)?;
    let l = chol.l(); // lower triangular

    // Step 2: Compute B' = L^{-1} B L^{-T}.
    // First Y = L^{-1} B  via solve_lower_triangular(L, B).
    let y = l
        .solve_lower_triangular(b)
        .expect("L is square + nonsingular by construction");
    // Then B' = Y * L^{-T}. Take the transpose:
    //   (Y * L^{-T})^T = L^{-1} * Y^T
    // so we solve L * Z = Y^T → Z = L^{-1} Y^T = (Y L^{-T})^T → B' = Z^T.
    let bp_t = l
        .solve_lower_triangular(&y.transpose())
        .expect("L is nonsingular");
    let bp = bp_t.transpose();

    // Step 3: ordinary symmetric eigh on B'.
    let SymmetricEigen {
        eigenvalues,
        eigenvectors,
    } = SymmetricEigen::new(bp);

    // Step 4: recover V = L^{-T} Y_eig. L^T is upper-triangular; solve
    // L^T V = Y_eig for V (each column independently).
    let l_t = l.transpose();
    let v = l_t
        .solve_upper_triangular(&eigenvectors)
        .expect("L^T is nonsingular");

    // Step 5: sort by eigenvalue descending. Build a permutation of
    // column indices, then materialise sorted matrix + vector.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_vals = DVector::<f64>::zeros(n);
    let mut sorted_vecs = DMatrix::<f64>::zeros(n, n);
    for (out_col, &src_col) in idx.iter().enumerate() {
        sorted_vals[out_col] = eigenvalues[src_col];
        sorted_vecs.set_column(out_col, &v.column(src_col));
    }

    Ok((sorted_vals, sorted_vecs))
}
