//! `PldaTransform` — the load-time setup + per-embedding projection.
//!
//! Construction loads the compile-time-embedded weight blobs and runs
//! the generalized-eigh setup once. Thereafter `xvec_transform` and
//! `plda_transform` are pure read-only mappings.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::{
  embed::NORM_EPSILON,
  plda::{
    EMBEDDING_DIMENSION, PLDA_DIMENSION,
    error::Error,
    loader::{PldaWeights, XvecWeights, load_plda, load_xvec},
  },
};

/// Raw, **unnormalized** WeSpeaker output destined for the PLDA
/// transform. Wrapping the `[f32; 256]` in a distinct type prevents
/// the most likely API misuse: feeding
/// [`dia::embed::Embedding::as_array`](crate::embed::Embedding::as_array),
/// which is L2-normalized.
///
/// Pyannote's `xvec_tf` operates on **raw** WeSpeaker outputs
/// (`pyannote/audio/pipelines/clustering.py:608` —
/// `fea = self.plda(train_embeddings)`, where `train_embeddings` is
/// the un-normalized output of `get_embeddings`; the
/// `train_embeddings_normed` copy is only used for AHC linkage). If a
/// caller feeds an L2-normalized vector here instead, the centering
/// `x - mean1` produces a different intermediate, the LDA projection
/// maps to the wrong subspace, and downstream VBx clustering silently
/// drifts off the captured pyannote distribution. See
/// `normalized_vs_raw_input_produce_materially_different_output` in
/// `src/plda/tests.rs`.
///
/// # Codex review HIGH
///
/// Phase 1 originally exposed `xvec_transform(&[f32; 256])` directly.
/// That allowed `plda.project(embedding.as_array())` to compile and
/// silently produce wrong-distribution PLDA features. This wrapper
/// makes the contract type-safe — `xvec_transform`'s signature now
/// requires `&RawEmbedding`, which can only be constructed from a
/// raw array via [`Self::from_raw_array`].
#[derive(Debug, Clone)]
pub struct RawEmbedding([f32; EMBEDDING_DIMENSION]);

impl RawEmbedding {
  /// Wrap a raw, **unnormalized** WeSpeaker embedding vector.
  ///
  /// Validates the array is finite (rejects NaN / `±inf`).
  ///
  /// **This constructor is gated behind the `plda-fixtures` feature**
  /// (also available under `cfg(test)`). Production builds without
  /// the feature cannot reach this function at all — the only entry
  /// for "I have a raw WeSpeaker vector, project it" is via
  /// `dia::embed::EmbedModel::embed_raw` once Phase 5 lands. The
  /// gate exists because the finite check alone is *necessary but
  /// not sufficient* — a deliberate caller could still wrap a
  /// normalized `Embedding::as_array()` and reach
  /// `xvec_transform`. Codex review HIGH.
  ///
  /// **Do NOT pass `dia::embed::Embedding::as_array()` here.** That
  /// vector has been L2-normalized — wrong distribution for PLDA —
  /// and the type system rejects the direct path (the doctest below
  /// is a `compile_fail` example proving it).
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] if any element is NaN, `+inf`, or
  ///   `-inf`.
  ///
  /// # Examples
  ///
  /// Construction from a raw array (requires `plda-fixtures`):
  ///
  /// ```
  /// # #[cfg(feature = "plda-fixtures")] {
  /// use dia::plda::RawEmbedding;
  /// let raw: [f32; 256] = [0.5; 256];
  /// let _ = RawEmbedding::from_raw_array(raw).expect("finite");
  /// # }
  /// ```
  ///
  /// `dia::embed::Embedding::as_array()` is the wrong-distribution
  /// type and the compiler rejects it:
  ///
  /// ```compile_fail
  /// use dia::embed::Embedding;
  /// use dia::plda::PldaTransform;
  /// let plda = PldaTransform::new().unwrap();
  /// let emb = Embedding::normalize_from([0.5; 256]).unwrap();
  /// // The type system rejects this — `xvec_transform` requires
  /// // `&RawEmbedding`, not `&[f32; 256]`.
  /// let _ = plda.xvec_transform(emb.as_array());
  /// ```
  #[cfg(any(feature = "plda-fixtures", test))]
  #[cfg_attr(docsrs, doc(cfg(feature = "plda-fixtures")))]
  pub fn from_raw_array(arr: [f32; EMBEDDING_DIMENSION]) -> Result<Self, Error> {
    if !arr.iter().all(|v| v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    Ok(Self(arr))
  }

  /// Borrow the underlying raw vector. Provided for parity tests and
  /// internal use; production callers should not need this. Also
  /// gated behind `plda-fixtures` so external consumers cannot
  /// extract a raw vector and re-wrap it as a different boundary
  /// type.
  #[cfg(any(feature = "plda-fixtures", test))]
  #[cfg_attr(docsrs, doc(cfg(feature = "plda-fixtures")))]
  pub fn as_array(&self) -> &[f32; EMBEDDING_DIMENSION] {
    &self.0
  }
}

/// Output of [`PldaTransform::xvec_transform`] / input to
/// [`PldaTransform::plda_transform`]. A 128-d f64 vector with norm
/// `sqrt(PLDA_DIMENSION) ≈ 11.31` — the intermediate distribution
/// that `plda_tf` is mathematically defined for.
///
/// Wrapping the `[f64; 128]` in a distinct type prevents the
/// stage-2 analogue of the `RawEmbedding` misuse: feeding
/// `plda_transform` a vector that wasn't produced by `xvec_transform`
/// (e.g. an L2-normalized 128-d vector with norm 1.0, a stale
/// pyannote capture from a different revision, or hand-constructed
/// input). Without this gate, `plda_transform` would whiten any
/// finite input and return `Ok` — VBx then clusters wrong-distribution
/// features without any error signal.
///
/// The legitimate ways to obtain a `PostXvecEmbedding` are:
/// - Call [`PldaTransform::xvec_transform`] (the runtime path).
/// - Call [`Self::from_pyannote_capture`] with a captured
///   `post_xvec` value (parity tests / offline tooling); the
///   constructor validates the norm matches `sqrt(D_out)`.
///
/// # Codex review HIGH
///
/// Phase 1 originally exposed `plda_transform(&[f64; 128])` directly.
/// That allowed callers to feed wrong-distribution data and silently
/// produce garbage whitened features. This wrapper makes the contract
/// type-safe.
#[derive(Debug, Clone)]
pub struct PostXvecEmbedding([f64; PLDA_DIMENSION]);

impl PostXvecEmbedding {
  /// Internal constructor for `xvec_transform`. Skips norm validation
  /// because the algorithm guarantees the invariant by construction.
  pub(super) fn from_xvec_output(arr: [f64; PLDA_DIMENSION]) -> Self {
    Self(arr)
  }

  /// External constructor for parity tests and offline tooling that
  /// load a `post_xvec` value from a captured pyannote run. Validates
  /// finite + norm within `1e-3` of `sqrt(PLDA_DIMENSION)`.
  ///
  /// **Gated behind the `plda-fixtures` feature** (also available
  /// under `cfg(test)`). The norm check alone is necessary but not
  /// sufficient — a synthetic 128-d vector scaled to `sqrt(128)`
  /// would still pass it. Production callers must reach
  /// `plda_transform` only via the value returned by
  /// [`PldaTransform::xvec_transform`], whose internal
  /// `from_xvec_output` constructor is the only un-gated entry.
  /// Codex review HIGH.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] on any NaN/`±inf` element.
  /// - [`Error::WrongPostXvecNorm`] if the norm is outside the
  ///   expected `sqrt(D_out) ± 1e-3` band — the input is not a
  ///   post-`xvec_tf` vector.
  ///
  /// # Examples
  ///
  /// Captured pyannote post_xvec (norm ≈ sqrt(128)) is accepted
  /// (requires `plda-fixtures`):
  ///
  /// ```
  /// # #[cfg(feature = "plda-fixtures")] {
  /// use dia::plda::PostXvecEmbedding;
  /// let mut arr = [0.0_f64; 128];
  /// // Construct a vector with the right norm.
  /// arr[0] = (128.0_f64).sqrt();
  /// let _ = PostXvecEmbedding::from_pyannote_capture(arr).expect("right norm");
  /// # }
  /// ```
  ///
  /// Raw `[f64; 128]` cannot be passed to `plda_transform` directly:
  ///
  /// ```compile_fail
  /// use dia::plda::PldaTransform;
  /// let plda = PldaTransform::new().unwrap();
  /// let arr = [0.0_f64; 128];
  /// // Doesn't compile — `plda_transform` requires `&PostXvecEmbedding`.
  /// let _ = plda.plda_transform(&arr);
  /// ```
  #[cfg(any(feature = "plda-fixtures", test))]
  #[cfg_attr(docsrs, doc(cfg(feature = "plda-fixtures")))]
  pub fn from_pyannote_capture(arr: [f64; PLDA_DIMENSION]) -> Result<Self, Error> {
    if !arr.iter().all(|v| v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    let norm: f64 = arr.iter().map(|v| v * v).sum::<f64>().sqrt();
    let expected = (PLDA_DIMENSION as f64).sqrt();
    let tolerance = 1.0e-3;
    if (norm - expected).abs() > tolerance {
      return Err(Error::WrongPostXvecNorm {
        actual: norm,
        expected,
        tolerance,
      });
    }
    Ok(Self(arr))
  }

  /// Borrow the underlying f64 vector. Gated behind `plda-fixtures`
  /// so external consumers cannot extract a vector and re-wrap it.
  #[cfg(any(feature = "plda-fixtures", test))]
  #[cfg_attr(docsrs, doc(cfg(feature = "plda-fixtures")))]
  pub fn as_array(&self) -> &[f64; PLDA_DIMENSION] {
    &self.0
  }
}

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
    let (eigenvalues_desc, eigenvectors_desc) = generalized_eigh_descending(&b, &w)?;

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
  ///
  /// `input` is a [`RawEmbedding`] — a raw, **unnormalized** WeSpeaker
  /// vector — not [`dia::embed::Embedding`](crate::embed::Embedding)
  /// (L2-normalized) which is the wrong distribution for PLDA.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] if a non-finite value appears in an
  ///   intermediate vector (the input is finite by `RawEmbedding`'s
  ///   construction-time invariant; this guards against arithmetic
  ///   overflows in the LDA projection).
  /// - [`Error::DegenerateInput`] if `‖input - mean1‖ < NORM_EPSILON`
  ///   (the input is essentially equal to `mean1` and L2-normalizing
  ///   the centered vector would amplify noise to dominate signal).
  pub fn xvec_transform(&self, input: &RawEmbedding) -> Result<PostXvecEmbedding, Error> {
    // Input finite-ness is enforced by `RawEmbedding::from_raw_array`,
    // so we don't re-validate here. Intermediate-vector checks happen
    // inside `checked_l2_normalize_in_place` below.

    // 1. Promote f32 input to f64 and center: x = input - mean1.
    let mut x =
      DVector::<f64>::from_iterator(EMBEDDING_DIMENSION, input.0.iter().map(|v| *v as f64));
    x -= &self.mean1;

    // 2. L2-normalize, then scale by sqrt(D_in). Validate the norm
    //    before dividing — pyannote's lambda would silently produce
    //    NaN here for zero-norm input.
    checked_l2_normalize_in_place(&mut x)?;
    x *= self.sqrt_in_dim;

    // 3. lda.T @ x  →  (PLDA_DIMENSION,)-shaped vector.
    //    nalgebra's `tr_mul` is matmul-with-transposed-lhs; avoids
    //    an explicit transpose copy.
    let mut y = self.lda.tr_mul(&x);

    // 4. Recenter: y -= mean2.
    y -= &self.mean2;

    // 5. L2-normalize, then scale by sqrt(D_out). Same validation
    //    as step 2 — guards against degenerate intermediates that
    //    could come from a corrupted upstream LDA matrix.
    checked_l2_normalize_in_place(&mut y)?;
    y *= self.sqrt_out_dim;

    let mut out = [0.0f64; PLDA_DIMENSION];
    for (slot, value) in out.iter_mut().zip(y.iter()) {
      *slot = *value;
    }
    // The algorithm guarantees `‖out‖ == sqrt(D_out)` by construction
    // — no need to re-validate via `from_pyannote_capture`.
    Ok(PostXvecEmbedding::from_xvec_output(out))
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
  ///
  /// `post_xvec` must be a [`PostXvecEmbedding`]. Distribution +
  /// finite-ness are enforced by that type — `plda_transform` itself
  /// does no validation. Codex review HIGH (stage-2 analogue of the
  /// `RawEmbedding` boundary).
  pub fn plda_transform(&self, post_xvec: &PostXvecEmbedding) -> [f64; PLDA_DIMENSION] {
    // 1. Center: x = post_xvec - plda_mu.
    let mut x = DVector::<f64>::from_iterator(PLDA_DIMENSION, post_xvec.0.iter().copied());
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

  /// Convenience: chain `xvec_transform` → `plda_transform`. Returns
  /// only the errors produced by stage 1 (`xvec_transform`); stage 2
  /// is now infallible because [`PostXvecEmbedding`] enforces its
  /// own preconditions.
  pub fn project(&self, input: &RawEmbedding) -> Result<[f64; PLDA_DIMENSION], Error> {
    let post_xvec = self.xvec_transform(input)?;
    Ok(self.plda_transform(&post_xvec))
  }

  /// Eigenvalue diagonal `phi` (descending) — `pyannote.audio.core.plda.PLDA.phi`.
  /// Consumed by VBx in Phase 2 as the across-class covariance diagonal.
  pub fn phi(&self) -> &[f64] {
    self.phi.as_slice()
  }
}

/// In-place L2 normalization with explicit error reporting. Returns
/// [`Error::NonFiniteInput`] if the norm is non-finite (input had
/// NaN/Inf that survived earlier checks; defense-in-depth) and
/// [`Error::DegenerateInput`] if the norm is below
/// `NORM_EPSILON` (dividing would amplify noise to dominate signal).
fn checked_l2_normalize_in_place(v: &mut DVector<f64>) -> Result<(), Error> {
  let n = v.norm();
  if !n.is_finite() {
    return Err(Error::NonFiniteInput);
  }
  if n < NORM_EPSILON as f64 {
    return Err(Error::DegenerateInput);
  }
  *v /= n;
  Ok(())
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
  let chol = w.clone().cholesky().ok_or(Error::WNotPositiveDefinite)?;
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

#[cfg(test)]
mod helper_tests {
  use super::*;

  /// Direct test of the near-zero-norm guard. Constructed at the
  /// helper level rather than the public-API level because real f32
  /// inputs cannot produce a centered f64 norm below `NORM_EPSILON`
  /// after the f32→f64 promotion round-trip noise (see
  /// `src/plda/tests.rs` comment for the analysis).
  #[test]
  fn checked_l2_normalize_rejects_near_zero() {
    let mut v = DVector::<f64>::from_iterator(4, [1e-15, 1e-15, 1e-15, 1e-15]);
    let n = v.norm();
    assert!(
      n < NORM_EPSILON as f64,
      "test input norm {n} must be < epsilon"
    );
    let result = checked_l2_normalize_in_place(&mut v);
    assert!(
      matches!(result, Err(Error::DegenerateInput)),
      "got {result:?}"
    );
  }

  #[test]
  fn checked_l2_normalize_rejects_nan() {
    let mut v = DVector::<f64>::from_iterator(3, [1.0, f64::NAN, 1.0]);
    let result = checked_l2_normalize_in_place(&mut v);
    assert!(
      matches!(result, Err(Error::NonFiniteInput)),
      "got {result:?}"
    );
  }

  #[test]
  fn checked_l2_normalize_rejects_inf() {
    let mut v = DVector::<f64>::from_iterator(3, [1.0, f64::INFINITY, 1.0]);
    let result = checked_l2_normalize_in_place(&mut v);
    assert!(
      matches!(result, Err(Error::NonFiniteInput)),
      "got {result:?}"
    );
  }

  #[test]
  fn checked_l2_normalize_succeeds_on_unit_input() {
    let mut v = DVector::<f64>::from_iterator(3, [3.0, 4.0, 0.0]);
    checked_l2_normalize_in_place(&mut v).expect("non-degenerate, finite");
    let n = v.norm();
    assert!((n - 1.0).abs() < 1e-15, "norm after normalize = {n}");
  }
}
