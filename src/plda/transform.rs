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
/// # Construction
///
/// Construction is `#[cfg(test)] pub(crate)` — production builds
/// cannot construct a `RawEmbedding` at all. The only production
/// path from a raw WeSpeaker vector to PLDA features will be via
/// `EmbedModel::embed_raw` once Phase 5 lands; that path will own
/// its own typed entry inside `dia::plda` so the boundary stays
/// sealed. (Codex review MEDIUM: a public `plda-fixtures` Cargo
/// feature was previously used as the gate, but additive features
/// are globally unified, so any downstream crate enabling it would
/// have re-exposed the constructor for the entire build.)
///
/// # Type-safety contract
///
/// `xvec_transform`'s signature requires `&RawEmbedding`, so passing
/// the L2-normalized `Embedding` vector is a compile error rather
/// than a silent distribution drift. The
/// `normalized_vs_raw_input_produce_materially_different_output`
/// test in `src/plda/tests.rs` is observable evidence the API
/// distinction matters: feeding the same vector raw vs L2-normalized
/// produces materially different `xvec_transform` outputs.
#[derive(Debug, Clone)]
pub struct RawEmbedding([f32; EMBEDDING_DIMENSION]);

impl RawEmbedding {
  /// Wrap a raw, **unnormalized** WeSpeaker embedding vector.
  /// `#[cfg(test)] pub(crate)` — see [`RawEmbedding`]'s type-level
  /// docs for the visibility rationale.
  ///
  /// Validates the array is finite **and** has non-trivial L2 norm.
  /// Both checks matter: `xvec_transform` centers `input - mean1`
  /// before its inner norm guard fires, so a degraded ONNX output of
  /// all zeros would pass the inner guard (centered norm = `‖mean1‖`)
  /// and silently produce a finite `sqrt(128)`-normed PLDA stage-1
  /// vector that downstream VBx would treat as legitimate speaker
  /// evidence. Rejecting at the **uncentered** input here catches
  /// that class. Codex review HIGH.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] if any element is NaN, `+inf`, or
  ///   `-inf`.
  /// - [`Error::DegenerateInput`] if `‖arr‖ < NORM_EPSILON` — the
  ///   input is effectively zero and any downstream PLDA output
  ///   would be fabricated speaker evidence rather than a real
  ///   embedding signal.
  #[cfg(test)]
  pub(crate) fn from_raw_array(arr: [f32; EMBEDDING_DIMENSION]) -> Result<Self, Error> {
    if !arr.iter().all(|v| v.is_finite()) {
      return Err(Error::NonFiniteInput);
    }
    // Reject degenerate input *before* `xvec_transform` centers it.
    // The norm is computed in f64 because squaring 256 small f32
    // values can lose precision near the threshold.
    let norm_sq: f64 = arr.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    if norm_sq.sqrt() < NORM_EPSILON as f64 {
      return Err(Error::DegenerateInput);
    }
    Ok(Self(arr))
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
/// finite input and return — VBx then clusters wrong-distribution
/// features without any error signal.
///
/// The only production path to a `PostXvecEmbedding` is calling
/// [`PldaTransform::xvec_transform`] (which constructs internally
/// via the `pub(super)` `from_xvec_output`). Parity tests use a
/// `#[cfg(test)] pub(crate)` constructor that loads from a captured
/// pyannote run and validates the norm; that constructor cannot be
/// reached from production builds or downstream crates. Codex
/// review HIGH.
///
/// # Type-safety contract
///
/// `plda_transform`'s signature requires `&PostXvecEmbedding`, so
/// passing a raw `[f64; 128]` is a compile error rather than a
/// silent distribution drift.
#[derive(Debug, Clone)]
pub struct PostXvecEmbedding([f64; PLDA_DIMENSION]);

impl PostXvecEmbedding {
  /// Internal constructor for `xvec_transform`. Skips norm validation
  /// because the algorithm guarantees the invariant by construction.
  pub(super) fn from_xvec_output(arr: [f64; PLDA_DIMENSION]) -> Self {
    Self(arr)
  }

  /// Internal constructor for parity tests that load a `post_xvec`
  /// value from a captured pyannote run. `#[cfg(test)] pub(crate)`
  /// — see [`PostXvecEmbedding`]'s type-level docs for why this is
  /// not reachable from production builds.
  ///
  /// Validates finite + norm within `1e-3` of `sqrt(PLDA_DIMENSION)`.
  /// The norm check is necessary but not sufficient — a synthetic
  /// 128-d vector scaled to `sqrt(128)` would still pass it — which
  /// is precisely why this constructor must remain test-only.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteInput`] on any NaN/`±inf` element.
  /// - [`Error::WrongPostXvecNorm`] if the norm is outside the
  ///   expected `sqrt(D_out) ± 1e-3` band — the input is not a
  ///   post-`xvec_tf` vector.
  #[cfg(test)]
  pub(crate) fn from_pyannote_capture(arr: [f64; PLDA_DIMENSION]) -> Result<Self, Error> {
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

  /// Borrow the underlying f64 vector. Gated alongside
  /// [`Self::from_pyannote_capture`] so the same visibility rules
  /// apply.
  #[cfg(test)]
  pub(crate) fn as_array(&self) -> &[f64; PLDA_DIMENSION] {
    &self.0
  }
}

/// Minimum allowed `‖input - mean1‖` after the first centering step.
///
/// Calibrated against the captured Phase-0 distribution rather than
/// f32 quantization noise: across the 654 raw WeSpeaker embeddings
/// in `tests/parity/fixtures/01_dialogue/raw_embeddings.npz`, the
/// observed centered-norm range is `[1.36, 7.08]` with median 2.45.
/// `0.1` sits ~14× below the empirical minimum (so a far-out-of-
/// distribution real input still passes) and ~2.86 million× above
/// the f32-roundtrip noise floor of `mean1` (~3.49e-8 for the
/// committed weights), so any centered norm in the
/// `[noise_floor, 0.1)` band is rejected.
///
/// # Why a constant rather than the previous noise-floor × 1000
///
/// The earlier threshold was `‖mean1 - mean1.astype(f32)‖ × 1000`
/// ≈ 3.5e-5. That left a ~38000× attack window between threshold
/// and real signal: an embedder collapsed to `mean1.astype(f32) +
/// jitter` with `‖jitter‖` anywhere in `(3.5e-5, 1.36)` would pass
/// the guard, the L2-normalize would amplify the attacker-chosen
/// jitter direction to unit norm, and the rest of the pipeline
/// would whiten that into a fabricated speaker-evidence vector
/// indistinguishable from a real embedding. Calibrating to the
/// data closes that window. Codex review HIGH (round 6).
///
/// If the model weights or the embedder are ever changed, this
/// constant must be re-validated against fresh captured data —
/// see `tests/parity/python/capture_intermediates.py`.
pub(crate) const XVEC_CENTERED_MIN_NORM: f64 = 0.1;

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
/// against the Phase-0 captured artifacts via `src/plda/parity_tests.rs`.
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
  /// - [`Error::DegenerateInput`] if `‖input - mean1‖` is below the
  ///   data-calibrated [`XVEC_CENTERED_MIN_NORM`] threshold (`0.1`
  ///   — see that constant's docs for the calibration), or if the
  ///   second-stage intermediate becomes degenerate. The first check
  ///   rejects both the `mean1.astype(f32)` collapse-to-mean attack
  ///   and the more sophisticated `mean1 + small_jitter` variants
  ///   that an earlier f32-quantization-noise-based threshold would
  ///   have admitted. Codex review HIGH (round 6).
  pub fn xvec_transform(&self, input: &RawEmbedding) -> Result<PostXvecEmbedding, Error> {
    // Input finite-ness is enforced by `RawEmbedding::from_raw_array`,
    // so we don't re-validate here. Intermediate-vector checks happen
    // inside `checked_l2_normalize_in_place` below.

    // 1. Promote f32 input to f64 and center: x = input - mean1.
    let mut x =
      DVector::<f64>::from_iterator(EMBEDDING_DIMENSION, input.0.iter().map(|v| *v as f64));
    x -= &self.mean1;

    // 2. L2-normalize, then scale by sqrt(D_in). Use the
    //    data-calibrated `XVEC_CENTERED_MIN_NORM` threshold here
    //    rather than the shared `NORM_EPSILON`. The threat model is
    //    a degraded or adversarial embedder returning `mean1 +
    //    jitter` for a small `jitter`: the centered f64 norm is
    //    `‖jitter‖`, the L2-normalize amplifies the (attacker-chosen)
    //    direction of `jitter` to unit norm, and the rest of the
    //    pipeline whitens that into a `sqrt(128)`-normed PLDA
    //    stage-1 vector indistinguishable from a real embedding.
    //    The threshold at `0.1` is calibrated against the captured
    //    real-input distribution (smallest observed centered norm
    //    1.36 across 654 raw embeddings); any below-threshold
    //    centered norm cannot be a real WeSpeaker output.
    checked_l2_normalize_in_place_with_min(&mut x, XVEC_CENTERED_MIN_NORM)?;
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
///
/// Used for the stage-2 (post-LDA) intermediate where the noise
/// floor is f64 quantization, well below `NORM_EPSILON`.
fn checked_l2_normalize_in_place(v: &mut DVector<f64>) -> Result<(), Error> {
  checked_l2_normalize_in_place_with_min(v, NORM_EPSILON as f64)
}

/// `checked_l2_normalize_in_place` with a caller-supplied minimum
/// norm. Used by `xvec_transform`'s first centering, where the
/// effective noise floor is `‖mean1.astype(f32) - mean1‖` (the
/// quantization noise of mean1 itself), ~3.5e-8 for the committed
/// weights — far above the shared `NORM_EPSILON = 1e-12`.
fn checked_l2_normalize_in_place_with_min(
  v: &mut DVector<f64>,
  min_norm: f64,
) -> Result<(), Error> {
  let n = v.norm();
  if !n.is_finite() {
    return Err(Error::NonFiniteInput);
  }
  if n < min_norm {
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
