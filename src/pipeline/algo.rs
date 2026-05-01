//! Pyannote `cluster_vbx` flow stages 2–7 wired end-to-end.

use crate::{
  cluster::{
    ahc::ahc_init,
    centroid::{SP_ALIVE_THRESHOLD, weighted_centroids},
    hungarian::{UNMATCHED, constrained_argmax},
    vbx::{StopReason, vbx_iterate},
  },
  pipeline::error::Error,
};
use nalgebra::{DMatrix, DVector};

/// Pyannote's `qinit` smoothing factor: each AHC label becomes a
/// `softmax(7.0 * one_hot)` row over `num_init` columns. Hardcoded in
/// pyannote (`utils/vbx.py:cluster_vbx`).
const QINIT_SMOOTHING: f64 = 7.0;

/// Inputs to [`assign_embeddings`]. Grouped to keep the function
/// signature manageable.
#[derive(Debug, Clone)]
pub struct AssignEmbeddingsInput<'a> {
  embeddings: &'a DMatrix<f64>,
  num_chunks: usize,
  num_speakers: usize,
  segmentations: &'a [f64],
  num_frames: usize,
  post_plda: &'a DMatrix<f64>,
  phi: &'a DVector<f64>,
  train_chunk_idx: &'a [usize],
  train_speaker_idx: &'a [usize],
  threshold: f64,
  fa: f64,
  fb: f64,
  max_iters: usize,
}

impl<'a> AssignEmbeddingsInput<'a> {
  /// Construct.
  ///
  /// Field meaning:
  /// - `embeddings`: raw per-(chunk, speaker) embeddings flattened to
  ///   `(num_chunks * num_speakers, embed_dim)`.
  /// - `segmentations`: per-`(chunk, frame, speaker)` activity flattened
  ///   `[c][f][s]`. Length `num_chunks * num_frames * num_speakers`.
  /// - `post_plda`: post-PLDA features for the active training subset,
  ///   shape `(num_train, plda_dim)`.
  /// - `phi`: eigenvalue diagonal (length `plda_dim`).
  /// - `train_chunk_idx` / `train_speaker_idx`: row-major active
  ///   indices, length `num_train`.
  /// - `threshold`/`fa`/`fb`/`max_iters`: AHC and VBx hyperparameters.
  ///   Community-1 defaults: 0.6, 0.07, 0.8, 20.
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    embeddings: &'a DMatrix<f64>,
    num_chunks: usize,
    num_speakers: usize,
    segmentations: &'a [f64],
    num_frames: usize,
    post_plda: &'a DMatrix<f64>,
    phi: &'a DVector<f64>,
    train_chunk_idx: &'a [usize],
    train_speaker_idx: &'a [usize],
    threshold: f64,
    fa: f64,
    fb: f64,
    max_iters: usize,
  ) -> Self {
    Self {
      embeddings,
      num_chunks,
      num_speakers,
      segmentations,
      num_frames,
      post_plda,
      phi,
      train_chunk_idx,
      train_speaker_idx,
      threshold,
      fa,
      fb,
      max_iters,
    }
  }

  /// Raw per-`(chunk, speaker)` embeddings.
  pub const fn embeddings(&self) -> &'a DMatrix<f64> {
    self.embeddings
  }
  /// Number of chunks.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Speaker slots per chunk.
  pub const fn num_speakers(&self) -> usize {
    self.num_speakers
  }
  /// Per-`(chunk, frame, speaker)` activity.
  pub const fn segmentations(&self) -> &'a [f64] {
    self.segmentations
  }
  /// Frames per chunk.
  pub const fn num_frames(&self) -> usize {
    self.num_frames
  }
  /// Post-PLDA features for the active training subset.
  pub const fn post_plda(&self) -> &'a DMatrix<f64> {
    self.post_plda
  }
  /// PLDA eigenvalue diagonal.
  pub const fn phi(&self) -> &'a DVector<f64> {
    self.phi
  }
  /// Active chunk indices (length `num_train`).
  pub const fn train_chunk_idx(&self) -> &'a [usize] {
    self.train_chunk_idx
  }
  /// Active speaker indices (length `num_train`).
  pub const fn train_speaker_idx(&self) -> &'a [usize] {
    self.train_speaker_idx
  }
  /// AHC linkage threshold.
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }
  /// VBx Fa.
  pub const fn fa(&self) -> f64 {
    self.fa
  }
  /// VBx Fb.
  pub const fn fb(&self) -> f64 {
    self.fb
  }
  /// VBx max iterations.
  pub const fn max_iters(&self) -> usize {
    self.max_iters
  }
}

/// Run pyannote's `cluster_vbx` flow (stages 2–7).
///
/// Returns `Vec<Vec<i32>>` of length `num_chunks`; each inner vector is
/// length `num_speakers`. Entries are alive-cluster indices in the
/// reduced (`sp > SP_ALIVE_THRESHOLD`) cluster space, or
/// [`crate::cluster::hungarian::UNMATCHED`] = `-2` for speakers with no
/// surviving cluster.
///
/// # Speaker-count constraints (currently unsupported)
///
/// Pyannote's `cluster_vbx` (`clustering.py:617-633`) supports
/// `num_clusters` / `min_clusters` / `max_clusters` constraints by
/// running a KMeans fallback over the L2-normalized training
/// embeddings *after* VBx, when auto-VBx's cluster count violates
/// the constraints. This Rust port currently only exposes the
/// auto-VBx path — there is no `num_clusters` field in
/// [`AssignEmbeddingsInput`]. All five captured fixtures used the
/// auto path, so existing parity tests are unaffected, but any
/// caller that needs a forced speaker count must either
/// post-process VBx output or wait for this feature to land.
///
/// **TODO** (Codex review HIGH round 2 of Phase 5): add
/// `num_clusters: Option<usize>`, `min_clusters: Option<usize>`,
/// `max_clusters: Option<usize>` to the input struct and port
/// pyannote's KMeans branch when an auto-VBx count violates the
/// constraints. Adding it will require:
///   1. A k-means++ implementation (or a `linfa-clustering` dep) on
///      L2-normalized embeddings — pyannote uses sklearn's KMeans
///      with `n_init=3, random_state=42`.
///   2. Centroid recomputation from the KMeans cluster assignment.
///   3. Disabling `constrained_assignment` in this branch (pyannote
///      does this to avoid artificial cluster inflation).
///   4. A new fixture captured with `num_clusters` forcing != auto.
pub fn assign_embeddings(input: &AssignEmbeddingsInput<'_>) -> Result<Vec<Vec<i32>>, Error> {
  // SIMD always on. Earlier review rounds gated this to aarch64 to
  // chase cross-arch ulp determinism; a later round caught that
  // nalgebra/matrixmultiply runs its own AVX/FMA/NEON GEMM inside
  // VBx outside the gate, so cross-arch bit-equality was never
  // actually deliverable. The gate was theatre. NEON matches
  // scalar bit-exact at the primitive level (verified end-to-end
  // by `assign_embeddings_scalar_and_simd_produce_identical_hard_clusters`
  // on aarch64); x86 SIMD diverges by ulps but tracks
  // matrixmultiply's GEMM precision. Algorithm robustness against
  // those ulp drifts is validated empirically by
  // `pipeline::parity_tests` and `offline::parity_tests` against
  // pyannote captures (DER ≤ 0.4% on all 6 captured fixtures).
  assign_embeddings_inner(input, true)
}

/// Test-only entrypoint: identical to [`assign_embeddings`] but
/// threads an explicit `use_simd` flag through every internal
/// `ops::*` callsite (`ahc_init`, `vbx_iterate`,
/// `weighted_centroids`, stage-6 cosine, `cosine_distance_pre_norm`).
/// Used by the end-to-end backend-forced differential test in
/// [`crate::pipeline::tests`] to prove that scalar and SIMD
/// produce *bit-identical* hard cluster assignments on aarch64
/// (Codex adversarial review, repeated rounds).
#[cfg(test)]
pub(crate) fn assign_embeddings_with_simd(
  input: &AssignEmbeddingsInput<'_>,
  use_simd: bool,
) -> Result<Vec<Vec<i32>>, Error> {
  assign_embeddings_inner(input, use_simd)
}

fn assign_embeddings_inner(
  input: &AssignEmbeddingsInput<'_>,
  use_simd: bool,
) -> Result<Vec<Vec<i32>>, Error> {
  let &AssignEmbeddingsInput {
    embeddings,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames,
    post_plda,
    phi,
    train_chunk_idx,
    train_speaker_idx,
    threshold,
    fa,
    fb,
    max_iters,
  } = input;

  // ── Boundary checks ────────────────────────────────────────────
  if num_chunks == 0 {
    return Err(Error::Shape("num_chunks must be at least 1"));
  }
  if num_speakers == 0 {
    return Err(Error::Shape("num_speakers must be at least 1"));
  }
  let embed_dim = embeddings.ncols();
  if embed_dim == 0 {
    return Err(Error::Shape("embeddings must have at least one column"));
  }
  // Use checked arithmetic at the public boundary: enormous dimension
  // products would otherwise wrap silently in release builds, letting
  // a malformed caller match the equality check with a tiny buffer
  // and reach allocation/index code with bogus shape metadata. Mirrors
  // `offline::algo`. Codex review HIGH round 7.
  let expected_emb_rows = num_chunks
    .checked_mul(num_speakers)
    .ok_or(Error::Shape("num_chunks * num_speakers overflows usize"))?;
  if embeddings.nrows() != expected_emb_rows {
    return Err(Error::Shape(
      "embeddings.nrows() must equal num_chunks * num_speakers",
    ));
  }
  if num_frames == 0 {
    return Err(Error::Shape("num_frames must be at least 1"));
  }
  let expected_seg_len = num_chunks
    .checked_mul(num_frames)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(Error::Shape(
      "num_chunks * num_frames * num_speakers overflows usize",
    ))?;
  if segmentations.len() != expected_seg_len {
    return Err(Error::Shape(
      "segmentations.len() must equal num_chunks * num_frames * num_speakers",
    ));
  }
  if train_chunk_idx.len() != train_speaker_idx.len() {
    return Err(Error::Shape(
      "train_chunk_idx and train_speaker_idx must have the same length",
    ));
  }
  let num_train = train_chunk_idx.len();
  if post_plda.nrows() != num_train {
    return Err(Error::Shape("post_plda.nrows() must equal num_train"));
  }
  let plda_dim = post_plda.ncols();
  if plda_dim == 0 {
    // Zero-column post_plda would let VBx iterate on no PLDA evidence
    // — `inv_l`, `alpha`, `log_p` all degenerate to empty/zero. The
    // resulting posterior is independent of the input embeddings,
    // producing plausible hard_clusters from pure prior. A schema
    // drift in PLDA capture or downstream feeding the wrong array
    // would silently yield wrong diarization. Codex review MEDIUM
    // round 7 of Phase 5.
    return Err(Error::Shape(
      "post_plda must have at least one column (PLDA dimension)",
    ));
  }
  if phi.len() != plda_dim {
    return Err(Error::Shape("phi.len() must equal post_plda.ncols()"));
  }
  // Validate train indices stay within bounds — out-of-range silently
  // poisons centroid math by reading garbage embeddings.
  for i in 0..num_train {
    let c = train_chunk_idx[i];
    let s = train_speaker_idx[i];
    if c >= num_chunks {
      return Err(Error::Shape("train_chunk_idx[i] out of range"));
    }
    if s >= num_speakers {
      return Err(Error::Shape("train_speaker_idx[i] out of range"));
    }
  }
  // Validate that *every* row of `embeddings` and *every* entry of
  // `segmentations` is finite. AHC and centroid only validate the
  // train subset (rows indexed by `train_chunk_idx`/`train_speaker_idx`),
  // but stage 6 reads ALL embedding rows for cosine scoring and stage
  // 7 reads ALL segmentations for the inactive-speaker mask. A NaN in
  // a non-train row would silently become a soft cost that
  // `constrained_argmax` rewrites to the global `nanmin` — yielding
  // a plausible-looking but wrong assignment with no surfaced error.
  // Codex adversarial review MEDIUM (this commit).
  for v in embeddings.iter() {
    if !v.is_finite() {
      return Err(Error::NonFinite("embeddings"));
    }
  }
  for v in segmentations.iter() {
    if !v.is_finite() {
      return Err(Error::NonFinite("segmentations"));
    }
  }
  // Pyannote one-cluster fast path (`clustering.py:588-594`): when
  // fewer than 2 active embeddings survive `filter_embeddings`,
  // pyannote skips AHC/VBx entirely and returns
  // `hard_clusters = np.zeros((num_chunks, num_speakers))` —
  // i.e. every speaker in every chunk gets cluster 0. This handles
  // short clips, sparse speech, or single-usable-speaker recordings
  // without erroring. Codex review HIGH round 1 of Phase 5.
  if num_train < 2 {
    return Ok(vec![vec![0i32; num_speakers]; num_chunks]);
  }
  if !(0.0..=f64::MAX).contains(&threshold) || !threshold.is_finite() || threshold <= 0.0 {
    return Err(Error::Shape("threshold must be a positive finite scalar"));
  }
  if max_iters == 0 {
    return Err(Error::Shape("max_iters must be at least 1"));
  }

  // ── Stage 2: AHC on active embeddings ──────────────────────────
  // Project the rows of `embeddings` selected by `(chunk_idx,
  // speaker_idx)` into a contiguous `(num_train, embed_dim)` matrix.
  let mut train_embeddings = DMatrix::<f64>::zeros(num_train, embed_dim);
  for i in 0..num_train {
    let c = train_chunk_idx[i];
    let s = train_speaker_idx[i];
    let row = c * num_speakers + s;
    for d in 0..embed_dim {
      train_embeddings[(i, d)] = embeddings[(row, d)];
    }
  }
  // In test builds, route through the `*_with_simd` variants so the
  // backend-forced differential test in `pipeline::tests` can A/B
  // both backends end-to-end. Production (release) builds always go
  // through the `true` path; the cfg branch compiles to the same
  // call.
  #[cfg(test)]
  let ahc_clusters =
    crate::cluster::ahc::algo::ahc_init_with_simd(&train_embeddings, threshold, use_simd)?;
  #[cfg(not(test))]
  let ahc_clusters = {
    let _ = use_simd;
    ahc_init(&train_embeddings, threshold)?
  };

  // ── Stage 3 (caller-supplied): post_plda is the VBx feature matrix.
  // ── Stage 4: VBx ──────────────────────────────────────────────
  let num_init = ahc_clusters.iter().copied().max().expect("num_train >= 2") + 1;
  let qinit = build_qinit(&ahc_clusters, num_init);
  #[cfg(test)]
  let vbx_out = crate::cluster::vbx::algo::vbx_iterate_with_simd(
    post_plda, phi, &qinit, fa, fb, max_iters, use_simd,
  )?;
  #[cfg(not(test))]
  let vbx_out = vbx_iterate(post_plda, phi, &qinit, fa, fb, max_iters)?;
  if vbx_out.stop_reason() == StopReason::MaxIterationsReached {
    // Pyannote silently accepts max_iters reached — it's the common
    // case in real data (16 of 20 captured iters converged but pyannote
    // doesn't check). The Rust port follows suit; downstream consumers
    // can inspect VbxOutput separately if they need the convergence flag.
  }

  // ── Stage 5: drop sp-squashed clusters, compute centroids ───────
  #[cfg(test)]
  let centroids = crate::cluster::centroid::algo::weighted_centroids_with_simd(
    vbx_out.gamma(),
    vbx_out.pi(),
    &train_embeddings,
    SP_ALIVE_THRESHOLD,
    use_simd,
  )?;
  #[cfg(not(test))]
  let centroids = weighted_centroids(
    vbx_out.gamma(),
    vbx_out.pi(),
    &train_embeddings,
    SP_ALIVE_THRESHOLD,
  )?;
  let num_alive = centroids.nrows();

  // ── Stage 6: cdist(embeddings, centroids, metric="cosine") ─────
  // Then `soft_clusters = 2 - e2k_distance`. Per pyannote.
  //
  // SIMD dot — bit-identical to scalar on aarch64 (see
  // `ops::scalar::dot` module docs). The cosine costs feed
  // `constrained_argmax` (Hungarian) at stage 7; cross-architecture
  // determinism on aarch64 is guaranteed by the scalar/NEON
  // bit-identical contract.
  //
  // nalgebra is column-major so `embeddings.row(r)` and
  // `centroids.row(k)` are strided. We pack all centroid rows into
  // one flat row-major buffer (`centroid_buf`, length
  // `num_alive * embed_dim`, single heap alloc) and reuse one
  // `emb_row` scratch buffer across the inner k-loop. `norm_sq`
  // factors are hoisted: `centroid_norm_sq[k]` is a stage-6
  // constant, `emb_norm_sq` is constant across the inner k-loop.
  let mut soft = vec![DMatrix::<f64>::zeros(num_speakers, num_alive); num_chunks];
  let mut centroid_buf: Vec<f64> = Vec::with_capacity(num_alive * embed_dim);
  for k in 0..num_alive {
    for d in 0..embed_dim {
      centroid_buf.push(centroids[(k, d)]);
    }
  }
  let centroid_norm_sq: Vec<f64> = centroid_buf
    .chunks_exact(embed_dim)
    .map(|row| crate::ops::dot(row, row, use_simd))
    .collect();
  let mut emb_row: Vec<f64> = vec![0.0; embed_dim];
  for (c, soft_c) in soft.iter_mut().enumerate() {
    for s in 0..num_speakers {
      let row = c * num_speakers + s;
      for d in 0..embed_dim {
        emb_row[d] = embeddings[(row, d)];
      }
      let emb_norm_sq = crate::ops::dot(&emb_row, &emb_row, use_simd);
      for (k, c_row) in centroid_buf.chunks_exact(embed_dim).enumerate() {
        let dist =
          cosine_distance_pre_norm(&emb_row, emb_norm_sq, c_row, centroid_norm_sq[k], use_simd);
        soft_c[(s, k)] = 2.0 - dist;
      }
    }
  }

  // ── Stage 7: constrained_assignment masking + Hungarian ────────
  // Pyannote: const = soft.min() - 1.0; soft[seg.sum(1) == 0] = const.
  // The mask is over (chunk, speaker) where every frame had zero
  // activity — equivalently, the speaker is "off" in this chunk.
  let mut soft_min = f64::INFINITY;
  for chunk in &soft {
    for v in chunk.iter() {
      if *v < soft_min {
        soft_min = *v;
      }
    }
  }
  let inactive_const = soft_min - 1.0;
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      // sum over frames of seg[c, f, s].
      let mut sum_activity = 0.0;
      for f in 0..num_frames {
        sum_activity += segmentations[(c * num_frames + f) * num_speakers + s];
      }
      if sum_activity == 0.0 {
        for k in 0..num_alive {
          soft[c][(s, k)] = inactive_const;
        }
      }
    }
  }
  let hard = constrained_argmax(&soft)?;

  // Sanity: hard.len() == num_chunks; each row has length num_speakers.
  debug_assert_eq!(hard.len(), num_chunks);
  for row in &hard {
    debug_assert_eq!(row.len(), num_speakers);
  }
  let _ = UNMATCHED; // doc reference

  Ok(hard)
}

/// Build pyannote's `qinit = scipy_softmax(one_hot(ahc_clusters) * 7.0)`
/// matrix. Shape `(num_train, num_init)` with each row a softmax of a
/// one-hot vector at column `ahc_clusters[i]`. Smoothing factor 7.0 is
/// hardcoded in `pyannote.audio.utils.vbx.cluster_vbx`.
fn build_qinit(ahc_clusters: &[usize], num_init: usize) -> DMatrix<f64> {
  let n = ahc_clusters.len();
  let on_logit = QINIT_SMOOTHING;
  // softmax over (one_hot * 7.0): row r has logits [0, …, 7 (at hot col), …, 0].
  // Numerator: exp(7.0) at hot col, exp(0) = 1 elsewhere.
  // Denominator: exp(7.0) + (num_init - 1).
  let on_exp = on_logit.exp();
  let denom = on_exp + (num_init - 1) as f64;
  let on_mass = on_exp / denom;
  let off_mass = 1.0 / denom;
  let mut q = DMatrix::<f64>::from_element(n, num_init, off_mass);
  for (r, &k) in ahc_clusters.iter().enumerate() {
    q[(r, k)] = on_mass;
  }
  q
}

/// Cosine distance between two rows of two matrices: `1 - dot / (|a| *
/// |b|)`. Matches `scipy.spatial.distance.cdist(metric="cosine")` for
/// finite vectors.
///
/// Zero-norm rows return `NaN` (matching scipy's 0/0 behavior). Stage
/// 7's `diarization::cluster::hungarian::constrained_argmax` rewrites NaN to the global
/// nanmin via `np.nan_to_num`, so a zero-norm active row gets the
/// worst possible cost and is NOT preferred over genuinely-similar
/// embeddings. Returning `1.0` (mid-similarity) instead — as the
/// previous version did — would have let a corrupt zero-vector
/// embedding tie or beat a real low-similarity match. Codex review
/// HIGH round 3 of Phase 5.
///
/// Cosine distance variant that takes pre-computed `||row||²` for
/// both inputs. Used by stage 6's hot inner loop where `norm_sq_b` is
/// constant across the k-iteration and `norm_sq_a` is constant across
/// the cluster loop — so the caller hoists both out and only pays for
/// one `ops::dot` per (c, s, k).
///
/// Operates on contiguous f64 slices (`row_a`, `row_b`) — the layout
/// `ops::dot` expects.
fn cosine_distance_pre_norm(
  row_a: &[f64],
  norm_sq_a: f64,
  row_b: &[f64],
  norm_sq_b: f64,
  use_simd: bool,
) -> f64 {
  debug_assert_eq!(row_a.len(), row_b.len());
  // SIMD dot: scalar/NEON bit-identical contract — see stage-6
  // comment block above.
  let dot = crate::ops::dot(row_a, row_b, use_simd);
  let denom = norm_sq_a.sqrt() * norm_sq_b.sqrt();
  if denom == 0.0 {
    return f64::NAN;
  }
  1.0 - dot / denom
}
