//! Pyannote `cluster_vbx` flow stages 2–7 wired end-to-end.

use crate::ahc::ahc_init;
use crate::centroid::{SP_ALIVE_THRESHOLD, weighted_centroids};
use crate::hungarian::{UNMATCHED, constrained_argmax};
use crate::pipeline::error::Error;
use crate::vbx::{StopReason, vbx_iterate};
use nalgebra::{DMatrix, DVector};

/// Pyannote's `qinit` smoothing factor: each AHC label becomes a
/// `softmax(7.0 * one_hot)` row over `num_init` columns. Hardcoded in
/// pyannote (`utils/vbx.py:cluster_vbx`).
const QINIT_SMOOTHING: f64 = 7.0;

/// Inputs to [`assign_embeddings`]. Grouped to keep the function
/// signature manageable.
#[derive(Debug, Clone)]
pub struct AssignEmbeddingsInput<'a> {
  /// Raw per-chunk per-speaker embeddings, flattened to
  /// `(num_chunks * num_speakers, embed_dim)`. Row
  /// `c * num_speakers + s` is the embedding for `(chunk c, speaker s)`.
  /// These feed stages 5 (centroid) and 6 (cdist).
  pub embeddings: &'a DMatrix<f64>,
  pub num_chunks: usize,
  pub num_speakers: usize,
  /// Per-chunk per-frame per-speaker activity. Flattened to a length
  /// `num_chunks * num_frames * num_speakers` slice with stride order
  /// `[c][f][s]`. Used to derive the inactive `(chunk, speaker)` mask
  /// that pyannote's constrained_assignment overrides at stage 7.
  pub segmentations: &'a [f64],
  pub num_frames: usize,
  /// Post-PLDA features for the active training subset, shape
  /// `(num_train, plda_dim)`. Pyannote computes this via
  /// `self.plda(train_embeddings)`. Phase 1's PLDA parity test
  /// already validates the projection; this pipeline accepts
  /// pre-projected features.
  pub post_plda: &'a DMatrix<f64>,
  /// Eigenvalue diagonal `phi` (`PldaTransform::phi()`). Length
  /// `plda_dim`. Consumed by VBx.
  pub phi: &'a DVector<f64>,
  /// Indices of active `(chunk, speaker)` pairs in row-major order
  /// matching `post_plda` rows. Length `num_train`. Pyannote
  /// computes these via `filter_embeddings`.
  pub train_chunk_idx: &'a [usize],
  pub train_speaker_idx: &'a [usize],
  /// AHC linkage threshold. Pyannote community-1 default: `0.6`.
  pub threshold: f64,
  /// VBx Fa (sufficient-statistics scale). Community-1: `0.07`.
  pub fa: f64,
  /// VBx Fb (speaker regularization). Community-1: `0.8`.
  pub fb: f64,
  /// VBx max iterations. Pyannote hardcodes `20`.
  pub max_iters: usize,
}

/// Run pyannote's `cluster_vbx` flow (stages 2–7).
///
/// Returns `Vec<Vec<i32>>` of length `num_chunks`; each inner vector is
/// length `num_speakers`. Entries are alive-cluster indices in the
/// reduced (`sp > SP_ALIVE_THRESHOLD`) cluster space, or
/// [`crate::hungarian::UNMATCHED`] = `-2` for speakers with no
/// surviving cluster.
pub fn assign_embeddings(input: &AssignEmbeddingsInput<'_>) -> Result<Vec<Vec<i32>>, Error> {
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
  if embeddings.nrows() != num_chunks * num_speakers {
    return Err(Error::Shape(
      "embeddings.nrows() must equal num_chunks * num_speakers",
    ));
  }
  if num_frames == 0 {
    return Err(Error::Shape("num_frames must be at least 1"));
  }
  if segmentations.len() != num_chunks * num_frames * num_speakers {
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
  if num_train < 2 {
    return Err(Error::TooFewActiveEmbeddings(num_train));
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
  let ahc_clusters = ahc_init(&train_embeddings, threshold)?;

  // ── Stage 3 (caller-supplied): post_plda is the VBx feature matrix.
  // ── Stage 4: VBx ──────────────────────────────────────────────
  let num_init = ahc_clusters.iter().copied().max().expect("num_train >= 2") + 1;
  let qinit = build_qinit(&ahc_clusters, num_init);
  let vbx_out = vbx_iterate(post_plda, phi, &qinit, fa, fb, max_iters)?;
  if vbx_out.stop_reason == StopReason::MaxIterationsReached {
    // Pyannote silently accepts max_iters reached — it's the common
    // case in real data (16 of 20 captured iters converged but pyannote
    // doesn't check). The Rust port follows suit; downstream consumers
    // can inspect VbxOutput separately if they need the convergence flag.
  }

  // ── Stage 5: drop sp-squashed clusters, compute centroids ───────
  let centroids = weighted_centroids(
    &vbx_out.gamma,
    &vbx_out.pi,
    &train_embeddings,
    SP_ALIVE_THRESHOLD,
  )?;
  let num_alive = centroids.nrows();

  // ── Stage 6: cdist(embeddings, centroids, metric="cosine") ─────
  // Then `soft_clusters = 2 - e2k_distance`. Per pyannote.
  let mut soft = vec![DMatrix::<f64>::zeros(num_speakers, num_alive); num_chunks];
  for (c, soft_c) in soft.iter_mut().enumerate() {
    for s in 0..num_speakers {
      let row = c * num_speakers + s;
      for k in 0..num_alive {
        let dist = cosine_distance_rows(embeddings, row, &centroids, k);
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
/// finite vectors. Returns `1.0` (max distance) for zero-norm rows to
/// avoid NaN — the caller is expected to have validated finite-ness
/// upstream, but a zero-norm row can still arise via centroid
/// computation if all weights are tiny.
fn cosine_distance_rows(a: &DMatrix<f64>, ra: usize, b: &DMatrix<f64>, rb: usize) -> f64 {
  let cols = a.ncols();
  debug_assert_eq!(b.ncols(), cols);
  let mut dot = 0.0;
  let mut na = 0.0;
  let mut nb = 0.0;
  for i in 0..cols {
    let av = a[(ra, i)];
    let bv = b[(rb, i)];
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  let denom = na.sqrt() * nb.sqrt();
  if denom == 0.0 {
    return 1.0;
  }
  1.0 - dot / denom
}
