//! Phase 5c offline diarization orchestrator.

use std::sync::Arc;

use crate::{
  cluster::centroid::SP_ALIVE_THRESHOLD,
  embed::EMBEDDING_DIM,
  pipeline::{AssignEmbeddingsInput, ChunkAssignment, assign_embeddings},
  plda::{PldaTransform, RawEmbedding},
  reconstruct::{ReconstructInput, RttmSpan, SlidingWindow, discrete_to_spans, reconstruct},
};
use nalgebra::{DMatrix, DVector};

/// Diarizer error type (re-exports the pipeline error since that's
/// where most failures surface in offline mode).
#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("offline: {0}")]
  Shape(&'static str),
  #[error("offline: pipeline: {0}")]
  Pipeline(#[from] crate::pipeline::Error),
  #[error("offline: reconstruct: {0}")]
  Reconstruct(#[from] crate::reconstruct::Error),
  #[error("offline: plda: {0}")]
  Plda(#[from] crate::plda::Error),
}

/// Inputs to [`diarize_offline`].
///
/// Caller has already produced segmentation + raw-embedding tensors
/// via their own ONNX inference. Tensors must follow the pyannote
/// community-1 layout.
pub struct OfflineInput<'a> {
  raw_embeddings: &'a [f32],
  num_chunks: usize,
  num_speakers: usize,
  segmentations: &'a [f64],
  num_frames_per_chunk: usize,
  count: &'a [u8],
  num_output_frames: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
  plda: &'a PldaTransform,
  threshold: f64,
  fa: f64,
  fb: f64,
  max_iters: usize,
  min_duration_off: f64,
  smoothing_epsilon: Option<f32>,
}

impl<'a> OfflineInput<'a> {
  /// Construct with `community-1` hyperparameter defaults
  /// (`threshold = 0.6`, `fa = 0.07`, `fb = 0.8`, `max_iters = 20`,
  /// `min_duration_off = 0.0`, `smoothing_epsilon = None`). Override
  /// individual hyperparameters via the `with_*` builders.
  ///
  /// Required data inputs:
  /// - `raw_embeddings`: pre-PLDA WeSpeaker raw embeddings, flattened
  ///   `[c][s][d]`. Length `num_chunks * num_speakers * EMBEDDING_DIM`.
  /// - `segmentations`: per-`(chunk, frame, speaker)` activity flattened
  ///   `[c][f][s]`. Length `num_chunks * num_frames_per_chunk * num_speakers`.
  /// - `count`: per-output-frame instantaneous speaker count.
  ///   Length `num_output_frames`.
  /// - `chunks_sw` / `frames_sw`: sliding-window timing.
  /// - `plda`: PLDA model.
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    raw_embeddings: &'a [f32],
    num_chunks: usize,
    num_speakers: usize,
    segmentations: &'a [f64],
    num_frames_per_chunk: usize,
    count: &'a [u8],
    num_output_frames: usize,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
    plda: &'a PldaTransform,
  ) -> Self {
    Self {
      raw_embeddings,
      num_chunks,
      num_speakers,
      segmentations,
      num_frames_per_chunk,
      count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      plda,
      // Community-1 defaults.
      threshold: 0.6,
      fa: 0.07,
      fb: 0.8,
      max_iters: 20,
      min_duration_off: 0.0,
      smoothing_epsilon: None,
    }
  }

  /// Set the AHC linkage threshold (builder).
  #[must_use]
  pub const fn with_threshold(mut self, threshold: f64) -> Self {
    self.threshold = threshold;
    self
  }

  /// Set the VBx Fa hyperparameter (builder).
  #[must_use]
  pub const fn with_fa(mut self, fa: f64) -> Self {
    self.fa = fa;
    self
  }

  /// Set the VBx Fb hyperparameter (builder).
  #[must_use]
  pub const fn with_fb(mut self, fb: f64) -> Self {
    self.fb = fb;
    self
  }

  /// Set the VBx max-iterations cap (builder).
  #[must_use]
  pub const fn with_max_iters(mut self, max_iters: usize) -> Self {
    self.max_iters = max_iters;
    self
  }

  /// Set the gap-merging threshold for span post-processing (builder).
  #[must_use]
  pub const fn with_min_duration_off(mut self, min_duration_off: f64) -> Self {
    self.min_duration_off = min_duration_off;
    self
  }

  /// Set the temporal-smoothing epsilon for reconstruct (builder).
  /// `None` = bit-exact pyannote argmax. `Some(0.1)` recommended for
  /// `OwnedDiarizationPipeline`.
  #[must_use]
  pub const fn with_smoothing_epsilon(mut self, smoothing_epsilon: Option<f32>) -> Self {
    self.smoothing_epsilon = smoothing_epsilon;
    self
  }

  /// Pre-PLDA WeSpeaker raw embeddings.
  pub const fn raw_embeddings(&self) -> &'a [f32] {
    self.raw_embeddings
  }
  /// Number of chunks.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Speaker slots per chunk.
  pub const fn num_speakers(&self) -> usize {
    self.num_speakers
  }
  /// Per-`(chunk, frame, speaker)` segmentation activity.
  pub const fn segmentations(&self) -> &'a [f64] {
    self.segmentations
  }
  /// Frames per chunk.
  pub const fn num_frames_per_chunk(&self) -> usize {
    self.num_frames_per_chunk
  }
  /// Per-output-frame speaker count.
  pub const fn count(&self) -> &'a [u8] {
    self.count
  }
  /// Output-frame grid length.
  pub const fn num_output_frames(&self) -> usize {
    self.num_output_frames
  }
  /// Outer (chunk-level) sliding window.
  pub const fn chunks_sw(&self) -> SlidingWindow {
    self.chunks_sw
  }
  /// Inner (frame-level) sliding window.
  pub const fn frames_sw(&self) -> SlidingWindow {
    self.frames_sw
  }
  /// PLDA model.
  pub const fn plda(&self) -> &'a PldaTransform {
    self.plda
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
  /// Gap merging threshold for span post-processing.
  pub const fn min_duration_off(&self) -> f64 {
    self.min_duration_off
  }
  /// Optional smoothing epsilon for reconstruct.
  pub const fn smoothing_epsilon(&self) -> Option<f32> {
    self.smoothing_epsilon
  }
}

/// Output of [`diarize_offline`].
///
/// Owned slices are `Arc<[T]>` so multiple downstream consumers
/// (RTTM emission, metric computation, visualization, etc.) can share
/// the same buffer with cheap `Arc::clone` rather than re-allocating.
#[derive(Debug, Clone)]
pub struct OfflineOutput {
  hard_clusters: Arc<[ChunkAssignment]>,
  discrete_diarization: Arc<[f32]>,
  num_clusters: usize,
  spans: Arc<[RttmSpan]>,
}

impl OfflineOutput {
  /// Construct.
  pub fn new(
    hard_clusters: Arc<[ChunkAssignment]>,
    discrete_diarization: Arc<[f32]>,
    num_clusters: usize,
    spans: Arc<[RttmSpan]>,
  ) -> Self {
    Self {
      hard_clusters,
      discrete_diarization,
      num_clusters,
      spans,
    }
  }

  /// Cheap-clone handle to the per-chunk hard speaker assignment.
  /// Each row is `[i32; MAX_SPEAKER_SLOTS]` (= 3) with `-2` for
  /// unmatched slots. Length = `num_chunks`.
  pub fn hard_clusters(&self) -> Arc<[ChunkAssignment]> {
    Arc::clone(&self.hard_clusters)
  }

  /// Borrow the per-chunk hard speaker assignment without cloning the
  /// `Arc`.
  pub fn hard_clusters_slice(&self) -> &[ChunkAssignment] {
    &self.hard_clusters
  }

  /// Cheap-clone handle to the frame-level binary diarization grid
  /// `(num_output_frames, num_clusters)`, flattened row-major
  /// `[t][k]`.
  pub fn discrete_diarization(&self) -> Arc<[f32]> {
    Arc::clone(&self.discrete_diarization)
  }

  /// Borrow the frame-level binary diarization grid without cloning
  /// the `Arc`.
  pub fn discrete_diarization_slice(&self) -> &[f32] {
    &self.discrete_diarization
  }

  /// Number of clusters in the output diarization grid.
  pub const fn num_clusters(&self) -> usize {
    self.num_clusters
  }

  /// Cheap-clone handle to the RTTM spans (uri-agnostic). Caller
  /// wraps with file id to format.
  pub fn spans(&self) -> Arc<[RttmSpan]> {
    Arc::clone(&self.spans)
  }

  /// Borrow the RTTM spans without cloning the `Arc`.
  pub fn spans_slice(&self) -> &[RttmSpan] {
    &self.spans
  }
}

/// Run the offline pyannote-equivalent diarization pipeline.
///
/// Mirrors `pyannote.audio.pipelines.clustering.VBxClustering.__call__`
/// plus `pyannote/audio/pipelines/speaker_diarization.SpeakerDiarization.apply`'s
/// reconstruction step. Pyannote-equivalent output on the captured
/// fixtures (parity-tested in `crate::offline::parity_tests`).
///
/// # Errors
///
/// - [`Error::Shape`] if any tensor dimension mismatches.
/// - [`Error::Plda`] if a (chunk, speaker) raw embedding is degenerate
///   (zero-norm / NaN — see [`RawEmbedding::from_raw_array`]).
/// - [`Error::Pipeline`] if `assign_embeddings` rejects a non-finite
///   intermediate or hits a shape gate.
/// - [`Error::Reconstruct`] for non-finite segmentations or invalid
///   sliding-window timing.
pub fn diarize_offline(input: &OfflineInput<'_>) -> Result<OfflineOutput, Error> {
  let &OfflineInput {
    raw_embeddings,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames_per_chunk,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    plda,
    threshold,
    fa,
    fb,
    max_iters,
    min_duration_off,
    smoothing_epsilon,
  } = input;

  // ── Boundary checks ────────────────────────────────────────────
  if num_chunks == 0 {
    return Err(Error::Shape("num_chunks must be at least 1"));
  }
  if num_speakers == 0 {
    return Err(Error::Shape("num_speakers must be at least 1"));
  }
  if num_frames_per_chunk == 0 {
    return Err(Error::Shape("num_frames_per_chunk must be at least 1"));
  }
  let expected_emb_len = num_chunks
    .checked_mul(num_speakers)
    .and_then(|n| n.checked_mul(EMBEDDING_DIM))
    .ok_or(Error::Shape("raw_embeddings size overflow"))?;
  if raw_embeddings.len() != expected_emb_len {
    return Err(Error::Shape(
      "raw_embeddings.len() must equal num_chunks * num_speakers * EMBEDDING_DIM",
    ));
  }
  let expected_seg_len = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(Error::Shape("segmentations size overflow"))?;
  if segmentations.len() != expected_seg_len {
    return Err(Error::Shape(
      "segmentations.len() must equal num_chunks * num_frames_per_chunk * num_speakers",
    ));
  }

  // ── Stage 1: filter active (chunk, speaker) pairs ──────────────
  //
  // Bit-exact port of `pyannote.audio.pipelines.clustering.
  // VBxClustering.filter_embeddings` (community-1):
  //
  //   single_active = sum(seg, axis=speaker) == 1     # per (c, f)
  //   clean[c, s] = sum_f (seg[c, f, s] * single_active[c, f])
  //   active[c, s] = clean[c, s] >= 0.2 * num_frames  # MIN_ACTIVE_RATIO
  //   chunk_idx, speaker_idx = where(active)
  //
  // The clean-frame criterion drops (chunk, speaker) pairs that are
  // ONLY active during overlap regions — where pyannote's powerset
  // segmentation has multiple slots active simultaneously. Their
  // embeddings are noisy mixtures and tend to corrupt AHC + VBx,
  // most catastrophically on 04_three_speaker (heavy 3-way overlap):
  // including them gave 38% DER, dropping them brings it to ~0%.
  //
  // The previous comment claimed pyannote uses a simple `sum > 0`
  // rule; that was wrong — `pyannote/audio/pipelines/clustering.py:
  // filter_embeddings:106-125` is unambiguous. The captured
  // `train_chunk_idx`/`train_speaker_idx` arrays in our fixtures
  // happened to match `sum > 0` for the easier fixtures
  // (01/02/03/05/06) because nearly every (c, s) with non-zero
  // activity also met the 20% clean-frame bar. 04 is the outlier.
  const MIN_ACTIVE_RATIO: f64 = 0.2;
  let min_clean_frames = MIN_ACTIVE_RATIO * num_frames_per_chunk as f64;
  let mut train_chunk_idx: Vec<usize> = Vec::new();
  let mut train_speaker_idx: Vec<usize> = Vec::new();
  for c in 0..num_chunks {
    // Per-frame: how many speakers active at this (c, f)?
    let mut single_active = vec![false; num_frames_per_chunk];
    for f in 0..num_frames_per_chunk {
      let mut active_count = 0u32;
      for s in 0..num_speakers {
        // Pyannote uses BINARIZED segmentations here. The
        // `_speaker_count` and `filter_embeddings` paths both
        // interpret nonzero seg values as active. We've already
        // run binarize upstream (via `>= onset` in the segmentation
        // step that produces the captured/streamed segmentations
        // tensor), so any nonzero entry here is binary-active.
        if segmentations[(c * num_frames_per_chunk + f) * num_speakers + s] > 0.0 {
          active_count += 1;
        }
      }
      single_active[f] = active_count == 1;
    }
    for s in 0..num_speakers {
      let mut clean_frames = 0.0_f64;
      for f in 0..num_frames_per_chunk {
        if single_active[f] {
          clean_frames += segmentations[(c * num_frames_per_chunk + f) * num_speakers + s];
        }
      }
      if clean_frames >= min_clean_frames {
        train_chunk_idx.push(c);
        train_speaker_idx.push(s);
      }
    }
  }
  let num_train = train_chunk_idx.len();

  // ── Stage 2: build full f64 embeddings DMatrix ─────────────────
  // shape (num_chunks * num_speakers, EMBEDDING_DIM).
  let mut embeddings = DMatrix::<f64>::zeros(num_chunks * num_speakers, EMBEDDING_DIM);
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      let row = c * num_speakers + s;
      let base = (c * num_speakers + s) * EMBEDDING_DIM;
      for d in 0..EMBEDDING_DIM {
        embeddings[(row, d)] = raw_embeddings[base + d] as f64;
      }
    }
  }

  // ── Stage 3: PLDA project active embeddings ────────────────────
  let plda_dim = plda.phi().len();
  let mut post_plda = DMatrix::<f64>::zeros(num_train, plda_dim);
  for (i, (&c, &s)) in train_chunk_idx
    .iter()
    .zip(train_speaker_idx.iter())
    .enumerate()
  {
    let base = (c * num_speakers + s) * EMBEDDING_DIM;
    let mut arr = [0.0_f32; EMBEDDING_DIM];
    arr.copy_from_slice(&raw_embeddings[base..base + EMBEDDING_DIM]);
    let raw = RawEmbedding::from_raw_array(arr)?;
    let projected = plda.project(&raw)?;
    for (d, v) in projected.iter().enumerate() {
      post_plda[(i, d)] = *v;
    }
  }
  let phi = DVector::<f64>::from_iterator(plda_dim, plda.phi().iter().copied());

  // ── Stage 4: assign_embeddings (AHC + VBx + centroid + Hungarian) ─
  let pipeline_input = AssignEmbeddingsInput::new(
    &embeddings,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames_per_chunk,
    &post_plda,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(threshold)
  .with_fa(fa)
  .with_fb(fb)
  .with_max_iters(max_iters);
  let hard_clusters = assign_embeddings(&pipeline_input)?;
  let _ = SP_ALIVE_THRESHOLD; // doc reference

  // ── Stage 5: reconstruct → frame-level diarization ──────────────
  //
  // Match `reconstruct`'s internal `num_clusters` computation
  // exactly: it pads up to `max(count)` so the top-K binarization
  // has enough cluster slots. If we under-count here, the
  // `discrete_to_spans` assertion `grid.len() == num_frames *
  // num_clusters` panics for fixtures where `count` peaks higher
  // than the number of distinct hard-cluster ids.
  let mut max_cluster_id = -1i32;
  for row in hard_clusters.iter() {
    for &k in row {
      if k > max_cluster_id {
        max_cluster_id = k;
      }
    }
  }
  let num_clusters_from_hard = if max_cluster_id < 0 {
    0
  } else {
    (max_cluster_id + 1) as usize
  };
  let max_count = count.iter().copied().max().unwrap_or(0) as usize;
  let num_clusters = num_clusters_from_hard.max(max_count.max(1));
  let recon_input = ReconstructInput::new(
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    &hard_clusters,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
  )
  .with_smoothing_epsilon(smoothing_epsilon);
  let discrete_diarization = reconstruct(&recon_input)?;

  // ── Stage 6: discrete diarization → RTTM spans ─────────────────
  let spans = discrete_to_spans(
    &discrete_diarization,
    num_output_frames,
    num_clusters,
    frames_sw,
    min_duration_off,
  );

  // `discrete_to_spans` builds via `Vec::push` because span count is
  // unknown a-priori; convert to `Arc<[RttmSpan]>` once at the
  // boundary. This is a one-time O(num_spans) copy (typically <1000
  // elements) — small price for the fan-out savings on every
  // downstream `Arc::clone`.
  let spans: Arc<[RttmSpan]> = Arc::from(spans);
  Ok(OfflineOutput::new(
    hard_clusters,
    discrete_diarization,
    num_clusters,
    spans,
  ))
}
