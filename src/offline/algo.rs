//! Phase 5c offline diarization orchestrator.

use crate::{
  cluster::centroid::SP_ALIVE_THRESHOLD,
  embed::EMBEDDING_DIM,
  pipeline::{AssignEmbeddingsInput, assign_embeddings},
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
  /// Pre-PLDA WeSpeaker raw embeddings, shape `(num_chunks,
  /// num_speakers, EMBEDDING_DIM = 256)`, flattened in row-major
  /// `[c][s][d]` order. Captured by pyannote as
  /// `raw_embeddings.npz["embeddings"]`. f32 to match ONNX output.
  pub raw_embeddings: &'a [f32],
  pub num_chunks: usize,
  pub num_speakers: usize,

  /// Per-chunk per-frame per-speaker segmentation activity, shape
  /// `(num_chunks, num_frames_per_chunk, num_speakers)`, flattened
  /// `[c][f][s]`. f64 to match pyannote's internal precision.
  pub segmentations: &'a [f64],
  pub num_frames_per_chunk: usize,

  /// Per-output-frame instantaneous speaker count from pyannote's
  /// segmentation binarization. Length `num_output_frames`. Drives
  /// reconstruction's top-`count` selection.
  pub count: &'a [u8],
  pub num_output_frames: usize,

  /// Sliding-window timing, both axes (chunk-level + frame-level).
  /// pyannote community-1 default: chunks `(0, 10s, 1s)`, frames
  /// derived from the segmentation model's frame rate.
  pub chunks_sw: SlidingWindow,
  pub frames_sw: SlidingWindow,

  /// PLDA model. Loaded via [`PldaTransform::new`]; weights are
  /// embedded in the binary via `include_bytes!`.
  pub plda: &'a PldaTransform,

  /// AHC linkage threshold (community-1 default: `0.6`).
  pub threshold: f64,
  /// VBx Fa (community-1 default: `0.07`).
  pub fa: f64,
  /// VBx Fb (community-1 default: `0.8`).
  pub fb: f64,
  /// VBx max iterations (community-1 hardcodes `20`).
  pub max_iters: usize,

  /// `min_duration_off` (seconds) for span post-processing — merge
  /// adjacent same-cluster spans separated by a gap `≤
  /// min_duration_off`. Community-1 default: `0.0` (no merging).
  pub min_duration_off: f64,

  /// Optional temporal smoothing epsilon for the reconstruct top-k
  /// selection. `None` = bit-exact pyannote (descending-activation
  /// argmax). `Some(0.1)` matches speakrs's
  /// `ReconstructMethod::Smoothed { epsilon: 0.1 }` default — when
  /// two clusters' activations differ by `< eps`, prefer the one
  /// selected at the previous frame. Reduces flicker between
  /// near-tied speakers; recommended for `OwnedDiarizationPipeline`
  /// since ONNX numerical drift produces more near-ties than
  /// pyannote's pure-PyTorch path.
  pub smoothing_epsilon: Option<f32>,
}

/// Output of [`diarize_offline`].
pub struct OfflineOutput {
  /// Hard speaker assignment per (chunk, speaker_slot). `-2` for
  /// unmatched. Length `num_chunks`; each inner vec has length
  /// `num_speakers`.
  pub hard_clusters: Vec<Vec<i32>>,
  /// Frame-level binary diarization grid `(num_output_frames,
  /// num_clusters)`, flattened row-major `[t][k]`.
  pub discrete_diarization: Vec<f32>,
  pub num_clusters: usize,
  /// RTTM spans (uri-agnostic). Caller wraps with file id to format.
  pub spans: Vec<RttmSpan>,
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
  let pipeline_input = AssignEmbeddingsInput {
    embeddings: &embeddings,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames: num_frames_per_chunk,
    post_plda: &post_plda,
    phi: &phi,
    train_chunk_idx: &train_chunk_idx,
    train_speaker_idx: &train_speaker_idx,
    threshold,
    fa,
    fb,
    max_iters,
  };
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
  for row in &hard_clusters {
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
  let recon_input = ReconstructInput {
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    hard_clusters: &hard_clusters,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    smoothing_epsilon,
  };
  let discrete_diarization = reconstruct(&recon_input)?;

  // ── Stage 6: discrete diarization → RTTM spans ─────────────────
  let spans = discrete_to_spans(
    &discrete_diarization,
    num_output_frames,
    num_clusters,
    frames_sw,
    min_duration_off,
  );

  Ok(OfflineOutput {
    hard_clusters,
    discrete_diarization,
    num_clusters,
    spans,
  })
}
