//! Reconstruction math: clustered_segmentations + overlap-add aggregate
//! + top-K binarize.

use std::sync::Arc;

use crate::{
  cluster::hungarian::{ChunkAssignment, UNMATCHED},
  reconstruct::error::Error,
};

/// Hard upper bound on the cluster-id range accepted in `hard_clusters`.
/// Pyannote's diarization pipeline emits ids bounded by the alive
/// cluster count after VBx (typically 1–4). `1024` is ~256× any
/// realistic speaker count; it stops a corrupt or malicious caller
/// from driving the `num_clusters * num_chunks * num_frames_per_chunk`
/// allocation into the multi-GB range.
pub const MAX_CLUSTER_ID: i32 = 1023;

/// Hard upper bound on `count[t]` (instantaneous active speaker count
/// per output frame). Pyannote derives `count` from
/// `aggregate(sum(binarized_seg, axis=-1))`, so the theoretical max is
/// `overlap_factor * num_speakers` ≈ 30 for the community-1 config
/// (10s chunk, 1s step, 3 speakers). Real fixtures observe max=2.
/// Capping at `64` allows comfortable headroom over realistic values
/// while catching `u8::MAX = 255`-style sentinel corruption that would
/// drive `num_clusters` and the top-K binarize past the actual
/// speaker space.
pub const MAX_COUNT_PER_FRAME: u8 = 64;

/// Pyannote `SlidingWindow` (start, duration, step), all in seconds.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindow {
  start: f64,
  duration: f64,
  step: f64,
}

impl SlidingWindow {
  /// Construct a sliding window. All values in seconds.
  pub const fn new(start: f64, duration: f64, step: f64) -> Self {
    Self {
      start,
      duration,
      step,
    }
  }

  /// First-frame center offset (seconds).
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// Per-frame receptive-field length (seconds).
  pub const fn duration(&self) -> f64 {
    self.duration
  }

  /// Stride between consecutive frame centers (seconds).
  pub const fn step(&self) -> f64 {
    self.step
  }

  /// Builder: replace `start`.
  #[must_use]
  pub const fn with_start(mut self, start: f64) -> Self {
    self.start = start;
    self
  }

  /// Builder: replace `duration`.
  #[must_use]
  pub const fn with_duration(mut self, duration: f64) -> Self {
    self.duration = duration;
    self
  }

  /// Builder: replace `step`.
  #[must_use]
  pub const fn with_step(mut self, step: f64) -> Self {
    self.step = step;
    self
  }

  /// `pyannote.core.SlidingWindow.closest_frame(t)` — round to the
  /// nearest frame index whose center is at `t`. Frame `i`'s center
  /// is at `start + duration / 2 + i * step`.
  fn closest_frame(&self, t: f64) -> i64 {
    ((t - self.start - self.duration / 2.0) / self.step).round() as i64
  }
}

/// Inputs to [`reconstruct`].
#[derive(Debug, Clone)]
pub struct ReconstructInput<'a> {
  segmentations: &'a [f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  hard_clusters: &'a [ChunkAssignment],
  count: &'a [u8],
  num_output_frames: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
  smoothing_epsilon: Option<f32>,
}

impl<'a> ReconstructInput<'a> {
  /// Construct with `smoothing_epsilon = None` (bit-exact pyannote
  /// argmax). Pass `Some(eps)` via [`Self::with_smoothing_epsilon`]
  /// to prefer the previous frame's selection when two clusters are
  /// within `eps` activation.
  ///
  /// All shape preconditions are re-verified by [`reconstruct`] —
  /// see its doc-comment for the validation rules.
  ///
  /// Required data inputs:
  /// - `segmentations`: per-`(chunk, frame, speaker)` activity flattened
  ///   `[c][f][s]`. Length `num_chunks * num_frames_per_chunk * num_speakers`.
  /// - `hard_clusters`: per-chunk hard cluster assignment (output of
  ///   `diarization::pipeline`). Length `num_chunks`; each inner vec has
  ///   length `num_speakers` with `-2` indicating an unmatched speaker.
  /// - `count`: per-output-frame instantaneous speaker count.
  ///   Length `num_output_frames`.
  /// - `chunks_sw` / `frames_sw`: outer / inner sliding windows.
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    segmentations: &'a [f64],
    num_chunks: usize,
    num_frames_per_chunk: usize,
    num_speakers: usize,
    hard_clusters: &'a [ChunkAssignment],
    count: &'a [u8],
    num_output_frames: usize,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Self {
    Self {
      segmentations,
      num_chunks,
      num_frames_per_chunk,
      num_speakers,
      hard_clusters,
      count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      smoothing_epsilon: None,
    }
  }

  /// Set the temporal-smoothing epsilon for top-k selection (builder).
  /// `None` = strict descending-activation argmax. `Some(eps)` =
  /// prefer the previous frame's selection when two clusters are
  /// within `eps` activation.
  #[must_use]
  pub const fn with_smoothing_epsilon(mut self, smoothing_epsilon: Option<f32>) -> Self {
    self.smoothing_epsilon = smoothing_epsilon;
    self
  }

  /// Per-`(chunk, frame, speaker)` activity, flattened `[c][f][s]`.
  pub const fn segmentations(&self) -> &'a [f64] {
    self.segmentations
  }
  /// Number of chunks.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Frames per chunk (segmentation model output).
  pub const fn num_frames_per_chunk(&self) -> usize {
    self.num_frames_per_chunk
  }
  /// Speaker slots per chunk.
  pub const fn num_speakers(&self) -> usize {
    self.num_speakers
  }
  /// Per-chunk hard cluster assignment.
  pub const fn hard_clusters(&self) -> &'a [ChunkAssignment] {
    self.hard_clusters
  }
  /// Per-output-frame instantaneous speaker count.
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
  /// Optional smoothing epsilon for top-k selection.
  pub const fn smoothing_epsilon(&self) -> Option<f32> {
    self.smoothing_epsilon
  }
}

/// Run pyannote's reconstruction.
///
/// Returns a binary `(num_output_frames * num_clusters)` flat vector
/// where row `t` has `1.0` at the top-`count[t]` cluster indices by
/// aggregated activation, `0.0` elsewhere.
///
/// `num_clusters` is derived as `max(hard_clusters) + 1`. If all
/// clusters are `UNMATCHED` (`-2`), returns an all-zero grid (no
/// clusters to assign).
///
/// # Errors
///
/// - [`Error::Shape`] for any dimension mismatch.
/// - [`Error::NonFinite`] if `segmentations` contains a non-finite
///   value (NaN handling is supported via [`Inference::aggregate`]'s
///   mask path; arbitrary `±inf` is rejected).
/// - [`Error::Timing`] for non-finite or non-positive sliding-window
///   parameters.
pub fn reconstruct(input: &ReconstructInput<'_>) -> Result<Arc<[f32]>, Error> {
  let &ReconstructInput {
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    hard_clusters,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    smoothing_epsilon,
  } = input;

  use crate::reconstruct::error::{NonFiniteField, ShapeError, TimingError};
  // ── Boundary checks ────────────────────────────────────────────
  if num_chunks == 0 {
    return Err(ShapeError::ZeroNumChunks.into());
  }
  if num_frames_per_chunk == 0 {
    return Err(ShapeError::ZeroNumFramesPerChunk.into());
  }
  if num_speakers == 0 {
    return Err(ShapeError::ZeroNumSpeakers.into());
  }
  // Use checked arithmetic at the public boundary: a malformed caller
  // could pick dimensions whose product wraps in release (e.g.
  // `num_frames_per_chunk = usize::MAX/2 + 1`, `num_speakers = 2`,
  // wrapping to a small value), match the wrapped count with a tiny
  // segmentations slice, and reach allocation/index code with bogus
  // shape metadata. Reject overflow before the equality check.
  let expected_seg_len = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(ShapeError::SegmentationsSizeOverflow)?;
  if segmentations.len() != expected_seg_len {
    return Err(ShapeError::SegmentationsLenMismatch.into());
  }
  if hard_clusters.len() != num_chunks {
    return Err(ShapeError::HardClustersLenMismatch.into());
  }
  // Each `hard_clusters[c]` is `[i32; MAX_SPEAKER_SLOTS]` by type, so
  // its length is statically equal to `MAX_SPEAKER_SLOTS = 3`. We
  // require `num_speakers <= MAX_SPEAKER_SLOTS` so the body's
  // `0..num_speakers` indexing stays in-bounds.
  if num_speakers > crate::segment::options::MAX_SPEAKER_SLOTS as usize {
    return Err(ShapeError::TooManySpeakers.into());
  }
  if num_output_frames == 0 {
    // Zero output frames with nonempty chunks/segmentations is a
    // schema/timing drift signal, not a valid input. Returning an
    // empty grid would make a downstream caller computing
    // `grid.len() / num_output_frames` divide by zero.
    return Err(ShapeError::ZeroNumOutputFrames.into());
  }
  if count.len() != num_output_frames {
    return Err(ShapeError::CountLenMismatch.into());
  }
  // count[t] = instantaneous active speaker count at output frame t.
  // Pyannote derives this from `aggregate(sum(binarized_seg, axis=-1))`
  // which sums per-chunk active counts over overlapping chunks. Real
  // fixtures observe max=2; theoretical max for community-1 is
  // overlap_factor * num_speakers ≈ 30. `MAX_COUNT_PER_FRAME = 64`
  // allows headroom while catching u8::MAX=255 sentinel corruption that
  // would expand `num_clusters` past the actual speaker space and
  // fabricate dummy speakers in the top-K binarize.
  for &c in count {
    if c > MAX_COUNT_PER_FRAME {
      return Err(ShapeError::CountAboveMax.into());
    }
  }
  for w in [chunks_sw, frames_sw] {
    if !w.duration.is_finite() || !w.step.is_finite() || !w.start.is_finite() {
      return Err(TimingError::NonFiniteParameter.into());
    }
    if w.duration <= 0.0 || w.step <= 0.0 {
      return Err(TimingError::NonPositiveDurationOrStep.into());
    }
  }
  // Reject all non-finite segmentation values (NaN and ±inf). Pyannote's
  // `Inference.aggregate` does `np.nan_to_num(score, nan=0.0)` and tracks
  // missingness via a parallel mask, but the realistic source of NaN is
  // upstream model corruption (torch nan-prop), and a silent fallback
  // here lets a degraded inference dependency produce plausible-but-
  // wrong RTTM output. Surfacing it as a clear typed error matches
  // `diarization::cluster::hungarian`'s ±inf rejection at the solver boundary.
  for &v in segmentations {
    if !v.is_finite() {
      return Err(NonFiniteField::Segmentations.into());
    }
  }

  // Validate cluster ids: `UNMATCHED` (-2) is allowed; non-negative
  // values must be in `[0, MAX_CLUSTER_ID]`.
  // round 4: a stray negative id (e.g. -1) silently dropped active
  // speech under the previous code (skipped by the speakers_in_k
  // filter), and a corrupt large positive id could drive the
  // num_clusters allocation into multi-GB range.
  for row in hard_clusters {
    for &k in row {
      if k == UNMATCHED {
        continue;
      }
      if k < 0 {
        return Err(ShapeError::HardClustersNegativeId.into());
      }
      if k > MAX_CLUSTER_ID {
        return Err(ShapeError::HardClustersIdAboveMax.into());
      }
    }
  }

  // Determine num_clusters from hard_clusters.
  let mut max_cluster = -1i32;
  for row in hard_clusters {
    for &k in row {
      if k > max_cluster {
        max_cluster = k;
      }
    }
  }
  if max_cluster < 0 {
    // No assigned clusters anywhere — return all-zero grid built
    // directly via the trusted-len iterator collect. `Range<usize>`
    // is `TrustedLen`, `Map` preserves it, so std's specialized
    // `<Arc<[T]> as FromIterator<T>>::from_iter` allocates the
    // `Arc<[f32]>` once and writes each element straight in.
    return Ok((0..num_output_frames).map(|_| 0.0_f32).collect());
  }
  let num_clusters_from_hard = (max_cluster + 1) as usize;

  // Pyannote pads num_clusters up to `max(count)` if needed (so the
  // top-K binarization can pull at least `count[t]` cluster slots).
  let max_count = count.iter().copied().max().unwrap_or(0) as usize;
  let num_clusters = num_clusters_from_hard.max(max_count.max(1));

  // ── Stage 1: clustered_segmentations ────────────────────────────
  // Initialized to NaN sentinel. We track NaN-ness via a parallel
  // bool mask to avoid f64::is_nan overhead in the aggregation loop.
  // Per-chunk: for each cluster k present in hard_clusters[c],
  // clustered[c, f, k] = max over speakers s where hard_clusters[c, s] == k
  //                       of segmentations[c, f, s].
  // Checked product: `num_clusters` derives from `max_cluster + 1`
  // which is bounded by MAX_CLUSTER_ID validation above, but the
  // multi-axis product can still overflow on adversarial dimensions
  // even if each axis individually is sane.
  let cs_size = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_clusters))
    .ok_or(ShapeError::ClusteredSizeOverflow)?;
  let mut clustered = vec![0.0f64; cs_size];
  let mut clustered_mask = vec![false; cs_size]; // true = valid (not NaN)

  for c in 0..num_chunks {
    for k_iter in 0..num_clusters_from_hard {
      let k = k_iter as i32;
      // Find speakers in this chunk assigned to cluster k.
      let speakers_in_k: Vec<usize> = hard_clusters[c]
        .iter()
        .enumerate()
        .filter_map(|(s, &kk)| (kk == k).then_some(s))
        .collect();
      if speakers_in_k.is_empty() {
        continue;
      }
      for f in 0..num_frames_per_chunk {
        let mut max_act = f64::NEG_INFINITY;
        for &s in &speakers_in_k {
          let v = segmentations[(c * num_frames_per_chunk + f) * num_speakers + s];
          if v > max_act {
            max_act = v;
          }
        }
        let cs_idx = (c * num_frames_per_chunk + f) * num_clusters + k_iter;
        clustered[cs_idx] = max_act;
        clustered_mask[cs_idx] = true;
      }
    }
  }
  // UNMATCHED speakers (k == -2) skipped — clustered_mask stays false
  // for those (cluster, frame) cells and aggregate treats them as NaN
  // (skipped contribution).

  // ── Stage 2: aggregate(skip_average=True) ──────────────────────
  // Pyannote's overlap-add: for each chunk c, find start_frame =
  // closest_frame(chunk_start_time + 0.5 * frame_duration), then
  //   aggregated[start_frame .. start_frame + npc, k] += clustered * mask
  // hamming + warm_up are all-ones in cluster_vbx's call path.
  let mut aggregated = vec![0.0f32; num_output_frames * num_clusters];
  let mut agg_mask = vec![false; num_output_frames * num_clusters];

  for c in 0..num_chunks {
    let chunk_start_time = chunks_sw.start + (c as f64) * chunks_sw.step;
    let center_offset = 0.5 * frames_sw.duration;
    let start_frame = frames_sw.closest_frame(chunk_start_time + center_offset);
    if start_frame < 0 {
      // Pyannote produces frames at non-negative indices; if a chunk
      // starts before the first output frame, clip its leading
      // frames out. start_frame_clamp = 0; clip leading
      // (-start_frame) of the chunk.
    }
    for f in 0..num_frames_per_chunk {
      let out_f = start_frame + f as i64;
      if out_f < 0 || out_f as usize >= num_output_frames {
        continue;
      }
      let out_f = out_f as usize;
      for k in 0..num_clusters_from_hard {
        let cs_idx = (c * num_frames_per_chunk + f) * num_clusters + k;
        if !clustered_mask[cs_idx] {
          continue;
        }
        let v = clustered[cs_idx] as f32;
        let agg_idx = out_f * num_clusters + k;
        aggregated[agg_idx] += v;
        agg_mask[agg_idx] = true;
      }
    }
  }
  // Cells that never received a contribution → leave as 0.0
  // (pyannote uses `missing=0.0` for to_diarization).
  for (i, &m) in agg_mask.iter().enumerate() {
    if !m {
      aggregated[i] = 0.0;
    }
  }

  // ── Stage 3: top-`count[t]` binarize per output frame ──────────
  //
  // Build the output `Arc<[f32]>` directly via
  // `Arc::new_uninit_slice` + `Arc::get_mut` for unique mutable
  // access (the freshly-allocated `Arc` has refcount 1, so
  // `get_mut` returns `Some`). Each cell is initialized to 0.0
  // first, then non-default selections overwrite. After all writes,
  // `assume_init` converts the `MaybeUninit` slice in place — no
  // `Vec`-then-`Arc` round-trip and no copy through the refcount
  // prefix.
  let total = num_output_frames * num_clusters;
  let mut arc_uninit: Arc<[std::mem::MaybeUninit<f32>]> = Arc::new_uninit_slice(total);
  // SAFETY: `arc_uninit` was just constructed and has refcount 1, so
  // `get_mut` returns `Some` with unique mutable access to the
  // backing storage.
  let out: &mut [std::mem::MaybeUninit<f32>] = Arc::get_mut(&mut arc_uninit)
    .expect("Arc::get_mut on freshly-allocated new_uninit_slice (unique owner) returns Some");
  for slot in out.iter_mut() {
    slot.write(0.0_f32);
  }
  // SAFETY: every element was just initialized via `slot.write(0.0)`.
  let out: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), total) };
  let mut prev_selected: Vec<usize> = Vec::new();
  for (t, &c_byte) in count.iter().enumerate().take(num_output_frames) {
    let c_count = c_byte as usize;
    if c_count == 0 {
      prev_selected.clear();
      continue;
    }
    // Sort cluster indices by descending activation at frame t.
    let row_start = t * num_clusters;
    let mut sorted: Vec<usize> = (0..num_clusters).collect();
    if let Some(eps) = smoothing_epsilon {
      // Speakrs-style tie-breaking: when two activations are within
      // `eps` of each other, prefer the one that was selected at
      // t-1. Falls back to descending-activation order otherwise.
      sorted.sort_by(|&a, &b| {
        let va = aggregated[row_start + a];
        let vb = aggregated[row_start + b];
        let diff = (va - vb).abs();
        if diff < eps {
          let a_was = prev_selected.contains(&a);
          let b_was = prev_selected.contains(&b);
          // Bias toward previously-selected (descending order: was-active first).
          b_was.cmp(&a_was)
        } else {
          vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        }
      });
    } else {
      sorted.sort_by(|&a, &b| {
        let va = aggregated[row_start + a];
        let vb = aggregated[row_start + b];
        // Descending: vb.partial_cmp(va). Stable tie-break by index
        // (sort_by is stable since 1.20).
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
      });
    }
    let now_selected: Vec<usize> = sorted.iter().take(c_count).copied().collect();
    for &k in &now_selected {
      out[row_start + k] = 1.0;
    }
    prev_selected = now_selected;
  }

  // SAFETY: every cell in `arc_uninit` was initialized to 0.0 above
  // (the `slot.write(0.0)` loop covered the full slice), and the
  // subsequent `out[..] = 1.0` writes are also fully initialized
  // f32 values. `assume_init` is the canonical conversion from
  // `Arc<[MaybeUninit<f32>]>` to `Arc<[f32]>`.
  let arc_init: Arc<[f32]> = unsafe { arc_uninit.assume_init() };
  // Reference UNMATCHED so the import isn't dead code.
  let _ = UNMATCHED;
  Ok(arc_init)
}
