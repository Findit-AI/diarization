//! Bit-exact pyannote count tensor computation.
//!
//! Mirrors `pyannote.audio.pipelines.utils.diarization.SpeakerDiarizationMixin.speaker_count`,
//! which itself calls `pyannote.audio.core.inference.Inference.aggregate`
//! with the specific argument set:
//!
//! ```python
//! trimmed = Inference.trim(binarized_segmentations, warm_up=(0.1, 0.1))
//! count = Inference.aggregate(
//!     np.sum(trimmed, axis=-1, keepdims=True),
//!     frames,
//!     hamming=False,
//!     missing=0.0,
//!     skip_average=False,
//! )
//! count.data = np.rint(count.data).astype(np.uint8)
//! ```
//!
//! Algorithmic shape:
//! - **Trim**: zero out the first/last 10% of each chunk's frames
//!   (the model's warm-up zone). Those positions don't contribute.
//! - **Uniform weights** (`hamming=False`): every non-trimmed
//!   per-chunk frame contributes with weight 1.0.
//! - **Divide by overlapping chunk count** (`skip_average=False`):
//!   per output frame, the aggregate is divided by the number of
//!   *non-trimmed* per-chunk frames that contributed.
//! - **`np.rint` then `uint8` cast**: banker's rounding of the
//!   floating-point average to integer count.
//!
//! Importantly, this is NOT the same aggregation pyannote uses to
//! produce per-speaker *activations* during reconstruction — that
//! path passes `hamming=True, skip_average=True` and a different
//! warm-up. We keep [`hamming_aggregate`] in this module for that
//! distinct use case (reconstruction-side aggregation), but
//! [`count_pyannote`] does not call it.

use std::sync::Arc;

use crate::reconstruct::SlidingWindow;

/// Errors returned by the fallible (`try_*`) variants of this module.
///
/// The non-fallible counterparts ([`count_pyannote`] /
/// [`hamming_aggregate`]) panic on the same conditions. Use the
/// fallible form when shape preconditions could come from untrusted
/// input.
#[derive(Debug, thiserror::Error)]
pub enum Error {
  /// Input slice length doesn't match the declared `(num_chunks, ...)`
  /// shape product, or geometry is invalid (zero / non-finite values).
  #[error("aggregate: shape: {0}")]
  Shape(#[from] ShapeError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  #[error("num_chunks must be at least 1")]
  ZeroNumChunks,
  #[error("num_frames_per_chunk must be at least 1")]
  ZeroNumFramesPerChunk,
  #[error("num_speakers must be at least 1")]
  ZeroNumSpeakers,
  #[error("chunks_sw.duration must be a positive finite scalar")]
  InvalidChunkDuration,
  #[error("chunks_sw.step must be a positive finite scalar")]
  InvalidChunkStep,
  #[error("frames_sw_template.duration must be a positive finite scalar")]
  InvalidFrameDuration,
  #[error("frames_sw_template.step must be a positive finite scalar")]
  InvalidFrameStep,
  #[error("onset must be finite")]
  NonFiniteOnset,
  #[error("num_chunks * num_frames_per_chunk * num_speakers overflows usize")]
  CountTensorSizeOverflow,
  #[error("segmentations.len() must equal num_chunks * num_frames_per_chunk * num_speakers")]
  SegmentationsLenMismatch,
  #[error("num_chunks * num_frames_per_chunk overflows usize")]
  HammingSizeOverflow,
  #[error("per_chunk_value.len() must equal num_chunks * num_frames_per_chunk")]
  HammingPerChunkValueLenMismatch,
  #[error("chunk_step must be a positive finite scalar")]
  InvalidHammingChunkStep,
  #[error("frame_step must be a positive finite scalar")]
  InvalidHammingFrameStep,
  #[error(
    "num_frames_per_chunk must be at least 2 for hamming aggregation \
     (length-1 windows divide by zero in the hamming formula)"
  )]
  HammingNumFramesPerChunkBelowTwo,
  #[error(
    "num_output_frames overflows usize (chunk_duration / frame_step too large \
     to represent or saturated past usize::MAX)"
  )]
  OutputFrameCountOverflow,
  #[error("segmentations contains non-finite values (NaN / +inf / -inf)")]
  NonFiniteSegmentations,
  #[error("per_chunk_value contains non-finite values (NaN / +inf / -inf)")]
  NonFinitePerChunkValue,
  /// Derived hamming chunk-start frame index `(c * chunk_step /
  /// frame_step).round_ties_even() as i64` falls outside the
  /// `[i64::MIN/2, i64::MAX/2]` safety range. Adversarial-but-finite
  /// `chunk_step / frame_step` values can saturate the float-to-int
  /// cast to `i64::MAX/MIN`; the subsequent `start_frame + cf`
  /// addition then panics in debug or wraps/skips in release. Same
  /// derived-index threat shape as `reconstruct`'s timing guard.
  #[error(
    "hamming derived chunk-start frame index out of i64 safety range; \
     finite-but-extreme chunk_step / frame_step would saturate the cast"
  )]
  HammingDerivedTimingOutOfRange,
}

/// Output of [`count_pyannote`] / [`try_count_pyannote`]: the
/// per-output-frame integer count tensor plus the matching
/// `SlidingWindow`.
///
/// `count` is `Arc<[u8]>` so multiple downstream consumers can share
/// the buffer without copying it. `Arc::clone` is two atomic ops;
/// independent passes (e.g. RTTM emission + offline pipeline reuse +
/// metric computation) each get a cheap handle.
#[derive(Debug, Clone)]
pub struct CountTensor {
  count: Arc<[u8]>,
  frames_sw: SlidingWindow,
}

impl CountTensor {
  /// Cheap-clone handle to the per-output-frame count of active
  /// speakers. Length = `frames_sw`'s expansion of the input chunk
  /// grid. Each call is one `Arc::clone` (atomic refcount bump).
  pub fn count(&self) -> Arc<[u8]> {
    Arc::clone(&self.count)
  }

  /// Borrow as a slice without cloning the `Arc`.
  pub fn count_slice(&self) -> &[u8] {
    &self.count
  }

  /// Output-frame sliding window — `start = 0.0`, `duration` and
  /// `step` from the `frames_sw_template` argument.
  pub const fn frames_sw(&self) -> SlidingWindow {
    self.frames_sw
  }

  /// Consume into the inner parts.
  pub fn into_parts(self) -> (Arc<[u8]>, SlidingWindow) {
    (self.count, self.frames_sw)
  }
}

/// Hamming-weighted, skip-average aggregation across overlapping chunks.
///
/// Mirrors `pyannote.audio.core.inference.Inference.aggregate` with
/// `hamming=True, skip_average=True, warm_up=(0.0, 0.0)` —
/// **NOT** the configuration used for the count tensor (see
/// [`count_pyannote`]). This is the configuration pyannote uses
/// elsewhere (per-speaker activation aggregation during
/// reconstruction).
///
/// All durations / steps are in seconds.
///
/// - `chunk_duration`: length of each chunk window (e.g. 10.0).
/// - `chunk_step`: distance between chunk starts (e.g. 1.0).
/// - `frame_step`: stride between consecutive output frames. Pyannote
///   community-1: 0.016875 s. Note this is **NOT** the same as
///   `chunk_duration / num_frames_per_chunk`.
/// - `num_output_frames`: matches pyannote's
///   `closest_frame(last_chunk_end + 0.5 * frame_duration) + 1`.
///
/// Per-chunk values are arranged as `(num_chunks, num_frames_per_chunk)`
/// flat. Each chunk's frame `cf` accumulates into output frame
/// `start_frame_c + cf`, where `start_frame_c = round(c * chunk_step
/// / frame_step)` (numpy banker's rounding).
///
/// `skip_average = true` (pyannote convention): returns the
/// **unnormalized** hamming-weighted sum (no division by total
/// weight).
///
/// # Panics
///
/// Panics if `per_chunk_value.len() != num_chunks *
/// num_frames_per_chunk`. Use [`try_hamming_aggregate`] to surface
/// the precondition as `Result<_, Error>` instead.
pub fn hamming_aggregate(
  per_chunk_value: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  chunk_step: f64,
  frame_step: f64,
  num_output_frames: usize,
) -> Vec<f64> {
  try_hamming_aggregate(
    per_chunk_value,
    num_chunks,
    num_frames_per_chunk,
    chunk_step,
    frame_step,
    num_output_frames,
  )
  .expect("hamming_aggregate: shape precondition violated; use try_hamming_aggregate to handle")
}

/// Fallible variant of [`hamming_aggregate`]. Returns
/// [`Error::Shape`] when `per_chunk_value.len() != num_chunks *
/// num_frames_per_chunk`; otherwise identical output.
pub fn try_hamming_aggregate(
  per_chunk_value: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  chunk_step: f64,
  frame_step: f64,
  num_output_frames: usize,
) -> Result<Vec<f64>, Error> {
  // `num_frames_per_chunk == 0` underflows `(... - 1) as f64` below.
  // `num_frames_per_chunk == 1` makes `n_minus_1 == 0.0`, the hamming
  // formula divides by zero and emits a NaN window, then the
  // accumulator quietly fills `out` with NaNs and returns `Ok(_)` from
  // a fallible API. Reject both at the boundary — a hamming window
  // over a single point isn't mathematically meaningful (no edges to
  // taper) and any caller that lands here has a shape bug that should
  // fail loudly. Non-positive / non-finite step values divide into a
  // non-finite start_frame that saturates to `i64::MAX` after the cast.
  if num_frames_per_chunk < 2 {
    return Err(ShapeError::HammingNumFramesPerChunkBelowTwo.into());
  }
  if !chunk_step.is_finite() || chunk_step <= 0.0 {
    return Err(ShapeError::InvalidHammingChunkStep.into());
  }
  if !frame_step.is_finite() || frame_step <= 0.0 {
    return Err(ShapeError::InvalidHammingFrameStep.into());
  }
  let expected = num_chunks
    .checked_mul(num_frames_per_chunk)
    .ok_or(ShapeError::HammingSizeOverflow)?;
  if per_chunk_value.len() != expected {
    return Err(ShapeError::HammingPerChunkValueLenMismatch.into());
  }
  // Reject non-finite input up front. Without this, NaN cells flow
  // through the multiply-add accumulator and the function returns
  // `Ok(Vec<NaN>)` from a fallible API — silent numeric corruption.
  // Mirrors the policy in `try_count_pyannote`.
  for &v in per_chunk_value {
    if !v.is_finite() {
      return Err(ShapeError::NonFinitePerChunkValue.into());
    }
  }
  // Validate the derived chunk-start frame index for both endpoints
  // (c=0 and c=num_chunks-1). The inner loop computes
  //   start_frame = (c * chunk_step / frame_step).round_ties_even() as i64
  //   ofr         = start_frame + cf
  // For finite-but-adversarial `chunk_step / frame_step`, the
  // float-to-int cast saturates to `i64::MAX/MIN`, after which
  // `start_frame + cf` panics in debug or wraps/skips in release.
  // Same threat shape as the reconstruct derived-timing guard;
  // bound the index well within `i64` so the addition is always safe.
  // The `c=0` endpoint is trivially `0 / step = 0`, but we check it
  // for symmetry and to catch a future code change that lets `c=0`
  // pull in a non-zero offset.
  let safe_lo = -(i64::MAX / 2) as f64;
  let safe_hi = (i64::MAX / 2) as f64;
  // First chunk: c = 0 → chunk_start_t = 0. normalized = 0 always.
  // Last chunk: c = num_chunks - 1.
  if num_chunks > 0 {
    let last_chunk_start_t = (num_chunks as f64 - 1.0) * chunk_step;
    if !last_chunk_start_t.is_finite() {
      return Err(ShapeError::HammingDerivedTimingOutOfRange.into());
    }
    let last_normalized = last_chunk_start_t / frame_step;
    if !last_normalized.is_finite() || !(safe_lo..=safe_hi).contains(&last_normalized) {
      return Err(ShapeError::HammingDerivedTimingOutOfRange.into());
    }
  }
  let mut out = vec![0.0_f64; num_output_frames];
  let n_minus_1 = (num_frames_per_chunk - 1) as f64;
  let hamming: Vec<f64> = (0..num_frames_per_chunk)
    .map(|n| 0.54 - 0.46 * (std::f64::consts::TAU * n as f64 / n_minus_1).cos())
    .collect();
  for c in 0..num_chunks {
    let chunk_start_t = c as f64 * chunk_step;
    let start_frame = (chunk_start_t / frame_step).round_ties_even() as i64;
    for cf in 0..num_frames_per_chunk {
      let ofr = start_frame + cf as i64;
      if ofr < 0 || (ofr as usize) >= num_output_frames {
        continue;
      }
      out[ofr as usize] += per_chunk_value[c * num_frames_per_chunk + cf] * hamming[cf];
    }
  }
  Ok(out)
}

/// Compute pyannote's exact `num_output_frames` for the given
/// chunking + output-frame timing parameters.
///
/// Pyannote 4.0.4 `Inference.aggregate` (verbatim, eliding obvious
/// substitutions):
/// ```text
/// last_chunk_end = chunks.start + chunks.duration + (num_chunks - 1) * chunks.step
/// num_frames     = frames.closest_frame(last_chunk_end + 0.5 * frames.duration) + 1
/// ```
/// where `closest_frame(t) = round((t - frames.start - 0.5 *
/// frames.duration) / frames.step)`. The `+0.5 * frames.duration` in
/// the call CANCELS the `-0.5 * frames.duration` inside
/// `closest_frame`, leaving `round(last_chunk_end / frames.step) + 1`
/// (with `frames.start = 0`).
///
/// Both `chunks.start` and `frames.start` are 0 in the community-1
/// pipeline.
///
/// # Panics
///
/// Panics if `num_chunks == 0` (subtraction overflow), if `frame_step
/// <= 0.0` (the divide produces a non-finite length), or if the
/// resulting frame count overflows `usize`. Callers should validate
/// inputs before calling — [`try_num_output_frames_pyannote`] surfaces
/// these as `Result<usize, ShapeError>` instead, and
/// [`try_count_pyannote`] uses the checked form at its boundary.
pub fn num_output_frames_pyannote(
  num_chunks: usize,
  chunk_duration: f64,
  chunk_step: f64,
  frame_step: f64,
) -> usize {
  try_num_output_frames_pyannote(num_chunks, chunk_duration, chunk_step, frame_step)
    .expect("num_output_frames_pyannote: precondition violated; use try_num_output_frames_pyannote to handle")
}

/// Fallible variant of [`num_output_frames_pyannote`]. Validates that
/// the geometry produces a finite, in-range output frame count.
///
/// # Errors
///
/// - [`ShapeError::ZeroNumChunks`] if `num_chunks == 0`.
/// - [`ShapeError::InvalidFrameStep`] if `frame_step` is not a positive
///   finite scalar.
/// - [`ShapeError::OutputFrameCountOverflow`] if `chunk_duration /
///   frame_step` is non-finite, negative, or rounds to a value that
///   does not fit in `usize` (or whose `+1` would overflow). Catches
///   pathological geometries like `chunk_duration = 1e15` with
///   `frame_step = 1e-15`, where the float division stays finite but
///   saturates `as usize` to `usize::MAX`.
pub fn try_num_output_frames_pyannote(
  num_chunks: usize,
  chunk_duration: f64,
  chunk_step: f64,
  frame_step: f64,
) -> Result<usize, ShapeError> {
  if num_chunks == 0 {
    return Err(ShapeError::ZeroNumChunks);
  }
  if !frame_step.is_finite() || frame_step <= 0.0 {
    return Err(ShapeError::InvalidFrameStep);
  }
  let last_chunk_end = chunk_duration + (num_chunks - 1) as f64 * chunk_step;
  let frames_f = (last_chunk_end / frame_step).round_ties_even();
  // Reject NaN/±inf and any value that would saturate `as usize` or
  // overflow the `+ 1`. `usize::MAX as f64` is exactly representable
  // (it's a power-of-two minus one rounded up to the nearest f64), so
  // this comparison is monotonic.
  if !frames_f.is_finite() || frames_f < 0.0 || frames_f >= usize::MAX as f64 {
    return Err(ShapeError::OutputFrameCountOverflow);
  }
  (frames_f as usize)
    .checked_add(1)
    .ok_or(ShapeError::OutputFrameCountOverflow)
}

/// Bit-exact pyannote `speaker_count`. Returns the per-output-frame
/// integer count of active speakers, ready to feed into
/// [`reconstruct`](crate::reconstruct::reconstruct).
///
/// Implements (verbatim from pyannote 4.0.4):
/// ```text
/// trimmed = trim(binarized, warm_up=(0.1, 0.1))         # NaN-mask
/// count = aggregate(sum(trimmed, axis=speaker),         # per-chunk integer count
///                   hamming=False,                       # uniform weights
///                   skip_average=False,                  # divide by overlapping count
///                   missing=0.0)                          # NaN cells → 0
/// count = np.rint(count).astype(np.uint8)
/// ```
///
/// `segmentations`: `(num_chunks, num_frames_per_chunk, num_speakers)`
/// flattened row-major in the [c][f][s] order pyannote uses.
///
/// Returns a [`CountTensor`] holding the per-output-frame count and
/// the matching `SlidingWindow`. `chunks_sw` describes the input
/// chunk grid (`duration` = chunk_duration, `step` = chunk_step).
/// `frames_sw_template` describes the output frame grid (`duration`
/// and `step`); its `start` is ignored — the returned `SlidingWindow`
/// always starts at 0.0 to match pyannote's convention.
///
/// # Panics
///
/// Panics if `segmentations.len() != num_chunks * num_frames_per_chunk
/// * num_speakers`. Use [`try_count_pyannote`] to surface the
/// precondition as `Result<_, Error>` instead.
pub fn count_pyannote(
  segmentations: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  onset: f64,
  chunks_sw: SlidingWindow,
  frames_sw_template: SlidingWindow,
) -> CountTensor {
  try_count_pyannote(
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    onset,
    chunks_sw,
    frames_sw_template,
  )
  .expect("count_pyannote: shape precondition violated; use try_count_pyannote to handle")
}

/// Fallible variant of [`count_pyannote`]. Returns [`Error::Shape`]
/// when `segmentations.len() != num_chunks * num_frames_per_chunk *
/// num_speakers` (or when that product overflows `usize`); otherwise
/// identical output.
pub fn try_count_pyannote(
  segmentations: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  onset: f64,
  chunks_sw: SlidingWindow,
  frames_sw_template: SlidingWindow,
) -> Result<CountTensor, Error> {
  // Reject empty / non-positive geometry up front. `num_chunks == 0`
  // would underflow `(num_chunks - 1) as f64` in
  // `num_output_frames_pyannote` and drive `aggregated`'s allocation
  // toward `usize::MAX`. `frame_step <= 0` divides into a non-finite
  // length that saturates the same allocation. `num_frames_per_chunk
  // == 0` and `num_speakers == 0` are technically fillable but produce
  // semantically meaningless empty outputs, so refuse them too.
  if num_chunks == 0 {
    return Err(ShapeError::ZeroNumChunks.into());
  }
  if num_frames_per_chunk == 0 {
    return Err(ShapeError::ZeroNumFramesPerChunk.into());
  }
  if num_speakers == 0 {
    return Err(ShapeError::ZeroNumSpeakers.into());
  }
  let chunk_duration = chunks_sw.duration();
  let chunk_step = chunks_sw.step();
  let frame_duration = frames_sw_template.duration();
  let frame_step = frames_sw_template.step();
  if !chunk_duration.is_finite() || chunk_duration <= 0.0 {
    return Err(ShapeError::InvalidChunkDuration.into());
  }
  if !chunk_step.is_finite() || chunk_step <= 0.0 {
    return Err(ShapeError::InvalidChunkStep.into());
  }
  if !frame_duration.is_finite() || frame_duration <= 0.0 {
    return Err(ShapeError::InvalidFrameDuration.into());
  }
  if !frame_step.is_finite() || frame_step <= 0.0 {
    return Err(ShapeError::InvalidFrameStep.into());
  }
  if !onset.is_finite() {
    return Err(ShapeError::NonFiniteOnset.into());
  }
  let expected = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(ShapeError::CountTensorSizeOverflow)?;
  if segmentations.len() != expected {
    return Err(ShapeError::SegmentationsLenMismatch.into());
  }
  // Reject non-finite segmentation values up front. The threshold
  // comparison `v >= onset` is asymmetric on non-finite inputs: NaN
  // compares false, -inf compares false (against a finite onset),
  // +inf compares true. A degraded segmentation backend producing
  // NaN/inf cells would silently fold into a finite-looking count
  // tensor, hiding the bad input from downstream reconstruct's
  // top-K logic. Same policy as
  // `crate::reconstruct::reconstruct`'s segmentation finite check.
  for &v in segmentations {
    if !v.is_finite() {
      return Err(ShapeError::NonFiniteSegmentations.into());
    }
  }

  // ── 1. Per-(chunk, frame) integer count of active speakers ─────
  //
  // SIMD-friendly form. The input layout is `[c][f][s]` (speakers
  // innermost), so per-frame counting strides by `num_speakers` —
  // typically 3, which is too narrow for vector loads. We rewrite as
  // an outer per-speaker accumulation: for each (chunk, speaker),
  // scan all frames contiguously, threshold-compare to onset, add
  // 0 or 1 to the per-frame count slot. Each per-speaker pass over
  // a chunk is a `num_frames_per_chunk`-long contiguous scan over
  // f64 with a strided gather — large enough (≥ 200) for the
  // compiler to autovectorize the threshold-cmp + add to NEON
  // `vcgeq_f64` + `vaddq_f64` and AVX2 `_mm256_cmp_pd` +
  // `_mm256_add_pd`. The branch (`if seg >= onset`) is rewritten
  // branchless as `(seg >= onset) as f64`-style SELECT for the same
  // reason. Verified by `aggregate::parity_tests` (bit-exact match
  // to pyannote's captured count tensor on all 6 fixtures, 0%
  // mismatch tolerance).
  let mut chunk_count: Vec<f64> = vec![0.0; num_chunks * num_frames_per_chunk];
  for c in 0..num_chunks {
    let chunk_count_row =
      &mut chunk_count[c * num_frames_per_chunk..(c + 1) * num_frames_per_chunk];
    for s in 0..num_speakers {
      let seg_base = c * num_frames_per_chunk * num_speakers + s;
      let stride = num_speakers;
      for (f, slot) in chunk_count_row.iter_mut().enumerate() {
        let v = segmentations[seg_base + f * stride];
        // Branchless threshold-add. Compiles to `vbsl_f64` (NEON)
        // or `_mm256_blendv_pd` (AVX2) — bit-identical to the
        // `if v >= onset { 1.0 } else { 0.0 }` form.
        let active = if v >= onset { 1.0_f64 } else { 0.0_f64 };
        *slot += active;
      }
    }
  }

  // ── 2. Trim warm-up zone ───────────────────────────────────────
  //
  // Pyannote 4.0.4 community-1 calls `speaker_count` with
  // `warm_up=(0.0, 0.0)` (see
  // `pyannote/audio/pipelines/speaker_diarization.py:611`), even
  // though `speaker_count`'s default is `(0.1, 0.1)`. So no trim
  // is applied on the community-1 path. We keep the structure
  // here in case a future caller wants to pass non-zero warm-up,
  // but parameterize it through an explicit argument; for now
  // the count-tensor path is fixed at zero warm-up.
  //
  // (If we ever need to expose this, surface a `warm_up: (f64, f64)`
  // arg and parameterize the active_frame mask.)
  let active_frame: Vec<bool> = vec![true; num_frames_per_chunk];

  // ── 3. Per-chunk start_frame ───────────────────────────────────
  // start_frame = closest_frame(chunk.start + 0.5 * frame_duration)
  //             = round((chunk.start + 0.5 * frame_duration - 0.5 * frame_duration) / frame_step)
  //             = round(chunk.start / frame_step)
  // (with frames.start = 0; the two 0.5 * frame_duration cancel.)
  let _ = frame_duration; // referenced in docs; cancels analytically here.
  // Use the checked variant so a pathological geometry (e.g. enormous
  // `chunk_duration` with tiny `frame_step`) surfaces as a typed
  // `ShapeError` instead of a saturating `as usize` cast that would
  // either OOM the `aggregated` Vec or overflow `+ 1` to wrap to zero.
  let num_output_frames =
    try_num_output_frames_pyannote(num_chunks, chunk_duration, chunk_step, frame_step)?;

  // ── 4. Aggregate (uniform weights, divide by overlapping count) ─
  let mut aggregated = vec![0.0_f64; num_output_frames];
  let mut overlapping_count = vec![0.0_f64; num_output_frames];
  for c in 0..num_chunks {
    let chunk_start_t = c as f64 * chunk_step;
    let start_frame = (chunk_start_t / frame_step).round_ties_even() as i64;
    for cf in 0..num_frames_per_chunk {
      if !active_frame[cf] {
        continue;
      }
      let ofr = start_frame + cf as i64;
      if ofr < 0 || (ofr as usize) >= num_output_frames {
        continue;
      }
      let ofr = ofr as usize;
      aggregated[ofr] += chunk_count[c * num_frames_per_chunk + cf];
      overlapping_count[ofr] += 1.0;
    }
  }

  // ── 5. count[t] = round(aggregated[t] / overlapping_count[t]) ──
  // Pyannote uses `np.maximum(overlapping_count, epsilon)` with
  // epsilon = 1e-12 to avoid divide-by-zero, then for cells where
  // `aggregated_mask == 0` (no contributing chunks), it injects
  // `missing=0.0`. Effectively: count is 0 where no chunk
  // contributed, else `np.rint(aggregated / overlapping_count)`.
  //
  // Build `Arc<[u8]>` directly via the trusted-len iterator collect:
  // `Range<usize>::map` preserves `TrustedLen`, so std's
  // specialized `<Arc<[T]> as FromIterator<T>>::from_iter` allocates
  // the `Arc` once and writes each element in place — no
  // `Vec`-then-`Arc` round-trip. Callers fan-out via cheap
  // `Arc::clone` (refcount bump).
  let epsilon = 1e-12_f64;
  let count: Arc<[u8]> = (0..num_output_frames)
    .map(|t| {
      if overlapping_count[t] > 0.0 {
        let avg = aggregated[t] / overlapping_count[t].max(epsilon);
        avg.round_ties_even().clamp(0.0, u8::MAX as f64) as u8
      } else {
        0
      }
    })
    .collect();

  let frames_sw = SlidingWindow::new(0.0, frame_duration, frame_step);

  Ok(CountTensor { count, frames_sw })
}

#[cfg(test)]
mod try_variant_tests {
  use super::*;

  fn sw(duration: f64, step: f64) -> SlidingWindow {
    SlidingWindow::new(0.0, duration, step)
  }

  #[test]
  fn try_count_pyannote_rejects_short_segmentations() {
    // Declared shape is 3 chunks * 4 frames * 2 speakers = 24 elements.
    let segs: Vec<f64> = vec![0.0; 23];
    let r = try_count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  fn try_count_pyannote_rejects_overflow() {
    // num_chunks * num_frames_per_chunk * num_speakers overflows usize.
    let segs: Vec<f64> = vec![0.0; 0];
    let r = try_count_pyannote(
      &segs,
      1 << 30,
      1 << 30,
      1 << 30,
      0.5,
      sw(10.0, 1.0),
      sw(0.062, 0.0169),
    );
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  /// `num_chunks == 0` would underflow `(num_chunks - 1) as f64` in
  /// `num_output_frames_pyannote` and saturate the `aggregated`
  /// allocation to `usize::MAX` in release builds.
  #[test]
  fn try_count_pyannote_rejects_zero_num_chunks() {
    let r = try_count_pyannote(&[], 0, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  fn try_count_pyannote_rejects_zero_num_frames_per_chunk() {
    let r = try_count_pyannote(&[], 3, 0, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  fn try_count_pyannote_rejects_zero_num_speakers() {
    let r = try_count_pyannote(&[], 3, 4, 0, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  /// `frame_step == 0` divides into a non-finite output-frame count.
  #[test]
  fn try_count_pyannote_rejects_zero_frame_step() {
    let segs: Vec<f64> = vec![0.0; 24];
    let r = try_count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  fn try_count_pyannote_rejects_negative_frame_step() {
    let segs: Vec<f64> = vec![0.0; 24];
    let r = try_count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, -0.0169));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  fn try_count_pyannote_rejects_non_finite_onset() {
    let segs: Vec<f64> = vec![0.0; 24];
    let r = try_count_pyannote(&segs, 3, 4, 2, f64::NAN, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  fn try_count_pyannote_rejects_non_finite_chunk_duration() {
    let segs: Vec<f64> = vec![0.0; 24];
    let r = try_count_pyannote(
      &segs,
      3,
      4,
      2,
      0.5,
      sw(f64::INFINITY, 1.0),
      sw(0.062, 0.0169),
    );
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  /// Pathological-but-finite geometry: enormous `chunk_duration` with
  /// tiny `frame_step`. The intermediate float division stays finite,
  /// but `as usize` saturates to `usize::MAX`, then `+ 1` would either
  /// panic in checked builds or wrap to 0 in release. The checked
  /// helper must reject this with a typed `OutputFrameCountOverflow`
  /// instead of OOMing the downstream Vec or producing junk output.
  #[test]
  fn try_num_output_frames_pyannote_rejects_overflow_geometry() {
    let r = try_num_output_frames_pyannote(1, 1.0e15, 1.0, 1.0e-15);
    assert!(
      matches!(r, Err(ShapeError::OutputFrameCountOverflow)),
      "got {r:?}"
    );
  }

  #[test]
  fn try_count_pyannote_rejects_overflow_geometry() {
    // 1 chunk, 4 frames, 2 speakers → segs len 8. `chunk_duration =
    // 1e15`, `frame_step = 1e-15` makes num_output_frames overflow.
    let segs: Vec<f64> = vec![0.0; 8];
    let r = try_count_pyannote(&segs, 1, 4, 2, 0.5, sw(1.0e15, 1.0), sw(0.062, 1.0e-15));
    assert!(
      matches!(r, Err(Error::Shape(ShapeError::OutputFrameCountOverflow))),
      "got {r:?}"
    );
  }

  /// `try_num_output_frames_pyannote` rejects bad inputs without
  /// panicking. Mirrors the panic contract of `num_output_frames_pyannote`
  /// but as `Result<_, ShapeError>`.
  #[test]
  fn try_num_output_frames_pyannote_rejects_zero_num_chunks() {
    let r = try_num_output_frames_pyannote(0, 10.0, 1.0, 0.0169);
    assert!(matches!(r, Err(ShapeError::ZeroNumChunks)), "got {r:?}");
  }

  #[test]
  fn try_num_output_frames_pyannote_rejects_zero_frame_step() {
    let r = try_num_output_frames_pyannote(3, 10.0, 1.0, 0.0);
    assert!(matches!(r, Err(ShapeError::InvalidFrameStep)), "got {r:?}");
  }

  #[test]
  fn try_hamming_aggregate_rejects_zero_num_frames_per_chunk() {
    let r = try_hamming_aggregate(&[], 3, 0, 1.0, 0.0169, 8);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::HammingNumFramesPerChunkBelowTwo))
      ),
      "got {r:?}"
    );
  }

  /// `num_frames_per_chunk == 1` makes the hamming formula divide by
  /// zero (`n_minus_1 == 0.0`); previously this returned `Ok(Vec<NaN>)`
  /// from a fallible API. Now rejected at the boundary.
  #[test]
  fn try_hamming_aggregate_rejects_single_frame_chunk() {
    let r = try_hamming_aggregate(&[0.5; 3], 3, 1, 1.0, 0.0169, 8);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::HammingNumFramesPerChunkBelowTwo))
      ),
      "got {r:?}"
    );
    // Belt-and-suspenders: even if the variant changes shape, the
    // output must never contain NaN for accepted input.
    if let Ok(v) = &r {
      assert!(
        v.iter().all(|x| !x.is_nan()),
        "hamming aggregate emitted NaN for 1-frame chunk: {v:?}"
      );
    }
  }

  #[test]
  fn try_hamming_aggregate_rejects_zero_frame_step() {
    let r = try_hamming_aggregate(&[0.0; 12], 3, 4, 1.0, 0.0, 8);
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  /// Threshold comparison `v >= onset` is asymmetric on non-finite
  /// inputs (NaN false, -inf false against finite onset, +inf true),
  /// so a degraded segmentation backend producing NaN/inf cells could
  /// silently fold into a finite-looking count tensor. The fallible
  /// API must reject the bad input up front instead.
  #[test]
  fn try_count_pyannote_rejects_nan_segmentation() {
    let mut segs: Vec<f64> = vec![0.5; 24];
    segs[7] = f64::NAN;
    let r = try_count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(
      matches!(r, Err(Error::Shape(ShapeError::NonFiniteSegmentations))),
      "got {r:?}"
    );
  }

  #[test]
  fn try_count_pyannote_rejects_pos_inf_segmentation() {
    let mut segs: Vec<f64> = vec![0.5; 24];
    segs[0] = f64::INFINITY;
    let r = try_count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(
      matches!(r, Err(Error::Shape(ShapeError::NonFiniteSegmentations))),
      "got {r:?}"
    );
  }

  #[test]
  fn try_count_pyannote_rejects_neg_inf_segmentation() {
    let mut segs: Vec<f64> = vec![0.5; 24];
    segs[15] = f64::NEG_INFINITY;
    let r = try_count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
    assert!(
      matches!(r, Err(Error::Shape(ShapeError::NonFiniteSegmentations))),
      "got {r:?}"
    );
  }

  /// `try_hamming_aggregate` has the same class of issue: a NaN cell
  /// in `per_chunk_value` flows through the multiply-add accumulator
  /// and the function returns `Ok(Vec<NaN>)` from a fallible API.
  #[test]
  fn try_hamming_aggregate_rejects_nan_per_chunk_value() {
    let mut vals: Vec<f64> = vec![0.5; 12];
    vals[5] = f64::NAN;
    let r = try_hamming_aggregate(&vals, 3, 4, 1.0, 0.0169, 8);
    assert!(
      matches!(r, Err(Error::Shape(ShapeError::NonFinitePerChunkValue))),
      "got {r:?}"
    );
  }

  #[test]
  fn try_hamming_aggregate_rejects_inf_per_chunk_value() {
    let mut vals: Vec<f64> = vec![0.5; 12];
    vals[0] = f64::INFINITY;
    let r = try_hamming_aggregate(&vals, 3, 4, 1.0, 0.0169, 8);
    assert!(
      matches!(r, Err(Error::Shape(ShapeError::NonFinitePerChunkValue))),
      "got {r:?}"
    );
  }

  #[test]
  #[should_panic(expected = "shape precondition violated")]
  fn count_pyannote_panics_on_short_input() {
    let segs: Vec<f64> = vec![0.0; 23];
    let _ = count_pyannote(&segs, 3, 4, 2, 0.5, sw(10.0, 1.0), sw(0.062, 0.0169));
  }

  #[test]
  fn try_hamming_aggregate_rejects_short_input() {
    let r = try_hamming_aggregate(&[0.0; 7], 3, 4, 1.0, 0.0169, 100);
    assert!(matches!(r, Err(Error::Shape(_))), "got {r:?}");
  }

  #[test]
  #[should_panic(expected = "shape precondition violated")]
  fn hamming_aggregate_panics_on_short_input() {
    let _ = hamming_aggregate(&[0.0; 7], 3, 4, 1.0, 0.0169, 100);
  }

  /// Round-20 [medium]: derived chunk-start frame index must be
  /// bounded within `[i64::MIN/2, i64::MAX/2]`. With finite-but-
  /// adversarial `chunk_step / frame_step`, the float-to-int cast
  /// `as i64` saturates, after which `start_frame + cf` panics in
  /// debug or wraps/skips in release. Same threat shape as the
  /// reconstruct derived-timing guard.
  #[test]
  fn try_hamming_aggregate_rejects_extreme_chunk_step_to_frame_step_ratio() {
    // chunk_step = f64::MAX, frame_step = 1.0 → last chunk normalized
    // = (num_chunks - 1) * f64::MAX → +inf or way past i64::MAX/2.
    let per_chunk = vec![1.0_f64; 2 * 4]; // 2 chunks, 4 frames/chunk.
    let r = try_hamming_aggregate(&per_chunk, 2, 4, f64::MAX, 1.0, 16);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::HammingDerivedTimingOutOfRange))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn try_hamming_aggregate_rejects_tiny_frame_step_makes_normalized_overflow_i64() {
    // chunk_step = 1e150, frame_step = 1e-150. Their ratio = 1e300,
    // multiplied by (num_chunks-1) overflows i64 safety bound.
    let per_chunk = vec![1.0_f64; 2 * 4];
    let r = try_hamming_aggregate(&per_chunk, 2, 4, 1e150, 1e-150, 16);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::HammingDerivedTimingOutOfRange))
      ),
      "got {r:?}"
    );
  }
}
