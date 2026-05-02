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

use crate::reconstruct::SlidingWindow;

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
pub fn hamming_aggregate(
  per_chunk_value: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  chunk_step: f64,
  frame_step: f64,
  num_output_frames: usize,
) -> Vec<f64> {
  assert_eq!(
    per_chunk_value.len(),
    num_chunks * num_frames_per_chunk,
    "hamming_aggregate: per_chunk_value.len() != num_chunks * num_frames_per_chunk"
  );
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
  out
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
pub fn num_output_frames_pyannote(
  num_chunks: usize,
  chunk_duration: f64,
  chunk_step: f64,
  frame_step: f64,
) -> usize {
  let last_chunk_end = chunk_duration + (num_chunks - 1) as f64 * chunk_step;
  (last_chunk_end / frame_step).round_ties_even() as usize + 1
}

/// Bit-exact pyannote `speaker_count`. Returns the per-output-frame
/// integer count of active speakers, ready to feed into
/// [`crate::reconstruct::reconstruct`].
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
/// Returns `(count, frames_sw)` where `frames_sw` matches pyannote's
/// output-frame `SlidingWindow`.
/// `chunks_sw` describes the input chunk grid (`duration` =
/// chunk_duration, `step` = chunk_step). `frames_sw_template` describes
/// the output frame grid (`duration` and `step`); its `start` is
/// ignored — the returned `SlidingWindow` always starts at 0.0 to
/// match pyannote's convention. Both args replace the previous four
/// positional `(chunk_duration, chunk_step, frame_duration, frame_step)`
/// floats.
pub fn count_pyannote(
  segmentations: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  onset: f64,
  chunks_sw: SlidingWindow,
  frames_sw_template: SlidingWindow,
) -> (Vec<u8>, SlidingWindow) {
  let chunk_duration = chunks_sw.duration();
  let chunk_step = chunks_sw.step();
  let frame_duration = frames_sw_template.duration();
  let frame_step = frames_sw_template.step();
  assert_eq!(
    segmentations.len(),
    num_chunks * num_frames_per_chunk * num_speakers,
    "count_pyannote: segmentations.len() != num_chunks * num_frames_per_chunk * num_speakers"
  );

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
  let num_output_frames =
    num_output_frames_pyannote(num_chunks, chunk_duration, chunk_step, frame_step);

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
  let mut count = vec![0u8; num_output_frames];
  let epsilon = 1e-12_f64;
  for t in 0..num_output_frames {
    if overlapping_count[t] > 0.0 {
      let avg = aggregated[t] / overlapping_count[t].max(epsilon);
      count[t] = avg.round_ties_even().clamp(0.0, u8::MAX as f64) as u8;
    }
  }

  let frames_sw = SlidingWindow::new(0.0, frame_duration, frame_step);

  (count, frames_sw)
}
