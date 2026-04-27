//! `exclude_overlap` mask construction. Spec §5.8.
//!
//! Per-window per-frame raw probabilities → binarized per-(slot, frame)
//! activity → per-frame "clean" mask (only this slot active) → sample-
//! rate `keep_mask` (frame-aligned via [`crate::segment::stitch::frame_to_sample`]).
//!
//! The output `keep_mask` feeds [`crate::embed::EmbedModel::embed_masked`]
//! to extract a per-speaker embedding without overlap contamination.

use crate::{
  embed::MIN_CLIP_SAMPLES,
  segment::{
    options::{FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS},
    stitch::frame_to_sample,
  },
};

/// Per-(slot, frame) binarized speaker activity for a single window.
///
/// `binarized[slot][frame] = raw_probs[slot][frame] > threshold`. Strict
/// `>` matches spec §5.8 / pyannote's `Binarize(onset=...)` rule.
pub(crate) fn binarize_per_frame(
  raw_probs: &[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
  threshold: f32,
) -> [[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize] {
  let mut out = [[false; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
  for s in 0..MAX_SPEAKER_SLOTS as usize {
    for f in 0..FRAMES_PER_WINDOW {
      out[s][f] = raw_probs[s][f] > threshold;
    }
  }
  out
}

/// Per-frame "n_active" — number of slots active at each frame
/// (sum across slots of `binarized[slot][frame]`).
pub(crate) fn count_active_per_frame(
  binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
) -> [u8; FRAMES_PER_WINDOW] {
  let mut out = [0u8; FRAMES_PER_WINDOW];
  for (f, slot_count) in out.iter_mut().enumerate() {
    let mut n = 0u8;
    for slot_row in binarized.iter() {
      if slot_row[f] {
        n += 1;
      }
    }
    *slot_count = n;
  }
  out
}

/// Per-frame "speaker mask" for slot `s`: `true` iff slot `s` is active
/// at that frame. Equivalent to `binarized[s]` but copied as
/// `[bool; FRAMES_PER_WINDOW]` for downstream uniformity.
pub(crate) fn speaker_mask_for_slot(
  binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
  slot: u8,
) -> [bool; FRAMES_PER_WINDOW] {
  binarized[slot as usize]
}

/// Per-frame "clean mask" for slot `s`: `true` iff slot `s` is active
/// at that frame AND no other slot is active there.
pub(crate) fn clean_mask_for_slot(
  binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
  n_active: &[u8; FRAMES_PER_WINDOW],
  slot: u8,
) -> [bool; FRAMES_PER_WINDOW] {
  let mut out = [false; FRAMES_PER_WINDOW];
  for f in 0..FRAMES_PER_WINDOW {
    out[f] = binarized[slot as usize][f] && n_active[f] == 1;
  }
  out
}

/// Outcome of the §5.8 mask decision: which mask is used + the
/// sample-rate keep_mask aligned with the activity range.
#[allow(dead_code)] // fields read by the Phase 11 pump (Task 43)
pub(crate) struct MaskDecision {
  /// Sample-rate keep_mask. `keep_mask.len() == s1 - s0`.
  pub(crate) keep_mask: Vec<bool>,
  /// `true` iff the clean (overlap-excluded) mask is being used;
  /// `false` if we fell back to the speaker-only mask.
  pub(crate) used_clean: bool,
}

/// Expand a per-frame mask to a per-sample mask, aligned with an
/// activity's absolute-sample range `[s0, s1)`.
///
/// Frame f covers sample range
/// `[window_start + frame_to_sample(f), window_start + frame_to_sample(f + 1))`.
/// Each sample in the activity range whose containing frame has
/// `frame_mask[f] == true` is set to `true` in the output.
#[allow(dead_code)] // wired into the Phase 11 pump (Task 43)
pub(crate) fn frame_mask_to_sample_keep_mask(
  frame_mask: &[bool; FRAMES_PER_WINDOW],
  window_start: u64,
  s0: u64,
  s1: u64,
) -> Vec<bool> {
  debug_assert!(s1 >= s0, "frame_mask_to_sample_keep_mask: s1 < s0");
  let activity_len = (s1 - s0) as usize;
  let mut keep = vec![false; activity_len];
  for (f, &active) in frame_mask.iter().enumerate() {
    if !active {
      continue;
    }
    let frame_s_start = window_start + frame_to_sample(f as u32) as u64;
    let frame_s_end = window_start + frame_to_sample((f + 1) as u32) as u64;
    let lo = frame_s_start.max(s0);
    let hi = frame_s_end.min(s1);
    if lo >= hi {
      continue;
    }
    let lo_in = (lo - s0) as usize;
    let hi_in = (hi - s0) as usize;
    keep[lo_in..hi_in].fill(true);
  }
  keep
}

/// Decide between the clean (overlap-excluded) mask and the speaker-only
/// mask, then build the sample-rate keep_mask. Returns the chosen mask
/// + a flag indicating which was used.
///
/// **Decision rule (spec §5.8):** prefer the clean mask if its
/// gathered-sample count >= `MIN_CLIP_SAMPLES`. Otherwise fall back to
/// the speaker-only mask. The third-tier fallback (skip the activity if
/// `embed_masked` returns `InvalidClip` even on the speaker-only mask,
/// matching pyannote `speaker_verification.py:611-612`) is handled in
/// Phase 11's pump (Task 43), not here.
#[allow(dead_code)] // wired into the Phase 11 pump (Task 43)
pub(crate) fn decide_keep_mask(
  raw_probs: &[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
  binarize_threshold: f32,
  slot: u8,
  window_start: u64,
  s0: u64,
  s1: u64,
) -> MaskDecision {
  let binarized = binarize_per_frame(raw_probs, binarize_threshold);
  let n_active = count_active_per_frame(&binarized);
  let speaker = speaker_mask_for_slot(&binarized, slot);
  let clean = clean_mask_for_slot(&binarized, &n_active, slot);

  let speaker_keep = frame_mask_to_sample_keep_mask(&speaker, window_start, s0, s1);
  let clean_keep = frame_mask_to_sample_keep_mask(&clean, window_start, s0, s1);

  let clean_count = clean_keep.iter().filter(|&&b| b).count();
  if clean_count >= MIN_CLIP_SAMPLES as usize {
    MaskDecision {
      keep_mask: clean_keep,
      used_clean: true,
    }
  } else {
    MaskDecision {
      keep_mask: speaker_keep,
      used_clean: false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn empty_probs() -> [[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize] {
    [[0.0; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize]
  }

  #[test]
  fn binarize_threshold_strict_greater_than() {
    let mut p = empty_probs();
    p[0][10] = 0.6; // > 0.5 → true
    p[0][20] = 0.4; // < 0.5 → false
    p[0][25] = 0.5; // == 0.5 → false (strict >)
    p[1][30] = 1.0; // > 0.5 → true
    let b = binarize_per_frame(&p, 0.5);
    assert!(b[0][10]);
    assert!(!b[0][20]);
    assert!(!b[0][25], "strict > means 0.5 is NOT binarized to true");
    assert!(b[1][30]);
  }

  #[test]
  fn count_active_per_frame_sums_slots() {
    let mut p = empty_probs();
    // Frame 100: slots 0 + 2 active. Frame 200: only slot 1.
    p[0][100] = 0.9;
    p[2][100] = 0.9;
    p[1][200] = 0.9;
    let b = binarize_per_frame(&p, 0.5);
    let n = count_active_per_frame(&b);
    assert_eq!(n[100], 2);
    assert_eq!(n[200], 1);
    assert_eq!(n[0], 0);
  }

  #[test]
  fn clean_mask_excludes_overlap_frames() {
    let mut p = empty_probs();
    // Frame 100: slot 0 alone. Frame 200: slot 0 + slot 1 (overlap).
    p[0][100] = 0.9;
    p[0][200] = 0.9;
    p[1][200] = 0.9;
    let b = binarize_per_frame(&p, 0.5);
    let n = count_active_per_frame(&b);
    let clean = clean_mask_for_slot(&b, &n, 0);
    assert!(clean[100], "frame 100: slot 0 alone → clean");
    assert!(!clean[200], "frame 200: slot 0 + slot 1 → not clean");
  }
}

#[cfg(test)]
mod expand_tests {
  use super::*;
  use crate::segment::options::WINDOW_SAMPLES;

  #[test]
  fn expand_full_window_mask_covers_full_range() {
    let mask = [true; FRAMES_PER_WINDOW];
    let keep = frame_mask_to_sample_keep_mask(&mask, 0, 0, WINDOW_SAMPLES as u64);
    let true_count = keep.iter().filter(|&&b| b).count();
    // Frames cover ≈ all of [0, WINDOW_SAMPLES) modulo frame-to-sample
    // rounding at the very ends.
    assert_eq!(keep.len(), WINDOW_SAMPLES as usize);
    let coverage = true_count as f64 / WINDOW_SAMPLES as f64;
    assert!(
      coverage > 0.95,
      "all-true frame mask should cover most samples; coverage = {coverage}"
    );
  }

  #[test]
  fn expand_partial_mask_covers_only_active_frames() {
    let mut mask = [false; FRAMES_PER_WINDOW];
    mask[100..200].fill(true);
    let keep = frame_mask_to_sample_keep_mask(&mask, 0, 0, WINDOW_SAMPLES as u64);
    let true_count = keep.iter().filter(|&&b| b).count();
    // Frames 100..200 cover ≈ 100 frames * (160_000/589) ≈ 27_165 samples.
    assert!(
      (25_000..30_000).contains(&true_count),
      "expected ≈27_165 true samples; got {true_count}"
    );
    // Sample at the start of frame 99 should be false.
    let frame_99_sample = frame_to_sample(99) as usize;
    assert!(
      !keep[frame_99_sample.saturating_sub(1)],
      "boundary sample before frame 100 should be false"
    );
  }

  #[test]
  fn decide_uses_clean_when_long_enough() {
    let mut p = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    // Slot 0 alone for 300 frames → ~80k samples >> MIN_CLIP_SAMPLES (400).
    p[0][100..400].fill(0.9);
    let d = decide_keep_mask(&p, 0.5, 0, 0, 0, WINDOW_SAMPLES as u64);
    assert!(d.used_clean);
  }

  #[test]
  fn decide_falls_back_when_clean_too_short() {
    let mut p = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    // Slot 0 alone for ONE frame only → ~272 samples < 400 (MIN_CLIP_SAMPLES).
    p[0][100] = 0.9;
    // Slot 0 + slot 1 active for 100 overlapping frames → contributes to
    // speaker mask but NOT clean mask.
    p[0][200..300].fill(0.9);
    p[1][200..300].fill(0.9);
    let d = decide_keep_mask(&p, 0.5, 0, 0, 0, WINDOW_SAMPLES as u64);
    assert!(!d.used_clean, "expected fallback to speaker-only mask");
    let true_count = d.keep_mask.iter().filter(|&&b| b).count();
    assert!(
      true_count > 25_000,
      "speaker-only mask should still cover frames 100 + 200..300; got {true_count}"
    );
  }
}
