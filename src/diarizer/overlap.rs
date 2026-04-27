//! `exclude_overlap` mask construction. Spec §5.8.
//!
//! Per-window per-frame raw probabilities → binarized per-(slot, frame)
//! activity → per-frame "clean" mask (only this slot active) → sample-
//! rate `keep_mask` (frame-aligned via [`crate::segment::stitch::frame_to_sample`]).
//!
//! The output `keep_mask` feeds [`crate::embed::EmbedModel::embed_masked`]
//! to extract a per-speaker embedding without overlap contamination.

use crate::segment::options::{FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS};

/// Per-(slot, frame) binarized speaker activity for a single window.
///
/// `binarized[slot][frame] = raw_probs[slot][frame] > threshold`. Strict
/// `>` matches spec §5.8 / pyannote's `Binarize(onset=...)` rule.
#[allow(dead_code)] // wired into decide_keep_mask (Task 38) and the pump (Task 43)
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
#[allow(dead_code)] // wired into decide_keep_mask (Task 38) and the pump (Task 43)
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
#[allow(dead_code)] // wired into decide_keep_mask (Task 38) and the pump (Task 43)
pub(crate) fn speaker_mask_for_slot(
  binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
  slot: u8,
) -> [bool; FRAMES_PER_WINDOW] {
  binarized[slot as usize]
}

/// Per-frame "clean mask" for slot `s`: `true` iff slot `s` is active
/// at that frame AND no other slot is active there.
#[allow(dead_code)] // wired into decide_keep_mask (Task 38) and the pump (Task 43)
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
