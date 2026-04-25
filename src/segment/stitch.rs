//! Overlap-add stitching of per-window voice probabilities.
//!
//! Each window contributes `FRAMES_PER_WINDOW` voice probabilities expanded
//! over its `WINDOW_SAMPLES` sample span. Overlapping windows are averaged
//! sample-by-sample.

extern crate alloc;

use alloc::{collections::VecDeque, vec::Vec};

use crate::segment::options::{FRAMES_PER_WINDOW, WINDOW_SAMPLES};

/// Convert a frame index in `0..=FRAMES_PER_WINDOW` to a sample offset in
/// `0..=WINDOW_SAMPLES` using rounded integer arithmetic.
#[inline]
pub(crate) const fn frame_to_sample(frame_idx: u32) -> u32 {
  // (frame_idx * WINDOW_SAMPLES + FRAMES_PER_WINDOW/2) / FRAMES_PER_WINDOW
  let n = frame_idx as u64 * WINDOW_SAMPLES as u64;
  let half = (FRAMES_PER_WINDOW as u64) / 2;
  ((n + half) / FRAMES_PER_WINDOW as u64) as u32
}

/// Stream-indexed accumulator for voice probability. Windows contribute via
/// [`add_window`]; finalized samples are exposed via [`take_finalized`].
pub(crate) struct VoiceStitcher {
  /// First absolute sample index represented in `sum` / `count`.
  base_sample: u64,
  /// Per-sample contribution sum.
  sum: VecDeque<f32>,
  /// Per-sample contribution count.
  count: VecDeque<u32>,
}

impl VoiceStitcher {
  pub(crate) fn new() -> Self {
    Self {
      base_sample: 0,
      sum: VecDeque::new(),
      count: VecDeque::new(),
    }
  }

  pub(crate) fn clear(&mut self) {
    self.base_sample = 0;
    self.sum.clear();
    self.count.clear();
  }

  /// Add one window of per-frame voice probabilities (length
  /// `FRAMES_PER_WINDOW`) starting at absolute sample `start_sample`.
  ///
  /// If `start_sample` is before the stitcher's `base_sample` (which can
  /// happen for an end-of-stream tail-anchor window that overlaps
  /// already-finalized samples), the prefix that lies in the finalized
  /// region is silently skipped — only the suffix `[base_sample, end)`
  /// contributes.
  pub(crate) fn add_window(&mut self, start_sample: u64, voice_per_frame: &[f32]) {
    debug_assert_eq!(voice_per_frame.len(), FRAMES_PER_WINDOW);

    // Entirely in the finalized region → nothing to do.
    let end_sample = start_sample + WINDOW_SAMPLES as u64;
    if end_sample <= self.base_sample {
      return;
    }

    // Ensure capacity covers [base_sample, end_sample).
    let needed_len = (end_sample - self.base_sample) as usize;
    while self.sum.len() < needed_len {
      self.sum.push_back(0.0);
      self.count.push_back(0);
    }

    // Expand each frame across its sample range, clipping to base_sample.
    // The index `f` is used both to compute frame->sample boundaries and to
    // pick the per-frame probability; iterator form is not cleaner here.
    #[allow(clippy::needless_range_loop)]
    for f in 0..FRAMES_PER_WINDOW {
      let s0 = frame_to_sample(f as u32) as u64;
      let s1 = frame_to_sample(f as u32 + 1) as u64;
      let p = voice_per_frame[f];
      for s in s0..s1 {
        let abs = start_sample + s;
        if abs < self.base_sample {
          continue;
        }
        let idx = (abs - self.base_sample) as usize;
        self.sum[idx] += p;
        self.count[idx] += 1;
      }
    }
  }

  /// Drain finalized samples up to but not including `up_to_sample`.
  /// Returns per-sample averaged voice probabilities and advances the base.
  pub(crate) fn take_finalized(&mut self, up_to_sample: u64) -> Vec<f32> {
    debug_assert!(up_to_sample >= self.base_sample);
    let n = (up_to_sample.saturating_sub(self.base_sample)) as usize;
    let n = n.min(self.sum.len());
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
      let s = self.sum.pop_front().unwrap();
      let c = self.count.pop_front().unwrap();
      out.push(if c == 0 { 0.0 } else { s / c as f32 });
    }
    self.base_sample += n as u64;
    out
  }

  pub(crate) fn base_sample(&self) -> u64 {
    self.base_sample
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn ones_window() -> Vec<f32> {
    vec![1.0; FRAMES_PER_WINDOW]
  }
  fn zeros_window() -> Vec<f32> {
    vec![0.0; FRAMES_PER_WINDOW]
  }

  #[test]
  fn frame_to_sample_endpoints() {
    assert_eq!(frame_to_sample(0), 0);
    assert_eq!(frame_to_sample(FRAMES_PER_WINDOW as u32), WINDOW_SAMPLES);
  }

  #[test]
  fn frame_to_sample_monotonic() {
    let mut prev = 0u32;
    for f in 1..=FRAMES_PER_WINDOW as u32 {
      let s = frame_to_sample(f);
      assert!(s >= prev);
      prev = s;
    }
  }

  #[test]
  fn single_window_finalize_all() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    let out = s.take_finalized(WINDOW_SAMPLES as u64);
    assert_eq!(out.len(), WINDOW_SAMPLES as usize);
    for v in out {
      assert!((v - 1.0).abs() < 1e-6);
    }
    assert_eq!(s.base_sample(), WINDOW_SAMPLES as u64);
  }

  #[test]
  fn two_overlapping_windows_average() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window()); // covers [0, 160_000)
    s.add_window(40_000, &zeros_window()); // covers [40_000, 200_000)
    // [0, 40_000) only window 1 contributed → 1.0
    // [40_000, 160_000) overlap → 0.5
    // [160_000, 200_000) only window 2 → 0.0
    let out = s.take_finalized(200_000);
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[39_999] - 1.0).abs() < 1e-6);
    assert!((out[40_000] - 0.5).abs() < 1e-6);
    assert!((out[159_999] - 0.5).abs() < 1e-6);
    assert!(out[160_000].abs() < 1e-6);
    assert!(out[199_999].abs() < 1e-6);
  }

  #[test]
  fn partial_finalize_advances_base() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    let part = s.take_finalized(40_000);
    assert_eq!(part.len(), 40_000);
    assert_eq!(s.base_sample(), 40_000);
    // Remaining samples still reachable.
    let rest = s.take_finalized(WINDOW_SAMPLES as u64);
    assert_eq!(rest.len(), 120_000);
    assert_eq!(s.base_sample(), WINDOW_SAMPLES as u64);
  }

  #[test]
  fn clear_resets() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    s.clear();
    assert_eq!(s.base_sample(), 0);
    assert!(s.take_finalized(100).is_empty());
  }
}
