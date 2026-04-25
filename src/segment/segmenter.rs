//! Layer-1 Sans-I/O speaker segmentation state machine.

extern crate alloc;

use alloc::{
  boxed::Box,
  collections::{BTreeMap, VecDeque},
  vec::Vec,
};

use mediatime::TimeRange;

use crate::segment::{
  hysteresis::Hysteresis,
  options::{SegmentOptions, SAMPLE_RATE_TB, WINDOW_SAMPLES},
  stitch::VoiceStitcher,
  types::{Action, WindowId},
};

/// Sans-I/O speaker segmentation state machine.
///
/// See the module docs for the high-level data flow. In short:
///
/// 1. Caller appends PCM via [`push_samples`](Self::push_samples).
/// 2. Caller drains [`Action`]s via [`poll`](Self::poll). When it sees
///    [`Action::NeedsInference`], it runs the model on the supplied samples
///    and calls [`push_inference`](Self::push_inference) with the scores.
/// 3. After all PCM is delivered, caller calls [`finish`](Self::finish) and
///    drains remaining actions.
///
/// `Segmenter` is `Send` but not `Sync`: use one per stream.
pub struct Segmenter {
  pub(crate) opts: SegmentOptions,

  /// Rolling sample buffer. Index 0 corresponds to absolute sample
  /// `consumed_samples`.
  pub(crate) input: VecDeque<f32>,
  pub(crate) consumed_samples: u64,

  /// Index of the next window to schedule (== how many windows already
  /// scheduled). Window k covers `[k * step_samples, k * step_samples + WINDOW_SAMPLES)`
  /// in absolute samples — *unless* it's the tail anchor.
  pub(crate) next_window_idx: u32,

  /// Pending inference round-trips: id → (window-start sample).
  pub(crate) pending: BTreeMap<WindowId, u64>,

  /// Output queue.
  pub(crate) pending_actions: VecDeque<Action>,

  /// Stream-indexed voice-probability accumulator.
  pub(crate) stitcher: VoiceStitcher,

  /// Online hysteresis cursor for the voice timeline.
  pub(crate) voice_hyst: Hysteresis,
  /// If voice is currently active, when did the run start?
  pub(crate) voice_run_start: Option<u64>,

  /// Once `finish()` has been called we may schedule the tail window.
  pub(crate) finished: bool,
  /// Final tail window has been emitted as `NeedsInference`.
  pub(crate) tail_emitted: bool,
  /// Total stream length latched at `finish()`.
  pub(crate) total_samples: u64,
}

impl Segmenter {
  /// Construct a new segmenter.
  pub fn new(opts: SegmentOptions) -> Self {
    let onset = opts.onset_threshold();
    let offset = opts.offset_threshold();
    Self {
      opts,
      input: VecDeque::new(),
      consumed_samples: 0,
      next_window_idx: 0,
      pending: BTreeMap::new(),
      pending_actions: VecDeque::new(),
      stitcher: VoiceStitcher::new(),
      voice_hyst: Hysteresis::new(onset, offset),
      voice_run_start: None,
      finished: false,
      tail_emitted: false,
      total_samples: 0,
    }
  }

  /// Read-only access to the configured options.
  pub fn options(&self) -> &SegmentOptions {
    &self.opts
  }

  /// Append 16 kHz mono float32 PCM samples. Arbitrary chunk size.
  ///
  /// Calling after [`finish`](Self::finish) is a programming bug; the call
  /// is silently ignored in release builds and panics in debug.
  pub fn push_samples(&mut self, samples: &[f32]) {
    debug_assert!(!self.finished, "push_samples after finish");
    if self.finished {
      return;
    }
    self.input.extend(samples.iter().copied());
    self.schedule_ready_windows();
  }

  /// Schedule any regular windows that are now fully buffered. Tail
  /// scheduling happens in `finish()`.
  fn schedule_ready_windows(&mut self) {
    let step = self.opts.step_samples() as u64;
    let win = WINDOW_SAMPLES as u64;
    loop {
      let start = self.next_window_idx as u64 * step;
      let end = start + win;
      // Buffered samples cover [consumed_samples, consumed_samples + input.len()).
      let buffered_end = self.consumed_samples + self.input.len() as u64;
      if buffered_end < end {
        return; // not enough audio yet
      }
      self.emit_window(start, /* zero_pad_to_window = */ false);
      self.next_window_idx += 1;
    }
  }

  /// Build a window starting at `start` (absolute samples), copy its
  /// samples (zero-padding if needed), enqueue `NeedsInference`, and
  /// trim the input buffer.
  pub(crate) fn emit_window(&mut self, start: u64, zero_pad_to_window: bool) {
    let win = WINDOW_SAMPLES as u64;
    let buffered_end = self.consumed_samples + self.input.len() as u64;

    // Copy samples [start, start + WINDOW_SAMPLES) from `input`, padding
    // with zeros past `buffered_end` if `zero_pad_to_window` is true.
    let mut samples: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
    let avail_end = if zero_pad_to_window {
      start + win
    } else {
      buffered_end.min(start + win)
    };

    let copy_from = (start.saturating_sub(self.consumed_samples)) as usize;
    let copy_until = (avail_end.saturating_sub(self.consumed_samples)) as usize;
    for i in copy_from..copy_until {
      samples.push(self.input[i]);
    }
    // Zero-pad tail if needed.
    while samples.len() < WINDOW_SAMPLES as usize {
      samples.push(0.0);
    }

    let id = WindowId::new(TimeRange::new(
      start as i64,
      (start + win) as i64,
      SAMPLE_RATE_TB,
    ));
    self.pending.insert(id, start);
    self.pending_actions.push_back(Action::NeedsInference {
      id,
      samples: Box::from(samples.as_slice()),
    });

    // Trim samples that no future window will need. Next window (if any)
    // starts at next_window_idx * step. Anything before that can go.
    let next_start = (self.next_window_idx + 1) as u64 * self.opts.step_samples() as u64;
    self.trim_input_to(next_start);
  }

  fn trim_input_to(&mut self, abs_sample: u64) {
    let target = abs_sample.min(self.consumed_samples + self.input.len() as u64);
    let drop_n = (target.saturating_sub(self.consumed_samples)) as usize;
    for _ in 0..drop_n {
      self.input.pop_front();
    }
    self.consumed_samples += drop_n as u64;
  }

  /// Drain the next pending action.
  pub fn poll(&mut self) -> Option<Action> {
    self.pending_actions.pop_front()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use mediatime::TimeRange;

  fn opts() -> SegmentOptions {
    SegmentOptions::default()
  }

  #[test]
  fn empty_no_actions() {
    let mut s = Segmenter::new(opts());
    assert!(s.poll().is_none());
  }

  #[test]
  fn first_window_emits_after_full_window_buffered() {
    let mut s = Segmenter::new(opts());
    // Push 80_000 samples — half a window. No action yet.
    s.push_samples(&vec![0.1f32; 80_000]);
    assert!(s.poll().is_none());
    // Push another 80_000 — now we have a full window.
    s.push_samples(&vec![0.2f32; 80_000]);
    match s.poll() {
      Some(Action::NeedsInference { id, samples }) => {
        assert_eq!(samples.len(), WINDOW_SAMPLES as usize);
        assert_eq!(id.range(), TimeRange::new(0, 160_000, SAMPLE_RATE_TB));
        // First half is 0.1, second half is 0.2.
        assert!((samples[0] - 0.1).abs() < 1e-6);
        assert!((samples[80_000] - 0.2).abs() < 1e-6);
      }
      other => panic!("expected NeedsInference, got {other:?}"),
    }
    assert!(s.poll().is_none());
  }

  #[test]
  fn second_window_emits_after_one_step_more_audio() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0f32; 160_000]);
    let _first = s.poll();
    // After 200_000 total samples we have second window covering [40_000, 200_000).
    s.push_samples(&vec![0.0f32; 40_000]);
    match s.poll() {
      Some(Action::NeedsInference { id, .. }) => {
        assert_eq!(id.range(), TimeRange::new(40_000, 200_000, SAMPLE_RATE_TB));
      }
      other => panic!("expected NeedsInference, got {other:?}"),
    }
  }
}
