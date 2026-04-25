//! Layer-1 Sans-I/O speaker segmentation state machine.

extern crate alloc;

use alloc::{
  boxed::Box,
  collections::{BTreeMap, VecDeque},
  vec,
  vec::Vec,
};

use mediatime::TimeRange;

use crate::segment::{
  error::Error,
  hysteresis::{runs_of_true, Hysteresis},
  options::{
    SegmentOptions, FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, SAMPLE_RATE_HZ,
    SAMPLE_RATE_TB, WINDOW_SAMPLES,
  },
  powerset::{powerset_to_speakers, softmax_row, voice_prob},
  stitch::{frame_to_sample, VoiceStitcher},
  types::{Action, SpeakerActivity, WindowId},
  window::plan_starts,
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
      self.emit_window(start);
      self.next_window_idx += 1;
    }
  }

  /// Build a window starting at `start` (absolute samples), copy its
  /// samples (zero-padding when the input buffer is shorter than
  /// `WINDOW_SAMPLES`), enqueue `NeedsInference`, and trim the input buffer.
  pub(crate) fn emit_window(&mut self, start: u64) {
    let win = WINDOW_SAMPLES as u64;
    let buffered_end = self.consumed_samples + self.input.len() as u64;

    // Copy samples [start, min(start+W, buffered_end)) from `input`. Anything
    // beyond `buffered_end` is zero-padded below.
    let mut samples: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
    let avail_end = buffered_end.min(start + win);

    let copy_from = (start.saturating_sub(self.consumed_samples)) as usize;
    let copy_until = (avail_end.saturating_sub(self.consumed_samples)) as usize;
    for i in copy_from..copy_until {
      samples.push(self.input[i]);
    }
    // Zero-pad tail if the buffer didn't reach `start + win`.
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

  /// Provide ONNX inference results for a previously-yielded window.
  ///
  /// `scores` must have length `FRAMES_PER_WINDOW * POWERSET_CLASSES = 4123`
  /// (powerset logits for each output frame).
  pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error> {
    let expected = FRAMES_PER_WINDOW * POWERSET_CLASSES;
    if scores.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: scores.len(),
      });
    }
    let start = self
      .pending
      .remove(&id)
      .ok_or(Error::UnknownWindow { id })?;

    // Decode powerset row by row.
    let mut speaker_probs: [Vec<f32>; MAX_SPEAKER_SLOTS as usize] = [
      vec![0.0; FRAMES_PER_WINDOW],
      vec![0.0; FRAMES_PER_WINDOW],
      vec![0.0; FRAMES_PER_WINDOW],
    ];
    let mut voice_per_frame: Vec<f32> = Vec::with_capacity(FRAMES_PER_WINDOW);

    for f in 0..FRAMES_PER_WINDOW {
      let row_start = f * POWERSET_CLASSES;
      let mut row = [0f32; POWERSET_CLASSES];
      row.copy_from_slice(&scores[row_start..row_start + POWERSET_CLASSES]);
      let probs = softmax_row(&row);
      voice_per_frame.push(voice_prob(&probs));
      let s = powerset_to_speakers(&probs);
      speaker_probs[0][f] = s[0];
      speaker_probs[1][f] = s[1];
      speaker_probs[2][f] = s[2];
    }

    // Emit per-window speaker activities.
    self.emit_speaker_activities(id, start, &speaker_probs);

    // Feed voice probabilities into the stitcher.
    self.stitcher.add_window(start, &voice_per_frame);

    // Finalize voice probabilities and emit any closed voice spans.
    self.process_voice_finalization();
    Ok(())
  }

  fn emit_speaker_activities(
    &mut self,
    id: WindowId,
    window_start: u64,
    speaker_probs: &[Vec<f32>; MAX_SPEAKER_SLOTS as usize],
  ) {
    let onset = self.opts.onset_threshold();
    let offset = self.opts.offset_threshold();
    let min_dur = self.opts.min_activity_duration();
    let min_samples = duration_to_samples(min_dur);

    for slot in 0..MAX_SPEAKER_SLOTS {
      let probs = &speaker_probs[slot as usize];
      // Per-window hysteresis (no carry — slots are window-local).
      let mut h = Hysteresis::new(onset, offset);
      let mask: Vec<bool> = probs.iter().map(|&p| h.push(p)).collect();
      for (f0, f1) in runs_of_true(&mask) {
        let s0 = window_start + frame_to_sample(f0 as u32) as u64;
        let s1 = window_start + frame_to_sample(f1 as u32) as u64;
        if s1 - s0 < min_samples {
          continue;
        }
        let range = TimeRange::new(s0 as i64, s1 as i64, SAMPLE_RATE_TB);
        self
          .pending_actions
          .push_back(Action::Activity(SpeakerActivity::new(id, slot, range)));
      }
    }
  }

  /// Pull finalized voice probabilities out of the stitcher and run them
  /// through the streaming hysteresis cursor, emitting closed voice spans.
  fn process_voice_finalization(&mut self) {
    let up_to = self.next_finalization_boundary();
    let probs = if up_to > self.stitcher.base_sample() {
      self.stitcher.take_finalized(up_to)
    } else {
      Vec::new()
    };
    let base_after = self.stitcher.base_sample();
    let base_before = base_after - probs.len() as u64;
    for (i, p) in probs.iter().enumerate() {
      let abs = base_before + i as u64;
      let was_active = self.voice_hyst.is_active();
      let now_active = self.voice_hyst.push(*p);
      match (was_active, now_active) {
        (false, true) => self.voice_run_start = Some(abs),
        (true, false) => {
          if let Some(start) = self.voice_run_start.take() {
            self.emit_voice_span(start, abs);
          }
        }
        _ => {}
      }
    }
    if self.finished && self.pending.is_empty() {
      if let Some(start) = self.voice_run_start.take() {
        self.emit_voice_span(start, self.total_samples);
        self.voice_hyst.reset();
      }
    }
  }

  /// Largest absolute sample finalized after current windows are processed.
  /// Pre-finish: no future regular window contributes to samples
  /// `< next_window_idx * step`. Post-finish (and after the tail window's
  /// scores are pushed), the boundary is `total_samples`.
  fn next_finalization_boundary(&self) -> u64 {
    if self.finished && self.pending.is_empty() {
      return self.total_samples;
    }
    let step = self.opts.step_samples() as u64;
    self.next_window_idx as u64 * step
  }

  fn emit_voice_span(&mut self, start_sample: u64, end_sample: u64) {
    let dur_samples = end_sample - start_sample;
    let min = duration_to_samples(self.opts.min_voice_duration());
    if dur_samples < min {
      return;
    }
    let range = TimeRange::new(start_sample as i64, end_sample as i64, SAMPLE_RATE_TB);
    self.pending_actions.push_back(Action::VoiceSpan(range));
  }

  /// Signal end-of-stream. Schedules a tail-anchored window if needed and
  /// causes any open voice span to close on subsequent `poll`s.
  pub fn finish(&mut self) {
    if self.finished {
      return;
    }
    self.finished = true;
    self.total_samples = self.consumed_samples + self.input.len() as u64;

    if self.tail_emitted {
      return;
    }
    if self.total_samples == 0 {
      return; // nothing to do
    }

    let starts = plan_starts(self.total_samples, self.opts.step_samples());
    let regular_emitted = self.next_window_idx as usize;
    // Anything in `starts` past what we've already emitted is the tail (or
    // a missed regular window plus tail; we just emit them in order).
    for &start in starts.iter().skip(regular_emitted) {
      self.emit_window(start);
    }
    self.tail_emitted = true;
    // If no tail was scheduled (e.g. total_samples is an exact multiple of
    // step that already produced a regular window covering the end), no
    // future `push_inference` will run, so flush voice finalization here.
    self.process_voice_finalization();
  }

  /// Reset to empty state. Internal allocations are reused.
  pub fn clear(&mut self) {
    self.input.clear();
    self.consumed_samples = 0;
    self.next_window_idx = 0;
    self.pending.clear();
    self.pending_actions.clear();
    self.stitcher.clear();
    self.voice_hyst.reset();
    self.voice_run_start = None;
    self.finished = false;
    self.tail_emitted = false;
    self.total_samples = 0;
  }
}

#[inline]
fn duration_to_samples(d: core::time::Duration) -> u64 {
  let nanos = d.as_nanos();
  // 16_000 samples/sec ⇒ samples = nanos * 16_000 / 1e9.
  (nanos * SAMPLE_RATE_HZ as u128 / 1_000_000_000u128) as u64
}

#[cfg(test)]
mod tests {
  use super::*;
  use mediatime::TimeRange;

  fn opts() -> SegmentOptions {
    SegmentOptions::default()
  }

  /// Build synthetic powerset logits where speaker A is "active" (class 1
  /// dominates) for frames in `active_frames`, otherwise silence (class 0
  /// dominates). All other classes are negligible.
  fn synth_logits(active_frames: core::ops::Range<usize>) -> Vec<f32> {
    let mut out = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    for f in 0..FRAMES_PER_WINDOW {
      let row_start = f * POWERSET_CLASSES;
      // Strong negative for everything, then boost the chosen class.
      for c in 0..POWERSET_CLASSES {
        out[row_start + c] = -10.0;
      }
      let active = active_frames.contains(&f);
      let dominant = if active { 1 } else { 0 }; // 1 = A only, 0 = silence
      out[row_start + dominant] = 10.0;
    }
    out
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

  #[test]
  fn push_inference_wrong_length_errors() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let action = s.poll().unwrap();
    let id = match action {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let bogus = vec![0.0f32; 100];
    match s.push_inference(id, &bogus) {
      Err(Error::InferenceShapeMismatch { expected, got }) => {
        assert_eq!(expected, FRAMES_PER_WINDOW * POWERSET_CLASSES);
        assert_eq!(got, 100);
      }
      other => panic!("unexpected: {other:?}"),
    }
  }

  #[test]
  fn push_inference_unknown_window_errors() {
    let mut s = Segmenter::new(opts());
    let bogus_id = WindowId::new(TimeRange::new(0, 160_000, SAMPLE_RATE_TB));
    let scores = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    match s.push_inference(bogus_id, &scores) {
      Err(Error::UnknownWindow { .. }) => {}
      other => panic!("unexpected: {other:?}"),
    }
  }

  #[test]
  fn one_window_speaker_a_active_emits_activity() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    // Speaker A active for a contiguous block of frames.
    let scores = synth_logits(100..200);
    s.push_inference(id, &scores).unwrap();

    // Drain actions; expect at least one Activity for slot 0 within
    // window [0, 160_000).
    let mut saw_activity = false;
    while let Some(a) = s.poll() {
      if let Action::Activity(act) = a {
        assert_eq!(act.window_id(), id);
        assert_eq!(act.speaker_slot(), 0);
        assert_eq!(act.range().timebase(), SAMPLE_RATE_TB);
        saw_activity = true;
      }
    }
    assert!(saw_activity, "expected at least one Activity for slot 0");
  }

  #[test]
  fn finish_short_clip_schedules_tail_window() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 50_000]); // less than one window
    assert!(s.poll().is_none());
    s.finish();
    match s.poll() {
      Some(Action::NeedsInference { samples, .. }) => {
        assert_eq!(samples.len(), WINDOW_SAMPLES as usize);
        // The 50_000 buffered samples should be at the start; the
        // rest is zero-padded.
        for i in 0..50_000 {
          assert_eq!(samples[i], 0.0);
        }
        for i in 50_000..160_000 {
          assert_eq!(samples[i], 0.0);
        }
      }
      other => panic!("unexpected: {other:?}"),
    }
  }

  #[test]
  fn clear_resets_state() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let _ = s.poll();
    s.clear();
    assert!(s.poll().is_none());
    // Push again — first window starts at sample 0 fresh.
    s.push_samples(&vec![0.0; 160_000]);
    match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => {
        assert_eq!(id.range().start_pts(), 0);
      }
      _ => unreachable!(),
    }
  }

  #[test]
  fn end_of_stream_closes_open_voice_span() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    // All frames "voiced" via class 1 (speaker A) — voice prob ≈ 1.0.
    let scores = synth_logits(0..FRAMES_PER_WINDOW);
    s.push_inference(id, &scores).unwrap();
    s.finish();
    // Tail window (anchored at 0) needs inference too.
    if let Some(Action::NeedsInference { id: tail_id, .. }) = s.poll() {
      s.push_inference(tail_id, &scores).unwrap();
    }

    let mut found_voice = false;
    while let Some(a) = s.poll() {
      if matches!(a, Action::VoiceSpan(_)) {
        found_voice = true;
      }
    }
    assert!(found_voice, "expected a closing voice span on finish");
  }
}
