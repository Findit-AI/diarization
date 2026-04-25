//! Public types emitted by the segmentation state machine.

use core::num::NonZeroU32;

use mediatime::{Timebase, TimeRange, Timestamp};

use crate::segment::options::SAMPLE_RATE_TB;

/// Stable correlation handle for one inference round-trip.
///
/// Equal to the window's sample range in `SAMPLE_RATE_TB`. Two `WindowId`s
/// with the same start and end compare equal and hash to the same value, so
/// it is safe to use as a `HashMap` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(TimeRange);

impl WindowId {
    pub(crate) const fn new(range: TimeRange) -> Self { Self(range) }
    /// Sample-range covered by the window in `SAMPLE_RATE_TB`.
    pub const fn range(&self) -> TimeRange { self.0 }
    /// Window start as a `Timestamp`.
    pub const fn start(&self) -> Timestamp { self.0.start() }
    /// Window end as a `Timestamp`.
    pub const fn end(&self) -> Timestamp { self.0.end() }
    /// Window duration (always 10 s for v0.1.0).
    pub fn duration(&self) -> core::time::Duration { self.0.duration() }
}

/// One window-local speaker activity.
///
/// `speaker_slot` ∈ `0..=2` is local to the emitting window — slot identity
/// does NOT cross windows. Cross-window speaker identity is the job of a
/// future clustering layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerActivity {
    window_id: WindowId,
    speaker_slot: u8,
    range: TimeRange,
}

impl SpeakerActivity {
    pub(crate) const fn new(window_id: WindowId, speaker_slot: u8, range: TimeRange) -> Self {
        Self { window_id, speaker_slot, range }
    }
    /// The window this activity was decoded from.
    pub const fn window_id(&self) -> WindowId { self.window_id }
    /// Window-local speaker slot (0, 1, or 2).
    pub const fn speaker_slot(&self) -> u8 { self.speaker_slot }
    /// Sample range of the activity within the stream, in `SAMPLE_RATE_TB`.
    pub const fn range(&self) -> TimeRange { self.range }
}

/// One output of the Layer-1 state machine.
#[derive(Debug, Clone)]
pub enum Action {
    /// The caller must run ONNX inference on `samples` and call
    /// [`Segmenter::push_inference`](crate::segment::Segmenter::push_inference)
    /// with the same `id`.
    NeedsInference {
        /// Correlation handle (the window's sample range).
        id: WindowId,
        /// Always `WINDOW_SAMPLES = 160_000` mono float32 samples at 16 kHz,
        /// zero-padded if the input stream is shorter.
        samples: alloc::boxed::Box<[f32]>,
    },
    /// A decoded window-local speaker activity.
    Activity(SpeakerActivity),
    /// A finalized speaker-agnostic voice region.
    VoiceSpan(TimeRange),
}

/// Layer-2 emission events (Layer 2 hides `NeedsInference` from the caller).
#[derive(Debug, Clone)]
pub enum Event {
    /// A decoded window-local speaker activity.
    Activity(SpeakerActivity),
    /// A finalized speaker-agnostic voice region.
    VoiceSpan(TimeRange),
}

#[cfg(feature = "std")]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(start: i64, end: i64) -> TimeRange {
        TimeRange::new(start, end, SAMPLE_RATE_TB)
    }

    #[test]
    fn window_id_accessors() {
        let id = WindowId::new(tr(0, 160_000));
        assert_eq!(id.range(), tr(0, 160_000));
        assert_eq!(id.start().pts(), 0);
        assert_eq!(id.end().pts(), 160_000);
        assert_eq!(id.duration(), core::time::Duration::from_secs(10));
    }

    #[test]
    fn window_id_hash_eq_value_semantic() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(WindowId::new(tr(0, 160_000)));
        assert!(s.contains(&WindowId::new(tr(0, 160_000))));
        assert!(!s.contains(&WindowId::new(tr(40_000, 200_000))));
    }

    #[test]
    fn speaker_activity_accessors() {
        let win = WindowId::new(tr(0, 160_000));
        let act = SpeakerActivity::new(win, 1, tr(8_000, 24_000));
        assert_eq!(act.window_id(), win);
        assert_eq!(act.speaker_slot(), 1);
        assert_eq!(act.range(), tr(8_000, 24_000));
    }
}
