//! Configuration constants and tunables for `dia::segment`.

use core::num::NonZeroU32;
use core::time::Duration;

use mediatime::Timebase;

/// Audio sample rate this module supports — 16 kHz.
///
/// pyannote/segmentation-3.0 was trained at 16 kHz only. Callers must
/// resample upstream.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// `mediatime` timebase for every sample-indexed `Timestamp` and `TimeRange`
/// emitted by this module: `1 / 16_000` seconds.
pub const SAMPLE_RATE_TB: Timebase =
    Timebase::new(1, NonZeroU32::new(SAMPLE_RATE_HZ).unwrap());

/// Sample count of one model window — 160 000 samples (10 s at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 160_000;

/// Output frames produced per window by the segmentation model.
pub const FRAMES_PER_WINDOW: usize = 589;

/// Powerset class count: silence, A, B, C, A+B, A+C, B+C.
pub const POWERSET_CLASSES: usize = 7;

/// Maximum simultaneous speakers per window.
pub const MAX_SPEAKER_SLOTS: u8 = 3;

/// Tunables for the segmenter. Defaults match the upstream pyannote pipeline.
#[derive(Debug, Clone)]
pub struct SegmentOptions {
    onset_threshold: f32,
    offset_threshold: f32,
    step_samples: u32,
    min_voice_duration: Duration,
    min_activity_duration: Duration,
    voice_merge_gap: Duration,
}

impl Default for SegmentOptions {
    fn default() -> Self { Self::new() }
}

impl SegmentOptions {
    /// Construct with pyannote defaults: onset 0.5, offset 0.357,
    /// step 40 000 samples (2.5 s), all duration filters disabled.
    pub const fn new() -> Self {
        Self {
            onset_threshold: 0.5,
            offset_threshold: 0.357,
            step_samples: 40_000,
            min_voice_duration: Duration::ZERO,
            min_activity_duration: Duration::ZERO,
            voice_merge_gap: Duration::ZERO,
        }
    }

    /// Onset (rising-edge) threshold for hysteresis binarization.
    pub const fn onset_threshold(&self) -> f32 { self.onset_threshold }
    /// Offset (falling-edge) threshold for hysteresis binarization.
    pub const fn offset_threshold(&self) -> f32 { self.offset_threshold }
    /// Sliding-window step in samples (default 40 000 = 2.5 s).
    pub const fn step_samples(&self) -> u32 { self.step_samples }
    /// Minimum voice-span duration; shorter spans are dropped (default 0).
    pub const fn min_voice_duration(&self) -> Duration { self.min_voice_duration }
    /// Minimum speaker-activity duration (default 0).
    pub const fn min_activity_duration(&self) -> Duration { self.min_activity_duration }
    /// Merge adjacent voice spans separated by at most this gap (default 0).
    pub const fn voice_merge_gap(&self) -> Duration { self.voice_merge_gap }

    /// Builder: set the onset threshold.
    pub const fn with_onset_threshold(mut self, v: f32) -> Self { self.onset_threshold = v; self }
    /// Builder: set the offset threshold.
    pub const fn with_offset_threshold(mut self, v: f32) -> Self { self.offset_threshold = v; self }
    /// Builder: set the sliding-window step in samples.
    pub const fn with_step_samples(mut self, v: u32) -> Self { self.step_samples = v; self }
    /// Builder: set the minimum voice-span duration.
    pub const fn with_min_voice_duration(mut self, v: Duration) -> Self { self.min_voice_duration = v; self }
    /// Builder: set the minimum speaker-activity duration.
    pub const fn with_min_activity_duration(mut self, v: Duration) -> Self { self.min_activity_duration = v; self }
    /// Builder: set the voice-span merge gap.
    pub const fn with_voice_merge_gap(mut self, v: Duration) -> Self { self.voice_merge_gap = v; self }

    /// Mutating: set the onset threshold.
    pub fn set_onset_threshold(&mut self, v: f32) -> &mut Self { self.onset_threshold = v; self }
    /// Mutating: set the offset threshold.
    pub fn set_offset_threshold(&mut self, v: f32) -> &mut Self { self.offset_threshold = v; self }
    /// Mutating: set the sliding-window step in samples.
    pub fn set_step_samples(&mut self, v: u32) -> &mut Self { self.step_samples = v; self }
    /// Mutating: set the minimum voice-span duration.
    pub fn set_min_voice_duration(&mut self, v: Duration) -> &mut Self { self.min_voice_duration = v; self }
    /// Mutating: set the minimum speaker-activity duration.
    pub fn set_min_activity_duration(&mut self, v: Duration) -> &mut Self { self.min_activity_duration = v; self }
    /// Mutating: set the voice-span merge gap.
    pub fn set_voice_merge_gap(&mut self, v: Duration) -> &mut Self { self.voice_merge_gap = v; self }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pyannote() {
        let o = SegmentOptions::default();
        assert_eq!(o.onset_threshold(), 0.5);
        assert!((o.offset_threshold() - 0.357).abs() < 1e-6);
        assert_eq!(o.step_samples(), 40_000);
        assert_eq!(o.min_voice_duration(), Duration::ZERO);
    }

    #[test]
    fn builder_round_trip() {
        let o = SegmentOptions::new()
            .with_onset_threshold(0.6)
            .with_offset_threshold(0.4)
            .with_step_samples(20_000)
            .with_min_voice_duration(Duration::from_millis(100))
            .with_min_activity_duration(Duration::from_millis(50))
            .with_voice_merge_gap(Duration::from_millis(30));

        assert_eq!(o.onset_threshold(), 0.6);
        assert_eq!(o.offset_threshold(), 0.4);
        assert_eq!(o.step_samples(), 20_000);
        assert_eq!(o.min_voice_duration(), Duration::from_millis(100));
        assert_eq!(o.min_activity_duration(), Duration::from_millis(50));
        assert_eq!(o.voice_merge_gap(), Duration::from_millis(30));
    }

    #[test]
    fn sample_rate_tb_matches_constant() {
        assert_eq!(SAMPLE_RATE_TB.den().get(), SAMPLE_RATE_HZ);
        assert_eq!(SAMPLE_RATE_TB.num(), 1);
    }
}
