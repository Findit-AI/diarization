//! Phase 5e: streaming VAD-gated diarization.
//!
//! Production flow: silero VAD continuously processes the input
//! audio, emitting bounded voice ranges. Each completed voice range
//! is passed through dia's offline diarization pipeline
//! ([`crate::offline::OwnedDiarizationPipeline`]), producing
//! pyannote-equivalent speaker assignments WITHIN that range. Cross-
//! range speaker identity is tracked via a session-wide centroid
//! bank.
//!
//! ## Why this matches pyannote accuracy
//!
//! The full-batch pyannote pipeline (`SpeakerDiarization.apply`) does
//! AHC + VBx clustering on the entire recording's embeddings. For
//! streaming use cases, waiting for the entire recording defeats the
//! purpose. The voice-range-gated pattern preserves pyannote's
//! algorithm (AHC + VBx + reconstruct) at the granularity of one
//! voice range — typically 0.5–60 seconds — and uses a lighter-
//! weight centroid-bank match across ranges to maintain global
//! speaker identity.
//!
//! Within each voice range, accuracy = `OwnedDiarizationPipeline`
//! accuracy (Phase 5d). Across ranges, identity tracking is similar
//! to the streaming `Diarizer::process_samples` cosine + EMA
//! approach but operates on per-range CENTROIDS (averages of clean
//! embeddings), which are far more stable than per-chunk-slot
//! embeddings.
//!
//! ## API
//!
//! - [`StreamingDiarizationPipeline::new`] — construct with the
//!   silero VAD options + dia's offline pipeline config.
//! - [`StreamingDiarizationPipeline::push_audio`] — push raw 16 kHz
//!   mono samples; emits diarized spans via callback as voice ranges
//!   close.
//! - [`StreamingDiarizationPipeline::finish`] — flush silero, run
//!   diarization on any trailing voice range.
//!
//! ## What this is NOT
//!
//! - Not bit-exact pyannote — within a voice range, accuracy is
//!   Phase 5d (1.77%–6.50% DER vs pyannote on 5 of 6 captured
//!   fixtures). The cross-range speaker bank adds another layer of
//!   approximation since pyannote clusters globally, not per-range.
//! - Not lower-latency than one voice range — the offline pipeline
//!   needs the full range to compute AHC dendrograms. For sub-range
//!   latency, use [`crate::diarizer::Diarizer::process_samples`]
//!   (online cosine + EMA, lower accuracy).

mod pipeline;

pub use pipeline::{
  StreamingDiarizationPipeline, StreamingDiarizedSpan, StreamingError, StreamingPipelineConfig,
};
