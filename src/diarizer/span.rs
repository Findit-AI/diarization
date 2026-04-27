//! Public output types for [`crate::diarizer::Diarizer`]. Spec §4.4.

use mediatime::TimeRange;

use crate::embed::Embedding;

/// Per-`(window, slot)` context retained during a diarization session.
/// Returned by `Diarizer::collected_embeddings()`. Carries everything
/// needed to correlate a post-hoc offline-clustering re-labeling
/// back to its source activity.
///
/// Granularity is **per-`(window, slot)`** — one entry per pre-
/// reconstruction `SpeakerActivity` from [`crate::segment`]. Finer-
/// grained than the post-reconstruction [`DiarizedSpan`] output (one
/// per closed cluster run). The two views are reconciled via
/// `online_speaker_id`.
///
/// Style note: `pub` fields, not accessors. Matches [`crate::embed::EmbeddingMeta`]
/// — this type is a debug/audit bag of pre-computed values, not a
/// type with internal invariants.
#[derive(Debug, Clone)]
pub struct CollectedEmbedding {
  /// Sample range of the source activity, in `crate::segment::SAMPLE_RATE_TB`.
  pub range: TimeRange,
  /// L2-normalized embedding extracted from the activity.
  pub embedding: Embedding,
  /// Online speaker id assigned by `Clusterer::submit` during streaming.
  pub online_speaker_id: u64,
  /// Window-local slot from `dia::segment::SpeakerActivity` (`0..MAX_SPEAKER_SLOTS`).
  pub speaker_slot: u8,
  /// Whether the embedding used the `exclude_overlap` clean mask
  /// (`true`) or fell back to the speaker-only mask (`false`).
  /// See spec §5.8 for fallback semantics.
  pub used_clean_mask: bool,
}

/// One closed speaker turn after reconstruction.
///
/// **Rev-6 shape:** `(range, speaker_id, is_new_speaker)` plus three
/// rev-7 quality metrics. Window-local concepts (similarity, slot)
/// don't apply to a stitched multi-window span; that context lives
/// on [`CollectedEmbedding`].
///
/// Style note: `pub(crate)` fields with accessors. Matches
/// `cluster::SpeakerCentroid` and `cluster::ClusterAssignment` —
/// types with semantic invariants that may evolve (rev-6 changed
/// the field set; without accessors this would have been a breaking
/// change).
#[derive(Debug, Clone, Copy)]
pub struct DiarizedSpan {
  pub(crate) range: TimeRange,
  pub(crate) speaker_id: u64,
  pub(crate) is_new_speaker: bool,
  pub(crate) average_activation: f32,
  pub(crate) activity_count: u32,
  pub(crate) clean_mask_fraction: f32,
}

impl DiarizedSpan {
  /// Sample range of this span, in `crate::segment::SAMPLE_RATE_TB`.
  pub fn range(&self) -> TimeRange {
    self.range
  }

  /// Global cluster id assigned by the online clusterer.
  pub fn speaker_id(&self) -> u64 {
    self.speaker_id
  }

  /// `true` iff this is the first time `speaker_id` has been emitted
  /// in the current `Diarizer` session (post `new`/`clear`).
  pub fn is_new_speaker(&self) -> bool {
    self.is_new_speaker
  }

  /// Mean per-frame normalized activation across the span's frames.
  /// Each frame's contribution is `activation_sum / activation_chunk_count`,
  /// so the value is in `[0.0, 1.0]`. Higher = more confident speaker
  /// assignment. Roughly comparable across spans. (Rev-7.)
  pub fn average_activation(&self) -> f32 {
    self.average_activation
  }

  /// Number of `(WindowId, slot)` segment activities that contributed
  /// to this span. (Rev-7.)
  pub fn activity_count(&self) -> u32 {
    self.activity_count
  }

  /// Of the contributing activities, the fraction whose embedding
  /// used the `exclude_overlap` clean mask. Range `[0.0, 1.0]`.
  /// Lower = more overlap-contaminated. (Rev-7.)
  pub fn clean_mask_fraction(&self) -> f32 {
    self.clean_mask_fraction
  }
}
