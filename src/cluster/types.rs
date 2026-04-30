//! Output types for `diarization::cluster`.

use crate::embed::Embedding;

/// A per-speaker centroid maintained by the online [`Clusterer`](crate::cluster::Clusterer).
///
/// Derives `Copy` because the centroid array is fixed-size (`[f32; 256]`),
/// making copies cheap compared to indirection.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerCentroid {
  pub(crate) speaker_id: u64,
  pub(crate) centroid: Embedding,
  pub(crate) assignment_count: u32,
}

impl SpeakerCentroid {
  /// The globally unique speaker identifier assigned by the clusterer.
  pub fn speaker_id(&self) -> u64 {
    self.speaker_id
  }

  /// The current centroid embedding for this speaker.
  pub fn centroid(&self) -> &Embedding {
    &self.centroid
  }

  /// Number of embeddings assigned to this speaker so far.
  pub fn assignment_count(&self) -> u32 {
    self.assignment_count
  }
}

/// The result of submitting one embedding to the online clusterer.
#[derive(Debug, Clone, Copy)]
pub struct ClusterAssignment {
  pub(crate) speaker_id: u64,
  pub(crate) is_new_speaker: bool,
  pub(crate) similarity: Option<f32>,
}

impl ClusterAssignment {
  /// The speaker identifier this embedding was assigned to.
  pub fn speaker_id(&self) -> u64 {
    self.speaker_id
  }

  /// `true` if this assignment opened a new speaker slot.
  pub fn is_new_speaker(&self) -> bool {
    self.is_new_speaker
  }

  /// Cosine similarity to the matched centroid, or `None` for the very
  /// first speaker (no prior centroid to compare against).
  pub fn similarity(&self) -> Option<f32> {
    self.similarity
  }
}
