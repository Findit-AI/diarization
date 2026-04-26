//! Online streaming speaker clustering.

use crate::{
  cluster::{ClusterAssignment, ClusterOptions, Error, SpeakerCentroid},
  embed::{EMBEDDING_DIM, Embedding},
};

// ── Internal per-speaker state ─────────────────────────────────────────────

/// Per-speaker state maintained inside the `Clusterer`.
///
/// Kept private; callers observe speakers only through [`SpeakerCentroid`]
/// and [`ClusterAssignment`].
struct SpeakerEntry {
  /// Globally unique speaker id (monotonically assigned).
  speaker_id: u64,
  /// Current centroid — always L2-normalized after each update.
  centroid: Embedding,
  /// Number of embeddings assigned so far (used by `RollingMean`).
  assignment_count: u32,
  /// Running accumulator used by `RollingMean` before re-normalization.
  ///
  /// For `Ema` this field is unused; the centroid IS the EMA.
  #[allow(dead_code)] // TODO(task-12): used by update_speaker RollingMean branch
  accumulator: [f32; EMBEDDING_DIM],
}

// ── Clusterer ─────────────────────────────────────────────────────────────

/// Online streaming speaker clusterer.
///
/// Maintains a set of per-speaker centroids.  Call [`submit`](Clusterer::submit)
/// for each `Embedding` produced by `dia::embed`; it returns a
/// [`ClusterAssignment`] containing the globally-unique speaker id and whether
/// a new speaker slot was opened.
///
/// # Thread safety
/// `Clusterer` is **not** `Sync`; wrap it in a `Mutex` if you need to share
/// it across threads.
pub struct Clusterer {
  opts: ClusterOptions,
  /// Active speaker slots.
  speakers: Vec<SpeakerEntry>,
  /// Monotonically increasing counter used to mint new speaker ids.
  next_id: u64,
}

impl Clusterer {
  /// Construct a new `Clusterer` with the supplied options.
  pub fn new(opts: ClusterOptions) -> Self {
    Self {
      opts,
      speakers: Vec::new(),
      next_id: 0,
    }
  }

  /// Return a snapshot of all current speaker centroids.
  pub fn speakers(&self) -> Vec<SpeakerCentroid> {
    self
      .speakers
      .iter()
      .map(|e| SpeakerCentroid {
        speaker_id: e.speaker_id,
        centroid: e.centroid,
        assignment_count: e.assignment_count,
      })
      .collect()
  }

  /// How many speaker slots are currently open.
  pub fn speaker_count(&self) -> usize {
    self.speakers.len()
  }

  /// Reset all speaker state (centroids, counts, id counter).
  pub fn reset(&mut self) {
    self.speakers.clear();
    self.next_id = 0;
  }

  // ── submit ─────────────────────────────────────────────────────────────

  /// Submit one embedding and receive a speaker assignment.
  ///
  /// # Algorithm
  /// 1. If no speakers exist, open the first slot (returns
  ///    `is_new_speaker = true`, `similarity = None`).
  /// 2. Otherwise compute cosine similarity to every centroid and find the
  ///    argmax.
  /// 3. If `argmax_similarity >= similarity_threshold`, assign to that
  ///    speaker and update its centroid.
  /// 4. Else if `speaker_count < max_speakers`, open a new slot.
  /// 5. Else apply the `OverflowStrategy`:
  ///    - `AssignClosest`: assign to the argmax speaker anyway.
  ///    - `Reject`: return `Err(Error::TooManySpeakers { limit })`.
  pub fn submit(&mut self, embedding: Embedding) -> Result<ClusterAssignment, Error> {
    // ── path 1: first speaker ─────────────────────────────────────────
    if self.speakers.is_empty() {
      let id = self.mint_id();
      let mut accumulator = [0.0f32; EMBEDDING_DIM];
      for (acc, &val) in accumulator.iter_mut().zip(embedding.0.iter()) {
        *acc = val;
      }
      self.speakers.push(SpeakerEntry {
        speaker_id: id,
        centroid: embedding,
        assignment_count: 1,
        accumulator,
      });
      return Ok(ClusterAssignment {
        speaker_id: id,
        is_new_speaker: true,
        similarity: None,
      });
    }

    // ── path 2+: find best match ───────────────────────────────────────
    // TODO(task-12): replace placeholder with argmax + EMA + overflow
    Err(Error::TooManySpeakers {
      limit: self.opts.max_speakers(),
    })
  }

  // ── helpers ────────────────────────────────────────────────────────────

  fn mint_id(&mut self) -> u64 {
    let id = self.next_id;
    self.next_id += 1;
    id
  }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  fn unit_embedding(dim: usize) -> Embedding {
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[dim] = 1.0;
    Embedding::normalize_from(v).unwrap()
  }

  #[test]
  fn first_submit_opens_speaker_zero() {
    let mut c = Clusterer::new(ClusterOptions::new());
    let e = unit_embedding(0);
    let a = c.submit(e).unwrap();
    assert_eq!(a.speaker_id(), 0);
    assert!(a.is_new_speaker());
    assert_eq!(a.similarity(), None);
    assert_eq!(c.speaker_count(), 1);
  }

  #[test]
  fn reset_clears_state() {
    let mut c = Clusterer::new(ClusterOptions::new());
    let e = unit_embedding(0);
    c.submit(e).unwrap();
    assert_eq!(c.speaker_count(), 1);
    c.reset();
    assert_eq!(c.speaker_count(), 0);
    // After reset, next submit opens speaker 0 again.
    let a = c.submit(e).unwrap();
    assert_eq!(a.speaker_id(), 0);
  }
}
