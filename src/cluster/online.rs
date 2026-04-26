//! Online streaming speaker clustering.

use crate::{
  cluster::{
    ClusterAssignment, ClusterOptions, Error, OverflowStrategy, SpeakerCentroid, UpdateStrategy,
  },
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

    // ── path 2+: argmax similarity ────────────────────────────────────
    let (best_idx, best_sim) = self
      .speakers
      .iter()
      .enumerate()
      .map(|(i, s)| (i, s.centroid.similarity(&embedding)))
      .fold((0usize, f32::NEG_INFINITY), |(bi, bs), (i, s)| {
        if s > bs { (i, s) } else { (bi, bs) }
      });

    let threshold = self.opts.similarity_threshold();
    let max_speakers = self.opts.max_speakers();

    if best_sim >= threshold {
      // ── path 3: assign to best match ─────────────────────────────
      let strategy = self.opts.update_strategy();
      Self::update_speaker(&mut self.speakers[best_idx], &embedding, strategy);
      Ok(ClusterAssignment {
        speaker_id: self.speakers[best_idx].speaker_id,
        is_new_speaker: false,
        similarity: Some(best_sim),
      })
    } else if (self.speakers.len() as u32) < max_speakers {
      // ── path 4: open a new slot ───────────────────────────────────
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
      Ok(ClusterAssignment {
        speaker_id: id,
        is_new_speaker: true,
        similarity: Some(best_sim),
      })
    } else {
      // ── path 5: overflow ─────────────────────────────────────────
      match self.opts.overflow_strategy() {
        OverflowStrategy::AssignClosest => {
          let strategy = self.opts.update_strategy();
          Self::update_speaker(&mut self.speakers[best_idx], &embedding, strategy);
          Ok(ClusterAssignment {
            speaker_id: self.speakers[best_idx].speaker_id,
            is_new_speaker: false,
            similarity: Some(best_sim),
          })
        }
        OverflowStrategy::Reject => Err(Error::TooManySpeakers {
          limit: max_speakers,
        }),
      }
    }
  }

  // ── helpers ────────────────────────────────────────────────────────────

  fn mint_id(&mut self) -> u64 {
    let id = self.next_id;
    self.next_id += 1;
    id
  }

  /// Update `entry`'s centroid and accumulator given a newly assigned
  /// `embedding`, using the specified `strategy`.
  ///
  /// After update the centroid is always L2-normalized (or left as-is if
  /// the updated vector is degenerate — that can only happen due to float
  /// underflow in pathological edge cases).
  fn update_speaker(entry: &mut SpeakerEntry, embedding: &Embedding, strategy: UpdateStrategy) {
    entry.assignment_count = entry.assignment_count.saturating_add(1);
    match strategy {
      UpdateStrategy::RollingMean => {
        // Accumulate then re-normalize.
        // (accumulator tracks the running unnormalized sum; centroid is
        // the normalized direction of that sum.)
        for (acc, &val) in entry.accumulator.iter_mut().zip(embedding.0.iter()) {
          *acc += val;
        }
        // Re-normalize accumulator to produce the new centroid.
        let sq: f32 = entry.accumulator.iter().map(|x| x * x).sum();
        let norm = sq.sqrt();
        if norm > 1e-12 {
          let mut new_centroid = [0.0f32; EMBEDDING_DIM];
          for (c, &a) in new_centroid.iter_mut().zip(entry.accumulator.iter()) {
            *c = a / norm;
          }
          entry.centroid = Embedding(new_centroid);
        }
        // If norm ≤ 1e-12 (degenerate), keep existing centroid.
      }
      UpdateStrategy::Ema(alpha) => {
        // centroid_new = (1 − α) * centroid_old + α * embedding
        // Then re-normalize.
        let mut blended = [0.0f32; EMBEDDING_DIM];
        for ((b, &old), &new) in blended
          .iter_mut()
          .zip(entry.centroid.0.iter())
          .zip(embedding.0.iter())
        {
          *b = (1.0 - alpha) * old + alpha * new;
        }
        let sq: f32 = blended.iter().map(|x| x * x).sum();
        let norm = sq.sqrt();
        if norm > 1e-12 {
          for b in blended.iter_mut() {
            *b /= norm;
          }
          entry.centroid = Embedding(blended);
        }
        // Also update accumulator to reflect current centroid (for
        // consistency; not read by EMA path, but keeps state coherent).
        for (acc, &c) in entry.accumulator.iter_mut().zip(entry.centroid.0.iter()) {
          *acc = c;
        }
      }
    }
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

  #[test]
  fn identical_embedding_stays_in_same_speaker() {
    let mut c = Clusterer::new(ClusterOptions::new());
    let e = unit_embedding(0);
    let a1 = c.submit(e).unwrap();
    let a2 = c.submit(e).unwrap();
    assert_eq!(a1.speaker_id(), a2.speaker_id());
    assert!(!a2.is_new_speaker());
    // similarity to self is 1.0
    assert!((a2.similarity().unwrap() - 1.0).abs() < 1e-5);
  }

  #[test]
  fn orthogonal_embeddings_open_new_speaker() {
    // Threshold = 0.5; orthogonal embeddings have sim = 0.0 → new speaker.
    let mut c = Clusterer::new(ClusterOptions::new());
    let e0 = unit_embedding(0);
    let e1 = unit_embedding(1);
    let a0 = c.submit(e0).unwrap();
    let a1 = c.submit(e1).unwrap();
    assert_ne!(a0.speaker_id(), a1.speaker_id());
    assert!(a1.is_new_speaker());
    assert_eq!(c.speaker_count(), 2);
  }

  #[test]
  fn overflow_assign_closest_does_not_open_new_speaker() {
    let opts = ClusterOptions::new()
      .with_max_speakers(1)
      .with_overflow_strategy(OverflowStrategy::AssignClosest);
    let mut c = Clusterer::new(opts);
    let e0 = unit_embedding(0);
    let e1 = unit_embedding(1);
    c.submit(e0).unwrap();
    // e1 is orthogonal → would open new speaker but max=1.
    let a1 = c.submit(e1).unwrap();
    assert!(!a1.is_new_speaker());
    assert_eq!(c.speaker_count(), 1);
  }

  #[test]
  fn overflow_reject_returns_error() {
    let opts = ClusterOptions::new()
      .with_max_speakers(1)
      .with_overflow_strategy(OverflowStrategy::Reject);
    let mut c = Clusterer::new(opts);
    let e0 = unit_embedding(0);
    let e1 = unit_embedding(1);
    c.submit(e0).unwrap();
    let err = c.submit(e1).unwrap_err();
    assert!(matches!(err, Error::TooManySpeakers { limit: 1 }));
  }

  #[test]
  fn ema_update_centroid_moves_toward_new_embedding() {
    // Start with speaker at dim=0. Submit same dim=0 → centroid stays at
    // dim=0. Then submit dim=1 at a high threshold so it stays in speaker 0.
    let opts = ClusterOptions::new()
      .with_similarity_threshold(0.0) // accept everything into first speaker
      .with_max_speakers(1)
      .with_update_strategy(UpdateStrategy::Ema(0.5));
    let mut c = Clusterer::new(opts);
    let e0 = unit_embedding(0);
    let e1 = unit_embedding(1);
    c.submit(e0).unwrap();
    c.submit(e1).unwrap();
    // After one EMA step: blended = (0.5 * e0 + 0.5 * e1) / ||...||
    // Normalized: [1/√2, 1/√2, 0…]
    let snaps = c.speakers();
    let centroid = snaps[0].centroid();
    // Both dim 0 and dim 1 should be ≈ 1/√2
    let expected = 1.0_f32 / 2.0_f32.sqrt();
    assert!((centroid.as_array()[0] - expected).abs() < 1e-5);
    assert!((centroid.as_array()[1] - expected).abs() < 1e-5);
  }

  #[test]
  fn rolling_mean_assignment_count_increments() {
    let opts = ClusterOptions::new()
      .with_similarity_threshold(0.0)
      .with_update_strategy(UpdateStrategy::RollingMean);
    let mut c = Clusterer::new(opts);
    let e = unit_embedding(0);
    c.submit(e).unwrap();
    c.submit(e).unwrap();
    c.submit(e).unwrap();
    let snaps = c.speakers();
    assert_eq!(snaps[0].assignment_count(), 3);
  }

  #[test]
  fn speakers_snapshot_reflects_all_centroids() {
    let mut c = Clusterer::new(ClusterOptions::new());
    c.submit(unit_embedding(0)).unwrap();
    c.submit(unit_embedding(1)).unwrap();
    let snaps = c.speakers();
    assert_eq!(snaps.len(), 2);
    let ids: Vec<u64> = snaps.iter().map(|s| s.speaker_id()).collect();
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
  }
}
