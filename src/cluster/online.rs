//! Online streaming speaker clustering.

use crate::{
  cluster::{
    ClusterAssignment, ClusterOptions, Error, OverflowStrategy, SpeakerCentroid, UpdateStrategy,
  },
  embed::{EMBEDDING_DIM, Embedding, NORM_EPSILON},
};

// ── Internal per-speaker state ─────────────────────────────────────────────

/// Per-speaker state maintained inside the `Clusterer`.
///
/// Kept private; callers observe speakers only through [`SpeakerCentroid`]
/// and [`ClusterAssignment`].
struct SpeakerEntry {
  /// Globally unique speaker id (monotonically assigned).
  speaker_id: u64,
  /// Running accumulator used by both `RollingMean` and `Ema` strategies.
  ///
  /// For `RollingMean` this is the unnormalized sum of all assigned
  /// embeddings. For `Ema` this is the current EMA value (unnormalized).
  accumulator: [f32; EMBEDDING_DIM],
  /// Cached L2-normalized centroid derived from `accumulator`.
  ///
  /// Updated lazily: only refreshed when `||accumulator|| >= NORM_EPSILON`.
  /// Retains the last-good value when the accumulator becomes degenerate
  /// (e.g., antipodal cancellation). Spec §5.4.
  cached_centroid: [f32; EMBEDDING_DIM],
  /// Number of embeddings assigned so far.
  assignment_count: u32,
}

// ── Clusterer ─────────────────────────────────────────────────────────────

/// Online streaming speaker clusterer.
///
/// Maintains a set of per-speaker centroids.  Call [`submit`](Clusterer::submit)
/// for each `Embedding` produced by `diarization::embed`; it returns a
/// [`ClusterAssignment`] containing the globally-unique speaker id and whether
/// a new speaker slot was opened.
///
/// # Thread safety
/// `Clusterer` is `Send + Sync`. However, `submit` takes `&mut self`, so
/// concurrent updates from multiple threads require external synchronization
/// (e.g., a `Mutex`).
pub struct Clusterer {
  opts: ClusterOptions,
  /// Active speaker slots.
  speakers: Vec<SpeakerEntry>,
  /// Monotonically increasing counter used to mint new speaker ids.
  next_speaker_id: u64,
}

impl Clusterer {
  /// Construct a new `Clusterer` with the supplied options.
  pub fn new(opts: ClusterOptions) -> Self {
    Self {
      opts,
      speakers: Vec::new(),
      next_speaker_id: 0,
    }
  }

  /// Borrow the configured options.
  pub fn options(&self) -> &ClusterOptions {
    &self.opts
  }

  /// Return a snapshot of all current speaker centroids.
  pub fn speakers(&self) -> Vec<SpeakerCentroid> {
    self
      .speakers
      .iter()
      .map(|e| SpeakerCentroid {
        speaker_id: e.speaker_id,
        centroid: Embedding(e.cached_centroid),
        assignment_count: e.assignment_count,
      })
      .collect()
  }

  /// How many speaker slots are currently open.
  pub fn num_speakers(&self) -> usize {
    self.speakers.len()
  }

  /// Look up a single speaker by id.
  pub fn speaker(&self, id: u64) -> Option<SpeakerCentroid> {
    self
      .speakers
      .iter()
      .find(|s| s.speaker_id == id)
      .map(|s| SpeakerCentroid {
        speaker_id: s.speaker_id,
        centroid: Embedding(s.cached_centroid),
        assignment_count: s.assignment_count,
      })
  }

  /// Clear all speaker state (centroids, counts, id counter).
  pub fn clear(&mut self) {
    self.speakers.clear();
    self.next_speaker_id = 0;
  }

  // ── submit ─────────────────────────────────────────────────────────────

  /// Submit one embedding and receive a speaker assignment.
  ///
  /// # Algorithm (spec §5.4)
  /// 1. If no speakers exist, open the first slot (returns
  ///    `is_new_speaker = true`, `similarity = None`).
  /// 2. Compute cosine similarity to every centroid via dot product on
  ///    unit vectors; find the argmax (lowest-index tie-break).
  /// 3. If `argmax_similarity >= similarity_threshold`, assign to that
  ///    speaker and update its centroid.
  /// 4. If `max_speakers` is set and the cap is reached, apply
  ///    `overflow_strategy`:
  ///    - `AssignClosest`: assign to argmax speaker, bump count only —
  ///      **no centroid update**.
  ///    - `Reject`: return `Err(Error::TooManySpeakers { cap })`.
  /// 5. Otherwise open a new speaker slot.
  pub fn submit(&mut self, embedding: &Embedding) -> Result<ClusterAssignment, Error> {
    // ── path 1: first speaker ─────────────────────────────────────────
    if self.speakers.is_empty() {
      self.speakers.push(SpeakerEntry {
        speaker_id: 0,
        accumulator: embedding.0,
        cached_centroid: embedding.0,
        assignment_count: 1,
      });
      self.next_speaker_id = 1;
      return Ok(ClusterAssignment {
        speaker_id: 0,
        is_new_speaker: true,
        similarity: None,
      });
    }

    // ── path 2+: argmax similarity ────────────────────────────────────
    // Dot product on unit vectors = cosine similarity.
    // Lowest-index tie-break: strict `>` keeps the first best found.
    let (best_idx, best_sim) = {
      let mut best_idx = 0usize;
      let mut best_sim = f32::NEG_INFINITY;
      for (i, s) in self.speakers.iter().enumerate() {
        let sim: f32 = s
          .cached_centroid
          .iter()
          .zip(embedding.0.iter())
          .map(|(a, b)| a * b)
          .sum();
        if sim > best_sim {
          best_sim = sim;
          best_idx = i;
        }
      }
      (best_idx, best_sim)
    };

    let threshold = self.opts.similarity_threshold();

    if best_sim >= threshold {
      // ── path 3: assign to best match ─────────────────────────────
      self.update_speaker(best_idx, embedding);
      return Ok(ClusterAssignment {
        speaker_id: self.speakers[best_idx].speaker_id,
        is_new_speaker: false,
        similarity: Some(best_sim),
      });
    }

    // ── path 4/5: below threshold — check overflow ───────────────────
    if let Some(cap) = self.opts.max_speakers()
      && self.speakers.len() as u32 >= cap
    {
      match self.opts.overflow_strategy() {
        OverflowStrategy::AssignClosest => {
          // Force-assign WITHOUT updating centroid (spec §5.4).
          // Updating would pull the centroid toward an outlier.
          self.speakers[best_idx].assignment_count += 1;
          return Ok(ClusterAssignment {
            speaker_id: self.speakers[best_idx].speaker_id,
            is_new_speaker: false,
            similarity: Some(best_sim),
          });
        }
        OverflowStrategy::Reject => {
          return Err(Error::TooManySpeakers { cap });
        }
      }
    }

    // ── path 5: open a new slot ───────────────────────────────────────
    let new_id = self.next_speaker_id;
    self.next_speaker_id += 1;
    self.speakers.push(SpeakerEntry {
      speaker_id: new_id,
      accumulator: embedding.0,
      cached_centroid: embedding.0,
      assignment_count: 1,
    });
    Ok(ClusterAssignment {
      speaker_id: new_id,
      is_new_speaker: true,
      similarity: Some(best_sim),
    })
  }

  // ── helpers ────────────────────────────────────────────────────────────

  /// Update `entry`'s accumulator and refresh `cached_centroid` lazily.
  ///
  /// Always updates the accumulator. Refreshes `cached_centroid` only when
  /// `||accumulator|| >= NORM_EPSILON`; otherwise leaves the cached centroid
  /// at its last-good value (spec §5.4 — handles antipodal cancellation).
  fn update_speaker(&mut self, idx: usize, e: &Embedding) {
    let entry = &mut self.speakers[idx];
    match self.opts.update_strategy() {
      UpdateStrategy::RollingMean => {
        for (a, x) in entry.accumulator.iter_mut().zip(e.0.iter()) {
          *a += *x;
        }
      }
      UpdateStrategy::Ema(alpha) => {
        let one_minus = 1.0 - alpha;
        for (a, x) in entry.accumulator.iter_mut().zip(e.0.iter()) {
          *a = one_minus * *a + alpha * *x;
        }
      }
    }
    entry.assignment_count += 1;

    // Refresh cached_centroid from accumulator IFF norm >= eps.
    // Otherwise leave at last-known-good value (spec §5.4).
    //
    // f64 accumulator: 256 squared-f32 terms can lose ~8 bits of mantissa
    // in f32 (sum of values ~1.0). Promote for stability, demote at the
    // end. Not perf-critical — runs once per assignment.
    let sq: f64 = entry
      .accumulator
      .iter()
      .map(|&a| (a as f64) * (a as f64))
      .sum();
    let n = sq.sqrt() as f32;
    if n >= NORM_EPSILON {
      let inv_n = n.recip();
      for (c, a) in entry
        .cached_centroid
        .iter_mut()
        .zip(entry.accumulator.iter())
      {
        *c = *a * inv_n;
      }
    }
    // else: cached_centroid retains its prior value.
  }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  fn unit(i: usize) -> Embedding {
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[i] = 1.0;
    Embedding::normalize_from(v).unwrap()
  }

  // ── Task 11 tests ──────────────────────────────────────────────────────

  #[test]
  fn first_submission_is_speaker_zero() {
    let mut c = Clusterer::new(ClusterOptions::default());
    let a = c.submit(&unit(0)).unwrap();
    assert_eq!(a.speaker_id(), 0);
    assert!(a.is_new_speaker());
    assert_eq!(a.similarity(), None);
    assert_eq!(c.num_speakers(), 1);
  }

  #[test]
  fn clear_resets_speaker_id() {
    let mut c = Clusterer::new(ClusterOptions::default());
    let _ = c.submit(&unit(0));
    c.clear();
    assert_eq!(c.num_speakers(), 0);
    let a = c.submit(&unit(0)).unwrap();
    assert_eq!(a.speaker_id(), 0); // restarts at 0
  }

  // ── Task 12 tests ──────────────────────────────────────────────────────

  #[test]
  fn second_similar_submission_assigned_same_speaker() {
    let mut c = Clusterer::new(ClusterOptions::default());
    let _ = c.submit(&unit(0));
    // Same direction, slightly perturbed but still cosine ≈ 1.
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = 0.99;
    v[1] = 0.01;
    let e = Embedding::normalize_from(v).unwrap();
    let a = c.submit(&e).unwrap();
    assert_eq!(a.speaker_id(), 0);
    assert!(!a.is_new_speaker());
    assert!(a.similarity().unwrap() > 0.5);
  }

  #[test]
  fn dissimilar_submission_spawns_new_speaker() {
    let mut c = Clusterer::new(ClusterOptions::default());
    let _ = c.submit(&unit(0));
    let a = c.submit(&unit(1)).unwrap(); // orthogonal, sim = 0 < 0.5
    assert_eq!(a.speaker_id(), 1);
    assert!(a.is_new_speaker());
    assert_eq!(a.similarity(), Some(0.0)); // not None — there was a prior speaker
    assert_eq!(c.num_speakers(), 2);
  }

  #[test]
  fn ema_update_changes_centroid() {
    let mut c = Clusterer::new(ClusterOptions::default());
    let _ = c.submit(&unit(0));
    // Submit a slightly off-axis embedding that still goes to speaker 0.
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = 0.99;
    v[1] = 0.14; // sim ≈ 0.99 > 0.5
    let e2 = Embedding::normalize_from(v).unwrap();
    let _ = c.submit(&e2).unwrap();
    let s0 = c.speaker(0).unwrap();
    // Centroid should have moved off the unit-x axis toward the new direction.
    assert!(s0.centroid().as_array()[1] > 0.0);
  }

  #[test]
  fn overflow_reject_returns_error() {
    let mut c = Clusterer::new(ClusterOptions::default().with_max_speakers(1));
    let _ = c.submit(&unit(0));
    let r = c.submit(&unit(1)); // orthogonal → would spawn new but cap=1
    assert!(matches!(r, Err(Error::TooManySpeakers { cap: 1 })));
  }

  #[test]
  fn overflow_assign_closest_no_centroid_update() {
    let mut c = Clusterer::new(
      ClusterOptions::default()
        .with_max_speakers(1)
        .with_overflow_strategy(OverflowStrategy::AssignClosest),
    );
    let _ = c.submit(&unit(0));
    let centroid_before = *c.speaker(0).unwrap().centroid().as_array();
    let r = c.submit(&unit(1)).unwrap();
    assert_eq!(r.speaker_id(), 0); // forced to existing speaker
    assert!(!r.is_new_speaker());
    let centroid_after = *c.speaker(0).unwrap().centroid().as_array();
    assert_eq!(
      centroid_before, centroid_after,
      "AssignClosest must NOT update centroid"
    );
    assert_eq!(c.speaker(0).unwrap().assignment_count(), 2);
  }

  #[test]
  fn argmax_tie_break_lowest_speaker_id_wins() {
    // Spec §5.4: when multiple centroids share the maximum similarity to
    // the query, the lowest-index speaker wins (`>` strict in argmax).
    let mut c = Clusterer::new(ClusterOptions::default());
    let _ = c.submit(&unit(0));
    let _ = c.submit(&unit(1));
    // Query = (1/√2)·unit(0) + (1/√2)·unit(1). Cosine similarity to
    // both centroids is exactly 1/√2 — a perfect tie. Lower-index
    // (speaker 0) must win.
    let mut v = [0.0f32; EMBEDDING_DIM];
    let s = core::f32::consts::FRAC_1_SQRT_2;
    v[0] = s;
    v[1] = s;
    let e = Embedding::normalize_from(v).unwrap();
    let a = c.submit(&e).unwrap();
    assert_eq!(
      a.speaker_id(),
      0,
      "tie-break must pick lower-index speaker id"
    );
    assert!(!a.is_new_speaker());
    let sim = a.similarity().expect("not the first-ever assignment");
    assert!(
      (sim - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
      "expected cosine ≈ 1/√2; got {sim}"
    );
  }

  #[test]
  fn antipodal_submission_within_speaker_does_not_panic() {
    // Spec §5.4 cached_centroid lazy-update: submit e, then -e to
    // the same speaker via threshold tweak. Cached centroid stays
    // at e (last good value).
    let mut c = Clusterer::new(
      ClusterOptions::default()
        .with_similarity_threshold(-1.0)
        .with_update_strategy(UpdateStrategy::RollingMean),
    );
    let e = unit(0);
    let _ = c.submit(&e).unwrap();
    let mut neg = [0.0f32; EMBEDDING_DIM];
    neg[0] = -1.0;
    let neg = Embedding::normalize_from(neg).unwrap();
    let _ = c.submit(&neg).unwrap(); // both go to speaker 0 (threshold = -1)
    // Accumulator is now ≈ [0; 256]. Cached centroid should NOT
    // be NaN — it's preserved as the previous good value.
    let s0 = c.speaker(0).unwrap();
    for x in s0.centroid().as_array() {
      assert!(
        x.is_finite(),
        "centroid component went NaN: {:?}",
        s0.centroid()
      );
    }
  }

  // ── Task 13 property tests ─────────────────────────────────────────────

  #[test]
  fn rolling_mean_accumulator_magnitude_bounded() {
    // Property (spec §9): for any sequence of Clusterer::submit
    // calls under RollingMean, after N assignments the accumulator
    // satisfies ||accumulator||₂ <= N (triangle inequality on a
    // sum of N unit vectors).
    let mut c = Clusterer::new(
      ClusterOptions::default()
        .with_update_strategy(UpdateStrategy::RollingMean)
        .with_similarity_threshold(-1.0), // force assignment to speaker 0
    );
    let n = 100;
    for i in 0..n {
      let mut v = [0.0f32; EMBEDDING_DIM];
      v[i % EMBEDDING_DIM] = 1.0;
      let e = Embedding::normalize_from(v).unwrap();
      c.submit(&e).unwrap();
    }
    let s0 = &c.speakers[0];
    let mut sq = 0.0f64;
    for k in 0..EMBEDDING_DIM {
      sq += (s0.accumulator[k] as f64) * (s0.accumulator[k] as f64);
    }
    let norm = sq.sqrt();
    assert!(
      norm <= n as f64,
      "||accumulator|| = {} exceeds N = {}",
      norm,
      n
    );
  }

  #[test]
  fn similarity_field_invariant_first_only_none() {
    // Spec §4.3: ClusterAssignment::similarity is None iff this is
    // the first-ever assignment in the Clusterer's lifetime.
    let mut c = Clusterer::new(ClusterOptions::default());
    let a0 = c.submit(&unit(0)).unwrap();
    assert_eq!(a0.similarity(), None);
    let a1 = c.submit(&unit(1)).unwrap();
    assert!(a1.similarity().is_some());
  }
}
