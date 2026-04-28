//! Constants and option types for `dia::cluster`.

// ── Constants ────────────────────────────────────────────────────────────────

/// Cosine-similarity threshold an incoming embedding must meet (or exceed)
/// against the best existing centroid to be assigned to that speaker;
/// embeddings whose best similarity falls *below* the threshold open a
/// new speaker slot (subject to `max_speakers` / `overflow_strategy`).
///
/// Practical range: `[-1.0, 1.0]` (the full cosine range). Setters
/// reject NaN / `±inf` and values outside this range; passing `-1.0`
/// is permitted and forces every embedding to merge with the nearest
/// centroid. Higher thresholds are stricter — fewer embeddings clear
/// the bar to be merged with an existing centroid, so MORE new
/// speakers get opened (and the clusterer fragments more). Lower
/// thresholds are more lenient — most embeddings merge with the
/// nearest existing centroid, so FEWER new speakers get opened (and
/// distinct speakers may collapse together).
///
/// The implementation in [`crate::cluster::Clusterer::submit`] uses
/// `best_sim >= threshold ⇒ assign to existing`, otherwise open new.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

/// Range check for any `similarity_threshold` setter. Centralized so
/// online and offline option types validate identically. Codex review
/// MEDIUM.
#[inline]
fn validate_similarity_threshold(v: f32) {
  assert!(
    v.is_finite() && (-1.0..=1.0).contains(&v),
    "similarity_threshold must be finite in [-1.0, 1.0]; got {v}"
  );
}

/// Range check for an `UpdateStrategy::Ema(alpha)` setter. Codex review
/// MEDIUM.
#[inline]
fn validate_update_strategy(s: UpdateStrategy) {
  if let UpdateStrategy::Ema(alpha) = s {
    assert!(
      alpha.is_finite() && alpha > 0.0 && alpha <= 1.0,
      "UpdateStrategy::Ema(alpha) requires finite alpha in (0.0, 1.0]; got {alpha}"
    );
  }
}

/// Default α for the Exponential Moving Average centroid update rule.
///
/// `centroid_new = (1 − α) * centroid_old + α * embedding`
///
/// Range: `(0.0, 1.0]`. Smaller values give older observations more weight.
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;

/// Hard upper bound on the auto-detected speaker count used by
/// [`cluster_offline`](crate::cluster::cluster_offline) when
/// [`OfflineClusterOptions::target_speakers`] is `None` (spec §4.3, §5.5).
/// Has no effect on the online [`Clusterer`](crate::cluster::Clusterer).
pub const MAX_AUTO_SPEAKERS: u32 = 15;

/// Hard upper bound on the number of input embeddings accepted by
/// [`cluster_offline`](crate::cluster::cluster_offline). Reached →
/// [`Error::InputTooLarge`](crate::cluster::Error::InputTooLarge).
///
/// Both supported offline methods allocate dense `N × N` matrices:
/// spectral builds the affinity matrix and computes eigendecomposition
/// (`O(N³)` time, `O(N²)` memory); average-/complete-/single-linkage
/// agglomerative builds the same affinity. At `N = 5_000`,
/// `5_000² × 4 B ≈ 100 MB` for `f32` and ~200 MB for the f64 affinity
/// matrix used by spectral — already well into "noticeable" territory.
/// The cap below is a defense-in-depth bound; callers reclustering
/// long sessions should down-sample to a representative subset rather
/// than feeding raw per-activity embeddings. Codex review MEDIUM.
pub const MAX_OFFLINE_INPUT: usize = 5_000;

// ── Online clustering options ─────────────────────────────────────────────

/// How the per-speaker centroid is updated each time an embedding is assigned
/// to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateStrategy {
  /// Maintain an unweighted rolling mean:
  /// `centroid_new = (centroid_old * n + embedding) / (n + 1)`
  RollingMean,
  /// Exponential Moving Average with the given α ∈ `(0, 1]`:
  /// `centroid_new = (1 − α) * centroid_old + α * embedding`
  Ema(f32),
}

impl Default for UpdateStrategy {
  fn default() -> Self {
    Self::Ema(DEFAULT_EMA_ALPHA)
  }
}

/// What to do when the number of open speaker slots has reached
/// [`ClusterOptions::max_speakers`] and a new embedding does not match any
/// existing speaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverflowStrategy {
  /// Reject the embedding and return
  /// [`Error::TooManySpeakers`](crate::cluster::Error::TooManySpeakers).
  /// This is the default.
  #[default]
  Reject,
  /// Assign the embedding to the closest existing speaker even if it does
  /// not meet the similarity threshold. **Does not update the centroid.**
  AssignClosest,
}

/// Runtime options for the online [`Clusterer`](crate::cluster::Clusterer).
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterOptions {
  similarity_threshold: f32,
  update_strategy: UpdateStrategy,
  max_speakers: Option<u32>,
  overflow_strategy: OverflowStrategy,
}

impl Default for ClusterOptions {
  fn default() -> Self {
    Self {
      similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
      update_strategy: UpdateStrategy::default(),
      max_speakers: None,
      overflow_strategy: OverflowStrategy::default(),
    }
  }
}

impl ClusterOptions {
  /// Construct with all defaults.
  pub fn new() -> Self {
    Self::default()
  }

  // ── Accessors ──────────────────────────────────────────────────────────

  /// Cosine-similarity threshold for opening a new speaker slot.
  pub fn similarity_threshold(&self) -> f32 {
    self.similarity_threshold
  }

  /// Centroid update rule applied on each assignment.
  pub fn update_strategy(&self) -> UpdateStrategy {
    self.update_strategy
  }

  /// Maximum number of auto-assigned speaker slots, or `None` for no cap.
  pub fn max_speakers(&self) -> Option<u32> {
    self.max_speakers
  }

  /// Behaviour when `max_speakers` is reached.
  pub fn overflow_strategy(&self) -> OverflowStrategy {
    self.overflow_strategy
  }

  // ── Builder (consuming with_*) ─────────────────────────────────────────

  /// Set the similarity threshold (builder).
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `[-1.0, 1.0]`. Codex review MEDIUM.
  pub fn with_similarity_threshold(mut self, v: f32) -> Self {
    validate_similarity_threshold(v);
    self.similarity_threshold = v;
    self
  }

  /// Set the update strategy (builder).
  ///
  /// # Panics
  /// Panics if the strategy is `Ema(alpha)` with non-finite alpha or
  /// alpha outside `(0.0, 1.0]`. Codex review MEDIUM.
  pub fn with_update_strategy(mut self, v: UpdateStrategy) -> Self {
    validate_update_strategy(v);
    self.update_strategy = v;
    self
  }

  /// Set the max-speakers cap (builder).
  ///
  /// # Panics
  /// Panics if `v == 0`. A cap of zero would forbid every assignment
  /// (no slot can ever be opened) and is almost certainly a bug — use
  /// `None` (the default) to opt out of the cap. Codex review MEDIUM.
  pub fn with_max_speakers(mut self, v: u32) -> Self {
    assert!(v > 0, "max_speakers must be > 0; pass `None` for no cap");
    self.max_speakers = Some(v);
    self
  }

  /// Set the overflow strategy (builder).
  pub fn with_overflow_strategy(mut self, v: OverflowStrategy) -> Self {
    self.overflow_strategy = v;
    self
  }

  // ── Mutators (in-place set_*) ───────────────────────────────────────────

  /// Set the similarity threshold (in-place).
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `[-1.0, 1.0]`. Codex review MEDIUM.
  pub fn set_similarity_threshold(&mut self, v: f32) -> &mut Self {
    validate_similarity_threshold(v);
    self.similarity_threshold = v;
    self
  }

  /// Set the update strategy (in-place).
  ///
  /// # Panics
  /// Panics if the strategy is `Ema(alpha)` with non-finite alpha or
  /// alpha outside `(0.0, 1.0]`. Codex review MEDIUM.
  pub fn set_update_strategy(&mut self, v: UpdateStrategy) -> &mut Self {
    validate_update_strategy(v);
    self.update_strategy = v;
    self
  }

  /// Set the max-speakers cap (in-place).
  ///
  /// # Panics
  /// Panics if `v == 0`. Codex review MEDIUM.
  pub fn set_max_speakers(&mut self, v: u32) -> &mut Self {
    assert!(v > 0, "max_speakers must be > 0; pass `None` for no cap");
    self.max_speakers = Some(v);
    self
  }

  /// Set the overflow strategy (in-place).
  pub fn set_overflow_strategy(&mut self, v: OverflowStrategy) -> &mut Self {
    self.overflow_strategy = v;
    self
  }
}

// ── Offline clustering options ────────────────────────────────────────────

/// HAC linkage criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Linkage {
  /// Nearest-neighbour linkage (minimum pairwise distance).
  Single,
  /// Farthest-neighbour linkage (maximum pairwise distance).
  Complete,
  /// Average pairwise distance (UPGMA).
  #[default]
  Average,
}

/// Offline clustering algorithm.
///
/// **Threshold semantics differ by variant** — `similarity_threshold` is
/// consumed by some methods and ignored by others (Codex review MEDIUM):
///
/// | Variant            | Reads `similarity_threshold` |
/// |--------------------|------------------------------|
/// | `Agglomerative {..}` | Yes — used as the merge stop criterion (`stop_dist = 1 - threshold`). |
/// | `Spectral`           | **No** — K is chosen from `target_speakers` or the eigengap heuristic. |
///
/// The N==1 / N==2 fast paths in
/// [`cluster_offline`](crate::cluster::cluster_offline) consult
/// `similarity_threshold` regardless of method.
///
/// If you switch to [`Spectral`](Self::Spectral) (the default) and rely
/// on tuning the threshold, your output will not change. Either pin
/// `target_speakers`, switch to [`Agglomerative`](Self::Agglomerative),
/// or open an issue if you need threshold-driven K selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OfflineMethod {
  /// Agglomerative Hierarchical Clustering with the given linkage.
  Agglomerative {
    /// The HAC linkage criterion.
    linkage: Linkage,
  },
  /// Spectral clustering (filled in by Phase 4).
  #[default]
  Spectral,
}

/// Options for the offline batch [`cluster_offline`](crate::cluster::cluster_offline) function.
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineClusterOptions {
  method: OfflineMethod,
  similarity_threshold: f32,
  target_speakers: Option<u32>,
  seed: Option<u64>,
}

impl Default for OfflineClusterOptions {
  fn default() -> Self {
    Self {
      method: OfflineMethod::default(),
      similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
      target_speakers: None,
      seed: None,
    }
  }
}

impl OfflineClusterOptions {
  /// Construct with all defaults.
  pub fn new() -> Self {
    Self::default()
  }

  // ── Accessors ──────────────────────────────────────────────────────────

  /// The offline clustering algorithm.
  pub fn method(&self) -> OfflineMethod {
    self.method
  }

  /// Cosine-similarity threshold used by the algorithm.
  ///
  /// **Not all [`OfflineMethod`] variants consume this.** See
  /// [`OfflineMethod`] for the per-variant table. Notably,
  /// [`OfflineMethod::Spectral`] (the default) ignores it for
  /// `N >= 3`. Codex review MEDIUM.
  pub fn similarity_threshold(&self) -> f32 {
    self.similarity_threshold
  }

  /// Target number of speaker clusters, or `None` for automatic.
  pub fn target_speakers(&self) -> Option<u32> {
    self.target_speakers
  }

  /// Optional RNG seed for reproducibility.
  pub fn seed(&self) -> Option<u64> {
    self.seed
  }

  // ── Builder (consuming with_*) ─────────────────────────────────────────

  /// Set the algorithm (builder).
  pub fn with_method(mut self, m: OfflineMethod) -> Self {
    self.method = m;
    self
  }

  /// Set the similarity threshold (builder).
  ///
  /// # Panics
  /// Panics if `t` is NaN/±inf or outside `[-1.0, 1.0]`. Codex review MEDIUM.
  pub fn with_similarity_threshold(mut self, t: f32) -> Self {
    validate_similarity_threshold(t);
    self.similarity_threshold = t;
    self
  }

  /// Set the target speaker count (builder).
  ///
  /// `n == 0` is accepted at this layer for API symmetry — it is
  /// rejected by [`cluster_offline`](crate::cluster::cluster_offline)
  /// with [`Error::TargetTooSmall`](crate::cluster::Error::TargetTooSmall)
  /// rather than panicking, so callers can store the option and
  /// surface the validation error themselves.
  pub fn with_target_speakers(mut self, n: u32) -> Self {
    self.target_speakers = Some(n);
    self
  }

  /// Set the RNG seed (builder).
  pub fn with_seed(mut self, s: u64) -> Self {
    self.seed = Some(s);
    self
  }

  // ── Mutators (in-place set_*) ───────────────────────────────────────────

  /// Set the algorithm (in-place).
  pub fn set_method(&mut self, m: OfflineMethod) -> &mut Self {
    self.method = m;
    self
  }

  /// Set the similarity threshold (in-place).
  ///
  /// # Panics
  /// Panics if `t` is NaN/±inf or outside `[-1.0, 1.0]`. Codex review MEDIUM.
  pub fn set_similarity_threshold(&mut self, t: f32) -> &mut Self {
    validate_similarity_threshold(t);
    self.similarity_threshold = t;
    self
  }

  /// Set the target speaker count (in-place).
  ///
  /// `n == 0` is accepted at this layer; see
  /// [`Self::with_target_speakers`] for rationale.
  pub fn set_target_speakers(&mut self, n: u32) -> &mut Self {
    self.target_speakers = Some(n);
    self
  }

  /// Set the RNG seed (in-place).
  pub fn set_seed(&mut self, s: u64) -> &mut Self {
    self.seed = Some(s);
    self
  }
}

#[cfg(test)]
mod validation_tests {
  //! Codex review MEDIUM: setters reject NaN / `±inf` / out-of-range values
  //! so a caller cannot install a configuration that silently corrupts
  //! the online clusterer's hot path (NaN comparisons, runaway accumulator
  //! updates, or zero-cap deadlocks).

  use super::*;

  // ── ClusterOptions (online) ──────────────────────────────────────────

  #[test]
  #[should_panic(expected = "similarity_threshold must be finite in [-1.0, 1.0]")]
  fn online_threshold_nan_panics() {
    let _ = ClusterOptions::new().with_similarity_threshold(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "similarity_threshold must be finite in [-1.0, 1.0]")]
  fn online_threshold_inf_panics() {
    let _ = ClusterOptions::new().with_similarity_threshold(f32::INFINITY);
  }

  #[test]
  #[should_panic(expected = "similarity_threshold must be finite in [-1.0, 1.0]")]
  fn online_threshold_out_of_range_panics() {
    let _ = ClusterOptions::new().with_similarity_threshold(1.5);
  }

  /// Negative threshold is permitted because tests use `-1.0` as a
  /// "force merge anything" sentinel — see
  /// `Clusterer::submit` test sites in `online.rs`.
  #[test]
  fn online_threshold_negative_one_ok() {
    let opts = ClusterOptions::new().with_similarity_threshold(-1.0);
    assert!((opts.similarity_threshold() + 1.0).abs() < 1e-7);
  }

  #[test]
  #[should_panic(expected = "UpdateStrategy::Ema(alpha) requires finite alpha in (0.0, 1.0]")]
  fn online_ema_alpha_nan_panics() {
    let _ = ClusterOptions::new().with_update_strategy(UpdateStrategy::Ema(f32::NAN));
  }

  #[test]
  #[should_panic(expected = "UpdateStrategy::Ema(alpha) requires finite alpha in (0.0, 1.0]")]
  fn online_ema_alpha_zero_panics() {
    // alpha = 0 means "never update" — degenerate and almost certainly
    // a bug. Match the documented (0.0, 1.0] range.
    let _ = ClusterOptions::new().with_update_strategy(UpdateStrategy::Ema(0.0));
  }

  #[test]
  #[should_panic(expected = "UpdateStrategy::Ema(alpha) requires finite alpha in (0.0, 1.0]")]
  fn online_ema_alpha_above_one_panics() {
    let _ = ClusterOptions::new().with_update_strategy(UpdateStrategy::Ema(1.5));
  }

  #[test]
  fn online_update_strategy_rolling_mean_ok_regardless_of_alpha_check() {
    // RollingMean has no alpha, so the validator must not falsely
    // reject it.
    let opts = ClusterOptions::new().with_update_strategy(UpdateStrategy::RollingMean);
    assert_eq!(opts.update_strategy(), UpdateStrategy::RollingMean);
  }

  #[test]
  #[should_panic(expected = "max_speakers must be > 0")]
  fn online_max_speakers_zero_panics() {
    let _ = ClusterOptions::new().with_max_speakers(0);
  }

  #[test]
  #[should_panic(expected = "similarity_threshold must be finite in [-1.0, 1.0]")]
  fn online_set_threshold_in_place_validates() {
    let mut opts = ClusterOptions::new();
    opts.set_similarity_threshold(f32::NAN);
  }

  // ── OfflineClusterOptions ────────────────────────────────────────────

  #[test]
  #[should_panic(expected = "similarity_threshold must be finite in [-1.0, 1.0]")]
  fn offline_threshold_nan_panics() {
    let _ = OfflineClusterOptions::new().with_similarity_threshold(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "similarity_threshold must be finite in [-1.0, 1.0]")]
  fn offline_threshold_neg_inf_panics() {
    let _ = OfflineClusterOptions::new().with_similarity_threshold(f32::NEG_INFINITY);
  }
}
