//! Constants and option types for `dia::cluster`.

// ── Constants ────────────────────────────────────────────────────────────────

/// Cosine-similarity threshold below which a new embedding is assigned to the
/// closest existing speaker and above which a new speaker slot is opened.
///
/// Range: `[0.0, 1.0]`. Values closer to 1.0 are more conservative
/// (fewer new speakers); values closer to 0.0 are more permissive.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

/// Default α for the Exponential Moving Average centroid update rule.
///
/// `centroid_new = (1 − α) * centroid_old + α * embedding`
///
/// Range: `(0.0, 1.0]`. Smaller values give older observations more weight.
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;

/// Hard cap on the number of auto-assigned speaker slots used as a fallback
/// when `ClusterOptions::max_speakers` is not `None`.
///
/// Kept for reference; the default for `ClusterOptions::max_speakers` is
/// `None` (no cap).
pub const MAX_AUTO_SPEAKERS: u32 = 15;

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
  pub fn with_similarity_threshold(mut self, v: f32) -> Self {
    self.similarity_threshold = v;
    self
  }

  /// Set the update strategy (builder).
  pub fn with_update_strategy(mut self, v: UpdateStrategy) -> Self {
    self.update_strategy = v;
    self
  }

  /// Set the max-speakers cap (builder).
  pub fn with_max_speakers(mut self, v: u32) -> Self {
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
  pub fn set_similarity_threshold(&mut self, v: f32) -> &mut Self {
    self.similarity_threshold = v;
    self
  }

  /// Set the update strategy (in-place).
  pub fn set_update_strategy(&mut self, v: UpdateStrategy) -> &mut Self {
    self.update_strategy = v;
    self
  }

  /// Set the max-speakers cap (in-place).
  pub fn set_max_speakers(&mut self, v: u32) -> &mut Self {
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
  pub fn with_similarity_threshold(mut self, t: f32) -> Self {
    self.similarity_threshold = t;
    self
  }

  /// Set the target speaker count (builder).
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
  pub fn set_similarity_threshold(&mut self, t: f32) -> &mut Self {
    self.similarity_threshold = t;
    self
  }

  /// Set the target speaker count (in-place).
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
