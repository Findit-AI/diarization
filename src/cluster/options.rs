//! Constants and option types for `diarization::cluster`.

// ── Constants ────────────────────────────────────────────────────────────────

/// Cosine-similarity threshold consumed by
/// [`OfflineMethod::Agglomerative`] as the merge stop criterion
/// (`stop_dist = 1 - threshold`). Range: `[-1.0, 1.0]`.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

/// Range check for any `similarity_threshold` setter.
#[inline]
fn validate_similarity_threshold(v: f32) {
  assert!(
    v.is_finite() && (-1.0..=1.0).contains(&v),
    "similarity_threshold must be finite in [-1.0, 1.0]; got {v}"
  );
}

/// Hard upper bound on the auto-detected speaker count used by
/// [`cluster_offline`](crate::cluster::cluster_offline) when
/// [`OfflineClusterOptions::target_speakers`] is `None` (spec §4.3, §5.5).
pub const MAX_AUTO_SPEAKERS: u32 = 15;

/// Hard upper bound on the number of input embeddings accepted by
/// [`cluster_offline`](crate::cluster::cluster_offline). Reached →
/// [`Error::InputTooLarge`](crate::cluster::Error::InputTooLarge).
///
/// Both supported offline methods allocate dense `N × N` matrices:
/// spectral builds the f64 affinity matrix and runs eigendecomposition
/// (`O(N³)` time, `O(N²)` memory); agglomerative builds the same
/// affinity in f32. At the chosen cap (`N = 1_000`):
///
/// - spectral affinity: `1_000² × 8 B ≈ 8 MB`
/// - intermediate Laplacian + identity: `~16 MB` more
/// - eigendecomposition working memory: another `~10 MB`
///
/// Total memory ≈ tens of MB, eigendecomposition a few seconds on a
/// modern CPU — comfortably within an interactive offline-recluster
/// budget. The previous cap of 5_000 allowed `5_000² × 8 B ≈ 200 MB`
/// per dense matrix and minutes of CPU; that was a documented
/// "defense in depth" bound but not actually safe. Codex review
/// MEDIUM.
///
/// Callers reclustering long sessions should down-sample collected
/// embeddings to a representative subset rather than feed every
/// per-activity embedding back through `cluster_offline`.
pub const MAX_OFFLINE_INPUT: usize = 1_000;

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
  use super::*;

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
