//! Builder + options for [`crate::diarizer::Diarizer`]. Spec §4.4.

use crate::{cluster::ClusterOptions, segment::SegmentOptions};

/// Configuration for [`crate::diarizer::Diarizer`].
///
/// All fields have sensible defaults. The two embedded option types
/// ([`SegmentOptions`] and [`ClusterOptions`]) themselves expose the
/// detailed knobs for the Layer-1 segmentation and the online
/// clustering — `DiarizerOptions` only carries the diarizer-specific
/// orchestration knobs.
#[derive(Debug, Clone)]
pub struct DiarizerOptions {
  pub(crate) segment: SegmentOptions,
  pub(crate) cluster: ClusterOptions,
  /// Retain per-activity context across the session for
  /// `Diarizer::collected_embeddings()`. Default: `true`.
  pub(crate) collect_embeddings: bool,
  /// Onset threshold for binarizing per-frame per-speaker raw
  /// probabilities. Used by §5.8 (`exclude_overlap`) and §5.10
  /// (speaker count). Default: 0.5 (matches pyannote's
  /// `Binarize(onset=...)` default).
  pub(crate) binarize_threshold: f32,
  /// Apply the `exclude_overlap` clean mask when extracting
  /// embeddings (spec §5.8). Default: `true`.
  pub(crate) exclude_overlap: bool,
}

impl Default for DiarizerOptions {
  fn default() -> Self {
    Self {
      segment: SegmentOptions::default(),
      cluster: ClusterOptions::default(),
      collect_embeddings: true,
      binarize_threshold: 0.5,
      exclude_overlap: true,
    }
  }
}

impl DiarizerOptions {
  /// Construct an instance with all defaults.
  pub fn new() -> Self {
    Self::default()
  }

  // ── Accessors ────────────────────────────────────────────────────────

  /// Borrow the embedded [`SegmentOptions`].
  pub fn segment_options(&self) -> &SegmentOptions {
    &self.segment
  }

  /// Borrow the embedded [`ClusterOptions`].
  pub fn cluster_options(&self) -> &ClusterOptions {
    &self.cluster
  }

  /// Whether per-`(window, slot)` embeddings are retained for
  /// `Diarizer::collected_embeddings()`.
  pub fn collect_embeddings(&self) -> bool {
    self.collect_embeddings
  }

  /// Onset threshold for binarizing per-frame raw probabilities.
  pub fn binarize_threshold(&self) -> f32 {
    self.binarize_threshold
  }

  /// Whether `exclude_overlap` masking is applied when extracting
  /// embeddings.
  pub fn exclude_overlap(&self) -> bool {
    self.exclude_overlap
  }

  // ── Builder (consuming with_*) ────────────────────────────────────────

  /// Replace the embedded [`SegmentOptions`].
  pub fn with_segment_options(mut self, opts: SegmentOptions) -> Self {
    self.segment = opts;
    self
  }

  /// Replace the embedded [`ClusterOptions`].
  pub fn with_cluster_options(mut self, opts: ClusterOptions) -> Self {
    self.cluster = opts;
    self
  }

  /// Toggle retention of per-`(window, slot)` embeddings.
  pub fn with_collect_embeddings(mut self, on: bool) -> Self {
    self.collect_embeddings = on;
    self
  }

  /// Set the onset threshold for binarizing per-frame raw probabilities.
  pub fn with_binarize_threshold(mut self, t: f32) -> Self {
    self.binarize_threshold = t;
    self
  }

  /// Toggle the `exclude_overlap` clean mask path.
  pub fn with_exclude_overlap(mut self, on: bool) -> Self {
    self.exclude_overlap = on;
    self
  }

  // ── Mutators (in-place set_*) ────────────────────────────────────────

  /// Replace the embedded [`SegmentOptions`] in place.
  pub fn set_segment_options(&mut self, opts: SegmentOptions) -> &mut Self {
    self.segment = opts;
    self
  }

  /// Replace the embedded [`ClusterOptions`] in place.
  pub fn set_cluster_options(&mut self, opts: ClusterOptions) -> &mut Self {
    self.cluster = opts;
    self
  }

  /// Toggle retention of per-`(window, slot)` embeddings in place.
  pub fn set_collect_embeddings(&mut self, on: bool) -> &mut Self {
    self.collect_embeddings = on;
    self
  }

  /// Set the onset threshold in place.
  pub fn set_binarize_threshold(&mut self, t: f32) -> &mut Self {
    self.binarize_threshold = t;
    self
  }

  /// Toggle the `exclude_overlap` clean mask path in place.
  pub fn set_exclude_overlap(&mut self, on: bool) -> &mut Self {
    self.exclude_overlap = on;
    self
  }
}

/// Builder for [`DiarizerOptions`].
///
/// `with_*` setters only — no `options(opts)` override (avoids the
/// rev-1 API inconsistency where the builder could overwrite earlier
/// `with_*` calls).
///
/// Note: `build()` returns [`DiarizerOptions`], NOT `Diarizer` —
/// `Diarizer::new(opts)` is the actual constructor. This keeps the
/// builder zero-cost when callers want to round-trip options
/// (e.g., for serde).
#[derive(Debug, Clone, Default)]
pub struct DiarizerBuilder {
  opts: DiarizerOptions,
}

impl DiarizerBuilder {
  /// Construct a builder with default option values.
  pub fn new() -> Self {
    Self::default()
  }

  /// Replace the embedded [`SegmentOptions`].
  pub fn with_segment_options(mut self, o: SegmentOptions) -> Self {
    self.opts.segment = o;
    self
  }

  /// Replace the embedded [`ClusterOptions`].
  pub fn with_cluster_options(mut self, o: ClusterOptions) -> Self {
    self.opts.cluster = o;
    self
  }

  /// Toggle retention of per-`(window, slot)` embeddings.
  pub fn with_collect_embeddings(mut self, on: bool) -> Self {
    self.opts.collect_embeddings = on;
    self
  }

  /// Set the onset threshold for binarizing per-frame raw probabilities.
  pub fn with_binarize_threshold(mut self, t: f32) -> Self {
    self.opts.binarize_threshold = t;
    self
  }

  /// Toggle the `exclude_overlap` clean mask path.
  pub fn with_exclude_overlap(mut self, on: bool) -> Self {
    self.opts.exclude_overlap = on;
    self
  }

  /// Finalize into a [`DiarizerOptions`] value.
  pub fn build(self) -> DiarizerOptions {
    self.opts
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_values() {
    let o = DiarizerOptions::default();
    assert!(o.collect_embeddings());
    assert!((o.binarize_threshold() - 0.5).abs() < 1e-7);
    assert!(o.exclude_overlap());
  }

  #[test]
  fn builder_round_trip() {
    let o = DiarizerBuilder::new()
      .with_collect_embeddings(false)
      .with_binarize_threshold(0.7)
      .with_exclude_overlap(false)
      .build();
    assert!(!o.collect_embeddings());
    assert!((o.binarize_threshold() - 0.7).abs() < 1e-7);
    assert!(!o.exclude_overlap());
  }

  #[test]
  fn set_methods_mutate_in_place() {
    let mut o = DiarizerOptions::default();
    o.set_binarize_threshold(0.4).set_collect_embeddings(false);
    assert!((o.binarize_threshold() - 0.4).abs() < 1e-7);
    assert!(!o.collect_embeddings());
  }
}
