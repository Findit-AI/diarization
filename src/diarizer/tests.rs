//! Cross-component diarizer tests. Spec §9.

use super::*;

#[test]
fn new_diarizer_default_state() {
  let d = Diarizer::new(DiarizerOptions::default());
  assert_eq!(d.pending_inferences(), 0);
  assert_eq!(d.buffered_samples(), 0);
  assert_eq!(d.total_samples_pushed(), 0);
  assert_eq!(d.num_speakers(), 0);
  assert!(d.collected_embeddings().is_empty());
}

#[test]
fn builder_round_trip() {
  let opts = Diarizer::builder()
    .with_collect_embeddings(false)
    .with_binarize_threshold(0.6)
    .build();
  assert!(!opts.collect_embeddings());
  assert!((opts.binarize_threshold() - 0.6).abs() < 1e-7);
}

#[test]
fn clear_resets_state_but_preserves_collected() {
  let mut d = Diarizer::new(DiarizerOptions::default());
  // total_samples_pushed bumps via push_audio (added in Task 36).
  // For Task 35 just verify clear() doesn't panic on a fresh instance.
  d.clear();
  assert_eq!(d.total_samples_pushed(), 0);
}
