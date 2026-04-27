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
fn clear_resets_collected_embeddings_too() {
  // Codex review post-rev-9 HIGH: clear() must drop collected_embeddings
  // for tenant-isolation safety in pooled diarizers. Behaviour change
  // from rev-9 (which preserved them on the assumption that callers
  // would offline-re-cluster after clear()).
  let mut d = Diarizer::new(DiarizerOptions::default());
  d.collected_embeddings
    .push(crate::diarizer::CollectedEmbedding {
      range: mediatime::TimeRange::new(0, 32_000, crate::segment::options::SAMPLE_RATE_TB),
      embedding: crate::embed::Embedding::normalize_from({
        let mut v = [0.0f32; 256];
        v[0] = 1.0;
        v
      })
      .unwrap(),
      online_speaker_id: 0,
      speaker_slot: 0,
      used_clean_mask: true,
    });
  assert_eq!(d.collected_embeddings().len(), 1);
  d.clear();
  assert_eq!(d.total_samples_pushed(), 0);
  assert!(
    d.collected_embeddings().is_empty(),
    "clear() must drop collected_embeddings (Codex review fix)"
  );
}

#[test]
fn take_collected_returns_and_drops() {
  let mut d = Diarizer::new(DiarizerOptions::default());
  // The collected_embeddings field is pub(crate); we can populate
  // it directly from inside the diarizer module's tests.
  d.collected_embeddings
    .push(crate::diarizer::CollectedEmbedding {
      range: mediatime::TimeRange::new(0, 32_000, crate::segment::options::SAMPLE_RATE_TB),
      embedding: crate::embed::Embedding::normalize_from({
        let mut v = [0.0f32; 256];
        v[0] = 1.0;
        v
      })
      .unwrap(),
      online_speaker_id: 0,
      speaker_slot: 0,
      used_clean_mask: true,
    });
  assert_eq!(d.collected_embeddings().len(), 1);
  let taken = d.take_collected();
  assert_eq!(taken.len(), 1);
  assert!(d.collected_embeddings().is_empty());
}

#[test]
#[ignore = "requires segment + embed ONNX models"]
fn process_samples_skeleton_pumps_segment_without_panicking() {
  use crate::{embed::EmbedModel, segment::SegmentModel};
  use std::path::PathBuf;

  let seg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/segmentation-3.0.onnx");
  let embed_path =
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx");
  if !seg_path.exists() || !embed_path.exists() {
    return;
  }

  let mut d = Diarizer::new(DiarizerOptions::default());
  let mut seg = SegmentModel::from_file(&seg_path).expect("segment model");
  let mut emb = EmbedModel::from_file(&embed_path).expect("embed model");
  let samples = vec![0.001f32; 320_000]; // 20 s of near-silence.
  d.process_samples(&mut seg, &mut emb, &samples, |_span| {})
    .expect("pump completes without panicking");

  assert_eq!(d.total_samples_pushed(), 320_000);
  // Phase 11 will assert spans emitted; for now, success means no panic.
}

#[cfg(feature = "ort")]
#[test]
#[ignore = "requires both ONNX models"]
fn end_to_end_pump_runs_without_panicking() {
  use std::path::PathBuf;

  use crate::{embed::EmbedModel, segment::SegmentModel};

  let seg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/segmentation-3.0.onnx");
  let embed_path =
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx");
  if !seg_path.exists() || !embed_path.exists() {
    return; // skip if models aren't available
  }

  let mut d = Diarizer::new(DiarizerOptions::default());
  let mut seg = SegmentModel::from_file(&seg_path).expect("segment model");
  let mut emb = EmbedModel::from_file(&embed_path).expect("embed model");

  // 30 s of low-amplitude noise. Real diarization tests use real audio
  // (Task 44+); this verifies the wiring runs without panicking on
  // synthetic input. With no clear speakers, we expect either zero
  // spans (everything below threshold) or a small number on the noise.
  let samples = vec![0.001f32; 16_000 * 30];
  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &samples, |span| spans.push(span))
    .expect("process_samples completes");
  d.finish_stream(&mut seg, &mut emb, |span| spans.push(span))
    .expect("finish_stream completes");

  assert_eq!(d.total_samples_pushed(), 16_000 * 30);
  // Don't assert on span count — synthetic noise is below the segment
  // model's voice-detection threshold; spans count is 0 or small.
  eprintln!(
    "synthetic-input pump: {} spans, {} speakers, {} collected",
    spans.len(),
    d.num_speakers(),
    d.collected_embeddings().len()
  );
}
