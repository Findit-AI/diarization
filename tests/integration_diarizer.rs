//! End-to-end integration tests for `dia::Diarizer`.
//!
//! All tests are `#[ignore]` because they require:
//! - The pyannote/segmentation-3.0 ONNX model at
//!   `models/segmentation-3.0.onnx` (or set `DIA_SEGMENT_MODEL_PATH` —
//!   `scripts/download-model.sh` fetches this).
//! - The WeSpeaker ResNet34-LM ONNX at
//!   `models/wespeaker_resnet34_lm.onnx` (or set `DIA_EMBED_MODEL_PATH` —
//!   `scripts/download-embed-model.sh` fetches this).
//! - Optionally `tests/fixtures/diarize_test_30s.wav` for the
//!   end-to-end pump test (`scripts/download-test-fixtures.sh`
//!   generates a synthetic stand-in).
//!
//! Run with:
//!   cargo test --features ort --test integration_diarizer -- --ignored

#![cfg(feature = "ort")]

use std::path::PathBuf;

use dia::{
  diarizer::{Diarizer, DiarizerOptions},
  embed::EmbedModel,
  segment::SegmentModel,
};

fn segment_model_path() -> PathBuf {
  if let Ok(p) = std::env::var("DIA_SEGMENT_MODEL_PATH") {
    return PathBuf::from(p);
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/segmentation-3.0.onnx")
}

fn embed_model_path() -> PathBuf {
  if let Ok(p) = std::env::var("DIA_EMBED_MODEL_PATH") {
    return PathBuf::from(p);
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
}

/// Returns `(segment, embed)` if both models are present; `None` if either is missing.
/// Tests early-return on `None` so they pass cleanly in CI without the models.
fn try_load_models() -> Option<(SegmentModel, EmbedModel)> {
  let seg_path = segment_model_path();
  let emb_path = embed_model_path();
  if !seg_path.exists() || !emb_path.exists() {
    return None;
  }
  let seg = SegmentModel::from_file(&seg_path).expect("segment model loads");
  let emb = EmbedModel::from_file(&emb_path).expect("embed model loads");
  Some((seg, emb))
}

#[test]
#[ignore = "requires both ONNX models"]
fn end_to_end_pump_completes_on_synthetic_clip() {
  // 30 s of low-amplitude noise — meant to verify the pump runs end-
  // to-end without panicking. With near-silence, the segment model
  // typically detects no voice → no spans. That's fine; the test is
  // about the wiring.
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());
  let samples = vec![0.001f32; 16_000 * 30];

  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &samples, |s| spans.push(s))
    .expect("process_samples completes");
  d.finish_stream(&mut seg, &mut emb, |s| spans.push(s))
    .expect("finish_stream completes");

  assert_eq!(d.total_samples_pushed(), 16_000 * 30);
  eprintln!(
    "synthetic-30s pump: {} spans, {} speakers, {} collected",
    spans.len(),
    d.num_speakers(),
    d.collected_embeddings().len()
  );
}

#[test]
#[ignore = "requires both ONNX models AND tests/fixtures/diarize_test_30s.wav"]
fn end_to_end_pump_30s_fixture() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let fixture =
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diarize_test_30s.wav");
  if !fixture.exists() {
    eprintln!(
      "fixture not found at {}; run scripts/download-test-fixtures.sh",
      fixture.display()
    );
    return;
  }

  let mut reader = hound::WavReader::open(&fixture).expect("open fixture");
  assert_eq!(reader.spec().sample_rate, 16_000);
  assert_eq!(reader.spec().channels, 1);
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| s.unwrap() as f32 / i16::MAX as f32)
    .collect();
  let n_samples = samples.len() as u64;

  let mut d = Diarizer::new(DiarizerOptions::default());
  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &samples, |s| spans.push(s))
    .expect("process_samples completes");
  d.finish_stream(&mut seg, &mut emb, |s| spans.push(s))
    .expect("finish_stream completes");

  assert_eq!(d.total_samples_pushed(), n_samples);
  eprintln!(
    "fixture-30s pump: {} spans, {} speakers",
    spans.len(),
    d.num_speakers()
  );
  // The synthetic-tone fixture produces no real speech; a real
  // multi-speaker fixture would yield > 0 spans.
}
