//! End-to-end integration tests for `diarization::Diarizer`.
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

use diarization::{
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

#[test]
#[ignore = "requires both ONNX models"]
fn empty_push_is_noop() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());
  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &[], |s| spans.push(s))
    .expect("empty push completes");

  assert_eq!(d.total_samples_pushed(), 0);
  assert!(spans.is_empty());
  assert_eq!(d.pending_inferences(), 0);
}

#[test]
#[ignore = "requires both ONNX models"]
fn sub_window_push_no_spans_until_finish() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());
  let half_second = vec![0.001f32; 8_000];

  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &half_second, |s| spans.push(s))
    .expect("sub-window push completes");
  // 0.5 s is far below the 10 s segment window; no inference scheduled.
  assert_eq!(d.pending_inferences(), 0);
  assert!(spans.is_empty());

  // finish_stream's tail-anchor processes the partial buffer.
  d.finish_stream(&mut seg, &mut emb, |s| spans.push(s))
    .expect("finish_stream completes");

  assert_eq!(d.total_samples_pushed(), 8_000);
  // Span count depends on the model — 0 for silent input is fine.
}

#[test]
#[ignore = "requires both ONNX models"]
fn multiple_short_pushes_accumulate() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());

  for _ in 0..5 {
    let chunk = vec![0.001f32; 16_000]; // 1 s × 5 = 5 s
    d.process_samples(&mut seg, &mut emb, &chunk, |_| {})
      .expect("push completes");
  }

  assert_eq!(d.total_samples_pushed(), 5 * 16_000);
  // Still under one window's worth (10 s); no inference scheduled.
  assert_eq!(d.pending_inferences(), 0);
}

#[test]
#[ignore = "requires both ONNX models"]
fn long_single_push_processes_multiple_windows() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());
  let sixty_seconds = vec![0.001f32; 60 * 16_000];

  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &sixty_seconds, |s| spans.push(s))
    .expect("60 s push completes");

  // Synchronous pump drains all scheduled inferences.
  assert_eq!(d.pending_inferences(), 0);
  // For 60 s with the default 2.5 s segment step: ~20 regular windows.
  // No assertion on span count (silent input).
  eprintln!(
    "60s pump: {} spans, {} speakers",
    spans.len(),
    d.num_speakers()
  );
}

#[test]
#[ignore = "requires both ONNX models"]
fn total_samples_pushed_monotonic_resets_on_clear() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());

  d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 10_000], |_| {})
    .expect("first push");
  d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 7_000], |_| {})
    .expect("second push");
  assert_eq!(d.total_samples_pushed(), 17_000);

  d.clear();
  assert_eq!(d.total_samples_pushed(), 0);

  d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 5_000], |_| {})
    .expect("post-clear push");
  assert_eq!(d.total_samples_pushed(), 5_000);
}

#[test]
#[ignore = "requires both ONNX models"]
fn buffered_frames_steady_state() {
  let Some((mut seg, mut emb)) = try_load_models() else {
    return;
  };
  let mut d = Diarizer::new(DiarizerOptions::default());

  d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 30 * 16_000], |_| {})
    .expect("30s push");

  // After 30 s of steady-state streaming, the reconstruction buffer
  // holds the un-finalized frames. ~10 s of audio × 58.9 fps ≈ 589
  // frames bounded above by the segment lookback.
  let buffered = d.buffered_frames();
  eprintln!("buffered_frames after 30s push: {buffered}");
  // Lower bound: at least one window's worth of frames hasn't yet
  // finalized (peek_next_window_start hasn't advanced past them).
  assert!(buffered > 0, "expected un-finalized frames; got {buffered}");
  // Upper bound: bounded by the segmenter's lookback (~589 frames per
  // window plus some slack for in-flight windows).
  assert!(
    buffered < 1500,
    "expected steady-state ≈ 589 frames; got {buffered}"
  );
}
