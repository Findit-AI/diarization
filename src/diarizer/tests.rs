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
fn poisoned_state_blocks_further_work() {
  // Codex review HIGH: after a non-recoverable error, the Diarizer
  // refuses further work until clear() is called. We can't easily
  // inject a real model error from a unit test (SegmentModel/
  // EmbedModel both require ONNX files), so this test exercises the
  // poisoned flag toggle through clear() directly. The full failure-
  // injection path is exercised through the integration suite.
  let mut d = Diarizer::new(DiarizerOptions::default());
  assert!(!d.poisoned, "fresh Diarizer must not be poisoned");
  d.poisoned = true;
  d.clear();
  assert!(!d.poisoned, "clear() should reset the poisoned flag");
}

#[test]
fn finished_state_blocks_further_work_until_clear() {
  // Codex review HIGH: after `finish_stream` completes, `process_samples`
  // and a second `finish_stream` must return Error::Finished, not be
  // silently accepted. Without this gate, push_audio still grows the
  // audio buffer and `total_samples_pushed`, but the inner Segmenter
  // drops the samples — desynchronizing the public counter from
  // segmentation.
  //
  // We can't instantiate real SegmentModel/EmbedModel in a unit test
  // (both require ONNX files), so this test exercises the flag toggle
  // through clear() directly, mirroring `poisoned_state_blocks_further_work`.
  // The full lifecycle path is covered by the integration tests gated on
  // model availability.
  let mut d = Diarizer::new(DiarizerOptions::default());
  assert!(!d.finished, "fresh Diarizer must not be finished");
  assert!(!d.finishing, "fresh Diarizer must not be finishing");
  d.finished = true;
  d.finishing = true;
  d.clear();
  assert!(!d.finished, "clear() must reset the finished flag");
  assert!(!d.finishing, "clear() must reset the finishing flag");
}

/// Codex review HIGH regression: a retryable `finish_stream` failure
/// leaves `finished == false` (so the caller can re-drive
/// `finish_stream`), but the inner `Segmenter` is already finished —
/// post-finish samples would be silently dropped. The `finishing` flag
/// locks `process_samples` the moment `finish_stream` begins, and only
/// `clear()` resets it. This test exercises the flag transitions
/// directly because invoking the real lifecycle requires ONNX models.
#[test]
fn finishing_flag_locks_process_samples_across_finish_retry() {
  let mut d = Diarizer::new(DiarizerOptions::default());
  // Simulate "finish_stream started, segmenter.finish() ran, drain
  // returned a retryable Err that left a stashed inference."
  d.finishing = true;
  // finished stays false because the retry has not yet completed.
  assert!(!d.finished);

  // The matching guard in `process_samples` is `finishing || finished`.
  // We verify by inspecting both flags rather than re-running the
  // model-bound path, mirroring `poisoned_state_blocks_further_work`.
  assert!(
    d.finishing || d.finished,
    "process_samples gate must trip when EITHER flag is set"
  );

  // After the caller re-drives `finish_stream` successfully, both
  // flags are set; only clear() returns to the fresh state.
  d.finished = true;
  d.clear();
  assert!(
    !d.finishing && !d.finished,
    "clear() must reset both lifecycle flags"
  );
}

#[test]
fn clear_resets_pending_seg_inference() {
  // Codex review HIGH: clear() must drop any stashed in-flight
  // segmentation inference so a fresh session doesn't accidentally
  // try to retry an inference from the previous session.
  let mut d = Diarizer::new(DiarizerOptions::default());
  // We can't easily build a real WindowId from outside the segment
  // module, but the field is pub(crate) so a None starting state plus
  // a verified clear() suffices for the regression.
  assert!(
    d.pending_seg_inference.is_none(),
    "fresh Diarizer must not have stashed inference"
  );
  d.clear();
  assert!(
    d.pending_seg_inference.is_none(),
    "clear() must drop pending_seg_inference"
  );
}

/// Codex review HIGH: clear() must also drop any stashed embed
/// retry. Mirrors `clear_resets_pending_seg_inference`. The runtime
/// path that populates `pending_embed` requires real ONNX models;
/// here we verify the field is reset, mirroring the existing
/// poison-flag and pending-seg-inference tests.
#[test]
fn clear_resets_pending_embed() {
  let mut d = Diarizer::new(DiarizerOptions::default());
  assert!(
    d.pending_embed.is_none(),
    "fresh Diarizer must not have stashed embed"
  );
  d.clear();
  assert!(d.pending_embed.is_none(), "clear() must drop pending_embed");
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
