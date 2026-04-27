//! End-to-end integration tests for `dia::embed`.
//!
//! Exercises only the **public** API surface — no `pub(crate)` access.
//! These tests are `#[ignore]`-d because they require the WeSpeaker
//! ResNet34-LM ONNX model. Download with:
//!
//!   ./scripts/download-embed-model.sh
//!   cargo test --features ort --test integration_embed -- --ignored
//!
//! Or point at an arbitrary location via `DIA_EMBED_MODEL_PATH`.

#![cfg(feature = "ort")]

use std::path::PathBuf;

use dia::embed::{EMBED_WINDOW_SAMPLES, EmbedModel};

fn model_path() -> PathBuf {
  if let Ok(p) = std::env::var("DIA_EMBED_MODEL_PATH") {
    return PathBuf::from(p);
  }
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
}

fn skip_if_missing() -> Option<EmbedModel> {
  let path = model_path();
  if !path.exists() {
    return None;
  }
  Some(EmbedModel::from_file(&path).expect("load model"))
}

#[test]
#[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
fn embed_round_trips_on_2s_clip() {
  let Some(mut model) = skip_if_missing() else {
    return;
  };
  let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
  let r = model.embed(&samples).expect("embed succeeds");
  let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
  let norm = n_sq.sqrt();
  assert!(
    (norm - 1.0).abs() < 1e-5,
    "||embedding|| = {norm}, expected 1.0 ± 1e-5"
  );
  assert_eq!(r.windows_used(), 1);
}

#[test]
#[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
fn embed_returns_unit_norm_for_5s_clip() {
  let Some(mut model) = skip_if_missing() else {
    return;
  };
  // 5 s = 80_000 samples. plan_starts: regular grid k_max = (80k-32k)/16k = 3
  // → [0, 16k, 32k, 48k]; tail = 48k. After dedup → 4 windows.
  let samples = vec![0.001f32; 5 * 16_000];
  let r = model.embed(&samples).expect("embed succeeds");
  let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
  assert!((n_sq.sqrt() - 1.0).abs() < 1e-5);
  assert_eq!(r.windows_used(), 4, "5s clip → 4 sliding windows");
}

#[test]
#[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
fn embed_weighted_with_uniform_probs_matches_plain_direction() {
  let Some(mut model) = skip_if_missing() else {
    return;
  };
  let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
  let probs = vec![1.0f32; EMBED_WINDOW_SAMPLES as usize];
  let plain = model.embed(&samples).unwrap();
  let weighted = model.embed_weighted(&samples, &probs).unwrap();
  let cos: f32 = plain
    .embedding()
    .as_array()
    .iter()
    .zip(weighted.embedding().as_array().iter())
    .map(|(a, b)| a * b)
    .sum();
  assert!(
    (cos - 1.0).abs() < 1e-5,
    "cosine(plain, weighted-uniform) = {cos}"
  );
}

#[test]
#[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
fn embed_masked_full_mask_matches_plain() {
  // keep_mask = all true → identical to plain embed.
  let Some(mut model) = skip_if_missing() else {
    return;
  };
  let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
  let mask = vec![true; EMBED_WINDOW_SAMPLES as usize];
  let plain = model.embed(&samples).unwrap();
  let masked = model.embed_masked(&samples, &mask).unwrap();
  assert_eq!(
    plain.embedding().as_array(),
    masked.embedding().as_array(),
    "all-true mask should match plain embed bit-exactly"
  );
}
