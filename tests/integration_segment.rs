//! Smoke test against a real pyannote/segmentation-3.0 ONNX model. Skipped
//! by default (`#[ignore]`); run with:
//!
//!     cargo test --test integration_segment -- --ignored
//!
//! Requires `models/segmentation-3.0.onnx` next to this repo's root.

#![cfg(feature = "ort")]

use dia::segment::{SegmentModel, SegmentOptions, Segmenter};

#[test]
#[ignore = "requires model file at models/segmentation-3.0.onnx"]
fn smoke_test_runs_inference_on_synthetic_audio() {
  let mut model =
    SegmentModel::from_file("models/segmentation-3.0.onnx").expect("model file present");
  let mut seg = Segmenter::new(SegmentOptions::default());

  // 12 seconds of low-amplitude noise — exercise tail anchoring.
  let mut pcm = vec![0.0f32; 16_000 * 12];
  for (i, x) in pcm.iter_mut().enumerate() {
    *x = ((i as f32) * 0.0001).sin() * 0.01;
  }

  let mut events: usize = 0;
  seg
    .process_samples(&mut model, &pcm, |_| events += 1)
    .expect("ok");
  seg.finish_stream(&mut model, |_| events += 1).expect("ok");

  // We don't assert specific events on synthetic noise (the model may
  // emit none); the point is that the pipeline runs end-to-end without
  // panicking and the inference contract holds.
  let _ = events;
}
