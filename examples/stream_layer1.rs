//! Demonstrates the Sans-I/O Segmenter API with a synthetic inferencer that
//! returns logits for "speaker A continuously voiced." Run with:
//!
//!     cargo run --no-default-features --example stream_layer1
//!
//! No model file required (the Sans-I/O state machine is exercisable
//! with synthetic inputs without `ort`).

use diarization::segment::{
  Action, FRAMES_PER_WINDOW, POWERSET_CLASSES, SegmentOptions, Segmenter, WINDOW_SAMPLES,
};

fn synth_scores_voiced() -> Vec<f32> {
  let mut out = vec![-10.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
  for f in 0..FRAMES_PER_WINDOW {
    out[f * POWERSET_CLASSES + 1] = 10.0; // class 1 = speaker A only
  }
  out
}

fn main() -> Result<(), diarization::segment::Error> {
  let mut seg = Segmenter::new(SegmentOptions::default());

  // Simulate a streaming source: 25 chunks of 10 000 samples (250 000 total).
  for chunk in (0..25).map(|_| vec![0.0f32; 10_000]) {
    seg.push_samples(&chunk);
    while let Some(action) = seg.poll() {
      match action {
        Action::NeedsInference { id, samples } => {
          println!(
            "inference request: id={:?}, len={}",
            id.range(),
            samples.len()
          );
          let scores = synth_scores_voiced();
          seg.push_inference(id, &scores)?;
        }
        Action::Activity(a) => {
          println!(
            "activity: window={:?} slot={} range={:?}",
            a.window_id().range(),
            a.speaker_slot(),
            a.range()
          );
        }
        Action::VoiceSpan(r) => println!("voice span: {r:?}"),
        _ => {}
      }
    }
  }

  seg.finish();
  while let Some(action) = seg.poll() {
    match action {
      Action::NeedsInference { id, samples } => {
        println!("tail inference: id={:?}, len={}", id.range(), samples.len());
        let _ = WINDOW_SAMPLES; // sanity reference
        let scores = synth_scores_voiced();
        seg.push_inference(id, &scores)?;
      }
      Action::Activity(a) => println!(
        "tail activity: slot={} range={:?}",
        a.speaker_slot(),
        a.range()
      ),
      Action::VoiceSpan(r) => println!("tail voice span: {r:?}"),
      _ => {}
    }
  }
  Ok(())
}
