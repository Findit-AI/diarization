//! Streams a 16 kHz mono WAV file through the segmenter using the bundled
//! ort driver. Run with:
//!
//!     cargo run --example stream_from_wav -- path/to/audio.wav
//!
//! Requires `models/segmentation-3.0.onnx` (run `scripts/download-model.sh`).

#[cfg(feature = "ort")]
fn main() -> anyhow::Result<()> {
  use dia::segment::{Event, SegmentModel, SegmentOptions, Segmenter};

  let path = std::env::args()
    .nth(1)
    .expect("usage: stream_from_wav <file.wav>");
  let pcm = read_wav_mono_16k(&path)?;
  let mut model = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
  let mut seg = Segmenter::new(SegmentOptions::default());

  // Feed in 100 ms chunks (1_600 samples) to simulate streaming.
  for chunk in pcm.chunks(1_600) {
    seg.process_samples(&mut model, chunk, |event| match event {
      Event::Activity(a) => println!(
        "activity: window={:?} slot={} range={:?}",
        a.window_id().range(),
        a.speaker_slot(),
        a.range()
      ),
      Event::VoiceSpan(r) => println!("voice: {r:?} ({:?})", r.duration()),
    })?;
  }
  seg.finish_stream(&mut model, |event| match event {
    Event::Activity(a) => println!(
      "tail activity: window={:?} slot={} range={:?}",
      a.window_id().range(),
      a.speaker_slot(),
      a.range()
    ),
    Event::VoiceSpan(r) => println!("tail voice: {r:?}"),
  })?;
  Ok(())
}

#[cfg(feature = "ort")]
fn read_wav_mono_16k(path: &str) -> anyhow::Result<Vec<f32>> {
  let mut reader = hound::WavReader::open(path)?;
  let spec = reader.spec();
  anyhow::ensure!(
    spec.sample_rate == 16_000,
    "expected 16 kHz, got {}",
    spec.sample_rate
  );
  anyhow::ensure!(spec.channels == 1, "expected mono, got {}", spec.channels);
  let samples: Result<Vec<f32>, _> = match spec.sample_format {
    hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    hound::SampleFormat::Int => reader
      .samples::<i32>()
      .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
      .collect(),
  };
  Ok(samples?)
}

#[cfg(not(feature = "ort"))]
fn main() {
  eprintln!("This example requires the `ort` feature: cargo run --example stream_from_wav");
}
