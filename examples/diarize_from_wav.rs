//! Run [`dia::diarizer::Diarizer`] on a 16 kHz mono WAV file and print
//! emitted [`DiarizedSpan`](dia::diarizer::DiarizedSpan)s.
//!
//! ```sh
//! ./scripts/download-model.sh        # pyannote/segmentation-3.0
//! ./scripts/download-embed-model.sh  # WeSpeaker ResNet34
//! cargo run --release --features ort --example diarize_from_wav -- path/to/clip.wav
//! ```
//!
//! Resolves the segment + embed models via env vars
//! `DIA_SEGMENT_MODEL_PATH` / `DIA_EMBED_MODEL_PATH` or falls back to
//! `models/segmentation-3.0.onnx` / `models/wespeaker_resnet34_lm.onnx`.

#![cfg(feature = "ort")]

use std::{env, error::Error, path::PathBuf};

use dia::{
  diarizer::{Diarizer, DiarizerOptions},
  embed::EmbedModel,
  segment::SegmentModel,
};

fn segment_model_path() -> PathBuf {
  if let Ok(p) = env::var("DIA_SEGMENT_MODEL_PATH") {
    return PathBuf::from(p);
  }
  PathBuf::from("models/segmentation-3.0.onnx")
}

fn embed_model_path() -> PathBuf {
  if let Ok(p) = env::var("DIA_EMBED_MODEL_PATH") {
    return PathBuf::from(p);
  }
  PathBuf::from("models/wespeaker_resnet34_lm.onnx")
}

fn main() -> Result<(), Box<dyn Error>> {
  let args: Vec<String> = env::args().collect();
  let clip_path = args.get(1).ok_or("usage: diarize_from_wav <clip.wav>")?;

  let mut reader = hound::WavReader::open(clip_path)?;
  if reader.spec().sample_rate != 16_000 {
    return Err(format!("input must be 16 kHz; got {} Hz", reader.spec().sample_rate).into());
  }
  if reader.spec().channels != 1 {
    return Err(
      format!(
        "input must be mono; got {} channels",
        reader.spec().channels
      )
      .into(),
    );
  }
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
    .collect::<Result<Vec<_>, _>>()?;

  let mut seg = SegmentModel::from_file(segment_model_path())?;
  let mut emb = EmbedModel::from_file(embed_model_path())?;
  let mut d = Diarizer::new(DiarizerOptions::default());

  println!(
    "# Streaming {} samples ({:.2} s) in 5 s chunks (simulating VAD output)",
    samples.len(),
    samples.len() as f64 / 16_000.0
  );

  let mut span_count = 0u32;
  let chunk_size = 16_000 * 5; // 5 s per push
  for chunk in samples.chunks(chunk_size) {
    d.process_samples(&mut seg, &mut emb, chunk, |span| {
      print_span(span, false);
      span_count += 1;
    })?;
  }
  d.finish_stream(&mut seg, &mut emb, |span| {
    print_span(span, true);
    span_count += 1;
  })?;

  println!(
    "# Done. {} spans, {} speakers, {} samples processed.",
    span_count,
    d.num_speakers(),
    d.total_samples_pushed()
  );
  Ok(())
}

fn print_span(span: dia::diarizer::DiarizedSpan, flushed: bool) {
  let suffix = if flushed { " [flushed]" } else { "" };
  println!(
    "[{:.3}s — {:.3}s] speaker {} (avg_act={:.3}, n_act={}, clean={:.2}){}",
    span.range().start_pts() as f64 / 16_000.0,
    span.range().end_pts() as f64 / 16_000.0,
    span.speaker_id(),
    span.average_activation(),
    span.activity_count(),
    span.clean_mask_fraction(),
    suffix
  );
}
