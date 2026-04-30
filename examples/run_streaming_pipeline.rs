//! Phase 5e CLI: silero VAD + dia offline diarization on a 16 kHz
//! mono WAV. Streams audio through silero in chunks, runs the
//! offline pipeline on each closed voice range, prints RTTM lines
//! to stdout.
//!
//! ```sh
//! cargo run --example run_streaming_pipeline --features ort --release -- clip_16k.wav > hyp.rttm
//! ```

use diarization::{
  embed::EmbedModel,
  plda::PldaTransform,
  segment::SegmentModel,
  streaming::StreamingDiarizationPipeline,
};
use silero::Session as VadSession;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
    eprintln!("usage: run_streaming_pipeline <clip.wav>");
    std::process::exit(1);
  }
  let clip = &args[1];

  let mut reader = hound::WavReader::open(clip)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    return Err(format!("expected 16 kHz; got {} Hz", spec.sample_rate).into());
  }
  if spec.channels != 1 {
    return Err(format!("expected mono; got {} channels", spec.channels).into());
  }
  let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
    (hound::SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
      .collect::<Result<Vec<_>, _>>()?,
    (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    _ => return Err("unsupported wav format".into()),
  };

  let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let mut seg = SegmentModel::from_file(crate_root.join("models/segmentation-3.0.onnx"))?;
  let mut emb = EmbedModel::from_file(crate_root.join("models/wespeaker_resnet34_lm.onnx"))?;
  let plda = PldaTransform::new()?;
  let vad = VadSession::from_memory(silero::BUNDLED_MODEL)?;

  let mut pipeline = StreamingDiarizationPipeline::new(vad);
  let uri = std::path::Path::new(clip)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("audio");

  let emit_span = |span: diarization::streaming::StreamingDiarizedSpan| {
    let start = span.start_sample as f64 / 16_000.0;
    let dur = (span.end_sample - span.start_sample) as f64 / 16_000.0;
    println!(
      "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> SPEAKER_{:02} <NA> <NA>",
      uri, start, dur, span.speaker_id
    );
  };

  // Push the audio in 1-second chunks to simulate streaming. The
  // VAD itself processes silero-frame-sized blocks internally.
  let chunk = 16_000;
  let mut span_count = 0u32;
  let mut spans_emitter = |span: diarization::streaming::StreamingDiarizedSpan| {
    span_count += 1;
    emit_span(span);
  };
  for window in samples.chunks(chunk) {
    pipeline.push_audio(&mut seg, &mut emb, &plda, window, &mut spans_emitter)?;
  }
  pipeline.finish(&mut seg, &mut emb, &plda, &mut spans_emitter)?;

  eprintln!(
    "# streaming dia (Phase 5e): {} speakers tracked, {} spans emitted (samples={}, secs={:.1})",
    pipeline.num_speakers(),
    span_count,
    samples.len(),
    samples.len() as f64 / 16_000.0
  );
  Ok(())
}
