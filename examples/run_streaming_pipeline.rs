//! Streaming voice-range-driven diarization on a 16 kHz mono WAV.
//!
//! Caller drives silero VAD externally and pushes one voice range at
//! a time into [`StreamingOfflineDiarizer`]. At end-of-stream,
//! `finalize` runs global pyannote-equivalent clustering and prints
//! original-timeline RTTM spans.
//!
//! ```sh
//! cargo run --example run_streaming_pipeline --features ort --release -- clip_16k.wav > hyp.rttm
//! ```

use diarization::{
  embed::EmbedModel,
  plda::PldaTransform,
  segment::SegmentModel,
  streaming::{StreamingOfflineDiarizer, StreamingOfflineOptions},
};
use silero::{
  Session as VadSession, SpeechOptions, SpeechSegment, SpeechSegmenter, StreamState as VadStream,
};
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

  // Embedding model: honor `DIA_EMBED_MODEL_PATH` if set, otherwise
  // fall back to the conventional `<crate-root>/models/` location.
  let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let emb_path: PathBuf = std::env::var_os("DIA_EMBED_MODEL_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|| crate_root.join("models/wespeaker_resnet34_lm.onnx"));
  let mut seg = SegmentModel::bundled()?;
  let mut emb = EmbedModel::from_file(&emb_path)
    .map_err(|e| format!("load embed model from {}: {}", emb_path.display(), e))?;
  let plda = PldaTransform::new()?;
  let mut vad_session = VadSession::from_memory(silero::BUNDLED_MODEL)?;
  let vad_opts = SpeechOptions::default()
    .with_min_silence_duration(std::time::Duration::from_millis(1500))
    .with_min_speech_duration(std::time::Duration::from_millis(250))
    .with_max_speech_duration(std::time::Duration::from_secs(60));
  let mut vad_stream = VadStream::new(vad_opts.sample_rate());
  let mut vad_segmenter = SpeechSegmenter::new(vad_opts);

  let mut diarizer = StreamingOfflineDiarizer::new(StreamingOfflineOptions::default());

  // Stream the audio through silero to discover voice ranges, then
  // push each range's PCM through the diarizer eagerly. The voice-
  // range-to-PCM mapping is straightforward because we already have
  // `samples` fully buffered; in a true streaming setting (e.g.
  // ffmpeg → stdin) the caller would maintain a rolling buffer.
  let chunk = 16_000;
  let mut emitted: Vec<SpeechSegment> = Vec::new();
  for window in samples.chunks(chunk) {
    vad_segmenter.process_samples(&mut vad_session, &mut vad_stream, window, |s| {
      emitted.push(s);
    })?;
  }
  vad_segmenter.finish_stream(&mut vad_session, &mut vad_stream, |s| {
    emitted.push(s);
  })?;

  for seg_span in &emitted {
    let start = seg_span.start_sample() as usize;
    let end = (seg_span.end_sample() as usize).min(samples.len());
    if end <= start {
      continue;
    }
    diarizer.push_voice_range(
      &mut seg,
      &mut emb,
      seg_span.start_sample(),
      &samples[start..end],
    )?;
  }

  let spans = diarizer.finalize(&plda)?;
  let uri = std::path::Path::new(clip)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("audio");
  for span in spans.iter() {
    let start = span.start_sample() as f64 / 16_000.0;
    let dur = (span.end_sample() - span.start_sample()) as f64 / 16_000.0;
    println!(
      "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> SPEAKER_{:02} <NA> <NA>",
      uri,
      start,
      dur,
      span.speaker_id()
    );
  }

  eprintln!(
    "# streaming dia: {} voice ranges, {} spans emitted (samples={}, secs={:.1})",
    diarizer.num_ranges(),
    spans.len(),
    samples.len(),
    samples.len() as f64 / 16_000.0,
  );
  Ok(())
}
