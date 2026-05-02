//! Run `diarization::streaming::StreamingOfflineDiarizer` on a fixed audio clip
//! and dump RTTM (NIST format) to stdout. Pair with `python/reference.py` for
//! the pyannote.audio reference + `python/score.py` for DER computation.
//!
//! Pushes the entire clip as a single voice range so the streaming-offline
//! path is exercised end-to-end on the same input the offline pipeline sees.
//! With one voice range covering the whole clip, the result must match the
//! offline pipeline modulo plumbing.
//!
//! Usage: `cargo run --release --manifest-path tests/parity/Cargo.toml -- <clip.wav>`
//! (run from the dia crate root).

use anyhow::{Context, Result, bail};
use diarization::{
  embed::EmbedModel,
  plda::PldaTransform,
  segment::SegmentModel,
  streaming::{StreamingOfflineOptions, StreamingOfflineDiarizer},
};

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
    bail!("usage: dia-parity <clip.wav>");
  }
  let clip_path = &args[1];

  let mut reader = hound::WavReader::open(clip_path).context("open clip")?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    bail!("expected 16 kHz; clip has {} Hz", spec.sample_rate);
  }
  if spec.channels != 1 {
    bail!("expected mono; clip has {} channels", spec.channels);
  }

  let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
    (hound::SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
      .collect::<Result<Vec<_>, _>>()?,
    (hound::SampleFormat::Float, 32) => reader
      .samples::<f32>()
      .collect::<Result<Vec<_>, _>>()?,
    other => bail!(
      "unsupported WAV sample format: {:?} ({}-bit); use s16le or f32le",
      other.0,
      other.1
    ),
  };

  // Segmentation ships bundled in the crate. Embedding model is BYO
  // (27 MB, doesn't fit under the crates.io 10 MB cap).
  let mut seg = SegmentModel::bundled().context("load bundled segment model")?;
  let emb_path = std::env::var("DIA_EMBED_MODEL_PATH")
    .unwrap_or_else(|_| "models/wespeaker_resnet34_lm.onnx".into());
  let mut emb = EmbedModel::from_file(&emb_path).context("load embed model")?;
  let plda = PldaTransform::new().context("load plda")?;

  let mut diarizer = StreamingOfflineDiarizer::new(StreamingOfflineOptions::default());
  diarizer
    .push_voice_range(&mut seg, &mut emb, 0, &samples)
    .context("push_voice_range")?;
  let spans = diarizer.finalize(&plda).context("finalize")?;

  let uri = std::path::Path::new(clip_path)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("clip");
  for s in spans.iter() {
    let start_sec = s.start_sample() as f64 / 16_000.0;
    let dur_sec = (s.end_sample() - s.start_sample()) as f64 / 16_000.0;
    println!(
      "SPEAKER {uri} 1 {:.3} {:.3} <NA> <NA> SPK_{:02} <NA> <NA>",
      start_sec,
      dur_sec,
      s.speaker_id()
    );
  }
  eprintln!(
    "# dia (streaming-offline): {} spans, {} voice ranges, total_samples = {}",
    spans.len(),
    diarizer.num_ranges(),
    samples.len(),
  );
  Ok(())
}
