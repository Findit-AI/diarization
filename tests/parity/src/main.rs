//! Run `dia::Diarizer` on a fixed audio clip and dump RTTM (NIST format)
//! to stdout. Pair with `python/reference.py` for the pyannote.audio
//! reference + `python/score.py` for DER computation.
//!
//! Usage: `cargo run --release --manifest-path tests/parity/Cargo.toml -- <clip.wav>`
//! (run from the dia crate root).

use anyhow::{Context, Result, bail};
use dia::{
  diarizer::{Diarizer, DiarizerOptions},
  embed::EmbedModel,
  segment::SegmentModel,
};

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
    bail!("usage: dia-parity <clip.wav>");
  }
  let clip_path = &args[1];

  let mut reader = hound::WavReader::open(clip_path).context("open clip")?;
  if reader.spec().sample_rate != 16_000 {
    bail!(
      "expected 16 kHz; clip has {} Hz",
      reader.spec().sample_rate
    );
  }
  if reader.spec().channels != 1 {
    bail!("expected mono; clip has {} channels", reader.spec().channels);
  }

  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
    .collect::<Result<Vec<_>, _>>()?;

  // Segment + embed models. Paths can be overridden via env vars.
  let seg_path = std::env::var("DIA_SEGMENT_MODEL_PATH")
    .unwrap_or_else(|_| "models/segmentation-3.0.onnx".into());
  let emb_path = std::env::var("DIA_EMBED_MODEL_PATH")
    .unwrap_or_else(|_| "models/wespeaker_resnet34_lm.onnx".into());
  let mut seg = SegmentModel::from_file(&seg_path).context("load segment model")?;
  let mut emb = EmbedModel::from_file(&emb_path).context("load embed model")?;

  let mut d = Diarizer::new(DiarizerOptions::default());
  let mut spans = Vec::new();
  d.process_samples(&mut seg, &mut emb, &samples, |s| spans.push(s))
    .context("process_samples")?;
  d.finish_stream(&mut seg, &mut emb, |s| spans.push(s))
    .context("finish_stream")?;

  // Dump RTTM (NIST format).
  let uri = std::path::Path::new(clip_path)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("clip");
  for s in &spans {
    let start_sec = s.range().start_pts() as f64 / 16_000.0;
    let dur_sec = (s.range().end_pts() - s.range().start_pts()) as f64 / 16_000.0;
    println!(
      "SPEAKER {uri} 1 {:.3} {:.3} <NA> <NA> SPK_{:02} <NA> <NA>",
      start_sec,
      dur_sec,
      s.speaker_id()
    );
  }
  eprintln!(
    "# dia: {} spans, {} speakers, total_samples_pushed = {}",
    spans.len(),
    d.num_speakers(),
    d.total_samples_pushed()
  );
  Ok(())
}
