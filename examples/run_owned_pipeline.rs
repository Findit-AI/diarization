//! Phase 5d entrypoint: run `OwnedDiarizationPipeline` on a 16 kHz
//! mono WAV and print RTTM lines to stdout. Mirrors the existing
//! `tests/parity/src/main.rs` entry but uses the new offline path
//! (full pyannote `community-1` clustering) instead of the streaming
//! online clusterer.
//!
//! ```sh
//! cargo run --example run_owned_pipeline --features ort --release -- \
//!   path/to/clip_16k.wav > hyp.rttm
//! ```
//!
//! Pair with `tests/parity/python/score.py reference.rttm hyp.rttm`
//! to compute DER vs pyannote.

use diarization::{
  embed::EmbedModel, offline::OwnedDiarizationPipeline, plda::PldaTransform,
  reconstruct::spans_to_rttm_lines, segment::SegmentModel,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
    eprintln!("usage: run_owned_pipeline <clip.wav>");
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

  // Models live in <crate-root>/models/.
  let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let mut seg = SegmentModel::from_file(crate_root.join("models/segmentation-3.0.onnx"))?;
  let mut emb = EmbedModel::from_file(crate_root.join("models/wespeaker_resnet34_lm.onnx"))?;
  let plda = PldaTransform::new()?;

  let pipeline = OwnedDiarizationPipeline::new();
  let out = pipeline.run(&mut seg, &mut emb, &plda, &samples)?;

  // Use clip basename as the RTTM uri.
  let uri = std::path::Path::new(clip)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("audio");

  for line in spans_to_rttm_lines(&out.spans(), uri) {
    println!("{line}");
  }

  eprintln!(
    "# dia (Phase 5d offline): {} spans, {} clusters",
    out.spans().len(),
    out.num_clusters()
  );
  Ok(())
}
