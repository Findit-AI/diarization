//! Same as run_owned_pipeline but uses the `tch` (libtorch) embedding
//! backend instead of ORT. Build with the `tch` feature:
//!
//! ```sh
//! LIBTORCH=$(pwd)/tests/parity/python/.venv/lib/python3.12/site-packages/torch \
//!   LIBTORCH_BYPASS_VERSION_CHECK=1 \
//!   cargo run --release --no-default-features --features ort,tch \
//!   --example run_owned_pipeline_tch tests/parity/fixtures/04_three_speaker/clip_16k.wav > hyp.rttm
//! ```

use diarization::{
  embed::EmbedModel,
  offline::OwnedDiarizationPipeline,
  plda::PldaTransform,
  reconstruct::spans_to_rttm_lines,
  segment::SegmentModel,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
    eprintln!("usage: run_owned_pipeline_tch <clip.wav>");
    std::process::exit(1);
  }
  let clip = &args[1];

  let mut reader = hound::WavReader::open(clip)?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 || spec.channels != 1 {
    return Err("expected 16 kHz mono".into());
  }
  let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
    (hound::SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
      .collect::<Result<Vec<_>, _>>()?,
    (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    _ => return Err("unsupported wav".into()),
  };

  let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let mut seg = SegmentModel::from_file(crate_root.join("models/segmentation-3.0.onnx"))?;
  let mut emb = EmbedModel::from_torchscript_file(crate_root.join("models/wespeaker_resnet34_lm.pt"))?;
  let plda = PldaTransform::new()?;

  let pipeline = OwnedDiarizationPipeline::new();
  let out = pipeline.run(&mut seg, &mut emb, &plda, &samples)?;

  let uri = std::path::Path::new(clip)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("audio");
  for line in spans_to_rttm_lines(&out.spans, uri) {
    println!("{line}");
  }
  eprintln!(
    "# tch embed: {} spans, {} clusters",
    out.spans.len(),
    out.num_clusters
  );
  Ok(())
}
