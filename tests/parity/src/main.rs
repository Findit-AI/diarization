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
  embed::{EmbedModel, EmbedModelOptions},
  ep::CoreML,
  plda::PldaTransform,
  segment::{SegmentModel, SegmentModelOptions},
  streaming::{StreamingOfflineOptions, StreamingOfflineDiarizer},
};
use ort::ep::coreml::{ComputeUnits, ModelFormat};

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

  // EP dispatch knobs — useful for isolating CoreML correctness
  // regressions per model:
  //   DIA_DISABLE_AUTO_PROVIDERS=1   — force CPU on both seg + emb
  //   DIA_FORCE_CPU_SEG=1            — force CPU on seg only
  //   DIA_FORCE_CPU_EMB=1            — force CPU on emb only
  //   DIA_COREML_COMPUTE_UNITS=cpu|gpu|ane|all   — when CoreML auto-
  //     registers, pin the compute unit selection. Useful for
  //     debugging which dispatch produces NaN (the ANE is FP16-only
  //     on M-series and is the most likely culprit for precision
  //     regressions). Default = "all" (CoreML's own picker).
  // Default (all unset) auto-registers `dia::ep::auto_providers()`
  // for both — at build time with `--features coreml`, the CoreML EP.
  let disable_auto = std::env::var("DIA_DISABLE_AUTO_PROVIDERS").ok().as_deref() == Some("1");
  let force_cpu_seg =
    disable_auto || std::env::var("DIA_FORCE_CPU_SEG").ok().as_deref() == Some("1");
  let force_cpu_emb =
    disable_auto || std::env::var("DIA_FORCE_CPU_EMB").ok().as_deref() == Some("1");
  let compute_units = match std::env::var("DIA_COREML_COMPUTE_UNITS").ok().as_deref() {
    Some("cpu") => Some(ComputeUnits::CPUOnly),
    Some("gpu") => Some(ComputeUnits::CPUAndGPU),
    Some("ane") => Some(ComputeUnits::CPUAndNeuralEngine),
    Some("all") | None => None, // None = CoreML's default = ALL
    Some(other) => bail!(
      "DIA_COREML_COMPUTE_UNITS must be one of: cpu, gpu, ane, all (got {other:?})"
    ),
  };
  // Additional CoreML knobs for debugging the WeSpeaker NaN.
  //   DIA_COREML_MODEL_FORMAT=mlprogram|nn   default = nn
  //   DIA_COREML_STATIC_SHAPES=1             require static shapes
  let model_format = match std::env::var("DIA_COREML_MODEL_FORMAT").ok().as_deref() {
    Some("mlprogram") => Some(ModelFormat::MLProgram),
    Some("nn") | None => None,
    Some(other) => bail!(
      "DIA_COREML_MODEL_FORMAT must be 'mlprogram' or 'nn' (got {other:?})"
    ),
  };
  let static_shapes = std::env::var("DIA_COREML_STATIC_SHAPES").ok().as_deref() == Some("1");
  let coreml_provider = || {
    let mut ep = CoreML::default();
    if let Some(u) = compute_units {
      ep = ep.with_compute_units(u);
    }
    if let Some(f) = model_format {
      ep = ep.with_model_format(f);
    }
    if static_shapes {
      ep = ep.with_static_input_shapes(true);
    }
    ep.build()
  };
  let emb_path = std::env::var("DIA_EMBED_MODEL_PATH")
    .unwrap_or_else(|_| "models/wespeaker_resnet34_lm.onnx".into());
  let plda = PldaTransform::new().context("load plda")?;
  let mut seg = if force_cpu_seg {
    SegmentModel::bundled_with_options(SegmentModelOptions::default())
      .context("load bundled segment model (CPU)")?
  } else if compute_units.is_some() {
    // Caller pinned a compute unit — explicitly construct the EP
    // with that pin and pass via `_with_options`. Default
    // `bundled()` would auto_providers() with CoreML's defaults.
    let opts = SegmentModelOptions::default().with_providers(vec![coreml_provider()]);
    SegmentModel::bundled_with_options(opts).context("load bundled segment model (CoreML pinned)")?
  } else {
    SegmentModel::bundled().context("load bundled segment model (auto)")?
  };
  let mut emb = if force_cpu_emb {
    EmbedModel::from_file_with_options(&emb_path, EmbedModelOptions::default())
      .context("load embed model (CPU)")?
  } else if compute_units.is_some() {
    let opts = EmbedModelOptions::default().with_providers(vec![coreml_provider()]);
    EmbedModel::from_file_with_options(&emb_path, opts).context("load embed model (CoreML pinned)")?
  } else {
    EmbedModel::from_file(&emb_path).context("load embed model (auto)")?
  };
  let cu_label = compute_units
    .map(|u| match u {
      ComputeUnits::CPUOnly => "cpu",
      ComputeUnits::CPUAndGPU => "gpu",
      ComputeUnits::CPUAndNeuralEngine => "ane",
      ComputeUnits::All => "all",
    })
    .unwrap_or("default");
  eprintln!(
    "# dia: seg={} emb={} coreml_cu={}",
    if force_cpu_seg { "CPU" } else { "auto" },
    if force_cpu_emb { "CPU" } else { "auto" },
    cu_label,
  );
  // Suppress unused-import warning (the explicit `_with_options`
  // path keeps the type alive; CoreML import is for downstream use
  // by callers reading this binary as an integration example).
  let _ = (
    SegmentModelOptions::default(),
    EmbedModelOptions::default(),
    CoreML::default(),
  );

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
