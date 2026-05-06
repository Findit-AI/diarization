// Spike: validate kaldi-native-fbank crate parity with torchaudio.compliance.kaldi.fbank.
//
// Reads `test_clip.wav` (5-second 16 kHz mono), computes 80-mel kaldi fbank,
// emits CSV (header + one row per frame) on stdout. Intended to be diffed
// frame-by-frame, coefficient-by-coefficient against the Python reference.

use anyhow::{Context, Result, bail};
use hound::WavReader;
use kaldi_native_fbank::{
  fbank::{FbankComputer, FbankOptions},
  online::{FeatureComputer, OnlineFeature},
};

fn main() -> Result<()> {
  // 1) Load the test WAV.
  let mut reader = WavReader::open("test_clip.wav").context("open test_clip.wav")?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    bail!("expected 16 kHz sample rate, got {}", spec.sample_rate);
  }
  if spec.channels != 1 {
    bail!("expected mono (1 channel), got {}", spec.channels);
  }
  if spec.bits_per_sample != 16 {
    bail!("expected 16-bit PCM, got {}", spec.bits_per_sample);
  }
  // torchaudio.compliance.kaldi.fbank expects waveform in the int16 range
  // (signed-int amplitudes from −32768..=32767), per the Kaldi convention.
  // soundfile's dtype="float32" path normalizes to [-1.0, 1.0), so the
  // Python sidecar multiplies by 32768.0 to undo that. On the Rust side
  // we read i16 PCM directly and just widen to f32 — same int16 magnitudes.
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| s.map(|v| v as f32))
    .collect::<Result<Vec<_>, _>>()?;

  // 2) Configure 80-mel kaldi fbank to match torchaudio defaults + spike overrides.
  //
  // torchaudio.compliance.kaldi.fbank defaults (we DO NOT override unless noted):
  //   sample_frequency=16000, frame_length=25 ms, frame_shift=10 ms,
  //   preemphasis_coefficient=0.97, snip_edges=True, low_freq=20.0,
  //   high_freq=0.0, use_energy=False, raw_energy=True, remove_dc_offset=True,
  //   htk_compat=False, round_to_power_of_two=True, blackman_coeff=0.42,
  //   energy_floor=1.0, vtln_warp=1.0.
  //
  // Spike overrides (matched on both sides):
  //   num_mel_bins=80, dither=0.0, window_type="hamming".
  //
  // kaldi-native-fbank 0.1.0 defaults DIFFER from torchaudio in several ways
  // we must override here: dither=0.00003 (random!), window_type="povey",
  // use_energy=true, energy_floor=0.0, MelOptions::num_bins=25.
  let mut opts = FbankOptions::default();
  opts.frame_opts.samp_freq = 16_000.0;
  opts.frame_opts.frame_length_ms = 25.0;
  opts.frame_opts.frame_shift_ms = 10.0;
  opts.frame_opts.dither = 0.0;
  opts.frame_opts.preemph_coeff = 0.97;
  opts.frame_opts.remove_dc_offset = true;
  opts.frame_opts.window_type = "hamming".to_string();
  opts.frame_opts.round_to_power_of_two = true;
  opts.frame_opts.blackman_coeff = 0.42;
  opts.frame_opts.snip_edges = true;
  opts.mel_opts.num_bins = 80;
  opts.mel_opts.low_freq = 20.0;
  opts.mel_opts.high_freq = 0.0;
  opts.use_energy = false;
  opts.raw_energy = true;
  opts.htk_compat = false;
  opts.energy_floor = 1.0;
  opts.use_log_fbank = true;
  opts.use_power = true;

  let computer = FbankComputer::new(opts).map_err(|e| anyhow::anyhow!(e))?;
  let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));
  online.accept_waveform(16_000.0, &samples);
  online.input_finished();

  // 3) Dump CSV: `frame,mel0,mel1,…,mel79`.
  let n = online.num_frames_ready();
  let header: Vec<String> = std::iter::once("frame".to_string())
    .chain((0..80).map(|i| format!("mel{i}")))
    .collect();
  println!("{}", header.join(","));
  for f in 0..n {
    let frame = online
      .get_frame(f)
      .ok_or_else(|| anyhow::anyhow!("frame {f} unexpectedly missing"))?;
    let row: Vec<String> = std::iter::once(f.to_string())
      .chain(frame.iter().map(|x| format!("{x}")))
      .collect();
    println!("{}", row.join(","));
  }
  Ok(())
}
