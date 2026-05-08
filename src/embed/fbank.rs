//! Kaldi-compatible fbank feature extraction. Spec §4.2.
//!
//! Bit-near-exact port of `torchaudio.compliance.kaldi.fbank`
//! (see `src/embed/torchaudio_fbank.rs`) wrapped to match the
//! WeSpeaker / pyannote conventions:
//! - 16 kHz mono input
//! - 80 mel bins
//! - 25 ms frame length, 10 ms frame shift
//! - hamming window
//! - dither = 0 (deterministic; default is 0.00003)
//! - DC offset removal, preemphasis 0.97, snip_edges true
//! - Power spectrum + log magnitude
//!
//! Per-clip post-processing matches pyannote's
//! `pyannote/audio/pipelines/speaker_verification.py` (line 549, 566):
//! - Input is scaled by `1 << 15` so torchaudio-style int16-magnitude
//!   computation matches WeSpeaker's reference.
//! - Output is mean-subtracted across frames.
//!
//! Verified against `torchaudio.compliance.kaldi.fbank` per Task 1 spike
//! (max |Δ| ~ 2.4e-4 on f32; spec §15 #43).

use crate::embed::{
  error::Error,
  options::{FBANK_FRAMES, FBANK_NUM_MELS, MIN_CLIP_SAMPLES},
};

/// Compute the kaldi-compatible fbank for a clip and pad / center-crop
/// to exactly `[FBANK_FRAMES, FBANK_NUM_MELS] = [200, 80]`.
///
/// Used by `EmbedModel::embed*` in the per-window inner loop.
///
/// # Errors
/// - [`Error::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES` (< 25 ms).
/// - [`Error::NonFiniteInput`] if any sample is NaN/inf.
///
/// # Numerical contract
/// Verified against `torchaudio.compliance.kaldi.fbank`: max abs
/// element error 2.2e-4 on the worst frame of a 23.6-min Mandarin
/// recording, but propagates to ≤1e-5 max abs in the WeSpeaker
/// embedding (vs 0.66 with the prior `kaldi-native-fbank` backend).
/// 95% of cells agree below 1e-5; the residual is f32 FFT
/// reduction-order noise (rustfft radix-2 vs PyTorch's pocketfft).
pub fn compute_fbank(samples: &[f32]) -> Result<Box<[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]>, Error> {
  if samples.len() < MIN_CLIP_SAMPLES as usize {
    return Err(Error::InvalidClip {
      len: samples.len(),
      min: MIN_CLIP_SAMPLES as usize,
    });
  }
  if samples.iter().any(|s| !s.is_finite()) {
    return Err(Error::NonFiniteInput);
  }

  // Bit-near-exact port of `torchaudio.compliance.kaldi.fbank` —
  // see `compute_full_fbank` and `src/embed/torchaudio_fbank.rs`.
  let raw = super::torchaudio_fbank::compute_fbank(samples);
  let n_avail = raw.len() / FBANK_NUM_MELS;
  let mut out = Box::new([[0.0f32; FBANK_NUM_MELS]; FBANK_FRAMES]);

  if n_avail >= FBANK_FRAMES {
    // Center-crop. Diarizer-level masking is applied via embed_masked
    // BEFORE compute_fbank, so center-cropping here only ever drops
    // already-masked-or-padded audio.
    let start = (n_avail - FBANK_FRAMES) / 2;
    for (f, out_row) in out.iter_mut().enumerate() {
      let src = &raw[(start + f) * FBANK_NUM_MELS..(start + f + 1) * FBANK_NUM_MELS];
      out_row.copy_from_slice(src);
    }
  } else {
    // Zero-pad symmetrically.
    let pad_left = (FBANK_FRAMES - n_avail) / 2;
    for (f, out_row) in out.iter_mut().skip(pad_left).take(n_avail).enumerate() {
      let src = &raw[f * FBANK_NUM_MELS..(f + 1) * FBANK_NUM_MELS];
      out_row.copy_from_slice(src);
    }
  }

  // Mean-subtract across frames (per pyannote line 566:
  // `return features - torch.mean(features, dim=1, keepdim=True)`).
  // f64 accumulator: 200 squared-f32 terms can lose mantissa bits in f32.
  let mut mean_per_mel = [0.0f64; FBANK_NUM_MELS];
  for row in out.iter() {
    for (m, &v) in row.iter().enumerate() {
      mean_per_mel[m] += v as f64;
    }
  }
  for m in mean_per_mel.iter_mut() {
    *m /= FBANK_FRAMES as f64;
  }
  for row in out.iter_mut() {
    for (m, v) in row.iter_mut().enumerate() {
      *v -= mean_per_mel[m] as f32;
    }
  }

  Ok(out)
}

/// Compute a kaldi-style fbank for an arbitrary-length clip,
/// returning a flat row-major `(num_frames, FBANK_NUM_MELS)` Vec.
///
/// Same kaldi parameters as [`compute_fbank`], same int16 scaling,
/// same per-(batch, mel) mean centering across frames. Used by the
/// ORT backend for the 10s chunk + frame-mask path
/// ([`crate::embed::EmbedModel::embed_chunk_with_frame_mask`]) where
/// the output frame count varies with the input length and the
/// fixed-size [`compute_fbank`] return type doesn't fit.
pub fn compute_full_fbank(samples: &[f32]) -> Result<Vec<f32>, Error> {
  if samples.len() < MIN_CLIP_SAMPLES as usize {
    return Err(Error::InvalidClip {
      len: samples.len(),
      min: MIN_CLIP_SAMPLES as usize,
    });
  }
  if samples.iter().any(|s| !s.is_finite()) {
    return Err(Error::NonFiniteInput);
  }

  // Bit-near-exact port of `torchaudio.compliance.kaldi.fbank` (see
  // src/embed/torchaudio_fbank.rs). Replaces the prior
  // `kaldi-native-fbank` C++ backend whose f32 reduction order
  // diverged from torchaudio by ~2.4e-4 on Mandarin
  // `08_luyu_jinjing_freedom`, amplifying through ResNet34 to ~0.66
  // absolute embedding-element error and an extra spurious cluster.
  let mut out = super::torchaudio_fbank::compute_fbank(samples);
  let num_frames = out.len() / FBANK_NUM_MELS;
  if num_frames == 0 {
    return Ok(out);
  }

  // Mean-subtract per-(batch, mel) across frames (matches pyannote
  // line 566: `return features - torch.mean(features, dim=1, keepdim=True)`).
  let mut mean_per_mel = [0.0f64; FBANK_NUM_MELS];
  for f in 0..num_frames {
    for m in 0..FBANK_NUM_MELS {
      mean_per_mel[m] += out[f * FBANK_NUM_MELS + m] as f64;
    }
  }
  for m in mean_per_mel.iter_mut() {
    *m /= num_frames as f64;
  }
  for f in 0..num_frames {
    for m in 0..FBANK_NUM_MELS {
      out[f * FBANK_NUM_MELS + m] -= mean_per_mel[m] as f32;
    }
  }

  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embed::options::EMBED_WINDOW_SAMPLES;

  #[test]
  fn rejects_too_short() {
    let r = compute_fbank(&[0.1; 100]);
    assert!(
      matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })),
      "expected InvalidClip {{ len: 100, min: 400 }}, got {r:?}"
    );
  }

  #[test]
  fn rejects_nan() {
    // Build a long-enough clip so the length check doesn't fire first.
    let r = compute_fbank(&[f32::NAN; 32_000]);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "expected NonFiniteInput, got {r:?}"
    );
  }

  #[test]
  fn produces_correct_shape_for_2s_clip() {
    // 2 seconds of near-silence: 32_000 samples → ~200 fbank frames.
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
    assert_eq!(f[0].len(), FBANK_NUM_MELS);
    // After mean-subtraction, all values must be finite.
    for row in f.iter() {
      for &v in row.iter() {
        assert!(v.is_finite(), "fbank coefficient went non-finite: {v}");
      }
    }
  }

  #[test]
  fn produces_correct_shape_for_short_clip_with_padding() {
    // MIN_CLIP_SAMPLES + 100 ≈ 31 ms → only ~1-2 fbank frames available.
    // The pad_left branch should fire and out is FBANK_FRAMES (200) rows.
    let samples = vec![0.001f32; MIN_CLIP_SAMPLES as usize + 100];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
  }

  #[test]
  fn accepts_min_clip_samples_exactly() {
    // Boundary: exactly MIN_CLIP_SAMPLES = 400 samples = 25 ms = 1 frame.
    let samples = vec![0.001f32; MIN_CLIP_SAMPLES as usize];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
    assert_eq!(f[0].len(), FBANK_NUM_MELS);
  }

  #[test]
  fn produces_correct_shape_for_long_clip_with_center_crop() {
    // 4 seconds of audio → ~398 fbank frames > FBANK_FRAMES = 200 → exercises
    // the center-crop branch (start = (n_avail - 200) / 2).
    let samples = vec![0.001f32; 2 * EMBED_WINDOW_SAMPLES as usize];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
    assert_eq!(f[0].len(), FBANK_NUM_MELS);
    // After mean-subtraction, all values must be finite (regression guard
    // for the center-crop branch specifically).
    for row in f.iter() {
      for &v in row.iter() {
        assert!(v.is_finite(), "center-crop branch produced non-finite: {v}");
      }
    }
  }

  #[test]
  fn full_fbank_rejects_too_short() {
    let r = compute_full_fbank(&[0.1; 100]);
    assert!(
      matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })),
      "expected InvalidClip {{ len: 100, min: 400 }}, got {r:?}"
    );
  }

  #[test]
  fn full_fbank_rejects_non_finite() {
    let r = compute_full_fbank(&[f32::NAN; 32_000]);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "expected NonFiniteInput for NaN samples, got {r:?}"
    );
    let r = compute_full_fbank(&[f32::INFINITY; 32_000]);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "expected NonFiniteInput for +inf samples, got {r:?}"
    );
  }

  #[test]
  fn full_fbank_shape_scales_with_input_length() {
    // 10s @ 16 kHz, 25 ms frame / 10 ms shift, snip_edges = true →
    // num_frames = floor((160_000 - 400) / 160) + 1 = 998.
    // Output is row-major (num_frames, FBANK_NUM_MELS), so total length
    // is num_frames * FBANK_NUM_MELS. Pin the contract used by the ORT
    // backend's `embed_chunk_with_frame_mask` path, which divides
    // `fbank.len()` by `FBANK_NUM_MELS` to recover the frame count.
    let samples = vec![0.001f32; 160_000];
    let out = compute_full_fbank(&samples).unwrap();
    assert!(!out.is_empty());
    assert_eq!(out.len() % FBANK_NUM_MELS, 0);
    let frames = out.len() / FBANK_NUM_MELS;
    assert_eq!(frames, 998);
    for v in &out {
      assert!(v.is_finite(), "fbank coefficient went non-finite: {v}");
    }
  }

  #[test]
  fn full_fbank_is_mean_centered_per_mel() {
    // Mean-subtraction at fbank.rs:201-215 zeros each mel band's
    // mean across frames. Verifying this directly catches a future
    // refactor that drops or reorders the centering pass — the
    // resulting embeddings would be biased and silently mis-cluster.
    let samples: Vec<f32> = (0..32_000)
      .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
      .collect();
    let out = compute_full_fbank(&samples).unwrap();
    let frames = out.len() / FBANK_NUM_MELS;
    assert!(frames > 1);
    for m in 0..FBANK_NUM_MELS {
      let mean: f64 = (0..frames)
        .map(|f| f64::from(out[f * FBANK_NUM_MELS + m]))
        .sum::<f64>()
        / frames as f64;
      // f32 → f64 mean accumulator over up to ~200 frames; tolerance
      // covers the f32 rounding of the per-(batch, mel) subtraction.
      assert!(
        mean.abs() < 1e-3,
        "mel {m} mean = {mean} (should be ≈ 0 after mean-subtraction)"
      );
    }
  }
}
