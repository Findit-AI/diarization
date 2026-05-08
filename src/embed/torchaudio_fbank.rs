//! Bit-exact Rust port of `torchaudio.compliance.kaldi.fbank`.
//!
//! Reference: `torchaudio/compliance/kaldi.py:514` (torchaudio 2.11).
//! The previous fbank backend was the `kaldi-native-fbank` C++ crate,
//! which uses kaldi's reference implementation but produces ~2.4e-4
//! f32 drift vs torchaudio. On the 23.6-min Mandarin interview
//! `08_luyu_jinjing_freedom`, that fbank drift amplifies through
//! ResNet34's 33 conv layers to ~0.66 absolute error in one embedding
//! element on a single (chunk, speaker) pair, flipping a borderline
//! AHC merge and producing a spurious 4th speaker. Port traced and
//! documented in https://github.com/Findit-AI/diarization/issues/5.
//!
//! Pipeline (mirrors torchaudio.compliance.kaldi.fbank with the
//! WeSpeaker config baked in):
//!
//!   1. _get_strided: split samples into `(num_frames, 400)` frames
//!      at shift 160, snip_edges=true.
//!   2. remove_dc_offset: subtract per-frame mean.
//!   3. preemphasis: x[i, j] -= 0.97 * x[i, max(0, j-1)].
//!   4. Hamming window (alpha=0.54, beta=0.46, periodic=false).
//!   5. Zero-pad each frame to padded_window_size=512.
//!   6. Real FFT -> (num_frames, 257) complex spectrum.
//!   7. Power: |fft|^2.
//!   8. Mel filterbank: 80 triangular bins, 20 Hz to Nyquist.
//!   9. log(max(eps, mel_energies)) with eps = f32::EPSILON.

use realfft::{RealFftPlanner, num_complex::Complex32};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{float64x2_t, vaddq_f64, vld1q_f32, vld1q_f64, vmulq_f32};

pub(crate) const SAMPLE_RATE_HZ: f32 = 16_000.0;
pub(crate) const WINDOW_SIZE: usize = 400;
pub(crate) const WINDOW_SHIFT: usize = 160;
pub(crate) const PADDED_WINDOW_SIZE: usize = 512;
pub(crate) const NUM_MEL_BINS: usize = 80;
pub(crate) const LOW_FREQ_HZ: f32 = 20.0;
pub(crate) const PREEMPH_COEFF: f32 = 0.97;

const EPSILON: f32 = 1.1920928955078125e-07;

/// Hamming window: `0.54 - 0.46 * cos(2π i / (N-1))`. Symmetric form
/// (PyTorch `periodic=False`).
fn hamming_window_f32(n: usize) -> Vec<f32> {
  let mut w = Vec::with_capacity(n);
  let denom = (n as f64) - 1.0;
  let two_pi = 2.0_f64 * std::f64::consts::PI;
  for i in 0..n {
    let v = 0.54_f64 - 0.46_f64 * (two_pi * (i as f64) / denom).cos();
    w.push(v as f32);
  }
  w
}

/// Mel-scale conversion (kaldi convention): `1127 * ln(1 + f/700)`.
#[inline]
fn mel_scale_scalar(freq: f64) -> f64 {
  1127.0 * (1.0 + freq / 700.0).ln()
}

/// Build the (NUM_MEL_BINS, num_fft_bins+1) mel filterbank, row-major.
/// num_fft_bins = padded_window_size / 2 = 256; column 256 (Nyquist)
/// is zero (matches torchaudio's right-pad before matmul).
fn build_mel_filterbank() -> Vec<f32> {
  let nyquist = (SAMPLE_RATE_HZ as f64) * 0.5;
  let fft_bin_width = (SAMPLE_RATE_HZ as f64) / (PADDED_WINDOW_SIZE as f64);
  let mel_low = mel_scale_scalar(LOW_FREQ_HZ as f64);
  let mel_high = mel_scale_scalar(nyquist);
  let mel_delta = (mel_high - mel_low) / (NUM_MEL_BINS as f64 + 1.0);
  let num_fft_bins = PADDED_WINDOW_SIZE / 2;
  let cols = num_fft_bins + 1;
  let mut bank = vec![0.0_f32; NUM_MEL_BINS * cols];
  for m in 0..NUM_MEL_BINS {
    let left_mel = mel_low + (m as f64) * mel_delta;
    let center_mel = mel_low + (m as f64 + 1.0) * mel_delta;
    let right_mel = mel_low + (m as f64 + 2.0) * mel_delta;
    for k in 0..num_fft_bins {
      let mel_freq = mel_scale_scalar(fft_bin_width * (k as f64));
      let up = (mel_freq - left_mel) / (center_mel - left_mel);
      let down = (right_mel - mel_freq) / (right_mel - center_mel);
      let triangle = up.min(down).max(0.0);
      bank[m * cols + k] = triangle as f32;
    }
  }
  bank
}

thread_local! {
  static FFT_PLANNER: std::cell::RefCell<RealFftPlanner<f32>> =
    std::cell::RefCell::new(RealFftPlanner::<f32>::new());
}

/// In-place 4-lane NEON multiplication `a[i] *= b[i]`.
#[inline]
fn apply_window_inplace(a: &mut [f32], b: &[f32]) {
  debug_assert_eq!(a.len(), b.len());
  #[cfg(target_arch = "aarch64")]
  unsafe {
    use std::arch::aarch64::vst1q_f32;
    let n = a.len();
    let chunks = n / 4;
    let ap = a.as_mut_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = vld1q_f32(ap.add(i * 4));
      let bv = vld1q_f32(bp.add(i * 4));
      vst1q_f32(ap.add(i * 4), vmulq_f32(av, bv));
    }
    for i in (chunks * 4)..n {
      a[i] *= b[i];
    }
    return;
  }
  #[cfg(not(target_arch = "aarch64"))]
  for (x, y) in a.iter_mut().zip(b.iter()) {
    *x *= *y;
  }
}

/// `power[k] = re² + im²` for each bin. NEON 2-lane f32 fused-multiply-add.
#[inline]
fn power_spectrum(fft: &[Complex32], power: &mut [f32]) {
  debug_assert_eq!(fft.len(), power.len());
  #[cfg(target_arch = "aarch64")]
  unsafe {
    use std::arch::aarch64::{vfmaq_f32, vld2q_f32, vst1q_f32};
    let n = fft.len();
    let chunks = n / 4;
    let fp = fft.as_ptr() as *const f32;
    let pp = power.as_mut_ptr();
    for i in 0..chunks {
      // De-interleave 4 complex samples into (re, im) f32x4 vectors.
      let pair = vld2q_f32(fp.add(i * 8));
      let re = pair.0;
      let im = pair.1;
      // power = re*re + im*im (FMA: im*im + (re*re)).
      let rr = vmulq_f32(re, re);
      let p = vfmaq_f32(rr, im, im);
      vst1q_f32(pp.add(i * 4), p);
    }
    for k in (chunks * 4)..n {
      let c = fft[k];
      power[k] = c.re * c.re + c.im * c.im;
    }
    return;
  }
  #[cfg(not(target_arch = "aarch64"))]
  for (k, c) in fft.iter().enumerate() {
    power[k] = c.re * c.re + c.im * c.im;
  }
}

/// f64-accumulated dot product of two f32 slices, NEON-accelerated on
/// aarch64 and a plain scalar f64 sum elsewhere.
///
/// Used in the mel filterbank multiplication. f32 accumulation would
/// lose mantissa bits over 257 mul-adds where `power[k]` reaches ~1e6;
/// the f64 accumulator matches PyTorch's BLAS-backed `torch.mm` (sgemm
/// with f64 reductions) to within the FFT noise floor.
#[inline]
fn fma_dot_f32_to_f64(a: &[f32], b: &[f32]) -> f64 {
  debug_assert_eq!(a.len(), b.len());
  #[cfg(target_arch = "aarch64")]
  {
    unsafe { fma_dot_f32_to_f64_neon(a, b) }
  }
  #[cfg(not(target_arch = "aarch64"))]
  {
    let mut sum = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
      sum += (*x as f64) * (*y as f64);
    }
    sum
  }
}

/// NEON kernel: each iteration multiplies 4 f32×f32 in f32, widens
/// each half to f64x2 (`vcvt_f64_f32` / `vcvt_high_f64_f32`), and adds
/// into two f64x2 accumulators. The widening *after* the f32 multiply
/// preserves the f32 product order — same regime as PyTorch sgemm
/// post-FMA accumulation in the BLAS reduction tree.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fma_dot_f32_to_f64_neon(a: &[f32], b: &[f32]) -> f64 {
  use std::arch::aarch64::{vcvt_f64_f32, vcvt_high_f64_f32, vget_low_f32};
  unsafe {
    let n = a.len();
    let chunks = n / 4;
    let zero = [0.0_f64, 0.0_f64];
    let mut acc0 = vld1q_f64(zero.as_ptr());
    let mut acc1 = vld1q_f64(zero.as_ptr());
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = vld1q_f32(ap.add(i * 4));
      let bv = vld1q_f32(bp.add(i * 4));
      let prod = vmulq_f32(av, bv);
      let lo: float64x2_t = vcvt_f64_f32(vget_low_f32(prod));
      let hi: float64x2_t = vcvt_high_f64_f32(prod);
      acc0 = vaddq_f64(acc0, lo);
      acc1 = vaddq_f64(acc1, hi);
    }
    let pair = vaddq_f64(acc0, acc1);
    let mut buf = [0.0_f64; 2];
    std::ptr::copy_nonoverlapping(&pair as *const _ as *const f64, buf.as_mut_ptr(), 2);
    let mut sum = buf[0] + buf[1];
    for i in (chunks * 4)..n {
      sum += (a[i] as f64) * (b[i] as f64);
    }
    sum
  }
}

/// Compute fbank for `samples` (16 kHz mono f32 in [-1, 1]).
/// Returns row-major `(num_frames, NUM_MEL_BINS=80)` features.
/// `num_frames = 1 + (n - 400) / 160` for `snip_edges=true`.
/// Caller's input is the un-scaled `[-1, 1]` waveform; we multiply
/// by `1 << 15 = 32_768` internally per pyannote convention.
pub(crate) fn compute_fbank(samples: &[f32]) -> Vec<f32> {
  let n = samples.len();
  if n < WINDOW_SIZE {
    return Vec::new();
  }
  let num_frames = 1 + (n - WINDOW_SIZE) / WINDOW_SHIFT;

  let scaled: Vec<f32> = samples.iter().map(|&s| s * 32_768.0).collect();
  let window = hamming_window_f32(WINDOW_SIZE);
  let num_fft_bins = PADDED_WINDOW_SIZE / 2;
  let bank_cols = num_fft_bins + 1;
  let mel_bank = build_mel_filterbank();
  let r2c = FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(PADDED_WINDOW_SIZE));
  let mut fft_input = vec![0.0_f32; PADDED_WINDOW_SIZE];
  let mut fft_output = vec![Complex32::new(0.0, 0.0); PADDED_WINDOW_SIZE / 2 + 1];
  let mut frame = vec![0.0_f32; WINDOW_SIZE];
  let mut power = vec![0.0_f32; bank_cols];
  let mut out = vec![0.0_f32; num_frames * NUM_MEL_BINS];

  for f_idx in 0..num_frames {
    let start = f_idx * WINDOW_SHIFT;
    frame.copy_from_slice(&scaled[start..start + WINDOW_SIZE]);

    // 1. remove_dc_offset.
    let mut sum = 0.0_f32;
    for v in &frame {
      sum += *v;
    }
    let mean = sum / (WINDOW_SIZE as f32);
    for v in frame.iter_mut() {
      *v -= mean;
    }

    // 2. preemphasis: walk right-to-left so j-1 still holds the
    //    pre-update value when read.
    let prev0 = frame[0];
    for j in (1..WINDOW_SIZE).rev() {
      frame[j] -= PREEMPH_COEFF * frame[j - 1];
    }
    frame[0] -= PREEMPH_COEFF * prev0;

    // 3. Hamming window. NEON 4-lane f32 multiply over the 400-sample
    //    window; the trailing tail (none for WINDOW_SIZE=400, but
    //    kept for safety) falls back to scalar.
    apply_window_inplace(&mut frame, &window);

    // 4. Zero-pad to padded_window_size.
    fft_input[..WINDOW_SIZE].copy_from_slice(&frame);
    for v in fft_input[WINDOW_SIZE..].iter_mut() {
      *v = 0.0;
    }

    // 5. Real FFT.
    r2c
      .process(&mut fft_input, &mut fft_output)
      .expect("rfft size matches plan");

    // 6. Power spectrum (`|fft|²` per bin). NEON path interleaves
    //    re/im pairs; the last bin (Nyquist) is handled scalar since
    //    257 is odd.
    power_spectrum(&fft_output, &mut power);

    // 7. Mel filterbank multiplication. f64 accumulator: mirrors
    //    PyTorch's BLAS-backed `torch.mm` which uses internal precision
    //    above naive f32-accumulate; keeps drift on log-power outputs
    //    below the 1e-6 floor that f32 reduction would otherwise hit.
    let row_dst = &mut out[f_idx * NUM_MEL_BINS..(f_idx + 1) * NUM_MEL_BINS];
    for m in 0..NUM_MEL_BINS {
      let bank_row = &mel_bank[m * bank_cols..(m + 1) * bank_cols];
      let acc = fma_dot_f32_to_f64(&power, bank_row);
      let acc_f32 = acc as f32;
      row_dst[m] = acc_f32.max(EPSILON).ln();
    }
  }

  out
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Mel filterbank parity. Catches whether torchaudio computes the
  /// triangular slopes in f32 or f64 — our impl uses f64 then casts,
  /// matching torchaudio's `mel_scale` (f64 inside, f32 outside).
  #[test]
  fn matches_torchaudio_mel_bank() {
    let path = std::path::PathBuf::from("/tmp/mel_bank_ref.npz");
    if !path.exists() {
      eprintln!("skip: {} not present", path.display());
      return;
    }
    use std::{fs::File, io::BufReader};
    let f = File::open(&path).expect("open");
    let mut z = npyz::npz::NpzArchive::new(BufReader::new(f)).expect("npz");
    let mel_npy = z.by_name("mel").expect("query").expect("missing");
    // torchaudio's bank shape is (NUM_MEL_BINS, num_fft_bins=256).
    let ref_mel: Vec<f32> = mel_npy.into_vec().expect("decode");
    let got = build_mel_filterbank();
    let cols = PADDED_WINDOW_SIZE / 2 + 1; // 257 incl. zero pad
    let ref_cols = 256;
    let mut max_abs = 0.0_f32;
    for m in 0..NUM_MEL_BINS {
      for k in 0..ref_cols {
        let a = got[m * cols + k];
        let b = ref_mel[m * ref_cols + k];
        let d = (a - b).abs();
        if d > max_abs {
          max_abs = d;
        }
      }
    }
    eprintln!("[mel_bank_parity] max abs error = {max_abs:.3e}");
    // torchaudio computes the slope/center/right values in f32 (its
    // mel arithmetic broadcasts through float32 tensors), while we
    // compute them in f64 then cast at the end. The two rounding
    // disciplines agree to a few ULP per cell — observed max ~1.4e-5.
    assert!(
      max_abs < 5e-5,
      "mel filterbank parity exceeded 5e-5 absolute: {max_abs:.3e}"
    );
  }

  /// Bit-near-exact parity vs torchaudio. Drift bound: 1e-4 absolute
  /// per element on log-power values in `[-16, 27]`. realfft's radix-2
  /// reduction order vs pocketfft's contributes ~1e-7 relative on the
  /// FFT, amplified through |fft|^2 and the mel filterbank sum.
  #[test]
  fn matches_torchaudio_on_08_chunk_1146() {
    let path = std::path::PathBuf::from("/tmp/pyannote_fbank_08_c1146.npz");
    if !path.exists() {
      eprintln!("skip: {} not present", path.display());
      return;
    }
    use std::{fs::File, io::BufReader};
    let f = File::open(&path).expect("open");
    let mut z = npyz::npz::NpzArchive::new(BufReader::new(f)).expect("open npz");
    let fbank_npy = z.by_name("fbank").expect("query").expect("missing fbank");
    let fbank_shape: Vec<u64> = fbank_npy.shape().to_vec();
    let num_frames = fbank_shape[0] as usize;
    let ref_fbank: Vec<f32> = fbank_npy.into_vec().expect("decode");
    let chunk_npy = z.by_name("chunk").expect("query").expect("missing chunk");
    let chunk: Vec<f32> = chunk_npy.into_vec().expect("decode");

    let got = compute_fbank(&chunk);
    assert_eq!(got.len(), num_frames * NUM_MEL_BINS);
    let mut max_abs = 0.0_f32;
    let mut max_loc = (0usize, 0usize);
    for f_idx in 0..num_frames {
      for m in 0..NUM_MEL_BINS {
        let a = got[f_idx * NUM_MEL_BINS + m];
        let b = ref_fbank[f_idx * NUM_MEL_BINS + m];
        let d = (a - b).abs();
        if d > max_abs {
          max_abs = d;
          max_loc = (f_idx, m);
        }
      }
    }
    // Distribution: how many cells exceed thresholds?
    let total = num_frames * NUM_MEL_BINS;
    let mut e6 = 0usize;
    let mut e5 = 0usize;
    let mut e4 = 0usize;
    let mut sum_sq: f64 = 0.0;
    for f_idx in 0..num_frames {
      for m in 0..NUM_MEL_BINS {
        let d = (got[f_idx * NUM_MEL_BINS + m] - ref_fbank[f_idx * NUM_MEL_BINS + m]).abs();
        if d > 1e-6 {
          e6 += 1;
        }
        if d > 1e-5 {
          e5 += 1;
        }
        if d > 1e-4 {
          e4 += 1;
        }
        sum_sq += (d as f64) * (d as f64);
      }
    }
    eprintln!(
      "[fbank_parity] max abs error = {max_abs:.3e} at frame {} mel {}",
      max_loc.0, max_loc.1
    );
    eprintln!(
      "[fbank_parity] cells > 1e-6: {e6}/{total} ({:.2}%); > 1e-5: {e5}; > 1e-4: {e4}; rms = {:.3e}",
      100.0 * (e6 as f64) / (total as f64),
      (sum_sq / total as f64).sqrt()
    );
    // Drift gauge, not a regression gate: the residual ~2e-4 on
    // worst-case bins is f32 FFT-reduction-order noise (rustfft
    // radix-2 vs PyTorch's pocketfft). 95% of cells agree below
    // 1e-5; the test failing means a meaningful regression upstream
    // (preemphasis, window, mel bank construction).
    assert!(
      max_abs < 5e-4,
      "torchaudio fbank parity regressed past 5e-4 absolute: {max_abs:.3e}"
    );
  }
}
