//! Run `offline::diarize_offline` against the captured pyannote
//! intermediates of a fixture (raw_embeddings, segmentations,
//! count, etc.) and emit RTTM.
//!
//! Use to measure dia's lower-bound DER vs pyannote — the
//! output diverges from `reference.rttm` only by:
//!  - PLDA self-projection ulp drift (we project from raw f32 vs
//!    pyannote's captured f64 post_plda).
//!  - Span emission ordering / formatting differences.
//!
//! ```sh
//! cargo run --example run_offline_from_captures --release -- \
//!   tests/parity/fixtures/01_dialogue > hyp.rttm
//! ```

use diarization::{
  offline::{OfflineInput, diarize_offline},
  plda::PldaTransform,
  reconstruct::{SlidingWindow, spans_to_rttm_lines},
};
use npyz::npz::NpzArchive;
use std::{fs::File, io::BufReader, path::PathBuf};

fn read_npz<T: npyz::Deserialize>(
  path: &PathBuf,
  key: &str,
) -> Result<(Vec<T>, Vec<u64>), Box<dyn std::error::Error>> {
  let f = File::open(path)?;
  let mut z = NpzArchive::new(BufReader::new(f))?;
  let npy = z
    .by_name(key)?
    .ok_or_else(|| format!("missing key {key}"))?;
  let shape = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec()?;
  Ok((data, shape))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
    eprintln!("usage: run_offline_from_captures <fixture-dir>");
    std::process::exit(1);
  }
  let base = PathBuf::from(&args[1]);

  let (raw_flat, raw_shape) = read_npz::<f32>(&base.join("raw_embeddings.npz"), "embeddings")?;
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;

  let (seg_flat_f32, seg_shape) =
    read_npz::<f32>(&base.join("segmentations.npz"), "segmentations")?;
  let num_frames_per_chunk = seg_shape[1] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  let (count_u8, count_shape) = read_npz::<u8>(&base.join("reconstruction.npz"), "count")?;
  let num_output_frames = count_shape[0] as usize;
  let (chunk_start, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "chunk_start")?;
  let (chunk_dur, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "chunk_duration")?;
  let (chunk_step, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "chunk_step")?;
  let (frame_start, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "frame_start")?;
  let (frame_dur, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "frame_duration")?;
  let (frame_step, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "frame_step")?;
  let (min_dur_off, _) = read_npz::<f64>(&base.join("reconstruction.npz"), "min_duration_off")?;
  let chunks_sw = SlidingWindow::new(chunk_start[0], chunk_dur[0], chunk_step[0]);
  let frames_sw = SlidingWindow::new(frame_start[0], frame_dur[0], frame_step[0]);

  let (threshold, _) = read_npz::<f64>(&base.join("ahc_state.npz"), "threshold")?;
  let (fa, _) = read_npz::<f64>(&base.join("vbx_state.npz"), "fa")?;
  let (fb, _) = read_npz::<f64>(&base.join("vbx_state.npz"), "fb")?;
  let (max_iters, _) = read_npz::<i64>(&base.join("vbx_state.npz"), "max_iters")?;

  let plda = PldaTransform::new()?;

  let input = OfflineInput::new(
    &raw_flat,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames_per_chunk,
    &count_u8,
    num_output_frames,
    chunks_sw,
    frames_sw,
    &plda,
  )
  .with_threshold(threshold[0])
  .with_fa(fa[0])
  .with_fb(fb[0])
  .with_max_iters(max_iters[0] as usize)
  .with_min_duration_off(min_dur_off[0]);
  let out = diarize_offline(&input)?;
  for line in spans_to_rttm_lines(out.spans_slice(), "clip_16k") {
    println!("{line}");
  }
  eprintln!(
    "# offline (captured tensors): {} spans, {} clusters",
    out.spans_slice().len(),
    out.num_clusters()
  );
  Ok(())
}
