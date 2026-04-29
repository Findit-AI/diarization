//! End-to-end parity test: `dia::reconstruct_pyannote::reconstruct`
//! against pyannote's captured `discrete_diarization` (Phase-0 fixture).

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

use crate::{
  pipeline::{AssignEmbeddingsInput, assign_embeddings},
  reconstruct_pyannote::{ReconstructInput, SlidingWindow, reconstruct},
};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures(fixture_dir: &str) {
  let required: Vec<String> = [
    "raw_embeddings.npz",
    "segmentations.npz",
    "plda_embeddings.npz",
    "ahc_state.npz",
    "vbx_state.npz",
    "clustering.npz",
    "reconstruction.npz",
  ]
  .iter()
  .map(|f| format!("tests/parity/fixtures/{fixture_dir}/{f}"))
  .collect();
  let missing: Vec<&str> = required
    .iter()
    .map(String::as_str)
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "reconstruct_pyannote parity fixtures missing: {missing:?}. \
     Re-run `tests/parity/python/capture_intermediates.py` to regenerate."
  );
}

fn read_npz_array<T>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>)
where
  T: npyz::Deserialize,
{
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_01_dialogue() {
  run_reconstruct_parity("01_dialogue");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_02_pyannote_sample() {
  run_reconstruct_parity("02_pyannote_sample");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_03_dual_speaker() {
  run_reconstruct_parity("03_dual_speaker");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_04_three_speaker() {
  run_reconstruct_parity("04_three_speaker");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_05_four_speaker() {
  run_reconstruct_parity("05_four_speaker");
}

fn run_reconstruct_parity(fixture_dir: &str) {
  require_fixtures(fixture_dir);
  let base = format!("tests/parity/fixtures/{fixture_dir}");

  // ── Stage 5a: produce hard_clusters via the assign_embeddings port ──
  let raw_path = fixture(&format!("{base}/raw_embeddings.npz"));
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  let mut embeddings = DMatrix::<f64>::zeros(num_chunks * num_speakers, embed_dim);
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      let row = c * num_speakers + s;
      let base = (c * num_speakers + s) * embed_dim;
      for d in 0..embed_dim {
        embeddings[(row, d)] = raw_flat[base + d] as f64;
      }
    }
  }

  let seg_path = fixture(&format!("{base}/segmentations.npz"));
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  let num_frames_per_chunk = seg_shape[1] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  let plda_path = fixture(&format!("{base}/plda_embeddings.npz"));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda = DMatrix::<f64>::from_row_slice(num_train, plda_dim, &post_plda_flat);
  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);
  let (chunk_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();

  let ahc_path = fixture(&format!("{base}/ahc_state.npz"));
  let (threshold_data, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let threshold = threshold_data[0];
  let vbx_path = fixture(&format!("{base}/vbx_state.npz"));
  let (fa_arr, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_arr, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_arr, _) = read_npz_array::<i64>(&vbx_path, "max_iters");

  let pipeline_input = AssignEmbeddingsInput {
    embeddings: &embeddings,
    num_chunks,
    num_speakers,
    segmentations: &segmentations,
    num_frames: num_frames_per_chunk,
    post_plda: &post_plda,
    phi: &phi,
    train_chunk_idx: &train_chunk_idx,
    train_speaker_idx: &train_speaker_idx,
    threshold,
    fa: fa_arr[0],
    fb: fb_arr[0],
    max_iters: max_iters_arr[0] as usize,
  };
  let hard_clusters = assign_embeddings(&pipeline_input).expect("assign_embeddings");

  // ── Stage 5b: reconstruct ──────────────────────────────────────
  let recon_path = fixture(&format!("{base}/reconstruction.npz"));
  let (count_u8, count_shape) = read_npz_array::<u8>(&recon_path, "count");
  assert_eq!(count_shape.len(), 2);
  let num_output_frames = count_shape[0] as usize;
  // count is (num_output_frames, 1) → flatten.
  assert_eq!(count_shape[1], 1);
  let (chunk_start_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_start");
  let (chunk_dur_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_duration");
  let (chunk_step_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_step");
  let (frame_start_arr, _) = read_npz_array::<f64>(&recon_path, "frame_start");
  let (frame_dur_arr, _) = read_npz_array::<f64>(&recon_path, "frame_duration");
  let (frame_step_arr, _) = read_npz_array::<f64>(&recon_path, "frame_step");
  let chunks_sw = SlidingWindow {
    start: chunk_start_arr[0],
    duration: chunk_dur_arr[0],
    step: chunk_step_arr[0],
  };
  let frames_sw = SlidingWindow {
    start: frame_start_arr[0],
    duration: frame_dur_arr[0],
    step: frame_step_arr[0],
  };

  let recon_input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    hard_clusters: &hard_clusters,
    count: &count_u8,
    num_output_frames,
    chunks_sw,
    frames_sw,
  };
  let got = reconstruct(&recon_input).expect("reconstruct");

  // ── Compare to captured discrete_diarization ────────────────────
  let (want_f32, want_shape) = read_npz_array::<f32>(&recon_path, "discrete_diarization");
  assert_eq!(want_shape.len(), 2);
  let want_frames = want_shape[0] as usize;
  let want_clusters = want_shape[1] as usize;
  assert_eq!(want_frames, num_output_frames);

  // Our `got` has num_clusters columns (= max(hard_clusters)+1, padded
  // up to max(count) if needed). Pyannote's `want` has `want_clusters`
  // columns. They should match.
  let got_clusters = got.len() / num_output_frames;
  assert_eq!(
    got_clusters, want_clusters,
    "cluster count mismatch: got {got_clusters}, want {want_clusters}"
  );

  // Element-wise: count mismatched cells. For pyannote-equivalent
  // behavior we expect ZERO mismatches (both binary outputs).
  let mut mismatch = 0usize;
  let mut first_mismatch = None;
  for t in 0..num_output_frames {
    for k in 0..want_clusters {
      let g = got[t * got_clusters + k];
      let w = want_f32[t * want_clusters + k];
      if g != w {
        mismatch += 1;
        if first_mismatch.is_none() {
          first_mismatch = Some((t, k, g, w));
        }
      }
    }
  }
  let total_cells = num_output_frames * want_clusters;
  let mismatch_pct = mismatch as f64 / total_cells as f64 * 100.0;
  eprintln!(
    "[parity_reconstruct] mismatches: {mismatch}/{total_cells} ({mismatch_pct:.4}%); first: {first_mismatch:?}"
  );
  assert!(
    mismatch == 0,
    "discrete_diarization parity failed: {mismatch}/{total_cells} cells diverge ({mismatch_pct:.4}%); \
     first: {first_mismatch:?}"
  );
}
