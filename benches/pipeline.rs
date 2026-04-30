//! End-to-end `assign_embeddings` throughput baseline.
//!
//! Times `diarization::pipeline::assign_embeddings` — the full
//! pyannote `cluster_vbx` flow stages 2-7 (AHC + VBx + centroid +
//! cosine + Hungarian). This is the integration-level measurement;
//! the per-stage benches isolate individual primitives.
//!
//! Per-fixture shape varies; the captured fixtures cover 30s–5min
//! recordings with 1–2 alive speakers. `01_dialogue` is the dominant
//! cost (218 chunks × 3 speakers × 256 dim).
//!
//! Run: `cargo bench --bench pipeline --features _bench`.

use std::{fs::File, hint::black_box, io::BufReader, path::PathBuf};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use diarization::pipeline::{AssignEmbeddingsInput, assign_embeddings};
use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

const FIXTURES: &[&str] = &[
  "01_dialogue",
  "02_pyannote_sample",
  "03_dual_speaker",
  "04_three_speaker",
  "05_four_speaker",
  "06_long_recording",
];

fn fixture(name: &str, file: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/parity/fixtures")
    .join(name)
    .join(file)
}

fn read_npz<T: npyz::Deserialize>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>) {
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape = npy.shape().to_vec();
  let data = npy.into_vec().expect("decode array");
  (data, shape)
}

struct PipelineInputs {
  embeddings: DMatrix<f64>,
  num_chunks: usize,
  num_speakers: usize,
  segmentations: Vec<f64>,
  num_frames: usize,
  post_plda: DMatrix<f64>,
  phi: DVector<f64>,
  train_chunk_idx: Vec<usize>,
  train_speaker_idx: Vec<usize>,
  threshold: f64,
  fa: f64,
  fb: f64,
  max_iters: usize,
}

fn load(fixture_name: &str) -> PipelineInputs {
  let raw_path = fixture(fixture_name, "raw_embeddings.npz");
  let (raw_flat, raw_shape) = read_npz::<f32>(&raw_path, "embeddings");
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

  let seg_path = fixture(fixture_name, "segmentations.npz");
  let (seg_flat_f32, seg_shape) = read_npz::<f32>(&seg_path, "segmentations");
  let num_frames = seg_shape[1] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  let plda_path = fixture(fixture_name, "plda_embeddings.npz");
  let (post_plda_flat, post_plda_shape) = read_npz::<f64>(&plda_path, "post_plda");
  let num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda = DMatrix::<f64>::from_row_slice(num_train, plda_dim, &post_plda_flat);
  let (phi_flat, _) = read_npz::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);
  let (chunk_idx_i64, _) = read_npz::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz::<i64>(&plda_path, "train_speaker_idx");
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();

  let ahc_path = fixture(fixture_name, "ahc_state.npz");
  let threshold = read_npz::<f64>(&ahc_path, "threshold").0[0];
  let vbx_path = fixture(fixture_name, "vbx_state.npz");
  let fa = read_npz::<f64>(&vbx_path, "fa").0[0];
  let fb = read_npz::<f64>(&vbx_path, "fb").0[0];
  let max_iters = read_npz::<i64>(&vbx_path, "max_iters").0[0] as usize;

  PipelineInputs {
    embeddings,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames,
    post_plda,
    phi,
    train_chunk_idx,
    train_speaker_idx,
    threshold,
    fa,
    fb,
    max_iters,
  }
}

fn bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("assign_embeddings");
  for &name in FIXTURES {
    let inp = load(name);
    group.bench_with_input(BenchmarkId::from_parameter(name), &inp, |b, inp| {
      b.iter(|| {
        let input = AssignEmbeddingsInput {
          embeddings: &inp.embeddings,
          num_chunks: inp.num_chunks,
          num_speakers: inp.num_speakers,
          segmentations: &inp.segmentations,
          num_frames: inp.num_frames,
          post_plda: &inp.post_plda,
          phi: &inp.phi,
          train_chunk_idx: &inp.train_chunk_idx,
          train_speaker_idx: &inp.train_speaker_idx,
          threshold: inp.threshold,
          fa: inp.fa,
          fb: inp.fb,
          max_iters: inp.max_iters,
        };
        let hard = assign_embeddings(black_box(&input)).expect("assign_embeddings");
        black_box(hard);
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
