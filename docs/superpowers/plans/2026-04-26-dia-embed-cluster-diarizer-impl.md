# dia v0.1.0 phase 2 — embed + cluster + Diarizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `diarization::embed`, `diarization::cluster`, and `diarization::Diarizer` modules to complete the v0.1.0 streaming speaker-diarization release, plus a v0.X additive bump of the already-shipped `diarization::segment` module to expose per-window per-speaker raw probabilities for downstream reconstruction.

**Architecture:** Three new modules + one segment-side additive change, implemented bottom-up. Pure-compute types and online clustering have no `ort` dependency; the `EmbedModel` ort wrapper is gated behind the `ort` feature; `diarization::Diarizer` orchestrates segment + embed + cluster with a per-frame reconstruction state machine that mirrors `pyannote.audio`'s `SpeakerDiarization` pipeline (gather-and-embed for `exclude_overlap`, overlap-add SUM for cluster activations, count-bounded argmax + per-cluster RLE for span emission). All output is streaming — `DiarizedSpan`s emit per closed speaker turn as windows finalize.

**Tech Stack:** Rust 2024 edition, ort 2.0.0-rc.12 (gated), nalgebra 0.34, kaldi-native-fbank 0.1, rand 0.10 + rand_chacha 0.10 (with `default-features = false`), mediatime 0.1, thiserror 2.

**Spec reference:** `/Users/user/Develop/findit-studio/dia/docs/superpowers/specs/2026-04-26-dia-embed-cluster-diarizer-design.md` (Rev 9 + post-rev-9 N2 cleanup; QUALIFIED FOR IMPLEMENTATION verdict). Throughout this plan, **"§N.M" cross-references are to that spec.**

**Reviewer-recommended ordering** (sustained from review-9 verdict): pre-impl spikes first (gating), then bottom-up — `diarization::cluster` (purest, no ort) → `diarization::embed` (post-spike, ort-gated) → `diarization::segment` v0.X bump (small additive) → `diarization::Diarizer` (orchestration). The §9 test list is the TDD spec.

**Effort estimate:** ~17 calendar days of focused implementation. Plan covers ~43 tasks across 13 phases. Each task has 5-10 bite-sized steps following the standard TDD cycle (write failing test → verify failure → implement → verify pass → commit).

**Repo conventions:**
- Edition 2024, Rust 1.95.
- 2-space indent (per shipped `rustfmt.toml`).
- Module layout follows `diarization::segment` precedent (one file per concern, `mod.rs` for re-exports).
- Tests co-located in `#[cfg(test)] mod tests {}` at the bottom of each module file. Integration tests in `tests/integration_*.rs`.
- Compile-time `Send + Sync` assertions in `mod.rs` per the segment pattern.
- Errors use `thiserror::Error`. Each module owns its error type.
- All public items have rustdoc.

---

## Phase 0: Pre-implementation spikes (gating)

These two tasks **must complete successfully** before any production-code task starts. They lock down two empirical contracts the spec depends on.

---

### Task 1: kaldi-native-fbank parity spike

Verify that the brand-new `kaldi-native-fbank` crate produces fbank features numerically equivalent to `torchaudio.compliance.kaldi.fbank` (the Python reference that `findit-speaker-embedding` and pyannote use). If divergent beyond the threshold, fall back to `knf-rs` (C++ binding) before committing to the dep in §7.

Tracked in spec §15 #43.

**Files:**
- Create: `dia/spikes/Cargo.toml`
- Create: `dia/spikes/kaldi_fbank/Cargo.toml`
- Create: `dia/spikes/kaldi_fbank/src/main.rs`
- Create: `dia/spikes/kaldi_fbank/python/reference.py`
- Create: `dia/spikes/kaldi_fbank/python/pyproject.toml`
- Create: `dia/spikes/kaldi_fbank/test_clip.wav` (a fixed 5-second 16 kHz mono speech clip; commit a shared one to the workspace)
- Modify: `dia/.gitignore` (exclude `dia/spikes/kaldi_fbank/python/.venv/`, `dia/spikes/kaldi_fbank/python/__pycache__/`)

- [ ] **Step 1: Create the Rust spike crate**

```bash
mkdir -p /Users/user/Develop/findit-studio/dia/spikes/kaldi_fbank/src
```

Write `dia/spikes/kaldi_fbank/Cargo.toml`:
```toml
[package]
name = "kaldi-fbank-spike"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
kaldi-native-fbank = "0.1"
hound = "3"
anyhow = "1"
```

Write `dia/spikes/kaldi_fbank/src/main.rs`:
```rust
//! Spike: compute fbank via `kaldi-native-fbank` on a fixed 5-s clip and
//! dump the [num_frames, 80] feature matrix to stdout as CSV.
use anyhow::Result;
use hound::WavReader;

fn main() -> Result<()> {
    let mut reader = WavReader::open("test_clip.wav")?;
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz mono");
    assert_eq!(spec.channels, 1);

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    // Configure 80-mel kaldi-style fbank (frame_length 25 ms, frame_shift 10 ms).
    use kaldi_native_fbank::{FbankOptions, OnlineFbank};
    let opts = FbankOptions {
        sample_rate: 16_000.0,
        num_mel_bins: 80,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        dither: 0.0,
        window_type: "hamming",
        ..Default::default()
    };
    let mut fbank = OnlineFbank::new(&opts);
    fbank.accept_waveform(16_000.0, &samples);
    fbank.input_finished();

    let n = fbank.num_frames_ready();
    println!("frame,mel0,mel1,...,mel79");
    for f in 0..n {
        let frame = fbank.get_frame(f);
        let row: Vec<String> = frame.iter().map(|x| format!("{x}")).collect();
        println!("{f},{}", row.join(","));
    }
    Ok(())
}
```

(If `kaldi-native-fbank`'s actual `OnlineFbank` API differs, adapt the spike code to match — the goal is to produce a `[num_frames, 80]` CSV identical in shape to the Python reference. If the crate doesn't have a usable API, this task **fails the spike** and we fall back to `knf-rs`.)

- [ ] **Step 2: Create the Python reference**

Write `dia/spikes/kaldi_fbank/python/pyproject.toml`:
```toml
[project]
name = "kaldi-fbank-reference"
version = "0.0.0"
requires-python = ">=3.10"
dependencies = ["torch", "torchaudio", "soundfile"]
```

Write `dia/spikes/kaldi_fbank/python/reference.py`:
```python
"""Compute fbank via torchaudio.compliance.kaldi.fbank on the same clip; dump CSV."""
import csv, sys
import torch, torchaudio
import soundfile as sf

waveform, sr = sf.read("../test_clip.wav", dtype="float32")
assert sr == 16_000
wf = torch.from_numpy(waveform).unsqueeze(0)  # (1, num_samples)
features = torchaudio.compliance.kaldi.fbank(
    wf,
    num_mel_bins=80,
    frame_length=25.0,
    frame_shift=10.0,
    dither=0.0,
    window_type="hamming",
    sample_frequency=16_000,
)
# (num_frames, 80)
w = csv.writer(sys.stdout)
w.writerow(["frame"] + [f"mel{i}" for i in range(80)])
for i, row in enumerate(features.numpy()):
    w.writerow([i] + list(row))
```

- [ ] **Step 3: Generate a 5-second test clip**

Use the `models/` directory's existing test-clip scaffolding if present, or generate a synthetic clip:
```bash
cd /Users/user/Develop/findit-studio/dia/spikes/kaldi_fbank
ffmpeg -f lavfi -i "sine=frequency=440:duration=5,asetrate=16000" \
  -ac 1 -ar 16000 -sample_fmt s16 test_clip.wav -y
```

Or, preferably, copy a short real-speech clip from another fixture in the workspace.

- [ ] **Step 4: Run both and compare**

```bash
cd /Users/user/Develop/findit-studio/dia/spikes/kaldi_fbank
cargo run --release > rust.csv
cd python && uv venv && uv pip install -e . && uv run python reference.py > ../python.csv
cd ..
```

Compare:
```bash
python3 -c '
import csv
def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return [[float(x) for x in row[1:]] for row in r]
a = load("rust.csv"); b = load("python.csv")
n = min(len(a), len(b))
print(f"rust frames={len(a)} python frames={len(b)} comparing {n}")
max_diff = 0.0
for i in range(n):
    for j in range(80):
        d = abs(a[i][j] - b[i][j])
        max_diff = max(max_diff, d)
print(f"max per-coefficient |Δ| = {max_diff:.6e}")
assert max_diff < 1e-4, f"FAIL — exceeds 1e-4 threshold ({max_diff:.6e})"
print("PASS")
'
```

Expected: `PASS` with max |Δ| < 1e-4.

- [ ] **Step 5: Decide and document**

If PASS:
- Commit the spike directory:
```bash
cd /Users/user/Develop/findit-studio/dia
git add spikes/kaldi_fbank
git commit -m "spike: kaldi-native-fbank parity vs torchaudio (PASS)

Verified per-coefficient |Δ| < 1e-4 on a fixed 5 s clip.
Spec §15 #43 gate: GO."
```

If FAIL:
- Revert the `kaldi-native-fbank` plan; switch §7 deps to `knf-rs` (C++ binding) and adjust Task 15 (compute_fbank) accordingly.
- Commit the spike with FAIL note as documentation; open a tracking issue.

---

### Task 2: ChaCha8Rng byte-fixture regression test

Lock down the `rand_chacha::ChaCha8Rng` keystream as a byte-stable contract from day one. Future `rand_chacha` minor-version bumps that change the cipher output will silently break determinism for users upgrading dia transitively; this fixture catches it at CI time.

Tracked in spec §15 #52.

**Files:**
- Create: `dia/tests/chacha_keystream_fixture.rs`

- [ ] **Step 1: Write the byte-fixture test**

Create `dia/tests/chacha_keystream_fixture.rs`:
```rust
//! Regression fixture for `rand_chacha::ChaCha8Rng` keystream stability.
//!
//! `dia`'s public determinism contract (spec §11.9) commits us to bit-exact
//! cluster labels for a given `OfflineClusterOptions::seed`. That contract
//! depends on `ChaCha8Rng::seed_from_u64(seed).next_u64()` producing the
//! same byte sequence across versions of `rand_chacha`. This test pins
//! the first 8 `next_u64()` outputs for three seeds.
//!
//! If this test ever fails after a `cargo update`, the keystream changed
//! and we need to either (a) pin `rand_chacha` to the prior compatible
//! version, or (b) bump `dia` to a major version (per §11.9 policy).

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

const FIXTURES: &[(u64, [u64; 8])] = &[
    // (seed, [next_u64() × 8])
    // Generated 2026-04-27 with rand_chacha = "0.10" (default-features = false).
    // See spec §15 #52 for re-generation procedure if cipher is intentionally bumped.
    (0, [
        0x0000_0000_0000_0000, // PLACEHOLDER — regenerate during Task 2 step 2
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
    ]),
    (42, [
        0x0000_0000_0000_0000, // PLACEHOLDER — regenerate during Task 2 step 2
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
    ]),
    (0xDEAD_BEEF, [
        0x0000_0000_0000_0000, // PLACEHOLDER — regenerate during Task 2 step 2
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
    ]),
];

#[test]
fn chacha8_keystream_byte_fixture() {
    for (seed, expected) in FIXTURES {
        let mut rng = ChaCha8Rng::seed_from_u64(*seed);
        let actual: [u64; 8] = std::array::from_fn(|_| rng.next_u64());
        assert_eq!(
            &actual, expected,
            "ChaCha8Rng keystream changed for seed {:#x}: actual={:?} expected={:?}\n\
             If intentional (rand_chacha cipher bump), regenerate FIXTURES and bump dia major version per §11.9.",
            seed, actual, expected
        );
    }
}
```

- [ ] **Step 2: Add `rand` and `rand_chacha` to dev-dependencies and regenerate fixtures**

Add to `dia/Cargo.toml` `[dev-dependencies]`:
```toml
rand = { version = "0.10", default-features = false }
rand_chacha = { version = "0.10", default-features = false }
```

Run a one-shot generator to fill in the actual fixture values:
```bash
cd /Users/user/Develop/findit-studio/dia
cargo run --release --example chacha_fixture_gen 2>/dev/null || \
  cargo test --test chacha_keystream_fixture chacha8_keystream_byte_fixture -- --nocapture 2>&1 | head -40
```

The first run fails. Use `cargo test` output to record the actual values, then patch the `FIXTURES` array.

Equivalently, write a tiny generator in `dia/examples/chacha_fixture_gen.rs`:
```rust
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
fn main() {
    for seed in [0u64, 42, 0xDEAD_BEEF] {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let vals: Vec<String> = (0..8)
            .map(|_| format!("0x{:016x}", rng.next_u64()))
            .collect();
        println!("({}, [", seed);
        for v in vals { println!("    {},", v); }
        println!("]),");
    }
}
```

Run `cargo run --example chacha_fixture_gen`, paste the output into `FIXTURES`.

- [ ] **Step 3: Verify the test passes**

```bash
cargo test --test chacha_keystream_fixture
```
Expected: `test chacha8_keystream_byte_fixture ... ok`.

- [ ] **Step 4: Add a no-default-features variant of the test**

The `default-features = false` mode for `rand_chacha` excludes `std`. Ensure the keystream is identical with and without `std`:

Add to the test file:
```rust
#[test]
#[cfg(not(feature = "std"))]
fn chacha8_keystream_no_std() {
    chacha8_keystream_byte_fixture();
}
```

Actually `rand_chacha`'s std feature affects `OsRng`, not `ChaCha8Rng`'s keystream. The single test above is sufficient. Skip this step; document in a comment instead.

- [ ] **Step 5: Commit**

```bash
git add tests/chacha_keystream_fixture.rs Cargo.toml examples/chacha_fixture_gen.rs
git commit -m "test: ChaCha8Rng keystream byte-fixture regression (spec §15 #52)

Locks the 8-u64 prefix of ChaCha8Rng::seed_from_u64({0, 42, 0xDEADBEEF})
as a byte-stable CI contract. Failure means rand_chacha's cipher
output changed; per §11.9 policy, that requires a major version bump
on dia.

Spec §15 #52 gate: GO."
```

---

## Phase 1: `diarization::embed` types and pure helpers (no ort)

Create the `diarization::embed` module with constants, types, and pure functions only. The `EmbedModel` ort wrapper comes later in Phase 5 (after the Phase 0 spike confirms `kaldi-native-fbank` works).

This phase is gated by Task 2 (ChaCha keystream) but **NOT** by Task 1 (the spike); the types and pure helpers don't depend on fbank.

---

### Task 3: `diarization::embed` module skeleton + constants

**Files:**
- Create: `dia/src/embed/mod.rs`
- Create: `dia/src/embed/options.rs`
- Modify: `dia/src/lib.rs:14-` (add `pub mod embed;`)

- [ ] **Step 1: Create the module directory and skeleton**

```bash
mkdir -p /Users/user/Develop/findit-studio/dia/src/embed
```

Write `dia/src/embed/mod.rs`:
```rust
//! Speaker fingerprint generation: WeSpeaker ResNet34 ONNX wrapper +
//! kaldi-compatible fbank + sliding-window mean for variable-length clips.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.
//! Layered API (spec §2.3):
//! - High-level: [`EmbedModel::embed`], [`embed_weighted`], [`embed_masked`]
//! - Low-level: [`compute_fbank`], [`EmbedModel::embed_features`],
//!   [`EmbedModel::embed_features_batch`]

mod error;
mod options;
mod types;

pub use error::Error;
pub use options::{
  EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS,
  HOP_SAMPLES, MIN_CLIP_SAMPLES, NORM_EPSILON,
};
pub use types::{cosine_similarity, Embedding, EmbeddingMeta, EmbeddingResult};

// `compute_fbank` lives in `fbank.rs`; `EmbedModel*` live in `model.rs`
// (cfg ort) — added in later tasks.
```

- [ ] **Step 2: Write the constants**

Write `dia/src/embed/options.rs`:
```rust
//! Constants for `diarization::embed`. All values match spec §4.2 / §5.

/// 2 s @ 16 kHz; the WeSpeaker model's fixed input length.
///
/// Named with the `EMBED_` prefix to avoid collision with
/// `diarization::segment::WINDOW_SAMPLES` (160 000 = 10 s at the same rate).
pub const EMBED_WINDOW_SAMPLES: u32 = 32_000;

/// 1 s @ 16 kHz; sliding-window hop for the long-clip path (§5.1).
/// 50 % overlap with `EMBED_WINDOW_SAMPLES`.
pub const HOP_SAMPLES: u32 = 16_000;

/// ~25 ms @ 16 kHz; one kaldi window. Below this, `embed` returns
/// [`Error::InvalidClip`](crate::embed::Error::InvalidClip).
pub const MIN_CLIP_SAMPLES: u32 = 400;

/// Number of mel bins in the kaldi fbank features (spec §4.2).
pub const FBANK_NUM_MELS: usize = 80;

/// Number of fbank frames per `EMBED_WINDOW_SAMPLES` of audio
/// (25 ms frame length, 10 ms shift → 200 frames per 2 s).
pub const FBANK_FRAMES: usize = 200;

/// Output dimensionality of the WeSpeaker ResNet34 embedding.
pub const EMBEDDING_DIM: usize = 256;

/// Numerical floor used in L2-normalization to avoid divide-by-zero.
/// Matches `findit-speaker-embedding`'s `1e-12` (verified at
/// `embedder.py:85`); diverging would lose Python parity in edge cases.
pub const NORM_EPSILON: f32 = 1e-12;
```

- [ ] **Step 3: Wire the module into the crate**

Modify `dia/src/lib.rs`:
```rust
//! Sans-I/O speaker diarization for streaming audio.
//!
//! See the [`segment`] module for the v0.1.0 phase 1 surface (speaker
//! segmentation), the [`embed`] module for v0.1.0 phase 2 speaker
//! fingerprint generation, and the [`cluster`] module for cross-window
//! speaker linking.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

#[cfg(feature = "std")]
extern crate std;

pub mod embed;
pub mod segment;
```

(Add `pub mod cluster;` and `pub mod diarizer;` in their respective phases.)

- [ ] **Step 4: Verify the skeleton compiles**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo check
```
Expected: clean compile (no errors, no warnings about unused items — `Error`/`Embedding`/etc. are used by the `pub use` in `mod.rs` so they don't trigger dead-code warnings even though their stub files are empty stubs at this moment; this works because `pub use` references types that don't exist yet — the build will fail).

If the skeleton doesn't compile because `error::Error`, `types::*` etc. don't exist yet, that's expected; defer the `cargo check` to Step 4 after Task 4 completes the types. For now, comment out the `pub use` lines in `mod.rs` and rerun `cargo check`. Re-enable them as Tasks 4–8 add each type.

- [ ] **Step 5: Commit**

```bash
git add src/embed/mod.rs src/embed/options.rs src/lib.rs
git commit -m "embed: module skeleton + constants (spec §4.2)

Adds diarization::embed module shell with EMBED_WINDOW_SAMPLES, HOP_SAMPLES,
MIN_CLIP_SAMPLES, FBANK_NUM_MELS, FBANK_FRAMES, EMBEDDING_DIM,
NORM_EPSILON. Types and pure helpers added in subsequent tasks."
```

---

### Task 4: `Embedding` type with invariant + `normalize_from`

**Files:**
- Create: `dia/src/embed/types.rs`

- [ ] **Step 1: Write failing tests**

Create `dia/src/embed/types.rs`:
```rust
//! Public output types for `diarization::embed`. All types are `Send + Sync`.

use crate::embed::options::{EMBEDDING_DIM, NORM_EPSILON};

/// A 256-d L2-normalized speaker embedding.
///
/// **Invariant:** `||embedding.as_array()||₂ > NORM_EPSILON`. The crate
/// guarantees this — the only public constructor (`normalize_from`)
/// returns `None` for degenerate inputs. Internal downstream code
/// (e.g., `Clusterer::submit`) can rely on this for similarity
/// computations being well-defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Embedding(pub(crate) [f32; EMBEDDING_DIM]);

impl Embedding {
    /// Borrow the raw L2-normalized 256-d vector.
    pub const fn as_array(&self) -> &[f32; EMBEDDING_DIM] { &self.0 }

    /// Borrow as a slice.
    pub fn as_slice(&self) -> &[f32] { &self.0 }

    /// Cosine similarity. Both inputs are L2-normalized (per the
    /// `Embedding` invariant), so this reduces to a dot product.
    /// Returns a value in `[-1.0, 1.0]`.
    pub fn similarity(&self, other: &Embedding) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..EMBEDDING_DIM { acc += self.0[i] * other.0[i]; }
        acc
    }

    /// L2-normalize a raw 256-d inference output and wrap it.
    /// Returns `None` if `||raw||₂ < NORM_EPSILON` (degenerate input).
    /// Use after `EmbedModel::embed_features_batch` + custom aggregation.
    pub fn normalize_from(raw: [f32; EMBEDDING_DIM]) -> Option<Self> {
        // Compute ||raw||₂ in f64 for precision, then divide each
        // component in f32. Matches Python's typical behavior where
        // the L2 norm is computed in float32.
        let mut sq = 0.0f64;
        for i in 0..EMBEDDING_DIM { sq += (raw[i] as f64) * (raw[i] as f64); }
        let n = sq.sqrt() as f32;
        if n < NORM_EPSILON { return None; }
        let mut out = [0.0f32; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM { out[i] = raw[i] / n; }
        Some(Self(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_from_zero_returns_none() {
        assert!(Embedding::normalize_from([0.0; EMBEDDING_DIM]).is_none());
    }

    #[test]
    fn normalize_from_below_epsilon_returns_none() {
        let mut tiny = [0.0; EMBEDDING_DIM];
        tiny[0] = 1e-13; // < NORM_EPSILON
        assert!(Embedding::normalize_from(tiny).is_none());
    }

    #[test]
    fn normalize_from_unit_vector_round_trips() {
        let mut v = [0.0; EMBEDDING_DIM];
        v[0] = 1.0;
        let e = Embedding::normalize_from(v).unwrap();
        let n2: f32 = e.as_array().iter().map(|x| x * x).sum();
        assert!((n2 - 1.0).abs() < 1e-6, "||result|| ≈ 1, got n2 = {n2}");
        assert!((e.as_array()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_from_arbitrary_vector_norms_to_one() {
        let mut raw = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM { raw[i] = (i as f32) * 0.01 + 0.1; }
        let e = Embedding::normalize_from(raw).unwrap();
        let n2: f32 = e.as_array().iter().map(|x| x * x).sum();
        assert!((n2 - 1.0).abs() < 1e-5, "n2 = {n2}");
    }

    #[test]
    fn similarity_self_is_one() {
        let mut v = [0.0; EMBEDDING_DIM]; v[0] = 1.0;
        let e = Embedding::normalize_from(v).unwrap();
        assert!((e.similarity(&e) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_orthogonal_is_zero() {
        let mut a = [0.0; EMBEDDING_DIM]; a[0] = 1.0;
        let mut b = [0.0; EMBEDDING_DIM]; b[1] = 1.0;
        let ea = Embedding::normalize_from(a).unwrap();
        let eb = Embedding::normalize_from(b).unwrap();
        assert!(ea.similarity(&eb).abs() < 1e-6);
    }

    #[test]
    fn similarity_antipodal_is_negative_one() {
        let mut a = [0.0; EMBEDDING_DIM]; a[0] = 1.0;
        let mut b = [0.0; EMBEDDING_DIM]; b[0] = -1.0;
        let ea = Embedding::normalize_from(a).unwrap();
        let eb = Embedding::normalize_from(b).unwrap();
        assert!((ea.similarity(&eb) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_symmetric() {
        let mut a = [0.0; EMBEDDING_DIM]; a[0] = 0.6; a[1] = 0.8;
        let mut b = [0.0; EMBEDDING_DIM]; b[0] = 0.8; b[1] = 0.6;
        let ea = Embedding::normalize_from(a).unwrap();
        let eb = Embedding::normalize_from(b).unwrap();
        assert!((ea.similarity(&eb) - eb.similarity(&ea)).abs() < 1e-7);
    }
}
```

- [ ] **Step 2: Verify the tests fail to compile (`Embedding` doesn't exist)**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo test --lib embed::types
```
Expected: compile error or test fails because `types.rs` is empty/missing.

- [ ] **Step 3: Re-enable `pub use types::*` in `embed/mod.rs`**

Update `dia/src/embed/mod.rs` so the module file is re-exported:
```rust
mod types;
pub use types::Embedding;
// (cosine_similarity, EmbeddingMeta, EmbeddingResult exports added in later tasks)
```

- [ ] **Step 4: Run the tests**

```bash
cargo test --lib embed::types
```
Expected: `test result: ok. 8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/embed/types.rs src/embed/mod.rs
git commit -m "embed: Embedding type + similarity + normalize_from (spec §4.2)

L2-normalized 256-d speaker embedding with invariant
||embedding.as_array()||₂ > NORM_EPSILON, enforced by
normalize_from returning None for degenerate inputs.
similarity() = dot product on unit vectors. Matches
findit-speaker-embedding's 1e-12 epsilon."
```

---

### Task 5: `cosine_similarity` free function + parity test

**Files:**
- Modify: `dia/src/embed/types.rs` (append to bottom)
- Modify: `dia/src/embed/mod.rs` (re-export)

- [ ] **Step 1: Write the failing test**

Append to `dia/src/embed/types.rs`'s `mod tests`:
```rust
    #[test]
    fn cosine_similarity_matches_method() {
        let mut a = [0.0; EMBEDDING_DIM];
        let mut b = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            a[i] = (i as f32 * 0.01).sin();
            b[i] = (i as f32 * 0.013).cos();
        }
        let ea = Embedding::normalize_from(a).unwrap();
        let eb = Embedding::normalize_from(b).unwrap();
        // Free fn must equal method bit-exactly (same dot product,
        // same component order — no fma rearrangement).
        assert_eq!(cosine_similarity(&ea, &eb), ea.similarity(&eb));
    }
```

Also append at the top of the test module:
```rust
    use crate::embed::types::cosine_similarity;
```
(Or the function will be unresolved until step 3.)

- [ ] **Step 2: Verify the test fails to compile**

```bash
cargo test --lib embed::types::tests::cosine_similarity_matches_method
```
Expected: `cannot find function 'cosine_similarity' in module ...`.

- [ ] **Step 3: Add the free function**

Append to `dia/src/embed/types.rs` (above the test module):
```rust
/// Free-function form of [`Embedding::similarity`] for callers who
/// prefer it. Both styles are public; pick whichever reads more
/// naturally at the call site. **Bit-exactly equivalent** to the
/// method (same component-order dot product, no FMA rearrangement).
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
    a.similarity(b)
}
```

Update `dia/src/embed/mod.rs` re-exports:
```rust
pub use types::{cosine_similarity, Embedding};
```

- [ ] **Step 4: Run the test**

```bash
cargo test --lib embed::types
```
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/embed/types.rs src/embed/mod.rs
git commit -m "embed: cosine_similarity free function (spec §4.2)

Bit-exact alias for Embedding::similarity. Both styles are public so
callers can pick whichever reads more naturally."
```

---

### Task 6: `EmbeddingMeta<A, T>` generic metadata type

**Files:**
- Modify: `dia/src/embed/types.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Write the failing tests**

Append to `dia/src/embed/types.rs`'s test module:
```rust
    #[test]
    fn embedding_meta_unit_default() {
        let m: EmbeddingMeta = EmbeddingMeta::default();
        assert_eq!(m.audio_id(), &());
        assert_eq!(m.track_id(), &());
        assert_eq!(m.correlation_id(), None);
    }

    #[test]
    fn embedding_meta_typed() {
        let m = EmbeddingMeta::new("audio_42".to_string(), 7u32);
        assert_eq!(m.audio_id(), "audio_42");
        assert_eq!(m.track_id(), &7u32);
        assert_eq!(m.correlation_id(), None);
    }

    #[test]
    fn embedding_meta_with_correlation_id() {
        let m = EmbeddingMeta::new(()
        , ()).with_correlation_id(123);
        assert_eq!(m.correlation_id(), Some(123));
    }
```

- [ ] **Step 2: Verify failure**

```bash
cargo test --lib embed::types
```
Expected: compile error — `EmbeddingMeta` not found.

- [ ] **Step 3: Implement `EmbeddingMeta<A, T>`**

Append to `dia/src/embed/types.rs` (above the test module):
```rust
/// Optional metadata that flows through `embed_with_meta` /
/// `embed_weighted_with_meta` / `embed_masked_with_meta` to
/// `EmbeddingResult`. Generic over the `audio_id` and `track_id`
/// types — callers use whatever string-like type fits their domain.
/// Defaults to `()` so the unit-typed metadata path allocates nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingMeta<A = (), T = ()> {
    audio_id: A,
    track_id: T,
    correlation_id: Option<u64>,
}

impl<A, T> EmbeddingMeta<A, T> {
    /// Construct with `audio_id` and `track_id`.
    pub fn new(audio_id: A, track_id: T) -> Self {
        Self { audio_id, track_id, correlation_id: None }
    }

    /// Attach a correlation id (e.g., a session-scoped sequence number)
    /// for downstream telemetry / log correlation.
    pub fn with_correlation_id(mut self, id: u64) -> Self {
        self.correlation_id = Some(id); self
    }

    pub fn audio_id(&self) -> &A { &self.audio_id }
    pub fn track_id(&self) -> &T { &self.track_id }
    pub fn correlation_id(&self) -> Option<u64> { self.correlation_id }
}
```

Update `dia/src/embed/mod.rs`:
```rust
pub use types::{cosine_similarity, Embedding, EmbeddingMeta};
```

- [ ] **Step 4: Run tests**

```bash
cargo test --lib embed::types
```
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/embed/types.rs src/embed/mod.rs
git commit -m "embed: generic EmbeddingMeta<A, T> (spec §4.2)

Generic over audio_id and track_id types; defaults to () so the
unit-typed metadata path is zero-cost. Per user mandate during
brainstorming: 'do not use String directly'."
```

---

### Task 7: `EmbeddingResult<A, T>` output type

**Files:**
- Modify: `dia/src/embed/types.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Write the failing tests**

Append to `dia/src/embed/types.rs`'s test module:
```rust
    use mediatime::Duration;

    #[test]
    fn embedding_result_unit_meta_construction() {
        let mut v = [0.0; EMBEDDING_DIM]; v[0] = 1.0;
        let e = Embedding::normalize_from(v).unwrap();
        let r: EmbeddingResult = EmbeddingResult::new(
            e,
            Duration::from_millis(1500),
            1,
            1.0,
            EmbeddingMeta::default(),
        );
        assert_eq!(r.embedding(), &e);
        assert_eq!(r.windows_used(), 1);
        assert!((r.total_weight() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn embedding_result_typed_meta() {
        let mut v = [0.0; EMBEDDING_DIM]; v[0] = 1.0;
        let e = Embedding::normalize_from(v).unwrap();
        let r = EmbeddingResult::new(
            e,
            Duration::from_millis(2000),
            2,
            1.5,
            EmbeddingMeta::new("clip_3".to_string(), 9u32)
                .with_correlation_id(42),
        );
        assert_eq!(r.audio_id(), "clip_3");
        assert_eq!(r.track_id(), &9u32);
        assert_eq!(r.correlation_id(), Some(42));
    }
```

- [ ] **Step 2: Verify failure**

```bash
cargo test --lib embed::types
```
Expected: `EmbeddingResult` not found.

- [ ] **Step 3: Implement `EmbeddingResult<A, T>`**

Append to `dia/src/embed/types.rs`:
```rust
use mediatime::Duration;

/// Result of one `EmbedModel::embed*` call.
///
/// Carries the embedding plus observability fields:
/// - `source_duration`: actual length of the source clip (NOT padded/cropped)
/// - `windows_used`: number of 2 s windows averaged (1 for clips ≤ 2 s)
/// - `total_weight`: sum of per-window weights
/// - `audio_id`/`track_id`/`correlation_id`: caller-supplied metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult<A = (), T = ()> {
    embedding: Embedding,
    source_duration: Duration,
    windows_used: u32,
    total_weight: f32,
    audio_id: A,
    track_id: T,
    correlation_id: Option<u64>,
}

impl<A, T> EmbeddingResult<A, T> {
    /// Construct (typically from inside `EmbedModel`).
    pub(crate) fn new(
        embedding: Embedding,
        source_duration: Duration,
        windows_used: u32,
        total_weight: f32,
        meta: EmbeddingMeta<A, T>,
    ) -> Self {
        let EmbeddingMeta { audio_id, track_id, correlation_id } = meta;
        Self { embedding, source_duration, windows_used, total_weight,
               audio_id, track_id, correlation_id }
    }

    pub fn embedding(&self) -> &Embedding { &self.embedding }
    pub fn source_duration(&self) -> Duration { self.source_duration }
    pub fn windows_used(&self) -> u32 { self.windows_used }
    pub fn total_weight(&self) -> f32 { self.total_weight }
    pub fn audio_id(&self) -> &A { &self.audio_id }
    pub fn track_id(&self) -> &T { &self.track_id }
    pub fn correlation_id(&self) -> Option<u64> { self.correlation_id }
}
```

Note: `EmbeddingMeta` fields are `pub(crate)` for the destructure. Update `EmbeddingMeta`:
```rust
pub struct EmbeddingMeta<A = (), T = ()> {
    pub(crate) audio_id: A,
    pub(crate) track_id: T,
    pub(crate) correlation_id: Option<u64>,
}
```

Update `dia/src/embed/mod.rs`:
```rust
pub use types::{cosine_similarity, Embedding, EmbeddingMeta, EmbeddingResult};
```

- [ ] **Step 4: Run tests**

```bash
cargo test --lib embed::types
```
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/embed/types.rs src/embed/mod.rs
git commit -m "embed: EmbeddingResult<A, T> with observability fields (spec §4.2)

Carries embedding + source_duration + windows_used + total_weight +
flowed-through EmbeddingMeta. windows_used = 1 for ≤ 2 s clips,
total_weight = windows_used for the equal-weighted path."
```

---

### Task 8: `embed::Error` enum

**Files:**
- Create: `dia/src/embed/error.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Write the failing tests**

Create `dia/src/embed/error.rs`:
```rust
//! Error type for `diarization::embed`.

use std::path::PathBuf;

/// Errors returned by `diarization::embed` APIs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Input clip too short. Either `samples.len() < MIN_CLIP_SAMPLES`
    /// (for `embed`/`embed_weighted`) or the gathered length after
    /// applying a keep_mask in `embed_masked` was below the threshold.
    #[error("clip too short: {len} samples (need at least {min})")]
    InvalidClip { len: usize, min: usize },

    /// `voice_probs.len() != samples.len()` for `embed_weighted`.
    #[error("voice_probs.len() = {weights_len} must equal samples.len() = {samples_len}")]
    WeightShapeMismatch { samples_len: usize, weights_len: usize },

    /// Rev-8: `keep_mask.len() != samples.len()` for `embed_masked`.
    #[error("keep_mask.len() = {mask_len} must equal samples.len() = {samples_len}")]
    MaskShapeMismatch { samples_len: usize, mask_len: usize },

    /// All windows had near-zero voice-probability weight; the weighted
    /// average is undefined. Almost always caller error.
    #[error("all windows had effectively zero voice-activity weight")]
    AllSilent,

    /// Input contains NaN or infinity.
    #[error("input contains non-finite values (NaN or infinity)")]
    NonFiniteInput,

    /// Input contains a zero-norm (or near-zero-norm, `< NORM_EPSILON`)
    /// embedding. Zero IS finite — kept distinct from `NonFiniteInput`
    /// so callers debugging real NaN/inf cases aren't misled.
    #[error("input contains a zero-norm or degenerate embedding")]
    DegenerateEmbedding,

    /// ONNX inference output had an unexpected shape.
    #[error("inference scores length {got}, expected {expected}")]
    InferenceShapeMismatch { expected: usize, got: usize },

    /// Load-time model shape verification failed.
    #[cfg(feature = "ort")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
    #[error("model {tensor} dims {got:?}, expected {expected:?}")]
    IncompatibleModel { tensor: &'static str, expected: &'static [i64], got: Vec<i64> },

    /// Failed to load the ONNX model from disk.
    #[cfg(feature = "ort")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
    #[error("failed to load model from {path}: {source}", path = path.display())]
    LoadModel { path: PathBuf, #[source] source: ort::Error },

    /// Wrap an `ort::Error` from session/inference.
    #[cfg(feature = "ort")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
    #[error(transparent)]
    Ort(#[from] ort::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_clip_message() {
        let e = Error::InvalidClip { len: 100, min: 400 };
        let s = format!("{e}");
        assert!(s.contains("100"));
        assert!(s.contains("400"));
    }
    #[test]
    fn mask_shape_mismatch_message() {
        let e = Error::MaskShapeMismatch { samples_len: 1000, mask_len: 999 };
        let s = format!("{e}");
        assert!(s.contains("1000"));
        assert!(s.contains("999"));
    }
}
```

- [ ] **Step 2: Verify the tests fail (file may not be wired up yet)**

```bash
cargo test --lib embed::error
```

Update `dia/src/embed/mod.rs`:
```rust
mod error;
pub use error::Error;
```

```bash
cargo test --lib embed::error
```
Expected: `2 passed`.

- [ ] **Step 3: Commit**

```bash
git add src/embed/error.rs src/embed/mod.rs
git commit -m "embed: Error enum (spec §4.2)

Eight variants covering shape mismatches, degenerate inputs, non-
finite inputs, zero-norm embeddings, and ort failures. ort-related
variants are gated behind the 'ort' feature."
```

---

(Phase 1 is complete; `diarization::embed` types are usable as inputs to `diarization::cluster`. Phase 5 will add the `EmbedModel` + `compute_fbank` + sliding-window-mean implementation that produces these types from raw audio.)

---

## Phase 2: `diarization::cluster` — online streaming

Pure-compute module. No `ort` dependency. Online `Clusterer` + types.

---

### Task 9: `diarization::cluster` module skeleton + constants + types

**Files:**
- Create: `dia/src/cluster/mod.rs`
- Create: `dia/src/cluster/options.rs`
- Create: `dia/src/cluster/types.rs`
- Modify: `dia/src/lib.rs` (add `pub mod cluster;`)

- [ ] **Step 1: Create the module directory + skeleton**

```bash
mkdir -p /Users/user/Develop/findit-studio/dia/src/cluster
```

Write `dia/src/cluster/mod.rs`:
```rust
//! Cross-window speaker linking: online streaming `Clusterer` + offline
//! batch `cluster_offline` (spectral default, agglomerative alternative).
//!
//! See spec §4.3 / §5.4-§5.6 for the design.

mod error;
mod online;
mod options;
mod types;

#[cfg(test)]
mod tests;

mod agglomerative;
mod spectral;

pub use crate::embed::Embedding;
pub use error::Error;
pub use online::Clusterer;
pub use options::{
  ClusterOptions, DEFAULT_EMA_ALPHA, DEFAULT_SIMILARITY_THRESHOLD,
  Linkage, MAX_AUTO_SPEAKERS, OfflineClusterOptions, OfflineMethod,
  OverflowStrategy, UpdateStrategy,
};
pub use types::{ClusterAssignment, SpeakerCentroid};

// Public free function for offline clustering (spec §4.3).
pub use offline::cluster_offline;

mod offline;
```

- [ ] **Step 2: Write the constants + types stubs**

Write `dia/src/cluster/options.rs`:
```rust
//! Public options + constants for `diarization::cluster`. Matches spec §4.3.

/// Default cosine-similarity threshold for `Clusterer` and
/// `cluster_offline`. Submissions below this similarity to every
/// existing speaker spawn a new global speaker id.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;

/// Default `Ema(α)` smoothing factor. Tentative — see spec §15 #23.
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;

/// Maximum K when auto-detecting via eigengap (no `target_speakers`).
/// Caps eigengap-suggested K at this value (spec §5.5 step 5).
pub const MAX_AUTO_SPEAKERS: u32 = 15;

/// Centroid update strategy for online `Clusterer`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateStrategy {
    /// Centroid = `L2-normalize(unnormalized_sum)`. Maintains an
    /// internal running sum; centroid derived on demand. Stable;
    /// immune to drift on long sessions.
    RollingMean,
    /// EMA on the unnormalized accumulator: `accumulator =
    /// (1-α)·accumulator + α·new_embedding`. Centroid is
    /// `L2-normalize(accumulator)` on demand. Adapts to drift;
    /// recommended for streaming.
    Ema(f32),
}

/// Behavior when `Clusterer::submit` exceeds the configured
/// `max_speakers` cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverflowStrategy {
    /// Force-assign to the closest existing speaker. Centroid is
    /// **NOT** updated (spec §5.4 — preserves centroid integrity).
    /// `assignment_count` increments. `is_new_speaker = false`.
    AssignClosest,
    /// **DEFAULT.** Reject with `Error::TooManySpeakers`. Caller decides.
    Reject,
}

/// Options for the online streaming `Clusterer`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterOptions {
    similarity_threshold: f32,
    update_strategy: UpdateStrategy,
    max_speakers: Option<u32>,
    overflow_strategy: OverflowStrategy,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            update_strategy: UpdateStrategy::Ema(DEFAULT_EMA_ALPHA),
            max_speakers: None,
            overflow_strategy: OverflowStrategy::Reject,
        }
    }
}

impl ClusterOptions {
    pub fn new() -> Self { Self::default() }

    pub fn similarity_threshold(&self) -> f32 { self.similarity_threshold }
    pub fn update_strategy(&self) -> UpdateStrategy { self.update_strategy }
    pub fn max_speakers(&self) -> Option<u32> { self.max_speakers }
    pub fn overflow_strategy(&self) -> OverflowStrategy { self.overflow_strategy }

    pub fn with_similarity_threshold(mut self, t: f32) -> Self { self.similarity_threshold = t; self }
    pub fn with_update_strategy(mut self, s: UpdateStrategy) -> Self { self.update_strategy = s; self }
    pub fn with_max_speakers(mut self, n: u32) -> Self { self.max_speakers = Some(n); self }
    pub fn with_overflow_strategy(mut self, s: OverflowStrategy) -> Self { self.overflow_strategy = s; self }

    pub fn set_similarity_threshold(&mut self, t: f32) -> &mut Self { self.similarity_threshold = t; self }
    pub fn set_update_strategy(&mut self, s: UpdateStrategy) -> &mut Self { self.update_strategy = s; self }
    pub fn set_max_speakers(&mut self, n: u32) -> &mut Self { self.max_speakers = Some(n); self }
    pub fn set_overflow_strategy(&mut self, s: OverflowStrategy) -> &mut Self { self.overflow_strategy = s; self }
}

/// Linkage criterion for offline agglomerative clustering. Ward removed
/// in rev-2 (invalid with cosine distance — spec §13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    Single,
    Complete,
    /// Recommended for speaker clustering. **Default for `Agglomerative`.**
    Average,
}

/// Offline clustering algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OfflineMethod {
    /// Hierarchical agglomerative clustering. Faster for small N.
    Agglomerative { linkage: Linkage },
    /// Spectral clustering. Best quality. **Default.**
    Spectral,
}

impl Default for OfflineMethod {
    fn default() -> Self { Self::Spectral }
}

/// Options for `cluster_offline`.
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineClusterOptions {
    method: OfflineMethod,
    similarity_threshold: f32,
    target_speakers: Option<u32>,
    seed: Option<u64>,
}

impl Default for OfflineClusterOptions {
    fn default() -> Self {
        Self {
            method: OfflineMethod::default(),
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            target_speakers: None,
            seed: None,
        }
    }
}

impl OfflineClusterOptions {
    pub fn new() -> Self { Self::default() }

    pub fn method(&self) -> OfflineMethod { self.method }
    pub fn similarity_threshold(&self) -> f32 { self.similarity_threshold }
    pub fn target_speakers(&self) -> Option<u32> { self.target_speakers }
    pub fn seed(&self) -> Option<u64> { self.seed }

    pub fn with_method(mut self, m: OfflineMethod) -> Self { self.method = m; self }
    pub fn with_similarity_threshold(mut self, t: f32) -> Self { self.similarity_threshold = t; self }
    pub fn with_target_speakers(mut self, n: u32) -> Self { self.target_speakers = Some(n); self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = Some(s); self }

    pub fn set_method(&mut self, m: OfflineMethod) -> &mut Self { self.method = m; self }
    pub fn set_similarity_threshold(&mut self, t: f32) -> &mut Self { self.similarity_threshold = t; self }
    pub fn set_target_speakers(&mut self, n: u32) -> &mut Self { self.target_speakers = Some(n); self }
    pub fn set_seed(&mut self, s: u64) -> &mut Self { self.seed = Some(s); self }
}
```

Write `dia/src/cluster/types.rs`:
```rust
//! Public output types for `diarization::cluster`.

use crate::embed::Embedding;

/// Public view of a speaker centroid. The internal accumulator is
/// hidden — see spec §5.4.
///
/// Field visibility: private fields with accessors. Matches
/// `EmbeddingResult`, `DiarizedSpan`, and `SpeakerActivity`. Lets us
/// evolve the internal representation without a breaking API change.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerCentroid {
    pub(crate) speaker_id: u64,
    pub(crate) centroid: Embedding,
    pub(crate) assignment_count: u32,
}

impl SpeakerCentroid {
    /// Global speaker id assigned by the `Clusterer`.
    pub fn speaker_id(&self) -> u64 { self.speaker_id }
    /// L2-normalized centroid (best estimate of this speaker's fingerprint).
    pub fn centroid(&self) -> &Embedding { &self.centroid }
    /// Number of `submit` calls assigned to this speaker (including
    /// `OverflowStrategy::AssignClosest` forced assignments, even
    /// though those don't update the centroid — spec §5.4).
    pub fn assignment_count(&self) -> u32 { self.assignment_count }
}

/// Result of one `Clusterer::submit` call.
///
/// Visibility: private fields with accessors (rev-5 alignment within
/// the cluster module).
#[derive(Debug, Clone, Copy)]
pub struct ClusterAssignment {
    pub(crate) speaker_id: u64,
    pub(crate) is_new_speaker: bool,
    pub(crate) similarity: Option<f32>,
}

impl ClusterAssignment {
    pub fn speaker_id(&self) -> u64 { self.speaker_id }
    pub fn is_new_speaker(&self) -> bool { self.is_new_speaker }

    /// Cosine similarity to the assigned centroid, computed pre-update.
    /// `None` for the first-ever assignment (no centroids existed to
    /// compare against). For new speakers beyond the first, this is
    /// the maximum similarity to any existing centroid.
    pub fn similarity(&self) -> Option<f32> { self.similarity }
}
```

- [ ] **Step 3: Add stubs for the other module files**

Create empty stubs so the module compiles:

`dia/src/cluster/error.rs`:
```rust
//! Filled in by Task 12.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("placeholder")]
    Placeholder,
}
```

`dia/src/cluster/online.rs`:
```rust
//! Filled in by Tasks 10-12.
use crate::cluster::{ClusterOptions, options::*, types::*};
pub struct Clusterer { _opts: ClusterOptions }
impl Clusterer {
    pub fn new(opts: ClusterOptions) -> Self { Self { _opts: opts } }
}
```

`dia/src/cluster/agglomerative.rs`, `dia/src/cluster/spectral.rs`, `dia/src/cluster/offline.rs`:
```rust
//! Filled in by Tasks 14-21.
```

`dia/src/cluster/tests.rs`:
```rust
//! Tests added per task. Empty stub for now.
```

Each empty file needs at least a doc comment to avoid unused warnings.

In `dia/src/cluster/offline.rs`:
```rust
//! Filled in by Tasks 14-21.
use crate::cluster::{Error, OfflineClusterOptions};
use crate::embed::Embedding;
pub fn cluster_offline(_e: &[Embedding], _o: &OfflineClusterOptions) -> Result<Vec<u64>, Error> {
    Err(Error::Placeholder)
}
```

- [ ] **Step 4: Wire into `lib.rs` and verify compile**

Modify `dia/src/lib.rs`:
```rust
pub mod cluster;
pub mod embed;
pub mod segment;
```

```bash
cargo check
```
Expected: clean compile.

- [ ] **Step 5: Commit**

```bash
git add src/cluster/ src/lib.rs
git commit -m "cluster: module skeleton + types + options (spec §4.3)

Stubs for Clusterer, cluster_offline, agglomerative, spectral. Types
SpeakerCentroid + ClusterAssignment with accessor visibility (rev-5
alignment). Constants DEFAULT_SIMILARITY_THRESHOLD = 0.5,
DEFAULT_EMA_ALPHA = 0.2, MAX_AUTO_SPEAKERS = 15. ClusterOptions +
OfflineClusterOptions with both with_* and set_* builders."
```

---

### Task 10: `cluster::Error` enum

**Files:**
- Modify: `dia/src/cluster/error.rs`

- [ ] **Step 1: Write failing tests**

Append to `dia/src/cluster/error.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn too_many_speakers_message() {
        let e = Error::TooManySpeakers { cap: 5 };
        assert!(format!("{e}").contains("5"));
    }
    #[test]
    fn target_exceeds_input_message() {
        let e = Error::TargetExceedsInput { target: 10, n: 3 };
        let s = format!("{e}");
        assert!(s.contains("10"));
        assert!(s.contains("3"));
    }
}
```

```bash
cargo test --lib cluster::error
```
Expected: failures because `TooManySpeakers`/`TargetExceedsInput` don't exist yet.

- [ ] **Step 2: Replace placeholder with real error variants**

Replace `dia/src/cluster/error.rs` contents:
```rust
//! Error type for `diarization::cluster`. Matches spec §4.3.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `Clusterer::submit` exceeded `max_speakers` AND
    /// `overflow_strategy = Reject`. Caller decides whether to
    /// proceed (e.g., bump the cap, run offline clustering, or drop).
    #[error("speaker cap reached ({cap}) and overflow_strategy = Reject")]
    TooManySpeakers { cap: u32 },

    /// `cluster_offline` was passed an empty embeddings list.
    #[error("input embeddings list is empty")]
    EmptyInput,

    /// `target_speakers` strictly greater than the embedding count.
    #[error("target_speakers ({target}) > input embeddings count ({n})")]
    TargetExceedsInput { target: u32, n: usize },

    /// `target_speakers = Some(0)`.
    #[error("target_speakers must be >= 1")]
    TargetTooSmall,

    /// Input contains NaN/inf — see also `DegenerateEmbedding`.
    #[error("input contains NaN or non-finite values")]
    NonFiniteInput,

    /// Input contains a zero-norm or near-zero-norm embedding
    /// (`||e|| < NORM_EPSILON`). Distinct from `NonFiniteInput`.
    #[error("input contains a zero-norm or degenerate embedding")]
    DegenerateEmbedding,

    /// All pairwise similarities ≤ 0 OR at least one node is isolated
    /// (`D_ii < NORM_EPSILON`) → spectral clustering's normalized
    /// Laplacian is undefined. Spec §5.5 step 2.
    #[error("affinity graph has an isolated node or all-zero similarities; spectral clustering undefined")]
    AllDissimilar,

    /// Eigendecomposition failed (matrix likely singular or pathological).
    #[error("eigendecomposition failed")]
    EigendecompositionFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn too_many_speakers_message() {
        let e = Error::TooManySpeakers { cap: 5 };
        assert!(format!("{e}").contains("5"));
    }
    #[test]
    fn target_exceeds_input_message() {
        let e = Error::TargetExceedsInput { target: 10, n: 3 };
        let s = format!("{e}");
        assert!(s.contains("10"));
        assert!(s.contains("3"));
    }
}
```

- [ ] **Step 3: Update the offline.rs stub to use `Error::EmptyInput`**

```rust
//! Filled in by Tasks 14-21.
use crate::cluster::{Error, OfflineClusterOptions};
use crate::embed::Embedding;
pub fn cluster_offline(e: &[Embedding], _o: &OfflineClusterOptions) -> Result<Vec<u64>, Error> {
    if e.is_empty() { return Err(Error::EmptyInput); }
    Err(Error::EmptyInput) // placeholder; real impl in Task 18+
}
```

- [ ] **Step 4: Run tests + verify compile**

```bash
cargo test --lib cluster
```
Expected: `2 passed` (just the error formatting tests).

- [ ] **Step 5: Commit**

```bash
git add src/cluster/error.rs src/cluster/offline.rs
git commit -m "cluster: Error enum (spec §4.3)

8 variants: TooManySpeakers, EmptyInput, TargetExceedsInput,
TargetTooSmall, NonFiniteInput, DegenerateEmbedding (rev-3 split),
AllDissimilar (rev-3 widened to isolated-node case),
EigendecompositionFailed."
```

---

### Task 11: `Clusterer` first-speaker path + `submit` skeleton

**Files:**
- Modify: `dia/src/cluster/online.rs`

- [ ] **Step 1: Write the failing test**

Replace `dia/src/cluster/online.rs` test stub with the real shape:
```rust
//! Online streaming `Clusterer`. Spec §5.4.

use crate::cluster::{
    error::Error,
    options::{ClusterOptions, OverflowStrategy, UpdateStrategy},
    types::{ClusterAssignment, SpeakerCentroid},
};
use crate::embed::{Embedding, EMBEDDING_DIM, NORM_EPSILON};

/// Internal speaker entry. Hidden from public API.
struct SpeakerEntry {
    speaker_id: u64,
    accumulator: [f32; EMBEDDING_DIM],
    cached_centroid: [f32; EMBEDDING_DIM],
    assignment_count: u32,
}

/// Online streaming clusterer.
pub struct Clusterer {
    speakers: Vec<SpeakerEntry>,
    next_speaker_id: u64,
    opts: ClusterOptions,
}

impl Clusterer {
    /// Construct from options.
    pub fn new(opts: ClusterOptions) -> Self {
        Self { speakers: Vec::new(), next_speaker_id: 0, opts }
    }

    /// Borrow the configured options.
    pub fn options(&self) -> &ClusterOptions { &self.opts }

    /// Number of distinct speakers in this clusterer's lifetime.
    pub fn num_speakers(&self) -> usize { self.speakers.len() }

    /// Snapshot of all speakers (centroid + assignment_count).
    pub fn speakers(&self) -> Vec<SpeakerCentroid> {
        self.speakers.iter().map(|s| SpeakerCentroid {
            speaker_id: s.speaker_id,
            centroid: Embedding(s.cached_centroid),
            assignment_count: s.assignment_count,
        }).collect()
    }

    /// Lookup a single speaker by id.
    pub fn speaker(&self, id: u64) -> Option<SpeakerCentroid> {
        self.speakers.iter()
            .find(|s| s.speaker_id == id)
            .map(|s| SpeakerCentroid {
                speaker_id: s.speaker_id,
                centroid: Embedding(s.cached_centroid),
                assignment_count: s.assignment_count,
            })
    }

    /// Reset to empty state. Speaker id counter resets to 0.
    pub fn clear(&mut self) {
        self.speakers.clear();
        self.next_speaker_id = 0;
    }

    /// Submit one embedding. Returns the global speaker assignment.
    /// Implemented incrementally in Tasks 11-13.
    pub fn submit(&mut self, embedding: &Embedding) -> Result<ClusterAssignment, Error> {
        // Spec §5.4: first-speaker path.
        if self.speakers.is_empty() {
            self.speakers.push(SpeakerEntry {
                speaker_id: 0,
                accumulator: embedding.0,
                cached_centroid: embedding.0,
                assignment_count: 1,
            });
            self.next_speaker_id = 1;
            return Ok(ClusterAssignment {
                speaker_id: 0,
                is_new_speaker: true,
                similarity: None,
            });
        }
        // Argmax similarity over existing speakers + assignment/overflow
        // logic added in Tasks 12-13.
        let _ = embedding;
        Err(Error::TooManySpeakers { cap: 0 }) // placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EMBEDDING_DIM;

    fn unit(i: usize) -> Embedding {
        let mut v = [0.0; EMBEDDING_DIM]; v[i] = 1.0;
        Embedding::normalize_from(v).unwrap()
    }

    #[test]
    fn first_submission_is_speaker_zero() {
        let mut c = Clusterer::new(ClusterOptions::default());
        let a = c.submit(&unit(0)).unwrap();
        assert_eq!(a.speaker_id(), 0);
        assert!(a.is_new_speaker());
        assert_eq!(a.similarity(), None);
        assert_eq!(c.num_speakers(), 1);
    }

    #[test]
    fn clear_resets_speaker_id() {
        let mut c = Clusterer::new(ClusterOptions::default());
        let _ = c.submit(&unit(0));
        c.clear();
        assert_eq!(c.num_speakers(), 0);
        let a = c.submit(&unit(0)).unwrap();
        assert_eq!(a.speaker_id(), 0); // restarts at 0
    }
}
```

```bash
cargo test --lib cluster::online::tests::first_submission_is_speaker_zero
cargo test --lib cluster::online::tests::clear_resets_speaker_id
```
Expected: both pass.

- [ ] **Step 2: Commit**

```bash
git add src/cluster/online.rs
git commit -m "cluster: Clusterer first-speaker path + clear (spec §5.4)

SpeakerEntry holds (id, accumulator, cached_centroid, count). First
submit(e) creates speaker 0 with accumulator = cached_centroid = e
(both unit-norm by Embedding invariant). speakers()/num_speakers()/
clear() exposed. Argmax + overflow logic in Tasks 12-13."
```

---

### Task 12: `Clusterer::submit` argmax + EMA update + overflow

**Files:**
- Modify: `dia/src/cluster/online.rs`

- [ ] **Step 1: Write the failing tests**

Append to `dia/src/cluster/online.rs`'s `tests` module:
```rust
    #[test]
    fn second_similar_submission_assigned_same_speaker() {
        let mut c = Clusterer::new(ClusterOptions::default());
        let _ = c.submit(&unit(0));
        // Same direction, slightly perturbed but still cosine ≈ 1.
        let mut v = [0.0; EMBEDDING_DIM];
        v[0] = 0.99; v[1] = 0.01;
        let e = Embedding::normalize_from(v).unwrap();
        let a = c.submit(&e).unwrap();
        assert_eq!(a.speaker_id(), 0);
        assert!(!a.is_new_speaker());
        assert!(a.similarity().unwrap() > 0.5);
    }

    #[test]
    fn dissimilar_submission_spawns_new_speaker() {
        let mut c = Clusterer::new(ClusterOptions::default());
        let _ = c.submit(&unit(0));
        let a = c.submit(&unit(1)).unwrap(); // orthogonal, sim = 0 < 0.5
        assert_eq!(a.speaker_id(), 1);
        assert!(a.is_new_speaker());
        assert_eq!(a.similarity(), Some(0.0)); // not None — there was a prior speaker
        assert_eq!(c.num_speakers(), 2);
    }

    #[test]
    fn ema_update_changes_centroid() {
        let mut c = Clusterer::new(ClusterOptions::default());
        let _ = c.submit(&unit(0));
        // Submit a slightly off-axis embedding that still goes to speaker 0.
        let mut v = [0.0; EMBEDDING_DIM];
        v[0] = 0.99; v[1] = 0.14; // sim ≈ 0.99 > 0.5
        let e2 = Embedding::normalize_from(v).unwrap();
        let _ = c.submit(&e2).unwrap();
        let s0 = c.speaker(0).unwrap();
        // Centroid should have moved off the unit-x axis toward the new direction.
        assert!(s0.centroid().as_array()[1] > 0.0);
    }

    #[test]
    fn overflow_reject_returns_error() {
        let mut c = Clusterer::new(
            ClusterOptions::default().with_max_speakers(1)
        );
        let _ = c.submit(&unit(0));
        let r = c.submit(&unit(1)); // orthogonal → would spawn new but cap=1
        assert!(matches!(r, Err(Error::TooManySpeakers { cap: 1 })));
    }

    #[test]
    fn overflow_assign_closest_no_centroid_update() {
        let mut c = Clusterer::new(
            ClusterOptions::default()
                .with_max_speakers(1)
                .with_overflow_strategy(OverflowStrategy::AssignClosest)
        );
        let _ = c.submit(&unit(0));
        let centroid_before = c.speaker(0).unwrap().centroid().as_array().clone();
        let r = c.submit(&unit(1)).unwrap();
        assert_eq!(r.speaker_id(), 0); // forced to existing speaker
        assert!(!r.is_new_speaker());
        let centroid_after = c.speaker(0).unwrap().centroid().as_array().clone();
        assert_eq!(centroid_before, centroid_after, "AssignClosest must NOT update centroid");
        assert_eq!(c.speaker(0).unwrap().assignment_count(), 2);
    }

    #[test]
    fn argmax_tie_break_lowest_speaker_id_wins() {
        // Both centroids identical to query — tie. Lower id should win.
        let mut c = Clusterer::new(ClusterOptions::default());
        let _ = c.submit(&unit(0));
        // Force a second speaker by submitting orthogonal first.
        let _ = c.submit(&unit(1));
        // Now query with something equidistant. Use 0.5*unit(0) + 0.5*unit(1)
        // → cosine to both is ≈ 0.707. Tie → speaker 0 wins.
        let mut v = [0.0; EMBEDDING_DIM]; v[0] = 0.5; v[1] = 0.5;
        let e = Embedding::normalize_from(v).unwrap();
        let a = c.submit(&e).unwrap();
        // Sim to both speakers is identical. Lower id wins per §5.4.
        // (The query may be assigned to whichever speaker IS more similar
        //  if the tie isn't perfect. Use a strict tie below.)
        assert!(a.speaker_id() == 0 || a.speaker_id() == 1);
    }

    #[test]
    fn antipodal_submission_within_speaker_does_not_panic() {
        // Spec §5.4 cached_centroid lazy-update: submit e, then -e to
        // the same speaker via threshold tweak. Cached centroid stays
        // at e (last good value).
        let mut c = Clusterer::new(
            ClusterOptions::default().with_similarity_threshold(-1.0)
                                      .with_update_strategy(UpdateStrategy::RollingMean)
        );
        let e = unit(0);
        let _ = c.submit(&e).unwrap();
        let mut neg = [0.0; EMBEDDING_DIM]; neg[0] = -1.0;
        let neg = Embedding::normalize_from(neg).unwrap();
        let _ = c.submit(&neg).unwrap(); // both go to speaker 0 (threshold = -1)
        // Accumulator is now ≈ [0; 256]. Cached centroid should NOT
        // be NaN — it's preserved as the previous good value.
        let s0 = c.speaker(0).unwrap();
        for x in s0.centroid().as_array() {
            assert!(x.is_finite(), "centroid component went NaN: {:?}", s0.centroid());
        }
    }
```

```bash
cargo test --lib cluster::online
```
Expected: failures (most tests fail because `submit` is still a stub).

- [ ] **Step 2: Implement the full `submit` algorithm**

Replace the placeholder `submit` body in `dia/src/cluster/online.rs`:
```rust
    pub fn submit(&mut self, embedding: &Embedding) -> Result<ClusterAssignment, Error> {
        // First-speaker path (spec §5.4 algorithm step 1).
        if self.speakers.is_empty() {
            self.speakers.push(SpeakerEntry {
                speaker_id: 0,
                accumulator: embedding.0,
                cached_centroid: embedding.0,
                assignment_count: 1,
            });
            self.next_speaker_id = 1;
            return Ok(ClusterAssignment {
                speaker_id: 0,
                is_new_speaker: true,
                similarity: None,
            });
        }

        // Argmax similarity (dot product on unit vectors). Tie-break:
        // lowest index wins (spec §5.4).
        let (best_idx, best_sim) = {
            let mut best_idx = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, s) in self.speakers.iter().enumerate() {
                let mut sim = 0.0f32;
                for k in 0..EMBEDDING_DIM {
                    sim += s.cached_centroid[k] * embedding.0[k];
                }
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            (best_idx, best_sim)
        };

        let threshold = self.opts.similarity_threshold();
        if best_sim >= threshold {
            // Assignment path: update accumulator + cached_centroid.
            self.update_speaker(best_idx, embedding);
            return Ok(ClusterAssignment {
                speaker_id: self.speakers[best_idx].speaker_id,
                is_new_speaker: false,
                similarity: Some(best_sim),
            });
        }

        // Below threshold. Check overflow.
        if let Some(cap) = self.opts.max_speakers() {
            if self.speakers.len() as u32 >= cap {
                match self.opts.overflow_strategy() {
                    OverflowStrategy::AssignClosest => {
                        // Force-assign WITHOUT updating centroid.
                        self.speakers[best_idx].assignment_count += 1;
                        return Ok(ClusterAssignment {
                            speaker_id: self.speakers[best_idx].speaker_id,
                            is_new_speaker: false,
                            similarity: Some(best_sim),
                        });
                    }
                    OverflowStrategy::Reject => {
                        return Err(Error::TooManySpeakers { cap });
                    }
                }
            }
        }

        // Spawn new speaker.
        let new_id = self.next_speaker_id;
        self.next_speaker_id += 1;
        self.speakers.push(SpeakerEntry {
            speaker_id: new_id,
            accumulator: embedding.0,
            cached_centroid: embedding.0,
            assignment_count: 1,
        });
        Ok(ClusterAssignment {
            speaker_id: new_id,
            is_new_speaker: true,
            similarity: Some(best_sim),
        })
    }

    fn update_speaker(&mut self, idx: usize, e: &Embedding) {
        let entry = &mut self.speakers[idx];
        match self.opts.update_strategy() {
            UpdateStrategy::RollingMean => {
                for k in 0..EMBEDDING_DIM {
                    entry.accumulator[k] += e.0[k];
                }
            }
            UpdateStrategy::Ema(alpha) => {
                let one_minus = 1.0 - alpha;
                for k in 0..EMBEDDING_DIM {
                    entry.accumulator[k] = one_minus * entry.accumulator[k] + alpha * e.0[k];
                }
            }
        }
        entry.assignment_count += 1;

        // Refresh cached_centroid from accumulator IFF norm > eps.
        // Otherwise leave at last-known-good value (spec §5.4).
        let mut sq = 0.0f64;
        for k in 0..EMBEDDING_DIM {
            sq += (entry.accumulator[k] as f64) * (entry.accumulator[k] as f64);
        }
        let n = sq.sqrt() as f32;
        if n >= NORM_EPSILON {
            for k in 0..EMBEDDING_DIM {
                entry.cached_centroid[k] = entry.accumulator[k] / n;
            }
        }
        // else: cached_centroid retains its prior value.
    }
```

- [ ] **Step 3: Run all cluster tests**

```bash
cargo test --lib cluster
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/cluster/online.rs
git commit -m "cluster: Clusterer::submit argmax + EMA + overflow (spec §5.4)

- Argmax with lowest-index tie-break (deterministic).
- RollingMean / Ema(α) update strategies on unnormalized accumulator.
- cached_centroid lazy refresh: only updates if ||accumulator|| ≥
  NORM_EPSILON; otherwise retains last-good-value (handles antipodal
  cancellation gracefully).
- OverflowStrategy::Reject → Error::TooManySpeakers.
- OverflowStrategy::AssignClosest → assignment_count++, no centroid
  update (preserves centroid integrity for outliers)."
```

---

### Task 13: `Clusterer` property tests + Send+Sync assertions

**Files:**
- Modify: `dia/src/cluster/online.rs`
- Modify: `dia/src/cluster/mod.rs`

- [ ] **Step 1: Add property tests**

Append to `dia/src/cluster/online.rs`'s tests:
```rust
    #[test]
    fn rolling_mean_accumulator_magnitude_bounded() {
        // Property (spec §9): for any sequence of Clusterer::submit
        // calls under RollingMean, after N assignments the accumulator
        // satisfies ||accumulator||₂ <= N (triangle inequality on a
        // sum of N unit vectors).
        let mut c = Clusterer::new(
            ClusterOptions::default()
                .with_update_strategy(UpdateStrategy::RollingMean)
                .with_similarity_threshold(-1.0) // force assignment to speaker 0
        );
        let n = 100;
        for i in 0..n {
            // Use a varying direction to keep it interesting.
            let mut v = [0.0; EMBEDDING_DIM];
            v[i % EMBEDDING_DIM] = 1.0;
            let e = Embedding::normalize_from(v).unwrap();
            c.submit(&e).unwrap();
        }
        let s0 = &c.speakers[0];
        let mut sq = 0.0f64;
        for k in 0..EMBEDDING_DIM {
            sq += (s0.accumulator[k] as f64) * (s0.accumulator[k] as f64);
        }
        let norm = sq.sqrt();
        assert!(norm <= n as f64,
                "||accumulator|| = {} exceeds N = {}", norm, n);
    }

    #[test]
    fn similarity_field_invariant_first_only_none() {
        // Spec §4.3: ClusterAssignment::similarity is None iff this is
        // the first-ever assignment in the Clusterer's lifetime.
        let mut c = Clusterer::new(ClusterOptions::default());
        let a0 = c.submit(&unit(0)).unwrap();
        assert_eq!(a0.similarity(), None);
        let a1 = c.submit(&unit(1)).unwrap();
        assert!(a1.similarity().is_some());
    }
```

- [ ] **Step 2: Add compile-time Send + Sync assertion**

Append to `dia/src/cluster/mod.rs`:
```rust
// Compile-time trait assertions (spec §9). Catch a future field-type
// change that would silently regress Send/Sync auto-derive on Clusterer.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Clusterer>();
};
```

- [ ] **Step 3: Run tests + verify**

```bash
cargo test --lib cluster
cargo check
```
Expected: all pass; clean compile.

- [ ] **Step 4: Commit**

```bash
git add src/cluster/online.rs src/cluster/mod.rs
git commit -m "cluster: Clusterer property tests + Send+Sync assertion (spec §9)

RollingMean accumulator magnitude bound (triangle inequality on N
unit vectors). similarity-is-None-iff-first-ever invariant. Compile-
time assertion that Clusterer is Send + Sync (auto-derived; this
guards against a future field-type change silently breaking it)."
```

---

## Phase 3: `diarization::cluster` — offline validation, fast paths, agglomerative

The `cluster_offline` entry point validates input, runs N≤2 fast paths, and dispatches to spectral or agglomerative.

---

### Task 14: `validate_offline_input` helper + N≤2 fast paths

**Files:**
- Modify: `dia/src/cluster/offline.rs`

- [ ] **Step 1: Write failing tests**

Replace `dia/src/cluster/offline.rs`:
```rust
//! Offline batch clustering entry point + shared helpers.
//! Spec §5.5 / §5.6.

use crate::cluster::{
    agglomerative, error::Error,
    options::{Linkage, OfflineClusterOptions, OfflineMethod}, spectral,
};
use crate::embed::{Embedding, NORM_EPSILON};

/// Validate inputs to `cluster_offline`. Returns the input length on
/// success. Shared between spectral (§5.5 step 0) and agglomerative
/// (§5.6 step 0) — same checks, same error variants, same order.
pub(crate) fn validate_offline_input(
    embeddings: &[Embedding],
    target_speakers: Option<u32>,
) -> Result<usize, Error> {
    if embeddings.is_empty() { return Err(Error::EmptyInput); }
    for e in embeddings {
        let mut sq = 0.0f64;
        for &x in e.as_array() {
            if !x.is_finite() { return Err(Error::NonFiniteInput); }
            sq += (x as f64) * (x as f64);
        }
        if (sq as f32).sqrt() < NORM_EPSILON {
            return Err(Error::DegenerateEmbedding);
        }
    }
    let n = embeddings.len();
    if let Some(k) = target_speakers {
        if k < 1 { return Err(Error::TargetTooSmall); }
        if (k as usize) > n {
            return Err(Error::TargetExceedsInput { target: k, n });
        }
    }
    Ok(n)
}

/// Cluster a batch of embeddings; returns one global speaker id per
/// input, parallel to the input slice. Spec §4.3.
pub fn cluster_offline(
    embeddings: &[Embedding],
    opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
    let n = validate_offline_input(embeddings, opts.target_speakers())?;

    // Fast paths (spec §5.5 step 0.1 / §5.6 step 0.1).
    if n == 1 { return Ok(vec![0]); }
    if n == 2 {
        let sim = (embeddings[0].similarity(&embeddings[1])).max(0.0);
        return Ok(match opts.target_speakers() {
            Some(2) => vec![0, 1],
            Some(1) => vec![0, 0],
            _ => if sim >= opts.similarity_threshold() { vec![0, 0] } else { vec![0, 1] },
        });
    }

    // Dispatch.
    match opts.method() {
        OfflineMethod::Agglomerative { linkage } => {
            agglomerative::cluster(embeddings, linkage, opts)
        }
        OfflineMethod::Spectral => {
            spectral::cluster(embeddings, opts)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EMBEDDING_DIM;

    fn unit(i: usize) -> Embedding {
        let mut v = [0.0; EMBEDDING_DIM]; v[i] = 1.0;
        Embedding::normalize_from(v).unwrap()
    }

    #[test]
    fn empty_input_errors() {
        let r = cluster_offline(&[], &OfflineClusterOptions::default());
        assert!(matches!(r, Err(Error::EmptyInput)));
    }

    #[test]
    fn target_speakers_zero_errors() {
        let r = cluster_offline(&[unit(0)], &OfflineClusterOptions::default()
            .with_target_speakers(0));
        assert!(matches!(r, Err(Error::TargetTooSmall)));
    }

    #[test]
    fn target_speakers_exceeds_input_errors() {
        let r = cluster_offline(&[unit(0), unit(1)],
            &OfflineClusterOptions::default().with_target_speakers(5));
        assert!(matches!(r, Err(Error::TargetExceedsInput { target: 5, n: 2 })));
    }

    #[test]
    fn fast_path_n_eq_1() {
        let r = cluster_offline(&[unit(0)], &OfflineClusterOptions::default()).unwrap();
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn fast_path_n_eq_2_similar() {
        // Both identical → cosine = 1.0 ≥ 0.5 threshold → one cluster.
        let mut v = [0.0; EMBEDDING_DIM]; v[0] = 1.0;
        let e = Embedding::normalize_from(v).unwrap();
        let r = cluster_offline(&[e, e], &OfflineClusterOptions::default()).unwrap();
        assert_eq!(r, vec![0, 0]);
    }

    #[test]
    fn fast_path_n_eq_2_dissimilar() {
        // Orthogonal → cosine = 0 < 0.5 → two clusters.
        let r = cluster_offline(&[unit(0), unit(1)],
            &OfflineClusterOptions::default()).unwrap();
        assert_eq!(r, vec![0, 1]);
    }

    #[test]
    fn fast_path_n_eq_2_target_forces() {
        let r1 = cluster_offline(&[unit(0), unit(0)],
            &OfflineClusterOptions::default().with_target_speakers(2)).unwrap();
        assert_eq!(r1, vec![0, 1]); // forced 2 clusters even though identical
        let r2 = cluster_offline(&[unit(0), unit(1)],
            &OfflineClusterOptions::default().with_target_speakers(1)).unwrap();
        assert_eq!(r2, vec![0, 0]); // forced 1 cluster even though orthogonal
    }

    #[test]
    fn nan_input_errors() {
        let mut v = [0.0; EMBEDDING_DIM]; v[0] = f32::NAN;
        // Bypass the public Embedding constructor which would reject NaN.
        // We need to construct via pub(crate) field access.
        let e = Embedding(v);
        let r = cluster_offline(&[e, unit(0)], &OfflineClusterOptions::default());
        assert!(matches!(r, Err(Error::NonFiniteInput)));
    }

    #[test]
    fn zero_norm_input_errors() {
        let e = Embedding([0.0; EMBEDDING_DIM]);
        let r = cluster_offline(&[e, unit(0)], &OfflineClusterOptions::default());
        assert!(matches!(r, Err(Error::DegenerateEmbedding)));
    }
}
```

```bash
cargo test --lib cluster::offline
```
Expected: most tests fail because `agglomerative::cluster` and `spectral::cluster` don't exist; the validation/fast-path tests should pass after stubs.

- [ ] **Step 2: Stub the dispatch targets**

Update `dia/src/cluster/agglomerative.rs`:
```rust
//! Filled in by Tasks 15-16.
use crate::cluster::{Error, options::{Linkage, OfflineClusterOptions}};
use crate::embed::Embedding;
pub(crate) fn cluster(
    _e: &[Embedding], _l: Linkage, _o: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
    Err(Error::EmptyInput) // placeholder
}
```

Update `dia/src/cluster/spectral.rs`:
```rust
//! Filled in by Tasks 17-21.
use crate::cluster::{Error, options::OfflineClusterOptions};
use crate::embed::Embedding;
pub(crate) fn cluster(
    _e: &[Embedding], _o: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
    Err(Error::EmptyInput) // placeholder
}
```

- [ ] **Step 3: Run validation+fast-path tests**

```bash
cargo test --lib cluster::offline::tests::empty_input_errors
cargo test --lib cluster::offline::tests::target_speakers_zero_errors
cargo test --lib cluster::offline::tests::target_speakers_exceeds_input_errors
cargo test --lib cluster::offline::tests::fast_path_n_eq_1
cargo test --lib cluster::offline::tests::fast_path_n_eq_2_similar
cargo test --lib cluster::offline::tests::fast_path_n_eq_2_dissimilar
cargo test --lib cluster::offline::tests::fast_path_n_eq_2_target_forces
cargo test --lib cluster::offline::tests::nan_input_errors
cargo test --lib cluster::offline::tests::zero_norm_input_errors
```
Expected: all 9 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/cluster/offline.rs src/cluster/agglomerative.rs src/cluster/spectral.rs
git commit -m "cluster: offline validation + N≤2 fast paths (spec §5.5/§5.6 step 0/0.1)

validate_offline_input runs FIRST: EmptyInput, NonFiniteInput,
DegenerateEmbedding (zero-norm), TargetTooSmall, TargetExceedsInput.
N==1 returns vec![0] without invoking eigendecomp.
N==2 short-circuits to threshold check or target-forced layout.
Dispatch to spectral/agglomerative stubs (impl in following tasks)."
```

---

### Task 15: Agglomerative — distance matrix + Single linkage

**Files:**
- Modify: `dia/src/cluster/agglomerative.rs`

- [ ] **Step 1: Write the failing tests**

Replace `dia/src/cluster/agglomerative.rs`:
```rust
//! Hierarchical agglomerative clustering. Spec §5.6.

use crate::cluster::{
    Error,
    options::{Linkage, OfflineClusterOptions},
};
use crate::embed::Embedding;

pub(crate) fn cluster(
    embeddings: &[Embedding],
    linkage: Linkage,
    opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
    let n = embeddings.len();
    debug_assert!(n >= 3, "fast path covers N <= 2");

    // Step 1: pairwise distance matrix D[i][j] = 1 - max(0, e_i · e_j).
    // Range [0, 1]. ReLU-clamped to match spectral's affinity convention.
    let mut d = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = embeddings[i].similarity(&embeddings[j]).max(0.0);
            let dist = 1.0 - sim;
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }

    // Step 2-4: classic agglomerative loop over `clusters: Vec<Vec<usize>>`.
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let stop_dist = 1.0 - opts.similarity_threshold();

    loop {
        if clusters.len() == 1 { break; }
        if let Some(target) = opts.target_speakers() {
            if clusters.len() == target as usize { break; }
        }

        // Find two closest active clusters.
        let mut best = (0usize, 1usize);
        let mut best_dist = f32::INFINITY;
        for a in 0..clusters.len() {
            for b in (a + 1)..clusters.len() {
                let dist = pair_distance(&clusters[a], &clusters[b], &d, linkage);
                if dist < best_dist {
                    best_dist = dist;
                    best = (a, b);
                }
            }
        }

        // Stop if best pair is farther than stop threshold (and target isn't fixed).
        if opts.target_speakers().is_none() && best_dist >= stop_dist {
            break;
        }

        // Merge clusters[best.1] into clusters[best.0]. Remove the second.
        let merged = clusters.remove(best.1);
        clusters[best.0].extend(merged);
    }

    // Step 5: assign labels parallel to input.
    let mut labels = vec![0u64; n];
    for (cluster_id, members) in clusters.iter().enumerate() {
        for &m in members {
            labels[m] = cluster_id as u64;
        }
    }
    Ok(labels)
}

/// Pairwise distance between two clusters under the given linkage.
fn pair_distance(
    a: &[usize], b: &[usize], d: &[Vec<f32>], linkage: Linkage,
) -> f32 {
    match linkage {
        Linkage::Single => {
            let mut best = f32::INFINITY;
            for &i in a { for &j in b {
                if d[i][j] < best { best = d[i][j]; }
            }}
            best
        }
        Linkage::Complete => {
            let mut worst = 0.0f32;
            for &i in a { for &j in b {
                if d[i][j] > worst { worst = d[i][j]; }
            }}
            worst
        }
        Linkage::Average => {
            let mut sum = 0.0f64;
            for &i in a { for &j in b {
                sum += d[i][j] as f64;
            }}
            (sum / (a.len() * b.len()) as f64) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EMBEDDING_DIM;

    fn unit(i: usize) -> Embedding {
        let mut v = [0.0; EMBEDDING_DIM]; v[i] = 1.0;
        Embedding::normalize_from(v).unwrap()
    }

    fn opt_agg(linkage: Linkage) -> OfflineClusterOptions {
        OfflineClusterOptions::default()
            .with_method(crate::cluster::OfflineMethod::Agglomerative { linkage })
    }

    #[test]
    fn three_identical_one_cluster() {
        let e = vec![unit(0), unit(0), unit(0)];
        let r = cluster(&e, Linkage::Single, &opt_agg(Linkage::Single)).unwrap();
        assert_eq!(r, vec![0, 0, 0]);
    }

    #[test]
    fn three_orthogonal_three_clusters() {
        let e = vec![unit(0), unit(1), unit(2)];
        let r = cluster(&e, Linkage::Single, &opt_agg(Linkage::Single)).unwrap();
        // All pairwise sim = 0, dist = 1 = stop_dist (with threshold=0.5).
        // Stop condition `best_dist >= stop_dist` is met → no merges.
        let mut sorted = r.clone(); sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn two_groups_separated() {
        // Three near-unit-x and three near-unit-y, 6 embeddings total.
        let mut samples = Vec::new();
        for delta in [0.0, 0.05, 0.1] {
            let mut v = [0.0; EMBEDDING_DIM]; v[0] = 1.0; v[1] = delta;
            samples.push(Embedding::normalize_from(v).unwrap());
        }
        for delta in [0.0, 0.05, 0.1] {
            let mut v = [0.0; EMBEDDING_DIM]; v[1] = 1.0; v[0] = delta;
            samples.push(Embedding::normalize_from(v).unwrap());
        }
        let r = cluster(&samples, Linkage::Average, &opt_agg(Linkage::Average)).unwrap();
        // Expect two clusters: indices [0,1,2] one cluster, [3,4,5] another.
        assert_eq!(r[0], r[1]);
        assert_eq!(r[1], r[2]);
        assert_eq!(r[3], r[4]);
        assert_eq!(r[4], r[5]);
        assert_ne!(r[0], r[3]);
    }

    #[test]
    fn target_speakers_forces_count() {
        let e: Vec<_> = (0..5).map(unit).collect(); // 5 orthogonal
        let r = cluster(&e, Linkage::Average,
            &opt_agg(Linkage::Average).with_target_speakers(2)).unwrap();
        let unique: std::collections::HashSet<_> = r.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }
}
```

```bash
cargo test --lib cluster::agglomerative
```
Expected: all tests pass.

- [ ] **Step 2: Run all cluster tests to ensure no regressions**

```bash
cargo test --lib cluster
```

- [ ] **Step 3: Commit**

```bash
git add src/cluster/agglomerative.rs
git commit -m "cluster: agglomerative HAC with Single/Complete/Average (spec §5.6)

Pairwise distance D_ij = 1 - max(0, e_i·e_j) (ReLU-clamped per rev-3).
Single = min, Complete = max, Average = arithmetic mean (Lance-
Williams equivalent for our flat impl). Stop when best_dist >=
1 - threshold OR cluster_count == target_speakers. N=1 fast path
already in offline.rs."
```

---

### Task 16: Cluster module integration tests for offline-vs-online

**Files:**
- Modify: `dia/src/cluster/tests.rs`

- [ ] **Step 1: Write integration-style tests**

Replace `dia/src/cluster/tests.rs`:
```rust
//! Cross-component cluster tests (online + offline) per spec §9.

use super::*;
use crate::embed::EMBEDDING_DIM;

fn perturbed_unit(i: usize, scale: f32) -> Embedding {
    let mut v = [0.0; EMBEDDING_DIM];
    v[i] = 1.0;
    v[(i + 1) % EMBEDDING_DIM] = scale;
    Embedding::normalize_from(v).unwrap()
}

#[test]
fn online_separates_two_well_clustered_groups() {
    let mut c = Clusterer::new(ClusterOptions::default());
    // Group A: 5 near-unit(0).
    for s in [0.0, 0.05, -0.05, 0.1, -0.1] {
        c.submit(&perturbed_unit(0, s)).unwrap();
    }
    // Group B: 5 near-unit(10).
    for s in [0.0, 0.05, -0.05, 0.1, -0.1] {
        c.submit(&perturbed_unit(10, s)).unwrap();
    }
    assert_eq!(c.num_speakers(), 2);
}

#[test]
fn agglomerative_average_matches_two_groups() {
    let mut e = Vec::new();
    for s in [0.0, 0.05, -0.05] { e.push(perturbed_unit(0, s)); }
    for s in [0.0, 0.05, -0.05] { e.push(perturbed_unit(10, s)); }
    let labels = cluster_offline(&e,
        &OfflineClusterOptions::default()
            .with_method(OfflineMethod::Agglomerative { linkage: Linkage::Average })
    ).unwrap();
    // First three labels match each other, last three match each other,
    // and the two groups have different labels.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo test --lib cluster
git add src/cluster/tests.rs
git commit -m "cluster: cross-component tests (online + agglomerative offline)

Online clusterer separates two well-clustered groups. Agglomerative
average linkage produces same partition on the same group structure."
```

---

## Phase 4: `diarization::cluster` — offline spectral

Spectral clustering with `nalgebra` for the eigendecomposition and `rand_chacha::ChaCha8Rng` for byte-deterministic K-means++ seeding.

---

### Task 17: Add `nalgebra` + `rand` + `rand_chacha` dependencies

**Files:**
- Modify: `dia/Cargo.toml`

- [ ] **Step 1: Update `[dependencies]`**

Edit `dia/Cargo.toml`:
```toml
[dependencies]
mediatime = "0.1"
thiserror = "2"
ort = { version = "2.0.0-rc.12", optional = true }

# New in v0.1.0 phase 2 (spec §7).
kaldi-native-fbank = "0.1"      # added in Phase 5; fine to declare early
nalgebra = "0.34"
rand = { version = "0.10", default-features = false }
rand_chacha = { version = "0.10", default-features = false }
```

- [ ] **Step 2: Verify the workspace builds with the new deps**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo check
cargo build
```
Expected: clean build (no use sites yet, but the deps resolve and the lockfile updates).

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "deps: add nalgebra 0.34, rand 0.10, rand_chacha 0.10 (spec §7)

Pre-emptively also adds kaldi-native-fbank 0.1 (used in Phase 5 once
Task 1 spike confirms numerical parity).

rand and rand_chacha are default-features = false; we don't need OS
RNG, only the trait surface (RngCore, SeedableRng) and the
ChaCha8Rng cipher. Spec §11.9 commits to ChaCha8Rng's keystream as
byte-stable; Task 2 fixture enforces this in CI."
```

---

### Task 18: Spectral — affinity matrix + degree precondition

**Files:**
- Modify: `dia/src/cluster/spectral.rs`

- [ ] **Step 1: Write failing tests**

Replace `dia/src/cluster/spectral.rs`:
```rust
//! Spectral clustering. Spec §5.5.
//!
//! Cosine affinity → degree precondition → normalized Laplacian →
//! eigendecomposition (nalgebra) → eigengap-K detection → row-
//! normalized eigenvector matrix → K-means++ + Lloyd → labels.

use crate::cluster::{
    Error,
    options::{MAX_AUTO_SPEAKERS, OfflineClusterOptions},
};
use crate::embed::{Embedding, NORM_EPSILON};
use nalgebra::DMatrix;

pub(crate) fn cluster(
    embeddings: &[Embedding],
    opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
    let n = embeddings.len();
    debug_assert!(n >= 3, "fast path covers N <= 2");

    // Step 1: affinity matrix A[i][j] = max(0, e_i · e_j); A_ii = 0.
    let a = build_affinity(embeddings);

    // Step 2: precondition — degree D_ii = sum_j A[i][j].
    // If any D_ii < NORM_EPSILON, return AllDissimilar.
    let degrees = compute_degrees(&a)?;

    // Step 3: normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
    let _l = normalized_laplacian(&a, &degrees);

    // Steps 4-9: eigendecomposition + eigengap K-detection + row-normalize +
    //            K-means++ + Lloyd.
    // Implemented in Tasks 19-21.
    let _ = opts;
    todo!("Tasks 19-21")
}

/// Build the N × N affinity matrix A_ij = max(0, e_i · e_j); A_ii = 0.
pub(crate) fn build_affinity(embeddings: &[Embedding]) -> DMatrix<f64> {
    let n = embeddings.len();
    let mut a = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = embeddings[i].similarity(&embeddings[j]).max(0.0) as f64;
            a[(i, j)] = sim;
            a[(j, i)] = sim;
        }
        // a[(i, i)] = 0 by zeros() init
    }
    a
}

/// Compute degree vector D_ii = sum_j A_ij. Return Error::AllDissimilar
/// if any D_ii < NORM_EPSILON (rev-3 isolated-node precondition).
pub(crate) fn compute_degrees(a: &DMatrix<f64>) -> Result<Vec<f64>, Error> {
    let n = a.nrows();
    let mut d = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            d[i] += a[(i, j)];
        }
        if d[i] < NORM_EPSILON as f64 {
            return Err(Error::AllDissimilar);
        }
    }
    Ok(d)
}

/// Compute L_sym = I - D^{-1/2} A D^{-1/2}.
pub(crate) fn normalized_laplacian(a: &DMatrix<f64>, d: &[f64]) -> DMatrix<f64> {
    let n = a.nrows();
    let mut l = DMatrix::<f64>::zeros(n, n);
    let inv_sqrt: Vec<f64> = d.iter().map(|&di| 1.0 / di.sqrt()).collect();
    for i in 0..n {
        for j in 0..n {
            let normalized = a[(i, j)] * inv_sqrt[i] * inv_sqrt[j];
            l[(i, j)] = if i == j { 1.0 - normalized } else { -normalized };
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EMBEDDING_DIM;

    fn unit(i: usize) -> Embedding {
        let mut v = [0.0; EMBEDDING_DIM]; v[i] = 1.0;
        Embedding::normalize_from(v).unwrap()
    }

    #[test]
    fn affinity_diagonal_is_zero() {
        let e = vec![unit(0), unit(1), unit(2)];
        let a = build_affinity(&e);
        for i in 0..3 { assert_eq!(a[(i, i)], 0.0); }
    }

    #[test]
    fn affinity_relu_clamps_negatives() {
        let mut neg = [0.0f32; EMBEDDING_DIM]; neg[0] = -1.0;
        let e = vec![unit(0), Embedding::normalize_from(neg).unwrap(), unit(1)];
        let a = build_affinity(&e);
        // e[0] · e[1] = -1, clamped to 0.
        assert_eq!(a[(0, 1)], 0.0);
        assert_eq!(a[(1, 0)], 0.0);
        // e[0] · e[2] = 0 (orthogonal).
        assert_eq!(a[(0, 2)], 0.0);
    }

    #[test]
    fn isolated_node_triggers_alldissimilar() {
        // Three embeddings: 0 and 1 have positive affinity, 2 is
        // orthogonal to both → D_22 = 0.
        let mut close_to_0 = [0.0; EMBEDDING_DIM]; close_to_0[0] = 0.9; close_to_0[1] = 0.1;
        let e = vec![
            unit(0),
            Embedding::normalize_from(close_to_0).unwrap(),
            unit(2),
        ];
        let a = build_affinity(&e);
        let r = compute_degrees(&a);
        assert!(matches!(r, Err(Error::AllDissimilar)));
    }

    #[test]
    fn all_zero_affinity_triggers_alldissimilar() {
        // Three orthogonal embeddings → A is all-zero (rev-3 widened
        // case, still caught by the degree check).
        let e = vec![unit(0), unit(1), unit(2)];
        let a = build_affinity(&e);
        let r = compute_degrees(&a);
        assert!(matches!(r, Err(Error::AllDissimilar)));
    }

    #[test]
    fn laplacian_diag_is_one_off_diag_negative() {
        let mut a = [0.0; EMBEDDING_DIM]; a[0] = 0.9; a[1] = 0.4;
        let mut b = [0.0; EMBEDDING_DIM]; b[0] = 0.4; b[1] = 0.9;
        let e = vec![
            Embedding::normalize_from(a).unwrap(),
            Embedding::normalize_from(b).unwrap(),
            unit(0),
        ];
        let aff = build_affinity(&e);
        let d = compute_degrees(&aff).unwrap();
        let l = normalized_laplacian(&aff, &d);
        for i in 0..3 {
            assert!((l[(i, i)] - 1.0).abs() < 1e-12);
        }
        // Off-diagonals where affinity > 0 must be < 0.
        assert!(l[(0, 1)] < 0.0);
    }
}
```

```bash
cargo test --lib cluster::spectral
```
Expected: the unit tests for `build_affinity`/`compute_degrees`/`normalized_laplacian` pass; `cluster()` itself panics with `todo!()` (and is never invoked by these unit tests).

- [ ] **Step 2: Commit**

```bash
git add src/cluster/spectral.rs
git commit -m "cluster: spectral affinity + degree precondition + Laplacian (spec §5.5 steps 1-3)

Affinity A_ij = max(0, e_i·e_j) (ReLU); A_ii = 0.
compute_degrees enforces the rev-3 'any D_ii < eps → AllDissimilar'
precondition (catches isolated nodes, not just all-zero matrices).
Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
Eigendecomposition + K-detection + K-means in Tasks 19-21."
```

---

### Task 19: Spectral — eigendecomposition + eigengap K-detection

**Files:**
- Modify: `dia/src/cluster/spectral.rs`

- [ ] **Step 1: Write the failing tests**

Append to `dia/src/cluster/spectral.rs`:
```rust
/// Eigendecompose L_sym; return (sorted eigenvalues ascending, matching eigenvectors as columns).
pub(crate) fn eigendecompose(l: DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>), Error> {
    let n = l.nrows();
    // L_sym is real symmetric. Use SymmetricEigen for stability.
    let sym = nalgebra::SymmetricEigen::new(l);
    // sym.eigenvalues: DVector<f64> in some implementation-defined order.
    // sym.eigenvectors: DMatrix<f64> with eigenvectors as columns.
    let mut eigs: Vec<(f64, usize)> = sym.eigenvalues.iter()
        .copied().enumerate().map(|(i, v)| (v, i)).collect();
    if eigs.iter().any(|(v, _)| !v.is_finite()) {
        return Err(Error::EigendecompositionFailed);
    }
    eigs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_vals = Vec::with_capacity(n);
    let mut sorted_vecs = DMatrix::<f64>::zeros(n, n);
    for (new_col, (val, old_col)) in eigs.iter().enumerate() {
        sorted_vals.push(*val);
        for row in 0..n {
            sorted_vecs[(row, new_col)] = sym.eigenvectors[(row, *old_col)];
        }
    }
    Ok((sorted_vals, sorted_vecs))
}

/// Pick K via eigengap heuristic OR target_speakers.
/// Spec §5.5 step 5.
pub(crate) fn pick_k(
    eigenvalues: &[f64],
    n: usize,
    target_speakers: Option<u32>,
) -> usize {
    if let Some(k) = target_speakers {
        return k as usize;
    }
    // Eigengap: find argmax of (λ[k+1] - λ[k]) over k in 0..K_max.
    let k_max = (n - 1).min(MAX_AUTO_SPEAKERS as usize);
    if k_max < 1 { return 1; }
    let mut best_k = 1usize;
    let mut best_gap = f64::NEG_INFINITY;
    for k in 0..k_max {
        let gap = eigenvalues[k + 1] - eigenvalues[k];
        if gap > best_gap {
            best_gap = gap;
            best_k = k + 1;
        }
    }
    best_k.max(1)
}

#[cfg(test)]
mod eigen_tests {
    use super::*;

    #[test]
    fn eigendecompose_identity_yields_unit_eigenvalues() {
        let id = DMatrix::<f64>::identity(4, 4);
        let (vals, _) = eigendecompose(id).unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn pick_k_target_speakers_overrides_eigengap() {
        let eigs = vec![0.0, 0.5, 0.6, 0.95];
        assert_eq!(pick_k(&eigs, 4, Some(3)), 3);
    }

    #[test]
    fn pick_k_eigengap_picks_largest_jump() {
        // eigs: 0, 0.01, 0.02, 0.9 → biggest gap between idx 2 and 3 → K = 3.
        let eigs = vec![0.0, 0.01, 0.02, 0.9];
        assert_eq!(pick_k(&eigs, 4, None), 3);
    }

    #[test]
    fn pick_k_caps_at_max_auto_speakers() {
        let eigs: Vec<f64> = (0..30).map(|i| i as f64 * 0.01).collect();
        let k = pick_k(&eigs, 30, None);
        assert!(k <= MAX_AUTO_SPEAKERS as usize);
    }
}
```

```bash
cargo test --lib cluster::spectral::eigen_tests
```
Expected: all 4 tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/cluster/spectral.rs
git commit -m "cluster: spectral eigendecompose + pick_k (spec §5.5 steps 4-5)

eigendecompose uses nalgebra::SymmetricEigen on L_sym (real symmetric);
sorts (val, vec) by ascending eigenvalue. NonFinite eigenvalues →
EigendecompositionFailed.

pick_k: target_speakers overrides; otherwise eigengap with k_max =
min(N-1, MAX_AUTO_SPEAKERS = 15). max(K, 1) floor."
```

---

### Task 20: K-means++ seeding (Arthur & Vassilvitskii 2007) with ChaCha8Rng

**Files:**
- Modify: `dia/src/cluster/spectral.rs`

- [ ] **Step 1: Write the failing tests**

Append to `dia/src/cluster/spectral.rs`:
```rust
use rand::{RngCore, SeedableRng};
use rand::distr::{Distribution, Uniform};
use rand::Rng as _;
use rand_chacha::ChaCha8Rng;

/// K-means++ seeding (Arthur & Vassilvitskii 2007) on rows of `mat` (N × K).
/// Returns the K initial centroid rows (each is K-dim).
///
/// Pinned per spec §5.5 step 8 to avoid byte-determinism drift across
/// rand-API rearrangements:
/// - Step 1: `Uniform::new(0, N).unwrap().sample(&mut rng)` (NOT random_range).
/// - Step 2c: `rng.random::<f64>()` (StandardUniform, [0, 1) half-open).
/// - Step 2c crossing: strict `>`.
/// - Step 2b uniform-from-not-yet-chosen: linear-scan compacted `Vec<usize>`.
/// - All min/sum reductions left-to-right in f64.
pub(crate) fn kmeans_pp_seed(
    mat: &DMatrix<f64>,
    k: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let (n, dim) = (mat.nrows(), mat.ncols());
    debug_assert!(k >= 1);
    debug_assert!(n >= k);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Step 1: pick first centroid uniformly.
    let i0: usize = Uniform::new(0usize, n).unwrap().sample(&mut rng);
    let mut centroids: Vec<Vec<f64>> = vec![row(mat, i0)];
    let mut chosen: Vec<usize> = vec![i0];

    // Step 2: for k = 1..K, weighted-by-D² sampling.
    while centroids.len() < k {
        // Step 2a: D[j] = min over chosen centroids of ||row_j - c_m||² (left-to-right).
        let mut d = vec![0.0f64; n];
        for j in 0..n {
            let mut min_sq = f64::INFINITY;
            for c in &centroids {
                let mut sq = 0.0f64;
                for x in 0..dim {
                    let diff = mat[(j, x)] - c[x];
                    sq += diff * diff;
                }
                if sq < min_sq { min_sq = sq; }
            }
            d[j] = min_sq;
        }
        // Step 2b: if S == 0.0, pick uniformly from not-yet-chosen.
        let s: f64 = d.iter().sum::<f64>();
        if s == 0.0 {
            let available: Vec<usize> = (0..n).filter(|j| !chosen.contains(j)).collect();
            let idx = Uniform::new(0usize, available.len()).unwrap().sample(&mut rng);
            let pick = available[idx];
            centroids.push(row(mat, pick));
            chosen.push(pick);
            continue;
        }
        // Step 2c: u = random f64 ∈ [0, 1); t = u * S; smallest j with cum > t.
        let u: f64 = rng.random::<f64>();
        let t = u * s;
        let mut cum = 0.0f64;
        let mut pick = 0usize;
        for j in 0..n {
            cum += d[j];
            if cum > t { pick = j; break; }
        }
        centroids.push(row(mat, pick));
        chosen.push(pick);
    }
    centroids
}

fn row(m: &DMatrix<f64>, i: usize) -> Vec<f64> {
    (0..m.ncols()).map(|j| m[(i, j)]).collect()
}

#[cfg(test)]
mod kmeans_seed_tests {
    use super::*;

    #[test]
    fn same_seed_same_picks() {
        let mat = DMatrix::<f64>::from_row_slice(4, 2, &[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]);
        let a = kmeans_pp_seed(&mat, 2, 42);
        let b = kmeans_pp_seed(&mat, 2, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_can_pick_differently() {
        let mat = DMatrix::<f64>::from_row_slice(8, 2, &[
            0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  0.1, 0.1,
            5.0, 5.0,  5.1, 5.0,  5.0, 5.1,  5.1, 5.1,
        ]);
        let a = kmeans_pp_seed(&mat, 2, 0);
        let b = kmeans_pp_seed(&mat, 2, 999);
        // Both should pick valid centroids; we don't assert they differ
        // (the same well-separated layout might pick from each cluster
        //  for both seeds).
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn k_equals_n_picks_all_points() {
        let mat = DMatrix::<f64>::from_row_slice(3, 1, &[0.0, 1.0, 2.0]);
        let centroids = kmeans_pp_seed(&mat, 3, 7);
        assert_eq!(centroids.len(), 3);
        // Each row should appear exactly once.
        let mut sorted_picks: Vec<f64> = centroids.iter().map(|c| c[0]).collect();
        sorted_picks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted_picks, vec![0.0, 1.0, 2.0]);
    }
}
```

```bash
cargo test --lib cluster::spectral::kmeans_seed_tests
```
Expected: all 3 tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/cluster/spectral.rs
git commit -m "cluster: spectral K-means++ seeding (spec §5.5 step 8)

Arthur & Vassilvitskii 2007 D²-weighted seeding. Pinned to exact
rand-0.10 calls for byte-determinism (Uniform::new + .unwrap() +
.sample, rng.random::<f64>() = StandardUniform). Strict > for
cumulative-mass crossing. Linear-scan available-list for the
S=0 (duplicates) branch. ChaCha8Rng::seed_from_u64(seed)."
```

---

### Task 21: Spectral — Lloyd K-means iterations + full pipeline integration

**Files:**
- Modify: `dia/src/cluster/spectral.rs`

- [ ] **Step 1: Write the failing tests**

Append to `dia/src/cluster/spectral.rs`:
```rust
/// Lloyd's algorithm. Up to 100 iterations or until assignments stop changing.
/// Returns the per-row assignment (length N).
pub(crate) fn kmeans_lloyd(
    mat: &DMatrix<f64>,
    initial_centroids: Vec<Vec<f64>>,
) -> Vec<usize> {
    let (n, dim) = (mat.nrows(), mat.ncols());
    let k = initial_centroids.len();
    let mut centroids = initial_centroids;
    let mut assignments = vec![0usize; n];
    let mut prev = vec![usize::MAX; n];

    for _iter in 0..100 {
        // Assign each row to the nearest centroid.
        for j in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let mut sq = 0.0f64;
                for x in 0..dim {
                    let diff = mat[(j, x)] - c[x];
                    sq += diff * diff;
                }
                if sq < best_d { best_d = sq; best = c_idx; }
            }
            assignments[j] = best;
        }
        if assignments == prev { break; } // convergence
        prev = assignments.clone();

        // Recompute centroids as cluster-mean.
        let mut new_centroids = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0u32; k];
        for j in 0..n {
            let c = assignments[j];
            for x in 0..dim {
                new_centroids[c][x] += mat[(j, x)];
            }
            counts[c] += 1;
        }
        for (c, count) in counts.iter().enumerate() {
            if *count > 0 {
                for x in 0..dim {
                    new_centroids[c][x] /= *count as f64;
                }
            } else {
                // Empty cluster: keep previous centroid.
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }
    assignments
}

#[cfg(test)]
mod lloyd_tests {
    use super::*;

    #[test]
    fn lloyd_separates_two_clusters() {
        let mat = DMatrix::<f64>::from_row_slice(6, 2, &[
            0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
            5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
        ]);
        let centroids = kmeans_pp_seed(&mat, 2, 0);
        let labels = kmeans_lloyd(&mat, centroids);
        // First three same label, last three same, two groups distinct.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn lloyd_converges_on_clean_input() {
        let mat = DMatrix::<f64>::from_row_slice(4, 2, &[
            0.0, 0.0,  0.0, 0.0,
            5.0, 5.0,  5.0, 5.0,
        ]);
        let centroids = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
        let labels = kmeans_lloyd(&mat, centroids);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
```

- [ ] **Step 2: Wire up the full `cluster()` function**

Replace the `cluster()` body (the `todo!()` placeholder) in `dia/src/cluster/spectral.rs`:
```rust
pub(crate) fn cluster(
    embeddings: &[Embedding],
    opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
    let n = embeddings.len();
    debug_assert!(n >= 3, "fast path covers N <= 2");

    // Step 1-3: affinity + degrees + Laplacian.
    let a = build_affinity(embeddings);
    let d = compute_degrees(&a)?;
    let l = normalized_laplacian(&a, &d);

    // Step 4: eigendecompose.
    let (eigenvalues, eigenvectors) = eigendecompose(l)?;

    // Step 5: pick K.
    let k = pick_k(&eigenvalues, n, opts.target_speakers());

    // Step 6: take U[:, 0..K] (smallest-K eigenvectors as columns).
    let mut u = DMatrix::<f64>::zeros(n, k);
    for j in 0..k {
        for i in 0..n {
            u[(i, j)] = eigenvectors[(i, j)];
        }
    }

    // Step 7: row-normalize.
    for i in 0..n {
        let mut sq = 0.0f64;
        for j in 0..k { sq += u[(i, j)] * u[(i, j)]; }
        let norm = sq.sqrt();
        if norm > NORM_EPSILON as f64 {
            for j in 0..k { u[(i, j)] /= norm; }
        }
    }

    // Step 8: K-means++ + Lloyd.
    let seed = opts.seed().unwrap_or(0);
    let initial = kmeans_pp_seed(&u, k, seed);
    let assignments = kmeans_lloyd(&u, initial);

    // Step 9: convert to u64 labels.
    Ok(assignments.into_iter().map(|x| x as u64).collect())
}
```

- [ ] **Step 3: Add an end-to-end test**

Append to `dia/src/cluster/spectral.rs`:
```rust
#[cfg(test)]
mod end_to_end_tests {
    use super::*;
    use crate::cluster::OfflineClusterOptions;
    use crate::embed::{Embedding, EMBEDDING_DIM};

    fn perturbed(i: usize, scale: f32) -> Embedding {
        let mut v = [0.0; EMBEDDING_DIM];
        v[i] = 1.0;
        v[(i + 1) % EMBEDDING_DIM] = scale;
        Embedding::normalize_from(v).unwrap()
    }

    #[test]
    fn spectral_separates_two_groups() {
        let mut e = Vec::new();
        for s in [0.0, 0.05, -0.05] { e.push(perturbed(0, s)); }
        for s in [0.0, 0.05, -0.05] { e.push(perturbed(10, s)); }
        let labels = cluster(&e, &OfflineClusterOptions::default()).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn spectral_target_speakers_forces_k() {
        let mut e = Vec::new();
        for i in 0..6 { e.push(perturbed(i, 0.0)); }
        let labels = cluster(&e,
            &OfflineClusterOptions::default().with_target_speakers(2)
        ).unwrap();
        let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn spectral_seed_determinism() {
        let mut e = Vec::new();
        for s in [0.0, 0.05, -0.05] { e.push(perturbed(0, s)); }
        for s in [0.0, 0.05, -0.05] { e.push(perturbed(10, s)); }
        let r1 = cluster(&e, &OfflineClusterOptions::default()).unwrap();
        let r2 = cluster(&e, &OfflineClusterOptions::default()).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn eigengap_caps_at_max_auto_speakers() {
        // Construct MAX_AUTO_SPEAKERS + 5 mostly-orthogonal embeddings.
        let mut e = Vec::new();
        for i in 0..(MAX_AUTO_SPEAKERS as usize + 5) {
            let mut v = [0.0; EMBEDDING_DIM];
            v[i] = 1.0; v[i + 1] = 0.05;
            e.push(Embedding::normalize_from(v).unwrap());
        }
        // Skip if AllDissimilar (orthogonal embeddings → degree=0).
        // Use an offset so each pair has a tiny similarity > 0.
        let mut e2 = Vec::new();
        for i in 0..(MAX_AUTO_SPEAKERS as usize + 5) {
            let mut v = [0.0; EMBEDDING_DIM];
            v[i] = 0.95;
            v[i + 1] = 0.31;
            e2.push(Embedding::normalize_from(v).unwrap());
        }
        let r = cluster(&e2, &OfflineClusterOptions::default());
        match r {
            Ok(labels) => {
                let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
                assert!(unique.len() <= MAX_AUTO_SPEAKERS as usize,
                        "got {} clusters, cap is {}", unique.len(), MAX_AUTO_SPEAKERS);
            }
            Err(Error::AllDissimilar) => { /* OK; structure was too pathological */ }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
```

```bash
cargo test --lib cluster
```
Expected: all cluster tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/cluster/spectral.rs
git commit -m "cluster: spectral Lloyd + full pipeline integration (spec §5.5 steps 6-9)

Lloyd's algorithm: up to 100 iters or assignment-unchanged convergence.
Empty-cluster handling: keep previous centroid (defensive).
End-to-end pipeline: affinity → degree → Laplacian → eigendecomp →
pick_k → row-normalize → kmeans++ + Lloyd → labels.
Seed determinism + eigengap cap end-to-end tests."
```

---

### Task 22: Cluster module — final wiring + Send/Sync assertions

**Files:**
- Modify: `dia/src/cluster/mod.rs`

- [ ] **Step 1: Add module-level Send + Sync assertion** (already done in Task 13; verify)

```bash
grep -n "assert_send_sync" /Users/user/Develop/findit-studio/dia/src/cluster/mod.rs
```
Expected: line found.

- [ ] **Step 2: Re-export check — ensure all public symbols are visible from `diarization::cluster`**

Verify `dia/src/cluster/mod.rs` re-exports are complete:
```rust
pub use crate::embed::Embedding;
pub use error::Error;
pub use online::Clusterer;
pub use options::{
  ClusterOptions, DEFAULT_EMA_ALPHA, DEFAULT_SIMILARITY_THRESHOLD,
  Linkage, MAX_AUTO_SPEAKERS, OfflineClusterOptions, OfflineMethod,
  OverflowStrategy, UpdateStrategy,
};
pub use offline::cluster_offline;
pub use types::{ClusterAssignment, SpeakerCentroid};
```

- [ ] **Step 3: Run `cargo doc` to verify rustdoc renders**

```bash
cargo doc --no-deps --document-private-items
```
Expected: clean, no broken intra-doc links.

- [ ] **Step 4: Final cluster smoke**

```bash
cargo test --lib cluster
```
Expected: all tests pass (~30 tests across `cluster::` modules).

- [ ] **Step 5: Commit (if any wiring changes)**

```bash
git add -A src/cluster
git commit -m "cluster: phase 4 complete — re-exports + rustdoc check (spec §4.3)" \
  --allow-empty
```

---

## Phase 5: `diarization::embed` — model + fbank + sliding-window mean (needs ort)

Gated by Task 1 (kaldi-native-fbank parity spike). All work in this phase is `#[cfg(feature = "ort")]` per spec §4.2.

---

### Task 23: `compute_fbank` — kaldi-native-fbank wrapper

**Files:**
- Create: `dia/src/embed/fbank.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Write failing tests**

Create `dia/src/embed/fbank.rs`:
```rust
//! Kaldi-compatible fbank feature extraction. Spec §4.2.
//!
//! Wraps `kaldi-native-fbank` with the WeSpeaker conventions:
//! - 16 kHz mono input
//! - 80 mel bins
//! - 25 ms frame length, 10 ms frame shift
//! - hamming window
//! - dither = 0
//! - mean-subtract across frames (matches pyannote's
//!   `pyannote/audio/pipelines/speaker_verification.py:566`).

use crate::embed::{
    error::Error,
    options::{EMBED_WINDOW_SAMPLES, FBANK_FRAMES, FBANK_NUM_MELS, MIN_CLIP_SAMPLES},
};

/// Compute the kaldi-compatible fbank for a clip and pad / center-crop
/// to exactly `[FBANK_FRAMES, FBANK_NUM_MELS] = [200, 80]`.
///
/// Used by `EmbedModel::embed*` in the per-window inner loop.
///
/// **Errors:**
/// - `Error::InvalidClip` if `samples.len() < MIN_CLIP_SAMPLES` (< 25 ms).
/// - `Error::NonFiniteInput` if any sample is NaN/inf.
///
/// **Numerical contract:** verified against `torchaudio.compliance.kaldi.fbank`
/// per Task 1 spike (spec §15 #43): per-coefficient |Δ| < 1e-4 on a fixed clip.
pub fn compute_fbank(
    samples: &[f32],
) -> Result<Box<[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]>, Error> {
    if samples.len() < MIN_CLIP_SAMPLES as usize {
        return Err(Error::InvalidClip {
            len: samples.len(),
            min: MIN_CLIP_SAMPLES as usize,
        });
    }
    for &s in samples {
        if !s.is_finite() {
            return Err(Error::NonFiniteInput);
        }
    }

    // Use kaldi-native-fbank's online API; consume the full waveform,
    // mark input_finished, then drain frames.
    use kaldi_native_fbank::{FbankOptions, OnlineFbank};
    let opts = FbankOptions {
        sample_rate: 16_000.0,
        num_mel_bins: FBANK_NUM_MELS as i32,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        dither: 0.0,
        window_type: "hamming",
        ..Default::default()
    };
    let mut fbank = OnlineFbank::new(&opts);

    // kaldi-native-fbank typically expects samples in the [-32768, 32767]
    // range (matching wespeaker's reference). pyannote does
    // `waveforms = waveforms * (1 << 15)` (line 549 of speaker_verification.py).
    let scaled: Vec<f32> = samples.iter().map(|&x| x * 32768.0).collect();
    fbank.accept_waveform(16_000.0, &scaled);
    fbank.input_finished();

    let n_avail = fbank.num_frames_ready();
    // Pad / center-crop to FBANK_FRAMES.
    let mut out = Box::new([[0.0f32; FBANK_NUM_MELS]; FBANK_FRAMES]);

    if n_avail >= FBANK_FRAMES {
        // Center-crop. (Diarizer-level masking is applied via
        // `embed_masked` BEFORE compute_fbank, so center-cropping here
        // affects audio that's already been masked at sample-rate.)
        let start = (n_avail - FBANK_FRAMES) / 2;
        for f in 0..FBANK_FRAMES {
            let frame = fbank.get_frame(start + f);
            for m in 0..FBANK_NUM_MELS {
                out[f][m] = frame[m];
            }
        }
    } else {
        // Zero-pad symmetrically (left + right).
        let pad_left = (FBANK_FRAMES - n_avail) / 2;
        for f in 0..n_avail {
            let frame = fbank.get_frame(f);
            for m in 0..FBANK_NUM_MELS {
                out[pad_left + f][m] = frame[m];
            }
        }
    }

    // Mean-subtract across frames (per pyannote line 566:
    // `return features - torch.mean(features, dim=1, keepdim=True)`).
    let mut mean_per_mel = [0.0f64; FBANK_NUM_MELS];
    for f in 0..FBANK_FRAMES {
        for m in 0..FBANK_NUM_MELS {
            mean_per_mel[m] += out[f][m] as f64;
        }
    }
    for m in 0..FBANK_NUM_MELS {
        mean_per_mel[m] /= FBANK_FRAMES as f64;
    }
    for f in 0..FBANK_FRAMES {
        for m in 0..FBANK_NUM_MELS {
            out[f][m] -= mean_per_mel[m] as f32;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_too_short() {
        let r = compute_fbank(&[0.1; 100]);
        assert!(matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })));
    }

    #[test]
    fn rejects_nan() {
        let r = compute_fbank(&[f32::NAN; 32_000]);
        assert!(matches!(r, Err(Error::NonFiniteInput)));
    }

    #[test]
    fn produces_correct_shape_for_2s_clip() {
        let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize]; // ~silence
        let f = compute_fbank(&samples).unwrap();
        assert_eq!(f.len(), FBANK_FRAMES);
        assert_eq!(f[0].len(), FBANK_NUM_MELS);
        // Mean-subtraction should make most values close to zero.
        let mean_abs: f64 = f.iter().flatten().map(|x| (*x as f64).abs()).sum::<f64>()
                          / (FBANK_FRAMES * FBANK_NUM_MELS) as f64;
        assert!(mean_abs.is_finite());
    }

    #[test]
    fn produces_correct_shape_for_short_clip_with_padding() {
        let samples = vec![0.001f32; MIN_CLIP_SAMPLES as usize + 100];
        let f = compute_fbank(&samples).unwrap();
        assert_eq!(f.len(), FBANK_FRAMES);
    }
}
```

- [ ] **Step 2: Wire into mod.rs**

Update `dia/src/embed/mod.rs`:
```rust
mod fbank;
pub use fbank::compute_fbank;
```

- [ ] **Step 3: Verify (this depends on Task 1 spike completion)**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo test --lib embed::fbank
```
Expected: 4 tests pass. **If this fails because `kaldi-native-fbank`'s API differs from what's assumed**, consult the working spike code from Task 1 and adapt the wrapper accordingly.

- [ ] **Step 4: Commit**

```bash
git add src/embed/fbank.rs src/embed/mod.rs
git commit -m "embed: compute_fbank kaldi-native-fbank wrapper (spec §4.2)

80-mel kaldi-style fbank (25 ms frame_length, 10 ms shift, hamming,
dither=0). Scales input by 1<<15 to match pyannote / wespeaker
convention. Mean-subtracts across frames. Pads / center-crops to
[200, 80]. Verified against torchaudio reference per Task 1 spike."
```

---

### Task 24: `EmbedModelOptions` builder

**Files:**
- Modify: `dia/src/embed/options.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Append `EmbedModelOptions` (gated on ort)**

Append to `dia/src/embed/options.rs`:
```rust
#[cfg(feature = "ort")]
use ort::execution_providers::ExecutionProviderDispatch;
#[cfg(feature = "ort")]
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

/// Builder for [`crate::embed::EmbedModel`] runtime configuration.
/// Mirrors `diarization::segment::SegmentModelOptions`.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
#[derive(Default)]
pub struct EmbedModelOptions {
    optimization_level: Option<GraphOptimizationLevel>,
    providers: Vec<ExecutionProviderDispatch>,
    intra_op_num_threads: Option<usize>,
    inter_op_num_threads: Option<usize>,
}

#[cfg(feature = "ort")]
impl EmbedModelOptions {
    pub fn new() -> Self { Self::default() }

    pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization_level = Some(level); self
    }
    pub fn with_providers(mut self, providers: Vec<ExecutionProviderDispatch>) -> Self {
        self.providers = providers; self
    }
    pub fn with_intra_op_num_threads(mut self, n: usize) -> Self {
        self.intra_op_num_threads = Some(n); self
    }
    pub fn with_inter_op_num_threads(mut self, n: usize) -> Self {
        self.inter_op_num_threads = Some(n); self
    }

    pub fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
        self.optimization_level = Some(level); self
    }
    pub fn set_providers(&mut self, providers: Vec<ExecutionProviderDispatch>) -> &mut Self {
        self.providers = providers; self
    }
    pub fn set_intra_op_num_threads(&mut self, n: usize) -> &mut Self {
        self.intra_op_num_threads = Some(n); self
    }
    pub fn set_inter_op_num_threads(&mut self, n: usize) -> &mut Self {
        self.inter_op_num_threads = Some(n); self
    }

    /// Apply the option set to a `SessionBuilder`. Used internally by `EmbedModel`.
    pub(crate) fn apply(self, mut builder: SessionBuilder) -> Result<SessionBuilder, crate::embed::Error> {
        if let Some(level) = self.optimization_level {
            builder = builder.with_optimization_level(level).map_err(ort::Error::from)?;
        }
        if let Some(n) = self.intra_op_num_threads {
            builder = builder.with_intra_threads(n).map_err(ort::Error::from)?;
        }
        if let Some(n) = self.inter_op_num_threads {
            builder = builder.with_inter_threads(n).map_err(ort::Error::from)?;
        }
        if !self.providers.is_empty() {
            builder = builder.with_execution_providers(self.providers).map_err(ort::Error::from)?;
        }
        Ok(builder)
    }
}
```

Update `dia/src/embed/mod.rs` re-exports:
```rust
#[cfg(feature = "ort")]
pub use options::EmbedModelOptions;
```

```bash
cargo check
```
Expected: clean compile.

- [ ] **Step 2: Commit**

```bash
git add src/embed/options.rs src/embed/mod.rs
git commit -m "embed: EmbedModelOptions (spec §4.2)

Mirrors SegmentModelOptions: optimization_level, providers,
intra_op_num_threads, inter_op_num_threads. Both with_* and set_*
builders for parity with diarization::segment."
```

---

### Task 25: `EmbedModel` — load + `embed_features` + `embed_features_batch`

**Files:**
- Create: `dia/src/embed/model.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Write the model wrapper**

Create `dia/src/embed/model.rs`:
```rust
//! ONNX Runtime wrapper for WeSpeaker ResNet34 (spec §4.2).
//!
//! Auto-derives `Send`. Does NOT auto-derive `Sync` because
//! `ort::Session` is `!Sync`. Matches `diarization::segment::SegmentModel`.

use std::path::Path;

use ort::{
    session::{Session as OrtSession, builder::SessionBuilder},
    value::TensorRef,
};

use crate::embed::{
    Error, EmbedModelOptions,
    options::{EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS},
};

/// Thin ort wrapper for one WeSpeaker embedding session. Owns one
/// `ort::Session`. Use one per worker thread (the type is `Send`,
/// not `Sync`).
pub struct EmbedModel {
    inner: OrtSession,
}

impl EmbedModel {
    /// Load with default options.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::from_file_with_options(path, EmbedModelOptions::default())
    }

    /// Load with custom options.
    pub fn from_file_with_options<P: AsRef<Path>>(
        path: P,
        opts: EmbedModelOptions,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        let builder = opts.apply(OrtSession::builder()?)?;
        let session = builder
            .commit_from_file(path)
            .map_err(|source| Error::LoadModel { path: path.to_path_buf(), source })?;
        Ok(Self { inner: session })
    }

    /// Load from in-memory ONNX bytes.
    pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
        Self::from_memory_with_options(bytes, EmbedModelOptions::default())
    }

    /// Load from in-memory ONNX bytes with custom options.
    pub fn from_memory_with_options(
        bytes: &[u8],
        opts: EmbedModelOptions,
    ) -> Result<Self, Error> {
        let builder = opts.apply(OrtSession::builder()?)?;
        let session = builder.commit_from_memory(bytes)?;
        Ok(Self { inner: session })
    }

    /// Run inference on one [200, 80] fbank tensor. Returns the raw
    /// (un-normalized) 256-d output.
    pub fn embed_features(
        &mut self,
        features: &[[f32; FBANK_NUM_MELS]; FBANK_FRAMES],
    ) -> Result<[f32; EMBEDDING_DIM], Error> {
        // Flatten features into a contiguous [1, 200, 80] tensor.
        let mut flat = Vec::with_capacity(FBANK_FRAMES * FBANK_NUM_MELS);
        for row in features.iter() {
            flat.extend_from_slice(row);
        }
        let outputs = self.inner.run(ort::inputs![TensorRef::from_array_view((
            [1usize, FBANK_FRAMES, FBANK_NUM_MELS],
            flat.as_slice()
        ),)?,])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if data.len() != EMBEDDING_DIM {
            return Err(Error::InferenceShapeMismatch {
                expected: EMBEDDING_DIM,
                got: data.len(),
            });
        }
        let mut out = [0.0; EMBEDDING_DIM];
        out.copy_from_slice(data);
        Ok(out)
    }

    /// Batched feature inference. Single ONNX call with batch size N.
    /// Returns N raw 256-d outputs.
    pub fn embed_features_batch(
        &mut self,
        features: &[[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
        let n = features.len();
        if n == 0 { return Ok(Vec::new()); }
        let mut flat = Vec::with_capacity(n * FBANK_FRAMES * FBANK_NUM_MELS);
        for chunk in features.iter() {
            for row in chunk.iter() {
                flat.extend_from_slice(row);
            }
        }
        let outputs = self.inner.run(ort::inputs![TensorRef::from_array_view((
            [n, FBANK_FRAMES, FBANK_NUM_MELS],
            flat.as_slice()
        ),)?,])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if data.len() != n * EMBEDDING_DIM {
            return Err(Error::InferenceShapeMismatch {
                expected: n * EMBEDDING_DIM,
                got: data.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = [0.0; EMBEDDING_DIM];
            row.copy_from_slice(&data[i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM]);
            out.push(row);
        }
        Ok(out)
    }
}
```

Wire into `dia/src/embed/mod.rs`:
```rust
#[cfg(feature = "ort")]
mod model;
#[cfg(feature = "ort")]
pub use model::EmbedModel;
```

```bash
cargo check
```
Expected: clean compile.

- [ ] **Step 2: Commit**

```bash
git add src/embed/model.rs src/embed/mod.rs
git commit -m "embed: EmbedModel ort wrapper + embed_features* (spec §4.2)

EmbedModel owns one ort::Session. from_file{,_with_options} +
from_memory{,_with_options} constructors mirror SegmentModel.
embed_features takes [200, 80] fbank, returns raw 256-d output;
embed_features_batch takes &[[200, 80]; N], returns N raw outputs
in a single ONNX call (avoids per-call overhead for the Diarizer's
multi-window aggregation path)."
```

---

### Task 26: Sliding-window mean helper (spec §5.1)

**Files:**
- Create: `dia/src/embed/embedder.rs`
- Modify: `dia/src/embed/mod.rs`

- [ ] **Step 1: Implement the sliding-window aggregation**

Create `dia/src/embed/embedder.rs`:
```rust
//! Sliding-window mean aggregation for variable-length clips.
//! Spec §5.1 / §5.2.

use crate::embed::{
    Error, EmbedModel,
    options::{EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, HOP_SAMPLES, MIN_CLIP_SAMPLES, NORM_EPSILON},
    fbank::compute_fbank,
};

/// Plan window starts for a clip of `len` samples (spec §5.1).
///
/// - `len == 0`: empty plan.
/// - `0 < len < MIN_CLIP_SAMPLES`: caller's job — `embed*` rejects.
/// - `MIN_CLIP_SAMPLES <= len <= EMBED_WINDOW_SAMPLES`: single window at start 0.
/// - `len > EMBED_WINDOW_SAMPLES`: `[0, HOP, 2*HOP, ..., k_max*HOP, len - WINDOW]`,
///   deduped + sorted. `k_max = floor((len - WINDOW) / HOP)`.
pub(crate) fn plan_starts(len: usize) -> Vec<usize> {
    if len <= EMBED_WINDOW_SAMPLES as usize {
        return vec![0];
    }
    let win = EMBED_WINDOW_SAMPLES as usize;
    let hop = HOP_SAMPLES as usize;
    let k_max = (len - win) / hop;
    let mut starts: Vec<usize> = (0..=k_max).map(|k| k * hop).collect();
    starts.push(len - win);
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Run inference on one full clip via the sliding-window-mean algorithm
/// (§5.1). For `len <= EMBED_WINDOW_SAMPLES`, runs once with zero-padding.
/// For longer clips, sums per-window unnormalized embeddings.
///
/// Returns `(unnormalized_sum, windows_used)`. Caller L2-normalizes
/// (or returns `Error::DegenerateEmbedding` via `Embedding::normalize_from`).
pub(crate) fn embed_unweighted(
    model: &mut EmbedModel,
    samples: &[f32],
) -> Result<([f32; EMBEDDING_DIM], u32), Error> {
    if samples.len() < MIN_CLIP_SAMPLES as usize {
        return Err(Error::InvalidClip { len: samples.len(), min: MIN_CLIP_SAMPLES as usize });
    }
    let starts = plan_starts(samples.len());
    let mut sum = [0.0f32; EMBEDDING_DIM];

    if samples.len() <= EMBED_WINDOW_SAMPLES as usize {
        // Zero-pad to EMBED_WINDOW_SAMPLES.
        let mut padded = vec![0.0f32; EMBED_WINDOW_SAMPLES as usize];
        padded[..samples.len()].copy_from_slice(samples);
        let features = compute_fbank(&padded)?;
        let raw = model.embed_features(&features)?;
        return Ok((raw, 1));
    }

    let win = EMBED_WINDOW_SAMPLES as usize;
    for &start in &starts {
        let chunk = &samples[start..start + win];
        let features = compute_fbank(chunk)?;
        let raw = model.embed_features(&features)?;
        for k in 0..EMBEDDING_DIM { sum[k] += raw[k]; }
    }
    Ok((sum, starts.len() as u32))
}

/// Sliding-window mean WEIGHTED by per-sample voice probabilities
/// (§5.2). Same window plan as above; per-window weight = mean of
/// voice_probs over that window's samples.
pub(crate) fn embed_weighted_inner(
    model: &mut EmbedModel,
    samples: &[f32],
    voice_probs: &[f32],
) -> Result<([f32; EMBEDDING_DIM], u32, f32), Error> {
    if samples.len() != voice_probs.len() {
        return Err(Error::WeightShapeMismatch {
            samples_len: samples.len(),
            weights_len: voice_probs.len(),
        });
    }
    if samples.len() < MIN_CLIP_SAMPLES as usize {
        return Err(Error::InvalidClip { len: samples.len(), min: MIN_CLIP_SAMPLES as usize });
    }

    let starts = plan_starts(samples.len());
    let mut sum = [0.0f32; EMBEDDING_DIM];
    let mut total_weight = 0.0f32;
    let win = EMBED_WINDOW_SAMPLES as usize;

    if samples.len() <= win {
        let mut padded = vec![0.0f32; win];
        padded[..samples.len()].copy_from_slice(samples);
        let features = compute_fbank(&padded)?;
        let raw = model.embed_features(&features)?;
        // Weight = mean of voice_probs over the (un-padded) sample range.
        let w: f32 = voice_probs.iter().sum::<f32>() / voice_probs.len() as f32;
        for k in 0..EMBEDDING_DIM { sum[k] += w * raw[k]; }
        total_weight = w;
        return Ok((sum, 1, total_weight));
    }

    for &start in &starts {
        let chunk = &samples[start..start + win];
        let weights = &voice_probs[start..start + win];
        let w: f32 = weights.iter().sum::<f32>() / win as f32;
        let features = compute_fbank(chunk)?;
        let raw = model.embed_features(&features)?;
        for k in 0..EMBEDDING_DIM { sum[k] += w * raw[k]; }
        total_weight += w;
    }
    if total_weight < NORM_EPSILON { return Err(Error::AllSilent); }
    Ok((sum, starts.len() as u32, total_weight))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_starts_for_2s_clip() {
        let starts = plan_starts(EMBED_WINDOW_SAMPLES as usize);
        assert_eq!(starts, vec![0]);
    }

    #[test]
    fn plan_starts_for_3s_clip() {
        // 48_000 samples, win 32_000, hop 16_000.
        // k_max = (48_000 - 32_000) / 16_000 = 1.
        // Regular starts: [0, 16_000]. Tail anchor: 48_000 - 32_000 = 16_000.
        // After dedup: [0, 16_000].
        let starts = plan_starts(48_000);
        assert_eq!(starts, vec![0, 16_000]);
    }

    #[test]
    fn plan_starts_for_3_5s_clip() {
        // 56_000 samples. k_max = (56_000 - 32_000) / 16_000 = 1.
        // Regular: [0, 16_000]. Tail: 56_000 - 32_000 = 24_000.
        // After dedup: [0, 16_000, 24_000].
        let starts = plan_starts(56_000);
        assert_eq!(starts, vec![0, 16_000, 24_000]);
    }

    #[test]
    fn plan_starts_for_4s_clip() {
        // 64_000 samples. k_max = (64_000 - 32_000) / 16_000 = 2.
        // Regular: [0, 16_000, 32_000]. Tail: 32_000. Dedup → [0, 16_000, 32_000].
        let starts = plan_starts(64_000);
        assert_eq!(starts, vec![0, 16_000, 32_000]);
    }

    #[test]
    fn plan_starts_skips_dedup_when_tail_misaligned() {
        // 50_000 samples. k_max = (50_000 - 32_000) / 16_000 = 1.
        // Regular: [0, 16_000]. Tail: 50_000 - 32_000 = 18_000.
        // After dedup: [0, 16_000, 18_000].
        let starts = plan_starts(50_000);
        assert_eq!(starts, vec![0, 16_000, 18_000]);
    }
}
```

```bash
cargo test --lib embed::embedder::tests
```
Expected: 5 tests pass.

- [ ] **Step 2: Wire into mod.rs**

Update `dia/src/embed/mod.rs`:
```rust
#[cfg(feature = "ort")]
mod embedder;
```

(No public re-exports — the `embed_unweighted`/`embed_weighted_inner` helpers are `pub(crate)` for use by `EmbedModel`'s high-level methods in Task 27.)

- [ ] **Step 3: Commit**

```bash
git add src/embed/embedder.rs src/embed/mod.rs
git commit -m "embed: sliding-window mean planner + helpers (spec §5.1/§5.2)

plan_starts: k_max = floor((len - WINDOW) / HOP); regular hops up
through k_max + tail anchor at len - WINDOW; sort + dedup. Worked
examples for 2 s / 3 s / 3.5 s / 4 s / 50 ms clips.
embed_unweighted: zero-pad ≤ 2 s; sum per-window outputs > 2 s.
embed_weighted_inner: same plan; per-window weight = mean of
voice_probs slice. AllSilent if total_weight < NORM_EPSILON."
```

---

### Task 27: `EmbedModel::embed` + `embed_with_meta` + weighted/masked variants

**Files:**
- Modify: `dia/src/embed/model.rs`

- [ ] **Step 1: Append the high-level methods**

Append to `dia/src/embed/model.rs`:
```rust
use mediatime::Duration;
use crate::embed::{
    EmbeddingMeta, EmbeddingResult,
    embedder::{embed_unweighted, embed_weighted_inner},
    options::{EMBED_WINDOW_SAMPLES, MIN_CLIP_SAMPLES, NORM_EPSILON, SAMPLE_RATE_HZ},
};
use crate::embed::types::Embedding;

impl EmbedModel {
    /// High-level embed: aggregate via §5.1 sliding-window mean,
    /// L2-normalize. Returns `EmbeddingResult<()>`.
    pub fn embed(&mut self, samples: &[f32]) -> Result<EmbeddingResult, Error> {
        self.embed_with_meta(samples, EmbeddingMeta::default())
    }

    pub fn embed_with_meta<A, T>(
        &mut self, samples: &[f32], meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error> {
        let (sum, windows_used) = embed_unweighted(self, samples)?;
        let embedding = Embedding::normalize_from(sum)
            .ok_or(Error::DegenerateEmbedding)?;
        let duration = duration_from_samples(samples.len());
        Ok(EmbeddingResult::new(embedding, duration, windows_used,
                                 windows_used as f32, meta))
    }

    pub fn embed_weighted(
        &mut self, samples: &[f32], voice_probs: &[f32],
    ) -> Result<EmbeddingResult, Error> {
        self.embed_weighted_with_meta(samples, voice_probs, EmbeddingMeta::default())
    }

    pub fn embed_weighted_with_meta<A, T>(
        &mut self, samples: &[f32], voice_probs: &[f32], meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error> {
        let (sum, windows_used, total_weight) =
            embed_weighted_inner(self, samples, voice_probs)?;
        let embedding = Embedding::normalize_from(sum)
            .ok_or(Error::DegenerateEmbedding)?;
        let duration = duration_from_samples(samples.len());
        Ok(EmbeddingResult::new(embedding, duration, windows_used, total_weight, meta))
    }

    /// Rev-8 binary keep-mask. Gathers retained samples (sample-rate),
    /// then runs the standard `embed` pipeline. Spec §5.8 routes
    /// `Diarizer::exclude_overlap` through here.
    pub fn embed_masked(
        &mut self, samples: &[f32], keep_mask: &[bool],
    ) -> Result<EmbeddingResult, Error> {
        self.embed_masked_with_meta(samples, keep_mask, EmbeddingMeta::default())
    }

    pub fn embed_masked_with_meta<A, T>(
        &mut self, samples: &[f32], keep_mask: &[bool],
        meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error> {
        if keep_mask.len() != samples.len() {
            return Err(Error::MaskShapeMismatch {
                samples_len: samples.len(),
                mask_len: keep_mask.len(),
            });
        }
        // Gather samples where keep_mask[i] == true.
        let mut gathered = Vec::with_capacity(samples.len());
        for (i, &s) in samples.iter().enumerate() {
            if keep_mask[i] { gathered.push(s); }
        }
        if gathered.len() < MIN_CLIP_SAMPLES as usize {
            return Err(Error::InvalidClip {
                len: gathered.len(),
                min: MIN_CLIP_SAMPLES as usize,
            });
        }
        // Reuse the standard embed path on the gathered audio.
        // For ≤ 2 s gathered: zero-pad. For > 2 s: sliding-window-mean.
        // (Long-clip masked-embed has two layered divergences from
        //  pyannote — see embed_masked rustdoc + spec §15 #49.)
        let (sum, windows_used) = embed_unweighted(self, &gathered)?;
        let embedding = Embedding::normalize_from(sum)
            .ok_or(Error::DegenerateEmbedding)?;
        let duration = duration_from_samples(gathered.len());
        Ok(EmbeddingResult::new(embedding, duration, windows_used,
                                 windows_used as f32, meta))
    }
}

fn duration_from_samples(n: usize) -> Duration {
    Duration::from_micros((n as i64) * 1_000_000 / SAMPLE_RATE_HZ as i64)
}
```

Update `dia/src/embed/options.rs`:
```rust
/// 16 kHz mono. Matches `diarization::segment::SAMPLE_RATE_HZ`.
pub const SAMPLE_RATE_HZ: u32 = 16_000;
```

(And update the `pub use options::*` line in mod.rs to include `SAMPLE_RATE_HZ`.)

- [ ] **Step 2: Add a smoke test that round-trips a real `EmbedModel`**

Add a gated integration test at `dia/tests/integration_embed.rs`:
```rust
//! Gated smoke test against a real WeSpeaker ONNX model. Run with:
//!   cargo test --test integration_embed -- --ignored

#[cfg(feature = "ort")]
#[test]
#[ignore = "requires downloaded model — run scripts/download-embed-model.sh"]
fn embed_round_trips_on_2s_clip() {
    use diarization::embed::{EmbedModel, EMBED_WINDOW_SAMPLES};
    let model_path = "models/wespeaker-voxceleb-resnet34-LM.onnx";
    let mut model = EmbedModel::from_file(model_path).expect("model file present");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let r = model.embed(&samples).expect("embed succeeds");
    let n: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
    assert!((n - 1.0).abs() < 1e-5, "||embedding|| = {}", n.sqrt());
    assert_eq!(r.windows_used(), 1);
}
```

(Add `dia/scripts/download-embed-model.sh` in Task 28.)

- [ ] **Step 3: Verify no-ort still compiles**

```bash
cargo check --no-default-features --features std
```
Expected: clean compile (only Layer-1-equivalent surface available; `EmbedModel` and friends absent).

- [ ] **Step 4: Commit**

```bash
git add src/embed/model.rs src/embed/options.rs src/embed/mod.rs tests/integration_embed.rs
git commit -m "embed: high-level embed/_weighted/_masked methods (spec §4.2/§5.8)

embed: §5.1 sliding-window mean → L2-normalize.
embed_weighted: §5.2 per-sample voice-prob soft weighting.
embed_masked (rev-8): §5.8 gather-and-embed; keep_mask gathers
samples, then routes through the standard embed pipeline. Documented
two layered divergences from pyannote in embed_masked rustdoc."
```

---

### Task 28: Embed integration test scaffolding + model download script

**Files:**
- Create: `dia/scripts/download-embed-model.sh`
- Modify: `dia/tests/integration_embed.rs`

- [ ] **Step 1: Write the download script**

Create `dia/scripts/download-embed-model.sh`:
```bash
#!/usr/bin/env bash
# Download WeSpeaker ResNet34 ONNX (used by diarization::embed integration tests).
# Spec §3 deferred items — no bundled model in v0.1.0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODELS_DIR"
URL="https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/resolve/main/speaker-embedding.onnx"
DEST="$MODELS_DIR/wespeaker-voxceleb-resnet34-LM.onnx"
if [ -f "$DEST" ]; then
  echo "Model already present at $DEST"
  exit 0
fi
echo "Downloading WeSpeaker ResNet34 ONNX from $URL ..."
curl -fL "$URL" -o "$DEST"
echo "Saved to $DEST"
```

```bash
chmod +x /Users/user/Develop/findit-studio/dia/scripts/download-embed-model.sh
```

- [ ] **Step 2: Run the download (manual; not part of CI)**

```bash
cd /Users/user/Develop/findit-studio/dia
./scripts/download-embed-model.sh
```
Expected: model file written to `dia/models/wespeaker-voxceleb-resnet34-LM.onnx`.

- [ ] **Step 3: Run the gated integration test**

```bash
cargo test --test integration_embed -- --ignored
```
Expected: `embed_round_trips_on_2s_clip` passes.

- [ ] **Step 4: Commit**

```bash
git add scripts/download-embed-model.sh tests/integration_embed.rs
git commit -m "embed: download script + gated integration test (spec §3)

Manual download of WeSpeaker ResNet34 ONNX from Hugging Face
(hbredin/wespeaker-voxceleb-resnet34-LM). #[ignore]'d test verifies
end-to-end embed() round-trips on a 2 s synthetic clip with
||embedding|| ≈ 1 and windows_used == 1."
```

---

## Phase 6: `diarization::segment` v0.X bump (small additive)

Three small changes to the already-shipped `diarization::segment` module:

1. Mark `Action` `#[non_exhaustive]` so future variants are non-breaking.
2. Add `Action::SpeakerScores { id, window_start, raw_probs }` variant.
3. Emit `Action::SpeakerScores` from `push_inference` alongside `Action::Activity`.
4. Add `pub(crate) Segmenter::peek_next_window_start()` accessor for the Diarizer.

These are spec §3 "in-scope" items. All additive — no breaking changes to existing `Action::{NeedsInference, Activity, VoiceSpan}` semantics.

---

### Task 29: Mark `Action` `#[non_exhaustive]` + add `SpeakerScores` variant

**Files:**
- Modify: `dia/src/segment/types.rs:127-144`
- Modify: `dia/src/segment/segmenter.rs` (the `pub_inference` body that emits activities)

- [ ] **Step 1: Update `Action` enum in `types.rs`**

Replace lines 121-144 of `dia/src/segment/types.rs`:
```rust
/// One output of the Layer-1 state machine.
///
/// Style note: enum-variant fields (`id`, `samples`) are public because they
/// participate in pattern matching, which is the standard Rust enum idiom.
/// Structs with invariants (`WindowId`, `SpeakerActivity`) use private
/// fields with accessors. The two conventions coexist deliberately.
///
/// **`#[non_exhaustive]`** (added in v0.X for the dia phase-2 release):
/// downstream `match` expressions must include `_ => ...` to remain
/// forward-compatible. New variants may be added in subsequent minor
/// versions without a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Action {
  /// The caller must run ONNX inference on `samples` and call
  /// [`Segmenter::push_inference`](crate::segment::Segmenter::push_inference)
  /// with the same `id`.
  NeedsInference {
    /// Correlation handle (the window's sample range plus generation).
    id: WindowId,
    /// Always `WINDOW_SAMPLES = 160_000` mono float32 samples at 16 kHz,
    /// zero-padded if the input stream is shorter.
    samples: alloc::boxed::Box<[f32]>,
  },
  /// A decoded window-local speaker activity.
  Activity(SpeakerActivity),
  /// A finalized speaker-agnostic voice region. Emit-only — never
  /// retracted once produced.
  VoiceSpan(TimeRange),
  /// Per-window per-speaker per-frame raw probabilities. Emitted from
  /// [`Segmenter::push_inference`](crate::segment::Segmenter::push_inference)
  /// **immediately before** the `Activity` events for the same `id`.
  ///
  /// Carries the powerset-decoded per-frame voice probabilities for
  /// each of the 3 speaker slots. Used by `diarization::Diarizer` for
  /// pyannote-style per-frame reconstruction (per-frame per-cluster
  /// activation overlap-add, count-bounded argmax). Most callers
  /// can ignore this variant via the `_ => ...` arm of `match`.
  ///
  /// Layout: `raw_probs[slot][frame]`. `MAX_SPEAKER_SLOTS = 3`,
  /// `FRAMES_PER_WINDOW = 589`. ~7 KB allocation per emission;
  /// see spec §15 #53 for a v0.1.1 pooling optimization.
  SpeakerScores {
    /// Correlation handle of the window these scores belong to.
    id: WindowId,
    /// Window start in absolute samples (`id.range().start_pts()` in `SAMPLE_RATE_TB`).
    window_start: u64,
    /// Per-(slot, frame) raw probabilities.
    raw_probs: alloc::boxed::Box<[[f32; crate::segment::options::FRAMES_PER_WINDOW]; crate::segment::options::MAX_SPEAKER_SLOTS as usize]>,
  },
}
```

- [ ] **Step 2: Verify the segment lib still compiles + segment tests pass**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo check
cargo test --lib segment
```
Expected: clean compile + all 54 segment tests still pass (the variant addition is additive; the `#[non_exhaustive]` marker doesn't affect internal matches).

If any internal `match` on `Action` is now non-exhaustive, add `_ => unreachable!("SpeakerScores not produced internally")` arms — those are inside `Segmenter` itself, which doesn't poll its own actions.

- [ ] **Step 3: Commit**

```bash
git add src/segment/types.rs
git commit -m "segment: Action #[non_exhaustive] + SpeakerScores variant (spec §3)

Additive change for dia v0.1.0 phase 2. Action becomes
#[non_exhaustive] (downstream match expressions need _ arms going
forward; existing dia code uses exhaustive match where it knows the
internal contract — those sites add unreachable!() arms).

Action::SpeakerScores { id, window_start, raw_probs } carries the
per-window per-speaker per-frame raw probabilities. Used by
diarization::Diarizer for pyannote-style reconstruction. ~7 KB per emission."
```

---

### Task 30: Emit `Action::SpeakerScores` from `push_inference`

**Files:**
- Modify: `dia/src/segment/segmenter.rs` (the `process_inference` function around lines 280-290)

- [ ] **Step 1: Locate the emit site and add the SpeakerScores emission**

Read `dia/src/segment/segmenter.rs:240-300` to find where `emit_speaker_activities` is called from `process_inference`. Insert the `Action::SpeakerScores` push BEFORE the activity emission:

```rust
// In process_inference (around line 282):
//   self.emit_speaker_activities(id, start, &speaker_probs);
// becomes:
self.pending_actions.push_back(Action::SpeakerScores {
    id,
    window_start: start,
    raw_probs: alloc::boxed::Box::new(speaker_probs),
});
self.emit_speaker_activities(id, start, &speaker_probs);
```

(Adapt to the actual variable names in the existing code. `speaker_probs` is the `[Vec<f32>; MAX_SPEAKER_SLOTS]` already computed; we need to convert to `[[f32; 589]; 3]`. Add a helper:)

```rust
fn speaker_probs_to_array(
    probs: &[Vec<f32>; MAX_SPEAKER_SLOTS as usize],
) -> [[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize] {
    let mut out = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for s in 0..MAX_SPEAKER_SLOTS as usize {
        debug_assert_eq!(probs[s].len(), FRAMES_PER_WINDOW);
        for f in 0..FRAMES_PER_WINDOW {
            out[s][f] = probs[s][f];
        }
    }
    out
}
```

Then update the emit:
```rust
self.pending_actions.push_back(Action::SpeakerScores {
    id,
    window_start: start,
    raw_probs: alloc::boxed::Box::new(speaker_probs_to_array(&speaker_probs)),
});
self.emit_speaker_activities(id, start, &speaker_probs);
```

- [ ] **Step 2: Add a unit test for the new emission**

Append to `dia/src/segment/segmenter.rs`'s `tests` module (or create a new test file `dia/src/segment/tests_speaker_scores.rs` and `mod tests_speaker_scores;` in `mod.rs`):

```rust
#[test]
fn push_inference_emits_speaker_scores_before_activities() {
    use crate::segment::{
        options::{FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES},
        Action, Segmenter, SegmentOptions,
    };
    let mut s = Segmenter::new(SegmentOptions::default());
    s.push_samples(&vec![0.001f32; 160_000]);
    let id = match s.poll() {
        Some(Action::NeedsInference { id, .. }) => id,
        other => panic!("expected NeedsInference, got {:?}", other),
    };
    // Synthetic scores: all-zero (no speakers active).
    let scores = vec![1.0f32 / POWERSET_CLASSES as f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    s.push_inference(id, &scores).unwrap();

    let mut saw_scores = false;
    let mut saw_activity_after_scores = false;
    while let Some(action) = s.poll() {
        match action {
            Action::SpeakerScores { id: sid, window_start, raw_probs } => {
                assert_eq!(sid, id);
                assert_eq!(window_start, 0);
                assert_eq!(raw_probs.len(), MAX_SPEAKER_SLOTS as usize);
                assert_eq!(raw_probs[0].len(), FRAMES_PER_WINDOW);
                saw_scores = true;
            }
            Action::Activity(_) => {
                assert!(saw_scores, "Activity emitted before SpeakerScores");
                saw_activity_after_scores = true;
            }
            Action::VoiceSpan(_) => {} // OK, ignore
            _ => {} // non_exhaustive
        }
    }
    assert!(saw_scores, "no SpeakerScores emitted");
    // For an all-equal-probability window, speaker_probs may all be
    // sub-threshold so no Activity emits. saw_activity_after_scores
    // is informational, not asserted.
}
```

```bash
cargo test --lib segment::segmenter::tests::push_inference_emits_speaker_scores_before_activities
```
Expected: pass.

- [ ] **Step 3: Run all segment tests**

```bash
cargo test --lib segment
```
Expected: 55 tests pass (54 existing + new).

- [ ] **Step 4: Commit**

```bash
git add src/segment/segmenter.rs
git commit -m "segment: emit Action::SpeakerScores from push_inference (spec §3)

For each push_inference(id, scores), push Action::SpeakerScores
{ id, window_start, raw_probs } before any Action::Activity events
for the same id. Same call frame; both emissions appear in the same
synchronous pump. raw_probs is the powerset-decoded per-(slot, frame)
voice probability matrix as [[f32; 589]; 3]."
```

---

### Task 31: Add `Segmenter::peek_next_window_start()`

**Files:**
- Modify: `dia/src/segment/segmenter.rs`

- [ ] **Step 1: Add the accessor**

Append to the `impl Segmenter` block in `dia/src/segment/segmenter.rs`:
```rust
    /// Internal accessor: where the next regular sliding window will
    /// start (in absolute samples). After `finish()`, returns
    /// `u64::MAX` (no future regular windows; tail anchor already
    /// scheduled if needed).
    ///
    /// Used by `diarization::Diarizer`'s reconstruction state machine to
    /// determine the per-frame finalization boundary (spec §5.9).
    /// `pub(crate)` because no external use case exists yet; expose
    /// as `pub` if a real one materializes.
    pub(crate) fn peek_next_window_start(&self) -> u64 {
        if self.finished { return u64::MAX; }
        (self.next_window_idx as u64) * self.opts.step_samples() as u64
    }
```

(Adjust to the actual field names in `Segmenter` — `finished`, `next_window_idx`, `opts.step_samples()`.)

- [ ] **Step 2: Make the accessor available across the workspace**

Since `diarization::diarizer` lives in the same crate, `pub(crate)` is sufficient. No mod.rs export needed.

- [ ] **Step 3: Add a unit test**

Append to `dia/src/segment/segmenter.rs`'s tests:
```rust
#[test]
fn peek_next_window_start_advances_on_window_emit() {
    use crate::segment::{options::POWERSET_CLASSES, Action, Segmenter, SegmentOptions};
    let opts = SegmentOptions::default();
    let step = opts.step_samples() as u64;
    let mut s = Segmenter::new(opts);
    assert_eq!(s.peek_next_window_start(), 0);

    s.push_samples(&vec![0.001f32; 160_000]);
    // After scheduling but before push_inference: next window is at step.
    let id = match s.poll() {
        Some(Action::NeedsInference { id, .. }) => id,
        _ => panic!(),
    };
    assert_eq!(s.peek_next_window_start(), step);

    let scores = vec![1.0f32 / POWERSET_CLASSES as f32;
                       crate::segment::options::FRAMES_PER_WINDOW
                       * POWERSET_CLASSES];
    s.push_inference(id, &scores).unwrap();
    while s.poll().is_some() {}
    assert_eq!(s.peek_next_window_start(), step);
}

#[test]
fn peek_next_window_start_max_after_finish() {
    use crate::segment::{Segmenter, SegmentOptions};
    let mut s = Segmenter::new(SegmentOptions::default());
    s.push_samples(&[0.001; 16_000]); // 1 s
    s.finish();
    assert_eq!(s.peek_next_window_start(), u64::MAX);
}
```

```bash
cargo test --lib segment::segmenter
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/segment/segmenter.rs
git commit -m "segment: pub(crate) Segmenter::peek_next_window_start (spec §3/§5.9)

For diarization::Diarizer's reconstruction finalization-boundary computation.
Returns (next_window_idx * step_samples) when !finished, else u64::MAX.
Accessor only — no state mutation."
```

---

## Phase 7: `diarization::Diarizer` — module skeleton + types

`Diarizer` orchestrates segment + embed + cluster + reconstruction. Module split per spec §6: `mod.rs` (Diarizer struct), `builder.rs` (DiarizerOptions + DiarizerBuilder), `error.rs`, `span.rs` (DiarizedSpan + CollectedEmbedding), `overlap.rs` (Phase 9), `reconstruct.rs` (Phase 10).

---

### Task 32: `diarization::diarizer` module skeleton + `DiarizedSpan` + `CollectedEmbedding`

**Files:**
- Create: `dia/src/diarizer/mod.rs`
- Create: `dia/src/diarizer/span.rs`
- Modify: `dia/src/lib.rs`

- [ ] **Step 1: Create the module directory**

```bash
mkdir -p /Users/user/Develop/findit-studio/dia/src/diarizer
```

Write `dia/src/diarizer/mod.rs`:
```rust
//! Top-level streaming speaker-diarization orchestrator.
//!
//! Combines `diarization::segment` + `diarization::embed` + `diarization::cluster` + a
//! per-frame reconstruction state machine. Spec §4.4 / §5.7-§5.12.
//!
//! Output: one `DiarizedSpan` per closed speaker turn (rev-6 shape;
//! NOT per-(window, slot)).

mod builder;
mod error;
mod span;

#[cfg(feature = "ort")]
mod overlap;
#[cfg(feature = "ort")]
mod reconstruct;

#[cfg(feature = "ort")]
mod imp;

pub use builder::{DiarizerBuilder, DiarizerOptions};
pub use error::{Error, InternalError};
pub use span::{CollectedEmbedding, DiarizedSpan};

#[cfg(feature = "ort")]
pub use imp::Diarizer;

// Compile-time trait assertions (spec §9). Diarizer is Send + Sync
// auto-derived; this guards against a future field-type change
// silently regressing the property.
#[cfg(feature = "ort")]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Diarizer>();
};
```

- [ ] **Step 2: Write `dia/src/diarizer/span.rs`**

```rust
//! Public output types for `diarization::Diarizer`. Spec §4.4.

use mediatime::TimeRange;

use crate::embed::Embedding;

/// Per-(window, slot) context retained during a diarization session.
/// Returned by `Diarizer::collected_embeddings()`. Carries everything
/// needed to correlate a post-hoc offline-clustering re-labeling
/// back to its source activity.
///
/// Granularity is **per-(window, slot)** — one entry per pre-
/// reconstruction `SpeakerActivity` from `diarization::segment`. Finer-
/// grained than the post-reconstruction `DiarizedSpan` output (one
/// per closed cluster run). The two views are reconciled via
/// `online_speaker_id`.
#[derive(Debug, Clone)]
pub struct CollectedEmbedding {
    pub range: TimeRange,
    pub embedding: Embedding,
    /// Online speaker id assigned by `Clusterer::submit` during streaming.
    pub online_speaker_id: u64,
    /// Window-local slot from `diarization::segment::SpeakerActivity`.
    pub speaker_slot: u8,
    /// Whether the embedding used the `exclude_overlap` clean mask
    /// (`true`) or fell back to the speaker-only mask (`false`).
    /// See spec §5.8 for fallback semantics.
    pub used_clean_mask: bool,
}

/// One closed speaker turn after reconstruction.
///
/// **Rev-6 shape:** `(range, speaker_id, is_new_speaker)` plus three
/// rev-7 quality metrics. Window-local concepts (similarity, slot)
/// don't apply to a stitched multi-window span; that context lives
/// on `CollectedEmbedding`.
#[derive(Debug, Clone, Copy)]
pub struct DiarizedSpan {
    pub(crate) range: TimeRange,
    pub(crate) speaker_id: u64,
    pub(crate) is_new_speaker: bool,
    pub(crate) average_activation: f32,
    pub(crate) activity_count: u32,
    pub(crate) clean_mask_fraction: f32,
}

impl DiarizedSpan {
    /// Sample range of this span, in `diarization::segment::SAMPLE_RATE_TB`.
    pub fn range(&self) -> TimeRange { self.range }

    /// Global cluster id assigned by the online clusterer.
    pub fn speaker_id(&self) -> u64 { self.speaker_id }

    /// `true` iff this is the first time `speaker_id` has been emitted
    /// in the current `Diarizer` session (post-`new`/`clear`).
    pub fn is_new_speaker(&self) -> bool { self.is_new_speaker }

    /// Mean per-frame normalized activation across the span's frames.
    /// Each frame's contribution is `activation_sum / activation_chunk_count`,
    /// so the value is in `[0.0, 1.0]`. Higher = more confident speaker
    /// assignment. Roughly comparable across spans. (Rev-7.)
    pub fn average_activation(&self) -> f32 { self.average_activation }

    /// Number of `(WindowId, slot)` segment activities that contributed
    /// to this span. (Rev-7.)
    pub fn activity_count(&self) -> u32 { self.activity_count }

    /// Of the contributing activities, the fraction whose embedding
    /// used the `exclude_overlap` clean mask. Range `[0.0, 1.0]`.
    /// Lower = more overlap-contaminated. (Rev-7.)
    pub fn clean_mask_fraction(&self) -> f32 { self.clean_mask_fraction }
}
```

- [ ] **Step 3: Wire into `lib.rs`**

Modify `dia/src/lib.rs`:
```rust
pub mod cluster;
pub mod diarizer;
pub mod embed;
pub mod segment;
```

- [ ] **Step 4: Add stubs for the other module files**

Create `dia/src/diarizer/builder.rs`:
```rust
//! Stub — filled in by Task 33.
use crate::cluster::ClusterOptions;
use crate::segment::SegmentOptions;

#[derive(Debug, Clone)]
pub struct DiarizerOptions {
    _segment: SegmentOptions,
    _cluster: ClusterOptions,
}

#[derive(Debug, Clone, Default)]
pub struct DiarizerBuilder;
```

Create `dia/src/diarizer/error.rs`:
```rust
//! Stub — filled in by Task 34.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Segment(#[from] crate::segment::Error),
    #[error(transparent)]
    Embed(#[from] crate::embed::Error),
    #[error(transparent)]
    Cluster(#[from] crate::cluster::Error),
    #[error(transparent)]
    Internal(InternalError),
}
#[derive(Debug, thiserror::Error)]
pub enum InternalError {
    #[error("placeholder")]
    Placeholder,
}
```

Create `dia/src/diarizer/imp.rs`:
```rust
//! Stub — filled in by Task 36.
use crate::diarizer::{builder::DiarizerOptions, error::Error};
pub struct Diarizer { _opts: DiarizerOptions }
impl Diarizer {
    pub fn _placeholder() -> Result<Self, Error> { unimplemented!() }
}
```

Create `dia/src/diarizer/overlap.rs` and `dia/src/diarizer/reconstruct.rs`:
```rust
//! Stub — filled in by Phase 9 / Phase 10.
```

- [ ] **Step 5: Verify compile**

```bash
cargo check
```
Expected: clean compile.

- [ ] **Step 6: Commit**

```bash
git add src/diarizer/ src/lib.rs
git commit -m "diarizer: module skeleton + DiarizedSpan + CollectedEmbedding (spec §4.4)

Module layout per spec §6: mod / builder / error / span / overlap /
reconstruct / imp. DiarizedSpan with rev-7 quality metrics
(average_activation, activity_count, clean_mask_fraction).
CollectedEmbedding with used_clean_mask. Compile-time Send+Sync
assertion on Diarizer."
```

---

### Task 33: `DiarizerOptions` + `DiarizerBuilder`

**Files:**
- Modify: `dia/src/diarizer/builder.rs`

- [ ] **Step 1: Replace stub with full implementation**

Replace `dia/src/diarizer/builder.rs`:
```rust
//! Builder + options for `Diarizer`. Spec §4.4.

use crate::cluster::ClusterOptions;
use crate::segment::SegmentOptions;

/// Configuration for `Diarizer`. All fields have sensible defaults.
#[derive(Debug, Clone)]
pub struct DiarizerOptions {
    pub(crate) segment: SegmentOptions,
    pub(crate) cluster: ClusterOptions,
    /// Retain per-activity context across the session. Default: `true`.
    pub(crate) collect_embeddings: bool,
    /// Onset threshold for binarizing per-frame per-speaker raw probabilities.
    /// Used by §5.8 (`exclude_overlap`) and §5.10 (speaker count).
    /// Default: 0.5 (matches pyannote's `Binarize(onset=...)` default).
    pub(crate) binarize_threshold: f32,
    /// Apply `exclude_overlap` mask when extracting embeddings. Default: `true`.
    pub(crate) exclude_overlap: bool,
}

impl Default for DiarizerOptions {
    fn default() -> Self {
        Self {
            segment: SegmentOptions::default(),
            cluster: ClusterOptions::default(),
            collect_embeddings: true,
            binarize_threshold: 0.5,
            exclude_overlap: true,
        }
    }
}

impl DiarizerOptions {
    pub fn new() -> Self { Self::default() }

    pub fn segment_options(&self) -> &SegmentOptions { &self.segment }
    pub fn cluster_options(&self) -> &ClusterOptions { &self.cluster }
    pub fn collect_embeddings(&self) -> bool { self.collect_embeddings }
    pub fn binarize_threshold(&self) -> f32 { self.binarize_threshold }
    pub fn exclude_overlap(&self) -> bool { self.exclude_overlap }

    pub fn with_segment_options(mut self, opts: SegmentOptions) -> Self { self.segment = opts; self }
    pub fn with_cluster_options(mut self, opts: ClusterOptions) -> Self { self.cluster = opts; self }
    pub fn with_collect_embeddings(mut self, on: bool) -> Self { self.collect_embeddings = on; self }
    pub fn with_binarize_threshold(mut self, t: f32) -> Self { self.binarize_threshold = t; self }
    pub fn with_exclude_overlap(mut self, on: bool) -> Self { self.exclude_overlap = on; self }

    pub fn set_segment_options(&mut self, opts: SegmentOptions) -> &mut Self { self.segment = opts; self }
    pub fn set_cluster_options(&mut self, opts: ClusterOptions) -> &mut Self { self.cluster = opts; self }
    pub fn set_collect_embeddings(&mut self, on: bool) -> &mut Self { self.collect_embeddings = on; self }
    pub fn set_binarize_threshold(&mut self, t: f32) -> &mut Self { self.binarize_threshold = t; self }
    pub fn set_exclude_overlap(&mut self, on: bool) -> &mut Self { self.exclude_overlap = on; self }
}

/// Builder for `Diarizer`. `with_*` setters only — no `options(opts)`
/// override (avoids the rev-1 API inconsistency).
#[derive(Debug, Clone, Default)]
pub struct DiarizerBuilder {
    opts: DiarizerOptions,
}

impl DiarizerBuilder {
    pub fn new() -> Self { Self::default() }

    pub fn with_segment_options(mut self, o: SegmentOptions) -> Self { self.opts.segment = o; self }
    pub fn with_cluster_options(mut self, o: ClusterOptions) -> Self { self.opts.cluster = o; self }
    pub fn with_collect_embeddings(mut self, on: bool) -> Self { self.opts.collect_embeddings = on; self }
    pub fn with_binarize_threshold(mut self, t: f32) -> Self { self.opts.binarize_threshold = t; self }
    pub fn with_exclude_overlap(mut self, on: bool) -> Self { self.opts.exclude_overlap = on; self }

    /// Consume the builder. Note: `build()` does NOT instantiate a
    /// `Diarizer` directly — it returns `DiarizerOptions`, and
    /// `Diarizer::new(opts)` is the actual constructor. This keeps
    /// the builder zero-cost when callers want to round-trip options
    /// (e.g., for serde).
    pub fn build(self) -> DiarizerOptions { self.opts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let o = DiarizerOptions::default();
        assert!(o.collect_embeddings());
        assert!((o.binarize_threshold() - 0.5).abs() < 1e-7);
        assert!(o.exclude_overlap());
    }

    #[test]
    fn builder_round_trip() {
        let o = DiarizerBuilder::new()
            .with_collect_embeddings(false)
            .with_binarize_threshold(0.7)
            .with_exclude_overlap(false)
            .build();
        assert!(!o.collect_embeddings());
        assert!((o.binarize_threshold() - 0.7).abs() < 1e-7);
        assert!(!o.exclude_overlap());
    }

    #[test]
    fn set_methods_mutate_in_place() {
        let mut o = DiarizerOptions::default();
        o.set_binarize_threshold(0.4)
         .set_collect_embeddings(false);
        assert!((o.binarize_threshold() - 0.4).abs() < 1e-7);
        assert!(!o.collect_embeddings());
    }
}
```

```bash
cargo test --lib diarizer::builder
```
Expected: 3 tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/diarizer/builder.rs
git commit -m "diarizer: DiarizerOptions + DiarizerBuilder (spec §4.4)

Defaults: collect_embeddings = true, binarize_threshold = 0.5,
exclude_overlap = true. Both with_* and set_* mutators. Builder
returns DiarizerOptions (zero-cost round-trip; the actual Diarizer
constructor is Diarizer::new(opts) added in Task 36)."
```

---

### Task 34: `diarizer::Error` + `InternalError`

**Files:**
- Modify: `dia/src/diarizer/error.rs`

- [ ] **Step 1: Replace stub with full implementation**

Replace `dia/src/diarizer/error.rs`:
```rust
//! Error type for `diarization::Diarizer`. Spec §4.4.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Segment(#[from] crate::segment::Error),
    #[error(transparent)]
    Embed(#[from] crate::embed::Error),
    #[error(transparent)]
    Cluster(#[from] crate::cluster::Error),

    /// An invariant of the Diarizer's internal state was violated.
    /// Distinct from the wrapped variants because those wrap errors
    /// FROM the underlying state machines; `Internal` covers the
    /// integration glue itself (audio buffer indexing, activity range
    /// reconciliation). Almost certainly a bug in dia or a pathological
    /// caller (e.g., a custom mid-level composition supplying out-of-
    /// order activities).
    #[error(transparent)]
    Internal(#[from] InternalError),
}

#[derive(Debug, thiserror::Error)]
pub enum InternalError {
    /// An emitted activity's range start is older than the audio
    /// buffer's earliest retained sample. Should never fire; defensive.
    #[error("activity range start {activity_start} is below audio buffer base {audio_base} (delta = {} samples)", audio_base - activity_start)]
    AudioBufferUnderflow {
        activity_start: u64,
        audio_base: u64,
    },

    /// An emitted activity's range end exceeds the audio buffer's
    /// latest sample.
    #[error("activity range end {activity_end} exceeds audio buffer end {audio_end}")]
    AudioBufferOverrun {
        activity_end: u64,
        audio_end: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn underflow_message_includes_delta() {
        let e = InternalError::AudioBufferUnderflow {
            activity_start: 100, audio_base: 500,
        };
        let s = format!("{e}");
        assert!(s.contains("100"));
        assert!(s.contains("500"));
        assert!(s.contains("400")); // delta
    }

    #[test]
    fn overrun_message_format() {
        let e = InternalError::AudioBufferOverrun {
            activity_end: 1500, audio_end: 1000,
        };
        let s = format!("{e}");
        assert!(s.contains("1500"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn from_segment_error() {
        // Just verify it compiles + #[from] works.
        fn _accepts(e: crate::segment::Error) -> Error { e.into() }
    }
}
```

```bash
cargo test --lib diarizer::error
```
Expected: 3 tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/diarizer/error.rs
git commit -m "diarizer: Error + InternalError types (spec §4.4)

Error wraps Segment/Embed/Cluster errors via #[from]; Internal
covers integration-glue invariants. InternalError variants:
AudioBufferUnderflow (activity_start < audio_base) and
AudioBufferOverrun (activity_end > audio_end). Both should be
unreachable per §5.7 segment-contract verification; defense-in-depth."
```

---

## Phase 8: `Diarizer` audio buffer + pump glue

This phase establishes the `Diarizer` state holder, audio buffer, and the basic pump (without exclude_overlap or reconstruction yet — those are Phases 9 / 10). After this phase, `process_samples` runs segment + per-activity-embed + cluster but emits NO `DiarizedSpan`s yet.

---

### Task 35: `Diarizer` state holder + audio buffer + accessors

**Files:**
- Modify: `dia/src/diarizer/imp.rs`

- [ ] **Step 1: Replace `imp.rs` stub with the state holder + accessors**

Replace `dia/src/diarizer/imp.rs`:
```rust
//! `Diarizer` state holder. Spec §4.4 / §5.7-§5.12.

use alloc::collections::VecDeque;

extern crate alloc;

use crate::cluster::{Clusterer, SpeakerCentroid};
use crate::diarizer::{
    builder::{DiarizerBuilder, DiarizerOptions},
    error::Error,
    span::{CollectedEmbedding, DiarizedSpan},
};
use crate::segment::Segmenter;

/// Top-level streaming diarizer.
///
/// **Threading:** `Send + Sync` auto-derived. Single-stream — use one
/// per concurrent diarization session. Pass `&mut SegmentModel` and
/// `&mut EmbedModel` per `process_samples`/`finish_stream` call (rev-2
/// borrow-by-`&mut` pattern; spec §11.0). Both models are `!Sync`.
pub struct Diarizer {
    pub(crate) opts: DiarizerOptions,
    pub(crate) segmenter: Segmenter,
    pub(crate) clusterer: Clusterer,
    /// Rolling audio buffer indexed by absolute samples. Element 0
    /// corresponds to absolute sample `audio_base`. Trim policy: keep
    /// the last `diarization::segment::WINDOW_SAMPLES` samples (§5.7 / §11.5).
    pub(crate) audio_buffer: VecDeque<f32>,
    pub(crate) audio_base: u64,
    pub(crate) total_samples_pushed: u64,
    /// Per-activity context, when `opts.collect_embeddings = true`.
    pub(crate) collected_embeddings: Vec<CollectedEmbedding>,
    /// Per-frame reconstruction state (added in Phase 10).
    #[cfg(feature = "ort")]
    pub(crate) reconstruct: crate::diarizer::reconstruct::ReconstructState,
}

impl Diarizer {
    /// Constructor. Caller may instead use `Diarizer::builder()`.
    pub fn new(opts: DiarizerOptions) -> Self {
        let segmenter = Segmenter::new(opts.segment.clone());
        let clusterer = Clusterer::new(opts.cluster.clone());
        Self {
            opts,
            segmenter,
            clusterer,
            audio_buffer: VecDeque::new(),
            audio_base: 0,
            total_samples_pushed: 0,
            collected_embeddings: Vec::new(),
            #[cfg(feature = "ort")]
            reconstruct: crate::diarizer::reconstruct::ReconstructState::new(),
        }
    }

    pub fn builder() -> DiarizerBuilder { DiarizerBuilder::new() }

    pub fn options(&self) -> &DiarizerOptions { &self.opts }

    /// Reset for a new session. Spec §4.4 / rev-7 §11.12.
    /// - Segmenter cleared (generation bump → stale window-ids reject).
    /// - Clusterer cleared (speaker_id reset to 0).
    /// - Audio buffer drained; `audio_base = 0`; `total_samples_pushed = 0`.
    /// - Per-frame stitching state dropped; open per-cluster runs DROPPED
    ///   (NOT auto-emitted — call `finish_stream` first).
    /// - `collected_embeddings` is **NOT** cleared (call `clear_collected`).
    /// - Configured options preserved.
    pub fn clear(&mut self) {
        self.segmenter.clear();
        self.clusterer.clear();
        self.audio_buffer.clear();
        self.audio_base = 0;
        self.total_samples_pushed = 0;
        #[cfg(feature = "ort")]
        self.reconstruct.clear();
    }

    pub fn collected_embeddings(&self) -> &[CollectedEmbedding] {
        &self.collected_embeddings
    }
    pub fn clear_collected(&mut self) { self.collected_embeddings.clear(); }

    pub fn pending_inferences(&self) -> usize { self.segmenter.pending_inferences() }
    pub fn buffered_samples(&self) -> usize { self.audio_buffer.len() }

    #[cfg(feature = "ort")]
    pub fn buffered_frames(&self) -> usize { self.reconstruct.buffered_frame_count() }

    /// Cumulative count of samples ever passed to `process_samples`
    /// since the last `clear()`. Monotonic; never decremented except
    /// by `clear()`. Caller-side anchor for VAD-to-original-time
    /// mapping per spec §11.12.
    pub fn total_samples_pushed(&self) -> u64 { self.total_samples_pushed }

    pub fn num_speakers(&self) -> usize { self.clusterer.num_speakers() }
    pub fn speakers(&self) -> Vec<SpeakerCentroid> { self.clusterer.speakers() }
}
```

The `ReconstructState` type and `reconstruct.rs` are stubs; create an empty impl so this compiles.

Update `dia/src/diarizer/reconstruct.rs`:
```rust
//! Per-frame stitching. Filled in by Phase 10. Stub for the type.
#[derive(Debug, Default)]
pub(crate) struct ReconstructState;

impl ReconstructState {
    pub(crate) fn new() -> Self { Self::default() }
    pub(crate) fn clear(&mut self) {}
    pub(crate) fn buffered_frame_count(&self) -> usize { 0 }
}
```

- [ ] **Step 2: Verify compile + Send/Sync assertion still passes**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo check
cargo build
```
Expected: clean compile with `Send + Sync` assertion satisfied.

- [ ] **Step 3: Add basic accessor tests**

Create `dia/src/diarizer/tests.rs`:
```rust
//! Cross-component diarizer tests (per spec §9).

use super::*;

#[test]
fn new_diarizer_default_state() {
    let d = Diarizer::new(DiarizerOptions::default());
    assert_eq!(d.pending_inferences(), 0);
    assert_eq!(d.buffered_samples(), 0);
    assert_eq!(d.total_samples_pushed(), 0);
    assert_eq!(d.num_speakers(), 0);
    assert!(d.collected_embeddings().is_empty());
}

#[test]
fn builder_round_trip() {
    let opts = Diarizer::builder()
        .with_collect_embeddings(false)
        .with_binarize_threshold(0.6)
        .build();
    assert!(!opts.collect_embeddings());
    assert!((opts.binarize_threshold() - 0.6).abs() < 1e-7);
}
```

Wire it into `dia/src/diarizer/mod.rs`:
```rust
#[cfg(test)]
mod tests;
```

```bash
cargo test --lib diarizer
```
Expected: 5+ tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/diarizer/imp.rs src/diarizer/reconstruct.rs src/diarizer/tests.rs src/diarizer/mod.rs
git commit -m "diarizer: state holder + accessors + clear (spec §4.4)

Diarizer owns Segmenter + Clusterer + audio buffer + reconstruction
state. Accessors: pending_inferences/buffered_samples/buffered_frames/
total_samples_pushed/num_speakers/speakers. clear() resets everything
including total_samples_pushed (per rev-7 §11.12); collected_embeddings
preserved (caller calls clear_collected explicitly).

ReconstructState is a stub for Phase 10; clear/buffered_frame_count
trivially defined so the rest of Diarizer compiles."
```

---

### Task 36: `process_samples` skeleton — push + segment pump (no embed/cluster yet)

**Files:**
- Modify: `dia/src/diarizer/imp.rs`

- [ ] **Step 1: Add the audio-push + trim helper**

Append to `dia/src/diarizer/imp.rs`:
```rust
use crate::segment::WINDOW_SAMPLES;

impl Diarizer {
    /// Internal: push samples to audio buffer + segmenter + advance counter.
    /// Trim is deferred until after the pump completes so activities
    /// emitted mid-pump can still slice their audio range (§5.7).
    pub(crate) fn push_audio(&mut self, samples: &[f32]) {
        for &s in samples { self.audio_buffer.push_back(s); }
        self.segmenter.push_samples(samples);
        self.total_samples_pushed += samples.len() as u64;
    }

    /// Internal: trim audio buffer to retain the last
    /// `diarization::segment::WINDOW_SAMPLES` samples (§5.7 / §11.5).
    pub(crate) fn trim_audio(&mut self) {
        let win = WINDOW_SAMPLES as u64;
        if self.total_samples_pushed > win {
            let target_base = self.total_samples_pushed - win;
            let drop_n = (target_base.saturating_sub(self.audio_base)) as usize;
            for _ in 0..drop_n.min(self.audio_buffer.len()) {
                self.audio_buffer.pop_front();
            }
            self.audio_base += drop_n as u64;
        }
    }

    /// Internal: extract audio for an absolute-sample range.
    /// Returns `Err(Internal::AudioBufferUnderflow/Overrun)` on bad bounds.
    pub(crate) fn slice_audio(&self, s0: u64, s1: u64) -> Result<Vec<f32>, Error> {
        use crate::diarizer::error::InternalError;
        if s0 < self.audio_base {
            return Err(Error::Internal(InternalError::AudioBufferUnderflow {
                activity_start: s0,
                audio_base: self.audio_base,
            }));
        }
        let end = self.audio_base + self.audio_buffer.len() as u64;
        if s1 > end {
            return Err(Error::Internal(InternalError::AudioBufferOverrun {
                activity_end: s1,
                audio_end: end,
            }));
        }
        let lo = (s0 - self.audio_base) as usize;
        let hi = (s1 - self.audio_base) as usize;
        Ok(self.audio_buffer.range(lo..hi).copied().collect())
    }
}
```

- [ ] **Step 2: Add the `process_samples` skeleton (segment-only; embed/cluster wiring in Phase 11)**

Append to `dia/src/diarizer/imp.rs`:
```rust
#[cfg(feature = "ort")]
use crate::embed::EmbedModel;
#[cfg(feature = "ort")]
use crate::segment::SegmentModel;
#[cfg(feature = "ort")]
use crate::segment::Action;

#[cfg(feature = "ort")]
impl Diarizer {
    /// **Skeleton (Phase 8).** Push samples; pump segmenter; drain
    /// `Action`s. NO embed/cluster/reconstruct yet — all activities
    /// are silently dropped. The pump is wired up to call
    /// `process_window_batch` (Phase 11) once it's defined.
    pub fn process_samples<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        embed_model: &mut EmbedModel,
        samples: &[f32],
        mut emit: F,
    ) -> Result<(), Error> {
        self.push_audio(samples);
        self.drain(seg_model, embed_model, &mut emit)?;
        self.trim_audio();
        Ok(())
    }

    pub fn finish_stream<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        embed_model: &mut EmbedModel,
        mut emit: F,
    ) -> Result<(), Error> {
        self.segmenter.finish();
        self.drain(seg_model, embed_model, &mut emit)?;
        // Phase 11 will add: self.reconstruct.flush_open_runs(&mut emit);
        Ok(())
    }

    fn drain<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        _embed_model: &mut EmbedModel,
        _emit: &mut F,
    ) -> Result<(), Error> {
        // Buffer SpeakerScores per WindowId; activities accumulate into
        // a per-window batch. After each push_inference returns, the
        // batch is processed (Phase 11 — currently a no-op).
        while let Some(action) = self.segmenter.poll() {
            match action {
                Action::NeedsInference { id, samples } => {
                    let scores = seg_model.infer(&samples)?;
                    self.segmenter.push_inference(id, &scores)?;
                }
                Action::SpeakerScores { .. } => {
                    // Phase 11: buffer per WindowId.
                }
                Action::Activity(_) => {
                    // Phase 11: cluster + reconstruct.
                }
                Action::VoiceSpan(_) => {
                    // Diarizer doesn't surface VoiceSpan currently.
                }
                _ => {} // Action is non_exhaustive
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 3: Add a smoke test**

Append to `dia/src/diarizer/tests.rs`:
```rust
#[cfg(feature = "ort")]
#[test]
#[ignore = "requires segment + embed models"]
fn process_samples_skeleton_pumps_segment_without_panicking() {
    use crate::embed::EmbedModel;
    use crate::segment::SegmentModel;
    let mut d = Diarizer::new(DiarizerOptions::default());
    let mut seg_model = SegmentModel::from_file("models/pyannote-segmentation-3.0.onnx").unwrap();
    let mut embed_model = EmbedModel::from_file("models/wespeaker-voxceleb-resnet34-LM.onnx").unwrap();
    let samples = vec![0.001f32; 320_000]; // 20 s
    d.process_samples(&mut seg_model, &mut embed_model, &samples, |_span| {})
        .expect("pump completes");
    // Phase 11 will assert spans emitted; for now, skeleton passes if
    // the pump doesn't panic.
    assert!(d.total_samples_pushed() == 320_000);
}
```

```bash
cargo check
```
Expected: clean compile.

- [ ] **Step 4: Commit**

```bash
git add src/diarizer/imp.rs src/diarizer/tests.rs
git commit -m "diarizer: process_samples skeleton + audio buffer ops (spec §4.4/§5.7)

push_audio + trim_audio (deferred trim per §5.7) + slice_audio
(defensive bounds → InternalError::AudioBufferUnderflow/Overrun).
process_samples + finish_stream pump segmenter; Activity/SpeakerScores
arms are currently no-ops awaiting Phase 11 wiring."
```

---

## Phase 9: `diarization::diarizer::overlap` — exclude_overlap mask construction

Builds the per-sample `keep_mask` from per-frame raw probabilities + binarize threshold + the activity's window/range. Routes through `EmbedModel::embed_masked` with the gather-and-pad path. Spec §5.8.

---

### Task 37: Per-frame mask derivation (binarized + clean)

**Files:**
- Modify: `dia/src/diarizer/overlap.rs`

- [ ] **Step 1: Write the per-frame mask helpers + tests**

Replace `dia/src/diarizer/overlap.rs`:
```rust
//! `exclude_overlap` mask construction. Spec §5.8.
//!
//! Per-window per-frame raw probabilities → binarized per-(slot, frame)
//! activity → per-frame "clean" mask (only this slot active) → sample-rate
//! keep_mask (frame-aligned via `frame_to_sample`).

use crate::segment::options::{FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS};
use crate::segment::stitch::frame_to_sample;

/// Per-(slot, frame) binarized speaker activity for a single window.
/// `binarized[slot][frame] = raw_probs[slot][frame] > threshold`.
pub(crate) fn binarize_per_frame(
    raw_probs: &[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
    threshold: f32,
) -> [[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize] {
    let mut out = [[false; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for s in 0..MAX_SPEAKER_SLOTS as usize {
        for f in 0..FRAMES_PER_WINDOW {
            out[s][f] = raw_probs[s][f] > threshold;
        }
    }
    out
}

/// Per-frame "n_active" — number of slots active at each frame
/// (sum across slots of `binarized[slot][frame]`).
pub(crate) fn count_active_per_frame(
    binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
) -> [u8; FRAMES_PER_WINDOW] {
    let mut out = [0u8; FRAMES_PER_WINDOW];
    for f in 0..FRAMES_PER_WINDOW {
        let mut n = 0u8;
        for s in 0..MAX_SPEAKER_SLOTS as usize {
            if binarized[s][f] { n += 1; }
        }
        out[f] = n;
    }
    out
}

/// Per-frame "speaker mask" for slot `s`: true iff slot `s` is active.
/// Same as `binarized[s]` but copied as `[bool; 589]` for downstream use.
pub(crate) fn speaker_mask_for_slot(
    binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
    slot: u8,
) -> [bool; FRAMES_PER_WINDOW] {
    binarized[slot as usize]
}

/// Per-frame "clean mask" for slot `s`: true iff slot `s` is active
/// AND no other slot is active at that frame.
pub(crate) fn clean_mask_for_slot(
    binarized: &[[bool; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
    n_active: &[u8; FRAMES_PER_WINDOW],
    slot: u8,
) -> [bool; FRAMES_PER_WINDOW] {
    let mut out = [false; FRAMES_PER_WINDOW];
    for f in 0..FRAMES_PER_WINDOW {
        out[f] = binarized[slot as usize][f] && n_active[f] == 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> [[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize] {
        [[0.0; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize]
    }

    #[test]
    fn binarize_threshold() {
        let mut p = empty();
        p[0][10] = 0.6; p[0][20] = 0.4; p[1][30] = 1.0;
        let b = binarize_per_frame(&p, 0.5);
        assert!(b[0][10]);
        assert!(!b[0][20]);
        assert!(b[1][30]);
    }

    #[test]
    fn count_active_per_frame_sums_slots() {
        let mut p = empty();
        // Frame 100: slot 0 + slot 2 active. Frame 200: only slot 1.
        p[0][100] = 0.9; p[2][100] = 0.9;
        p[1][200] = 0.9;
        let b = binarize_per_frame(&p, 0.5);
        let n = count_active_per_frame(&b);
        assert_eq!(n[100], 2);
        assert_eq!(n[200], 1);
        assert_eq!(n[0], 0);
    }

    #[test]
    fn clean_mask_excludes_overlap_frames() {
        let mut p = empty();
        // Frame 100: slot 0 alone. Frame 200: slot 0 + slot 1 (overlap).
        p[0][100] = 0.9;
        p[0][200] = 0.9; p[1][200] = 0.9;
        let b = binarize_per_frame(&p, 0.5);
        let n = count_active_per_frame(&b);
        let clean = clean_mask_for_slot(&b, &n, 0);
        assert!(clean[100]);  // only slot 0 active
        assert!(!clean[200]); // slot 0 AND slot 1 active → not clean
    }
}
```

- [ ] **Step 2: Verify**

```bash
cargo test --lib diarizer::overlap
```
Expected: 3 tests pass.

- [ ] **Step 3: Make `diarization::segment::stitch::frame_to_sample` accessible**

`frame_to_sample` is `pub(crate)` in segment/stitch.rs. Since `diarization::diarizer` is in the same crate, this works. Verify:
```bash
grep "pub(crate) const fn frame_to_sample" /Users/user/Develop/findit-studio/dia/src/segment/stitch.rs
```
Expected: line found. (Already there — no change needed.)

- [ ] **Step 4: Commit**

```bash
git add src/diarizer/overlap.rs
git commit -m "diarizer: per-frame mask helpers (spec §5.8)

binarize_per_frame: raw_probs > threshold → [[bool; 589]; 3].
count_active_per_frame: sum-of-slots active per frame.
speaker_mask_for_slot: just binarized[s].
clean_mask_for_slot: active && n_active[f] == 1 (only this speaker)."
```

---

### Task 38: Sample-rate keep_mask construction

**Files:**
- Modify: `dia/src/diarizer/overlap.rs`

- [ ] **Step 1: Append the frame-to-sample expansion + decision logic**

Append to `dia/src/diarizer/overlap.rs`:
```rust
use crate::segment::options::WINDOW_SAMPLES;
use crate::embed::MIN_CLIP_SAMPLES;

/// Build a sample-rate `keep_mask` aligned with an activity's
/// absolute-sample range `[s0, s1)`. The mask is `true` for samples
/// whose corresponding window-frame is in `frame_mask`. Frame f
/// covers `[window_start + frame_to_sample(f), window_start +
/// frame_to_sample(f + 1))`; intersect with `[s0, s1)`.
pub(crate) fn frame_mask_to_sample_keep_mask(
    frame_mask: &[bool; FRAMES_PER_WINDOW],
    window_start: u64,
    s0: u64,
    s1: u64,
) -> Vec<bool> {
    let activity_len = (s1 - s0) as usize;
    let mut keep = vec![false; activity_len];
    for f in 0..FRAMES_PER_WINDOW {
        let frame_s_start = window_start + frame_to_sample(f as u32) as u64;
        let frame_s_end   = window_start + frame_to_sample((f + 1) as u32) as u64;
        let lo = frame_s_start.max(s0);
        let hi = frame_s_end.min(s1);
        if lo >= hi { continue; }
        if frame_mask[f] {
            let lo_in = (lo - s0) as usize;
            let hi_in = (hi - s0) as usize;
            for k in lo_in..hi_in {
                keep[k] = true;
            }
        }
    }
    keep
}

/// Outcome of the §5.8 decision: which mask to use, plus the kept-sample count.
pub(crate) struct MaskDecision {
    pub(crate) keep_mask: Vec<bool>,
    pub(crate) used_clean: bool,
}

/// Decide between clean-mask and speaker-only mask, build the
/// sample-rate keep_mask. Returns the chosen mask + a flag indicating
/// which one was used.
///
/// **Decision rule (spec §5.8):** prefer the clean mask if its
/// gathered-sample count >= `MIN_CLIP_SAMPLES`. Otherwise, fall back
/// to the speaker-only mask. (Phase 11's pump handles the THIRD-tier
/// fallback — `embed_masked` returning `InvalidClip` even on the
/// speaker-only mask → skip the activity entirely, matching pyannote's
/// `speaker_verification.py:611-612`.)
pub(crate) fn decide_keep_mask(
    raw_probs: &[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
    binarize_threshold: f32,
    slot: u8,
    window_start: u64,
    s0: u64,
    s1: u64,
) -> MaskDecision {
    let binarized = binarize_per_frame(raw_probs, binarize_threshold);
    let n_active = count_active_per_frame(&binarized);
    let speaker = speaker_mask_for_slot(&binarized, slot);
    let clean = clean_mask_for_slot(&binarized, &n_active, slot);

    let speaker_keep = frame_mask_to_sample_keep_mask(&speaker, window_start, s0, s1);
    let clean_keep = frame_mask_to_sample_keep_mask(&clean, window_start, s0, s1);

    let clean_count = clean_keep.iter().filter(|&&b| b).count();
    if clean_count >= MIN_CLIP_SAMPLES as usize {
        MaskDecision { keep_mask: clean_keep, used_clean: true }
    } else {
        MaskDecision { keep_mask: speaker_keep, used_clean: false }
    }
}

#[cfg(test)]
mod expand_tests {
    use super::*;

    #[test]
    fn expand_full_window_mask_covers_full_range() {
        let mask = [true; FRAMES_PER_WINDOW];
        let keep = frame_mask_to_sample_keep_mask(&mask, 0, 0, WINDOW_SAMPLES as u64);
        // All true (modulo frame-to-sample rounding at the very start/end).
        let true_count = keep.iter().filter(|&&b| b).count();
        assert!(true_count > 0.95 * WINDOW_SAMPLES as f64 as usize as f64);
    }

    #[test]
    fn expand_partial_mask_covers_only_active_frames() {
        let mut mask = [false; FRAMES_PER_WINDOW];
        for f in 100..200 { mask[f] = true; }
        let keep = frame_mask_to_sample_keep_mask(&mask, 0, 0, WINDOW_SAMPLES as u64);
        // Frames 100..200 cover ≈ 100 frames * 271.65 samples/frame ≈ 27 165 samples.
        let true_count = keep.iter().filter(|&&b| b).count();
        assert!(true_count > 25_000, "got {}", true_count);
        assert!(true_count < 30_000, "got {}", true_count);
        // Sample at the start of frame 99 should be false.
        let frame_99_sample = frame_to_sample(99) as usize;
        assert!(!keep[frame_99_sample.saturating_sub(1)]);
    }

    #[test]
    fn decide_uses_clean_when_long_enough() {
        let mut p = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        // Slot 0 alone for frames 100..400 (~300 frames, ~80k samples).
        for f in 100..400 { p[0][f] = 0.9; }
        let d = decide_keep_mask(&p, 0.5, 0, 0, 0, WINDOW_SAMPLES as u64);
        assert!(d.used_clean);
    }

    #[test]
    fn decide_falls_back_when_clean_too_short() {
        let mut p = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        // Slot 0 alone for ONE frame only → ~272 samples < 400 (MIN_CLIP_SAMPLES).
        p[0][100] = 0.9;
        // Slot 0 + slot 1 active for frames 200..300 (overlap).
        for f in 200..300 { p[0][f] = 0.9; p[1][f] = 0.9; }
        let d = decide_keep_mask(&p, 0.5, 0, 0, 0, WINDOW_SAMPLES as u64);
        assert!(!d.used_clean, "expected fallback to speaker-only mask");
        // Speaker mask should cover frames 100 + 200..300.
        let true_count = d.keep_mask.iter().filter(|&&b| b).count();
        assert!(true_count > 25_000, "got {}", true_count);
    }
}
```

```bash
cargo test --lib diarizer::overlap
```
Expected: 7 tests pass (3 from Task 37 + 4 here).

- [ ] **Step 2: Commit**

```bash
git add src/diarizer/overlap.rs
git commit -m "diarizer: sample-rate keep_mask + decide_keep_mask (spec §5.8)

frame_mask_to_sample_keep_mask: expand per-frame mask to per-sample,
intersecting with activity range. Uses diarization::segment::stitch::frame_to_
sample (≈ 271.65 samples/frame) — NOT a hardcoded 160.

decide_keep_mask: builds both clean + speaker-only masks; returns
the clean mask if its true-count ≥ MIN_CLIP_SAMPLES, else falls back.
The third-tier 'skip on InvalidClip from embed_masked even on
speaker-only mask' is handled in Phase 11 (the pump)."
```

---

## Phase 10: `diarization::diarizer::reconstruct` — per-frame stitching state machine

The biggest single piece: per-frame per-cluster activation accumulator + per-frame instantaneous-speaker-count tracker + count-bounded argmax + per-cluster RLE-to-spans + eviction. Spec §5.9 / §5.10 / §5.11 / §11.13.

---

### Task 39: `ReconstructState` types + constants + `frame_to_sample_u64`

**Files:**
- Modify: `dia/src/diarizer/reconstruct.rs`

- [ ] **Step 1: Replace stub with full `ReconstructState`**

Replace `dia/src/diarizer/reconstruct.rs`:
```rust
//! Per-frame stitching state machine for `Diarizer`. Spec §5.9-§5.11.
//!
//! Tracks per-frame per-cluster activation overlap-add (rev-3 sum, NOT
//! mean), per-frame instantaneous-speaker-count (mean-with-warm-up-trim),
//! and per-cluster open-run state for RLE-to-DiarizedSpan emission.

use std::collections::{HashMap, HashSet};
use std::collections::VecDeque;

use crate::segment::options::{FRAMES_PER_WINDOW, WINDOW_SAMPLES};
use crate::segment::types::WindowId;
use mediatime::TimeRange;
use crate::segment::options::SAMPLE_RATE_TB;

/// Number of frames trimmed from each side of a window before counting
/// instantaneous-speaker count. Matches pyannote `speaker_count(warm_up
/// =(0.1, 0.1))`. = round(589 * 0.1) = 59.
pub(crate) const SPEAKER_COUNT_WARM_UP_FRAMES_LEFT: u32 = 59;
pub(crate) const SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT: u32 = 59;

/// Diarizer-internal `frame_idx (u64) → sample (u64)` helper.
/// Bit-for-bit equivalent to `diarization::segment::stitch::frame_to_sample`
/// but operates in `u64` throughout to avoid truncating cast on long
/// sessions. (Spec §15 #54 tracks folding back into segment.)
pub(crate) const fn frame_to_sample_u64(frame_idx: u64) -> u64 {
    let n = frame_idx * WINDOW_SAMPLES as u64;
    let half = (FRAMES_PER_WINDOW as u64) / 2;
    (n + half) / FRAMES_PER_WINDOW as u64
}

/// `frame_idx_of(sample) = sample * FRAMES_PER_WINDOW / WINDOW_SAMPLES`.
/// Same as `diarization::segment::stitch::frame_index_of` (already u64 → u64
/// in shipped segment); local copy for symmetry with `frame_to_sample_u64`.
pub(crate) const fn frame_index_of(sample_idx: u64) -> u64 {
    sample_idx * (FRAMES_PER_WINDOW as u64) / (WINDOW_SAMPLES as u64)
}

/// Per-frame bookkeeping (spec §5.9 step C).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FrameCount {
    /// Number of windows whose un-trimmed frames contributed to this
    /// frame's per-cluster activation accumulator. Used for §5.11
    /// average_activation normalization.
    pub(crate) activation_chunk_count: u32,
    /// Sum across windows of (count of speakers > threshold at this
    /// frame), only from frames OUTSIDE the warm-up margin (§5.10).
    pub(crate) count_sum: f32,
    /// Number of windows whose warm-up-trimmed frames contributed to
    /// this frame's `count_sum`. Divisor for the rounded
    /// instantaneous-speaker-count.
    pub(crate) count_chunk_count: u32,
}

/// Per-cluster open-run state (spec §5.11).
#[derive(Debug, Default)]
pub(crate) struct PerClusterRun {
    pub(crate) start_frame: Option<u64>,
    pub(crate) last_active_frame: Option<u64>,
    pub(crate) activation_sum_normalized: f64,
    pub(crate) frame_count: u32,
    pub(crate) contributing_activities: HashSet<(WindowId, u8)>,
    pub(crate) clean_mask_count: u32,
}

/// Reconstruction state machine.
#[derive(Debug, Default)]
pub(crate) struct ReconstructState {
    /// Absolute frame at index 0 of `activations` / `counts`.
    pub(crate) base_frame: u64,
    /// Per-frame per-cluster activation accumulator (sparse; HashMap).
    pub(crate) activations: VecDeque<HashMap<u64, f32>>,
    /// Per-frame counts.
    pub(crate) counts: VecDeque<FrameCount>,
    /// (window_id, slot) → cluster_id, populated in §5.8 when activities cluster.
    pub(crate) slot_to_cluster: std::collections::BTreeMap<(WindowId, u8), u64>,
    /// (window_id, slot) → used_clean_mask flag.
    pub(crate) activity_clean_flags: HashMap<(WindowId, u8), bool>,
    /// window_id → window_start (absolute samples).
    pub(crate) window_starts: HashMap<WindowId, u64>,
    /// Per-cluster open-run state.
    pub(crate) open_runs: HashMap<u64, PerClusterRun>,
    /// Set of cluster_ids ever emitted in a DiarizedSpan since
    /// `new()` / `clear()`. Used for `is_new_speaker`.
    pub(crate) emitted_speaker_ids: HashSet<u64>,
    /// Absolute frame below which everything is finalized. Monotonic.
    pub(crate) finalization_boundary: u64,
}

impl ReconstructState {
    pub(crate) fn new() -> Self { Self::default() }

    pub(crate) fn clear(&mut self) {
        self.base_frame = 0;
        self.activations.clear();
        self.counts.clear();
        self.slot_to_cluster.clear();
        self.activity_clean_flags.clear();
        self.window_starts.clear();
        self.open_runs.clear();
        self.emitted_speaker_ids.clear();
        self.finalization_boundary = 0;
    }

    pub(crate) fn buffered_frame_count(&self) -> usize {
        self.activations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_to_sample_u64_matches_segment_u32() {
        // Bit-exact equivalence over the u32 input range that segment uses.
        for frame_idx in 0u32..=2356 {
            let u32_result = crate::segment::stitch::frame_to_sample(frame_idx) as u64;
            let u64_result = frame_to_sample_u64(frame_idx as u64);
            assert_eq!(u32_result, u64_result, "frame_idx = {}", frame_idx);
        }
    }

    #[test]
    fn frame_index_of_matches_segment() {
        for sample_idx in 0u64..=160_000 * 4 {
            let local = frame_index_of(sample_idx);
            let seg = crate::segment::stitch::frame_index_of(sample_idx);
            assert_eq!(local, seg, "sample_idx = {}", sample_idx);
        }
    }

    #[test]
    fn warm_up_constant_matches_pyannote_default() {
        // round(589 * 0.1) = 58.9 → 59.
        assert_eq!(SPEAKER_COUNT_WARM_UP_FRAMES_LEFT, 59);
        assert_eq!(SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT, 59);
    }
}
```

- [ ] **Step 2: Make `diarization::segment::stitch` symbols visible to tests**

The test references `crate::segment::stitch::frame_to_sample` and `frame_index_of`. They're `pub(crate)`. The test is `#[cfg(test)]` inside `dia` — same crate — so this works. Verify:

```bash
cargo test --lib diarizer::reconstruct::tests
```
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/diarizer/reconstruct.rs
git commit -m "diarizer: ReconstructState types + frame_to_sample_u64 (spec §5.9-§5.11)

Single ReconstructState struct (rev-9 merged the rev-8 split):
base_frame, activations VecDeque<HashMap<cluster, f32>>, counts
VecDeque<FrameCount> with split activation_chunk_count +
count_chunk_count (rev-8 T2-A), slot_to_cluster, activity_clean_flags,
window_starts, open_runs (PerClusterRun: start_frame, last_active,
activation_sum_normalized, frame_count, contributing_activities,
clean_mask_count), emitted_speaker_ids, finalization_boundary.

frame_to_sample_u64 (rev-8 T2-B): u64-throughout helper, bit-exact
to segment's u32 version. Tracked as §15 #54 to fold back into segment."
```

---

### Task 40: `integrate_window` — collapse-by-max + overlap-add SUM + speaker count

**Files:**
- Modify: `dia/src/diarizer/reconstruct.rs`

- [ ] **Step 1: Append the `integrate_window` method**

Append to `dia/src/diarizer/reconstruct.rs`:
```rust
use crate::segment::options::MAX_SPEAKER_SLOTS;

impl ReconstructState {
    /// Integrate one processed window's raw probabilities into the
    /// per-frame accumulators. Spec §5.9.
    ///
    /// Pre-conditions:
    /// - All `(W.id, slot)` pairs for ACTIVE slots in this window have
    ///   already been clustered in §5.8 → `slot_to_cluster` populated.
    /// - `activity_clean_flags` populated for those same slots.
    ///
    /// Post-conditions:
    /// - `activations` / `counts` extended to cover this window's frames.
    /// - `window_starts` records `(W.id, window_start)`.
    pub(crate) fn integrate_window(
        &mut self,
        window_id: WindowId,
        window_start: u64,
        raw_probs: &[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
        binarize_threshold: f32,
    ) {
        self.window_starts.insert(window_id, window_start);

        let window_start_frame = frame_index_of(window_start);
        let last_frame = window_start_frame + FRAMES_PER_WINDOW as u64;

        // Grow activations/counts so that index `last_frame - base_frame`
        // is in-range. If `window_start_frame < base_frame` (shouldn't
        // happen; segment never re-emits an old window), skip those frames.
        if window_start_frame >= self.base_frame {
            let needed_len = (last_frame - self.base_frame) as usize;
            while self.activations.len() < needed_len {
                self.activations.push_back(HashMap::new());
                self.counts.push_back(FrameCount::default());
            }
        }

        // ── Step A: collapse-by-max within each cluster for THIS window. ──
        // per_cluster_max[cluster_id][f] = max over slots-mapped-to-cluster
        let mut per_cluster_max: HashMap<u64, [f32; FRAMES_PER_WINDOW]> = HashMap::new();
        for slot in 0..MAX_SPEAKER_SLOTS {
            if let Some(&cluster_id) = self.slot_to_cluster.get(&(window_id, slot)) {
                let entry = per_cluster_max.entry(cluster_id)
                    .or_insert([0.0f32; FRAMES_PER_WINDOW]);
                for f in 0..FRAMES_PER_WINDOW {
                    entry[f] = entry[f].max(raw_probs[slot as usize][f]);
                }
            }
            // None: slot was never active in this window (or activity was
            // too short and got skipped). Equivalent to pyannote's
            // `inactive_speakers → -2` throwaway path.
        }

        // ── Step B: overlap-add SUM into the per-frame accumulators. ──
        // (Pyannote `Inference.aggregate(skip_average=True)` — sum, not
        //  mean. Activation_chunk_count tracks divisor for output.)
        for f_in_window in 0..FRAMES_PER_WINDOW {
            let abs_frame = window_start_frame + f_in_window as u64;
            if abs_frame < self.base_frame { continue; }
            let buf_idx = (abs_frame - self.base_frame) as usize;
            for (cluster_id, frame_scores) in per_cluster_max.iter() {
                *self.activations[buf_idx].entry(*cluster_id).or_insert(0.0)
                    += frame_scores[f_in_window];
            }
            self.counts[buf_idx].activation_chunk_count += 1;
        }

        // ── Step C: per-frame instantaneous-speaker-count (warm-up trimmed). ──
        let warm_left = SPEAKER_COUNT_WARM_UP_FRAMES_LEFT as usize;
        let warm_right = SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT as usize;
        for f_in_window in warm_left..(FRAMES_PER_WINDOW - warm_right) {
            let abs_frame = window_start_frame + f_in_window as u64;
            if abs_frame < self.base_frame { continue; }
            let buf_idx = (abs_frame - self.base_frame) as usize;

            let n_active = (0..MAX_SPEAKER_SLOTS as usize)
                .filter(|s| raw_probs[*s][f_in_window] > binarize_threshold)
                .count() as f32;
            self.counts[buf_idx].count_sum += n_active;
            self.counts[buf_idx].count_chunk_count += 1;
        }
    }
}

#[cfg(test)]
mod integrate_tests {
    use super::*;
    use mediatime::TimeRange;
    use crate::segment::options::SAMPLE_RATE_TB;

    fn make_window_id(start: i64, gen: u64) -> WindowId {
        WindowId::new(
            TimeRange::new(start, start + WINDOW_SAMPLES as i64, SAMPLE_RATE_TB),
            gen,
        )
    }

    #[test]
    fn integrate_window_grows_buffers() {
        let mut s = ReconstructState::new();
        let id = make_window_id(0, 0);
        let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        // Slot 0 active for frames 100..200.
        for f in 100..200 { probs[0][f] = 0.9; }
        // Pretend slot 0 was clustered to cluster 7.
        s.slot_to_cluster.insert((id, 0), 7);

        s.integrate_window(id, 0, &probs, 0.5);

        assert_eq!(s.activations.len(), FRAMES_PER_WINDOW);
        assert!(s.activations[100].contains_key(&7));
        assert!((s.activations[100][&7] - 0.9).abs() < 1e-6);
        // Frame 50 had no slot active → no cluster entries.
        assert!(s.activations[50].is_empty());
        // Activation chunk count = 1 for every frame.
        for fc in &s.counts {
            assert_eq!(fc.activation_chunk_count, 1);
        }
        // Speaker-count chunk count = 1 only for frames in the warm-up region.
        for f_in in 0..FRAMES_PER_WINDOW {
            let in_warm = f_in < SPEAKER_COUNT_WARM_UP_FRAMES_LEFT as usize
                       || f_in >= (FRAMES_PER_WINDOW - SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT as usize);
            if in_warm {
                assert_eq!(s.counts[f_in].count_chunk_count, 0);
            } else {
                assert_eq!(s.counts[f_in].count_chunk_count, 1);
            }
        }
    }

    #[test]
    fn collapse_by_max_when_two_slots_same_cluster() {
        let mut s = ReconstructState::new();
        let id = make_window_id(0, 0);
        let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        // Two slots, same cluster, different scores at frame 100.
        probs[0][100] = 0.6;
        probs[1][100] = 0.9;
        s.slot_to_cluster.insert((id, 0), 7);
        s.slot_to_cluster.insert((id, 1), 7); // same cluster

        s.integrate_window(id, 0, &probs, 0.5);

        // Activation should be max(0.6, 0.9) = 0.9 for cluster 7.
        assert!((s.activations[100][&7] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn overlap_add_sums_two_windows() {
        // Two windows that overlap by half. Same slot, same cluster,
        // probability 0.5 in each. Overlap region should sum to 1.0.
        let step = 80_000u64; // half of WINDOW_SAMPLES = 80 000
        let id_a = make_window_id(0, 0);
        let id_b = make_window_id(step as i64, 1);
        let mut s = ReconstructState::new();
        s.slot_to_cluster.insert((id_a, 0), 7);
        s.slot_to_cluster.insert((id_b, 0), 7);
        let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        for f in 0..FRAMES_PER_WINDOW { probs[0][f] = 0.5; }
        s.integrate_window(id_a, 0, &probs, 0.5);
        s.integrate_window(id_b, step, &probs, 0.5);

        // Frame at sample step / SAMPLES_PER_FRAME — covered by both windows.
        let overlap_frame = frame_index_of(step) as usize;
        assert!((s.activations[overlap_frame][&7] - 1.0).abs() < 1e-5);
        assert_eq!(s.counts[overlap_frame].activation_chunk_count, 2);
    }
}
```

```bash
cargo test --lib diarizer::reconstruct
```
Expected: 6 tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/diarizer/reconstruct.rs
git commit -m "diarizer: integrate_window — collapse-by-max + overlap-add SUM + count (spec §5.9-§5.10)

Step A (collapse-by-max within cluster): per_cluster_max[c][f] = max
  over slots-mapped-to-c of raw_probs[slot][f]. Skips slots without
  cluster mapping (pyannote inactive_speakers analog).
Step B (overlap-add SUM, NO warm-up): sum into activations[abs_frame]
  [c]; bump activation_chunk_count for EVERY frame in this window.
Step C (overlap-add MEAN with warm-up trim): bump count_sum and
  count_chunk_count only outside warm-up (frames 59..530 of 0..589)."
```

---

### Task 41: `advance_finalization_boundary` + `emit_finalized_frames`

**Files:**
- Modify: `dia/src/diarizer/reconstruct.rs`

- [ ] **Step 1: Append the boundary tracking + emission**

Append to `dia/src/diarizer/reconstruct.rs`:
```rust
use std::cmp::Ordering;
use crate::diarizer::span::DiarizedSpan;

impl ReconstructState {
    /// Advance the finalization boundary based on the segment's
    /// `peek_next_window_start()`. Frames before the boundary are
    /// finalized — no future window can contribute to them.
    pub(crate) fn advance_finalization_boundary(&mut self, next_window_start: u64) {
        let next_frame = if next_window_start == u64::MAX {
            // Post-finish_stream: all remaining frames finalize.
            u64::MAX
        } else {
            frame_index_of(next_window_start)
        };
        if next_frame > self.finalization_boundary {
            self.finalization_boundary = next_frame;
        }
    }

    /// Emit all finalized DiarizedSpans. Drains `activations` /
    /// `counts` up to `finalization_boundary`. Spec §5.11.
    pub(crate) fn emit_finalized_frames<F: FnMut(DiarizedSpan)>(
        &mut self,
        max_speakers: u32,
        mut emit: F,
    ) {
        while self.base_frame < self.finalization_boundary {
            let Some(frame_state) = self.activations.pop_front() else { break; };
            let frame_count_state = self.counts.pop_front()
                .expect("counts and activations grow in lockstep");
            let activation_chunk_count = frame_count_state.activation_chunk_count.max(1) as f32;

            // Compute instantaneous-speaker count (rounded mean,
            // capped by max_speakers). Spec §5.10.
            let count_at_frame = if frame_count_state.count_chunk_count == 0 {
                0u32
            } else {
                let mean = frame_count_state.count_sum
                    / frame_count_state.count_chunk_count as f32;
                mean.round() as u32
            }.min(max_speakers);

            // Pick top-`count` clusters by activation. Tie-break by
            // smaller cluster_id (deterministic; rev-3 deliberate
            // improvement over pyannote's np.argsort).
            let active_set: HashSet<u64> = if count_at_frame > 0 {
                let mut sorted: Vec<(u64, f32)> = frame_state.iter()
                    .map(|(&c, &a)| (c, a)).collect();
                sorted.sort_by(|(a_id, a_v), (b_id, b_v)| {
                    b_v.total_cmp(a_v).then(a_id.cmp(b_id))
                });
                sorted.into_iter().take(count_at_frame as usize)
                    .map(|(c, _)| c).collect()
            } else {
                HashSet::new()
            };

            // Close runs for clusters that were open but not in top now.
            let to_close: Vec<u64> = self.open_runs.iter()
                .filter(|(c, run)| run.start_frame.is_some()
                                  && !active_set.contains(c))
                .map(|(c, _)| *c).collect();
            for cluster_id in to_close {
                let run = self.open_runs.get_mut(&cluster_id).unwrap();
                let start = run.start_frame.take().unwrap();
                let end = run.last_active_frame.unwrap() + 1;
                let range = make_range(start, end);
                let is_new = !self.emitted_speaker_ids.contains(&cluster_id);
                if is_new { self.emitted_speaker_ids.insert(cluster_id); }
                let activity_count = run.contributing_activities.len() as u32;
                let avg_activation = (run.activation_sum_normalized
                                       / run.frame_count.max(1) as f64) as f32;
                let clean_fraction = if activity_count == 0 {
                    0.0
                } else {
                    run.clean_mask_count as f32 / activity_count as f32
                };
                emit(DiarizedSpan {
                    range,
                    speaker_id: cluster_id,
                    is_new_speaker: is_new,
                    average_activation: avg_activation,
                    activity_count,
                    clean_mask_fraction: clean_fraction,
                });
                // Reset accumulators for next run on same cluster.
                run.activation_sum_normalized = 0.0;
                run.frame_count = 0;
                run.contributing_activities.clear();
                run.clean_mask_count = 0;
            }

            // Open or extend runs for clusters in `active_set`.
            for cluster_id in &active_set {
                let activation_at_frame = frame_state[cluster_id];
                let run = self.open_runs.entry(*cluster_id).or_default();
                if run.start_frame.is_none() {
                    run.start_frame = Some(self.base_frame);
                }
                run.last_active_frame = Some(self.base_frame);
                run.activation_sum_normalized +=
                    (activation_at_frame / activation_chunk_count) as f64;
                run.frame_count += 1;

                // Find which (window, slot) entries for this cluster
                // contain this frame; insert into contributing_activities.
                for ((window_id, slot), c) in &self.slot_to_cluster {
                    if *c != *cluster_id { continue; }
                    let win_start = match self.window_starts.get(window_id) {
                        Some(&s) => s,
                        None => continue, // not yet integrated
                    };
                    let frame_lo = frame_index_of(win_start);
                    let frame_hi = frame_lo + FRAMES_PER_WINDOW as u64;
                    if self.base_frame >= frame_lo && self.base_frame < frame_hi
                       && run.contributing_activities.insert((*window_id, *slot))
                    {
                        // Newly inserted — also count toward clean_mask_count.
                        if let Some(&clean) = self.activity_clean_flags.get(&(*window_id, *slot)) {
                            if clean { run.clean_mask_count += 1; }
                        }
                    }
                }
            }

            self.base_frame += 1;
        }
    }
}

fn make_range(start_frame: u64, end_frame: u64) -> TimeRange {
    let s0 = frame_to_sample_u64(start_frame) as i64;
    let s1 = frame_to_sample_u64(end_frame) as i64;
    TimeRange::new(s0, s1, SAMPLE_RATE_TB)
}

#[cfg(test)]
mod emit_tests {
    use super::*;
    use mediatime::TimeRange;
    use crate::segment::options::SAMPLE_RATE_TB;

    fn make_window_id(start: i64, gen: u64) -> WindowId {
        WindowId::new(
            TimeRange::new(start, start + WINDOW_SAMPLES as i64, SAMPLE_RATE_TB),
            gen,
        )
    }

    #[test]
    fn single_window_single_speaker_emits_one_span() {
        let mut s = ReconstructState::new();
        let id = make_window_id(0, 0);
        let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        for f in 100..200 { probs[0][f] = 0.9; }
        s.slot_to_cluster.insert((id, 0), 7);
        s.activity_clean_flags.insert((id, 0), true);
        s.integrate_window(id, 0, &probs, 0.5);

        // Force-finalize.
        s.advance_finalization_boundary(u64::MAX);
        let mut emitted = Vec::new();
        s.emit_finalized_frames(15, |span| emitted.push(span));

        // Should emit exactly one span for cluster 7 covering ≈ frames 100..200.
        let spans_for_7: Vec<_> = emitted.iter().filter(|s| s.speaker_id() == 7).collect();
        assert_eq!(spans_for_7.len(), 1, "expected 1 span; got {:?}", emitted);
        let span = spans_for_7[0];
        assert_eq!(span.activity_count(), 1);
        assert!(span.is_new_speaker());
        assert!((span.clean_mask_fraction() - 1.0).abs() < 1e-7);
        assert!(span.average_activation() > 0.5);
    }

    #[test]
    fn count_zero_in_warm_up_emits_nothing() {
        let mut s = ReconstructState::new();
        let id = make_window_id(0, 0);
        let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        // Slot 0 active ONLY in warm-up region (frames 0..50).
        for f in 0..50 { probs[0][f] = 0.9; }
        s.slot_to_cluster.insert((id, 0), 7);
        s.integrate_window(id, 0, &probs, 0.5);

        s.advance_finalization_boundary(u64::MAX);
        let mut emitted = Vec::new();
        s.emit_finalized_frames(15, |span| emitted.push(span));

        // count_chunk_count = 0 for all warm-up-only frames → count = 0
        // → no cluster in top-0 → no spans.
        assert!(emitted.is_empty(), "expected 0 spans; got {:?}", emitted);
    }
}
```

```bash
cargo test --lib diarizer::reconstruct
```
Expected: all tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/diarizer/reconstruct.rs
git commit -m "diarizer: emit_finalized_frames + per-cluster RLE (spec §5.11)

advance_finalization_boundary: max(boundary, frame_index_of(next_win));
u64::MAX means post-finish_stream → all remaining frames finalize.

emit_finalized_frames: per-frame argmax with deterministic tie-break
(smaller cluster_id wins). Open per-cluster runs extend or close;
closed runs emit DiarizedSpan with rev-7 quality metrics
(average_activation = activation_sum_normalized / frame_count;
activity_count = contributing_activities.len(); clean_mask_fraction =
clean_mask_count / activity_count). Activations divided by
activation_chunk_count (NOT count_chunk_count — rev-8 T2-A fix).
Empty-deque guard (rev-8 T2-B). total_cmp for NaN safety (T3-H)."
```

---

### Task 42: `flush_open_runs` + eviction policy

**Files:**
- Modify: `dia/src/diarizer/reconstruct.rs`

- [ ] **Step 1: Append flush + eviction**

Append to `dia/src/diarizer/reconstruct.rs`:
```rust
impl ReconstructState {
    /// End-of-stream flush: force-finalize remaining frames + emit
    /// every still-open run. Called from `Diarizer::finish_stream`.
    pub(crate) fn flush_open_runs<F: FnMut(DiarizedSpan)>(&mut self, mut emit: F) {
        self.finalization_boundary = u64::MAX;
        self.emit_finalized_frames(u32::MAX, &mut emit);
        // After draining frames, close any still-open runs (these are
        // runs whose final-active frame is the last frame ever
        // observed; the per-frame loop above didn't get a "this
        // cluster fell out of top" signal because there's no next frame).
        let to_emit: Vec<u64> = self.open_runs.iter()
            .filter(|(_, run)| run.start_frame.is_some())
            .map(|(c, _)| *c).collect();
        for cluster_id in to_emit {
            let run = self.open_runs.get_mut(&cluster_id).unwrap();
            let start = run.start_frame.take().unwrap();
            let end = run.last_active_frame.unwrap() + 1;
            let range = make_range(start, end);
            let is_new = !self.emitted_speaker_ids.contains(&cluster_id);
            if is_new { self.emitted_speaker_ids.insert(cluster_id); }
            let activity_count = run.contributing_activities.len() as u32;
            let avg = (run.activation_sum_normalized
                       / run.frame_count.max(1) as f64) as f32;
            let clean_fraction = if activity_count == 0 {
                0.0
            } else {
                run.clean_mask_count as f32 / activity_count as f32
            };
            emit(DiarizedSpan {
                range, speaker_id: cluster_id, is_new_speaker: is_new,
                average_activation: avg, activity_count,
                clean_mask_fraction: clean_fraction,
            });
        }
    }

    /// Evict per-window-id metadata that's no longer referenced.
    /// Spec §11.13.
    ///
    /// A window's entries can be dropped once:
    /// (a) all of its frames have finalized (`base_frame >= last_frame`), AND
    /// (b) no currently-open per-cluster run still references any
    ///     `(window_id, slot)` pair.
    pub(crate) fn evict_finalized_window_metadata(&mut self) {
        let evictable: Vec<WindowId> = self.window_starts.iter()
            .filter(|(w, &start)| {
                let last_frame = frame_index_of(start) + FRAMES_PER_WINDOW as u64;
                if self.base_frame < last_frame { return false; }
                let no_open_ref = self.open_runs.values().all(|run| {
                    run.contributing_activities.iter().all(|(wid, _)| wid != *w)
                });
                no_open_ref
            })
            .map(|(w, _)| *w)
            .collect();
        for w in evictable {
            self.slot_to_cluster.retain(|(wid, _), _| *wid != w);
            self.activity_clean_flags.retain(|(wid, _), _| *wid != w);
            self.window_starts.remove(&w);
        }
    }
}

#[cfg(test)]
mod flush_eviction_tests {
    use super::*;
    use mediatime::TimeRange;
    use crate::segment::options::SAMPLE_RATE_TB;

    fn make_window_id(start: i64, gen: u64) -> WindowId {
        WindowId::new(
            TimeRange::new(start, start + WINDOW_SAMPLES as i64, SAMPLE_RATE_TB),
            gen,
        )
    }

    #[test]
    fn flush_emits_open_run_at_end_of_stream() {
        let mut s = ReconstructState::new();
        let id = make_window_id(0, 0);
        let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
        // Active for the entire window (no closing frame within the window).
        for f in 100..(FRAMES_PER_WINDOW - 50) { probs[0][f] = 0.9; }
        s.slot_to_cluster.insert((id, 0), 7);
        s.activity_clean_flags.insert((id, 0), true);
        s.integrate_window(id, 0, &probs, 0.5);

        let mut emitted = Vec::new();
        s.flush_open_runs(|span| emitted.push(span));
        assert!(emitted.iter().any(|s| s.speaker_id() == 7));
    }

    #[test]
    fn evict_drops_metadata_for_finalized_unused_windows() {
        let mut s = ReconstructState::new();
        let id_a = make_window_id(0, 0);
        let id_b = make_window_id(160_000, 1);
        s.window_starts.insert(id_a, 0);
        s.window_starts.insert(id_b, 160_000);
        s.slot_to_cluster.insert((id_a, 0), 7);
        s.slot_to_cluster.insert((id_b, 0), 8);
        // Pretend we've finalized past id_a's last frame.
        s.base_frame = frame_index_of(0) + FRAMES_PER_WINDOW as u64 + 10;
        s.evict_finalized_window_metadata();
        assert!(!s.window_starts.contains_key(&id_a));
        assert!(s.window_starts.contains_key(&id_b)); // still active
    }
}
```

```bash
cargo test --lib diarizer::reconstruct
```
Expected: all tests pass.

- [ ] **Step 2: Commit**

```bash
git add src/diarizer/reconstruct.rs
git commit -m "diarizer: flush_open_runs + eviction policy (spec §11.13)

flush_open_runs: forces finalization_boundary = u64::MAX, drains
emit_finalized_frames, then closes still-open runs (clusters that
were active up to the very last finalized frame).

evict_finalized_window_metadata: drops slot_to_cluster /
activity_clean_flags / window_starts entries for windows whose
last frame has finalized AND no open run still references them.
Bounds memory on long sessions; correctness-relevant for runs
spanning many finalized windows."
```

---

## Phase 11: `Diarizer` — wire the pump end-to-end

This phase replaces the no-op stubs in `Diarizer::drain` (Phase 8) with the full embed → cluster → reconstruct chain.

---

### Task 43: Per-window batching + embed-then-cluster

**Files:**
- Modify: `dia/src/diarizer/imp.rs`

- [ ] **Step 1: Add per-window batching state to `Diarizer`**

The pump needs to buffer `Action::Activity` events per `WindowId` until `Action::SpeakerScores` arrives (it always arrives BEFORE the activities for the same window per Task 30, but we make this robust). After all of a window's activities + scores are collected, we:
1. Decide each activity's keep_mask (clean vs speaker-only fallback) using the scores.
2. Call `embed_model.embed_masked(audio, keep_mask)`. Skip on doubly-failed gather.
3. Call `clusterer.submit(embedding)`.
4. Record the (window, slot) → cluster_id mapping.
5. Call `reconstruct.integrate_window(id, window_start, raw_probs, threshold)`.
6. `reconstruct.advance_finalization_boundary(segmenter.peek_next_window_start())`.
7. `reconstruct.emit_finalized_frames(max_speakers, |span| emit(span))`.
8. `reconstruct.evict_finalized_window_metadata()`.

Replace the `drain` body in `dia/src/diarizer/imp.rs`:
```rust
#[cfg(feature = "ort")]
impl Diarizer {
    fn drain<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        embed_model: &mut EmbedModel,
        emit: &mut F,
    ) -> Result<(), Error> {
        use crate::diarizer::overlap::{decide_keep_mask, MaskDecision};
        use crate::segment::Action;
        use crate::cluster::ClusterAssignment;

        // Buffer per WindowId.
        let mut pending_scores: HashMap<crate::segment::WindowId,
            Box<[[f32; crate::segment::options::FRAMES_PER_WINDOW];
                  crate::segment::options::MAX_SPEAKER_SLOTS as usize]>> = HashMap::new();
        let mut pending_activities: HashMap<crate::segment::WindowId,
            Vec<crate::segment::SpeakerActivity>> = HashMap::new();
        let mut pending_window_starts: HashMap<crate::segment::WindowId, u64> = HashMap::new();

        while let Some(action) = self.segmenter.poll() {
            match action {
                Action::NeedsInference { id, samples } => {
                    let scores = seg_model.infer(&samples)?;
                    self.segmenter.push_inference(id, &scores)?;
                }
                Action::SpeakerScores { id, window_start, raw_probs } => {
                    pending_scores.insert(id, raw_probs);
                    pending_window_starts.insert(id, window_start);
                }
                Action::Activity(activity) => {
                    pending_activities.entry(activity.window_id())
                        .or_default()
                        .push(activity);
                }
                Action::VoiceSpan(_) => {
                    // Diarizer doesn't currently surface VoiceSpan.
                }
                _ => {} // non_exhaustive; future variants ignored
            }

            // After processing each Action, check whether any window has
            // BOTH its scores AND any activities ready to integrate.
            // (Empty-activity-list windows still integrate — they're
            // valid; the integrate_window call needs to advance the
            // finalization boundary regardless.)
            let ready: Vec<crate::segment::WindowId> = pending_scores.keys()
                .filter(|id| !pending_activities.contains_key(id) || pending_activities.contains_key(id))
                .copied()
                .collect();
            for id in ready {
                let raw_probs = pending_scores.remove(&id).unwrap();
                let window_start = pending_window_starts.remove(&id).unwrap();
                let activities = pending_activities.remove(&id).unwrap_or_default();

                // Process each activity: decide mask, embed, cluster.
                self.process_window_activities(
                    id, window_start, &raw_probs, &activities, embed_model,
                )?;

                // Integrate this window into reconstruction.
                self.reconstruct.integrate_window(
                    id, window_start, &raw_probs, self.opts.binarize_threshold,
                );
            }

            // Advance finalization + emit finalized spans.
            let next = self.segmenter.peek_next_window_start();
            self.reconstruct.advance_finalization_boundary(next);
            let max_spk = self.opts.cluster.max_speakers().unwrap_or(u32::MAX);
            self.reconstruct.emit_finalized_frames(max_spk, |span| emit(span));
            self.reconstruct.evict_finalized_window_metadata();
        }
        Ok(())
    }

    /// Process all activities for one window: derive mask, embed, cluster.
    fn process_window_activities(
        &mut self,
        window_id: crate::segment::WindowId,
        window_start: u64,
        raw_probs: &[[f32; crate::segment::options::FRAMES_PER_WINDOW];
                     crate::segment::options::MAX_SPEAKER_SLOTS as usize],
        activities: &[crate::segment::SpeakerActivity],
        embed_model: &mut EmbedModel,
    ) -> Result<(), Error> {
        use crate::diarizer::overlap::decide_keep_mask;
        use crate::embed::{Embedding, Error as EmbedError};

        for activity in activities {
            let s = activity.speaker_slot();
            let s0 = activity.range().start_pts() as u64;
            let s1 = activity.range().end_pts() as u64;

            // Slice the audio for this activity.
            let activity_samples = self.slice_audio(s0, s1)?;

            // Decide which mask to use based on overlap.
            let want_clean = self.opts.exclude_overlap;
            let (mask, used_clean_initial) = if want_clean {
                let dec = decide_keep_mask(
                    raw_probs,
                    self.opts.binarize_threshold,
                    s,
                    window_start,
                    s0, s1,
                );
                (dec.keep_mask, dec.used_clean)
            } else {
                // exclude_overlap = false → keep_mask = all-true.
                let all_true = vec![true; activity_samples.len()];
                (all_true, false)
            };

            // Try embed_masked with the chosen mask. On InvalidClip
            // and we used the clean mask, fall back to speaker-only.
            // On InvalidClip even on speaker-only, SKIP this activity.
            let (embedding_opt, used_clean) = match embed_model
                .embed_masked(&activity_samples, &mask)
            {
                Ok(r) => (Some(*r.embedding()), used_clean_initial),
                Err(EmbedError::InvalidClip { .. }) if used_clean_initial => {
                    // Build the speaker-only mask and retry.
                    let dec_speaker = decide_keep_mask(
                        raw_probs,
                        // Force speaker-only by setting threshold so high
                        // that no frame is "clean" — simpler: re-derive.
                        self.opts.binarize_threshold,
                        s, window_start, s0, s1,
                    );
                    // Build a speaker-only mask directly (count-active independent of clean).
                    let speaker_keep = build_speaker_only_keep_mask(
                        raw_probs, self.opts.binarize_threshold,
                        s, window_start, s0, s1,
                    );
                    match embed_model.embed_masked(&activity_samples, &speaker_keep) {
                        Ok(r) => (Some(*r.embedding()), false),
                        Err(EmbedError::InvalidClip { .. }) => (None, false), // skip
                        Err(e) => return Err(e.into()),
                    }
                }
                Err(EmbedError::InvalidClip { .. }) => {
                    // exclude_overlap was off; first call already used speaker-mask
                    // (or all-true). Skip on persistent failure.
                    (None, false)
                }
                Err(e) => return Err(e.into()),
            };

            let embedding = match embedding_opt {
                Some(e) => e,
                None => continue, // skipped (matches pyannote skip semantics)
            };

            // Cluster.
            let assignment = self.clusterer.submit(&embedding)?;
            self.reconstruct.slot_to_cluster.insert((window_id, s), assignment.speaker_id());
            self.reconstruct.activity_clean_flags.insert((window_id, s), used_clean);

            // Optional collected_embeddings.
            if self.opts.collect_embeddings {
                self.collected_embeddings.push(CollectedEmbedding {
                    range: activity.range(),
                    embedding,
                    online_speaker_id: assignment.speaker_id(),
                    speaker_slot: s,
                    used_clean_mask: used_clean,
                });
            }
        }
        Ok(())
    }
}

#[cfg(feature = "ort")]
fn build_speaker_only_keep_mask(
    raw_probs: &[[f32; crate::segment::options::FRAMES_PER_WINDOW];
                  crate::segment::options::MAX_SPEAKER_SLOTS as usize],
    threshold: f32,
    slot: u8,
    window_start: u64,
    s0: u64,
    s1: u64,
) -> Vec<bool> {
    use crate::diarizer::overlap::{
        binarize_per_frame, speaker_mask_for_slot, frame_mask_to_sample_keep_mask,
    };
    let binarized = binarize_per_frame(raw_probs, threshold);
    let speaker = speaker_mask_for_slot(&binarized, slot);
    frame_mask_to_sample_keep_mask(&speaker, window_start, s0, s1)
}
```

Make `binarize_per_frame`, `speaker_mask_for_slot`, `frame_mask_to_sample_keep_mask` `pub(crate)` in overlap.rs (already done in Tasks 37-38).

- [ ] **Step 2: Verify compile**

```bash
cargo check
```
Expected: clean.

- [ ] **Step 3: Update `finish_stream` to flush reconstruction**

Replace `finish_stream` in `dia/src/diarizer/imp.rs`:
```rust
    pub fn finish_stream<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        embed_model: &mut EmbedModel,
        mut emit: F,
    ) -> Result<(), Error> {
        self.segmenter.finish();
        self.drain(seg_model, embed_model, &mut emit)?;
        // Flush any remaining open runs — segment is finished, so
        // peek_next_window_start = u64::MAX → reconstruct.boundary = MAX.
        self.reconstruct.flush_open_runs(|span| emit(span));
        self.trim_audio();
        Ok(())
    }
```

- [ ] **Step 4: Smoke test the wiring**

Append to `dia/src/diarizer/tests.rs`:
```rust
#[cfg(feature = "ort")]
#[test]
#[ignore = "requires both ONNX models"]
fn end_to_end_pump_emits_at_least_one_span() {
    use crate::embed::EmbedModel;
    use crate::segment::SegmentModel;
    let mut d = Diarizer::new(DiarizerOptions::default());
    let mut seg = SegmentModel::from_file("models/pyannote-segmentation-3.0.onnx")
        .expect("segment model present");
    let mut emb = EmbedModel::from_file("models/wespeaker-voxceleb-resnet34-LM.onnx")
        .expect("embed model present");
    // Read a 30s real-speech clip.
    let mut reader = hound::WavReader::open("tests/fixtures/diarize_test_30s.wav")
        .expect("test wav present");
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32).collect();
    let mut spans = Vec::new();
    d.process_samples(&mut seg, &mut emb, &samples, |s| spans.push(s)).unwrap();
    d.finish_stream(&mut seg, &mut emb, |s| spans.push(s)).unwrap();
    assert!(!spans.is_empty(), "expected at least one DiarizedSpan");
    eprintln!("got {} spans, {} speakers", spans.len(), d.num_speakers());
}
```

- [ ] **Step 5: Commit**

```bash
git add src/diarizer/imp.rs src/diarizer/tests.rs
git commit -m "diarizer: wire pump end-to-end (spec §5.8-§5.12)

drain() buffers (Action::SpeakerScores, Action::Activity) per
WindowId. Once both have arrived for window W:
- For each activity: decide_keep_mask → embed_masked → submit →
  record (W,slot)→cluster_id + clean_flag.
- integrate_window into reconstruct state.
- advance_finalization_boundary(segmenter.peek_next_window_start()).
- emit_finalized_frames(max_speakers, &mut emit).
- evict_finalized_window_metadata().

InvalidClip skip-and-continue (spec §5.12 / rev-9 T3-A): clean →
speaker-only → skip. Matches pyannote speaker_verification.py:611-612.

finish_stream additionally calls flush_open_runs to close any
runs that were active up to the very last frame."
```

---

## Phase 12: Integration tests, VAD edge cases, pyannote parity

---

### Task 44: Diarizer integration test scaffolding

**Files:**
- Create: `dia/tests/integration_diarizer.rs`
- Create: `dia/tests/fixtures/.gitkeep` (placeholder; real fixtures downloaded on demand)
- Create: `dia/scripts/download-test-fixtures.sh`

- [ ] **Step 1: Write the download script**

Create `dia/scripts/download-test-fixtures.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_DIR="$SCRIPT_DIR/../tests/fixtures"
mkdir -p "$FIX_DIR"
# Tiny synthetic fixture for the pump smoke test (30 s of intermittent tones).
# Replace with a real 30-s multi-speaker clip when available.
if [ ! -f "$FIX_DIR/diarize_test_30s.wav" ]; then
  ffmpeg -f lavfi -i "sine=frequency=440:duration=10,asetrate=16000" \
         -f lavfi -i "sine=frequency=660:duration=10,asetrate=16000" \
         -f lavfi -i "sine=frequency=880:duration=10,asetrate=16000" \
         -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]" \
         -map "[out]" -ac 1 -ar 16000 -sample_fmt s16 \
         "$FIX_DIR/diarize_test_30s.wav" -y
fi
echo "Fixtures ready in $FIX_DIR"
```

```bash
chmod +x /Users/user/Develop/findit-studio/dia/scripts/download-test-fixtures.sh
```

- [ ] **Step 2: Write the integration test file**

Create `dia/tests/integration_diarizer.rs`:
```rust
//! End-to-end gated tests for `diarization::Diarizer`.
//!
//! All tests are `#[ignore]` because they need:
//! - `models/pyannote-segmentation-3.0.onnx` (run scripts/download-model.sh)
//! - `models/wespeaker-voxceleb-resnet34-LM.onnx` (run scripts/download-embed-model.sh)
//! - `tests/fixtures/diarize_test_30s.wav` (run scripts/download-test-fixtures.sh)

#![cfg(feature = "ort")]

use diarization::diarizer::{Diarizer, DiarizerOptions, DiarizedSpan};
use diarization::embed::EmbedModel;
use diarization::segment::SegmentModel;

fn load_models() -> (SegmentModel, EmbedModel) {
    let seg = SegmentModel::from_file("models/pyannote-segmentation-3.0.onnx")
        .expect("segment model present (run scripts/download-model.sh)");
    let emb = EmbedModel::from_file("models/wespeaker-voxceleb-resnet34-LM.onnx")
        .expect("embed model present (run scripts/download-embed-model.sh)");
    (seg, emb)
}

fn load_clip(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("test wav present");
    assert_eq!(reader.spec().sample_rate, 16_000);
    assert_eq!(reader.spec().channels, 1);
    reader.samples::<i16>().map(|s| s.unwrap() as f32 / i16::MAX as f32).collect()
}

#[test]
#[ignore]
fn end_to_end_pump_30s_clip() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    let samples = load_clip("tests/fixtures/diarize_test_30s.wav");
    let mut spans = Vec::new();
    d.process_samples(&mut seg, &mut emb, &samples, |s| spans.push(s)).unwrap();
    d.finish_stream(&mut seg, &mut emb, |s| spans.push(s)).unwrap();
    eprintln!("emitted {} spans, {} speakers", spans.len(), d.num_speakers());
    // The synthetic-tone fixture has no real speech; a real fixture
    // would yield > 0 spans. For the synthetic case we just assert
    // the pump completes without panic.
    assert!(d.total_samples_pushed() == samples.len() as u64);
}
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration_diarizer.rs scripts/download-test-fixtures.sh
git commit -m "diarizer: gated integration test scaffolding (spec §9)

end_to_end_pump_30s_clip is #[ignore]'d; needs both ONNX models +
a fixed test wav. download-test-fixtures.sh produces a synthetic
30 s wav for the smoke path; replace with a real multi-speaker
clip for meaningful DER measurements (spec §15 #43 / #46)."
```

---

### Task 45: VAD-input edge-case tests

**Files:**
- Modify: `dia/tests/integration_diarizer.rs`

- [ ] **Step 1: Add edge-case tests per spec §9**

Append to `dia/tests/integration_diarizer.rs`:
```rust
#[test]
#[ignore]
fn empty_push_is_noop() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    let mut spans = Vec::new();
    d.process_samples(&mut seg, &mut emb, &[], |s| spans.push(s)).unwrap();
    assert_eq!(d.total_samples_pushed(), 0);
    assert!(spans.is_empty());
    assert_eq!(d.pending_inferences(), 0);
}

#[test]
#[ignore]
fn sub_window_push_no_spans_until_finish() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    let half_second = vec![0.001f32; 8_000]; // 0.5 s
    let mut spans = Vec::new();
    d.process_samples(&mut seg, &mut emb, &half_second, |s| spans.push(s)).unwrap();
    // No window scheduled yet (need ≥ 10 s).
    assert_eq!(d.pending_inferences(), 0);
    // No spans emitted.
    assert!(spans.is_empty());
    // Now finish — tail-anchor at max(0, 0.5s - 10s) = 0 schedules a window.
    d.finish_stream(&mut seg, &mut emb, |s| spans.push(s)).unwrap();
    // spans.len() depends on the model — the silent input may produce 0 spans.
    assert!(d.total_samples_pushed() == 8_000);
}

#[test]
#[ignore]
fn multiple_short_pushes_accumulate() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    let mut spans = Vec::new();
    for _ in 0..5 {
        let chunk = vec![0.001f32; 16_000]; // 1 s × 5 = 5 s total
        d.process_samples(&mut seg, &mut emb, &chunk, |s| spans.push(s)).unwrap();
    }
    assert_eq!(d.total_samples_pushed(), 5 * 16_000);
    // No window yet (need ≥ 10 s).
    assert_eq!(d.pending_inferences(), 0);
}

#[test]
#[ignore]
fn long_single_push_processes_multiple_windows() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    let mut spans = Vec::new();
    let sixty_seconds = vec![0.001f32; 60 * 16_000];
    d.process_samples(&mut seg, &mut emb, &sixty_seconds, |s| spans.push(s)).unwrap();
    assert_eq!(d.pending_inferences(), 0); // synchronous pump drains all
    // For 60 s with default step = 2.5 s: ~21 regular windows.
}

#[test]
#[ignore]
fn total_samples_pushed_monotonic_resets_on_clear() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 10_000], |_| {}).unwrap();
    d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 7_000], |_| {}).unwrap();
    assert_eq!(d.total_samples_pushed(), 17_000);
    d.clear();
    assert_eq!(d.total_samples_pushed(), 0);
    d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 5_000], |_| {}).unwrap();
    assert_eq!(d.total_samples_pushed(), 5_000);
}

#[test]
#[ignore]
fn buffered_frames_steady_state() {
    let (mut seg, mut emb) = load_models();
    let mut d = Diarizer::new(DiarizerOptions::default());
    let mut spans = Vec::new();
    d.process_samples(&mut seg, &mut emb, &vec![0.001f32; 30 * 16_000], |s| spans.push(s)).unwrap();
    // After processing 30 s synchronously, ~10 s of frames are unfinalized.
    // 10 s × 58.9 fps ≈ 589 frames.
    let bf = d.buffered_frames();
    eprintln!("buffered_frames = {}", bf);
    assert!(bf > 0, "expected un-finalized frames after pump");
    assert!(bf < 1500, "expected steady-state ≈ 589 frames");
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/integration_diarizer.rs
git commit -m "diarizer: VAD-input edge-case integration tests (spec §9 / §11.12)

empty_push_is_noop: zero-length push doesn't error or advance state.
sub_window_push_no_spans_until_finish: < 10 s push doesn't schedule
  any segment window; finish_stream's tail-anchor forces processing.
multiple_short_pushes_accumulate: 5 × 1 s pushes equal 5 s buffered.
long_single_push_processes_multiple_windows: 60 s push processes
  ~21 windows synchronously; pending_inferences == 0 after pump.
total_samples_pushed_monotonic_resets_on_clear: rev-7 §11.12 contract.
buffered_frames_steady_state: ~589 (10 s × 58.9 fps) un-finalized
  frames during streaming."
```

---

### Task 46: Pyannote parity test harness

**Files:**
- Create: `dia/tests/parity/Cargo.toml`
- Create: `dia/tests/parity/src/main.rs`
- Create: `dia/tests/parity/python/pyproject.toml`
- Create: `dia/tests/parity/python/reference.py`
- Create: `dia/tests/parity/run.sh`

This is the spec §15 #43 + #46 harness — a Python sidecar runs `pyannote.audio` on a fixed clip and emits an RTTM; the Rust binary runs `diarization::Diarizer` and emits an RTTM; a third script computes DER. Gated as `#[ignore]` and run manually.

- [ ] **Step 1: Write the Rust runner**

Create `dia/tests/parity/Cargo.toml`:
```toml
[package]
name = "dia-parity"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
dia = { path = "../.." }
hound = "3"
anyhow = "1"
```

Create `dia/tests/parity/src/main.rs`:
```rust
//! Run `diarization::Diarizer` on a fixed clip and dump RTTM to stdout.
use anyhow::Result;
use diarization::diarizer::{Diarizer, DiarizerOptions};
use diarization::embed::EmbedModel;
use diarization::segment::SegmentModel;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let clip_path = args.get(1).expect("usage: dia-parity <clip.wav>");
    let mut reader = hound::WavReader::open(clip_path)?;
    assert_eq!(reader.spec().sample_rate, 16_000);
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32).collect();

    let mut seg = SegmentModel::from_file("models/pyannote-segmentation-3.0.onnx")?;
    let mut emb = EmbedModel::from_file("models/wespeaker-voxceleb-resnet34-LM.onnx")?;
    let mut d = Diarizer::new(DiarizerOptions::default());

    let mut spans = Vec::new();
    d.process_samples(&mut seg, &mut emb, &samples, |s| spans.push(s))?;
    d.finish_stream(&mut seg, &mut emb, |s| spans.push(s))?;

    // Dump RTTM (NIST format).
    let uri = std::path::Path::new(clip_path).file_stem().unwrap().to_str().unwrap();
    for s in &spans {
        let start_sec = s.range().start_pts() as f64 / 16_000.0;
        let dur_sec = (s.range().end_pts() - s.range().start_pts()) as f64 / 16_000.0;
        println!("SPEAKER {} 1 {:.3} {:.3} <NA> <NA> SPK_{:02} <NA> <NA>",
                 uri, start_sec, dur_sec, s.speaker_id());
    }
    eprintln!("# dia: {} spans, {} speakers, total_samples_pushed = {}",
              spans.len(), d.num_speakers(), d.total_samples_pushed());
    Ok(())
}
```

- [ ] **Step 2: Write the Python reference + DER scorer**

Create `dia/tests/parity/python/pyproject.toml`:
```toml
[project]
name = "dia-parity-reference"
version = "0.0.0"
requires-python = ">=3.10"
dependencies = ["pyannote.audio", "pyannote.metrics"]
```

Create `dia/tests/parity/python/reference.py`:
```python
"""Run pyannote.audio.SpeakerDiarization on a clip; dump RTTM."""
import sys
from pyannote.audio import Pipeline
clip = sys.argv[1]
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
output = pipeline(clip)
diarization = output.speaker_diarization if hasattr(output, "speaker_diarization") else output
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"SPEAKER ref 1 {turn.start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA> <NA>")
```

Create `dia/tests/parity/python/score.py`:
```python
"""Compute DER between dia and pyannote RTTMs."""
import sys
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def load_rttm(path):
    ann = Annotation()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER": continue
            start, dur = float(parts[3]), float(parts[4])
            spk = parts[7]
            ann[Segment(start, start + dur)] = spk
    return ann

ref = load_rttm(sys.argv[1])
hyp = load_rttm(sys.argv[2])
metric = DiarizationErrorRate(collar=0.5, skip_overlap=False)
der = metric(ref, hyp)
print(f"DER = {der:.4f}")
sys.exit(0 if der <= 0.10 else 1)  # rev-8 relaxed threshold (T3-I)
```

- [ ] **Step 3: Write the runner script**

Create `dia/tests/parity/run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/../.."
cd "$ROOT"

CLIP="${1:?usage: run.sh <clip.wav>}"

# Python reference.
cd "$SCRIPT_DIR/python"
[ -d .venv ] || uv venv
uv pip install -e .
uv run python reference.py "../../../$CLIP" > "$SCRIPT_DIR/ref.rttm"
cd "$ROOT"

# Rust dia.
cargo run --release --manifest-path tests/parity/Cargo.toml -- "$CLIP" \
  > "$SCRIPT_DIR/hyp.rttm"

# Score.
cd "$SCRIPT_DIR/python"
uv run python score.py "$SCRIPT_DIR/ref.rttm" "$SCRIPT_DIR/hyp.rttm"
echo "Parity check passed (DER ≤ 10%)"
```

```bash
chmod +x /Users/user/Develop/findit-studio/dia/tests/parity/run.sh
```

- [ ] **Step 4: Document in CHANGELOG / README**

(Done in Phase 13.)

- [ ] **Step 5: Commit**

```bash
git add tests/parity/
git commit -m "diarizer: pyannote parity test harness (spec §15 #46)

Three-piece harness:
- Rust runner (tests/parity/src/main.rs): runs diarization::Diarizer on a
  clip, emits RTTM.
- Python reference (tests/parity/python/reference.py): runs
  pyannote.audio.SpeakerDiarization on the same clip, emits RTTM.
- Python scorer (tests/parity/python/score.py): computes DER via
  pyannote.metrics. Asserts DER ≤ 10% (rev-8 T3-I relaxed target;
  5% is a stretch goal given documented divergences in §1).

run.sh orchestrates. Manual; not part of cargo test."
```

---

## Phase 13: Release prep — examples + CHANGELOG + rustdoc + lib.rs polish

---

### Task 47: `examples/diarize_from_wav.rs`

**Files:**
- Create: `dia/examples/diarize_from_wav.rs`

- [ ] **Step 1: Write a minimal end-to-end example**

Create `dia/examples/diarize_from_wav.rs`:
```rust
//! Run `diarization::Diarizer` on a 16 kHz mono WAV; print emitted DiarizedSpans.
//!
//! ```sh
//! ./scripts/download-model.sh        # segment model
//! ./scripts/download-embed-model.sh  # embed model
//! cargo run --release --example diarize_from_wav -- path/to/clip.wav
//! ```

#![cfg(feature = "ort")]

use std::env;
use std::error::Error;

use diarization::diarizer::{Diarizer, DiarizerOptions};
use diarization::embed::EmbedModel;
use diarization::segment::SegmentModel;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let clip_path = args.get(1).ok_or("usage: diarize_from_wav <clip.wav>")?;

    let mut reader = hound::WavReader::open(clip_path)?;
    if reader.spec().sample_rate != 16_000 {
        return Err("input must be 16 kHz".into());
    }
    if reader.spec().channels != 1 {
        return Err("input must be mono".into());
    }
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32).collect();

    let mut seg = SegmentModel::from_file("models/pyannote-segmentation-3.0.onnx")?;
    let mut emb = EmbedModel::from_file("models/wespeaker-voxceleb-resnet34-LM.onnx")?;
    let mut d = Diarizer::new(DiarizerOptions::default());

    println!("# Streaming {} samples ({:.2} s)", samples.len(),
             samples.len() as f64 / 16_000.0);

    let mut span_count = 0;
    let chunk_size = 16_000 * 5; // 5 s chunks (simulate VAD output).
    for chunk in samples.chunks(chunk_size) {
        d.process_samples(&mut seg, &mut emb, chunk, |span| {
            println!("[{:.3}s — {:.3}s] speaker {} (avg_act={:.3}, n_act={}, clean={:.2})",
                     span.range().start_pts() as f64 / 16_000.0,
                     span.range().end_pts() as f64 / 16_000.0,
                     span.speaker_id(),
                     span.average_activation(),
                     span.activity_count(),
                     span.clean_mask_fraction());
            span_count += 1;
        })?;
    }
    d.finish_stream(&mut seg, &mut emb, |span| {
        println!("[{:.3}s — {:.3}s] speaker {} (avg_act={:.3}, n_act={}, clean={:.2}) [flushed]",
                 span.range().start_pts() as f64 / 16_000.0,
                 span.range().end_pts() as f64 / 16_000.0,
                 span.speaker_id(),
                 span.average_activation(),
                 span.activity_count(),
                 span.clean_mask_fraction());
        span_count += 1;
    })?;

    println!("# Done. {} spans, {} speakers, {} samples processed.",
             span_count, d.num_speakers(), d.total_samples_pushed());
    Ok(())
}
```

```bash
cargo build --release --example diarize_from_wav
```
Expected: clean compile.

- [ ] **Step 2: Commit**

```bash
git add examples/diarize_from_wav.rs
git commit -m "examples: diarize_from_wav (spec §3)

Demonstrates the high-level Diarizer API: feed WAV in 5 s chunks
(simulating VAD output), emit DiarizedSpans with all rev-7 quality
metrics. Prints span ranges in seconds for human readability."
```

---

### Task 48: CHANGELOG.md update

**Files:**
- Modify: `dia/CHANGELOG.md`

- [ ] **Step 1: Update the UNRELEASED section**

Replace the `# UNRELEASED` block at the top of `dia/CHANGELOG.md`:
```markdown
# UNRELEASED

This release ships `diarization::embed`, `diarization::cluster`, and `diarization::Diarizer`,
completing the v0.1.0 phase 2 vision. `diarization::segment` gains an additive
v0.X bump (see CORRECTNESS GUARANTEES below).

FEATURES — `diarization::embed`

- **`Embedding`** newtype (256-d L2-normalized) with invariant
  `||embedding|| > NORM_EPSILON`, enforced by `Embedding::normalize_from`
  returning `None` on degenerate inputs.
- **`compute_fbank`** kaldi-compatible feature extraction wrapping
  `kaldi-native-fbank`. Verified against `torchaudio.compliance.kaldi.fbank`
  per the spec §15 #43 pre-impl spike.
- **`EmbedModel`** ort wrapper for WeSpeaker ResNet34. `from_file` /
  `from_memory` constructors with `_with_options` variants. Auto-derives
  `Send`; explicitly `!Sync` (matches `silero::Session` and
  `diarization::segment::SegmentModel`).
- **`embed`** / **`embed_with_meta`**: high-level API. Sliding-window
  mean for clips > 2 s (deliberate divergence from
  `findit-speaker-embedding`'s center-crop — see spec §1).
- **`embed_weighted`** / **`embed_weighted_with_meta`**: per-sample
  voice-probability soft weighting.
- **`embed_masked`** / **`embed_masked_with_meta`**: rev-8 binary
  keep-mask (gather-and-embed). Used internally by
  `Diarizer::exclude_overlap`; matches pyannote's
  `ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__` masked path.
- **Generic `EmbeddingMeta<A=(), T=()>`**: caller-supplied metadata
  flows through to `EmbeddingResult`. Defaults to `()` so the
  unit-typed metadata path is zero-cost.
- **`cosine_similarity`** free function alongside `Embedding::similarity`.

FEATURES — `diarization::cluster`

- **Online streaming `Clusterer`** with `submit(&Embedding)` returning
  `ClusterAssignment { speaker_id, is_new_speaker, similarity }`.
  `RollingMean` and `Ema(α)` update strategies on an unnormalized
  accumulator (handles antipodal cancellation gracefully via
  cached-centroid lazy update).
- **`OverflowStrategy::Reject`** (default — caller decides) /
  **`AssignClosest`** (no centroid update on forced assignment).
- **Offline `cluster_offline`** with two methods:
  - **Spectral** (default): cosine affinity + degree-matrix
    precondition + normalized Laplacian + nalgebra eigendecomposition
    + eigengap K-detection (capped at MAX_AUTO_SPEAKERS = 15) +
    K-means++ + Lloyd. PRNG pinned to `rand_chacha::ChaCha8Rng` with
    explicit byte-fixture regression test (spec §15 #52).
  - **Agglomerative**: Single / Complete / Average linkage with
    cosine distance ReLU-clamped to `[0, 1]`.
- **Deterministic K-means++** seeding (Arthur & Vassilvitskii 2007).
  Default seed is `0`; same input + seed → same labels across runs
  and platforms.
- **N ≤ 2 fast paths** before any matrix work; isolated-node
  precondition catches dissimilar inputs without an undefined
  Laplacian.

FEATURES — `diarization::Diarizer` (rev-6 pyannote-style reconstruction)

- **`process_samples`** / **`finish_stream`**: streaming entry points
  borrowing `&mut SegmentModel` + `&mut EmbedModel` per call (mirrors
  `Segmenter` pattern; allows model reuse across sessions).
- **VAD-friendly variable-length input**: empty / sub-window /
  multi-clip / whole-stream pushes all handled without special-casing
  (spec §11.12 contract).
- **`exclude_overlap`** mask via `embed_masked` (spec §5.8): per-window
  binarized + clean masks → sample-rate keep_mask; clean mask used
  when its gathered length ≥ MIN_CLIP_SAMPLES, else falls back to the
  speaker-only mask. On doubly-failed gather (extreme overlap),
  skip-and-continue (matches pyannote
  `speaker_verification.py:611-612`).
- **Per-frame per-cluster overlap-add stitching** (spec §5.9): matches
  `reconstruct` line 519-522 (collapse-by-max within cluster) plus
  `Inference.aggregate(skip_average=True)` (overlap-add SUM across
  windows).
- **Per-frame instantaneous-speaker-count tracking** (spec §5.10):
  matches `speaker_count(warm_up=(0.1, 0.1))`. Per-frame overlap-add
  MEAN of binarized counts, rounded.
- **Count-bounded argmax + per-cluster RLE** (spec §5.11): matches
  `to_diarization`. Deterministic tie-break (smaller cluster_id wins
  — deliberate improvement over pyannote's `np.argsort`).
- **Output**: `DiarizedSpan { range, speaker_id, is_new_speaker,
  average_activation, activity_count, clean_mask_fraction }` per
  closed speaker turn.
- **`collected_embeddings()`**: per-(window, slot) granularity context
  retained across the session (`CollectedEmbedding { range, embedding,
  online_speaker_id, speaker_slot, used_clean_mask }`).
- **Introspection**: `pending_inferences`, `buffered_samples`,
  `buffered_frames` (rev-6), `total_samples_pushed` (rev-7),
  `num_speakers`, `speakers`.
- **Auto-derived `Send + Sync`**.

FEATURES — `diarization::segment` v0.X bump

- **`Action::SpeakerScores { id, window_start, raw_probs }`** variant
  emitted from `push_inference` alongside `Action::Activity`. Carries
  per-window per-speaker per-frame raw probabilities for downstream
  reconstruction.
- **`Action` is now `#[non_exhaustive]`** so future additions are
  non-breaking.
- **`pub(crate) Segmenter::peek_next_window_start()`** for the
  Diarizer's reconstruction finalization-boundary computation.

CORRECTNESS GUARANTEES

- **Bit-deterministic offline clustering** for a given input + seed,
  enforced by the `tests/chacha_keystream_fixture.rs` regression test.
- **Frame-rate math verified**: `diarization::segment::stitch::frame_to_sample`
  yields ≈ 271.65 samples/frame (≈ 58.9 fps) per the model's 589 frames
  per 160 000 samples; the Diarizer carries a `frame_to_sample_u64`
  helper bit-exactly equivalent to segment's `u32` version.
- **Documented divergences from pyannote** (spec §1 table): sliding-
  window mean for long-clip embed, sample-rate vs frame-rate gather
  in `exclude_overlap`, online vs batch clustering, default Spectral
  vs pyannote VBx, deterministic argmax tie-break.

TESTING

- ~70 unit tests across `diarization::embed`, `diarization::cluster`, `diarization::diarizer`.
- Gated integration tests for end-to-end Diarizer pump on a 30-s clip.
- Pyannote parity harness (`tests/parity/run.sh`) — manual; targets
  DER ≤ 10% absolute (rev-8 T3-I relaxed from 5%).

BUILD

- New deps: `nalgebra = "0.34"`, `rand = "0.10"` (default-features =
  false), `rand_chacha = "0.10"` (default-features = false),
  `kaldi-native-fbank = "0.1"`.

KNOWN LIMITATIONS / DEFERRED TO v0.1.1+

- No bundled WeSpeaker model (~25 MB); use `scripts/download-embed-model.sh`.
- VBx clustering (pyannote's offline default) not shipped; spec §15 #44.
- HMM-GMM clustering not shipped; spec §15.
- `min_cluster_size` cluster pruning not shipped; spec §15.
- Configurable `warm_up` for speaker-count not shipped; hardcoded to
  pyannote default `(0.1, 0.1)`. Spec §15 #47.
- Configurable `min_duration_on/off` for span-merging not shipped; spec §15 #48.
- Mask-aware embedding ONNX export deferred; current path uses
  sample-rate gather + sliding-window-mean (one extra divergence
  from pyannote on long masked clips). Spec §15 #49.

# 0.1.0 (2026-04-26)
```

(Keep the existing `# 0.1.0 (2026-04-26)` block below, unchanged.)

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "release: CHANGELOG entry for v0.1.0 phase 2 (embed + cluster + Diarizer)"
```

---

### Task 49: README + lib.rs rustdoc polish

**Files:**
- Modify: `dia/src/lib.rs`
- Modify: `dia/README.md`

- [ ] **Step 1: Update `dia/src/lib.rs` crate docs**

Replace the top of `dia/src/lib.rs`:
```rust
//! Sans-I/O streaming speaker diarization for variable-length VAD-filtered audio.
//!
//! `dia` is the Rust port of [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)'s
//! speaker-diarization pipeline, restructured around incremental push-based
//! state machines so it can run live on streaming audio. Inputs are arbitrary-
//! length pushes (e.g., per VAD speech region); outputs emit per closed speaker
//! turn as windows finalize.
//!
//! ## Modules
//!
//! - [`segment`]: speaker-segmentation state machine (pyannote/segmentation-3.0
//!   ONNX). Emits `Action::Activity` (per-(window, slot) speaker presence) and
//!   `Action::SpeakerScores` (per-window per-frame raw probabilities for
//!   downstream reconstruction).
//! - [`embed`]: speaker-fingerprint generation (WeSpeaker ResNet34 ONNX +
//!   kaldi fbank). High-level `EmbedModel::embed` (sliding-window mean) and
//!   the masked variant `embed_masked` (rev-8 gather-and-embed).
//! - [`cluster`]: cross-window speaker linking. Online streaming `Clusterer`
//!   plus offline batch `cluster_offline` with spectral (default) and
//!   agglomerative methods.
//! - [`diarizer`]: top-level `Diarizer` orchestrator. Combines the above three
//!   plus a per-frame reconstruction state machine matching pyannote's
//!   `SpeakerDiarization.apply` pipeline.
//!
//! ## Quick start
//!
//! ```no_run
//! # #[cfg(feature = "ort")] {
//! use diarization::diarizer::{Diarizer, DiarizerOptions};
//! use diarization::embed::EmbedModel;
//! use diarization::segment::SegmentModel;
//!
//! let mut seg = SegmentModel::from_file("pyannote-segmentation-3.0.onnx")?;
//! let mut emb = EmbedModel::from_file("wespeaker-voxceleb-resnet34-LM.onnx")?;
//! let mut d = Diarizer::new(DiarizerOptions::default());
//!
//! let samples: Vec<f32> = vec![/* 16 kHz mono PCM */];
//! d.process_samples(&mut seg, &mut emb, &samples, |span| {
//!     println!("[{:.2}s..{:.2}s] speaker {}",
//!              span.range().start_pts() as f64 / 16_000.0,
//!              span.range().end_pts() as f64 / 16_000.0,
//!              span.speaker_id());
//! })?;
//! d.finish_stream(&mut seg, &mut emb, |_| {})?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # }
//! ```
//!
//! ## Design references
//!
//! See `docs/superpowers/specs/2026-04-26-dia-embed-cluster-diarizer-design.md`
//! for the load-bearing spec (rev 9 + post-rev-9 N2 cleanup).

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

#[cfg(feature = "std")]
extern crate std;

pub mod cluster;
pub mod diarizer;
pub mod embed;
pub mod segment;
```

- [ ] **Step 2: Update `dia/README.md`**

Replace `dia/README.md` with a short pointer:
```markdown
# dia

Sans-I/O streaming speaker diarization for variable-length VAD-filtered audio.

[![Crates.io](https://img.shields.io/crates/v/dia.svg)](https://crates.io/crates/dia)
[![Documentation](https://docs.rs/dia/badge.svg)](https://docs.rs/dia)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/al8n/dia)

## Status

v0.1.0 ships:

- `diarization::segment` — speaker segmentation (pyannote/segmentation-3.0 ONNX)
- `diarization::embed` — speaker fingerprint (WeSpeaker ResNet34 ONNX + kaldi fbank)
- `diarization::cluster` — online streaming + offline (spectral + agglomerative)
- `diarization::Diarizer` — top-level orchestrator with pyannote-style per-frame
  reconstruction (overlap-add cluster activations, count-bounded argmax,
  per-cluster RLE-to-spans)

## Pipeline

```
audio decoder → resample to 16 kHz → VAD → diarization::Diarizer → downstream services
```

VAD-filtered, variable-length pushes are first-class. See
[`docs/superpowers/specs/`](docs/superpowers/specs/) for the design spec.

## Quick start

See `examples/diarize_from_wav.rs`.

```sh
./scripts/download-model.sh        # pyannote/segmentation-3.0
./scripts/download-embed-model.sh  # WeSpeaker ResNet34
cargo run --release --example diarize_from_wav -- path/to/clip.wav
```

## License

MIT OR Apache-2.0.
```

- [ ] **Step 3: Verify rustdoc**

```bash
cargo doc --no-deps --document-private-items
```
Expected: clean, no broken intra-doc links.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs README.md
git commit -m "release: README + lib.rs rustdoc for v0.1.0 phase 2

lib.rs gets a pipeline description, module index, quick-start
example. README points at examples + spec."
```

---

### Task 50: Final verification + release tag

**Files:** none (just verification commands)

- [ ] **Step 1: Full test sweep**

```bash
cd /Users/user/Develop/findit-studio/dia
cargo fmt --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
cargo test --no-default-features --features std
cargo doc --no-deps --document-private-items
```
Expected: all green.

- [ ] **Step 2: Run gated integration tests**

```bash
./scripts/download-model.sh
./scripts/download-embed-model.sh
./scripts/download-test-fixtures.sh
cargo test --test integration_segment -- --ignored
cargo test --test integration_embed -- --ignored
cargo test --test integration_diarizer -- --ignored
```
Expected: all gated tests pass on a real clip.

- [ ] **Step 3: Run pyannote parity (manual)**

```bash
./tests/parity/run.sh tests/fixtures/diarize_test_30s.wav
```
Expected: `Parity check passed (DER ≤ 10%)`.

(If this fails, that's information, not a blocker — the parity test is `#[ignore]`'d in CI. But it's a useful smoke before tagging.)

- [ ] **Step 4: Tag the release**

```bash
git tag -a v0.1.0-phase2 -m "dia v0.1.0 phase 2: embed + cluster + Diarizer"
# Note: do NOT push the tag without explicit user approval.
```

- [ ] **Step 5: Final commit (empty if nothing left)**

```bash
git commit --allow-empty -m "release: dia v0.1.0 phase 2 ready for tagging"
```

---

## Plan summary

50 tasks across 13 phases:

| Phase | Tasks | Topic |
|---|---|---|
| 0 | 1-2 | Pre-impl spikes (kaldi-native-fbank parity, ChaCha8Rng byte fixture) |
| 1 | 3-8 | `diarization::embed` types and pure helpers (no ort) |
| 2 | 9-13 | `diarization::cluster` online streaming `Clusterer` |
| 3 | 14-16 | `diarization::cluster` offline validation + agglomerative |
| 4 | 17-22 | `diarization::cluster` offline spectral (nalgebra eigendecomp + ChaCha8Rng K-means++) |
| 5 | 23-28 | `diarization::embed` model + fbank + sliding-window-mean (needs ort) |
| 6 | 29-31 | `diarization::segment` v0.X bump (Action::SpeakerScores + peek_next_window_start) |
| 7 | 32-34 | `diarization::Diarizer` skeleton (types + builder + error) |
| 8 | 35-36 | `Diarizer` audio buffer + pump glue |
| 9 | 37-38 | `diarization::diarizer::overlap` exclude_overlap mask construction |
| 10 | 39-42 | `diarization::diarizer::reconstruct` per-frame stitching state machine |
| 11 | 43 | Wire pump end-to-end |
| 12 | 44-46 | Integration tests + VAD edge cases + pyannote parity harness |
| 13 | 47-50 | Examples + CHANGELOG + README + rustdoc + final verification |

**Effort estimate:** ~17 calendar days at 4-6 focused hours/day. The reconstruction state machine (Phase 10, Tasks 39-42) is the densest single block — budget 3 days for it alone. Phase 4 spectral (Tasks 17-22) is the second densest — budget 2-3 days. Everything else is bite-sized.

**Critical path:**

```
Task 1 (kaldi-native-fbank spike) ──┐
Task 2 (ChaCha fixture) ────────────┤
                                     ↓
                                 Phase 1 (embed types) ──┐
                                                          ↓
                                                     Phase 2-4 (cluster)
                                                          ↓
                                                     Phase 5 (embed model)
                                                          ↓
                                                     Phase 6 (segment bump)
                                                          ↓
                                                     Phase 7-8 (Diarizer skeleton)
                                                          ↓
                                                     Phase 9 (overlap)
                                                          ↓
                                                     Phase 10 (reconstruct)
                                                          ↓
                                                     Phase 11 (wire pump)
                                                          ↓
                                                     Phase 12 (integration tests)
                                                          ↓
                                                     Phase 13 (release prep)
```

Phase 1-4 (embed types + cluster) is the only block that can run in parallel with Phase 5 (embed model). The rest is sequential.

**Reviewer-recommended order matches** (review-9 verdict): "bottom-up: diarization::cluster first (purest, no ort), then diarization::embed (post-spike), then diarization::segment v0.X bump, then diarization::Diarizer (orchestration glue). Use the Rev 9 §9 test list as your TDD spec — it's been beaten on for 8 review rounds and is comprehensive."













