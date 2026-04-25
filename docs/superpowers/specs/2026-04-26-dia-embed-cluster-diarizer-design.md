# dia — embed + cluster + Diarizer design (v0.1.0 phase 2)

**Revision 2** (2026-04-26, post-first-adversarial-review).
**Status:** ready for re-review.
**Scope:** Three new modules to ship in dia v0.1.0 alongside `dia::segment`:
- `dia::embed` — speaker fingerprint generation (port of `findit-speaker-embedding` + improvements)
- `dia::cluster` — cross-window speaker linking (online streaming + offline batch)
- `dia::Diarizer` — top-level orchestrator combining segment + embed + cluster

> **Revision 2 fixes** (full list in §13 / §15): renames `WINDOW_SAMPLES`
> to `EMBED_WINDOW_SAMPLES` so it doesn't collide with the segment
> constant; rewrites the online clusterer's centroid math to use an
> unnormalized accumulator (the rev-1 formula could produce NaN on
> antipodal updates); drops `Linkage::Ward` (mathematically invalid with
> cosine distance) and `Affinity::Gaussian` (no defensible default);
> caps the eigengap auto-K and adds an all-zero-affinity precondition
> check for spectral; flips `OverflowStrategy` default to `Reject` and
> changes `AssignClosest` to not update centroids on forced assignments;
> adds a deterministic K-means seed; matches Python's `1e-12` epsilon;
> uses `Option<f32>` for `ClusterAssignment::similarity`; replaces
> `(TimeRange, Embedding)` tuples with a `CollectedEmbedding` struct
> carrying full context; documents the sliding-window-vs-Python-center-crop
> divergence; keeps `Diarizer`'s borrow-by-`&mut` model pattern but
> replaces the misleading "ort-independence" rationale with the real
> reasons (mirror `Segmenter` pattern, allow model reuse). Plus a sibling
> comparison table (§2.1), enriched decision log, edge-case enumeration,
> rejected findings (§14), and an action list (§15).

## 1. Context

`dia::segment` answers "who speaks when, within a window" but only with
window-local speaker slot IDs (slot 0 in window 5 may or may not be the
same person as slot 0 in window 6). For real-world diarization output —
"speaker A spoke at 1.2 s and 8.7 s, speaker B at 3.4 s" — we need
**cross-window speaker linking**, which requires:

1. **A fingerprint per speech segment** — a 256-d vector that's similar
   for the same speaker and dissimilar for different speakers. This is
   `dia::embed` (M2).
2. **Cluster the fingerprints** — group segments by speaker, assign
   global IDs. This is `dia::cluster` (M3).
3. **Wire it together** — feed segment outputs through embedding and
   clustering, emit globally-labeled spans. This is the `Diarizer`
   orchestrator.

The Python project that inspired this work is `findit-speaker-embedding`
(WeSpeaker ResNet34 ONNX, kaldi-compatible fbank, 200-frame fixed input,
256-d L2-normalized output). dia v0.1.0 ports the embedding pipeline
faithfully but improves the long-clip handling and bundles the
clustering layer that Python deferred to a separate component.

User-facing pipeline this spec assumes:

```
audio decoder → resample to 16 kHz → VAD → dia::Diarizer → downstream services
```

Inside `Diarizer`: `Segmenter` → `EmbedModel` → `Clusterer` → emit
`DiarizedSpan`s with global speaker IDs.

## 2. Architecture overview

### 2.1 What we kept and what we changed vs siblings

| Pattern                                        | scenesdetect       | silero                               | soundevents                    | dia::segment (rev 5)                              | dia::embed / cluster / Diarizer (this spec)                                              |
| ---------------------------------------------- | ------------------ | ------------------------------------ | ------------------------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Owns model session?                            | n/a                | yes                                  | yes                            | no (`SegmentModel` is separate)                   | **no** — `EmbedModel` separate; `Diarizer` borrows both `SegmentModel` + `EmbedModel`    |
| State machine and session in one type?         | n/a                | yes                                  | n/a                            | no                                                | **no** — `Segmenter`, `Clusterer`, `Diarizer` are state machines; ort wrappers separate  |
| Output shape                                   | `Option<Timestamp>`| per-frame `f32`                      | `Vec<EventPrediction>`         | `Action` (Layer 1) + `Event` (Layer 2)            | `Embedding` / `EmbeddingResult` / `ClusterAssignment` / `DiarizedSpan`                    |
| Streaming push API                             | per-frame          | `Session::process_stream(callback)`  | none (batch-only)              | `Segmenter::process_samples`                      | `Diarizer::process_samples`; `EmbedModel::embed` is per-clip (no streaming state)        |
| Sans-I/O state-machine separation              | n/a                | partial (push_probability)           | n/a                            | full (`Segmenter` is `ort`-free)                  | `Clusterer` is fully Sans-I/O (pure compute, no audio); `EmbedModel` is the I/O boundary |
| Uses mediatime?                                | yes                | no                                   | no                             | yes                                               | yes (in `DiarizedSpan` and the segment-derived ranges)                                   |
| `Send` (state holder)                          | yes (auto)         | yes (auto)                           | yes (auto)                     | yes (auto, on `Segmenter`)                        | yes (auto, on `Clusterer` and `Diarizer`)                                                |
| `Sync` (state holder)                          | yes (auto)         | n/a                                  | yes (auto)                     | yes (auto, on `Segmenter`)                        | yes (auto, on `Clusterer` and `Diarizer`)                                                |
| `Sync` (ort wrapper)                           | n/a                | **no** (`!Sync`)                     | n/a (Classifier `!Sync`)       | **no** (`SegmentModel: !Sync`)                    | **no** (`EmbedModel: !Sync` — `ort::Session` is `!Sync`)                                 |
| Uses `nalgebra`                                | no                 | no                                   | no                             | no                                                | **yes** (spectral clustering); novel for the suite                                       |
| Generic metadata types                         | n/a                | no                                   | no                             | no                                                | **yes** — `EmbeddingMeta<A, T>` with `()` defaults (per user request)                    |

The novel pieces vs all four siblings: `nalgebra` dependency for spectral
clustering, generic metadata types, and the three-state-machine
orchestration pattern in `Diarizer` (segment + embed + cluster).

### 2.2 Module relationships

```
dia::embed       ←─── borrows nothing from siblings (uses mediatime + kaldi fbank)
   ↓ (Embedding)
dia::cluster     ←─── re-exports Embedding from dia::embed; otherwise self-contained
   ↑
dia::segment     ←─── shipped, unchanged
   ↓ (Segmenter, SpeakerActivity, SegmentModel, TimeRange)
dia::Diarizer    ←─── orchestrates all three above
```

| Module          | Depends on                              | Pulls in `ort` | Pulls in `nalgebra` |
| --------------- | --------------------------------------- | -------------- | ------------------- |
| `dia::segment`  | mediatime                               | yes (gated)    | no                  |
| `dia::embed`    | mediatime, kaldi-native-fbank           | yes (gated)    | no                  |
| `dia::cluster`  | dia::embed (Embedding type)             | no             | yes                 |
| `dia::Diarizer` | all of the above                        | yes (gated)    | yes                 |

### 2.3 Three layers of API

Following `dia::segment`'s precedent of giving callers escape hatches:

- **High-level (most users):** `dia::Diarizer::process_samples(samples, |span| ...)`. Push audio in, get `DiarizedSpan`s with global speaker IDs out.
- **Mid-level (custom orchestration):** Compose `Segmenter` + `EmbedModel` + `Clusterer` yourself. ~30 lines of glue; needed when you want a custom audio-buffer strategy, custom span filtering, or to feed in pre-segmented activities.
- **Low-level (custom inference):** `compute_fbank` + `EmbedModel::embed_features` + `cluster_offline`. Pure functions and ONNX hooks; needed when you want to batch inference across thousands of clips, apply non-default aggregation, or run offline-only on saved embeddings.

## 3. Scope for v0.1.0 phase 2

### In scope

**`dia::embed`:**
- `EmbedModel` ort wrapper for WeSpeaker ResNet34
- `EmbedModelOptions` mirroring `SegmentModelOptions`
- Pure-Rust `kaldi-native-fbank` for feature extraction
- Variable-length clip handling: zero-pad < 2 s, **sliding-window mean for > 2 s**
  (this is a deliberate IMPROVEMENT over Python's center-crop — see §5.1)
- Voice-weighted variant accepting caller-provided per-sample voice probabilities
- Two-tier API: low-level (raw `embed_features` / `embed_features_batch`) + high-level (`embed` / `embed_with_meta` / `embed_weighted` / `embed_weighted_with_meta`)
- `Embedding` newtype with `.similarity()` and `.normalize_from(raw)` methods
- Generic `EmbeddingMeta<A, T>` with `()` defaults
- `EmbeddingResult` carries observability fields (`windows_used`, `total_weight`, `source_duration`)

**`dia::cluster`:**
- Online streaming `Clusterer` (threshold + EMA centroid update via unnormalized accumulator)
- Offline `cluster_offline` with two methods:
  - **Spectral** (default): cosine affinity → normalized Laplacian → eigendecomposition → seeded K-means
  - **Agglomerative**: Single / Complete / Average linkage (Ward removed — invalid with cosine)
- Auto-K via eigengap (capped) or explicit `target_speakers`
- Deterministic K-means seed (`OfflineClusterOptions::seed`)
- All-zero-affinity precondition check
- Re-uses `dia::embed::Embedding`

**`dia::Diarizer`:**
- Builder API: `Diarizer::builder()` with `with_*` setters only (no `options()` setter)
- Internal audio buffer with bounded retention (`EMBED_WINDOW_SAMPLES` worth — see §5.7)
- `process_samples(seg_model, embed_model, samples, emit)` — synchronous: push → segment → embed → cluster → emit
- `finish_stream(seg_model, embed_model, emit)` — flush
- `clear()` — reset for new session
- `collected_embeddings()` — accessor returning `&[CollectedEmbedding]` with full context
- `pending_inferences()`, `buffered_samples()`, `num_speakers()`, `speakers()` — introspection
- Auto-derived `Send + Sync`

### Deferred — explicitly out of v0.1.0

| Item | Reason |
|---|---|
| Bundled WeSpeaker model | ~25 MB; matches soundevents posture (download script) |
| F1 parity tests vs `findit-speaker-embedding` Python output | Need Python infra; smoke tests cover basics |
| Threading / async service layer | Caller wraps with their own runtime |
| Per-frame voice prob from `dia::segment` integrated path | Segment-side API addition; v0.2 |
| `try_push_samples` soft-cap backpressure | Introspection covers it |
| Voice-weighted with internally-computed VAD | Forces a model dep |
| Median / top-K-window aggregation | No theoretical justification stronger than mean |
| Configurable hop | Hardcoded 1 s = 50 % overlap (Resemblyzer/pyannote default) |
| Batched high-level `embed_batch(&[&[f32]])` | `embed_features_batch` covers inner-loop batching |
| Speaker re-identification across sessions | Persistent global IDs are a v0.2 feature |
| `Affinity::Gaussian` | No defensible default σ; `Cosine` works without tuning |
| `Linkage::Ward` | Mathematically invalid with cosine distance |
| EMA(α) sensitivity analysis | The `Ema(0.2)` default is **tentative**; v0.2 may change after empirical tuning |
| Async out-of-order `push_inference` for embed | Diarizer assumes synchronous segment inference (§5.7) |

### Out of scope (handled elsewhere)

- VAD — caller's responsibility (silero or similar)
- Audio decoding / resampling — caller delivers 16 kHz mono float32 PCM
- Threading — caller wraps with their own runtime

## 4. Public API surface

### 4.1 Crate-level constants and re-exports

```rust
// Already exposed by dia::segment; canonical home stays in dia::segment.
// Both modules re-export to avoid forcing callers to import from segment.
pub const SAMPLE_RATE_HZ: u32 = 16_000;
```

### 4.2 `dia::embed`

#### Constants

```rust
/// 2 s @ 16 kHz; the WeSpeaker model's fixed input.
///
/// Named with the `EMBED_` prefix to avoid collision with
/// `dia::segment::WINDOW_SAMPLES` (which is 160_000 = 10 s at the same
/// rate). Same crate, different module — same short name would be a
/// footgun.
pub const EMBED_WINDOW_SAMPLES: u32 = 32_000;

/// 1 s, 50 % overlap (hardcoded).
pub const HOP_SAMPLES: u32 = 16_000;

/// ~25 ms; one kaldi window. Below this, `embed` returns `Error::InvalidClip`.
pub const MIN_CLIP_SAMPLES: u32 = 400;

pub const FBANK_NUM_MELS: usize = 80;
pub const FBANK_FRAMES: usize = 200;
pub const EMBEDDING_DIM: usize = 256;

/// Numerical floor used in L2-normalization to avoid divide-by-zero.
/// Matches `findit-speaker-embedding`'s `1e-12` (verified at
/// `embedder.py:85`); diverging would lose Python parity in edge cases.
pub const NORM_EPSILON: f32 = 1e-12;
```

#### Types

```rust
/// A 256-d L2-normalized speaker embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Embedding(pub(crate) [f32; EMBEDDING_DIM]);

impl Embedding {
    pub const fn as_array(&self) -> &[f32; EMBEDDING_DIM];
    pub fn as_slice(&self) -> &[f32];

    /// Cosine similarity. Both inputs are L2-normalized, so this reduces
    /// to a dot product. Returns a value in `[-1.0, 1.0]`.
    pub fn similarity(&self, other: &Embedding) -> f32;

    /// L2-normalize a raw 256-d inference output and wrap it.
    /// Returns `None` if `||raw|| < NORM_EPSILON` (degenerate input).
    /// Use after `EmbedModel::embed_features_batch` + custom aggregation.
    pub fn normalize_from(raw: [f32; EMBEDDING_DIM]) -> Option<Self>;
}

pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32;

/// Optional metadata that flows through `embed_with_meta` /
/// `embed_weighted_with_meta` to `EmbeddingResult`. Generic over the
/// `audio_id` and `track_id` types — callers use whatever string-like
/// type fits their domain. Defaults to `()` so the unit-typed metadata
/// path allocates nothing.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingMeta<A = (), T = ()> {
    audio_id: A,
    track_id: T,
    correlation_id: Option<u64>,
}

impl<A, T> EmbeddingMeta<A, T> {
    pub fn new(audio_id: A, track_id: T) -> Self;
    pub fn with_correlation_id(self, id: u64) -> Self;
    pub fn audio_id(&self) -> &A;
    pub fn track_id(&self) -> &T;
    pub fn correlation_id(&self) -> Option<u64>;
}

#[derive(Debug, Clone)]
pub struct EmbeddingResult<A = (), T = ()> {
    embedding: Embedding,
    /// Actual length of the source clip (NOT the padded/cropped 2 s).
    source_duration: Duration,
    /// Number of 2 s windows averaged. 1 for clips ≤ 2 s.
    windows_used: u32,
    /// Sum of per-window weights. Equals `windows_used as f32` for the
    /// equal-weighted path; lower for `embed_weighted` calls where some
    /// windows had low voice probability.
    total_weight: f32,
    audio_id: A,
    track_id: T,
    correlation_id: Option<u64>,
}

impl<A, T> EmbeddingResult<A, T> {
    pub fn embedding(&self) -> &Embedding;
    pub fn source_duration(&self) -> Duration;
    pub fn windows_used(&self) -> u32;
    pub fn total_weight(&self) -> f32;
    pub fn audio_id(&self) -> &A;
    pub fn track_id(&self) -> &T;
    pub fn correlation_id(&self) -> Option<u64>;
}
```

#### Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("clip too short: {len} samples (need at least {min})")]
    InvalidClip { len: usize, min: usize },

    #[error("voice_probs.len() = {weights_len} must equal samples.len() = {samples_len}")]
    WeightShapeMismatch { samples_len: usize, weights_len: usize },

    /// All windows had near-zero voice-probability weight; the weighted
    /// average is undefined. Almost always caller error.
    #[error("all windows had effectively zero voice-activity weight")]
    AllSilent,

    /// Input contains NaN or non-finite values.
    #[error("input contains non-finite values (NaN or infinity)")]
    NonFiniteInput,

    #[error("inference scores length {got}, expected {expected}")]
    InferenceShapeMismatch { expected: usize, got: usize },

    #[cfg(feature = "ort")]
    #[error("model {tensor} dims {got:?}, expected {expected:?}")]
    IncompatibleModel { tensor: &'static str, expected: &'static [i64], got: Vec<i64> },

    #[cfg(feature = "ort")]
    #[error("failed to load model from {path}: {source}", path = path.display())]
    LoadModel { path: PathBuf, #[source] source: ort::Error },

    #[cfg(feature = "ort")]
    #[error(transparent)]
    Ort(#[from] ort::Error),
}
```

#### Pure functions

```rust
/// Compute the kaldi-compatible fbank for a clip and pad/center-crop to
/// exactly `[FBANK_FRAMES, FBANK_NUM_MELS] = [200, 80]`.
///
/// Returns `Error::InvalidClip` if `samples.len() < MIN_CLIP_SAMPLES`.
/// Returns `Error::NonFiniteInput` if any sample is NaN/infinity.
pub fn compute_fbank(
    samples: &[f32],
) -> Result<[[f32; FBANK_NUM_MELS]; FBANK_FRAMES], Error>;
```

#### `EmbedModel` (cfg ort)

```rust
#[cfg(feature = "ort")]
pub struct EmbedModelOptions { /* mirrors SegmentModelOptions */ }

#[cfg(feature = "ort")]
pub struct EmbedModel { /* private; owns ort::Session + scratch */ }

#[cfg(feature = "ort")]
impl EmbedModel {
    pub fn from_file / from_file_with_options / from_memory / from_memory_with_options;

    // ── Low-level: pre-computed features ──────────
    /// Run ONNX inference on one set of [200, 80] fbank features.
    /// Returns the **raw, un-normalized** 256-d output.
    /// Wrap with `Embedding::normalize_from(raw)` to L2-normalize.
    pub fn embed_features(
        &mut self,
        features: &[[f32; FBANK_NUM_MELS]; FBANK_FRAMES],
    ) -> Result<[f32; EMBEDDING_DIM], Error>;

    /// Batched feature inference. Single ONNX call with batch size N.
    /// Returns raw outputs.
    pub fn embed_features_batch(
        &mut self,
        features: &[[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error>;

    // ── High-level ────────────────────────────────
    pub fn embed(&mut self, samples: &[f32]) -> Result<EmbeddingResult, Error>;
    pub fn embed_with_meta<A, T>(
        &mut self, samples: &[f32], meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error>;

    pub fn embed_weighted(
        &mut self, samples: &[f32], voice_probs: &[f32],
    ) -> Result<EmbeddingResult, Error>;
    pub fn embed_weighted_with_meta<A, T>(
        &mut self, samples: &[f32], voice_probs: &[f32], meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error>;
}
```

`EmbedModel` auto-derives `Send`; does NOT auto-derive `Sync` because
`ort::Session` is `!Sync`. Matches `silero::Session` and
`dia::segment::SegmentModel`.

### 4.3 `dia::cluster`

#### Constants

```rust
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;
/// Tentative — see decision log §12 entry "EMA(0.2) tentative".
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;
/// Max K when auto-detecting via eigengap (no `target_speakers`).
pub const MAX_AUTO_SPEAKERS: u32 = 15;
```

#### Types

```rust
pub use crate::embed::Embedding;

/// Public view of a speaker centroid. The internal accumulator is
/// hidden — see §5.4 for the algorithmic reasoning.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerCentroid {
    speaker_id: u64,
    centroid: Embedding,           // L2-normalized; lazily derived from accumulator
    assignment_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ClusterAssignment {
    pub speaker_id: u64,
    pub is_new_speaker: bool,
    /// Cosine similarity to the assigned centroid, computed pre-update.
    /// `None` for the first-ever assignment (no centroids existed to
    /// compare against). For new speakers beyond the first, this is
    /// the maximum similarity to any existing centroid.
    pub similarity: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ClusterOptions {
    similarity_threshold: f32,        // DEFAULT_SIMILARITY_THRESHOLD
    update_strategy: UpdateStrategy,  // Ema(DEFAULT_EMA_ALPHA)
    max_speakers: Option<u32>,        // None
    overflow_strategy: OverflowStrategy, // Reject
}

#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    /// Centroid = L2-normalize(unnormalized_sum). Maintains an internal
    /// running sum; centroid is derived on demand. Stable; immune to
    /// drift on long sessions.
    RollingMean,
    /// Centroid update via EMA on the unnormalized accumulator:
    /// `accumulator = (1-α) · accumulator + α · new_embedding`.
    /// The exposed centroid is `L2-normalize(accumulator)` on demand.
    /// Adapts to drift; recommended for streaming.
    Ema(f32),
}

#[derive(Debug, Clone, Copy)]
pub enum OverflowStrategy {
    /// Force-assign to the closest existing speaker. The centroid is
    /// **NOT updated** for this assignment (the new embedding doesn't
    /// match the speaker well; updating would corrupt the centroid).
    /// `assignment_count` increments. `is_new_speaker = false`.
    AssignClosest,
    /// **DEFAULT.** Reject with `Error::TooManySpeakers`. Caller decides.
    Reject,
}

impl Default for ClusterOptions { /* threshold 0.5, Ema(0.2), no cap, Reject */ }
impl ClusterOptions {
    pub fn new() -> Self;
    pub fn with_similarity_threshold(self, t: f32) -> Self;
    pub fn with_update_strategy(self, s: UpdateStrategy) -> Self;
    pub fn with_max_speakers(self, n: u32) -> Self;
    pub fn with_overflow_strategy(self, s: OverflowStrategy) -> Self;
}
```

#### Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("speaker cap reached ({cap}) and overflow_strategy = Reject")]
    TooManySpeakers { cap: u32 },

    #[error("input embeddings list is empty")]
    EmptyInput,

    #[error("target_speakers ({target}) > input embeddings count ({n})")]
    TargetExceedsInput { target: u32, n: usize },

    #[error("target_speakers must be >= 1")]
    TargetTooSmall,

    #[error("input contains NaN or non-finite values")]
    NonFiniteInput,

    /// All pairwise similarities ≤ 0 → spectral clustering's affinity
    /// matrix is all-zero, Laplacian undefined.
    #[error("all pairwise similarities are non-positive; spectral clustering undefined")]
    AllDissimilar,

    #[error("eigendecomposition failed (matrix likely singular or pathological)")]
    EigendecompositionFailed,
}
```

#### Online streaming `Clusterer`

```rust
pub struct Clusterer { /* private; speakers, options */ }

impl Clusterer {
    pub fn new(opts: ClusterOptions) -> Self;
    pub fn options(&self) -> &ClusterOptions;

    /// Submit one embedding; returns the global speaker assignment.
    /// Updates the assigned speaker's accumulator via the configured strategy.
    pub fn submit(&mut self, embedding: &Embedding) -> Result<ClusterAssignment, Error>;

    pub fn speakers(&self) -> Vec<SpeakerCentroid>;
    pub fn speaker(&self, id: u64) -> Option<SpeakerCentroid>;
    pub fn num_speakers(&self) -> usize;

    pub fn clear(&mut self);
}
```

#### Offline batch clustering

```rust
#[derive(Debug, Clone)]
pub struct OfflineClusterOptions {
    method: OfflineMethod,            // Spectral (default)
    similarity_threshold: f32,        // DEFAULT_SIMILARITY_THRESHOLD
    target_speakers: Option<u32>,     // None (auto-detect)
    /// K-means seed for spectral clustering. `None` = derive
    /// deterministically from a hash of the input embeddings; `Some(s)`
    /// = use `s` directly. Default: `None` (deterministic by input).
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub enum OfflineMethod {
    /// Hierarchical agglomerative clustering. Faster for small N.
    Agglomerative { linkage: Linkage },
    /// Spectral clustering. Best quality. **Default.**
    Spectral,  // affinity is fixed at Cosine; Gaussian removed in rev 2
}

#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    Single,
    Complete,
    /// Recommended for speaker clustering. **Default.**
    Average,
    // Ward removed in rev 2: invalid with cosine distance.
}

impl Default for OfflineClusterOptions { /* method=Spectral, threshold=0.5, target=None, seed=None */ }
impl OfflineClusterOptions {
    pub fn new() -> Self;
    pub fn with_method(self, m: OfflineMethod) -> Self;
    pub fn with_similarity_threshold(self, t: f32) -> Self;
    pub fn with_target_speakers(self, n: u32) -> Self;
    pub fn with_seed(self, s: u64) -> Self;
}

/// Cluster a batch of embeddings; returns one global speaker id per
/// input, parallel to the input slice.
pub fn cluster_offline(
    embeddings: &[Embedding],
    opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error>;
```

### 4.4 `dia::Diarizer`

#### Types

```rust
pub struct DiarizerOptions {
    segment: SegmentOptions,
    cluster: ClusterOptions,
    collect_embeddings: bool,         // default true
}
impl Default for DiarizerOptions { /* sensible defaults */ }
impl DiarizerOptions {
    pub fn new() -> Self;
    pub fn with_segment_options(self, opts: SegmentOptions) -> Self;
    pub fn with_cluster_options(self, opts: ClusterOptions) -> Self;
    pub fn with_collect_embeddings(self, on: bool) -> Self;
}

/// Per-activity context retained during a diarization session.
/// Returned by `Diarizer::collected_embeddings()`. Carries everything
/// needed to correlate the offline-clustering re-labeling back to its
/// source activity.
#[derive(Debug, Clone)]
pub struct CollectedEmbedding {
    pub range: TimeRange,
    pub embedding: Embedding,
    /// Online speaker id assigned by `Clusterer::submit` during streaming.
    pub online_speaker_id: u64,
    /// Window-local slot from `dia::segment::SpeakerActivity`.
    pub speaker_slot: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct DiarizedSpan {
    range: TimeRange,
    speaker_id: u64,
    /// Cosine similarity to the assigned speaker's centroid.
    /// `None` for the first-ever assignment (no prior speakers).
    similarity: Option<f32>,
    is_new_speaker: bool,
    /// Window-local slot from segment (preserved for debugging).
    speaker_slot: u8,
}

impl DiarizedSpan {
    pub fn range(&self) -> TimeRange;
    pub fn speaker_id(&self) -> u64;
    pub fn similarity(&self) -> Option<f32>;
    pub fn is_new_speaker(&self) -> bool;
    pub fn speaker_slot(&self) -> u8;
}

pub struct Diarizer { /* private */ }
pub struct DiarizerBuilder { /* private */ }

impl Diarizer {
    pub fn builder() -> DiarizerBuilder;
    pub fn options(&self) -> &DiarizerOptions;

    #[cfg(feature = "ort")]
    pub fn process_samples<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        embed_model: &mut EmbedModel,
        samples: &[f32],
        emit: F,
    ) -> Result<(), Error>;

    #[cfg(feature = "ort")]
    pub fn finish_stream<F: FnMut(DiarizedSpan)>(
        &mut self,
        seg_model: &mut SegmentModel,
        embed_model: &mut EmbedModel,
        emit: F,
    ) -> Result<(), Error>;

    pub fn clear(&mut self);

    pub fn collected_embeddings(&self) -> &[CollectedEmbedding];
    pub fn clear_collected(&mut self);

    pub fn pending_inferences(&self) -> usize;
    pub fn buffered_samples(&self) -> usize;
    pub fn num_speakers(&self) -> usize;
    pub fn speakers(&self) -> Vec<SpeakerCentroid>;
}

impl DiarizerBuilder {
    pub fn new() -> Self;
    pub fn with_segment_options(self, opts: SegmentOptions) -> Self;
    pub fn with_cluster_options(self, opts: ClusterOptions) -> Self;
    pub fn with_collect_embeddings(self, on: bool) -> Self;
    pub fn build(self) -> Diarizer;
}
```

The builder exposes only `with_*` setters (per-field) — there's no
`options(opts)` setter that takes a whole `DiarizerOptions`, because
`DiarizerOptions` itself uses the `with_*` builder pattern (so callers
who want a pre-built options struct can do
`diarizer_options.into_diarizer()` → not provided either; they pass the
fields one at a time on the builder). This avoids the rev-1 API
inconsistency where `options(opts)` overlapped with `segment_options(o)`.

#### Diarizer error type

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Segment(#[from] crate::segment::Error),
    #[error(transparent)]
    Embed(#[from] crate::embed::Error),
    #[error(transparent)]
    Cluster(#[from] crate::cluster::Error),
}
```

Each source module keeps its own error; the `Diarizer` wraps via `#[from]`
so callers can use `?` cleanly. Multi-error rationale: each module is
independently usable (and shippable to crates.io as a sub-crate someday),
so each owns its own error type. The Diarizer's wrapper is the integration
point.

## 5. Algorithm semantics (load-bearing decisions)

### 5.1 Sliding-window mean for variable-length clips

> **Important divergence from `findit-speaker-embedding`:** Python
> center-crops clips longer than 2 s. dia uses sliding-window mean.
> **Embeddings produced by dia::embed are NOT directly comparable with
> Python output for clips > 2 s.** Cosine similarity between
> dia-produced and Python-produced embeddings of the *same* clip will
> typically land between 0.92 and 0.99 (still "same speaker" but
> noticeably lower than identity). Document this clearly in the README
> so users migrating from Python don't get surprised.

Algorithm:

```
if len(samples) < MIN_CLIP_SAMPLES:
    return Error::InvalidClip
elif len(samples) <= EMBED_WINDOW_SAMPLES:
    # one window with zero padding
    padded = samples ++ [0.0; EMBED_WINDOW_SAMPLES - len(samples)]
    features = compute_fbank(padded)
    raw = onnx(features)
    return l2_normalize(raw, eps=NORM_EPSILON)
else:
    # multiple overlapping windows, equal-weighted mean
    starts = [k * HOP_SAMPLES for k in 0..K-1] + [len(samples) - EMBED_WINDOW_SAMPLES]
    starts = dedup_and_sort(starts)
    sum = [0.0; 256]
    for s in starts:
        chunk = samples[s..s + EMBED_WINDOW_SAMPLES]
        features = compute_fbank(chunk)
        sum += onnx(features)
    return l2_normalize(sum, eps=NORM_EPSILON)
```

The `+ [len - EMBED_WINDOW_SAMPLES]` and dedup ensures the last window
aligns with end-of-clip.

### 5.2 Voice-weighted variant

Same algorithm as §5.1, with two changes:
- For each window: `weight = mean(voice_probs[s..s + EMBED_WINDOW_SAMPLES])`
- `sum += weight * raw`, `total_weight += weight`
- `if total_weight < NORM_EPSILON: return Error::AllSilent`
- Result: `l2_normalize(sum, eps=NORM_EPSILON)` (note: the division by
  total_weight is a scalar that cancels out under L2-normalization, so
  we don't actually divide — just normalize the weighted sum)

### 5.3 Single-window weighting collapse

For clips ≤ 2 s, only one window runs. `embed_weighted` and `embed`
produce identical results — a scalar weight on a single vector cancels
under L2-normalization. Documented in `embed_weighted` rustdoc.

### 5.4 Online clustering: centroid algebra (REWRITTEN in rev 2)

The rev-1 spec stored an L2-normalized centroid and proposed updating it
via arithmetic mean then re-normalizing. That math is unsound on
antipodal updates: `[1, 0] + [-1, 0] = [0, 0]`, `||0|| = 0`, divide by
zero. Replaced in rev 2 with an **unnormalized accumulator** approach.

**Internal state per speaker** (private, not exposed):

```rust
struct SpeakerEntry {
    speaker_id: u64,
    /// Unnormalized running accumulator. Centroid = L2-normalize(this).
    accumulator: [f32; EMBEDDING_DIM],
    assignment_count: u32,
}
```

**Public-facing `SpeakerCentroid`** carries the L2-normalized centroid
derived on demand from the internal accumulator. Callers see clean
unit vectors; the accumulator is hidden.

**Update strategies:**

```
RollingMean (on assignment of new embedding e):
    accumulator += e            // pure addition; e is L2-normalized
    assignment_count += 1
    centroid_for_query = l2_normalize(accumulator, NORM_EPSILON)

Ema(α) (on assignment of new embedding e):
    accumulator = (1 - α) * accumulator + α * e
    assignment_count += 1
    centroid_for_query = l2_normalize(accumulator, NORM_EPSILON)
```

If `||accumulator|| < NORM_EPSILON` after update, the centroid is
degenerate; the speaker entry retains the previous centroid (last good
value cached). This handles the antipodal case gracefully.

**Submit algorithm:**

```
for each submitted embedding e (L2-normalized):
    if speakers.is_empty():
        new_id = 0
        speakers.push(SpeakerEntry {
            speaker_id: 0,
            accumulator: e,
            assignment_count: 1,
        })
        return ClusterAssignment {
            speaker_id: 0,
            is_new_speaker: true,
            similarity: None,        // no prior speakers; sentinel-free
        }

    sims = [centroid_of(s).similarity(e) for s in speakers]
    (best_idx, best_sim) = argmax_lowest_index(sims)   // tie-breaking: lowest index wins

    if best_sim >= threshold:
        speakers[best_idx].update(e, strategy)
        return ClusterAssignment {
            speaker_id: speakers[best_idx].speaker_id,
            is_new_speaker: false,
            similarity: Some(best_sim),
        }
    else if max_speakers.is_some_and(|cap| speakers.len() >= cap):
        match overflow_strategy:
            AssignClosest:
                # NO centroid update — assignment is low-confidence;
                # updating would pull the centroid toward an outlier.
                speakers[best_idx].assignment_count += 1
                return ClusterAssignment {
                    speaker_id: speakers[best_idx].speaker_id,
                    is_new_speaker: false,
                    similarity: Some(best_sim),
                }
            Reject:
                return Error::TooManySpeakers { cap }
    else:
        new_id = speakers.last().unwrap().speaker_id + 1
        speakers.push(SpeakerEntry {
            speaker_id: new_id,
            accumulator: e,
            assignment_count: 1,
        })
        return ClusterAssignment {
            speaker_id: new_id,
            is_new_speaker: true,
            similarity: Some(best_sim),
        }
```

**Tie-breaking rule:** when multiple centroids have identical similarity
to the query embedding, the lowest-index speaker (earliest assigned)
wins. Documented for reproducibility.

### 5.5 Offline spectral clustering

```
1. Build affinity matrix A ∈ R^{N×N}:
     A_ij = max(0, e_i · e_j)    // ReLU of cosine similarity
     A_ii = 0

2. Precondition check:
     if A.iter().all(|x| x < NORM_EPSILON):
         return Error::AllDissimilar

3. Compute degree matrix D where D_ii = sum_j A_ij.
4. Compute normalized graph Laplacian:
     L_sym = I - D^{-1/2} A D^{-1/2}

5. Eigendecompose L_sym; let λ = sorted eigenvalues, U = corresponding eigenvectors.

6. Determine K:
     if let Some(k) = target_speakers:
         if k < 1: return Error::TargetTooSmall
         if (k as usize) > N: return Error::TargetExceedsInput { target: k, n: N }
         K = k
     else:
         # Eigengap heuristic with cap.
         K_max = min(N - 1, MAX_AUTO_SPEAKERS)
         K = argmax(λ[k+1] - λ[k] for k in 1..=K_max)
         K = max(K, 1)

7. Take U[:, 0..K] (smallest-K eigenvectors).
8. Row-normalize U.
9. K-means on rows of U:
     - K-means++ seeding with deterministic seed:
         if let Some(s) = seed: use s
         else: derive from FxHash of input embeddings' bytes
     - 100 iterations or until convergence.
10. Return cluster assignments.
```

### 5.5.1 Edge cases (NEW in rev 2)

For both `cluster_offline` and `Clusterer::submit`:

| Edge case | Behavior |
|---|---|
| `embeddings.is_empty()` (offline) | `Error::EmptyInput` |
| `target_speakers = Some(0)` | `Error::TargetTooSmall` |
| `target_speakers > N` (offline) | `Error::TargetExceedsInput` |
| `N == 1` (offline) | Returns `vec![0]` immediately; no clustering |
| `N == 2` (offline, spectral) | Returns `vec![0, 1]` if similarity < threshold else `vec![0, 0]`; eigengap is degenerate at N=2 so we skip directly to the threshold check |
| All embeddings identical (offline) | One cluster: `vec![0; N]` |
| Affinity matrix all-zero (spectral) | `Error::AllDissimilar` |
| NaN in any embedding | `Error::NonFiniteInput` (validated at entry) |
| All-zero embedding (norm = 0) | `Error::NonFiniteInput` (we treat zero-norm as invalid since it can't be a real fingerprint) |
| Eigendecomposition NaN-out | `Error::EigendecompositionFailed` |

Tests in §9 cover each row.

### 5.6 Offline agglomerative clustering

Algorithm (HAC with Average linkage by default):

```
1. Validate input (Error::EmptyInput if empty, NonFiniteInput if any NaN/zero).
2. Compute pairwise distance matrix D ∈ R^{N×N}:
     D_ij = 1 - cos_similarity(e_i, e_j)   // 0 = identical, 2 = opposite
3. Initialize: each embedding is its own cluster.
4. Repeat:
     - Find the two most similar non-merged clusters (smallest D_ij).
     - if their distance ≥ (1 - threshold) AND target_speakers = None: stop.
     - if cluster count == target_speakers: stop.
     - Merge them.
     - Update distances per linkage:
         Single: D[merged, k] = min(D[a, k], D[b, k])
         Complete: D[merged, k] = max(D[a, k], D[b, k])
         Average: weighted by cluster sizes (Lance-Williams formula)
         (Ward removed: invalid with cosine distance.)
5. Return labels.
```

### 5.7 Diarizer audio buffer policy

The Diarizer maintains a `VecDeque<f32>` indexed by absolute sample
position. **This policy assumes synchronous segment inference, as
performed inside `Diarizer::process_samples`.** Async out-of-order
`push_inference` is not supported on the Diarizer path; users needing
async must drop down to mid-level composition (§2.3) and manage
buffer retention manually.

After every `process_samples` returns:

- Drop samples below `total_samples_pushed - EMBED_WINDOW_SAMPLES`.
- Why this bound: the latest emitted activity is in the most recent
  segment window (covered by the synchronous segment+embed pipeline);
  the earliest sample we still need is the start of the most-recent
  embed window (= 2 s back from now).

Steady-state memory: `EMBED_WINDOW_SAMPLES * 4 bytes = 128 KB`.
(Note: the rev-1 spec said 640 KB based on `WINDOW_SAMPLES = 160_000`,
which was the segment constant. With `EMBED_WINDOW_SAMPLES = 32_000`
the bound is much smaller. Verified by re-derivation: the segment
tail-anchor reach-back is irrelevant because by the time
`process_samples` returns, the segment has already produced any tail
activity inline.)

For a `SpeakerActivity` with range `[s0, s1)`, the slice
`audio_buffer[s0 - audio_base..s1 - audio_base]` is fed to
`embed_model.embed(...)`. **Defensive bounds check**: if the slice
underflows (e.g., a pathological out-of-order activity from a buggy
caller), return `Error::Embed(Error::InvalidClip { len: 0, min:
MIN_CLIP_SAMPLES as usize })` rather than panicking.

### 5.8 Diarizer error handling policy

When `embed_model.embed(slice)` returns `Err` for a particular activity:
- The error is **propagated** via `process_samples`'s `Result<(), Error>` return.
- Activities NOT YET PROCESSED in the same call are lost.
- The Diarizer's internal state is left consistent (audio buffer
  trimmed up to the failed activity; `Clusterer` and
  `collected_embeddings` not updated for the failed activity).
- Caller can call `process_samples` again with the next chunk; the
  Diarizer continues from where it left off.

Same policy for `cluster.submit` errors.

## 6. Module layout

```
src/
├── lib.rs                                 # crate-level constants, re-exports, Diarizer
├── diarizer.rs                            # Diarizer + builder + DiarizedSpan + CollectedEmbedding
├── segment/                               # SHIPPED (unchanged)
├── embed/
│   ├── mod.rs                             # pub re-exports
│   ├── types.rs                           # Embedding, EmbeddingMeta, EmbeddingResult
│   ├── options.rs                         # constants, EmbedModelOptions
│   ├── error.rs
│   ├── fbank.rs                           # compute_fbank wrapper
│   ├── embedder.rs                        # sliding-window logic
│   └── model.rs                           # cfg(ort) — EmbedModel
└── cluster/
    ├── mod.rs                             # pub re-exports
    ├── options.rs                         # ClusterOptions, OfflineClusterOptions
    ├── error.rs
    ├── online.rs                          # Clusterer
    ├── agglomerative.rs                   # offline HAC
    └── spectral.rs                        # offline spectral (uses nalgebra)
```

## 7. Crate metadata

```toml
[dependencies]
# Existing
mediatime = "0.1"
thiserror = "2"
ort = { version = "2.0.0-rc.12", optional = true }

# New in v0.1.0 phase 2
kaldi-native-fbank = "0.1"
nalgebra = "0.33"
```

Both new deps are unconditional. No new feature flags. The `cluster-spectral`
feature flag mentioned during brainstorming is not introduced — `nalgebra`
is needed for the default `OfflineMethod::Spectral`, so gating it would
break the default path.

## 8. Streaming flow examples

(Same as rev 1 §8, with renamed `WINDOW_SAMPLES` → `EMBED_WINDOW_SAMPLES`
and `(TimeRange, Embedding)` tuples replaced with `CollectedEmbedding`
struct accesses. Specifically the offline-refinement snippet is now:)

```rust
let collected = diar.collected_embeddings();
let embeds: Vec<Embedding> = collected.iter().map(|c| c.embedding).collect();
let refined: Vec<u64> = dia::cluster::cluster_offline(
    &embeds,
    &OfflineClusterOptions::default(),
)?;
// `refined[i]` is the offline label for `collected[i]`. Reconcile with
// `collected[i].online_speaker_id` if needed (e.g., majority-vote map).
```

## 9. Testing strategy

### Unit tests

**`dia::embed`:** as in rev 1, plus:
- NaN-input rejection: `compute_fbank(&[f32::NAN; 32_000])` → `Error::NonFiniteInput`.
- Antipodal embedding handling: edge case in `Embedding::normalize_from`.

**`dia::cluster`:**
- All §5.5.1 edge cases.
- Antipodal centroid update: `Clusterer::submit([1,0,...])` then `submit([-1,0,...])` doesn't panic; second submission becomes new speaker (not assigned).
- `OverflowStrategy::AssignClosest` does NOT change centroid (only `assignment_count`).
- `OverflowStrategy::Reject` returns `Error::TooManySpeakers`.
- Tie-breaking: two centroids equidistant from query → lowest speaker_id wins.
- K-means seed reproducibility: same seed → same labels.
- K-means seed default: same input twice with `seed = None` → same labels.
- `cluster_offline` agglomerative-vs-spectral on a hand-built 6-embedding 2-cluster set produces the same labels.
- Eigengap K cap: synthetic `MAX_AUTO_SPEAKERS + 5` clusters → returns `MAX_AUTO_SPEAKERS`.

**`dia::Diarizer`:**
- All from rev 1.
- Buffer underflow on pathological activity range → `Error::Embed(Error::InvalidClip)` (no panic).
- `num_speakers()` and `speakers()` reflect Clusterer state mid-stream.

### Integration tests (gated)

As in rev 1.

### Compile-time trait assertions

```rust
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<dia::cluster::Clusterer>();
    assert_send_sync::<dia::Diarizer>();

    #[cfg(feature = "ort")]
    fn assert_send<T: Send>() {}
    #[cfg(feature = "ort")]
    assert_send::<dia::embed::EmbedModel>();
};
```

## 10. Threading and lifecycle

- **`EmbedModel`**: auto-derives `Send` (from `ort::Session`); does NOT
  auto-derive `Sync`. Use one per worker. Same posture as
  `dia::segment::SegmentModel` and `silero::Session`.
- **`Clusterer`**: auto-derives `Send + Sync`. State machine behind `&mut`.
- **`Diarizer`**: auto-derives `Send + Sync`. ort sessions are passed by
  `&mut` per call rather than owned.
- No internal locks. No async. Callers wrap with their own runtime.
- `Diarizer::clear()` advances the underlying `Segmenter`'s generation
  counter (via `Segmenter::clear`); the `Clusterer`'s speaker IDs reset
  to 0. Does not discard or reload the ONNX sessions — those are
  external.
- **Determinism**: same caveats as `dia::segment` — set
  `intra_op_num_threads = 1` on both `SegmentModelOptions` and
  `EmbedModelOptions` for bit-exact reproducibility. K-means in spectral
  clustering uses a deterministic seed (default: derived from input).

## 11. Resolved contracts

(Mostly as rev 1; key updates:)

### 11.0 Why Diarizer borrows `&mut SegmentModel` and `&mut EmbedModel` (UPDATED)

The rev-1 spec claimed "ort-independence — users can construct a
Diarizer for offline replay without `ort`." That was a phantom feature:
without `ort`, there's no constructor for fresh embeddings, and the
spec doesn't ship serde for saved embeddings. Honest reasons to keep
the borrow pattern:

1. **Mirror `dia::segment::Segmenter`**: that state machine also
   borrows `SegmentModel` per call. Following the same pattern in
   `Diarizer` keeps the suite consistent.
2. **Model reuse across sequential sessions in one thread**: a server
   processing many short audio files can construct one `SegmentModel`
   and one `EmbedModel`, then construct/clear/reuse a `Diarizer` per
   file. With owned models the `Diarizer` would need to be reconstructed
   each session (or we'd have to add `with_models` getters). Borrowing
   is cleaner.
3. **Construction is independent of ort feature gating**: the
   `Diarizer` itself has no `cfg(feature = "ort")` on its struct
   definition; only `process_samples` and `finish_stream` are gated.
   This is mostly aesthetic but does mean `Diarizer::new`, `clear`,
   `collected_embeddings` etc. work on builds without ort (e.g., tests
   that build with `--no-default-features`).

### 11.1 Online speaker IDs are per-`Clusterer`-instance

Same as rev 1.

### 11.2 Online vs offline IDs are different ID spaces

Same as rev 1.

### 11.3 Voice-weighted result for short clips

Same as rev 1.

### 11.4 `compute_fbank` configuration is fixed

Same as rev 1.

### 11.5 Diarizer audio buffer is bounded at `EMBED_WINDOW_SAMPLES`

(Updated bound per §5.7; ~128 KB instead of rev 1's incorrect 640 KB.)

### 11.6 `collected_embeddings` retention

By default, `Diarizer` retains every `CollectedEmbedding` for the full
session. Size: ~1 KB per entry (256 × 4 bytes embedding + range +
small ids). For a 30-min session at 1 activity/s: ~1.8 MB. Reasonable.

For long sessions, set `collect_embeddings = false` (skips collection)
or call `clear_collected()` periodically.

### 11.7 Sample-rate validation

Same as rev 1.

### 11.8 `target_speakers` and edge cases

(Documented in §5.5.1.)

### 11.9 K-means deterministic seeding

`OfflineClusterOptions::seed`: `None` derives a deterministic seed by
hashing the byte representation of the input embeddings; `Some(s)` uses
`s` directly. Calling `cluster_offline` twice on the same input with
default options produces the same speaker IDs.

### 11.10 OverflowStrategy semantics

(Documented in §5.4.)

### 11.11 Argmax tie-breaking

(Documented in §5.4 — "lowest-index speaker wins.")

## 12. Decision log

From this brainstorm:

- Three new modules: `dia::embed`, `dia::cluster`, `dia::Diarizer`.
- `dia::embed` is feature-parity with `findit-speaker-embedding` plus
  sliding-window mean for long clips. Better quality than Python's
  center-crop. **Embeddings are NOT directly comparable with Python
  output for clips > 2 s** (cosine ≈ 0.92–0.99 typical for same speaker).
- Pure-Rust `kaldi-native-fbank` for fbank. Lighter build than the C++
  `knf-rs` binding. Numerical parity vs Python's `torchaudio.compliance.kaldi.fbank`
  is not asserted in v0.1.0; a parity test against the C++ original
  is a v0.1.1 follow-up (see §15 #22).
- Two-step pure transform (`compute_fbank` + `embed_features` +
  high-level `embed`) — no Sans-I/O state machine for embedding.
- `&[f32]` zero-copy throughout; `EmbeddingMeta<A, T>` generic with
  `()` defaults; consumed by value into the result.
- Voice-weighted variant accepts caller-provided per-sample voice
  probabilities; `dia::embed` does not run its own VAD.
- Sliding-window mean for clips > 2 s, zero-pad for clips ≤ 2 s,
  reject clips below `MIN_CLIP_SAMPLES`. Hardcoded 1 s hop.
- `Embedding` newtype with `.similarity()` method; cosine = dot product.
- Cross-window speaker linking IS in scope. New `dia::cluster` module.
- Online streaming `Clusterer` with EMA-on-unnormalized-accumulator
  centroid update (rev-2 fix; rev-1's L2-then-arithmetic-mean was
  unsound on antipodal updates). `Reject` is the default
  `OverflowStrategy` (rev-2 change; `AssignClosest` would corrupt
  centroids).
- Offline `cluster_offline` supports both **spectral** (default) and
  **agglomerative** clustering. Spectral via `nalgebra`. `Linkage::Ward`
  removed (rev-2; invalid with cosine). `Affinity::Gaussian` removed
  (rev-2; no defensible default σ for v0.1.0).
- Eigengap K is capped at `min(N - 1, MAX_AUTO_SPEAKERS = 15)` (rev-2
  fix; rev-1 was unbounded → could pick K = N).
- All-zero affinity matrix in spectral → `Error::AllDissimilar`
  precondition check (rev-2).
- K-means uses a deterministic seed: explicit `seed: Option<u64>` on
  `OfflineClusterOptions`; default derives from input hash (rev-2 fix).
- `nalgebra` always in deps (no feature flag).
- `NORM_EPSILON = 1e-12` matches Python's value (verified in
  `findit-speaker-embedding/embedder.py:85`).
- `ClusterAssignment::similarity` is `Option<f32>` not `f32 = -1.0`
  sentinel (rev-2; -1.0 is a valid cosine value).
- `WINDOW_SAMPLES` renamed to `EMBED_WINDOW_SAMPLES` to avoid collision
  with `dia::segment::WINDOW_SAMPLES = 160_000` (rev-2).
- `collected_embeddings()` returns `&[CollectedEmbedding]` struct slice
  (rev-2; rev-1's `&[(TimeRange, Embedding)]` tuples lacked context
  for offline reconciliation).
- `DiarizerBuilder` uses only `with_*` setters (rev-2; rev-1 mixed
  `options(opts)` and `with_*` which was inconsistent).
- Top-level `dia::Diarizer` orchestrator. Borrows `&mut SegmentModel`
  and `&mut EmbedModel` per call. Rationale (rev-2): mirror Segmenter
  pattern + allow model reuse across sessions; the rev-1
  "ort-independence" rationale was a phantom feature.
- `Diarizer::collected_embeddings()` retains per-activity context for
  hybrid online+offline workflows.
- Sample-rate validation deferred to caller.
- No bundled WeSpeaker model.
- `DEFAULT_EMA_ALPHA = 0.2` is **tentative** — chosen by Resemblyzer
  precedent without empirical tuning. v0.2 sensitivity analysis pinned
  in §15 #20.

## 13. Revision history

- **Revision 1** (2026-04-26): initial spec from brainstorming.
- **Revision 2** (2026-04-26, this document): incorporates first
  adversarial-review feedback. Critical algorithmic fixes:
  centroid math (unnormalized accumulator), Ward removal, eigengap K
  cap, all-zero affinity precondition, K-means seed, OverflowStrategy
  default flip + no-update-on-forced-assign. API consistency fixes:
  `EMBED_WINDOW_SAMPLES` rename, `Option<f32>` for similarity,
  `CollectedEmbedding` struct, builder API cleanup. Documentation:
  Python center-crop divergence callout, edge-case enumeration,
  Diarizer ownership rationale corrected, sibling comparison table,
  rejected findings (§14), action list (§15).

## 14. Findings rejected from review 1

For traceability:

- **T2-C** (drop generic `EmbeddingMeta<A, T>` to "match
  `dia::segment::Clip`"): **rejected.** There is no `dia::segment::Clip`
  type — `Segmenter::push_samples(&[f32])` takes raw samples directly.
  The reviewer's "consistency" argument is based on a type that doesn't
  exist. More importantly, **the user explicitly requested generic
  `EmbeddingMeta` during brainstorming** ("Make EmbeddingMeta generic
  over audio_id and track_id, do not use String directly"). Reverting
  that based on a false premise would be wrong. The cost the reviewer
  flags (generics propagating through `EmbeddingResult<A, T>` and the
  embed methods) is real but bounded — the meta-free path
  (`embed`, `embed_weighted`) returns concrete `EmbeddingResult` (=
  `EmbeddingResult<(), ()>`) so typical callers never see the type
  parameters.

## 15. Action list for v0.1.1+ follow-ups

These don't block v0.1.0 but should be tracked:

| #  | Item                                                                                              | Severity |
| -- | ------------------------------------------------------------------------------------------------- | -------- |
| 1  | Add `EmbeddingMeta::audio_id_mut` / `track_id_mut` if a real use case appears                     | low      |
| 22 | Numerical parity test: compare `dia::embed::compute_fbank` against `kaldi-native-fbank` C++ reference and against `torchaudio.compliance.kaldi.fbank` (Python) on a fixed input vector. Threshold: per-coefficient |Δ| < 1e-4. | high  |
| 23 | EMA(α) sensitivity analysis: empirically tune α on a known-labeled multi-speaker dataset; revisit `DEFAULT_EMA_ALPHA = 0.2` if a better value emerges. | medium |
| 24 | Persistent global speaker IDs across sessions (re-identification): match new-session embeddings against saved centroids from a prior session.                                                                          | low |
| 25 | Per-frame voice prob exposure from `dia::segment` so `embed_weighted` can be wired internally (no external VAD needed when using the Diarizer)                                                                         | medium |
| 26 | F1-style numerical parity test plan against `findit-speaker-embedding` Python output on clips ≤ 2 s (where dia and Python should agree, modulo floating-point noise)                                                  | medium |
| 27 | `Diarizer` integration test on a real multi-speaker recording (e.g., a 5-minute podcast clip with 2-3 speakers) — manual inspection of speaker IDs                                                                    | high |
| 28 | Investigate sub-quadratic spectral clustering (Nyström approximation or sparse Laplacian) for very long sessions (N > 5000)                                                                                            | low |
