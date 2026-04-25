# dia — embed + cluster + Diarizer design (v0.1.0 phase 2)

**Revision 1** (2026-04-26).
**Status:** ready for review.
**Scope:** Three new modules to ship in dia v0.1.0 alongside `dia::segment`:
- `dia::embed` — speaker fingerprint generation (port of `findit-speaker-embedding` + improvements)
- `dia::cluster` — cross-window speaker linking (online streaming + offline batch)
- `dia::Diarizer` — top-level orchestrator combining segment + embed + cluster

> This spec covers the second half of dia v0.1.0. `dia::segment` is already
> shipped (commits `25f817c` … `0cfba96`); this work plugs the remaining
> pieces of full speaker diarization onto that foundation.

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

### 2.1 Three modules + an orchestrator

```
src/
├── segment/          # SHIPPED v0.1.0 phase 1, unchanged
├── embed/            # NEW — fingerprint generation
│   ├── mod.rs
│   ├── types.rs        # Embedding, EmbeddingMeta<A,T>, EmbeddingResult<A,T>
│   ├── options.rs      # constants, EmbedModelOptions
│   ├── error.rs        # Error
│   ├── fbank.rs        # compute_fbank wrapper around kaldi-native-fbank
│   ├── embedder.rs     # embed_features / sliding-window logic
│   └── model.rs        # EmbedModel + ort wrapper (cfg ort)
├── cluster/          # NEW — speaker linking
│   ├── mod.rs
│   ├── options.rs      # ClusterOptions, OfflineClusterOptions, enums
│   ├── error.rs        # Error
│   ├── online.rs       # Clusterer (streaming)
│   ├── agglomerative.rs # offline HAC
│   └── spectral.rs     # offline spectral
├── diarizer.rs       # NEW — top-level orchestrator
└── lib.rs            # re-exports
```

### 2.2 Module relationships

| Module | Depends on | Pulls in `ort` | Pulls in `nalgebra` |
|---|---|---|---|
| `dia::segment` | mediatime | yes (gated) | no |
| `dia::embed` | mediatime, kaldi-native-fbank | yes (gated) | no |
| `dia::cluster` | dia::embed (Embedding type) | no | yes (for spectral) |
| `dia::Diarizer` | all of the above | yes (gated) | yes |

`dia::cluster` borrows the `Embedding` type from `dia::embed` (re-exported
for convenience) but stays inference-free — no audio, no model, no ort.

`dia::Diarizer` owns the audio buffer, drives the Segmenter state machine,
calls EmbedModel and Clusterer, and emits the high-level diarization
events. ort sessions (`SegmentModel`, `EmbedModel`) are passed by mutable
reference into Diarizer's methods, **not** owned by Diarizer — same
pattern as `Segmenter::process_samples`.

### 2.3 Three layers of API

Following `dia::segment`'s precedent of "give callers escape hatches":

- **High-level (most users):** `dia::Diarizer::process_samples(samples, |span| ...)`. Push audio in, get `DiarizedSpan`s with global speaker IDs out.
- **Mid-level (custom orchestration):** Compose `Segmenter` + `EmbedModel` + `Clusterer` yourself. ~30 lines of glue code; needed when you want a different audio-buffer strategy, custom span filtering, or to feed in pre-segmented activities.
- **Low-level (custom inference):** `compute_fbank` + `EmbedModel::embed_features` + `cluster_offline`. Pure functions and ONNX hooks; needed when you want to batch inference across thousands of clips, or apply a non-default aggregation, or run offline-only on saved embeddings.

## 3. Scope for v0.1.0 phase 2

### In scope

**`dia::embed`:**
- `EmbedModel` ort wrapper for WeSpeaker ResNet34 (`from_file`, `from_memory`, `*_with_options`)
- `EmbedModelOptions` mirroring `SegmentModelOptions` (graph optimization, providers, intra/inter threads)
- Pure-Rust `kaldi-native-fbank` for feature extraction
- Variable-length clip handling: zero-pad < 2 s, sliding-window mean for > 2 s
- Hardcoded 1 s hop (50 % overlap)
- Voice-weighted variant (`embed_weighted`) consuming caller-provided per-sample voice probabilities
- Two-tier API: low-level `compute_fbank` + `embed_features` / `embed_features_batch`, high-level `embed` / `embed_with_meta` / `embed_weighted` / `embed_weighted_with_meta`
- `Embedding` newtype with `.similarity()` method (cosine = dot product, since both are L2-normalized)
- Generic `EmbeddingMeta<A, T>` with `()` defaults; consumed by value into `EmbeddingResult<A, T>` to avoid `Clone` bounds
- `EmbeddingResult` carries observability fields (`windows_used`, `total_weight`, `source_duration`)

**`dia::cluster`:**
- Online streaming `Clusterer` (threshold-based incremental, EMA centroid updates)
- Offline `cluster_offline` with two methods:
  - **Spectral clustering** (default): affinity matrix → normalized Laplacian → eigendecomposition → K-means on top-K eigenvectors → cluster IDs. The pyannote.audio approach.
  - **Agglomerative clustering**: pairwise distance matrix → repeated argmax-and-merge with configurable linkage (Single / Complete / Average / Ward). Faster for small N, no eigendecomposition.
- Auto-K detection via eigengap (spectral) or threshold (agglomerative); explicit `target_speakers` override available
- Re-uses `dia::embed::Embedding`

**`dia::Diarizer`:**
- Builder API: `Diarizer::builder().build()`
- Internal audio buffer with bounded retention (`WINDOW_SAMPLES` worth = 10 s = ~640 KB)
- `process_samples(seg_model, embed_model, samples, emit)` — synchronous: push audio, segment, embed, cluster, emit `DiarizedSpan`s
- `finish_stream(seg_model, embed_model, emit)` — flush
- `clear()` — reset for new session
- `collected_embeddings()` — accessor for offline re-clustering (hybrid online+offline pattern)
- `pending_inferences()`, `buffered_samples()` — introspection mirroring Segmenter
- Auto-derived `Send + Sync`

### Deferred — explicitly out of v0.1.0

| Item | Reason |
|---|---|
| Bundled WeSpeaker model | ~25 MB; matches soundevents posture (download script) |
| F1 parity tests vs `findit-speaker-embedding` Python output | Need Python infra; numerical-quality smoke tests cover the basics |
| Threading / async service layer | Caller wraps with their own runtime; matches dia::segment |
| Per-frame voice prob from `dia::segment` integrated path | Segment-side API addition; v0.2 segment can expose this cleanly |
| `try_push_samples` soft-cap backpressure | Introspection covers it |
| Voice-weighted mode using internally-computed voice probs | Would force a VAD model dep; caller-provides keeps modules independent |
| Median / top-K-window aggregation | Mean is the only aggregation with strong theoretical backing |
| Configurable hop | Hardcoded 1 s = 50 % overlap (Resemblyzer/pyannote default); add `EmbedOptions` if a real use case appears |
| Batched high-level `embed_batch(&[&[f32]])` | `embed_features_batch` covers the inference batching; multi-clip orchestration is a v0.2 ergonomics feature |
| Speaker re-identification across sessions | Out of scope; `Clusterer::clear()` resets generation; persistent global IDs are a v0.2 feature |

### Out of scope (handled elsewhere)

- VAD — caller's responsibility (silero or similar) before audio reaches `Diarizer`.
- Audio decoding / resampling — caller delivers 16 kHz mono float32 PCM.
- Threading — caller wraps with their own runtime.

## 4. Public API surface

### 4.1 `dia::embed`

#### Constants

```rust
pub const SAMPLE_RATE_HZ: u32 = 16_000;
pub const WINDOW_SAMPLES: u32 = 32_000;     // 2 s @ 16 kHz; model's fixed input
pub const HOP_SAMPLES: u32 = 16_000;        // 1 s, 50 % overlap (hardcoded)
pub const MIN_CLIP_SAMPLES: u32 = 400;      // ~25 ms; one kaldi window
pub const FBANK_NUM_MELS: usize = 80;
pub const FBANK_FRAMES: usize = 200;
pub const EMBEDDING_DIM: usize = 256;
```

#### Types

```rust
/// A 256-d L2-normalized speaker embedding. Newtype gives type safety
/// and a place to attach helpers like `similarity()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Embedding(pub(crate) [f32; EMBEDDING_DIM]);

impl Embedding {
    pub const fn as_array(&self) -> &[f32; EMBEDDING_DIM];
    pub fn as_slice(&self) -> &[f32];
    /// Cosine similarity. Both inputs are L2-normalized, so this reduces
    /// to a dot product. Returns a value in `[-1.0, 1.0]`.
    pub fn similarity(&self, other: &Embedding) -> f32;

    /// L2-normalize a raw 256-d inference output and wrap it as an
    /// `Embedding`. Use after [`EmbedModel::embed_features_batch`]
    /// + custom aggregation (e.g. mean across windows) to produce the
    /// final normalized embedding.
    pub fn normalize_from(raw: [f32; EMBEDDING_DIM]) -> Self;
}

/// Free-function form for callers who prefer it.
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32;

/// Optional metadata that flows through `embed_with_meta` /
/// `embed_weighted_with_meta` to `EmbeddingResult`. Generic over the
/// audio_id and track_id types so callers can use whatever string-like
/// type fits their domain (`String`, `&'static str`, custom enum, …).
/// Defaults to `()` so the unit-typed metadata path allocates nothing.
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

/// Output of a clip-level embedding call.
#[derive(Debug, Clone)]
pub struct EmbeddingResult<A = (), T = ()> {
    embedding: Embedding,
    /// Actual length of the source clip (NOT the padded/cropped 2 s).
    source_duration: Duration,
    /// Number of 2 s windows averaged. 1 for clips ≤ 2 s, larger for
    /// longer clips.
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

    /// All sliding windows had near-zero voice-probability weight; the
    /// weighted average is undefined. Almost always caller error
    /// (passing a pure-silence clip).
    #[error("all windows had effectively zero voice-activity weight")]
    AllSilent,

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
///
/// Pure CPU; no `ort` dependency. Use for batched inference pipelines
/// (compute features for many clips, then call `embed_features_batch`)
/// or for parity testing against another fbank implementation.
pub fn compute_fbank(
    samples: &[f32],
) -> Result<[[f32; FBANK_NUM_MELS]; FBANK_FRAMES], Error>;
```

Internally wraps `kaldi-native-fbank` with the WeSpeaker-trained config
(25 ms frame length, 10 ms hop, Povey window, pre-emphasis 0.97, no
dither, no DC-offset removal, no edge-snipping) and runs the same
pad/center-crop logic the Python project uses for fixed-size inputs.

#### `EmbedModel` (cfg ort)

```rust
#[cfg(feature = "ort")]
pub struct EmbedModelOptions { /* private */ }

#[cfg(feature = "ort")]
impl EmbedModelOptions {
    pub fn new() -> Self;
    pub fn with_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub fn with_providers(self, providers: Vec<ExecutionProviderDispatch>) -> Self;
    pub fn with_intra_op_num_threads(self, n: usize) -> Self;
    pub fn with_inter_op_num_threads(self, n: usize) -> Self;
}

#[cfg(feature = "ort")]
pub struct EmbedModel { /* private; owns ort::Session + scratch */ }

#[cfg(feature = "ort")]
impl EmbedModel {
    // ── Constructors ──────────────────────────────
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error>;
    pub fn from_file_with_options(path: impl AsRef<Path>, opts: EmbedModelOptions)
        -> Result<Self, Error>;
    pub fn from_memory(bytes: &[u8]) -> Result<Self, Error>;
    pub fn from_memory_with_options(bytes: &[u8], opts: EmbedModelOptions)
        -> Result<Self, Error>;

    // ── Low-level: pre-computed features ──────────
    /// Run ONNX inference on one set of [200, 80] kaldi fbank features.
    /// Returns the **raw, un-normalized** 256-d output of the WeSpeaker
    /// model. Wrap with [`Embedding::normalize_from`] to produce an
    /// L2-normalized `Embedding`.
    ///
    /// Returning raw output (rather than an `Embedding`) lets callers
    /// aggregate across multiple windows manually before applying the
    /// final L2-normalize — that's the standard recipe for
    /// sliding-window-mean speaker embeddings.
    pub fn embed_features(
        &mut self,
        features: &[[f32; FBANK_NUM_MELS]; FBANK_FRAMES],
    ) -> Result<[f32; EMBEDDING_DIM], Error>;

    /// Batched feature inference. Single ONNX call with batch size N.
    /// Real speedup on GPU; on CPU with `intra_threads = 1`, similar to
    /// N sequential calls. Returns raw (un-normalized) outputs; wrap each
    /// with `Embedding::normalize_from` after any cross-window
    /// aggregation.
    pub fn embed_features_batch(
        &mut self,
        features: &[[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error>;

    // ── High-level: equal-weighted sliding mean ───
    /// Embed a clip with sliding 2 s windows (`HOP_SAMPLES` hop), equal
    /// weights, mean-then-L2-normalize. Zero-pads clips ≤ 2 s.
    pub fn embed(
        &mut self,
        samples: &[f32],
    ) -> Result<EmbeddingResult, Error>;

    pub fn embed_with_meta<A, T>(
        &mut self,
        samples: &[f32],
        meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error>;

    // ── High-level: voice-weighted sliding mean ───
    /// `voice_probs.len() == samples.len()`. Each value in `[0.0, 1.0]`
    /// is the probability that the corresponding sample is voice. Per-
    /// window weight = mean of the voice_probs over that window's sample
    /// range. Returns `Error::AllSilent` if total weight < epsilon.
    pub fn embed_weighted(
        &mut self,
        samples: &[f32],
        voice_probs: &[f32],
    ) -> Result<EmbeddingResult, Error>;

    pub fn embed_weighted_with_meta<A, T>(
        &mut self,
        samples: &[f32],
        voice_probs: &[f32],
        meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error>;
}
```

#### Re-exports from `dia::embed`

```rust
#[cfg(feature = "ort")]
pub use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "ort")]
pub use ort::execution_providers::ExecutionProviderDispatch;
```

Same pattern as `dia::segment`, including the deliberate divergence from
silero on `ExecutionProviderDispatch` (justified because we expose
`with_providers`).

### 4.2 `dia::cluster`

#### Constants

```rust
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;
```

#### Types

```rust
pub use crate::embed::Embedding;   // re-export for convenience

#[derive(Debug, Clone, Copy)]
pub struct SpeakerCentroid {
    speaker_id: u64,
    centroid: Embedding,           // L2-normalized running mean
    assignment_count: u32,
}
impl SpeakerCentroid {
    pub fn speaker_id(&self) -> u64;
    pub fn centroid(&self) -> &Embedding;
    pub fn assignment_count(&self) -> u32;
}

#[derive(Debug, Clone, Copy)]
pub struct ClusterAssignment {
    pub speaker_id: u64,
    pub is_new_speaker: bool,
    /// Cosine similarity to the assigned centroid, computed pre-update.
    /// For new speakers, this is the max similarity to any existing
    /// centroid (or `-1.0` if there were none).
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct ClusterOptions {
    similarity_threshold: f32,        // default DEFAULT_SIMILARITY_THRESHOLD
    update_strategy: UpdateStrategy,  // default Ema(DEFAULT_EMA_ALPHA)
    max_speakers: Option<u32>,        // default None
    overflow_strategy: OverflowStrategy, // default AssignClosest
}

#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    /// Centroid = arithmetic mean of all assigned embeddings, re-normalized.
    /// Best for offline / batch use; immune to drift in long sessions.
    RollingMean,
    /// EMA: centroid = (1-α) · centroid + α · new, re-normalized.
    /// Adapts to drift over time; recommended for streaming.
    Ema(f32),
}

#[derive(Debug, Clone, Copy)]
pub enum OverflowStrategy {
    /// When `max_speakers` is reached and no centroid passes the
    /// threshold, force-assign to the closest existing speaker.
    AssignClosest,
    /// Reject with `Error::TooManySpeakers`. Caller decides what to do.
    Reject,
}

impl Default for ClusterOptions { /* sensible defaults */ }
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

    #[error("eigendecomposition failed (matrix likely singular or pathological)")]
    EigendecompositionFailed,
}
```

#### Online streaming `Clusterer`

```rust
pub struct Clusterer { /* private */ }

impl Clusterer {
    pub fn new(opts: ClusterOptions) -> Self;
    pub fn options(&self) -> &ClusterOptions;

    /// Submit one embedding; returns the global speaker assignment.
    /// Updates the assigned centroid via the configured strategy.
    pub fn submit(&mut self, embedding: &Embedding) -> Result<ClusterAssignment, Error>;

    pub fn speakers(&self) -> &[SpeakerCentroid];
    pub fn speaker(&self, id: u64) -> Option<&SpeakerCentroid>;
    pub fn num_speakers(&self) -> usize;

    pub fn clear(&mut self);
}
```

`Clusterer` auto-derives `Send + Sync` (all fields are `Send + Sync`).

#### Offline batch clustering

```rust
#[derive(Debug, Clone)]
pub struct OfflineClusterOptions {
    method: OfflineMethod,            // default Spectral { affinity: Cosine }
    similarity_threshold: f32,        // default DEFAULT_SIMILARITY_THRESHOLD
    target_speakers: Option<u32>,     // default None (auto-detect)
}

#[derive(Debug, Clone, Copy)]
pub enum OfflineMethod {
    /// Hierarchical agglomerative clustering. Faster for small N.
    Agglomerative { linkage: Linkage },
    /// Spectral clustering. Recommended for max quality. The default.
    Spectral { affinity: Affinity },
}

#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    Single,
    Complete,
    /// Recommended for speaker clustering.
    Average,
    Ward,
}

#[derive(Debug, Clone, Copy)]
pub enum Affinity {
    /// Direct cosine similarity (= dot product on L2-normalized vectors).
    /// Recommended; works well without tuning.
    Cosine,
    /// Gaussian kernel: exp(-||a-b||² / 2σ²). Adds a bandwidth knob.
    Gaussian { sigma: f32 },
}

impl Default for OfflineClusterOptions { /* method = Spectral { Cosine } */ }
impl OfflineClusterOptions {
    pub fn new() -> Self;
    pub fn with_method(self, m: OfflineMethod) -> Self;
    pub fn with_similarity_threshold(self, t: f32) -> Self;
    pub fn with_target_speakers(self, n: u32) -> Self;
}

/// Cluster a batch of embeddings; returns one global speaker id per
/// input, parallel to the input slice.
///
/// More accurate than `Clusterer::submit` because it considers the entire
/// collection at once. Use for post-processing recorded sessions where
/// streaming latency isn't a concern.
///
/// `Error::EmptyInput` if `embeddings.is_empty()`.
/// `Error::EigendecompositionFailed` if spectral fails (rare; falls back
/// to agglomerative-with-average-linkage internally? No — return the
/// error and let the caller switch methods).
pub fn cluster_offline(
    embeddings: &[Embedding],
    opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error>;
```

### 4.3 `dia::Diarizer`

```rust
pub struct DiarizerOptions {
    segment: SegmentOptions,           // default SegmentOptions::default()
    cluster: ClusterOptions,           // default ClusterOptions::default()
    /// Retain every emitted (range, embedding) pair internally so the
    /// user can call `cluster_offline` on `collected_embeddings()` later.
    /// Default true.
    collect_embeddings: bool,
}

impl Default for DiarizerOptions { /* sensible defaults */ }

#[derive(Debug, Clone, Copy)]
pub struct DiarizedSpan {
    range: TimeRange,
    speaker_id: u64,
    /// Cosine similarity to the assigned speaker's centroid. For new
    /// speakers, max similarity to any existing centroid (or -1.0).
    similarity: f32,
    /// True iff this span created a new global speaker id.
    is_new_speaker: bool,
    /// Window-local slot from `dia::segment` (preserved for debugging).
    speaker_slot: u8,
}

impl DiarizedSpan {
    pub fn range(&self) -> TimeRange;
    pub fn speaker_id(&self) -> u64;
    pub fn similarity(&self) -> f32;
    pub fn is_new_speaker(&self) -> bool;
    pub fn speaker_slot(&self) -> u8;
}

pub struct Diarizer { /* private */ }

pub struct DiarizerBuilder { /* private */ }

impl Diarizer {
    pub fn builder() -> DiarizerBuilder;
    pub fn options(&self) -> &DiarizerOptions;

    /// Push samples through segmenter → embedder → clusterer.
    /// Emits one `DiarizedSpan` per `SpeakerActivity` produced by the
    /// segmenter.
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

    /// Reset to empty state. Generation counters are advanced (matching
    /// dia::segment behavior). Internal allocations are reused.
    pub fn clear(&mut self);

    /// Per-activity embeddings collected during this session, in
    /// emission order. Empty if `collect_embeddings = false`. Useful
    /// for hybrid online+offline workflows: stream → finish_stream →
    /// `cluster_offline(diar.collected_embeddings(), ...)` for refined IDs.
    pub fn collected_embeddings(&self) -> &[(TimeRange, Embedding)];

    /// Discard collected embeddings to free memory mid-session.
    pub fn clear_collected(&mut self);

    pub fn pending_inferences(&self) -> usize;
    pub fn buffered_samples(&self) -> usize;
}

impl DiarizerBuilder {
    pub fn new() -> Self;
    pub fn options(self, opts: DiarizerOptions) -> Self;
    pub fn segment_options(self, opts: SegmentOptions) -> Self;
    pub fn cluster_options(self, opts: ClusterOptions) -> Self;
    pub fn collect_embeddings(self, on: bool) -> Self;
    pub fn build(self) -> Diarizer;
}
```

`Diarizer` auto-derives `Send + Sync`. ort sessions are passed by `&mut`
into method calls (not owned by Diarizer), so the `Diarizer` itself is
ort-feature-independent and can be constructed without ort enabled.
This allows users to construct a Diarizer for offline replay /
re-clustering of collected embeddings even on builds without ort.

#### Diarizer-side errors

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

Each source module keeps its own error type; the Diarizer's error wraps
them via `#[from]` so callers can use `?` cleanly.

## 5. Algorithm semantics (load-bearing decisions)

### 5.1 Sliding-window mean for variable-length clips

For input length `n`:

```
if n < MIN_CLIP_SAMPLES:
    return Error::InvalidClip
elif n <= WINDOW_SAMPLES:
    # one window with zero padding
    padded = samples ++ [0.0; WINDOW_SAMPLES - n]
    features = compute_fbank(padded)
    raw = onnx(features)             # [256], unnormalized
    return l2_normalize(raw)
else:
    # multiple overlapping windows, equal-weighted mean
    starts = [k * HOP_SAMPLES for k = 0..K-1] ++ [n - WINDOW_SAMPLES]
    starts = dedup_and_sort(starts)
    sum = [0.0; 256]
    for s in starts:
        chunk = samples[s..s + WINDOW_SAMPLES]
        features = compute_fbank(chunk)
        sum += onnx(features)
    return l2_normalize(sum / K)
```

The `+ [n - WINDOW_SAMPLES]` and dedup ensures the last window aligns
with end-of-clip, regardless of where the regular grid ended.
Mathematically equivalent to silero / dia::segment's tail-anchor pattern.

### 5.2 Voice-weighted variant

Same algorithm, with two changes:
- For each window, `weight = mean(voice_probs[s..s + WINDOW_SAMPLES])`
- `sum += weight * raw`, accumulate `total_weight += weight`
- `if total_weight < EPSILON: return Error::AllSilent`
- Result: `l2_normalize(sum / total_weight)`

`EPSILON = 1e-6` (avoids divide-by-zero on numeric noise).

### 5.3 Single-window weighting collapse

For clips ≤ 2 s, only one window runs. A scalar weight on a single vector
followed by L2-normalize is a no-op (the scalar cancels in normalization).
So `embed_weighted` and `embed` produce identical results for short clips.
Document this clearly so callers don't think "weighting isn't working" on
short inputs.

### 5.4 Online clustering update

```
for each submitted embedding e:
    if speakers.is_empty():
        new_id = 0; speakers.push(SpeakerCentroid::new(new_id, e))
        return ClusterAssignment { speaker_id: 0, is_new_speaker: true,
                                   similarity: -1.0 }

    sims = [s.centroid.similarity(e) for s in speakers]
    (best_idx, best_sim) = argmax(sims)

    if best_sim >= threshold:
        # update existing speaker
        speakers[best_idx].update(e, strategy)
        speakers[best_idx].assignment_count += 1
        return ClusterAssignment { speaker_id: speakers[best_idx].id,
                                   is_new_speaker: false,
                                   similarity: best_sim }
    else if max_speakers.is_some_and(|cap| speakers.len() >= cap):
        # at cap; behavior depends on overflow_strategy
        match overflow_strategy:
            AssignClosest:
                speakers[best_idx].update(e, strategy)
                return ClusterAssignment { ..., is_new_speaker: false, ... }
            Reject:
                return Error::TooManySpeakers { cap }
    else:
        # new speaker
        new_id = next_speaker_id  # process-monotonic? per-instance? See §11.
        speakers.push(SpeakerCentroid::new(new_id, e))
        return ClusterAssignment { ..., is_new_speaker: true, similarity: best_sim }
```

#### Centroid update strategies

- **`UpdateStrategy::RollingMean`**: keep the assigned-embedding count `c`
  alongside the centroid. On update: `centroid = (centroid * c + e) / (c + 1)`,
  L2-normalize, increment count.
- **`UpdateStrategy::Ema(α)`**: `centroid = (1-α) * centroid + α * e`,
  L2-normalize.

Both maintain L2-normalized centroids so similarity comparisons remain
cosine.

### 5.5 Offline spectral clustering

Algorithm (Ng-Jordan-Weiss, 2002):

```
1. Build affinity matrix A ∈ R^{N×N}:
     - Cosine: A_ij = max(0, e_i · e_j)    (ReLU of dot product, since negatives
       break the Laplacian's spectral properties)
     - Gaussian: A_ij = exp(-||e_i - e_j||² / 2σ²)
     A_ii = 0  (no self-loops)

2. Compute degree matrix D where D_ii = sum_j A_ij.

3. Compute normalized graph Laplacian:
     L_sym = I - D^{-1/2} A D^{-1/2}

4. Eigendecompose L_sym; take K smallest eigenvectors as columns of
   matrix U ∈ R^{N×K}.

5. Determine K:
     - If target_speakers = Some(k): use k.
     - Else: eigengap heuristic — K is the index k that maximizes
       λ_{k+1} - λ_k among eigenvalues of L_sym (k ≥ 1).

6. Row-normalize U: each row to unit length.

7. K-means with K clusters on the rows of U. Use K-means++ seeding for
   determinism reproducibility (seed by hash of input). 100 iterations
   max, or until convergence.

8. Return cluster assignments parallel to input embeddings.
```

Implementation: `nalgebra` for the linear algebra. `nalgebra::SymmetricEigen`
for step 4. K-means is ~30 LOC implemented in-crate.

**Failure mode:** if the eigendecomposition fails (singular matrix,
NaN inputs), return `Error::EigendecompositionFailed`. Caller can fall
back to agglomerative.

### 5.6 Offline agglomerative clustering

Algorithm (HAC, average linkage by default):

```
1. Compute pairwise distance matrix D ∈ R^{N×N}:
     D_ij = 1 - cos_similarity(e_i, e_j)   (so 0 = identical, 2 = opposite)

2. Initialize clusters: each embedding is its own cluster.
3. Repeat:
     - Find the two most similar clusters (smallest D_ij among non-merged).
     - If their distance ≥ (1 - threshold) AND target_speakers = None: stop.
     - If cluster count == target_speakers: stop.
     - Merge them.
     - Update distances (depending on linkage):
         Single: D[merged, k] = min(D[a, k], D[b, k])
         Complete: D[merged, k] = max(D[a, k], D[b, k])
         Average: weighted average by cluster sizes
         Ward: variance-minimizing update (Lance-Williams formula)

4. Assign labels.
```

Implementation: ~100 LOC. No `nalgebra` dependency — uses plain `Vec<f32>`.

### 5.7 Diarizer audio buffer policy

The Diarizer maintains a `VecDeque<f32>` indexed by absolute sample
position. After every `process_samples` returns:

- Drop samples below `total_samples_pushed - WINDOW_SAMPLES` (= last 10 s).
- This is the maximum reach-back of a future tail-anchored segment window
  at `finish()` time, plus a safety margin.

Steady-state memory: ~640 KB (10 s × 16 kHz × 4 bytes).

For a `SpeakerActivity` with range `[s0, s1)`, the slice
`audio_buffer[s0 - audio_base..s1 - audio_base]` is fed to
`embed_model.embed(...)`.

### 5.8 Online vs offline result reconciliation

When a caller does both online (during streaming) and offline (after
finish), the speaker IDs may differ — they were assigned by different
algorithms with different views. Reconciliation policy:

- The two ID spaces are independent; dia does not auto-reconcile.
- Caller can build a mapping by majority vote: for each offline-cluster,
  find the mode of its members' online IDs.
- Document this in the `cluster_offline` rustdoc.

## 6. Module layout

```
src/
├── lib.rs                                 # re-exports + Diarizer
├── diarizer.rs                            # Diarizer + builder + DiarizedSpan
├── segment/                               # SHIPPED (unchanged)
├── embed/
│   ├── mod.rs                             # pub re-exports
│   ├── types.rs                           # Embedding, EmbeddingMeta, EmbeddingResult
│   ├── options.rs                         # constants, EmbedModelOptions
│   ├── error.rs
│   ├── fbank.rs                           # compute_fbank wrapper
│   ├── embedder.rs                        # sliding-window logic, embed_features dispatch
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

Adding to `Cargo.toml`:

```toml
[dependencies]
# Existing
mediatime = "0.1"
thiserror = "2"
ort = { version = "2.0.0-rc.12", optional = true }

# New
kaldi-native-fbank = "0.1"   # pure-Rust kaldi fbank
nalgebra = "0.33"            # spectral clustering eigendecomposition
```

No new feature flags. Both new deps are unconditional (always pulled in).
Rationale:
- `kaldi-native-fbank` is needed by `dia::embed`, which is part of v0.1.0.
- `nalgebra` is needed by `dia::cluster::cluster_offline` with `Spectral`
  method (the default). Gating it behind a feature would leak through to
  the `OfflineClusterOptions::default()` path, which would be surprising.

Lints, profile, and other Cargo metadata are unchanged from
`dia::segment`.

## 8. Streaming flow examples

### 8.1 Easy mode: Diarizer

```rust
use dia::{Diarizer, DiarizerOptions, DiarizedSpan};
use dia::segment::{SegmentModel, SegmentOptions};
use dia::embed::EmbedModel;

let mut seg_model = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
let mut embed_model = EmbedModel::from_file("models/wespeaker_resnet34.onnx")?;
let mut diar = Diarizer::builder()
    .segment_options(SegmentOptions::default())
    .cluster_options(ClusterOptions::default())
    .build();

for chunk in audio_in {
    diar.process_samples(&mut seg_model, &mut embed_model, &chunk, |span| {
        println!("{:?}: speaker {} (similarity {:.2})",
                 span.range(), span.speaker_id(), span.similarity());
    })?;
}
diar.finish_stream(&mut seg_model, &mut embed_model, |span| { ... })?;

// Optional: refined offline labels for higher quality
let refined: Vec<u64> = dia::cluster::cluster_offline(
    &diar.collected_embeddings()
         .iter().map(|(_, e)| *e).collect::<Vec<_>>(),
    &OfflineClusterOptions::default(),
)?;
```

### 8.2 Mid-level: compose modules manually

```rust
use dia::segment::{Segmenter, SegmentModel, SegmentOptions};
use dia::embed::EmbedModel;
use dia::cluster::{Clusterer, ClusterOptions};

let mut seg = Segmenter::new(SegmentOptions::default());
let mut clusterer = Clusterer::new(ClusterOptions::default());
let mut audio_buf: VecDeque<f32> = VecDeque::new();
let mut audio_base: u64 = 0;

while let Some(chunk) = audio_in.next() {
    audio_buf.extend(chunk.iter().copied());
    seg.process_samples(&mut seg_model, &chunk, |event| {
        if let dia::segment::Event::Activity(activity) = event {
            let r = activity.range();
            let lo = (r.start_pts() as u64 - audio_base) as usize;
            let hi = (r.end_pts() as u64 - audio_base) as usize;
            let clip: Vec<f32> = audio_buf.range(lo..hi).copied().collect();
            let result = embed_model.embed(&clip)?;
            let assignment = clusterer.submit(result.embedding())?;
            output.send(my_diarized_span(activity, assignment));
        }
    })?;
}
```

### 8.3 Low-level: batched inference

```rust
use dia::embed::{compute_fbank, EmbedModel};

let activities: Vec<&[f32]> = ...;       // many clips
let features: Vec<_> = activities.iter()
    .map(|c| compute_fbank(c))
    .collect::<Result<_, _>>()?;
let raw_embeddings = embed_model.embed_features_batch(&features)?;
// caller does any further aggregation / clustering
```

## 9. Testing strategy

### 9.1 Unit tests (no model, no ort)

**`dia::embed`:**
- `compute_fbank`: known input vector → expected output shape (200, 80).
- Sliding-window math: 2 s clip → 1 window, 5 s clip → 4 windows (with end-aligned tail), 30 s clip → 29 windows.
- Voice-weighted weight aggregation: per-window weight = mean of corresponding voice_probs slice.
- `EmbeddingMeta` generic type-parameter combinations compile.
- `Embedding::similarity` matches dot product on hand-built unit vectors.

**`dia::cluster`:**
- `Clusterer::submit` on a known sequence of embeddings produces expected speaker assignments.
- Threshold boundary: similarity exactly equals threshold → assigned (≥, not >).
- `UpdateStrategy::RollingMean` with `assignment_count = 5` averages correctly.
- `UpdateStrategy::Ema(0.2)` updates correctly.
- `OverflowStrategy` paths.
- `cluster_offline` with `Agglomerative` + `Average` linkage on a 6-embedding hand-built set produces expected clusters.
- `cluster_offline` with `Spectral` + `Cosine` on the same set produces the same clusters (sanity check).
- Eigengap auto-K detection on a 4-cluster synthetic set picks K=4.

**`dia::Diarizer`:**
- Audio buffer trim: after each `process_samples`, samples older than
  `total_samples_pushed - WINDOW_SAMPLES` are dropped.
- `clear()` resets generation, audio buffer, collected_embeddings, clusterer.
- `collected_embeddings()` length matches the count of emitted DiarizedSpans.

### 9.2 Integration tests (gated)

`tests/integration_diarize.rs` with `#[ignore]`. Requires both ONNX models
under `models/`. Pushes 60 seconds of synthetic audio (mixture of two
"speakers" represented by different sine-wave fundamentals) through
`Diarizer::process_samples`. Asserts:
- At least 2 distinct `speaker_id` values are emitted.
- All `DiarizedSpan` ranges are within the input audio's bounds.
- `collected_embeddings().len()` matches the count of emitted spans.
- `cluster_offline` on the collected embeddings returns a cluster
  assignment of the same length.

### 9.3 Compile-time trait assertions

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
  `dia::segment::SegmentModel`.
- **`Clusterer`**: auto-derives `Send + Sync`. State machine behind `&mut`
  (each call needs `&mut self`).
- **`Diarizer`**: auto-derives `Send + Sync`. ort sessions are passed in
  per call rather than owned, so `Diarizer` itself doesn't carry the
  `!Sync` baggage.
- No internal locks. No async. Callers wrap with their own runtime.
- `Diarizer::clear()` advances the underlying `Segmenter`'s generation
  counter (via `Segmenter::clear`); the `Clusterer`'s speaker IDs reset
  to 0. Does not discard or warm down the ONNX sessions.
- **Determinism**: same caveats as `dia::segment` — set
  `intra_op_num_threads = 1` on both `SegmentModelOptions` and
  `EmbedModelOptions` for bit-exact reproducibility. K-means in
  spectral clustering uses a deterministic seed for reproducibility.

## 11. Resolved contracts

### 11.1 Online speaker IDs are per-`Clusterer`-instance

`Clusterer::submit` assigns IDs starting from 0 and incrementing.
`clear()` resets to 0. These IDs are NOT process-wide unique like
`WindowId`'s generation counter — speaker identity is by definition
session-scoped.

If a caller needs cross-session speaker re-identification, they
maintain their own mapping (e.g., persist centroids and match by
similarity at session-start time). Out of scope for v0.1.0.

### 11.2 Online vs offline IDs are different ID spaces

`Clusterer` (online) and `cluster_offline` produce IDs independently.
The same audio session run through both will likely produce different
ID assignments. Caller reconciles if needed (e.g., by majority-vote
mapping, see §5.8).

`Diarizer` emits **online** IDs in `DiarizedSpan`s. To get offline
IDs, call `cluster_offline(diar.collected_embeddings())` after
`finish_stream`.

### 11.3 Voice-weighted result for short clips

For clips ≤ 2 s (single window), `embed_weighted` is mathematically
equivalent to `embed`: the per-window weight is a scalar that cancels
out under L2-normalization. Documented in `embed_weighted` rustdoc
so callers don't think "weighting isn't working."

### 11.4 `compute_fbank` configuration is fixed

Frame length 25 ms, hop 10 ms, Povey window, pre-emphasis 0.97, no
dither, no DC-offset removal, no edge-snipping. These match what
WeSpeaker was trained with. Exposing them as options would only let
users break parity. No `FbankOptions` for v0.1.0.

### 11.5 Diarizer audio buffer is bounded at `WINDOW_SAMPLES`

Steady-state memory for the audio buffer is ~640 KB. After each
`process_samples`, samples older than `total_samples_pushed -
WINDOW_SAMPLES` are dropped. This is the maximum reach-back of a
tail-anchored window at `finish()` time, plus zero margin (but
`WINDOW_SAMPLES` IS the safety: tail starts at `total - WINDOW_SAMPLES`,
so we keep exactly `WINDOW_SAMPLES` samples at all times).

### 11.6 `collected_embeddings` retention

By default, `Diarizer` retains all (range, embedding) pairs for the
full session. Embedding size: 256 × 4 = 1 KB per activity. For a
30-minute session with 1 activity per second: ~1.8 MB. Reasonable.

For very long sessions, set `collect_embeddings = false` in
`DiarizerOptions` (skips collection) or call `clear_collected()`
periodically (drops what's been collected).

### 11.7 Sample-rate validation

Like `dia::segment::Segmenter::push_samples`, `Diarizer::process_samples`
does NOT validate sample rate. Callers feeding non-16 kHz audio get
silent corruption. Documented; matches the rest of dia's posture.

### 11.8 ONNX session ownership

`Diarizer` does not own the ort sessions; they are borrowed per call.
This decouples Diarizer construction from ort feature gating: a build
without the `ort` feature can still construct a `Diarizer` for offline
replay over collected embeddings.

The downside: caller juggles three things (`SegmentModel`, `EmbedModel`,
`Diarizer`). The convenience savings come from the audio-buffer
management and the segment→embed→cluster wiring.

### 11.9 Cluster overflow handling

When `max_speakers` is set and a new embedding doesn't match any
existing centroid:
- `OverflowStrategy::AssignClosest`: force-assign to the closest
  existing centroid. Update its centroid as usual. The
  `is_new_speaker` flag is `false`. Use this when speaker count is
  bounded by domain knowledge (e.g., known meeting attendee count).
- `OverflowStrategy::Reject`: return `Error::TooManySpeakers`. Caller
  decides — drop the embedding, expand the cap, etc.

Default is `AssignClosest` (more forgiving).

## 12. Decision log

From this brainstorm:

- Three new modules: `dia::embed`, `dia::cluster`, `dia::Diarizer`.
- `dia::embed` is feature-parity with `findit-speaker-embedding` plus
  sliding-window mean for long clips (better quality than Python's
  center-crop).
- Pure-Rust `kaldi-native-fbank` for fbank (vs C++-binding `knf-rs`).
  Lighter build, sufficient parity for v0.1.0.
- Two-step pure transform (`compute_fbank` + `embed_features` +
  high-level `embed`) — no Sans-I/O state machine for embedding
  because there's no streaming state to maintain.
- `&[f32]` zero-copy throughout; `EmbeddingMeta<A, T>` generic with
  `()` defaults and consumed by value into the result.
- Voice-weighted variant accepts caller-provided per-sample voice
  probabilities; `dia::embed` does not run its own VAD.
- Sliding-window mean for clips > 2 s, zero-pad for clips ≤ 2 s,
  reject clips below `MIN_CLIP_SAMPLES`.
- Hardcoded 1 s hop (50 % overlap); not configurable in v0.1.0.
- `Embedding` newtype with `.similarity()` method; cosine = dot product.
- Cross-window speaker linking IS in scope for v0.1.0. New
  `dia::cluster` module.
- Online streaming `Clusterer` with EMA centroid update, threshold-based
  assignment, configurable overflow strategy.
- Offline `cluster_offline` supports both **spectral** (default) and
  **agglomerative** clustering. Spectral via `nalgebra`.
- `nalgebra` always in deps (no feature flag).
- Top-level `dia::Diarizer` orchestrator. Owns audio buffer, drives
  segment + embed + cluster. ort sessions passed by `&mut` per call,
  not owned.
- `Diarizer::collected_embeddings()` retains per-activity embeddings
  for hybrid online+offline workflows.
- Sample-rate validation deferred to caller; matches segment's posture.
- No bundled WeSpeaker model (~25 MB); provide download script.

## 13. Revision history

- **Revision 1** (2026-04-26): initial spec from brainstorming.
