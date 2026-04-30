# dia — segmentation sub-project design (v0.1.0)

**Revision 5** (2026-04-25, post-fourth-adversarial-review — reviewer
declared rev 4 "near-qualified" with one must-fix; this revision applies
that fix plus the line-edit polish items the reviewer flagged).

**Status:** ready for re-review (reviewer suggested this is the last round).
**Scope:** The `diarization::segment` module only. `diarization::embed` is a separate sub-project to be specced after this one ships.

> **Revision 5 changes** (full list in §12 / §15): defines the
> `frame_index_of(sample) -> u64` companion to `frame_to_sample` with
> explicit floor rounding for boundary safety (the rev-4 §5.4.1 formula
> referenced this function without defining it); restores the
> `push_inference` rustdoc enumeration that was condensed away in rev 4;
> pins `total_samples` semantics to "cumulative counter, never
> decremented"; adds `static_assertions`-style compile-time `Send`/`Sync`
> tests to §9; documents the `Relaxed`-ordering rationale in §11.9;
> tweaks several wordings.

## 1. Context

`dia` is a Rust port of two findit-studio Python projects: `findit-pyannote-seg` (speaker
segmentation, M1) and `findit-speaker-embedding` (speaker embedding, M2). The single
`dia` crate will host both as modules: `diarization::segment` and `diarization::embed`. **This spec
covers `diarization::segment` only.**

The crate sits in the findit-studio Rust suite alongside `silero` (VAD),
`soundevents` (sound-event classification), `scenesdetect` (scene/shot
boundaries), and `mediatime` (rational time types). It targets stream-processing
audio applications that ingest 16 kHz mono PCM samples frame-by-frame and consume
diarization events.

`dia` adopts `mediatime` for every time/range/duration value crossing the API.
This **shares the choice with `scenesdetect`**; `silero` and `soundevents` use raw
`u64` sample indices and `core::time::Duration` instead. The design rationale for
following `scenesdetect` is that segmentation, like scene detection, produces
sample-range outputs that benefit from explicit timebase-tagged types — VAD and
sound-event classification have a flatter output shape that doesn't.

## 2. Architecture: Sans-I/O dual layer

The architectural philosophy is borrowed from `scenesdetect` (state-machine-only,
zero internal I/O) but adapted to acknowledge that segmentation MUST run a neural
network. The compromise: a Sans-I/O state machine with no `ort` dependency
(Layer 1), plus a separate Layer 2 that pairs it with an `ort::Session`.

### 2.1 What we kept and what we changed vs siblings

| Pattern                                | scenesdetect  | silero                                                                      | soundevents                                                              | dia (this spec)                                                                                                                  |
| -------------------------------------- | ------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Owns model session?                    | n/a (no ML)   | yes (`Session`)                                                             | yes (`Classifier`)                                                       | **No** — Layer 1 has no `ort` dep; Layer 2's `SegmentModel` does                                                                 |
| State machine and session in one type? | n/a           | yes                                                                         | n/a (no streaming state)                                                 | **No** — `Segmenter` (Layer 1) and `SegmentModel` (Layer 2) are separate                                                         |
| Streaming push API                     | `process(frame) -> Option<Cut>` | `Session::process_stream(stream, samples, FnMut(f32))`            | none — batch only (`classify(samples) -> Vec<...>`)                      | `Segmenter::push_samples` + `poll` (Layer 1) plus `Segmenter::process_samples(model, samples, FnMut(Event))` (Layer 2)           |
| Inference request as an explicit action| n/a (no model)| no (internal)                                                               | no (internal)                                                            | **Yes** — `Action::NeedsInference` round-trips via the caller                                                                    |
| Output type                            | `Option<Timestamp>` | `f32` per frame                                                       | `Vec<EventPrediction>` per call                                          | **Two** — `Action` (Layer 1, includes inference requests) / `Event` (Layer 2, emission only)                                     |
| Worker-pool friendliness               | yes (no I/O)  | yes (one session per worker)                                                | yes (one classifier per worker)                                          | yes (one Segmenter per stream; one SegmentModel per worker)                                                                      |
| Uses mediatime?                        | **yes**       | no (raw `u64` samples)                                                      | no (`core::time::Duration`)                                              | **yes** — follows scenesdetect                                                                                                   |
| `Send` (state holder)                  | yes (auto)    | yes (auto)                                                                  | yes (auto)                                                               | yes (auto, on `Segmenter`)                                                                                                       |
| `Sync` (state holder)                  | yes (auto)    | **no** — silero docs `Session is Send but not Sync` because `ort::Session` is `!Sync` (silero/src/session.rs line 61) | yes (`Classifier` has no `OrtSession` field shown to be `!Sync`)         | **`Segmenter`: yes (auto)** — Layer 1 holds no `!Sync` field. **`SegmentModel`: no (auto)** — wraps `ort::Session`, matches silero |

Novel pieces relative to all three siblings: the `Action` / `Event` enum split
(so inference is the caller's concern, not the library's), the explicit
`WindowId` round-trip with a stale-id-rejection generation counter (§4 / §11.9),
and the `Segmenter`/`SegmentModel` separation (so Layer 1 is `ort`-free).

### 2.2 Layer 1 — Sans-I/O `Segmenter` (no `ort` dependency)

```rust
let mut seg = Segmenter::new(SegmentOptions::default());

seg.push_samples(&pcm_chunk);
while let Some(action) = seg.poll() {
    match action {
        Action::NeedsInference { id, samples } => {
            let scores = my_inferencer.run(&samples)?;     // caller's I/O
            seg.push_inference(id, &scores)?;
        }
        Action::Activity(a)  => emit_activity(a),
        Action::VoiceSpan(r) => emit_voice_span(r),
    }
}
seg.finish();
while let Some(action) = seg.poll() { /* drain tail */ }
```

The state machine never reaches outside itself. All buffering, window scheduling,
powerset decoding, hysteresis, voice-span stitching, and final emission are pure
CPU work owned by the `Segmenter`. Layer 1 has zero `ort` dependency and is
exercisable in unit tests with synthetic scores — no model file required.

### 2.3 Layer 2 — `SegmentModel` + streaming wrappers (gated on `ort` feature, default-on)

```rust
let mut model = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
let mut seg   = Segmenter::new(SegmentOptions::default());

while let Some(frame) = audio_in.next().await {
    seg.process_samples(&mut model, &frame, |event| match event {
        Event::Activity(a)  => emit_activity(a),
        Event::VoiceSpan(r) => emit_voice_span(r),
    })?;
}
seg.finish_stream(&mut model, |event| { /* drain */ })?;
```

`process_samples` is sugar for the Layer 1 loop with `model.infer` filling each
`NeedsInference`. `Event` is `Action` minus the `NeedsInference` variant.

## 3. Scope for v0.1.0

### In scope

- Layer 1 `Segmenter` with `push_samples`, `poll`, `push_inference`, `finish`,
  `clear`, plus introspection (`pending_inferences`, `buffered_samples`).
- **Process-wide unique `WindowId` generation counter** (one
  `static AtomicU64`; bumped on every `Segmenter::new` and on every `clear`)
  that prevents stale-id corruption both within a single segmenter (across
  `clear`) and across multiple segmenters in the same process. Details in
  §11.9.
- **Finalization boundary that incorporates pending windows** so out-of-order
  `push_inference` cannot finalize frames whose other contributing windows
  have not yet reported. Details in §5.4.
- Layer 2 `SegmentModel` (ort wrapper) + `Segmenter::process_samples` /
  `finish_stream` convenience methods, behind the `ort` feature.
- Sliding-window scheduling with configurable step (default 40 000 samples =
  2.5 s) and a tail window anchored to end-of-stream when needed, with
  deduplication when the tail-anchor start equals the last regular window's
  start.
- Powerset decoding (7-class → 3-speaker, additive sum of class probabilities).
- Hysteresis binarization with onset/offset thresholds (sequential state
  machine, **runs on the per-FRAME timeline**).
- Voice-timeline stitching with a **per-FRAME** `(sum, count)` accumulator
  (overlap-add mean of `1 - p[silence]`), then per-frame hysteresis, then RLE,
  then frame-to-sample conversion at emission time.
- Voice-span post-processing: **merge BEFORE min-duration filter** (matches
  Python).
- Window-local `SpeakerActivity` records, sample range clamped to actual stream
  length on tail windows.
- ONNX model shape verification at `from_file`/`from_memory` load time
  (returns `Error::IncompatibleModel` if input or output dims don't match
  `[*, 1, 160000]` / `[*, 589, 7]`).
- `SegmentModel` configuration for execution providers and graph optimization
  level (CPU + level 3 by default), via re-exported `ort` types (see §4 for
  the divergence from silero, which only re-exports `GraphOptimizationLevel`).
- Pure-CPU unit tests on the state machine with hand-crafted scores
  (no model file required), plus the streaming-specific cases listed in §9.
- Two `examples/`: a Layer-1 example with a synthetic inferencer and a
  Layer-2 example streaming a WAV file.
- A gated `#[ignore]` integration test against a downloaded model.
- A throughput bench on Layer 1.

### Deferred — out of v0.1.0

| Item                                            | Reason                                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `include_frame_probabilities`                   | Adds another `Action` variant; ship when a caller actually needs it.                                  |
| Bundled ONNX model (`include_bytes!`)           | Model is ~5.7 MB; licensing review can come later. Provide a download script.                         |
| Reference parity test vs `pyannote.audio`       | Requires Python + HF_TOKEN. Layer 1 unit tests cover the logic; F1 lands later.                       |
| Cross-window speaker clustering / global IDs    | Future M3 layer (downstream of `diarization::embed`); not segmentation's job.                                 |
| `infer_batch` for cross-stream batching         | Not needed at CPU `intra_threads=1` (batch=1 is already the fastest path on M1). Add for GPU/HT-CPU.  |
| `IoBinding` / `infer_into(&mut [f32; 4123])`    | ~16 KB allocation per 2.5 s of audio; not a hot path. Add when a profile points here.                 |
| `Arc<[f32]>` / borrow-token in `NeedsInference` | Same — alloc cost is small; design churn is large.                                                    |
| **`serde` feature for output types**            | **Deferred entirely from v0.1.0.** Re-introducing the feature requires `#[cfg_attr(serde, derive(Serialize, Deserialize))]` on `WindowId`, `SpeakerActivity`, `Action`, `Event`, and threading `mediatime/serde`. Half-wiring it is worse than not having it. |
| `ControlFlow<E>` / `Result<()>` user callbacks  | Silero ships `FnMut(f32)` (no error propagation); we follow that idiom. Users `.set` an `Option` from inside the closure if they need to surface an error after the call returns. (Citation verified at silero/src/session.rs:297.) |
| `step_samples` typed as `Duration` rather than `u32` | Aesthetic cleanup; keeps churn out of v0.1.0. Pick at v0.2 / v1.0.                              |
| `drain()` iterator adapter                      | `while let Some(action) = seg.poll()` is fine and explicit. Add when ergonomics demand it.            |
| `SamplesAt16k<'_>` typed input                  | Type-safe but pollutes every call site. Caller-must-enforce matches silero/scenesdetect/soundevents.  |
| Soft-cap `try_push_samples` returning `Backpressure` | `pending_inferences()` and `buffered_samples()` are introspection-only in v0.1.0. Add a cap if a caller actually hits a runaway-buffer scenario. |
| `Segmenter::default()` impl                     | Always paired with explicit `SegmentOptions`. Add if convenience demands.                             |
| `alloc` Cargo feature                           | Removed — the crate is unconditionally `alloc`-required (uses `Vec`, `Box`, `BTreeMap`, `VecDeque`). |

### Out of scope (handled elsewhere)

- Speaker embedding — `diarization::embed`, separate sub-project.
- Resampling — input must be 16 kHz mono float32; caller enforces.
- Audio-file decoding — caller's responsibility.
- Threading / worker pools — caller's responsibility.

## 4. Public API surface

### Constants

```rust
/// 16 kHz monolithic timebase used for every `TimeRange` and `Timestamp`
/// emitted by `diarization::segment`. pyannote-seg-3.0 only operates at 16 kHz.
pub const SAMPLE_RATE_TB: Timebase;

pub const SAMPLE_RATE_HZ: u32 = 16_000;
pub const WINDOW_SAMPLES: u32 = 160_000;        // 10 s @ 16 kHz
pub const FRAMES_PER_WINDOW: usize = 589;
pub const POWERSET_CLASSES: usize = 7;          // silence, A, B, C, A+B, A+C, B+C
pub const MAX_SPEAKER_SLOTS: u8 = 3;
```

The actual schema-drift detector is runtime ONNX shape verification at
`SegmentModel::from_file`. No `static_assert!` blocks.

### `ort` re-exports (feature-gated)

```rust
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::execution_providers::ExecutionProviderDispatch;
```

`GraphOptimizationLevel` matches silero's re-export precisely.
`ExecutionProviderDispatch` is **deliberately re-exported beyond what silero
does** — silero only ships level configuration, not provider configuration.
We re-export it because `SegmentModel`'s `with_providers` builder takes a
`Vec<ExecutionProviderDispatch>`; without the re-export, callers would be
forced to import `ort` directly even after picking the dia surface.

### Types

```rust
/// Stable correlation handle for one inference round-trip. Carries the
/// window's sample range in `SAMPLE_RATE_TB` plus an opaque generation
/// token minted from a process-wide counter (§11.9). Two `WindowId`s
/// compare equal iff both their range AND generation match.
///
/// The generation counter eliminates two corruption scenarios:
/// 1. **Within one segmenter**, a stale `push_inference` from before a
///    `clear()` cannot match a new pending entry with the same range.
/// 2. **Across segmenters in the same process**, an `id` accidentally
///    fed to the wrong `Segmenter` cannot match because each
///    `Segmenter::new` consumes a fresh counter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId { /* private: range + generation */ }

impl WindowId {
    pub const fn range(&self)    -> TimeRange;
    pub const fn start(&self)    -> Timestamp;
    pub const fn end(&self)      -> Timestamp;
    pub const fn duration(&self) -> Duration;
    // No public `generation()` — the counter is opaque. `Debug` exposes it
    // for diagnostics. If a future caller demonstrates need for stable
    // introspection, we'll add `pub fn generation_for_diagnostics(&self)`
    // with explicit unstable-API framing.
}

// Ord by (generation, start_pts). Within one generation, ordering is
// "by sample position" and meaningful. Across generations, ordering is
// deterministic (suitable for `BTreeMap` lookup) but semantically
// meaningless — cross-stream IDs should not be compared by `<` / `>`.
impl Ord        for WindowId { /* by (generation, start_pts) */ }
impl PartialOrd for WindowId { /* delegates */ }

/// One window-local speaker activity. `speaker_slot` ∈ {0, 1, 2} is local
/// to the emitting window — slot identity does NOT cross windows.
/// Cross-window speaker identity is the job of a future clustering layer (M3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerActivity { /* fields private */ }
impl SpeakerActivity {
    pub const fn window_id(&self)    -> WindowId;
    pub const fn speaker_slot(&self) -> u8;
    pub const fn range(&self)        -> TimeRange;
}

/// One output of the Layer 1 state machine.
///
/// Style note: enum-variant fields (`id`, `samples`) are public because
/// they participate in pattern matching. Structs with invariants
/// (`WindowId`, `SpeakerActivity`) use private fields with accessors.
/// The two conventions coexist deliberately.
#[derive(Debug, Clone)]
pub enum Action {
    /// Caller must run ONNX inference on `samples` and call
    /// `Segmenter::push_inference(id, scores)`. `samples.len() == WINDOW_SAMPLES`.
    NeedsInference { id: WindowId, samples: Box<[f32]> },
    /// A decoded window-local speaker activity.
    Activity(SpeakerActivity),
    /// A finalized speaker-agnostic voice region. Emit-only — never retracted
    /// once produced.
    VoiceSpan(TimeRange),
}

/// Layer 2 emission events (Layer 2 hides `NeedsInference` from the caller).
#[derive(Debug, Clone)]
pub enum Event {
    Activity(SpeakerActivity),
    VoiceSpan(TimeRange),
}
```

### Options

```rust
#[derive(Debug, Clone)]
pub struct SegmentOptions { /* fields private */ }

impl SegmentOptions {
    pub const fn new() -> Self;          // pyannote defaults
    pub const fn default() -> Self;      // same

    // accessors (all `const fn`)
    pub const fn onset_threshold(&self)        -> f32;
    pub const fn offset_threshold(&self)       -> f32;
    pub const fn step_samples(&self)           -> u32;
    pub const fn min_voice_duration(&self)     -> Duration;
    pub const fn min_activity_duration(&self)  -> Duration;
    pub const fn voice_merge_gap(&self)        -> Duration;

    // builders (`with_*`) and mutating `set_*` variants for every field.
}
```

Defaults match `findit-pyannote-seg.RequestOptions`: onset 0.5, offset 0.357,
step 40 000 samples, all duration filters disabled. Builders/setters are
infallible; `Error::InvalidOptions` is reserved for future eager validation.

### `Segmenter` (Layer 1)

```rust
pub struct Segmenter { /* private */ }

impl Segmenter {
    pub fn new(opts: SegmentOptions) -> Self;
    pub fn options(&self) -> &SegmentOptions;

    /// Append 16 kHz mono float32 PCM. Arbitrary chunk size. Caller must
    /// enforce sample rate — there is no runtime guard.
    ///
    /// `samples.len() == 0` is a no-op: the call is accepted but does NOT
    /// count as a "non-empty push" for the §11.7 tail-window threshold.
    pub fn push_samples(&mut self, samples: &[f32]);

    /// Drain the next action. `None` means "nothing ready right now"; it
    /// does NOT mean "done" — the caller decides when the stream ends by
    /// calling `finish()`.
    pub fn poll(&mut self) -> Option<Action>;

    /// Provide ONNX scores for a previously-yielded `NeedsInference`.
    /// `scores.len()` must equal `FRAMES_PER_WINDOW * POWERSET_CLASSES = 4123`.
    ///
    /// Returns `Error::UnknownWindow` if `id` is not in the pending set.
    /// This covers four scenarios:
    /// 1. `id` was never yielded by `poll`.
    /// 2. `id` was already consumed by an earlier `push_inference` call
    ///    (each pending entry is consumed exactly once).
    /// 3. `id` came from a previous stream that was reset by `clear()`
    ///    (caught by the generation counter — see `WindowId` and §11.9).
    /// 4. `id` was minted by a different `Segmenter` instance whose sample
    ///    range happens to match a current pending window's range
    ///    (different generation; see §11.9).
    ///
    /// Returns `Error::InferenceShapeMismatch` if `scores.len() != 4123`.
    pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error>;

    /// Signal end-of-stream. Schedules a tail-anchored window if needed
    /// (deduplicated against the last regular window's start). The closing
    /// voice span (if any) is emitted as soon as the last pending inference
    /// is fulfilled (or immediately, if there is none).
    pub fn finish(&mut self);

    /// Reset to empty state for a new stream:
    /// - input buffer cleared,
    /// - pending inferences dropped (their IDs become unknown — see §11.9),
    /// - voice/hysteresis state reset,
    /// - `finished` / `tail_emitted` flags cleared,
    /// - process-wide generation counter advanced.
    ///
    /// Internal allocations are reused. Does NOT discard or warm down a
    /// paired `SegmentModel`.
    pub fn clear(&mut self);

    // Introspection — for backpressure and debugging.

    /// Number of `NeedsInference` actions yielded but not yet fulfilled
    /// via `push_inference`. Stays at zero in a steady state.
    pub fn pending_inferences(&self) -> usize;

    /// Number of input samples currently buffered (i.e. pushed via
    /// `push_samples` but not yet released because they're still part of
    /// some not-yet-scheduled or in-flight window).
    pub fn buffered_samples(&self) -> usize;
}
```

Auto-derived `Send + Sync`. All fields are `Send + Sync` primitives or
`Send + Sync` collections (`Vec`, `VecDeque`, `BTreeMap`, `Box<[f32]>`,
`Option<u64>`). The state machine is normally driven through `&mut self`,
so `Sync` is incidental — sharing one `Segmenter` between threads buys
nothing because every API call needs `&mut self`.

> Note: `Segmenter`'s `Sync`-ness differs from `silero::Session`'s. Silero's
> `Session` wraps `ort::Session` directly and is `!Sync` because of it.
> `Segmenter` (Layer 1) has no `ort` field and so picks up `Sync` for free.
> The Layer-2 `SegmentModel` is the dia type that mirrors silero's `Session`
> shape, and (correctly) is `Send` but `!Sync` (see below).

### `SegmentModel` and Layer 2 helpers (feature `ort`, default on)

```rust
#[cfg(feature = "ort")]
pub struct SegmentModelOptions { /* private */ }

#[cfg(feature = "ort")]
impl SegmentModelOptions {
    pub const fn new() -> Self;
    pub fn with_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    /// Configure execution providers in priority order. Default: CPU only.
    /// CoreML on macOS is known to degrade segmentation-3.0 numerics; do not
    /// enable without measuring.
    pub fn with_providers(self, providers: Vec<ExecutionProviderDispatch>) -> Self;
    pub fn with_intra_op_num_threads(self, n: usize) -> Self;
    pub fn with_inter_op_num_threads(self, n: usize) -> Self;
}

#[cfg(feature = "ort")]
pub struct SegmentModel { /* owns ort::Session + reusable scratch */ }

#[cfg(feature = "ort")]
impl SegmentModel {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error>;
    pub fn from_file_with_options(path: impl AsRef<Path>, opts: SegmentModelOptions) -> Result<Self, Error>;
    pub fn from_memory(bytes: &[u8]) -> Result<Self, Error>;
    pub fn from_memory_with_options(bytes: &[u8], opts: SegmentModelOptions) -> Result<Self, Error>;

    pub fn infer(&mut self, samples: &[f32]) -> Result<Vec<f32>, Error>;
}

#[cfg(feature = "ort")]
impl Segmenter {
    pub fn process_samples<F: FnMut(Event)>(
        &mut self, model: &mut SegmentModel, samples: &[f32], emit: F,
    ) -> Result<(), Error>;

    pub fn finish_stream<F: FnMut(Event)>(
        &mut self, model: &mut SegmentModel, emit: F,
    ) -> Result<(), Error>;
}
```

`from_memory` / `from_memory_with_options` accept `&[u8]` and **copy** the bytes
internally (via `ort::Session::Builder::commit_from_memory`, sibling-validated
at silero/src/session.rs:120). The buffer can be dropped immediately after the
constructor returns.

`SegmentModel` validates input/output shapes at construction:
- input dims must match `[*, 1, 160000]` (dynamic batch axis allowed)
- output dims must match `[*, 589, 7]`

A wrong-architecture model fails at load with `Error::IncompatibleModel`, not
on first inference.

`SegmentModel` auto-derives `Send`. It does **not auto-derive `Sync`**
because `ort::Session` is `!Sync` (and therefore at least one field of
`SegmentModel` is `!Sync`). **Matches `silero::Session` exactly**
(silero/src/session.rs line 61 documents the same property). Use one per
worker.

**Scratch ownership:** `SegmentModel` owns the input scratch (`Vec<f32>` reused
across `infer` calls). Output `Vec<f32>` is allocated per call; reuse is
deferred (see §3 "Deferred"). `Segmenter` owns its own pre-allocated buffers
for the state machine; no scratch is shared between layers.

### Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reserved for future eager validation; not currently emitted (v0.1.0
    /// stores option values verbatim).
    #[error("invalid segment options: {0}")]
    InvalidOptions(&'static str),

    #[error("inference scores length {got}, expected {expected}")]
    InferenceShapeMismatch { expected: usize, got: usize },

    /// `push_inference` was called with a `WindowId` that is not in the
    /// pending set. See `Segmenter::push_inference` rustdoc for the full
    /// list of scenarios this covers.
    #[error("inference scores received for unknown WindowId {id:?}")]
    UnknownWindow { id: WindowId },

    #[cfg(feature = "ort")]
    #[error("model {tensor} dims {got:?}, expected {expected:?}")]
    IncompatibleModel { tensor: &'static str, expected: &'static [i64], got: Vec<i64> },

    #[cfg(feature = "ort")]
    #[error("failed to load model from {path}: {source}")]
    LoadModel { path: PathBuf, #[source] source: ort::Error },

    #[cfg(feature = "ort")]
    #[error(transparent)]
    Ort(#[from] ort::Error),
}
```

Cargo dependency: `thiserror = "2"` (defaults on, **matches silero**). The
previous revision specified `default-features = false` to align with
`scenesdetect`, but silero is the larger sibling and `thiserror`'s default
features don't conflict with our build configuration.

## 5. Algorithm semantics (load-bearing decisions)

### 5.1 Powerset decoding

For each output frame:

1. Apply numerically stable softmax to the 7-class logits row.
2. Per-speaker probability is the **additive sum** of marginals — speaker A is
   active iff class 1 (A only), 4 (A+B), or 5 (A+C) fired:
   ```
   p(A) = p[1] + p[4] + p[5]
   p(B) = p[2] + p[4] + p[6]
   p(C) = p[3] + p[5] + p[6]
   ```
3. Voice probability (any-speaker active) is computed **separately** as
   `1 - p[silence] = 1 - p[0]`, not derived from the per-speaker marginals
   (which would over-count overlap classes).

Reference: findit-pyannote-seg/powerset.py, segmenter.py:178-186.

### 5.2 Frame ↔ sample conversion

Python uses float `SAMPLES_PER_FRAME = 160000 / 589 ≈ 271.6469` and converts
frame → sample via `int(round(f * SAMPLES_PER_FRAME))`. Rust uses pure integer
arithmetic.

#### 5.2.1 frame → sample (rounded)

```rust
const fn frame_to_sample(frame_idx: u32) -> u32 {
    let n = frame_idx as u64 * WINDOW_SAMPLES as u64;
    let half = (FRAMES_PER_WINDOW as u64) / 2;
    ((n + half) / FRAMES_PER_WINDOW as u64) as u32
}
```

Bit-for-bit equivalent to `round(f * 160000 / 589)` for any integer `f` —
`f * 160000` cannot land exactly on a half-integer multiple of `589` (589 is
odd), so the rounding mode is unambiguous.

#### 5.2.2 sample → frame (floor)

For computing the finalization boundary in §5.4.1 we need the inverse
direction. The boundary's correctness condition is "no future or pending
window can still contribute to a frame at or below the boundary." That
demands **floor** rounding — anything else either over-finalizes (admits a
frame that a future window will still touch) or under-finalizes (delays
draining without need).

```rust
const fn frame_index_of(sample_idx: u64) -> u64 {
    sample_idx * (FRAMES_PER_WINDOW as u64) / (WINDOW_SAMPLES as u64)
}
```

Worked example exposing the half-integer collision (where the sample-domain
position lands exactly between two frames):
- `step_samples = 40_000`, window-start at `k * step_samples`.
- For `k = 2`: `sample = 80_000`. `80_000 * 589 / 160_000 = 47_120_000 /
  160_000 = 294.5`. **Floor = 294.** Round-half-up would give 295,
  banker's-rounding 294. Floor is the only safe choice for boundary
  computation, and we use it here.

This direction is **not** the inverse of `frame_to_sample`; the two are an
asymmetric pair, and that is intentional.

### 5.3 Hysteresis

Two-threshold sequential state machine:

```
state = inactive
for p in probabilities:
    state = if state == active: p >= offset else: p >= onset
    yield state
```

At `p == offset` exactly, an active state stays active (the condition is
`p >= offset`, not `p > offset`). Matches Python.

### 5.4 Voice-timeline stitching (PER-FRAME)

Storage and computation happen at **frame rate**, not sample rate. The
accumulator is `(sum: VecDeque<f32>, count: VecDeque<u32>)` indexed by absolute
frame number.

**Per-hour storage:** `(3600 s × 16000 Hz) / 271.65 samples/frame ≈ 212 037
frames`. Each frame stores `f32` sum + `u32` count = 8 bytes, so per-hour
storage is **~1.7 MB**. (Per-sample storage would be 460 MB — a 270× larger
worst case.) The implementation MUST use per-frame storage.

After a window's frame range is appended, frames whose absolute index is below
the **finalization boundary** are eligible for draining via `mean = sum / count`
and downstream processing. Drained frames are popped from the front, freeing
memory.

**Tail-window overlap with the finalized region is silently dropped.** A tail
anchor whose start frame is before the current finalization boundary
contributes only to the suffix `[base_frame, end_frame)`.

**`warm_up_ratio` center-cropping** (Python's `stitching.py:94`) is **not
implemented** in v0.1.0 because the Python `Segmenter.segment` call site does
not pass it (default 0.0). Adding it would diverge from observed Python
behavior.

#### 5.4.1 Finalization boundary

The boundary is the smallest frame index that no future or pending window can
still contribute to:

```
boundary_frame = min(
    frame_index_of(next_window_idx * step_samples),    // smallest start of any future scheduled window
    min over pending: frame_index_of(window_start),    // smallest start of any in-flight pending window
)
```

…with one terminal case: if `finished == true` AND `pending` is empty, the
boundary is `total_frames`, defined as

```
total_frames = ceil(total_samples_pushed * FRAMES_PER_WINDOW / WINDOW_SAMPLES)
             = (total_samples_pushed * FRAMES_PER_WINDOW + WINDOW_SAMPLES - 1)
                / WINDOW_SAMPLES
```

— i.e. the smallest absolute frame index whose start sample is at or past the
end of pushed audio.

**Type-width note:** all window-index and sample arithmetic at this call
site is performed in `u64` (cast `next_window_idx as u64 * step_samples as u64`).
A pure-`u32` multiplication would overflow at `next_window_idx ≈ 107_374`, i.e.
roughly 75 hours of audio at default `step_samples = 40_000`.

This is critical for correctness under out-of-order `push_inference`: if
windows 0, 1, 2 are all pending and the caller's async pipeline returns scores
for window 2 first, the boundary stays clamped to window 0's start (the
earliest pending) — frames covered by both windows 1 and 2 are NOT drained
yet, because window 1 hasn't reported. Once window 0 (and 1) finally arrive,
the boundary advances and the now-fully-contributed frames drain. Without
this, an out-of-order pipeline would corrupt the running mean.

### 5.5 Activity extraction

For each window, after powerset decoding:

1. For each speaker slot, run a fresh hysteresis at frame rate (no
   cross-window state).
2. RLE the boolean mask into `[start_frame, end_frame)` runs.
3. Convert frame indices to absolute sample indices via `frame_to_sample` and
   the window's `start_sample`.
4. **Clamp `[start_sample, end_sample)` to `[0, total_samples]`** for tail
   windows after `finish()`. Without `finish()`, regular windows already
   guarantee they fall within buffered audio.
5. **Then** filter by `min_activity_duration`. Order matters: a clamp that
   shrinks a span below threshold drops it.

### 5.6 Voice-span finalization (PER-FRAME, then sample conversion)

A streaming hysteresis cursor maintains `Option<u64>` "current run start
frame" between drain passes. The cursor's hysteresis state (active/inactive)
also carries between drains.

Per `push_inference`:

1. Decode powerset → voice probabilities for the new window's frames, append
   to stitcher.
2. Recompute the finalization boundary (§5.4.1).
3. Drain finalized frames `[old_boundary, new_boundary)` from the stitcher.
4. For each drained frame, step the streaming hysteresis. Rising edge sets
   `cursor.run_start = Some(frame)`. Falling edge emits a span via §5.6.5
   below and clears the run start.
5. (No early emission while the run is open; the run continues across
   `push_inference` calls until a falling edge OR end-of-stream.)

Per `finish()`:

1. Schedule the tail window if needed (deduplicated against the last regular
   window's start).
2. Each subsequent `push_inference` follows the per-window flow above.

End-of-stream span closure (triggered on the LAST `push_inference` after
`finish` has been called AND `pending` is empty after that call):

1. Recompute the boundary; with `finished && pending.is_empty()` it is
   `total_frames`.
2. Drain all remaining frames; step hysteresis.
3. If the cursor still has `run_start = Some(start_frame)`, emit a final span
   `[start_frame, total_frames)` and clear.
4. Reset hysteresis to `inactive`.
5. **Flush the §5.6.5 merge cursor's pending span** (if any). The merge cursor
   buffers one span waiting to see whether the next span is close enough to
   merge — at end-of-stream there is no next span, so the buffered span must
   be emitted (subject to `min_voice_duration`).

#### 5.6.5 Span post-processing (per emission)

When the hysteresis state machine emits a `[start_frame, end_frame)` run:

1. Convert frame indices to absolute sample indices via `frame_to_sample`.
2. **Merge** spans separated by ≤ `voice_merge_gap` samples (the cursor
   buffers one pending span; emits it when the next span starts farther away
   than the gap, or on end-of-stream).
3. **Then** filter by `min_voice_duration`.

Order of merge vs filter matches Python (segmenter.py:249-263).

### 5.7 Voice spans are commit-only

Once `Action::VoiceSpan(r)` (or `Event::VoiceSpan(r)`) has been yielded by
`poll` / `process_samples`, the segmenter does **not** retract or revise it.
There is no `VoiceSpanRetract` event.

### 5.8 Voice-span emission timing

Latency measured here is **audio-time buffering latency only** — the time
between sample `e` arriving via `push_samples` and the segmenter being ready
to emit a voice span ending at `e` on the next `poll`. End-to-end emission
latency observed by the caller adds:

- ONNX inference time (one or more `model.infer` calls, each ~tens of ms),
- caller's async-pipeline scheduling (if `push_inference` is dispatched off
  the segmenter's thread),
- out-of-order completion (under T1-A above, the boundary stalls until the
  earliest pending window's scores arrive).

The audio-time bound:

```
latency_audio_time = WINDOW_SAMPLES - (e mod step_samples)
```

With defaults `WINDOW_SAMPLES = 160_000` and `step_samples = 40_000`, this
ranges from 7.5 s (when `e mod step_samples = step_samples - 1`) to **10 s**
(when `e mod step_samples = 0`).

Derivation: every sample `e` is covered by `⌈WINDOW_SAMPLES / step_samples⌉ =
4` overlapping windows. The LAST window covering `e` starts at
`floor(e / step_samples) * step_samples`. To emit that window as
`NeedsInference`, the segmenter needs buffered audio up to `start +
WINDOW_SAMPLES`, which is up to `e + (WINDOW_SAMPLES - (e mod step_samples))`.
Once that window's `push_inference` has been called and (under §5.4.1) all
earlier-pending windows have also reported, the boundary advances past `e`
and any voice run ending at `e` is emitted on the next `poll`.

The `min_voice_duration` filter does not extend this — once a span is
finalized it is emitted or dropped immediately.

## 6. Module layout

```
src/
├── lib.rs                         # crate-level docs, pub mod segment;
└── segment/
    ├── mod.rs                     # pub use of all public items + ort re-exports
    ├── types.rs                   # WindowId (with generation), SpeakerActivity, Action, Event
    ├── options.rs                 # SegmentOptions, SAMPLE_RATE_TB
    ├── error.rs                   # Error
    ├── powerset.rs                # softmax + 7→3 marginals + voice_prob (pure)
    ├── hysteresis.rs              # streaming Hysteresis + RLE (pure)
    ├── stitch.rs                  # frame_to_sample + per-frame VoiceStitcher (pure)
    ├── window.rs                  # plan_starts (pure, with tail dedup)
    ├── segmenter.rs               # Segmenter (Layer 1) — owns generation counter consumer
    └── model.rs                   # SegmentModel + Layer 2; cfg(feature = "ort")
```

The process-wide generation counter (`static GENERATION: AtomicU64`) lives in
`segmenter.rs`. `Segmenter::new` and `Segmenter::clear` consume it via
`fetch_add(1, Relaxed)`.

## 7. Crate metadata

- `name = "dia"`, `version = "0.1.0"`, `edition = "2024"`, `rust-version = "1.95"`
- License: `MIT OR Apache-2.0` (matches all four siblings)
- Lints (verbatim, copied from silero):
  ```toml
  [lints.rust]
  rust_2018_idioms = "warn"
  single_use_lifetimes = "warn"
  unexpected_cfgs = { level = "warn", check-cfg = [
    'cfg(all_tests)',
    'cfg(tarpaulin)',
  ] }
  ```
  No `missing_docs` lint.
- `[profile.bench]`: copy from any of the siblings (all identical).
- Dependencies:
  - `mediatime = "0.1"`
  - `thiserror = "2"` *(matches silero; `default-features` left on)*
  - `ort = { version = "2.0.0-rc.12", optional = true }`
- Features:
  - `default = ["std", "ort"]`
  - `std = []`
  - `ort = ["dep:ort"]`
- Dev-deps: `criterion = "0.8"`, `tempfile = "3"`, `hound = "3"`, `anyhow = "1"`
- Delete template stubs: `examples/foo.rs`, `benches/foo.rs`, `tests/foo.rs`

## 8. Streaming flow (canonical examples)

`examples/stream_layer1.rs` (no model, no ort): drives `Segmenter` directly
with a synthetic inferencer. Header comment documents the run command:

```
//! Run with:
//!     cargo run --no-default-features --features std --example stream_layer1
```

`examples/stream_from_wav.rs` (gated on `ort`): streams a 16 kHz mono WAV
file in 100 ms chunks. Default features include `ort`, so the run command is
just `cargo run --example stream_from_wav -- path/to/audio.wav`.

## 9. Testing strategy

### Unit tests (no model, no `ort`)

- `options.rs`, `powerset.rs`, `hysteresis.rs` (incl. `p == offset` boundary),
  `window.rs` (incl. step-aligned dedup), `stitch.rs` (incl. tail-window
  prefix-skip when start-frame < base-frame).
- `segmenter.rs`:
  - empty input → no actions.
  - first window emits after one full window of audio.
  - second window emits one step later.
  - `push_inference` with wrong shape → `InferenceShapeMismatch`.
  - `push_inference` with unknown id → `UnknownWindow`.
  - `push_inference` twice with same id: first succeeds, second
    `UnknownWindow`.
  - `push_inference` after `clear()` with a stale id: `UnknownWindow`
    (generation mismatch).
  - **Cross-segmenter id collision**: two `Segmenter::new` instances both
    yield `WindowId(range=0..160000)`; using `seg_a`'s id with
    `seg_b.push_inference` returns `UnknownWindow` (different generations).
  - Synthetic-A-active scores → at least one slot-0 `Activity`.
  - `finish()` on a sub-window clip schedules zero-padded tail.
  - `clear()` resets all state; new generation gets a different counter.
  - End-of-stream closes an open voice span.
  - Tail-window activity range is clamped to `total_samples`.
  - `voice_merge_gap` merges before `min_voice_duration` drops.
  - `pending_inferences()` and `buffered_samples()` track state correctly.
  - **Out-of-order push_inference**: schedule windows 0, 1, 2; push
    inference for window 2 first; assert no events emitted until window 0
    (and 1) push_inference complete. Then drain.
  - Voice-span emission timing: feed scores progressively; assert each span
    appears no earlier than after its last covering window's
    `push_inference` AND any earlier-pending windows' `push_inference`.
  - Voice span at exact stream boundary: `finish` while a run is active →
    span closes at `total_samples`.

### Integration test (gated)

`tests/integration_segment.rs` with `#[ignore]`. Asserts no panics + output
shape (slots in 0..3, voice spans monotonic). Does NOT assert specific
content — model output on noise is unspecified.

### Compile-time trait assertions

Add a small block (e.g. `tests/trait_bounds.rs` or a `cfg(test)` module
in `lib.rs`) that fails the build if the auto-derive story drifts:

```rust
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<diarization::segment::Segmenter>();

    #[cfg(feature = "ort")]
    assert_send::<diarization::segment::SegmentModel>();
};
```

The `!Sync` property of `SegmentModel` is enforced by `ort::Session`'s
own `!Sync`-ness; if a future field change accidentally gives
`SegmentModel` a `Sync` impl, asserting the negative is awkward without
the `static_assertions` crate. We accept that risk in v0.1.0 — the spec
inherits silero's posture and any drift would be a deliberate code change
caught in review.

### Benches

`benches/segment.rs` — Layer 1 throughput on one minute of audio with
synthetic scores.

## 10. Threading and lifecycle

- **`Segmenter`** (Layer 1): auto-derived `Send + Sync`. All fields are
  `Send + Sync`. Primarily driven through `&mut self` so `Sync` is incidental.
  Differs from `silero::Session` (which is `!Sync` because of
  `ort::Session`) — see §2.1 table.
- **`SegmentModel`** (Layer 2): auto-derives `Send`; **does NOT auto-derive
  `Sync`** because `ort::Session` is `!Sync`. **Matches `silero::Session`
  exactly** (silero/src/session.rs line 61: "Send but not Sync"). Use one
  per worker.
- No internal locks. No async. Callers wrap with their own runtime if desired.
- `clear()` resets `Segmenter` state, increments the process-wide generation
  counter, but does **not** discard or reload the paired `SegmentModel`.
  The ort session stays warm — for services processing many short clips this
  is the intended pattern.
- **Determinism**: ort inference on CPU with `intra_op_num_threads > 1` is
  not bit-exact across runs (parallel reductions). Set `intra_op_num_threads
  = 1` in `SegmentModelOptions` for reproducible results.

## 11. Resolved contracts

### 11.1 Voice-span emission timing

§5.8. Audio-time worst-case latency `WINDOW_SAMPLES - (e mod step_samples)`,
ranging 7.5–10 s with defaults. Excludes inference and async-pipeline
delays.

### 11.2 Backpressure

`push_samples` allocates without bound. Detect via
`Segmenter::buffered_samples()`. `pending_inferences()` reports the
count of NeedsInference actions yielded but not yet fulfilled. Canonical
caller pattern documented in `Segmenter` rustdoc.

### 11.3 `min_voice_duration` interaction with stream emission

Spans are filtered (or merged-then-filtered, per §5.6) at finalization,
not held back. A span whose finalized duration is below the threshold is
dropped immediately.

### 11.4 Sample-rate validation

`Segmenter::push_samples(&[f32])` does not validate sample rate. Caller's
responsibility. Matches silero/scenesdetect/soundevents practice.

### 11.5 Callback error propagation

`process_samples<F: FnMut(Event)>` and `finish_stream<F: FnMut(Event)>`
mirror `silero::Session::process_stream<F: FnMut(f32)>` — callbacks can't
return errors. Callers can capture an `Option<MyError>` by mutable reference
from inside the closure if they need to surface a failure post-call.

### 11.6 Streaming chunk size

Any chunk size accepted by `push_samples`; the segmenter buffers internally
and schedules `NeedsInference` actions as full windows become available.

### 11.7 Sub-window-length clips and the empty/sub-window threshold

A clip shorter than `WINDOW_SAMPLES` is handled by `finish()`: a tail
window is anchored at sample 0 with available samples followed by zero
padding, and inference runs on the padded buffer.

**Threshold rule:** the segmenter maintains a private cumulative counter
`total_samples_pushed: u64` that is incremented on every `push_samples`
call by `samples.len()`. The counter is **never decremented** —
window-driven trimming of the input buffer (which removes consumed
samples) does not affect it. `finish()` schedules a tail window iff
`total_samples_pushed > 0`.

Equivalent formulation: at least one non-empty `push_samples` call must
have happened before `finish()`. An empty stream — zero `push_samples`
calls, or only empty-slice calls — produces zero windows.

`clear()` resets `total_samples_pushed` to 0.

### 11.8 Multi-stream `clear()` warmup

`clear()` does not discard or warm-down the ONNX session — that lives on
the `SegmentModel`. Construct the model once, reuse a `Segmenter` via
`clear` per clip.

### 11.9 Stale-id rejection (process-wide)

The dia process holds one `static GENERATION: AtomicU64 = AtomicU64::new(0)`
counter. Every `Segmenter::new` and every `Segmenter::clear` performs
`GENERATION.fetch_add(1, Relaxed)` and stores the result on `self`. Each
yielded `WindowId` carries that current value.

**Memory ordering:** `Relaxed` is sufficient because the counter values are
not used to synchronize any other memory; their only purpose is to provide
a unique token. Each `Segmenter` reads the value once at construction or
clear, stores it in a local field, and consults it from then on under
`&mut self`. There is no happens-before relationship across `Segmenter`
instances that needs to be established by the atomic.

`push_inference` checks `id.generation == self.generation`. Mismatch ⇒
`Error::UnknownWindow`. This closes both corruption scenarios:

1. **Stale-after-clear in the same `Segmenter`:** old generation, mismatch,
   reject.
2. **Cross-`Segmenter` ID collision** (two segmenters both yielding
   `WindowId(range=0..160000)` because both started at `next_window_idx=0`):
   they got distinct generations from the atomic, so the structural ranges
   may match but generations don't — reject.

The counter wraps at `2^64`. At 10⁹ `clear()` calls per second the wrap
takes ~600 years; we treat overflow as not-a-concern. (`fetch_add` wraps
silently; if a wrap ever lined up with surviving stale `WindowId`s the
mismatch behavior would degrade, but the timescale precludes it.)

`WindowId::generation()` is **not exposed** on the public API — the counter
is opaque. `Debug` exposes it for diagnostics. `Ord` orders by
`(generation, start_pts)`; cross-generation ordering is deterministic but
semantically meaningless and should not be used for sample-position comparisons.

### 11.10 Empty-stream behavior

`Segmenter::new(opts)` followed by `finish()` with no `push_samples` calls
(or only empty-slice calls) produces zero `Action`s and zero `Event`s.
`pending_inferences()` and `buffered_samples()` stay at 0. See §11.7 for
the precise threshold.

## 12. Decision log (cumulative)

From original brainstorming:
- Single crate `dia`, two modules (`segment`, `embed`); not a workspace.
- Segmentation first; embedding is a separate sub-project.
- Sans-I/O dual layer: state machine has zero `ort` dep; `ort` is a default-on
  feature for the convenience driver.
- `WindowId` is a newtype-ish handle backed by `TimeRange`.
- `VoiceSpan` is just `TimeRange` directly.
- Public types use private fields with accessor methods.
- `mediatime` provides every time/range/duration value crossing the API.
- Edition 2024, Rust 1.95.

Added in Revision 2 (post-review-1):
- `WindowId` impls `Ord`/`PartialOrd` manually.
- Algorithm semantics promoted to load-bearing §5.
- Voice spans are commit-only.
- Voice-span emission latency is bounded.
- Layer 2 gains `SegmentModelOptions`.
- ONNX shape verification at model load.
- `pending_inferences()`, `buffered_samples()` introspection.

Added/fixed in Revision 3 (post-review-2):
- `WindowId` gains a generation counter (per-Segmenter at the time).
- Stitching/hysteresis/RLE pinned to per-frame.
- Latency bound corrected to `WINDOW_SAMPLES - (e mod step)`.
- Tautological `static_assert`s removed.
- `Error` regains `#[derive(thiserror::Error)]`.
- `Sync` claim added (incorrectly — corrected in rev 4).
- `serde` feature deferred; `alloc` feature removed.
- §2.1 sibling comparison table introduced.
- Streaming RLE / open-span lifecycle, empty-stream, push_inference idempotence
  documented.
- `ort` types re-exported.

Added/fixed in Revision 5 (post-review-4):
- §5.2 gains a `frame_index_of(sample) -> u64` definition with explicit
  floor rounding and a worked example exposing the half-integer collision
  for window starts at multiples of `step_samples`. Rev 4's §5.4.1
  boundary formula referenced this function without defining it.
- §4 `push_inference` rustdoc restored — the four `UnknownWindow`
  scenarios (never-yielded / already-consumed / stale-after-clear /
  cross-segmenter-collision) are now spelled out at the API definition,
  not only buried in §11.9. Rev 4 had condensed this away.
- §11.7 threshold rule pinned to a precise `total_samples_pushed: u64`
  cumulative counter that is never decremented and resets only on
  `clear()`. Rev 4's "input buffer length" wording was ambiguous about
  whether window-driven trimming counted.
- §9 gains a compile-time `Send + Sync` assertion block for `Segmenter`
  and a `Send` assertion for `SegmentModel`. The `!Sync` of
  `SegmentModel` rides on `ort::Session` and is not asserted explicitly
  (would require `static_assertions` dev-dep — not worth it).
- §11.9 spells out why `Relaxed` ordering is correct (the counter only
  provides uniqueness; no other memory is synchronized through it).
- §10 + §4 wording: "auto-derives `Send`; does NOT auto-derive `Sync`"
  replaces "auto-derived `Send`, NOT `Sync`" — the previous phrasing read
  as if `!Sync` were being actively asserted, which auto-derive does not do.
- §4 `push_samples` docstring: explicit "empty slice is a no-op and does
  NOT count toward the §11.7 tail-window threshold."
- §15 #10 rewords "Send + !Sync" (which would be invalid Rust syntax) to
  "Send (and not Sync)".

Added/fixed in Revision 4 (post-review-3):
- **Generation counter is now process-wide via `AtomicU64`**, not per-Segmenter,
  so cross-segmenter ID collisions are also rejected (§11.9).
- **Finalization boundary now incorporates pending windows' starts**
  (§5.4.1), preventing out-of-order `push_inference` from advancing the
  boundary past frames whose other contributing windows haven't reported.
- **`Sync` story corrected**: `Segmenter` (Layer 1, no ort) is auto
  `Send + Sync`. `SegmentModel` (Layer 2, owns `ort::Session`) is auto
  `Send` but **not `Sync`** — matches `silero::Session` precisely
  (silero/src/session.rs line 61). The previous "Sync everywhere" framing
  was wrong.
- **Per-hour memory math corrected** from "~849 KB" to "~1.7 MB" (off by
  2× because the previous figure didn't include the `count` array).
- **`ExecutionProviderDispatch` re-export documented as a deliberate
  divergence from silero** (silero re-exports only `GraphOptimizationLevel`).
- **`thiserror = "2"` keeps default features on** (matches silero), reverting
  rev 3's `default-features = false` (which only matched scenesdetect).
- **§1 framing corrected**: dia adopts `mediatime` to share the choice with
  `scenesdetect`; `silero` and `soundevents` use raw types.
- **§2.1 table gains a `soundevents` column** showing its batch-only / no-mediatime
  divergence.
- **`WindowId::generation()` removed from public API** (now opaque, exposed
  only via `Debug`).
- **§5.6 lifecycle sequenced explicitly** for `push_inference` and `finish()`
  flows, including the end-of-stream span closure trigger.
- **§5.8 latency claim re-scoped** as audio-time-buffering only, with the
  additive caller-side components (inference time, async scheduling,
  out-of-order completion) called out.
- **§11.7 / §11.10 threshold rule** between empty-stream and sub-window-stream
  pinned to a precise check.
- **Generation overflow policy** stated explicitly (wraps at 2^64, ~600
  years at 10⁹ clears/s).
- §15 smoke-test item dropped (silero source confirms `commit_from_memory`
  works in rc.12; no validation needed).

## 13. Revision history

- **Revision 1** (2026-04-25): Initial spec; implemented in commits
  `25f817c` … `18634b5`.
- **Revision 2**: review-1 feedback — algorithm semantics § promoted to
  load-bearing, open questions resolved, Layer 2 options added.
- **Revision 3**: review-2 feedback — generation counter, per-frame
  stitching, latency bound corrected, static asserts dropped, thiserror
  restored, serde deferred.
- **Revision 4**: review-3 feedback. Generation counter promoted to
  process-wide atomic; finalization boundary made pending-aware; `Sync`
  story corrected against silero/src/session.rs:61; memory math fixed.
- **Revision 5** (this document): review-4 feedback fixed
  (`frame_index_of`, `push_inference` rustdoc, `total_samples`
  semantics, Send/Sync wording, etc.). Plus a small post-sign-off polish
  pass adding items 24-27 to §15 (formal `total_frames` definition,
  u64-arithmetic note, end-of-stream merge-cursor flush, mediatime
  const-fn verification). Review 5 verdict: **QUALIFIED. Ship it.** No
  further revisions planned — implementation feedback rolls into a
  CHANGELOG, not a Rev 6.

## 14. Findings rejected (cumulative)

### From review 4

- **T3-B** (generation counter location, `segmenter.rs` vs `types.rs`):
  rejected as a spec-level concern. The static lives where its consumer
  lives; the implementation is free to relocate. Spec stays neutral.
- **T3-C** (AtomicU64 const init): non-issue. Implementer-confirmation
  note from the reviewer.

(T2-C "frame index width" is not listed here because the reviewer
*confirmed* `u64`; that's a non-finding, captured implicitly in §5
where every absolute frame/sample index is already typed `u64`.)

### From review 3

- **CONSENSUS-C (`ExecutionProviderDispatch` re-export gratuitous)**:
  partially rejected. We keep the re-export because the public API takes
  `Vec<ExecutionProviderDispatch>` and forcing callers to import `ort`
  directly defeats the point. The "silero doesn't do this" critique is
  valid and now documented in §4 as a deliberate divergence.
- **T2-A-rejection (latency claim)**: not rejected — adopted (rescoped to
  audio-time-only, additive components documented).
- **T3-A (`generation()` public)**: adopted — hidden from public API.
- **T3-B (overflow)**: adopted — documented in §11.9.
- **T3-C (cross-gen Ord)**: adopted — documented in §11.9.
- **T3-D (smoke test redundant)**: adopted — §15 item dropped.

Findings carried over from review 2 (still rejected):

- T1-2, T3-5, T3-6 (silero `commit_from_memory` / `FnMut` / `from_memory`
  unverified): silero/src/session.rs:120 / :297 confirm both. Sibling-validated.
- T2-7 (no soft-cap enforcement): introspection-only matches silero.
- T4-1 (`Segmenter::default`): not worth API churn for a one-keystroke save.
- T4-6 (edition 2024 unused): sibling consistency.

## 15. Action list for v0.1.1 patches

Spec → impl reconciliation. Items 1–4 are correctness fixes; the rest are
API/documentation cleanups.

| #  | Item                                                                                              | Severity |
| -- | ------------------------------------------------------------------------------------------------- | -------- |
| 1  | `WindowId` carries a `generation: u64`; **process-wide** `static AtomicU64` bumped on `Segmenter::new` AND `clear()`; `push_inference` rejects on generation mismatch. (§11.9)                       | critical |
| 2  | `VoiceStitcher` is per-frame (not per-sample). Hysteresis and RLE run on frames; `frame_to_sample` happens at emission only. (§5.4)                                                                  | critical |
| 3  | **Finalization boundary** = `min(next_window_start_in_frames, min over pending of start_frame)` (with `total_frames` after finished+empty). (§5.4.1)                                                  | critical |
| 4  | Implement `voice_merge_gap` (currently dead option). (§5.6.5)                                    | critical |
| 5  | Clamp tail-window activity ranges to `total_samples`. (§5.5)                                     | critical |
| 6  | Add ONNX shape verification at `SegmentModel::from_file` / `from_memory`. (§4)                    | high     |
| 7  | Add `Segmenter::pending_inferences()` and `buffered_samples()`. (§4)                              | high     |
| 8  | Add `SegmentModelOptions` with `with_optimization_level`, `with_providers`, `with_intra_op_num_threads`, `with_inter_op_num_threads`; re-export `GraphOptimizationLevel` and `ExecutionProviderDispatch` from `diarization::segment`. (§4) | high     |
| 9  | Restore `#[derive(thiserror::Error)]` on `Error`; remove manual `Display`/`std::error::Error` impls. (§4)                                                                                            | high     |
| 10 | **`SegmentModel` should be `Send` (and not `Sync`)**, matching silero. The current impl is auto-derived from `ort::Session` which is already `!Sync`, so confirm rather than change. **No `PhantomData` needed for either type** (`Segmenter` stays `Send + Sync` via auto-derive). (§10) | high     |
| 11 | `Cargo.toml` revert to `thiserror = "2"` (drop `default-features = false`). (§7)                  | medium   |
| 12 | Simplify `WindowId::Ord` to `(generation, start_pts)`; document cross-generation ordering as deterministic-but-not-semantic. (§4)                                                                    | medium   |
| 13 | Make `WindowId::duration()` `const fn`; **remove the `pub fn generation()` accessor** (keep generation visible via `Debug` only). (§4)                                                               | medium   |
| 14 | Remove the `alloc` Cargo feature. The `serde` Cargo feature stays out of v0.1.0. (§7)             | medium   |
| 15 | Add the streaming-specific tests listed in §9: out-of-order `push_inference`, cross-segmenter id collision, exact-step-boundary tail dedup, voice-span exact-boundary closure on `finish`.            | medium   |
| 16 | Document `push_inference` semantics, `clear()` drops pending, canonical backpressure pattern in rustdoc. (§4, §11.2)                                                                                  | low      |
| 17 | Implement `frame_index_of(sample) -> u64` (floor) per §5.2.2. Add a unit test pinning the half-integer collision case: `frame_index_of(80_000) == 294`.                                              | critical |
| 18 | The `push_inference` rustdoc must enumerate the four `UnknownWindow` scenarios (never-yielded / already-consumed / stale-after-clear / cross-segmenter-collision). Verbatim from §4.                  | high     |
| 19 | Implement `total_samples_pushed: u64` cumulative counter on `Segmenter`; increment on every `push_samples`; reset on `clear()`; never decrement. Use it for the §11.7 threshold check.                | high     |
| 20 | **Verify** `pub use ort::execution_providers::ExecutionProviderDispatch;` resolves on `ort = "2.0.0-rc.12"`. The path was not exercised by silero (which only re-exports `GraphOptimizationLevel`); if rc.12 has shifted the module, find the correct path. | high     |
| 21 | Add the `Relaxed`-ordering rationale comment near the `static GENERATION` definition (per §11.9).                                                                                                     | medium   |
| 22 | Add the compile-time `Send` / `Send + Sync` assertion block from §9 ("Compile-time trait assertions").                                                                                                | medium   |
| 23 | Document `push_samples(&[])` as no-op in the rustdoc (per §4).                                                                                                                                        | low      |
| 24 | Implement `total_frames = ceil(total_samples_pushed * FRAMES_PER_WINDOW / WINDOW_SAMPLES)` per §5.4.1 terminal-case definition.                                                                       | low      |
| 25 | Cast `next_window_idx` and `step_samples` to `u64` at the boundary call site (already done in the current impl; verify on impl review). Per §5.4.1's type-width note.                                  | low      |
| 26 | At end-of-stream span closure (§5.6 step 5), flush the §5.6.5 merge cursor's pending span before exiting the drain.                                                                                   | low      |
| 27 | Verify that `mediatime`'s `TimeRange` / `Timestamp` / `core::time::Duration` accessors used by `WindowId` / `SpeakerActivity` / `SegmentOptions` are usable from `const fn` context in `mediatime = "0.1"`. If any aren't, drop `const` from those accessors in §4. | low      |

Items 1, 2, 3, 4, 5, 10, 17 are the correctness must-haves. The rest are
ergonomic/documentation polish surfaced by reviews 1-5.
