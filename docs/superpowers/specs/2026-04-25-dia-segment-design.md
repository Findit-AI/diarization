# dia — segmentation sub-project design (v0.1.0)

**Revision 3** (2026-04-25, post-second-adversarial-review).
**Status:** ready for re-review.
**Scope:** The `dia::segment` module only. `dia::embed` is a separate sub-project to be specced after this one ships.

> **Revision 3 fixes** are summarized in §14, full action list and pushbacks
> in §15. The big ones: `WindowId` gains a generation counter so stale
> async push_inference calls can no longer corrupt a `clear()`'d stream;
> stitching and hysteresis are pinned to per-FRAME storage (not per-sample,
> which the previous wording incorrectly suggested); voice-span emission
> latency bound is corrected from `step` to `WINDOW_SAMPLES`; the
> mathematically-tautological static asserts are removed; `Sync` is now
> auto-derived (matching silero) instead of falsely claimed "not Sync"; the
> `serde` feature is deferred to v0.2.

## 1. Context

`dia` is a Rust port of two findit-studio Python projects: `findit-pyannote-seg` (speaker
segmentation, M1) and `findit-speaker-embedding` (speaker embedding, M2). The single
`dia` crate will host both as modules: `dia::segment` and `dia::embed`. **This spec
covers `dia::segment` only.**

The crate sits alongside `silero` (VAD), `soundevents` (sound-event classification),
`scenesdetect` (scene/shot boundaries), and `mediatime` (rational time types) in the
findit-studio Rust suite. It targets stream-processing audio applications that ingest
16 kHz mono PCM samples frame-by-frame and consume diarization events.

## 2. Architecture: Sans-I/O dual layer

The architectural philosophy is borrowed from `scenesdetect` (state-machine-only,
zero internal I/O) but adapted to acknowledge that segmentation MUST run a neural
network. The compromise: a Sans-I/O state machine with no `ort` dependency
(Layer 1), plus a separate Layer 2 that pairs it with an `ort::Session`.

### 2.1 What we kept and what we changed vs siblings

| Pattern                                | scenesdetect  | silero         | dia (this spec)                                                            |
| -------------------------------------- | ------------- | -------------- | -------------------------------------------------------------------------- |
| Owns model session?                    | n/a (no ML)   | yes (`Session`)| **No** — Layer 1 has no `ort` dep; Layer 2's `SegmentModel` does           |
| State machine and session in one type? | n/a           | yes            | **No** — `Segmenter` (Layer 1) and `SegmentModel` (Layer 2) are separate   |
| Streaming push API                     | `process(frame) -> Option<Cut>` | `Session::process_stream(stream, samples, FnMut(f32))` | `Segmenter::push_samples` + `poll` (Layer 1) plus `Segmenter::process_samples(model, samples, FnMut(Event))` (Layer 2)  |
| Inference request as an explicit action| n/a (no model)| no (internal) | **Yes** — `Action::NeedsInference` round-trips via the caller              |
| Output type                            | `Option<Timestamp>` | `f32` per frame | **Two** — `Action` (Layer 1, includes inference requests) / `Event` (Layer 2, emission only) |
| Worker-pool friendliness               | yes (no I/O)  | yes (one session per worker) | yes (one Segmenter per stream; one SegmentModel per worker) |
| `Send`                                 | yes (auto)    | yes (auto)     | yes (auto)                                                                 |
| `Sync`                                 | yes (auto)    | yes (auto)     | yes (auto, matches silero — see §10)                                       |

The novel pieces are the `Action` / `Event` enum split (so inference is the
caller's concern, not the library's) and the explicit `WindowId` round-trip
(below). Everything else is the silero shape.

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
- Stale-id rejection across `clear()` via a generation counter on `WindowId`
  (§4 / §11.9).
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
- `mediatime`-based time types throughout.
- ONNX model shape verification at `from_file`/`from_memory` load time
  (returns `Error::IncompatibleModel` if input or output dims don't match
  `[*, 1, 160000]` / `[*, 589, 7]`).
- `SegmentModel` configuration for execution providers and graph optimization
  level (CPU + level 3 by default), via re-exported `ort` types.
- Pure-CPU unit tests on the state machine with hand-crafted scores
  (no model file required).
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
| Cross-window speaker clustering / global IDs    | Future M3 layer (downstream of `dia::embed`); not segmentation's job.                                 |
| `infer_batch` for cross-stream batching         | Not needed at CPU `intra_threads=1` (batch=1 is already the fastest path on M1). Add for GPU/HT-CPU.  |
| `IoBinding` / `infer_into(&mut [f32; 4123])`    | ~16 KB allocation per 2.5 s of audio; not a hot path. Add when a profile points here.                 |
| `Arc<[f32]>` / borrow-token in `NeedsInference` | Same — alloc cost is small; design churn is large.                                                    |
| **`serde` feature for output types**            | **Deferred entirely from v0.1.0.** Re-introducing the feature requires `#[cfg_attr(serde, derive(Serialize, Deserialize))]` on `WindowId`, `SpeakerActivity`, `Action`, `Event`, and threading `mediatime/serde`. Half-wiring it is worse than not having it. |
| `ControlFlow<E>` / `Result<()>` user callbacks  | Silero ships `FnMut(f32)` (no error propagation); we follow that idiom. Users `.set` an `Option` from inside the closure if they need to surface an error after the call returns. (Citation verified at [silero/src/session.rs:297](../../../../silero/src/session.rs).) |
| `step_samples` typed as `Duration` rather than `u32` | Aesthetic cleanup; keeps churn out of v0.1.0. Pick at v0.2 / v1.0.                              |
| `drain()` iterator adapter                      | `while let Some(action) = seg.poll()` is fine and explicit. Add when ergonomics demand it.            |
| `SamplesAt16k<'_>` typed input                  | Type-safe but pollutes every call site. Caller-must-enforce matches silero/scenesdetect/soundevents.  |
| Soft-cap `try_push_samples` returning `Backpressure` | `pending_inferences()` and `buffered_samples()` are introspection-only in v0.1.0. Add a cap if a caller actually hits a runaway-buffer scenario. |
| `Segmenter::default()` impl                     | Always paired with explicit `SegmentOptions`. Add if convenience demands.                             |
| `alloc` Cargo feature                           | Removed — the crate is unconditionally `alloc`-required (uses `Vec`, `Box`, `BTreeMap`, `VecDeque`). The `default = ["std", "ort"]` features are sufficient. |

### Out of scope (handled elsewhere)

- Speaker embedding — `dia::embed`, separate sub-project.
- Resampling — input must be 16 kHz mono float32; caller enforces.
- Audio-file decoding — caller's responsibility.
- Threading / worker pools — caller's responsibility.

## 4. Public API surface

### Constants

```rust
/// 16 kHz monolithic timebase used for every `TimeRange` and `Timestamp`
/// emitted by `dia::segment`. pyannote-seg-3.0 only operates at 16 kHz.
pub const SAMPLE_RATE_TB: Timebase;

pub const SAMPLE_RATE_HZ: u32 = 16_000;
pub const WINDOW_SAMPLES: u32 = 160_000;        // 10 s @ 16 kHz
pub const FRAMES_PER_WINDOW: usize = 589;
pub const POWERSET_CLASSES: usize = 7;          // silence, A, B, C, A+B, A+C, B+C
pub const MAX_SPEAKER_SLOTS: u8 = 3;
```

No tautological static asserts. The actual schema-drift detector is
runtime ONNX shape verification at `SegmentModel::from_file`.

### `ort` re-exports (feature-gated)

```rust
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::execution_providers::ExecutionProviderDispatch;
```

We re-export `ort` types rather than wrap them so callers can compose
provider dispatches without a translation layer. Trade-off: enabling the `ort`
feature pulls `ort` into the public type graph.

### Types

```rust
/// Stable correlation handle for one inference round-trip. Carries (1) the
/// window's sample range in `SAMPLE_RATE_TB` and (2) a generation counter
/// that increments on every `Segmenter::clear()`. Two `WindowId`s minted by
/// the same generation compare equal iff their ranges are equal; across
/// generations they always compare unequal.
///
/// The generation counter prevents the following async-pipeline bug: an
/// inference response from a previous (cleared) stream silently matching a
/// new stream's pending window because both have the same sample range
/// `(0, 160_000)`. After `clear()`, every previously-issued `WindowId`
/// fails `push_inference` with `Error::UnknownWindow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId { /* private: range + generation */ }

impl WindowId {
    pub const fn range(&self)    -> TimeRange;
    pub const fn start(&self)    -> Timestamp;
    pub const fn end(&self)      -> Timestamp;
    pub const fn duration(&self) -> Duration;
    /// Internal generation counter exposed for diagnostics. Equal across
    /// the lifetime of one stream; bumped on `clear()`.
    pub const fn generation(&self) -> u64;
}

// Ord by (generation, start_pts). End-PTS is always start_pts + W and so
// adds no information. (Generation orders cross-stream IDs deterministically
// even though they should never compare semantically.)
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
/// **Style note:** enum-variant fields (`id`, `samples`) are public because
/// they participate in pattern matching, which is the standard Rust enum
/// idiom. Structs with invariants (`WindowId`, `SpeakerActivity`) use
/// private fields with accessors. The two conventions coexist deliberately.
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
step 40 000 samples, all duration filters disabled.

**Validation:** `SegmentOptions` builders/setters are infallible — they
store values verbatim. Bad values (e.g., offset >= onset) produce wrong
output but no panics or errors. `Error::InvalidOptions` is reserved for
future eager validation; v0.1.0 ships without it. Callers that want
validation can wrap construction in their own check.

### `Segmenter` (Layer 1)

```rust
pub struct Segmenter { /* private */ }

impl Segmenter {
    pub fn new(opts: SegmentOptions) -> Self;
    pub fn options(&self) -> &SegmentOptions;

    /// Append 16 kHz mono float32 PCM. Arbitrary chunk size.
    /// CALLER MUST ENFORCE SAMPLE RATE — there is no runtime guard.
    pub fn push_samples(&mut self, samples: &[f32]);

    /// Drain the next action. `None` means "nothing ready right now"; it
    /// does NOT mean "done" — the caller decides when the stream ends by
    /// calling `finish()`.
    pub fn poll(&mut self) -> Option<Action>;

    /// Provide ONNX scores for a previously-yielded `NeedsInference`.
    /// `scores.len()` must equal `FRAMES_PER_WINDOW * POWERSET_CLASSES = 4123`.
    ///
    /// Returns `Error::UnknownWindow` if `id` does not match a currently
    /// pending window. This covers four scenarios:
    /// 1. `id` was never yielded by `poll`.
    /// 2. `id` was already consumed by an earlier `push_inference` call
    ///    (each pending entry is consumed exactly once).
    /// 3. `id` came from a previous stream that was reset by `clear()`
    ///    (caught by the generation counter — see `WindowId`).
    /// 4. `id` happens to share a sample range with a current pending
    ///    window but has a different generation.
    pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error>;

    /// Signal end-of-stream. Schedules a tail-anchored window if needed
    /// (deduplicated against the last regular window's start). The closing
    /// voice span (if any) is emitted as soon as the last pending inference
    /// is fulfilled (or immediately, if there is none).
    pub fn finish(&mut self);

    /// Reset to empty state for a new stream:
    /// - input buffer cleared,
    /// - pending inferences dropped (subsequent `push_inference` for those
    ///   IDs returns `Error::UnknownWindow` because the generation counter
    ///   has advanced),
    /// - voice/hysteresis state reset,
    /// - `finished` / `tail_emitted` flags cleared.
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
    /// some not-yet-scheduled or in-flight window). Useful for detecting
    /// pathological caller backpressure (e.g. `push_samples` without ever
    /// calling `poll`).
    pub fn buffered_samples(&self) -> usize;
}
```

Auto-derived `Send` and **`Sync`** (matches silero — see §10). The state
machine is behind `&mut self`, so no synchronization protocol is implied
by `Sync`; we don't impose one.

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

    /// Run inference on one 160 000-sample window. Returns the flattened
    /// `[FRAMES_PER_WINDOW * POWERSET_CLASSES] = [4123]` logits.
    /// Heap-allocates the output `Vec<f32>`; v0.1.0 ships this surface.
    /// A reusable-buffer fast path is deferred (see §3 "Deferred").
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
internally (via `ort::Session::Builder::commit_from_memory`, sibling-validated at
[silero/src/session.rs](../../../../silero/src/session.rs)). The buffer can be
dropped immediately after the constructor returns.

`SegmentModel` validates input/output shapes at construction:
- input dims must match `[*, 1, 160000]` (dynamic batch axis allowed)
- output dims must match `[*, 589, 7]`

A wrong-architecture model fails at load with `Error::IncompatibleModel`, not
on first inference.

`SegmentModel` is `Send` (auto-derived from `ort::Session`) but not `Sync`. Use
one per worker.

**Scratch ownership:** `SegmentModel` owns the input scratch (`Vec<f32>` reused
across `infer` calls). Output `Vec<f32>` is allocated per call; reuse is
deferred (see §3 "Deferred"). `Segmenter` owns its own pre-allocated buffers
for the state machine; no scratch is shared between layers.

### Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Construction-time validation of `SegmentOptions`. Reserved for
    /// future eager validation; not currently emitted (v0.1.0 stores
    /// option values verbatim).
    #[error("invalid segment options: {0}")]
    InvalidOptions(&'static str),

    /// `push_inference` received the wrong number of scores.
    #[error("inference scores length {got}, expected {expected}")]
    InferenceShapeMismatch { expected: usize, got: usize },

    /// `push_inference` was called with a `WindowId` that is not in the
    /// pending set. See `Segmenter::push_inference` rustdoc for the four
    /// scenarios this covers.
    #[error("inference scores received for unknown WindowId {id:?}")]
    UnknownWindow { id: WindowId },

    /// Loaded model's input/output dims don't match what we expect.
    #[cfg(feature = "ort")]
    #[error("model {tensor} dims {got:?}, expected {expected:?}")]
    IncompatibleModel { tensor: &'static str, expected: &'static [i64], got: Vec<i64> },

    #[cfg(feature = "ort")]
    #[error("failed to load model from {path}: {source}")]
    LoadModel {
        path: PathBuf,                   // implicitly std-only via cfg(ort)
        #[source] source: ort::Error,
    },

    #[cfg(feature = "ort")]
    #[error(transparent)]
    Ort(#[from] ort::Error),
}
```

`thiserror`'s no_std support is pulled in via `default-features = false` so
the derive works whether or not `feature = "std"` is set on dia. The
`std::error::Error` impl follows `cfg(feature = "std")`.

## 5. Algorithm semantics (load-bearing decisions)

Every decision below was deliberately copied from the Python reference. Future
work that diverges from these must be deliberate and documented.

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
3. Voice probability (any-speaker active) is computed **separately** from the
   per-speaker marginals, as `1 - p[silence] = 1 - p[0]`. It is *not* derived
   from the per-speaker probabilities (which would over-count overlap classes).

Reference: [findit-pyannote-seg/powerset.py](../../../../findit-pyannote-seg/src/findit_pyannote_seg/powerset.py),
[segmenter.py:178-186](../../../../findit-pyannote-seg/src/findit_pyannote_seg/segmenter.py).

### 5.2 Frame-to-sample conversion

Python uses float `SAMPLES_PER_FRAME = 160000 / 589 ≈ 271.6468` and converts
via `int(round(f * SAMPLES_PER_FRAME))`. Rust uses pure integer arithmetic:

```rust
const fn frame_to_sample(frame_idx: u32) -> u32 {
    let n = frame_idx as u64 * WINDOW_SAMPLES as u64;
    let half = (FRAMES_PER_WINDOW as u64) / 2;
    ((n + half) / FRAMES_PER_WINDOW as u64) as u32
}
```

This is bit-for-bit equivalent to `round(f * 160000 / 589)` for any integer
`f` because `f * 160000` can never land exactly on a half-integer multiple of
`589` (589 is odd). No rounding-mode divergence between Rust and Python is
possible.

### 5.3 Hysteresis

Two-threshold state machine, sequential by construction (cannot be vectorized
without changing semantics):

```
state = inactive
for p in probabilities:
    state = if state == active: p >= offset else: p >= onset
    yield state
```

**Boundary:** at `p == offset` exactly, an active state stays active (the
condition is `p >= offset`, not `p > offset`). This matches Python.
A unit test pins the boundary.

### 5.4 Voice-timeline stitching (PER-FRAME)

Storage and computation happen at **frame rate**, not sample rate. For each
window, the model emits `[FRAMES_PER_WINDOW] = [589]` voice probabilities;
these go into a stream-indexed `(sum, count)` accumulator with one entry per
frame, not per sample. The window-to-stream frame offset is
`round(start_sample / SAMPLES_PER_FRAME)`, computed via the integer math in
§5.2.

Per-hour-of-audio storage at frame rate is ~849 KB (`(3600 * 16000 / 271.65) *
(4 + 4) = 848 568 B`), versus ~460 MB if we stored per-sample. The
implementation MUST use per-frame storage.

After a window's frame range is appended to the accumulator, frames whose
absolute index is below the finalization boundary are eligible for draining
via mean (= sum / count) and downstream processing. The frames are popped
from the front of the accumulator on drain, freeing memory.

**Tail-window overlap with the finalized region is silently dropped.** A tail
anchor whose start frame is before the current finalization boundary
contributes only to the suffix `[base_frame, end_frame)`.

`warm_up_ratio` center-cropping (Python's `stitching.py:94`) is **not
implemented** in v0.1.0 because the Python `Segmenter.segment` call site does
not pass it (default 0.0). Adding it would diverge from observed Python
behavior.

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

Voice spans are finalized by:

1. Drain the per-frame accumulator from the front up to the finalization
   boundary (in frames).
2. Run streaming hysteresis on those drained frames (the cursor's state
   carries across windows).
3. RLE the resulting boolean mask into `[start_frame, end_frame)` runs.
4. Convert frame indices to absolute sample indices via `frame_to_sample`.
5. **Merge** spans separated by ≤ `voice_merge_gap` samples.
6. **Then** filter by `min_voice_duration`.

Order of steps 5 vs 6 matches Python ([segmenter.py:249-263](../../../../findit-pyannote-seg/src/findit_pyannote_seg/segmenter.py#L249)).

**Open-span lifecycle:** the streaming hysteresis cursor maintains an
`Option<u64>` "current run start sample" between drains. While probabilities
keep crossing/staying above the offset, the run grows; on a falling edge a
voice span emits and the cursor's run-start clears. On `finish()` with no
pending inferences, an open run closes at `total_samples` and emits.

### 5.7 Voice spans are commit-only

Once `Action::VoiceSpan(r)` (or `Event::VoiceSpan(r)`) has been yielded by
`poll` / `process_samples`, the segmenter does **not** retract or revise it.
There is no `VoiceSpanRetract` event. Implementations that need to revoke
spans should buffer them on the caller side and apply their own logic.

### 5.8 Voice-span emission timing

A voice run that ends at frame `f_e` (sample `e ≈ f_e * SAMPLES_PER_FRAME`)
is emitted once the finalization boundary advances past `f_e`. The boundary
in frames is `round(next_window_idx * step_samples / SAMPLES_PER_FRAME)`
pre-finish, and the equivalent of `total_samples` post-finish-with-no-pending-
inference.

**Worst-case audio-time latency between sample `e` arriving via
`push_samples` and the corresponding voice span being emitted by `poll`:**

```
latency_audio_time = WINDOW_SAMPLES - (e mod step_samples)
```

With the defaults `WINDOW_SAMPLES = 160_000` and `step_samples = 40_000`,
latency ranges from 7.5 s (when `e mod step_samples = step_samples - 1`) to
**10 s** (when `e mod step_samples = 0`). The bound is `WINDOW_SAMPLES = 10
s`, not `step_samples = 2.5 s` as the previous revision claimed.

This bound is achievable because every sample `e` is covered by ⌈
WINDOW_SAMPLES / step_samples ⌉ = 4 windows. The LAST window covering `e`
starts at `floor(e / step_samples) * step_samples`. To emit that window as
`NeedsInference`, the segmenter needs buffered audio up to that start +
`WINDOW_SAMPLES`, which is up to `e + (WINDOW_SAMPLES - (e mod
step_samples))` = up to `e + WINDOW_SAMPLES` in the worst case. Once that
window's `push_inference` has been called, the boundary advances past `e`
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
    ├── segmenter.rs               # Segmenter (Layer 1) — owns generation counter
    └── model.rs                   # SegmentModel + Layer 2; cfg(feature = "ort")
```

## 7. Crate metadata

- `name = "dia"`, `version = "0.1.0"`, `edition = "2024"`, `rust-version = "1.95"`
- License: `MIT OR Apache-2.0` (matches all four siblings)
- Lints (verbatim, copied from `silero/Cargo.toml`):
  ```toml
  [lints.rust]
  rust_2018_idioms = "warn"
  single_use_lifetimes = "warn"
  unexpected_cfgs = { level = "warn", check-cfg = [
    'cfg(all_tests)',
    'cfg(tarpaulin)',
  ] }
  ```
  No `missing_docs` lint — siblings don't have it.
- `[profile.bench]`: copy from any of the siblings (all identical).
- Dependencies:
  - `mediatime = "0.1"`
  - `thiserror = { version = "2", default-features = false }`
  - `ort = { version = "2.0.0-rc.12", optional = true }`
- Features:
  - `default = ["std", "ort"]`
  - `std = []`
  - `ort = ["dep:ort"]`
  - **No `serde` feature** in v0.1.0 (deferred — see §3).
  - **No `alloc` feature** in v0.1.0 (the crate is unconditionally `alloc`-required).
- Dev-deps: `criterion = "0.8"`, `tempfile = "3"`, `hound = "3"`, `anyhow = "1"`
- Delete template stubs: `examples/foo.rs`, `benches/foo.rs`, `tests/foo.rs`

## 8. Streaming flow (canonical examples)

`examples/stream_layer1.rs` (no model, no ort): drives `Segmenter` directly
with a synthetic inferencer that returns "speaker A continuously voiced"
logits. The example header comment documents the run command:

```
//! Run with:
//!     cargo run --no-default-features --features std --example stream_layer1
```

This is the example the README leads with — it proves the Sans-I/O contract
holds.

`examples/stream_from_wav.rs` (gated on `ort`): streams a 16 kHz mono WAV
file in 100 ms chunks through `process_samples` / `finish_stream`. Requires
`models/segmentation-3.0.onnx` (provided by `scripts/download-model.sh`).
Default features include `ort`, so the run command is just
`cargo run --example stream_from_wav -- path/to/audio.wav`.

## 9. Testing strategy

### Unit tests (no model, no `ort`)

- `options.rs`: defaults, builder round-trip, timebase consistency.
- `powerset.rs`: softmax sum-to-1 / numerical stability under extreme logits;
  marginals on hand-built rows.
- `hysteresis.rs`: rising/falling transitions; `p == offset` boundary stays
  active; flicker rejection; empty input; RLE edge cases (empty, all-true,
  all-false, trailing open run); streaming hysteresis matches batch
  hysteresis.
- `window.rs`: empty stream → empty plan; sub-window stream; exact
  one-window stream; regular grid + tail anchor (with and without dedup);
  step == window; **stream length exactly aligned to step grid (the dedup
  case).**
- `stitch.rs`: frame-to-sample endpoints + monotonicity; single-window
  finalize-all; two overlapping windows averaged at frame rate; partial
  finalize advances base; clear resets; **tail window with start-frame
  before base-frame contributes only the suffix.**
- `segmenter.rs`:
  - empty input → no actions, no panics.
  - first window emits after one full window of audio buffered.
  - second window emits one step later.
  - `push_inference` with wrong shape → `InferenceShapeMismatch`.
  - `push_inference` with unknown id → `UnknownWindow`.
  - **`push_inference` called twice with the same id: first succeeds,
    second returns `UnknownWindow`.**
  - **`push_inference` after `clear()` with an id from before the clear:
    returns `UnknownWindow` because the generation has advanced.**
  - One window with synthetic "speaker A active" scores → at least one
    `Activity` for slot 0.
  - `finish()` on a sub-window clip schedules a tail with samples zero-padded
    past `total_samples`.
  - `clear()` resets all state including pending and generation.
  - End-of-stream closes an open voice span.
  - **Tail-window activity range is clamped to `total_samples`**.
  - **`voice_merge_gap` merges adjacent spans BEFORE `min_voice_duration`
    drops them**.
  - `pending_inferences()` and `buffered_samples()` track state correctly
    across pushes / poll / push_inference / clear.
  - **Voice-span emission timing:** feed scores progressively for windows
    0..N; assert that a span ending at sample `e` only emits once the
    finalization boundary has advanced past the frame for `e`.
  - **Voice span at exact stream boundary** (`finish()` called when an open
    run is active) closes at `total_samples`.

### Integration test (gated)

`tests/integration_segment.rs` with `#[ignore]`. Requires
`models/segmentation-3.0.onnx`. Smoke-tests Layer 2 end-to-end on synthetic
noise: assert no panic and that any emitted events have the right shape
(activities have `speaker_slot ∈ 0..3`, voice spans have `start <= end`).
Does NOT assert specific event content — model output on noise is not
specified.

### Benches

`benches/segment.rs` — Layer 1 throughput on one minute of audio with
synthetic scores. Establishes a state-machine baseline before Layer 2.

## 10. Threading and lifecycle

- `Segmenter`: auto-derived `Send` and `Sync`. All fields are `Send + Sync`
  primitives or `Send + Sync` collections (`Vec`, `VecDeque`, `BTreeMap`,
  `Box<[f32]>`, `Option<u64>`). The state machine is normally driven through
  `&mut self`, so `Sync` is incidental — sharing one `Segmenter` between
  threads buys nothing because every API call needs `&mut self`. Matches
  silero's `Session`/`StreamState` (also auto-`Sync`).
- `SegmentModel`: auto-derived `Send` (from `ort::Session`), not `Sync`. One
  per worker thread. Multiple `Segmenter`s can be driven against one
  `SegmentModel` *serially* in a single thread.
- No internal locks. No async. Callers wrap with their own runtime if desired.
- `clear()` resets `Segmenter` state but does **not** discard or reload the
  paired `SegmentModel` — the ort session stays warm. For services
  processing many short clips, this is the intended pattern.
- **Determinism**: ort inference on CPU with `intra_op_num_threads > 1` is
  not bit-exact across runs (parallel reductions). Set `intra_op_num_threads
  = 1` in `SegmentModelOptions` for reproducible results. Default is the ort
  default.

## 11. Resolved contracts

### 11.1 Voice-span emission timing

Specified in §5.8. Worst-case audio-time latency: `WINDOW_SAMPLES - (e mod
step_samples)`, ranging from 7.5 s to 10 s with default options. No
additional latency from `min_voice_duration`.

### 11.2 Backpressure

`push_samples` allocates without bound. There is no soft cap. Callers detect
runaway buffering via `Segmenter::buffered_samples()` and throttle at their
end. `pending_inferences()` reports how many `NeedsInference` actions have
been yielded without `push_inference` follow-up — useful for detecting a
caller-side bug where the inference loop was skipped.

A canonical caller pattern is documented in the `Segmenter` rustdoc:

```rust
const MAX_PENDING: usize = 16;
if seg.pending_inferences() > MAX_PENDING {
    // throttle: do not push more samples until inference catches up
}
```

### 11.3 `min_voice_duration` interaction with stream emission

Spans are filtered (or merged-then-filtered, per §5.6) **at finalization
time**, not held back. A span whose finalized duration is below the threshold
is dropped immediately, not held in case it grows — voice timelines do not
grow after finalization.

### 11.4 Sample-rate validation

`Segmenter::push_samples(&[f32])` does **not** validate sample rate. Callers
who feed non-16 kHz audio get silent corruption. Documented in the rustdoc;
matches silero/scenesdetect/soundevents practice. A typed
`SamplesAt16k<'_>` newtype was considered and rejected as call-site noise
(see §3 "Deferred").

### 11.5 Callback error propagation

`process_samples<F: FnMut(Event)>` and `finish_stream<F: FnMut(Event)>` use
`FnMut(Event)` callbacks that cannot return errors, mirroring
[`silero::Session::process_stream<F: FnMut(f32)>`](../../../../silero/src/session.rs).
Callers who need to surface an error from inside the closure can capture an
`Option<MyError>` by mutable reference and check it after the call returns.

### 11.6 Streaming chunk size

Any chunk size is accepted by `push_samples`. The segmenter buffers
internally and schedules `NeedsInference` actions as full windows become
available.

### 11.7 Sub-window-length clips

A clip shorter than `WINDOW_SAMPLES` is handled by `finish()`: a tail window
is anchored at sample 0 with the available samples followed by zero padding,
and inference runs on the padded buffer. Hysteresis applies to the resulting
voice probabilities exactly as with longer clips.

### 11.8 Multi-stream `clear()` warmup

`clear()` does not discard or warm-down the ONNX session — that lives on the
`SegmentModel`, not the `Segmenter`. A service that processes N short clips
constructs the `SegmentModel` once, then constructs (or reuses via `clear`) a
`Segmenter` per clip.

### 11.9 Stale-id rejection across `clear()` (NEW)

Every `WindowId` carries a generation counter; `Segmenter::clear()`
increments the counter. Any `WindowId` minted in a previous generation
fails `push_inference` with `Error::UnknownWindow`, even if its sample
range exactly matches a current pending window's range (e.g. both
`(0, 160_000)`).

This closes the async-pipeline corruption scenario:

```
poll() → NeedsInference { id: W0 (gen=0) }   // stream A
                                              ─→ tokio::spawn inference
[Segmenter::clear() — generation now 1]
poll() → NeedsInference { id: W0' (gen=1) }   // stream B, same range
push_samples(...) etc.
spawned task delivers: push_inference(W0 gen=0, scores_for_A)
                       ↓
                       Error::UnknownWindow — generation mismatch
                       ↓
                       stream B's W0' (gen=1) is untouched
```

Without the generation counter, the call would silently match `W0'` and
corrupt stream B's output.

### 11.10 Empty-stream behavior

`Segmenter::new(opts)` on an empty stream and `finish()` without any
`push_samples` calls produces zero `Action`s and zero `Event`s. No
`NeedsInference` is scheduled because there is no audio. `pending_inferences()`
stays at 0; `buffered_samples()` stays at 0.

## 12. Decision log (cumulative)

From original brainstorming:
- Single crate `dia`, two modules (`segment`, `embed`); not a workspace.
- Segmentation first; embedding is a separate sub-project.
- Sans-I/O dual layer: state machine has zero `ort` dep; `ort` is a default-on
  feature for the convenience driver.
- `WindowId` is a newtype over `TimeRange` (type safety on the inference round-trip).
- `VoiceSpan` is just `TimeRange` directly — no wrapper.
- Public types use private fields with accessor methods.
- `mediatime` provides every time/range/duration value crossing the API.
- Edition 2024, Rust 1.95.

Added in Revision 2 (post-review):
- `WindowId` impls `Ord`/`PartialOrd` manually so it works as a `BTreeMap`
  key.
- Algorithm semantics are pinned in §5 and treated as load-bearing contracts.
- Voice spans are commit-only (no retraction).
- Voice-span emission latency is bounded.
- Layer 2 gains `SegmentModelOptions` with provider list, optimization level,
  and thread-count knobs; defaults to CPU + level 3.
- ONNX shape verification at model load (`Error::IncompatibleModel`).
- `Segmenter::pending_inferences()` and `buffered_samples()` for
  backpressure introspection.

Added/fixed in Revision 3 (post-second review):
- `WindowId` gains a generation counter to prevent stale-id corruption
  across `clear()` (§11.9).
- Stitching, hysteresis, and RLE pinned to **per-frame** storage and
  computation — Revision 2's "sample-by-sample" wording was inaccurate
  (§5.4, §5.6).
- Voice-span emission latency bound corrected from `step_samples` to
  `WINDOW_SAMPLES` (§5.8, §11.1).
- `Sync` is `auto`-derived (matches silero); previous "not Sync" claim
  removed (§10).
- The tautological `static asserts` on the model-derived constants are
  removed; the actual schema-drift check is runtime ONNX shape verification.
- `Error` regains `#[derive(thiserror::Error)]`; manual `Display` /
  `std::error::Error` impls dropped.
- `WindowId::duration()` is now `const fn`.
- `WindowId` `Ord` simplifies to `(generation, start_pts)`; `end_pts` adds no
  information because `end == start + WINDOW_SAMPLES` always.
- `push_inference` semantics for repeat / out-of-order calls are documented
  (§4 rustdoc + §11).
- `clear()` semantics on pending inferences are documented.
- Tail-window deduplication is documented in §5 and tested in §9.
- `ort` types (`GraphOptimizationLevel`, `ExecutionProviderDispatch`) are
  re-exported from `dia::segment`; chosen over wrapping or hard-coding.
- `serde` feature deferred entirely from v0.1.0 (was half-wired).
- `alloc` Cargo feature removed (the crate is unconditionally
  `alloc`-required).
- `Action` enum-variant style (public fields) vs `WindowId`/`SpeakerActivity`
  struct style (private + accessors) is documented as deliberate.
- §2.1 added a "what we kept / what we changed" table for siblings.
- Streaming RLE / open-span lifecycle documented in §5.6.
- Empty-stream behavior documented in §11.10.
- Test list expanded to include emission-timing, stale-id, tail-dedup, and
  exact-boundary-finish cases.

## 13. Revision history

- **Revision 1** (2026-04-25): Initial spec from brainstorming. Implemented
  in commits `25f817c` … `18634b5`. Adversarial review identified four tiers
  of issues.
- **Revision 2** (2026-04-25): incorporated review-1 feedback. Algorithm
  semantics promoted to a load-bearing §5; open questions resolved to
  written contracts in §11; Layer 2 surface gains provider/optimization
  knobs; tail-window activity clamping and `voice_merge_gap` implementation
  promoted from "discovered as bugs during implementation" to "explicit
  algorithm decisions"; "exactly like silero" claims softened.
- **Revision 3** (this document): incorporates review-2 feedback. Generation
  counter on `WindowId`; per-frame stitching/hysteresis (the impl currently
  uses per-sample and must be ported); emission-latency bound corrected;
  `Sync` accepted; static asserts dropped; thiserror restored; serde
  feature deferred; ort types re-exported; numerous documentation gaps
  closed (push_inference semantics, clear semantics, tail dedup, ort type
  policy, scratch ownership, callback citation, sibling-comparison table,
  streaming RLE / open-span lifecycle, empty-stream behavior).

## 14. Findings rejected from review 2

For traceability — items the review raised but the spec does not adopt,
with reasons:

- **T1-2 (`commit_from_memory` unverified)**: rejected. silero's
  [`Session::from_memory_with_options`](../../../../silero/src/session.rs)
  uses `commit_from_memory(model_bytes)`. Sibling-validated.
- **T3-5 ("FnMut citation fabricated")**: rejected. silero's
  [`Session::process_stream<F: FnMut(f32)>`](../../../../silero/src/session.rs)
  exists. The review's Agent 2 was looking at a different API surface.
- **T3-6 (`from_memory` "forward-looking")**: rejected for the same reason
  as T1-2 — silero ships it.
- **NEW-2 (per-sample buffer lifecycle)**: subsumed by the per-frame fix
  (§5.4). The buffer trims via front-pop on every drain; per-frame storage
  reduces 1-hour memory from ~460 MB to ~849 KB.
- **T2-7 (no soft-cap enforcement)**: rejected for v0.1.0. Introspection-only
  matches silero; soft-cap is listed in §3 "Deferred" for v0.2 if a real
  caller hits the runaway scenario.
- **T4-1 (`Segmenter::default()`)**: rejected for v0.1.0. `SegmentOptions`
  has `Default`; `Segmenter::new(opts)` is one keystroke more. Not worth
  the API churn now.
- **T4-6 (edition 2024 unused)**: rejected. Edition 2024 is consistent with
  all four siblings; we're not going to diverge for cosmetics.

## 15. Action list for implementation patches

Spec is now consistent with the current branch (`25f817c` … `18634b5`)
*except* for these items, which the v0.1.1 patch must apply:

| #  | Item                                                            | Severity |
| -- | --------------------------------------------------------------- | -------- |
| 1  | Add `generation: u64` to `WindowId`; bump on `clear()`; check on `push_inference`. (§11.9) | critical |
| 2  | Switch `VoiceStitcher` from per-sample to per-frame storage. Hysteresis and RLE run on frames; `frame_to_sample` happens at emission only. (§5.4, §5.6) | critical |
| 3  | Implement `voice_merge_gap` (currently dead option). (§5.6)    | critical |
| 4  | Clamp tail-window activity ranges to `total_samples`. (§5.5)   | critical |
| 5  | Add ONNX shape verification at `SegmentModel::from_file` / `from_memory`. (§4) | high     |
| 6  | Add `Segmenter::pending_inferences()` and `buffered_samples()`. (§4) | high     |
| 7  | Add `SegmentModelOptions` with `with_optimization_level`, `with_providers`, `with_intra_op_num_threads`, `with_inter_op_num_threads`; re-export `GraphOptimizationLevel` and `ExecutionProviderDispatch` from `dia::segment`. (§4) | high     |
| 8  | Restore `#[derive(thiserror::Error)]` on `Error`; remove manual `Display`/`std::error::Error` impls. (§4) | high     |
| 9  | Simplify `WindowId::Ord` to `(generation, start_pts)`. (§4)    | medium   |
| 10 | Make `WindowId::duration()` `const fn`. (§4)                   | medium   |
| 11 | Remove the tautological `const _: () = assert!(...)` blocks. (§4) | medium   |
| 12 | Remove the unused `alloc` Cargo feature. Remove the half-wired `serde` Cargo feature. (§7) | medium   |
| 13 | Add boundary tests for `p == offset` (§5.3), tail dedup (§9),  emission timing, stale-id rejection. (§9) | medium   |
| 14 | Document `push_inference` semantics in rustdoc; document `clear()` drops pending; document caller backpressure pattern. (§4, §11.2) | low      |
| 15 | Run `cargo build --features ort` and confirm `commit_from_memory` compiles against `ort = "2.0.0-rc.12"` (sanity check, expected to pass). | low      |

Items 1-4 are correctness fixes; the rest are API/documentation cleanups
surfaced by review. None of them require redesign.
