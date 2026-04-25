# dia — segmentation sub-project design (v0.1.0)

**Revision 2** (2026-04-25, post-adversarial-review).
**Status:** ready for re-review.
**Scope:** The `dia::segment` module only. `dia::embed` is a separate sub-project to be specced after this one ships.

> **Revision 2 changes** are listed in §13. The big ones: every algorithm
> ordering and rounding decision is now written down in §5 (Algorithm
> semantics); the three "open questions" in the original §10 have been
> resolved into concrete contracts (§11); the Layer 2 `SegmentModel` gains a
> small builder for providers + optimization level; the API gains
> introspection methods for backpressure; the spec drops "exactly like
> silero" claims it cannot honor.

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
(Layer 1), plus a separate Layer 2 that pairs it with an `ort::Session`. The state
machine's outputs include explicit `NeedsInference` actions; the caller fulfils
those any way they want (sync, async, batched, mocked). Layer 2 is the canned
synchronous-`ort` integration.

This is a **deliberate divergence from `scenesdetect`**, which has no model
dependency at all. It is **inspired by `silero`'s** `process_stream(stream,
samples, |prob| ...)` callback shape (see [silero/src/session.rs:297](../../../../silero/src/session.rs)),
not a verbatim copy: silero's state machine and ort session are coupled in one
`Session` type, while dia keeps them in separate types so Layer 1 is testable
without `ort`.

### Layer 1 — Sans-I/O `Segmenter` (no `ort` dependency)

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

### Layer 2 — `SegmentModel` + streaming wrappers (gated on `ort` feature, default-on)

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
- Layer 2 `SegmentModel` (ort wrapper) + `Segmenter::process_samples` /
  `finish_stream` convenience methods, behind the `ort` feature.
- Sliding-window scheduling with configurable step (default 40 000 samples =
  2.5 s) and a tail window anchored to end-of-stream when needed.
- Powerset decoding (7-class → 3-speaker, additive sum of class probabilities).
- Hysteresis binarization with onset/offset thresholds (sequential state machine).
- Voice-span stitching across overlapping windows (overlap-add mean of
  `1 - p[silence]`, then hysteresis on the stitched timeline).
- Voice-span post-processing: **merge BEFORE min-duration filter** (matches
  Python).
- Window-local `SpeakerActivity` records, sample range clamped to actual stream
  length on tail windows.
- `mediatime`-based time types throughout.
- ONNX model shape verification at `from_file`/`from_memory` load time
  (returns `Error::IncompatibleModel` if input or output dims don't match
  `[*, 1, 160000]` / `[*, 589, 7]`).
- `SegmentModel` configuration for execution providers and graph optimization
  level (CPU + level 3 by default).
- Static asserts on `FRAMES_PER_WINDOW == 589` and `POWERSET_CLASSES == 7` to
  surface model schema changes at build time.
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
| `serde` derives for output types                | Only `SegmentOptions` gets `serde` in v0.1.0; output-type serialization can come with the first user. |
| `ControlFlow<E>` / `Result<()>` user callbacks  | Silero ships `FnMut(f32)` (no error propagation); we follow that idiom. Users `.set` an `Option` from inside the closure if they need to surface an error after the call returns. |
| `step_samples` typed as `Duration` rather than `u32` | Aesthetic cleanup; keeps churn out of v0.1.0. Pick at v0.2 / v1.0.                              |
| `drain()` iterator adapter                      | `while let Some(action) = seg.poll()` is fine and explicit. Add when ergonomics demand it.            |
| `SamplesAt16k<'_>` typed input                  | Type-safe but pollutes every call site. Caller-must-enforce matches silero/scenesdetect/soundevents.  |

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

// Static asserts catch a future model schema change at build time.
const _: () = assert!(FRAMES_PER_WINDOW == 589);
const _: () = assert!(POWERSET_CLASSES == 7);
const _: () = assert!(MAX_SPEAKER_SLOTS == 3);
```

### Types

```rust
/// Stable correlation handle for one inference round-trip. Equal to the
/// window's sample range in `SAMPLE_RATE_TB`. Safe to use as a `BTreeMap`
/// key (Ord/PartialOrd manually impl'd; mediatime's TimeRange has Hash+Eq
/// but not Ord).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(/* private */);
impl WindowId {
    pub const fn range(&self)    -> TimeRange;
    pub const fn start(&self)    -> Timestamp;
    pub const fn end(&self)      -> Timestamp;
    pub fn duration(&self)       -> Duration;
}
impl Ord        for WindowId { /* by (start_pts, end_pts) */ }
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
    pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error>;

    /// Signal end-of-stream. Schedules a tail-anchored window if needed.
    /// The closing voice span (if any) is emitted as soon as the last
    /// pending inference is fulfilled (or immediately, if there is none).
    pub fn finish(&mut self);

    /// Reset to empty state for a new stream. Internal allocations are
    /// reused. Does NOT discard or warm down a paired `SegmentModel`.
    pub fn clear(&mut self);

    // Introspection — for backpressure and debugging.

    /// Number of `NeedsInference` actions yielded but not yet fulfilled
    /// via `push_inference`. Stays at zero in a steady state.
    pub fn pending_inferences(&self) -> usize;

    /// Bytes currently buffered, divided by 4. Useful for detecting
    /// pathological caller backpressure (e.g. `push_samples` without
    /// ever calling `poll`).
    pub fn buffered_samples(&self) -> usize;
}
```

Auto-derived `Send` (no interior mutability that would defeat it). Not `Sync` —
a `Segmenter` represents one stream's state. Use one per concurrent stream.
Note: silero relies on the same auto-derive behavior; we don't `unsafe impl`
either marker explicitly.

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
internally (using ort's `commit_from_memory`, not `commit_from_memory_directly`).
The buffer can be dropped immediately after the constructor returns.

`SegmentModel` validates input/output shapes at construction:
- input dims must match `[*, 1, 160000]` (dynamic batch axis allowed)
- output dims must match `[*, 589, 7]`

A wrong-architecture model fails at load with `Error::IncompatibleModel`, not
on first inference.

`SegmentModel` is `Send` (auto-derived from `ort::Session`) but not `Sync`. Use
one per worker.

### Errors

```rust
#[derive(Debug)]
pub enum Error {
    /// Construction-time validation of `SegmentOptions`.
    InvalidOptions(&'static str),

    /// `push_inference` received the wrong number of scores.
    InferenceShapeMismatch { expected: usize, got: usize },

    /// `push_inference` was called with a `WindowId` that was never
    /// yielded (or has already been consumed).
    UnknownWindow { id: WindowId },

    /// Loaded model's input/output dims don't match what we expect.
    #[cfg(feature = "ort")]
    IncompatibleModel { tensor: &'static str, expected: &'static [i64], got: Vec<i64> },

    #[cfg(feature = "ort")]
    LoadModel { path: PathBuf, source: ort::Error },

    #[cfg(feature = "ort")]
    Ort(ort::Error),
}
```

`Error` impls `Display` + `std::error::Error` (the latter behind `feature = "std"`).

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

### 5.4 Voice-timeline stitching

For each window's per-frame voice probabilities `[589]`, expand each frame to
its sample range using `frame_to_sample` and accumulate into a stream-indexed
`(sum, count)` buffer. On finalization, mean = sum / count.

**Tail-window overlap with the finalized region is silently dropped.** A tail
anchor whose start sample is before the current finalization boundary
contributes only to the suffix `[base_sample, end)`.

`warm_up_ratio` center-cropping (Python's `stitching.py:94`) is **not
implemented** in v0.1.0 because the Python `Segmenter.segment` call site does
not pass it (default 0.0). Adding it would diverge from observed Python
behavior.

### 5.5 Activity extraction

For each window, after powerset decoding:

1. For each speaker slot, run a fresh hysteresis (no cross-window state).
2. RLE the boolean mask into `[start_frame, end_frame)` runs.
3. Convert frame indices to absolute sample indices via `frame_to_sample` and
   the window's `start_sample`.
4. **Clamp `[start_sample, end_sample)` to `[0, total_samples]`** for tail
   windows after `finish()`. Without `finish()`, regular windows already
   guarantee they fall within buffered audio.
5. **Then** filter by `min_activity_duration`. Order matters: a clamp that
   shrinks a span below threshold drops it.

### 5.6 Voice-span finalization

Voice spans are finalized from the stitched timeline by:

1. Hysteresis on the per-sample mean voice probability (online, sample-by-sample).
2. RLE → `[start, end)` sample ranges.
3. **Merge** spans separated by ≤ `voice_merge_gap` samples.
4. **Then** filter by `min_voice_duration`.

Order of step 3 vs 4 matches Python ([segmenter.py:249-263](../../../../findit-pyannote-seg/src/findit_pyannote_seg/segmenter.py#L249)).

### 5.7 Voice spans are commit-only

Once `Action::VoiceSpan(r)` (or `Event::VoiceSpan(r)`) has been yielded by
`poll` / `process_samples`, the segmenter does **not** retract or revise it.
There is no `VoiceSpanRetract` event. Implementations that need to revoke
spans should buffer them on the caller side and apply their own logic.

### 5.8 Voice-span emission timing

A voice run that ends at sample `e` is emitted once the finalization boundary
passes `e`, where the boundary is `next_window_idx * step_samples` pre-finish
or `total_samples` post-finish-with-no-pending-inference. With the default
`step = 40 000`, this means a span ending at sample `e` emits at most 2.5 s
(one step) after the window covering `e` has been processed by `push_inference`.

The `min_voice_duration` filter does not change this — once a span is finalized
it is either emitted or dropped immediately.

## 6. Module layout

```
src/
├── lib.rs                         # crate-level docs, pub mod segment;
└── segment/
    ├── mod.rs                     # pub use of all public items
    ├── types.rs                   # WindowId (incl. Ord), SpeakerActivity, Action, Event
    ├── options.rs                 # SegmentOptions, SAMPLE_RATE_TB, static asserts
    ├── error.rs                   # Error
    ├── powerset.rs                # softmax + 7→3 marginals + voice_prob (pure)
    ├── hysteresis.rs              # streaming Hysteresis + RLE (pure)
    ├── stitch.rs                  # frame_to_sample + VoiceStitcher (pure)
    ├── window.rs                  # plan_starts (pure)
    ├── segmenter.rs               # Segmenter (Layer 1)
    └── model.rs                   # SegmentModel + Layer 2; cfg(feature = "ort")
```

## 7. Crate metadata

- `name = "dia"`, `version = "0.1.0"`, `edition = "2024"`, `rust-version = "1.95"`
- License: `MIT OR Apache-2.0` (matches all four siblings)
- Lints: copy from `silero/Cargo.toml`'s `[lints.rust]` block (which scenesdetect
  and mediatime also use): `rust_2018_idioms = "warn"`, `single_use_lifetimes
  = "warn"`, `unexpected_cfgs = { level = "warn", check-cfg = [...] }`. **No
  `missing_docs` lint** — siblings don't have it.
- `[profile.bench]`: copy from any of the siblings (all identical).
- Dependencies:
  - `mediatime = "0.1"`
  - `thiserror = "2"`
  - `ort = { version = "2.0.0-rc.12", optional = true }`
  - `serde = { version = "1", optional = true, default-features = false, features = ["derive"] }`
- Features: `default = ["std", "ort"]`, `std = []`, `alloc = []`, `ort = ["dep:ort"]`,
  `serde = ["dep:serde", "mediatime/serde"]`
- Dev-deps: `criterion = "0.8"`, `tempfile = "3"`, `hound = "3"`, `anyhow = "1"`
- Delete template stubs: `examples/foo.rs`, `benches/foo.rs`, `tests/foo.rs`

## 8. Streaming flow (canonical examples)

`examples/stream_layer1.rs` (no model, no ort): drives `Segmenter` directly
with a synthetic inferencer that returns "speaker A continuously voiced"
logits. Run with `cargo run --no-default-features --features std --example
stream_layer1`. This is the example the README leads with — it proves the
Sans-I/O contract holds.

`examples/stream_from_wav.rs` (gated on `ort`): streams a 16 kHz mono WAV
file in 100 ms chunks through `process_samples` / `finish_stream`. Requires
`models/segmentation-3.0.onnx` (provided by `scripts/download-model.sh`).

## 9. Testing strategy

### Unit tests (no model, no `ort`)

- `options.rs`: defaults, builder round-trip, timebase consistency.
- `powerset.rs`: softmax sum-to-1 / numerical stability under extreme logits;
  marginals on hand-built rows.
- `hysteresis.rs`: rising/falling transitions; flicker rejection; empty input;
  RLE edge cases (empty, all-true, all-false, trailing open run); streaming
  hysteresis matches batch hysteresis.
- `window.rs`: empty stream, sub-window stream, exact one-window stream,
  regular grid + tail anchor (with and without dedup), step == window.
- `stitch.rs`: frame-to-sample endpoints + monotonicity; single-window
  finalize-all; two overlapping windows averaged; partial finalize advances
  base; clear resets.
- `segmenter.rs`:
  - empty input → no actions.
  - first window emits after one full window of audio buffered.
  - second window emits one step later.
  - `push_inference` with wrong shape → `InferenceShapeMismatch`.
  - `push_inference` with unknown id → `UnknownWindow`.
  - One window with synthetic "speaker A active" scores → at least one
    `Activity` for slot 0.
  - `finish()` on a sub-window clip schedules a tail with samples zero-padded
    past `total_samples`.
  - `clear()` resets all state.
  - End-of-stream closes an open voice span.
  - **Tail-window activity range is clamped to `total_samples`** (regression
    test for the v0.1.0 fix-up).
  - **`voice_merge_gap` merges adjacent spans BEFORE `min_voice_duration`
    drops them** (regression test for the v0.1.0 fix-up).
  - `pending_inferences()` and `buffered_samples()` track state correctly.

### Integration test (gated)

`tests/integration_segment.rs` with `#[ignore]`. Requires
`models/segmentation-3.0.onnx`. Smoke-tests Layer 2 end-to-end on synthetic
noise: confirms the ONNX shape contract holds and `process_samples` /
`finish_stream` run without panicking.

### Static asserts

`const _: () = assert!(...)` in `options.rs` for `FRAMES_PER_WINDOW`,
`POWERSET_CLASSES`, `MAX_SPEAKER_SLOTS`, and `WINDOW_SAMPLES`. Catches a
future model schema change at compile time.

### Benches

`benches/segment.rs` — Layer 1 throughput on one minute of audio with
synthetic scores. Establishes a state-machine baseline before Layer 2.

## 10. Threading and lifecycle

- `Segmenter`: auto-derived `Send` (no interior mutability), not `Sync`. One
  per stream.
- `SegmentModel`: auto-derived `Send` (from `ort::Session`), not `Sync`. One
  per worker thread. Multiple `Segmenter`s can be driven against one `SegmentModel`
  *serially* in a single thread.
- No internal locks. No async. Callers wrap with their own runtime if desired.
- `clear()` resets `Segmenter` state but does **not** discard or reload the
  paired `SegmentModel` — the ort session stays warm. For services
  processing many short clips, this is the intended pattern.
- **Determinism**: ort inference on CPU with `intra_op_num_threads > 1` is
  not bit-exact across runs (parallel reductions). Set `intra_op_num_threads
  = 1` in `SegmentModelOptions` for reproducible results. Default is the ort
  default.

## 11. Resolved contracts (formerly "open questions")

### 11.1 Voice-span emission timing

Specified in §5.8. Worst-case latency: one step (default 2.5 s) past the
window covering the span's end. No additional latency from `min_voice_duration`.

### 11.2 Backpressure

`push_samples` allocates without bound. There is no soft cap. Callers detect
runaway buffering via `Segmenter::buffered_samples()` and throttle at their
end. `pending_inferences()` reports how many `NeedsInference` actions have
been yielded without `push_inference` follow-up — useful for detecting a
caller-side bug where the inference loop was skipped.

### 11.3 `min_voice_duration` interaction with stream emission

Spans are filtered (or merged-then-filtered, per §5.6) **at finalization
time**, not held back. A span whose finalized duration is below the threshold
is dropped immediately, not held in case it grows — voice timelines do not
grow after finalization.

### 11.4 Sample-rate validation

`Segmenter::push_samples(&[f32])` does **not** validate sample rate. Callers
who feed non-16 kHz audio get silent corruption. Documented in the rustdoc;
matches silero/scenesdetect/soundevents practice. A typed
`SamplesAt16k<'_>` newtype was considered and rejected as call-site noise.

### 11.5 Callback error propagation

`process_samples<F: FnMut(Event)>` and `finish_stream<F: FnMut(Event)>` use
`FnMut(Event)` callbacks that cannot return errors, mirroring
`silero::Session::process_stream<F: FnMut(f32)>`. Callers who need to surface
an error from inside the closure can capture an `Option<MyError>` by mutable
reference and check it after the call returns.

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
- `WindowId` impls `Ord`/`PartialOrd` manually so it works as a `BTreeMap` key
  in `no_std` (mediatime's `TimeRange` lacks `Ord`).
- Algorithm semantics are pinned in §5 and treated as load-bearing contracts.
- Voice spans are commit-only (no retraction).
- Voice-span emission latency is bounded at one step past the covering window.
- Layer 2 gains `SegmentModelOptions` with provider list, optimization level,
  and thread-count knobs; defaults to CPU + level 3.
- ONNX shape verification at model load (`Error::IncompatibleModel`).
- `Segmenter::pending_inferences()` and `buffered_samples()` for
  backpressure introspection.
- Static asserts on `FRAMES_PER_WINDOW`, `POWERSET_CLASSES`, `WINDOW_SAMPLES`,
  `MAX_SPEAKER_SLOTS`.
- The "exactly like silero" / "exactly like scenesdetect" claims are softened
  to "inspired by"; deliberate divergences are listed in §2.
- `Send` / `Sync` are auto-derived; we don't claim to "mirror silero" because
  silero doesn't `unsafe impl` either marker either.
- Tail-window activity sample ranges are clamped to `total_samples`.
- `voice_merge_gap` is implemented (was specified but not wired in
  Revision 1's algorithm details).

## 13. Revision history

- **Revision 1** (2026-04-25): Initial spec from brainstorming. Implemented
  in commits `25f817c` … `18634b5`. Adversarial review identified four tiers
  of issues (see review document).
- **Revision 2** (2026-04-25, this document): incorporates review feedback.
  Algorithm semantics promoted to a load-bearing §5; open questions resolved
  to written contracts in §11; Layer 2 surface gains provider/optimization
  knobs; tail-window activity clamping and `voice_merge_gap` implementation
  promoted from "discovered as bugs during implementation" to "explicit
  algorithm decisions"; "exactly like silero" claims softened to deliberate
  divergences; introspection methods added; static asserts called out;
  several v0.2 candidates explicitly enumerated in §3 "Deferred" so they're
  not lost.
