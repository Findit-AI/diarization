# dia — segmentation sub-project design (v0.1.0)

**Status:** approved through brainstorming, pending spec review
**Date:** 2026-04-25
**Scope:** The `dia::segment` module only. The `dia::embed` module is a separate sub-project to be brainstormed and specced after this one ships.

## 1. Context

`dia` is a Rust port of two findit-studio Python projects: `findit-pyannote-seg` (speaker
segmentation, M1) and `findit-speaker-embedding` (speaker embedding, M2). The single
`dia` crate will host both as modules: `dia::segment` and `dia::embed`. **This spec
covers `dia::segment` only.**

The crate sits alongside `silero` (VAD) and `soundevents` (sound-event classification)
in the findit-studio Rust suite. It targets stream-processing audio applications that
ingest 16 kHz mono PCM samples frame-by-frame and consume diarization events.

## 2. Architecture: Sans-I/O dual layer

The same architectural philosophy as `scenesdetect`: the core state machine has no
I/O — no model loading, no inference, no threading, no file or network access. The
state machine accumulates samples, schedules windows for inference, decodes posterior
probabilities into events, and stitches voice spans across overlapping windows.

Because dia *must* run a neural network somewhere, ONNX inference lives in a separate
**Layer 2** that pairs the Layer 1 state machine with an `ort::Session`. Both layers
are public; streaming users use Layer 2; tests, async, batched, and mocked-inference
callers drop down to Layer 1.

### Layer 1 — Sans-I/O `Segmenter` (no `ort` dependency)

```rust
let mut seg = Segmenter::new(SegmentOptions::default());

seg.push_samples(&pcm_chunk);
while let Some(action) = seg.poll() {
    match action {
        Action::NeedsInference { id, samples } => {
            let scores = my_inferencer.run(&samples)?; // caller's I/O
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
powerset decoding, hysteresis, and voice-span stitching are pure CPU work owned by
the `Segmenter`. Layer 1 has zero `ort` dependency and is therefore exercisable in
unit tests with synthetic scores — no model file required, no network, no Python.

### Layer 2 — `SegmentModel` + streaming wrappers (gated on `ort` feature, default-on)

```rust
let mut model = SegmentModel::from_file("models/seg-3.0.onnx")?;
let mut seg   = Segmenter::new(SegmentOptions::default());

while let Some(frame) = audio_in.next().await {
    seg.process_samples(&mut model, &frame, |event| match event {
        Event::Activity(a)  => emit_activity(a),
        Event::VoiceSpan(r) => emit_voice_span(r),
    })?;
}
seg.finish_stream(&mut model, |event| { /* drain */ })?;
```

Internally `process_samples` is the loop shown for Layer 1 with the inference
fulfilled by `model.infer(&samples)`. The `Event` enum is `Action` minus the
`NeedsInference` variant (only emission events, never requests).

This mirrors `silero::SpeechSegmenter::process_samples` exactly.

## 3. Scope for v0.1.0

### In scope

- Layer 1 state machine: `Segmenter` with `push_samples`, `poll`, `push_inference`,
  `finish`, `clear`.
- Layer 2 driver: `SegmentModel` (ort wrapper) + `Segmenter::process_samples` /
  `finish_stream` convenience methods, behind `ort` feature.
- Sliding-window scheduling at the model's native window size (160 000 samples =
  10 s @ 16 kHz) with configurable step (default 40 000 samples = 2.5 s) and a
  tail window anchored to end-of-stream when needed.
- Powerset decoding (7-class → 3-speaker probabilities).
- Hysteresis binarization with onset/offset thresholds.
- Voice-span stitching across overlapping windows (overlap-add mean of the
  speaker-agnostic voice probability).
- Window-local `SpeakerActivity` records with stable `WindowId`.
- `mediatime`-based time types throughout.
- Pure-CPU unit tests on the state machine with hand-crafted scores.
- An `ort`-feature integration example using a real model file.

### Deferred — out of v0.1.0

| Python feature                           | Reason for deferral                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `include_frame_probabilities`            | Adds another `Action` variant; ship when a caller actually needs it.                |
| Bundled ONNX model (`include_bytes!`)    | Model is ~5.7 MB; licensing review can come later. Provide a download script.       |
| Reference parity test vs `pyannote.audio`| Requires Python + HF_TOKEN. Layer 1 unit tests cover the logic; F1 lands later.     |
| Cross-window speaker clustering / global IDs | This is the future M3 layer (downstream of `dia::embed`); not segmentation's job. |

### Out of scope (handled elsewhere)

- Speaker embedding — `dia::embed`, separate sub-project.
- Resampling — input must be 16 kHz mono float32.
- Audio-file decoding — caller's responsibility.
- Threading / worker pools — caller's responsibility, exactly like `silero`.

## 4. Public API surface

### Constants

```rust
/// 16 kHz monolithic timebase used for every `TimeRange` and `Timestamp`
/// emitted by `dia::segment`. pyannote-seg-3.0 only operates at 16 kHz.
pub const SAMPLE_RATE_TB: Timebase;

/// Sample count of one model window (160 000 = 10 s @ 16 kHz).
pub const WINDOW_SAMPLES: u32 = 160_000;

/// Per-window powerset class count (silence, A, B, C, A+B, A+C, B+C).
pub const POWERSET_CLASSES: usize = 7;

/// Per-window output frame count.
pub const FRAMES_PER_WINDOW: usize = 589;

/// Maximum simultaneous speakers per window.
pub const MAX_SPEAKER_SLOTS: u8 = 3;
```

### Types

```rust
/// Stable correlation handle for one inference round-trip. Equal to the window's
/// sample range in `SAMPLE_RATE_TB`. Safe to use as a `HashMap` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(TimeRange);
impl WindowId {
    pub fn range(&self) -> TimeRange;
    pub fn start(&self) -> Timestamp;
    pub fn end(&self)   -> Timestamp;
    pub fn duration(&self) -> Duration;
}

/// One window-local speaker activity. `speaker_slot` ∈ {0, 1, 2} is local to the
/// emitting window — slot identity does NOT cross windows. Cross-window speaker
/// identity is the job of a future clustering layer (M3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerActivity { /* fields private */ }
impl SpeakerActivity {
    pub fn window_id(&self)    -> WindowId;
    pub fn speaker_slot(&self) -> u8;
    pub fn range(&self)        -> TimeRange;
}

/// One output of the Layer 1 state machine.
#[derive(Debug, Clone)]
pub enum Action {
    /// The caller must run ONNX inference on `samples` and call
    /// `Segmenter::push_inference(id, scores)`.
    NeedsInference { id: WindowId, samples: Box<[f32]> },
    Activity(SpeakerActivity),
    /// Speaker-agnostic voice region, stitched across overlapping windows.
    VoiceSpan(TimeRange),
}

/// Layer 2 emission events (no inference request — that's handled internally).
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
    pub fn new() -> Self;                                  // pyannote defaults
    pub const fn default() -> Self;                        // same
    // accessors
    pub fn onset_threshold(&self)        -> f32;
    pub fn offset_threshold(&self)       -> f32;
    pub fn step_samples(&self)           -> u32;
    pub fn min_voice_duration(&self)     -> Duration;
    pub fn min_activity_duration(&self)  -> Duration;
    pub fn voice_merge_gap(&self)        -> Duration;
    // builders
    pub fn with_onset_threshold(self, v: f32)    -> Self;
    pub fn with_offset_threshold(self, v: f32)   -> Self;
    pub fn with_step_samples(self, v: u32)       -> Self;
    pub fn with_min_voice_duration(self, v: Duration)    -> Self;
    pub fn with_min_activity_duration(self, v: Duration) -> Self;
    pub fn with_voice_merge_gap(self, v: Duration)       -> Self;
    // mutating variants: set_*
}
```

Defaults match the Python project: onset 0.5, offset 0.357, step 40 000 samples,
all duration filters 0 (disabled).

### `Segmenter` (Layer 1)

```rust
pub struct Segmenter { /* private */ }

impl Segmenter {
    pub fn new(opts: SegmentOptions) -> Self;

    /// Append 16 kHz mono float32 PCM samples to the input buffer. Arbitrary chunk
    /// size; the segmenter does its own windowing.
    pub fn push_samples(&mut self, samples: &[f32]);

    /// Signal end-of-stream. Causes the tail window to be scheduled (if needed)
    /// and any open voice span to close on the next `poll`.
    pub fn finish(&mut self);

    /// Drain the next action. Returns `None` when nothing is ready (caller should
    /// push more samples, call `finish`, or stop).
    pub fn poll(&mut self) -> Option<Action>;

    /// Provide ONNX scores for a previously-yielded `NeedsInference` action.
    /// `scores` must have length `FRAMES_PER_WINDOW * POWERSET_CLASSES = 4123`.
    pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error>;

    /// Reset for a new stream. Internal allocations are reused.
    pub fn clear(&mut self);

    pub fn options(&self) -> &SegmentOptions;
}
```

Threading: `Segmenter` is `Send` (no interior locks) but **not** `Sync` (one
segmenter per stream). Mirrors silero's `StreamState`/`SpeechSegmenter`.

### `SegmentModel` and Layer 2 helpers (feature `ort`, default on)

```rust
#[cfg(feature = "ort")]
pub struct SegmentModel { /* private; owns ort::Session + scratch */ }

#[cfg(feature = "ort")]
impl SegmentModel {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error>;
    pub fn from_memory(bytes: &[u8])         -> Result<Self, Error>;

    /// Run inference on one 160 000-sample window. Returns the flattened
    /// `[FRAMES_PER_WINDOW * POWERSET_CLASSES] = [4123]` logits.
    /// Exposed for advanced callers who want to combine Layer 1's state machine
    /// with their own batching or scheduling around `SegmentModel`.
    pub fn infer(&mut self, samples: &[f32]) -> Result<Vec<f32>, Error>;
}

#[cfg(feature = "ort")]
impl Segmenter {
    /// Push samples, drain actions, run inference for any `NeedsInference` via
    /// `model`, and emit `Event`s through `emit`.
    pub fn process_samples<F: FnMut(Event)>(
        &mut self,
        model: &mut SegmentModel,
        samples: &[f32],
        emit: F,
    ) -> Result<(), Error>;

    /// Equivalent to `finish` + drain, with inference fulfilled by `model`.
    pub fn finish_stream<F: FnMut(Event)>(
        &mut self,
        model: &mut SegmentModel,
        emit: F,
    ) -> Result<(), Error>;
}
```

`SegmentModel` is `Send` but not `Sync`, exactly like silero's `Session`.

### Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid segment options: {0}")]
    InvalidOptions(&'static str),

    #[error("inference scores length {got}, expected {expected}")]
    InferenceShapeMismatch { expected: usize, got: usize },

    #[error("inference scores received for unknown WindowId {id:?}")]
    UnknownWindow { id: WindowId },

    #[cfg(feature = "ort")]
    #[error("failed to load model from {path}: {source}")]
    LoadModel { path: PathBuf, #[source] source: ort::Error },

    #[cfg(feature = "ort")]
    #[error(transparent)]
    Ort(#[from] ort::Error),
}
```

## 5. Module layout

```
src/
├── lib.rs                         # crate-level docs, pub mod segment;
└── segment/
    ├── mod.rs                     # pub use of all public items
    ├── types.rs                   # WindowId, SpeakerActivity, Action, Event
    ├── options.rs                 # SegmentOptions, SAMPLE_RATE_TB constant
    ├── error.rs                   # Error
    ├── powerset.rs                # 7-class -> 3-speaker decode (pure)
    ├── hysteresis.rs              # binarize + RLE (pure)
    ├── stitch.rs                  # voice probability aggregation (pure)
    ├── window.rs                  # sliding window planning (pure)
    ├── segmenter.rs               # Segmenter (Layer 1)
    └── model.rs                   # SegmentModel + Layer 2 wrappers; cfg(feature = "ort")
```

`embed/` will appear under `src/` later as a sibling.

## 6. Crate metadata changes

The dia repo is currently a `template-rs` clone. As part of v0.1.0:

- `[package].name` → `"dia"`
- `[package].version` → `"0.1.0"` (from 0.0.0)
- `[package].edition` → `"2024"` (from 2021; matches silero/soundevents/scenesdetect)
- `[package].rust-version` → `"1.95"` *(already done)*
- `[package].description` → real description
- `[package].repository` / `homepage` / `documentation` → real URLs (placeholders OK if not yet known)
- Copy `[lints]` and `[profile.bench]` config from `soundevents/Cargo.toml`
- Add deps:
  - `mediatime = "0.1"`
  - `thiserror = "2"`
  - `ort = { version = "2.0.0-rc.12", optional = true }`
  - `serde = { version = "1", optional = true, features = ["derive"] }` *(optional — for future option serialization, parallels silero/soundevents)*
- Features:
  - `default = ["std", "ort"]`
  - `std = []`
  - `alloc = []`
  - `ort = ["dep:ort"]`
  - `serde = ["dep:serde"]`
- Dev-deps:
  - `criterion = "0.8"` *(already present)*
  - `tempfile = "3"` *(already present)*
  - `hound = "3"` *(new — used by the WAV-streaming example, matches silero's choice)*
- Delete template stubs: `examples/foo.rs`, `benches/foo.rs`, `tests/foo.rs`
- Replace `README.md` with a real one (template language → dia description)
- Replace `CHANGELOG.md` with `# 0.1.0 (unreleased)` header

`build.rs` and the `ci/` scripts can stay as-is; siblings keep similar files.

## 7. Streaming flow (canonical example)

A single `examples/stream_from_wav.rs` (gated on `ort`, plus a tiny WAV reader as a
dev-dep — `hound` like silero):

```rust
use dia::segment::{Segmenter, SegmentModel, SegmentOptions, Event};

fn main() -> anyhow::Result<()> {
    let path  = std::env::args().nth(1).expect("usage: stream_from_wav <file.wav>");
    let pcm   = read_wav_mono_16k(&path)?;             // helper in the example file
    let mut model = SegmentModel::from_file("models/seg-3.0.onnx")?;
    let mut seg   = Segmenter::new(SegmentOptions::default());

    // Simulate a streaming source by feeding 100 ms chunks.
    for chunk in pcm.chunks(1_600) {
        seg.process_samples(&mut model, chunk, |event| match event {
            Event::Activity(a)  => println!("act window={:?} slot={} {:?}",
                                            a.window_id().range(), a.speaker_slot(), a.range()),
            Event::VoiceSpan(r) => println!("voice {:?} ({:?})", r, r.duration()),
        })?;
    }
    seg.finish_stream(&mut model, |event| { /* same match as above */ })?;
    Ok(())
}
```

A second example, `examples/stream_layer1.rs`, demonstrates the same flow against
the bare `Segmenter` with a stub inferencer that returns synthetic logits — this is
the example the README leads with, because it shows the Sans-I/O contract.

## 8. Testing strategy

### Unit tests (no model, no `ort`)

- `powerset.rs`: 7-class → 3-speaker decoding correctness over a few hand-built
  logit vectors; softmax stability.
- `hysteresis.rs`: onset/offset transitions; min-duration cutoff; RLE edge cases
  (empty input, all-true, all-false, alternating).
- `stitch.rs`: overlap-add mean for two windows; correct handling of the tail
  window; min-voice-duration filter.
- `window.rs`: sliding-window planner with various clip lengths, including the
  case where the clip is shorter than one window and the tail-anchor case.
- `segmenter.rs`:
  - `push_samples` then `poll` yields `NeedsInference` with correct `WindowId`
    range and 160 000 samples (zero-padded for a short clip).
  - Round-tripping synthetic scores through `push_inference` produces expected
    `Activity` and `VoiceSpan` events.
  - `finish` produces the tail window when needed; `clear` resets state.
  - `push_inference` with wrong shape → `InferenceShapeMismatch`.
  - `push_inference` with unknown id → `UnknownWindow`.

### Integration tests (require model file; `--features ort`, gated `#[ignore]`)

- Round-trip a known short audio buffer through Layer 2 and assert that some
  voice span and at least one activity are emitted. Looser than the Python F1
  parity test (which is deferred); just smoke coverage of the ort wiring.

### Benches (`benches/segment.rs`)

- Throughput of `Segmenter` only (synthetic scores), to establish a Layer 1
  baseline before Layer 2 is plugged in.

## 9. Threading and lifecycle

- `Segmenter`: `Send`, not `Sync`. One per stream. Cheap to construct; cheaper to
  reuse via `clear()`.
- `SegmentModel`: `Send`, not `Sync`. One per worker thread, exactly like
  `silero::Session`.
- No internal locks. No async. Callers wrap with their own runtime if desired.

## 10. Open questions

These are *not* blockers for the implementation plan — flagging them so they
don't get lost.

1. **Voice-span emission latency.** With 10 s windows and a 2.5 s step, a voice
   span that ends inside window N can only be confirmed once window N+3 has been
   processed (so we know the offset threshold has stayed below for ≥ 7.5 s of
   subsequent windows). The plan should pin down exactly when a span emits.
2. **Backpressure.** If a caller pushes huge buffers without polling, the
   pending-inference queue grows. We document this and leave it as the caller's
   problem; alternative would be a hard cap returning an error.
3. **`min_voice_duration` interaction with stream emission.** Filtering by
   duration means a span is held until it meets the threshold. Document the
   latency this adds.

## 11. Decision log (from brainstorming)

- Single crate `dia`, two modules (`segment`, `embed`); not a workspace.
- Segmentation first; embedding is a separate sub-project.
- Sans-I/O dual layer: state machine has zero `ort` dep; `ort` is a default-on
  feature for the convenience driver.
- `WindowId` is a newtype over `TimeRange` (type safety on the inference round-trip).
- `VoiceSpan` is just `TimeRange` directly — no wrapper.
- Public types use private fields with accessor methods.
- `mediatime` provides every time/range/duration value crossing the API.
- Edition 2024, Rust 1.95.
