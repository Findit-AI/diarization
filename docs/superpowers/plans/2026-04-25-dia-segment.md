# diarization::segment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `diarization::segment` Sans-I/O speaker-segmentation module — a pure-CPU state machine that ingests 16 kHz mono PCM frame-by-frame and emits `SpeakerActivity` and voice-span events, with an optional `ort`-feature convenience layer that drives ONNX inference.

**Architecture:** Two layers. Layer 1 is a Sans-I/O `Segmenter` with no `ort` dependency: caller pumps samples in via `push_samples`, drains `Action`s from `poll`, runs inference externally, and pushes scores back via `push_inference`. Layer 2 (gated on `ort` feature, default-on) pairs Layer 1 with a `SegmentModel` ONNX wrapper and provides silero-shaped `process_samples` / `finish_stream` streaming methods.

**Tech Stack:** Rust 2024 edition, Rust 1.95+, `mediatime` for time/range types, `ort 2.0.0-rc.12` for ONNX, `thiserror` 2 for errors, pyannote/segmentation-3.0 model.

**Spec:** `docs/superpowers/specs/2026-04-25-dia-segment-design.md`

---

## File Structure

**Created:**
- `src/segment/mod.rs` — module entry, re-exports
- `src/segment/types.rs` — `WindowId`, `SpeakerActivity`, `Action`, `Event`
- `src/segment/options.rs` — `SegmentOptions`, `SAMPLE_RATE_TB`, all constants
- `src/segment/error.rs` — `Error`
- `src/segment/powerset.rs` — softmax + 7→3 powerset decode (pure)
- `src/segment/hysteresis.rs` — onset/offset binarization + RLE (pure)
- `src/segment/stitch.rs` — overlap-add voice probability stitching (pure)
- `src/segment/window.rs` — sliding-window planner (pure)
- `src/segment/segmenter.rs` — Layer 1 `Segmenter` state machine
- `src/segment/model.rs` — Layer 2 `SegmentModel` (ort wrapper) + streaming helpers
- `examples/stream_layer1.rs` — sans-I/O example with synthetic inferencer
- `examples/stream_from_wav.rs` — full pipeline against a WAV file (`ort` feature)
- `tests/integration_segment.rs` — gated integration test against a real model
- `benches/segment.rs` — Layer 1 throughput bench
- `scripts/download-model.sh` — download `segmentation-3.0.onnx` from HuggingFace

**Modified:**
- `Cargo.toml` — name, version, edition, deps, features, dev-deps, lints, profile
- `src/lib.rs` — declare and re-export `segment`
- `README.md` — replace template content
- `CHANGELOG.md` — replace template content

**Deleted:**
- `examples/foo.rs`, `benches/foo.rs`, `tests/foo.rs` — template stubs

---

## Task 1: Crate metadata, dependencies, features, lints

**Files:**
- Modify: `Cargo.toml` (whole file)
- Delete: `examples/foo.rs`, `benches/foo.rs`, `tests/foo.rs`

- [ ] **Step 1.1: Replace Cargo.toml with the dia version**

Overwrite `/Users/user/Develop/findit-studio/dia/Cargo.toml` with:

```toml
[package]
name = "dia"
version = "0.1.0"
edition = "2024"
rust-version = "1.95"
license = "MIT OR Apache-2.0"
repository = "https://github.com/al8n/dia"
homepage = "https://github.com/al8n/dia"
documentation = "https://docs.rs/dia"
description = "Sans-I/O speaker diarization for streaming audio (segmentation; embedding to follow)."
readme = "README.md"

[features]
default = ["std", "ort"]
std = []
alloc = []
ort = ["dep:ort"]
serde = ["dep:serde", "mediatime/serde"]

[dependencies]
mediatime = "0.1"
thiserror = "2"
ort = { version = "2.0.0-rc.12", optional = true }
serde = { version = "1", optional = true, default-features = false, features = ["derive"] }

[dev-dependencies]
criterion = "0.8"
tempfile = "3"
hound = "3"

[[bench]]
path = "benches/segment.rs"
name = "segment"
harness = false

[profile.bench]
opt-level = 3
debug = false
codegen-units = 1
lto = "thin"
incremental = false
debug-assertions = false
overflow-checks = false
rpath = false

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[lints.rust]
rust_2018_idioms = "warn"
single_use_lifetimes = "warn"
unexpected_cfgs = { level = "warn", check-cfg = [
  'cfg(all_tests)',
  'cfg(tarpaulin)',
] }
```

- [ ] **Step 1.2: Delete the three template stub files**

```bash
rm /Users/user/Develop/findit-studio/dia/examples/foo.rs \
   /Users/user/Develop/findit-studio/dia/benches/foo.rs \
   /Users/user/Develop/findit-studio/dia/tests/foo.rs
```

- [ ] **Step 1.3: Verify the crate builds**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo build --no-default-features --features std
```

Expected: builds successfully with no errors. (`--no-default-features` avoids needing ort right now.)

- [ ] **Step 1.4: Verify default-feature build pulls ort**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo build
```

Expected: builds with `ort` linked; warning-free.

- [ ] **Step 1.5: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add Cargo.toml && \
  git rm examples/foo.rs benches/foo.rs tests/foo.rs && \
  git commit -m "chore: dia crate metadata and clean template stubs"
```

---

## Task 2: README and CHANGELOG

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 2.1: Write README.md**

Replace `/Users/user/Develop/findit-studio/dia/README.md` with:

````markdown
# dia

Sans-I/O speaker diarization for streaming audio.

`dia` is a Rust port of two findit-studio Python projects: `findit-pyannote-seg`
(speaker segmentation) and `findit-speaker-embedding` (speaker embedding). The
v0.1.0 release ships **segmentation only** (`diarization::segment`); embedding and
cross-window clustering will follow in subsequent releases.

## Design

The core (`diarization::segment::Segmenter`) is a Sans-I/O state machine: it never
opens a file, runs a model, or spawns a thread. Callers push 16 kHz mono PCM
in, drain `Action`s out, run ONNX inference themselves, and push scores back.
This makes the segmenter testable with synthetic scores (no model file
required) and lets callers choose any inference policy — synchronous, async,
batched across streams, GPU, mocked.

A convenience layer (gated on the default `ort` feature) provides
`process_samples` / `finish_stream` methods that drive a bundled
`SegmentModel` ONNX wrapper, mirroring `silero`'s streaming idiom.

## Quickstart

```rust
use diarization::segment::{Segmenter, SegmentModel, SegmentOptions, Event};

let mut model = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
let mut seg   = Segmenter::new(SegmentOptions::default());

while let Some(frame) = audio_in.next() {
    seg.process_samples(&mut model, &frame, |event| match event {
        Event::Activity(a)  => println!("{:?}", a),
        Event::VoiceSpan(r) => println!("{:?}", r),
    })?;
}
seg.finish_stream(&mut model, |_| {})?;
# Ok::<(), diarization::segment::Error>(())
```

For Sans-I/O usage (no `ort` dependency), see `examples/stream_layer1.rs`.

## License

`dia` is dual-licensed under MIT or Apache 2.0 at your option.
````

- [ ] **Step 2.2: Write CHANGELOG.md**

Replace `/Users/user/Develop/findit-studio/dia/CHANGELOG.md` with:

```markdown
# 0.1.0 (unreleased)

FEATURES

- `diarization::segment` module: Sans-I/O speaker segmentation backed by
  pyannote/segmentation-3.0 ONNX. Two layers: a pure state-machine `Segmenter`
  with no `ort` dependency, plus a default-on `ort` feature exposing a
  `SegmentModel` ONNX wrapper and `process_samples` / `finish_stream`
  streaming helpers.
```

- [ ] **Step 2.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add README.md CHANGELOG.md && \
  git commit -m "docs: dia README and CHANGELOG for v0.1.0"
```

---

## Task 3: Module skeleton and constants

**Files:**
- Modify: `src/lib.rs`
- Create: `src/segment/mod.rs`, `src/segment/options.rs`

- [ ] **Step 3.1: Rewrite src/lib.rs**

Replace `/Users/user/Develop/findit-studio/dia/src/lib.rs` with:

```rust
//! Sans-I/O speaker diarization for streaming audio.
//!
//! See the [`segment`] module for the v0.1.0 surface (speaker segmentation).
//! Future releases will add an `embed` module for speaker embedding and a
//! clustering layer that turns window-local speaker slots into global
//! speaker identities.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

pub mod segment;
```

- [ ] **Step 3.2: Create src/segment/mod.rs**

Create `/Users/user/Develop/findit-studio/dia/src/segment/mod.rs`:

```rust
//! Speaker segmentation: Sans-I/O state machine + optional ort driver.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.

mod error;
mod hysteresis;
mod options;
mod powerset;
mod segmenter;
mod stitch;
mod types;
mod window;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
mod model;

pub use error::Error;
pub use options::{
    FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, SAMPLE_RATE_HZ,
    SAMPLE_RATE_TB, SegmentOptions, WINDOW_SAMPLES,
};
pub use segmenter::Segmenter;
pub use types::{Action, Event, SpeakerActivity, WindowId};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use model::SegmentModel;
```

- [ ] **Step 3.3: Create src/segment/options.rs with constants and SegmentOptions**

Create `/Users/user/Develop/findit-studio/dia/src/segment/options.rs`:

```rust
//! Configuration constants and tunables for `diarization::segment`.

use core::num::NonZeroU32;
use core::time::Duration;

use mediatime::Timebase;

/// Audio sample rate this module supports — 16 kHz.
///
/// pyannote/segmentation-3.0 was trained at 16 kHz only. Callers must
/// resample upstream.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// `mediatime` timebase for every sample-indexed `Timestamp` and `TimeRange`
/// emitted by this module: `1 / 16_000` seconds.
pub const SAMPLE_RATE_TB: Timebase =
    Timebase::new(1, NonZeroU32::new(SAMPLE_RATE_HZ).unwrap());

/// Sample count of one model window — 160 000 samples (10 s at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 160_000;

/// Output frames produced per window by the segmentation model.
pub const FRAMES_PER_WINDOW: usize = 589;

/// Powerset class count: silence, A, B, C, A+B, A+C, B+C.
pub const POWERSET_CLASSES: usize = 7;

/// Maximum simultaneous speakers per window.
pub const MAX_SPEAKER_SLOTS: u8 = 3;

/// Tunables for the segmenter. Defaults match the upstream pyannote pipeline.
#[derive(Debug, Clone)]
pub struct SegmentOptions {
    onset_threshold: f32,
    offset_threshold: f32,
    step_samples: u32,
    min_voice_duration: Duration,
    min_activity_duration: Duration,
    voice_merge_gap: Duration,
}

impl Default for SegmentOptions {
    fn default() -> Self { Self::new() }
}

impl SegmentOptions {
    /// Construct with pyannote defaults: onset 0.5, offset 0.357,
    /// step 40 000 samples (2.5 s), all duration filters disabled.
    pub const fn new() -> Self {
        Self {
            onset_threshold: 0.5,
            offset_threshold: 0.357,
            step_samples: 40_000,
            min_voice_duration: Duration::ZERO,
            min_activity_duration: Duration::ZERO,
            voice_merge_gap: Duration::ZERO,
        }
    }

    /// Onset (rising-edge) threshold for hysteresis binarization.
    pub const fn onset_threshold(&self) -> f32 { self.onset_threshold }
    /// Offset (falling-edge) threshold for hysteresis binarization.
    pub const fn offset_threshold(&self) -> f32 { self.offset_threshold }
    /// Sliding-window step in samples (default 40 000 = 2.5 s).
    pub const fn step_samples(&self) -> u32 { self.step_samples }
    /// Minimum voice-span duration; shorter spans are dropped (default 0).
    pub const fn min_voice_duration(&self) -> Duration { self.min_voice_duration }
    /// Minimum speaker-activity duration (default 0).
    pub const fn min_activity_duration(&self) -> Duration { self.min_activity_duration }
    /// Merge adjacent voice spans separated by at most this gap (default 0).
    pub const fn voice_merge_gap(&self) -> Duration { self.voice_merge_gap }

    /// Builder: set the onset threshold.
    pub const fn with_onset_threshold(mut self, v: f32) -> Self { self.onset_threshold = v; self }
    /// Builder: set the offset threshold.
    pub const fn with_offset_threshold(mut self, v: f32) -> Self { self.offset_threshold = v; self }
    /// Builder: set the sliding-window step in samples.
    pub const fn with_step_samples(mut self, v: u32) -> Self { self.step_samples = v; self }
    /// Builder: set the minimum voice-span duration.
    pub const fn with_min_voice_duration(mut self, v: Duration) -> Self { self.min_voice_duration = v; self }
    /// Builder: set the minimum speaker-activity duration.
    pub const fn with_min_activity_duration(mut self, v: Duration) -> Self { self.min_activity_duration = v; self }
    /// Builder: set the voice-span merge gap.
    pub const fn with_voice_merge_gap(mut self, v: Duration) -> Self { self.voice_merge_gap = v; self }

    /// Mutating: set the onset threshold.
    pub fn set_onset_threshold(&mut self, v: f32) -> &mut Self { self.onset_threshold = v; self }
    /// Mutating: set the offset threshold.
    pub fn set_offset_threshold(&mut self, v: f32) -> &mut Self { self.offset_threshold = v; self }
    /// Mutating: set the sliding-window step in samples.
    pub fn set_step_samples(&mut self, v: u32) -> &mut Self { self.step_samples = v; self }
    /// Mutating: set the minimum voice-span duration.
    pub fn set_min_voice_duration(&mut self, v: Duration) -> &mut Self { self.min_voice_duration = v; self }
    /// Mutating: set the minimum speaker-activity duration.
    pub fn set_min_activity_duration(&mut self, v: Duration) -> &mut Self { self.min_activity_duration = v; self }
    /// Mutating: set the voice-span merge gap.
    pub fn set_voice_merge_gap(&mut self, v: Duration) -> &mut Self { self.voice_merge_gap = v; self }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pyannote() {
        let o = SegmentOptions::default();
        assert_eq!(o.onset_threshold(), 0.5);
        assert!((o.offset_threshold() - 0.357).abs() < 1e-6);
        assert_eq!(o.step_samples(), 40_000);
        assert_eq!(o.min_voice_duration(), Duration::ZERO);
    }

    #[test]
    fn builder_round_trip() {
        let o = SegmentOptions::new()
            .with_onset_threshold(0.6)
            .with_offset_threshold(0.4)
            .with_step_samples(20_000)
            .with_min_voice_duration(Duration::from_millis(100))
            .with_min_activity_duration(Duration::from_millis(50))
            .with_voice_merge_gap(Duration::from_millis(30));

        assert_eq!(o.onset_threshold(), 0.6);
        assert_eq!(o.offset_threshold(), 0.4);
        assert_eq!(o.step_samples(), 20_000);
        assert_eq!(o.min_voice_duration(), Duration::from_millis(100));
        assert_eq!(o.min_activity_duration(), Duration::from_millis(50));
        assert_eq!(o.voice_merge_gap(), Duration::from_millis(30));
    }

    #[test]
    fn sample_rate_tb_matches_constant() {
        assert_eq!(SAMPLE_RATE_TB.den().get(), SAMPLE_RATE_HZ);
        assert_eq!(SAMPLE_RATE_TB.num(), 1);
    }
}
```

- [ ] **Step 3.4: Create stub modules so the crate compiles**

Each of these gets one line so `mod.rs` doesn't break the build. We'll fill them in subsequent tasks. For each path below, create a one-line file:

```rust
// src/segment/error.rs
//! Error types for `diarization::segment` (filled in Task 4).
```

```rust
// src/segment/types.rs
//! Public types for `diarization::segment` (filled in Task 5).
```

```rust
// src/segment/powerset.rs
//! Powerset decoding (filled in Task 6).
```

```rust
// src/segment/hysteresis.rs
//! Hysteresis binarization (filled in Task 7).
```

```rust
// src/segment/stitch.rs
//! Voice-probability stitching (filled in Task 8).
```

```rust
// src/segment/window.rs
//! Sliding-window planning (filled in Task 9).
```

```rust
// src/segment/segmenter.rs
//! Layer 1 Segmenter (filled in Task 10).
```

Each file goes under `/Users/user/Develop/findit-studio/dia/src/segment/`.

- [ ] **Step 3.5: Adjust mod.rs to only expose what exists yet**

Temporarily comment out the `pub use` lines that reference items not yet defined. Edit `src/segment/mod.rs`:

```rust
//! Speaker segmentation: Sans-I/O state machine + optional ort driver.

mod error;
mod hysteresis;
mod options;
mod powerset;
mod segmenter;
mod stitch;
mod types;
mod window;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
mod model;

// pub use error::Error;
pub use options::{
    FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, SAMPLE_RATE_HZ,
    SAMPLE_RATE_TB, SegmentOptions, WINDOW_SAMPLES,
};
// pub use segmenter::Segmenter;
// pub use types::{Action, Event, SpeakerActivity, WindowId};

// #[cfg(feature = "ort")]
// #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
// pub use model::SegmentModel;
```

We'll uncomment as each module lands.

- [ ] **Step 3.6: Also stub model.rs (cfg ort)**

Create `/Users/user/Develop/findit-studio/dia/src/segment/model.rs`:

```rust
//! Layer 2 SegmentModel (filled in Task 11).
```

- [ ] **Step 3.7: Run option tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std options::tests
```

Expected: 3 tests pass.

- [ ] **Step 3.8: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/ && \
  git commit -m "feat(segment): module skeleton, constants, SegmentOptions"
```

---

## Task 4: Public types — WindowId, SpeakerActivity, Action, Event

> Note: Tasks 4 and 5 were swapped after the original plan was written. The
> error type references `WindowId` from this module, so types must land first.

**Files:**
- Modify: `src/segment/types.rs`, `src/segment/mod.rs`

- [ ] **Step 4.1: Write the type definitions and tests**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/types.rs` with:

```rust
//! Public types emitted by the segmentation state machine.

extern crate alloc;

use mediatime::{TimeRange, Timestamp};

/// Stable correlation handle for one inference round-trip.
///
/// Equal to the window's sample range in `SAMPLE_RATE_TB`. Two `WindowId`s
/// with the same start and end compare equal and hash to the same value, so
/// it is safe to use as a `HashMap` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(TimeRange);

impl WindowId {
    pub(crate) const fn new(range: TimeRange) -> Self { Self(range) }
    /// Sample-range covered by the window in `SAMPLE_RATE_TB`.
    pub const fn range(&self) -> TimeRange { self.0 }
    /// Window start as a `Timestamp`.
    pub const fn start(&self) -> Timestamp { self.0.start() }
    /// Window end as a `Timestamp`.
    pub const fn end(&self) -> Timestamp { self.0.end() }
    /// Window duration (always 10 s for v0.1.0).
    pub fn duration(&self) -> core::time::Duration { self.0.duration() }
}

/// One window-local speaker activity.
///
/// `speaker_slot` ∈ `0..=2` is local to the emitting window — slot identity
/// does NOT cross windows. Cross-window speaker identity is the job of a
/// future clustering layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerActivity {
    window_id: WindowId,
    speaker_slot: u8,
    range: TimeRange,
}

impl SpeakerActivity {
    pub(crate) const fn new(window_id: WindowId, speaker_slot: u8, range: TimeRange) -> Self {
        Self { window_id, speaker_slot, range }
    }
    /// The window this activity was decoded from.
    pub const fn window_id(&self) -> WindowId { self.window_id }
    /// Window-local speaker slot (0, 1, or 2).
    pub const fn speaker_slot(&self) -> u8 { self.speaker_slot }
    /// Sample range of the activity within the stream, in `SAMPLE_RATE_TB`.
    pub const fn range(&self) -> TimeRange { self.range }
}

/// One output of the Layer-1 state machine.
#[derive(Debug, Clone)]
pub enum Action {
    /// The caller must run ONNX inference on `samples` and call
    /// [`Segmenter::push_inference`](crate::segment::Segmenter::push_inference)
    /// with the same `id`.
    NeedsInference {
        /// Correlation handle (the window's sample range).
        id: WindowId,
        /// Always `WINDOW_SAMPLES = 160_000` mono float32 samples at 16 kHz,
        /// zero-padded if the input stream is shorter.
        samples: alloc::boxed::Box<[f32]>,
    },
    /// A decoded window-local speaker activity.
    Activity(SpeakerActivity),
    /// A finalized speaker-agnostic voice region.
    VoiceSpan(TimeRange),
}

/// Layer-2 emission events (Layer 2 hides `NeedsInference` from the caller).
#[derive(Debug, Clone)]
pub enum Event {
    /// A decoded window-local speaker activity.
    Activity(SpeakerActivity),
    /// A finalized speaker-agnostic voice region.
    VoiceSpan(TimeRange),
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::segment::options::SAMPLE_RATE_TB;

    fn tr(start: i64, end: i64) -> TimeRange {
        TimeRange::new(start, end, SAMPLE_RATE_TB)
    }

    #[test]
    fn window_id_accessors() {
        let id = WindowId::new(tr(0, 160_000));
        assert_eq!(id.range(), tr(0, 160_000));
        assert_eq!(id.start().pts(), 0);
        assert_eq!(id.end().pts(), 160_000);
        assert_eq!(id.duration(), core::time::Duration::from_secs(10));
    }

    #[test]
    fn window_id_hash_eq_value_semantic() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(WindowId::new(tr(0, 160_000)));
        assert!(s.contains(&WindowId::new(tr(0, 160_000))));
        assert!(!s.contains(&WindowId::new(tr(40_000, 200_000))));
    }

    #[test]
    fn speaker_activity_accessors() {
        let win = WindowId::new(tr(0, 160_000));
        let act = SpeakerActivity::new(win, 1, tr(8_000, 24_000));
        assert_eq!(act.window_id(), win);
        assert_eq!(act.speaker_slot(), 1);
        assert_eq!(act.range(), tr(8_000, 24_000));
    }
}
```

- [ ] **Step 4.2: Uncomment the type re-exports in mod.rs**

In `src/segment/mod.rs`, change:

```rust
// pub use types::{Action, Event, SpeakerActivity, WindowId};
```

to:

```rust
pub use types::{Action, Event, SpeakerActivity, WindowId};
```

- [ ] **Step 4.3: Run the type tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std types::tests
```

Expected: 3 tests pass.

- [ ] **Step 4.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/types.rs src/segment/mod.rs && \
  git commit -m "feat(segment): public types — WindowId, SpeakerActivity, Action, Event"
```

---

## Task 5: Error type

**Files:**
- Modify: `src/segment/error.rs`, `src/segment/mod.rs`

- [ ] **Step 5.1: Write src/segment/error.rs**

```rust
//! Error type for the segmentation module.

use core::fmt;

#[cfg(feature = "std")]
use std::path::PathBuf;

use crate::segment::types::WindowId;

/// All errors produced by `diarization::segment`.
#[derive(Debug)]
pub enum Error {
    /// Construction-time validation failure for `SegmentOptions`.
    InvalidOptions(&'static str),

    /// `push_inference` received a `scores` slice of the wrong length.
    InferenceShapeMismatch {
        /// Expected element count: `FRAMES_PER_WINDOW * POWERSET_CLASSES`.
        expected: usize,
        /// Actual length received.
        got: usize,
    },

    /// `push_inference` was called with a `WindowId` that wasn't yielded by
    /// `poll` (or that has already been consumed).
    UnknownWindow {
        /// The unknown id.
        id: WindowId,
    },

    /// The `ort::Session` failed to load the model file.
    #[cfg(feature = "ort")]
    LoadModel {
        /// Path passed to `from_file`.
        path: PathBuf,
        /// Underlying ort error.
        source: ort::Error,
    },

    /// Generic ort runtime error from `SegmentModel::infer` or session ops.
    #[cfg(feature = "ort")]
    Ort(ort::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "invalid segment options: {msg}"),
            Self::InferenceShapeMismatch { expected, got } => {
                write!(f, "inference scores length {got}, expected {expected}")
            }
            Self::UnknownWindow { id } => {
                write!(f, "inference scores received for unknown WindowId {id:?}")
            }
            #[cfg(feature = "ort")]
            Self::LoadModel { path, source } => {
                write!(f, "failed to load model from {}: {source}", path.display())
            }
            #[cfg(feature = "ort")]
            Self::Ort(e) => write!(f, "ort runtime error: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "ort")]
            Self::LoadModel { source, .. } => Some(source),
            #[cfg(feature = "ort")]
            Self::Ort(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "ort")]
impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self { Self::Ort(e) }
}
```

We don't use `thiserror` here because `WindowId` from `types.rs` is forward-referenced and we want zero macro magic in the no-std path. (Switch to `thiserror` later if it proves cleaner; both compile fine.)

- [ ] **Step 5.2: Uncomment the Error re-export in mod.rs**

Edit `src/segment/mod.rs` and change:

```rust
// pub use error::Error;
```

to:

```rust
pub use error::Error;
```

- [ ] **Step 5.3: Verify build**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo build --no-default-features --features std
```

Expected: still builds. (`Error::LoadModel` / `Error::Ort` variants vanish without `ort`.)

- [ ] **Step 5.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/error.rs src/segment/mod.rs && \
  git commit -m "feat(segment): Error type"
```

---

## Task 6: Powerset decoding

**Files:**
- Modify: `src/segment/powerset.rs`

- [ ] **Step 6.1: Write the failing tests**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/powerset.rs` with:

```rust
//! Powerset → per-speaker probability decoding.
//!
//! pyannote/segmentation-3.0 outputs 7 logits per output frame, encoding
//! every subset of up to 3 simultaneous speakers:
//!
//! | class | meaning |
//! |-------|---------|
//! | 0     | silence |
//! | 1     | speaker A only |
//! | 2     | speaker B only |
//! | 3     | speaker C only |
//! | 4     | A + B   |
//! | 5     | A + C   |
//! | 6     | B + C   |
//!
//! Per-speaker probability is the marginal: speaker A is active iff class
//! 1, 4, or 5 fired. Voice (any speaker) probability is `1 - p(silence)`.

use crate::segment::options::POWERSET_CLASSES;

/// Numerically stable softmax over one row of [`POWERSET_CLASSES`] logits.
pub(crate) fn softmax_row(logits: &[f32; POWERSET_CLASSES]) -> [f32; POWERSET_CLASSES] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0f32; POWERSET_CLASSES];
    let mut sum = 0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        out[i] = e;
        sum += e;
    }
    debug_assert!(sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

/// Per-speaker probabilities `[p(A), p(B), p(C)]` from a softmaxed
/// [`POWERSET_CLASSES`] row.
pub(crate) fn powerset_to_speakers(probs: &[f32; POWERSET_CLASSES]) -> [f32; 3] {
    [
        probs[1] + probs[4] + probs[5],
        probs[2] + probs[4] + probs[6],
        probs[3] + probs[5] + probs[6],
    ]
}

/// Voice probability (= `1 - p(silence)`) for one softmaxed row.
pub(crate) fn voice_prob(probs: &[f32; POWERSET_CLASSES]) -> f32 {
    1.0 - probs[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_row_sums_to_one() {
        let logits = [-1.0, 2.0, 0.5, 1.5, -0.3, 0.0, 0.7];
        let p = softmax_row(&logits);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        for &v in &p {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn softmax_row_stable_with_extreme_logits() {
        let logits = [1000.0, 1001.0, 999.0, 1000.5, 998.0, 1000.2, 999.8];
        let p = softmax_row(&logits);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn powerset_pure_silence() {
        let probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s = powerset_to_speakers(&probs);
        assert_eq!(s, [0.0, 0.0, 0.0]);
        assert_eq!(voice_prob(&probs), 0.0);
    }

    #[test]
    fn powerset_pure_speaker_a() {
        let probs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s = powerset_to_speakers(&probs);
        assert_eq!(s, [1.0, 0.0, 0.0]);
        assert_eq!(voice_prob(&probs), 1.0);
    }

    #[test]
    fn powerset_a_and_b_overlap() {
        // 50% A+B, 50% silence
        let probs = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
        let s = powerset_to_speakers(&probs);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 0.5).abs() < 1e-6);
        assert_eq!(s[2], 0.0);
        assert!((voice_prob(&probs) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn powerset_marginals_sum_correctly() {
        // 0.1 silence, 0.2 A, 0.1 B, 0.05 C, 0.3 A+B, 0.15 A+C, 0.1 B+C
        let probs = [0.1, 0.2, 0.1, 0.05, 0.3, 0.15, 0.1];
        let s = powerset_to_speakers(&probs);
        // p(A) = 0.2 + 0.3 + 0.15 = 0.65
        // p(B) = 0.1 + 0.3 + 0.10 = 0.50
        // p(C) = 0.05 + 0.15 + 0.10 = 0.30
        assert!((s[0] - 0.65).abs() < 1e-6);
        assert!((s[1] - 0.50).abs() < 1e-6);
        assert!((s[2] - 0.30).abs() < 1e-6);
        assert!((voice_prob(&probs) - 0.9).abs() < 1e-6);
    }
}
```

- [ ] **Step 6.2: Run the powerset tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std powerset::tests
```

Expected: 5 tests pass.

- [ ] **Step 6.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/powerset.rs && \
  git commit -m "feat(segment): powerset decoding (softmax + 7→3 marginals)"
```

---

## Task 7: Hysteresis binarization + RLE

**Files:**
- Modify: `src/segment/hysteresis.rs`

- [ ] **Step 7.1: Write tests + implementation**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/hysteresis.rs` with:

```rust
//! Two-threshold hysteresis state machine and run-length encoding.
//!
//! `binarize` walks a probability sequence with state. The state goes
//! inactive → active when `p >= onset`, and active → inactive when
//! `p < offset`. With `offset < onset` this gives stable boundaries.
//!
//! `runs_of_true` extracts half-open `[start, end)` index ranges where the
//! mask is true.

extern crate alloc;
use alloc::vec::Vec;

/// Stateful hysteresis cursor. Use [`Hysteresis::push`] for streaming use,
/// or [`binarize`] for whole-buffer use.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hysteresis {
    onset: f32,
    offset: f32,
    active: bool,
}

impl Hysteresis {
    pub(crate) const fn new(onset: f32, offset: f32) -> Self {
        Self { onset, offset, active: false }
    }
    /// Step one sample. Returns the new active state.
    pub(crate) fn push(&mut self, p: f32) -> bool {
        self.active = if self.active { p >= self.offset } else { p >= self.onset };
        self.active
    }
    pub(crate) fn is_active(&self) -> bool { self.active }
    pub(crate) fn reset(&mut self) { self.active = false; }
}

/// Apply hysteresis to a probability sequence (no carried state).
pub(crate) fn binarize(probs: &[f32], onset: f32, offset: f32) -> Vec<bool> {
    let mut h = Hysteresis::new(onset, offset);
    probs.iter().map(|&p| h.push(p)).collect()
}

/// RLE of a boolean mask into half-open `[start, end)` index ranges of true.
pub(crate) fn runs_of_true(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &b) in mask.iter().enumerate() {
        match (b, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, mask.len()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binarize_simple_step() {
        let probs = [0.0, 0.4, 0.6, 0.5, 0.4, 0.3, 0.0];
        // onset 0.5, offset 0.4. State: 0,0,1,1,1,0,0 (active until p<0.4 at index 5).
        let m = binarize(&probs, 0.5, 0.4);
        assert_eq!(m, [false, false, true, true, true, false, false]);
    }

    #[test]
    fn binarize_hysteresis_prevents_flicker() {
        // probabilities oscillate between 0.45 and 0.55 around the onset.
        // With onset 0.5, offset 0.4, once active we stay active because
        // p >= 0.4 throughout.
        let probs = [0.55, 0.45, 0.55, 0.45, 0.55];
        let m = binarize(&probs, 0.5, 0.4);
        assert_eq!(m, [true, true, true, true, true]);
    }

    #[test]
    fn binarize_empty() {
        let m = binarize(&[], 0.5, 0.4);
        assert!(m.is_empty());
    }

    #[test]
    fn binarize_all_below_onset_stays_inactive() {
        let probs = [0.0, 0.1, 0.2, 0.3, 0.49];
        let m = binarize(&probs, 0.5, 0.4);
        assert_eq!(m, [false, false, false, false, false]);
    }

    #[test]
    fn runs_basic() {
        let m = [false, true, true, false, true, false, true, true, true];
        assert_eq!(runs_of_true(&m), vec![(1, 3), (4, 5), (6, 9)]);
    }

    #[test]
    fn runs_all_false() {
        let m = [false; 5];
        assert!(runs_of_true(&m).is_empty());
    }

    #[test]
    fn runs_all_true() {
        let m = [true; 4];
        assert_eq!(runs_of_true(&m), vec![(0, 4)]);
    }

    #[test]
    fn runs_trailing_open_run_closes() {
        let m = [false, true, true];
        assert_eq!(runs_of_true(&m), vec![(1, 3)]);
    }

    #[test]
    fn runs_empty() {
        assert!(runs_of_true(&[]).is_empty());
    }

    #[test]
    fn streaming_hysteresis_matches_batch() {
        let probs = [0.0, 0.4, 0.6, 0.5, 0.4, 0.3, 0.0];
        let mut h = Hysteresis::new(0.5, 0.4);
        let online: Vec<bool> = probs.iter().map(|&p| h.push(p)).collect();
        assert_eq!(online, binarize(&probs, 0.5, 0.4));
    }
}
```

- [ ] **Step 7.2: Run the hysteresis tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std hysteresis::tests
```

Expected: 9 tests pass.

- [ ] **Step 7.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/hysteresis.rs && \
  git commit -m "feat(segment): hysteresis binarization + RLE"
```

---

## Task 8: Sliding-window planner

**Files:**
- Modify: `src/segment/window.rs`

- [ ] **Step 8.1: Write tests + implementation**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/window.rs` with:

```rust
//! Sliding-window scheduling.
//!
//! Windows step at `step_samples` intervals. If the regular grid does not
//! cover the entire stream, a final tail window is anchored to end-of-stream
//! so the last `WINDOW_SAMPLES` samples are always processed.

extern crate alloc;
use alloc::vec::Vec;

use crate::segment::options::WINDOW_SAMPLES;

/// Plan output: the start sample of each scheduled window. Each window
/// covers `[start, start + WINDOW_SAMPLES)`.
///
/// `total_samples` is the full stream length.
/// Returns at minimum one window (anchored at 0, possibly padded) when
/// `total_samples > 0`. Empty streams yield an empty plan.
pub(crate) fn plan_starts(total_samples: u64, step_samples: u32) -> Vec<u64> {
    if total_samples == 0 {
        return Vec::new();
    }
    let step = step_samples as u64;
    debug_assert!(step > 0, "step_samples must be > 0");
    let win = WINDOW_SAMPLES as u64;

    let mut out = Vec::new();
    let mut s: u64 = 0;
    // Schedule regular windows that fully fit.
    while s + win <= total_samples {
        out.push(s);
        s += step;
    }
    // Tail anchor: ensure the final window ends at total_samples (or covers
    // [0, total_samples) if total < window).
    let tail_start = total_samples.saturating_sub(win);
    if out.last().copied() != Some(tail_start) {
        out.push(tail_start);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_no_windows() {
        assert!(plan_starts(0, 40_000).is_empty());
    }

    #[test]
    fn shorter_than_one_window_yields_one_anchored_window() {
        let p = plan_starts(50_000, 40_000);
        assert_eq!(p, vec![0]); // tail_start = 0 (50_000 - 160_000 saturates).
    }

    #[test]
    fn exact_one_window_no_tail_duplicate() {
        let p = plan_starts(160_000, 40_000);
        // Regular schedule places a window at 0; tail_start is also 0.
        assert_eq!(p, vec![0]);
    }

    #[test]
    fn regular_grid_then_tail_anchor() {
        // 200_000 samples, step 40_000: regular fits at 0 and 40_000
        // (since 40_000 + 160_000 = 200_000 == total). Next would be 80_000
        // (80_000 + 160_000 = 240_000 > 200_000), so stop. tail_start = 40_000,
        // already last → no duplicate.
        let p = plan_starts(200_000, 40_000);
        assert_eq!(p, vec![0, 40_000]);
    }

    #[test]
    fn regular_grid_with_separate_tail() {
        // 230_000 samples, step 40_000: regular windows at 0, 40_000.
        // 80_000 + 160_000 = 240_000 > 230_000, stop. tail_start = 70_000,
        // distinct from 40_000 → push as tail.
        let p = plan_starts(230_000, 40_000);
        assert_eq!(p, vec![0, 40_000, 70_000]);
    }

    #[test]
    fn step_equal_to_window_no_overlap() {
        // step == window, total = 320_000 → windows at 0 and 160_000, tail same as last.
        let p = plan_starts(320_000, 160_000);
        assert_eq!(p, vec![0, 160_000]);
    }
}
```

- [ ] **Step 8.2: Run the window tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std window::tests
```

Expected: 6 tests pass.

- [ ] **Step 8.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/window.rs && \
  git commit -m "feat(segment): sliding-window planner with tail anchor"
```

---

## Task 9: Voice-probability stitcher

**Files:**
- Modify: `src/segment/stitch.rs`

The stitcher accumulates per-window voice probabilities (one float per window-frame at `samples_per_frame ≈ 271.64`) into a continuous stream-indexed buffer where overlapping windows are averaged, then converts that to a per-sample voice probability the segmenter can hysteresis-binarize.

To keep arithmetic exact and align with pyannote's frame layout, we use the integer recipe:
`frame_idx → sample_idx = (frame_idx * WINDOW_SAMPLES) / FRAMES_PER_WINDOW` (rounded). For two consecutive frames the boundary sample is `frame_to_sample(f+1)`. We expand each frame's probability over `[frame_to_sample(f), frame_to_sample(f+1))`. The stitcher stores per-sample sum and contribution count so we can finalize windows incrementally.

- [ ] **Step 9.1: Write tests + implementation**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/stitch.rs` with:

```rust
//! Overlap-add stitching of per-window voice probabilities.
//!
//! Each window contributes `FRAMES_PER_WINDOW` voice probabilities expanded
//! over its `WINDOW_SAMPLES` sample span. Overlapping windows are averaged
//! sample-by-sample.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::segment::options::{FRAMES_PER_WINDOW, WINDOW_SAMPLES};

/// Convert a frame index in `0..=FRAMES_PER_WINDOW` to a sample offset in
/// `0..=WINDOW_SAMPLES` using rounded integer arithmetic.
#[inline]
pub(crate) const fn frame_to_sample(frame_idx: u32) -> u32 {
    // (frame_idx * WINDOW_SAMPLES + FRAMES_PER_WINDOW/2) / FRAMES_PER_WINDOW
    let n = frame_idx as u64 * WINDOW_SAMPLES as u64;
    let half = (FRAMES_PER_WINDOW as u64) / 2;
    ((n + half) / FRAMES_PER_WINDOW as u64) as u32
}

/// Stream-indexed accumulator for voice probability. Windows contribute via
/// [`add_window`]; finalized samples are exposed via [`take_finalized`].
pub(crate) struct VoiceStitcher {
    /// First absolute sample index represented in `sum` / `count`.
    base_sample: u64,
    /// Per-sample contribution sum.
    sum: VecDeque<f32>,
    /// Per-sample contribution count.
    count: VecDeque<u32>,
}

impl VoiceStitcher {
    pub(crate) fn new() -> Self {
        Self { base_sample: 0, sum: VecDeque::new(), count: VecDeque::new() }
    }

    pub(crate) fn clear(&mut self) {
        self.base_sample = 0;
        self.sum.clear();
        self.count.clear();
    }

    /// Add one window of per-frame voice probabilities (length
    /// `FRAMES_PER_WINDOW`) starting at absolute sample `start_sample`.
    pub(crate) fn add_window(&mut self, start_sample: u64, voice_per_frame: &[f32]) {
        debug_assert_eq!(voice_per_frame.len(), FRAMES_PER_WINDOW);
        debug_assert!(start_sample >= self.base_sample);

        // Ensure capacity covers [start_sample, start_sample + WINDOW_SAMPLES).
        let end_sample = start_sample + WINDOW_SAMPLES as u64;
        let needed_len = (end_sample - self.base_sample) as usize;
        while self.sum.len() < needed_len {
            self.sum.push_back(0.0);
            self.count.push_back(0);
        }

        // Expand each frame across its sample range.
        for f in 0..FRAMES_PER_WINDOW {
            let s0 = frame_to_sample(f as u32) as u64;
            let s1 = frame_to_sample(f as u32 + 1) as u64;
            let p = voice_per_frame[f];
            for s in s0..s1 {
                let abs = start_sample + s;
                let idx = (abs - self.base_sample) as usize;
                self.sum[idx] += p;
                self.count[idx] += 1;
            }
        }
    }

    /// Drain finalized samples up to but not including `up_to_sample`.
    /// Returns per-sample averaged voice probabilities and advances the base.
    pub(crate) fn take_finalized(&mut self, up_to_sample: u64) -> Vec<f32> {
        debug_assert!(up_to_sample >= self.base_sample);
        let n = (up_to_sample.saturating_sub(self.base_sample)) as usize;
        let n = n.min(self.sum.len());
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let s = self.sum.pop_front().unwrap();
            let c = self.count.pop_front().unwrap();
            out.push(if c == 0 { 0.0 } else { s / c as f32 });
        }
        self.base_sample += n as u64;
        out
    }

    pub(crate) fn base_sample(&self) -> u64 { self.base_sample }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones_window() -> Vec<f32> { vec![1.0; FRAMES_PER_WINDOW] }
    fn zeros_window() -> Vec<f32> { vec![0.0; FRAMES_PER_WINDOW] }

    #[test]
    fn frame_to_sample_endpoints() {
        assert_eq!(frame_to_sample(0), 0);
        assert_eq!(frame_to_sample(FRAMES_PER_WINDOW as u32), WINDOW_SAMPLES);
    }

    #[test]
    fn frame_to_sample_monotonic() {
        let mut prev = 0u32;
        for f in 1..=FRAMES_PER_WINDOW as u32 {
            let s = frame_to_sample(f);
            assert!(s >= prev);
            prev = s;
        }
    }

    #[test]
    fn single_window_finalize_all() {
        let mut s = VoiceStitcher::new();
        s.add_window(0, &ones_window());
        let out = s.take_finalized(WINDOW_SAMPLES as u64);
        assert_eq!(out.len(), WINDOW_SAMPLES as usize);
        for v in out { assert!((v - 1.0).abs() < 1e-6); }
        assert_eq!(s.base_sample(), WINDOW_SAMPLES as u64);
    }

    #[test]
    fn two_overlapping_windows_average() {
        let mut s = VoiceStitcher::new();
        s.add_window(0, &ones_window());            // covers [0, 160_000)
        s.add_window(40_000, &zeros_window());      // covers [40_000, 200_000)
        // [0, 40_000) only window 1 contributed → 1.0
        // [40_000, 160_000) overlap → 0.5
        // [160_000, 200_000) only window 2 → 0.0
        let out = s.take_finalized(200_000);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[39_999] - 1.0).abs() < 1e-6);
        assert!((out[40_000] - 0.5).abs() < 1e-6);
        assert!((out[159_999] - 0.5).abs() < 1e-6);
        assert!(out[160_000].abs() < 1e-6);
        assert!(out[199_999].abs() < 1e-6);
    }

    #[test]
    fn partial_finalize_advances_base() {
        let mut s = VoiceStitcher::new();
        s.add_window(0, &ones_window());
        let part = s.take_finalized(40_000);
        assert_eq!(part.len(), 40_000);
        assert_eq!(s.base_sample(), 40_000);
        // Remaining samples still reachable.
        let rest = s.take_finalized(WINDOW_SAMPLES as u64);
        assert_eq!(rest.len(), 120_000);
        assert_eq!(s.base_sample(), WINDOW_SAMPLES as u64);
    }

    #[test]
    fn clear_resets() {
        let mut s = VoiceStitcher::new();
        s.add_window(0, &ones_window());
        s.clear();
        assert_eq!(s.base_sample(), 0);
        assert!(s.take_finalized(100).is_empty());
    }
}
```

- [ ] **Step 9.2: Run the stitch tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std stitch::tests
```

Expected: 5 tests pass.

- [ ] **Step 9.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/stitch.rs && \
  git commit -m "feat(segment): voice-probability stitcher with overlap-add average"
```

---

## Task 10: Segmenter — state machine, push_samples, poll

**Files:**
- Modify: `src/segment/segmenter.rs`

This task lands the data structure plus `push_samples` / `poll` for the `NeedsInference` half. `push_inference` and emission come in Task 11; `finish` and `clear` in Task 12.

- [ ] **Step 10.1: Write the data structure and constructor**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/segmenter.rs` with:

```rust
//! Layer-1 Sans-I/O speaker segmentation state machine.

extern crate alloc;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

use mediatime::TimeRange;

use crate::segment::error::Error;
use crate::segment::hysteresis::{Hysteresis, runs_of_true};
use crate::segment::options::{
    FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, SAMPLE_RATE_TB, SegmentOptions,
    WINDOW_SAMPLES,
};
use crate::segment::powerset::{powerset_to_speakers, softmax_row, voice_prob};
use crate::segment::stitch::VoiceStitcher;
use crate::segment::types::{Action, SpeakerActivity, WindowId};
use crate::segment::window::plan_starts;

/// Sans-I/O speaker segmentation state machine.
///
/// See the module docs for the high-level data flow. In short:
///
/// 1. Caller appends PCM via [`push_samples`](Self::push_samples).
/// 2. Caller drains [`Action`]s via [`poll`](Self::poll). When it sees
///    [`Action::NeedsInference`], it runs the model on the supplied samples
///    and calls [`push_inference`](Self::push_inference) with the scores.
/// 3. After all PCM is delivered, caller calls [`finish`](Self::finish) and
///    drains remaining actions.
///
/// `Segmenter` is `Send` but not `Sync`: use one per stream.
pub struct Segmenter {
    opts: SegmentOptions,

    /// Rolling sample buffer. Index 0 corresponds to absolute sample
    /// `consumed_samples`.
    input: VecDeque<f32>,
    consumed_samples: u64,

    /// Index of the next window to schedule (== how many windows already
    /// scheduled). Window k covers `[k * step_samples, k * step_samples + WINDOW_SAMPLES)`
    /// in absolute samples — *unless* it's the tail anchor.
    next_window_idx: u32,

    /// Pending inference round-trips: id → (window-start sample).
    pending: BTreeMap<WindowId, u64>,

    /// Output queue.
    pending_actions: VecDeque<Action>,

    /// Stream-indexed voice-probability accumulator.
    stitcher: VoiceStitcher,

    /// Online hysteresis cursor for the voice timeline.
    voice_hyst: Hysteresis,
    /// If voice is currently active, when did the run start?
    voice_run_start: Option<u64>,

    /// Once `finish()` has been called we may schedule the tail window.
    finished: bool,
    /// Final tail window has been emitted as `NeedsInference`.
    tail_emitted: bool,
    /// Total stream length latched at `finish()`.
    total_samples: u64,
}

impl Segmenter {
    /// Construct a new segmenter.
    pub fn new(opts: SegmentOptions) -> Self {
        let onset = opts.onset_threshold();
        let offset = opts.offset_threshold();
        Self {
            opts,
            input: VecDeque::new(),
            consumed_samples: 0,
            next_window_idx: 0,
            pending: BTreeMap::new(),
            pending_actions: VecDeque::new(),
            stitcher: VoiceStitcher::new(),
            voice_hyst: Hysteresis::new(onset, offset),
            voice_run_start: None,
            finished: false,
            tail_emitted: false,
            total_samples: 0,
        }
    }

    /// Read-only access to the configured options.
    pub fn options(&self) -> &SegmentOptions { &self.opts }
}
```

- [ ] **Step 10.2: Add push_samples and the internal scheduler**

Append to the same file:

```rust
impl Segmenter {
    /// Append 16 kHz mono float32 PCM samples. Arbitrary chunk size.
    ///
    /// Calling after [`finish`](Self::finish) is a programming bug; the call
    /// is silently ignored in release builds and panics in debug.
    pub fn push_samples(&mut self, samples: &[f32]) {
        debug_assert!(!self.finished, "push_samples after finish");
        if self.finished {
            return;
        }
        self.input.extend(samples.iter().copied());
        self.schedule_ready_windows();
    }

    /// Schedule any regular windows that are now fully buffered. Tail
    /// scheduling happens in `finish()`.
    fn schedule_ready_windows(&mut self) {
        let step = self.opts.step_samples() as u64;
        let win = WINDOW_SAMPLES as u64;
        loop {
            let start = self.next_window_idx as u64 * step;
            let end = start + win;
            // Buffered samples cover [consumed_samples, consumed_samples + input.len()).
            let buffered_end = self.consumed_samples + self.input.len() as u64;
            if buffered_end < end {
                return; // not enough audio yet
            }
            self.emit_window(start, /* zero_pad_to_window = */ false);
            self.next_window_idx += 1;
        }
    }

    /// Build a window starting at `start` (absolute samples), copy its
    /// samples (zero-padding if needed), enqueue `NeedsInference`, and
    /// trim the input buffer.
    fn emit_window(&mut self, start: u64, zero_pad_to_window: bool) {
        let win = WINDOW_SAMPLES as u64;
        let buffered_end = self.consumed_samples + self.input.len() as u64;

        // Copy samples [start, start + WINDOW_SAMPLES) from `input`, padding
        // with zeros past `buffered_end` if `zero_pad_to_window` is true.
        let mut samples: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
        let avail_end = if zero_pad_to_window { start + win } else { buffered_end.min(start + win) };

        let copy_from = (start.saturating_sub(self.consumed_samples)) as usize;
        let copy_until = (avail_end.saturating_sub(self.consumed_samples)) as usize;
        for i in copy_from..copy_until {
            samples.push(self.input[i]);
        }
        // Zero-pad tail if needed.
        while samples.len() < WINDOW_SAMPLES as usize {
            samples.push(0.0);
        }

        let id = WindowId::new(TimeRange::new(start as i64, (start + win) as i64, SAMPLE_RATE_TB));
        self.pending.insert(id, start);
        self.pending_actions.push_back(Action::NeedsInference {
            id,
            samples: samples.into_boxed_slice(),
        });

        // Trim samples that no future window will need. Next window (if any)
        // starts at next_window_idx * step. Anything before that can go.
        let next_start = (self.next_window_idx + 1) as u64 * self.opts.step_samples() as u64;
        self.trim_input_to(next_start);
    }

    fn trim_input_to(&mut self, abs_sample: u64) {
        let target = abs_sample.min(self.consumed_samples + self.input.len() as u64);
        let drop_n = (target.saturating_sub(self.consumed_samples)) as usize;
        for _ in 0..drop_n {
            self.input.pop_front();
        }
        self.consumed_samples += drop_n as u64;
    }

    /// Drain the next pending action.
    pub fn poll(&mut self) -> Option<Action> {
        self.pending_actions.pop_front()
    }
}
```

- [ ] **Step 10.3: Add unit tests for push_samples + poll**

Append to the same file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mediatime::TimeRange;

    fn opts() -> SegmentOptions { SegmentOptions::default() }

    #[test]
    fn empty_no_actions() {
        let mut s = Segmenter::new(opts());
        assert!(s.poll().is_none());
    }

    #[test]
    fn first_window_emits_after_full_window_buffered() {
        let mut s = Segmenter::new(opts());
        // Push 80_000 samples — half a window. No action yet.
        s.push_samples(&vec![0.1f32; 80_000]);
        assert!(s.poll().is_none());
        // Push another 80_000 — now we have a full window.
        s.push_samples(&vec![0.2f32; 80_000]);
        match s.poll() {
            Some(Action::NeedsInference { id, samples }) => {
                assert_eq!(samples.len(), WINDOW_SAMPLES as usize);
                assert_eq!(id.range(), TimeRange::new(0, 160_000, SAMPLE_RATE_TB));
                // First half is 0.1, second half is 0.2.
                assert!((samples[0] - 0.1).abs() < 1e-6);
                assert!((samples[80_000] - 0.2).abs() < 1e-6);
            }
            other => panic!("expected NeedsInference, got {other:?}"),
        }
        assert!(s.poll().is_none());
    }

    #[test]
    fn second_window_emits_after_one_step_more_audio() {
        let mut s = Segmenter::new(opts());
        s.push_samples(&vec![0.0f32; 160_000]);
        let _first = s.poll();
        // After 200_000 total samples we have second window covering [40_000, 200_000).
        s.push_samples(&vec![0.0f32; 40_000]);
        match s.poll() {
            Some(Action::NeedsInference { id, .. }) => {
                assert_eq!(id.range(), TimeRange::new(40_000, 200_000, SAMPLE_RATE_TB));
            }
            other => panic!("expected NeedsInference, got {other:?}"),
        }
    }
}
```

- [ ] **Step 10.4: Uncomment Segmenter re-export**

Edit `src/segment/mod.rs`, change `// pub use segmenter::Segmenter;` to `pub use segmenter::Segmenter;`.

- [ ] **Step 10.5: Run the segmenter tests so far**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std segmenter::tests
```

Expected: 3 tests pass.

- [ ] **Step 10.6: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/segmenter.rs src/segment/mod.rs && \
  git commit -m "feat(segment): Segmenter struct, push_samples, poll"
```

---

## Task 11: Segmenter — push_inference and event emission

**Files:**
- Modify: `src/segment/segmenter.rs`

- [ ] **Step 11.1: Add push_inference and emission helpers**

Append the following block to `src/segment/segmenter.rs` (after the existing `impl Segmenter` blocks, before the `tests` module):

```rust
impl Segmenter {
    /// Provide ONNX inference results for a previously-yielded window.
    ///
    /// `scores` must have length `FRAMES_PER_WINDOW * POWERSET_CLASSES = 4123`
    /// (powerset logits for each output frame).
    pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error> {
        let expected = FRAMES_PER_WINDOW * POWERSET_CLASSES;
        if scores.len() != expected {
            return Err(Error::InferenceShapeMismatch { expected, got: scores.len() });
        }
        let start = self.pending.remove(&id).ok_or(Error::UnknownWindow { id })?;

        // Decode powerset row by row.
        let mut speaker_probs: [Vec<f32>; MAX_SPEAKER_SLOTS as usize] =
            [vec![0.0; FRAMES_PER_WINDOW], vec![0.0; FRAMES_PER_WINDOW], vec![0.0; FRAMES_PER_WINDOW]];
        let mut voice_per_frame: Vec<f32> = Vec::with_capacity(FRAMES_PER_WINDOW);

        for f in 0..FRAMES_PER_WINDOW {
            let row_start = f * POWERSET_CLASSES;
            let mut row = [0f32; POWERSET_CLASSES];
            row.copy_from_slice(&scores[row_start..row_start + POWERSET_CLASSES]);
            let probs = softmax_row(&row);
            voice_per_frame.push(voice_prob(&probs));
            let s = powerset_to_speakers(&probs);
            speaker_probs[0][f] = s[0];
            speaker_probs[1][f] = s[1];
            speaker_probs[2][f] = s[2];
        }

        // Emit per-window speaker activities.
        self.emit_speaker_activities(id, start, &speaker_probs);

        // Feed voice probabilities into the stitcher.
        self.stitcher.add_window(start, &voice_per_frame);

        // Finalize voice probabilities and emit any closed voice spans.
        self.process_voice_finalization();
        Ok(())
    }

    fn emit_speaker_activities(
        &mut self,
        id: WindowId,
        window_start: u64,
        speaker_probs: &[Vec<f32>; MAX_SPEAKER_SLOTS as usize],
    ) {
        let onset = self.opts.onset_threshold();
        let offset = self.opts.offset_threshold();
        let min_dur = self.opts.min_activity_duration();
        let min_samples = duration_to_samples(min_dur);

        for slot in 0..MAX_SPEAKER_SLOTS {
            let probs = &speaker_probs[slot as usize];
            // Per-window hysteresis (no carry — slots are window-local).
            let mut h = Hysteresis::new(onset, offset);
            let mask: Vec<bool> = probs.iter().map(|&p| h.push(p)).collect();
            for (f0, f1) in runs_of_true(&mask) {
                let s0 = window_start + crate::segment::stitch::frame_to_sample(f0 as u32) as u64;
                let s1 = window_start + crate::segment::stitch::frame_to_sample(f1 as u32) as u64;
                if s1 - s0 < min_samples {
                    continue;
                }
                let range = TimeRange::new(s0 as i64, s1 as i64, SAMPLE_RATE_TB);
                self.pending_actions
                    .push_back(Action::Activity(SpeakerActivity::new(id, slot, range)));
            }
        }
    }

    /// Pull finalized voice probabilities out of the stitcher and run them
    /// through the streaming hysteresis cursor, emitting closed voice spans.
    fn process_voice_finalization(&mut self) {
        let up_to = self.next_finalization_boundary();
        if up_to <= self.stitcher.base_sample() {
            return;
        }
        let probs = self.stitcher.take_finalized(up_to);
        let base = self.stitcher.base_sample() - probs.len() as u64; // base before this drain
        for (i, p) in probs.iter().enumerate() {
            let abs = base + i as u64;
            let was_active = self.voice_hyst.is_active();
            let now_active = self.voice_hyst.push(*p);
            match (was_active, now_active) {
                (false, true) => self.voice_run_start = Some(abs),
                (true, false) => {
                    if let Some(start) = self.voice_run_start.take() {
                        self.emit_voice_span(start, abs);
                    }
                }
                _ => {}
            }
        }
    }

    /// The largest absolute sample that no future regular window could
    /// influence. After processing window N, samples `< (N+1) * step` are
    /// finalized (because window N+1 starts there).
    fn next_finalization_boundary(&self) -> u64 {
        let step = self.opts.step_samples() as u64;
        let next_pending_window_start = self.next_window_idx as u64 * step;
        // Windows up to (but not including) next_window_idx have been emitted.
        // After window N processes, samples < (N+1) * step are finalized.
        // Equivalently, `next_pending_window_start` itself is the boundary.
        next_pending_window_start
    }

    fn emit_voice_span(&mut self, start_sample: u64, end_sample: u64) {
        let dur_samples = end_sample - start_sample;
        let min = duration_to_samples(self.opts.min_voice_duration());
        if dur_samples < min {
            return;
        }
        let range = TimeRange::new(start_sample as i64, end_sample as i64, SAMPLE_RATE_TB);
        self.pending_actions.push_back(Action::VoiceSpan(range));
    }
}

#[inline]
fn duration_to_samples(d: core::time::Duration) -> u64 {
    let nanos = d.as_nanos();
    // 16_000 samples/sec ⇒ samples = nanos * 16_000 / 1e9.
    (nanos * crate::segment::options::SAMPLE_RATE_HZ as u128 / 1_000_000_000u128) as u64
}
```

You'll also need to make `frame_to_sample` accessible from `segmenter.rs`. Edit `src/segment/stitch.rs` and change:

```rust
pub(crate) const fn frame_to_sample(frame_idx: u32) -> u32 {
```

(it's already `pub(crate)`, so no edit needed unless your earlier copy made it non-public — verify and adjust).

- [ ] **Step 11.2: Add tests for push_inference**

Append inside the `tests` module in `src/segment/segmenter.rs`:

```rust
    /// Build synthetic powerset logits where speaker A is "active" (class 1
    /// dominates) for frames in `active_frames`, otherwise silence (class 0
    /// dominates). All other classes are negligible.
    fn synth_logits(active_frames: std::ops::Range<usize>) -> Vec<f32> {
        let mut out = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
        for f in 0..FRAMES_PER_WINDOW {
            let row_start = f * POWERSET_CLASSES;
            // Strong negative for everything, then boost the chosen class.
            for c in 0..POWERSET_CLASSES { out[row_start + c] = -10.0; }
            let active = active_frames.contains(&f);
            let dominant = if active { 1 } else { 0 }; // 1 = A only, 0 = silence
            out[row_start + dominant] = 10.0;
        }
        out
    }

    #[test]
    fn push_inference_wrong_length_errors() {
        let mut s = Segmenter::new(opts());
        s.push_samples(&vec![0.0; 160_000]);
        let action = s.poll().unwrap();
        let id = match action {
            Action::NeedsInference { id, .. } => id,
            _ => unreachable!(),
        };
        let bogus = vec![0.0f32; 100];
        match s.push_inference(id, &bogus) {
            Err(Error::InferenceShapeMismatch { expected, got }) => {
                assert_eq!(expected, FRAMES_PER_WINDOW * POWERSET_CLASSES);
                assert_eq!(got, 100);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn push_inference_unknown_window_errors() {
        let mut s = Segmenter::new(opts());
        let bogus_id = WindowId::new(TimeRange::new(0, 160_000, SAMPLE_RATE_TB));
        let scores = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
        match s.push_inference(bogus_id, &scores) {
            Err(Error::UnknownWindow { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn one_window_speaker_a_active_emits_activity() {
        let mut s = Segmenter::new(opts());
        s.push_samples(&vec![0.0; 160_000]);
        let id = match s.poll().unwrap() {
            Action::NeedsInference { id, .. } => id,
            _ => unreachable!(),
        };
        // Speaker A active for a contiguous block of frames.
        let scores = synth_logits(100..200);
        s.push_inference(id, &scores).unwrap();

        // Drain actions; expect at least one Activity for slot 0 within
        // window [0, 160_000).
        let mut saw_activity = false;
        while let Some(a) = s.poll() {
            if let Action::Activity(act) = a {
                assert_eq!(act.window_id(), id);
                assert_eq!(act.speaker_slot(), 0);
                assert_eq!(act.range().timebase(), SAMPLE_RATE_TB);
                saw_activity = true;
            }
        }
        assert!(saw_activity, "expected at least one Activity for slot 0");
    }
}
```

- [ ] **Step 11.3: Run the segmenter tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std segmenter::tests
```

Expected: 6 tests pass (3 from Task 10 + 3 new).

- [ ] **Step 11.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/segmenter.rs && \
  git commit -m "feat(segment): push_inference, speaker activities, voice span emission"
```

---

## Task 12: Segmenter — finish, clear, voice-span tail finalization

**Files:**
- Modify: `src/segment/segmenter.rs`

- [ ] **Step 12.1: Add finish and clear methods**

Append to the `impl Segmenter` block (before the test module):

```rust
impl Segmenter {
    /// Signal end-of-stream. Schedules a tail-anchored window if needed and
    /// causes any open voice span to close on subsequent `poll`s.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.total_samples = self.consumed_samples + self.input.len() as u64;

        if self.tail_emitted {
            return;
        }
        if self.total_samples == 0 {
            return; // nothing to do
        }

        let starts = plan_starts(self.total_samples, self.opts.step_samples());
        let regular_emitted = self.next_window_idx as usize;
        // Anything in `starts` past what we've already emitted is the tail (or
        // a missed regular window plus tail; we just emit them in order).
        for &start in starts.iter().skip(regular_emitted) {
            self.emit_window(start, /* zero_pad_to_window = */ true);
        }
        self.tail_emitted = true;
        // Note: voice finalization for the tail happens after push_inference
        // for the tail window arrives. We close any still-open run there.
    }

    /// Reset to empty state. Internal allocations are reused.
    pub fn clear(&mut self) {
        self.input.clear();
        self.consumed_samples = 0;
        self.next_window_idx = 0;
        self.pending.clear();
        self.pending_actions.clear();
        self.stitcher.clear();
        self.voice_hyst.reset();
        self.voice_run_start = None;
        self.finished = false;
        self.tail_emitted = false;
        self.total_samples = 0;
    }
}
```

- [ ] **Step 12.2: Update finalization helpers to flush at end-of-stream**

The two methods `next_finalization_boundary` and `process_voice_finalization` from Task 11.1 need updates so that, after `finish()` has been called and the tail window's scores have been pushed, the segmenter advances the boundary all the way to `total_samples` and closes any still-open voice run.

In `src/segment/segmenter.rs`, replace `next_finalization_boundary` with the following final form:

```rust
    /// Largest absolute sample finalized after current windows are processed.
    /// Pre-finish: no future regular window contributes to samples
    /// `< next_window_idx * step`. Post-finish (and after the tail window's
    /// scores are pushed), the boundary is `total_samples`.
    fn next_finalization_boundary(&self) -> u64 {
        if self.finished && self.pending.is_empty() {
            return self.total_samples;
        }
        let step = self.opts.step_samples() as u64;
        self.next_window_idx as u64 * step
    }
```

And replace the entire `process_voice_finalization` method with the following final form (this lifts the early-return so the end-of-stream block is always reachable, and closes any open voice span when the stream is done):

```rust
    fn process_voice_finalization(&mut self) {
        let up_to = self.next_finalization_boundary();
        let probs = if up_to > self.stitcher.base_sample() {
            self.stitcher.take_finalized(up_to)
        } else {
            alloc::vec::Vec::new()
        };
        let base_after = self.stitcher.base_sample();
        let base_before = base_after - probs.len() as u64;
        for (i, p) in probs.iter().enumerate() {
            let abs = base_before + i as u64;
            let was_active = self.voice_hyst.is_active();
            let now_active = self.voice_hyst.push(*p);
            match (was_active, now_active) {
                (false, true) => self.voice_run_start = Some(abs),
                (true, false) => {
                    if let Some(start) = self.voice_run_start.take() {
                        self.emit_voice_span(start, abs);
                    }
                }
                _ => {}
            }
        }
        if self.finished && self.pending.is_empty() {
            if let Some(start) = self.voice_run_start.take() {
                self.emit_voice_span(start, self.total_samples);
                self.voice_hyst.reset();
            }
        }
    }
```

- [ ] **Step 12.3: Add tests for finish + clear**

Append to the `tests` module:

```rust
    #[test]
    fn finish_short_clip_schedules_tail_window() {
        let mut s = Segmenter::new(opts());
        s.push_samples(&vec![0.0; 50_000]); // less than one window
        assert!(s.poll().is_none());
        s.finish();
        match s.poll() {
            Some(Action::NeedsInference { samples, .. }) => {
                assert_eq!(samples.len(), WINDOW_SAMPLES as usize);
                // The 50_000 buffered samples should be at the start; the
                // rest is zero-padded.
                for i in 0..50_000 { assert_eq!(samples[i], 0.0); }
                for i in 50_000..160_000 { assert_eq!(samples[i], 0.0); }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn clear_resets_state() {
        let mut s = Segmenter::new(opts());
        s.push_samples(&vec![0.0; 160_000]);
        let _ = s.poll();
        s.clear();
        assert!(s.poll().is_none());
        // Push again — first window starts at sample 0 fresh.
        s.push_samples(&vec![0.0; 160_000]);
        match s.poll().unwrap() {
            Action::NeedsInference { id, .. } => {
                assert_eq!(id.range().start_pts(), 0);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn end_of_stream_closes_open_voice_span() {
        let mut s = Segmenter::new(opts());
        s.push_samples(&vec![0.0; 160_000]);
        let id = match s.poll().unwrap() {
            Action::NeedsInference { id, .. } => id,
            _ => unreachable!(),
        };
        // All frames "voiced" via class 1 (speaker A) — voice prob ≈ 1.0.
        let scores = synth_logits(0..FRAMES_PER_WINDOW);
        s.push_inference(id, &scores).unwrap();
        s.finish();

        let mut found_voice = false;
        while let Some(a) = s.poll() {
            if matches!(a, Action::VoiceSpan(_)) { found_voice = true; }
        }
        assert!(found_voice, "expected a closing voice span on finish");
    }
```

- [ ] **Step 12.4: Run the segmenter tests**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std segmenter::tests
```

Expected: 9 tests pass.

- [ ] **Step 12.5: Run the whole sans-I/O test suite**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std
```

Expected: all module tests pass (options, types, powerset, hysteresis, stitch, window, segmenter).

- [ ] **Step 12.6: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/segmenter.rs && \
  git commit -m "feat(segment): finish + clear + tail voice-span finalization"
```

---

## Task 13: Layer 2 — SegmentModel ort wrapper

**Files:**
- Modify: `src/segment/model.rs`, `src/segment/mod.rs`

- [ ] **Step 13.1: Implement SegmentModel**

Replace `/Users/user/Develop/findit-studio/dia/src/segment/model.rs` with:

```rust
//! ONNX Runtime wrapper for pyannote/segmentation-3.0.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

use crate::segment::error::Error;
use crate::segment::options::{FRAMES_PER_WINDOW, POWERSET_CLASSES, WINDOW_SAMPLES};

/// Thin ort wrapper for one segmentation model session.
///
/// Owns one `ort::Session` plus reusable input/output scratch. `Send` but
/// not `Sync` — use one per worker thread. Mirrors `silero::Session`.
pub struct SegmentModel {
    session: Session,
    /// Reusable input buffer holding `[1, WINDOW_SAMPLES]` float32 audio.
    input_scratch: Vec<f32>,
}

impl SegmentModel {
    /// Load the model from disk.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let session = Session::builder()
            .map_err(Error::Ort)?
            .commit_from_file(&path)
            .map_err(|source| Error::LoadModel { path: path.clone(), source })?;
        Ok(Self { session, input_scratch: Vec::with_capacity(WINDOW_SAMPLES as usize) })
    }

    /// Load the model from an in-memory ONNX byte buffer.
    pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
        let session = Session::builder()
            .map_err(Error::Ort)?
            .commit_from_memory(bytes)
            .map_err(Error::Ort)?;
        Ok(Self { session, input_scratch: Vec::with_capacity(WINDOW_SAMPLES as usize) })
    }

    /// Run inference on one 160 000-sample window. Returns flattened
    /// `[FRAMES_PER_WINDOW * POWERSET_CLASSES]` logits. `samples.len()` must
    /// equal `WINDOW_SAMPLES`.
    pub fn infer(&mut self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        debug_assert_eq!(samples.len(), WINDOW_SAMPLES as usize);

        // Build input tensor [1, 1, WINDOW_SAMPLES] (the published model
        // shape; pyannote uses a unit channel dim).
        self.input_scratch.clear();
        self.input_scratch.extend_from_slice(samples);
        let shape = [1i64, 1i64, WINDOW_SAMPLES as i64];
        let input = Tensor::from_array((shape, self.input_scratch.as_slice()))
            .map_err(Error::Ort)?;

        // Find the first input name (model exporters tend to name it
        // "audio" or "input"; we don't hard-code).
        let input_name = self
            .session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| Error::Ort(ort::Error::new("model has no inputs")))?;

        let outputs = self
            .session
            .run(ort::inputs![input_name => input])
            .map_err(Error::Ort)?;

        // Output 0 is shape [1, FRAMES_PER_WINDOW, POWERSET_CLASSES] for
        // segmentation-3.0.
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::Ort)?;
        let expected = FRAMES_PER_WINDOW * POWERSET_CLASSES;
        if data.len() != expected {
            return Err(Error::InferenceShapeMismatch { expected, got: data.len() });
        }
        Ok(data.to_vec())
    }
}
```

> **Note:** the `ort` 2.0.0-rc.12 API is still moving; if `Tensor::from_array`,
> `ort::inputs!`, or the output extraction signature has shifted between rc
> bumps, mirror what `silero/src/session.rs` does in this repo (it's already
> calling the same APIs successfully).

- [ ] **Step 13.2: Uncomment SegmentModel re-export in mod.rs**

In `src/segment/mod.rs`, change:

```rust
// #[cfg(feature = "ort")]
// #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
// pub use model::SegmentModel;
```

to:

```rust
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use model::SegmentModel;
```

- [ ] **Step 13.3: Verify the default-feature build**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo build
```

Expected: builds successfully with `ort` linked.

- [ ] **Step 13.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/model.rs src/segment/mod.rs && \
  git commit -m "feat(segment): SegmentModel ort wrapper (Layer 2)"
```

---

## Task 14: Layer 2 — process_samples, finish_stream

**Files:**
- Modify: `src/segment/model.rs`

- [ ] **Step 14.1: Add the streaming convenience methods on Segmenter**

Append to `/Users/user/Develop/findit-studio/dia/src/segment/model.rs`:

```rust
use crate::segment::segmenter::Segmenter;
use crate::segment::types::{Action, Event};

impl Segmenter {
    /// Push samples and drive the state machine to a quiescent state by
    /// fulfilling each `NeedsInference` via `model.infer`. `emit` is called
    /// for every emitted [`Event`].
    ///
    /// This is the streaming entry point that mirrors
    /// `silero::SpeechSegmenter::process_samples`.
    #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
    pub fn process_samples<F>(
        &mut self,
        model: &mut SegmentModel,
        samples: &[f32],
        mut emit: F,
    ) -> Result<(), Error>
    where
        F: FnMut(Event),
    {
        self.push_samples(samples);
        self.drain(model, &mut emit)
    }

    /// Equivalent to `finish` followed by draining all remaining actions
    /// (running inference for any unprocessed window).
    #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
    pub fn finish_stream<F>(&mut self, model: &mut SegmentModel, mut emit: F) -> Result<(), Error>
    where
        F: FnMut(Event),
    {
        self.finish();
        self.drain(model, &mut emit)
    }

    fn drain<F>(&mut self, model: &mut SegmentModel, emit: &mut F) -> Result<(), Error>
    where
        F: FnMut(Event),
    {
        while let Some(action) = self.poll() {
            match action {
                Action::NeedsInference { id, samples } => {
                    let scores = model.infer(&samples)?;
                    self.push_inference(id, &scores)?;
                }
                Action::Activity(a) => emit(Event::Activity(a)),
                Action::VoiceSpan(r) => emit(Event::VoiceSpan(r)),
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 14.2: Verify the build**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo build
```

Expected: clean build, no warnings.

- [ ] **Step 14.3: Verify all tests still pass**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std
cd /Users/user/Develop/findit-studio/dia && cargo test
```

Expected: both pass. Layer 2 has no unit tests yet — they come via the integration test in Task 17.

- [ ] **Step 14.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add src/segment/model.rs && \
  git commit -m "feat(segment): process_samples + finish_stream (Layer 2)"
```

---

## Task 15: Sans-I/O example with synthetic inferencer

**Files:**
- Create: `examples/stream_layer1.rs`

- [ ] **Step 15.1: Write the example**

Create `/Users/user/Develop/findit-studio/dia/examples/stream_layer1.rs`:

```rust
//! Demonstrates the Sans-I/O Segmenter API with a synthetic inferencer that
//! returns logits for "speaker A continuously voiced." Run with:
//!
//!     cargo run --no-default-features --features std --example stream_layer1
//!
//! No model file required.

use diarization::segment::{
    Action, FRAMES_PER_WINDOW, POWERSET_CLASSES, SegmentOptions, Segmenter, WINDOW_SAMPLES,
};

fn synth_scores_voiced() -> Vec<f32> {
    let mut out = vec![-10.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    for f in 0..FRAMES_PER_WINDOW {
        out[f * POWERSET_CLASSES + 1] = 10.0; // class 1 = speaker A only
    }
    out
}

fn main() -> Result<(), diarization::segment::Error> {
    let mut seg = Segmenter::new(SegmentOptions::default());

    // Simulate a streaming source: 25 chunks of 10 000 samples (250 000 total).
    for chunk in (0..25).map(|_| vec![0.0f32; 10_000]) {
        seg.push_samples(&chunk);
        // Drain anything ready.
        while let Some(action) = seg.poll() {
            match action {
                Action::NeedsInference { id, samples } => {
                    println!("inference request: id={:?}, len={}", id.range(), samples.len());
                    let scores = synth_scores_voiced();
                    seg.push_inference(id, &scores)?;
                }
                Action::Activity(a) => {
                    println!(
                        "activity: window={:?} slot={} range={:?}",
                        a.window_id().range(),
                        a.speaker_slot(),
                        a.range()
                    );
                }
                Action::VoiceSpan(r) => println!("voice span: {r:?}"),
            }
        }
    }

    seg.finish();
    while let Some(action) = seg.poll() {
        match action {
            Action::NeedsInference { id, samples } => {
                println!("tail inference: id={:?}, len={}", id.range(), samples.len());
                let _ = WINDOW_SAMPLES; // sanity reference
                let scores = synth_scores_voiced();
                seg.push_inference(id, &scores)?;
            }
            Action::Activity(a) => println!("tail activity: slot={} range={:?}", a.speaker_slot(), a.range()),
            Action::VoiceSpan(r) => println!("tail voice span: {r:?}"),
        }
    }
    Ok(())
}
```

- [ ] **Step 15.2: Run the example**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  cargo run --no-default-features --features std --example stream_layer1
```

Expected: prints inference requests, activities for slot 0, and at least one voice span.

- [ ] **Step 15.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add examples/stream_layer1.rs && \
  git commit -m "docs(segment): Sans-I/O streaming example with synthetic inferencer"
```

---

## Task 16: WAV streaming example (ort feature)

**Files:**
- Create: `examples/stream_from_wav.rs`

- [ ] **Step 16.1: Write the example**

Create `/Users/user/Develop/findit-studio/dia/examples/stream_from_wav.rs`:

```rust
//! Streams a 16 kHz mono WAV file through the segmenter using the bundled
//! ort driver. Run with:
//!
//!     cargo run --example stream_from_wav -- path/to/audio.wav
//!
//! Requires `models/segmentation-3.0.onnx` (run `scripts/download-model.sh`).

#[cfg(feature = "ort")]
fn main() -> anyhow::Result<()> {
    use diarization::segment::{Event, SegmentModel, SegmentOptions, Segmenter};

    let path = std::env::args().nth(1).expect("usage: stream_from_wav <file.wav>");
    let pcm = read_wav_mono_16k(&path)?;
    let mut model = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
    let mut seg = Segmenter::new(SegmentOptions::default());

    // Feed in 100 ms chunks (1_600 samples) to simulate streaming.
    for chunk in pcm.chunks(1_600) {
        seg.process_samples(&mut model, chunk, |event| match event {
            Event::Activity(a) => println!(
                "activity: window={:?} slot={} range={:?}",
                a.window_id().range(),
                a.speaker_slot(),
                a.range()
            ),
            Event::VoiceSpan(r) => println!("voice: {r:?} ({:?})", r.duration()),
        })?;
    }
    seg.finish_stream(&mut model, |event| match event {
        Event::Activity(a) => println!(
            "tail activity: window={:?} slot={} range={:?}",
            a.window_id().range(),
            a.speaker_slot(),
            a.range()
        ),
        Event::VoiceSpan(r) => println!("tail voice: {r:?}"),
    })?;
    Ok(())
}

#[cfg(feature = "ort")]
fn read_wav_mono_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    anyhow::ensure!(spec.sample_rate == 16_000, "expected 16 kHz, got {}", spec.sample_rate);
    anyhow::ensure!(spec.channels == 1, "expected mono, got {}", spec.channels);
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect(),
    };
    Ok(samples?)
}

#[cfg(not(feature = "ort"))]
fn main() {
    eprintln!("This example requires the `ort` feature: cargo run --example stream_from_wav");
}
```

- [ ] **Step 16.2: Add anyhow to dev-dependencies**

Edit `/Users/user/Develop/findit-studio/dia/Cargo.toml` to extend `[dev-dependencies]`:

```toml
[dev-dependencies]
anyhow = "1"
criterion = "0.8"
hound = "3"
tempfile = "3"
```

- [ ] **Step 16.3: Verify the example compiles**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo build --example stream_from_wav
```

Expected: builds. (We don't run it here because it needs a model file.)

- [ ] **Step 16.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add examples/stream_from_wav.rs Cargo.toml && \
  git commit -m "docs(segment): WAV streaming example using ort driver"
```

---

## Task 17: Integration test (gated)

**Files:**
- Create: `tests/integration_segment.rs`

- [ ] **Step 17.1: Write the integration test**

Create `/Users/user/Develop/findit-studio/dia/tests/integration_segment.rs`:

```rust
//! Smoke test against a real pyannote/segmentation-3.0 ONNX model. Skipped
//! by default (`#[ignore]`); run with:
//!
//!     cargo test --test integration_segment -- --ignored
//!
//! Requires `models/segmentation-3.0.onnx` next to this repo's root.

#![cfg(feature = "ort")]

use diarization::segment::{Event, SegmentModel, SegmentOptions, Segmenter};

#[test]
#[ignore = "requires model file at models/segmentation-3.0.onnx"]
fn smoke_test_runs_inference_on_synthetic_audio() {
    let mut model = SegmentModel::from_file("models/segmentation-3.0.onnx")
        .expect("model file present");
    let mut seg = Segmenter::new(SegmentOptions::default());

    // 12 seconds of low-amplitude noise — exercise tail anchoring.
    let mut pcm = vec![0.0f32; 16_000 * 12];
    for (i, x) in pcm.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0001).sin() * 0.01;
    }

    let mut events: usize = 0;
    seg.process_samples(&mut model, &pcm, |_| events += 1).expect("ok");
    seg.finish_stream(&mut model, |_| events += 1).expect("ok");

    // We don't assert specific events on synthetic noise (the model may
    // emit none); the point is that the pipeline runs end-to-end without
    // panicking and the inference contract holds.
    let _ = events;
}
```

- [ ] **Step 17.2: Verify it compiles**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --test integration_segment --no-run
```

Expected: builds.

- [ ] **Step 17.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add tests/integration_segment.rs && \
  git commit -m "test(segment): gated integration smoke test"
```

---

## Task 18: Layer 1 throughput bench

**Files:**
- Create: `benches/segment.rs`

- [ ] **Step 18.1: Write the bench**

Create `/Users/user/Develop/findit-studio/dia/benches/segment.rs`:

```rust
//! Layer-1 throughput bench. Runs `Segmenter` with synthetic scores so we
//! measure state-machine cost only (no ort).

use criterion::{Criterion, criterion_group, criterion_main, BatchSize};
use diarization::segment::{Action, FRAMES_PER_WINDOW, POWERSET_CLASSES, SegmentOptions, Segmenter};

fn synth_scores() -> Vec<f32> {
    let mut out = vec![-10.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    for f in 0..FRAMES_PER_WINDOW {
        out[f * POWERSET_CLASSES + 1] = 10.0;
    }
    out
}

fn bench_one_minute(c: &mut Criterion) {
    let scores = synth_scores();
    let pcm = vec![0.0f32; 16_000 * 60]; // one minute at 16 kHz
    c.bench_function("segmenter_one_minute_layer1", |b| {
        b.iter_batched(
            || Segmenter::new(SegmentOptions::default()),
            |mut seg| {
                for chunk in pcm.chunks(1_600) {
                    seg.push_samples(chunk);
                    while let Some(a) = seg.poll() {
                        match a {
                            Action::NeedsInference { id, .. } => {
                                seg.push_inference(id, &scores).unwrap();
                            }
                            Action::Activity(_) | Action::VoiceSpan(_) => {}
                        }
                    }
                }
                seg.finish();
                while let Some(a) = seg.poll() {
                    if let Action::NeedsInference { id, .. } = a {
                        seg.push_inference(id, &scores).unwrap();
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_one_minute);
criterion_main!(benches);
```

- [ ] **Step 18.2: Verify the bench compiles**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo bench --no-run
```

Expected: builds.

- [ ] **Step 18.3: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add benches/segment.rs && \
  git commit -m "bench(segment): Layer-1 throughput measurement"
```

---

## Task 19: Model download script

**Files:**
- Create: `scripts/download-model.sh`

- [ ] **Step 19.1: Write the download script**

Create `/Users/user/Develop/findit-studio/dia/scripts/download-model.sh`:

```bash
#!/usr/bin/env bash
# Download the pyannote/segmentation-3.0 ONNX model into `models/`.
# Mirror sources are listed in priority order; the script tries each until
# one succeeds. Re-run is idempotent.

set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
DEST_FILE="$DEST_DIR/segmentation-3.0.onnx"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_FILE" ]]; then
  echo "model already present: $DEST_FILE"
  exit 0
fi

URLS=(
  "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.onnx"
  "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx"
)

for url in "${URLS[@]}"; do
  echo "trying $url"
  if curl -fL --retry 3 --retry-delay 2 -o "$DEST_FILE.tmp" "$url"; then
    mv "$DEST_FILE.tmp" "$DEST_FILE"
    echo "downloaded to $DEST_FILE"
    exit 0
  fi
done

echo "failed to download model from any mirror" >&2
rm -f "$DEST_FILE.tmp"
exit 1
```

- [ ] **Step 19.2: Make it executable**

```bash
chmod +x /Users/user/Develop/findit-studio/dia/scripts/download-model.sh
```

- [ ] **Step 19.3: Add models/ to .gitignore**

Append to `/Users/user/Develop/findit-studio/dia/.gitignore`:

```
/models/*.onnx
```

- [ ] **Step 19.4: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  git add scripts/download-model.sh .gitignore && \
  git commit -m "build(segment): model download script + ignore models/*.onnx"
```

---

## Task 20: Final verification

- [ ] **Step 20.1: Sans-I/O test pass (no ort)**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test --no-default-features --features std
```

Expected: all module tests pass (~30 tests across options, types, powerset, hysteresis, stitch, window, segmenter).

- [ ] **Step 20.2: Default-feature test pass**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo test
```

Expected: same set of tests + the integration test compiles (and is `#[ignore]`'d at runtime).

- [ ] **Step 20.3: Layer-1 example runs**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  cargo run --no-default-features --features std --example stream_layer1
```

Expected: prints inference requests, activities, voice span, and exits cleanly.

- [ ] **Step 20.4: Clippy clean**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo clippy --all-targets --all-features -- -D warnings
```

Expected: no clippy warnings.

- [ ] **Step 20.5: Format check**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo fmt --check
```

Expected: no formatting drift.

- [ ] **Step 20.6: Docs build**

```bash
cd /Users/user/Develop/findit-studio/dia && cargo doc --all-features --no-deps
```

Expected: docs build with no warnings.

- [ ] **Step 20.7: If integration test machine has the model, run it**

```bash
cd /Users/user/Develop/findit-studio/dia && \
  ./scripts/download-model.sh && \
  cargo test --test integration_segment -- --ignored
```

Expected (if model downloaded): the smoke test passes. If the download fails, this step is acceptable to skip — the unit tests already cover the state machine.

---

## Notes for the implementer

- **Read the spec first.** `docs/superpowers/specs/2026-04-25-dia-segment-design.md` is the source of truth; this plan operationalizes it.
- **Reference the siblings.** `silero/src/session.rs` and `soundevents/soundevents/src/lib.rs` show the exact `ort` 2.0.0-rc.12 idioms; mimic them on any API friction.
- **Watch the `ort` rc API.** If `Tensor::from_array`, `ort::inputs!`, or output extraction in Task 13 doesn't compile, look at silero's working code in this repo — it's a guaranteed-good reference.
- **One commit per task minimum.** The plan ends each task with a commit step; don't batch.
- **Don't add features past the spec.** `include_frame_probabilities`, bundled models, F1 parity testing are explicitly deferred; if you find yourself adding them, stop.
- **TDD discipline.** Each task lands tests first (or alongside the small structs). Don't run integration tests until the unit tests pass.
