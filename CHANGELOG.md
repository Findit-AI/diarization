# UNRELEASED

# 0.1.0 (2026-04-26)

Initial release. Ships the `dia::segment` module — Sans-I/O speaker
segmentation backed by `pyannote/segmentation-3.0` ONNX.

FEATURES

- **Sans-I/O state machine** (`dia::segment::Segmenter`) with no `ort`
  dependency. Caller pumps audio in via `push_samples`, drains `Action`s
  via `poll`, runs ONNX inference externally, and pushes scores back via
  `push_inference`. The state machine is exercisable in unit tests with
  synthetic scores — no model file required.
- **Layer 2 streaming driver** (`Segmenter::process_samples` and
  `finish_stream`) gated on the default `ort` feature. Mirrors silero's
  `Session::process_stream` callback idiom.
- **`SegmentModel`** wraps `ort::Session` for `pyannote/segmentation-3.0`
  with `from_file`, `from_memory`, and `*_with_options` constructors.
- **`SegmentModelOptions`** builder for `GraphOptimizationLevel`,
  `ExecutionProviderDispatch`, intra/inter thread counts. Both `ort`
  types are re-exported from `dia::segment`.
- **`mediatime`-based time types** (`TimeRange`, `Timestamp`, `Duration`)
  for every sample range and duration crossing the public API.
- **Sliding-window scheduling** with configurable step (default 2.5 s)
  and tail-anchored window for end-of-stream coverage.
- **Powerset decoding** (7-class → 3-speaker additive marginals + voice
  probability), **per-frame voice-timeline stitching** (overlap-add mean,
  ~1.7 MB/hour storage), **streaming hysteresis** with onset/offset
  thresholds, **window-local `SpeakerActivity`** records, and
  **`voice_merge_gap`** post-processing.

CORRECTNESS GUARANTEES

- **Generation-counter `WindowId`** (process-wide `AtomicU64`): stale
  inference results from before a `clear()` and cross-`Segmenter` ID
  collisions both reject as `Error::UnknownWindow`.
- **Pending-aware finalization boundary**: out-of-order `push_inference`
  cannot prematurely finalize frames whose other contributing windows
  haven't yet reported.
- **Tail-window activity clamping** to `total_samples`.
- **Frame-to-sample conversion** uses integer-rounded division
  (`frame_to_sample`) bit-for-bit equivalent to Python's
  `int(round(...))`. **Sample-to-frame conversion** uses floor
  (`frame_index_of`) for boundary safety.

OBSERVABILITY

- `Segmenter::pending_inferences()` and `Segmenter::buffered_samples()`
  introspection for backpressure detection.
- Compile-time `Send + Sync` assertion on `Segmenter`; compile-time `Send`
  assertion on `SegmentModel` (which is `!Sync` because `ort::Session` is).

EXAMPLES, TESTS, BENCHES

- `examples/stream_layer1.rs`: Sans-I/O usage with synthetic inferencer
  (no model file, no `ort` feature).
- `examples/stream_from_wav.rs`: full Layer-2 pipeline streaming a 16 kHz
  mono WAV file in 100 ms chunks.
- `tests/integration_segment.rs`: gated `#[ignore]` smoke test against a
  real downloaded model.
- `benches/segment.rs`: Layer-1 throughput on one minute of audio.
- 54 unit tests covering options, powerset, hysteresis, RLE, sliding-window
  planning, per-frame stitching, segmenter end-to-end, out-of-order
  `push_inference`, cross-`Segmenter` ID collision, stale-id rejection,
  empty-stream handling, tail-window activity clamping.

BUILD

- Edition 2024, Rust 1.95.
- Default features `["std", "ort"]`. `--no-default-features --features
  std` builds without `ort` and exposes only Layer 1.
- Lints aligned with sibling crates (silero, soundevents, scenesdetect,
  mediatime).

KNOWN LIMITATIONS

- **No load-time ONNX shape verification.** The `ort` 2.0.0-rc.12 metadata
  API doesn't expose dimensions in a way matching the spec's assumption;
  shape mismatches surface on first inference as
  `Error::InferenceShapeMismatch`. The `Error::IncompatibleModel` variant
  is reserved for the eventual load-time check. Matches silero's pragmatic
  stance.
- **Sample-rate is the caller's responsibility.** `push_samples` accepts
  `&[f32]` without validating that the input is 16 kHz mono. Feeding the
  wrong rate produces silently corrupted output.
- **No bundled model.** Run `scripts/download-model.sh` to fetch
  `pyannote/segmentation-3.0` from Hugging Face.

DEFERRED FOR v0.2

- `dia::embed` module (speaker embedding via WeSpeaker ResNet34).
- `infer_batch` for cross-stream batching, `IoBinding`-based
  reusable-output-buffer fast path, `Arc<[f32]>` in `Action::NeedsInference`.
- `serde` derives on output types.
- `step_samples` typed as `Duration`.
- Soft-cap `try_push_samples` for backpressure enforcement.
- Bundled model behind a Cargo feature.
- F1 numerical-parity tests vs `pyannote.audio`.
