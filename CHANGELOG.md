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
  `Send`; explicitly `!Sync` (matches `diarization::segment::SegmentModel`).
- **`embed`** / **`embed_with_meta`**: high-level API. Sliding-window
  mean for clips > 2 s.
- **`embed_weighted`** / **`embed_weighted_with_meta`**: per-sample
  voice-probability soft weighting.
- **`embed_masked`** / **`embed_masked_with_meta`**: rev-8 binary
  keep-mask (gather-and-embed). Used by `Diarizer::exclude_overlap`.
- **Generic `EmbeddingMeta<A=(), T=()>`**: caller-supplied metadata
  flows through `EmbeddingResult`. Defaults to `()` so the unit-typed
  metadata path is zero-cost.
- **`cosine_similarity`** free function alongside `Embedding::similarity`.

FEATURES — `diarization::cluster`

- **Online streaming `Clusterer`** with `submit(&Embedding)` returning
  `ClusterAssignment { speaker_id, is_new_speaker, similarity }`.
  `RollingMean` and `Ema(α)` update strategies on an unnormalized
  accumulator (handles antipodal cancellation gracefully via lazy
  `cached_centroid` refresh).
- **`OverflowStrategy::Reject`** (default — caller decides) /
  **`AssignClosest`** (no centroid update on forced assignment).
- **Offline `cluster_offline`** with two methods:
  - **Spectral** (default): cosine affinity + degree-matrix
    precondition + normalized Laplacian + nalgebra eigendecomposition
    + eigengap K-detection (capped at `MAX_AUTO_SPEAKERS = 15`) +
    K-means++ + Lloyd. PRNG pinned to `rand_chacha::ChaCha8Rng` with
    explicit byte-fixture regression test.
  - **Agglomerative**: Single / Complete / Average linkage with
    cosine distance ReLU-clamped to `[0, 1]`.
- **Deterministic K-means++** seeding (Arthur & Vassilvitskii 2007).
  Default seed `0`; same input + seed → same labels across runs.
- **N ≤ 2 fast paths** before any matrix work; isolated-node
  precondition catches dissimilar inputs without an undefined Laplacian.

FEATURES — `diarization::Diarizer` (rev-6 pyannote-style reconstruction)

- **`process_samples`** / **`finish_stream`**: streaming entry points
  borrowing `&mut SegmentModel` + `&mut EmbedModel` per call.
- **VAD-friendly variable-length input**: empty / sub-window /
  multi-clip / whole-stream pushes all handled without special-casing.
- **`exclude_overlap`** mask (spec §5.8): per-window binarized + clean
  masks → sample-rate `keep_mask`; clean used when its gathered length
  ≥ `MIN_CLIP_SAMPLES`, else falls back to speaker-only. On doubly-
  failed gather, skip-and-continue (matches pyannote
  `speaker_verification.py:611-612`).
- **Per-frame per-cluster overlap-add stitching** (spec §5.9):
  collapse-by-max within cluster + overlap-add SUM across windows.
- **Per-frame instantaneous-speaker-count tracking** (spec §5.10):
  per-frame overlap-add MEAN with warm-up trim
  (`speaker_count(warm_up=(0.1, 0.1))`), rounded.
- **Count-bounded argmax + per-cluster RLE** (spec §5.11):
  deterministic tie-break (smaller cluster_id wins).
- **Output**: `DiarizedSpan { range, speaker_id, is_new_speaker,
  average_activation, activity_count, clean_mask_fraction }` per
  closed speaker turn.
- **`collected_embeddings()`**: per-(window, slot) granularity context
  retained across the session.
- **Introspection**: `pending_inferences`, `buffered_samples`,
  `buffered_frames`, `total_samples_pushed`, `num_speakers`, `speakers`.
- **Auto-derived `Send + Sync`**.

FEATURES — `diarization::segment` v0.X bump

- **`Action::SpeakerScores { id, window_start, raw_probs }`** variant
  emitted from `push_inference` alongside `Action::Activity`.
- **`Action` is now `#[non_exhaustive]`** so future additions are
  non-breaking.
- **`pub(crate) Segmenter::peek_next_window_start()`** for the
  Diarizer's reconstruction finalization-boundary computation.

CORRECTNESS GUARANTEES

- **Bit-deterministic offline clustering** for a given input + seed,
  enforced by `tests/chacha_keystream_fixture.rs` regression test.
- **Frame-rate math verified**: `diarization::segment::stitch::frame_to_sample`
  yields ≈ 271.65 samples/frame (≈ 58.9 fps); the Diarizer carries a
  `frame_to_sample_u64` helper bit-exactly equivalent to segment's
  `u32` version.
- **Documented divergences from pyannote** (spec §1): sliding-window
  mean for long-clip embed, sample-rate vs frame-rate gather in
  `exclude_overlap`, online vs batch clustering, default Spectral vs
  pyannote VBx, deterministic argmax tie-break.

TESTING

- ~175 unit tests across `diarization::embed`, `diarization::cluster`, `diarization::diarizer`.
- 149 lib tests pass on `--no-default-features --features std` (no ort).
- Gated integration tests for end-to-end Diarizer pump on a 30-s clip
  (8 #[ignore]'d tests in `tests/integration_diarizer.rs`).
- Pyannote parity harness (`tests/parity/run.sh`) — manual; targets
  DER ≤ 10% absolute (rev-8 T3-I relaxed from 5%).

BUILD

- New deps: `nalgebra = "0.34"`, `rand = "0.10"` (default-features =
  false), `rand_chacha = "0.10"` (default-features = false),
  `kaldi-native-fbank = "0.1"`.

KNOWN LIMITATIONS / DEFERRED TO v0.1.1+

- No bundled WeSpeaker model (~25 MB); use
  `scripts/download-embed-model.sh`.
- VBx clustering (pyannote's offline default) not shipped; spec §15 #44.
- HMM-GMM clustering not shipped; spec §15.
- `min_cluster_size` cluster pruning not shipped; spec §15.
- Configurable `warm_up` for speaker-count not shipped; hardcoded to
  pyannote default `(0.1, 0.1)`. Spec §15 #47.
- Configurable `min_duration_on/off` for span-merging not shipped;
  spec §15 #48.
- Mask-aware embedding ONNX export deferred; current path uses
  sample-rate gather + sliding-window-mean (one extra divergence
  from pyannote on long masked clips). Spec §15 #49.

# 0.1.0 (2026-04-26)

Initial release. Ships the `diarization::segment` module — Sans-I/O speaker
segmentation backed by `pyannote/segmentation-3.0` ONNX.

FEATURES

- **Sans-I/O state machine** (`diarization::segment::Segmenter`) with no `ort`
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
  types are re-exported from `diarization::segment`.
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

- `diarization::embed` module (speaker embedding via WeSpeaker ResNet34).
- `infer_batch` for cross-stream batching, `IoBinding`-based
  reusable-output-buffer fast path, `Arc<[f32]>` in `Action::NeedsInference`.
- `serde` derives on output types.
- `step_samples` typed as `Duration`.
- Soft-cap `try_push_samples` for backpressure enforcement.
- Bundled model behind a Cargo feature.
- F1 numerical-parity tests vs `pyannote.audio`.
