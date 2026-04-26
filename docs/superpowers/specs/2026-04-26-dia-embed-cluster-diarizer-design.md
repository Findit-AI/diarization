# dia — embed + cluster + Diarizer design (v0.1.0 phase 2)

**Revision 9** (2026-04-27, post-sixth-adversarial-review pseudocode-consistency patch).
**Status:** ready for implementation (pending §15 #43 pre-impl spike + §15 #52 ChaCha keystream stability).
**Scope:** Three new modules to ship in dia v0.1.0 alongside `dia::segment`,
**plus a v0.X bump of `dia::segment` to expose per-window per-speaker per-frame
raw probabilities** (required for reconstruction):
- `dia::segment` v0.X — adds `Action::SpeakerScores` variant + marks `Action` as `#[non_exhaustive]`
- `dia::embed` — speaker fingerprint generation with **`embed_masked` (rev-8) and `embed_weighted` primitives**
- `dia::cluster` — cross-window speaker linking (online streaming + offline batch)
- `dia::Diarizer` — top-level orchestrator with **pyannote-style per-frame
  reconstruction** (overlap-add cluster-activation stitching, count-bounded
  argmax, per-cluster RLE-to-spans). Output is one `DiarizedSpan` per closed
  speaker turn — not per-(window, speaker).

> **Revision 8 fixes load-bearing correctness errors** flagged by the
> fifth adversarial review:
> - **T1-A (CRITICAL):** §5.8's `exclude_overlap` mechanism was wrong.
>   Rev-7 routed it through `embed_weighted`, which §5.3 says collapses
>   to a no-op for clips ≤ 2 s (the most common VAD-filtered activity
>   length). Pyannote actually does **gather-and-embed**: drop the
>   masked-out frames before feature extraction, run ONNX on the
>   shorter cleaned signal (verified at
>   `pyannote/audio/pipelines/speaker_verification.py:568-619`,
>   `ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__`). Rev-8 adds a
>   new `EmbedModel::embed_masked(samples, keep_mask)` primitive that
>   gathers retained samples, zero-pads to `EMBED_WINDOW_SAMPLES`, and
>   embeds. §5.8 routes through it. `embed_weighted` is preserved for
>   callers who want soft per-sample weights (e.g., direct VAD
>   probability streams).
> - **T1-C (HIGH):** the per-frame rate was wrong throughout §5.8 /
>   §4.4 / §5.10. `pyannote/segmentation-3.0` produces 589 frames per
>   `WINDOW_SAMPLES = 160 000`, i.e. ≈ 271.65 samples/frame ≈ 16.97 ms/frame
>   ≈ 58.9 fps — verified by reading `dia::segment::stitch::frame_to_sample`
>   (the live constant `WINDOW_SAMPLES * frame_idx / FRAMES_PER_WINDOW`).
>   Rev-7 said "100 fps / 160 samples / 10 ms" repeatedly, which is
>   wrong. Rev-8 corrects: `MIN_CLEAN_FRAMES = ceil(MIN_CLIP_SAMPLES /
>   SAMPLES_PER_FRAME) = ceil(400/271.65) = 2`; the per-frame
>   accumulator steady-state is ≈ 589, not 1000.
> - **T2-A (HIGH):** `average_activation` normalization in §5.11 used
>   the warm-up-trimmed `chunk_count` from §5.10's speaker-count
>   bookkeeping, but pyannote's `Inference.aggregate(skip_average=True)`
>   maintains its own un-trimmed overlapping-chunk count
>   (`pyannote/audio/core/inference.py:602-604`). Rev-8 separates the
>   two counters in `FrameCount`: `activation_chunk_count` (no warm-up,
>   for activation normalization) and `count_chunk_count` (warm-up
>   trimmed, for speaker-count rounding only).
> - **T2-B/C/D (HIGH):** pseudocode bugs in the new §5.9-§5.11
>   reconstruction state machine — `emit_finalized_frames` flush could
>   panic at end-of-stream (`pop_front()` on an empty `VecDeque`);
>   `window_starts` was referenced but never declared in
>   `ReconstructState`; `activity_clean_flags` insertion was missing
>   from §5.8's pseudocode. All three corrected in rev-8.
> - **T1-B (rejected):** the reviewer claimed pyannote's default
>   clusterer is `HiddenMarkovModelClustering`, citing
>   `speaker_diarization.py:115`. Verified against the actual source:
>   line 115 is part of an unrelated docstring; the actual default at
>   `speaker_diarization.py:210` is `clustering: str = "VBxClustering"`.
>   §15 #44 was correct. Pushback documented in §14.
>
> Plus a fan-out of polish: §3.1 reference fix, `min_duration_off`
> wording fix in §4.4, §12 decision log refresh, §14 review-5..7
> rejected sections, §11.13 (NEW) on `slot_to_cluster` eviction, NaN
> handling in §5.11 argmax (`total_cmp`), u64 in frame arithmetic, and
> §15 #52 (ChaCha8Rng byte-fixture regression test).
>
> **Revision 7 nails down the VAD-prerequisite contract** that's been
> implicit since the brainstorm. The user's pipeline is `audio decoder
> → resample → VAD → dia → downstream`, so dia's input is **VAD-filtered
> speech**, fed in **variable-length pushes** — not necessarily 2 s, not
> necessarily 10 s, not necessarily aligned to any window boundary. The
> rev-6 algorithm already handles this correctly (segment buffers samples
> lazily, embed accepts 25 ms – any-length clips, reconstruction
> finalizes per-frame as windows complete) but the spec did not document
> the contract explicitly. Rev 7 documents it (§3, §11.12), enriches
> `DiarizedSpan` with quality metrics for downstream storage and analysis
> (`average_activation`, `activity_count`, `clean_mask_fraction`), adds a
> `Diarizer::total_samples_pushed()` accessor for caller-side timeline
> mapping, and adds §9 tests covering the variable-length VAD scenarios.
>
> **Revision 6 expands scope** to include pyannote-audio-style reconstruction
> in `dia::Diarizer`. Per the user, the `pyannote.audio` clustering pipeline
> "has been verified in prod," so we match its algorithm as closely as is
> practical given streaming constraints. Specifically:
> - `dia::segment` gains `Action::SpeakerScores { id, window_start, raw_probs
>   }`, emitted alongside `Action::Activity` from `push_inference`.
> - `dia::embed` adds an `exclude_overlap` mask path: when extracting the
>   embedding for speaker S in a window, mask out samples where another
>   speaker is also active (matches `pyannote/pipelines/speaker_diarization
>   .py:375-425`).
> - `dia::Diarizer` runs per-frame per-cluster overlap-add stitching
>   (matches `Inference.aggregate(..., skip_average=True)` on
>   cluster-collapsed-by-max scores) plus per-frame speaker-count tracking
>   (matches `speaker_count(..., warm_up=(0.1, 0.1))`) plus per-frame
>   count-bounded argmax (matches `to_diarization`) plus per-cluster
>   RLE-to-spans (matches `to_annotation`).
> - `DiarizedSpan` simplifies to `(range, speaker_id, is_new_speaker)`.
>   Drops `similarity` and `speaker_slot` (those are window-local concepts;
>   a stitched span spans multiple windows). Per-activity context still
>   accessible via `Diarizer::collected_embeddings()`.
> - **`Action` becomes `#[non_exhaustive]`** in `dia::segment` v0.X. This
>   is technically a breaking change but `dia::segment` is workspace-only
>   and never published to crates.io, so external impact is zero.
> - New parity test (§9) compares `dia::Diarizer` output against
>   `pyannote.audio` on a held-out 5-minute multi-speaker clip; target
>   diarization-error-rate (DER) < 5% absolute vs the reference.
>
> **Revision 5 closes the remaining implementation-blocking polish
> from review 4.** `nalgebra` bumped from 0.33 → 0.34 (verified
> current). K-means++ algorithm (§5.5 step 8) tightened with exact
> rand-0.10 calls for all four sources of byte ambiguity left in
> rev-4 prose: step-1 sampling method, step-2a min reduction order
> (left-to-right f64), step-2b "uniform from not-yet-chosen" method,
> step-2c f64 distribution + "crosses" semantics + un-normalized
> threshold computation. `SpeakerCentroid` gains its missing
> accessor `impl` block; `ClusterAssignment` converted from `pub`
> fields to private + accessors for visibility consistency. §5.7 /
> §11.5 audio-buffer reach-back claim now backed by line-number
> code-trace verification into shipped `dia::segment::segmenter.rs`
> (rejecting the reviewer's hypothetical "200 000" but accepting the
> meta-point that the assertion needed justification). `Diarizer::
> Error::Internal(AudioBufferUnderflow / AudioBufferOverrun)` variant
> tree replaces rev-1..rev-4's `InvalidClip { len: 0 }` sentinel.
> §15 grows a HIGH-severity pre-impl gating spike for `kaldi-native-
> fbank` (verify against `torchaudio.compliance.kaldi.fbank` BEFORE
> committing to the dep — the crate is brand-new). Plus assorted
> polish: numbering disambiguation, property-test wording alignment,
> per-linkage impact note for the rev-3 distance-clamp change.
>
> **Revision 4 closes the implementation-blocking gaps from review 3.**
> §7 grows `rand = "0.10"` and `rand_chacha = "0.10"` (without these,
> §5.5 step 8's pinned PRNG was un-implementable). §5.5 step 8 now
> spells out the K-means++ algorithm explicitly (Arthur & Vassilvitskii
> 2007 D²-weighted seeding) so two implementations cannot diverge on
> the same input + seed. The PRNG is pinned exactly to
> `rand_chacha::ChaCha8Rng` in §5.5 / §11.9 / §10 (rev 3 said "or
> equivalent reproducible PRNG," which left a backdoor for breaking
> output stability). Plus assorted polish: §13 records the rev-3
> agglomerative distance-clamp change that was shipped but not
> documented; §11.9 attribution corrected; §14 review-2 count fixed;
> §1 wording matches §5.1's "deliberate divergence" framing; §9 grows
> a property test for `RollingMean` accumulator magnitude bounds;
> §15 grows two more action items (float32 precision; cross-platform
> K-means determinism).
>
> **Revision 3 fixes the rev-2 audio-buffer regression** (the trim bound
> mistakenly used the *embed* window size when it should use the
> *segment* window size — would underflow on the first push that
> triggers a non-zero-start segment window). Plus a tighter spectral
> precondition (catches isolated nodes, not just all-zero affinity);
> N ≤ 1 / N == 2 fast paths in spectral; explicit K-means seed source
> (`Option<u64>` with `None` → constant `0`); explicit K-means
> convergence criterion; lazy-cached-centroid semantics on degenerate
> updates; expanded `clear()` rustdoc; `set_*` mutating builder methods
> for parity with `dia::segment`; explicit method enumeration on
> `EmbedModelOptions`; new `DegenerateEmbedding` error variant in both
> `dia::embed::Error` and `dia::cluster::Error` for zero-norm cases
> (rev 2 misnamed these as `NonFiniteInput`).
>
> **Revision 2 changes** (preserved from prior review): renamed
> `WINDOW_SAMPLES` to `EMBED_WINDOW_SAMPLES` so it doesn't collide with
> the segment constant; rewrote the online clusterer's centroid math to
> use an unnormalized accumulator (the rev-1 formula could produce NaN
> on antipodal updates); dropped `Linkage::Ward` (mathematically invalid
> with cosine distance) and `Affinity::Gaussian` (no defensible
> default); capped the eigengap auto-K; flipped `OverflowStrategy`
> default to `Reject` and changed `AssignClosest` to not update
> centroids on forced assignments; matches Python's `1e-12` epsilon;
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
256-d L2-normalized output) — and, after the rev-6 reconstruction scope
expansion, `pyannote.audio`'s `SpeakerDiarization` pipeline. dia v0.1.0
matches the algorithm shape of both faithfully but **diverges in
several documented places**:

| Divergence | Why | Where |
|---|---|---|
| Long-clip embedding aggregation: sliding-window mean instead of pyannote's gather-then-embed | Our fixed-200-frame fbank model can't accept variable-length features; pyannote's WeSpeaker ONNX export does | §5.1 (general path) |
| `exclude_overlap` mask: applied at sample-rate before fbank, not frame-rate after fbank | Same fixed-fbank-shape constraint | §5.8 (rev-6 / rev-8) |
| Online streaming clustering instead of pyannote's batch HAC / VBx / HMM-GMM | Streaming is the whole point of dia | §5.4 |
| Default offline clusterer: Spectral, not pyannote's VBx | VBx needs PLDA + bundled weights + batch-only; deferred to v0.2 (§15 #44) | §5.5 |
| K-means++ tie-break: smallest cluster-id wins (deterministic) | Pyannote uses `np.argsort` (implementation-defined for ties); ours is byte-stable | §5.5 step 8 |
| `Action::SpeakerScores` exposed; reconstruction runs externally to segment | Streaming requires per-window emission of raw probs; pyannote's pipeline holds them internally | §3 (rev-6 segment v0.X bump) |
| Reconstruction emits per-cluster runs as they close, not as a batch `Annotation` at end-of-stream | Streaming output | §5.11 |
| No `min_duration_on/off` merging | Deferred to v0.1.1 (§15 #48) | §4.4, §15 #48 |

Numerical-parity-sensitive callers should review the divergence
table above plus §5.1 / §5.8 before adopting. The clustering layer
itself is **novel** vs `findit-speaker-embedding` —
`findit-speaker-embedding` does not provide one; it stops at producing
fingerprints and leaves cross-window linking to its caller.

User-facing pipeline this spec assumes:

```
audio decoder → resample to 16 kHz → VAD → dia::Diarizer → downstream services
                                      │
                                      └─ ranges of human speech, variable-length,
                                         fed to dia incrementally as `process_samples`
                                         pushes.
```

**Critical prerequisite (rev-7).** dia's input is **VAD-filtered speech
in variable-length chunks**. Each `process_samples(samples)` push can be
any length:
- Sub-millisecond (e.g., 16 samples — won't trigger anything; buffered)
- Sub-clip (e.g., 0.5 s — buffered; spans emit only after enough audio
  accumulates or `finish_stream` is called)
- Single-clip (e.g., 2.3 s — typical VAD output for one utterance)
- Multi-clip (e.g., 30 s of concatenated VAD ranges)
- Whole-stream (e.g., 60 minutes pushed at once)

The algorithm handles all of these without special-casing — segment
lazily schedules windows when buffered samples reach
`k * step + WINDOW`; embed accepts any clip ≥ 25 ms; reconstruction
finalizes per-frame as the windows that contribute to each frame
complete. The variable-length contract is formalized in §11.12.

**The caller is responsible for mapping dia's input timeline back to
original audio time.** dia operates in "samples-pushed" coordinates
(monotonic from 0 at `clear()`); if the caller fed VAD-filtered audio
with silences removed, dia's sample positions don't correspond to
original-audio sample positions. The caller's VAD layer knows the
mapping; it owns the join.

Inside `Diarizer`, the flow matches `pyannote.audio`'s `SpeakerDiarization
.apply` pipeline (rev-6 scope expansion), adapted to streaming:

```
Segmenter → (per-window per-speaker raw probs)
          → exclude_overlap masked EmbedModel → (per-(window, speaker) embedding)
          → Clusterer.submit                  → (per-(window, speaker) cluster_id)
          → per-frame per-cluster overlap-add stitching
          → per-frame speaker-count tracking
          → count-bounded argmax + per-cluster RLE
          → emit one DiarizedSpan per closed speaker turn
```

This gives one merged span per speaker turn (matching pyannote's
`Annotation` output shape), not one span per (window, speaker) detection.

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
dia::segment     ←─── needs v0.X bump for Action::SpeakerScores (rev-6 scope expansion)
   ↓ (Segmenter, SpeakerActivity, SegmentModel, TimeRange, Action::SpeakerScores)
dia::Diarizer    ←─── orchestrates all three above; runs reconstruction
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

**`dia::segment` v0.X bump (rev-6 scope expansion):**
- New `Action::SpeakerScores { id: WindowId, window_start: u64, raw_probs: Box<[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize]> }` variant. Emitted from `push_inference(id, scores)` immediately before `Action::Activity` events for the same window. Carries the per-frame per-speaker raw probabilities used for downstream stitching.
- Mark `Action` as `#[non_exhaustive]` so future variants are non-breaking.
- No other public API changes; existing `Activity`, `VoiceSpan`, `NeedsInference` semantics unchanged.

**`dia::embed`:**
- `EmbedModel` ort wrapper for WeSpeaker ResNet34
- `EmbedModelOptions` mirroring `SegmentModelOptions`
- Pure-Rust `kaldi-native-fbank` for feature extraction
- Variable-length clip handling: zero-pad < 2 s, **sliding-window mean for > 2 s**
  (this is a deliberate IMPROVEMENT over Python's center-crop — see §5.1)
- Voice-weighted variant accepting caller-provided per-sample voice probabilities (rev-6: useful for direct VAD-probability streams; **NOT** used by Diarizer's `exclude_overlap` after rev-8 — see `embed_masked` below)
- **Rev-8: binary-mask variant** (`embed_masked` / `embed_masked_with_meta`): gather samples where `keep_mask[i] == true`, drop others, then run the standard embedding pipeline. Used by `Diarizer::exclude_overlap` (§5.8) to match pyannote's gather-and-embed mechanism.
- Two-tier API: low-level (raw `embed_features` / `embed_features_batch`) + high-level (`embed` / `embed_with_meta` / `embed_weighted` / `embed_weighted_with_meta` / `embed_masked` / `embed_masked_with_meta`)
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

**`dia::Diarizer` (rev-6: pyannote-style reconstruction):**
- Builder API: `Diarizer::builder()` with `with_*` setters only (no `options()` setter)
- Internal audio buffer with bounded retention (`dia::segment::WINDOW_SAMPLES` worth = 640 KB; see §5.7)
- `process_samples(seg_model, embed_model, samples, emit)` — synchronous: push → segment → embed (`exclude_overlap`-masked) → cluster → reconstruct → emit
- `finish_stream(seg_model, embed_model, emit)` — flush; closes any open per-cluster runs
- **`exclude_overlap` embedding**: when extracting an embedding for speaker `S` in window `W`, the per-sample mask zeroes out frames where any other speaker in `W` is also active. Falls back to the speaker-only mask if the clean mask is too short to embed. Matches `pyannote.audio/pipelines/speaker_diarization.py:375-425`. See §5.8.
- **Per-frame per-cluster overlap-add stitching**: as windows process, per-(window, slot) raw probabilities are collapsed-by-max within their cluster (matches `reconstruct` line 519-522), then summed into a global per-frame per-cluster activation accumulator (matches `Inference.aggregate(skip_average=True)`). See §5.9.
- **Per-frame instantaneous-speaker-count tracking**: per-window binarized speaker counts (sum across slots > onset) are aggregated via overlap-add MEAN and rounded (matches `speaker_count(warm_up=(0.1, 0.1))`). See §5.10.
- **Count-bounded argmax + per-cluster RLE**: as frames finalize (no future window can contribute), each frame picks its top-`count` clusters by activation; per-cluster runs are extended/closed; closed runs emit as `DiarizedSpan`. Matches `to_diarization` + `to_annotation`. See §5.11.
- `clear()` — reset for new session (also clears per-frame stitching state)
- `collected_embeddings()` — accessor returning `&[CollectedEmbedding]` with full context (per-(window, slot) granularity; preserved across reconstruction)
- `pending_inferences()`, `buffered_samples()`, `num_speakers()`, `speakers()`, **`buffered_frames()`** (new — number of un-finalized frames in the per-frame accumulator) — introspection
- Auto-derived `Send + Sync`

### Deferred — explicitly out of v0.1.0

| Item | Reason |
|---|---|
| Bundled WeSpeaker model | ~25 MB; matches soundevents posture (download script) |
| F1 parity tests vs `findit-speaker-embedding` Python output | Need Python infra; smoke tests cover basics |
| Threading / async service layer | Caller wraps with their own runtime |
| User-tunable `warm_up` for speaker count | Hardcoded to pyannote default `(0.1, 0.1)` for parity; expose as `DiarizerOptions::with_speaker_count_warm_up` if real demand emerges |
| User-tunable per-frame `min_duration_on/off` for output spans | We use existing voice-merge-gap-style filtering; see §5.11 |
| `min_cluster_size` cluster pruning | pyannote drops small clusters; we don't. v0.1.1 followup (§15) |
| VBx clustering (pyannote default) | §15 #44; needs PLDA model + batch-only |
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
///
/// **Invariant:** `||embedding.as_array()|| > NORM_EPSILON`. The crate
/// guarantees this — the only public constructor (`normalize_from`)
/// returns `None` for degenerate inputs. External callers and internal
/// downstream code (e.g., `Clusterer::submit`) can rely on this for
/// similarity computations being well-defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Embedding(pub(crate) [f32; EMBEDDING_DIM]);

impl Embedding {
    pub const fn as_array(&self) -> &[f32; EMBEDDING_DIM];
    pub fn as_slice(&self) -> &[f32];

    /// Cosine similarity. Both inputs are L2-normalized (per the
    /// `Embedding` invariant), so this reduces to a dot product. Returns
    /// a value in `[-1.0, 1.0]`.
    pub fn similarity(&self, other: &Embedding) -> f32;

    /// L2-normalize a raw 256-d inference output and wrap it.
    /// Returns `None` if `||raw|| < NORM_EPSILON` (degenerate input —
    /// callers should treat this as a bug or as input not meaningfully
    /// different from silence). Use after
    /// `EmbedModel::embed_features_batch` + custom aggregation.
    pub fn normalize_from(raw: [f32; EMBEDDING_DIM]) -> Option<Self>;
}

/// Free-function form of [`Embedding::similarity`] for callers who
/// prefer it. Both styles are public; pick whichever reads more
/// naturally at the call site.
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

    /// Rev-8: `keep_mask.len()` must equal `samples.len()` for `embed_masked`.
    #[error("keep_mask.len() = {mask_len} must equal samples.len() = {samples_len}")]
    MaskShapeMismatch { samples_len: usize, mask_len: usize },

    /// All windows had near-zero voice-probability weight; the weighted
    /// average is undefined. Almost always caller error.
    #[error("all windows had effectively zero voice-activity weight")]
    AllSilent,

    /// Input contains NaN or infinity.
    #[error("input contains non-finite values (NaN or infinity)")]
    NonFiniteInput,

    /// Input contains a zero-norm (or near-zero-norm, < `NORM_EPSILON`)
    /// embedding. Zero IS finite — kept distinct from `NonFiniteInput`
    /// so callers debugging real NaN/inf cases aren't misled.
    #[error("input contains a zero-norm or degenerate embedding")]
    DegenerateEmbedding,

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
#[derive(Default)]
pub struct EmbedModelOptions { /* private */ }

#[cfg(feature = "ort")]
impl EmbedModelOptions {
    pub fn new() -> Self;
    pub fn with_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub fn with_providers(self, providers: Vec<ExecutionProviderDispatch>) -> Self;
    pub fn with_intra_op_num_threads(self, n: usize) -> Self;
    pub fn with_inter_op_num_threads(self, n: usize) -> Self;
    // Mutating variants for parity with silero / dia::segment.
    pub fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self;
    pub fn set_providers(&mut self, providers: Vec<ExecutionProviderDispatch>) -> &mut Self;
    pub fn set_intra_op_num_threads(&mut self, n: usize) -> &mut Self;
    pub fn set_inter_op_num_threads(&mut self, n: usize) -> &mut Self;
}

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

    /// Soft per-sample weighting. Each element of `voice_probs` in `[0.0, 1.0]`
    /// scales its corresponding sample's contribution. For clips ≤ 2 s
    /// (single-window path), the weight collapses under L2-normalization
    /// — see §5.3 — so this primitive is best for the multi-window
    /// (>2 s) path or for callers passing direct VAD-probability streams.
    /// **NOT used for `Diarizer::exclude_overlap`** — see `embed_masked`
    /// below for the binary-keep-mask path.
    pub fn embed_weighted(
        &mut self, samples: &[f32], voice_probs: &[f32],
    ) -> Result<EmbeddingResult, Error>;
    pub fn embed_weighted_with_meta<A, T>(
        &mut self, samples: &[f32], voice_probs: &[f32], meta: EmbeddingMeta<A, T>,
    ) -> Result<EmbeddingResult<A, T>, Error>;

    /// **Rev-8.** Binary keep-mask: gather samples where `keep_mask[i] ==
    /// true`, drop others, then run the standard embedding pipeline (§5.1)
    /// on the gathered audio. Matches pyannote's
    /// `ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__` masked path
    /// (`pyannote/audio/pipelines/speaker_verification.py:568-619`),
    /// adapted for our fixed-200-frame fbank.
    ///
    /// **Two layered divergences from pyannote** (rev-9 expanded per
    /// review-8 T3-B):
    /// 1. **Mask resolution.** Pyannote interpolates the input mask to
    ///    fbank-frame rate, binarizes at 0.5, then gathers FRAMES (post
    ///    -fbank). We gather SAMPLES (pre-fbank) using the input mask
    ///    directly. Numerically equivalent up to fbank-window-boundary
    ///    effects when mask transitions land on 25 ms boundaries.
    /// 2. **Long-clip aggregation.** When `gathered.len() >
    ///    EMBED_WINDOW_SAMPLES` (= 32 000 samples), our `embed_masked`
    ///    routes through §5.1's sliding-window mean over the gathered
    ///    audio. Pyannote runs ONNX once on the variable-length
    ///    feature sequence (their WeSpeaker ONNX export accepts
    ///    variable-length input). For typical VAD-filtered single
    ///    utterances (≤ 2 s after gather) this divergence doesn't
    ///    matter; for long clips with sparse mask coverage it can
    ///    produce different embeddings.
    ///
    /// Net: short-clip masked embeddings are close to pyannote (one
    /// divergence — sample-vs-frame-rate gather). Long-clip masked
    /// embeddings have two layered divergences. §15 #49 tracks the
    /// fix for divergence #2 (mask-aware ONNX export).
    ///
    /// **Errors:**
    /// - `Error::MaskShapeMismatch` if `keep_mask.len() != samples.len()`.
    /// - `Error::InvalidClip` if the gathered length < `MIN_CLIP_SAMPLES`.
    /// - `Error::NonFiniteInput` if any sample is NaN/inf.
    ///
    /// **Caller responsibility:** if `embed_masked` returns `InvalidClip`
    /// because the gathered length was too short, the caller may decide
    /// to fall back (e.g., re-call with a less-restrictive mask). The
    /// `Diarizer` does this internally — see §5.8 fallback rule. After
    /// the second `InvalidClip`, the Diarizer skips the activity (rev-9
    /// per review-8 T3-A; matches pyannote's skip-and-continue).
    pub fn embed_masked(
        &mut self, samples: &[f32], keep_mask: &[bool],
    ) -> Result<EmbeddingResult, Error>;
    pub fn embed_masked_with_meta<A, T>(
        &mut self, samples: &[f32], keep_mask: &[bool], meta: EmbeddingMeta<A, T>,
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
///
/// Field visibility: private fields with accessors. Matches
/// `EmbeddingResult`, `DiarizedSpan`, and `SpeakerActivity` (sibling
/// output types across dia). Private fields let us evolve the
/// internal representation without a breaking API change — e.g., a
/// future minor version could expose `f64` similarity or carry the
/// originating window id.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerCentroid {
    speaker_id: u64,
    centroid: Embedding,           // L2-normalized; lazily derived from accumulator
    assignment_count: u32,
}

impl SpeakerCentroid {
    /// Global speaker ID assigned by the `Clusterer`.
    pub fn speaker_id(&self) -> u64;

    /// L2-normalized centroid (the public-facing best estimate of
    /// this speaker's fingerprint).
    pub fn centroid(&self) -> &Embedding;

    /// Number of `submit` calls assigned to this speaker (including
    /// `OverflowStrategy::AssignClosest` forced assignments, even
    /// though those don't update the centroid — see §5.4).
    pub fn assignment_count(&self) -> u32;
}

/// Result of one `Clusterer::submit` call.
///
/// Field visibility: private fields with accessors, mirroring
/// `SpeakerCentroid` (rev-5 alignment; rev-1..rev-4 had `pub` fields,
/// which mixed two visibility patterns within the same module).
#[derive(Debug, Clone, Copy)]
pub struct ClusterAssignment {
    speaker_id: u64,
    is_new_speaker: bool,
    similarity: Option<f32>,
}

impl ClusterAssignment {
    pub fn speaker_id(&self) -> u64;
    pub fn is_new_speaker(&self) -> bool;

    /// Cosine similarity to the assigned centroid, computed pre-update.
    /// `None` for the first-ever assignment (no centroids existed to
    /// compare against). For new speakers beyond the first, this is
    /// the maximum similarity to any existing centroid.
    ///
    /// **Invariant:** `similarity().is_none()` iff this is the
    /// first-ever assignment in the `Clusterer`'s lifetime (post
    /// `new()` or post `clear()`).
    pub fn similarity(&self) -> Option<f32>;
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
    // Mutating variants for parity with silero / dia::segment.
    pub fn set_similarity_threshold(&mut self, t: f32) -> &mut Self;
    pub fn set_update_strategy(&mut self, s: UpdateStrategy) -> &mut Self;
    pub fn set_max_speakers(&mut self, n: u32) -> &mut Self;
    pub fn set_overflow_strategy(&mut self, s: OverflowStrategy) -> &mut Self;
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

    /// Input contains a zero-norm or near-zero-norm embedding
    /// (||e|| < `NORM_EPSILON`). Distinct from `NonFiniteInput` (which
    /// is for NaN/inf). Almost certainly caller error: a real embedding
    /// produced by `EmbedModel::embed` is L2-normalized and cannot be
    /// degenerate. Likely cause: caller hand-constructed an embedding
    /// or read it from a corrupted store.
    #[error("input contains a zero-norm or degenerate embedding")]
    DegenerateEmbedding,

    /// All pairwise similarities ≤ 0, OR at least one node is isolated
    /// (its degree D_ii < `NORM_EPSILON`) → spectral clustering's
    /// normalized Laplacian is undefined. See §5.5 step 2.
    #[error("affinity graph has an isolated node or all-zero similarities; spectral clustering undefined")]
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
    /// K-means seed for spectral clustering.
    ///
    /// - `None` (default): use the constant `0` as the seed. This gives
    ///   deterministic output for a given input AND deterministic K-means
    ///   initialization across calls. Matches Python NumPy's
    ///   `np.random.RandomState(0)` convention for reproducibility.
    /// - `Some(s)`: use `s` directly. Useful when running multiple
    ///   independent clusterings on the same data (e.g., for
    ///   reproducibility studies) where varying seed checks robustness.
    ///
    /// We do NOT derive the seed from a hash of the input — doing so
    /// would require committing to a specific hash function (FxHash,
    /// SipHash, FNV, …), which is a public API decision the spec
    /// shouldn't take lightly. The constant default is simpler, equally
    /// deterministic for "same input → same output," and avoids the
    /// dependency / version-stability question.
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
    // Mutating variants.
    pub fn set_method(&mut self, m: OfflineMethod) -> &mut Self;
    pub fn set_similarity_threshold(&mut self, t: f32) -> &mut Self;
    pub fn set_target_speakers(&mut self, n: u32) -> &mut Self;
    pub fn set_seed(&mut self, s: u64) -> &mut Self;
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
    /// Onset threshold for binarizing per-frame per-speaker raw probabilities.
    /// Used by `exclude_overlap` (§5.8) and speaker-count tracking (§5.10).
    /// Matches pyannote's `Binarize(onset=...)` default in
    /// `pyannote/audio/utils/signal.py:235`. Default: 0.5.
    binarize_threshold: f32,
    /// Apply `exclude_overlap` mask when extracting embeddings (§5.8).
    /// Matches pyannote's `embedding_exclude_overlap` parameter.
    /// Default: `true`.
    exclude_overlap: bool,
}
impl Default for DiarizerOptions { /* sensible defaults */ }
impl DiarizerOptions {
    pub fn new() -> Self;
    pub fn with_segment_options(self, opts: SegmentOptions) -> Self;
    pub fn with_cluster_options(self, opts: ClusterOptions) -> Self;
    pub fn with_collect_embeddings(self, on: bool) -> Self;
    pub fn with_binarize_threshold(self, t: f32) -> Self;
    pub fn with_exclude_overlap(self, on: bool) -> Self;
    pub fn set_segment_options(&mut self, opts: SegmentOptions) -> &mut Self;
    pub fn set_cluster_options(&mut self, opts: ClusterOptions) -> &mut Self;
    pub fn set_collect_embeddings(&mut self, on: bool) -> &mut Self;
    pub fn set_binarize_threshold(&mut self, t: f32) -> &mut Self;
    pub fn set_exclude_overlap(&mut self, on: bool) -> &mut Self;
}

/// Per-activity context retained during a diarization session.
/// Returned by `Diarizer::collected_embeddings()`. Carries everything
/// needed to correlate the offline-clustering re-labeling back to its
/// source activity.
///
/// Granularity is **per-(window, slot)** — one `CollectedEmbedding` per
/// pre-reconstruction `SpeakerActivity` from `dia::segment`. This is
/// finer-grained than the post-reconstruction `DiarizedSpan` output
/// (one per closed cluster run). The two views are reconciled via
/// `online_speaker_id` (the cluster id assigned at embed/cluster time,
/// matching the `speaker_id` of the eventually-emitted `DiarizedSpan`s
/// for that cluster).
#[derive(Debug, Clone)]
pub struct CollectedEmbedding {
    pub range: TimeRange,
    pub embedding: Embedding,
    /// Online speaker id assigned by `Clusterer::submit` during streaming.
    pub online_speaker_id: u64,
    /// Window-local slot from `dia::segment::SpeakerActivity`.
    pub speaker_slot: u8,
    /// Whether the embedding used the `exclude_overlap` clean mask
    /// (`true`) or fell back to the speaker-only mask (`false`).
    /// See §5.8 for fallback semantics.
    pub used_clean_mask: bool,
}

/// One closed speaker turn after reconstruction.
///
/// **Rev-6 simplification:** rev-1..rev-5 had a per-(window, slot)
/// `DiarizedSpan` with `similarity` and `speaker_slot` fields. Rev-6
/// reconstruction emits per-cluster runs that span multiple windows,
/// so window-local concepts (slot, per-submission similarity) are no
/// longer well-defined for a single span. Window-local context
/// remains available via `Diarizer::collected_embeddings()`.
///
/// **Rev-7 enrichment:** added `average_activation`, `activity_count`,
/// and `clean_mask_fraction` for downstream storage and analysis (per
/// the user's "we want all important information" mandate). All three
/// are computed during reconstruction (§5.11) at zero algorithmic cost
/// (they're already-available accumulator values).
///
/// **Emission timing:** a `DiarizedSpan` for cluster `K` is emitted
/// when its current per-frame run closes — i.e., the first frame
/// where cluster `K` is NOT in the top-`count` finalized frames after
/// having been in them in the immediately previous finalized frame.
/// **Rev-8: there is NO `min_duration_off` merging in v0.1.0.**
/// Adjacent same-cluster runs separated by ≥ 1 finalized-frame gap
/// emit as separate spans. Pyannote's `to_annotation(min_duration_off
/// =...)` merging is deferred to v0.1.1 — see §15 #48. Callers who
/// want short-gap merging today can post-process the emitted span
/// stream themselves with a 1-element lookback (≈ 5 lines).
#[derive(Debug, Clone, Copy)]
pub struct DiarizedSpan {
    range: TimeRange,
    speaker_id: u64,
    /// `true` iff this is the first time `speaker_id` is emitted in
    /// the current `Diarizer` session (post-`new`/`clear`).
    is_new_speaker: bool,
    /// Mean per-frame normalized activation for this cluster across
    /// the span's frames. Each frame's contribution is
    /// `activation_sum / activation_chunk_count`, so the value is in `[0.0, 1.0]`
    /// (1.0 = every contributing window said "this speaker active here
    /// at probability 1.0"). Higher = more confident the cluster was
    /// active for this turn. Roughly comparable across spans.
    /// (Rev-7.)
    average_activation: f32,
    /// Number of `(WindowId, slot)` segment activities that contributed
    /// to this span (i.e., that were clustered into this span's
    /// `speaker_id` and whose frames overlap this span's `range`).
    /// (Rev-7.)
    activity_count: u32,
    /// Of the contributing activities, the fraction whose embedding
    /// used the `exclude_overlap` clean mask (vs falling back to the
    /// speaker-only mask). Range `[0.0, 1.0]`. Lower = more
    /// overlap-contaminated audio → less confident speaker
    /// attribution. (Rev-7.)
    clean_mask_fraction: f32,
}

impl DiarizedSpan {
    pub fn range(&self) -> TimeRange;
    pub fn speaker_id(&self) -> u64;
    pub fn is_new_speaker(&self) -> bool;
    pub fn average_activation(&self) -> f32;
    pub fn activity_count(&self) -> u32;
    pub fn clean_mask_fraction(&self) -> f32;
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

    /// Reset the Diarizer for a new session. After this call:
    ///
    /// - Internal `Segmenter` is cleared (`Segmenter::clear`): pending
    ///   inferences invalidated via generation-counter bump,
    ///   sample-position counter reset to 0, sample buffer drained,
    ///   per-frame voice timeline cleared, hysteresis state reset.
    /// - Internal `Clusterer` is cleared: all speaker entries removed,
    ///   next assigned `speaker_id` resets to 0.
    /// - **`total_samples_pushed()` resets to 0** (rev-7); the
    ///   "samples-pushed" timeline restarts. Caller-side
    ///   VAD-to-original-time mapping logs (per §11.12) must be
    ///   reset alongside.
    /// - Audio buffer is drained.
    /// - **Per-frame stitching state is dropped** (rev-6): the
    ///   per-frame per-cluster activation accumulator, per-frame
    ///   speaker-count accumulator, slot-to-cluster mapping, per-cluster
    ///   open-run state, and emitted-speaker tracking (for `is_new_speaker`).
    /// - **Open per-cluster runs are NOT emitted** — they're discarded.
    ///   Use `finish_stream` first if you want them flushed.
    /// - `collected_embeddings()` is **NOT** cleared by `clear()` —
    ///   call `clear_collected()` separately if you want to drop the
    ///   accumulated context (this matches how callers may want to
    ///   keep collected embeddings around for offline refinement
    ///   *after* the streaming session ends).
    /// - Configured options (segment / cluster / collect_embeddings /
    ///   binarize_threshold / exclude_overlap) are preserved.
    pub fn clear(&mut self);

    pub fn collected_embeddings(&self) -> &[CollectedEmbedding];
    pub fn clear_collected(&mut self);

    pub fn pending_inferences(&self) -> usize;
    pub fn buffered_samples(&self) -> usize;
    /// Number of un-finalized frames currently held in the per-frame
    /// per-cluster activation accumulator. Steady-state ≈ 589
    /// (one full window's worth of frames at the model's ≈ 58.9 fps);
    /// spikes during pumps that schedule multiple new windows. Useful
    /// for backpressure detection. (Rev-6; rev-8 corrected the
    /// frame-rate math from "100 fps × 10 s = 1000" to "58.9 fps,
    /// which gives ≈ 589 frames per 10 s window.")
    pub fn buffered_frames(&self) -> usize;
    /// Cumulative count of samples passed to `process_samples` since
    /// the last `clear()` (or since construction). Monotonic, never
    /// reset by `finish_stream`. Useful for callers mapping dia's
    /// "samples-pushed" timeline back to original-audio time when
    /// VAD-filtered audio is fed in: maintain a parallel
    /// `Vec<(dia_offset, original_offset)>` log keyed by this counter.
    /// (Rev-7.)
    pub fn total_samples_pushed(&self) -> u64;
    pub fn num_speakers(&self) -> usize;
    pub fn speakers(&self) -> Vec<SpeakerCentroid>;
}

impl DiarizerBuilder {
    pub fn new() -> Self;
    pub fn with_segment_options(self, opts: SegmentOptions) -> Self;
    pub fn with_cluster_options(self, opts: ClusterOptions) -> Self;
    pub fn with_collect_embeddings(self, on: bool) -> Self;
    pub fn with_binarize_threshold(self, t: f32) -> Self;
    pub fn with_exclude_overlap(self, on: bool) -> Self;
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

    /// An invariant of the Diarizer's internal state was violated.
    /// Distinct from `Embed`/`Segment`/`Cluster` because those wrap
    /// errors *from* the underlying state machines; `Internal` covers
    /// the integration glue itself (audio buffer indexing, activity
    /// range reconciliation). Almost certainly a bug in dia or a
    /// pathological caller (e.g., a custom mid-level composition that
    /// supplies out-of-order activities).
    #[error(transparent)]
    Internal(InternalError),
}

#[derive(Debug, thiserror::Error)]
pub enum InternalError {
    /// An emitted activity's range start is older than the audio
    /// buffer's earliest retained sample. Should never fire in
    /// practice — the §5.7 trim bound is exactly tight against the
    /// `dia::segment::Segmenter` reach-back. If this fires, either
    /// (a) Segmenter's reach-back contract regressed, (b) caller is
    /// using a hand-rolled mid-level composition that supplies
    /// out-of-order activities, or (c) the Diarizer's audio buffer
    /// was externally corrupted.
    #[error("activity range start {activity_start} is below audio buffer base {audio_base} (delta = {} samples)", audio_base - activity_start)]
    AudioBufferUnderflow {
        activity_start: u64,
        audio_base: u64,
    },

    /// An emitted activity's range end exceeds the audio buffer's
    /// latest sample. Should also never fire — the segment only emits
    /// activities for windows whose samples have been fully pushed.
    /// Same diagnosis space as `AudioBufferUnderflow`.
    #[error("activity range end {activity_end} exceeds audio buffer end {audio_end}")]
    AudioBufferOverrun {
        activity_end: u64,
        audio_end: u64,
    },
}
```

Each source module keeps its own error; the `Diarizer` wraps via `#[from]`
so callers can use `?` cleanly. Multi-error rationale: each module is
independently usable (and shippable to crates.io as a sub-crate someday),
so each owns its own error type. The Diarizer's wrapper is the integration
point.

**Rev-5 change:** rev-1..rev-4 returned `Error::Embed(Error::InvalidClip
{ len: 0, min: ... })` when the audio buffer underflowed for a
pathological activity range. That used `len: 0` as a sentinel for "this
isn't actually a clip-length issue, this is internal Diarizer state
inconsistency." Rev 5 splits that into a dedicated
`Error::Internal(InternalError::AudioBufferUnderflow)` so callers
debugging real `InvalidClip` errors aren't misled by sentinels.

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
    # multiple overlapping windows, equal-weighted mean.
    #
    # Regular hop count: as many full windows as fit before the end.
    # k_max = floor((len - EMBED_WINDOW_SAMPLES) / HOP_SAMPLES)
    # gives k in {0, 1, ..., k_max}. The window starting at
    # k_max * HOP_SAMPLES is fully contained but may not end at len.
    #
    # Then append a tail-anchored window at len - EMBED_WINDOW_SAMPLES
    # so the very end of the clip is covered. Dedup handles the case
    # where a regular hop already lands exactly on the tail anchor.
    let k_max = (len(samples) - EMBED_WINDOW_SAMPLES) / HOP_SAMPLES;  # integer floor
    starts = [k * HOP_SAMPLES for k in 0..=k_max] ++ [len(samples) - EMBED_WINDOW_SAMPLES]
    starts = dedup_and_sort(starts)
    sum = [0.0; 256]
    for s in starts:
        chunk = samples[s..s + EMBED_WINDOW_SAMPLES]
        features = compute_fbank(chunk)
        sum += onnx(features)
    return l2_normalize(sum, eps=NORM_EPSILON)
```

The trailing `[len - EMBED_WINDOW_SAMPLES]` plus dedup ensures the last
window covers the end of the clip, even when `len` isn't a clean
multiple of `HOP_SAMPLES`.

Worked examples (for the test cases in §9):
- `len = 32 000` (2.0 s): single-window path; `windows_used = 1`.
- `len = 48 000` (3.0 s): `k_max = 1` → starts `[0, 16 000]`, tail anchor `[16 000]`, dedup → `[0, 16 000]`; `windows_used = 2`.
- `len = 56 000` (3.5 s): `k_max = 1` → starts `[0, 16 000]`, tail anchor `[24 000]`, dedup → `[0, 16 000, 24 000]`; `windows_used = 3`.
- `len = 64 000` (4.0 s): `k_max = 2` → starts `[0, 16 000, 32 000]`, tail anchor `[32 000]`, dedup → `[0, 16 000, 32 000]`; `windows_used = 3`.

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
    /// Last-known-good L2-normalized centroid. Updated atomically with
    /// `accumulator` whenever the post-update accumulator norm is
    /// `>= NORM_EPSILON`. Held at the prior good value when the
    /// post-update accumulator is degenerate (see "Degenerate update"
    /// below). Initialized to the first submitted embedding (which is
    /// itself L2-normalized by `Embedding`'s invariant), so `cached_centroid`
    /// is always populated from the moment a SpeakerEntry exists.
    ///
    /// **Invariant:** `||cached_centroid|| > NORM_EPSILON`. This is what
    /// makes wrapping it in `Embedding` (via `pub(crate)` constructor)
    /// safe: the public `Embedding` invariant is preserved.
    cached_centroid: [f32; EMBEDDING_DIM],
    assignment_count: u32,
}
```

**Public-facing `SpeakerCentroid`** wraps `cached_centroid` (via
`Embedding`'s `pub(crate)` constructor). Callers see L2-normalized unit
vectors; the accumulator and the lazy-update bookkeeping are hidden.

**Update strategies:**

```
RollingMean (on assignment of new embedding e):
    accumulator += e            // pure addition; e is L2-normalized
    assignment_count += 1
    update_cached_centroid()    // see "Degenerate update" below

Ema(α) (on assignment of new embedding e):
    accumulator = (1 - α) * accumulator + α * e
    assignment_count += 1
    update_cached_centroid()
```

**Degenerate update** (handles antipodal accumulators):

```
fn update_cached_centroid(&mut self):
    let norm = l2_norm(&self.accumulator)
    if norm >= NORM_EPSILON:
        self.cached_centroid = self.accumulator / norm
    // else: leave cached_centroid at its prior value.
    //
    // Rationale: the new accumulator is degenerate (e.g., RollingMean
    // saw an exactly-antipodal pair and summed to 0). Re-normalizing
    // would return a meaningless direction (or NaN with eps=0). The
    // prior centroid is the best available estimate of the speaker
    // until a non-cancelling update arrives.
```

This is "lazy" in the sense that we don't recompute `cached_centroid`
from scratch on `speakers()` queries — it's already up to date.
"Lazy" here refers to the public `SpeakerCentroid` being derived from
private state on demand, not to deferred computation of
`cached_centroid` itself.

**Submit algorithm:**

```
for each submitted embedding e (L2-normalized; ||e|| ≈ 1 by Embedding invariant):
    if speakers.is_empty():
        speakers.push(SpeakerEntry {
            speaker_id: 0,
            accumulator: e,             // also has unit norm
            cached_centroid: e,         // first centroid is the embedding itself
            assignment_count: 1,
        })
        return ClusterAssignment {
            speaker_id: 0,
            is_new_speaker: true,
            similarity: None,           // no prior speakers; sentinel-free
        }

    sims = [s.cached_centroid · e for s in speakers]    // dot product = cosine sim (both unit)
    (best_idx, best_sim) = argmax_lowest_index(sims)    // tie-breaking: lowest index wins

    if best_sim >= threshold:
        speakers[best_idx].update(e, strategy)          // also updates cached_centroid
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
            cached_centroid: e,
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
0. Validate input (runs FIRST, before any matrix work):
     if embeddings.is_empty():
         return Error::EmptyInput
     if any embedding contains NaN or non-finite component:
         return Error::NonFiniteInput
     if any embedding has ||e|| < NORM_EPSILON:
         return Error::DegenerateEmbedding

   Then resolve target_speakers up front (cheap; informs fast paths):
     if let Some(k) = target_speakers:
         if k < 1: return Error::TargetTooSmall
         if (k as usize) > N: return Error::TargetExceedsInput { target: k, n: N }

0.1. Fast paths for tiny N (BEFORE building A or running eigendecomp):
     if N == 1:
         return Ok(vec![0])
     if N == 2:
         # Eigengap is degenerate at N=2 (only one gap, no information).
         # Fall back to threshold check.
         sim = max(0.0, embeddings[0] · embeddings[1])
         if let Some(2) = target_speakers:
             return Ok(vec![0, 1])             // user forced 2
         if let Some(1) = target_speakers:
             return Ok(vec![0, 0])             // user forced 1
         # Auto: threshold decides.
         return Ok(if sim >= similarity_threshold { vec![0, 0] } else { vec![0, 1] })

1. Build affinity matrix A ∈ R^{N×N}:
     A_ij = max(0, e_i · e_j)    // ReLU of cosine similarity
     A_ii = 0

2. Compute degree matrix D where D_ii = sum_j A_ij.
   PRECONDITION CHECK (rev-3 fix):
     if any D_ii < NORM_EPSILON:
         # An "isolated node": this embedding has no positive cosine
         # similarity to any other. The normalized Laplacian
         # L_sym = I - D^{-1/2} A D^{-1/2} contains a 1/sqrt(D_ii)
         # term, which would explode to infinity (or NaN with eps=0).
         #
         # The rev-2 spec checked only `A.iter().all(|x| < eps)`
         # (i.e., the *whole* affinity matrix is zero). That misses
         # the case where some pairs have positive similarity but at
         # least one node is dissimilar to everyone — the matrix is
         # still partially populated yet the Laplacian is ill-defined.
         return Error::AllDissimilar

3. Compute normalized graph Laplacian:
     L_sym = I - D^{-1/2} A D^{-1/2}

4. Eigendecompose L_sym; let λ = sorted eigenvalues (ascending),
   U = corresponding eigenvectors as columns.
     if eigendecomp fails or returns NaN:
         return Error::EigendecompositionFailed

5. Determine K:
     if let Some(k) = target_speakers:
         K = k                                // already validated in step 0
     else:
         # Eigengap heuristic with cap.
         K_max = min(N - 1, MAX_AUTO_SPEAKERS as usize)
         if K_max < 1:
             K = 1
         else:
             # gap[k] = λ[k+1] - λ[k]; pick the k with largest gap.
             K = 1 + argmax(λ[k+1] - λ[k] for k in 0..K_max)
             K = max(K, 1)

6. Take U[:, 0..K] (smallest-K eigenvectors as columns → N×K matrix).
7. Row-normalize the N×K matrix (each row to unit L2 norm; rows that
   are all-zero stay all-zero — they'll cluster arbitrarily and the
   eigendecomp would have already flagged a real degeneracy via
   step 2's precondition).
8. Run K-means on rows of the row-normalized matrix:
     - **Seed source** (deterministic):
         let rng_seed: u64 = match opts.seed {
             Some(s) => s,
             None    => 0,                    // documented constant default
         }
       The PRNG is **pinned to `rand_chacha::ChaCha8Rng`** (constructed
       via `ChaCha8Rng::seed_from_u64(rng_seed)`). NOT a different
       reproducible PRNG: different PRNGs produce different sequences
       from the same seed, which means different cluster labels for
       the same input — a user-observable breaking change. Swapping
       the PRNG (or upgrading `rand_chacha` to a major version that
       changes the stream algorithm) is a breaking change requiring
       a major version bump on dia. `rand`'s `thread_rng` is
       **never** used because it's seeded from OS entropy and
       non-deterministic across runs.

     - **K-means++ seeding** (Arthur & Vassilvitskii 2007) — exact
       algorithm pinned in spec to remove implementer ambiguity. Each
       step calls out the precise rand 0.10 API to use, because
       different APIs (e.g., `random_range` vs `Uniform::sample` vs
       manual modulo) consume different numbers of bytes from
       ChaCha8Rng's keystream and would shift the entire downstream
       state, producing different cluster labels for the same seed.
       ```
       Inputs: rows R = [r_0, ..., r_{N-1}] in R^K (the row-normalized
               eigenvector matrix), K target centroids, ChaCha8Rng `rng`.
       Output: K initial centroids c_0, ..., c_{K-1} in R^K.
       Implementation note: all f64 arithmetic below MUST execute
       strictly left-to-right (no SIMD reductions, no parallelization,
       no associativity rewrites). Floating-point sums and min reductions
       are non-associative; differing reduction orders flip near-tie
       comparisons by ≤ ULP, which is enough to change which row is
       picked at step 2a's argmin or step 2c's cumulative-mass
       crossing. Pin: write the loop as a sequential accumulator.

       1. Pick first centroid:
          // requires `use rand::distr::{Distribution, Uniform};`
          let i_0: usize = Uniform::new(0usize, N).unwrap().sample(&mut rng);
          c_0 = R[i_0].

          (`Uniform::new` returns `Result<Uniform<usize>, Error>` in
          rand 0.10; `.unwrap()` is safe here because N >= 1 is
          enforced by `Error::EmptyInput` in step 0. Do NOT use
          `rng.random_range(0usize..N)` — it's currently `Uniform`
          internally but that's an unstable implementation detail.)

       2. For k = 1..K:
            a. For each j in 0..N (left-to-right):
                 Compute D_j = min{ ||r_j - c_m||²  for m in 0..k }
                 by iterating m = 0, 1, ..., k-1 in order, tracking
                 running min. Use f64 for both the per-component
                 squared-difference accumulator and the min reduction.
            b. Let S = Σ_{j=0..N} D_j (sum left-to-right in f64).
               If `S == 0.0` exactly (all rows coincide with already-
               chosen centroids — possible when input has duplicates):
                 Build `available: Vec<usize>` by scanning j = 0..N in
                 order and including j iff it is NOT in the set of
                 already-chosen centroid indices (linear scan, NOT a
                 hash set, for byte-deterministic order). Pick:
                   let idx: usize =
                     Uniform::new(0usize, available.len())
                       .unwrap()
                       .sample(&mut rng);
                 Set j = available[idx], c_k = R[j], continue.
            c. Otherwise (S > 0):
                 // requires `use rand::RngExt;` (rand 0.10 split
                 // `random` out of `Rng` into `RngExt`)
                 Draw u via `let u: f64 = rng.random::<f64>();` (this
                 is `StandardUniform.sample(&mut rng)` in rand 0.10:
                 53-bit-mantissa f64 in [0.0, 1.0), half-open).
                 Compute threshold `t = u * S` (NOT `u`; we sample
                 against the un-normalized cumulative to avoid a
                 division-rounding step).
                 Walk j = 0, 1, 2, ..., maintaining a running prefix
                 sum `cum = Σ_{i=0..=j} D_i` (left-to-right in f64).
                 Pick the **smallest** j such that `cum > t` (strict
                 `>`, not `>=`; the strict comparison correctly skips
                 zero-mass rows whose D_j == 0 and ensures `u == 0`
                 still picks the first non-zero-mass row).
                 c_k = R[j].
       ```
       This pinning closes four byte-determinism gaps that the rev-3
       and rev-4 prose descriptions left ambiguous (step-1 sampling
       method, step-2b "uniformly from not-yet-chosen" method, step-2c
       f64 distribution, step-2c "crosses" semantics).

     - **Lloyd's iterations** with these termination conditions, in order:
         a. Convergence: every cluster assignment is unchanged from
            the previous iteration. Equivalently, the per-iteration
            "move count" is 0.
         b. Hard iteration cap: 100 iterations.
       Whichever fires first wins. Both are documented in rustdoc
       on `OfflineClusterOptions`.
9. Return per-input cluster labels in `[0, K)`. Labels are NOT
   guaranteed to be stable across input orderings — caller must not
   compare label values across two `cluster_offline` calls; use only
   for grouping.
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
| Affinity graph has any isolated node — i.e., some `D_ii < NORM_EPSILON` (spectral) | `Error::AllDissimilar` (this includes the degenerate sub-case where the entire affinity matrix is zero) |
| NaN in any embedding | `Error::NonFiniteInput` (validated at entry) |
| All-zero embedding (norm < `NORM_EPSILON`) | `Error::DegenerateEmbedding` (validated at entry; zero IS finite — distinct from NaN/inf) |
| Eigendecomposition NaN-out | `Error::EigendecompositionFailed` |

Tests in §9 cover each row.

### 5.6 Offline agglomerative clustering

Algorithm (HAC with Average linkage by default):

```
0. Validate input (runs FIRST, before any distance work):
     if embeddings.is_empty():
         return Error::EmptyInput
     if any embedding contains NaN or non-finite component:
         return Error::NonFiniteInput
     if any embedding has ||e|| < NORM_EPSILON:
         return Error::DegenerateEmbedding
     if let Some(k) = target_speakers:
         if k < 1: return Error::TargetTooSmall
         if (k as usize) > N: return Error::TargetExceedsInput { target: k, n: N }

0.1. Fast paths:
     if N == 1:
         return Ok(vec![0])
     if N == 2:
         sim = max(0.0, embeddings[0] · embeddings[1])
         if let Some(2) = target_speakers: return Ok(vec![0, 1])
         if let Some(1) = target_speakers: return Ok(vec![0, 0])
         return Ok(if sim >= similarity_threshold { vec![0, 0] } else { vec![0, 1] })

1. Compute pairwise distance matrix D ∈ R^{N×N}:
     D_ij = 1 - max(0, e_i · e_j)    // 0 = identical, 1 = orthogonal/anti
                                      // (clamped via ReLU to match spectral)
2. Initialize: each embedding is its own cluster.
3. Repeat:
     - Find the two most similar non-merged clusters (smallest D_ij).
     - if their distance >= (1 - threshold) AND target_speakers = None: stop.
     - if cluster count == target_speakers: stop.
     - Merge them.
     - Update distances per linkage:
         Single:   D[merged, k] = min(D[a, k], D[b, k])
         Complete: D[merged, k] = max(D[a, k], D[b, k])
         Average:  weighted by cluster sizes (Lance-Williams formula)
         (Ward removed: invalid with cosine distance.)
4. Return labels in `[0, num_clusters)`.
```

**Note on validation ordering vs spectral:** §5.5 step 0 and §5.6 step 0
are intentionally identical — both validate input fully BEFORE any
matrix computation. Same error variants in the same order. Single
implementation function `validate_offline_input(...)` shared by both
(see implementation plan).

### 5.7 Diarizer audio buffer policy

The Diarizer maintains a `VecDeque<f32>` indexed by absolute sample
position. **This policy assumes synchronous segment inference, as
performed inside `Diarizer::process_samples`.** Async out-of-order
`push_inference` is not supported on the Diarizer path; users needing
async must drop down to mid-level composition (§2.3) and manage
buffer retention manually.

After every `process_samples` returns:

- **Drop samples below `total_samples_pushed - dia::segment::WINDOW_SAMPLES`**
  (i.e., 160 000 samples = 10 s back). **CRITICAL:** The bound is the
  *segment* window size, not the *embed* window size.

**Why the segment window size, not the embed window size?**

The rev-2 spec used `EMBED_WINDOW_SAMPLES` (32 000 = 2 s) here, claiming
"the earliest sample we still need is the start of the most-recent
embed window." That reasoning is wrong: it confuses what
`embed_model.embed(...)` reads from the buffer with what the *next*
`process_samples` call will need.

The actual constraint comes from `dia::segment::Segmenter`. Code trace
on the v0.1.0-shipped segmenter (verified 2026-04-26):

- `schedule_ready_windows` (`segmenter.rs:161-173`) emits a window with
  start `k * step_samples` only when `total_samples_pushed >= k *
  step_samples + WINDOW_SAMPLES`. So after a synchronous push-and-pump
  cycle ending at total `T`, the largest emitted window has index
  `k_max = floor((T - WINDOW_SAMPLES) / step_samples)` and start
  `k_max * step_samples ≤ T - WINDOW_SAMPLES`.
- `emit_speaker_activities` (`segmenter.rs:282-329`) is invoked from
  `push_inference` and produces activities with absolute-sample range
  in `[k * step_samples, k * step_samples + WINDOW_SAMPLES)` for the
  window-`k` push. So the latest activity's range start is `k_max *
  step_samples`.
- The next process_samples call, pushing more audio, will (eventually)
  schedule window `k_max + 1` whose start is `(k_max + 1) *
  step_samples`. By the inequality above this is `> T -
  WINDOW_SAMPLES`. The activities from that window have range start
  `>= (k_max + 1) * step_samples > T - WINDOW_SAMPLES`.
- `finish()` (`segmenter.rs:453-474`) and `plan_starts` (`window.rs:36`)
  schedule the tail-anchor at `total_samples - WINDOW_SAMPLES`. The
  activity range from the tail can start exactly at `T - WINDOW_SAMPLES`.

Therefore the trim bound `T - WINDOW_SAMPLES`:
- is ≤ every future regular-window activity's start (no underflow);
- equals the tail-anchor activity's earliest start (exactly tight).

So WINDOW_SAMPLES is the **correct** bound — not WINDOW_SAMPLES + step
(which would over-allocate by `step_samples`) and not just
`step_samples` (which would underflow on `finish_stream`'s tail).
Reviewers in early revisions hypothesized other bounds; this one is
verified by the code trace above and by the integration tests in §9.

This is the same bound the `Segmenter` uses internally for its own
input buffer trim (`emit_window` line 207-208 trims to `(next_window_idx
+ 1) * step_samples`, which is at most `T - WINDOW_SAMPLES + step_samples`,
i.e., it retains `WINDOW_SAMPLES - step_samples` samples — close to but
slightly less than the Diarizer's bound, because the Segmenter doesn't
need the tail-anchor margin until `finish()` is called).

**Steady-state memory:** `dia::segment::WINDOW_SAMPLES * 4 bytes =
160 000 × 4 = 640 KB`. Plus `Diarizer`'s own `VecDeque` overhead and
the `CollectedEmbedding` vec (when `collect_embeddings = true`).

For a `SpeakerActivity` with absolute-sample range `[s0, s1)`, the slice
`audio_buffer[s0 - audio_base..s1 - audio_base]` is fed to
`embed_model.embed(...)`. **Defensive bounds checks** (return errors
rather than panicking):
- `s0 < audio_base` → `Error::Internal(InternalError::AudioBufferUnderflow
  { activity_start: s0, audio_base })`.
- `s1 > audio_base + audio_buffer.len()` →
  `Error::Internal(InternalError::AudioBufferOverrun { activity_end: s1,
  audio_end: audio_base + audio_buffer.len() as u64 })`.

Both should be unreachable under the segment contract verified in
§5.7. Their existence is purely defense-in-depth for malformed mid-level
compositions or future segment regressions.

### 5.8 Diarizer `exclude_overlap` embedding mask

Matches `pyannote/audio/pipelines/speaker_diarization.py:375-425` (the
`get_embeddings(file, binary_segmentations, exclude_overlap=True)`
caller path) plus `pyannote/audio/pipelines/speaker_verification.py:568-619`
(`ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__`'s actual mask
mechanism).

**Purpose.** When extracting speaker `S`'s embedding from a window that
also contains another speaker, the audio at frames where multiple
speakers are simultaneously active "contaminates" S's fingerprint with
the other speaker's voice. Masking those frames before feature
extraction gives a cleaner embedding, which improves cluster
separability — the single largest pre-clustering quality lever in
pyannote's pipeline.

**Mechanism (rev-8, corrected from rev-6/7).** Pyannote's
`ONNXWeSpeakerPretrainedSpeakerEmbedding` does **gather-and-embed**:
interpolate the input mask to fbank-frame rate, binarize at 0.5,
boolean-index-gather the fbank features (drop masked-out frames), then
run ONNX inference on the cleaned (variable-length) features.

Rev-6/7 mistakenly routed our path through `embed_weighted`, which is
a different primitive — per-sample soft-weighted aggregation that
collapses to a no-op for clips ≤ 2 s under L2-normalization (§5.3).
That made `exclude_overlap = true` silently ineffective for the most
common case (VAD-filtered single-utterance activities, typically
0.5-3 s). The rev-5 adversarial review caught this. Rev-8 routes
through `EmbedModel::embed_masked` (§4.2 rev-8 addition) which gathers
samples at sample-rate before fbank, then embeds normally.

**Numerical divergence from pyannote (documented).** Pyannote gathers
at frame-rate (after fbank); we gather at sample-rate (before fbank).
Equivalent up to fbank-window-boundary effects when mask transitions
land on 25 ms boundaries (the kaldi fbank window width). Differences
manifest only when the mask transitions in the middle of a fbank
window — the gathered audio is contiguous in our path, vs the
gathered features are contiguous in pyannote's.

**Frame-rate verified.** `dia::segment::stitch::frame_to_sample`
defines `WINDOW_SAMPLES * frame_idx / FRAMES_PER_WINDOW = 160 000 *
frame_idx / 589`, so a frame is ≈ 271.65 samples, ≈ 16.97 ms,
≈ 58.9 fps — **NOT** 100 fps / 10 ms / 160 samples (rev-6/7 prose
was wrong; rev-5 review caught this).

**Algorithm** (per pump, per window `W` that just had its activities
emitted):

```
Inputs (drained from `Segmenter`):
  raw_probs[slot][frame]: [[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS]
                          (from Action::SpeakerScores)
  activities[]: Vec<SpeakerActivity>
                (from Action::Activity emissions for window W)
  window_start: u64    (W.window_start, absolute samples; first sample of frame 0)

# Step 1: per-frame binarized speaker masks (frame-rate).
binarized: [[bool; MAX_SPEAKER_SLOTS]; FRAMES_PER_WINDOW]
for f in 0..FRAMES_PER_WINDOW:
    for s in 0..MAX_SPEAKER_SLOTS:
        binarized[f][s] = raw_probs[s][f] > binarize_threshold

# Step 2: per-frame "n_active" (number of slots active at this frame).
n_active: [u8; FRAMES_PER_WINDOW]
for f in 0..FRAMES_PER_WINDOW:
    n_active[f] = binarized[f].iter().filter(|&&b| b).count() as u8

# Step 3: per-activity mask construction + embedding.
for activity in activities:
    let s = activity.speaker_slot
    let s0 = activity.range.start    // absolute samples
    let s1 = activity.range.end

    # Per-frame masks (frame-rate, indexed in [0, FRAMES_PER_WINDOW)):
    speaker_mask_frames: [bool; FRAMES_PER_WINDOW]
    clean_mask_frames:   [bool; FRAMES_PER_WINDOW]
    for f in 0..FRAMES_PER_WINDOW:
        speaker_mask_frames[f] = binarized[f][s]
        clean_mask_frames[f]   = binarized[f][s] && n_active[f] == 1

    # Convert per-frame masks to per-sample keep_masks aligned with the
    # activity's absolute-sample range [s0, s1).
    # Frame f covers absolute samples
    #     [window_start + frame_to_sample(f),
    #      window_start + frame_to_sample(f + 1)).
    # We expand the frame mask, restricted to overlap with [s0, s1).
    let activity_len = (s1 - s0) as usize
    let mut speaker_keep = vec![false; activity_len]
    let mut clean_keep   = vec![false; activity_len]
    for f in 0..FRAMES_PER_WINDOW:
        let frame_s_start_abs = window_start + frame_to_sample(f as u32) as u64
        let frame_s_end_abs   = window_start + frame_to_sample((f + 1) as u32) as u64
        let lo_abs = frame_s_start_abs.max(s0)
        let hi_abs = frame_s_end_abs.min(s1)
        if lo_abs >= hi_abs { continue; }
        let lo_in_act = (lo_abs - s0) as usize
        let hi_in_act = (hi_abs - s0) as usize
        if speaker_mask_frames[f] { speaker_keep[lo_in_act..hi_in_act].fill(true) }
        if clean_mask_frames[f]   { clean_keep  [lo_in_act..hi_in_act].fill(true) }

    # Decide which mask: clean if it would gather enough samples, else fall back.
    let clean_kept_count = clean_keep.iter().filter(|&&b| b).count()
    let used_clean = clean_kept_count >= MIN_CLIP_SAMPLES as usize
    let keep_mask = if used_clean { &clean_keep } else { &speaker_keep }

    # Embed via gather-and-pad (§4.2 rev-8 embed_masked).
    # Two-step fallback chain (rev-9 per review-8 T3-A):
    #   1. Try clean_keep (if used_clean=true).
    #   2. If InvalidClip, fall back to speaker_keep.
    #   3. If InvalidClip AGAIN, skip the activity entirely and
    #      continue (do NOT propagate — matches pyannote's skip).
    let activity_samples = &audio_buffer[s0 - audio_base ..= s1 - audio_base - 1]
    let mut effective_used_clean = used_clean

    let result = match embed_model.embed_masked(activity_samples, keep_mask) {
        Ok(r) => Some(r),
        Err(Error::InvalidClip { .. }) if used_clean => {
            # Clean gather too short. Fall back to speaker-only mask.
            effective_used_clean = false
            match embed_model.embed_masked(activity_samples, &speaker_keep) {
                Ok(r) => Some(r),
                Err(Error::InvalidClip { .. }) => {
                    # Even the speaker-only mask gathers fewer than
                    # MIN_CLIP_SAMPLES. Skip this activity entirely.
                    # Reconstruction (§5.9) will see no contribution
                    # for (W.id, s); the frames effectively get
                    # count=0 from this slot.
                    #
                    # Optional: log at debug level (e.g., tracing::debug!).
                    None
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(Error::InvalidClip { .. }) => {
            # used_clean was already false (entered the speaker-only
            # branch). Same skip semantics as above.
            None
        }
        Err(e) => return Err(e.into()),
    }

    let Some(result) = result else { continue; }   # skip on doubly-failed gather
    let embedding = *result.embedding()
    let assignment = clusterer.submit(&embedding)?

    # Track for reconstruction (§5.9), span-quality metrics (§5.11), and observability.
    slot_to_cluster.insert((W.id, s), assignment.speaker_id())
    activity_clean_flags.insert((W.id, s), effective_used_clean)
    if collect_embeddings:
        collected_embeddings.push(CollectedEmbedding {
            range: activity.range,
            embedding,
            online_speaker_id: assignment.speaker_id(),
            speaker_slot: s,
            used_clean_mask: effective_used_clean,
        })
```

**`MIN_CLIP_SAMPLES`** as the threshold (rev-8 simplification). Rev-7
introduced a separate `MIN_CLEAN_FRAMES` constant computed from
`MIN_CLIP_SAMPLES / SAMPLES_PER_FRAME`. Rev-8 drops it: the
gather-and-pad path lets us check the gathered-sample count directly,
which is exactly the constraint that matters (`embed_masked` returns
`InvalidClip` if gathered < `MIN_CLIP_SAMPLES`). The frame-count
threshold was a proxy that would have been numerically equivalent
(`ceil(400 / 271.65) = 2` frames) but is now redundant.

**Pyannote's equivalent.** `pyannote/audio/pipelines/speaker_verification.py:611-612`:
```python
if masked_feature.shape[0] < self.min_num_frames:
    continue   # skip the embedding for this (chunk, speaker)
```
We diverge: rev-8 falls back to the speaker-only mask rather than
skipping. Skipping would lose the embedding for activities with very
short clean regions, which is precisely the case where overlap-aware
masking matters most. Falling back to the speaker-only mask preserves
the embedding (with some contamination) and lets `clean_mask_fraction`
in the output `DiarizedSpan` (§4.4) signal the quality degradation to
the caller.

**Observability.** `CollectedEmbedding::used_clean_mask` and
`DiarizedSpan::clean_mask_fraction` (§4.4) let callers detect
heavy-overlap audio: if a high fraction of activities fell back to the
speaker-only mask, embedding quality (and downstream cluster purity)
is likely degraded for that span.

### 5.9 Per-frame per-cluster activation stitching

Matches `pyannote/audio/pipelines/speaker_diarization.py:480-528`
(`reconstruct`) plus `pyannote/audio/core/inference.py:499-620`
(`Inference.aggregate(skip_average=True)`).

**Goal.** From the per-(window, slot) raw probabilities and the
per-(window, slot) cluster assignments produced in §5.8, build a
per-absolute-frame per-cluster activation accumulator. Frames close
out (finalize) when no future window can contribute to them; finalized
frames feed §5.11's count-bounded argmax.

**State.**

```rust
struct ReconstructState {
    /// Absolute frame at index 0 of the per-frame buffers.
    base_frame: u64,

    /// Per-(absolute frame - base_frame) per-cluster activation sum.
    /// Outer Vec indexed by `frame - base_frame`; inner HashMap is
    /// keyed by global cluster id (sparse — most clusters are inactive
    /// at any given frame). Cluster ids never get re-keyed when
    /// new clusters appear; we just grow the HashMap.
    activations: VecDeque<HashMap<u64, f32>>,

    /// Per-frame bookkeeping for §5.10 (one `FrameCount` per absolute
    /// frame, indexed by `frame - base_frame`).
    counts: VecDeque<FrameCount>,

    /// Per-(window_id, slot) → cluster id. Populated in §5.8 when
    /// activities are clustered. Consumed (per window) when that
    /// window's `SpeakerScores` arrive and reconstruction integration
    /// runs (§5.9 step A). Persists across the integration so §5.11
    /// can attribute open-run frames to contributing activities (used
    /// to populate `DiarizedSpan::activity_count`). Eviction in §11.13.
    slot_to_cluster: BTreeMap<(WindowId, u8), u64>,

    /// Per-(window_id, slot) → bool flag indicating whether the
    /// activity used the clean mask. Populated in §5.8, consumed by
    /// §5.11 to populate `DiarizedSpan::clean_mask_fraction`. Same
    /// eviction lifecycle as `slot_to_cluster` (§11.13).
    activity_clean_flags: HashMap<(WindowId, u8), bool>,

    /// Per-window_id → window_start (absolute samples). Populated
    /// during integration in §5.9 step A; consumed by §5.11 to map
    /// frame indices back to absolute samples and to determine which
    /// (window, slot) entries can be evicted (§11.13).
    window_starts: HashMap<WindowId, u64>,

    /// Per-cluster open-run state. Updated in §5.11 each time
    /// `emit_finalized_frames` advances `base_frame`.
    open_runs: HashMap<u64 /* cluster_id */, PerClusterRun>,

    /// Set of `cluster_id`s that have ever been emitted in a
    /// `DiarizedSpan` since `new()` or `clear()`. Used to set
    /// `DiarizedSpan::is_new_speaker`.
    emitted_speaker_ids: HashSet<u64>,

    /// Absolute frame below which everything is finalized. Monotonic.
    finalization_boundary: u64,
}

struct FrameCount {
    // ── Activation bookkeeping (no warm-up trim, used in §5.11
    //     average_activation normalization). ──
    /// Number of windows whose un-trimmed frames contributed to this
    /// frame's per-cluster activation accumulator. **NEVER** identical
    /// to `count_chunk_count`; the activation accumulator includes
    /// warm-up frames whereas the speaker-count accumulator excludes
    /// them. Rev-7 conflated these two; rev-8 split them per
    /// review-7 T2-A, matching pyannote's
    /// `Inference.aggregate(skip_average=True)` overlapping_chunk_count
    /// independence from `speaker_count`'s separate trimmed count.
    activation_chunk_count: u32,

    // ── Speaker-count bookkeeping (warm-up trimmed, used in §5.10 +
    //     §5.11 count-bounded argmax). ──
    /// Sum across windows of (count of speakers > threshold at this
    /// frame), only from frames OUTSIDE each contributing window's
    /// warm-up margin (see §5.10).
    count_sum: f32,
    /// Number of windows whose warm-up-trimmed frames contributed to
    /// this frame's `count_sum`. Used as the divisor when rounding to
    /// the integer instantaneous speaker count.
    count_chunk_count: u32,
}
```

**Rev-9 organization fix.** Rev-8 split this into `ReconstructState`
+ `ReconstructStateExtras` (a holdover from an earlier draft), which
caused the rev-8 review to flag `slot_to_cluster` and
`activity_clean_flags` as duplicate-declared. Rev-9 merges everything
into a single `ReconstructState` so there's one place to read it.
`PerClusterRun` is defined in §5.11 below; field documentation for
each member is co-located with its first use.

**Per-window integration** (called from §5.8's pump loop after a
window's activities have all been clustered AND its `Action::SpeakerScores`
have arrived):

**Integration trigger** (rev-8 explicit): in the Diarizer's pump loop
(§5.8), buffer each `Action::SpeakerScores` per `WindowId`. After each
`Segmenter::push_inference(id, scores)` returns, drain the queued
`Action::Activity` events for `id` (clustering each — §5.8), then
invoke `integrate_window(id)`. This is well-defined because both the
`SpeakerScores` and the `Activity` events for window `id` are emitted
synchronously in `push_inference`'s call frame; both have arrived
before the next `Action::NeedsInference` would be polled.

```
fn integrate_window(W, raw_probs[slot][frame], slot_to_cluster):
    let window_start = W.window_start as u64    // absolute samples
    let window_start_frame = frame_index_of(window_start)
    window_starts.insert(W.id, window_start)    // (T2-C, populated here)
    grow activations / counts buffers up to window_start_frame + FRAMES_PER_WINDOW

    # Step A: collapse-by-max within each cluster for THIS window.
    # (Matches reconstruct line 519-522: clustered_segmentations[c, :, k] =
    #  max over slots-in-cluster-k of segmentation[c, :, slot].)
    let mut per_cluster_max: HashMap<u64, [f32; FRAMES_PER_WINDOW]>
    for slot in 0..MAX_SPEAKER_SLOTS:
        match slot_to_cluster.get((W.id, slot)):
            Some(cluster_id):
                for f in 0..FRAMES_PER_WINDOW:
                    let v = raw_probs[slot][f]
                    let entry = per_cluster_max.entry(*cluster_id).or_insert([0.0; FRAMES_PER_WINDOW])
                    entry[f] = entry[f].max(v)
            None:
                # Slot was never active in this window (or activity was
                # too short). Skip — it does NOT contribute to any cluster.
                # Equivalent to pyannote's `inactive_speakers` → cluster -2 path.
                continue

    # Step B: overlap-add SUM into the global per-frame accumulator
    # (Matches Inference.aggregate(skip_average=True) line 598-600).
    # NOTE: pyannote sets warm_up=(0.0, 0.0) for the activation aggregate,
    # so we count EVERY frame (no warm-up trim). This is independent of
    # the speaker_count overlap-add MEAN below (Step C), which DOES trim.
    for (cluster_id, frame_scores) in per_cluster_max:
        for f_in_window in 0..FRAMES_PER_WINDOW:
            let abs_frame = window_start_frame + f_in_window
            let buf_idx = (abs_frame - base_frame) as usize
            let entry = activations[buf_idx].entry(cluster_id).or_insert(0.0)
            *entry += frame_scores[f_in_window]
    # In a parallel loop (or in the loop above), bump the activation
    # chunk-count for every frame in the window (no warm-up trim).
    for f_in_window in 0..FRAMES_PER_WINDOW:
        let abs_frame = window_start_frame + f_in_window
        let buf_idx = (abs_frame - base_frame) as usize
        counts[buf_idx].activation_chunk_count += 1

    # Step C: speaker-count bookkeeping — for each frame, accumulate the
    # binarized speaker count from THIS window for §5.10.
    # WITH warm-up trimming (matches pyannote `speaker_count(warm_up=
    # (0.1, 0.1))`).
    for f_in_window in 0..FRAMES_PER_WINDOW:
        let abs_frame = window_start_frame + f_in_window
        let buf_idx = (abs_frame - base_frame) as usize

        let in_warm_up = f_in_window < SPEAKER_COUNT_WARM_UP_FRAMES_LEFT
                       || f_in_window >= (FRAMES_PER_WINDOW - SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT)
        if !in_warm_up:
            let n_active_here = (0..MAX_SPEAKER_SLOTS)
                .filter(|s| raw_probs[s][f_in_window] > binarize_threshold)
                .count() as f32
            counts[buf_idx].count_sum += n_active_here
            counts[buf_idx].count_chunk_count += 1

    # Step D: advance finalization boundary and emit (§5.11).
    advance_finalization_boundary(W.id)
    emit_finalized_frames()

    # Step E: evict (W.id, *) entries from slot_to_cluster /
    # activity_clean_flags / window_starts whose window's last frame
    # has finalized AND whose run is no longer open. See §11.13.
    evict_finalized_window_metadata()
```

**Finalization boundary**. A frame `f` is finalized when no future
window can contribute to it. Following the same logic as
`dia::segment::stitch::next_finalization_boundary`: a window with start
`s` covers frames `[frame_index_of(s), frame_index_of(s + WINDOW_SAMPLES))`.
The smallest start of any *future* window is the next
`Segmenter`-scheduled start, which is `(next_window_idx) * step_samples`
**at the moment integration runs**. So:

```
fn advance_finalization_boundary(W_id):
    let next_window_start_frame = frame_index_of(
        segmenter.peek_next_window_start()
    )
    self.finalization_boundary = max(self.finalization_boundary, next_window_start_frame)
    // Equivalently: at end-of-stream (post-finish_stream), set to total_frames.
```

The `Segmenter::peek_next_window_start()` accessor is added as part
of the rev-6 segment v0.X bump (§3 in-scope items, "`dia::segment`
v0.X bump" bullet). Returns `(next_window_idx as u64) * step_samples`
if `!finished`, else `u64::MAX` (no future windows).

### 5.10 Per-frame instantaneous-speaker-count tracking

Matches `pyannote/audio/pipelines/utils/diarization.py:150-186`
(`speaker_count`) with default `warm_up=(0.1, 0.1)`.

**Definition.** At each absolute frame `f`, `count[f]` is the rounded
overlap-add MEAN of binarized speaker counts across all windows
that contribute to `f` (after trimming each window's warm-up
left/right margins).

The binarization (`raw_probs[slot][f] > binarize_threshold`) is
per-window per-frame. The warm-up trimming drops contributions from
the first/last 10% of each window (= 58 frames for `FRAMES_PER_WINDOW
= 589`); pyannote's intuition is that the model's first/last frames
are less reliable due to lack of left/right context.

**Constants** (rev-6, derived from pyannote defaults):

```rust
/// Pyannote's `Inference.trim` warm-up ratio. We hardcode this to the
/// pyannote default; configurability is deferred to v0.1.1 if needed.
pub const SPEAKER_COUNT_WARM_UP_RATIO_LEFT: f32 = 0.1;
pub const SPEAKER_COUNT_WARM_UP_RATIO_RIGHT: f32 = 0.1;

/// Number of frames trimmed from each side of a window before counting.
/// = round(FRAMES_PER_WINDOW * 0.1) = 59.
pub const SPEAKER_COUNT_WARM_UP_FRAMES_LEFT: u32 = 59;
pub const SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT: u32 = 59;
```

(These are private constants in `dia::diarizer::reconstruct`. The
caller never sees them.)

**Computation.** Done inline in §5.9 step C; see that block for the
exact loop. The §5.10-specific bit is the warm-up trimming guard
(`f_in_window` outside `[SPEAKER_COUNT_WARM_UP_FRAMES_LEFT,
FRAMES_PER_WINDOW - SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT)`) and the
counter being `count_chunk_count` (warm-up trimmed) — distinct from
§5.9 step B's `activation_chunk_count` (un-trimmed).

(Rev-9 removed a duplicated copy of the integration loop that lived
here in rev-8; the duplicate had a stale `chunk_count` field name —
see review-8 T2-A. Single source of truth now lives in §5.9 step C.)

**Finalization** (called from §5.11):

```
fn finalized_count_at(frame): u32 {
    let c = counts[frame - base_frame]
    if c.count_chunk_count == 0 { return 0 }    // frame in warm-up of all contributing windows
    return (c.count_sum / c.count_chunk_count as f32).round() as u32
}
```

Capped at runtime by `max_speakers` from `ClusterOptions::max_speakers`
(matching `pyannote/audio/pipelines/speaker_diarization.py:676`'s
`count.data = min(count.data, max_speakers)`).

### 5.11 Count-bounded argmax + per-cluster RLE-to-spans

Matches `pyannote/audio/pipelines/utils/diarization.py:221-268`
(`to_diarization`) plus the run-length-encoding done by `Binarize`
in `to_annotation`.

**State.**

```rust
struct PerClusterRun {
    cluster_id: u64,
    /// Absolute frame at which the current run started. None = no open run.
    start_frame: Option<u64>,
    /// Last absolute frame where this cluster was in the top-c. Used for
    /// merge-gap logic (see below).
    last_active_frame: Option<u64>,

    // ── Rev-7 quality-metric accumulators (per-run) ──
    /// Sum over this run's finalized frames of the cluster's
    /// per-frame normalized activation (= activation_sum / chunk_count).
    /// On run close, `mean = activation_sum_normalized / frame_count`
    /// is the `DiarizedSpan::average_activation`.
    activation_sum_normalized: f64,
    /// Number of finalized frames included in the run so far.
    /// On run close, divides `activation_sum_normalized`.
    frame_count: u32,
    /// Set of `(WindowId, slot)` activities whose frames have
    /// contributed to this run. Sized typically <= 10 entries per run.
    /// On run close, count = `DiarizedSpan::activity_count`.
    contributing_activities: HashSet<(WindowId, u8)>,
    /// Of the contributing activities so far, how many used the clean
    /// mask. On run close, `clean_count / activity_count =
    /// DiarizedSpan::clean_mask_fraction`.
    clean_mask_count: u32,
}

// `open_runs`, `emitted_speaker_ids`, and `activity_clean_flags`
// live on `ReconstructState` declared in §5.9 — see that block for
// the full layout. (Rev-9 merged the three duplicate `ReconstructState
// + ReconstructStateExtras` snippets that rev-8 left scattered across
// §5.9 / §5.11.)
```

**Algorithm** (run as part of finalization, after each window is
integrated in §5.9). Rev-8 fixed three bugs surfaced by review-7
T2-B, T3-F, T3-H: empty-deque guard on the flush loop, `u64`
arithmetic for frame indices throughout (no truncating cast to
`u32`), and `total_cmp` instead of `partial_cmp` for the argmax
comparison so NaN-defense doesn't silently demote NaN clusters to
the back.

```
fn emit_finalized_frames(emit: &mut FnMut(DiarizedSpan)):
    while base_frame < finalization_boundary {
        // ── Rev-8 T2-B: guard the deque pop. activations may be empty
        //    when finalization_boundary is set to u64::MAX during
        //    flush_open_runs but the buffer was already drained.
        let Some(frame_state): Option<HashMap<u64, f32>> = activations.pop_front() else { break; };
        let frame_count_state: FrameCount = counts.pop_front()
            .expect("counts and activations grow in lockstep");
        // ── Rev-8 T2-A: use ACTIVATION chunk count (un-trimmed) here.
        //    speaker-count chunk count is for §5.10 only.
        let activation_chunk_count = frame_count_state.activation_chunk_count.max(1) as f32

        let count = finalized_count_at(base_frame).min(max_speakers as u32)

        # Pick top-`count` clusters by activation at this frame.
        # (Matches to_diarization line 261-267: argsort(-activations,
        # axis=-1)[:count].)
        let top: Vec<(u64, f32)> = if count > 0 {
            let mut sorted: Vec<(u64, f32)> = frame_state.iter()
                .map(|(&c, &a)| (c, a)).collect()
            // Rev-8 T3-H: use total_cmp for NaN safety. partial_cmp
            // returns None for NaN comparisons; unwrap_or(Equal) would
            // silently demote NaN values inconsistently. total_cmp
            // gives a strict total ordering with NaN at the highest
            // (or lowest) end. Activations from the integration step
            // SHOULD always be finite — defense-in-depth only.
            // Tie-break: lower cluster_id wins (deterministic).
            sorted.sort_by(|(a_id, a_v), (b_id, b_v)| {
                b_v.total_cmp(a_v)            // descending by activation
                    .then(a_id.cmp(b_id))    // ascending by id on tie
            })
            sorted.into_iter().take(count as usize).collect()
        } else {
            Vec::new()
        }

        # For each cluster: extend or open a run if active here; close
        # any open run for clusters not in `top`.
        let active_set: HashSet<u64> = top.iter().map(|&(id, _)| id).collect()

        # Close runs for clusters that were open but are NOT in top now.
        for (cluster_id, run) in self.open_runs.iter_mut() {
            if !active_set.contains(cluster_id) {
                if let Some(start) = run.start_frame.take() {
                    let end_frame = run.last_active_frame.unwrap() + 1
                    let range = make_range(start, end_frame)
                    let is_new = !emitted_speaker_ids.contains(cluster_id)
                    if is_new { emitted_speaker_ids.insert(*cluster_id); }
                    emit(DiarizedSpan {
                        range,
                        speaker_id: *cluster_id,
                        is_new_speaker: is_new,
                        average_activation: (run.activation_sum_normalized
                                             / run.frame_count.max(1) as f64)
                                             as f32,
                        activity_count: run.contributing_activities.len() as u32,
                        clean_mask_fraction: if run.contributing_activities.is_empty() {
                            0.0
                        } else {
                            run.clean_mask_count as f32
                                / run.contributing_activities.len() as f32
                        },
                    })
                    // Reset metric accumulators for next run on this cluster.
                    run.activation_sum_normalized = 0.0
                    run.frame_count = 0
                    run.contributing_activities.clear()
                    run.clean_mask_count = 0
                }
            }
        }

        # Open or extend runs for clusters in `top`.
        for &(cluster_id, activation_sum) in &top {
            let run = self.open_runs.entry(cluster_id).or_insert_with(|| PerClusterRun {
                cluster_id,
                start_frame: None,
                last_active_frame: None,
                activation_sum_normalized: 0.0,
                frame_count: 0,
                contributing_activities: HashSet::new(),
                clean_mask_count: 0,
            })
            if run.start_frame.is_none() {
                run.start_frame = Some(base_frame)
            }
            run.last_active_frame = Some(base_frame)

            # Quality-metric accumulation. Use the un-trimmed
            # activation_chunk_count (bound at the top of this loop
            # iteration) for normalization — see §5.10 for why this
            # is distinct from the warm-up-trimmed count_chunk_count.
            run.activation_sum_normalized += (activation_sum / activation_chunk_count) as f64
            run.frame_count += 1

            # Find activities (window, slot) whose frame_range contains base_frame
            # AND that map to this cluster_id, then add to contributing_activities.
            for ((window_id, slot), c) in &slot_to_cluster {
                if *c != cluster_id { continue; }
                let win = window_starts.get(window_id).expect("window seen during integration")
                let frame_lo = frame_index_of(*win)
                let frame_hi = frame_lo + FRAMES_PER_WINDOW as u64
                if base_frame >= frame_lo && base_frame < frame_hi {
                    if run.contributing_activities.insert((*window_id, *slot)) {
                        // Newly inserted — also count toward clean_mask if applicable.
                        if let Some(&clean) = activity_clean_flags.get(&(*window_id, *slot)) {
                            if clean { run.clean_mask_count += 1; }
                        }
                    }
                }
            }
        }

        base_frame += 1
```

**Note on metric-accumulator memory.** `slot_to_cluster` is bounded by
`(open_windows × MAX_SPEAKER_SLOTS)` ≈ a few dozen entries during the
~10 s rolling window of un-finalized frames. Lookups are linear over
this small set, which is comparable to a HashMap lookup at this scale
and avoids the hash overhead. `activity_clean_flags` and the
per-(window_id, slot) entries in `slot_to_cluster` are evicted once
all frames in their window have finalized AND the run that referenced
them has closed (eviction is bookkeeping that happens after each
`emit_finalized_frames` call; not shown in the pseudo-code above for
clarity).

**End-of-stream flush** (called from `finish_stream` after the segment's
tail-anchor activities have integrated):

```
fn flush_open_runs(emit: &mut FnMut(DiarizedSpan)):
    # Force-finalize remaining frames up to total_frames.
    self.finalization_boundary = u64::MAX
    emit_finalized_frames(emit)
    # Then close every still-open run with the same quality-metric
    # population as in emit_finalized_frames above.
    for (cluster_id, mut run) in std::mem::take(&mut self.open_runs) {
        if let Some(start) = run.start_frame {
            let end_frame = run.last_active_frame.unwrap() + 1
            let range = make_range(start, end_frame)
            let is_new = !emitted_speaker_ids.contains(&cluster_id)
            if is_new { emitted_speaker_ids.insert(cluster_id); }
            emit(DiarizedSpan {
                range,
                speaker_id: cluster_id,
                is_new_speaker: is_new,
                average_activation: (run.activation_sum_normalized
                                     / run.frame_count.max(1) as f64)
                                     as f32,
                activity_count: run.contributing_activities.len() as u32,
                clean_mask_fraction: if run.contributing_activities.is_empty() {
                    0.0
                } else {
                    run.clean_mask_count as f32
                        / run.contributing_activities.len() as f32
                },
            })
        }
    }
```

**Frame-to-sample conversion** (`make_range`). Rev-8's changelog
overstated what the spec actually did — it kept a `as u32`
truncating cast on `start_frame`/`end_frame` (review-8 T2-B caught
this). Rev-9 fixes by using a Diarizer-internal `u64`-throughout
helper that's bit-for-bit equivalent to
`dia::segment::stitch::frame_to_sample` but doesn't truncate.

```rust
/// `dia::diarizer::reconstruct::frame_to_sample_u64`. Internal
/// helper. Bit-for-bit equivalent of
/// `dia::segment::stitch::frame_to_sample(frame_idx: u32) -> u32`
/// (rounded division), but operates on `u64` throughout to avoid
/// truncating frame indices on long sessions. The segment's
/// internal helper is `pub(crate)` and signatures u32 → u32 because
/// segment never produces frame indices > FRAMES_PER_WINDOW × N
/// for any single window's frame loop. The Diarizer holds an
/// absolute frame count and needs u64.
const fn frame_to_sample_u64(frame_idx: u64) -> u64 {
    let n = frame_idx * WINDOW_SAMPLES as u64;
    let half = (FRAMES_PER_WINDOW as u64) / 2;
    (n + half) / FRAMES_PER_WINDOW as u64
}

fn make_range(start_frame: u64, end_frame: u64) -> TimeRange {
    let s0 = frame_to_sample_u64(start_frame) as i64;
    let s1 = frame_to_sample_u64(end_frame) as i64;
    TimeRange::new(s0, s1, SAMPLE_RATE_TB)
}
```

Bit-exact equivalence to `dia::segment::stitch::frame_to_sample` is
asserted by a unit test that runs both functions on `frame_idx in
0..=FRAMES_PER_WINDOW * 4` (i.e., `0..=2356`, which covers the u32
input range that segment uses) and checks they produce identical
outputs after the appropriate u32 → u64 widening.

(Once `dia::segment` adds a `pub(crate) frame_to_sample_u64` helper
of its own — out of scope for v0.X bump but a v0.1.1 candidate —
the Diarizer copy can be deleted in favor of segment's. Tracked as
§15 #54.)

**Tie-break determinism.** When two clusters have exactly equal
activation at a frame, `sort_by(b_v.cmp(a_v).then(a_id.cmp(b_id)))`
picks the one with the smaller `cluster_id` (== the one assigned
earlier in the session). This makes streaming output deterministic
under reordering of clustering operations. Pyannote uses
`np.argsort` which has implementation-defined tie-break; we pick a
documented behavior.

### 5.12 Diarizer error handling policy

When `embed_model.embed_masked(slice, keep_mask)` (rev-8 path; was
`embed_weighted` in rev-1..rev-7 — review-8 T2-C caught the stale
reference) returns `Err` for a particular activity:
- **`Error::InvalidClip` (gathered length too short)** — rev-9 per
  review-8 T3-A: the Diarizer **does NOT propagate** this error if
  it fires after the §5.8 fallback chain (clean → speaker-only)
  has already been exhausted. Instead: the activity is **skipped**
  (no embedding submitted, no `slot_to_cluster` entry, no
  `CollectedEmbedding` recorded), and the pump continues to the
  next activity. This matches pyannote's
  `speaker_verification.py:611-612` skip-and-continue behavior. The
  skipped activity's frames will produce zero per-cluster
  activation in §5.9 step B (no slot mapping → no contribution),
  so reconstruction degrades gracefully — those frames may be
  count=0 if no other speakers contribute.
- **All other errors** — `MaskShapeMismatch`, `NonFiniteInput`,
  `InferenceShapeMismatch`, ort errors, etc.: the error is
  **propagated** via `process_samples`'s `Result<(), Error>` return.
- Activities NOT YET PROCESSED in the same call are lost.
- The Diarizer's internal state is left consistent (audio buffer
  trimmed up to the failed activity; `Clusterer`,
  `collected_embeddings`, and per-frame stitching state not updated
  for the failed activity).
- **Open per-cluster runs are NOT auto-closed** on error — they
  continue and may be extended on the next successful pump. Caller
  can call `finish_stream` to flush.
- Caller can call `process_samples` again with the next chunk; the
  Diarizer continues from where it left off.

Same policy for `cluster.submit` errors. Same policy for
reconstruction internal errors (which should be unreachable per
§5.7's defensive-bounds-check contract).

## 6. Module layout

```
src/
├── lib.rs                                 # crate-level constants, re-exports, Diarizer
├── diarizer/
│   ├── mod.rs                             # pub re-exports (Diarizer, DiarizedSpan, CollectedEmbedding, DiarizerOptions)
│   ├── builder.rs                         # DiarizerBuilder + DiarizerOptions
│   ├── error.rs                           # diarizer::Error + InternalError (rev-5 §T4-A)
│   ├── span.rs                            # DiarizedSpan + CollectedEmbedding
│   ├── overlap.rs                         # exclude_overlap mask construction (rev-6 §5.8)
│   └── reconstruct.rs                     # per-frame stitching + speaker count + RLE (rev-6 §5.9-5.11)
├── segment/                               # v0.1.0 phase 1 + rev-6 v0.X bump:
│                                          #   - new Action::SpeakerScores variant
│                                          #   - mark Action #[non_exhaustive]
│                                          #   - pub(crate) Segmenter::peek_next_window_start()
├── embed/
│   ├── mod.rs                             # pub re-exports
│   ├── types.rs                           # Embedding, EmbeddingMeta, EmbeddingResult
│   ├── options.rs                         # constants, EmbedModelOptions
│   ├── error.rs
│   ├── fbank.rs                           # compute_fbank wrapper
│   ├── embedder.rs                        # sliding-window logic + embed_weighted
│   └── model.rs                           # cfg(ort) — EmbedModel
└── cluster/
    ├── mod.rs                             # pub re-exports
    ├── options.rs                         # ClusterOptions, OfflineClusterOptions
    ├── error.rs
    ├── online.rs                          # Clusterer
    ├── agglomerative.rs                   # offline HAC
    └── spectral.rs                        # offline spectral (uses nalgebra)
```

### Why split `diarizer/` into multiple files

The `Diarizer` is now meaningfully larger than rev-5's design (the
reconstruction logic is ~500 lines of state-machine code on top of
existing audio buffer + cluster orchestration). Splitting into
`builder.rs` / `span.rs` / `overlap.rs` / `reconstruct.rs` / `error.rs`
keeps each file focused on one responsibility:

- **`builder.rs`**: pure-data `DiarizerOptions` + `DiarizerBuilder`. No
  algorithm logic.
- **`span.rs`**: output types (`DiarizedSpan`, `CollectedEmbedding`).
  Stable; rarely changes.
- **`overlap.rs`**: `exclude_overlap` mask construction (frame
  binarization, clean-vs-fallback decision, sample-mask expansion).
  Pure functions; testable without ort.
- **`reconstruct.rs`**: per-frame per-cluster activation accumulator,
  speaker-count tracking, count-bounded argmax, per-cluster RLE.
  The largest single file (~400 lines); pure compute, testable
  without ort or audio.
- **`mod.rs`**: the `Diarizer` struct itself (state holder + pump
  glue). Pulls together the above modules. Pump is the only place
  that orchestrates segment + embed + cluster + reconstruct in
  sequence.
- **`error.rs`**: error types, kept separate so subordinate modules
  can `use crate::diarizer::error::Error` without circular deps.

## 7. Crate metadata

```toml
[dependencies]
# Existing
mediatime = "0.1"
thiserror = "2"
ort = { version = "2.0.0-rc.12", optional = true }

# New in v0.1.0 phase 2
kaldi-native-fbank = "0.1"
nalgebra = "0.34"
# K-means PRNG for spectral clustering (see §5.5 step 8 / §11.9).
# rand_chacha provides the pinned ChaCha8Rng implementation; rand
# provides convenience methods (`Rng::random_range`, distributions)
# on top of `rand_core`'s base `RngCore`/`SeedableRng` traits. Both
# at default-features=false to skip OS RNG / std features we don't
# need. (Implementer may drop `rand` if all sampling can be done
# directly through rand_core's traits, which rand_chacha depends on
# transitively. Spec-level recommendation: include both.)
rand = { version = "0.10", default-features = false }
rand_chacha = { version = "0.10", default-features = false }
```

All four new deps are unconditional. No new feature flags. The
`cluster-spectral` feature flag mentioned during brainstorming is not
introduced — `nalgebra` is needed for the default
`OfflineMethod::Spectral`, so gating it would break the default path.

**PRNG version pinning:** `rand_chacha = "0.10"` is pinned at the spec
level because the K-means spectral clustering output depends on the
exact byte sequence produced by `ChaCha8Rng`. A future bump to 0.11
(or any version that changes the streaming algorithm) would change
cluster labels for the same input — that is a breaking change for
users who pinned `dia` and expect stable output. See §11.9 for the
breaking-change policy on PRNG bumps.

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
- `Embedding::normalize_from(&[0.0; 256])` → `None` (degenerate; below `NORM_EPSILON`).
- `Embedding::normalize_from` round-trip: an already-normalized vector goes through with `||result|| ≈ 1.0` and per-component diff < `1e-6`.
- `Embedding::similarity` symmetry: `a.similarity(&b) == b.similarity(&a)` for any constructed embeddings.
- Free function parity: `cosine_similarity(&a, &b) == a.similarity(&b)` exactly.
- AllSilent: `embed_weighted` with all-zero `voice_probs` → `Error::AllSilent`.
- `embed` of a 32 000-sample (= exactly 2 s) clip uses 1 window, `windows_used = 1`.
- `embed` of a 48 000-sample (= 3 s) clip uses 2 windows (start 0 and start 16 000; start 16 000 is also the tail anchor `48 000 - 32 000`, so dedup leaves 2), `windows_used = 2`.
- `embed` of a 56 000-sample (= 3.5 s) clip uses 3 windows (start 0, start 16 000, tail anchor at start 24 000 = 56 000 - 32 000), `windows_used = 3`.

**`dia::cluster`:**
- All §5.5.1 edge cases.
- Antipodal centroid update: `Clusterer::submit([1,0,...])` then `submit([-1,0,...])` doesn't panic; second submission becomes new speaker (not assigned).
- Property: `RollingMean` accumulator magnitude bounded — for any sequence of `Clusterer::submit` calls under `RollingMean`, after `N` assignments any `SpeakerEntry::accumulator` satisfies `||accumulator||₂ <= N` (triangle inequality on a sum of `N` unit vectors). This is the load-bearing invariant — `cached_centroid`'s validity (`||cached_centroid|| > NORM_EPSILON` per §5.4) requires that `||accumulator||₂` not catastrophically zero out, and the upper bound complements it. Sanity check on `f32` precision behavior (no infinity, no NaN, no runaway growth).
- Degenerate-update lazy-cache: in `RollingMean`, submit `e1`, force-assign `−e1` (via threshold tweak so it's still assigned to the same speaker), assert `cached_centroid` stays at `e1` (the prior good value) rather than NaN.
- `OverflowStrategy::AssignClosest` does NOT change centroid (only `assignment_count`).
- `OverflowStrategy::Reject` returns `Error::TooManySpeakers`.
- Tie-breaking: two centroids equidistant from query → lowest speaker_id wins.
- K-means seed reproducibility: same seed → same labels.
- K-means seed default: same input twice with `seed = None` → same labels (= seed `0`).
- K-means convergence: input where labels stabilize at iteration 5 finishes in ≤ 6 iterations (assert via instrumented PRNG/iter counter).
- `cluster_offline` agglomerative-vs-spectral on a hand-built 6-embedding 2-cluster set produces the same labels.
- Eigengap K cap: synthetic `MAX_AUTO_SPEAKERS + 5` clusters → returns `MAX_AUTO_SPEAKERS`.
- N == 1 fast path: `cluster_offline(&[e], _)` returns `vec![0]` without invoking eigendecomp.
- N == 2 fast path: `cluster_offline(&[e, e], default)` returns `vec![0, 0]` (above threshold); `cluster_offline(&[e, -e], default)` returns `vec![0, 1]` (below threshold).
- Isolated-node precondition: 3 embeddings where one is orthogonal to both others → `Error::AllDissimilar` (rev-2 spec would have *not* caught this; rev-3 does via degree-matrix check).
- Zero-norm rejection: `cluster_offline(&[zero_embedding], _)` → `Error::DegenerateEmbedding`.

**`dia::Diarizer`:**
- All from rev 1.
- Buffer underflow on pathological activity range → `Error::Internal(InternalError::AudioBufferUnderflow { .. })` (no panic).
- Buffer overrun on pathological activity range → `Error::Internal(InternalError::AudioBufferOverrun { .. })` (no panic).
- `num_speakers()` and `speakers()` reflect Clusterer state mid-stream.

**`dia::Diarizer` reconstruction (rev-6):**

Pure-compute tests (no ort, synthetic raw_probs):
- **`exclude_overlap` clean-mask path**: window with two simultaneously-active slots, raw_probs designed so that frames 100-300 have *only* slot 0 active and frames 0-99 + 300-588 have both slots active. Activity for slot 0 covers all of [0, 588]. Verify `used_clean_mask = true`, `voice_probs` nonzero only at frames 100-300.
- **`exclude_overlap` fallback path**: same setup but only frames 200-201 have only slot 0 active (< MIN_CLEAN_FRAMES=3). Verify `used_clean_mask = false`, `voice_probs` matches the speaker-only mask (nonzero anywhere slot 0 is on).
- **Single-window single-speaker reconstruction**: synthetic raw_probs with slot 0 active in frames 100-200, zero elsewhere. After integration + finalization (post-finish_stream), exactly one DiarizedSpan with range = [frame_to_sample(100), frame_to_sample(200)], speaker_id = 0, is_new_speaker = true.
- **Two-window stitching of one speaker turn**: window 0 (frames 0-588 abs) has slot 0 in frames 200-588; window 1 (frames 100-688 abs) has slot 0 in frames 100-388 (= abs 200-488). After both integrate (overlap-add SUM correctly merges scores in abs frames 200-488), one DiarizedSpan with range = [frame_to_sample(200), frame_to_sample(589)] (window 0's end of run; window 1's slot maps to same cluster).
- **Two-speaker non-overlapping turns**: slot 0 active in window 0 only, slot 1 active in window 1 only. Two separate DiarizedSpans, distinct speaker_ids. Both `is_new_speaker = true`.
- **Two-speaker overlapping turns with count=2**: both slots simultaneously active in mid-window frames; count = 2 at those frames. Verify two overlapping DiarizedSpans (range overlap is allowed; output is a list per closed run).
- **Count clamps to count=1 when only one cluster passes**: synthetic case where two slots are active but get clustered to the *same* cluster (e.g., similar embeddings). Verify only one DiarizedSpan emits, not two, even though count was 2.
- **Speaker-count warm-up trimming**: scores[slot=0][frame=0..58] = 0.9 (in pyannote-defined warm-up); scores[slot=0][frame=58..588] = 0.0. Speaker count at abs_frame=0 should be 0 (warm-up frames excluded), not 1.
- **Tie-break**: two clusters with identical activations at a frame; smaller cluster_id wins. Verify deterministic across two runs.
- **`is_new_speaker` flag**: speaker_id 0 emits twice (first turn, then second turn); first emission has `is_new_speaker = true`, second has `false`.
- **`finish_stream` flushes open run**: speaker active to end-of-stream, no closing frame; `finish_stream` emits the open run before returning.
- **`clear()` discards open runs**: after `clear()`, `pending_inferences == 0`, `buffered_frames == 0`, no spans emitted.
- **`buffered_frames()` introspection**: during a pump, before finalization completes, equals the number of frames in the per-frame accumulator. After `finish_stream`, returns 0.
- **Slot-to-cluster mapping correctness**: simulate two windows where the same speaker appears in slot 0 of window 0 and slot 1 of window 1 (clustering puts them in the same cluster). Verify reconstruction max-collapses correctly across the slot reassignment.
- **Inactive-slot skip**: a slot with raw_probs all < binarize_threshold should NOT appear in slot_to_cluster (no embedding submitted), so reconstruct integration skips it (matches pyannote's `inactive_speakers → -2` throwaway).

VAD-input variable-length tests (rev-7, real `Segmenter` + synthetic inferencer):
- **Empty push is no-op**: `process_samples(_, _, &[], _)` returns `Ok(())`, no callbacks invoked, no state advances. `total_samples_pushed` unchanged.
- **Sub-window single push (0.5 s) without finish_stream**: `process_samples` returns `Ok(())`. `pending_inferences == 0`, `buffered_samples == 8000`, `buffered_frames == 0`. No DiarizedSpans emitted yet.
- **Sub-window single push followed by finish_stream**: tail-anchor schedules at `max(0, total - WINDOW) = 0`, padded window infers, real-audio activities cluster, DiarizedSpan(s) emit covering only the real-audio range (zero-padded frames produce count=0 → no spans).
- **Sub-MIN_CLIP_SAMPLES single push (e.g., 10 ms = 160 samples) followed by finish_stream**: padded window's only "real" frames are < 1 frame's worth (each frame is ≈ 271.65 samples ≈ 16.97 ms); activities, if any, would have ranges < MIN_CLIP_SAMPLES → embed_model returns `Error::InvalidClip`. Verify the error propagates out of `finish_stream` rather than silently producing garbage.
- **Multiple short pushes + finish_stream**: `process_samples(0.5 s) → process_samples(0.8 s) → process_samples(1.2 s) → finish_stream`. Total 2.5 s. Tail-anchor at 0, padded window infers, all activities have absolute-sample ranges within `[0, 2.5 s)`. Verify `total_samples_pushed == 2.5 * 16_000 == 40_000`.
- **Multiple medium pushes**: `process_samples(3 s) × 5 → finish_stream`. Total 15 s. Two regular windows + tail-anchor (or no tail if 15 s ≥ 10 s and last full window is at start = 5 s covering [5 s, 15 s)). Verify `pending_inferences` decrements as expected.
- **Long single push (60 s) without finish_stream**: many windows process during the single `process_samples` call. Verify `total_samples_pushed == 60 * 16_000`, `pending_inferences == 0` after pump completes (synchronous), and DiarizedSpans emit for already-finalized turns. Some frames remain un-finalized (≈ last 10 s); `buffered_frames > 0`.
- **Mixed lengths**: `process_samples` calls of `[0.5 s, 3 s, 0.2 s, 10 s, 5 s]` interleaved → no panics, no errors, total accumulates correctly. `finish_stream` flushes everything.
- **Push that brings total exactly to WINDOW_SAMPLES boundary**: push `10 s` exactly (160_000 samples). One window schedules at start = 0 covering `[0, 10 s)`. Verify activities + scores + integration occur. `pending_inferences` after pump completes = 0.
- **Consecutive `finish_stream + clear + process_samples` cycle**: simulate caller doing per-VAD-range diarization. After `finish_stream` flushes, `clear()` resets all state including `total_samples_pushed`. New `process_samples` starts fresh from sample 0. Verify speaker IDs reset to 0 (rev-7 confirmed by §4.4 `clear()` rustdoc).
- **`total_samples_pushed` monotonicity**: across many `process_samples` calls, `total_samples_pushed` is monotonically non-decreasing and equals exactly `sum(samples.len() for each push)`. Reset to 0 only by `clear()`.
- **Cross-VAD-range activity detection** (caller-driven, not a dia property): pseudo-test that pushes two separated VAD ranges as one stream and verifies that a single dia activity range can span both. Documented as expected behavior; caller's responsibility per §11.12.

DiarizedSpan quality-metric tests (rev-7):
- **`average_activation` of single-window-isolated turn**: synthetic raw_probs with one slot at 1.0 across all 589 frames; one window, no overlap. Activation per frame = 1.0 (single-chunk count). After RLE, `average_activation ≈ 1.0`.
- **`average_activation` of multi-window turn**: 4 overlapping windows, each contributing the same speaker at 0.8 average. Activation_sum_normalized per frame = 0.8 (after dividing by chunk_count = 4). After RLE, `average_activation ≈ 0.8`.
- **`activity_count` matches contributing activities**: synthetic 3-window run, each window contributes 1 activity for the same cluster. After run closes, `activity_count == 3`.
- **`clean_mask_fraction == 1.0`**: all contributing activities used clean mask (no overlap).
- **`clean_mask_fraction == 0.5`**: 2 of 4 contributing activities fell back to speaker-only mask (heavy-overlap regions).
- **`clean_mask_fraction == 0.0` defensible default**: edge case where a run somehow has zero contributing activities (shouldn't happen with correct integration; defensive default for divide-by-zero).

End-to-end ort-gated tests (`#[ignore]` smoke tests, real model):
- 30-second single-speaker clip → exactly one DiarizedSpan, speaker_id = 0.
- 30-second two-speaker alternating clip (no overlap) → 2-speaker output, alternating DiarizedSpans.
- Pyannote parity test: 5-minute multi-speaker held-out clip; run pyannote.audio reference (Python, separate harness) and dia. Compute DER (diarization error rate) using a standard scorer. **Rev-8: relaxed target to DER ≤ 10 % absolute** on a clean 2-3 speaker meeting clip (rev-7 had ≤ 5 %, which review-7 T3-I correctly flagged as aspirational given documented divergences: gather-mechanism difference §5.8, online-vs-batch clustering, default-clusterer difference §5.5 vs pyannote VBx, K-means seed source). 5 % remains a stretch goal. (See §15 #43 for the parity-test harness setup.)

### Integration tests (gated)

As in rev 1, plus the pyannote parity smoke test described above.

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
  clustering uses `rand_chacha::ChaCha8Rng` (pinned by version in §7;
  swap is a major-version-bump change), seeded from
  `OfflineClusterOptions::seed` (default `0`). See §11.9 and §5.5 step 8.

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

### 11.5 Diarizer audio buffer is bounded at `dia::segment::WINDOW_SAMPLES`

The audio buffer is trimmed to retain the last
`dia::segment::WINDOW_SAMPLES` (= 160 000 samples = 10 s) below
`total_samples_pushed`. **Steady-state memory: 640 KB** (160 000 × 4
bytes) plus `VecDeque` overhead and the `CollectedEmbedding` vec.

**Why the segment window, not the embed window:** see §5.7. Briefly:
the segment's tail-anchored sliding window can place the start of a
future window up to `dia::segment::WINDOW_SAMPLES` behind
`total_samples_pushed`, and the activity it produces may need any
sample within that range. Trimming below that bound makes a future
activity's audio range unrecoverable.

**Rev-history correction:** rev 2 mistakenly tightened this bound to
`EMBED_WINDOW_SAMPLES` (32 000 = 2 s = 128 KB), claiming the segment's
tail-anchor reach-back was "irrelevant because by the time
`process_samples` returns, the segment has already produced any tail
activity inline." That reasoning was wrong — see §5.7's "Why the
segment window size, not the embed window size?" for the corrected
analysis. Rev 1's 640 KB figure was correct (with the wrong bound name
attached); rev 3 restores the correct figure under the correct name.

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

`OfflineClusterOptions::seed`:
- `None` (default): the literal constant `0` is used as the K-means
  PRNG seed. Same input → same labels, deterministic across runs.
- `Some(s)`: `s` is used directly.

**The PRNG is pinned to `rand_chacha::ChaCha8Rng`.** Constructed via
`ChaCha8Rng::seed_from_u64(seed)`. `rand`'s `thread_rng` is **never**
used (seeded from OS entropy → non-deterministic).

**Breaking-change policy:** swapping the PRNG type, changing the
seed-derivation function, or upgrading `rand_chacha` to a version
that alters its keystream is a user-observable behavioral change
(same input → different cluster labels). All three require a major
version bump on dia. The dia team commits to NOT make this change
between minor versions.

Rev-3 change from rev 2: rev 2 proposed deriving the default seed
from a hash of input bytes (FxHash). Rev 3 uses the literal `0`
instead — same determinism, no public commitment to a specific hash
function.

Rev-4 change: pinned `rand_chacha::ChaCha8Rng` exactly (rev 3 said
"or equivalent reproducible PRNG," which left a backdoor for
breaking output stability).

### 11.10 OverflowStrategy semantics

(Documented in §5.4.)

### 11.11 Argmax tie-breaking

(Documented in §5.4 — "lowest-index speaker wins.")

### 11.13 Reconstruction state eviction policy (rev-8, presented out of numerical order — §11.12 below)

The Diarizer's reconstruction state machine (§5.9-§5.11) has three
per-window-id metadata maps that must be evicted to bound memory on
long sessions:

- `slot_to_cluster: BTreeMap<(WindowId, u8), u64>`
- `activity_clean_flags: HashMap<(WindowId, u8), bool>`
- `window_starts: HashMap<WindowId, u64>`

**Eviction rule.** After each `emit_finalized_frames()` call returns,
for each `WindowId w` with `window_starts[w] = window_start`:

```
let last_frame = frame_index_of(window_start) + FRAMES_PER_WINDOW as u64;
let no_open_run_references_w = open_runs.values().all(|run| {
    run.contributing_activities.iter().all(|(wid, _)| *wid != w)
});
if base_frame >= last_frame && no_open_run_references_w {
    slot_to_cluster.retain(|(wid, _), _| *wid != w);
    activity_clean_flags.retain(|(wid, _), _| *wid != w);
    window_starts.remove(&w);
}
```

In words: a window's metadata can be dropped once (a) all of its
frames have finalized — none of them can be in any future
contribution to a cluster's per-frame run since they're already past
the finalization boundary — AND (b) no currently-open per-cluster
run still has a contributing-activity entry referencing this window
(open runs may extend forward and emit a `DiarizedSpan` with
`activity_count` that includes the original (window, slot) pair).

**Memory bound for a 60-min session at default `step_samples = 40_000`:**
- Number of windows ever scheduled: ≈ `60 * 60 * 16_000 / 40_000 = 1440`
- Without eviction: `1440 windows × (3 slots + 1 window_start)` ≈ ~10 KB total — small but unbounded.
- With eviction: at most `~10 windows` of metadata (one window's worth ≈ 4 frames overlap × `step_samples / WINDOW_SAMPLES ratio`, plus open-run grace) ≈ ~80 bytes.
- The savings matter more for multi-hour streams; the eviction policy is correctness-relevant once an open run spans a long stretch of finalized windows.

**Open-run-aware aspect** (the `no_open_run_references_w` check): if a
speaker's run is open across windows W=10, W=11, W=12, ..., W=20 (an
extended turn), and W=10..W=15 have finalized but W=16..W=20 are still
contributing, we keep the metadata for W=10..W=15 even though their
frames have finalized — because `DiarizedSpan::activity_count` (§4.4)
counts all contributing (window, slot) pairs across the entire run,
and we'd lose count information if we evicted early. Eviction
happens at the moment the run closes (or the moment a window's last
contributing activity has been emitted into a closed span).

### 11.12 Variable-length VAD-filtered input contract (rev-7)

**The Diarizer accepts pushes of any length** via `process_samples`,
including:
- Empty pushes (`samples.len() == 0`): no-op, returns immediately.
- Sub-window pushes (e.g., 0.5 s of VAD speech): samples are buffered;
  no `Action::NeedsInference` is scheduled until enough samples
  accumulate. Caller can poll `pending_inferences() == 0` and
  `buffered_samples() > 0` to detect this state.
- Single-window pushes (~10 s exact): one window schedules,
  inference runs, activities + scores emit, integration happens, some
  frames finalize, possibly a `DiarizedSpan` emits.
- Multi-window pushes (≫ 10 s): many windows schedule in sequence;
  reconstruction integrates them in order.
- Whole-stream pushes (60 minutes pushed at once): equivalent to the
  multi-window case, just larger; memory budget per §11.5 (≈ 640 KB
  audio buffer + ≈ 64 KB per-frame accumulator + ≈ 1 KB per
  `CollectedEmbedding`).

**Lazy window scheduling.** Segment schedules a window with start
`k * step_samples` only when `total_samples_pushed ≥ k * step + WINDOW
= k * step + 160 000`. So a caller pushing only 7 seconds of VAD audio
sees zero windows scheduled until either (a) more samples arrive or
(b) `finish_stream` is called (which schedules the segment's
tail-anchor at `total - WINDOW`, padding with zeros if needed).

**No special-casing for clip lengths.** The reconstruction algorithm
(§5.9–§5.11) is frame-driven, not window-driven. A 0.5-s VAD push
followed by `finish_stream` produces:
1. Segment's tail-anchor schedules at `max(0, total - WINDOW) = 0`,
   covering `[0, 10 s)` with 9.5 s of zero-padding.
2. Inference runs on the padded window. Voice/speaker probabilities
   are ~0 for the zero-padded frames (model output on silence).
3. Reconstruction integrates the window. Activations for zero-padded
   frames are ~0 → no cluster is in the top-`count` there →
   `count[f] == 0` → no DiarizedSpan emits for those frames.
4. Activities for the real 0.5 s emit normally (one or more).

**Activities can span original-VAD-range boundaries.** The segment
treats all input as contiguous speech. If the caller fed
`[VAD range A: 0–5 s of original time], [VAD range B: 8–12 s of
original time]` as one stream, dia sees 9 s of contiguous speech.
A segment activity can have a range that crosses the VAD-range
boundary in dia's input timeline (e.g., dia samples
`[60 000, 100 000)` = original-time `[3.75–5 s, 8–9.25 s]`). **The
caller is responsible for detecting and handling this** if it matters
for downstream — they have the VAD-to-original-time mapping; we don't.

**Recommended caller pattern for VAD-aware processing:**

```rust
let mut dia_to_original: Vec<(u64, u64)> = Vec::new(); // sorted by dia_offset
for vad_range in vad_layer.ranges_of_speech(audio) {
    let dia_offset_at_push = diarizer.total_samples_pushed();   // §4.4 accessor (rev-7)
    dia_to_original.push((dia_offset_at_push, vad_range.start));
    let speech_samples = audio.slice(vad_range);
    diarizer.process_samples(&mut seg_model, &mut embed_model, speech_samples,
        |span| {
            let original_range = map_dia_range_to_original(
                span.range(), &dia_to_original
            );
            store(original_range, span.speaker_id(), span.average_activation(), ...);
        }
    )?;
}
diarizer.finish_stream(&mut seg_model, &mut embed_model, |span| {
    // Same mapping as above.
    let original_range = map_dia_range_to_original(span.range(), &dia_to_original);
    store(...);
})?;
```

The `map_dia_range_to_original` helper is application code, not
provided by dia. (Action item §15 #50 considers whether dia should
ship a `dia::utils::TimelineMap` helper for this — open question;
might bloat the surface for limited gain.)

**`Diarizer` introspection accessors useful for VAD-aware callers:**
- `total_samples_pushed() -> u64`: cumulative count of samples ever
  pushed (rev-7 addition; never decremented except by `clear()`).
- `pending_inferences() -> usize`: how many windows are queued for
  inference. Non-zero means more spans will emit on the next pump.
- `buffered_samples() -> usize`: audio still in the rolling buffer.
- `buffered_frames() -> usize`: un-finalized frames in the per-frame
  accumulator (rev-6 addition).
- `num_speakers() -> usize`: distinct global cluster IDs assigned so far.
- `speakers() -> Vec<SpeakerCentroid>`: per-cluster summary (centroid
  + assignment_count) for storage / inspection at any point.
- `collected_embeddings() -> &[CollectedEmbedding]`: per-(window, slot)
  granularity output, kept across the session (until `clear_collected`).

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
- Spectral precondition: `Error::AllDissimilar` is returned if any
  degree `D_ii < NORM_EPSILON` (rev-3 tightened from rev-2's narrower
  "all-of-A < eps" check; isolated nodes are now caught even when
  the rest of the affinity graph is non-degenerate).
- K-means uses a deterministic seed: explicit `seed: Option<u64>` on
  `OfflineClusterOptions`; default is the literal `0` (rev-3
  simplified rev-2's hash-of-input-bytes proposal). PRNG pinned to
  `rand_chacha::ChaCha8Rng` (rev-4 tightened from rev-3's "or
  equivalent reproducible PRNG").
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

### Rev-6 / rev-7 / rev-8 additions to the decision log

- **Reconstruction adoption (rev-6):** match `pyannote.audio`'s
  `SpeakerDiarization` pipeline shape (segmentation → exclude-overlap
  embedding → clustering → reconstruct → annotate). Reasoning: pyannote
  is the published / validated production reference; matching its
  algorithm gives us a defensible quality baseline. Streaming
  adaptation (per-frame finalization driven by segment's window
  schedule) is novel.
- **`Action::SpeakerScores` segment v0.X bump (rev-6):** segment must
  expose per-window per-speaker raw probabilities for downstream
  reconstruction. Implemented as a new `Action` variant emitted from
  `push_inference` alongside `Activity`. `Action` marked
  `#[non_exhaustive]` to make future additions non-breaking.
- **`exclude_overlap` mechanism (rev-6 wrong, rev-8 corrected):**
  rev-6/7 routed it through `embed_weighted` — a no-op for clips ≤ 2 s
  (review-7 T1-A caught this). Rev-8 introduces `embed_masked` (gather
  retained samples + zero-pad to 2 s + embed normally), matching
  pyannote's `ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__` masked
  path. Documented divergence: pyannote gathers at frame-rate post-fbank;
  we gather at sample-rate pre-fbank. Equivalent up to fbank-window
  boundary alignment.
- **DiarizedSpan field set (rev-7):** simplified to `(range, speaker_id,
  is_new_speaker)` plus three quality metrics (`average_activation`,
  `activity_count`, `clean_mask_fraction`) for downstream storage and
  analysis. Window-local concepts (similarity, slot) live on
  `CollectedEmbedding` instead.
- **VAD prerequisite formalization (rev-7):** Diarizer accepts
  variable-length pushes (`samples.len()` from 0 to whole-stream). The
  rev-6 algorithm already handled this; rev-7 made it explicit in
  §11.12 with a recommended caller pattern for VAD-to-original timeline
  mapping.
- **Frame rate ≈ 58.9 fps (rev-8 correction):** rev-6/7 prose said
  "100 fps / 10 ms / 160 samples" repeatedly. Verified by reading
  shipped `dia::segment::stitch::frame_to_sample`: `WINDOW_SAMPLES *
  frame_idx / FRAMES_PER_WINDOW = 160_000 * frame_idx / 589` ≈ 271.65
  samples/frame ≈ 16.97 ms/frame ≈ 58.9 fps. Rev-8 corrects throughout.
- **Argmax tie-break: smallest cluster_id wins (rev-6 / rev-8 highlight
  per review-7 T4-B):** pyannote uses `np.argsort(-activations)` which
  has implementation-defined tie behavior (typically quicksort,
  non-deterministic for ties). We pick the deterministic
  smaller-cluster_id tie-break to make streaming output byte-stable
  across runs and platforms — a deliberate IMPROVEMENT over pyannote,
  not a divergence.
- **Activation vs speaker-count chunk-counts split (rev-8 per
  review-7 T2-A):** pyannote's `Inference.aggregate(skip_average=True)`
  has its own un-trimmed overlapping_chunk_count; `speaker_count` has
  a separate warm-up-trimmed count. Rev-7 conflated them into a single
  `FrameCount.chunk_count`; rev-8 splits into `activation_chunk_count`
  (no trim, used for `average_activation` normalization in §5.11) and
  `count_chunk_count` (warm-up trimmed, used for speaker-count
  rounding in §5.10).
- **Reconstruction state eviction (rev-8 per review-7 T3-J):** the
  three per-window-id metadata maps (`slot_to_cluster`,
  `activity_clean_flags`, `window_starts`) are evicted once a window's
  last frame has finalized AND no open per-cluster run still references
  any (window, slot) entry. Documented in §11.13. Bounds memory on
  long sessions.
- **DER target ≤ 10 % (rev-8 per review-7 T3-I):** rev-7 had ≤ 5 %
  absolute on a 5-min held-out clip. Reviewer correctly observed this
  is aspirational given documented divergences. Rev-8 relaxes to
  ≤ 10 % for the CI gate; 5 % remains a stretch goal for
  numerical-parity-quality work later.

## 13. Revision history

- **Revision 1** (2026-04-26): initial spec from brainstorming.
- **Revision 2** (2026-04-26): incorporates first adversarial-review
  feedback. Critical algorithmic fixes: centroid math (unnormalized
  accumulator), Ward removal, eigengap K cap, all-zero affinity
  precondition, K-means seed, OverflowStrategy default flip +
  no-update-on-forced-assign. API consistency fixes:
  `EMBED_WINDOW_SAMPLES` rename, `Option<f32>` for similarity,
  `CollectedEmbedding` struct, builder API cleanup. Documentation:
  Python center-crop divergence callout, edge-case enumeration,
  Diarizer ownership rationale corrected, sibling comparison table,
  rejected findings (§14), action list (§15).
  - **Note retracted in rev 3:** rev 2 also tightened the Diarizer
    audio buffer bound from `WINDOW_SAMPLES` (segment-style 640 KB) to
    `EMBED_WINDOW_SAMPLES` (~128 KB), claiming the segment's
    tail-anchor reach-back was irrelevant. That tightening was
    incorrect; it underflows on real workloads. Rev 3 restores the
    `dia::segment::WINDOW_SAMPLES` bound (640 KB).
- **Revision 3** (2026-04-26): incorporates second adversarial-review
  feedback.
  - **Critical fix (T1-A):** §5.7 / §11.5 audio buffer trim bound
    reverted from `EMBED_WINDOW_SAMPLES` to
    `dia::segment::WINDOW_SAMPLES`; steady-state memory restored to
    640 KB. Adds explanatory rationale on the segment tail-anchor
    reach-back.
  - **Algorithmic fix (T1-B):** §5.5 spectral clustering precondition
    tightened from "all-of-A < eps" to "any D_ii < eps" (catches
    isolated nodes, not just the all-zero matrix). Built on top of an
    explicit degree-matrix computation step.
  - **Spectral fast paths:** §5.5 / §5.6 add explicit N == 1 and
    N == 2 short-circuit branches before any matrix work; eigengap
    is degenerate at N=2.
  - **Validation timing:** §5.5 / §5.6 explicitly state validation
    runs first (before matrix computation), with a shared helper.
  - **K-means seed source + convergence:** §5.5 step 8 specifies
    `rand_chacha::ChaCha8Rng` (or equivalent reproducible PRNG;
    `thread_rng` forbidden), and lists the two termination conditions
    (assignment-unchanged convergence, 100-iteration cap) in order.
    §11.9 updated: default seed is the literal `0`, not a hash of
    input bytes.
  - **Cached centroid semantics (§5.4):** internal `SpeakerEntry`
    grows a `cached_centroid` field; degenerate-update behavior
    documented (centroid stays at last-known-good value).
  - **`Diarizer::clear()` rustdoc (§4.4):** enumerates exactly what
    is and isn't reset, including the deliberate non-clearing of
    `collected_embeddings`.
  - **Builder parity:** `EmbedModelOptions`, `ClusterOptions`,
    `OfflineClusterOptions`, `DiarizerOptions` all gain `set_*`
    mutating methods alongside the consuming `with_*` builders, for
    parity with `silero` and `dia::segment`.
  - **`Embedding` invariant:** rustdoc on `Embedding` makes the
    L2-norm > NORM_EPSILON invariant explicit and points to
    `normalize_from` as the only constructor that can break it
    (returns `None` instead).
  - **`DegenerateEmbedding` error variant:** new variant in BOTH
    `dia::embed::Error` and `dia::cluster::Error` for
    zero-norm/near-zero-norm cases — rev 2 (in `dia::cluster`) and the
    initial design (in `dia::embed`) both misnamed these as
    `NonFiniteInput` (zero IS finite).
  - **Misc:** §1 wording (`findit-speaker-embedding` does not provide
    clustering — clustering layer is "novel," not "deferred"), free
    function `cosine_similarity` rationale.
  - **Agglomerative distance clamped to [0, 1] (NEW in rev 3, missed
    from the rev-3 entry above and surfaced by review 3):** §5.6
    step 1 changed from `D_ij = 1 - cos_similarity(e_i, e_j)` (range
    `[0, 2]`) to `D_ij = 1 - max(0, e_i · e_j)` (range `[0, 1]`,
    ReLU-clamped to match spectral's affinity convention). Affects
    cluster output when any pair has negative cosine similarity:
    rev-2 saw `dist = 1.5` for cos = -0.5; rev-3 sees `dist = 1.0`.
    Threshold semantics (`stop if dist >= 1 - threshold`) are
    unchanged in form but the underlying distance space is now
    bounded — pre-existing implementations following rev-2's formula
    would silently produce different clusters when "upgraded" to
    rev 3.

    **Per-linkage impact:**
    - `Linkage::Single` (smallest distance wins): essentially
      unaffected. Single linkage always merges the closest pair, and
      clamping a *larger* distance from 1.5 to 1.0 doesn't change
      which pair is closest (the closest-pair distance is `<= 1.0`
      in both rev-2 and rev-3 because some non-anti-correlated pair
      always exists in real-world inputs).
    - `Linkage::Complete` (largest distance wins): **affected.** The
      max-of-pair-distances becomes `<= 1.0` in rev-3, so merge
      ordering downstream of any negatively-correlated pair shifts.
    - `Linkage::Average` (Lance-Williams weighted mean): **affected.**
      Cluster-vs-cluster average distance includes contributions
      from negatively-correlated pairs; clamping shifts the average,
      which shifts merge ordering.

    Net: any caller using Complete or Average linkage on inputs with
    negatively-correlated embeddings will see different cluster
    output between rev-2 and rev-3 of this spec. Single-linkage
    callers are unaffected. The rev-3 change is the right one
    (matches spectral's affinity convention; `[0, 1]` is the
    bounded-distance convention used in the speaker-clustering
    literature) but is documented here so the divergence isn't a
    silent surprise.
- **Revision 4** (2026-04-26): incorporates third adversarial-review
  feedback. Implementation-blocking gaps closed.
  - **§7 dependencies (review-3 item 42):** added `rand = "0.10"` and
    `rand_chacha = "0.10"` (with `default-features = false`).
    Without these, §5.5 step 8's pinned PRNG was un-implementable
    from §7 alone. (Reviewer suggested "0.9", but the current
    crates.io versions for both are 0.10 as of 2026-04-26 — used
    those.)
  - **K-means++ algorithm pinned (review-3 item 43):** §5.5 step 8
    now spells out the Arthur & Vassilvitskii 2007 D²-weighted seeding
    algorithm explicitly (uniform first centroid; subsequent centroids
    sampled with probability ∝ squared distance to nearest existing
    centroid; tie-break via smallest-index cumulative-mass crossing).
    Without this, two implementations could produce different cluster
    labels from the same input + seed (k-means|| variant,
    farthest-first traversal, uniform initialization, etc.).
  - **PRNG pinning tightened (review-3 item 44):** §5.5 / §11.9 / §10
    now pin `rand_chacha::ChaCha8Rng` exactly, drop "or equivalent
    reproducible PRNG," and add an explicit breaking-change-policy
    statement (PRNG swap or `rand_chacha` major-version bump that
    changes keystream → major version bump on dia).
  - **§13 rev-3 entry corrected (review-3 item 45):** added the
    missing agglomerative distance-clamp note above (the change
    shipped in rev 3 but wasn't in rev 3's history bullet list).
  - **§11.9 attribution fix (review-3 item 46):** "rev-3 change from
    rev 2" not "from rev 1" (the hash-of-input-bytes seed proposal
    was introduced in rev 2, not rev 1).
  - **§14 count fix (review-3 item 47):** review-2 paragraph
    rephrased to avoid undercount — "all action items in review 2
    are accepted" instead of "twelve."
  - **§1 wording (review-3 item 48):** "improves the long-clip
    handling" → "with one deliberate divergence: long-clip handling
    uses sliding-window mean (see §5.1 for the trade-off)." Removes
    the value-judgment framing; matches §5.1's "Important divergence"
    callout.
  - **§9 property test (review-3 item 49):** added "RollingMean
    accumulator bounded magnitude" property test (assert
    `||accumulator|| <= N` after `N` submissions of unit-norm
    embeddings).
  - **§5.4 cached_centroid invariant comment (T3-B):** added a
    one-line comment on the field documenting that
    `||cached_centroid|| > NORM_EPSILON` always holds (initialized
    from a unit vector; degenerate updates leave it unchanged).
  - **§15 float32 precision item (T3-C):** added action item to
    investigate `f32` precision impact on `RollingMean` for sessions
    with `N > 10⁶` assignments (not blocking v0.1.0; flagged for
    v0.1.1+).
- **Revision 5** (2026-04-26): incorporates fourth adversarial-review
  feedback. Two implementation-blocking polish items closed; broader
  byte-determinism specification.
  - **nalgebra 0.33 → 0.34 (review-4 T1-A):** verified via
    `cargo info nalgebra` that 0.34.2 is current; the rev-4
    "0.33" pin was stale.
  - **K-means++ byte-determinism pinned (review-4 T1-B):** §5.5
    step 8 now spells out exact rand-0.10 calls for all four
    sources of byte ambiguity: step-1 first-centroid sampling
    (`Uniform::new(0, N).unwrap().sample`); step-2a min reduction
    order (left-to-right f64); step-2b "uniform from not-yet-chosen"
    (compacted-index `Vec` + `Uniform::new`); step-2c f64 sampling
    (`rng.random::<f64>()` = `StandardUniform`, half-open [0, 1));
    step-2c "crosses" semantics (strict `>`, not `>=`); step-2c
    threshold computation (`u * S` rather than `D_j / S` to avoid
    division-rounding). These four ambiguities, left in rev-4 prose,
    would have produced different cluster labels across independent
    correct implementations of the spec. Pinning them removes that.
  - **SpeakerCentroid accessor block (review-4 T1-C):** added the
    missing `impl SpeakerCentroid { ... }`. The struct had private
    fields and no accessor methods, so `Clusterer::speakers() ->
    Vec<SpeakerCentroid>` returned read-only-but-unreadable values.
    Also converted `ClusterAssignment` from `pub` fields to private
    + accessors, for visibility consistency within the cluster
    module (review-4 T3-A).
  - **§5.7 / §11.5 code-trace verification (review-4 T2-A):** the
    `total_samples_pushed - dia::segment::WINDOW_SAMPLES` trim bound
    is now justified by line-number references into the shipped
    `dia::segment::segmenter.rs` and `window.rs`. Pushed back on
    the reviewer's "could be `step_samples + WINDOW_SAMPLES = 200 000`"
    claim — code trace shows the bound is exactly `WINDOW_SAMPLES`,
    not `WINDOW_SAMPLES + step_samples`.
  - **§13 vs §15 numbering disambiguation (review-4 T2-B):**
    rev-4's §13 entries used "(item NN)" referring to review-3
    finding numbers; §15 used "#NN" as its own action-item local
    namespace. Rev-5 §13 entries explicitly say "(review-3 item NN)"
    to disambiguate. §15 #43 (which mislabeled itself as "rev-3
    review item 49") was removed — review-3 item 49 was the
    property-test request, already addressed in §9 + §13; the
    "cross-platform K-means determinism CI" idea was a duplicate
    of §15 #32 anyway (review-4 T3-C).
  - **§9 / §13 property-test wording aligned (review-4 T2-C):** both
    sections now describe the L2-norm bound `||accumulator||₂ <= N`
    (the load-bearing invariant for `cached_centroid` validity), not
    "per-component absolute value <= N" (which is also true but not
    the load-bearing one).
  - **§15 #43 kaldi-native-fbank pre-impl spike (review-4 T2-D):**
    added a HIGH-severity gating action item to verify
    `kaldi-native-fbank = "0.1"` (brand-new, single-version,
    ~1.4k downloads as of 2026-04-26) against `torchaudio.compliance.
    kaldi.fbank` BEFORE writing the embed module. Falls back to
    `knf-rs` if the spike fails.
  - **§13 distance-clamp linkage-impact nuance (review-4 T3-D):**
    the rev-3 distance-clamp note now spells out per-linkage impact
    (Single unaffected; Complete and Average affected on inputs with
    negatively-correlated pairs). Polish, not a correctness change.
  - **`Diarizer::Error::Internal(InternalError::AudioBufferUnderflow
    { .. })` variant (review-4 T4-A):** rev-1..rev-4 returned
    `Error::Embed(Error::InvalidClip { len: 0, min: ... })` for
    audio-buffer underflow on a pathological activity. Rev-5 splits
    this into a dedicated `Error::Internal` variant tree with two
    sub-variants (`AudioBufferUnderflow`, `AudioBufferOverrun`) so
    callers debugging real `InvalidClip` errors aren't misled by
    sentinels.
- **Revision 6** (2026-04-26): scope expansion to match
  `pyannote.audio`'s reconstruction pipeline. Per the user's
  request — "match pyannote audio behavior as close as possible
  because it has been verified in prod." Diarizer output semantics
  change: spans are stitched per closed speaker turn, not per
  (window, slot) detection.
  - **`dia::segment` v0.X bump:** new `Action::SpeakerScores { id,
    window_start, raw_probs }` variant emitted from `push_inference`
    alongside `Action::Activity`. Carries the per-frame per-speaker
    raw probabilities used for downstream stitching. `Action` marked
    `#[non_exhaustive]`. New `pub(crate) Segmenter::peek_next_window_start()`
    accessor for the Diarizer's finalization-boundary computation.
  - **`exclude_overlap` embedding (§5.8, new):** matches
    `pyannote/audio/pipelines/speaker_diarization.py:375-425`. Per
    activity, build a per-frame "clean" mask (only this speaker
    active, no overlap), expand to per-sample, fall back to
    speaker-only mask if clean mask < `MIN_CLEAN_FRAMES` (≈ 30 ms).
    Mask is fed to existing `embed_weighted`. Documented divergence
    from pyannote: pyannote's mask is consumed inside the embedding
    model (PyTorch); ours is at the sliding-window aggregation layer
    (because WeSpeaker ONNX export doesn't accept a mask input).
    `CollectedEmbedding::used_clean_mask` flag for observability.
  - **Per-frame per-cluster activation stitching (§5.9, new):**
    matches `reconstruct` (collapse-by-max within cluster, per
    chunk) plus `Inference.aggregate(skip_average=True)` (overlap-add
    SUM across chunks). State machine in `dia::diarizer::reconstruct`.
  - **Speaker-count tracking (§5.10, new):** matches `speaker_count`
    with `warm_up=(0.1, 0.1)`. Per-frame overlap-add MEAN of
    binarized speaker counts, rounded. Hardcoded warm-up matches
    pyannote default (configurability deferred to v0.1.1).
  - **Count-bounded argmax + per-cluster RLE (§5.11, new):** matches
    `to_diarization` + `to_annotation`. As frames finalize, top-`count`
    clusters by activation are picked; per-cluster open runs extend or
    close. Closed runs emit as `DiarizedSpan`. Tie-break by smaller
    `cluster_id` (deterministic).
  - **`DiarizedSpan` simplified:** dropped `similarity` and
    `speaker_slot` fields (window-local concepts not well-defined for
    a stitched multi-window span). Window-local context still
    available via `Diarizer::collected_embeddings()`. Added
    `is_new_speaker` flag.
  - **`DiarizerOptions` adds:** `binarize_threshold` (default 0.5;
    matches pyannote's `Binarize` default), `exclude_overlap`
    (default true; matches pyannote's `embedding_exclude_overlap`).
  - **`Diarizer::buffered_frames()` accessor:** new introspection
    method returning the count of un-finalized frames in the
    per-frame accumulator. (Rev-7 said steady-state was "≈ 1000 =
    10 s × 100 fps," which is wrong — rev-8 corrected to ≈ 589 frames
    at the model's ≈ 58.9 fps frame rate.)
  - **`Diarizer::clear()` rustdoc updated** to enumerate the rev-6
    per-frame stitching state being dropped, and to clarify that
    open per-cluster runs are NOT auto-emitted (use `finish_stream`
    first).
  - **§6 module layout:** `diarizer.rs` split into `diarizer/{mod,
    builder, error, span, overlap, reconstruct}.rs` to keep each
    file focused (reconstruction is ~400 lines on top of existing
    pump glue).
  - **§9 reconstruction tests:** comprehensive new tests covering
    `exclude_overlap` (clean + fallback paths), single/two-speaker
    stitching, count-bounded argmax, warm-up trimming, tie-break,
    `is_new_speaker`, `finish_stream` flush, `clear()` discards,
    `buffered_frames()` introspection, slot-to-cluster reassignment,
    inactive-slot skip. Plus pyannote parity smoke test (rev-7
    target: DER ≤ 5%; rev-8 relaxed to ≤ 10% per review-7 T3-I).
  - **§3 deferred items:** updated to remove "Per-frame voice prob
    from `dia::segment` integrated path" (now in scope as
    `Action::SpeakerScores`); added `min_cluster_size` cluster
    pruning, VBx clustering, configurable warm_up, configurable
    per-frame `min_duration_on/off`.
- **Revision 7** (2026-04-26): nails down the
  VAD-prerequisite contract that's been implicit since the brainstorm
  but never formalized; enriches `DiarizedSpan` for downstream
  storage/analysis. No algorithmic changes — the rev-6 algorithm
  already handled variable-length VAD inputs correctly; rev 7
  documents that and tests it.
  - **§1 / §3 / §11.12 (NEW): variable-length VAD-filtered input
    contract.** dia accepts `process_samples` pushes of any size
    (empty / sub-clip / multi-clip / whole-stream). Spelled out the
    behavior for each, the lazy-window-scheduling semantics, and the
    "activities can span original-VAD-range boundaries" caveat. Added
    a recommended caller pattern showing how to map dia's
    "samples-pushed" timeline back to original-audio time using a
    `Vec<(dia_offset, original_offset)>` log keyed off
    `Diarizer::total_samples_pushed()`.
  - **§4.4 `DiarizedSpan` enrichment:** added three quality-metric
    fields (`average_activation: f32`, `activity_count: u32`,
    `clean_mask_fraction: f32`). All three are computed during
    reconstruction at zero algorithmic cost (already-available
    accumulator values). User mandate: "we want all important
    information so that we can store the data for analyze or other
    stuff."
  - **§4.4 `Diarizer::total_samples_pushed()` accessor:** new
    introspection method returning the cumulative count of samples
    ever passed to `process_samples` since the last `clear()`.
    Caller-side tool for VAD-range timeline mapping.
  - **§5.11 RLE algorithm extended** to populate the new
    `DiarizedSpan` fields. State growth: `PerClusterRun` gains
    `activation_sum_normalized: f64`, `frame_count: u32`,
    `contributing_activities: HashSet<(WindowId, u8)>`,
    `clean_mask_count: u32`. `ReconstructState` gains
    `activity_clean_flags: HashMap<(WindowId, u8), bool>`.
  - **§9 tests:** ~12 new test cases covering empty pushes,
    sub-window, sub-MIN_CLIP_SAMPLES, multiple short pushes, multiple
    medium pushes, long single push, mixed lengths, exactly-window
    boundary, finish_stream + clear cycle,
    `total_samples_pushed` monotonicity, cross-VAD-range activity
    detection, plus 6 quality-metric tests for the three new
    DiarizedSpan fields.
  - **§15 #50 (deferred):** open question whether dia should ship a
    `dia::utils::TimelineMap` helper for VAD-to-original timeline
    mapping. Defer to v0.1.1+ pending real-world need.
  - **No algorithmic changes.** The rev-6 algorithm already handled
    variable-length VAD-filtered input correctly (segment buffers
    samples lazily, embed accepts 25 ms – any-length clips,
    reconstruction finalizes per-frame as windows complete). Rev 7
    is documentation + observability + tests — no behavior change
    for any existing caller.
- **Revision 8** (2026-04-26): incorporates fifth
  adversarial-review feedback (review 7). Three load-bearing
  correctness fixes plus a fan-out of pseudocode bugs in the rev-6
  reconstruction state machine. The most consequential change: rev-7
  spec routed `Diarizer`'s `exclude_overlap` mask through
  `embed_weighted`, which §5.3 says is a no-op for clips ≤ 2 s — i.e.,
  silently ineffective for the most common VAD-filtered activity case.
  Rev-8 adds a new `EmbedModel::embed_masked` primitive matching
  pyannote's actual gather-and-embed mechanism and routes §5.8 through
  it.
  - **§4.2 / §5.8 (review-7 T1-A, CRITICAL):** new
    `EmbedModel::embed_masked(samples, keep_mask: &[bool])` API.
    Pyannote's `ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__`
    interpolates the input mask to fbank-frame rate, binarizes at
    0.5, gathers retained features, runs ONNX on the variable-length
    cleaned features. Our equivalent: gather samples at sample-rate
    (pre-fbank), zero-pad to `EMBED_WINDOW_SAMPLES`, embed normally.
    Numerically equivalent up to fbank-window-boundary effects when
    mask transitions land on 25 ms boundaries. §5.8 entirely rewritten
    to use the new path.
  - **§4.4 / §5.8 / §5.10 (review-7 T1-C):** SAMPLES_PER_FRAME is
    `WINDOW_SAMPLES / FRAMES_PER_WINDOW = 160_000 / 589 ≈ 271.65`,
    not the "160 / 100 fps / 10 ms" rev-7 prose claimed. Verified
    against shipped `dia::segment::stitch::frame_to_sample`. §5.8 now
    uses correct frame-to-sample expansion via the actual
    `frame_to_sample` function (not a hardcoded constant).
    `MIN_CLEAN_FRAMES` (rev-7's separate constant) dropped in favor
    of direct gathered-sample-count check against `MIN_CLIP_SAMPLES`,
    which is the constraint that actually matters for `embed_masked`.
  - **§5.9 / §5.11 (review-7 T2-A):** `FrameCount` split into
    `activation_chunk_count` (un-trimmed; for `average_activation`
    normalization) and `count_chunk_count` (warm-up trimmed; for
    speaker-count rounding). Rev-7 conflated them. Pyannote's
    `Inference.aggregate(skip_average=True)` and `speaker_count`
    maintain independent counts.
  - **§5.11 (review-7 T2-B):** `emit_finalized_frames` flush loop
    now Option-aware (`let Some(...) else break;`) so the
    `flush_open_runs` end-of-stream path doesn't panic when the
    deque is empty but `finalization_boundary == u64::MAX`.
  - **§5.9 (review-7 T2-C):** `ReconstructStateExtras::window_starts`
    declared explicitly (rev-7 referenced it without declaring).
  - **§5.8 (review-7 T2-D):** `activity_clean_flags.insert((W.id, s),
    used_clean)` now visible in the §5.8 pseudocode (rev-7 declared
    the field but didn't show insertion).
  - **§15 #52 (review-7 T2-E):** new HIGH-severity action item for
    a pre-impl ChaCha8Rng byte-fixture regression test.
  - **§11.13 (review-7 T3-J, NEW):** explicit eviction policy for
    `slot_to_cluster` / `activity_clean_flags` / `window_starts`.
    Bounds memory on long sessions; correctness-relevant when an
    open run extends across many finalized windows.
  - **§5.11 (review-7 T3-F + T3-H):** `u64` everywhere in frame
    arithmetic (no truncating `as u32` casts) and `total_cmp` instead
    of `partial_cmp` in argmax for NaN safety.
  - **§4.4 (review-7 T3-B):** `DiarizedSpan` rustdoc no longer
    references the non-existent `min_duration_off` option;
    explicitly states "no merging in v0.1.0; see §15 #48 for v0.1.1".
  - **§9 / §13 (review-7 T3-I):** pyannote parity DER target relaxed
    from ≤ 5% to ≤ 10% absolute. Reviewer correctly observed that
    5% was aspirational given the documented divergences (gather
    mechanism, online-vs-batch clustering, default clusterer
    differences). 5% remains a stretch goal for v0.1.1+.
  - **§1 (review-7 T4-C):** divergence list expanded from "one
    deliberate divergence" to a table of seven. Honestifies the
    framing.
  - **§12 (review-7 T3-C, T4-B):** decision log now includes rev-6,
    rev-7, rev-8 additions. T4-B's deterministic tie-break called
    out as a deliberate IMPROVEMENT over pyannote's
    implementation-defined `np.argsort`, not a divergence.
  - **§14 (review-7 T3-D):** rejected-finding subsections added for
    reviews 5, 6, 7. Review-7 T1-B explicitly rejected with
    line-number evidence (reviewer cited line 115 for default
    clusterer; actual line is 210, default is `VBxClustering`).
  - **§15 #53 (review-7 T4-A):** new low-severity action item for
    `Action::SpeakerScores` allocation-pool optimization.
- **Revision 9** (2026-04-27, this document): incorporates sixth
  adversarial-review feedback. Reviewer's verdict: NEAR-QUALIFIED
  with three small pseudocode-consistency bugs, all genuine and
  caught by their cargo-check-by-eye. Plus a graceful retraction of
  their prior T1-B claim (§14 already had this rejection;
  reviewer-8 verified independently that VBx IS pyannote's default
  and apologized). All review-8 items applied; nothing rejected.
  - **§5.10 finalized_count_at (review-8 T1-A):** removed the stale
    `if c.chunk_count == 0` guard left over from the rev-8
    `FrameCount` split. Single guard on `c.count_chunk_count`.
  - **§5.11 emit_finalized_frames (review-8 T1-B):** the body of
    the loop referenced an undefined local `chunk_count`; rev-9
    renames to the bound `activation_chunk_count` (matches the
    rev-8 split). Same root cause as T1-A — incomplete rename.
  - **§5.9 / §5.11 ReconstructState organization (review-8 T1-C):**
    rev-8 had `slot_to_cluster` and `activity_clean_flags` declared
    in two different structs (`ReconstructState` +
    `ReconstructStateExtras`), with `§5.11` adding a third stub.
    Rev-9 merges everything into a single `ReconstructState` in
    §5.9, deletes `ReconstructStateExtras`, and removes the §5.11
    duplicate stub. Single source of truth.
  - **§5.10 stale duplicate (review-8 T2-A):** the "copied here for
    clarity" computation snippet had the old `chunk_count` field
    name. Rev-9 removes the duplicate entirely; the canonical loop
    in §5.9 step C is the single source.
  - **§5.11 make_range u64 (review-8 T2-B):** rev-8's changelog
    claimed "u64 everywhere in frame arithmetic," but `make_range`
    still had `frame_to_sample(start_frame as u32) as i64`. Rev-9
    introduces a Diarizer-internal `frame_to_sample_u64(u64) -> u64`
    helper (bit-for-bit equivalent to segment's `u32` version) and
    routes `make_range` through it. Tracked as §15 #54 to fold
    back into segment v0.1.1 once that bumps.
  - **§5.12 stale embed_weighted reference (review-8 T2-C):** the
    error-handling-policy prose still mentioned `embed_weighted`
    after rev-8's mechanism switch to `embed_masked`. Rev-9 fixes
    plus extends the policy per review-8 T3-A.
  - **§5.8 / §5.12 InvalidClip skip-and-continue (review-8 T3-A):**
    when both the clean mask AND the speaker-only fallback gather
    < `MIN_CLIP_SAMPLES` (rare; segment shouldn't produce activities
    that short, but possible under heavy overlap), the Diarizer
    now SKIPS the activity and continues the pump rather than
    propagating `InvalidClip`. Matches pyannote's
    `speaker_verification.py:611-612` skip-and-continue.
  - **§4.2 / §5.8 long-clip masked-embed double divergence
    (review-8 T3-B):** explicit documentation that long-clip
    masked embeddings have TWO layered divergences from pyannote
    (sample-vs-frame-rate gather + sliding-window-mean vs
    variable-length-ONNX). For typical short VAD utterances this
    doesn't matter. §15 #49 tracks the v0.2 mask-aware ONNX
    export fix for divergence #2.
  - **§14 review-8 acknowledgment:** the reviewer-8 explicitly
    retracted their prior rev-7 T1-B claim ("HMM-GMM is the
    default") after independent verification. Rev-9 §14 records
    this in the existing review-7 rejection block.
  - **§15 #54 (review-8 T2-B follow-up):** new low-severity action
    item to add `pub(crate) frame_to_sample_u64` to
    `dia::segment::stitch` in v0.1.1 so the Diarizer can drop its
    internal copy.

## 14. Findings rejected from reviews

For traceability across both adversarial-review rounds.

### From review 1 (rev 1 → rev 2)

- **T2-C** (drop generic `EmbeddingMeta<A, T>` to "match
  `dia::segment::Clip`"): **rejected, sustained in rev 3.** There is
  no `dia::segment::Clip` type — `Segmenter::push_samples(&[f32])`
  takes raw samples directly (verified by re-reading
  `dia/src/segment/segmenter.rs` during rev-3 work). The reviewer's
  "consistency" argument is based on a type that doesn't exist. More
  importantly, **the user explicitly requested generic
  `EmbeddingMeta` during brainstorming** ("Make EmbeddingMeta
  generic over audio_id and track_id, do not use String directly").
  Reverting that based on a false premise would be wrong. The cost
  the reviewer flags (generics propagating through
  `EmbeddingResult<A, T>` and the embed methods) is real but bounded
  — the meta-free path (`embed`, `embed_weighted`) returns concrete
  `EmbeddingResult` (= `EmbeddingResult<(), ()>`) so typical callers
  never see the type parameters.

### From review 2 (rev 2 → rev 3)

- **None outright rejected.** All action items in the second review
  are accepted in some form, either as direct spec edits (T1-A, T1-B,
  cached centroid, fast paths, validation timing, K-means seed,
  builder parity, `clear()` rustdoc, Embedding invariant,
  `DegenerateEmbedding` variant, §1 wording, `cosine_similarity`
  rationale) or as v0.1.1+ action items (see §15). Where the
  reviewer's preferred phrasing differed from what we ended up
  shipping, we noted the difference inline (e.g., reviewer wrote
  `NonFiniteInput` for zero-norm cases; we split that into a separate
  `DegenerateEmbedding` variant since zero IS finite).

### From review 3 (rev 3 → rev 4)

- **None outright rejected.** All eight items in the third review are
  accepted as rev-4 spec edits (items 42–49 above; T3-B and T3-C also
  applied). Items #42 and #43 were the only implementation-blocking
  gaps; the rest are documentation polish and are now folded into the
  spec rather than the implementation patch list.

### From review 4 (rev 4 → rev 5)

- **T2-A specific number rejected; meta-point accepted.** Review 4
  claimed the §5.7 audio-buffer reach-back "could be `step_samples
  + WINDOW_SAMPLES = 200 000` not 160 000" and that the
  `WINDOW_SAMPLES` bound was an unverified design assertion. The
  specific number is wrong: code trace on shipped
  `dia::segment::segmenter.rs` (`schedule_ready_windows` at lines
  161-173, `emit_speaker_activities` at lines 282-329, `finish` at
  lines 453-474) and `window.rs` (`plan_starts` at line 36) shows
  the bound is exactly `WINDOW_SAMPLES`. The reviewer's hypothetical
  "200 000" would over-allocate by 40 000 samples (one step). The
  meta-point — that the spec asserted this without justification —
  was correct, and §5.7 / §11.5 now contain the line-by-line code
  trace.
- **All other review-4 items accepted.** T1-A (nalgebra 0.34), T1-B
  (K-means++ pins), T1-C (SpeakerCentroid accessors), T2-B
  (numbering disambiguation), T2-C (property test wording), T2-D
  (kaldi-native-fbank spike), T3-A (visibility consistency), T3-B
  (left-to-right reduction pin), T3-C (de-dup #32 vs #43), T3-D
  (linkage-impact nuance), T4-A (Diarizer::Error::Internal variant).
  T4-B (§15 numbering gaps) is acknowledged but deferred — the
  numbering reflects the historical accumulation of items across
  reviews and renumbering would invalidate cross-references in the
  prior revision-history bullets.

### From review 5 (rev 5 → rev 6) — none rejected

All review-5 items folded into the rev-6 scope expansion (pyannote
reconstruction). No findings rejected.

### From review 6 (rev 6 → rev 7) — none rejected

The rev-6 → rev-7 transition was not driven by an adversarial review
but by a user-mandate scope clarification (VAD-prerequisite contract
+ richer DiarizedSpan output). No findings rejected; not applicable.

### From review 7 (rev 7 → rev 8)

- **T1-B (default clusterer is HMM-GMM, not VBx) — REJECTED.** The
  reviewer cited `speaker_diarization.py:115` as evidence that
  pyannote's default is `clustering: str = "HiddenMarkovModelClustering"`.
  Verified against the actual source file at
  `/Users/user/Develop/findit-studio/pyannote-audio/src/pyannote/audio/pipelines/speaker_diarization.py`:
  line 115 is part of an unrelated docstring/output-formatting
  block in the `diarize_one` method. The actual default
  `clustering: str = "VBxClustering"` lives in the `__init__`
  signature (around line 210 in the working tree at the time of
  rev-8 verification; line numbers shift with edits so we cite the
  signature rather than a fixed line). The `Clustering` enum at
  `clustering.py:853-857` does NOT contain
  `HiddenMarkovModelClustering` — it has `AgglomerativeClustering`,
  `KMeansClustering`, `VBxClustering`, `OracleClustering`. §15 #44
  ("VBx is pyannote-audio's default offline clusterer") is
  therefore **correct**.

  **Reviewer-8 acknowledged this rejection** in their rev-8
  review's T4-A: *"Verified independently against current
  pyannote-audio main (last commit 2025-10-08): the default at the
  __init__ signature is `clustering: str = "VBxClustering"`. The
  Clustering enum does NOT contain HiddenMarkovModelClustering...
  My Rev 7 review's T1-B was wrong; my agent likely cited a
  stale/non-existent source. The rev-8 author's pushback was
  correct."* No further action; sustaining the rejection.
- **All other review-7 items accepted** as rev-8 spec edits. T1-A
  (gather-and-embed via new `embed_masked` API + §5.8 rewrite),
  T1-C (SAMPLES_PER_FRAME math: ≈ 271.65 not 160; ≈ 58.9 fps not
  100; MIN_CLEAN_FRAMES dropped in favor of direct gathered-sample
  count check). T2-A (split FrameCount into activation_chunk_count
  and count_chunk_count), T2-B (Option-aware flush loop), T2-C
  (`window_starts` declared in `ReconstructStateExtras`), T2-D
  (`activity_clean_flags` insertion shown in §5.8 pseudocode),
  T2-E (§15 #52 ChaCha8Rng byte-fixture regression test),
  T2-F (resolved by T1-A fix — gather-and-pad path is no longer
  no-op for clips ≤ 2 s). T3-A (§3.1 phantom reference removed),
  T3-B (§4.4 min_duration_off note honestified to "v0.1.0: no
  merging; see §15 #48 for v0.1.1"), T3-E (`SAMPLE_RATE_TB` referenced
  via `dia::segment` re-exports throughout), T3-F (`u64` everywhere
  in frame arithmetic; no `as u32` truncating casts), T3-G
  (§5.9 explicit "buffer per WindowId; integrate when both
  Activity events and SpeakerScores have arrived"), T3-H
  (`total_cmp` for NaN safety in argmax), T3-I (DER target relaxed
  to ≤ 10 % absolute on a clean clip; 5 % was aspirational),
  T3-J (eviction policy pinned in §11.13). T4-A (segment
  `Action::SpeakerScores` allocation churn — accepted as a v0.1.1
  optimization noted in §15 #53), T4-B (deterministic tie-break
  documented as a deliberate improvement in §12 decision log),
  T4-C (§1 divergence list expanded — "one deliberate divergence"
  → bulleted list).

### From review 8 (rev 8 → rev 9) — none rejected

All review-8 items applied as rev-9 spec edits:
- T1-A (§5.10 stale `chunk_count` guard removed),
- T1-B (§5.11 `chunk_count` → `activation_chunk_count`),
- T1-C (`ReconstructStateExtras` merged into single
  `ReconstructState` in §5.9; §5.11 duplicate stub removed),
- T2-A (§5.10 stale "copied here" snippet removed; §5.9 step C is
  the single source),
- T2-B (rev-8 changelog overstated u64 cleanup; rev-9 actually
  adds a Diarizer-internal `frame_to_sample_u64` helper),
- T2-C (§5.12 `embed_weighted` reference → `embed_masked`),
- T3-A (`InvalidClip` skip-and-continue rather than propagate;
  matches pyannote `speaker_verification.py:611-612`),
- T3-B (§4.2 `embed_masked` rustdoc grew explicit "two layered
  divergences from pyannote" note for long-clip masked embeds).
- T4-A is acknowledged in §14 review-7 entry.

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
| 29 | (rev-2 review) Property test: `Embedding::similarity` is symmetric and bounded in `[-1, 1]` for any pair of `Embedding`s constructed via `normalize_from`. Quickcheck-style.                                          | low |
| 30 | (rev-2 review) Stress test: 10 000 streaming `Clusterer::submit` calls with synthetic embeddings drawn from K=5 well-separated clusters; assert eventual recovery of K speakers and per-cluster purity > 0.95.        | medium |
| 31 | (rev-2 review) Spectral clustering benchmark on N ∈ {10, 100, 500, 1000} with timing budget; flag any super-quadratic blow-up.                                                                                          | low |
| 32 | (rev-2 review) Cross-platform determinism check: same input → same K-means labels on macOS/Linux/x86_64/aarch64. Pinpoints any non-determinism in `nalgebra`'s eigensolver.                                            | medium |
| 33 | (rev-2 review) Document the `embed_weighted` AllSilent threshold semantics: clarify in rustdoc that AllSilent triggers when `total_weight < NORM_EPSILON` (not when individual weights are zero).                     | low |
| 34 | (rev-2 review) Add a Diarizer end-to-end "no audio" test: zero-length input, all-silent input, sub-MIN_CLIP_SAMPLES input — assert no panics, no spurious activities, no embeddings collected.                        | medium |
| 35 | (rev-2 review) Add a `DiarizerOptions::with_offline_cluster_options` plumbing for callers who want to control the offline-refinement defaults from the top-level builder. Currently they call `cluster_offline` directly with their own `OfflineClusterOptions`.                                                                          | low |
| 36 | (rev-2 review) Fuzzing pass on `compute_fbank` and `cluster_offline` (cargo-fuzz). Especially valuable on the eigendecomp path which can NaN-out on pathological inputs.                                              | medium |
| 37 | (rev-2 review) `Clusterer::submit_batch(&[Embedding])` convenience method that loops over `submit` and returns `Vec<Result<ClusterAssignment>>` — saves the caller a hot loop.                                          | low |
| 38 | (rev-2 review) Documentation: a "porting from `findit-speaker-embedding`" appendix in the README listing every behavioral divergence (sliding-window mean, threshold defaults, error variants, etc.).                  | medium |
| 39 | (rev-2 review) Property: spectral clustering is invariant under input permutation (cluster *count* and *purity* must match, even if label values differ). Test with shuffle.                                          | medium |
| 40 | (rev-2 review) Telemetry: optional `tracing` spans inside `Diarizer::process_samples` for per-activity embedding-and-clustering timings. Gated behind a feature flag to avoid forcing the dep on no-tracing users.    | low |
| 41 | (rev-2 review) Verify `kaldi-native-fbank` numerical agreement with the C++ `knf-rs` binding on a 30-second test clip. If divergent beyond 1e-3, investigate which is more correct vs Python's torchaudio reference. | medium |
| 42 | (rev-3 review T3-C) Float32 precision impact on `RollingMean` for sessions with `N > 10⁶` assignments — once `||accumulator||` reaches ~10⁶, per-component lower bits of the `f32` mantissa (≈7 decimal digits) can drift, biasing the L2-normalized centroid direction. Investigate periodic accumulator renormalization (e.g., `accumulator /= ||accumulator||` every 10 000 assignments to keep magnitude near 1) or upgrading the accumulator to `f64` internally. Not a v0.1.0 issue (typical sessions are well under 10⁶); flagged for tracking.                                                                                                                                                                                       | low |
| 43 | (rev-4 review T2-D) **PRE-IMPLEMENTATION SPIKE** (run before writing the `dia::embed` module): integrate `kaldi-native-fbank = "0.1"` against `findit-speaker-embedding`'s reference embeddings on a fixed 16 kHz test clip and assert per-coefficient agreement with `torchaudio.compliance.kaldi.fbank` to `< 1e-4`. The crate is brand-new (0.1.0 published 2026-01-12 by RustedBytes, single version, ~1.4k downloads) and has not yet been validated in production. If the spike fails, fall back to `knf-rs` (C++ binding) **before** committing the dep in §7. This protects against shipping v0.1.0 with an fbank crate that subtly disagrees with the Python reference. Distinct from item 41 (post-ship continuous verification); this item is a go/no-go gate. | high |
| 44 | **VBx (Variational Bayes HMM Clustering of x-vectors)** as a third `OfflineMethod` for `cluster_offline`. VBx is `pyannote-audio`'s **default** offline clusterer (`speaker_diarization.py:210`); it pairs Agglomerative Hierarchical Clustering for initialization with a Variational Bayes EM refinement using PLDA scoring (Landini et al., "Bayesian HMM clustering of x-vector sequences in the LIUM speaker diarization system," 2022). Better quality than spectral on long meetings with many speakers, but: (a) batch-only — no streaming variant exists in literature; (b) requires a pre-trained PLDA model (separate training pipeline + bundled weights); (c) ~3–5× slower than spectral. Adds significant scope. Defer to v0.2 once spectral has shipped and we have a comparison baseline. (See §13 history for why we did NOT port pyannote-audio wholesale — they're a 20k-line file-in/file-out PyTorch toolkit, but VBx is a specific algorithm worth implementing on top of our streaming-friendly substrate.) | medium |
| 45 | **Threshold tuning A/B**: empirically compare `cluster_offline` defaults against pyannote-audio's defaults (Agglomerative `centroid`-linkage threshold 0.7 vs our `Average`-linkage threshold 0.5; spectral threshold tuning). On a held-out multi-speaker dataset, measure DER (diarization error rate) across the 2D grid of (linkage, threshold). May change `DEFAULT_SIMILARITY_THRESHOLD` in v0.1.1 if a clearly-better default emerges. | low |
| 46 | (rev-6) **Pyannote parity-test harness setup** (referenced from §9): wire up a Python sidecar (uv-managed) that runs `pyannote.audio.SpeakerDiarization` on a fixed 5-minute multi-speaker WAV, exports the reference Annotation as RTTM, and a Rust integration test that runs `dia::Diarizer` on the same WAV, exports its DiarizedSpans as RTTM, and runs `pyannote.metrics.diarization.DiarizationErrorRate` to compute DER. Assertion: DER ≤ 5% absolute. Gated via `#[ignore]` (requires Python toolchain and downloaded models; not part of default `cargo test`). The harness also doubles as the kaldi-native-fbank validation harness (§15 #43) since both need a fixed reference clip + Python reference. | high |
| 47 | (rev-6) **Configurable speaker-count `warm_up`**: rev-6 hardcodes `(0.1, 0.1)` to match pyannote default. Expose via `DiarizerOptions::with_speaker_count_warm_up_ratio` if user tuning becomes a real ask. | low |
| 48 | (rev-6) **Configurable `min_duration_on/off`** for per-cluster RLE: rev-6 emits any closed run as a DiarizedSpan. Pyannote applies `min_duration_on`/`min_duration_off` thresholds in `to_annotation`. Adding ours is straightforward but defer until we have user feedback on whether tiny spans / micro-gaps cause downstream issues. | low |
| 49 | (rev-6) **Mask-aware embedding ONNX export**: WeSpeaker ONNX export currently takes only waveform. Pyannote's mask is consumed inside the model. We approximate via per-window mean weighting in `embed_weighted`. If we ever re-export WeSpeaker with a mask input, switch the §5.8 mask path to feed it directly to the model — closer to pyannote behavior. Significant scope (re-export + maintain alongside the current export). | medium |
| 50 | (rev-7) **`dia::utils::TimelineMap` helper for VAD-to-original mapping**: caller-side `Vec<(dia_offset, original_offset)>` log + `map_dia_range_to_original(TimeRange) -> TimeRange` helper. Currently the caller writes this themselves (~30 lines, see §11.12 example). Could ship a small `dia::utils::TimelineMap` struct with `record(dia_offset, original_offset)` / `lookup(dia_range)` API. Defer until we have evidence that >1 caller has implemented this independently — premature standardization. | low |
| 51 | (rev-7) **Per-VAD-range diarization with shared speakers**: API for callers who want each VAD range to produce its own clean DiarizedSpans (no spans crossing original-VAD-range boundaries) while keeping speaker IDs consistent across ranges. Currently requires manual mid-level orchestration. Could provide `Diarizer::checkpoint() / restore_after_clear` or `Diarizer::process_chunk_isolated(samples)` that finishes-then-resumes without speaker-state loss. Defer; the §11.12 caller pattern (concatenate + post-process) is sufficient for v0.1.0. | medium |
| 52 | (rev-8 review-7 T2-E) **ChaCha8Rng keystream byte-fixture regression test** (PRE-IMPL): assert that `ChaCha8Rng::seed_from_u64(42).next_u64()` (and a small handful of other seeds × draws) produces the exact `u64` values we record now. Test runs under both default features and `default-features = false`. Goal: catch a future `rand_chacha` minor-version bump that quietly changes the cipher's stream output, since our public determinism contract (§11.9) depends on byte-stable output. The fixture is ~6 lines of test code; the alarm value is high (silent reproducibility break across users upgrading dia transitively). | high |
| 53 | (rev-8 review-7 T4-A) **`Action::SpeakerScores` allocation churn**: the variant carries `Box<[[f32; 589]; 3]>` ≈ 7 KB per emission. For a 60-min stream at default `step_samples = 40_000`, that's ~1440 windows × 7 KB = ~10 MB of allocations through the lifetime, all short-lived (consumed by the Diarizer's pump within the same `process_samples` call). Could pool the buffers (`pub(crate) Segmenter::take_speaker_score_buf` returning a `Box` from a small-pool freelist that the Diarizer returns post-consume) or pass `&[[f32; 589]; 3]` borrowed from segmenter scratch. Defer until profiling shows allocator pressure. | low |
| 54 | (rev-9 review-8 T2-B) **`dia::segment::stitch::frame_to_sample_u64` helper** in `dia::segment` v0.1.1: add `pub(crate) const fn frame_to_sample_u64(frame_idx: u64) -> u64` alongside the existing `u32`-typed helper. The Diarizer currently carries a copy of this function (§5.11 `make_range`) to avoid the truncating cast. Once segment ships its own `u64` helper, the Diarizer copy can be deleted. Bit-exact equivalence is asserted by a unit test in §9. | low |
