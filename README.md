# dia

Sans-I/O speaker diarization with pyannote-equivalent accuracy.

[![Crates.io](https://img.shields.io/crates/v/diarization.svg)](https://crates.io/crates/diarization)
[![Documentation](https://docs.rs/diarization/badge.svg)](https://docs.rs/diarization)
[![License](https://img.shields.io/badge/license-(MIT_OR_Apache--2.0)_AND_MIT_AND_CC--BY--4.0-blue.svg)](https://github.com/al8n/diarization)

## Status

v0.1.0 ships:

- `diarization::segment` — speaker segmentation (pyannote/segmentation-3.0).
  **Bundled by default** (~6 MB, MIT) via `SegmentModel::bundled()`.
- `diarization::embed` — speaker fingerprint (WeSpeaker ResNet34 ONNX +
  kaldi fbank). **Not bundled** — 27 MB exceeds the crates.io 10 MB cap;
  caller fetches via `scripts/download-embed-model.sh` or sets
  `DIA_EMBED_MODEL_PATH`.
- `diarization::plda` — pyannote/speaker-diarization-community-1 PLDA
  whitening. **Bundled by default** (CC-BY-4.0) via `PldaTransform::new()`.
- `diarization::cluster` + `pipeline` — pyannote `cluster_vbx` primitives
  (PLDA → AHC → VBx → centroid → cosine → Hungarian → reconstruct).
- `diarization::offline::OwnedDiarizationPipeline` — owned-audio batch
  entrypoint.
- `diarization::streaming::StreamingOfflineDiarizer` — voice-range-driven
  streaming entrypoint with the same per-fixture DER as offline (caller
  drives a VAD; heavy stages 1+2 run eagerly, global clustering deferred
  to `finalize`).

## Pipeline

```
audio decoder → resample to 16 kHz → VAD → diarization → downstream services
```

See [`docs/superpowers/specs/`](docs/superpowers/specs/) for the design
spec.

## Quick start

The segmentation model and PLDA weights ship inside the crate — only the
WeSpeaker ResNet34-LM embedding ONNX is BYO (~26 MB; above the
crates.io 10 MB hard limit, so it cannot be bundled). Fetch it from the
[FinDIT-Studio/dia-models](https://huggingface.co/FinDIT-Studio/dia-models)
HuggingFace bundle in either of two ways:

```sh
# Option A: huggingface_hub CLI (recommended; handles caching + auth).
hf download FinDIT-Studio/dia-models wespeaker_resnet34_lm.onnx \
  --local-dir models

# Option B: plain curl, no extra tools.
mkdir -p models
curl -L -o models/wespeaker_resnet34_lm.onnx \
  https://huggingface.co/FinDIT-Studio/dia-models/resolve/main/wespeaker_resnet34_lm.onnx
```

(Workspace developers can also run `./scripts/download-embed-model.sh`,
which wraps the same URL and verifies the SHA-256. The script is
omitted from the published crate tarball.)

Then run an end-to-end example. The simplest needs only the `ort`
feature:

```sh
cargo run --release --features ort --example run_owned_pipeline -- \
  path/to/clip_16k.wav > hyp.rttm
```

For the streaming pipeline (uses `silero-vad` to detect voice ranges
on the fly), enable the matching feature:

```sh
cargo run --release --features ort,silero-vad --example run_streaming_pipeline -- \
  path/to/clip.wav
```

`DIA_EMBED_MODEL_PATH` overrides the default `models/wespeaker_resnet34_lm.onnx`
location if you keep the model elsewhere.

## License

The `dia` source is dual-licensed: **MIT OR Apache-2.0** (caller's
choice). See `LICENSE-MIT` / `LICENSE-APACHE`.

### Bundled-model attributions propagate to downstream binaries

`dia` embeds two third-party model artifacts into every compiled
binary via `include_bytes!`:

| File | License | Source |
|---|---|---|
| `models/segmentation-3.0.onnx` (bundled when `bundled-segmentation` feature is on, default) | **MIT** | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| `models/plda/*.bin` | **CC-BY-4.0** | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) |

The full SPDX expression is therefore
`(MIT OR Apache-2.0) AND MIT AND CC-BY-4.0`. When you redistribute a
binary that depends on `dia`, reproduce the attributions from
[`NOTICE`](NOTICE) somewhere a recipient can find — for instance, in
your application's "About" or third-party-licenses page. Full
provenance: [`models/SOURCE.md`](models/SOURCE.md) (segmentation),
[`models/plda/SOURCE.md`](models/plda/SOURCE.md) (PLDA).

To opt out of the segmentation bundling (e.g. to ship a fine-tuned
variant), disable default features: `diarization = { version = "...",
default-features = false, features = ["ort"] }`. You then load via
`SegmentModel::from_file` / `from_memory` directly.

## Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `ort` | yes | The ONNX-runtime-backed `SegmentModel` and `EmbedModel` types. |
| `bundled-segmentation` | yes | Embeds `models/segmentation-3.0.onnx` (~6 MB) into the binary. Exposes `SegmentModel::bundled()`. Implies `ort`. Disable to ship a fine-tuned segmentation model separately. |
| `tch` | no | TorchScript embedding backend (libtorch ≈600 MB). Bit-exact pyannote on heavy-overlap fixtures where ONNX→ORT diverges. |
| `silero-vad` | no | Path-dep on the sister `silero` crate; only used by `examples/run_streaming_pipeline.rs`. |

The PLDA parity test runs as part of the regular test suite — no
feature flag required:

```bash
cargo test plda::parity_tests
```

It auto-skips when `tests/parity/fixtures/01_dialogue/*.npz` is absent
(checked-in for this repo, but a fresh checkout from a model-only
mirror would have to regenerate them via the Phase-0 capture script).
