# dia

Sans-I/O streaming speaker diarization for variable-length VAD-filtered audio.

[![Crates.io](https://img.shields.io/crates/v/dia.svg)](https://crates.io/crates/dia)
[![Documentation](https://docs.rs/dia/badge.svg)](https://docs.rs/dia)
[![License](https://img.shields.io/badge/license-(MIT_OR_Apache--2.0)_AND_CC--BY--4.0-blue.svg)](https://github.com/al8n/dia)

## Status

v0.1.0 ships:

- `dia::segment` — speaker segmentation (pyannote/segmentation-3.0 ONNX)
- `dia::embed` — speaker fingerprint (WeSpeaker ResNet34 ONNX + kaldi fbank)
- `dia::cluster` — online streaming + offline (spectral + agglomerative)
- `dia::Diarizer` — top-level orchestrator with pyannote-style per-frame
  reconstruction (overlap-add cluster activations, count-bounded argmax,
  per-cluster RLE-to-spans)

## Pipeline

```
audio decoder → resample to 16 kHz → VAD → dia::Diarizer → downstream services
```

VAD-filtered, variable-length pushes are first-class. See
[`docs/superpowers/specs/`](docs/superpowers/specs/) for the design spec.

## Quick start

```sh
./scripts/download-model.sh         # pyannote/segmentation-3.0
./scripts/download-embed-model.sh   # WeSpeaker ResNet34
cargo run --release --features ort --example diarize_from_wav -- path/to/clip.wav
```

See `examples/diarize_from_wav.rs` for the full source.

## License

The `dia` source is dual-licensed: **MIT OR Apache-2.0** (caller's
choice). See `LICENSE-MIT` / `LICENSE-APACHE`.

### CC-BY-4.0 attribution required

`dia` embeds a small set of third-party PLDA weight matrices from
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
into every compiled binary via `include_bytes!`. Those matrices are
licensed under **CC-BY-4.0**, and the attribution requirement
**propagates to any downstream binary that links `dia`**.

The full SPDX expression is therefore `(MIT OR Apache-2.0) AND CC-BY-4.0`.

When you redistribute a binary that depends on `dia` (e.g. as part of a
larger application), you must reproduce the attribution from
[`NOTICE`](NOTICE) somewhere a recipient can find — for instance, in
your application's "About" or third-party-licenses page. The full
provenance, including the upstream snapshot revision and the
`models/plda/*.bin` layout, lives in
[`models/plda/SOURCE.md`](models/plda/SOURCE.md).

If you want to opt out of the CC-BY-4.0 obligation: do not call
anything in `dia::plda`, do not link in code that does, and we
recommend gating the dependency such that the `dia::plda` module is
never reached. (At the time of writing, no other `dia` module
depends on `dia::plda`; that changes once the Phase-2/3 VBx +
constrained Hungarian work lands.)

## Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `ort` | yes | The ONNX-runtime-backed `SegmentModel` and `EmbedModel` types. |
| `plda-fixtures` | **no** | Public constructors for `dia::plda::RawEmbedding` (`from_raw_array`) and `dia::plda::PostXvecEmbedding` (`from_pyannote_capture`). Enabling this is the explicit signal that the caller is parity-test code or offline tooling — production builds without the feature cannot construct PLDA boundary inputs at all, eliminating a class of silent distribution-drift bugs. The Phase-1 parity test (`tests/parity_plda.rs`) declares `required-features = ["plda-fixtures"]` and is automatically skipped when the feature is not enabled. |

To run the PLDA parity test:

```bash
cargo test --features plda-fixtures --test parity_plda
```
