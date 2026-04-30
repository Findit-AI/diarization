# dia

Sans-I/O streaming speaker diarization for variable-length VAD-filtered audio.

[![Crates.io](https://img.shields.io/crates/v/dia.svg)](https://crates.io/crates/dia)
[![Documentation](https://docs.rs/dia/badge.svg)](https://docs.rs/dia)
[![License](https://img.shields.io/badge/license-(MIT_OR_Apache--2.0)_AND_CC--BY--4.0-blue.svg)](https://github.com/al8n/dia)

## Status

v0.1.0 ships:

- `diarization::segment` — speaker segmentation (pyannote/segmentation-3.0 ONNX)
- `diarization::embed` — speaker fingerprint (WeSpeaker ResNet34 ONNX + kaldi fbank)
- `diarization::cluster` — online streaming + offline (spectral + agglomerative)
- `diarization::Diarizer` — top-level orchestrator with pyannote-style per-frame
  reconstruction (overlap-add cluster activations, count-bounded argmax,
  per-cluster RLE-to-spans)

## Pipeline

```
audio decoder → resample to 16 kHz → VAD → diarization::Diarizer → downstream services
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

Note: in v0.1.0 the `diarization::plda` module is crate-private (no
public-API caller can construct a `RawEmbedding`, by design — see
`src/lib.rs:62-72`), but `include_bytes!` still embeds the
weights into every linked binary, so the attribution requirement
above applies regardless of which `dia` modules you actually
call. The module flips back to `pub` once the Phase-2/3 VBx +
constrained Hungarian work lands with a typed entry from
`EmbedModel`.

## Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `ort` | yes | The ONNX-runtime-backed `SegmentModel` and `EmbedModel` types. |

The PLDA parity test runs as part of the regular test suite — no
feature flag required:

```bash
cargo test plda::parity_tests
```

It auto-skips when `tests/parity/fixtures/01_dialogue/*.npz` is absent
(checked-in for this repo, but a fresh checkout from a model-only
mirror would have to regenerate them via the Phase-0 capture script).
