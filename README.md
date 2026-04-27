# dia

Sans-I/O streaming speaker diarization for variable-length VAD-filtered audio.

[![Crates.io](https://img.shields.io/crates/v/dia.svg)](https://crates.io/crates/dia)
[![Documentation](https://docs.rs/dia/badge.svg)](https://docs.rs/dia)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/al8n/dia)

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

MIT OR Apache-2.0.
