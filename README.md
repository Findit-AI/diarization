# dia

Sans-I/O speaker diarization for streaming audio.

`dia` is a Rust port of two findit-studio Python projects: `findit-pyannote-seg`
(speaker segmentation) and `findit-speaker-embedding` (speaker embedding). The
v0.1.0 release ships **segmentation only** (`dia::segment`); embedding and
cross-window clustering will follow in subsequent releases.

## Design

The core (`dia::segment::Segmenter`) is a Sans-I/O state machine: it never
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
use dia::segment::{Segmenter, SegmentModel, SegmentOptions, Event};

let mut model = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
let mut seg   = Segmenter::new(SegmentOptions::default());

while let Some(frame) = audio_in.next() {
    seg.process_samples(&mut model, &frame, |event| match event {
        Event::Activity(a)  => println!("{:?}", a),
        Event::VoiceSpan(r) => println!("{:?}", r),
    })?;
}
seg.finish_stream(&mut model, |_| {})?;
# Ok::<(), dia::segment::Error>(())
```

For Sans-I/O usage (no `ort` dependency), see `examples/stream_layer1.rs`.

## License

`dia` is dual-licensed under MIT or Apache 2.0 at your option.
