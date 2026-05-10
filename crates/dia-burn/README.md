# dia-burn

Pure-Rust [`burn`](https://burn.dev) inference backend for the
[`dia`](../..) speaker diarization pipeline. A drop-in alternative
to `dia-ort` for cross-compile targets where ONNX Runtime can't
ship prebuilt binaries (`powerpc64`, `powerpc64le`, `riscv64`,
`s390x`, `i686`, `wasm32-*`).

> **Status — placeholder.** The runtime types are stable and the
> API is forward-compatible, but inference itself is not wired up
> yet. Both dia ONNX models hit upstream `burn-onnx` codegen bugs;
> see "Why this is a stub" below.

## Why a `burn` backend?

`dia-ort` is the default backend and is bit-exact with pyannote on
the supported targets, but ORT only ships prebuilt binaries for
mainstream OS / arch combinations. `dia-tch` has the same
prebuilt-binary problem on top of a ~600 MB libtorch dependency.
Anything outside x86_64 / aarch64 — embedded targets, WebAssembly,
exotic server architectures — has no inference path today.

`burn` is a pure-Rust deep learning framework. `burn-onnx`
translates ONNX models into Rust at build time; the runtime uses
`burn-ndarray` (no system deps, no prebuilts, no CUDA / Metal /
wgpu unless you opt in). That makes it portable to anywhere
`rustc` runs.

## Why this is a stub

As of `burn-onnx` 0.21.0-pre.5 (the latest published; 0.21 stable
doesn't ship `burn-onnx` yet) **both** dia models fail end-to-end:

### `pyannote/segmentation-3.0` — codegen failure

```text
error: Conv1d expects input tensor of rank 3, got rank 4
```

The model has an `If` op feeding the first `Conv1d`. `burn-onnx`'s
type-inference pass doesn't propagate ranks across `If`'s two
branches, so the `Conv1d` translator sees the wrong rank and
exits before producing any Rust.

**Fix path**: ONNX surgery to inline the static branch (most
likely workable today), *or* upstream `burn-onnx` support for
`If`-op rank propagation.

### `wespeaker_resnet34_lm` — emitted Rust doesn't compile

`burn-onnx` codegen *succeeds*: 606 lines of Rust + a 25 MB `.bpk`
weights blob. The emitted code does not compile:

```rust
let concat2_out1: [i64; 3usize] = [&shape3_out1[..], &[add2_out1][..]]
    .concat()
    .try_into()
    .unwrap();
let resize1_out1 = {
    let target_height = concat2_out1[2] as usize;
    let target_width = concat2_out1[3] as usize;  // ← OOB on length-3 array
    burn::tensor::module::interpolate(unsqueeze2_out1, ...)
};
```

The `Resize` op is lowered into an `interpolate` call indexed by an
out-of-bounds array element.

**Fix path**: upstream `burn-onnx` `Resize`-op codegen patch.

## Reproducing the codegen failures

```bash
cd crates/dia-burn
cargo build --features unstable-onnx-codegen
```

`build.rs` runs the wespeaker codegen, which today fails to
compile downstream. Pointing `build.rs` at
`models/segmentation-3.0.onnx` instead reproduces the rank
inference failure earlier — the codegen step itself panics.

## When does the stub get replaced?

When `burn-onnx` ships fixes for both ops. The crate's public
surface is intentionally tiny so the swap will be a non-breaking
internal change:

- `BurnEmbedModel::from_embedded()` constructs the model
- `.embed_chunk_with_frame_mask(samples, mask)` returns a 256-d
  raw embedding (matches `dia-ort`'s contract)
- `consts::{EMBEDDING_DIM, FRAMES_PER_WINDOW, WINDOW_SAMPLES}`
  are re-exported from `dia-core`

Today every inference call returns `Error::NotYetImplemented`.

## License

MIT OR Apache-2.0, matching the rest of the dia workspace.
