//! Pure-Rust `burn` inference backend for the dia diarization
//! pipeline.
//!
//! # Why this crate exists
//!
//! `dia-ort` (the default backend) shells out to ONNX Runtime,
//! which only ships prebuilt binaries for x86_64 / aarch64 on
//! mainstream OSes. On targets ort-sys can't reach — `powerpc64`,
//! `powerpc64le`, `riscv64`, `s390x`, `i686`, anything `wasm32-*`
//! — `dia-ort` is a non-starter. `dia-tch` is also out (libtorch
//! has the same prebuilt-binary problem, plus is ~600 MB even on
//! supported targets). `dia-burn` aims to plug that gap with a
//! pure-Rust inference path: `burn-onnx` translates the dia ONNX
//! models into a Rust burn graph at build time, the runtime uses
//! `burn-ndarray` (no system deps).
//!
//! # Status: blocked on upstream burn-onnx
//!
//! As of `burn-onnx` 0.21.0-pre.5 (the latest pre-release; 0.21
//! stable doesn't ship burn-onnx yet) BOTH dia models fail end-to-
//! end:
//!
//! - **pyannote/segmentation-3.0** — burn-onnx fails to type-infer
//!   the SincNet first block. The model has an `If` op feeding the
//!   first `Conv1d`; burn-onnx's translator reports
//!   "Conv1d expects input tensor of rank 3, got rank 4". Codegen
//!   exits before producing any Rust. Fix path: ONNX surgery to
//!   inline the static branch, *or* upstream burn-onnx support for
//!   `If`-op rank propagation.
//!
//! - **wespeaker_resnet34_lm** (256-d speaker embedder) — burn-onnx
//!   codegen *succeeds* and produces 606 lines of Rust + a 25 MB
//!   `.bpk` weights blob. The emitted Rust does not compile: the
//!   `Resize` op is lowered into an `interpolate` call indexed by
//!   an out-of-bounds array element (length-3 `[i64]`, indexed at
//!   `[3]`). Fix path: upstream `Resize`-op codegen patch.
//!
//! Both blockers are upstream-track. This crate ships a documented
//! placeholder today (a working library with the public types
//! but no inference); when burn-onnx lands fixes for both ops the
//! `unstable-onnx-codegen` feature gets promoted to the default
//! and the stub gets replaced with the generated graph wrappers.
//!
//! # Reproducing the codegen failures
//!
//! ```text
//! cd crates/dia-burn
//! cargo build --features unstable-onnx-codegen
//! ```
//!
//! `build.rs` runs the wespeaker codegen, which today fails to
//! compile downstream. Switching `build.rs` to point at
//! `models/segmentation-3.0.onnx` reproduces the rank-inference
//! failure at codegen time instead.

#![cfg_attr(not(test), no_std)]

use diarization::embed::{EMBEDDING_DIM, Error as EmbedError};

/// Errors returned by the burn embedding stub.
///
/// The `Embed` variant is reserved for the eventual real
/// implementation — keeping it in the type today means swapping
/// the stub for a working backend won't be a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// Forwards `dia-core`'s `embed::Error` for fbank/shape
  /// validation failures (used once codegen is unstuck).
  #[error(transparent)]
  Embed(#[from] EmbedError),

  /// `dia-burn`'s inference path is currently a placeholder. See
  /// the crate-level docs for the burn-onnx blockers.
  #[error(
    "dia-burn inference is not yet wired up — burn-onnx 0.21.0-pre.5 fails to codegen the dia models. See dia_burn crate docs."
  )]
  NotYetImplemented,
}

/// WeSpeaker ResNet34-LM speaker-embedding model on a pure-Rust
/// `burn` backend.
///
/// Today this is a documented stub: every inference call returns
/// [`Error::NotYetImplemented`]. The struct + method shapes are
/// stable (they mirror `dia-ort`'s embed model so the planned
/// meta-crate can dispatch over them without per-backend
/// branching), and the constants below let downstream code that
/// integrates against the eventual API compile against `dia-burn`
/// today.
#[derive(Debug, Default)]
pub struct BurnEmbedModel {
  // Zero-sized today; will hold the generated `Model<NdArray>` +
  // `Device` once burn-onnx codegen produces compiling Rust.
  _placeholder: (),
}

impl BurnEmbedModel {
  /// Construct a stub model. Returns successfully so callers can
  /// build their pipeline + plumbing today — the failure surfaces
  /// at the inference boundary, not at construction.
  pub const fn from_embedded() -> Self {
    Self { _placeholder: () }
  }

  /// Pyannote-style frame-mask embedding. **Currently always
  /// returns [`Error::NotYetImplemented`]** — see crate docs.
  ///
  /// Signature is fixed so the eventual real implementation can
  /// drop in without breaking callers:
  /// - `chunk_samples`: a `WINDOW_SAMPLES` (= 160 000) mono
  ///   16 kHz chunk
  /// - `frame_mask`: a `FRAMES_PER_WINDOW` (= 589) bool slice
  /// - returns the raw 256-d (`EMBEDDING_DIM`) embedding, NOT
  ///   L2-normalized (matching `dia-ort`'s contract).
  pub fn embed_chunk_with_frame_mask(
    &self,
    _chunk_samples: &[f32],
    _frame_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    Err(Error::NotYetImplemented)
  }
}

/// Re-exported shape constants so downstream crates can compile
/// against `dia-burn` without also depending on `dia-core` just to
/// reach for these values.
pub mod consts {
  pub use diarization::embed::EMBEDDING_DIM;
  pub use diarization::segment::{FRAMES_PER_WINDOW, WINDOW_SAMPLES};
}

#[cfg(test)]
mod tests {
  use super::*;
  use diarization::segment::{FRAMES_PER_WINDOW, WINDOW_SAMPLES};

  #[test]
  fn stub_constructs_and_errors_on_inference() {
    let model = BurnEmbedModel::from_embedded();
    let samples = vec![0.0f32; WINDOW_SAMPLES as usize];
    let mask = vec![true; FRAMES_PER_WINDOW];
    let err = model
      .embed_chunk_with_frame_mask(&samples, &mask)
      .expect_err("stub should error");
    assert!(matches!(err, Error::NotYetImplemented));
  }

  #[test]
  fn re_exported_consts_match_dia_core() {
    assert_eq!(consts::EMBEDDING_DIM, EMBEDDING_DIM);
    assert_eq!(consts::FRAMES_PER_WINDOW, FRAMES_PER_WINDOW);
    assert_eq!(consts::WINDOW_SAMPLES, WINDOW_SAMPLES);
  }
}
