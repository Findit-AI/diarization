//! Build-time ONNX → burn Rust codegen.
//!
//! Currently a no-op: gated entirely behind the
//! `unstable-onnx-codegen` feature. As of `burn-onnx` 0.21.0-pre.5
//! BOTH dia models hit upstream codegen bugs:
//!
//! - **pyannote/segmentation-3.0** — burn-onnx fails to type-infer
//!   the SincNet first block. The model routes a tensor through an
//!   `If` op into `Conv1d`; burn-onnx's translator reports
//!   "Conv1d expects input tensor of rank 3, got rank 4" and exits.
//!   Fix path: ONNX surgery to drop the dynamic branch *or*
//!   upstream `If`-op rank propagation.
//!
//! - **wespeaker_resnet34_lm** — burn-onnx codegen *succeeds* (606
//!   lines + a 25 MB `.bpk` blob) but the emitted Rust does not
//!   compile. The `Resize` lowering produces an out-of-bounds
//!   array index inside the interpolate call. Fix path: upstream
//!   `Resize`-op codegen patch.
//!
//! Both are upstream-track. dia-burn ships a documented stub today
//! and will wire up real models when those land. The codegen
//! implementation below is preserved so contributors can flip the
//! feature flag and reproduce the failures with one command.

#[cfg(feature = "unstable-onnx-codegen")]
fn run_codegen() {
  use std::path::PathBuf;

  let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
  // WeSpeaker ResNet34-LM: ~26 MB ONNX, NOT in-repo (excluded by
  // crates.io 10 MB ceiling — fetched by
  // `dia-core/scripts/download-embed-model.sh`).
  let wespeaker = manifest_dir
    .parent()
    .expect("dia-burn parent")
    .join("dia-core/models/wespeaker_resnet34_lm.onnx");
  println!("cargo:rerun-if-changed={}", wespeaker.display());

  if !wespeaker.exists() {
    println!(
      "cargo:warning={} missing — skipping burn codegen. Run dia-core/scripts/download-embed-model.sh.",
      wespeaker.display()
    );
    return;
  }

  let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
  burn_onnx::ModelGen::new()
    .input(wespeaker.to_str().expect("ONNX path is UTF-8"))
    .out_dir(out_dir.to_str().expect("OUT_DIR is UTF-8"))
    .run_from_script();
}

fn main() {
  #[cfg(feature = "unstable-onnx-codegen")]
  run_codegen();
}
