//! ONNX Runtime execution providers — opt-in hardware acceleration.
//!
//! Each `ep-*` cargo feature in `Cargo.toml` toggles the matching
//! ORT execution provider (EP). When the feature is on, the EP type
//! is re-exported here so callers can construct a provider and pass
//! it to [`crate::segment::SegmentModelOptions::with_providers`] or
//! [`crate::embed::EmbedModelOptions::with_providers`] without taking
//! a direct `ort` dependency.
//!
//! Names match `ort::ep::*` (e.g. `dia::ep::CoreML`, `dia::ep::CUDA`).
//! The older `*ExecutionProvider`-suffixed aliases that lived in
//! `ort::execution_providers` were deprecated upstream in
//! ort 2.0.0-rc.12; we follow the new convention and do not re-export
//! the deprecated aliases.
//!
//! ## Example: register a single provider
//!
//! ```ignore
//! # // ignored: requires the `coreml` cargo feature + Apple host.
//! use diarization::{
//!   ep::CoreML,
//!   segment::{SegmentModel, SegmentModelOptions},
//! };
//!
//! let seg_opts = SegmentModelOptions::default()
//!   .with_providers(vec![CoreML::default().build()]);
//! let mut seg = SegmentModel::bundled_with_options(seg_opts)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! **Do not** copy that pattern for `EmbedModel`: ORT's CoreML EP
//! mistranslates the WeSpeaker ResNet34-LM graph and emits NaN/Inf
//! on most realistic inputs across every CoreML compute unit
//! (`cpu` / `gpu` / `ane` / `all`), every model format
//! (`NeuralNetwork` / `MLProgram`), and the static-shape knob.
//! `EmbedModel::from_file` deliberately does NOT auto-register
//! providers; if you call `with_providers([CoreML::default().build()])`
//! on the embed options yourself you will get hard pipeline failures
//! on most clips. CUDA / TensorRT / DirectML / ROCm / OpenVINO have
//! NOT been parity-validated on this model — verify on your data
//! before enabling.
//!
//! ## Example: ship a single binary that auto-picks GPU
//!
//! Build with the `gpu` meta-feature (`--features gpu`); the helper
//! returns whichever EPs were compiled in, in a priority order. ORT
//! registers each as `MayUse` — the first one whose ops match runs
//! and the rest stay dormant on CPU fallback.
//!
//! Note: [`auto_providers`] is what
//! [`crate::segment::SegmentModel::bundled`] already calls; you
//! normally never invoke it directly. It is `pub` for callers who
//! want to build the same provider list and apply it through the
//! `_with_options` paths (e.g. on `EmbedModel`, where the no-arg
//! constructor stays on CPU by design).
//!
//! ```ignore
//! # // ignored: depends on which `ep-*` features are compiled in.
//! use diarization::ep::auto_providers;
//! let seg_opts = diarization::segment::SegmentModelOptions::default()
//!   .with_providers(auto_providers());
//! ```
//!
//! ## Runtime requirements
//!
//! The `ep-*` cargo features only enable the *bindings*. Each EP
//! still needs the matching native library on the host:
//!
//! - `coreml` — Apple Silicon / macOS, no extra install (ships in
//!   the system).
//! - `cuda` — NVIDIA CUDA toolkit + cuDNN.
//! - `tensorrt` — NVIDIA TensorRT (also pulls CUDA).
//! - `directml` — Windows 10+ with DirectX 12.
//! - `rocm` / `migraphx` — AMD ROCm runtime.
//! - `openvino` — Intel OpenVINO toolkit.
//! - `webgpu` — a WebGPU-capable native runtime (Dawn / wgpu).
//! - `xnnpack` — ARM/x86 SIMD CPU EP, no extra install.
//! - others (`onednn`, `cann`, `acl`, `qnn`, `nnapi`, `tvm`, `azure`)
//!   follow vendor-specific install paths.
//!
//! The `ort` crate's default `download-binaries` feature ships a
//! CPU-only build; vendor EPs typically require either a vendor build
//! of onnxruntime or `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` pointing
//! at the vendor libs. See
//! <https://ort.pyke.io/perf/execution-providers> for setup details.
//!
//! ## EP determinism
//!
//! Different EPs can produce slightly different f32/f16 outputs from
//! the same model — vendor kernels round differently, fuse ops
//! differently, and use different math libraries. The dia parity
//! tests assert against pyannote's CPU reference; switching EPs may
//! perturb DER by a small amount but should not regress the partition
//! shape on realistic inputs *for models that the EP can compile
//! correctly* — see the WeSpeaker / CoreML caveat above for an EP
//! that does not satisfy that assumption.

pub use ort::ep::ExecutionProviderDispatch;

#[cfg(feature = "coreml")]
#[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
pub use ort::ep::CoreML;

#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub use ort::ep::CUDA;

#[cfg(feature = "tensorrt")]
#[cfg_attr(docsrs, doc(cfg(feature = "tensorrt")))]
pub use ort::ep::TensorRT;

#[cfg(feature = "directml")]
#[cfg_attr(docsrs, doc(cfg(feature = "directml")))]
pub use ort::ep::DirectML;

#[cfg(feature = "rocm")]
#[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
pub use ort::ep::ROCm;

#[cfg(feature = "migraphx")]
#[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
pub use ort::ep::MIGraphX;

#[cfg(feature = "openvino")]
#[cfg_attr(docsrs, doc(cfg(feature = "openvino")))]
pub use ort::ep::OpenVINO;

#[cfg(feature = "webgpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "webgpu")))]
pub use ort::ep::WebGPU;

#[cfg(feature = "xnnpack")]
#[cfg_attr(docsrs, doc(cfg(feature = "xnnpack")))]
pub use ort::ep::XNNPACK;

#[cfg(feature = "onednn")]
#[cfg_attr(docsrs, doc(cfg(feature = "onednn")))]
pub use ort::ep::OneDNN;

#[cfg(feature = "cann")]
#[cfg_attr(docsrs, doc(cfg(feature = "cann")))]
pub use ort::ep::CANN;

#[cfg(feature = "acl")]
#[cfg_attr(docsrs, doc(cfg(feature = "acl")))]
pub use ort::ep::ACL;

#[cfg(feature = "qnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "qnn")))]
pub use ort::ep::QNN;

#[cfg(feature = "nnapi")]
#[cfg_attr(docsrs, doc(cfg(feature = "nnapi")))]
pub use ort::ep::NNAPI;

#[cfg(feature = "tvm")]
#[cfg_attr(docsrs, doc(cfg(feature = "tvm")))]
pub use ort::ep::TVM;

#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub use ort::ep::Azure;

/// Build a provider list from whichever `ep-*` features are compiled in.
///
/// Order is "most-likely-to-accelerate first":
/// `TensorRT → CUDA → CoreML → DirectML → ROCm → MIGraphX →
/// OpenVINO → WebGPU → OneDNN → XNNPACK → CANN → QNN → ACL →
/// NNAPI → TVM → Azure`. ORT registers each as `MayUse`,
/// so the first whose ops match accelerates and the rest stay
/// dormant on CPU fallback.
///
/// Returns an empty `Vec` if no `ep-*` features are enabled, in which
/// case ORT runs on its default CPU dispatch.
///
/// # Example
///
/// ```ignore
/// # // ignored: depends on which `ep-*` features are compiled in.
/// use diarization::ep::auto_providers;
///
/// let seg_opts = diarization::segment::SegmentModelOptions::default()
///   .with_providers(auto_providers());
/// ```
#[must_use]
pub fn auto_providers() -> Vec<ExecutionProviderDispatch> {
  #[allow(unused_mut)]
  let mut out: Vec<ExecutionProviderDispatch> = Vec::new();
  #[cfg(feature = "tensorrt")]
  out.push(TensorRT::default().build());
  #[cfg(feature = "cuda")]
  out.push(CUDA::default().build());
  #[cfg(feature = "coreml")]
  out.push(CoreML::default().build());
  #[cfg(feature = "directml")]
  out.push(DirectML::default().build());
  #[cfg(feature = "rocm")]
  out.push(ROCm::default().build());
  #[cfg(feature = "migraphx")]
  out.push(MIGraphX::default().build());
  #[cfg(feature = "openvino")]
  out.push(OpenVINO::default().build());
  #[cfg(feature = "webgpu")]
  out.push(WebGPU::default().build());
  #[cfg(feature = "onednn")]
  out.push(OneDNN::default().build());
  #[cfg(feature = "xnnpack")]
  out.push(XNNPACK::default().build());
  #[cfg(feature = "cann")]
  out.push(CANN::default().build());
  #[cfg(feature = "qnn")]
  out.push(QNN::default().build());
  #[cfg(feature = "acl")]
  out.push(ACL::default().build());
  #[cfg(feature = "nnapi")]
  out.push(NNAPI::default().build());
  #[cfg(feature = "tvm")]
  out.push(TVM::default().build());
  #[cfg(feature = "azure")]
  out.push(Azure::default().build());
  out
}
