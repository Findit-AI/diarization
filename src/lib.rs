//! Sans-I/O speaker diarization for streaming audio.
//!
//! - [`segment`]: v0.1.0 speaker segmentation (window-local speaker slots).
//! - [`embed`]: speaker fingerprint generation (WeSpeaker ResNet34 ONNX).
//!
//! A clustering layer that turns window-local speaker slots into global
//! speaker identities will follow in a later release.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

#[cfg(feature = "std")]
extern crate std;

pub mod embed;
pub mod segment;
