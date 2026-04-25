//! Sans-I/O speaker diarization for streaming audio.
//!
//! See the [`segment`] module for the v0.1.0 surface (speaker segmentation).
//! Future releases will add an `embed` module for speaker embedding and a
//! clustering layer that turns window-local speaker slots into global
//! speaker identities.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

pub mod segment;
