//! Speaker segmentation: Sans-I/O state machine + optional ort driver.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.

mod error;
mod hysteresis;
mod options;
mod powerset;
mod segmenter;
mod stitch;
mod types;
mod window;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
mod model;

// pub use error::Error;
pub use options::{
    FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, SAMPLE_RATE_HZ,
    SAMPLE_RATE_TB, SegmentOptions, WINDOW_SAMPLES,
};
// pub use segmenter::Segmenter;
// pub use types::{Action, Event, SpeakerActivity, WindowId};

// #[cfg(feature = "ort")]
// #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
// pub use model::SegmentModel;
