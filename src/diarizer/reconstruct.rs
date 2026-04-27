//! Per-frame per-cluster reconstruction state machine.
//! Stub for Phase 10 (Tasks 39-42); `Diarizer::new` / `clear` /
//! `buffered_frames` reach into this module.

#[derive(Debug, Default)]
pub(crate) struct ReconstructState;

impl ReconstructState {
  pub(crate) fn new() -> Self {
    Self
  }

  pub(crate) fn clear(&mut self) {}

  pub(crate) fn buffered_frame_count(&self) -> usize {
    0
  }
}
