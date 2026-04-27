//! Builder + options for [`crate::diarizer::Diarizer`]. Filled in by Task 33.

use crate::{cluster::ClusterOptions, segment::SegmentOptions};

#[derive(Debug, Clone, Default)]
pub struct DiarizerOptions {
  #[allow(dead_code)]
  segment: SegmentOptions,
  #[allow(dead_code)]
  cluster: ClusterOptions,
}

#[derive(Debug, Clone, Default)]
pub struct DiarizerBuilder;
