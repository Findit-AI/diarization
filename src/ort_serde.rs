//! Serde bridges for foreign `ort` types that don't carry `Serialize`/
//! `Deserialize` impls themselves. Mirrors the sister silero crate's
//! pattern (`silero/src/options.rs::graph_optimization_level`):
//! introduce a snake-case-tagged mirror enum, plug it into
//! `serde(with = ...)`, supply a `default()` for stability across ort
//! versions.
//!
//! Only compiled when **both** `ort` and `serde` features are enabled
//! (the wrappers are useless without the underlying foreign type).

#[cfg(all(feature = "ort", feature = "serde"))]
pub(crate) mod graph_optimization_level {
  use ort::session::builder::GraphOptimizationLevel;
  use serde::{Deserialize, Deserializer, Serialize, Serializer};

  /// Snake-case-tagged mirror of `GraphOptimizationLevel` for the
  /// serde-serialized form (e.g. `"level3"` in JSON). The default
  /// matches silero's choice — `Disable` is stable across ort
  /// versions, whereas ort's own runtime default has shifted between
  /// release lines.
  #[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
  #[serde(rename_all = "snake_case")]
  enum OptLevel {
    #[default]
    Disable,
    Level1,
    Level2,
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptLevel {
    fn from(v: GraphOptimizationLevel) -> Self {
      match v {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptLevel> for GraphOptimizationLevel {
    fn from(v: OptLevel) -> Self {
      match v {
        OptLevel::Disable => Self::Disable,
        OptLevel::Level1 => Self::Level1,
        OptLevel::Level2 => Self::Level2,
        OptLevel::Level3 => Self::Level3,
        OptLevel::All => Self::All,
      }
    }
  }

  pub fn serialize<S>(v: &GraphOptimizationLevel, ser: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptLevel::from(*v).serialize(ser)
  }

  pub fn deserialize<'de, D>(de: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptLevel::deserialize(de).map(Into::into)
  }
}
