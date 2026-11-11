//! Validated layouts and import of the existing text `experts.idx` format.

mod error;
mod index;
mod layout;
mod parser;

pub use error::FormatError;
pub use index::ExpertIndex;
pub use layout::{Layer, Part};

/// Maximum accepted text index size (4 MiB).
pub const MAX_INDEX_BYTES: usize = 4 * 1024 * 1024;
/// Maximum allocated block size (256 MiB per expert).
pub const MAX_EXPERT_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum number of layers in a pack.
pub const MAX_LAYERS: usize = 512;
/// Maximum number of experts per layer and slots per cache.
pub const MAX_EXPERTS: usize = 4096;

/// Alignment of layer offsets and expert block strides in the pack.
pub const PACK_ALIGNMENT: u64 = 4096;
