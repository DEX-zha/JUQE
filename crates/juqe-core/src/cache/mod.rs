//! Per-layer cache ownership and load publication.
//!
//! The cache owns metadata only. Callers must finish a write before completing or
//! failing its ticket, and hold every lease until the consumer has finished.

mod error;
mod handles;
mod slot;
mod state;

pub use error::CacheError;
pub use handles::{Lease, LoadTicket, Request};
pub use state::Cache;
