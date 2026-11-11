//! Bounded expert loading with explicit batch ownership.
//!
//! The runtime owns cache metadata and workers. Storage owns destination memory.
//! A prepared batch holds leases until the inference engine finishes consuming it.
#![forbid(unsafe_code)]

mod batch;
mod config;
mod dispatch;
mod error;
mod pool;
mod runtime;
mod storage;
mod worker;

pub use batch::{Batch, BatchReport};
pub use config::Config;
pub use error::RuntimeError;
pub use runtime::Runtime;
pub use storage::ExpertStorage;
