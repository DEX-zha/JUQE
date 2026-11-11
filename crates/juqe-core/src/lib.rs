//! Validated expert metadata and deterministic cache bookkeeping.
//!
//! No filesystem, GPU, pruning, or implicit background work lives in this crate.
#![forbid(unsafe_code)]

pub mod cache;
pub mod format;
