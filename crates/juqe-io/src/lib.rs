//! Exact expert reads into exclusively borrowed host buffers.
//!
//! Success means that all three matrices are present. On error, destinations may
//! contain partial writes and must not be published. Linux direct mode bypasses
//! the page cache; ordinary buffered reads are available on Unix.
#![cfg(unix)]

mod buffer;
mod index;
#[cfg(target_os = "linux")]
mod linux;
mod positional;
mod reader;
mod report;
mod source;

pub use buffer::AlignedBuffer;
pub use index::read_index;
pub use reader::ExpertReader;
pub use report::{Fallback, OpenMode, ReadPath, ReadReport};
