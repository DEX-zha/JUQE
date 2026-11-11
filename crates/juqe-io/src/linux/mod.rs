//! Linux direct I/O; raw pointers remain within synchronous borrowed reads.

mod alignment;
mod scatter;

pub(super) use alignment::{Alignment, alignment, validate_staging};
pub(super) use scatter::scatter;

use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

pub(super) fn open_direct(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}
