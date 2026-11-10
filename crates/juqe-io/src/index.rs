use juqe_core::format::{ExpertIndex, MAX_INDEX_BYTES};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

/// Read and validate an index without unbounded file allocation.
pub fn read_index(path: impl AsRef<Path>) -> io::Result<ExpertIndex> {
    let mut text = String::new();
    File::open(path)?
        .take((MAX_INDEX_BYTES + 1) as u64)
        .read_to_string(&mut text)?;
    text.parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
