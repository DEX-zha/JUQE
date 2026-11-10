use juqe_core::format::{ExpertIndex, Layer};
use std::{io, path::Path, time::Instant};

use crate::{AlignedBuffer, OpenMode, ReadReport, source::Source};

/// Shared positional reader with caller-owned scratch and destination buffers.
#[derive(Debug)]
pub struct ExpertReader {
    source: Source,
}

impl ExpertReader {
    /// Open an immutable weight file and validate its length against the index.
    ///
    /// Length validation does not authenticate model identity.
    ///
    /// # Errors
    /// Returns the underlying file error, an invalid-data error for truncation,
    /// or an unsupported error when direct I/O is unavailable.
    pub fn open(path: impl AsRef<Path>, index: &ExpertIndex, mode: OpenMode) -> io::Result<Self> {
        let source = Source::open(path.as_ref(), mode)?;
        index
            .validate_file_size(source.len()?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(Self { source })
    }

    /// Read one expert into exclusive destinations in gate, up, down order.
    ///
    /// Capture `requested_at` when submitting the request, before it waits for a
    /// worker. Preallocate scratch and destinations before submission.
    ///
    /// # Errors
    /// Rejects unknown experts, dense layers, invalid buffer lengths, and future
    /// timestamps. I/O errors may leave partial bytes; do not publish them.
    pub fn read_into(
        &self,
        layer: &Layer,
        expert: usize,
        mut destinations: [&mut [u8]; 3],
        scratch: &mut AlignedBuffer,
        requested_at: Instant,
    ) -> io::Result<ReadReport> {
        let started = Instant::now();
        if requested_at > started {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "request timestamp is in the future",
            ));
        }
        let offset = layer
            .expert_offset(expert)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        validate_buffers(layer, &destinations, scratch)?;

        let mut bytes_read = 0;
        let (path, fallback) = self.source.read(
            layer,
            offset,
            &mut destinations,
            scratch.as_mut_slice(),
            &mut bytes_read,
        )?;
        let ready = Instant::now();
        Ok(ReadReport {
            path,
            fallback,
            bytes_read,
            queue_wait: started - requested_at,
            service_time: ready - started,
            request_to_ready: ready - requested_at,
        })
    }
}

fn validate_buffers(
    layer: &Layer,
    destinations: &[&mut [u8]; 3],
    scratch: &AlignedBuffer,
) -> io::Result<()> {
    if (scratch.as_slice().len() as u64) < layer.stride()
        || destinations
            .iter()
            .zip(layer.parts())
            .any(|(buffer, part)| buffer.len() as u64 != part.bytes())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "scratch or destination size mismatch",
        ));
    }
    Ok(())
}
