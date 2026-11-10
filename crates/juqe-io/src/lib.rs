//! Exact expert loading. Success means all three matrices are present.
//!
//! The caller must not publish a destination slot on error: partial writes may exist.
//! Buffered mode works on Unix; Linux direct mode bypasses the OS page cache.
//! No CUDA, inference, asynchronous scheduler, or pruning is implemented here.
#![cfg(unix)]
mod buffer;
#[cfg(target_os = "linux")]
mod linux;

pub use buffer::AlignedBuffer;
use juqe_core::format::{ExpertIndex, Layer, MAX_INDEX_BYTES};
use std::{
    fs::File,
    io::{self, Read},
    os::unix::fs::FileExt,
    path::Path,
    time::{Duration, Instant},
};

/// How to open the weight file. Direct mode is explicitly Linux-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    /// Portable positional reads through the filesystem cache.
    Buffered,
    /// Linux O_DIRECT; fail explicitly if opening direct I/O is unsupported.
    Direct,
}
/// Actual successful path, never silently reported as direct when buffered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPath {
    /// Read one block into scratch and copy matrices, using buffered file I/O.
    BufferedStaging,
    /// Read one block into aligned scratch and copy matrices, using O_DIRECT.
    DirectStaging,
    /// Scatter the block into the three matrix destinations, using O_DIRECT.
    DirectScatter,
}
/// Direct scatter was unavailable for this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// Destination addresses or file segments did not satisfy DIO alignment.
    Alignment,
    /// Kernel rejected vectored direct I/O; a complete staging read succeeded.
    Unsupported,
}
/// Measurements for one successful request. Allocate buffers before submitting it.
#[derive(Debug)]
pub struct ReadReport {
    /// Path actually used.
    pub path: ReadPath,
    /// Why a direct request used staging instead of scatter.
    pub fallback: Option<Fallback>,
    /// Bytes successfully read from the file, including partial fallback attempts.
    pub bytes_read: u64,
    /// Time from caller submission to entry into the loader.
    pub queue_wait: Duration,
    /// Time spent inside the loader, including validation, read and any copies.
    pub service_time: Duration,
    /// Caller submission through completion; use this for the 10 ms objective.
    pub request_to_ready: Duration,
}

/// Read and validate an index without unbounded file allocation.
pub fn read_index(path: impl AsRef<Path>) -> io::Result<ExpertIndex> {
    let mut text = String::new();
    File::open(path)?
        .take((MAX_INDEX_BYTES + 1) as u64)
        .read_to_string(&mut text)?;
    text.parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Shared positional reader. Use separate scratch and destinations per worker.
#[derive(Debug)]
pub struct ExpertReader {
    file: File,
    mode: OpenMode,
    #[cfg(target_os = "linux")]
    alignment: linux::Alignment,
}
impl ExpertReader {
    /// Open a file and reject truncation against the supplied index.
    ///
    /// Length validation does not authenticate model identity; use a verified pack.
    pub fn open(path: impl AsRef<Path>, index: &ExpertIndex, mode: OpenMode) -> io::Result<Self> {
        let file = match mode {
            OpenMode::Buffered => File::open(path)?,
            OpenMode::Direct => {
                #[cfg(target_os = "linux")]
                {
                    linux::open_direct(path.as_ref())?
                }
                #[cfg(not(target_os = "linux"))]
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "direct I/O requires Linux",
                    ));
                }
            }
        };
        index
            .validate_file_size(file.metadata()?.len())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        #[cfg(target_os = "linux")]
        let alignment = linux::alignment(&file);
        Ok(Self {
            file,
            mode,
            #[cfg(target_os = "linux")]
            alignment,
        })
    }

    /// Read one expert into disjoint exclusive destinations. No publication occurs here.
    ///
    /// `requested_at` must be captured when the job enters its queue, not when a worker
    /// begins reading. An error may leave partial destination bytes; never mark them ready.
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
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let bytes = usize::try_from(layer.stride())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "expert too large"))?;
        if scratch.as_slice().len() < bytes
            || destinations
                .iter()
                .zip(layer.parts())
                .any(|(d, p)| d.len() as u64 != p.bytes())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "scratch or destination size mismatch",
            ));
        }
        let mut actual = 0;
        let (path, fallback) = {
            #[cfg(target_os = "linux")]
            {
                if self.mode == OpenMode::Direct {
                    match linux::scatter(
                        &self.file,
                        self.alignment,
                        layer,
                        offset,
                        &mut destinations,
                        scratch.as_mut_slice(),
                        &mut actual,
                    ) {
                        Ok(true) => (ReadPath::DirectScatter, None),
                        Ok(false) => (ReadPath::DirectStaging, Some(Fallback::Alignment)),
                        Err(e)
                            if matches!(
                                e.raw_os_error(),
                                Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
                            ) =>
                        {
                            (ReadPath::DirectStaging, Some(Fallback::Unsupported))
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    (ReadPath::BufferedStaging, None)
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                (ReadPath::BufferedStaging, None)
            }
        };
        #[cfg(not(target_os = "linux"))]
        let _ = self.mode;
        if path != ReadPath::DirectScatter {
            #[cfg(target_os = "linux")]
            if self.mode == OpenMode::Direct {
                linux::validate_staging(self.alignment, offset, &scratch.as_slice()[..bytes])?;
            }
            read_exact_at(
                &self.file,
                &mut scratch.as_mut_slice()[..bytes],
                offset,
                &mut actual,
            )?;
            for (dst, part) in destinations.iter_mut().zip(layer.parts()) {
                let start = part.offset() as usize;
                dst.copy_from_slice(&scratch.as_slice()[start..start + dst.len()]);
            }
        }
        let ready = Instant::now();
        Ok(ReadReport {
            path,
            fallback,
            bytes_read: actual,
            queue_wait: started - requested_at,
            service_time: ready - started,
            request_to_ready: ready - requested_at,
        })
    }
}

fn read_exact_at(
    file: &File,
    mut buffer: &mut [u8],
    mut offset: u64,
    actual: &mut u64,
) -> io::Result<()> {
    while !buffer.is_empty() {
        match file.read_at(buffer, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short expert read",
                ));
            }
            Ok(n) => {
                offset += n as u64;
                *actual += n as u64;
                buffer = &mut buffer[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
