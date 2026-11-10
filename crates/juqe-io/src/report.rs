use std::time::Duration;

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
