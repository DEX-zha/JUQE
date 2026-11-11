use juqe_io::{AlignedBuffer, ReadReport};
use std::{io, time::Instant};

/// Destination storage shared by runtime workers and held alive by batch leases.
///
/// Implementations must write only the requested slot and return success only when
/// all three matrices are present. The runtime prevents concurrent writes to the
/// same slot and prevents writes while a batch holds its lease. Implementations
/// remain responsible for bounds checks and their own memory safety.
pub trait ExpertStorage: Send + Sync {
    /// Load one expert into the reserved slot, using worker-private scratch.
    ///
    /// `requested_at` precedes queue wait. No destination is published until this
    /// method returns successfully. Errors may leave partial destination bytes.
    fn load(
        &self,
        layer: usize,
        expert: usize,
        slot: usize,
        scratch: &mut AlignedBuffer,
        requested_at: Instant,
    ) -> io::Result<ReadReport>;
}
