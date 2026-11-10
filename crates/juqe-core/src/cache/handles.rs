use std::sync::Arc;

/// Permission to publish one load, bound to its cache, slot, and generation.
#[derive(Debug)]
#[must_use = "complete or fail the ticket after its writer has stopped"]
pub struct LoadTicket {
    pub(super) owner: u64,
    pub(super) slot: usize,
    pub(super) generation: u64,
    pub(super) expert: usize,
}

impl LoadTicket {
    /// Destination slot reserved for the writer.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Expert assigned to this write.
    pub fn expert(&self) -> usize {
        self.expert
    }
}

/// A ready slot protected until all clones are dropped after consumption.
#[derive(Debug, Clone)]
#[must_use = "hold the lease until the consumer completes"]
pub struct Lease {
    slot: usize,
    _pin: Arc<()>,
}

impl Lease {
    pub(super) fn new(slot: usize, pin: &Arc<()>) -> Self {
        Self {
            slot,
            _pin: Arc::clone(pin),
        }
    }

    /// Slot whose bytes may be consumed.
    pub fn slot(&self) -> usize {
        self.slot
    }
}

/// Result of requesting an expert without blocking.
#[derive(Debug)]
pub enum Request {
    /// The expert is ready. Retain the lease until consumption ends.
    Ready(Lease),
    /// Submit this load, then complete or fail it after the writer returns.
    Load(LoadTicket),
    /// This expert is already loading. No duplicate read is needed.
    Pending,
}
