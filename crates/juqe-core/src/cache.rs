//! Per-layer LRU metadata. Callers must hold leases until GPU consumers finish.
//!
//! This module neither owns weight buffers nor submits I/O. Completing/failing a load
//! is legal only after its writer has stopped. The future backend must enforce that fence.
use std::{
    error::Error,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_CACHE: AtomicU64 = AtomicU64::new(1);

/// Cache operation rejected without evicting a protected slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    /// Invalid count, expert identifier or handle.
    Invalid,
    /// Every slot is loading or leased; caller must apply backpressure.
    Busy,
    /// A monotonic identifier would wrap.
    Capacity,
}
impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cache: {self:?}")
    }
}
impl Error for CacheError {}

/// Exclusive permission to publish one load. Not cloneable; tied to a cache and generation.
#[derive(Debug)]
#[must_use = "complete or fail the ticket after its writer has stopped"]
pub struct LoadTicket {
    owner: u64,
    slot: usize,
    generation: u64,
    expert: usize,
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

/// A ready slot pinned until all clones are dropped after consumption completes.
#[derive(Debug, Clone)]
#[must_use = "hold the lease until the consumer completes"]
pub struct Lease {
    slot: usize,
    _pin: Arc<()>,
}
impl Lease {
    /// Slot whose bytes may be consumed.
    pub fn slot(&self) -> usize {
        self.slot
    }
}

/// Outcome of requesting an expert.
#[derive(Debug)]
pub enum Request {
    /// Bytes were already ready; retain this lease until consumption ends.
    Ready(Lease),
    /// Submit one load and publish or fail it when its writer returns.
    Load(LoadTicket),
    /// Another request is loading this expert; no duplicate read is needed.
    Pending,
}

#[derive(Debug, Clone, Default)]
struct Slot {
    expert: Option<usize>,
    ready: bool,
    generation: u64,
    last: u64,
    pin: Arc<()>,
}

/// Single-owner metadata state machine. Synchronization belongs to the scheduler.
#[derive(Debug)]
pub struct Cache {
    owner: u64,
    slots: Vec<Slot>,
    map: Vec<Option<usize>>,
    clock: u64,
}
impl Cache {
    /// Allocate bounded metadata (at most 4096 experts and slots per layer).
    pub fn new(experts: usize, slots: usize) -> Result<Self, CacheError> {
        if experts == 0 || experts > 4096 || slots == 0 || slots > experts {
            return Err(CacheError::Invalid);
        }
        let owner = NEXT_CACHE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| CacheError::Capacity)?;
        Ok(Self {
            owner,
            slots: vec![Slot::default(); slots],
            map: vec![None; experts],
            clock: 0,
        })
    }
    fn tick(&mut self) -> Result<u64, CacheError> {
        self.clock = self.clock.checked_add(1).ok_or(CacheError::Capacity)?;
        Ok(self.clock)
    }
    /// Look up, coalesce, or reserve a load. A busy cache never evicts leased/loading slots.
    pub fn request(&mut self, expert: usize) -> Result<Request, CacheError> {
        if expert >= self.map.len() {
            return Err(CacheError::Invalid);
        }
        if let Some(s) = self.map[expert] {
            if !self.slots[s].ready {
                return Ok(Request::Pending);
            }
            let now = self.tick()?;
            let slot = &mut self.slots[s];
            slot.last = now;
            return Ok(Request::Ready(Lease {
                slot: s,
                _pin: Arc::clone(&slot.pin),
            }));
        }
        let victim = self
            .slots
            .iter()
            .position(|s| s.expert.is_none())
            .or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.ready && Arc::strong_count(&s.pin) == 1)
                    .min_by_key(|(_, s)| s.last)
                    .map(|(i, _)| i)
            })
            .ok_or(CacheError::Busy)?;
        let generation = self.slots[victim]
            .generation
            .checked_add(1)
            .ok_or(CacheError::Capacity)?;
        let now = self.tick()?;
        if let Some(old) = self.slots[victim].expert {
            self.map[old] = None;
        }
        self.slots[victim] = Slot {
            expert: Some(expert),
            ready: false,
            generation,
            last: now,
            pin: Arc::new(()),
        };
        self.map[expert] = Some(victim);
        Ok(Request::Load(LoadTicket {
            owner: self.owner,
            slot: victim,
            generation,
            expert,
        }))
    }
    fn validate(&self, ticket: &LoadTicket) -> Result<(), CacheError> {
        let slot = self.slots.get(ticket.slot).ok_or(CacheError::Invalid)?;
        if ticket.owner != self.owner
            || ticket.generation != slot.generation
            || slot.expert != Some(ticket.expert)
            || slot.ready
        {
            return Err(CacheError::Invalid);
        }
        Ok(())
    }
    /// Publish a complete successful write and pin its slot for the requesting consumer.
    pub fn complete(&mut self, ticket: &LoadTicket) -> Result<Lease, CacheError> {
        self.validate(ticket)?;
        let slot = &mut self.slots[ticket.slot];
        slot.ready = true;
        Ok(Lease {
            slot: ticket.slot,
            _pin: Arc::clone(&slot.pin),
        })
    }
    /// Discard a failed write after I/O has stopped; its partial contents remain inaccessible.
    pub fn fail(&mut self, ticket: &LoadTicket) -> Result<(), CacheError> {
        self.validate(ticket)?;
        self.map[ticket.expert] = None;
        let slot = &mut self.slots[ticket.slot];
        slot.expert = None;
        slot.ready = false;
        Ok(())
    }
}
