use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::{
    CacheError, Lease, LoadTicket, Request,
    slot::{Slot, SlotState},
};
use crate::format::MAX_EXPERTS;

static NEXT_CACHE: AtomicU64 = AtomicU64::new(1);

/// Single-owner cache metadata. The scheduler supplies synchronization and buffers.
#[derive(Debug)]
pub struct Cache {
    owner: u64,
    slots: Vec<Slot>,
    expert_slots: Vec<Option<usize>>,
    clock: u64,
}

impl Cache {
    /// Allocate metadata for `1 <= slots <= experts <= MAX_EXPERTS`.
    ///
    /// # Errors
    /// Returns [`CacheError::InvalidCapacity`] for invalid counts or
    /// [`CacheError::SequenceExhausted`] if cache identifiers are exhausted.
    pub fn new(experts: usize, slots: usize) -> Result<Self, CacheError> {
        if experts == 0 || experts > MAX_EXPERTS || slots == 0 || slots > experts {
            return Err(CacheError::InvalidCapacity { experts, slots });
        }
        let owner = NEXT_CACHE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| CacheError::SequenceExhausted)?;
        Ok(Self {
            owner,
            slots: (0..slots).map(|_| Slot::default()).collect(),
            expert_slots: vec![None; experts],
            clock: 0,
        })
    }

    /// Look up, coalesce, or reserve a load without evicting protected slots.
    ///
    /// # Errors
    /// Returns [`CacheError::InvalidExpert`] for an unknown expert,
    /// [`CacheError::Busy`] when every slot is protected, or
    /// [`CacheError::SequenceExhausted`] before an access counter can wrap.
    pub fn request(&mut self, expert: usize) -> Result<Request, CacheError> {
        if expert >= self.expert_slots.len() {
            return Err(CacheError::InvalidExpert {
                expert,
                experts: self.expert_slots.len(),
            });
        }
        if let Some(slot_id) = self.expert_slots[expert] {
            if matches!(self.slots[slot_id].state, SlotState::Loading { .. }) {
                return Ok(Request::Pending);
            }
            let now = self.tick()?;
            let slot = &mut self.slots[slot_id];
            if let SlotState::Ready { pin, .. } = &slot.state {
                slot.last_used = now;
                return Ok(Request::Ready(Lease::new(slot_id, pin)));
            }
        }
        let slot_id = self.victim().ok_or(CacheError::Busy)?;
        self.reserve(slot_id, expert).map(Request::Load)
    }

    /// Publish a successful complete write and lease its slot to the caller.
    ///
    /// Call only after the writer has stopped. This method does not wait for I/O.
    ///
    /// # Errors
    /// Rejects tickets from another cache and tickets already completed or failed.
    pub fn complete(&mut self, ticket: &LoadTicket) -> Result<Lease, CacheError> {
        self.validate(ticket)?;
        let pin = Arc::new(());
        let lease = Lease::new(ticket.slot, &pin);
        self.slots[ticket.slot].state = SlotState::Ready {
            expert: ticket.expert,
            pin,
        };
        Ok(lease)
    }

    /// Discard a failed load after the writer has stopped.
    ///
    /// Partial bytes remain inaccessible. Dropping a ticket alone does not release
    /// its slot because the cache cannot know whether its writer is still active.
    ///
    /// # Errors
    /// Rejects tickets from another cache and tickets already completed or failed.
    pub fn fail(&mut self, ticket: &LoadTicket) -> Result<(), CacheError> {
        self.validate(ticket)?;
        self.expert_slots[ticket.expert] = None;
        self.slots[ticket.slot].state = SlotState::Empty;
        Ok(())
    }

    fn victim(&self) -> Option<usize> {
        self.slots.iter().position(Slot::is_empty).or_else(|| {
            self.slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.is_evictable())
                .min_by_key(|(_, slot)| slot.last_used)
                .map(|(id, _)| id)
        })
    }

    fn reserve(&mut self, slot_id: usize, expert: usize) -> Result<LoadTicket, CacheError> {
        let generation = self.slots[slot_id]
            .generation
            .checked_add(1)
            .ok_or(CacheError::SequenceExhausted)?;
        let now = self.tick()?;
        if let Some(previous) = self.slots[slot_id].expert() {
            self.expert_slots[previous] = None;
        }
        self.slots[slot_id] = Slot {
            state: SlotState::Loading { expert },
            generation,
            last_used: now,
        };
        self.expert_slots[expert] = Some(slot_id);
        Ok(LoadTicket {
            owner: self.owner,
            slot: slot_id,
            generation,
            expert,
        })
    }

    fn validate(&self, ticket: &LoadTicket) -> Result<(), CacheError> {
        if ticket.owner != self.owner {
            return Err(CacheError::ForeignTicket);
        }
        let slot = self.slots.get(ticket.slot).ok_or(CacheError::StaleTicket)?;
        if ticket.generation != slot.generation
            || !matches!(slot.state, SlotState::Loading { expert } if expert == ticket.expert)
        {
            return Err(CacheError::StaleTicket);
        }
        Ok(())
    }

    fn tick(&mut self) -> Result<u64, CacheError> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or(CacheError::SequenceExhausted)?;
        Ok(self.clock)
    }
}
