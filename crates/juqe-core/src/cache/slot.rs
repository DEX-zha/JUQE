use std::sync::Arc;

#[derive(Debug, Default)]
pub(super) enum SlotState {
    #[default]
    Empty,
    Loading {
        expert: usize,
    },
    Ready {
        expert: usize,
        pin: Arc<()>,
    },
}

#[derive(Debug, Default)]
pub(super) struct Slot {
    pub state: SlotState,
    pub generation: u64,
    pub last_used: u64,
}

impl Slot {
    pub fn expert(&self) -> Option<usize> {
        match self.state {
            SlotState::Empty => None,
            SlotState::Loading { expert } | SlotState::Ready { expert, .. } => Some(expert),
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self.state, SlotState::Empty)
    }

    pub fn is_evictable(&self) -> bool {
        matches!(&self.state, SlotState::Ready { pin, .. } if Arc::strong_count(pin) == 1)
    }
}
