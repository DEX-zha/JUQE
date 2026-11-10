use std::{error::Error, fmt};

/// A rejected cache operation leaves protected slots intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    /// Counts must satisfy `1 <= slots <= experts <= MAX_EXPERTS`.
    InvalidCapacity {
        /// Requested expert count.
        experts: usize,
        /// Requested slot count.
        slots: usize,
    },
    /// An expert identifier is outside the cache's configured range.
    InvalidExpert {
        /// Requested expert identifier.
        expert: usize,
        /// Configured expert count.
        experts: usize,
    },
    /// Every slot is loading or leased; the caller must apply backpressure.
    Busy,
    /// The completion ticket belongs to another cache.
    ForeignTicket,
    /// The ticket no longer identifies the slot's current load.
    StaleTicket,
    /// A monotonic cache identifier, generation, or access counter would wrap.
    SequenceExhausted,
}

impl fmt::Display for CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity { experts, slots } => write!(
                formatter,
                "invalid cache capacity: {slots} slots for {experts} experts"
            ),
            Self::InvalidExpert { expert, experts } => {
                write!(formatter, "expert {expert} is outside 0..{experts}")
            }
            Self::Busy => formatter.write_str("all cache slots are loading or leased"),
            Self::ForeignTicket => formatter.write_str("load ticket belongs to another cache"),
            Self::StaleTicket => {
                formatter.write_str("load ticket no longer identifies an active load")
            }
            Self::SequenceExhausted => formatter.write_str("cache sequence counter exhausted"),
        }
    }
}

impl Error for CacheError {}
