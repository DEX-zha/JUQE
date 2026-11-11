use crate::{ExpertStorage, RuntimeError};
use juqe_core::cache::{Cache, CacheError, Lease, LoadTicket, Request};
use std::{sync::Arc, time::Duration};

/// Slot assignments protected until the batch is dropped after GPU completion.
///
/// Dropping this object does not wait for a GPU fence. The engine must synchronize
/// consumption first. A batch keeps its storage alive even if its runtime is dropped.
pub struct Batch {
    pub(crate) slots: Vec<usize>,
    pub(crate) report: BatchReport,
    pub(crate) _leases: Vec<Option<Lease>>,
    pub(crate) _storage: Arc<dyn ExpertStorage>,
}

impl Batch {
    /// Slots in the same order as the expert identifiers passed to `prepare`.
    pub fn slots(&self) -> &[usize] {
        &self.slots
    }

    /// Loading counters and full preparation time for this batch.
    pub fn report(&self) -> &BatchReport {
        &self.report
    }
}

/// Counters for one prepared batch; expert counts refer to distinct identifiers.
#[derive(Debug, Default)]
pub struct BatchReport {
    /// Distinct requested experts that were already ready.
    pub cache_hits: usize,
    /// Distinct experts loaded for this batch.
    pub loads: usize,
    /// Actual bytes read, including partial reads before fallback.
    pub bytes_read: u64,
    /// Submission through successful cache publication, including queue wait.
    pub request_to_ready: Duration,
}

pub(crate) struct Reservations {
    pub leases: Vec<Option<Lease>>,
    pub tickets: Vec<LoadTicket>,
    pub hits: usize,
}

impl Reservations {
    pub fn new(
        cache: &mut Cache,
        layer: usize,
        experts: &[usize],
        count: usize,
        capacity: usize,
    ) -> Result<Self, RuntimeError> {
        let mut requested = vec![false; count];
        let mut unique = Vec::new();
        for &expert in experts {
            if expert >= count {
                return Err(RuntimeError::Cache {
                    layer,
                    source: CacheError::InvalidExpert {
                        expert,
                        experts: count,
                    },
                });
            }
            if !requested[expert] {
                requested[expert] = true;
                unique.push(expert);
            }
        }
        if unique.len() > capacity {
            return Err(RuntimeError::BatchTooLarge {
                layer,
                requested: unique.len(),
                capacity,
            });
        }
        let mut reservations = Self {
            leases: vec![None; count],
            tickets: Vec::new(),
            hits: 0,
        };
        // Protect every hit before a missing expert can evict any requested slot.
        for &expert in &unique {
            let lease = cache
                .ready(expert)
                .map_err(|source| RuntimeError::Cache { layer, source })?;
            if lease.is_some() {
                reservations.hits += 1;
            }
            reservations.leases[expert] = lease;
        }
        for expert in unique {
            if reservations.leases[expert].is_some() {
                continue;
            }
            match cache.request(expert) {
                Ok(Request::Load(ticket)) => reservations.tickets.push(ticket),
                Ok(Request::Ready(lease)) => reservations.leases[expert] = Some(lease),
                result => {
                    // No I/O has been submitted, so every reservation can be released.
                    for ticket in &reservations.tickets {
                        cache
                            .fail(ticket)
                            .map_err(|source| RuntimeError::Cache { layer, source })?;
                    }
                    let source = match result {
                        Err(error) => error,
                        _ => CacheError::Busy,
                    };
                    return Err(RuntimeError::Cache { layer, source });
                }
            }
        }
        Ok(reservations)
    }
}
