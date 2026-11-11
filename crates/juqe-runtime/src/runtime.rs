use crate::{
    Batch, Config, ExpertStorage, RuntimeError,
    batch::{BatchReport, Reservations},
    dispatch,
    pool::WorkerPool,
};
use juqe_core::{cache::Cache, format::ExpertIndex};
use std::{sync::Arc, time::Instant};

/// Single-owner control path with a bounded pool of parallel I/O workers.
///
/// Calls to `prepare` are serialized through `&mut self`. Earlier batches may stay
/// alive; their leased slots remain protected. Dropping the runtime joins workers.
pub struct Runtime {
    index: ExpertIndex,
    config: Config,
    caches: Vec<Option<Cache>>,
    pool: WorkerPool,
    storage: Arc<dyn ExpertStorage>,
    stopped: bool,
}

impl Runtime {
    /// Validate resource limits, allocate worker scratch, and start workers.
    ///
    /// # Errors
    /// Rejects invalid layouts or capacities and reports allocation/thread failures.
    pub fn new(
        index: ExpertIndex,
        config: Config,
        storage: Arc<dyn ExpertStorage>,
    ) -> Result<Self, RuntimeError> {
        config.validate(&index)?;
        let scratch_bytes = index
            .layers()
            .iter()
            .map(|layer| layer.stride())
            .max()
            .unwrap_or(0);
        if scratch_bytes == 0 {
            return Err(RuntimeError::Configuration(
                "pack has no routed expert layers",
            ));
        }
        let caches = config
            .layer_slots
            .iter()
            .enumerate()
            .map(|(layer, &slots)| {
                if slots == 0 {
                    Ok(None)
                } else {
                    Cache::new(index.experts(), slots)
                        .map(Some)
                        .map_err(|source| RuntimeError::Cache { layer, source })
                }
            })
            .collect::<Result<_, _>>()?;
        let pool = WorkerPool::new(&config, scratch_bytes as usize, Arc::clone(&storage))
            .map_err(RuntimeError::Worker)?;
        Ok(Self {
            index,
            config,
            caches,
            pool,
            storage,
            stopped: false,
        })
    }

    /// Load all distinct experts and return protected slot assignments.
    ///
    /// The returned slots preserve input order, including repeated expert IDs.
    /// Hold the batch until the inference engine's consumers have completed.
    ///
    /// # Errors
    /// Rejects invalid identifiers and batches exceeding capacity before submitting
    /// I/O. On load failure, drains all submitted writes before returning an error.
    pub fn prepare(&mut self, layer: usize, experts: &[usize]) -> Result<Batch, RuntimeError> {
        if self.stopped {
            return Err(RuntimeError::Stopped);
        }
        let requested_at = Instant::now();
        let cache = self
            .caches
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or(RuntimeError::InvalidLayer(layer))?;
        let reservations = Reservations::new(
            cache,
            layer,
            experts,
            self.index.experts(),
            self.config.layer_slots[layer],
        )?;
        let mut leases = reservations.leases;
        let mut report = BatchReport {
            cache_hits: reservations.hits,
            ..BatchReport::default()
        };
        if let Err(error) = dispatch::load(
            &mut self.pool,
            cache,
            layer,
            reservations.tickets,
            &mut leases,
            requested_at,
            &mut report,
        ) {
            if matches!(error, RuntimeError::Worker(_) | RuntimeError::Stopped) {
                self.stopped = true;
            }
            return Err(error);
        }
        let slots = experts
            .iter()
            .map(|&expert| {
                leases[expert]
                    .as_ref()
                    .map(|lease| lease.slot())
                    .ok_or(RuntimeError::Stopped)
            })
            .collect::<Result<Vec<_>, _>>()?;
        report.request_to_ready = requested_at.elapsed();
        Ok(Batch {
            slots,
            report,
            _leases: leases,
            _storage: Arc::clone(&self.storage),
        })
    }

    /// Stop submissions and join every worker after its queued writes finish.
    ///
    /// Existing batches retain their leases and storage. Repeated calls are harmless.
    pub fn shutdown(&mut self) {
        self.stopped = true;
        self.pool.shutdown();
    }
}
