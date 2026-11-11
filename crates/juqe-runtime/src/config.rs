use crate::RuntimeError;
use juqe_core::format::ExpertIndex;

/// Resource limits for one runtime instance.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of I/O workers, from 1 through 128.
    pub workers: usize,
    /// Maximum jobs waiting for a worker, from 1 through 4096.
    pub queue_capacity: usize,
    /// Slots for each layer; dense layers require zero slots.
    pub layer_slots: Vec<usize>,
}

impl Config {
    pub(crate) fn validate(&self, index: &ExpertIndex) -> Result<(), RuntimeError> {
        if !(1..=128).contains(&self.workers) {
            return Err(RuntimeError::Configuration("workers must be in 1..=128"));
        }
        if !(1..=4096).contains(&self.queue_capacity) {
            return Err(RuntimeError::Configuration(
                "queue capacity must be in 1..=4096",
            ));
        }
        if self.layer_slots.len() != index.layers().len() {
            return Err(RuntimeError::Configuration(
                "slot counts must cover every layer",
            ));
        }
        for (layout, &slots) in index.layers().iter().zip(&self.layer_slots) {
            if (layout.stride() == 0 && slots != 0)
                || (layout.stride() != 0 && !(1..=index.experts()).contains(&slots))
            {
                return Err(RuntimeError::Configuration(
                    "invalid slot count for layer layout",
                ));
            }
        }
        Ok(())
    }
}
