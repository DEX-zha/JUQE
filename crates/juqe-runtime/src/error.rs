use juqe_core::cache::CacheError;
use std::{error::Error, fmt, io};

/// Failure to initialize or prepare an expert batch.
#[derive(Debug)]
pub enum RuntimeError {
    /// A resource setting is invalid.
    Configuration(&'static str),
    /// The requested layer is dense or outside the pack.
    InvalidLayer(usize),
    /// The batch needs more distinct experts than the layer can hold.
    BatchTooLarge {
        /// Layer identifier.
        layer: usize,
        /// Distinct experts in the batch.
        requested: usize,
        /// Configured slot capacity.
        capacity: usize,
    },
    /// A cache ownership constraint rejected the operation.
    Cache {
        /// Layer identifier.
        layer: usize,
        /// Underlying cache rejection.
        source: CacheError,
    },
    /// A worker could not load an expert.
    Load {
        /// Layer identifier.
        layer: usize,
        /// Expert identifier.
        expert: usize,
        /// Underlying I/O failure.
        source: io::Error,
    },
    /// Worker startup or communication failed.
    Worker(io::Error),
    /// The worker pool is no longer available.
    Stopped,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) => {
                write!(formatter, "invalid runtime configuration: {message}")
            }
            Self::InvalidLayer(layer) => write!(
                formatter,
                "layer {layer} is not a routed layer in this pack"
            ),
            Self::BatchTooLarge {
                layer,
                requested,
                capacity,
            } => write!(
                formatter,
                "layer {layer} needs {requested} experts but has {capacity} slots; increase cache capacity or reduce the microbatch"
            ),
            Self::Cache { layer, source } => write!(formatter, "layer {layer}: {source}"),
            Self::Load {
                layer,
                expert,
                source,
            } => write!(
                formatter,
                "cannot load layer {layer} expert {expert}: {source}"
            ),
            Self::Worker(source) => write!(formatter, "expert worker failed: {source}"),
            Self::Stopped => formatter.write_str("expert runtime is stopped"),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cache { source, .. } => Some(source),
            Self::Load { source, .. } | Self::Worker(source) => Some(source),
            _ => None,
        }
    }
}
