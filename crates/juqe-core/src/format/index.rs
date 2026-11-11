use super::{FormatError, Layer, error::invalid};

/// Immutable validated index; constructing arbitrary unchecked layers is not exposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertIndex {
    pub(super) experts: usize,
    pub(super) layers: Vec<Layer>,
    pub(super) file_bytes: u64,
}
impl ExpertIndex {
    /// Number of experts in each routed layer.
    pub fn experts(&self) -> usize {
        self.experts
    }
    /// Layers in model order, including dense placeholders.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
    /// Minimum backing file length; gaps between layers are permitted.
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    /// Reject a backing file shorter than any declared expert extent.
    pub fn validate_file_size(&self, bytes: u64) -> Result<(), FormatError> {
        if bytes < self.file_bytes {
            return Err(invalid("expert file is truncated"));
        }
        Ok(())
    }
}
