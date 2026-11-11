use super::{FormatError, error::invalid};

/// A quantized matrix within one packed expert. Type codes are opaque ggml codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub(super) kind: u32,
    pub(super) dimensions: [u64; 2],
    pub(super) offset: u64,
    pub(super) bytes: u64,
}
impl Part {
    /// Original ggml type identifier; this importer does not decode quantization.
    pub fn kind(&self) -> u32 {
        self.kind
    }
    /// Matrix dimensions as stored by the prototype.
    pub fn dimensions(&self) -> [u64; 2] {
        self.dimensions
    }
    /// Start relative to the expert block.
    pub fn offset(&self) -> u64 {
        self.offset
    }
    /// Matrix payload length, excluding padding.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Validated expert layout for a layer. Zero stride denotes a dense layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub(super) offset: u64,
    pub(super) stride: u64,
    pub(super) parts: [Part; 3],
    pub(super) experts: usize,
}
impl Layer {
    /// File offset of this layer's first expert.
    pub fn offset(&self) -> u64 {
        self.offset
    }
    /// Total block length, including padding (zero for dense layers).
    pub fn stride(&self) -> u64 {
        self.stride
    }
    /// Matrices in gate, up, down order.
    pub fn parts(&self) -> &[Part; 3] {
        &self.parts
    }
    /// Absolute block offset, checked against the expert count and dense layers.
    pub fn expert_offset(&self, expert: usize) -> Result<u64, FormatError> {
        if self.stride == 0 || expert >= self.experts {
            return Err(invalid("expert outside routed layer"));
        }
        // The parser checks the entire layer extent; keep arithmetic checked here too.
        (expert as u64)
            .checked_mul(self.stride)
            .and_then(|x| self.offset.checked_add(x))
            .ok_or_else(|| invalid("expert offset overflow"))
    }
}
