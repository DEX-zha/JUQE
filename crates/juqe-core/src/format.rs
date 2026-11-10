//! Checked reader for the prototype's text `experts.idx` format.
use std::{error::Error, fmt, str::FromStr};

/// Maximum text index accepted by this importer (four MiB).
pub const MAX_INDEX_BYTES: usize = 4 * 1024 * 1024;
/// Maximum block allocated by the first runtime (256 MiB per expert).
pub const MAX_EXPERT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_LAYERS: usize = 512;
const MAX_EXPERTS: usize = 4096;

/// Invalid or unsupported expert metadata, with source-line context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError(String);
impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Error for FormatError {}
fn invalid(message: impl Into<String>) -> FormatError {
    FormatError(message.into())
}

/// A quantized matrix within one packed expert. Type codes are opaque ggml codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    kind: u32,
    dimensions: [u64; 2],
    offset: u64,
    bytes: u64,
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
    offset: u64,
    stride: u64,
    parts: [Part; 3],
    experts: usize,
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

/// Immutable validated index; constructing arbitrary unchecked layers is not exposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertIndex {
    experts: usize,
    layers: Vec<Layer>,
    file_bytes: u64,
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

impl FromStr for ExpertIndex {
    type Err = FormatError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.len() > MAX_INDEX_BYTES {
            return Err(invalid("index exceeds size limit"));
        }
        let mut lines = text.lines();
        let parse = |line: &str, expected: usize, number: usize| -> Result<Vec<u64>, FormatError> {
            let fields: Vec<_> = line.split_whitespace().take(expected + 1).collect();
            if fields.len() != expected {
                return Err(invalid(format!(
                    "line {number}: expected {expected} fields"
                )));
            }
            fields
                .iter()
                .map(|s| {
                    s.parse::<u64>()
                        .map_err(|_| invalid(format!("line {number}: invalid unsigned integer")))
                })
                .collect()
        };
        let header = parse(lines.next().ok_or_else(|| invalid("missing header"))?, 2, 1)?;
        if !(1..=MAX_LAYERS as u64).contains(&header[0])
            || !(1..=MAX_EXPERTS as u64).contains(&header[1])
        {
            return Err(invalid("layer/expert count outside supported limits"));
        }
        let experts = header[1] as usize;
        let mut layers = Vec::new();
        let mut end = 0_u64;
        for l in 0..header[0] as usize {
            let row = parse(
                lines.next().ok_or_else(|| invalid("missing layer"))?,
                18,
                l + 2,
            )?;
            if row[0] != l as u64 || row[1] % 4096 != 0 || row[2] % 4096 != 0 || row[1] < end {
                return Err(invalid(format!(
                    "layer {l}: invalid order, alignment or overlapping file regions"
                )));
            }
            if row[2] > MAX_EXPERT_BYTES {
                return Err(invalid(format!(
                    "layer {l}: expert block exceeds allocation limit"
                )));
            }
            let mut parts = Vec::new();
            let mut part_end = 0;
            for p in 0..3 {
                let f = &row[3 + p * 5..8 + p * 5];
                let kind = u32::try_from(f[0]).map_err(|_| invalid("ggml type outside u32"))?;
                if row[2] == 0 {
                    if f.iter().any(|&v| v != 0) {
                        return Err(invalid("dense layer has expert data"));
                    }
                } else if f[1] == 0
                    || f[2] == 0
                    || f[4] == 0
                    || f[3] < part_end
                    || f[3].checked_add(f[4]).is_none_or(|v| v > row[2])
                {
                    return Err(invalid(format!(
                        "layer {l}: invalid matrix bounds or overlap"
                    )));
                }
                part_end = f[3] + f[4]; // checked above, or all zero for dense
                parts.push(Part {
                    kind,
                    dimensions: [f[1], f[2]],
                    offset: f[3],
                    bytes: f[4],
                });
            }
            end = row[2]
                .checked_mul(experts as u64)
                .and_then(|n| row[1].checked_add(n))
                .filter(|n| *n <= i64::MAX as u64)
                .ok_or_else(|| invalid("file extent exceeds positional I/O range"))?;
            let parts: [Part; 3] = parts
                .try_into()
                .map_err(|_| invalid("expected three matrices"))?;
            layers.push(Layer {
                offset: row[1],
                stride: row[2],
                parts,
                experts,
            });
        }
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(invalid("unexpected trailing index data"));
        }
        Ok(Self {
            experts,
            layers,
            file_bytes: end,
        })
    }
}
