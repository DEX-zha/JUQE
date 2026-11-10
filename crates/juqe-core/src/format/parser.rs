use std::str::FromStr;

use super::{
    ExpertIndex, FormatError, Layer, MAX_EXPERT_BYTES, MAX_EXPERTS, MAX_INDEX_BYTES, MAX_LAYERS,
    PACK_ALIGNMENT, Part, error::invalid,
};

const HEADER_FIELDS: usize = 2;
const LAYER_FIELDS: usize = 18;
const MATRIX_FIELDS: usize = 5;

impl FromStr for ExpertIndex {
    type Err = FormatError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.len() > MAX_INDEX_BYTES {
            return Err(invalid("index exceeds size limit"));
        }
        let mut lines = text.lines();
        let header = parse_fields(lines.next(), HEADER_FIELDS, 1)?;
        let layer_count = header[0];
        let expert_count = header[1];
        if !(1..=MAX_LAYERS as u64).contains(&layer_count)
            || !(1..=MAX_EXPERTS as u64).contains(&expert_count)
        {
            return Err(invalid("layer/expert count outside supported limits"));
        }

        let experts = expert_count as usize;
        let mut layers = Vec::with_capacity(layer_count as usize);
        let mut file_bytes = 0;
        for layer_id in 0..layer_count as usize {
            let line = layer_id + 2;
            let fields = parse_fields(lines.next(), LAYER_FIELDS, line)?;
            let (layer, end) = parse_layer(&fields, layer_id, experts, file_bytes)
                .map_err(|error| invalid(format!("line {line}: {error}")))?;
            layers.push(layer);
            file_bytes = end;
        }
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(invalid("unexpected trailing index data"));
        }
        Ok(Self {
            experts,
            layers,
            file_bytes,
        })
    }
}

fn parse_fields(
    line: Option<&str>,
    expected: usize,
    number: usize,
) -> Result<Vec<u64>, FormatError> {
    let line = line.ok_or_else(|| invalid(format!("line {number}: missing record")))?;
    let fields: Vec<_> = line.split_whitespace().take(expected + 1).collect();
    if fields.len() != expected {
        return Err(invalid(format!(
            "line {number}: expected {expected} fields"
        )));
    }
    fields
        .into_iter()
        .map(|field| {
            field
                .parse()
                .map_err(|_| invalid(format!("line {number}: invalid unsigned integer")))
        })
        .collect()
}

fn parse_layer(
    fields: &[u64],
    layer_id: usize,
    experts: usize,
    previous_end: u64,
) -> Result<(Layer, u64), FormatError> {
    let stored_id = fields[0];
    let offset = fields[1];
    let stride = fields[2];
    if stored_id != layer_id as u64
        || !offset.is_multiple_of(PACK_ALIGNMENT)
        || !stride.is_multiple_of(PACK_ALIGNMENT)
        || offset < previous_end
    {
        return Err(invalid(
            "invalid layer order, alignment, or overlapping file regions",
        ));
    }
    if stride > MAX_EXPERT_BYTES {
        return Err(invalid("expert block exceeds allocation limit"));
    }

    let mut parts = Vec::with_capacity(3);
    let mut previous_part_end = 0;
    for matrix in fields[3..].as_chunks::<MATRIX_FIELDS>().0 {
        let part = parse_part(matrix, stride, previous_part_end)?;
        // parse_part checked this sum, including the all-zero dense placeholder.
        previous_part_end = part.offset + part.bytes;
        parts.push(part);
    }
    let end = stride
        .checked_mul(experts as u64)
        .and_then(|bytes| offset.checked_add(bytes))
        .filter(|end| *end <= i64::MAX as u64)
        .ok_or_else(|| invalid("file extent exceeds positional I/O range"))?;
    let parts = parts
        .try_into()
        .map_err(|_| invalid("expected three matrices"))?;
    Ok((
        Layer {
            offset,
            stride,
            parts,
            experts,
        },
        end,
    ))
}

fn parse_part(
    fields: &[u64; MATRIX_FIELDS],
    stride: u64,
    previous_end: u64,
) -> Result<Part, FormatError> {
    let kind = u32::try_from(fields[0]).map_err(|_| invalid("ggml type outside u32"))?;
    let dimensions = [fields[1], fields[2]];
    let offset = fields[3];
    let bytes = fields[4];
    if stride == 0 {
        if fields.iter().any(|&field| field != 0) {
            return Err(invalid("dense layer has expert data"));
        }
    } else if dimensions.contains(&0)
        || bytes == 0
        || offset < previous_end
        || offset.checked_add(bytes).is_none_or(|end| end > stride)
    {
        return Err(invalid("invalid matrix bounds or overlap"));
    }
    Ok(Part {
        kind,
        dimensions,
        offset,
        bytes,
    })
}
