use crate::{LayerPool, boundary::invalid};
use juqe_core::format::ExpertIndex;
use std::{io, ptr::NonNull};

pub(super) struct Region {
    pointer: NonNull<u8>,
    bytes: usize,
}

impl Region {
    pub unsafe fn new(pointer: *mut u8, bytes: usize) -> io::Result<Self> {
        let pointer = NonNull::new(pointer).ok_or_else(|| invalid("pool pointer is null"))?;
        Ok(Self { pointer, bytes })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: construction verifies lengths and disjoint regions; the engine
        // keeps the allocation live. A slot mutex excludes other writers, and the
        // runtime never submits a write while a GPU consumer holds the slot lease.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.bytes) }
    }
}

// SAFETY: regions are engine-owned stable allocations. Moving the descriptor does
// not move bytes. Exclusive slot mutexes and runtime leases govern every access.
unsafe impl Send for Region {}

pub(super) fn validate(index: &ExpertIndex, pools: &[LayerPool]) -> io::Result<()> {
    if pools.len() != index.layers().len() {
        return Err(invalid("pool count differs from layer count"));
    }
    let mut ranges = Vec::new();
    for (layout, pool) in index.layers().iter().zip(pools) {
        if layout.stride() == 0 {
            if pool.slots != 0
                || pool
                    .parts
                    .iter()
                    .any(|part| !part.data.is_null() || part.bytes != 0)
            {
                return Err(invalid("dense layers must not have expert pools"));
            }
            continue;
        }
        if !(1..=index.experts()).contains(&(pool.slots as usize)) {
            return Err(invalid("invalid layer slot count"));
        }
        for (matrix, part) in layout.parts().iter().zip(&pool.parts) {
            let required = matrix
                .bytes()
                .checked_mul(pool.slots as u64)
                .ok_or_else(|| invalid("pool extent overflow"))?;
            if part.data.is_null() || part.bytes < required || part.bytes > isize::MAX as u64 {
                return Err(invalid(
                    "pool is null, too small, or outside the addressable range",
                ));
            }
            let start = part.data as usize;
            let end = start
                .checked_add(part.bytes as usize)
                .ok_or_else(|| invalid("pool address overflow"))?;
            ranges.push(start..end);
        }
    }
    ranges.sort_by_key(|range| range.start);
    if ranges.windows(2).any(|pair| pair[0].end > pair[1].start) {
        return Err(invalid("expert pools overlap"));
    }
    Ok(())
}
