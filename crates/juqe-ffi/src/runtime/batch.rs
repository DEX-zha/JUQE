use super::RuntimeHandle;
use crate::{BatchInfo, boundary};

const MAX_BATCH_ENTRIES: usize = 1_048_576;

/// Prepare a complete expert batch and retain its slot leases in the runtime.
///
/// # Safety
/// `handle` must be live and exclusively borrowed. `experts` must contain `count`
/// initialized IDs; `slots` must be writable for `count` IDs. `info` must be writable
/// and disjoint from all inputs and outputs. No active batch may be overwritten.
/// Slot buffers may be consumed only after success and before batch release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_runtime_prepare(
    handle: *mut RuntimeHandle,
    layer: u32,
    experts: *const i32,
    count: usize,
    slots: *mut i32,
    info: *mut BatchInfo,
) -> i32 {
    boundary::run(|| {
        // SAFETY: the caller supplies a live exclusively borrowed handle.
        let handle = unsafe { handle.as_mut() }
            .ok_or_else(|| boundary::invalid("runtime handle is null"))?;
        if handle.active.is_some() {
            return Err(
                boundary::invalid("release the previous batch after GPU completion").into(),
            );
        }
        if count == 0
            || count > MAX_BATCH_ENTRIES
            || experts.is_null()
            || slots.is_null()
            || info.is_null()
        {
            return Err(boundary::invalid("invalid batch pointers or entry count").into());
        }
        // SAFETY: caller provides count initialized IDs. Copying finishes before any
        // output writes, so IDs and slots may share an in-place engine buffer.
        let experts = unsafe { std::slice::from_raw_parts(experts, count) }
            .iter()
            .map(|&expert| {
                usize::try_from(expert).map_err(|_| boundary::invalid("negative expert identifier"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = handle.runtime.prepare(layer as usize, &experts)?;
        for (position, &slot) in batch.slots().iter().enumerate() {
            // SAFETY: caller provides count output entries; slots fit i32 because
            // validated cache capacity is at most 4096. Input borrows have ended.
            unsafe {
                slots.add(position).write(slot as i32);
            }
        }
        let report = batch.report();
        let result = BatchInfo {
            cache_hits: report.cache_hits as u64,
            loads: report.loads as u64,
            bytes_read: report.bytes_read,
            request_to_ready_seconds: report.request_to_ready.as_secs_f64(),
        };
        // SAFETY: caller provides exclusive writable output distinct from other arguments.
        unsafe {
            boundary::write(info, result)?;
        }
        handle.active = Some(batch);
        Ok(())
    })
}

/// Release the active batch after the engine's completion fence. Repeated release is valid.
///
/// # Safety
/// `handle` must be live and exclusively borrowed. All GPU and CPU readers of the
/// active batch's foreign slots must have finished before this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_runtime_release(handle: *mut RuntimeHandle) -> i32 {
    boundary::run(|| {
        // SAFETY: caller supplies exclusive access after synchronizing all consumers.
        let handle = unsafe { handle.as_mut() }
            .ok_or_else(|| boundary::invalid("runtime handle is null"))?;
        handle.active.take();
        Ok(())
    })
}
