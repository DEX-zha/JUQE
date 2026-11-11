use super::RuntimeHandle;
use crate::{LayerPool, PackHandle, RuntimeConfig, boundary, memory::MemoryStorage};
use juqe_io::OpenMode;
use juqe_runtime::{Config, Runtime};
use std::{ffi::c_char, sync::Arc};

/// Create a runtime borrowing engine-owned destination pools.
///
/// # Safety
/// Input pointers must be live, aligned, and initialized for this call. `pools`
/// must contain `layers` descriptors. Every described allocation must be writable,
/// disjoint, and stable until runtime close. The engine may consume a slot only
/// after prepare succeeds and must synchronize before releasing that batch.
/// `output` must be an exclusive writable pointer slot, disjoint from inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_runtime_create(
    pack: *const PackHandle,
    weights: *const c_char,
    config: *const RuntimeConfig,
    pools: *const LayerPool,
    layers: usize,
    output: *mut *mut RuntimeHandle,
) -> i32 {
    boundary::run(|| {
        // SAFETY: the caller supplies exclusive writable output storage.
        unsafe {
            boundary::write(output, std::ptr::null_mut())?;
        }
        // SAFETY: inputs satisfy the exported function's lifetime and alignment contract.
        let (pack, weights, config) = unsafe {
            (
                boundary::input(pack)?,
                boundary::path(weights)?,
                boundary::input(config)?,
            )
        };
        if layers != pack.index.layers().len() || pools.is_null() {
            return Err(boundary::invalid("pool descriptors must cover every layer").into());
        }
        if !(1..=128).contains(&config.workers) || !(1..=4096).contains(&config.queue_capacity) {
            return Err(boundary::invalid("invalid worker count or queue capacity").into());
        }
        let mode = match config.direct {
            0 => OpenMode::Buffered,
            1 => OpenMode::Direct,
            _ => return Err(boundary::invalid("direct mode must be zero or one").into()),
        };
        // SAFETY: layer count is bounded by the validated index; caller provides that
        // many initialized descriptors. Their lifetime covers this construction.
        let pools = unsafe { std::slice::from_raw_parts(pools, layers) };
        let runtime_config = Config {
            workers: config.workers as usize,
            queue_capacity: config.queue_capacity as usize,
            layer_slots: pools.iter().map(|pool| pool.slots as usize).collect(),
        };
        // SAFETY: the engine retains pool allocations and follows the batch protocol.
        // MemoryStorage additionally checks all lengths, ranges, and overlaps.
        let storage = unsafe { MemoryStorage::new(pack.index.clone(), &weights, mode, pools)? };
        let runtime = Runtime::new(pack.index.clone(), runtime_config, Arc::new(storage))?;
        let handle = Box::new(RuntimeHandle {
            runtime,
            active: None,
        });
        // SAFETY: output was checked before construction and remains exclusively writable.
        unsafe {
            boundary::write(output, Box::into_raw(handle))?;
        }
        Ok(())
    })
}
