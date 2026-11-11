mod batch;
mod create;

pub use batch::{juqe_runtime_prepare, juqe_runtime_release};
pub use create::juqe_runtime_create;

use crate::boundary;
use juqe_runtime::{Batch, Runtime};

/// Opaque runtime handle, exclusive to one engine control thread.
pub struct RuntimeHandle {
    runtime: Runtime,
    active: Option<Batch>,
}

/// Join workers and release the runtime. Null is accepted.
///
/// # Safety
/// The handle must be live and exclusively owned. Synchronize all GPU consumers
/// first, and keep the foreign pools allocated until this function returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_runtime_close(handle: *mut RuntimeHandle) -> i32 {
    boundary::run(|| {
        if !handle.is_null() {
            // SAFETY: caller transfers the original Box back once, after GPU completion.
            let mut handle = unsafe { Box::from_raw(handle) };
            handle.runtime.shutdown();
            handle.active.take();
        }
        Ok(())
    })
}
