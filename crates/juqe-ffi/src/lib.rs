//! C ABI for engine-owned pools and synchronized inference callbacks.
//!
//! Handles are exclusive to one control thread. Foreign pool memory must remain
//! allocated until runtime destruction and may be read by the GPU only while its
//! batch is active. Release the batch only after synchronizing GPU consumption.

mod boundary;
mod memory;
mod pack;
mod runtime;
mod types;

pub use boundary::juqe_last_error;
pub use pack::{PackHandle, juqe_pack_close, juqe_pack_info, juqe_pack_layer, juqe_pack_open};
pub use runtime::{
    RuntimeHandle, juqe_runtime_close, juqe_runtime_create, juqe_runtime_prepare,
    juqe_runtime_release,
};
pub use types::{BatchInfo, LayerInfo, LayerPool, PackInfo, PartInfo, PoolPart, RuntimeConfig};
