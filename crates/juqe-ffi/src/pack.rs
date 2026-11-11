use crate::{LayerInfo, PackInfo, PartInfo, boundary};
use juqe_core::format::ExpertIndex;
use std::ffi::c_char;

/// Opaque immutable pack metadata. Access only through the exported functions.
pub struct PackHandle {
    pub(crate) index: ExpertIndex,
}

/// Open and validate an expert index.
///
/// # Safety
/// `path` must be a live NUL-terminated string. `output` must be a writable,
/// exclusively borrowed pointer slot. Close each successful handle once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_pack_open(path: *const c_char, output: *mut *mut PackHandle) -> i32 {
    boundary::run(|| {
        // SAFETY: pointers meet the exported function's call contract.
        let path = unsafe { boundary::path(path)? };
        // SAFETY: caller supplies an exclusive writable pointer slot.
        unsafe {
            boundary::write(output, std::ptr::null_mut())?;
        }
        let index = juqe_io::read_index(path)?;
        // SAFETY: output was checked above and remains exclusively writable.
        unsafe {
            boundary::write(output, Box::into_raw(Box::new(PackHandle { index })))?;
        }
        Ok(())
    })
}

/// Copy pack dimensions into the caller's output.
///
/// # Safety
/// `pack` must be live. `output` must be writable, aligned, and disjoint from it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_pack_info(pack: *const PackHandle, output: *mut PackInfo) -> i32 {
    boundary::run(|| {
        // SAFETY: the caller provides a live handle and exclusive output.
        let pack = unsafe { boundary::input(pack)? };
        let info = PackInfo {
            layers: pack.index.layers().len() as u32,
            experts: pack.index.experts() as u32,
        };
        // SAFETY: the caller supplies exclusive writable output storage.
        unsafe {
            boundary::write(output, info)?;
        }
        Ok(())
    })
}

/// Copy a layer's matrix geometry into the caller's output.
///
/// # Safety
/// `pack` must be live. `output` must be writable, aligned, and disjoint from it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_pack_layer(
    pack: *const PackHandle,
    layer: u32,
    output: *mut LayerInfo,
) -> i32 {
    boundary::run(|| {
        // SAFETY: the caller provides a live handle and exclusive output.
        let pack = unsafe { boundary::input(pack)? };
        let layer = pack
            .index
            .layers()
            .get(layer as usize)
            .ok_or_else(|| boundary::invalid("layer out of range"))?;
        let info = LayerInfo {
            stride: layer.stride(),
            parts: std::array::from_fn(|part| {
                let part = &layer.parts()[part];
                PartInfo {
                    kind: part.kind(),
                    dimensions: part.dimensions(),
                    bytes: part.bytes(),
                }
            }),
        };
        // SAFETY: the caller supplies exclusive writable output storage.
        unsafe {
            boundary::write(output, info)?;
        }
        Ok(())
    })
}

/// Close a pack handle. Null is accepted.
///
/// # Safety
/// A non-null handle must come from `juqe_pack_open`, remain unclosed, and have no
/// concurrent users. Runtime instances keep their own metadata copies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_pack_close(pack: *mut PackHandle) -> i32 {
    boundary::run(|| {
        if !pack.is_null() {
            // SAFETY: the caller transfers the original, uniquely owned Box back once.
            drop(unsafe { Box::from_raw(pack) });
        }
        Ok(())
    })
}
