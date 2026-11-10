//! Small audited boundary for aligned, initialized, owned scratch memory.
use juqe_core::format::MAX_EXPERT_BYTES;
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    io,
    ptr::NonNull,
};

/// Owned zero-initialized scratch. Preallocate one per worker outside timed reads.
///
/// This is ordinary host memory, not CUDA-pinned memory.
#[derive(Debug)]
pub struct AlignedBuffer {
    pointer: NonNull<u8>,
    layout: Layout,
}
impl AlignedBuffer {
    /// Allocate at most 256 MiB with a caller-selected power-of-two alignment.
    pub fn new(bytes: usize, alignment: usize) -> io::Result<Self> {
        if bytes == 0 || bytes as u64 > MAX_EXPERT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "scratch size outside limits",
            ));
        }
        let layout = Layout::from_size_align(bytes, alignment).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "invalid allocation layout")
        })?;
        // SAFETY: layout has nonzero size and a valid power-of-two alignment.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "scratch allocation failed")
        })?;
        Ok(Self { pointer, layout })
    }
    /// Entire initialized allocation, exclusively borrowed for this call.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: allocation is initialized, live, and uniquely owned; &mut self
        // prevents other safe access while the returned slice is live.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
    /// Entire initialized allocation, immutably borrowed.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: allocation remains live and initialized; borrow excludes mutation.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }
}
impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: this allocation was created using precisely this layout and is
        // freed once; no slice can outlive its owner in safe Rust.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}
// SAFETY: ownership can move between workers; slices borrow the owner and no raw
// pointer is exposed. There is no concurrent mutation of the owned allocation.
unsafe impl Send for AlignedBuffer {}
