use super::Alignment;
use juqe_core::format::Layer;
use std::{fs::File, io, os::fd::AsRawFd};

pub(crate) fn scatter(
    file: &File,
    a: Alignment,
    layer: &Layer,
    offset: u64,
    destinations: &mut [&mut [u8]; 3],
    scratch: &mut [u8],
    actual: &mut u64,
) -> io::Result<bool> {
    if !offset.is_multiple_of(a.offset) || !layer.stride().is_multiple_of(a.offset) {
        return Ok(false);
    }
    let mut vectors: Vec<libc::iovec> = Vec::with_capacity(7);
    let mut cursor = 0;
    let mut used = 0;
    let scratch_base = scratch.as_mut_ptr();
    let scratch_len = scratch.len();
    let mut gap = |bytes: usize, vectors: &mut Vec<libc::iovec>| -> bool {
        if bytes == 0 {
            return true;
        }
        let pad = (a.memory - used % a.memory) % a.memory;
        let Some(start) = used.checked_add(pad) else {
            return false;
        };
        let Some(end) = start.checked_add(bytes).filter(|&end| end <= scratch_len) else {
            return false;
        };
        if !(bytes as u64).is_multiple_of(a.offset) {
            return false;
        }
        // SAFETY: start..end was checked inside live scratch; separate padding ranges
        // advance monotonically and cannot overlap one another or destinations.
        let pointer = unsafe { scratch_base.add(start) };
        if !(pointer as usize).is_multiple_of(a.memory) {
            return false;
        }
        vectors.push(libc::iovec {
            iov_base: pointer.cast(),
            iov_len: bytes,
        });
        used = end;
        true
    };
    for (part, dst) in layer.parts().iter().zip(destinations.iter_mut()) {
        if !part.bytes().is_multiple_of(a.offset)
            || !(dst.as_ptr() as usize).is_multiple_of(a.memory)
            || !gap((part.offset() - cursor) as usize, &mut vectors)
        {
            return Ok(false);
        }
        vectors.push(libc::iovec {
            iov_base: dst.as_mut_ptr().cast(),
            iov_len: dst.len(),
        });
        cursor = part.offset() + part.bytes();
    }
    if !gap((layer.stride() - cursor) as usize, &mut vectors) {
        return Ok(false);
    }
    let mut offset = offset as libc::off_t; // index validation caps extents at i64::MAX
    let mut first = 0;
    while first < vectors.len() {
        // SAFETY: all iovecs refer to disjoint writable borrowed slices or scratch gaps,
        // validated above and alive throughout this synchronous syscall. At most 7
        // vectors; offset is within the validated extent and no fd cursor is mutated.
        let got = unsafe {
            libc::preadv(
                file.as_raw_fd(),
                vectors[first..].as_ptr(),
                (vectors.len() - first) as libc::c_int,
                offset,
            )
        };
        if got < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if got == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short scatter read",
            ));
        }
        *actual += got as u64;
        offset += got as libc::off_t;
        let mut left = got as usize;
        while first < vectors.len() && left >= vectors[first].iov_len {
            left -= vectors[first].iov_len;
            first += 1;
        }
        if left > 0 && first < vectors.len() {
            // SAFETY: partial read consumed fewer than iov_len bytes; updated pointer
            // remains in the same allocation, and its remaining length is reduced.
            vectors[first].iov_base =
                unsafe { vectors[first].iov_base.cast::<u8>().add(left).cast() };
            vectors[first].iov_len -= left;
        }
    }
    Ok(true)
}
