use std::{fs::File, io, os::fd::AsRawFd};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Alignment {
    pub(super) memory: usize,
    pub(super) offset: u64,
}

pub(crate) fn alignment(file: &File) -> Alignment {
    // SAFETY: statx is a plain C struct for which zero is a valid initial representation.
    let mut info: libc::statx = unsafe { std::mem::zeroed() };
    // SAFETY: live fd, NUL-terminated empty path, valid writable statx object. The syscall
    // borrows these pointers only until it returns; no aliasing with expert destinations.
    let result = unsafe {
        libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            libc::STATX_DIOALIGN,
            &mut info,
        )
    };
    if result == 0
        && info.stx_mask & libc::STATX_DIOALIGN != 0
        && info.stx_dio_mem_align > 0
        && info.stx_dio_offset_align > 0
    {
        Alignment {
            memory: info.stx_dio_mem_align as usize,
            offset: u64::from(info.stx_dio_offset_align),
        }
    } else {
        Alignment {
            memory: 4096,
            offset: 4096,
        }
    }
}

pub(crate) fn validate_staging(a: Alignment, offset: u64, scratch: &[u8]) -> io::Result<()> {
    if !(scratch.as_ptr() as usize).is_multiple_of(a.memory)
        || !offset.is_multiple_of(a.offset)
        || !(scratch.len() as u64).is_multiple_of(a.offset)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "scratch does not satisfy direct I/O alignment",
        ));
    }
    Ok(())
}
