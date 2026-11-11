use std::{
    cell::RefCell,
    ffi::{CStr, c_char},
    io,
    os::unix::ffi::OsStrExt,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
};

thread_local! { static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn run(operation: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => {
            LAST_ERROR.with(|error| error.borrow_mut().clear());
            0
        }
        Ok(Err(error)) => {
            LAST_ERROR.with(|message| *message.borrow_mut() = error.to_string());
            1
        }
        Err(_) => {
            LAST_ERROR
                .with(|message| *message.borrow_mut() = "panic at JUQE ABI boundary".to_owned());
            2
        }
    }
}

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(crate) unsafe fn path(pointer: *const c_char) -> Result<PathBuf> {
    if pointer.is_null() {
        return Err(invalid("path is null").into());
    }
    // SAFETY: the caller supplies a live NUL-terminated path for this call.
    let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes();
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

pub(crate) unsafe fn input<'a, T>(pointer: *const T) -> Result<&'a T> {
    // SAFETY: callers establish alignment, initialization, and validity for the call.
    unsafe { pointer.as_ref() }.ok_or_else(|| invalid("input pointer is null").into())
}

pub(crate) unsafe fn write<T>(pointer: *mut T, value: T) -> Result<()> {
    if pointer.is_null() {
        return Err(invalid("output pointer is null").into());
    }
    // SAFETY: the caller supplies aligned, exclusive writable memory. Writing
    // through the raw pointer does not require the output to be initialized.
    unsafe {
        pointer.write(value);
    }
    Ok(())
}

/// Copy the current thread's last error and return the required capacity including NUL.
///
/// # Safety
/// If non-null, `buffer` must be writable for `capacity` bytes. The buffer must not
/// overlap other live Rust borrows. A null buffer queries the required size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn juqe_last_error(buffer: *mut c_char, capacity: usize) -> usize {
    LAST_ERROR.with(|error| {
        let error = error.borrow();
        if !buffer.is_null() && capacity > 0 {
            let copied = error.len().min(capacity - 1);
            // SAFETY: caller provides capacity bytes; source is separate thread-local
            // storage. copied + 1 fits, and the trailing byte terminates the C string.
            unsafe {
                std::ptr::copy_nonoverlapping(error.as_ptr(), buffer.cast(), copied);
                *buffer.add(copied) = 0;
            }
        }
        error.len() + 1
    })
}
