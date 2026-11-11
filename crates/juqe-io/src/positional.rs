use std::{fs::File, io, os::unix::fs::FileExt};

pub(crate) fn read_exact_at(
    file: &File,
    mut buffer: &mut [u8],
    mut offset: u64,
    actual: &mut u64,
) -> io::Result<()> {
    while !buffer.is_empty() {
        match file.read_at(buffer, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short expert read",
                ));
            }
            Ok(n) => {
                offset += n as u64;
                *actual += n as u64;
                buffer = &mut buffer[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
