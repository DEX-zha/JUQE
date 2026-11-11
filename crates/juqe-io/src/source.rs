use juqe_core::format::Layer;
use std::{fs::File, io, path::Path};

#[cfg(target_os = "linux")]
use crate::linux;
use crate::{Fallback, OpenMode, ReadPath, positional::read_exact_at};

#[derive(Debug)]
pub(crate) enum Source {
    Buffered(File),
    #[cfg(target_os = "linux")]
    Direct {
        file: File,
        alignment: linux::Alignment,
    },
}

impl Source {
    pub fn open(path: &Path, mode: OpenMode) -> io::Result<Self> {
        match mode {
            OpenMode::Buffered => File::open(path).map(Self::Buffered),
            OpenMode::Direct => {
                #[cfg(target_os = "linux")]
                {
                    let file = linux::open_direct(path)?;
                    let alignment = linux::alignment(&file);
                    Ok(Self::Direct { file, alignment })
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "direct I/O requires Linux",
                    ))
                }
            }
        }
    }

    pub fn len(&self) -> io::Result<u64> {
        self.file().metadata().map(|metadata| metadata.len())
    }

    pub fn read(
        &self,
        layer: &Layer,
        offset: u64,
        destinations: &mut [&mut [u8]; 3],
        scratch: &mut [u8],
        bytes_read: &mut u64,
    ) -> io::Result<(ReadPath, Option<Fallback>)> {
        match self {
            Self::Buffered(file) => {
                read_staged(file, layer, offset, destinations, scratch, bytes_read)?;
                Ok((ReadPath::BufferedStaging, None))
            }
            #[cfg(target_os = "linux")]
            Self::Direct { file, alignment } => {
                let fallback = match linux::scatter(
                    file,
                    *alignment,
                    layer,
                    offset,
                    destinations,
                    scratch,
                    bytes_read,
                ) {
                    Ok(true) => return Ok((ReadPath::DirectScatter, None)),
                    Ok(false) => Fallback::Alignment,
                    Err(error)
                        if matches!(
                            error.raw_os_error(),
                            Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
                        ) =>
                    {
                        Fallback::Unsupported
                    }
                    Err(error) => return Err(error),
                };
                linux::validate_staging(*alignment, offset, &scratch[..layer.stride() as usize])?;
                read_staged(file, layer, offset, destinations, scratch, bytes_read)?;
                Ok((ReadPath::DirectStaging, Some(fallback)))
            }
        }
    }

    fn file(&self) -> &File {
        match self {
            Self::Buffered(file) => file,
            #[cfg(target_os = "linux")]
            Self::Direct { file, .. } => file,
        }
    }
}

fn read_staged(
    file: &File,
    layer: &Layer,
    offset: u64,
    destinations: &mut [&mut [u8]; 3],
    scratch: &mut [u8],
    bytes_read: &mut u64,
) -> io::Result<()> {
    // The index caps blocks at 256 MiB, and ExpertReader checks every buffer length.
    read_exact_at(
        file,
        &mut scratch[..layer.stride() as usize],
        offset,
        bytes_read,
    )?;
    for (destination, part) in destinations.iter_mut().zip(layer.parts()) {
        let start = part.offset() as usize;
        destination.copy_from_slice(&scratch[start..start + destination.len()]);
    }
    Ok(())
}
