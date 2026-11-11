use crate::{Result, args::VerifyArgs};
use juqe_core::format::{Layer, PACK_ALIGNMENT};
use juqe_io::{AlignedBuffer, ExpertReader, ReadReport, read_index};
use std::{fs::File, io, os::unix::fs::FileExt, time::Instant};

pub(super) fn run(arguments: &VerifyArgs) -> Result<ReadReport> {
    let index = read_index(&arguments.index)?;
    let layer = index
        .layers()
        .get(arguments.layer)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "layer out of range"))?;
    let offset = layer.expert_offset(arguments.expert)?;
    let reader = ExpertReader::open(&arguments.weights, &index, arguments.mode)?;
    let mut scratch = allocate(layer.stride())?;
    let mut gate = allocate(layer.parts()[0].bytes())?;
    let mut up = allocate(layer.parts()[1].bytes())?;
    let mut down = allocate(layer.parts()[2].bytes())?;

    let report = reader.read_into(
        layer,
        arguments.expert,
        [gate.as_mut_slice(), up.as_mut_slice(), down.as_mut_slice()],
        &mut scratch,
        Instant::now(),
    )?;

    // Reference I/O and comparisons run after the loader's timer has stopped.
    File::open(&arguments.weights)?.read_exact_at(scratch.as_mut_slice(), offset)?;
    compare_matrices(
        layer,
        scratch.as_slice(),
        [gate.as_slice(), up.as_slice(), down.as_slice()],
    )?;
    Ok(report)
}

fn allocate(bytes: u64) -> io::Result<AlignedBuffer> {
    // Validated layouts cap each allocation at 256 MiB.
    AlignedBuffer::new(bytes as usize, PACK_ALIGNMENT as usize)
}

fn compare_matrices(layer: &Layer, reference: &[u8], matrices: [&[u8]; 3]) -> io::Result<()> {
    for (part, loaded) in layer.parts().iter().zip(matrices) {
        let start = part.offset() as usize;
        if reference[start..start + loaded.len()] != *loaded {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "matrix bytes differ from reference",
            ));
        }
    }
    Ok(())
}
