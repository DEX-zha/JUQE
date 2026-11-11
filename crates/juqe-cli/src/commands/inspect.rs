use crate::{Result, output};
use juqe_io::read_index;
use std::{io::Write, path::Path};

pub(super) fn run(path: &Path, writer: &mut impl Write) -> Result<()> {
    let index = read_index(path)?;
    output::index(writer, &index)?;
    Ok(())
}
