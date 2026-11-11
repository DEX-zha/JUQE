use juqe_io::OpenMode;
use std::{
    ffi::{OsStr, OsString},
    io,
    path::PathBuf,
};

pub(crate) const USAGE: &str = "\
Usage:
  juqe inspect <experts.idx>
  juqe verify <experts.idx> <experts.bin> <layer> <expert> [--direct]

verify compares each loaded matrix against an independent buffered positional read.
It prints loader latency (not model throughput). Direct I/O requires Linux.";

#[derive(Debug)]
pub(crate) enum Command {
    Help,
    Inspect { index: PathBuf },
    Verify(VerifyArgs),
}

#[derive(Debug)]
pub(crate) struct VerifyArgs {
    pub index: PathBuf,
    pub weights: PathBuf,
    pub layer: usize,
    pub expert: usize,
    pub mode: OpenMode,
}

impl Command {
    pub fn parse(arguments: impl IntoIterator<Item = OsString>) -> io::Result<Self> {
        let arguments: Vec<_> = arguments.into_iter().collect();
        let Some(command) = arguments.first() else {
            return Ok(Self::Help);
        };
        let command = text(command)?;
        match (command, &arguments[1..]) {
            ("--help" | "-h", []) => Ok(Self::Help),
            ("inspect", [index]) => Ok(Self::Inspect {
                index: index.into(),
            }),
            ("verify", [index, weights, layer, expert]) => {
                Self::verify(index, weights, layer, expert, OpenMode::Buffered)
            }
            ("verify", [index, weights, layer, expert, flag]) if flag == "--direct" => {
                Self::verify(index, weights, layer, expert, OpenMode::Direct)
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE)),
        }
    }

    fn verify(
        index: &OsStr,
        weights: &OsStr,
        layer: &OsStr,
        expert: &OsStr,
        mode: OpenMode,
    ) -> io::Result<Self> {
        Ok(Self::Verify(VerifyArgs {
            index: index.into(),
            weights: weights.into(),
            layer: identifier(layer, "layer")?,
            expert: identifier(expert, "expert")?,
            mode,
        }))
    }
}

fn text(value: &OsStr) -> io::Result<&str> {
    value.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "command arguments must be valid UTF-8",
        )
    })
}

fn identifier(value: &OsStr, name: &str) -> io::Result<usize> {
    text(value)?.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be a nonnegative integer"),
        )
    })
}
