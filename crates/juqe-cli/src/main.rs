//! Small diagnostic CLI; it does not perform model inference or manage services.
#![forbid(unsafe_code)]
use juqe_io::{AlignedBuffer, ExpertReader, OpenMode, read_index};
use std::{error::Error, io, process::ExitCode, time::Instant};

const USAGE: &str = "Usage:\n  juqe inspect <experts.idx>\n  juqe verify <experts.idx> <experts.bin> <layer> <expert> [--direct]\n\nverify compares each loaded matrix against an independent buffered positional read.\nIt prints loader latency (not model throughput). Direct I/O requires Linux.";

fn run(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    if args.is_empty() || args == ["--help"] || args == ["-h"] {
        println!("{USAGE}");
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "inspect") && args.len() == 2 {
        let index = read_index(&args[1])?;
        println!(
            "layers={} experts_per_layer={} required_file_bytes={}",
            index.layers().len(),
            index.experts(),
            index.file_bytes()
        );
        for (i, l) in index.layers().iter().enumerate() {
            println!(
                "layer={i} offset={} expert_bytes={} dense={}",
                l.offset(),
                l.stride(),
                l.stride() == 0
            );
        }
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "verify")
        && (args.len() == 5 || (args.len() == 6 && args[5] == "--direct"))
    {
        let index = read_index(&args[1])?;
        let layer_id: usize = args[3].parse()?;
        let expert: usize = args[4].parse()?;
        let layer = index.layers().get(layer_id).ok_or("layer out of range")?;
        let offset = layer.expert_offset(expert)?;
        let mode = if args.len() == 6 {
            OpenMode::Direct
        } else {
            OpenMode::Buffered
        };
        let reader = ExpertReader::open(&args[2], &index, mode)?;
        let mut scratch = AlignedBuffer::new(layer.stride() as usize, 4096)?;
        let mut gate = AlignedBuffer::new(layer.parts()[0].bytes() as usize, 4096)?;
        let mut up = AlignedBuffer::new(layer.parts()[1].bytes() as usize, 4096)?;
        let mut down = AlignedBuffer::new(layer.parts()[2].bytes() as usize, 4096)?;
        let report = reader.read_into(
            layer,
            expert,
            [gate.as_mut_slice(), up.as_mut_slice(), down.as_mut_slice()],
            &mut scratch,
            Instant::now(),
        )?;
        // Independent read and byte comparison, deliberately outside the loader timer.
        use std::os::unix::fs::FileExt;
        let reference = std::fs::File::open(&args[2])?;
        reference.read_exact_at(scratch.as_mut_slice(), offset)?;
        for (part, got) in
            layer
                .parts()
                .iter()
                .zip([gate.as_slice(), up.as_slice(), down.as_slice()])
        {
            let start = part.offset() as usize;
            if scratch.as_slice()[start..start + got.len()] != *got {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "matrix bytes differ from reference",
                )
                .into());
            }
        }
        println!(
            "verified=true layer={layer_id} expert={expert} path={:?} fallback={:?} bytes_read={} queue_ms={:.3} service_ms={:.3} request_ready_ms={:.3}",
            report.path,
            report.fallback,
            report.bytes_read,
            report.queue_wait.as_secs_f64() * 1000.,
            report.service_time.as_secs_f64() * 1000.,
            report.request_to_ready.as_secs_f64() * 1000.
        );
        return Ok(());
    }
    Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE).into())
}

fn main() -> ExitCode {
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| {
            arg.into_string().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "arguments must be valid UTF-8").into()
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>();
    match args.and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("juqe: {e}");
            ExitCode::FAILURE
        }
    }
}
