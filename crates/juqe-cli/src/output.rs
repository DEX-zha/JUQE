use juqe_core::format::ExpertIndex;
use juqe_io::ReadReport;
use std::io::{self, Write};

pub(crate) fn index(writer: &mut impl Write, index: &ExpertIndex) -> io::Result<()> {
    writeln!(
        writer,
        "layers={} experts_per_layer={} required_file_bytes={}",
        index.layers().len(),
        index.experts(),
        index.file_bytes()
    )?;
    for (id, layer) in index.layers().iter().enumerate() {
        writeln!(
            writer,
            "layer={id} offset={} expert_bytes={} dense={}",
            layer.offset(),
            layer.stride(),
            layer.stride() == 0
        )?;
    }
    Ok(())
}

pub(crate) fn verification(
    writer: &mut impl Write,
    layer: usize,
    expert: usize,
    report: &ReadReport,
) -> io::Result<()> {
    writeln!(
        writer,
        "verified=true layer={layer} expert={expert} path={:?} fallback={:?} bytes_read={} queue_ms={:.3} service_ms={:.3} request_ready_ms={:.3}",
        report.path,
        report.fallback,
        report.bytes_read,
        report.queue_wait.as_secs_f64() * 1000.0,
        report.service_time.as_secs_f64() * 1000.0,
        report.request_to_ready.as_secs_f64() * 1000.0,
    )
}
