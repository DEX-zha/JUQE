//! End-to-end byte correctness, file mutation and positional concurrency.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use juqe_core::format::ExpertIndex;
use juqe_io::{AlignedBuffer, ExpertReader, OpenMode, ReadPath};
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    path: PathBuf,
    index: ExpertIndex,
    bytes: Vec<u8>,
}
impl Fixture {
    fn new(gate_offset: usize) -> Self {
        let path = std::env::temp_dir().join(format!(
            "juqe-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let bytes: Vec<u8> = (0..12288)
            .map(|i| ((i * 17 + i / 13) % 251) as u8)
            .collect();
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let index =
            format!("1 3\n0 0 4096 0 128 1 {gate_offset} 512 0 128 1 1024 512 0 128 1 2560 512\n")
                .parse()
                .unwrap();
        Self { path, index, bytes }
    }
    fn check(&self, reader: &ExpertReader, expert: usize) -> juqe_io::ReadReport {
        let mut scratch = AlignedBuffer::new(4096, 4096).unwrap();
        // Extra bytes around each borrowed destination must remain untouched.
        let mut gate = AlignedBuffer::new(1536, 4096).unwrap();
        let mut up = AlignedBuffer::new(1536, 4096).unwrap();
        let mut down = AlignedBuffer::new(1536, 4096).unwrap();
        for p in [&mut gate, &mut up, &mut down] {
            p.as_mut_slice().fill(0xcc);
        }
        let layer = &self.index.layers()[0];
        let requested = Instant::now() - Duration::from_millis(3);
        let report = reader
            .read_into(
                layer,
                expert,
                [
                    &mut gate.as_mut_slice()[512..1024],
                    &mut up.as_mut_slice()[512..1024],
                    &mut down.as_mut_slice()[512..1024],
                ],
                &mut scratch,
                requested,
            )
            .unwrap();
        for (p, buf) in layer.parts().iter().zip([gate, up, down]) {
            let start = expert * 4096 + p.offset() as usize;
            assert_eq!(&buf.as_slice()[512..1024], &self.bytes[start..start + 512]);
            assert!(
                buf.as_slice()[..512]
                    .iter()
                    .chain(&buf.as_slice()[1024..])
                    .all(|b| *b == 0xcc)
            );
        }
        assert!(report.queue_wait >= Duration::from_millis(3));
        assert_eq!(
            report.request_to_ready,
            report.queue_wait + report.service_time
        );
        report
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn portable_reads_match_bytes_and_preserve_adjacent_slots() {
    let f = Fixture::new(0);
    let reader = ExpertReader::open(&f.path, &f.index, OpenMode::Buffered).unwrap();
    for expert in 0..3 {
        let r = f.check(&reader, expert);
        assert_eq!(r.path, ReadPath::BufferedStaging);
        assert_eq!(r.bytes_read, 4096);
    }
}
#[test]
fn simultaneous_requests_do_not_share_a_file_cursor() {
    let f = Fixture::new(0);
    let reader = ExpertReader::open(&f.path, &f.index, OpenMode::Buffered).unwrap();
    std::thread::scope(|scope| {
        for i in 0..8 {
            let (f, r) = (&f, &reader);
            scope.spawn(move || {
                for _ in 0..4 {
                    f.check(r, i % 3);
                }
            });
        }
    });
}
#[test]
fn truncation_after_open_is_not_successful_publication() {
    let f = Fixture::new(0);
    let reader = ExpertReader::open(&f.path, &f.index, OpenMode::Buffered).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&f.path)
        .unwrap()
        .set_len(512)
        .unwrap();
    let mut scratch = AlignedBuffer::new(4096, 4096).unwrap();
    let (mut a, mut b, mut c) = ([0; 512], [0; 512], [0; 512]);
    let e = reader
        .read_into(
            &f.index.layers()[0],
            0,
            [&mut a, &mut b, &mut c],
            &mut scratch,
            Instant::now(),
        )
        .unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    assert!(ExpertReader::open(&f.path, &f.index, OpenMode::Buffered).is_err());
}
#[test]
fn wrong_buffer_sizes_and_future_timestamps_fail() {
    let f = Fixture::new(0);
    let reader = ExpertReader::open(&f.path, &f.index, OpenMode::Buffered).unwrap();
    let mut scratch = AlignedBuffer::new(4096, 4096).unwrap();
    let (mut short, mut b, mut c) = ([0; 511], [0; 512], [0; 512]);
    let wrong_size = reader
        .read_into(
            &f.index.layers()[0],
            0,
            [&mut short, &mut b, &mut c],
            &mut scratch,
            Instant::now(),
        )
        .unwrap_err();
    assert_eq!(wrong_size.kind(), io::ErrorKind::InvalidInput);
    let mut a = [0; 512];
    let future_timestamp = reader
        .read_into(
            &f.index.layers()[0],
            0,
            [&mut a, &mut b, &mut c],
            &mut scratch,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap_err();
    assert_eq!(future_timestamp.kind(), io::ErrorKind::InvalidInput);
    assert!(AlignedBuffer::new(0, 4096).is_err());
    assert!(AlignedBuffer::new(4096, 3).is_err());
}
#[test]
fn index_import_is_bounded_and_rejects_binary_input() {
    let f = Fixture::new(0);
    assert!(juqe_io::read_index(&f.path).is_err());
    File::create(&f.path)
        .unwrap()
        .write_all(&vec![b' '; juqe_core::format::MAX_INDEX_BYTES + 1])
        .unwrap();
    assert!(juqe_io::read_index(&f.path).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn linux_scatter_and_unaligned_part_fallback_are_exact() {
    for offset in [0, 256] {
        let f = Fixture::new(offset);
        let reader = match ExpertReader::open(&f.path, &f.index, OpenMode::Direct) {
            Ok(reader) => reader,
            Err(e)
                if matches!(
                    e.raw_os_error(),
                    Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
                ) && std::env::var_os("JUQE_REQUIRE_DIRECT").is_none() =>
            {
                eprintln!("filesystem does not support O_DIRECT: {e}");
                return;
            }
            Err(e) => panic!("direct open failed: {e}"),
        };
        for expert in 0..3 {
            let r = f.check(&reader, expert);
            assert!(matches!(
                r.path,
                ReadPath::DirectScatter | ReadPath::DirectStaging
            ));
            if std::env::var_os("JUQE_REQUIRE_DIRECT").is_some() {
                assert_eq!(
                    r.path,
                    if offset == 0 {
                        ReadPath::DirectScatter
                    } else {
                        ReadPath::DirectStaging
                    }
                );
            }
        }
    }
}
