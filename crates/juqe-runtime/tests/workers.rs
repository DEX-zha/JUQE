//! Worker failure, parallel execution, and completion draining.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use juqe_io::{AlignedBuffer, ReadPath, ReadReport};
use juqe_runtime::{Config, ExpertStorage, Runtime, RuntimeError};
use std::{
    io,
    sync::{Arc, Condvar, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

struct ControlledStorage {
    started: mpsc::Sender<usize>,
    release: (Mutex<bool>, Condvar),
}

impl ExpertStorage for ControlledStorage {
    fn load(
        &self,
        _: usize,
        expert: usize,
        _: usize,
        _: &mut AlignedBuffer,
        _: Instant,
    ) -> io::Result<ReadReport> {
        self.started.send(expert).unwrap();
        if expert == 0 {
            return Err(io::Error::other("injected read failure"));
        }
        let (gate, signal) = &self.release;
        let (_released, timeout) = signal
            .wait_timeout_while(gate.lock().unwrap(), Duration::from_secs(5), |released| {
                !*released
            })
            .unwrap();
        if timeout.timed_out() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "test gate timed out",
            ));
        }
        Ok(report())
    }
}

fn runtime(storage: Arc<dyn ExpertStorage>) -> Runtime {
    let index = "1 4\n0 0 4096 0 128 1 0 512 0 128 1 1024 512 0 128 1 2560 512\n"
        .parse()
        .unwrap();
    Runtime::new(
        index,
        Config {
            workers: 2,
            queue_capacity: 1,
            layer_slots: vec![4],
        },
        storage,
    )
    .unwrap()
}

fn report() -> ReadReport {
    ReadReport {
        path: ReadPath::BufferedStaging,
        fallback: None,
        bytes_read: 4096,
        queue_wait: Duration::ZERO,
        service_time: Duration::ZERO,
        request_to_ready: Duration::ZERO,
    }
}

#[test]
fn an_error_does_not_return_while_another_worker_is_writing() {
    let (started, starts) = mpsc::channel();
    let storage = Arc::new(ControlledStorage {
        started,
        release: (Mutex::new(false), Condvar::new()),
    });
    let mut runtime = runtime(storage.clone());
    let (done, result) = mpsc::channel();
    let worker = thread::spawn(move || {
        done.send(matches!(
            runtime.prepare(0, &[0, 1, 2]),
            Err(RuntimeError::Load { .. })
        ))
        .unwrap();
    });
    for _ in 0..3 {
        starts.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    assert!(matches!(
        result.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    *storage.release.0.lock().unwrap() = true;
    storage.release.1.notify_all();
    assert!(result.recv_timeout(Duration::from_secs(5)).unwrap());
    worker.join().unwrap();
}

struct PanickingStorage;
impl ExpertStorage for PanickingStorage {
    fn load(
        &self,
        _: usize,
        expert: usize,
        _: usize,
        _: &mut AlignedBuffer,
        _: Instant,
    ) -> io::Result<ReadReport> {
        if expert == 0 {
            panic!("injected storage panic");
        }
        Ok(report())
    }
}

#[test]
fn a_loader_panic_releases_its_slot_and_keeps_workers_available() {
    let mut runtime = runtime(Arc::new(PanickingStorage));
    assert!(matches!(
        runtime.prepare(0, &[0, 1]),
        Err(RuntimeError::Load { .. })
    ));
    let batch = runtime.prepare(0, &[1, 2, 3]).unwrap();
    assert_eq!(batch.report().cache_hits, 1);
    assert_eq!(batch.report().loads, 2);
}
