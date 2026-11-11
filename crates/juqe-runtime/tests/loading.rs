//! Complete batch loading, byte correctness, and lease protection.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;

use juqe_core::cache::CacheError;
use juqe_runtime::RuntimeError;
use std::{
    fs::{self, OpenOptions},
    sync::atomic::Ordering,
};
use support::Fixture;

#[test]
fn duplicates_share_a_load_and_hits_are_protected_before_misses() {
    let fixture = Fixture::new(2);
    let mut runtime = fixture.runtime(2);
    let first = runtime.prepare(0, &[2, 1, 2]).unwrap();
    assert_eq!(first.slots()[0], first.slots()[2]);
    assert_eq!(first.report().loads, 2);
    assert_eq!(first.report().bytes_read, 8192);
    let retained_slot = first.slots()[0];
    fixture.check(retained_slot, 2);
    fixture.check(first.slots()[1], 1);
    drop(first);

    let second = runtime.prepare(0, &[0, 2]).unwrap();
    assert_eq!(second.slots()[1], retained_slot);
    assert_eq!(second.report().cache_hits, 1);
    assert_eq!(second.report().loads, 1);
    fixture.check(second.slots()[0], 0);
    fixture.check(second.slots()[1], 2);
    assert_eq!(fixture.storage.loads.load(Ordering::Relaxed), 3);
}

#[test]
fn held_batches_block_eviction_and_failed_reservations_are_reusable() {
    let fixture = Fixture::new(2);
    let mut runtime = fixture.runtime(2);
    let held = runtime.prepare(0, &[0]).unwrap();
    assert!(matches!(
        runtime.prepare(0, &[1, 2]),
        Err(RuntimeError::Cache {
            source: CacheError::Busy,
            ..
        })
    ));
    assert_eq!(fixture.storage.loads.load(Ordering::Relaxed), 1);
    fixture.check(held.slots()[0], 0);
    drop(held);
    let batch = runtime.prepare(0, &[1, 2]).unwrap();
    assert_eq!(batch.report().loads, 2);
    fixture.check(batch.slots()[0], 1);
    fixture.check(batch.slots()[1], 2);
}

#[test]
fn oversized_and_invalid_requests_do_not_submit_io() {
    let fixture = Fixture::new(2);
    let mut runtime = fixture.runtime(2);
    assert!(matches!(
        runtime.prepare(0, &[0, 1, 2]),
        Err(RuntimeError::BatchTooLarge { .. })
    ));
    assert!(matches!(
        runtime.prepare(1, &[0]),
        Err(RuntimeError::InvalidLayer(1))
    ));
    assert!(matches!(
        runtime.prepare(0, &[4]),
        Err(RuntimeError::Cache {
            source: CacheError::InvalidExpert { .. },
            ..
        })
    ));
    assert_eq!(fixture.storage.loads.load(Ordering::Relaxed), 0);
}

#[test]
fn failed_reads_do_not_publish_and_slots_can_retry() {
    let fixture = Fixture::new(2);
    let mut runtime = fixture.runtime(2);
    let weights = fixture.path.join("experts.bin");
    OpenOptions::new()
        .write(true)
        .open(&weights)
        .unwrap()
        .set_len(512)
        .unwrap();
    assert!(matches!(
        runtime.prepare(0, &[0, 1]),
        Err(RuntimeError::Load { .. })
    ));
    fs::write(&weights, &fixture.bytes).unwrap();
    let batch = runtime.prepare(0, &[0, 1]).unwrap();
    assert_eq!(batch.report().cache_hits, 0);
    assert_eq!(batch.report().loads, 2);
    fixture.check(batch.slots()[0], 0);
    fixture.check(batch.slots()[1], 1);
}

#[test]
fn shutdown_preserves_live_batch_and_rejects_new_requests() {
    let fixture = Fixture::new(1);
    let mut runtime = fixture.runtime(1);
    let batch = runtime.prepare(0, &[3]).unwrap();
    runtime.shutdown();
    runtime.shutdown();
    assert!(matches!(
        runtime.prepare(0, &[3]),
        Err(RuntimeError::Stopped)
    ));
    drop(runtime);
    fixture.check(batch.slots()[0], 3);
}
