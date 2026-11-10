//! Cache publication, lease lifetime, and eviction invariants.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use juqe_core::cache::{Cache, CacheError, LoadTicket, Request};

fn load(cache: &mut Cache, expert: usize) -> LoadTicket {
    match cache.request(expert).unwrap() {
        Request::Load(t) => t,
        r => panic!("expected load, got {r:?}"),
    }
}

#[test]
fn coalesces_pending_and_never_evicts_inflight() {
    let mut c = Cache::new(3, 1).unwrap();
    let t = load(&mut c, 0);
    assert!(matches!(c.request(0), Ok(Request::Pending)));
    assert!(matches!(c.request(1), Err(CacheError::Busy)));
    c.fail(&t).unwrap();
    let replacement = load(&mut c, 1);
    assert_eq!(replacement.slot(), t.slot());
    assert!(c.complete(&t).is_err());
    assert!(c.complete(&replacement).is_ok());
}
#[test]
fn all_cloned_leases_must_finish_before_eviction() {
    let mut c = Cache::new(2, 1).unwrap();
    let t = load(&mut c, 0);
    let a = c.complete(&t).unwrap();
    let b = a.clone();
    drop(a);
    assert!(matches!(c.request(1), Err(CacheError::Busy)));
    drop(b);
    assert!(matches!(c.request(1), Ok(Request::Load(_))));
}
#[test]
fn lru_retains_recently_used_ready_expert() {
    let mut c = Cache::new(3, 2).unwrap();
    let t0 = load(&mut c, 0);
    drop(c.complete(&t0).unwrap());
    let t1 = load(&mut c, 1);
    drop(c.complete(&t1).unwrap());
    let hit = c.request(0).unwrap();
    assert!(matches!(hit, Request::Ready(_)));
    drop(hit);
    let t2 = load(&mut c, 2);
    assert_eq!(t2.slot(), t1.slot());
    assert!(matches!(c.request(0), Ok(Request::Ready(_))));
}
#[test]
fn completion_handles_cannot_cross_cache_or_publish_twice() {
    let mut a = Cache::new(2, 1).unwrap();
    let mut b = Cache::new(2, 1).unwrap();
    let ta = load(&mut a, 0);
    let tb = load(&mut b, 0);
    assert!(matches!(b.complete(&ta), Err(CacheError::ForeignTicket)));
    assert!(a.complete(&ta).is_ok());
    assert!(matches!(a.complete(&ta), Err(CacheError::StaleTicket)));
    assert!(a.fail(&ta).is_err());
    assert!(b.complete(&tb).is_ok());
}
#[test]
fn bounded_cache_rejects_invalid_inputs() {
    for (e, s) in [(0, 1), (3, 0), (2, 3), (4097, 1)] {
        assert!(Cache::new(e, s).is_err());
    }
    assert!(Cache::new(2, 1).unwrap().request(2).is_err());
}
