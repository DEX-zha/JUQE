//! Cache ownership and hostile metadata regressions.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use juqe_core::{
    cache::{Cache, CacheError, LoadTicket, Request},
    format::ExpertIndex,
};

const ROW: &str = "0 0 4096 0 128 1 0 512 0 128 1 1024 512 0 128 1 2560 512";
fn load(cache: &mut Cache, expert: usize) -> LoadTicket {
    match cache.request(expert).unwrap() {
        Request::Load(t) => t,
        r => panic!("expected load, got {r:?}"),
    }
}

#[test]
fn parses_legacy_layout_and_bounds_experts() {
    let index: ExpertIndex = format!("1 3\n{ROW}\n").parse().unwrap();
    assert_eq!(index.file_bytes(), 12288);
    assert_eq!(index.layers()[0].expert_offset(2).unwrap(), 8192);
    assert!(index.layers()[0].expert_offset(3).is_err());
    assert!(index.validate_file_size(12287).is_err());
    assert!(index.validate_file_size(12288).is_ok());
}
#[test]
fn accepts_dense_placeholder_but_never_reads_it() {
    let row = ROW.replacen("0 0", "1 0", 1);
    let index: ExpertIndex = format!("2 3\n0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n{row}\n")
        .parse()
        .unwrap();
    assert!(index.layers()[0].expert_offset(0).is_err());
    assert_eq!(index.layers()[1].expert_offset(0).unwrap(), 0);
}
#[test]
fn rejects_overlaps_truncation_negative_and_unbounded_metadata() {
    for text in [
        "".to_owned(),
        "1 3\n".to_owned(),
        format!("1 3\n{ROW}\njunk"),
        format!("1 3\n{}", ROW.replace("1024 512", "256 512")),
        format!("1 3\n{}", ROW.replace("2560 512", "4096 512")),
        format!("1 3\n{}", ROW.replace("4096", "18446744073709551615")),
        format!(
            "1 3\n{}",
            ROW.replace("0 0 4096", "0 9223372036854771712 4096")
        ),
        format!("1 3\n{}", ROW.replace("0 0 4096", "0 -1 4096")),
        format!("513 3\n{ROW}"),
        format!("1 4097\n{ROW}"),
    ] {
        assert!(text.parse::<ExpertIndex>().is_err(), "accepted {text}");
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
    assert!(b.complete(&ta).is_err());
    assert!(a.complete(&ta).is_ok());
    assert!(a.complete(&ta).is_err());
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
