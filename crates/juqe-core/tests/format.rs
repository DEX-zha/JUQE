//! Validated layouts and hostile metadata rejection.
#![allow(clippy::unwrap_used)]
use juqe_core::format::ExpertIndex;

const ROW: &str = "0 0 4096 0 128 1 0 512 0 128 1 1024 512 0 128 1 2560 512";
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
