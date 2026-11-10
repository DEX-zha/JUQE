//! CLI errors must remain actionable and must not panic on Unix arguments.
#![allow(clippy::unwrap_used)]
use std::{ffi::OsString, os::unix::ffi::OsStringExt, process::Command};

#[test]
fn help_and_invalid_commands_have_distinct_exit_status() {
    let help = Command::new(env!("CARGO_BIN_EXE_juqe"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("juqe verify"));
    let invalid = Command::new(env!("CARGO_BIN_EXE_juqe"))
        .arg("verify")
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("Usage:"));
}

#[test]
fn invalid_utf8_argument_returns_an_error_without_panic() {
    let output = Command::new(env!("CARGO_BIN_EXE_juqe"))
        .arg(OsString::from_vec(vec![0xff]))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("valid UTF-8"));
    assert!(!error.contains("panicked"));
}
