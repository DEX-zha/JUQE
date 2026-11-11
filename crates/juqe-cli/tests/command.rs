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

#[test]
fn inspect_and_verify_work_with_real_files_and_native_paths() {
    use std::fs;
    let path = std::env::temp_dir().join(format!("juqe-cli-{}", std::process::id()));
    fs::create_dir(&path).unwrap();
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let directory = Directory(path);
    let index = directory.0.join("experts.idx");
    #[cfg(target_os = "linux")]
    let filename = OsString::from_vec(b"weights-\xff.bin".to_vec());
    #[cfg(not(target_os = "linux"))]
    let filename = OsString::from("weights with spaces.bin");
    let weights = directory.0.join(filename);
    fs::write(
        &index,
        "1 3\n0 0 4096 0 128 1 0 512 0 128 1 1024 512 0 128 1 2560 512\n",
    )
    .unwrap();
    let bytes: Vec<u8> = (0..12288).map(|offset| (offset % 251) as u8).collect();
    fs::write(&weights, bytes).unwrap();
    let inspect = Command::new(env!("CARGO_BIN_EXE_juqe"))
        .arg("inspect")
        .arg(&index)
        .output()
        .unwrap();
    assert!(inspect.status.success());
    assert!(String::from_utf8_lossy(&inspect.stdout).contains("required_file_bytes=12288"));
    let verify = Command::new(env!("CARGO_BIN_EXE_juqe"))
        .arg("verify")
        .arg(&index)
        .arg(&weights)
        .args(["0", "2"])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    assert!(
        String::from_utf8_lossy(&verify.stdout)
            .contains("verified=true layer=0 expert=2 path=BufferedStaging")
    );
    let invalid = Command::new(env!("CARGO_BIN_EXE_juqe"))
        .arg("verify")
        .arg(&index)
        .arg(&weights)
        .args(["-1", "2"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(
        String::from_utf8_lossy(&invalid.stderr).contains("layer must be a nonnegative integer")
    );
}
