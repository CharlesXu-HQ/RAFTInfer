use std::process::Command;

#[test]
fn info_reports_the_host_backend() {
    let output = Command::new(env!("CARGO_BIN_EXE_brt-cli"))
        .arg("info")
        .output()
        .expect("run brt-cli");

    assert!(output.status.success());
    assert_eq!(output.stdout, b"backend=host\n");
}

#[test]
fn host_info_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_brt-cli"))
        .arg("host-info")
        .output()
        .expect("run brt-cli");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command: host-info"));
}
