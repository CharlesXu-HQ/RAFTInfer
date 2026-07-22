use std::process::Command;

#[test]
fn host_info_reports_the_host_backend() {
    let output = Command::new(env!("CARGO_BIN_EXE_brt-cli"))
        .arg("host-info")
        .output()
        .expect("run brt-cli");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "backend=host"
    );
}
