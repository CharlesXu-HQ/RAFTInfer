use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_raftinfer"))
        .args(args)
        .output()
        .expect("run raftinfer")
}

#[test]
fn info_reports_the_host_backend() {
    let output = run(&["info"]);

    assert!(output.status.success());
    assert_eq!(output.stdout, b"backend=host\n");
}

#[test]
fn host_info_is_rejected() {
    let output = run(&["host-info"]);

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command: host-info"));
}

#[test]
fn generate_requires_every_named_argument() {
    for (args, expected) in [
        (
            vec![
                "generate",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
            ],
            "--model is required",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
            ],
            "--prompt is required",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--context",
                "8",
            ],
            "--max-new-tokens is required",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
            ],
            "--context is required",
        ),
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn generate_rejects_unknown_duplicate_and_missing_values() {
    for (args, expected) in [
        (
            vec!["generate", "--wat", "value"],
            "unknown generate argument: --wat",
        ),
        (
            vec!["generate", "--model", "one.gguf", "--model", "two.gguf"],
            "--model may only be specified once",
        ),
        (vec!["generate", "--model"], "--model requires a value"),
        (
            vec!["generate", "--model", "--prompt", "hello"],
            "--model requires a value",
        ),
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn generate_requires_positive_numeric_limits() {
    for (flag, value, expected) in [
        (
            "--max-new-tokens",
            "0",
            "--max-new-tokens must be a positive integer",
        ),
        ("--context", "0", "--context must be a positive integer"),
        (
            "--context",
            "not-a-number",
            "--context must be a positive integer",
        ),
    ] {
        let mut args = vec![
            "generate",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--context",
            "8",
        ];
        let index = args
            .iter()
            .position(|argument| *argument == flag)
            .expect("flag in fixture");
        args[index + 1] = value;
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn generate_rejects_an_obvious_context_overflow_before_loading_a_model() {
    let output = run(&[
        "generate",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "9",
        "--context",
        "8",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("requested 9 new tokens exceed the 8-token context")
    );
}

#[test]
fn generate_reports_that_host_only_builds_cannot_execute_inference() {
    let output = run(&[
        "generate",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
        "--context",
        "8",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("generation requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn generate_accepts_json_output_format_before_backend_selection() {
    let output = run(&[
        "generate",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
        "--context",
        "8",
        "--output-format",
        "json",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("generation requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn generate_accepts_explicit_kv_policy_before_backend_selection() {
    let output = run(&[
        "generate",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
        "--context",
        "8",
        "--kv-cache-dtype",
        "bf16",
        "--kv-cache-layout",
        "head-major",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("generation requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn generate_accepts_json_output_with_explicit_kv_policy_before_backend_selection() {
    let output = run(&[
        "generate",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
        "--context",
        "8",
        "--output-format",
        "json",
        "--kv-cache-dtype",
        "bf16",
        "--kv-cache-layout",
        "head-major",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("generation requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn generate_rejects_invalid_and_duplicate_output_formats() {
    for (args, expected) in [
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
                "--output-format",
                "xml",
            ],
            "--output-format must be text or json",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
                "--output-format",
                "json",
                "--output-format",
                "text",
            ],
            "--output-format may only be specified once",
        ),
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn generate_rejects_invalid_duplicate_and_missing_kv_policy_values() {
    for (args, expected) in [
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
                "--kv-cache-dtype",
                "fp16",
            ],
            "--kv-cache-dtype must be f32 or bf16",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
                "--kv-cache-layout",
                "layer-major",
            ],
            "--kv-cache-layout must be token-major or head-major",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
                "--kv-cache-dtype",
                "f32",
                "--kv-cache-dtype",
                "bf16",
            ],
            "--kv-cache-dtype may only be specified once",
        ),
        (
            vec![
                "generate",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
                "--context",
                "8",
                "--kv-cache-layout",
            ],
            "--kv-cache-layout requires a value",
        ),
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn benchmark_accepts_a_complete_arm_before_backend_selection() {
    let output = run(&[
        "benchmark",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--prompt-tokens",
        "128",
        "--decode-tokens",
        "128",
        "--context",
        "4096",
        "--warmups",
        "5",
        "--iterations",
        "20",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("benchmark requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn benchmark_accepts_explicit_kv_policy_before_backend_selection() {
    let output = run(&[
        "benchmark",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--prompt-tokens",
        "128",
        "--decode-tokens",
        "128",
        "--context",
        "4096",
        "--warmups",
        "5",
        "--iterations",
        "20",
        "--kv-cache-dtype",
        "bf16",
        "--kv-cache-layout",
        "head-major",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("benchmark requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn benchmark_rejects_invalid_duplicate_and_missing_kv_policy_values() {
    for (args, expected) in [
        (
            vec![
                "benchmark",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--prompt-tokens",
                "128",
                "--decode-tokens",
                "128",
                "--context",
                "4096",
                "--warmups",
                "5",
                "--iterations",
                "20",
                "--kv-cache-dtype",
                "fp16",
            ],
            "--kv-cache-dtype must be f32 or bf16",
        ),
        (
            vec![
                "benchmark",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--prompt-tokens",
                "128",
                "--decode-tokens",
                "128",
                "--context",
                "4096",
                "--warmups",
                "5",
                "--iterations",
                "20",
                "--kv-cache-layout",
                "layer-major",
            ],
            "--kv-cache-layout must be token-major or head-major",
        ),
        (
            vec![
                "benchmark",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--prompt-tokens",
                "128",
                "--decode-tokens",
                "128",
                "--context",
                "4096",
                "--warmups",
                "5",
                "--iterations",
                "20",
                "--kv-cache-layout",
                "token-major",
                "--kv-cache-layout",
                "head-major",
            ],
            "--kv-cache-layout may only be specified once",
        ),
        (
            vec![
                "benchmark",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--prompt-tokens",
                "128",
                "--decode-tokens",
                "128",
                "--context",
                "4096",
                "--warmups",
                "5",
                "--iterations",
                "20",
                "--kv-cache-dtype",
            ],
            "--kv-cache-dtype requires a value",
        ),
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn benchmark_rejects_invalid_lengths_before_loading_a_model() {
    for (flag, value, expected) in [
        (
            "--prompt-tokens",
            "0",
            "--prompt-tokens must be a positive integer",
        ),
        (
            "--decode-tokens",
            "0",
            "--decode-tokens must be a positive integer",
        ),
        ("--warmups", "0", "--warmups must be a positive integer"),
        (
            "--iterations",
            "0",
            "--iterations must be a positive integer",
        ),
    ] {
        let mut args = vec![
            "benchmark",
            "--model",
            "missing.gguf",
            "--prompt",
            "hello",
            "--prompt-tokens",
            "128",
            "--decode-tokens",
            "128",
            "--context",
            "4096",
            "--warmups",
            "5",
            "--iterations",
            "20",
        ];
        let index = args
            .iter()
            .position(|argument| *argument == flag)
            .expect("flag in fixture");
        args[index + 1] = value;

        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn benchmark_rejects_context_overflow_before_loading_a_model() {
    let output = run(&[
        "benchmark",
        "--model",
        "missing.gguf",
        "--prompt",
        "hello",
        "--prompt-tokens",
        "128",
        "--decode-tokens",
        "128",
        "--context",
        "255",
        "--warmups",
        "5",
        "--iterations",
        "20",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("128 prompt tokens plus 128 generated tokens exceed the 255-token context")
    );
}

#[test]
fn benchmark_accepts_explicit_prompt_token_ids_before_backend_selection() {
    let output = run(&[
        "benchmark",
        "--model",
        "missing.gguf",
        "--prompt-token-ids",
        "10,11",
        "--prompt-tokens",
        "2",
        "--decode-tokens",
        "1",
        "--context",
        "3",
        "--warmups",
        "1",
        "--iterations",
        "1",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("benchmark requires a CUDA-enabled RAFTInfer build")
    );
}

#[test]
fn benchmark_rejects_ambiguous_or_malformed_prompt_token_ids() {
    for (extra, expected) in [
        (
            vec!["--prompt", "hello", "--prompt-token-ids", "10,11"],
            "exactly one of --prompt or --prompt-token-ids is required",
        ),
        (
            vec!["--prompt-token-ids", "10,-1"],
            "--prompt-token-ids must contain non-negative i32 values",
        ),
        (
            vec!["--prompt-token-ids", "10,,11"],
            "--prompt-token-ids must contain non-negative i32 values",
        ),
        (
            vec!["--prompt-token-ids", "10,11"],
            "--prompt-token-ids contains 2 tokens but --prompt-tokens is 128",
        ),
    ] {
        let mut args = vec![
            "benchmark",
            "--model",
            "missing.gguf",
            "--prompt-tokens",
            "128",
            "--decode-tokens",
            "128",
            "--context",
            "4096",
            "--warmups",
            "5",
            "--iterations",
            "20",
        ];
        args.extend(extra);

        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
