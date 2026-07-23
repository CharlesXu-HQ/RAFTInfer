use brt_runtime::{Engine, EngineConfig};

#[test]
fn creates_host_engine_and_reports_backend() {
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");
    assert!(!engine.cuda_enabled());
}

#[test]
fn rejects_zero_pool_size() {
    let error = Engine::new(EngineConfig {
        initial_pool_bytes: 0,
        ..EngineConfig::default()
    })
    .expect_err("zero pool must fail");
    assert!(error.to_string().contains("initial_pool_bytes"));
}

#[test]
fn smoke_reports_unavailable_without_cuda_backend() {
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");

    let error = engine
        .run_smoke()
        .expect_err("host-only build cannot run CUDA smoke");

    assert_eq!(error.code(), 2);
    assert!(error.to_string().contains("CUDA backend is not enabled"));
}
