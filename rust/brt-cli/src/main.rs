use brt_runtime::{Engine, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = std::env::args().nth(1).unwrap_or_else(|| "info".into());
    if command != "info" && command != "host-info" {
        return Err(format!("unknown command: {command}").into());
    }

    let engine = Engine::new(EngineConfig::default())?;
    println!(
        "backend={}",
        if engine.cuda_enabled() {
            "cuda"
        } else {
            "host"
        }
    );
    Ok(())
}
