use brt_runtime::{
    BenchmarkConfig, ChatGeneration, Engine, EngineConfig, ExecutionDiagnostics, GenerationConfig,
    KvCacheDType, KvCacheLayout, Qwen35AttentionImplementation, Qwen35ExecutionPolicy,
    SessionConfig, benchmark_session, generate_chat,
    tokenizer::{ChatMessage, ChatRole, Tokenizer},
};
use std::fmt::Write;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let (command, command_arguments) = arguments
        .split_first()
        .map_or(("info", &[][..]), |(command, rest)| {
            (command.as_str(), rest)
        });
    match command {
        "info" => {
            if let Some(argument) = command_arguments.first() {
                return Err(format!("unknown info argument: {argument}").into());
            }
            run_info()
        }
        "generate" => run_generate(parse_generate_arguments(command_arguments)?),
        "benchmark" => run_benchmark(parse_benchmark_arguments(command_arguments)?),
        _ => Err(format!("unknown command: {command}").into()),
    }
}

fn run_info() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default())?;
    let backend = if engine.cuda_enabled() {
        "cuda"
    } else {
        "host"
    };
    println!("backend={backend}");
    Ok(())
}

#[derive(Debug)]
struct GenerateArguments {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    context_tokens: usize,
    output_format: OutputFormat,
    kv_cache_dtype: KvCacheDType,
    kv_cache_layout: KvCacheLayout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug)]
struct BenchmarkArguments {
    model: PathBuf,
    prompt: BenchmarkPrompt,
    prompt_tokens: usize,
    generated_tokens: usize,
    context_tokens: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    kv_cache_dtype: KvCacheDType,
    kv_cache_layout: KvCacheLayout,
}

#[derive(Debug)]
enum BenchmarkPrompt {
    Text(String),
    TokenIds(Vec<i32>),
}

fn parse_generate_arguments(
    arguments: &[String],
) -> Result<GenerateArguments, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut prompt = None;
    let mut max_new_tokens = None;
    let mut context_tokens = None;
    let mut output_format = None;
    let mut kv_cache_dtype = None;
    let mut kv_cache_layout = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let destination = match flag {
            "--model" => &mut model,
            "--prompt" => &mut prompt,
            "--max-new-tokens" => &mut max_new_tokens,
            "--context" => &mut context_tokens,
            "--output-format" => &mut output_format,
            "--kv-cache-dtype" => &mut kv_cache_dtype,
            "--kv-cache-layout" => &mut kv_cache_layout,
            _ => return Err(format!("unknown generate argument: {flag}").into()),
        };
        if destination.is_some() {
            return Err(format!("{flag} may only be specified once").into());
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        if value.starts_with("--") {
            return Err(format!("{flag} requires a value").into());
        }
        *destination = Some(value.clone());
        index += 2;
    }

    let model = model.ok_or("--model is required")?;
    if model.is_empty() {
        return Err("--model must not be empty".into());
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    if prompt.is_empty() {
        return Err("--prompt must not be empty".into());
    }
    let max_new_tokens = parse_positive(
        &max_new_tokens.ok_or("--max-new-tokens is required")?,
        "--max-new-tokens",
    )?;
    let context_tokens =
        parse_positive(&context_tokens.ok_or("--context is required")?, "--context")?;
    if max_new_tokens > context_tokens {
        return Err(format!(
            "requested {max_new_tokens} new tokens exceed the {context_tokens}-token context"
        )
        .into());
    }
    if context_tokens > u32::MAX as usize {
        return Err(format!("--context must not exceed {}", u32::MAX).into());
    }
    let output_format = match output_format.as_deref().unwrap_or("text") {
        "text" => OutputFormat::Text,
        "json" => OutputFormat::Json,
        _ => return Err("--output-format must be text or json".into()),
    };
    let kv_cache_dtype = parse_kv_cache_dtype(kv_cache_dtype.as_deref().unwrap_or("f32"))?;
    let kv_cache_layout =
        parse_kv_cache_layout(kv_cache_layout.as_deref().unwrap_or("token-major"))?;

    Ok(GenerateArguments {
        model: model.into(),
        prompt,
        max_new_tokens,
        context_tokens,
        output_format,
        kv_cache_dtype,
        kv_cache_layout,
    })
}

fn parse_positive(value: &str, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be a positive integer").into());
    }
    Ok(parsed)
}

fn parse_benchmark_arguments(
    arguments: &[String],
) -> Result<BenchmarkArguments, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut prompt = None;
    let mut prompt_token_ids = None;
    let mut prompt_tokens = None;
    let mut generated_tokens = None;
    let mut context_tokens = None;
    let mut warmup_iterations = None;
    let mut measured_iterations = None;
    let mut kv_cache_dtype = None;
    let mut kv_cache_layout = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let destination = match flag {
            "--model" => &mut model,
            "--prompt" => &mut prompt,
            "--prompt-token-ids" => &mut prompt_token_ids,
            "--prompt-tokens" => &mut prompt_tokens,
            "--decode-tokens" => &mut generated_tokens,
            "--context" => &mut context_tokens,
            "--warmups" => &mut warmup_iterations,
            "--iterations" => &mut measured_iterations,
            "--kv-cache-dtype" => &mut kv_cache_dtype,
            "--kv-cache-layout" => &mut kv_cache_layout,
            _ => return Err(format!("unknown benchmark argument: {flag}").into()),
        };
        if destination.is_some() {
            return Err(format!("{flag} may only be specified once").into());
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        if value.starts_with("--") {
            return Err(format!("{flag} requires a value").into());
        }
        *destination = Some(value.clone());
        index += 2;
    }

    let model = model.ok_or("--model is required")?;
    if model.is_empty() {
        return Err("--model must not be empty".into());
    }
    let prompt_tokens = parse_positive(
        &prompt_tokens.ok_or("--prompt-tokens is required")?,
        "--prompt-tokens",
    )?;
    let generated_tokens = parse_positive(
        &generated_tokens.ok_or("--decode-tokens is required")?,
        "--decode-tokens",
    )?;
    let context_tokens =
        parse_positive(&context_tokens.ok_or("--context is required")?, "--context")?;
    let warmup_iterations = parse_positive(
        &warmup_iterations.ok_or("--warmups is required")?,
        "--warmups",
    )?;
    let measured_iterations = parse_positive(
        &measured_iterations.ok_or("--iterations is required")?,
        "--iterations",
    )?;
    let required_tokens = prompt_tokens
        .checked_add(generated_tokens)
        .ok_or("prompt and generated token counts overflow usize")?;
    if required_tokens > context_tokens {
        return Err(format!(
            "{prompt_tokens} prompt tokens plus {generated_tokens} generated tokens exceed the \
             {context_tokens}-token context"
        )
        .into());
    }
    if context_tokens > u32::MAX as usize {
        return Err(format!("--context must not exceed {}", u32::MAX).into());
    }
    let prompt = match (prompt, prompt_token_ids) {
        (Some(prompt), None) if !prompt.is_empty() => BenchmarkPrompt::Text(prompt),
        (Some(_), None) => return Err("--prompt must not be empty".into()),
        (None, Some(token_ids)) => {
            let token_ids = parse_token_ids(&token_ids)?;
            if token_ids.len() != prompt_tokens {
                return Err(format!(
                    "--prompt-token-ids contains {} tokens but --prompt-tokens is \
                     {prompt_tokens}",
                    token_ids.len()
                )
                .into());
            }
            BenchmarkPrompt::TokenIds(token_ids)
        }
        _ => return Err("exactly one of --prompt or --prompt-token-ids is required".into()),
    };
    let kv_cache_dtype = parse_kv_cache_dtype(kv_cache_dtype.as_deref().unwrap_or("f32"))?;
    let kv_cache_layout =
        parse_kv_cache_layout(kv_cache_layout.as_deref().unwrap_or("token-major"))?;

    Ok(BenchmarkArguments {
        model: model.into(),
        prompt,
        prompt_tokens,
        generated_tokens,
        context_tokens,
        warmup_iterations,
        measured_iterations,
        kv_cache_dtype,
        kv_cache_layout,
    })
}

fn parse_kv_cache_dtype(value: &str) -> Result<KvCacheDType, Box<dyn std::error::Error>> {
    match value {
        "f32" => Ok(KvCacheDType::F32),
        "bf16" => Ok(KvCacheDType::Bf16),
        _ => Err("--kv-cache-dtype must be f32 or bf16".into()),
    }
}

fn parse_kv_cache_layout(value: &str) -> Result<KvCacheLayout, Box<dyn std::error::Error>> {
    match value {
        "token-major" => Ok(KvCacheLayout::TokenMajor),
        "head-major" => Ok(KvCacheLayout::HeadMajor),
        _ => Err("--kv-cache-layout must be token-major or head-major".into()),
    }
}

fn parse_token_ids(value: &str) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    if value.is_empty() {
        return Err("--prompt-token-ids must contain non-negative i32 values".into());
    }
    value
        .split(',')
        .map(|part| {
            part.parse::<i32>()
                .ok()
                .filter(|token_id| *token_id >= 0)
                .ok_or_else(|| {
                    "--prompt-token-ids must contain non-negative i32 values"
                        .to_owned()
                        .into()
                })
        })
        .collect()
}

fn run_generate(arguments: GenerateArguments) -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default())?;
    if !engine.cuda_enabled() {
        return Err(
            "generation requires a CUDA-enabled BRT build; this binary is host-only".into(),
        );
    }
    let model = engine.load_model(&arguments.model)?;
    let tokenizer = Tokenizer::from_spec(&model.tokenizer_spec()?)?;
    let mut session = model.create_session(SessionConfig {
        max_context_tokens: arguments.context_tokens as u32,
        qwen35_policy: Qwen35ExecutionPolicy {
            kv_cache_dtype: arguments.kv_cache_dtype,
            kv_cache_layout: arguments.kv_cache_layout,
            ..Qwen35ExecutionPolicy::default()
        },
    })?;
    let output = generate_chat(
        &mut session,
        &tokenizer,
        &arguments.prompt,
        GenerationConfig {
            max_new_tokens: arguments.max_new_tokens,
            context_tokens: arguments.context_tokens,
        },
    )?;
    match arguments.output_format {
        OutputFormat::Text => print!("{}", output.text),
        OutputFormat::Json => print!("{}", generation_json(&output)),
    }
    Ok(())
}

fn run_benchmark(arguments: BenchmarkArguments) -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default())?;
    if !engine.cuda_enabled() {
        return Err("benchmark requires a CUDA-enabled BRT build; this binary is host-only".into());
    }
    let model = engine.load_model(&arguments.model)?;
    let prompt_tokens = match arguments.prompt {
        BenchmarkPrompt::Text(prompt) => {
            let tokenizer = Tokenizer::from_spec(&model.tokenizer_spec()?)?;
            let rendered_prompt =
                tokenizer.apply_chat_template(&[ChatMessage::new(ChatRole::User, prompt)], true)?;
            let mut token_ids = tokenizer.encode(&rendered_prompt, false)?;
            if token_ids.len() < arguments.prompt_tokens {
                return Err(format!(
                    "benchmark prompt encoded to {} tokens, fewer than requested {}; provide a \
                     longer --prompt",
                    token_ids.len(),
                    arguments.prompt_tokens
                )
                .into());
            }
            token_ids.truncate(arguments.prompt_tokens);
            token_ids
        }
        BenchmarkPrompt::TokenIds(token_ids) => token_ids,
    };

    let mut session = model.create_session(SessionConfig {
        max_context_tokens: arguments.context_tokens as u32,
        qwen35_policy: Qwen35ExecutionPolicy {
            kv_cache_dtype: arguments.kv_cache_dtype,
            kv_cache_layout: arguments.kv_cache_layout,
            ..Qwen35ExecutionPolicy::default()
        },
    })?;
    let timings = benchmark_session(
        &mut session,
        &prompt_tokens,
        BenchmarkConfig {
            warmup_iterations: arguments.warmup_iterations,
            measured_iterations: arguments.measured_iterations,
            generated_tokens: arguments.generated_tokens,
        },
    )?;
    let output = BenchmarkOutput {
        prompt_tokens: arguments.prompt_tokens,
        generated_tokens: arguments.generated_tokens,
        warmup_iterations: arguments.warmup_iterations,
        measured_iterations: arguments.measured_iterations,
        peak_allocated_gpu_bytes: engine.peak_allocated_gpu_bytes()?,
        execution: session.diagnostics()?,
        prefill: LatencySummary::from_microseconds(timings.prefill_microseconds),
        generation: LatencySummary::from_microseconds(timings.generation_microseconds),
    };
    print!("{}", benchmark_json(&output));
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LatencySummary {
    min_us: f64,
    median_us: f64,
    p95_us: f64,
    max_us: f64,
    coefficient_of_variation: f64,
}

impl LatencySummary {
    fn from_microseconds(mut values: Vec<u64>) -> Self {
        assert!(!values.is_empty(), "latency samples must not be empty");
        let mean_us = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
        let coefficient_of_variation = if mean_us == 0.0 {
            0.0
        } else {
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean_us;
                    delta * delta
                })
                .sum::<f64>()
                / values.len() as f64;
            variance.sqrt() / mean_us
        };
        values.sort_unstable();
        let middle = values.len() / 2;
        let median_us = if values.len().is_multiple_of(2) {
            (values[middle - 1] as f64 + values[middle] as f64) / 2.0
        } else {
            values[middle] as f64
        };
        let p95_index = (95 * values.len()).div_ceil(100) - 1;
        Self {
            min_us: values[0] as f64,
            median_us,
            p95_us: values[p95_index] as f64,
            max_us: values[values.len() - 1] as f64,
            coefficient_of_variation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BenchmarkOutput {
    prompt_tokens: usize,
    generated_tokens: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    peak_allocated_gpu_bytes: u64,
    execution: ExecutionDiagnostics,
    prefill: LatencySummary,
    generation: LatencySummary,
}

fn benchmark_json(benchmark: &BenchmarkOutput) -> String {
    let prefill_tokens_per_second =
        benchmark.prompt_tokens as f64 * 1_000_000.0 / benchmark.prefill.median_us;
    let generation_tokens_per_second =
        benchmark.generated_tokens as f64 * 1_000_000.0 / benchmark.generation.median_us;
    format!(
        "{{\"schema_version\":1,\"prompt_tokens\":{},\"generated_tokens\":{},\
         \"warmup_iterations\":{},\"measured_iterations\":{},\
         \"peak_allocated_gpu_bytes\":{},\
         \"execution\":{{\"attention\":\"{}\",\"kv_cache_dtype\":\"{}\",\
         \"kv_cache_layout\":\"{}\",\"decode_graph_enabled\":{},\
         \"decode_graph_captured\":{},\"decode_graph_replayed\":{},\
         \"attention_workspace_bytes\":{}}},\
         \"prefill\":{{\"min_us\":{},\"median_us\":{},\"p95_us\":{},\"max_us\":{},\
         \"coefficient_of_variation\":{},\
         \"tokens_per_second\":{}}},\
         \"generation\":{{\"min_us\":{},\"median_us\":{},\"p95_us\":{},\"max_us\":{},\
         \"coefficient_of_variation\":{},\
         \"tokens_per_second\":{}}}}}\n",
        benchmark.prompt_tokens,
        benchmark.generated_tokens,
        benchmark.warmup_iterations,
        benchmark.measured_iterations,
        benchmark.peak_allocated_gpu_bytes,
        attention_name(benchmark.execution.attention),
        kv_cache_dtype_name(benchmark.execution.kv_cache_dtype),
        kv_cache_layout_name(benchmark.execution.kv_cache_layout),
        benchmark.execution.decode_graph_enabled,
        benchmark.execution.decode_graph_captured,
        benchmark.execution.decode_graph_replayed,
        benchmark.execution.attention_workspace_bytes,
        benchmark.prefill.min_us,
        benchmark.prefill.median_us,
        benchmark.prefill.p95_us,
        benchmark.prefill.max_us,
        benchmark.prefill.coefficient_of_variation,
        prefill_tokens_per_second,
        benchmark.generation.min_us,
        benchmark.generation.median_us,
        benchmark.generation.p95_us,
        benchmark.generation.max_us,
        benchmark.generation.coefficient_of_variation,
        generation_tokens_per_second,
    )
}

fn attention_name(value: Qwen35AttentionImplementation) -> &'static str {
    match value {
        Qwen35AttentionImplementation::MaterializedReference => "materialized_reference",
        Qwen35AttentionImplementation::OnlineTiled => "online_tiled",
    }
}

fn kv_cache_dtype_name(value: KvCacheDType) -> &'static str {
    match value {
        KvCacheDType::F32 => "f32",
        KvCacheDType::Bf16 => "bf16",
    }
}

fn kv_cache_layout_name(value: KvCacheLayout) -> &'static str {
    match value {
        KvCacheLayout::TokenMajor => "token-major",
        KvCacheLayout::HeadMajor => "head-major",
    }
}

fn generation_json(generation: &ChatGeneration) -> String {
    let mut output = String::from("{\"schema_version\":1,\"prompt_token_ids\":[");
    append_token_ids(&mut output, &generation.prompt_token_ids);
    output.push_str("],\"generated_token_ids\":[");
    append_token_ids(&mut output, &generation.generated_token_ids);
    output.push_str("],\"text\":");
    append_json_string(&mut output, &generation.text);
    output.push_str("}\n");
    output
}

fn append_token_ids(output: &mut String, token_ids: &[i32]) {
    for (index, token_id) in token_ids.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{token_id}").expect("writing to a String cannot fail");
    }
}

fn append_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                write!(output, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkOutput, LatencySummary, benchmark_json, generation_json};
    use brt_runtime::{
        ChatGeneration, ExecutionDiagnostics, KvCacheDType, KvCacheLayout,
        Qwen35AttentionImplementation,
    };

    #[test]
    fn generation_json_preserves_token_ids_and_escapes_text() {
        let output = generation_json(&ChatGeneration {
            prompt_token_ids: vec![10, 11],
            generated_token_ids: vec![20, 21],
            text: "line\n\"quoted\"\\tail".to_owned(),
        });

        assert_eq!(
            output,
            "{\"schema_version\":1,\"prompt_token_ids\":[10,11],\
             \"generated_token_ids\":[20,21],\"text\":\"line\\n\\\"quoted\\\"\\\\tail\"}\n"
        );
    }

    #[test]
    fn latency_summary_uses_median_and_nearest_rank_p95() {
        let summary = LatencySummary::from_microseconds(vec![40, 10, 30, 20]);

        assert_eq!(
            summary,
            LatencySummary {
                min_us: 10.0,
                median_us: 25.0,
                p95_us: 40.0,
                max_us: 40.0,
                coefficient_of_variation: 0.447213595499958,
            }
        );
    }

    #[test]
    fn benchmark_json_reports_both_phases_and_throughput() {
        let output = benchmark_json(&BenchmarkOutput {
            prompt_tokens: 128,
            generated_tokens: 128,
            warmup_iterations: 5,
            measured_iterations: 20,
            peak_allocated_gpu_bytes: 18_000_000_000,
            execution: ExecutionDiagnostics {
                attention: Qwen35AttentionImplementation::OnlineTiled,
                kv_cache_dtype: KvCacheDType::Bf16,
                kv_cache_layout: KvCacheLayout::HeadMajor,
                decode_graph_enabled: true,
                decode_graph_captured: true,
                decode_graph_replayed: true,
                attention_workspace_bytes: 2_097_152,
            },
            prefill: LatencySummary {
                min_us: 900.0,
                median_us: 1_000.0,
                p95_us: 1_100.0,
                max_us: 1_200.0,
                coefficient_of_variation: 0.11180339887498948,
            },
            generation: LatencySummary {
                min_us: 1_900.0,
                median_us: 2_000.0,
                p95_us: 2_100.0,
                max_us: 2_200.0,
                coefficient_of_variation: 0.05590169943749474,
            },
        });

        assert_eq!(
            output,
            "{\"schema_version\":1,\"prompt_tokens\":128,\"generated_tokens\":128,\
             \"warmup_iterations\":5,\"measured_iterations\":20,\
             \"peak_allocated_gpu_bytes\":18000000000,\
             \"execution\":{\"attention\":\"online_tiled\",\"kv_cache_dtype\":\"bf16\",\
             \"kv_cache_layout\":\"head-major\",\"decode_graph_enabled\":true,\
             \"decode_graph_captured\":true,\"decode_graph_replayed\":true,\
             \"attention_workspace_bytes\":2097152},\
             \"prefill\":{\"min_us\":900,\"median_us\":1000,\"p95_us\":1100,\
             \"max_us\":1200,\"coefficient_of_variation\":0.11180339887498948,\
             \"tokens_per_second\":128000},\
             \"generation\":{\"min_us\":1900,\"median_us\":2000,\"p95_us\":2100,\
             \"max_us\":2200,\"coefficient_of_variation\":0.05590169943749474,\
             \"tokens_per_second\":64000}}\n"
        );
    }
}
