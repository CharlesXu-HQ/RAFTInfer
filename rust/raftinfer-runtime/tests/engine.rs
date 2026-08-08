use raftinfer_runtime::{
    BenchmarkConfig, Engine, EngineConfig, GenerationConfig, GenerationSession, KvCacheDType,
    KvCacheLayout, Model, Qwen35AttentionImplementation, Qwen35DecodeAttentionImplementation,
    Qwen35ExecutionPolicy, SessionConfig, TokenResult, benchmark_session, generate_token_ids,
};
use std::{
    collections::VecDeque,
    fs::File,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

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

#[test]
fn peak_gpu_allocation_reports_unavailable_without_cuda_backend() {
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");

    let error = engine
        .peak_allocated_gpu_bytes()
        .expect_err("host-only build cannot report GPU allocation");

    assert_eq!(error.code(), 2);
    assert!(error.to_string().contains("CUDA backend is not enabled"));
}

#[test]
fn qwen35_execution_policy_defaults_to_online_f32_token_major_graphs() {
    assert_eq!(
        Qwen35ExecutionPolicy::default(),
        Qwen35ExecutionPolicy {
            attention: Qwen35AttentionImplementation::OnlineTiled,
            kv_cache_dtype: KvCacheDType::F32,
            kv_cache_layout: KvCacheLayout::TokenMajor,
            decode_graph: true,
            grouped_input_casts: true,
        }
    );
    assert_eq!(
        SessionConfig::default().qwen35_policy,
        Qwen35ExecutionPolicy::default()
    );
}

#[test]
fn split_k_diagnostics_have_stable_native_values_and_ffi_field_order() {
    assert_eq!(
        raftinfer_sys::RAFTINFER_QWEN35_DECODE_ATTENTION_SINGLE_BLOCK,
        0
    );
    assert_eq!(raftinfer_sys::RAFTINFER_QWEN35_DECODE_ATTENTION_SPLIT_K, 1);
    assert_eq!(
        Qwen35DecodeAttentionImplementation::SingleBlock as u32,
        raftinfer_sys::RAFTINFER_QWEN35_DECODE_ATTENTION_SINGLE_BLOCK
    );
    assert_eq!(
        Qwen35DecodeAttentionImplementation::SplitK as u32,
        raftinfer_sys::RAFTINFER_QWEN35_DECODE_ATTENTION_SPLIT_K
    );

    let diagnostics = raftinfer_sys::RaftInferSessionDiagnostics::default();
    let base = std::ptr::addr_of!(diagnostics) as usize;
    let workspace = std::ptr::addr_of!(diagnostics.attention_workspace_bytes) as usize - base;
    let decode_attention = std::ptr::addr_of!(diagnostics.decode_attention) as usize - base;
    let partition =
        std::ptr::addr_of!(diagnostics.decode_attention_partition_tokens) as usize - base;
    let threshold =
        std::ptr::addr_of!(diagnostics.decode_attention_threshold_tokens) as usize - base;
    let bucket =
        std::ptr::addr_of!(diagnostics.decode_attention_context_bucket_tokens) as usize - base;
    let graph =
        std::ptr::addr_of!(diagnostics.decode_attention_split_k_graph_captured) as usize - base;
    assert!(workspace < decode_attention);
    assert!(decode_attention < partition);
    assert!(partition < threshold);
    assert!(threshold < bucket);
    assert!(bucket < graph);
}

#[test]
fn missing_model_is_an_atomic_load_failure() {
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");
    let error = engine
        .load_model("/missing/raftinfer-qwen35.gguf")
        .expect_err("missing model must fail");
    assert!(error.to_string().contains("raftinfer-qwen35.gguf"));
}

#[test]
fn session_wraps_coarse_native_prefill_decode_and_reset() {
    let fixture = TemporaryGguf::new();
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");
    let model = engine
        .load_model(fixture.path())
        .expect("fixture model load");
    let mut session = model
        .create_session(SessionConfig {
            max_context_tokens: 8,
            qwen35_policy: Qwen35ExecutionPolicy {
                kv_cache_dtype: KvCacheDType::Bf16,
                kv_cache_layout: KvCacheLayout::HeadMajor,
                ..Qwen35ExecutionPolicy::default()
            },
        })
        .expect("session creation");

    let diagnostics_error = session
        .diagnostics()
        .expect_err("host-only diagnostics require CUDA execution state");
    assert_eq!(diagnostics_error.code(), 2);
    assert!(
        diagnostics_error
            .to_string()
            .contains("execution diagnostics")
    );

    let decode_error = session
        .decode(7)
        .expect_err("host-only decode has no fallback inference");
    assert_eq!(decode_error.code(), 2);
    assert!(decode_error.to_string().contains("decode backend"));

    let prefill_error = session
        .prefill(&[1, 2, 3])
        .expect_err("host-only prefill has no fallback inference");
    assert_eq!(prefill_error.code(), 2);
    assert!(prefill_error.to_string().contains("prefill backend"));

    session.reset().expect("reset remains available");
}

#[test]
fn session_rejects_zero_context_in_rust() {
    let fixture = TemporaryGguf::new();
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");
    let model = engine
        .load_model(fixture.path())
        .expect("fixture model load");
    let error = model
        .create_session(SessionConfig {
            max_context_tokens: 0,
            ..SessionConfig::default()
        })
        .expect_err("zero context must fail");
    assert_eq!(error.code(), 1);
    assert!(error.to_string().contains("max_context_tokens"));
}

#[test]
fn greedy_generation_prefills_once_and_decodes_once_per_following_token() {
    let mut session = ScriptedSession::new(
        [Ok(token_result(20))],
        [Ok(token_result(21)), Ok(token_result(22))],
    );

    let generated = generate_token_ids(
        &mut session,
        &[10, 11],
        GenerationConfig {
            max_new_tokens: 3,
            context_tokens: 8,
        },
        |_| false,
    )
    .expect("generation");

    assert_eq!(generated, vec![20, 21, 22]);
    assert_eq!(session.reset_calls, 1);
    assert_eq!(session.prefill_inputs, vec![vec![10, 11]]);
    assert_eq!(session.decode_inputs, vec![20, 21]);
}

#[test]
fn greedy_generation_omits_eos_returned_by_prefill() {
    let mut session = ScriptedSession::new([Ok(token_result(99))], []);

    let generated = generate_token_ids(
        &mut session,
        &[10],
        GenerationConfig {
            max_new_tokens: 4,
            context_tokens: 8,
        },
        |token| token == 99,
    )
    .expect("generation");

    assert!(generated.is_empty());
    assert!(session.decode_inputs.is_empty());
}

#[test]
fn greedy_generation_stops_on_eot_returned_by_decode() {
    let mut session = ScriptedSession::new([Ok(token_result(20))], [Ok(token_result(98))]);

    let generated = generate_token_ids(
        &mut session,
        &[10],
        GenerationConfig {
            max_new_tokens: 4,
            context_tokens: 8,
        },
        |token| matches!(token, 98 | 99),
    )
    .expect("generation");

    assert_eq!(generated, vec![20]);
    assert_eq!(session.decode_inputs, vec![20]);
}

#[test]
fn greedy_generation_obeys_the_max_token_limit_without_an_extra_decode() {
    let mut session = ScriptedSession::new([Ok(token_result(20))], []);

    let generated = generate_token_ids(
        &mut session,
        &[10],
        GenerationConfig {
            max_new_tokens: 1,
            context_tokens: 8,
        },
        |_| false,
    )
    .expect("generation");

    assert_eq!(generated, vec![20]);
    assert!(session.decode_inputs.is_empty());
}

#[test]
fn greedy_generation_rejects_context_overflow_before_touching_the_session() {
    let mut session = ScriptedSession::new([], []);

    let error = generate_token_ids(
        &mut session,
        &[10, 11, 12],
        GenerationConfig {
            max_new_tokens: 2,
            context_tokens: 4,
        },
        |_| false,
    )
    .expect_err("three prompt and two generated tokens do not fit");

    assert!(
        error
            .to_string()
            .contains("3 prompt tokens plus 2 requested new tokens exceed the 4-token context")
    );
    assert_eq!(session.reset_calls, 0);
    assert!(session.prefill_inputs.is_empty());
}

#[test]
fn greedy_generation_resets_the_session_for_safe_reuse() {
    let mut session = ScriptedSession::new([Ok(token_result(20)), Ok(token_result(30))], []);
    let config = GenerationConfig {
        max_new_tokens: 1,
        context_tokens: 8,
    };

    let first =
        generate_token_ids(&mut session, &[10], config, |_| false).expect("first generation");
    let second =
        generate_token_ids(&mut session, &[11], config, |_| false).expect("second generation");

    assert_eq!(first, vec![20]);
    assert_eq!(second, vec![30]);
    assert_eq!(session.reset_calls, 2);
    assert_eq!(session.prefill_inputs, vec![vec![10], vec![11]]);
}

#[test]
fn benchmark_warms_up_then_measures_prefill_and_generated_tokens_in_one_session() {
    let mut session = ScriptedSession::new(
        [
            Ok(token_result(20)),
            Ok(token_result(30)),
            Ok(token_result(40)),
        ],
        [
            Ok(token_result(21)),
            Ok(token_result(22)),
            Ok(token_result(23)),
            Ok(token_result(31)),
            Ok(token_result(32)),
            Ok(token_result(33)),
            Ok(token_result(41)),
            Ok(token_result(42)),
            Ok(token_result(43)),
        ],
    );

    let timings = benchmark_session(
        &mut session,
        &[10, 11],
        BenchmarkConfig {
            warmup_iterations: 2,
            measured_iterations: 1,
            generated_tokens: 3,
        },
    )
    .expect("benchmark");

    assert_eq!(timings.prefill_microseconds.len(), 1);
    assert_eq!(timings.generation_microseconds.len(), 1);
    assert_eq!(session.reset_calls, 3);
    assert_eq!(
        session.prefill_inputs,
        vec![vec![10, 11], vec![10, 11], vec![10, 11]]
    );
    assert_eq!(
        session.decode_inputs,
        vec![20, 21, 22, 30, 31, 32, 40, 41, 42]
    );
}

#[test]
fn benchmark_uses_one_batched_decode_per_iteration_when_available() {
    let mut session = BatchedScriptedSession::new(
        [
            Ok(token_result(20)),
            Ok(token_result(30)),
            Ok(token_result(40)),
        ],
        [
            Ok((vec![21, 22, 23], token_result(23))),
            Ok((vec![31, 32, 33], token_result(33))),
            Ok((vec![41, 42, 43], token_result(43))),
        ],
    );

    let timings = benchmark_session(
        &mut session,
        &[10, 11],
        BenchmarkConfig {
            warmup_iterations: 2,
            measured_iterations: 1,
            generated_tokens: 3,
        },
    )
    .expect("benchmark");

    assert_eq!(timings.prefill_microseconds.len(), 1);
    assert_eq!(timings.generation_microseconds.len(), 1);
    assert_eq!(session.reset_calls, 3);
    assert_eq!(
        session.batched_decode_inputs,
        vec![(20, 3), (30, 3), (40, 3)]
    );
    assert!(session.decode_inputs.is_empty());
}

#[allow(dead_code)]
fn session_lifetime_is_tied_to_model<'engine, 'model>(
    model: &'model Model<'engine>,
) -> raftinfer_runtime::Session<'model, 'engine> {
    model
        .create_session(SessionConfig {
            max_context_tokens: 8,
            ..SessionConfig::default()
        })
        .expect("session creation")
}

fn token_result(token_id: i32) -> TokenResult {
    TokenResult {
        token_id,
        position: 0,
    }
}

struct ScriptedSession {
    reset_calls: usize,
    prefill_inputs: Vec<Vec<i32>>,
    decode_inputs: Vec<i32>,
    prefill_results: VecDeque<Result<TokenResult, &'static str>>,
    decode_results: VecDeque<Result<TokenResult, &'static str>>,
}

struct BatchedScriptedSession {
    reset_calls: usize,
    prefill_inputs: Vec<Vec<i32>>,
    decode_inputs: Vec<i32>,
    batched_decode_inputs: Vec<(i32, usize)>,
    prefill_results: VecDeque<Result<TokenResult, &'static str>>,
    batched_decode_results: VecDeque<Result<(Vec<i32>, TokenResult), &'static str>>,
}

impl BatchedScriptedSession {
    fn new(
        prefill_results: impl IntoIterator<Item = Result<TokenResult, &'static str>>,
        batched_decode_results: impl IntoIterator<Item = Result<(Vec<i32>, TokenResult), &'static str>>,
    ) -> Self {
        Self {
            reset_calls: 0,
            prefill_inputs: Vec::new(),
            decode_inputs: Vec::new(),
            batched_decode_inputs: Vec::new(),
            prefill_results: prefill_results.into_iter().collect(),
            batched_decode_results: batched_decode_results.into_iter().collect(),
        }
    }
}

impl ScriptedSession {
    fn new(
        prefill_results: impl IntoIterator<Item = Result<TokenResult, &'static str>>,
        decode_results: impl IntoIterator<Item = Result<TokenResult, &'static str>>,
    ) -> Self {
        Self {
            reset_calls: 0,
            prefill_inputs: Vec::new(),
            decode_inputs: Vec::new(),
            prefill_results: prefill_results.into_iter().collect(),
            decode_results: decode_results.into_iter().collect(),
        }
    }
}

impl GenerationSession for ScriptedSession {
    type Error = &'static str;

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.reset_calls += 1;
        Ok(())
    }

    fn prefill(&mut self, tokens: &[i32]) -> Result<TokenResult, Self::Error> {
        self.prefill_inputs.push(tokens.to_vec());
        self.prefill_results
            .pop_front()
            .expect("scripted prefill result")
    }

    fn decode(&mut self, token_id: i32) -> Result<TokenResult, Self::Error> {
        self.decode_inputs.push(token_id);
        self.decode_results
            .pop_front()
            .expect("scripted decode result")
    }
}

impl GenerationSession for BatchedScriptedSession {
    type Error = &'static str;

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.reset_calls += 1;
        Ok(())
    }

    fn prefill(&mut self, tokens: &[i32]) -> Result<TokenResult, Self::Error> {
        self.prefill_inputs.push(tokens.to_vec());
        self.prefill_results
            .pop_front()
            .expect("scripted prefill result")
    }

    fn decode(&mut self, token_id: i32) -> Result<TokenResult, Self::Error> {
        self.decode_inputs.push(token_id);
        Err("single decode should not be used")
    }

    fn decode_greedy(
        &mut self,
        first_token_id: i32,
        output_tokens: &mut [i32],
    ) -> Result<TokenResult, Self::Error> {
        self.batched_decode_inputs
            .push((first_token_id, output_tokens.len()));
        let (tokens, result) = self
            .batched_decode_results
            .pop_front()
            .expect("scripted batched decode result")?;
        assert_eq!(tokens.len(), output_tokens.len());
        output_tokens.copy_from_slice(&tokens);
        Ok(result)
    }
}

struct TemporaryGguf {
    path: PathBuf,
}

impl TemporaryGguf {
    fn new() -> Self {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        path.push(format!("raftinfer-qwen35-rust-{nanos}.gguf"));
        let bytes = make_qwen35_gguf_fixture();
        let mut file = File::create(&path).expect("create fixture");
        file.write_all(&bytes).expect("write fixture");
        Self { path }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TemporaryGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Clone)]
struct Tensor {
    name: String,
    dimensions: Vec<u64>,
    offset: u64,
}

fn make_qwen35_gguf_fixture() -> Vec<u8> {
    const VOCABULARY_SIZE: u32 = 16;
    const HIDDEN_SIZE: u32 = 8;
    const INTERMEDIATE_SIZE: u32 = 16;
    const FULL_HEAD_COUNT: u32 = 2;
    const FULL_KV_HEAD_COUNT: u32 = 1;
    const FULL_HEAD_DIMENSION: u32 = 4;
    const LINEAR_KEY_HEAD_COUNT: u32 = 1;
    const LINEAR_VALUE_HEAD_COUNT: u32 = 2;
    const LINEAR_HEAD_DIMENSION: u32 = 4;
    const LINEAR_KEY_WIDTH: u32 = LINEAR_KEY_HEAD_COUNT * LINEAR_HEAD_DIMENSION;
    const LINEAR_VALUE_WIDTH: u32 = LINEAR_VALUE_HEAD_COUNT * LINEAR_HEAD_DIMENSION;
    const LINEAR_QKV_WIDTH: u32 = LINEAR_KEY_WIDTH * 2 + LINEAR_VALUE_WIDTH;

    let mut metadata = Vec::new();
    let mut metadata_count = 0_u64;
    push_string_metadata(&mut metadata, "general.architecture", "qwen35");
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "general.alignment", 32);
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "qwen35.embedding_length", HIDDEN_SIZE);
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.feed_forward_length",
        INTERMEDIATE_SIZE,
    );
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "qwen35.context_length", 128);
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "qwen35.block_count", 4);
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.attention.head_count",
        FULL_HEAD_COUNT,
    );
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.attention.head_count_kv",
        FULL_KV_HEAD_COUNT,
    );
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.attention.key_length",
        FULL_HEAD_DIMENSION,
    );
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.attention.value_length",
        FULL_HEAD_DIMENSION,
    );
    metadata_count += 1;
    push_f32_metadata(
        &mut metadata,
        "qwen35.attention.layer_norm_rms_epsilon",
        1.0e-6,
    );
    metadata_count += 1;
    push_f32_metadata(&mut metadata, "qwen35.rope.freq_base", 10_000.0);
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "qwen35.rope.dimension_count", 2);
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "qwen35.ssm.conv_kernel", 4);
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.ssm.state_size",
        LINEAR_HEAD_DIMENSION,
    );
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.ssm.group_count",
        LINEAR_KEY_HEAD_COUNT,
    );
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.ssm.time_step_rank",
        LINEAR_VALUE_HEAD_COUNT,
    );
    metadata_count += 1;
    push_u32_metadata(
        &mut metadata,
        "qwen35.ssm.inner_size",
        LINEAR_VALUE_HEAD_COUNT * LINEAR_HEAD_DIMENSION,
    );
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "qwen35.full_attention_interval", 4);
    metadata_count += 1;
    push_string_metadata(&mut metadata, "tokenizer.ggml.model", "gpt2");
    metadata_count += 1;
    push_string_metadata(&mut metadata, "tokenizer.ggml.pre", "qwen2");
    metadata_count += 1;
    push_string_array_metadata(
        &mut metadata,
        "tokenizer.ggml.tokens",
        &[
            "<eos>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        ],
    );
    metadata_count += 1;
    push_string_array_metadata(&mut metadata, "tokenizer.ggml.merges", &["a b", "b c"]);
    metadata_count += 1;
    push_u32_metadata(&mut metadata, "tokenizer.ggml.eos_token_id", 0);
    metadata_count += 1;
    push_string_metadata(&mut metadata, "tokenizer.chat_template", "{{ messages }}");
    metadata_count += 1;

    let mut tensors = Vec::new();
    let mut next_offset = 0_u64;
    add_tensor(
        &mut tensors,
        &mut next_offset,
        "token_embd.weight",
        &[HIDDEN_SIZE as u64, VOCABULARY_SIZE as u64],
    );
    add_tensor(
        &mut tensors,
        &mut next_offset,
        "output_norm.weight",
        &[HIDDEN_SIZE as u64],
    );
    add_tensor(
        &mut tensors,
        &mut next_offset,
        "output.weight",
        &[HIDDEN_SIZE as u64, VOCABULARY_SIZE as u64],
    );

    for block in 0..4 {
        add_common_block_tensors(
            &mut tensors,
            &mut next_offset,
            block,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
        );
        if block == 3 {
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_q.weight"),
                &[
                    HIDDEN_SIZE as u64,
                    (FULL_HEAD_COUNT * FULL_HEAD_DIMENSION * 2) as u64,
                ],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_k.weight"),
                &[
                    HIDDEN_SIZE as u64,
                    (FULL_KV_HEAD_COUNT * FULL_HEAD_DIMENSION) as u64,
                ],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_v.weight"),
                &[
                    HIDDEN_SIZE as u64,
                    (FULL_KV_HEAD_COUNT * FULL_HEAD_DIMENSION) as u64,
                ],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_output.weight"),
                &[
                    (FULL_HEAD_COUNT * FULL_HEAD_DIMENSION) as u64,
                    HIDDEN_SIZE as u64,
                ],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_q_norm.weight"),
                &[FULL_HEAD_DIMENSION as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_k_norm.weight"),
                &[FULL_HEAD_DIMENSION as u64],
            );
        } else {
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_qkv.weight"),
                &[HIDDEN_SIZE as u64, LINEAR_QKV_WIDTH as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "attn_gate.weight"),
                &[HIDDEN_SIZE as u64, LINEAR_VALUE_WIDTH as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_conv1d.weight"),
                &[4, LINEAR_QKV_WIDTH as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_dt.bias"),
                &[LINEAR_VALUE_HEAD_COUNT as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_a"),
                &[LINEAR_VALUE_HEAD_COUNT as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_beta.weight"),
                &[HIDDEN_SIZE as u64, LINEAR_VALUE_HEAD_COUNT as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_alpha.weight"),
                &[HIDDEN_SIZE as u64, LINEAR_VALUE_HEAD_COUNT as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_norm.weight"),
                &[LINEAR_HEAD_DIMENSION as u64],
            );
            add_tensor(
                &mut tensors,
                &mut next_offset,
                &block_name(block, "ssm_out.weight"),
                &[LINEAR_VALUE_WIDTH as u64, HIDDEN_SIZE as u64],
            );
        }
    }

    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, tensors.len() as u64);
    push_u64(&mut bytes, metadata_count);
    bytes.extend_from_slice(&metadata);
    for tensor in &tensors {
        push_string(&mut bytes, &tensor.name);
        push_u32(&mut bytes, tensor.dimensions.len() as u32);
        for dimension in &tensor.dimensions {
            push_u64(&mut bytes, *dimension);
        }
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, tensor.offset);
    }
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    bytes.resize(bytes.len() + next_offset as usize, 0);
    bytes
}

fn add_common_block_tensors(
    tensors: &mut Vec<Tensor>,
    next_offset: &mut u64,
    block: u32,
    hidden_size: u32,
    intermediate_size: u32,
) {
    add_tensor(
        tensors,
        next_offset,
        &block_name(block, "attn_norm.weight"),
        &[hidden_size as u64],
    );
    add_tensor(
        tensors,
        next_offset,
        &block_name(block, "post_attention_norm.weight"),
        &[hidden_size as u64],
    );
    add_tensor(
        tensors,
        next_offset,
        &block_name(block, "ffn_gate.weight"),
        &[hidden_size as u64, intermediate_size as u64],
    );
    add_tensor(
        tensors,
        next_offset,
        &block_name(block, "ffn_down.weight"),
        &[intermediate_size as u64, hidden_size as u64],
    );
    add_tensor(
        tensors,
        next_offset,
        &block_name(block, "ffn_up.weight"),
        &[hidden_size as u64, intermediate_size as u64],
    );
}

fn add_tensor(tensors: &mut Vec<Tensor>, next_offset: &mut u64, name: &str, dimensions: &[u64]) {
    let elements = dimensions.iter().copied().product::<u64>();
    let byte_size = elements * 2;
    tensors.push(Tensor {
        name: name.to_owned(),
        dimensions: dimensions.to_vec(),
        offset: *next_offset,
    });
    *next_offset = (*next_offset + byte_size).div_ceil(32) * 32;
}

fn block_name(index: u32, suffix: &str) -> String {
    format!("blk.{index}.{suffix}")
}

fn push_key(bytes: &mut Vec<u8>, key: &str, ty: u32) {
    push_string(bytes, key);
    push_u32(bytes, ty);
}

fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_key(bytes, key, 4);
    push_u32(bytes, value);
}

fn push_f32_metadata(bytes: &mut Vec<u8>, key: &str, value: f32) {
    push_key(bytes, key, 6);
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_key(bytes, key, 8);
    push_string(bytes, value);
}

fn push_string_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
    push_key(bytes, key, 9);
    push_u32(bytes, 8);
    push_u64(bytes, values.len() as u64);
    for value in values {
        push_string(bytes, value);
    }
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
