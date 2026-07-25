use brt_runtime::{Engine, EngineConfig, Model, SessionConfig};
use std::{
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
fn missing_model_is_an_atomic_load_failure() {
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");
    let error = engine
        .load_model("/missing/brt-qwen35.gguf")
        .expect_err("missing model must fail");
    assert!(error.to_string().contains("brt-qwen35.gguf"));
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
        })
        .expect("session creation");

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
        })
        .expect_err("zero context must fail");
    assert_eq!(error.code(), 1);
    assert!(error.to_string().contains("max_context_tokens"));
}

#[allow(dead_code)]
fn session_lifetime_is_tied_to_model<'engine, 'model>(
    model: &'model Model<'engine>,
) -> brt_runtime::Session<'model, 'engine> {
    model
        .create_session(SessionConfig {
            max_context_tokens: 8,
        })
        .expect("session creation")
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
        path.push(format!("brt-qwen35-rust-{nanos}.gguf"));
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
