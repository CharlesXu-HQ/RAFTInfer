use std::ffi::{c_char, c_int};

pub const RAFTINFER_STATUS_UNSUPPORTED: c_int = 5;
pub const RAFTINFER_STATUS_RESOURCE_EXHAUSTED: c_int = 6;

pub const RAFTINFER_QWEN35_ATTENTION_MATERIALIZED_REFERENCE: u32 = 0;
pub const RAFTINFER_QWEN35_ATTENTION_ONLINE_TILED: u32 = 1;
pub const RAFTINFER_QWEN35_KV_CACHE_F32: u32 = 0;
pub const RAFTINFER_QWEN35_KV_CACHE_BF16: u32 = 1;
pub const RAFTINFER_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR: u32 = 0;
pub const RAFTINFER_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR: u32 = 1;
pub const RAFTINFER_QWEN35_DECODE_ATTENTION_SINGLE_BLOCK: u32 = 0;
pub const RAFTINFER_QWEN35_DECODE_ATTENTION_SPLIT_K: u32 = 1;

#[repr(C)]
pub struct RaftInferEngineHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RaftInferModelHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RaftInferSessionHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RaftInferEngineConfig {
    pub struct_size: usize,
    pub device_id: i32,
    pub initial_pool_bytes: u64,
}

#[repr(C)]
pub struct RaftInferStatus {
    pub code: c_int,
    pub message: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RaftInferSmokeResult {
    pub device_id: i32,
    pub element_count: u32,
    pub checksum: u64,
}

#[repr(C)]
pub struct RaftInferOwnedBuffer {
    pub struct_size: usize,
    pub version: u32,
    pub data: *mut u8,
    pub size: usize,
}

#[repr(C)]
pub struct RaftInferQwen35ExecutionPolicy {
    pub struct_size: usize,
    pub attention: u32,
    pub kv_cache_dtype: u32,
    pub kv_cache_layout: u32,
    pub decode_graph: i32,
    pub grouped_input_casts: i32,
}

#[repr(C)]
pub struct RaftInferSessionConfig {
    pub struct_size: usize,
    pub max_context_tokens: u32,
    pub qwen35_policy: *const RaftInferQwen35ExecutionPolicy,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RaftInferSessionDiagnostics {
    pub struct_size: usize,
    pub attention: u32,
    pub kv_cache_dtype: u32,
    pub kv_cache_layout: u32,
    pub decode_graph_enabled: i32,
    pub decode_graph_captured: i32,
    pub decode_graph_replayed: i32,
    pub attention_workspace_bytes: usize,
    pub decode_attention: u32,
    pub decode_attention_partition_tokens: usize,
    pub decode_attention_threshold_tokens: usize,
    pub decode_attention_context_bucket_tokens: usize,
    pub decode_attention_split_k_graph_captured: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RaftInferTokenResult {
    pub token_id: i32,
    pub position: u32,
}

unsafe extern "C" {
    pub fn raftinfer_engine_create(
        config: *const RaftInferEngineConfig,
        out_engine: *mut *mut RaftInferEngineHandle,
    ) -> RaftInferStatus;
    pub fn raftinfer_engine_destroy(engine: *mut RaftInferEngineHandle);
    pub fn raftinfer_engine_is_cuda_enabled(engine: *const RaftInferEngineHandle) -> i32;
    pub fn raftinfer_engine_run_smoke(
        engine: *mut RaftInferEngineHandle,
        out_result: *mut RaftInferSmokeResult,
    ) -> RaftInferStatus;
    pub fn raftinfer_engine_peak_allocated_gpu_bytes(
        engine: *mut RaftInferEngineHandle,
        out_peak_allocated_gpu_bytes: *mut u64,
    ) -> RaftInferStatus;
    pub fn raftinfer_engine_load_model(
        engine: *mut RaftInferEngineHandle,
        gguf_path: *const c_char,
        out_model: *mut *mut RaftInferModelHandle,
    ) -> RaftInferStatus;
    pub fn raftinfer_model_destroy(model: *mut RaftInferModelHandle);
    pub fn raftinfer_model_copy_tokenizer_spec(
        model: *const RaftInferModelHandle,
        out_buffer: *mut RaftInferOwnedBuffer,
    ) -> RaftInferStatus;
    pub fn raftinfer_session_create(
        model: *mut RaftInferModelHandle,
        config: *const RaftInferSessionConfig,
        out_session: *mut *mut RaftInferSessionHandle,
    ) -> RaftInferStatus;
    pub fn raftinfer_session_prefill(
        session: *mut RaftInferSessionHandle,
        tokens: *const i32,
        token_count: usize,
        out_result: *mut RaftInferTokenResult,
    ) -> RaftInferStatus;
    pub fn raftinfer_session_decode(
        session: *mut RaftInferSessionHandle,
        token_id: i32,
        out_result: *mut RaftInferTokenResult,
    ) -> RaftInferStatus;
    pub fn raftinfer_session_decode_greedy(
        session: *mut RaftInferSessionHandle,
        first_token_id: i32,
        out_token_ids: *mut i32,
        token_count: usize,
        out_result: *mut RaftInferTokenResult,
    ) -> RaftInferStatus;
    pub fn raftinfer_session_diagnostics(
        session: *mut RaftInferSessionHandle,
        out_diagnostics: *mut RaftInferSessionDiagnostics,
    ) -> RaftInferStatus;
    pub fn raftinfer_session_reset(session: *mut RaftInferSessionHandle) -> RaftInferStatus;
    pub fn raftinfer_session_destroy(session: *mut RaftInferSessionHandle);
    pub fn raftinfer_owned_buffer_free(buffer: *mut RaftInferOwnedBuffer);
    pub fn raftinfer_last_error_message() -> *const c_char;
}
