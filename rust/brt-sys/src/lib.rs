use std::ffi::{c_char, c_int};

pub const BRT_STATUS_UNSUPPORTED: c_int = 5;
pub const BRT_STATUS_RESOURCE_EXHAUSTED: c_int = 6;

pub const BRT_QWEN35_ATTENTION_MATERIALIZED_REFERENCE: u32 = 0;
pub const BRT_QWEN35_ATTENTION_ONLINE_TILED: u32 = 1;
pub const BRT_QWEN35_KV_CACHE_F32: u32 = 0;
pub const BRT_QWEN35_KV_CACHE_BF16: u32 = 1;
pub const BRT_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR: u32 = 0;
pub const BRT_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR: u32 = 1;

#[repr(C)]
pub struct BrtEngineHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct BrtModelHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct BrtSessionHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct BrtEngineConfig {
    pub struct_size: usize,
    pub device_id: i32,
    pub initial_pool_bytes: u64,
}

#[repr(C)]
pub struct BrtStatus {
    pub code: c_int,
    pub message: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BrtSmokeResult {
    pub device_id: i32,
    pub element_count: u32,
    pub checksum: u64,
}

#[repr(C)]
pub struct BrtOwnedBuffer {
    pub struct_size: usize,
    pub version: u32,
    pub data: *mut u8,
    pub size: usize,
}

#[repr(C)]
pub struct BrtQwen35ExecutionPolicy {
    pub struct_size: usize,
    pub attention: u32,
    pub kv_cache_dtype: u32,
    pub kv_cache_layout: u32,
    pub decode_graph: i32,
    pub grouped_input_casts: i32,
}

#[repr(C)]
pub struct BrtSessionConfig {
    pub struct_size: usize,
    pub max_context_tokens: u32,
    pub qwen35_policy: *const BrtQwen35ExecutionPolicy,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BrtSessionDiagnostics {
    pub struct_size: usize,
    pub attention: u32,
    pub kv_cache_dtype: u32,
    pub kv_cache_layout: u32,
    pub decode_graph_enabled: i32,
    pub decode_graph_captured: i32,
    pub decode_graph_replayed: i32,
    pub attention_workspace_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BrtTokenResult {
    pub token_id: i32,
    pub position: u32,
}

unsafe extern "C" {
    pub fn brt_engine_create(
        config: *const BrtEngineConfig,
        out_engine: *mut *mut BrtEngineHandle,
    ) -> BrtStatus;
    pub fn brt_engine_destroy(engine: *mut BrtEngineHandle);
    pub fn brt_engine_is_cuda_enabled(engine: *const BrtEngineHandle) -> i32;
    pub fn brt_engine_run_smoke(
        engine: *mut BrtEngineHandle,
        out_result: *mut BrtSmokeResult,
    ) -> BrtStatus;
    pub fn brt_engine_peak_allocated_gpu_bytes(
        engine: *mut BrtEngineHandle,
        out_peak_allocated_gpu_bytes: *mut u64,
    ) -> BrtStatus;
    pub fn brt_engine_load_model(
        engine: *mut BrtEngineHandle,
        gguf_path: *const c_char,
        out_model: *mut *mut BrtModelHandle,
    ) -> BrtStatus;
    pub fn brt_model_destroy(model: *mut BrtModelHandle);
    pub fn brt_model_copy_tokenizer_spec(
        model: *const BrtModelHandle,
        out_buffer: *mut BrtOwnedBuffer,
    ) -> BrtStatus;
    pub fn brt_session_create(
        model: *mut BrtModelHandle,
        config: *const BrtSessionConfig,
        out_session: *mut *mut BrtSessionHandle,
    ) -> BrtStatus;
    pub fn brt_session_prefill(
        session: *mut BrtSessionHandle,
        tokens: *const i32,
        token_count: usize,
        out_result: *mut BrtTokenResult,
    ) -> BrtStatus;
    pub fn brt_session_decode(
        session: *mut BrtSessionHandle,
        token_id: i32,
        out_result: *mut BrtTokenResult,
    ) -> BrtStatus;
    pub fn brt_session_diagnostics(
        session: *mut BrtSessionHandle,
        out_diagnostics: *mut BrtSessionDiagnostics,
    ) -> BrtStatus;
    pub fn brt_session_reset(session: *mut BrtSessionHandle) -> BrtStatus;
    pub fn brt_session_destroy(session: *mut BrtSessionHandle);
    pub fn brt_owned_buffer_free(buffer: *mut BrtOwnedBuffer);
    pub fn brt_last_error_message() -> *const c_char;
}
