use std::ffi::{c_char, c_int};

pub const BRT_STATUS_UNSUPPORTED: c_int = 5;
pub const BRT_STATUS_RESOURCE_EXHAUSTED: c_int = 6;

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
pub struct BrtSessionConfig {
    pub struct_size: usize,
    pub max_context_tokens: u32,
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
    pub fn brt_session_reset(session: *mut BrtSessionHandle) -> BrtStatus;
    pub fn brt_session_destroy(session: *mut BrtSessionHandle);
    pub fn brt_owned_buffer_free(buffer: *mut BrtOwnedBuffer);
    pub fn brt_last_error_message() -> *const c_char;
}
