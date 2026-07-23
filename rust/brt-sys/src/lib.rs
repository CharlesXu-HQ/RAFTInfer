use std::ffi::{c_char, c_int};

#[repr(C)]
pub struct BrtEngineHandle {
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
    pub fn brt_last_error_message() -> *const c_char;
}
