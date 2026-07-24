use std::ffi::{c_char, c_int};

pub const BRT_STATUS_OK: c_int = 0;
pub const BRT_STATUS_INVALID_ARGUMENT: c_int = 1;
pub const BRT_STATUS_UNAVAILABLE: c_int = 2;
pub const BRT_STATUS_CUDA_ERROR: c_int = 3;
pub const BRT_STATUS_INTERNAL: c_int = 4;
pub const BRT_STATUS_UNSUPPORTED: c_int = 5;
pub const BRT_STATUS_RESOURCE_EXHAUSTED: c_int = 6;

pub type BrtDataType = c_int;
pub const BRT_DTYPE_F32: BrtDataType = 1;
pub const BRT_DTYPE_F16: BrtDataType = 2;
pub const BRT_DTYPE_BF16: BrtDataType = 3;
pub const BRT_DTYPE_Q4_K: BrtDataType = 100;

pub type BrtQuantFormat = c_int;
pub const BRT_QUANT_NONE: BrtQuantFormat = 0;
pub const BRT_QUANT_Q4_K: BrtQuantFormat = 1;

pub type BrtMemoryType = c_int;
pub const BRT_MEMORY_HOST: BrtMemoryType = 0;
pub const BRT_MEMORY_CUDA_DEVICE: BrtMemoryType = 1;

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
pub struct BrtTensorDesc {
    pub data: *mut std::ffi::c_void,
    pub scales: *const std::ffi::c_void,
    pub zero_points: *const std::ffi::c_void,
    pub byte_size: usize,
    pub shape: [i64; 4],
    pub strides: [i64; 4],
    pub rank: u32,
    pub dtype: BrtDataType,
    pub quant: BrtQuantFormat,
    pub memory: BrtMemoryType,
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
