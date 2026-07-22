use std::{ffi::CStr, fmt, ptr::NonNull};

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub device_id: i32,
    pub initial_pool_bytes: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            initial_pool_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug)]
pub struct Error {
    code: i32,
    message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BRT error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct Engine {
    raw: NonNull<brt_sys::BrtEngineHandle>,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self, Error> {
        let native = brt_sys::BrtEngineConfig {
            struct_size: std::mem::size_of::<brt_sys::BrtEngineConfig>(),
            device_id: config.device_id,
            initial_pool_bytes: config.initial_pool_bytes,
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe { brt_sys::brt_engine_create(&native, &mut raw) };
        if status.code != 0 {
            let message = if status.message.is_null() {
                "unknown native error".to_owned()
            } else {
                unsafe { CStr::from_ptr(status.message) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(Error {
                code: status.code,
                message,
            });
        }

        Ok(Self {
            raw: NonNull::new(raw).expect("successful create returned null"),
        })
    }

    pub fn cuda_enabled(&self) -> bool {
        unsafe { brt_sys::brt_engine_is_cuda_enabled(self.raw.as_ptr()) != 0 }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { brt_sys::brt_engine_destroy(self.raw.as_ptr()) }
    }
}
