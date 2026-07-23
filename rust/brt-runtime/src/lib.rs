use std::{ffi::CStr, fmt, marker::PhantomData, ptr::NonNull, rc::Rc};

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

impl Error {
    pub fn code(&self) -> i32 {
        self.code
    }
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
    _not_send_sync: PhantomData<Rc<()>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SmokeResult {
    pub device_id: i32,
    pub element_count: u32,
    pub checksum: u64,
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
        status_to_result(status)?;

        Ok(Self {
            raw: NonNull::new(raw).expect("successful create returned null"),
            _not_send_sync: PhantomData,
        })
    }

    pub fn cuda_enabled(&self) -> bool {
        unsafe { brt_sys::brt_engine_is_cuda_enabled(self.raw.as_ptr()) != 0 }
    }

    pub fn run_smoke(&self) -> Result<SmokeResult, Error> {
        let mut native = brt_sys::BrtSmokeResult::default();
        let status = unsafe { brt_sys::brt_engine_run_smoke(self.raw.as_ptr(), &mut native) };
        status_to_result(status).map(|()| SmokeResult {
            device_id: native.device_id,
            element_count: native.element_count,
            checksum: native.checksum,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { brt_sys::brt_engine_destroy(self.raw.as_ptr()) }
    }
}

fn status_to_result(status: brt_sys::BrtStatus) -> Result<(), Error> {
    if status.code == 0 {
        return Ok(());
    }

    Err(Error {
        code: status.code,
        message: native_error_message(status),
    })
}

fn native_error_message(status: brt_sys::BrtStatus) -> String {
    let message = if status.message.is_null() {
        unsafe { brt_sys::brt_last_error_message() }
    } else {
        status.message
    };

    if message.is_null() {
        "unknown native error".to_owned()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }
}
