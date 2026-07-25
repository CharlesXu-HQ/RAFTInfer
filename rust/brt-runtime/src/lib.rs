use std::{
    ffi::{CStr, CString},
    fmt,
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

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

#[derive(Debug)]
pub struct Model<'engine> {
    raw: NonNull<brt_sys::BrtModelHandle>,
    _engine: PhantomData<&'engine Engine>,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[derive(Debug)]
pub struct Session<'model, 'engine> {
    raw: NonNull<brt_sys::BrtSessionHandle>,
    _model: PhantomData<&'model Model<'engine>>,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerSpec {
    pub version: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SmokeResult {
    pub device_id: i32,
    pub element_count: u32,
    pub checksum: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SessionConfig {
    pub max_context_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenResult {
    pub token_id: i32,
    pub position: u32,
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

    pub fn load_model<'engine>(
        &'engine self,
        gguf_path: impl AsRef<Path>,
    ) -> Result<Model<'engine>, Error> {
        let path = gguf_path.as_ref().to_string_lossy();
        let native_path = CString::new(path.as_bytes()).map_err(|_| Error {
            code: brt_sys::BRT_STATUS_UNSUPPORTED,
            message: "GGUF path contains an interior NUL byte".to_owned(),
        })?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            brt_sys::brt_engine_load_model(self.raw.as_ptr(), native_path.as_ptr(), &mut raw)
        };
        status_to_result(status)?;
        Ok(Model {
            raw: NonNull::new(raw).expect("successful model load returned null"),
            _engine: PhantomData,
            _not_send_sync: PhantomData,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { brt_sys::brt_engine_destroy(self.raw.as_ptr()) }
    }
}

impl<'engine> Model<'engine> {
    pub fn tokenizer_spec(&self) -> Result<TokenizerSpec, Error> {
        let mut native = brt_sys::BrtOwnedBuffer {
            struct_size: std::mem::size_of::<brt_sys::BrtOwnedBuffer>(),
            version: 0,
            data: std::ptr::null_mut(),
            size: 0,
        };
        let status =
            unsafe { brt_sys::brt_model_copy_tokenizer_spec(self.raw.as_ptr(), &mut native) };
        status_to_result(status)?;
        let bytes = if native.size == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(native.data, native.size) }.to_vec()
        };
        let version = native.version;
        unsafe { brt_sys::brt_owned_buffer_free(&mut native) };
        Ok(TokenizerSpec { version, bytes })
    }

    pub fn create_session<'model>(
        &'model self,
        config: SessionConfig,
    ) -> Result<Session<'model, 'engine>, Error> {
        let native = brt_sys::BrtSessionConfig {
            struct_size: std::mem::size_of::<brt_sys::BrtSessionConfig>(),
            max_context_tokens: config.max_context_tokens,
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe { brt_sys::brt_session_create(self.raw.as_ptr(), &native, &mut raw) };
        status_to_result(status)?;
        Ok(Session {
            raw: NonNull::new(raw).expect("successful session create returned null"),
            _model: PhantomData,
            _not_send_sync: PhantomData,
        })
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        unsafe { brt_sys::brt_model_destroy(self.raw.as_ptr()) }
    }
}

impl Session<'_, '_> {
    pub fn prefill(&mut self, tokens: &[i32]) -> Result<TokenResult, Error> {
        let mut native = brt_sys::BrtTokenResult::default();
        let status = unsafe {
            brt_sys::brt_session_prefill(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                &mut native,
            )
        };
        status_to_result(status).map(|()| TokenResult {
            token_id: native.token_id,
            position: native.position,
        })
    }

    pub fn decode(&mut self, token_id: i32) -> Result<TokenResult, Error> {
        let mut native = brt_sys::BrtTokenResult::default();
        let status =
            unsafe { brt_sys::brt_session_decode(self.raw.as_ptr(), token_id, &mut native) };
        status_to_result(status).map(|()| TokenResult {
            token_id: native.token_id,
            position: native.position,
        })
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let status = unsafe { brt_sys::brt_session_reset(self.raw.as_ptr()) };
        status_to_result(status)
    }
}

impl Drop for Session<'_, '_> {
    fn drop(&mut self) {
        unsafe { brt_sys::brt_session_destroy(self.raw.as_ptr()) }
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
