use std::{
    ffi::{CStr, CString},
    fmt,
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
    time::Instant,
};

use tokenizer::{ChatMessage, ChatRole, Tokenizer, TokenizerError};

pub mod tokenizer;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub context_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatGeneration {
    pub prompt_token_ids: Vec<i32>,
    pub generated_token_ids: Vec<i32>,
    pub text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub generated_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkTimings {
    pub prefill_microseconds: Vec<u64>,
    pub generation_microseconds: Vec<u64>,
}

pub trait GenerationSession {
    type Error;

    fn reset(&mut self) -> Result<(), Self::Error>;
    fn prefill(&mut self, tokens: &[i32]) -> Result<TokenResult, Self::Error>;
    fn decode(&mut self, token_id: i32) -> Result<TokenResult, Self::Error>;
}

#[derive(Debug)]
pub enum GenerationError<E> {
    InvalidInput(String),
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for GenerationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Backend(error) => write!(f, "generation backend failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for GenerationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInput(_) => None,
            Self::Backend(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum TextGenerationError {
    Tokenizer(TokenizerError),
    Generation(GenerationError<Error>),
}

impl fmt::Display for TextGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenizer(error) => write!(f, "tokenization failed: {error}"),
            Self::Generation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for TextGenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tokenizer(error) => Some(error),
            Self::Generation(error) => Some(error),
        }
    }
}

pub fn generate_token_ids<S, F>(
    session: &mut S,
    prompt_tokens: &[i32],
    config: GenerationConfig,
    mut is_stop_token: F,
) -> Result<Vec<i32>, GenerationError<S::Error>>
where
    S: GenerationSession,
    F: FnMut(i32) -> bool,
{
    if config.max_new_tokens == 0 {
        return Err(GenerationError::InvalidInput(
            "max_new_tokens must be positive".to_owned(),
        ));
    }
    if config.context_tokens == 0 {
        return Err(GenerationError::InvalidInput(
            "context_tokens must be positive".to_owned(),
        ));
    }
    if prompt_tokens.is_empty() {
        return Err(GenerationError::InvalidInput(
            "prompt must encode to at least one token".to_owned(),
        ));
    }
    let required_tokens = prompt_tokens
        .len()
        .checked_add(config.max_new_tokens)
        .ok_or_else(|| {
            GenerationError::InvalidInput(
                "prompt and requested generation length overflow usize".to_owned(),
            )
        })?;
    if required_tokens > config.context_tokens {
        return Err(GenerationError::InvalidInput(format!(
            "{} prompt tokens plus {} requested new tokens exceed the {}-token context",
            prompt_tokens.len(),
            config.max_new_tokens,
            config.context_tokens
        )));
    }

    session.reset().map_err(GenerationError::Backend)?;
    let mut result = session
        .prefill(prompt_tokens)
        .map_err(GenerationError::Backend)?;
    let mut generated = Vec::with_capacity(config.max_new_tokens);
    for index in 0..config.max_new_tokens {
        if is_stop_token(result.token_id) {
            break;
        }
        generated.push(result.token_id);
        if index + 1 < config.max_new_tokens {
            result = session
                .decode(result.token_id)
                .map_err(GenerationError::Backend)?;
        }
    }
    Ok(generated)
}

pub fn benchmark_session<S>(
    session: &mut S,
    prompt_tokens: &[i32],
    config: BenchmarkConfig,
) -> Result<BenchmarkTimings, GenerationError<S::Error>>
where
    S: GenerationSession,
{
    if prompt_tokens.is_empty() {
        return Err(GenerationError::InvalidInput(
            "benchmark prompt must contain at least one token".to_owned(),
        ));
    }
    if config.warmup_iterations == 0 {
        return Err(GenerationError::InvalidInput(
            "benchmark warmup_iterations must be positive".to_owned(),
        ));
    }
    if config.measured_iterations == 0 {
        return Err(GenerationError::InvalidInput(
            "benchmark measured_iterations must be positive".to_owned(),
        ));
    }
    if config.generated_tokens == 0 {
        return Err(GenerationError::InvalidInput(
            "benchmark generated_tokens must be positive".to_owned(),
        ));
    }
    let total_iterations = config
        .warmup_iterations
        .checked_add(config.measured_iterations)
        .ok_or_else(|| {
            GenerationError::InvalidInput("benchmark iteration count overflows usize".to_owned())
        })?;
    let mut timings = BenchmarkTimings {
        prefill_microseconds: Vec::with_capacity(config.measured_iterations),
        generation_microseconds: Vec::with_capacity(config.measured_iterations),
    };

    for iteration in 0..total_iterations {
        session.reset().map_err(GenerationError::Backend)?;

        let prefill_start = Instant::now();
        let mut result = session
            .prefill(prompt_tokens)
            .map_err(GenerationError::Backend)?;
        let prefill_microseconds = elapsed_microseconds(prefill_start);

        let generation_start = Instant::now();
        for _ in 1..config.generated_tokens {
            result = session
                .decode(result.token_id)
                .map_err(GenerationError::Backend)?;
        }
        let generation_microseconds = elapsed_microseconds(generation_start);

        if iteration >= config.warmup_iterations {
            timings.prefill_microseconds.push(prefill_microseconds);
            timings
                .generation_microseconds
                .push(generation_microseconds);
        }
    }
    Ok(timings)
}

fn elapsed_microseconds(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

pub fn generate_chat_text(
    session: &mut Session<'_, '_>,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: GenerationConfig,
) -> Result<String, TextGenerationError> {
    generate_chat(session, tokenizer, prompt, config).map(|generation| generation.text)
}

pub fn generate_chat(
    session: &mut Session<'_, '_>,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: GenerationConfig,
) -> Result<ChatGeneration, TextGenerationError> {
    if prompt.is_empty() {
        return Err(TextGenerationError::Generation(
            GenerationError::InvalidInput("prompt must not be empty".to_owned()),
        ));
    }
    let rendered = tokenizer
        .apply_chat_template(&[ChatMessage::new(ChatRole::User, prompt)], true)
        .map_err(TextGenerationError::Tokenizer)?;
    let prompt_tokens = tokenizer
        .encode(&rendered, false)
        .map_err(TextGenerationError::Tokenizer)?;
    let generated = generate_token_ids(session, &prompt_tokens, config, |token_id| {
        tokenizer.is_stop_token(token_id)
    })
    .map_err(TextGenerationError::Generation)?;
    let text = tokenizer
        .decode(&generated, true)
        .map_err(TextGenerationError::Tokenizer)?;
    Ok(ChatGeneration {
        prompt_token_ids: prompt_tokens,
        generated_token_ids: generated,
        text,
    })
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

    pub fn peak_allocated_gpu_bytes(&self) -> Result<u64, Error> {
        let mut peak_allocated_gpu_bytes = 0;
        let status = unsafe {
            brt_sys::brt_engine_peak_allocated_gpu_bytes(
                self.raw.as_ptr(),
                &mut peak_allocated_gpu_bytes,
            )
        };
        status_to_result(status).map(|()| peak_allocated_gpu_bytes)
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

impl GenerationSession for Session<'_, '_> {
    type Error = Error;

    fn reset(&mut self) -> Result<(), Self::Error> {
        Session::reset(self)
    }

    fn prefill(&mut self, tokens: &[i32]) -> Result<TokenResult, Self::Error> {
        Session::prefill(self, tokens)
    }

    fn decode(&mut self, token_id: i32) -> Result<TokenResult, Self::Error> {
        Session::decode(self, token_id)
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
