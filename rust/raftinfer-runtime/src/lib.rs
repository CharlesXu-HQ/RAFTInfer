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
        write!(f, "RAFTInfer error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct Engine {
    raw: NonNull<raftinfer_sys::RaftInferEngineHandle>,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[derive(Debug)]
pub struct Model<'engine> {
    raw: NonNull<raftinfer_sys::RaftInferModelHandle>,
    _engine: PhantomData<&'engine Engine>,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[derive(Debug)]
pub struct Session<'model, 'engine> {
    raw: NonNull<raftinfer_sys::RaftInferSessionHandle>,
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
    pub qwen35_policy: Qwen35ExecutionPolicy,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 4096,
            qwen35_policy: Qwen35ExecutionPolicy::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35AttentionImplementation {
    MaterializedReference,
    OnlineTiled,
}

impl Qwen35AttentionImplementation {
    fn to_native(self) -> u32 {
        match self {
            Self::MaterializedReference => {
                raftinfer_sys::RAFTINFER_QWEN35_ATTENTION_MATERIALIZED_REFERENCE
            }
            Self::OnlineTiled => raftinfer_sys::RAFTINFER_QWEN35_ATTENTION_ONLINE_TILED,
        }
    }

    fn from_native(value: u32) -> Result<Self, Error> {
        match value {
            raftinfer_sys::RAFTINFER_QWEN35_ATTENTION_MATERIALIZED_REFERENCE => {
                Ok(Self::MaterializedReference)
            }
            raftinfer_sys::RAFTINFER_QWEN35_ATTENTION_ONLINE_TILED => Ok(Self::OnlineTiled),
            _ => Err(Error {
                code: raftinfer_sys::RAFTINFER_STATUS_UNSUPPORTED,
                message: "native diagnostics returned an unknown attention implementation"
                    .to_owned(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheDType {
    F32,
    Bf16,
}

impl KvCacheDType {
    fn to_native(self) -> u32 {
        match self {
            Self::F32 => raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_F32,
            Self::Bf16 => raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_BF16,
        }
    }

    fn from_native(value: u32) -> Result<Self, Error> {
        match value {
            raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_F32 => Ok(Self::F32),
            raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_BF16 => Ok(Self::Bf16),
            _ => Err(Error {
                code: raftinfer_sys::RAFTINFER_STATUS_UNSUPPORTED,
                message: "native diagnostics returned an unknown KV cache dtype".to_owned(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheLayout {
    TokenMajor,
    HeadMajor,
}

impl KvCacheLayout {
    fn to_native(self) -> u32 {
        match self {
            Self::TokenMajor => raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR,
            Self::HeadMajor => raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR,
        }
    }

    fn from_native(value: u32) -> Result<Self, Error> {
        match value {
            raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR => Ok(Self::TokenMajor),
            raftinfer_sys::RAFTINFER_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR => Ok(Self::HeadMajor),
            _ => Err(Error {
                code: raftinfer_sys::RAFTINFER_STATUS_UNSUPPORTED,
                message: "native diagnostics returned an unknown KV cache layout".to_owned(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35ExecutionPolicy {
    pub attention: Qwen35AttentionImplementation,
    pub kv_cache_dtype: KvCacheDType,
    pub kv_cache_layout: KvCacheLayout,
    pub decode_graph: bool,
    pub grouped_input_casts: bool,
}

impl Default for Qwen35ExecutionPolicy {
    fn default() -> Self {
        Self {
            attention: Qwen35AttentionImplementation::OnlineTiled,
            kv_cache_dtype: KvCacheDType::F32,
            kv_cache_layout: KvCacheLayout::TokenMajor,
            decode_graph: true,
            grouped_input_casts: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionDiagnostics {
    pub attention: Qwen35AttentionImplementation,
    pub kv_cache_dtype: KvCacheDType,
    pub kv_cache_layout: KvCacheLayout,
    pub decode_graph_enabled: bool,
    pub decode_graph_captured: bool,
    pub decode_graph_replayed: bool,
    pub attention_workspace_bytes: usize,
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
    /// Number of timed decode calls after each prompt prefill.
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
    fn decode_greedy(
        &mut self,
        first_token_id: i32,
        output_tokens: &mut [i32],
    ) -> Result<TokenResult, Self::Error> {
        let mut result = TokenResult {
            token_id: first_token_id,
            position: 0,
        };
        for token in output_tokens {
            result = self.decode(result.token_id)?;
            *token = result.token_id;
        }
        Ok(result)
    }
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

    let mut decode_tokens = vec![0; config.generated_tokens];
    for iteration in 0..total_iterations {
        session.reset().map_err(GenerationError::Backend)?;

        let prefill_start = Instant::now();
        let result = session
            .prefill(prompt_tokens)
            .map_err(GenerationError::Backend)?;
        let prefill_microseconds = elapsed_microseconds(prefill_start);

        let generation_start = Instant::now();
        session
            .decode_greedy(result.token_id, &mut decode_tokens)
            .map_err(GenerationError::Backend)?;
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
        let native = raftinfer_sys::RaftInferEngineConfig {
            struct_size: std::mem::size_of::<raftinfer_sys::RaftInferEngineConfig>(),
            device_id: config.device_id,
            initial_pool_bytes: config.initial_pool_bytes,
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe { raftinfer_sys::raftinfer_engine_create(&native, &mut raw) };
        status_to_result(status)?;

        Ok(Self {
            raw: NonNull::new(raw).expect("successful create returned null"),
            _not_send_sync: PhantomData,
        })
    }

    pub fn cuda_enabled(&self) -> bool {
        unsafe { raftinfer_sys::raftinfer_engine_is_cuda_enabled(self.raw.as_ptr()) != 0 }
    }

    pub fn run_smoke(&self) -> Result<SmokeResult, Error> {
        let mut native = raftinfer_sys::RaftInferSmokeResult::default();
        let status =
            unsafe { raftinfer_sys::raftinfer_engine_run_smoke(self.raw.as_ptr(), &mut native) };
        status_to_result(status).map(|()| SmokeResult {
            device_id: native.device_id,
            element_count: native.element_count,
            checksum: native.checksum,
        })
    }

    pub fn peak_allocated_gpu_bytes(&self) -> Result<u64, Error> {
        let mut peak_allocated_gpu_bytes = 0;
        let status = unsafe {
            raftinfer_sys::raftinfer_engine_peak_allocated_gpu_bytes(
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
            code: raftinfer_sys::RAFTINFER_STATUS_UNSUPPORTED,
            message: "GGUF path contains an interior NUL byte".to_owned(),
        })?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            raftinfer_sys::raftinfer_engine_load_model(
                self.raw.as_ptr(),
                native_path.as_ptr(),
                &mut raw,
            )
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
        unsafe { raftinfer_sys::raftinfer_engine_destroy(self.raw.as_ptr()) }
    }
}

impl<'engine> Model<'engine> {
    pub fn tokenizer_spec(&self) -> Result<TokenizerSpec, Error> {
        let mut native = raftinfer_sys::RaftInferOwnedBuffer {
            struct_size: std::mem::size_of::<raftinfer_sys::RaftInferOwnedBuffer>(),
            version: 0,
            data: std::ptr::null_mut(),
            size: 0,
        };
        let status = unsafe {
            raftinfer_sys::raftinfer_model_copy_tokenizer_spec(self.raw.as_ptr(), &mut native)
        };
        status_to_result(status)?;
        let bytes = if native.size == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(native.data, native.size) }.to_vec()
        };
        let version = native.version;
        unsafe { raftinfer_sys::raftinfer_owned_buffer_free(&mut native) };
        Ok(TokenizerSpec { version, bytes })
    }

    pub fn create_session<'model>(
        &'model self,
        config: SessionConfig,
    ) -> Result<Session<'model, 'engine>, Error> {
        let native_policy = raftinfer_sys::RaftInferQwen35ExecutionPolicy {
            struct_size: std::mem::size_of::<raftinfer_sys::RaftInferQwen35ExecutionPolicy>(),
            attention: config.qwen35_policy.attention.to_native(),
            kv_cache_dtype: config.qwen35_policy.kv_cache_dtype.to_native(),
            kv_cache_layout: config.qwen35_policy.kv_cache_layout.to_native(),
            decode_graph: bool_to_native(config.qwen35_policy.decode_graph),
            grouped_input_casts: bool_to_native(config.qwen35_policy.grouped_input_casts),
        };
        let native = raftinfer_sys::RaftInferSessionConfig {
            struct_size: std::mem::size_of::<raftinfer_sys::RaftInferSessionConfig>(),
            max_context_tokens: config.max_context_tokens,
            qwen35_policy: &native_policy,
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            raftinfer_sys::raftinfer_session_create(self.raw.as_ptr(), &native, &mut raw)
        };
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
        unsafe { raftinfer_sys::raftinfer_model_destroy(self.raw.as_ptr()) }
    }
}

impl Session<'_, '_> {
    pub fn diagnostics(&self) -> Result<ExecutionDiagnostics, Error> {
        let mut native = raftinfer_sys::RaftInferSessionDiagnostics {
            struct_size: std::mem::size_of::<raftinfer_sys::RaftInferSessionDiagnostics>(),
            ..raftinfer_sys::RaftInferSessionDiagnostics::default()
        };
        let status =
            unsafe { raftinfer_sys::raftinfer_session_diagnostics(self.raw.as_ptr(), &mut native) };
        status_to_result(status)?;
        Ok(ExecutionDiagnostics {
            attention: Qwen35AttentionImplementation::from_native(native.attention)?,
            kv_cache_dtype: KvCacheDType::from_native(native.kv_cache_dtype)?,
            kv_cache_layout: KvCacheLayout::from_native(native.kv_cache_layout)?,
            decode_graph_enabled: native.decode_graph_enabled != 0,
            decode_graph_captured: native.decode_graph_captured != 0,
            decode_graph_replayed: native.decode_graph_replayed != 0,
            attention_workspace_bytes: native.attention_workspace_bytes,
        })
    }

    pub fn prefill(&mut self, tokens: &[i32]) -> Result<TokenResult, Error> {
        let mut native = raftinfer_sys::RaftInferTokenResult::default();
        let status = unsafe {
            raftinfer_sys::raftinfer_session_prefill(
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
        let mut native = raftinfer_sys::RaftInferTokenResult::default();
        let status = unsafe {
            raftinfer_sys::raftinfer_session_decode(self.raw.as_ptr(), token_id, &mut native)
        };
        status_to_result(status).map(|()| TokenResult {
            token_id: native.token_id,
            position: native.position,
        })
    }

    pub fn decode_greedy(
        &mut self,
        first_token_id: i32,
        output_tokens: &mut [i32],
    ) -> Result<TokenResult, Error> {
        let mut native = raftinfer_sys::RaftInferTokenResult::default();
        let status = unsafe {
            raftinfer_sys::raftinfer_session_decode_greedy(
                self.raw.as_ptr(),
                first_token_id,
                output_tokens.as_mut_ptr(),
                output_tokens.len(),
                &mut native,
            )
        };
        status_to_result(status).map(|()| TokenResult {
            token_id: native.token_id,
            position: native.position,
        })
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let status = unsafe { raftinfer_sys::raftinfer_session_reset(self.raw.as_ptr()) };
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

    fn decode_greedy(
        &mut self,
        first_token_id: i32,
        output_tokens: &mut [i32],
    ) -> Result<TokenResult, Self::Error> {
        Session::decode_greedy(self, first_token_id, output_tokens)
    }
}

impl Drop for Session<'_, '_> {
    fn drop(&mut self) {
        unsafe { raftinfer_sys::raftinfer_session_destroy(self.raw.as_ptr()) }
    }
}

fn bool_to_native(value: bool) -> i32 {
    if value { 1 } else { 0 }
}

fn status_to_result(status: raftinfer_sys::RaftInferStatus) -> Result<(), Error> {
    if status.code == 0 {
        return Ok(());
    }

    Err(Error {
        code: status.code,
        message: native_error_message(status),
    })
}

fn native_error_message(status: raftinfer_sys::RaftInferStatus) -> String {
    let message = if status.message.is_null() {
        unsafe { raftinfer_sys::raftinfer_last_error_message() }
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
