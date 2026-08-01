use crate::TokenizerSpec;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    fmt,
};

const MAGIC_PREFIX: &[u8; 7] = b"RIFTOK\0";
const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
/// Conservative cap for a single pre-tokenized UTF-8 piece before BPE.
///
/// The BPE merge loop is intentionally simple and allocation-heavy; this bound
/// keeps malformed or adversarial text from driving quadratic work on one huge
/// piece while remaining far above normal Qwen chat-tokenizer pieces.
const MAX_PRETOKEN_PIECE_BYTES: usize = 4096;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TokenizerError {
    message: String,
}

impl TokenizerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Vision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, i32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), usize>,
    protected_tokens: Vec<(String, i32)>,
    skip_special_ids: HashSet<i32>,
    stop_token_ids: HashSet<i32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    bos_token_id: Option<i32>,
    eos_token_id: Option<i32>,
    add_bos_token: bool,
    add_eos_token: bool,
}

impl Tokenizer {
    pub fn from_spec(spec: &TokenizerSpec) -> Result<Self, TokenizerError> {
        let metadata = parse_spec(spec)?;
        let model = metadata.required_string("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(TokenizerError::new(format!(
                "unsupported tokenizer.ggml.model: {model}"
            )));
        }
        let pre = metadata.required_string("tokenizer.ggml.pre")?;
        if !matches!(pre.as_str(), "qwen2" | "qwen35") {
            return Err(TokenizerError::new(format!(
                "unsupported tokenizer.ggml.pre: {pre}"
            )));
        }
        let chat_template = metadata.required_string("tokenizer.chat_template")?;
        if chat_template.is_empty() {
            return Err(TokenizerError::new(
                "tokenizer.chat_template must not be empty",
            ));
        }
        validate_qwen35_chat_template(chat_template)?;

        let tokens = metadata.required_string_array("tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(TokenizerError::new("tokenizer.ggml.tokens is empty"));
        }
        if tokens.len() > i32::MAX as usize {
            return Err(TokenizerError::new(format!(
                "tokenizer.ggml.tokens length {} exceeds i32 token id range",
                tokens.len()
            )));
        }
        let mut seen_tokens = HashSet::with_capacity(tokens.len());
        for token in tokens {
            if token.is_empty() {
                return Err(TokenizerError::new(
                    "tokenizer.ggml.tokens contains an empty token string",
                ));
            }
            if !seen_tokens.insert(token.as_bytes().to_vec()) {
                return Err(TokenizerError::new(format!(
                    "duplicate token bytes: {token:?}"
                )));
            }
        }

        let token_types = match metadata.optional_i32_array("tokenizer.ggml.token_type")? {
            Some(values) => values.clone(),
            None => vec![TOKEN_TYPE_NORMAL; tokens.len()],
        };
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::new(format!(
                "tokenizer.ggml.token_type length {} does not match tokenizer.ggml.tokens length {}",
                token_types.len(),
                tokens.len()
            )));
        }

        let merges = metadata.required_string_array("tokenizer.ggml.merges")?;
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let mut parts = merge.split(' ');
            let left = parts
                .next()
                .filter(|part| !part.is_empty())
                .ok_or_else(|| TokenizerError::new(format!("malformed merge: {merge:?}")))?;
            let right = parts
                .next()
                .filter(|part| !part.is_empty())
                .ok_or_else(|| TokenizerError::new(format!("malformed merge: {merge:?}")))?;
            if parts.next().is_some() {
                return Err(TokenizerError::new(format!("malformed merge: {merge:?}")));
            }
            if merge_ranks
                .insert((left.to_owned(), right.to_owned()), rank)
                .is_some()
            {
                return Err(TokenizerError::new(format!("duplicate merge: {merge:?}")));
            }
        }

        let mut vocab = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as i32);
        }

        let mut protected_ids = HashSet::new();
        let mut skip_special_ids = HashSet::new();
        let bos_token_id = metadata.optional_u32("tokenizer.ggml.bos_token_id")?;
        let eos_token_id = metadata.optional_u32("tokenizer.ggml.eos_token_id")?;
        let eot_token_id = metadata.optional_u32("tokenizer.ggml.eot_token_id")?;
        for (key, id) in [
            ("tokenizer.ggml.bos_token_id", bos_token_id),
            ("tokenizer.ggml.eos_token_id", eos_token_id),
            ("tokenizer.ggml.eot_token_id", eot_token_id),
            (
                "tokenizer.ggml.padding_token_id",
                metadata.optional_u32("tokenizer.ggml.padding_token_id")?,
            ),
        ] {
            if let Some(id) = id {
                if id as usize >= tokens.len() {
                    return Err(TokenizerError::new(format!(
                        "{key} {id} is out of range for {} tokens",
                        tokens.len()
                    )));
                }
                protected_ids.insert(id as i32);
                skip_special_ids.insert(id as i32);
            }
        }

        for (id, token_type) in token_types.iter().enumerate() {
            if matches!(*token_type, TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED) {
                protected_ids.insert(id as i32);
            }
            if *token_type == TOKEN_TYPE_CONTROL {
                skip_special_ids.insert(id as i32);
            }
        }

        let mut protected_tokens: Vec<(String, i32)> = protected_ids
            .iter()
            .map(|id| {
                let token = tokens[*id as usize].clone();
                (token, *id)
            })
            .collect();
        protected_tokens.sort_by_key(|entry| Reverse(entry.0.len()));

        let (byte_encoder, byte_decoder) = byte_maps();
        let stop_token_ids = [eos_token_id, eot_token_id]
            .into_iter()
            .flatten()
            .map(|id| id as i32)
            .collect();

        Ok(Self {
            vocab,
            id_to_token: tokens.clone(),
            merges: merge_ranks,
            protected_tokens,
            skip_special_ids,
            stop_token_ids,
            byte_encoder,
            byte_decoder,
            bos_token_id: bos_token_id.map(|id| id as i32),
            eos_token_id: eos_token_id.map(|id| id as i32),
            add_bos_token: metadata
                .optional_bool("tokenizer.ggml.add_bos_token")?
                .unwrap_or(false),
            add_eos_token: metadata
                .optional_bool("tokenizer.ggml.add_eos_token")?
                .unwrap_or(false),
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, TokenizerError> {
        let mut tokens = Vec::new();
        if add_special_tokens
            && self.add_bos_token
            && let Some(id) = self.bos_token_id
        {
            tokens.push(id);
        }

        let mut byte_index = 0;
        while byte_index < text.len() {
            if let Some((special, id)) = self.match_special(text, byte_index) {
                tokens.push(id);
                byte_index += special.len();
                continue;
            }
            let next_special = self
                .protected_tokens
                .iter()
                .filter_map(|(special, _)| {
                    text[byte_index..].find(special).map(|pos| byte_index + pos)
                })
                .min()
                .unwrap_or(text.len());
            for piece in pretokenize(&text[byte_index..next_special]) {
                if piece.len() > MAX_PRETOKEN_PIECE_BYTES {
                    return Err(TokenizerError::new(format!(
                        "pretoken piece is {} bytes, exceeding the {MAX_PRETOKEN_PIECE_BYTES}-byte BPE bound",
                        piece.len()
                    )));
                }
                self.encode_piece(piece, &mut tokens)?;
            }
            byte_index = next_special;
        }

        if add_special_tokens
            && self.add_eos_token
            && let Some(id) = self.eos_token_id
        {
            tokens.push(id);
        }
        Ok(tokens)
    }

    pub fn decode(
        &self,
        tokens: &[i32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for id in tokens {
            if *id < 0 || *id as usize >= self.id_to_token.len() {
                return Err(TokenizerError::new(format!(
                    "token id {id} is out of range"
                )));
            }
            if skip_special_tokens && self.skip_special_ids.contains(id) {
                continue;
            }
            let token = &self.id_to_token[*id as usize];
            if self
                .protected_tokens
                .iter()
                .any(|(_, token_id)| token_id == id)
            {
                bytes.extend(token.as_bytes());
                continue;
            }
            for ch in token.chars() {
                let byte = self.byte_decoder.get(&ch).ok_or_else(|| {
                    TokenizerError::new(format!(
                        "token {id} contains non byte-level character {ch:?}"
                    ))
                })?;
                bytes.push(*byte);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        if messages.is_empty() {
            return Err(TokenizerError::new("chat messages must not be empty"));
        }
        let last_user_index = messages
            .iter()
            .rposition(|message| message.role == ChatRole::User);
        let mut rendered = String::new();
        for (index, message) in messages.iter().enumerate() {
            if message.content.is_empty() {
                return Err(TokenizerError::new(
                    "chat message content must not be empty",
                ));
            }
            let role = match message.role {
                ChatRole::System if index == 0 => "system",
                ChatRole::System => {
                    return Err(TokenizerError::new(
                        "system chat message must be the first message",
                    ));
                }
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => {
                    return Err(TokenizerError::new(
                        "tool chat messages are unsupported by the text-only template",
                    ));
                }
                ChatRole::Vision => {
                    return Err(TokenizerError::new(
                        "vision chat messages are unsupported by the text-only template",
                    ));
                }
            };
            rendered.push_str("<|im_start|>");
            rendered.push_str(role);
            rendered.push('\n');
            if message.role == ChatRole::Assistant
                && last_user_index.is_some_and(|last_user| index > last_user)
            {
                rendered.push_str("<think>\n\n</think>\n\n");
            }
            rendered.push_str(&message.content);
            rendered.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            rendered.push_str("<|im_start|>assistant\n<think>");
        }
        Ok(rendered)
    }

    pub fn is_stop_token(&self, token_id: i32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    fn encode_piece(&self, piece: &str, tokens: &mut Vec<i32>) -> Result<(), TokenizerError> {
        let encoded: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|byte| self.byte_encoder[*byte as usize].to_string())
            .collect();
        for token in self.apply_bpe(encoded) {
            let id = self.vocab.get(&token).ok_or_else(|| {
                TokenizerError::new(format!("tokenizer vocabulary is missing token {token:?}"))
            })?;
            tokens.push(*id);
        }
        Ok(())
    }

    fn apply_bpe(&self, mut parts: Vec<String>) -> Vec<String> {
        while parts.len() > 1 {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    self.merges
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (*rank, index))
                })
                .min_by_key(|(rank, _)| *rank);
            let Some((_, index)) = best else {
                break;
            };
            let merged = format!("{}{}", parts[index], parts[index + 1]);
            parts.splice(index..=index + 1, [merged]);
        }
        parts
    }

    fn match_special<'a>(&'a self, text: &'a str, byte_index: usize) -> Option<(&'a str, i32)> {
        self.protected_tokens.iter().find_map(|(special, id)| {
            text[byte_index..]
                .starts_with(special)
                .then_some((special.as_str(), *id))
        })
    }
}

fn validate_qwen35_chat_template(template: &str) -> Result<(), TokenizerError> {
    let normalized: String = template
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(|ch| if ch == '"' { '\'' } else { ch })
        .collect();
    for fragment in [
        "macrorender_content(content",
        "last_query_index=messages|length-1",
        "formessageinmessages[::-1]",
        "formessageinmessages",
        "render_content(message.content",
        "message.role=='user'",
        "message.role=='assistant'",
        "'</think>'incontent",
        "ifadd_generation_prompt",
        "enable_thinkingisdefinedandenable_thinkingisfalse",
        "<|im_start|>",
        "<|im_end|>",
        "<|im_start|>assistant",
        "<think>",
        "</think>",
    ] {
        if !normalized.contains(fragment) {
            return Err(TokenizerError::new(format!(
                "tokenizer.chat_template does not match the supported Qwen3.5 no-tools signature: missing {fragment:?}"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Metadata {
    entries: HashMap<String, Value>,
}

impl Metadata {
    fn required_string(&self, key: &str) -> Result<&String, TokenizerError> {
        match self.entries.get(key) {
            Some(Value::String(value)) => Ok(value),
            Some(_) => Err(TokenizerError::new(format!("{key} must be a string"))),
            None => Err(TokenizerError::new(format!("missing {key}"))),
        }
    }

    fn required_string_array(&self, key: &str) -> Result<&Vec<String>, TokenizerError> {
        match self.entries.get(key) {
            Some(Value::StringArray(value)) => Ok(value),
            Some(_) => Err(TokenizerError::new(format!("{key} must be a string array"))),
            None => Err(TokenizerError::new(format!("missing {key}"))),
        }
    }

    fn optional_i32_array(&self, key: &str) -> Result<Option<&Vec<i32>>, TokenizerError> {
        match self.entries.get(key) {
            Some(Value::I32Array(value)) => Ok(Some(value)),
            Some(_) => Err(TokenizerError::new(format!("{key} must be an i32 array"))),
            None => Ok(None),
        }
    }

    fn optional_u32(&self, key: &str) -> Result<Option<u32>, TokenizerError> {
        match self.entries.get(key) {
            Some(Value::U32(value)) => Ok(Some(*value)),
            Some(_) => Err(TokenizerError::new(format!("{key} must be a u32 value"))),
            None => Ok(None),
        }
    }

    fn optional_bool(&self, key: &str) -> Result<Option<bool>, TokenizerError> {
        match self.entries.get(key) {
            Some(Value::Bool(value)) => Ok(Some(*value)),
            Some(_) => Err(TokenizerError::new(format!("{key} must be a bool value"))),
            None => Ok(None),
        }
    }
}

#[derive(Clone, Debug)]
enum Value {
    U32(u32),
    Bool(bool),
    String(String),
    StringArray(Vec<String>),
    I32Array(Vec<i32>),
    Ignored,
}

fn parse_spec(spec: &TokenizerSpec) -> Result<Metadata, TokenizerError> {
    if spec.version != 1 {
        return Err(TokenizerError::new(format!(
            "unsupported tokenizer spec version {}",
            spec.version
        )));
    }
    let mut reader = Reader::new(&spec.bytes);
    let magic = reader.read_bytes(MAGIC_PREFIX.len(), "magic")?;
    if magic != MAGIC_PREFIX {
        return Err(TokenizerError::new("invalid tokenizer spec magic"));
    }
    let version = reader.read_bytes(1, "version")?[0];
    if version != 1 {
        return Err(TokenizerError::new(format!(
            "unsupported tokenizer ABI version {version}"
        )));
    }
    let entry_count = reader.read_u32("entry count")? as usize;
    let min_entry_size = 8;
    if entry_count > reader.remaining_len() / min_entry_size {
        return Err(TokenizerError::new(format!(
            "entry count {entry_count} exceeds remaining tokenizer spec bytes"
        )));
    }
    let mut entries = HashMap::with_capacity(entry_count);
    let mut previous_key: Option<String> = None;
    for _ in 0..entry_count {
        let key = reader.read_string("metadata key")?;
        if let Some(previous) = &previous_key {
            if key == *previous {
                return Err(TokenizerError::new(format!("duplicate key: {key}")));
            }
            if key < *previous {
                return Err(TokenizerError::new(format!(
                    "tokenizer metadata keys must be sorted: {key} after {previous}"
                )));
            }
        }
        previous_key = Some(key.clone());
        let value_type = reader.read_u32(&format!("metadata {key} type"))?;
        let value = reader.read_value(value_type, &key)?;
        if entries.insert(key.clone(), value).is_some() {
            return Err(TokenizerError::new(format!("duplicate key: {key}")));
        }
    }
    if !reader.is_empty() {
        return Err(TokenizerError::new("tokenizer spec has trailing bytes"));
    }
    Ok(Metadata { entries })
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining_len(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_bytes(&mut self, count: usize, context: &str) -> Result<&'a [u8], TokenizerError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| TokenizerError::new(format!("{context} offset overflow")))?;
        let Some(bytes) = self.bytes.get(self.offset..end) else {
            return Err(TokenizerError::new(format!("truncated {context}")));
        };
        self.offset = end;
        Ok(bytes)
    }

    fn read_u32(&mut self, context: &str) -> Result<u32, TokenizerError> {
        let bytes: [u8; 4] = self
            .read_bytes(4, context)?
            .try_into()
            .expect("slice length is checked");
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self, context: &str) -> Result<i32, TokenizerError> {
        let bytes: [u8; 4] = self
            .read_bytes(4, context)?
            .try_into()
            .expect("slice length is checked");
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self, context: &str) -> Result<u64, TokenizerError> {
        let bytes: [u8; 8] = self
            .read_bytes(8, context)?
            .try_into()
            .expect("slice length is checked");
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_string(&mut self, context: &str) -> Result<String, TokenizerError> {
        let len = self.read_u32(&format!("{context} length"))? as usize;
        let bytes = self.read_bytes(len, context)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| TokenizerError::new(format!("{context} is not valid UTF-8")))
    }

    fn read_value(&mut self, value_type: u32, key: &str) -> Result<Value, TokenizerError> {
        match value_type {
            TYPE_U8 | TYPE_I8 => {
                let _ = self.read_bytes(1, key)?;
                Ok(Value::Ignored)
            }
            TYPE_U16 | TYPE_I16 => {
                let _ = self.read_bytes(2, key)?;
                Ok(Value::Ignored)
            }
            TYPE_U32 => Ok(Value::U32(self.read_u32(key)?)),
            TYPE_I32 => {
                let _ = self.read_i32(key)?;
                Ok(Value::Ignored)
            }
            TYPE_F32 => {
                let _ = self.read_bytes(4, key)?;
                Ok(Value::Ignored)
            }
            TYPE_BOOL => {
                let raw = self.read_bytes(1, key)?[0];
                match raw {
                    0 => Ok(Value::Bool(false)),
                    1 => Ok(Value::Bool(true)),
                    _ => Err(TokenizerError::new(format!("{key} has invalid bool value"))),
                }
            }
            TYPE_STRING => Ok(Value::String(self.read_string(key)?)),
            TYPE_ARRAY => self.read_array(key),
            TYPE_U64 | TYPE_I64 | TYPE_F64 => {
                let _ = self.read_bytes(8, key)?;
                Ok(Value::Ignored)
            }
            other => Err(TokenizerError::new(format!(
                "{key} has unsupported metadata type {other}"
            ))),
        }
    }

    fn read_array(&mut self, key: &str) -> Result<Value, TokenizerError> {
        let element_type = self.read_u32(&format!("{key} array element type"))?;
        if element_type == TYPE_ARRAY {
            return Err(TokenizerError::new(format!(
                "{key} nested arrays are unsupported"
            )));
        }
        let len = self.read_u64(&format!("{key} array length"))?;
        let len: usize = len
            .try_into()
            .map_err(|_| TokenizerError::new(format!("{key} array is too large")))?;
        if key == "tokenizer.ggml.tokens" && len > i32::MAX as usize {
            return Err(TokenizerError::new(format!(
                "{key} length {len} exceeds i32 token id range"
            )));
        }
        self.ensure_array_fits(key, element_type, len)?;
        match element_type {
            TYPE_STRING => {
                if is_used_string_array(key) {
                    let mut values = Vec::with_capacity(len);
                    for index in 0..len {
                        values.push(self.read_string(&format!("{key}[{index}]"))?);
                    }
                    Ok(Value::StringArray(values))
                } else {
                    for index in 0..len {
                        let _ = self.read_string(&format!("{key}[{index}]"))?;
                    }
                    Ok(Value::Ignored)
                }
            }
            TYPE_I32 => {
                if key == "tokenizer.ggml.token_type" {
                    let mut values = Vec::with_capacity(len);
                    for index in 0..len {
                        values.push(self.read_i32(&format!("{key}[{index}]"))?);
                    }
                    Ok(Value::I32Array(values))
                } else {
                    for index in 0..len {
                        let _ = self.read_i32(&format!("{key}[{index}]"))?;
                    }
                    Ok(Value::Ignored)
                }
            }
            TYPE_U8 | TYPE_I8 => {
                let _ = self.read_bytes(len, key)?;
                Ok(Value::Ignored)
            }
            TYPE_U16 | TYPE_I16 => {
                let _ = self.read_bytes(len * 2, key)?;
                Ok(Value::Ignored)
            }
            TYPE_U32 | TYPE_F32 => {
                let _ = self.read_bytes(len * 4, key)?;
                Ok(Value::Ignored)
            }
            TYPE_BOOL => {
                for index in 0..len {
                    let raw = self.read_bytes(1, &format!("{key}[{index}]"))?[0];
                    if raw > 1 {
                        return Err(TokenizerError::new(format!(
                            "{key}[{index}] has invalid bool value"
                        )));
                    }
                }
                Ok(Value::Ignored)
            }
            TYPE_U64 | TYPE_I64 | TYPE_F64 => {
                let _ = self.read_bytes(len * 8, key)?;
                Ok(Value::Ignored)
            }
            other => Err(TokenizerError::new(format!(
                "{key} has unsupported array element type {other}"
            ))),
        }
    }

    fn ensure_array_fits(
        &self,
        key: &str,
        element_type: u32,
        len: usize,
    ) -> Result<(), TokenizerError> {
        let min_element_size = match element_type {
            TYPE_U8 | TYPE_I8 | TYPE_BOOL => 1,
            TYPE_U16 | TYPE_I16 => 2,
            TYPE_U32 | TYPE_I32 | TYPE_F32 | TYPE_STRING => 4,
            TYPE_U64 | TYPE_I64 | TYPE_F64 => 8,
            TYPE_ARRAY => unreachable!("nested arrays rejected before bounds check"),
            other => {
                return Err(TokenizerError::new(format!(
                    "{key} has unsupported array element type {other}"
                )));
            }
        };
        if len > self.bytes.len().saturating_sub(self.offset) / min_element_size {
            return Err(TokenizerError::new(format!(
                "{key} array length exceeds remaining tokenizer spec bytes"
            )));
        }
        Ok(())
    }
}

fn is_used_string_array(key: &str) -> bool {
    matches!(key, "tokenizer.ggml.tokens" | "tokenizer.ggml.merges")
}

fn pretokenize(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut index = 0;
    while index < text.len() {
        let rest = &text[index..];
        if let Some(len) = contraction_len(rest) {
            pieces.push(&text[index..index + len]);
            index += len;
            continue;
        }

        let first = rest.chars().next().expect("index is in bounds");
        let next = rest[first.len_utf8()..]
            .chars()
            .next()
            .map(|ch| (first.len_utf8(), ch));
        if first != '\r'
            && first != '\n'
            && !is_letter_or_mark(first)
            && !first.is_numeric()
            && let Some((next_offset, next)) = next
            && is_letter_or_mark(next)
        {
            let end = index + next_offset + letter_run_len(&rest[next_offset..]);
            pieces.push(&text[index..end]);
            index = end;
            continue;
        }

        if first.is_numeric() {
            let end = index + first.len_utf8();
            pieces.push(&text[index..end]);
            index = end;
            continue;
        }

        if first == ' '
            && let Some((next_offset, next)) = next
            && is_punctuation(next)
        {
            let end = index + next_offset + punctuation_run_len(&rest[next_offset..]);
            pieces.push(&text[index..end]);
            index = end;
            continue;
        }

        if is_letter_or_mark(first) {
            let end = index + letter_run_len(rest);
            pieces.push(&text[index..end]);
            index = end;
        } else if first.is_whitespace() {
            let end = index + whitespace_piece_len(rest);
            pieces.push(&text[index..end]);
            index = end;
        } else {
            let end = index + punctuation_run_len(rest);
            pieces.push(&text[index..end]);
            index = end;
        }
    }
    pieces
}

fn contraction_len(text: &str) -> Option<usize> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let lower = text
        .chars()
        .take(3)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    CONTRACTIONS
        .iter()
        .find(|contraction| lower.starts_with(**contraction))
        .map(|contraction| contraction.len())
}

fn letter_run_len(text: &str) -> usize {
    run_len(text, is_letter_or_mark)
}

fn punctuation_run_len(text: &str) -> usize {
    let mut end = run_len(text, is_punctuation);
    end += newline_run_len(&text[end..]);
    end
}

fn whitespace_run_len(text: &str) -> usize {
    run_len(text, |ch| ch.is_whitespace() && ch != '\r' && ch != '\n')
}

fn whitespace_piece_len(text: &str) -> usize {
    let leading_non_newline = whitespace_run_len(text);
    let newline_len = newline_run_len(&text[leading_non_newline..]);
    if newline_len > 0 {
        leading_non_newline + newline_len
    } else if let Some(next) = text[leading_non_newline..].chars().next()
        && let Some(last_whitespace) = last_char_span(&text[..leading_non_newline])
        && last_whitespace.start > 0
        && whitespace_can_prefix(last_whitespace.ch, next)
    {
        last_whitespace.start
    } else {
        leading_non_newline
    }
}

#[derive(Clone, Copy)]
struct CharSpan {
    ch: char,
    start: usize,
}

fn last_char_span(text: &str) -> Option<CharSpan> {
    text.char_indices()
        .last()
        .map(|(start, ch)| CharSpan { ch, start })
}

fn whitespace_can_prefix(whitespace: char, next: char) -> bool {
    is_letter_or_mark(next) || (whitespace == ' ' && is_punctuation(next))
}

fn newline_run_len(text: &str) -> usize {
    run_len(text, |ch| ch == '\r' || ch == '\n')
}

fn run_len(text: &str, predicate: impl Fn(char) -> bool) -> usize {
    let mut end = 0;
    for (offset, ch) in text.char_indices() {
        if !predicate(ch) {
            break;
        }
        end = offset + ch.len_utf8();
    }
    end
}

fn is_letter_or_mark(ch: char) -> bool {
    ch.is_alphabetic() || is_mark(ch)
}

fn is_mark(ch: char) -> bool {
    matches!(
        ch,
        '\u{0300}'..='\u{036f}'
            | '\u{0483}'..='\u{0489}'
            | '\u{0591}'..='\u{05bd}'
            | '\u{05bf}'
            | '\u{05c1}'..='\u{05c2}'
            | '\u{05c4}'..='\u{05c5}'
            | '\u{05c7}'
            | '\u{0610}'..='\u{061a}'
            | '\u{064b}'..='\u{065f}'
            | '\u{0670}'
            | '\u{06d6}'..='\u{06dc}'
            | '\u{06df}'..='\u{06e4}'
            | '\u{06e7}'..='\u{06e8}'
            | '\u{06ea}'..='\u{06ed}'
            | '\u{1ab0}'..='\u{1aff}'
            | '\u{1dc0}'..='\u{1dff}'
            | '\u{20d0}'..='\u{20ff}'
            | '\u{fe00}'..='\u{fe0f}'
            | '\u{fe20}'..='\u{fe2f}'
            | '\u{e0100}'..='\u{e01ef}'
    )
}

fn is_punctuation(ch: char) -> bool {
    !ch.is_whitespace() && !is_letter_or_mark(ch) && !ch.is_numeric()
}

fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs: Vec<u32> = bs.iter().map(|byte| *byte as u32).collect();
    let mut next = 0_u32;
    for byte in 0_u8..=255 {
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + next);
            next += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    for (byte, codepoint) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(codepoint).expect("GPT-2 byte map codepoint");
        encoder[byte as usize] = ch;
        decoder.insert(ch, byte);
    }
    (encoder, decoder)
}
