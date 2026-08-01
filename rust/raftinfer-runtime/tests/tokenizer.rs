use raftinfer_runtime::{
    Engine, EngineConfig, TokenizerSpec,
    tokenizer::{ChatMessage, ChatRole, Tokenizer},
};

#[test]
fn encodes_bpe_and_decodes_after_concatenating_bytes() {
    let tokenizer = synthetic_tokenizer();

    assert_eq!(tokenizer.encode("abc!", false).expect("encode"), [5, 8]);
    assert_eq!(tokenizer.encode(" a", false).expect("encode"), [7]);
    assert_eq!(
        tokenizer
            .decode(&[5, 8], false)
            .expect("decode merged token bytes"),
        "abc!"
    );
}

#[test]
fn rejects_pretoken_pieces_above_the_bpe_bound() {
    let tokenizer = synthetic_tokenizer();
    let boundary = "a".repeat(4096);
    let too_long = "a".repeat(4097);

    assert_eq!(
        tokenizer
            .encode(&boundary, false)
            .expect("boundary piece encodes")
            .len(),
        4096
    );
    let error = tokenizer
        .encode(&too_long, false)
        .expect_err("oversized pretoken piece must fail");
    assert!(error.to_string().contains("pretoken piece"));
}

#[test]
fn accepts_official_qwen35_pretokenizer_alias() {
    let mut entries = valid_entries();
    replace_entry(&mut entries, entry_string("tokenizer.ggml.pre", "qwen35"));

    Tokenizer::from_spec(&spec(entries)).expect("official qwen35 pre-tokenizer alias");
}

#[test]
fn accepts_official_jinja_escaped_newline_template_signature() {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_string("tokenizer.chat_template", OFFICIAL_ESCAPED_CHAT_TEMPLATE),
    );

    Tokenizer::from_spec(&spec(entries))
        .expect("official Jinja template uses escaped newlines inside string literals");
}

#[test]
fn rejects_fragment_only_chat_template_signature() {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_string("tokenizer.chat_template", FRAGMENT_ONLY_CHAT_TEMPLATE),
    );

    let error = Tokenizer::from_spec(&spec(entries))
        .expect_err("output fragments alone do not prove compatible rendering semantics");
    assert!(error.to_string().contains("supported Qwen3.5"));
}

#[test]
fn encodes_unicode_numeric_codepoints_one_at_a_time() {
    let tokenizer = synthetic_tokenizer();

    assert_eq!(tokenizer.encode("١１", false).expect("encode"), [25, 30]);
}

#[test]
fn groups_qwen2_whitespace_alternatives() {
    let tokenizer = synthetic_tokenizer();

    assert_eq!(tokenizer.encode("a  \n", false).expect("encode"), [1, 33]);
    assert_eq!(tokenizer.encode("a  ", false).expect("encode"), [1, 32]);
    assert_eq!(tokenizer.encode("a  b", false).expect("encode"), [1, 6, 42]);
    assert_eq!(tokenizer.encode("a \t\n", false).expect("encode"), [1, 36]);
    assert_eq!(tokenizer.encode("a !", false).expect("encode"), [1, 43]);
}

#[test]
fn keeps_non_basic_combining_marks_in_letter_piece() {
    let tokenizer = synthetic_tokenizer();

    assert_eq!(tokenizer.encode("a\u{fe0f}", false).expect("encode"), [41]);
}

#[test]
fn protects_registered_special_tokens_during_encoding() {
    let tokenizer = synthetic_tokenizer();

    let tokens = tokenizer
        .encode("<|im_start|>user\nHi<|im_end|>\n", false)
        .expect("encode with specials");

    assert_eq!(tokens, [9, 18, 19, 20, 21, 14, 16, 17, 10, 14]);
    assert_eq!(
        tokenizer
            .decode(&tokens, false)
            .expect("decode with specials"),
        "<|im_start|>user\nHi<|im_end|>\n"
    );
}

#[test]
fn skip_special_tokens_does_not_drop_user_defined_added_tokens() {
    let tokenizer = synthetic_tokenizer();
    let tokens = tokenizer
        .encode("<custom><|im_start|>a<|im_end|>", false)
        .expect("encode protected user-defined and control tokens");

    assert_eq!(tokens, [22, 9, 1, 10]);
    assert_eq!(
        tokenizer
            .decode(&tokens, true)
            .expect("decode skips only control specials"),
        "<custom>a"
    );
}

#[test]
fn applies_qwen35_no_tools_text_chat_template() {
    let tokenizer = synthetic_tokenizer();
    let rendered = tokenizer
        .apply_chat_template(
            &[
                ChatMessage::new(ChatRole::System, "You are concise."),
                ChatMessage::new(ChatRole::User, "你好, GPU!"),
            ],
            true,
        )
        .expect("render chat");

    assert_eq!(
        rendered,
        "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\n你好, GPU!<|im_end|>\n<|im_start|>assistant\n<think>"
    );
}

#[test]
fn assistant_history_after_last_user_gets_empty_thinking_prefix() {
    let tokenizer = synthetic_tokenizer();
    let rendered = tokenizer
        .apply_chat_template(
            &[
                ChatMessage::new(ChatRole::System, "Rules"),
                ChatMessage::new(ChatRole::User, "First"),
                ChatMessage::new(ChatRole::Assistant, "Earlier answer"),
                ChatMessage::new(ChatRole::User, "Second"),
                ChatMessage::new(ChatRole::Assistant, "Final answer"),
            ],
            false,
        )
        .expect("render assistant history");

    assert_eq!(
        rendered,
        "<|im_start|>system\nRules<|im_end|>\n<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\nEarlier answer<|im_end|>\n<|im_start|>user\nSecond<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nFinal answer<|im_end|>\n"
    );
}

#[test]
fn rejects_chat_inputs_that_would_need_a_different_template() {
    let tokenizer = synthetic_tokenizer();

    for messages in [
        vec![ChatMessage::new(ChatRole::User, "")],
        vec![
            ChatMessage::new(ChatRole::User, "hi"),
            ChatMessage::new(ChatRole::System, "late"),
        ],
        vec![ChatMessage::new(ChatRole::Tool, "call")],
        vec![ChatMessage::new(ChatRole::Vision, "image")],
    ] {
        assert!(
            tokenizer.apply_chat_template(&messages, true).is_err(),
            "messages should be rejected: {messages:?}"
        );
    }
}

#[test]
fn rejects_corrupt_tokenizer_specs_with_actionable_errors() {
    let cases = [
        ("bad magic", corrupt_magic(), "magic"),
        ("bad version", corrupt_version(), "version"),
        ("trailing bytes", corrupt_trailing_bytes(), "trailing"),
        ("duplicate key", corrupt_duplicate_key(), "duplicate key"),
        ("unsorted key", corrupt_unsorted_key(), "sorted"),
        (
            "strict prefix truncation",
            corrupt_strict_prefixes(),
            "truncated",
        ),
        (
            "huge entry count",
            corrupt_huge_entry_count(),
            "entry count",
        ),
        (
            "huge token array length",
            corrupt_huge_token_array_length(),
            "tokenizer.ggml.tokens",
        ),
        (
            "missing tokens",
            corrupt_missing_tokens(),
            "missing tokenizer.ggml.tokens",
        ),
        (
            "empty control token",
            corrupt_empty_control_token(),
            "empty token",
        ),
        (
            "missing chat template",
            corrupt_missing_chat_template(),
            "missing tokenizer.chat_template",
        ),
        (
            "empty chat template",
            corrupt_empty_chat_template(),
            "tokenizer.chat_template",
        ),
        (
            "different chat template",
            corrupt_different_chat_template(),
            "tokenizer.chat_template",
        ),
        (
            "wrong model type",
            corrupt_wrong_model_type(),
            "tokenizer.ggml.model",
        ),
        (
            "wrong tokens type",
            corrupt_wrong_tokens_type(),
            "tokenizer.ggml.tokens",
        ),
        (
            "wrong merges array type",
            corrupt_wrong_merges_array_type(),
            "tokenizer.ggml.merges",
        ),
        (
            "duplicate token bytes",
            corrupt_duplicate_tokens(),
            "duplicate token",
        ),
        (
            "short token types",
            corrupt_short_token_types(),
            "token_type",
        ),
        ("malformed merge", corrupt_malformed_merge(), "merge"),
        (
            "special id out of range",
            corrupt_special_id(),
            "eos_token_id",
        ),
    ];

    for (name, spec, needle) in cases {
        let error = Tokenizer::from_spec(&spec).expect_err(name);
        assert!(
            error.to_string().contains(needle),
            "{name}: {error} did not contain {needle:?}"
        );
    }
}

#[test]
fn consumes_irrelevant_valid_gguf_metadata_types_without_rejecting() {
    let mut entries = valid_entries();
    entries.extend([
        entry_u8("tokenizer.ggml.extra_uint8", 7),
        entry_i8("tokenizer.ggml.extra_int8", -7),
        entry_u16("tokenizer.ggml.extra_uint16", 700),
        entry_i16("tokenizer.ggml.extra_int16", -700),
        entry_i32("tokenizer.ggml.extra_int32", -70_000),
        entry_f32("tokenizer.ggml.scores", 0.25),
        entry_u64("tokenizer.ggml.extra_uint64", 70_000),
        entry_i64("tokenizer.ggml.extra_int64", -70_000),
        entry_f64("tokenizer.ggml.extra_float64", 0.5),
        entry_f32_array("tokenizer.ggml.extra_float32_array", &[0.1, 0.2]),
        entry_u64_array("tokenizer.ggml.extra_uint64_array", &[1, 2]),
    ]);

    Tokenizer::from_spec(&spec(entries)).expect("valid irrelevant metadata should parse");
}

#[test]
fn rejects_huge_array_length_before_allocation() {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_array_header_only("tokenizer.ggml.merges", 8, u64::MAX),
    );

    let error = Tokenizer::from_spec(&spec(entries)).expect_err("huge array must fail");

    assert!(error.to_string().contains("array"));
}

#[test]
fn official_qwen35_vectors_match_when_artifact_is_available() {
    let Ok(path) = std::env::var("RAFTINFER_QWEN35_GGUF") else {
        eprintln!("skipping official tokenizer vectors: RAFTINFER_QWEN35_GGUF is not set");
        return;
    };
    let engine = Engine::new(EngineConfig::default()).expect("engine");
    let model = engine.load_model(path).expect("load model");
    let spec = model.tokenizer_spec().expect("tokenizer spec");
    let tokenizer = Tokenizer::from_spec(&spec).expect("parse tokenizer");

    for case in official_cases() {
        let tokens = tokenizer
            .encode(&case.text, false)
            .unwrap_or_else(|error| panic!("encode {}: {error}", case.name));
        assert_eq!(tokens, case.tokens, "{}", case.name);
        assert_eq!(
            tokenizer.decode(&case.tokens, false).expect("decode"),
            case.text,
            "{}",
            case.name
        );
    }

    let rendered = tokenizer
        .apply_chat_template(
            &[
                ChatMessage::new(ChatRole::System, "You are concise."),
                ChatMessage::new(ChatRole::User, "你好, GPU!"),
            ],
            true,
        )
        .expect("chat template");
    assert_eq!(rendered, OFFICIAL_CHAT_RENDERED);
    assert_eq!(
        tokenizer
            .encode(&rendered, false)
            .expect("encode rendered chat"),
        OFFICIAL_CHAT_TOKENS
    );
}

#[derive(Debug)]
struct OfficialCase {
    name: String,
    text: String,
    tokens: Vec<i32>,
}

const OFFICIAL_CHAT_RENDERED: &str = "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\n你好, GPU!<|im_end|>\n<|im_start|>assistant\n<think>";
const OFFICIAL_CHAT_TOKENS: &[i32] = &[
    248045, 8678, 198, 2523, 513, 61446, 13, 248046, 198, 248045, 846, 198, 109266, 11, 21966, 0,
    248046, 198, 248045, 74455, 198, 248068,
];

fn official_cases() -> Vec<OfficialCase> {
    let corpus = include_str!("../../../tests/parity/qwen35-tokenizer-corpus.jsonl");
    corpus
        .lines()
        .filter(|line| line.contains("\"kind\":\"encode\""))
        .map(parse_official_case)
        .collect()
}

fn parse_official_case(line: &str) -> OfficialCase {
    let name = extract_json_string(line, "name").expect("name");
    let text = extract_json_string(line, "text").expect("text");
    let tokens = extract_token_array(line);
    OfficialCase { name, text, tokens }
}

#[test]
fn json_string_parser_decodes_current_corpus_escapes() {
    let line = r#"{"text":"quote: \" slash: \\ newline:\n tab:\t mark:\u0651"}"#;

    assert_eq!(
        extract_json_string(line, "text").expect("json string"),
        "quote: \" slash: \\ newline:\n tab:\t mark:\u{0651}"
    );
}

fn extract_json_string(line: &str, key: &str) -> Result<String, String> {
    let key_literal = format!("\"{key}\"");
    let key_start = line
        .find(&key_literal)
        .ok_or_else(|| format!("missing JSON key {key:?}"))?;
    let after_key = &line[key_start + key_literal.len()..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| format!("missing colon after JSON key {key:?}"))?;
    let value = after_key[colon + 1..].trim_start();
    let mut chars = value.chars();
    if chars.next() != Some('"') {
        return Err(format!("JSON key {key:?} is not a string"));
    }
    parse_json_string_chars(chars)
}

fn parse_json_string_chars(chars: std::str::Chars<'_>) -> Result<String, String> {
    let mut output = String::new();
    let mut escaped = false;
    let mut unicode_escape: Option<String> = None;
    for ch in chars {
        if let Some(digits) = unicode_escape.as_mut() {
            digits.push(ch);
            if digits.len() == 4 {
                let codepoint = u32::from_str_radix(digits, 16)
                    .map_err(|_| format!("invalid unicode escape: {digits}"))?;
                let decoded = char::from_u32(codepoint)
                    .ok_or_else(|| format!("invalid unicode codepoint: {digits}"))?;
                output.push(decoded);
                unicode_escape = None;
            }
            continue;
        }
        if escaped {
            match ch {
                '"' => output.push('"'),
                '\\' => output.push('\\'),
                'n' => output.push('\n'),
                'r' => output.push('\r'),
                't' => output.push('\t'),
                'u' => unicode_escape = Some(String::new()),
                other => return Err(format!("unsupported JSON escape: \\{other}")),
            }
            escaped = false;
            continue;
        }
        match ch {
            '"' => return Ok(output),
            '\\' => escaped = true,
            other => output.push(other),
        }
    }
    Err("unterminated JSON string".to_owned())
}

fn extract_token_array(line: &str) -> Vec<i32> {
    let start = line.find("\"tokens\":[").expect("tokens") + "\"tokens\":[".len();
    let end = line[start..].find(']').expect("tokens end") + start;
    line[start..end]
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().expect("token id"))
        .collect()
}

fn synthetic_tokenizer() -> Tokenizer {
    Tokenizer::from_spec(&valid_spec()).expect("synthetic spec")
}

const OFFICIAL_ESCAPED_CHAT_TEMPLATE: &str = r#"
{%- macro render_content(content, do_vision_count, is_system_content=false) -%}
{{- content -}}
{%- endmacro -%}
{%- set ns = namespace(last_query_index=messages|length - 1) -%}
{%- for message in messages[::-1] -%}
{%- if message.role == "user" -%}
{%- set content = render_content(message.content, false)|trim -%}
{%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
{%- set content = render_content(message.content, true)|trim -%}
{%- if message.role == "user" -%}
{{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' -}}
{%- elif message.role == "assistant" -%}
{%- if '</think>' in content -%}
{{- '<|im_start|>' + message.role + '\n<think>\n' + content -}}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- if enable_thinking is defined and enable_thinking is false -%}
{{- '<think>\n\n</think>\n\n' -}}
{%- else -%}
{{- '<think>\n' -}}
{%- endif -%}
{%- endif -%}
"#;

const FRAGMENT_ONLY_CHAT_TEMPLATE: &str = r#"
{% if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{{- '<think>\n' }}
{{- '<think>\n\n</think>\n\n' }}
{% endif %}
<|im_end|>
"#;

const VALID_CHAT_TEMPLATE: &str = OFFICIAL_ESCAPED_CHAT_TEMPLATE;

fn valid_spec() -> TokenizerSpec {
    spec(vec![
        entry_bool("tokenizer.ggml.add_bos_token", false),
        entry_bool("tokenizer.ggml.add_eos_token", false),
        entry_u32("tokenizer.ggml.bos_token_id", 0),
        entry_string("tokenizer.ggml.model", "gpt2"),
        entry_string("tokenizer.ggml.pre", "qwen2"),
        entry_u32("tokenizer.ggml.eos_token_id", 0),
        entry_u32("tokenizer.ggml.eot_token_id", 10),
        entry_string_array(
            "tokenizer.ggml.merges",
            &[
                "a b",
                "ab c",
                "Ġ a",
                "Ù ¡",
                "ï ¼",
                "ï¼ ĳ",
                "Ù¡ ï¼ĳ",
                "Ġ Ġ",
                "ĠĠ Ċ",
                "Ġ ĉ",
                "Ġĉ Ċ",
                "a ï",
                "aï ¸",
                "aï¸ ı",
                "Ġ b",
                "Ġ !",
            ],
        ),
        entry_u32("tokenizer.ggml.padding_token_id", 0),
        entry_i32_array(
            "tokenizer.ggml.token_type",
            &[
                3, 1, 1, 1, 1, 1, 6, 1, 1, 3, 3, 1, 1, 1, 6, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
        ),
        entry_string_array(
            "tokenizer.ggml.tokens",
            &[
                "<|endoftext|>",
                "a",
                "b",
                "c",
                "ab",
                "abc",
                "Ġ",
                "Ġa",
                "!",
                "<|im_start|>",
                "<|im_end|>",
                "system",
                "user",
                "assistant",
                "Ċ",
                "<think>",
                "H",
                "i",
                "u",
                "s",
                "e",
                "r",
                "<custom>",
                "Ù",
                "¡",
                "Ù¡",
                "ï",
                "¼",
                "ĳ",
                "ï¼",
                "ï¼ĳ",
                "Ù¡ï¼ĳ",
                "ĠĠ",
                "ĠĠĊ",
                "ĉ",
                "Ġĉ",
                "ĠĉĊ",
                "¸",
                "ı",
                "aï",
                "aï¸",
                "aï¸ı",
                "Ġb",
                "Ġ!",
            ],
        ),
        entry_string("tokenizer.chat_template", VALID_CHAT_TEMPLATE),
    ])
}

fn corrupt_magic() -> TokenizerSpec {
    let mut spec = valid_spec();
    spec.bytes[0] = b'X';
    spec
}

fn corrupt_version() -> TokenizerSpec {
    let mut spec = valid_spec();
    spec.bytes[7] = 2;
    spec
}

fn corrupt_trailing_bytes() -> TokenizerSpec {
    let mut spec = valid_spec();
    spec.bytes.push(0);
    spec
}

fn corrupt_duplicate_key() -> TokenizerSpec {
    spec(vec![
        entry_string("tokenizer.ggml.model", "gpt2"),
        entry_string("tokenizer.ggml.model", "gpt2"),
    ])
}

fn corrupt_unsorted_key() -> TokenizerSpec {
    spec_in_order(vec![
        entry_string("tokenizer.ggml.tokens", "not first"),
        entry_string("tokenizer.ggml.model", "gpt2"),
    ])
}

fn corrupt_missing_tokens() -> TokenizerSpec {
    let mut entries = valid_entries();
    entries.retain(|entry| entry.key != "tokenizer.ggml.tokens");
    spec(entries)
}

fn corrupt_wrong_model_type() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(&mut entries, entry_u32("tokenizer.ggml.model", 7));
    spec(entries)
}

fn corrupt_duplicate_tokens() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_string_array("tokenizer.ggml.tokens", &["<|endoftext|>", "a", "a"]),
    );
    spec(entries)
}

fn corrupt_empty_control_token() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_string_array("tokenizer.ggml.tokens", &["", "a", "b"]),
    );
    replace_entry(
        &mut entries,
        entry_i32_array("tokenizer.ggml.token_type", &[3, 1, 1]),
    );
    spec(entries)
}

fn corrupt_short_token_types() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_i32_array("tokenizer.ggml.token_type", &[3, 1]),
    );
    spec(entries)
}

fn corrupt_malformed_merge() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_string_array("tokenizer.ggml.merges", &["a b c"]),
    );
    spec(entries)
}

fn corrupt_special_id() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(&mut entries, entry_u32("tokenizer.ggml.eos_token_id", 999));
    spec(entries)
}

fn corrupt_huge_entry_count() -> TokenizerSpec {
    let mut bytes = b"RIFTOK\0\x01".to_vec();
    push_u32(&mut bytes, u32::MAX);
    TokenizerSpec { version: 1, bytes }
}

fn corrupt_huge_token_array_length() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_array_header_only("tokenizer.ggml.tokens", 8, i32::MAX as u64 + 1),
    );
    spec(entries)
}

fn corrupt_strict_prefixes() -> TokenizerSpec {
    let valid = valid_spec();
    for len in 0..valid.bytes.len() {
        let prefix = TokenizerSpec {
            version: valid.version,
            bytes: valid.bytes[..len].to_vec(),
        };
        if Tokenizer::from_spec(&prefix).is_ok() {
            panic!("strict prefix of length {len} parsed successfully");
        }
    }
    TokenizerSpec {
        version: valid.version,
        bytes: valid.bytes[..valid.bytes.len() - 1].to_vec(),
    }
}

fn corrupt_missing_chat_template() -> TokenizerSpec {
    let mut entries = valid_entries();
    entries.retain(|entry| entry.key != "tokenizer.chat_template");
    spec(entries)
}

fn corrupt_empty_chat_template() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(&mut entries, entry_string("tokenizer.chat_template", ""));
    spec(entries)
}

fn corrupt_different_chat_template() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_string("tokenizer.chat_template", "plain nonempty template"),
    );
    spec(entries)
}

fn corrupt_wrong_tokens_type() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(&mut entries, entry_string("tokenizer.ggml.tokens", "abc"));
    spec(entries)
}

fn corrupt_wrong_merges_array_type() -> TokenizerSpec {
    let mut entries = valid_entries();
    replace_entry(
        &mut entries,
        entry_i32_array("tokenizer.ggml.merges", &[1, 2]),
    );
    spec(entries)
}

fn valid_entries() -> Vec<Entry> {
    decode_entries(valid_spec())
}

fn replace_entry(entries: &mut [Entry], replacement: Entry) {
    let index = entries
        .iter()
        .position(|entry| entry.key == replacement.key)
        .expect("entry exists");
    entries[index] = replacement;
}

#[derive(Clone)]
struct Entry {
    key: &'static str,
    bytes: Vec<u8>,
}

fn spec(mut entries: Vec<Entry>) -> TokenizerSpec {
    entries.sort_by_key(|entry| entry.key);
    spec_in_order(entries)
}

fn spec_in_order(entries: Vec<Entry>) -> TokenizerSpec {
    let mut bytes = b"RIFTOK\0\x01".to_vec();
    push_u32(&mut bytes, entries.len() as u32);
    for entry in entries {
        push_string(&mut bytes, entry.key);
        bytes.extend(entry.bytes);
    }
    TokenizerSpec { version: 1, bytes }
}

fn decode_entries(_spec: TokenizerSpec) -> Vec<Entry> {
    vec![
        entry_bool("tokenizer.ggml.add_bos_token", false),
        entry_bool("tokenizer.ggml.add_eos_token", false),
        entry_u32("tokenizer.ggml.bos_token_id", 0),
        entry_string("tokenizer.ggml.model", "gpt2"),
        entry_string("tokenizer.ggml.pre", "qwen2"),
        entry_u32("tokenizer.ggml.eos_token_id", 0),
        entry_u32("tokenizer.ggml.eot_token_id", 10),
        entry_string_array(
            "tokenizer.ggml.merges",
            &[
                "a b",
                "ab c",
                "Ġ a",
                "Ù ¡",
                "ï ¼",
                "ï¼ ĳ",
                "Ù¡ ï¼ĳ",
                "Ġ Ġ",
                "ĠĠ Ċ",
                "Ġ ĉ",
                "Ġĉ Ċ",
                "a ï",
                "aï ¸",
                "aï¸ ı",
                "Ġ b",
            ],
        ),
        entry_u32("tokenizer.ggml.padding_token_id", 0),
        entry_i32_array(
            "tokenizer.ggml.token_type",
            &[
                3, 1, 1, 1, 1, 1, 6, 1, 1, 3, 3, 1, 1, 1, 6, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
        ),
        entry_string_array(
            "tokenizer.ggml.tokens",
            &[
                "<|endoftext|>",
                "a",
                "b",
                "c",
                "ab",
                "abc",
                "Ġ",
                "Ġa",
                "!",
                "<|im_start|>",
                "<|im_end|>",
                "system",
                "user",
                "assistant",
                "Ċ",
                "<think>",
                "H",
                "i",
                "u",
                "s",
                "e",
                "r",
                "<custom>",
                "Ù",
                "¡",
                "Ù¡",
                "ï",
                "¼",
                "ĳ",
                "ï¼",
                "ï¼ĳ",
                "Ù¡ï¼ĳ",
                "ĠĠ",
                "ĠĠĊ",
                "ĉ",
                "Ġĉ",
                "ĠĉĊ",
                "¸",
                "ı",
                "aï",
                "aï¸",
                "aï¸ı",
                "Ġb",
            ],
        ),
        entry_string("tokenizer.chat_template", VALID_CHAT_TEMPLATE),
    ]
}

fn entry_u32(key: &'static str, value: u32) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 4);
    push_u32(&mut bytes, value);
    Entry { key, bytes }
}

fn entry_u8(key: &'static str, value: u8) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 0);
    bytes.push(value);
    Entry { key, bytes }
}

fn entry_i8(key: &'static str, value: i8) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 1);
    bytes.push(value as u8);
    Entry { key, bytes }
}

fn entry_u16(key: &'static str, value: u16) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 2);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_i16(key: &'static str, value: i16) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 3);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_i32(key: &'static str, value: i32) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 5);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_f32(key: &'static str, value: f32) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 6);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_bool(key: &'static str, value: bool) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 7);
    bytes.push(u8::from(value));
    Entry { key, bytes }
}

fn entry_u64(key: &'static str, value: u64) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 10);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_i64(key: &'static str, value: i64) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 11);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_f64(key: &'static str, value: f64) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 12);
    bytes.extend(value.to_le_bytes());
    Entry { key, bytes }
}

fn entry_string(key: &'static str, value: &str) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 8);
    push_string(&mut bytes, value);
    Entry { key, bytes }
}

fn entry_array_header_only(key: &'static str, element_type: u32, len: u64) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 9);
    push_u32(&mut bytes, element_type);
    push_u64(&mut bytes, len);
    Entry { key, bytes }
}

fn entry_f32_array(key: &'static str, values: &[f32]) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 9);
    push_u32(&mut bytes, 6);
    push_u64(&mut bytes, values.len() as u64);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    Entry { key, bytes }
}

fn entry_u64_array(key: &'static str, values: &[u64]) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 9);
    push_u32(&mut bytes, 10);
    push_u64(&mut bytes, values.len() as u64);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    Entry { key, bytes }
}

fn entry_string_array(key: &'static str, values: &[&str]) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 9);
    push_u32(&mut bytes, 8);
    push_u64(&mut bytes, values.len() as u64);
    for value in values {
        push_string(&mut bytes, value);
    }
    Entry { key, bytes }
}

fn entry_i32_array(key: &'static str, values: &[i32]) -> Entry {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 9);
    push_u32(&mut bytes, 5);
    push_u64(&mut bytes, values.len() as u64);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    Entry { key, bytes }
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    push_u32(bytes, value.len() as u32);
    bytes.extend(value.as_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}
