#include "../execution/qwen35_executor.hpp"
#include "../execution/workspace_arena.hpp"
#include "../foundation/device_context.hpp"
#include "../kernels/qwen35_primitives.cuh"
#include "../model/model.hpp"
#include "../reference/qwen35_executor.hpp"

#include "assert_enabled.hpp"
#include "qwen35_gguf_fixture.hpp"
#include "qwen35_nonzero_fixture.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/stream_ref>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace brt::test {

struct InputCastLaunches {
  std::size_t total{};
  std::vector<std::size_t> full_attention_projection;
  std::vector<std::size_t> linear_attention_projection;
  std::vector<std::size_t> ffn_gate_up;
};

void reset_qwen35_executor_input_cast_launches();
InputCastLaunches qwen35_executor_input_cast_launches();

} // namespace brt::test

namespace {

brt::model::Qwen35Config tiny_config() {
  return brt::model::Qwen35Config{
      .vocabulary_size = 16,
      .hidden_size = 8,
      .intermediate_size = 16,
      .context_length = 128,
      .full_attention_head_count = 2,
      .full_attention_kv_head_count = 1,
      .full_attention_head_dimension = 4,
      .linear_key_head_count = 1,
      .linear_value_head_count = 2,
      .linear_head_dimension = 4,
      .linear_convolution_width = 4,
      .rotary_dimension = 2,
      .rms_norm_epsilon = 1.0e-6F,
      .rope_frequency_base = 10000.0F,
      .blocks = {{0, brt::model::Qwen35BlockKind::linear_attention},
                 {1, brt::model::Qwen35BlockKind::linear_attention},
                 {2, brt::model::Qwen35BlockKind::linear_attention},
                 {3, brt::model::Qwen35BlockKind::full_attention}},
  };
}

brt::model::Qwen35Config release_attention_config() {
  auto config = tiny_config();
  config.hidden_size = 4096;
  config.full_attention_head_count = 16;
  config.full_attention_kv_head_count = 4;
  config.full_attention_head_dimension = 256;
  config.linear_key_head_count = 4;
  config.linear_value_head_count = 16;
  config.linear_head_dimension = 256;
  config.rotary_dimension = 64;
  config.blocks = {{0, brt::model::Qwen35BlockKind::full_attention}};
  return config;
}

brt::Qwen35ExecutionPolicy materialized_policy() {
  auto policy = brt::Qwen35ExecutionPolicy{};
  policy.attention = brt::Qwen35AttentionImplementation::materialized_reference;
  policy.decode_graph = false;
  policy.grouped_input_casts = false;
  return policy;
}

std::vector<std::uint8_t> make_release_attention_fixture() {
  constexpr std::uint32_t tensor_type = 30;
  constexpr std::uint32_t vocabulary_size = 16;
  constexpr std::uint32_t hidden_size = 4096;
  constexpr std::uint32_t intermediate_size = 16;
  constexpr std::uint32_t query_heads = 16;
  constexpr std::uint32_t kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;

  std::vector<std::uint8_t> metadata;
  std::uint64_t metadata_count = 0;
  const auto u32 = [&](const std::string &key, std::uint32_t value) {
    brt::test::detail::append_u32_metadata(metadata, key, value);
    ++metadata_count;
  };
  const auto f32 = [&](const std::string &key, float value) {
    brt::test::detail::append_float_metadata(metadata, key, value);
    ++metadata_count;
  };
  const auto string = [&](const std::string &key, const std::string &value) {
    brt::test::detail::append_string_metadata(metadata, key, value);
    ++metadata_count;
  };
  const auto strings = [&](const std::string &key,
                           const std::vector<std::string> &values) {
    brt::test::detail::append_string_array_metadata(metadata, key, values);
    ++metadata_count;
  };

  string("general.architecture", "qwen35");
  u32("general.alignment", 32);
  u32("qwen35.embedding_length", hidden_size);
  u32("qwen35.feed_forward_length", intermediate_size);
  u32("qwen35.context_length", 4);
  u32("qwen35.block_count", 1);
  u32("qwen35.attention.head_count", query_heads);
  u32("qwen35.attention.head_count_kv", kv_heads);
  u32("qwen35.attention.key_length", head_dim);
  u32("qwen35.attention.value_length", head_dim);
  f32("qwen35.attention.layer_norm_rms_epsilon", 1.0e-6F);
  f32("qwen35.rope.freq_base", 10'000.0F);
  u32("qwen35.rope.dimension_count", 64);
  u32("qwen35.ssm.conv_kernel", 4);
  u32("qwen35.ssm.state_size", head_dim);
  u32("qwen35.ssm.group_count", kv_heads);
  u32("qwen35.ssm.time_step_rank", query_heads);
  u32("qwen35.ssm.inner_size", hidden_size);
  u32("qwen35.full_attention_interval", 1);
  string("tokenizer.ggml.model", "gpt2");
  string("tokenizer.ggml.pre", "qwen2");
  strings("tokenizer.ggml.tokens", {"<eos>", "a", "b", "c", "d", "e", "f", "g",
                                    "h", "i", "j", "k", "l", "m", "n", "o"});
  strings("tokenizer.ggml.merges", {"a b", "b c"});
  u32("tokenizer.ggml.eos_token_id", 0);
  string("tokenizer.chat_template", "{{ messages }}");

  std::vector<brt::test::detail::Tensor> tensors;
  std::uint64_t next_offset = 0;
  const auto add = [&](std::string name,
                       std::vector<std::uint64_t> dimensions) {
    brt::test::detail::add_tensor(tensors, next_offset, std::move(name),
                                  std::move(dimensions), tensor_type);
  };
  add("token_embd.weight", {hidden_size, vocabulary_size});
  add("output_norm.weight", {hidden_size});
  add("output.weight", {hidden_size, vocabulary_size});
  add("blk.0.attn_norm.weight", {hidden_size});
  add("blk.0.post_attention_norm.weight", {hidden_size});
  add("blk.0.ffn_gate.weight", {hidden_size, intermediate_size});
  add("blk.0.ffn_down.weight", {intermediate_size, hidden_size});
  add("blk.0.ffn_up.weight", {hidden_size, intermediate_size});
  add("blk.0.attn_q.weight", {hidden_size, query_heads * head_dim * 2});
  add("blk.0.attn_k.weight", {hidden_size, kv_heads * head_dim});
  add("blk.0.attn_v.weight", {hidden_size, kv_heads * head_dim});
  add("blk.0.attn_output.weight", {query_heads * head_dim, hidden_size});
  add("blk.0.attn_q_norm.weight", {head_dim});
  add("blk.0.attn_k_norm.weight", {head_dim});

  std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
  brt::test::detail::append<std::uint32_t>(bytes, 3);
  brt::test::detail::append<std::uint64_t>(bytes, tensors.size());
  brt::test::detail::append<std::uint64_t>(bytes, metadata_count);
  bytes.insert(bytes.end(), metadata.begin(), metadata.end());
  for (const auto &tensor : tensors) {
    brt::test::detail::append_string(bytes, tensor.name);
    brt::test::detail::append<std::uint32_t>(bytes, tensor.dimensions.size());
    for (const auto dimension : tensor.dimensions)
      brt::test::detail::append(bytes, dimension);
    brt::test::detail::append<std::uint32_t>(bytes, tensor.type);
    brt::test::detail::append(bytes, tensor.offset);
  }
  while (bytes.size() % 32 != 0)
    bytes.push_back(0);
  bytes.resize(bytes.size() + next_offset);
  return bytes;
}

std::filesystem::path write_fixture(std::vector<std::uint8_t> bytes) {
  const auto path =
      std::filesystem::temp_directory_path() / "brt_qwen35_executor.gguf";
  std::ofstream output{path, std::ios::binary};
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  assert(output);
  return path;
}

void expect_executor_error(auto &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const brt::Qwen35ExecutorError &) {
    thrown = true;
  }
  assert(thrown);
}

void expect_executor_error_containing(auto &&fn, const std::string &expected) {
  try {
    fn();
  } catch (const brt::Qwen35ExecutorError &error) {
    assert(std::string{error.what()}.find(expected) != std::string::npos);
    return;
  }
  assert(false);
}

void expect_primitive_error_containing(auto &&fn, const std::string &expected) {
  try {
    fn();
  } catch (const brt::kernels::Qwen35PrimitiveError &error) {
    assert(std::string{error.what()}.find(expected) != std::string::npos);
    return;
  }
  assert(false);
}

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes == 0 ? 1 : bytes) {
    assert(cudaMalloc(&data_, bytes_) == cudaSuccess);
  }
  ~DeviceBuffer() noexcept {
    if (data_ != nullptr)
      (void)cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : bytes_(std::exchange(other.bytes_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}
  DeviceBuffer &operator=(DeviceBuffer &&) = delete;

  void *data() const noexcept { return data_; }

private:
  std::size_t bytes_{};
  void *data_{};
};

class Stream {
public:
  Stream() {
    assert(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
           cudaSuccess);
  }
  ~Stream() noexcept {
    if (stream_ != nullptr)
      (void)cudaStreamDestroy(stream_);
  }
  cudaStream_t get() const noexcept { return stream_; }

private:
  cudaStream_t stream_{};
};

template <typename T> struct DTypeTraits;

template <> struct DTypeTraits<__half> {
  static constexpr BrtDataType dtype = BRT_DTYPE_F16;
  static __half encode(float value) { return __float2half_rn(value); }
  static float decode(__half value) { return __half2float(value); }
};

template <> struct DTypeTraits<__nv_bfloat16> {
  static constexpr BrtDataType dtype = BRT_DTYPE_BF16;
  static __nv_bfloat16 encode(float value) {
    return __float2bfloat16_rn(value);
  }
  static float decode(__nv_bfloat16 value) { return __bfloat162float(value); }
};

template <typename T> std::vector<T> encode(std::span<const float> values) {
  std::vector<T> output(values.size());
  std::transform(values.begin(), values.end(), output.begin(),
                 DTypeTraits<T>::encode);
  return output;
}

template <typename T>
DeviceBuffer upload(std::span<const T> values, cudaStream_t stream) {
  DeviceBuffer output{values.size_bytes()};
  assert(cudaMemcpyAsync(output.data(), values.data(), values.size_bytes(),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
  return output;
}

template <typename T>
std::vector<T> download(const DeviceBuffer &buffer, std::size_t elements,
                        cudaStream_t stream) {
  std::vector<T> output(elements);
  assert(cudaMemcpyAsync(output.data(), buffer.data(), elements * sizeof(T),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  return output;
}

template <typename T>
void assert_close(std::span<const T> actual, std::span<const float> expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float got = DTypeTraits<T>::decode(actual[i]);
    assert(std::fabs(got - expected[i]) <= 2.0e-2F);
  }
}

constexpr float kExecutorLogitTolerance = 3.0e-2F;

void assert_reference_close(const char *stage, std::span<const float> actual,
                            std::span<const float> expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float absolute = std::fabs(actual[i] - expected[i]);
    const float denominator = std::max(std::fabs(expected[i]), 1.0e-12F);
    if (!(absolute <= kExecutorLogitTolerance ||
          absolute / denominator <= kExecutorLogitTolerance)) {
      std::fprintf(stderr,
                   "reference mismatch stage=%s index=%zu actual=%.9g "
                   "expected=%.9g absolute=%.9g relative=%.9g\n",
                   stage, i, static_cast<double>(actual[i]),
                   static_cast<double>(expected[i]),
                   static_cast<double>(absolute),
                   static_cast<double>(absolute / denominator));
    }
    assert(absolute <= kExecutorLogitTolerance ||
           absolute / denominator <= kExecutorLogitTolerance);
  }
}

void run_workspace_contract_tests() {
  const auto config = tiny_config();
  const std::size_t bytes = brt::Qwen35Executor::workspace_bytes(config, 17);
  assert(bytes > 0);
  assert(bytes % brt::Qwen35Executor::workspace_alignment == 0);
  assert(bytes == brt::Qwen35Executor::workspace_bytes(config, 17));
  assert(bytes >= brt::Qwen35Executor::workspace_bytes(config, 4));
  assert(brt::Qwen35Executor::workspace_bytes(config, 64) > bytes);
  assert(bytes == brt::Qwen35Executor::workspace_bytes(config, 17,
                                                       materialized_policy()));

  const auto release = release_attention_config();
  const auto reference = materialized_policy();
  const auto online = brt::Qwen35ExecutionPolicy{};
  const std::size_t reference_bytes =
      brt::Qwen35Executor::workspace_bytes(release, 17, reference);
  const std::size_t online_bytes =
      brt::Qwen35Executor::workspace_bytes(release, 17, online);
  assert(online_bytes < reference_bytes);
  assert(reference_bytes - online_bytes ==
         ((17 * release.full_attention_head_count * 17 * sizeof(float) +
           brt::Qwen35Executor::workspace_alignment - 1) /
          brt::Qwen35Executor::workspace_alignment) *
             brt::Qwen35Executor::workspace_alignment);
  auto invalid_reference = reference;
  invalid_reference.kv_cache = brt::Qwen35KvCacheDType::bf16;
  expect_executor_error([&] {
    (void)brt::Qwen35Executor::workspace_bytes(release, 17, invalid_reference);
  });
  auto unsupported_online = online;
  unsupported_online.kv_cache_layout = brt::Qwen35KvCacheLayout::head_major;
  assert(brt::Qwen35Executor::workspace_bytes(release, 1, unsupported_online) ==
         brt::Qwen35Executor::workspace_bytes(release, 1, reference));

  auto invalid = config;
  invalid.hidden_size = 0;
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(invalid, 17); });
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(config, 0); });
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(config, 129); });

  invalid = config;
  invalid.full_attention_head_dimension = 3;
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(invalid, 17); });
  invalid = config;
  invalid.linear_value_head_count = 1;
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(invalid, 17); });
  invalid = config;
  invalid.linear_key_head_count = 2;
  invalid.linear_value_head_count = 3;
  invalid.hidden_size =
      invalid.linear_value_head_count * invalid.linear_head_dimension;
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(invalid, 17); });
  invalid = config;
  invalid.blocks[1].index = 7;
  expect_executor_error(
      [&] { (void)brt::Qwen35Executor::workspace_bytes(invalid, 17); });

  invalid = config;
  invalid.hidden_size = std::numeric_limits<std::uint32_t>::max();
  invalid.full_attention_head_count = 1;
  invalid.full_attention_kv_head_count = 1;
  invalid.full_attention_head_dimension =
      std::numeric_limits<std::uint32_t>::max();
  invalid.linear_key_head_count = 1;
  invalid.linear_value_head_count = 1;
  invalid.linear_head_dimension = std::numeric_limits<std::uint32_t>::max();
  invalid.linear_convolution_width = std::numeric_limits<std::uint32_t>::max();
  invalid.context_length = std::numeric_limits<std::uint32_t>::max();
  expect_executor_error_containing(
      [&] {
        (void)brt::Qwen35Executor::workspace_bytes(
            invalid, std::numeric_limits<std::uint32_t>::max());
      },
      "attention cache shape overflow");
}

void run_host_validation_tests() {
  const auto config = tiny_config();
  const std::vector<int32_t> first_token{1};
  const std::vector<int32_t> last_token{15};
  brt::Qwen35Executor::validate_request(config, 0, first_token);
  brt::Qwen35Executor::validate_request(config, 127, last_token);

  expect_executor_error([&] {
    brt::Qwen35Executor::validate_request(config, 0,
                                          std::span<const int32_t>{});
  });
  expect_executor_error([&] {
    const std::vector<int32_t> tokens{-1};
    brt::Qwen35Executor::validate_request(config, 0, tokens);
  });
  expect_executor_error([&] {
    const std::vector<int32_t> tokens{16};
    brt::Qwen35Executor::validate_request(config, 0, tokens);
  });
  expect_executor_error([&] {
    const std::vector<int32_t> tokens{1, 2};
    brt::Qwen35Executor::validate_request(config, 127, tokens);
  });
}

template <typename T> void run_support_kernel_dtype_tests(cudaStream_t stream) {
  {
    const std::vector<float> logits{-1.0F, 5.0F, 0.0F, 5.0F, 3.0F};
    const auto typed = encode<T>(logits);
    auto device_logits = upload<T>(typed, stream);
    DeviceBuffer device_index{sizeof(std::int32_t)};
    brt::kernels::qwen35_argmax_typed(
        device_logits.data(), static_cast<std::int32_t *>(device_index.data()),
        logits.size(), DTypeTraits<T>::dtype, stream);
    const auto actual = download<std::int32_t>(device_index, 1, stream);
    assert(actual[0] == 1);
  }

  {
    constexpr std::size_t tokens = 2;
    constexpr std::size_t heads = 2;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t hidden = heads * head_dim;
    const std::vector<float> query_gate{1,   2,   3,   101, 102, 103, 4,   5,
                                        6,   104, 105, 106, 7,   8,   9,   107,
                                        108, 109, 10,  11,  12,  110, 111, 112};
    const auto typed = encode<T>(query_gate);
    auto device_input = upload<T>(typed, stream);
    DeviceBuffer device_query{tokens * hidden * sizeof(T)};
    DeviceBuffer device_gate{tokens * hidden * sizeof(T)};
    brt::kernels::qwen35_split_full_query_gate(
        device_input.data(), device_query.data(), device_gate.data(), tokens,
        heads, head_dim, DTypeTraits<T>::dtype, stream);
    const auto query = download<T>(device_query, tokens * hidden, stream);
    const auto gate = download<T>(device_gate, tokens * hidden, stream);
    const std::vector<float> expected_query{1, 2, 3, 4,  5,  6,
                                            7, 8, 9, 10, 11, 12};
    const std::vector<float> expected_gate{101, 102, 103, 104, 105, 106,
                                           107, 108, 109, 110, 111, 112};
    assert_close<T>(query, expected_query);
    assert_close<T>(gate, expected_gate);
  }

  {
    constexpr std::size_t tokens = 2;
    constexpr std::size_t qkv_width = 4;
    constexpr std::size_t beta_width = 2;
    constexpr std::size_t alpha_width = 2;
    constexpr std::size_t gate_width = 3;
    constexpr std::size_t packed_width =
        qkv_width + beta_width + alpha_width + gate_width;
    const std::vector<float> qkv{1, 2, 3, 4, 11, 12, 13, 14};
    const std::vector<float> beta{21, 22, 23, 24};
    const std::vector<float> alpha{31, 32, 33, 34};
    const std::vector<float> gate{41, 42, 43, 44, 45, 46};
    auto device_qkv = upload<T>(encode<T>(qkv), stream);
    auto device_beta = upload<T>(encode<T>(beta), stream);
    auto device_alpha = upload<T>(encode<T>(alpha), stream);
    auto device_gate = upload<T>(encode<T>(gate), stream);
    DeviceBuffer device_packed{tokens * packed_width * sizeof(T)};
    brt::kernels::qwen35_pack_linear_delta_input(
        device_qkv.data(), device_beta.data(), device_alpha.data(),
        device_gate.data(), device_packed.data(), tokens, qkv_width, beta_width,
        alpha_width, gate_width, DTypeTraits<T>::dtype, stream);
    const auto packed =
        download<T>(device_packed, tokens * packed_width, stream);
    const std::vector<float> expected{1,  2,  3,  4,  21, 22, 31, 32,
                                      41, 42, 43, 11, 12, 13, 14, 23,
                                      24, 33, 34, 44, 45, 46};
    assert_close<T>(packed, expected);
  }

  {
    constexpr auto width =
        std::numeric_limits<std::size_t>::max() / std::size_t{3} + 1;
    auto *pointer = reinterpret_cast<void *>(0x1000);
    auto stream_value = reinterpret_cast<cudaStream_t>(0x1000);
    expect_primitive_error_containing(
        [&] {
          brt::kernels::qwen35_pack_linear_delta_input(
              pointer, pointer, pointer, pointer, pointer, 1, width, width,
              width, width, DTypeTraits<T>::dtype, stream_value);
        },
        "width overflow");
  }
}

void run_support_kernel_tests() {
  assert(cudaSetDevice(0) == cudaSuccess);
  Stream stream;
  run_support_kernel_dtype_tests<__half>(stream.get());
  run_support_kernel_dtype_tests<__nv_bfloat16>(stream.get());
}

void run_executor_fixture_smoke() {
  assert(cudaSetDevice(0) == cudaSuccess);
  const auto path = write_fixture(brt::test::make_qwen35_gguf_fixture());
  brt::model::Model model{path.string()};
  const std::size_t max_context = 128;
  const std::size_t workspace =
      brt::Qwen35Executor::workspace_bytes(model.qwen35_config(), max_context);
  brt::DeviceContext device{0, 256U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);
  auto owner = device.create_execution_owner(workspace);
  auto context = owner->execution_context();
  brt::Qwen35Executor executor{context, model.qwen35_config(), *weights,
                               max_context};
  const auto diagnostics = executor.diagnostics();
  assert(diagnostics.attention ==
         brt::Qwen35AttentionImplementation::materialized_reference);
  assert(diagnostics.attention_workspace_bytes > 0);

  const std::vector<std::int32_t> prompt{1, 2, 3, 4, 5};
  const auto prefill = executor.prefill(prompt);
  assert(prefill.position == prompt.size() - 1);
  assert(prefill.token == 0);
  assert(executor.position() == prompt.size());
  assert(!executor.poisoned());

  const auto decoded = executor.decode(6);
  assert(decoded.position == prompt.size());
  assert(decoded.token == 0);
  assert(executor.position() == prompt.size() + 1);
  assert(!executor.poisoned());

  executor.reset();
  assert(executor.position() == 0);
  for (std::size_t i = 0; i < 32; ++i) {
    const auto result = executor.decode(static_cast<std::int32_t>(i % 7));
    assert(result.position == i);
    assert(result.token == 0);
  }
  assert(executor.position() == 32);
  assert(!executor.poisoned());

  executor.reset();
  executor.enable_trace(true);
  std::vector<std::int32_t> long_prompt(max_context);
  for (std::size_t index = 0; index < long_prompt.size(); ++index)
    long_prompt[index] = static_cast<std::int32_t>(index % 16);
  const auto long_prefill = executor.prefill(long_prompt);
  assert(long_prefill.position == long_prompt.size() - 1);
  assert(std::count_if(executor.trace().begin(), executor.trace().end(),
                       [](const brt::Qwen35TraceEntry &entry) {
                         return entry.name == "model.input_embed";
                       }) == 1);
  executor.enable_trace(false);
}

void run_executor_f32_auxiliary_smoke() {
  assert(cudaSetDevice(0) == cudaSuccess);
  const auto path =
      write_fixture(brt::test::make_qwen35_gguf_fixture(30, true));
  brt::model::Model model{path.string()};
  constexpr std::size_t max_context = 64;
  const std::size_t workspace =
      brt::Qwen35Executor::workspace_bytes(model.qwen35_config(), max_context);
  brt::DeviceContext device{0, 256U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);
  assert(weights->output_norm().type == brt::model::CudaWeightType::f32);
  auto owner = device.create_execution_owner(workspace);
  auto context = owner->execution_context();
  brt::Qwen35Executor executor{context, model.qwen35_config(), *weights,
                               max_context};
  const std::vector<std::int32_t> prompt{1, 2, 3, 4};
  const auto result = executor.prefill(prompt);
  assert(result.position == prompt.size() - 1);
  assert(!executor.poisoned());
}

void run_grouped_input_cast_tests() {
  assert(cudaSetDevice(0) == cudaSuccess);
  const auto path =
      write_fixture(brt::test::make_qwen35_nonzero_bf16_gguf_fixture());
  brt::model::Model model{path.string()};
  constexpr std::size_t max_context = 64;

  auto grouped_policy = materialized_policy();
  grouped_policy.grouped_input_casts = true;
  const auto ungrouped_policy = materialized_policy();
  const std::size_t workspace_bytes = brt::Qwen35Executor::workspace_bytes(
      model.qwen35_config(), max_context, grouped_policy);

  brt::DeviceContext device{0, 256U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);
  auto grouped_owner = device.create_execution_owner(workspace_bytes);
  auto ungrouped_owner = device.create_execution_owner(workspace_bytes);
  auto grouped_context = grouped_owner->execution_context();
  auto ungrouped_context = ungrouped_owner->execution_context();
  brt::Qwen35Executor grouped{grouped_context, model.qwen35_config(), *weights,
                              max_context, grouped_policy};
  brt::Qwen35Executor ungrouped{ungrouped_context, model.qwen35_config(),
                                *weights, max_context, ungrouped_policy};
  const std::vector<std::size_t> expected_grouped_full{1};
  const std::vector<std::size_t> expected_grouped_linear{1, 1, 1};
  const std::vector<std::size_t> expected_grouped_ffn{1, 1, 1, 1};
  const std::vector<std::size_t> expected_ungrouped_full{3};
  const std::vector<std::size_t> expected_ungrouped_linear{4, 4, 4};
  const std::vector<std::size_t> expected_ungrouped_ffn{2, 2, 2, 2};

  brt::test::reset_qwen35_executor_input_cast_launches();
  const auto grouped_result = grouped.decode(1);
  const auto grouped_casts = brt::test::qwen35_executor_input_cast_launches();
  assert(grouped_casts.total == 17);
  assert(grouped_casts.full_attention_projection == expected_grouped_full);
  assert(grouped_casts.linear_attention_projection == expected_grouped_linear);
  assert(grouped_casts.ffn_gate_up == expected_grouped_ffn);

  brt::test::reset_qwen35_executor_input_cast_launches();
  const auto ungrouped_result = ungrouped.decode(1);
  const auto ungrouped_casts = brt::test::qwen35_executor_input_cast_launches();
  assert(ungrouped_casts.total == 32);
  assert(ungrouped_casts.full_attention_projection == expected_ungrouped_full);
  assert(ungrouped_casts.linear_attention_projection ==
         expected_ungrouped_linear);
  assert(ungrouped_casts.ffn_gate_up == expected_ungrouped_ffn);

  assert(grouped_result.token == ungrouped_result.token);
  assert(grouped_result.position == ungrouped_result.position);
  std::vector<float> grouped_logits(model.qwen35_config().vocabulary_size);
  std::vector<float> ungrouped_logits(model.qwen35_config().vocabulary_size);
  grouped.copy_last_logits(grouped_logits);
  ungrouped.copy_last_logits(ungrouped_logits);
  assert(std::memcmp(grouped_logits.data(), ungrouped_logits.data(),
                     grouped_logits.size() * sizeof(grouped_logits.front())) ==
         0);
}

void run_executor_online_materialized_parity_tests() {
  assert(cudaSetDevice(0) == cudaSuccess);
  const auto path = write_fixture(make_release_attention_fixture());
  brt::model::Model model{path.string()};
  constexpr std::size_t max_context = 4;
  const auto reference_policy = materialized_policy();
  const auto online_policy = brt::Qwen35ExecutionPolicy{};
  const std::size_t reference_workspace_bytes =
      brt::Qwen35Executor::workspace_bytes(model.qwen35_config(), max_context,
                                           reference_policy);
  const std::size_t online_workspace_bytes =
      brt::Qwen35Executor::workspace_bytes(model.qwen35_config(), max_context,
                                           online_policy);

  brt::DeviceContext device{0, 256U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);

  raft::device_resources resources;
  const cudaStream_t stream =
      raft::resource::get_cuda_stream(resources).value();
  rmm::mr::cuda_memory_resource cuda_resource;
  rmm::mr::statistics_resource_adaptor statistics{cuda_resource};
  brt::WorkspaceArena reference_workspace{
      rmm::device_async_resource_ref{statistics}, cuda::stream_ref{stream},
      stream, reference_workspace_bytes};
  brt::WorkspaceArena online_workspace{
      rmm::device_async_resource_ref{statistics}, cuda::stream_ref{stream},
      stream, online_workspace_bytes};
  cudaDeviceProp properties{};
  assert(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
  assert(properties.sharedMemPerBlock <=
         static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const auto context_for = [&](brt::WorkspaceArena &workspace) {
    return brt::ExecutionContext{
        resources,
        rmm::device_async_resource_ref{statistics},
        stream,
        workspace,
        0,
        properties.major,
        properties.minor,
        static_cast<int>(properties.sharedMemPerBlock),
    };
  };
  auto reference_context = context_for(reference_workspace);
  auto online_context = context_for(online_workspace);
  brt::Qwen35Executor reference{reference_context, model.qwen35_config(),
                                *weights, max_context, reference_policy};
  brt::Qwen35Executor online{online_context, model.qwen35_config(), *weights,
                             max_context, online_policy};

  const auto reference_diagnostics = reference.diagnostics();
  const auto online_diagnostics = online.diagnostics();
  assert(reference_diagnostics.attention ==
         brt::Qwen35AttentionImplementation::materialized_reference);
  assert(reference_diagnostics.attention_workspace_bytes > 0);
  assert(online_diagnostics.attention ==
         brt::Qwen35AttentionImplementation::online_tiled);
  assert(online_diagnostics.attention_workspace_bytes == 0);

  const std::vector<std::int32_t> prompt{1, 2};
  const auto reference_prefill = reference.prefill(prompt);
  const auto online_prefill = online.prefill(prompt);
  assert(online_prefill.token == reference_prefill.token);
  assert(online_prefill.position == reference_prefill.position);
  std::vector<float> reference_logits(model.qwen35_config().vocabulary_size);
  std::vector<float> online_logits(model.qwen35_config().vocabulary_size);
  reference.copy_last_logits(reference_logits);
  online.copy_last_logits(online_logits);
  assert_reference_close("online-prefill", online_logits, reference_logits);

  const auto reference_decode = reference.decode(3);
  const auto online_decode = online.decode(3);
  assert(online_decode.token == reference_decode.token);
  assert(online_decode.position == reference_decode.position);
  reference.copy_last_logits(reference_logits);
  online.copy_last_logits(online_logits);
  assert_reference_close("online-decode", online_logits, reference_logits);

  (void)statistics.push_counters();
  for (std::size_t index = 0; index < 2; ++index) {
    reference.reset();
    online.reset();
    (void)reference.prefill(prompt);
    (void)online.prefill(prompt);
    (void)reference.decode(3);
    (void)online.decode(3);
  }
  const auto [bytes, allocations] = statistics.pop_counters();
  assert(bytes.value == 0);
  assert(bytes.peak == 0);
  assert(bytes.total == 0);
  assert(allocations.value == 0);
  assert(allocations.peak == 0);
  assert(allocations.total == 0);
}

void run_executor_reference_and_allocation_tests() {
  assert(cudaSetDevice(0) == cudaSuccess);
  const auto path =
      write_fixture(brt::test::make_qwen35_nonzero_bf16_gguf_fixture());
  brt::model::Model model{path.string()};
  constexpr std::size_t max_context = 64;
  const auto policy = materialized_policy();
  const std::size_t workspace_bytes = brt::Qwen35Executor::workspace_bytes(
      model.qwen35_config(), max_context, policy);

  brt::DeviceContext device{0, 256U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);

  raft::device_resources resources;
  const cudaStream_t stream =
      raft::resource::get_cuda_stream(resources).value();
  rmm::mr::cuda_memory_resource cuda_resource;
  rmm::mr::statistics_resource_adaptor statistics{cuda_resource};
  brt::WorkspaceArena workspace{rmm::device_async_resource_ref{statistics},
                                cuda::stream_ref{stream}, stream,
                                workspace_bytes};
  cudaDeviceProp properties{};
  assert(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
  assert(properties.sharedMemPerBlock <=
         static_cast<std::size_t>(std::numeric_limits<int>::max()));
  brt::ExecutionContext context{resources,
                                rmm::device_async_resource_ref{statistics},
                                stream,
                                workspace,
                                0,
                                properties.major,
                                properties.minor,
                                static_cast<int>(properties.sharedMemPerBlock)};
  brt::Qwen35Executor executor{context, model.qwen35_config(), *weights,
                               max_context, policy};
  const auto diagnostics = executor.diagnostics();
  assert(diagnostics.attention ==
         brt::Qwen35AttentionImplementation::materialized_reference);
  assert(diagnostics.kv_cache_dtype == brt::Qwen35KvCacheDType::f32);
  assert(diagnostics.kv_cache_layout == brt::Qwen35KvCacheLayout::token_major);
  assert(!diagnostics.decode_graph_captured);
  assert(!diagnostics.decode_graph_replayed);
  assert(diagnostics.attention_workspace_bytes > 0);

  const std::vector<std::int32_t> prompt{1, 2, 3, 4};
  const auto expected_prefill =
      brt::reference::qwen35_execute_model(model, prompt);
  const auto actual_prefill = executor.prefill(prompt);
  std::vector<float> actual_logits(model.qwen35_config().vocabulary_size);
  executor.copy_last_logits(actual_logits);
  assert(actual_prefill.position == prompt.size() - 1);
  assert(actual_prefill.token == expected_prefill.token);
  assert_reference_close("prefill", actual_logits, expected_prefill.logits);

  const std::vector<std::int32_t> prompt_and_decode{1, 2, 3, 4, 5};
  const auto expected_decode =
      brt::reference::qwen35_execute_model(model, prompt_and_decode);
  const auto actual_decode = executor.decode(5);
  executor.copy_last_logits(actual_logits);
  assert(actual_decode.position == prompt.size());
  assert(actual_decode.token == expected_decode.token);
  const auto split_decode_logits = actual_logits;
  assert_reference_close("decode", split_decode_logits, expected_decode.logits);

  executor.reset();
  const auto actual_batched = executor.prefill(prompt_and_decode);
  executor.copy_last_logits(actual_logits);
  assert(actual_batched.position == prompt_and_decode.size() - 1);
  assert(actual_batched.token == expected_decode.token);
  assert_reference_close("batched-five", actual_logits, expected_decode.logits);

  (void)statistics.push_counters();
  for (std::size_t index = 0; index < 8; ++index) {
    executor.reset();
    const auto repeated_prefill = executor.prefill(prompt);
    assert(repeated_prefill.position == prompt.size() - 1);
    const auto repeated_decode = executor.decode(5);
    assert(repeated_decode.position == prompt.size());
  }
  const auto [bytes, allocations] = statistics.pop_counters();
  assert(bytes.value == 0);
  assert(bytes.peak == 0);
  assert(bytes.total == 0);
  assert(allocations.value == 0);
  assert(allocations.peak == 0);
  assert(allocations.total == 0);
}

} // namespace

int main() {
  const char *opt_in = std::getenv("BRT_RUN_GPU_TESTS");
  if (opt_in == nullptr || std::string{opt_in} != "1") {
    return 77;
  }
  run_workspace_contract_tests();
  run_host_validation_tests();
  run_support_kernel_tests();
  run_executor_fixture_smoke();
  run_executor_f32_auxiliary_smoke();
  run_grouped_input_cast_tests();
  run_executor_online_materialized_parity_tests();
  run_executor_reference_and_allocation_tests();
}
