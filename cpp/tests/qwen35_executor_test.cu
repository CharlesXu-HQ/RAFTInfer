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
#include <cuda/stream_ref>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

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
      "full attention KV cache byte size overflow");
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
  const std::size_t max_context = 64;
  const std::size_t workspace =
      brt::Qwen35Executor::workspace_bytes(model.qwen35_config(), max_context);
  brt::DeviceContext device{0, 256U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);
  auto owner = device.create_execution_owner(workspace);
  auto context = owner->execution_context();
  brt::Qwen35Executor executor{context, model.qwen35_config(), *weights,
                               max_context};

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
}

void run_executor_reference_and_allocation_tests() {
  assert(cudaSetDevice(0) == cudaSuccess);
  const auto path =
      write_fixture(brt::test::make_qwen35_nonzero_bf16_gguf_fixture());
  brt::model::Model model{path.string()};
  constexpr std::size_t max_context = 64;
  const std::size_t workspace_bytes =
      brt::Qwen35Executor::workspace_bytes(model.qwen35_config(), max_context);

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
                               max_context};

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
  for (std::size_t index = 0; index < 32; ++index) {
    const auto result =
        executor.decode(static_cast<std::int32_t>((index + 6) % 16));
    assert(result.position == prompt_and_decode.size() + index);
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
  run_executor_reference_and_allocation_tests();
}
