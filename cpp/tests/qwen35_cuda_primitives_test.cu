#include "../execution/execution_context.hpp"
#include "../execution/workspace_arena.hpp"
#include "../kernels/qwen35_primitives.cuh"
#include "../reference/bf16.hpp"
#include "../reference/operators.hpp"
#include "../reference/qwen35.hpp"

#include "assert_enabled.hpp"

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <cuda/stream_ref>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

class TestResources {
 public:
  explicit TestResources(std::size_t pool_bytes)
      : pool_(cuda_resource_, pool_bytes), workspace_() {
    raft::resource::set_workspace_resource(resources_,
                                           raft::mr::device_resource{pool_});
    raft::resource::set_large_workspace_resource(
        resources_, raft::mr::device_resource{pool_});
    cudaDeviceProp properties{};
    assert(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    const auto stream = raft::resource::get_cuda_stream(resources_).value();
    workspace_ = std::make_unique<brt::WorkspaceArena>(
        rmm::device_async_resource_ref{pool_}, cuda::stream_ref{stream},
        stream, 1024 * 1024);
    context_ = std::make_unique<brt::ExecutionContext>(
        resources_, rmm::device_async_resource_ref{pool_}, stream, *workspace_,
        0, properties.major, properties.minor, properties.sharedMemPerBlock);
  }

  ~TestResources() noexcept {
    if (context_) {
      (void)cudaStreamSynchronize(context_->stream());
    }
    context_.reset();
    workspace_.reset();
  }

  brt::ExecutionContext& context() noexcept { return *context_; }

 private:
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
  raft::device_resources resources_;
  std::unique_ptr<brt::WorkspaceArena> workspace_;
  std::unique_ptr<brt::ExecutionContext> context_;
};

class DeviceBuffer {
 public:
  DeviceBuffer(brt::ExecutionContext& context, std::size_t bytes)
      : resource_(context.memory_resource()),
        stream_ref_(context.stream()),
        stream_(context.stream()),
        bytes_(bytes == 0 ? 1 : bytes),
        data_(resource_.allocate(stream_ref_, bytes_,
                                 rmm::CUDA_ALLOCATION_ALIGNMENT)) {}

  ~DeviceBuffer() noexcept {
    if (data_ == nullptr) return;
    (void)cudaStreamSynchronize(stream_);
    try {
      resource_.deallocate(stream_ref_, data_, bytes_,
                           rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (...) {
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : resource_(other.resource_),
        stream_ref_(other.stream_ref_),
        stream_(other.stream_),
        bytes_(std::exchange(other.bytes_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept = delete;

  void* data() const noexcept { return data_; }

 private:
  rmm::device_async_resource_ref resource_;
  cuda::stream_ref stream_ref_;
  cudaStream_t stream_{};
  std::size_t bytes_{};
  void* data_{};
};

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<__half> {
  static constexpr BrtDataType dtype = BRT_DTYPE_F16;

  static __half from_float(float value) { return __float2half_rn(value); }
  static float to_float(__half value) { return __half2float(value); }
};

template <>
struct DTypeTraits<__nv_bfloat16> {
  static constexpr BrtDataType dtype = BRT_DTYPE_BF16;

  static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
  static float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
};

bool close_enough(float actual, float expected) {
  const float abs = std::fabs(actual - expected);
  const float rel = abs / std::max(std::fabs(expected), 1.0e-6F);
  return abs <= 2.0e-2F || rel <= 2.0e-2F;
}

template <typename T>
std::vector<T> encode(std::span<const float> values) {
  std::vector<T> encoded(values.size());
  std::transform(values.begin(), values.end(), encoded.begin(),
                 DTypeTraits<T>::from_float);
  return encoded;
}

template <typename T>
std::vector<float> decode(std::span<const T> values) {
  std::vector<float> decoded(values.size());
  std::transform(values.begin(), values.end(), decoded.begin(),
                 DTypeTraits<T>::to_float);
  return decoded;
}

template <typename T>
DeviceBuffer upload(brt::ExecutionContext& context, std::span<const T> host) {
  DeviceBuffer device{context, host.size_bytes()};
  assert(cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                         cudaMemcpyHostToDevice,
                         context.stream()) == cudaSuccess);
  return device;
}

template <typename T>
std::vector<T> download(brt::ExecutionContext& context, const void* device,
                        std::size_t elements) {
  std::vector<T> host(elements);
  assert(cudaMemcpyAsync(host.data(), device, elements * sizeof(T),
                         cudaMemcpyDeviceToHost,
                         context.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(context.stream()) == cudaSuccess);
  return host;
}

void expect_primitive_error(auto&& fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const brt::kernels::Qwen35PrimitiveError&) {
    thrown = true;
  }
  assert(thrown);
}

std::vector<float> sequence(std::size_t elements, float scale, float bias) {
  std::vector<float> values(elements);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = bias + scale * static_cast<float>(
                           static_cast<int>((i * 13) % 17) - 8);
  }
  return values;
}

template <typename T>
void assert_matches(std::span<const T> actual, std::span<const float> expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    assert(close_enough(DTypeTraits<T>::to_float(actual[i]), expected[i]));
  }
}

void apply_qk_norm_rope_reference(std::span<const float> input,
                                  std::span<const float> weight,
                                  std::span<float> output,
                                  brt::kernels::QkNormRopeShape shape,
                                  float epsilon) {
  for (std::size_t token = 0; token < shape.tokens; ++token) {
    for (std::size_t head = 0; head < shape.heads; ++head) {
      const std::size_t base =
          (token * shape.heads + head) * shape.head_dim;
      double square_sum = 0.0;
      for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
        const float value = input[base + dim];
        square_sum += static_cast<double>(value) * value;
      }
      const float scale = 1.0F / std::sqrt(
                                      static_cast<float>(
                                          square_sum /
                                          static_cast<double>(shape.head_dim)) +
                                      epsilon);
      for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
        output[base + dim] = input[base + dim] * scale * (1.0F + weight[dim]);
      }
      const std::size_t pair_count = shape.rotary_dim / 2;
      for (std::size_t pair = 0; pair < pair_count; ++pair) {
        const double exponent =
            static_cast<double>(2 * pair) / static_cast<double>(shape.rotary_dim);
        const double theta =
            static_cast<double>(shape.position_offset + token) /
            std::pow(static_cast<double>(shape.rope_base), exponent);
        const float cos_theta = static_cast<float>(std::cos(theta));
        const float sin_theta = static_cast<float>(std::sin(theta));
        const std::size_t first = base + pair;
        const std::size_t second = base + pair + pair_count;
        const float x0 = output[first];
        const float x1 = output[second];
        output[first] = x0 * cos_theta - x1 * sin_theta;
        output[second] = x0 * sin_theta + x1 * cos_theta;
      }
    }
  }
}

template <typename T>
void run_dtype_tests(brt::ExecutionContext& context) {
  for (const std::size_t rows : {std::size_t{1}, std::size_t{2},
                                std::size_t{4}, std::size_t{17}}) {
    {
      constexpr std::size_t embedding_dim = 7;
      constexpr std::size_t vocab_size = 23;
      const std::vector<std::int32_t> tokens = [&] {
        std::vector<std::int32_t> ids(rows);
        for (std::size_t i = 0; i < rows; ++i) ids[i] = static_cast<int>(i % 5);
        return ids;
      }();
      const auto table_f32 = sequence(vocab_size * embedding_dim, 0.031F, -0.2F);
      const auto table = encode<T>(table_f32);
      const auto device_tokens = upload(context, std::span{tokens});
      const auto device_table = upload(context, std::span{table});
      DeviceBuffer device_output{context, rows * embedding_dim * sizeof(T)};
      brt::kernels::qwen35_embedding(
          static_cast<const std::int32_t*>(device_tokens.data()),
          device_table.data(), device_output.data(),
          brt::kernels::EmbeddingShape{rows, embedding_dim, vocab_size},
          DTypeTraits<T>::dtype, context.stream());
      const auto actual = download<T>(context, device_output.data(),
                                      rows * embedding_dim);
      std::vector<float> expected(rows * embedding_dim);
      brt::reference::embedding(std::span{tokens}, table_f32, expected,
                                brt::reference::EmbeddingShape{
                                    rows, embedding_dim, vocab_size});
      assert_matches<T>(actual, expected);
    }

    {
      constexpr std::size_t cols = 31;
      const auto input_f32 = sequence(rows * cols, 0.047F, 0.1F);
      const std::vector<float> zero_weight_f32(cols, 0.0F);
      const auto input = encode<T>(input_f32);
      const auto weight = encode<T>(zero_weight_f32);
      const auto device_input = upload(context, std::span{input});
      const auto device_weight = upload(context, std::span{weight});
      DeviceBuffer device_output{context, input.size() * sizeof(T)};
      brt::kernels::qwen35_rms_norm(
          device_input.data(), device_weight.data(), device_output.data(),
          brt::kernels::RmsNormShape{rows, cols}, 1.0e-5F,
          DTypeTraits<T>::dtype, context.stream());
      const auto actual = download<T>(context, device_output.data(), input.size());
      std::vector<float> expected(input_f32.size());
      brt::reference::qwen35_rms_norm(input_f32, zero_weight_f32, expected,
                                      rows, cols, 1.0e-5F);
      assert_matches<T>(actual, expected);
    }

    {
      constexpr std::size_t cols = 29;
      const auto lhs_f32 = sequence(rows * cols, 0.021F, 0.5F);
      const auto rhs_f32 = sequence(rows * cols, -0.017F, 0.25F);
      const auto lhs = encode<T>(lhs_f32);
      const auto rhs = encode<T>(rhs_f32);
      const auto device_lhs = upload(context, std::span{lhs});
      const auto device_rhs = upload(context, std::span{rhs});
      DeviceBuffer device_output{context, lhs.size() * sizeof(T)};
      brt::kernels::qwen35_residual_add(device_lhs.data(), device_rhs.data(),
                                        device_output.data(), lhs.size(),
                                        DTypeTraits<T>::dtype,
                                        context.stream());
      const auto actual = download<T>(context, device_output.data(), lhs.size());
      std::vector<float> expected(lhs_f32.size());
      brt::reference::add(lhs_f32, rhs_f32, expected,
                          brt::reference::AddShape{lhs_f32.size()});
      assert_matches<T>(actual, expected);
    }

    {
      constexpr std::size_t heads = 3;
      constexpr std::size_t head_dim = 13;
      constexpr std::size_t rotary_dim = 8;
      const auto input_f32 = sequence(rows * heads * head_dim, 0.019F, -0.4F);
      const auto weight_f32 = sequence(head_dim, 0.005F, 0.0F);
      const auto input = encode<T>(input_f32);
      const auto weight = encode<T>(weight_f32);
      const auto device_input = upload(context, std::span{input});
      const auto device_weight = upload(context, std::span{weight});
      DeviceBuffer device_output{context, input.size() * sizeof(T)};
      const brt::kernels::QkNormRopeShape shape{
          rows, heads, head_dim, rotary_dim, 5, 10000.0F};
      brt::kernels::qwen35_qk_norm_rope(
          device_input.data(), device_weight.data(), device_output.data(),
          shape, 1.0e-5F, DTypeTraits<T>::dtype, context.stream());
      const auto actual = download<T>(context, device_output.data(), input.size());
      std::vector<float> expected(input_f32.size());
      apply_qk_norm_rope_reference(input_f32, weight_f32, expected, shape,
                                   1.0e-5F);
      assert_matches<T>(actual, expected);
    }

    {
      constexpr std::size_t cols = 19;
      const auto values_f32 = sequence(rows * cols, 0.027F, 0.3F);
      const auto gates_f32 = sequence(rows * cols, -0.023F, -0.1F);
      const auto values = encode<T>(values_f32);
      const auto gates = encode<T>(gates_f32);
      const auto device_values = upload(context, std::span{values});
      const auto device_gates = upload(context, std::span{gates});
      DeviceBuffer device_output{context, values.size() * sizeof(T)};
      brt::kernels::qwen35_sigmoid_gate(
          device_values.data(), device_gates.data(), device_output.data(),
          values.size(), DTypeTraits<T>::dtype, context.stream());
      const auto actual =
          download<T>(context, device_output.data(), values.size());
      std::vector<float> expected(values.size());
      for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = values_f32[i] / (1.0F + std::exp(-gates_f32[i]));
      }
      assert_matches<T>(actual, expected);
    }

    {
      constexpr std::size_t cols = 23;
      const auto gate_f32 = sequence(rows * cols, 0.029F, -0.3F);
      const auto up_f32 = sequence(rows * cols, -0.011F, 0.7F);
      const auto gate = encode<T>(gate_f32);
      const auto up = encode<T>(up_f32);
      const auto device_gate = upload(context, std::span{gate});
      const auto device_up = upload(context, std::span{up});
      DeviceBuffer device_output{context, gate.size() * sizeof(T)};
      brt::kernels::qwen35_swiglu(device_gate.data(), device_up.data(),
                                  device_output.data(), gate.size(),
                                  DTypeTraits<T>::dtype, context.stream());
      const auto actual = download<T>(context, device_output.data(), gate.size());
      std::vector<float> expected(gate.size());
      brt::reference::swiglu(gate_f32, up_f32, expected,
                             brt::reference::SwiGluShape{gate.size()});
      assert_matches<T>(actual, expected);
    }
  }
}

void run_argmax_test(brt::ExecutionContext& context) {
  const std::vector<float> logits{
      -1.0F, 3.0F, -7.0F, 4.5F, 4.5F, 2.0F, -0.0F, -0.0F};
  const auto device_logits = upload(context, std::span{logits});
  DeviceBuffer device_index{context, sizeof(std::int32_t)};
  brt::kernels::qwen35_argmax(static_cast<const float*>(device_logits.data()),
                              static_cast<std::int32_t*>(device_index.data()),
                              logits.size(), context.stream());
  const auto actual =
      download<std::int32_t>(context, device_index.data(), std::size_t{1});
  assert(actual[0] == 3);
}

void run_token_validator_tests() {
  const std::vector<std::int32_t> valid{0, 2, 4};
  brt::kernels::qwen35_validate_token_ids(valid, 5);
  expect_primitive_error(
      [&] { brt::kernels::qwen35_validate_token_ids(
                 std::span<const std::int32_t>{}, 5); });
  expect_primitive_error(
      [&] { brt::kernels::qwen35_validate_token_ids(valid, 0); });
  expect_primitive_error([&] {
    const std::vector<std::int32_t> negative{0, -1, 2};
    brt::kernels::qwen35_validate_token_ids(negative, 5);
  });
  expect_primitive_error([&] {
    const std::vector<std::int32_t> out_of_range{0, 5};
    brt::kernels::qwen35_validate_token_ids(out_of_range, 5);
  });
}

void run_embedding_invalid_device_id_zeroes_output(
    brt::ExecutionContext& context) {
  const std::vector<std::int32_t> tokens{-1, 3};
  const auto table_f32 = sequence(3 * 4, 0.031F, -0.2F);
  const auto table = encode<__half>(table_f32);
  const auto device_tokens = upload(context, std::span{tokens});
  const auto device_table = upload(context, std::span{table});
  DeviceBuffer device_output{context, tokens.size() * 4 * sizeof(__half)};
  brt::kernels::qwen35_embedding(
      static_cast<const std::int32_t*>(device_tokens.data()),
      device_table.data(), device_output.data(),
      brt::kernels::EmbeddingShape{tokens.size(), 4, 3}, BRT_DTYPE_F16,
      context.stream());
  const auto actual =
      download<__half>(context, device_output.data(), tokens.size() * 4);
  for (const __half value : actual) {
    assert(DTypeTraits<__half>::to_float(value) == 0.0F);
  }
}

void run_invalid_shape_tests(brt::ExecutionContext& context) {
  auto* pointer = reinterpret_cast<void*>(0x1000);
  auto* const_pointer = reinterpret_cast<const void*>(0x1000);
  auto* tokens = reinterpret_cast<const std::int32_t*>(0x1000);
  auto* index = reinterpret_cast<std::int32_t*>(0x1000);
  auto* logits = reinterpret_cast<const float*>(0x1000);

  expect_primitive_error([&] {
    brt::kernels::qwen35_embedding(
        nullptr, const_pointer, pointer,
        brt::kernels::EmbeddingShape{1, 1, 1},
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_embedding(
        tokens, const_pointer, pointer,
        brt::kernels::EmbeddingShape{0, 1, 1},
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_embedding(
        tokens, const_pointer, pointer,
        brt::kernels::EmbeddingShape{
            1, 2,
            std::numeric_limits<std::size_t>::max() / std::size_t{2} + 1},
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_rms_norm(
        const_pointer, const_pointer, pointer, brt::kernels::RmsNormShape{1, 0},
        1.0e-5F, BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_rms_norm(
        const_pointer, const_pointer, pointer, brt::kernels::RmsNormShape{1, 1},
        -1.0F, BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_rms_norm(
        const_pointer, const_pointer, pointer, brt::kernels::RmsNormShape{1, 1},
        std::numeric_limits<float>::infinity(), BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_rms_norm(
        const_pointer, const_pointer, pointer, brt::kernels::RmsNormShape{1, 1},
        std::numeric_limits<float>::quiet_NaN(), BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{1, 1, 7, 9, 0, 10000.0F}, 1.0e-5F,
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{1, 1, 8, 5, 0, 10000.0F}, 1.0e-5F,
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{1, 1, 8, 0, 0, 10000.0F}, 1.0e-5F,
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{
            1, 1, 8, 4, 0, std::numeric_limits<float>::infinity()},
        1.0e-5F, BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{
            1, 1, 8, 4, 0, std::numeric_limits<float>::quiet_NaN()},
        1.0e-5F, BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{1, 1, 8, 4, 0, 10000.0F},
        std::numeric_limits<float>::infinity(), BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{1, 1, 8, 4, 0, 10000.0F},
        std::numeric_limits<float>::quiet_NaN(), BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_qk_norm_rope(
        const_pointer, const_pointer, pointer,
        brt::kernels::QkNormRopeShape{
            2, 1, 8, 4, std::numeric_limits<std::size_t>::max(), 10000.0F},
        1.0e-5F, BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_residual_add(
        const_pointer, const_pointer, pointer, 0, BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_residual_add(
        const_pointer, const_pointer, pointer,
        (static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1) *
            std::size_t{256},
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_swiglu(const_pointer, const_pointer, pointer, 1,
                                BRT_DTYPE_F32, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_argmax(logits, index, 0, context.stream());
  });
}

}  // namespace

int main() {
  assert(cudaSetDevice(0) == cudaSuccess);
  TestResources resources{64 * 1024 * 1024};
  auto& context = resources.context();

  run_dtype_tests<__half>(context);
  run_dtype_tests<__nv_bfloat16>(context);
  run_argmax_test(context);
  run_token_validator_tests();
  run_embedding_invalid_device_id_zeroes_output(context);
  run_invalid_shape_tests(context);
}
