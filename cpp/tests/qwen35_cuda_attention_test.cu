#include "../execution/execution_context.hpp"
#include "../execution/workspace_arena.hpp"
#include "../kernels/qwen35_attention.cuh"
#include "../kernels/qwen35_primitives.cuh"
#include "../reference/qwen35.hpp"

#include "assert_enabled.hpp"

#include <cuda/stream_ref>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
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
        rmm::device_async_resource_ref{pool_}, cuda::stream_ref{stream}, stream,
        8 * 1024 * 1024);
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

  brt::ExecutionContext &context() noexcept { return *context_; }

private:
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
  raft::device_resources resources_;
  std::unique_ptr<brt::WorkspaceArena> workspace_;
  std::unique_ptr<brt::ExecutionContext> context_;
};

class DeviceBuffer {
public:
  DeviceBuffer(brt::ExecutionContext &context, std::size_t bytes)
      : resource_(context.memory_resource()), stream_ref_(context.stream()),
        stream_(context.stream()), bytes_(bytes == 0 ? 1 : bytes),
        data_(resource_.allocate(stream_ref_, bytes_,
                                 rmm::CUDA_ALLOCATION_ALIGNMENT)) {}

  ~DeviceBuffer() noexcept {
    if (data_ == nullptr)
      return;
    (void)cudaStreamSynchronize(stream_);
    try {
      resource_.deallocate(stream_ref_, data_, bytes_,
                           rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (...) {
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : resource_(other.resource_), stream_ref_(other.stream_ref_),
        stream_(other.stream_), bytes_(std::exchange(other.bytes_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept = delete;

  void *data() const noexcept { return data_; }

private:
  rmm::device_async_resource_ref resource_;
  cuda::stream_ref stream_ref_;
  cudaStream_t stream_{};
  std::size_t bytes_{};
  void *data_{};
};

template <typename T> struct DTypeTraits;

template <> struct DTypeTraits<__half> {
  static constexpr BrtDataType dtype = BRT_DTYPE_F16;
  static __half from_float(float value) { return __float2half_rn(value); }
  static float to_float(__half value) { return __half2float(value); }
};

template <> struct DTypeTraits<__nv_bfloat16> {
  static constexpr BrtDataType dtype = BRT_DTYPE_BF16;
  static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
  static float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
};

template <> struct DTypeTraits<float> {
  static constexpr BrtDataType dtype = BRT_DTYPE_F32;
  static float from_float(float value) { return value; }
  static float to_float(float value) { return value; }
};

template <typename T> std::vector<T> encode(std::span<const float> values) {
  std::vector<T> encoded(values.size());
  std::transform(values.begin(), values.end(), encoded.begin(),
                 DTypeTraits<T>::from_float);
  return encoded;
}

template <typename T>
DeviceBuffer upload(brt::ExecutionContext &context, std::span<const T> host) {
  DeviceBuffer device{context, host.size_bytes()};
  assert(cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                         cudaMemcpyHostToDevice,
                         context.stream()) == cudaSuccess);
  return device;
}

template <typename T>
std::vector<T> download(brt::ExecutionContext &context, const void *device,
                        std::size_t elements) {
  std::vector<T> host(elements);
  assert(cudaMemcpyAsync(host.data(), device, elements * sizeof(T),
                         cudaMemcpyDeviceToHost,
                         context.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(context.stream()) == cudaSuccess);
  return host;
}

std::vector<float> sequence(std::size_t elements, float scale, float bias) {
  std::vector<float> values(elements);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] =
        bias + scale * static_cast<float>(static_cast<int>((i * 13) % 19) - 9);
  }
  return values;
}

bool close_enough(float actual, float expected) {
  const float abs = std::fabs(actual - expected);
  const float rel = abs / std::max(std::fabs(expected), 1.0e-6F);
  return abs <= 2.0e-2F || rel <= 2.0e-2F;
}

void expect_primitive_error(auto &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const brt::kernels::Qwen35PrimitiveError &) {
    thrown = true;
  }
  assert(thrown);
}

void apply_qk_norm_rope_reference(std::span<const float> input,
                                  std::span<const float> weight,
                                  std::span<float> output,
                                  brt::kernels::QkNormRopeShape shape,
                                  float epsilon) {
  for (std::size_t token = 0; token < shape.tokens; ++token) {
    for (std::size_t head = 0; head < shape.heads; ++head) {
      const std::size_t base = (token * shape.heads + head) * shape.head_dim;
      double square_sum = 0.0;
      for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
        const float value = input[base + dim];
        square_sum += static_cast<double>(value) * value;
      }
      const float scale =
          1.0F /
          std::sqrt(static_cast<float>(square_sum /
                                       static_cast<double>(shape.head_dim)) +
                    epsilon);
      for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
        output[base + dim] = input[base + dim] * scale * weight[dim];
      }
      const std::size_t pair_count = shape.rotary_dim / 2;
      for (std::size_t pair = 0; pair < pair_count; ++pair) {
        const double exponent = static_cast<double>(2 * pair) /
                                static_cast<double>(shape.rotary_dim);
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

std::vector<float> identity_matrix(std::size_t size) {
  std::vector<float> values(size * size, 0.0F);
  for (std::size_t i = 0; i < size; ++i) {
    values[i * size + i] = 1.0F;
  }
  return values;
}

void expect_matches_reference(std::span<const float> actual,
                              std::span<const float> expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    assert(close_enough(actual[i], expected[i]));
  }
}

template <typename T>
void run_prefill_case(brt::ExecutionContext &context, std::size_t tokens,
                      std::size_t query_heads, std::size_t kv_heads,
                      std::size_t head_dim) {
  const std::size_t hidden_size = query_heads * head_dim;
  const std::size_t q_size = hidden_size;
  const std::size_t kv_size = kv_heads * head_dim;
  const std::size_t stride = q_size + 2 * kv_size + hidden_size;
  const std::size_t rotary_dim = head_dim > 1 ? head_dim - (head_dim % 2) : 0;
  const auto q_raw = sequence(tokens * q_size, 0.031F, -0.2F);
  const auto k_raw = sequence(tokens * kv_size, -0.027F, 0.4F);
  const auto v_f32 = sequence(tokens * kv_size, 0.019F, -0.1F);
  auto gate_f32 = sequence(tokens * hidden_size, -0.023F, -0.7F);
  const std::vector<float> q_weight(head_dim, 1.0F);
  const std::vector<float> k_weight(head_dim, 1.0F);

  std::vector<float> q_norm(q_raw.size());
  std::vector<float> k_norm(k_raw.size());
  apply_qk_norm_rope_reference(
      q_raw, q_weight, q_norm,
      brt::kernels::QkNormRopeShape{tokens, query_heads, head_dim, rotary_dim,
                                    3, 10000.0F},
      1.0e-5F);
  apply_qk_norm_rope_reference(
      k_raw, k_weight, k_norm,
      brt::kernels::QkNormRopeShape{tokens, kv_heads, head_dim, rotary_dim, 3,
                                    10000.0F},
      1.0e-5F);

  std::vector<float> reference_input(tokens * stride);
  for (std::size_t token = 0; token < tokens; ++token) {
    std::copy_n(
        q_raw.begin() + static_cast<std::ptrdiff_t>(token * q_size), q_size,
        reference_input.begin() + static_cast<std::ptrdiff_t>(token * stride));
    std::copy_n(k_raw.begin() + static_cast<std::ptrdiff_t>(token * kv_size),
                kv_size,
                reference_input.begin() +
                    static_cast<std::ptrdiff_t>(token * stride + q_size));
    std::copy_n(
        v_f32.begin() + static_cast<std::ptrdiff_t>(token * kv_size), kv_size,
        reference_input.begin() +
            static_cast<std::ptrdiff_t>(token * stride + q_size + kv_size));
    std::copy_n(
        gate_f32.begin() + static_cast<std::ptrdiff_t>(token * hidden_size),
        hidden_size,
        reference_input.begin() +
            static_cast<std::ptrdiff_t>(token * stride + q_size + 2 * kv_size));
  }

  std::vector<float> expected(tokens * hidden_size);
  brt::reference::qwen35_gated_full_attention(
      reference_input, expected,
      brt::reference::FullAttentionReferenceWeights{
          .query_norm_weight = q_weight,
          .key_norm_weight = k_weight,
          .output_weight = identity_matrix(hidden_size)},
      brt::reference::FullAttentionReferenceArgs{.tokens = tokens,
                                                 .hidden_size = hidden_size,
                                                 .query_heads = query_heads,
                                                 .kv_heads = kv_heads,
                                                 .head_dim = head_dim,
                                                 .rotary_dim = rotary_dim,
                                                 .position_offset = 3,
                                                 .rope_base = 10000.0F,
                                                 .epsilon = 1.0e-5F});

  const auto q = encode<T>(q_norm);
  const auto k = encode<T>(k_norm);
  const auto q_raw_encoded = encode<T>(q_raw);
  const auto k_raw_encoded = encode<T>(k_raw);
  const auto q_weight_encoded = encode<T>(q_weight);
  const auto k_weight_encoded = encode<T>(k_weight);
  const auto v = encode<T>(v_f32);
  const auto gate = encode<T>(gate_f32);
  const auto device_q_raw = upload(context, std::span{q_raw_encoded});
  const auto device_k_raw = upload(context, std::span{k_raw_encoded});
  const auto device_q_weight = upload(context, std::span{q_weight_encoded});
  const auto device_k_weight = upload(context, std::span{k_weight_encoded});
  const auto device_v = upload(context, std::span{v});
  const auto device_gate = upload(context, std::span{gate});
  DeviceBuffer device_q{context, q.size() * sizeof(T)};
  DeviceBuffer device_k{context, k.size() * sizeof(T)};
  DeviceBuffer device_cache{context,
                            2 * tokens * kv_heads * head_dim * sizeof(float)};
  DeviceBuffer device_logits{
      context, brt::kernels::qwen35_attention_workspace_bytes(
                   brt::kernels::Qwen35AttentionShape{
                       tokens, query_heads, kv_heads, head_dim, tokens, 0})};
  DeviceBuffer device_output{context, tokens * hidden_size * sizeof(T)};

  brt::kernels::qwen35_qk_norm_rope(
      device_q_raw.data(), device_q_weight.data(), device_q.data(),
      brt::kernels::QkNormRopeShape{tokens, query_heads, head_dim, rotary_dim,
                                    3, 10000.0F},
      1.0e-5F, DTypeTraits<T>::dtype, DTypeTraits<T>::dtype, context.stream());
  brt::kernels::qwen35_qk_norm_rope(
      device_k_raw.data(), device_k_weight.data(), device_k.data(),
      brt::kernels::QkNormRopeShape{tokens, kv_heads, head_dim, rotary_dim, 3,
                                    10000.0F},
      1.0e-5F, DTypeTraits<T>::dtype, DTypeTraits<T>::dtype, context.stream());
  brt::kernels::qwen35_causal_attention(
      device_q.data(), device_k.data(), device_v.data(), device_gate.data(),
      device_output.data(), static_cast<float *>(device_cache.data()),
      static_cast<float *>(device_logits.data()),
      brt::kernels::qwen35_attention_workspace_floats(
          brt::kernels::Qwen35AttentionShape{tokens, query_heads, kv_heads,
                                             head_dim, tokens, 0}),
      brt::kernels::Qwen35AttentionShape{tokens, query_heads, kv_heads,
                                         head_dim, tokens, 0},
      DTypeTraits<T>::dtype, context.stream());

  const auto actual_encoded =
      download<T>(context, device_output.data(), tokens * hidden_size);
  std::vector<float> actual(actual_encoded.size());
  std::transform(actual_encoded.begin(), actual_encoded.end(), actual.begin(),
                 DTypeTraits<T>::to_float);
  expect_matches_reference(actual, expected);

  const auto cache = download<float>(context, device_cache.data(),
                                     2 * tokens * kv_heads * head_dim);
  for (std::size_t i = 0; i < k_norm.size(); ++i) {
    assert(close_enough(cache[i], k_norm[i]));
  }
  for (std::size_t i = 0; i < v_f32.size(); ++i) {
    assert(close_enough(cache[tokens * kv_heads * head_dim + i], v_f32[i]));
  }
}

template <typename T>
void run_prefill_dtype_cases(brt::ExecutionContext &context) {
  run_prefill_case<T>(context, 1, 2, 1, 7);
  run_prefill_case<T>(context, 2, 4, 2, 7);
  run_prefill_case<T>(context, 4, 4, 1, 7);
  run_prefill_case<T>(context, 17, 4, 2, 7);
}

template <typename T>
void run_decode_after_prefill_case(brt::ExecutionContext &context,
                                   std::size_t decode_tokens) {
  constexpr std::size_t prefill_tokens = 4;
  constexpr std::size_t query_heads = 4;
  constexpr std::size_t kv_heads = 2;
  constexpr std::size_t head_dim = 7;
  constexpr std::size_t hidden_size = query_heads * head_dim;
  constexpr std::size_t q_size = hidden_size;
  constexpr std::size_t kv_size = kv_heads * head_dim;
  constexpr std::size_t stride = q_size + 2 * kv_size + hidden_size;
  constexpr std::size_t rotary_dim = 6;
  const std::size_t total_tokens = prefill_tokens + decode_tokens;
  const auto q_raw = sequence(total_tokens * q_size, 0.017F, 0.1F);
  const auto k_raw = sequence(total_tokens * kv_size, -0.021F, 0.2F);
  const auto v_f32 = sequence(total_tokens * kv_size, 0.029F, -0.3F);
  const auto gate_f32 = sequence(total_tokens * hidden_size, -0.025F, -1.1F);
  const std::vector<float> unit_weight(head_dim, 1.0F);
  std::vector<float> q_norm(q_raw.size());
  std::vector<float> k_norm(k_raw.size());
  apply_qk_norm_rope_reference(
      q_raw, unit_weight, q_norm,
      brt::kernels::QkNormRopeShape{total_tokens, query_heads, head_dim,
                                    rotary_dim, 0, 10000.0F},
      1.0e-5F);
  apply_qk_norm_rope_reference(
      k_raw, unit_weight, k_norm,
      brt::kernels::QkNormRopeShape{total_tokens, kv_heads, head_dim,
                                    rotary_dim, 0, 10000.0F},
      1.0e-5F);

  std::vector<float> reference_input(total_tokens * stride);
  for (std::size_t token = 0; token < total_tokens; ++token) {
    std::copy_n(
        q_raw.begin() + static_cast<std::ptrdiff_t>(token * q_size), q_size,
        reference_input.begin() + static_cast<std::ptrdiff_t>(token * stride));
    std::copy_n(k_raw.begin() + static_cast<std::ptrdiff_t>(token * kv_size),
                kv_size,
                reference_input.begin() +
                    static_cast<std::ptrdiff_t>(token * stride + q_size));
    std::copy_n(
        v_f32.begin() + static_cast<std::ptrdiff_t>(token * kv_size), kv_size,
        reference_input.begin() +
            static_cast<std::ptrdiff_t>(token * stride + q_size + kv_size));
    std::copy_n(
        gate_f32.begin() + static_cast<std::ptrdiff_t>(token * hidden_size),
        hidden_size,
        reference_input.begin() +
            static_cast<std::ptrdiff_t>(token * stride + q_size + 2 * kv_size));
  }
  std::vector<float> expected_all(total_tokens * hidden_size);
  brt::reference::qwen35_gated_full_attention(
      reference_input, expected_all,
      brt::reference::FullAttentionReferenceWeights{
          .query_norm_weight = unit_weight,
          .key_norm_weight = unit_weight,
          .output_weight = identity_matrix(hidden_size)},
      brt::reference::FullAttentionReferenceArgs{.tokens = total_tokens,
                                                 .hidden_size = hidden_size,
                                                 .query_heads = query_heads,
                                                 .kv_heads = kv_heads,
                                                 .head_dim = head_dim,
                                                 .rotary_dim = rotary_dim,
                                                 .position_offset = 0,
                                                 .rope_base = 10000.0F,
                                                 .epsilon = 1.0e-5F});

  DeviceBuffer device_cache{context, 2 * total_tokens * kv_heads * head_dim *
                                         sizeof(float)};
  DeviceBuffer prefill_logits{context,
                              brt::kernels::qwen35_attention_workspace_bytes(
                                  brt::kernels::Qwen35AttentionShape{
                                      prefill_tokens, query_heads, kv_heads,
                                      head_dim, total_tokens, 0})};
  DeviceBuffer decode_logits{context,
                             brt::kernels::qwen35_attention_workspace_bytes(
                                 brt::kernels::Qwen35AttentionShape{
                                     decode_tokens, query_heads, kv_heads,
                                     head_dim, total_tokens, prefill_tokens})};
  DeviceBuffer prefill_output{context,
                              prefill_tokens * hidden_size * sizeof(T)};
  DeviceBuffer decode_output{context, decode_tokens * hidden_size * sizeof(T)};

  const auto q = encode<T>(q_norm);
  const auto k = encode<T>(k_norm);
  const auto q_raw_encoded = encode<T>(q_raw);
  const auto k_raw_encoded = encode<T>(k_raw);
  const auto unit_weight_encoded = encode<T>(unit_weight);
  const auto v = encode<T>(v_f32);
  const auto gate = encode<T>(gate_f32);
  const auto device_q_raw = upload(context, std::span{q_raw_encoded});
  const auto device_k_raw = upload(context, std::span{k_raw_encoded});
  const auto device_weight = upload(context, std::span{unit_weight_encoded});
  const auto device_v = upload(context, std::span{v});
  const auto device_gate = upload(context, std::span{gate});
  DeviceBuffer device_q{context, q.size() * sizeof(T)};
  DeviceBuffer device_k{context, k.size() * sizeof(T)};

  brt::kernels::qwen35_qk_norm_rope(
      device_q_raw.data(), device_weight.data(), device_q.data(),
      brt::kernels::QkNormRopeShape{total_tokens, query_heads, head_dim,
                                    rotary_dim, 0, 10000.0F},
      1.0e-5F, DTypeTraits<T>::dtype, DTypeTraits<T>::dtype, context.stream());
  brt::kernels::qwen35_qk_norm_rope(
      device_k_raw.data(), device_weight.data(), device_k.data(),
      brt::kernels::QkNormRopeShape{total_tokens, kv_heads, head_dim,
                                    rotary_dim, 0, 10000.0F},
      1.0e-5F, DTypeTraits<T>::dtype, DTypeTraits<T>::dtype, context.stream());

  brt::kernels::qwen35_causal_attention(
      device_q.data(), device_k.data(), device_v.data(), device_gate.data(),
      prefill_output.data(), static_cast<float *>(device_cache.data()),
      static_cast<float *>(prefill_logits.data()),
      brt::kernels::qwen35_attention_workspace_floats(
          brt::kernels::Qwen35AttentionShape{prefill_tokens, query_heads,
                                             kv_heads, head_dim, total_tokens,
                                             0}),
      brt::kernels::Qwen35AttentionShape{prefill_tokens, query_heads, kv_heads,
                                         head_dim, total_tokens, 0},
      DTypeTraits<T>::dtype, context.stream());

  const std::size_t q_offset = prefill_tokens * q_size;
  const std::size_t kv_offset = prefill_tokens * kv_size;
  const std::size_t hidden_offset = prefill_tokens * hidden_size;
  brt::kernels::qwen35_causal_attention(
      static_cast<const T *>(device_q.data()) + q_offset,
      static_cast<const T *>(device_k.data()) + kv_offset,
      static_cast<const T *>(device_v.data()) + kv_offset,
      static_cast<const T *>(device_gate.data()) + hidden_offset,
      decode_output.data(), static_cast<float *>(device_cache.data()),
      static_cast<float *>(decode_logits.data()),
      brt::kernels::qwen35_attention_workspace_floats(
          brt::kernels::Qwen35AttentionShape{decode_tokens, query_heads,
                                             kv_heads, head_dim, total_tokens,
                                             prefill_tokens}),
      brt::kernels::Qwen35AttentionShape{decode_tokens, query_heads, kv_heads,
                                         head_dim, total_tokens,
                                         prefill_tokens},
      DTypeTraits<T>::dtype, context.stream());

  const auto actual_encoded =
      download<T>(context, decode_output.data(), decode_tokens * hidden_size);
  std::vector<float> actual(actual_encoded.size());
  std::transform(actual_encoded.begin(), actual_encoded.end(), actual.begin(),
                 DTypeTraits<T>::to_float);
  expect_matches_reference(actual, std::span<const float>{expected_all}.subspan(
                                       prefill_tokens * hidden_size));

  const auto cache = download<float>(context, device_cache.data(),
                                     2 * total_tokens * kv_heads * head_dim);
  for (std::size_t i = 0; i < k_norm.size(); ++i) {
    assert(close_enough(cache[i], k_norm[i]));
  }
  for (std::size_t i = 0; i < v_f32.size(); ++i) {
    assert(
        close_enough(cache[total_tokens * kv_heads * head_dim + i], v_f32[i]));
  }
}

void run_invalid_shape_tests(brt::ExecutionContext &context) {
  auto *pointer = reinterpret_cast<void *>(0x1000);
  auto *const_pointer = reinterpret_cast<const void *>(0x1000);
  auto *floats = reinterpret_cast<float *>(0x1000);
  const brt::kernels::Qwen35AttentionShape valid{1, 2, 1, 7, 4, 0};
  expect_primitive_error([&] {
    brt::kernels::qwen35_causal_attention(
        nullptr, const_pointer, const_pointer, const_pointer, pointer, floats,
        floats, brt::kernels::qwen35_attention_workspace_floats(valid), valid,
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_causal_attention(
        const_pointer, const_pointer, const_pointer, const_pointer, pointer,
        floats, floats, brt::kernels::qwen35_attention_workspace_floats(valid),
        brt::kernels::Qwen35AttentionShape{0, 2, 1, 7, 4, 0}, BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_causal_attention(
        const_pointer, const_pointer, const_pointer, const_pointer, pointer,
        floats, floats, brt::kernels::qwen35_attention_workspace_floats(valid),
        brt::kernels::Qwen35AttentionShape{1, 3, 2, 7, 4, 0}, BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_causal_attention(
        const_pointer, const_pointer, const_pointer, const_pointer, pointer,
        floats, floats, brt::kernels::qwen35_attention_workspace_floats(valid),
        brt::kernels::Qwen35AttentionShape{2, 2, 1, 7, 2, 1}, BRT_DTYPE_F16,
        context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_causal_attention(
        const_pointer, const_pointer, const_pointer, const_pointer, pointer,
        floats, floats, brt::kernels::qwen35_attention_workspace_floats(valid),
        valid, BRT_DTYPE_Q4_K, context.stream());
  });
  expect_primitive_error([&] {
    brt::kernels::qwen35_causal_attention(
        const_pointer, const_pointer, const_pointer, const_pointer, pointer,
        floats, floats,
        brt::kernels::qwen35_attention_workspace_floats(valid) - 1, valid,
        BRT_DTYPE_F16, context.stream());
  });
  expect_primitive_error([&] {
    (void)brt::kernels::qwen35_attention_workspace_bytes(
        brt::kernels::Qwen35AttentionShape{
            2, 2, 1, 7, std::numeric_limits<std::size_t>::max(), 0});
  });
}

} // namespace

int main() {
  assert(cudaSetDevice(0) == cudaSuccess);
  TestResources resources{128 * 1024 * 1024};
  auto &context = resources.context();

  run_prefill_dtype_cases<__half>(context);
  run_prefill_dtype_cases<__nv_bfloat16>(context);
  run_prefill_dtype_cases<float>(context);
  run_decode_after_prefill_case<__half>(context, 1);
  run_decode_after_prefill_case<__half>(context, 2);
  run_decode_after_prefill_case<__nv_bfloat16>(context, 1);
  run_decode_after_prefill_case<__nv_bfloat16>(context, 2);
  run_decode_after_prefill_case<float>(context, 1);
  run_decode_after_prefill_case<float>(context, 2);
  run_invalid_shape_tests(context);
}
