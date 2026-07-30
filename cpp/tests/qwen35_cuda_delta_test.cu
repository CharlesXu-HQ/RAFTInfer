#include "../execution/execution_context.hpp"
#include "../execution/workspace_arena.hpp"
#include "../kernels/qwen35_delta.cuh"
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
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
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
    workspace_ = std::make_unique<raftinfer::WorkspaceArena>(
        rmm::device_async_resource_ref{pool_}, cuda::stream_ref{stream}, stream,
        4 * 1024 * 1024);
    context_ = std::make_unique<raftinfer::ExecutionContext>(
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

  raftinfer::ExecutionContext &context() noexcept { return *context_; }

private:
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
  raft::device_resources resources_;
  std::unique_ptr<raftinfer::WorkspaceArena> workspace_;
  std::unique_ptr<raftinfer::ExecutionContext> context_;
};

class DeviceBuffer {
public:
  DeviceBuffer(raftinfer::ExecutionContext &context, std::size_t bytes)
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
  DeviceBuffer &operator=(DeviceBuffer &&) = delete;

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
  static constexpr RaftInferDataType dtype = RAFTINFER_DTYPE_F16;
  static __half from_float(float value) { return __float2half_rn(value); }
  static float to_float(__half value) { return __half2float(value); }
};

template <> struct DTypeTraits<__nv_bfloat16> {
  static constexpr RaftInferDataType dtype = RAFTINFER_DTYPE_BF16;
  static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
  static float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
};

template <> struct DTypeTraits<float> {
  static constexpr RaftInferDataType dtype = RAFTINFER_DTYPE_F32;
  static float from_float(float value) { return value; }
  static float to_float(float value) { return value; }
};

template <typename T> std::vector<T> encode(std::span<const float> values) {
  std::vector<T> encoded(values.size());
  std::transform(values.begin(), values.end(), encoded.begin(),
                 DTypeTraits<T>::from_float);
  return encoded;
}

template <typename T> std::vector<float> decode(std::span<const T> values) {
  std::vector<float> decoded(values.size());
  std::transform(values.begin(), values.end(), decoded.begin(),
                 DTypeTraits<T>::to_float);
  return decoded;
}

template <typename T>
DeviceBuffer upload(raftinfer::ExecutionContext &context, std::span<const T> host) {
  DeviceBuffer buffer{context, host.size_bytes()};
  assert(cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                         cudaMemcpyHostToDevice,
                         context.stream()) == cudaSuccess);
  return buffer;
}

template <typename T>
std::vector<T> download(raftinfer::ExecutionContext &context, const void *device,
                        std::size_t elements) {
  std::vector<T> host(elements);
  assert(cudaMemcpyAsync(host.data(), device, elements * sizeof(T),
                         cudaMemcpyDeviceToHost,
                         context.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(context.stream()) == cudaSuccess);
  return host;
}

bool close_enough(float actual, float expected) {
  const float absolute = std::fabs(actual - expected);
  const float relative = absolute / std::max(std::fabs(expected), 1.0e-6F);
  return absolute <= 2.0e-2F || relative <= 2.0e-2F;
}

void assert_matches(std::span<const float> actual,
                    std::span<const float> expected) {
  assert(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    assert(close_enough(actual[index], expected[index]));
  }
}

void assert_finite(std::span<const float> values) {
  for (const float value : values) {
    assert(std::isfinite(value));
  }
}

std::vector<float> sequence(std::size_t elements, float scale, float bias) {
  std::vector<float> values(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    values[index] =
        bias +
        scale * static_cast<float>(static_cast<int>((index * 17) % 29) - 14);
  }
  return values;
}

raftinfer::kernels::GatedDeltaShape make_shape(std::size_t tokens) {
  return raftinfer::kernels::GatedDeltaShape{
      .tokens = tokens,
      .hidden_size = 28,
      .key_heads = 2,
      .value_heads = 4,
      .key_dim = 5,
      .value_dim = 7,
      .conv_width = 3,
      .epsilon = 1.0e-6F,
  };
}

raftinfer::reference::GatedDeltaReferenceArgs
make_reference_args(const raftinfer::kernels::GatedDeltaShape &shape) {
  return raftinfer::reference::GatedDeltaReferenceArgs{
      .tokens = shape.tokens,
      .hidden_size = shape.hidden_size,
      .key_heads = shape.key_heads,
      .value_heads = shape.value_heads,
      .key_dim = shape.key_dim,
      .value_dim = shape.value_dim,
      .conv_width = shape.conv_width,
      .epsilon = shape.epsilon,
  };
}

std::size_t key_size(const raftinfer::kernels::GatedDeltaShape &shape) {
  return shape.key_heads * shape.key_dim;
}

std::size_t value_size(const raftinfer::kernels::GatedDeltaShape &shape) {
  return shape.value_heads * shape.value_dim;
}

std::size_t conv_dim(const raftinfer::kernels::GatedDeltaShape &shape) {
  return 2 * key_size(shape) + value_size(shape);
}

std::size_t token_stride(const raftinfer::kernels::GatedDeltaShape &shape) {
  return conv_dim(shape) + 2 * shape.value_heads + shape.hidden_size;
}

struct Fixture {
  std::vector<float> input;
  std::vector<float> conv_weight;
  std::vector<float> recurrent_a;
  std::vector<float> dt_bias;
  std::vector<float> output_norm_weight;
};

Fixture make_fixture(const raftinfer::kernels::GatedDeltaShape &shape) {
  Fixture fixture{
      .input = sequence(shape.tokens * token_stride(shape), 0.017F, -0.03F),
      .conv_weight =
          sequence(conv_dim(shape) * shape.conv_width, 0.011F, 0.04F),
      .recurrent_a = sequence(shape.value_heads, 0.03F, -1.2F),
      .dt_bias = sequence(shape.value_heads, 0.025F, -0.4F),
      .output_norm_weight = sequence(shape.value_dim, 0.02F, 0.7F),
  };
  return fixture;
}

template <typename T, typename Weight = T>
Fixture quantize_fixture(const Fixture &fixture) {
  // Keep the CPU FP32 algorithm and CUDA kernel on identical encoded
  // operands; otherwise operand quantization is incorrectly charged to the
  // kernel's arithmetic error.
  return Fixture{
      .input = decode<T>(encode<T>(fixture.input)),
      .conv_weight = decode<Weight>(encode<Weight>(fixture.conv_weight)),
      .recurrent_a = decode<Weight>(encode<Weight>(fixture.recurrent_a)),
      .dt_bias = decode<Weight>(encode<Weight>(fixture.dt_bias)),
      .output_norm_weight =
          decode<Weight>(encode<Weight>(fixture.output_norm_weight)),
  };
}

template <typename T> struct DeviceRun {
  std::vector<float> output;
  std::vector<float> convolution;
  std::vector<float> recurrent;
};

template <typename T, typename Weight = T>
DeviceRun<T> run_cuda(raftinfer::ExecutionContext &context,
                      const raftinfer::kernels::GatedDeltaShape &shape,
                      const Fixture &fixture,
                      std::span<const float> initial_convolution = {},
                      std::span<const float> initial_recurrent = {},
                      raftinfer::kernels::GatedDeltaLaunchPolicy policy = {}) {
  const auto input = encode<T>(fixture.input);
  const auto conv_weight = encode<Weight>(fixture.conv_weight);
  const auto recurrent_a = encode<Weight>(fixture.recurrent_a);
  const auto dt_bias = encode<Weight>(fixture.dt_bias);
  const auto output_norm_weight = encode<Weight>(fixture.output_norm_weight);
  auto input_device = upload(context, std::span{input});
  auto conv_weight_device = upload(context, std::span{conv_weight});
  auto recurrent_a_device = upload(context, std::span{recurrent_a});
  auto dt_bias_device = upload(context, std::span{dt_bias});
  auto output_norm_weight_device =
      upload(context, std::span{output_norm_weight});
  DeviceBuffer output_device{context,
                             shape.tokens * shape.hidden_size * sizeof(T)};
  DeviceBuffer convolution_device{
      context, conv_dim(shape) * (shape.conv_width - 1) * sizeof(float)};
  DeviceBuffer recurrent_device{context, shape.value_heads * shape.key_dim *
                                             shape.value_dim * sizeof(float)};
  const std::size_t workspace_bytes =
      raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape);
  DeviceBuffer workspace_device{context, workspace_bytes};

  if (initial_convolution.empty()) {
    assert(cudaMemsetAsync(convolution_device.data(), 0,
                           conv_dim(shape) * (shape.conv_width - 1) *
                               sizeof(float),
                           context.stream()) == cudaSuccess);
  } else {
    assert(cudaMemcpyAsync(
               convolution_device.data(), initial_convolution.data(),
               initial_convolution.size_bytes(), cudaMemcpyHostToDevice,
               context.stream()) == cudaSuccess);
  }
  if (initial_recurrent.empty()) {
    assert(cudaMemsetAsync(recurrent_device.data(), 0,
                           shape.value_heads * shape.key_dim * shape.value_dim *
                               sizeof(float),
                           context.stream()) == cudaSuccess);
  } else {
    assert(cudaMemcpyAsync(recurrent_device.data(), initial_recurrent.data(),
                           initial_recurrent.size_bytes(),
                           cudaMemcpyHostToDevice,
                           context.stream()) == cudaSuccess);
  }

  raftinfer::kernels::qwen35_gated_delta(
      input_device.data(), conv_weight_device.data(), recurrent_a_device.data(),
      dt_bias_device.data(), output_norm_weight_device.data(),
      output_device.data(), static_cast<float *>(convolution_device.data()),
      static_cast<float *>(recurrent_device.data()), workspace_device.data(),
      workspace_bytes, shape, policy, DTypeTraits<T>::dtype,
      DTypeTraits<Weight>::dtype, context.stream());

  return DeviceRun<T>{
      .output = decode<T>(download<T>(context, output_device.data(),
                                      shape.tokens * shape.hidden_size)),
      .convolution = download<float>(context, convolution_device.data(),
                                     conv_dim(shape) * (shape.conv_width - 1)),
      .recurrent =
          download<float>(context, recurrent_device.data(),
                          shape.value_heads * shape.key_dim * shape.value_dim),
  };
}

template <typename T, typename Weight = T>
raftinfer::reference::GatedDeltaReferenceState run_reference(
    const raftinfer::kernels::GatedDeltaShape &shape, const Fixture &fixture,
    std::span<float> output,
    const raftinfer::reference::GatedDeltaReferenceState *initial = nullptr) {
  const auto quantized = quantize_fixture<T, Weight>(fixture);
  const auto args = make_reference_args(shape);
  raftinfer::reference::GatedDeltaReferenceState state{args};
  if (initial != nullptr) {
    state = *initial;
  }
  raftinfer::reference::qwen35_gated_delta_prefill(
      quantized.input, output,
      raftinfer::reference::GatedDeltaReferenceWeights{
          .conv_weight = quantized.conv_weight,
          .recurrent_a = quantized.recurrent_a,
          .dt_bias = quantized.dt_bias,
          .output_norm_weight = quantized.output_norm_weight,
      },
      args, state);
  return state;
}

template <typename T>
void check_prefill_length(raftinfer::ExecutionContext &context, std::size_t tokens) {
  const auto shape = make_shape(tokens);
  const auto fixture = make_fixture(shape);
  std::vector<float> expected_output(tokens * shape.hidden_size);
  const auto expected_state = run_reference<T>(shape, fixture, expected_output);
  const auto actual = run_cuda<T>(context, shape, fixture);
  assert_matches(actual.output, expected_output);
  assert_matches(actual.convolution, expected_state.convolution);
  assert_matches(actual.recurrent, expected_state.recurrent);
}

template <typename T>
void check_continued_prefill(raftinfer::ExecutionContext &context) {
  const auto first_shape = make_shape(4);
  const auto first_fixture = make_fixture(first_shape);
  std::vector<float> first_expected(first_shape.tokens *
                                    first_shape.hidden_size);
  const auto first_reference_state =
      run_reference<T>(first_shape, first_fixture, first_expected);
  const auto first_actual = run_cuda<T>(context, first_shape, first_fixture);
  assert_matches(first_actual.output, first_expected);

  const auto continued_shape = make_shape(2);
  auto continued_fixture = make_fixture(continued_shape);
  for (float &value : continued_fixture.input) {
    value += 0.037F;
  }
  std::vector<float> continued_expected(continued_shape.tokens *
                                        continued_shape.hidden_size);
  const auto continued_reference_state =
      run_reference<T>(continued_shape, continued_fixture, continued_expected,
                       &first_reference_state);
  const auto continued_actual =
      run_cuda<T>(context, continued_shape, continued_fixture,
                  first_actual.convolution, first_actual.recurrent);
  assert_matches(continued_actual.output, continued_expected);
  assert_matches(continued_actual.convolution,
                 continued_reference_state.convolution);
  assert_matches(continued_actual.recurrent,
                 continued_reference_state.recurrent);
}

template <typename T>
void check_decode_after_prefill(raftinfer::ExecutionContext &context) {
  const auto prefill_shape = make_shape(4);
  const auto prefill_fixture = make_fixture(prefill_shape);
  std::vector<float> prefill_expected(prefill_shape.tokens *
                                      prefill_shape.hidden_size);
  const auto prefill_reference_state =
      run_reference<T>(prefill_shape, prefill_fixture, prefill_expected);
  const auto prefill_actual =
      run_cuda<T>(context, prefill_shape, prefill_fixture);

  const auto decode_shape = make_shape(1);
  auto decode_fixture = make_fixture(decode_shape);
  for (float &value : decode_fixture.input) {
    value -= 0.019F;
  }
  std::vector<float> decode_expected(decode_shape.hidden_size);
  const auto decode_reference_state = run_reference<T>(
      decode_shape, decode_fixture, decode_expected, &prefill_reference_state);
  const auto decode_actual =
      run_cuda<T>(context, decode_shape, decode_fixture,
                  prefill_actual.convolution, prefill_actual.recurrent);
  assert_matches(decode_actual.output, decode_expected);
  assert_matches(decode_actual.convolution, decode_reference_state.convolution);
  assert_matches(decode_actual.recurrent, decode_reference_state.recurrent);
}

template <typename T> void check_reset(raftinfer::ExecutionContext &context) {
  const auto shape = make_shape(4);
  const auto fixture = make_fixture(shape);
  const auto first = run_cuda<T>(context, shape, fixture);
  const auto reset = run_cuda<T>(context, shape, fixture);
  assert_matches(reset.output, first.output);
  assert_matches(reset.convolution, first.convolution);
  assert_matches(reset.recurrent, first.recurrent);
}

template <typename T>
void check_device_resident_sequence(raftinfer::ExecutionContext &context) {
  const auto prefill_shape = make_shape(4);
  const auto decode_shape = make_shape(1);
  const auto continued_shape = make_shape(2);
  const auto prefill_fixture = make_fixture(prefill_shape);
  auto decode_fixture = make_fixture(decode_shape);
  auto continued_fixture = make_fixture(continued_shape);
  decode_fixture.conv_weight = prefill_fixture.conv_weight;
  decode_fixture.recurrent_a = prefill_fixture.recurrent_a;
  decode_fixture.dt_bias = prefill_fixture.dt_bias;
  decode_fixture.output_norm_weight = prefill_fixture.output_norm_weight;
  continued_fixture.conv_weight = prefill_fixture.conv_weight;
  continued_fixture.recurrent_a = prefill_fixture.recurrent_a;
  continued_fixture.dt_bias = prefill_fixture.dt_bias;
  continued_fixture.output_norm_weight = prefill_fixture.output_norm_weight;
  for (float &value : decode_fixture.input) {
    value -= 0.019F;
  }
  for (float &value : continued_fixture.input) {
    value += 0.037F;
  }

  std::vector<float> expected_prefill(prefill_shape.tokens *
                                      prefill_shape.hidden_size);
  const auto reference_after_prefill =
      run_reference<T>(prefill_shape, prefill_fixture, expected_prefill);
  std::vector<float> expected_decode(decode_shape.hidden_size);
  const auto reference_after_decode = run_reference<T>(
      decode_shape, decode_fixture, expected_decode, &reference_after_prefill);
  std::vector<float> expected_continued(continued_shape.tokens *
                                        continued_shape.hidden_size);
  const auto expected_final_state =
      run_reference<T>(continued_shape, continued_fixture, expected_continued,
                       &reference_after_decode);

  const auto encoded_prefill = encode<T>(prefill_fixture.input);
  const auto encoded_decode = encode<T>(decode_fixture.input);
  const auto encoded_continued = encode<T>(continued_fixture.input);
  const auto conv_weight = encode<T>(prefill_fixture.conv_weight);
  const auto recurrent_a = encode<T>(prefill_fixture.recurrent_a);
  const auto dt_bias = encode<T>(prefill_fixture.dt_bias);
  const auto output_norm_weight = encode<T>(prefill_fixture.output_norm_weight);
  auto prefill_input_device = upload(context, std::span{encoded_prefill});
  auto decode_input_device = upload(context, std::span{encoded_decode});
  auto continued_input_device = upload(context, std::span{encoded_continued});
  auto conv_weight_device = upload(context, std::span{conv_weight});
  auto recurrent_a_device = upload(context, std::span{recurrent_a});
  auto dt_bias_device = upload(context, std::span{dt_bias});
  auto output_norm_weight_device =
      upload(context, std::span{output_norm_weight});
  DeviceBuffer prefill_output_device{
      context, prefill_shape.tokens * prefill_shape.hidden_size * sizeof(T)};
  DeviceBuffer decode_output_device{context,
                                    decode_shape.hidden_size * sizeof(T)};
  DeviceBuffer continued_output_device{
      context,
      continued_shape.tokens * continued_shape.hidden_size * sizeof(T)};
  DeviceBuffer convolution_device{context, conv_dim(prefill_shape) *
                                               (prefill_shape.conv_width - 1) *
                                               sizeof(float)};
  DeviceBuffer recurrent_device{
      context, prefill_shape.value_heads * prefill_shape.key_dim *
                   prefill_shape.value_dim * sizeof(float)};
  const std::size_t workspace_bytes =
      raftinfer::kernels::qwen35_gated_delta_workspace_bytes(prefill_shape);
  DeviceBuffer workspace_device{context, workspace_bytes};
  assert(cudaMemsetAsync(convolution_device.data(), 0,
                         conv_dim(prefill_shape) *
                             (prefill_shape.conv_width - 1) * sizeof(float),
                         context.stream()) == cudaSuccess);
  assert(cudaMemsetAsync(recurrent_device.data(), 0,
                         prefill_shape.value_heads * prefill_shape.key_dim *
                             prefill_shape.value_dim * sizeof(float),
                         context.stream()) == cudaSuccess);

  const auto launch = [&](const void *input, void *output,
                          raftinfer::kernels::GatedDeltaShape shape) {
    raftinfer::kernels::qwen35_gated_delta(
        input, conv_weight_device.data(), recurrent_a_device.data(),
        dt_bias_device.data(), output_norm_weight_device.data(), output,
        static_cast<float *>(convolution_device.data()),
        static_cast<float *>(recurrent_device.data()), workspace_device.data(),
        workspace_bytes, shape, DTypeTraits<T>::dtype, DTypeTraits<T>::dtype,
        context.stream());
  };

  // All three calls share the same device state, workspace, and stream. No
  // host copy or synchronization occurs between them.
  launch(prefill_input_device.data(), prefill_output_device.data(),
         prefill_shape);
  launch(decode_input_device.data(), decode_output_device.data(), decode_shape);
  launch(continued_input_device.data(), continued_output_device.data(),
         continued_shape);

  const auto actual_prefill =
      decode<T>(download<T>(context, prefill_output_device.data(),
                            prefill_shape.tokens * prefill_shape.hidden_size));
  const auto actual_decode = decode<T>(download<T>(
      context, decode_output_device.data(), decode_shape.hidden_size));
  const auto actual_continued = decode<T>(
      download<T>(context, continued_output_device.data(),
                  continued_shape.tokens * continued_shape.hidden_size));
  const auto actual_convolution =
      download<float>(context, convolution_device.data(),
                      conv_dim(prefill_shape) * (prefill_shape.conv_width - 1));
  const auto actual_recurrent =
      download<float>(context, recurrent_device.data(),
                      prefill_shape.value_heads * prefill_shape.key_dim *
                          prefill_shape.value_dim);
  assert_matches(actual_prefill, expected_prefill);
  assert_matches(actual_decode, expected_decode);
  assert_matches(actual_continued, expected_continued);
  assert_matches(actual_convolution, expected_final_state.convolution);
  assert_matches(actual_recurrent, expected_final_state.recurrent);
}

template <typename T>
void check_saturation_edges(raftinfer::ExecutionContext &context) {
  const auto shape = make_shape(4);
  auto fixture = make_fixture(shape);
  fixture.recurrent_a = {-12.0F, -3.0F, -0.25F, -0.001F};
  fixture.dt_bias = {-25.0F, 0.0F, 5.0F, -5.0F};
  fixture.output_norm_weight = {8.0F, -7.0F, 6.0F, -5.0F, 4.0F, -3.0F, 2.0F};
  for (std::size_t token = 0; token < shape.tokens; ++token) {
    const std::size_t base = token * token_stride(shape);
    for (std::size_t head = 0; head < shape.value_heads; ++head) {
      fixture.input[base + conv_dim(shape) + head] =
          ((token + head) % 2 == 0) ? -30.0F : 30.0F;
      fixture.input[base + conv_dim(shape) + shape.value_heads + head] =
          head == 3 ? 10.0F : (head == 0 ? -10.0F : 0.5F);
    }
    const std::size_t gate_base =
        base + conv_dim(shape) + 2 * shape.value_heads;
    for (std::size_t index = 0; index < shape.hidden_size; ++index) {
      fixture.input[gate_base + index] =
          ((token + index) % 2 == 0) ? -18.0F : 18.0F;
    }
  }

  std::vector<float> expected_output(shape.tokens * shape.hidden_size);
  const auto expected_state = run_reference<T>(shape, fixture, expected_output);
  const auto actual = run_cuda<T>(context, shape, fixture);
  assert_finite(expected_output);
  assert_finite(expected_state.convolution);
  assert_finite(expected_state.recurrent);
  assert_finite(actual.output);
  assert_finite(actual.convolution);
  assert_finite(actual.recurrent);
  assert_matches(actual.output, expected_output);
  assert_matches(actual.convolution, expected_state.convolution);
  assert_matches(actual.recurrent, expected_state.recurrent);
}

template <typename T>
void check_f32_auxiliary_weights(raftinfer::ExecutionContext &context) {
  const auto shape = make_shape(4);
  const auto fixture = make_fixture(shape);
  std::vector<float> expected_output(shape.tokens * shape.hidden_size);
  const auto expected_state =
      run_reference<T, float>(shape, fixture, expected_output);
  const auto actual = run_cuda<T, float>(context, shape, fixture);
  assert_matches(actual.output, expected_output);
  assert_matches(actual.convolution, expected_state.convolution);
  assert_matches(actual.recurrent, expected_state.recurrent);
}

template <typename Weight>
void check_f32_activations(raftinfer::ExecutionContext &context) {
  const auto shape = make_shape(4);
  const auto fixture = make_fixture(shape);
  std::vector<float> expected_output(shape.tokens * shape.hidden_size);
  const auto expected_state =
      run_reference<float, Weight>(shape, fixture, expected_output);
  const auto actual = run_cuda<float, Weight>(context, shape, fixture);
  assert_matches(actual.output, expected_output);
  assert_matches(actual.convolution, expected_state.convolution);
  assert_matches(actual.recurrent, expected_state.recurrent);
}

void check_model_shape_register_path(raftinfer::ExecutionContext &context) {
  const raftinfer::kernels::GatedDeltaShape shape{
      .tokens = 3,
      .hidden_size = 512,
      .key_heads = 2,
      .value_heads = 4,
      .key_dim = 128,
      .value_dim = 128,
      .conv_width = 3,
      .epsilon = 1.0e-6F,
  };
  const auto fixture = make_fixture(shape);
  std::vector<float> expected_output(shape.tokens * shape.hidden_size);
  const auto expected_state =
      run_reference<float, float>(shape, fixture, expected_output);
  const auto actual = run_cuda<float, float>(context, shape, fixture);
  assert_matches(actual.output, expected_output);
  assert_matches(actual.convolution, expected_state.convolution);
  assert_matches(actual.recurrent, expected_state.recurrent);
}

raftinfer::kernels::GatedDeltaShape make_schedule_shape(std::size_t tokens,
                                                  std::size_t dim) {
  return raftinfer::kernels::GatedDeltaShape{
      .tokens = tokens,
      .hidden_size = 4 * dim,
      .key_heads = 2,
      .value_heads = 4,
      .key_dim = dim,
      .value_dim = dim,
      .conv_width = 3,
      .epsilon = 1.0e-6F,
  };
}

void assert_schedule_policy(const raftinfer::kernels::GatedDeltaLaunchPolicy &policy,
                            raftinfer::kernels::GatedDeltaSchedule schedule) {
  assert(policy.schedule == schedule);
  assert(policy.warps_per_block == 4);
  assert(!policy.transposed_boundary_state);
}

template <typename T>
void check_candidate_prefill_bucket(raftinfer::ExecutionContext &context,
                                    std::size_t dim, std::size_t tokens,
                                    raftinfer::kernels::GatedDeltaSchedule schedule) {
  const auto shape = make_schedule_shape(tokens, dim);
  const auto fixture = make_fixture(shape);
  std::vector<float> expected_output(tokens * shape.hidden_size);
  const auto expected_state = run_reference<T>(shape, fixture, expected_output);
  const auto policy =
      raftinfer::kernels::GatedDeltaLaunchPolicy{.schedule = schedule,
                                           .warps_per_block = 4,
                                           .transposed_boundary_state = false};
  const auto actual = run_cuda<T>(context, shape, fixture, {}, {}, policy);
  assert_matches(actual.output, expected_output);
  assert_matches(actual.convolution, expected_state.convolution);
  assert_matches(actual.recurrent, expected_state.recurrent);
}

template <typename T>
void check_candidate_continued_prefill_and_decode(
    raftinfer::ExecutionContext &context, std::size_t dim,
    raftinfer::kernels::GatedDeltaSchedule prefill_schedule,
    raftinfer::kernels::GatedDeltaSchedule decode_schedule) {
  const auto first_shape = make_schedule_shape(17, dim);
  const auto first_fixture = make_fixture(first_shape);
  std::vector<float> first_expected(first_shape.tokens *
                                    first_shape.hidden_size);
  const auto first_reference_state =
      run_reference<T>(first_shape, first_fixture, first_expected);
  const auto first_policy =
      raftinfer::kernels::GatedDeltaLaunchPolicy{.schedule = prefill_schedule,
                                           .warps_per_block = 4,
                                           .transposed_boundary_state = false};
  const auto first_actual =
      run_cuda<T>(context, first_shape, first_fixture, {}, {}, first_policy);
  assert_matches(first_actual.output, first_expected);
  assert_matches(first_actual.convolution, first_reference_state.convolution);
  assert_matches(first_actual.recurrent, first_reference_state.recurrent);

  const auto continued_shape = make_schedule_shape(128, dim);
  auto continued_fixture = make_fixture(continued_shape);
  for (float &value : continued_fixture.input) {
    value += 0.021F;
  }
  std::vector<float> continued_expected(continued_shape.tokens *
                                        continued_shape.hidden_size);
  auto reference_state =
      run_reference<T>(continued_shape, continued_fixture, continued_expected,
                       &first_reference_state);
  const auto continued_actual = run_cuda<T>(
      context, continued_shape, continued_fixture, first_actual.convolution,
      first_actual.recurrent, first_policy);
  assert_matches(continued_actual.output, continued_expected);
  assert_matches(continued_actual.convolution, reference_state.convolution);
  assert_matches(continued_actual.recurrent, reference_state.recurrent);

  auto convolution = continued_actual.convolution;
  auto recurrent = continued_actual.recurrent;
  for (std::size_t step = 0; step < 8; ++step) {
    const auto decode_shape = make_schedule_shape(1, dim);
    auto decode_fixture = make_fixture(decode_shape);
    for (float &value : decode_fixture.input) {
      value += 0.013F * static_cast<float>(step + 1);
    }
    std::vector<float> decode_expected(decode_shape.hidden_size);
    reference_state = run_reference<T>(decode_shape, decode_fixture,
                                       decode_expected, &reference_state);
    const auto decode_policy = raftinfer::kernels::GatedDeltaLaunchPolicy{
        .schedule = decode_schedule,
        .warps_per_block = 4,
        .transposed_boundary_state = false};
    const auto decode_actual =
        run_cuda<T>(context, decode_shape, decode_fixture, convolution,
                    recurrent, decode_policy);
    assert_matches(decode_actual.output, decode_expected);
    assert_matches(decode_actual.convolution, reference_state.convolution);
    assert_matches(decode_actual.recurrent, reference_state.recurrent);
    convolution = decode_actual.convolution;
    recurrent = decode_actual.recurrent;
  }
}

template <typename T>
void check_sm120_candidate_schedules(raftinfer::ExecutionContext &context) {
  for (const std::size_t dim : {std::size_t{64}, std::size_t{128}}) {
    const auto current = raftinfer::kernels::qwen35_gated_delta_select_policy(
        make_schedule_shape(128, dim), 128);
    assert_schedule_policy(
        current, raftinfer::kernels::GatedDeltaSchedule::register_resident_current);
    for (const std::size_t tokens :
         {std::size_t{17}, std::size_t{128}, std::size_t{512}}) {
      check_candidate_prefill_bucket<T>(
          context, dim, tokens,
          raftinfer::kernels::GatedDeltaSchedule::register_resident_prefill_sm120);
    }
    check_candidate_continued_prefill_and_decode<T>(
        context, dim,
        raftinfer::kernels::GatedDeltaSchedule::register_resident_prefill_sm120,
        raftinfer::kernels::GatedDeltaSchedule::register_resident_decode_sm120);
  }
}

void expect_delta_error(auto &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const raftinfer::kernels::Qwen35DeltaError &) {
    thrown = true;
  }
  assert(thrown);
}

void check_invalid_shapes(cudaStream_t stream) {
  auto shape = make_shape(1);
  const auto four_tokens = make_shape(4);
  assert(raftinfer::kernels::qwen35_gated_delta_workspace_bytes(four_tokens) ==
         4 * raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape));
  int dummy = 0;
  const auto launch = [&](raftinfer::kernels::GatedDeltaShape candidate,
                          std::size_t workspace_bytes =
                              std::numeric_limits<std::size_t>::max()) {
    raftinfer::kernels::qwen35_gated_delta(&dummy, &dummy, &dummy, &dummy, &dummy,
                                     &dummy, reinterpret_cast<float *>(&dummy),
                                     reinterpret_cast<float *>(&dummy), &dummy,
                                     workspace_bytes, candidate, RAFTINFER_DTYPE_F16,
                                     RAFTINFER_DTYPE_F16, stream);
  };

  auto zero_tokens = shape;
  zero_tokens.tokens = 0;
  expect_delta_error([&] { launch(zero_tokens); });
  auto bad_hidden = shape;
  ++bad_hidden.hidden_size;
  expect_delta_error([&] { launch(bad_hidden); });
  auto bad_heads = shape;
  bad_heads.value_heads = 3;
  expect_delta_error([&] { launch(bad_heads); });
  auto zero_width = shape;
  zero_width.conv_width = 0;
  expect_delta_error([&] { launch(zero_width); });
  auto bad_epsilon = shape;
  bad_epsilon.epsilon = std::numeric_limits<float>::quiet_NaN();
  expect_delta_error([&] { launch(bad_epsilon); });
  auto overflow = shape;
  overflow.tokens = std::numeric_limits<std::size_t>::max();
  expect_delta_error([&] { launch(overflow); });
  expect_delta_error([&] {
    launch(shape, raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape) - 1);
  });
  expect_delta_error([&] {
    raftinfer::kernels::qwen35_gated_delta(
        nullptr, &dummy, &dummy, &dummy, &dummy, &dummy,
        reinterpret_cast<float *>(&dummy), reinterpret_cast<float *>(&dummy),
        &dummy, raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape), shape,
        RAFTINFER_DTYPE_F16, RAFTINFER_DTYPE_F16, stream);
  });
  expect_delta_error([&] {
    raftinfer::kernels::qwen35_gated_delta(
        &dummy, &dummy, &dummy, &dummy, &dummy, &dummy,
        reinterpret_cast<float *>(&dummy), reinterpret_cast<float *>(&dummy),
        &dummy, raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape), shape,
        RAFTINFER_DTYPE_Q4_K, RAFTINFER_DTYPE_Q4_K, stream);
  });
  expect_delta_error([&] {
    raftinfer::kernels::qwen35_gated_delta(
        &dummy, &dummy, &dummy, &dummy, &dummy, &dummy,
        reinterpret_cast<float *>(&dummy), reinterpret_cast<float *>(&dummy),
        &dummy, raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape), shape,
        raftinfer::kernels::GatedDeltaLaunchPolicy{
            .schedule = static_cast<raftinfer::kernels::GatedDeltaSchedule>(99),
            .warps_per_block = 4,
            .transposed_boundary_state = false},
        RAFTINFER_DTYPE_F16, RAFTINFER_DTYPE_F16, stream);
  });
}

void check_misaligned_pointers(cudaStream_t stream) {
  const auto shape = make_shape(1);
  const std::size_t workspace_bytes =
      raftinfer::kernels::qwen35_gated_delta_workspace_bytes(shape);
  alignas(16) std::array<std::byte, 1024> storage{};
  void *const aligned = storage.data();
  void *const misaligned = storage.data() + 1;
  auto *const aligned_float = static_cast<float *>(aligned);
  auto *const misaligned_float = reinterpret_cast<float *>(storage.data() + 1);

  const auto invoke = [&](const void *input, const void *conv_weight,
                          const void *recurrent_a, const void *dt_bias,
                          const void *output_norm_weight, void *output,
                          float *convolution_state, float *recurrent_state,
                          void *workspace, RaftInferDataType dtype) {
    raftinfer::kernels::qwen35_gated_delta(
        input, conv_weight, recurrent_a, dt_bias, output_norm_weight, output,
        convolution_state, recurrent_state, workspace, workspace_bytes, shape,
        dtype, dtype, stream);
  };

  for (const RaftInferDataType dtype : {RAFTINFER_DTYPE_F16, RAFTINFER_DTYPE_BF16}) {
    expect_delta_error([&] {
      invoke(misaligned, aligned, aligned, aligned, aligned, aligned,
             aligned_float, aligned_float, aligned, dtype);
    });
    expect_delta_error([&] {
      invoke(aligned, misaligned, aligned, aligned, aligned, aligned,
             aligned_float, aligned_float, aligned, dtype);
    });
    expect_delta_error([&] {
      invoke(aligned, aligned, misaligned, aligned, aligned, aligned,
             aligned_float, aligned_float, aligned, dtype);
    });
    expect_delta_error([&] {
      invoke(aligned, aligned, aligned, misaligned, aligned, aligned,
             aligned_float, aligned_float, aligned, dtype);
    });
    expect_delta_error([&] {
      invoke(aligned, aligned, aligned, aligned, misaligned, aligned,
             aligned_float, aligned_float, aligned, dtype);
    });
    expect_delta_error([&] {
      invoke(aligned, aligned, aligned, aligned, aligned, misaligned,
             aligned_float, aligned_float, aligned, dtype);
    });
  }
  expect_delta_error([&] {
    invoke(aligned, aligned, aligned, aligned, aligned, aligned,
           misaligned_float, aligned_float, aligned, RAFTINFER_DTYPE_F16);
  });
  expect_delta_error([&] {
    invoke(aligned, aligned, aligned, aligned, aligned, aligned, aligned_float,
           misaligned_float, aligned, RAFTINFER_DTYPE_F16);
  });
  expect_delta_error([&] {
    invoke(aligned, aligned, aligned, aligned, aligned, aligned, aligned_float,
           aligned_float, misaligned, RAFTINFER_DTYPE_F16);
  });
}

template <typename T> void run_dtype_suite(raftinfer::ExecutionContext &context) {
  for (const std::size_t tokens : {1U, 2U, 4U, 17U}) {
    check_prefill_length<T>(context, tokens);
  }
  check_decode_after_prefill<T>(context);
  check_continued_prefill<T>(context);
  check_reset<T>(context);
  check_device_resident_sequence<T>(context);
  check_saturation_edges<T>(context);
}

} // namespace

int main() {
  TestResources resources{64 * 1024 * 1024};
  run_dtype_suite<__half>(resources.context());
  run_dtype_suite<__nv_bfloat16>(resources.context());
  run_dtype_suite<float>(resources.context());
  check_f32_auxiliary_weights<__half>(resources.context());
  check_f32_auxiliary_weights<__nv_bfloat16>(resources.context());
  check_f32_activations<__half>(resources.context());
  check_f32_activations<__nv_bfloat16>(resources.context());
  check_model_shape_register_path(resources.context());
  check_sm120_candidate_schedules<__nv_bfloat16>(resources.context());
  check_sm120_candidate_schedules<float>(resources.context());
  check_invalid_shapes(resources.context().stream());
  check_misaligned_pointers(resources.context().stream());
}
