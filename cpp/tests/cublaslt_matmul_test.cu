#include "../execution/cublaslt_matmul.hpp"

#include "assert_enabled.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    assert(cudaMalloc(&data_, bytes_ == 0 ? 1 : bytes_) == cudaSuccess);
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
  std::size_t bytes() const noexcept { return bytes_; }

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

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  cudaStream_t get() const noexcept { return stream_; }

private:
  cudaStream_t stream_{};
};

template <typename T> struct DTypeTraits;

template <> struct DTypeTraits<__half> {
  static constexpr RaftInferDataType dtype = RAFTINFER_DTYPE_F16;
  static __half encode(float value) { return __float2half_rn(value); }
  static float decode(__half value) { return __half2float(value); }
};

template <> struct DTypeTraits<__nv_bfloat16> {
  static constexpr RaftInferDataType dtype = RAFTINFER_DTYPE_BF16;
  static __nv_bfloat16 encode(float value) {
    return __float2bfloat16_rn(value);
  }
  static float decode(__nv_bfloat16 value) { return __bfloat162float(value); }
};

template <> struct DTypeTraits<float> {
  static constexpr RaftInferDataType dtype = RAFTINFER_DTYPE_F32;
  static float encode(float value) { return value; }
  static float decode(float value) { return value; }
};

template <typename T> std::vector<T> encode(std::span<const float> values) {
  std::vector<T> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(),
                 DTypeTraits<T>::encode);
  return result;
}

template <typename T>
DeviceBuffer upload(std::span<const T> values, cudaStream_t stream) {
  DeviceBuffer result{values.size_bytes()};
  assert(cudaMemcpyAsync(result.data(), values.data(), values.size_bytes(),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
  return result;
}

template <typename T>
std::vector<T> download(const DeviceBuffer &buffer, std::size_t elements,
                        cudaStream_t stream) {
  std::vector<T> result(elements);
  assert(cudaMemcpyAsync(result.data(), buffer.data(), elements * sizeof(T),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  return result;
}

bool close_enough(float actual, float expected) {
  const float abs = std::fabs(actual - expected);
  const float rel = abs / std::max(std::fabs(expected), 1.0e-6F);
  return abs <= 2.0e-2F || rel <= 2.0e-2F;
}

std::vector<float> sequence(std::size_t elements, float scale, float bias) {
  std::vector<float> result(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    result[i] =
        bias + scale * static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
  }
  return result;
}

std::vector<float> transpose(std::span<const float> input, std::size_t rows,
                             std::size_t cols) {
  std::vector<float> output(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      output[col * rows + row] = input[row * cols + col];
    }
  }
  return output;
}

std::vector<float> reference_matmul(std::span<const float> logical_input,
                                    std::span<const float> logical_weight,
                                    const raftinfer::CublasLtMatmulShape &shape) {
  std::vector<float> output(shape.m * shape.n);
  for (std::size_t row = 0; row < shape.m; ++row) {
    for (std::size_t col = 0; col < shape.n; ++col) {
      double sum = 0.0;
      for (std::size_t inner = 0; inner < shape.k; ++inner) {
        sum += static_cast<double>(logical_input[row * shape.k + inner]) *
               logical_weight[inner * shape.n + col];
      }
      output[row * shape.n + col] = static_cast<float>(sum);
    }
  }
  return output;
}

void expect_matmul_error(auto &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const raftinfer::CublasLtMatmulError &) {
    thrown = true;
  }
  assert(thrown);
}

raftinfer::CublasLtMatmulConfig
config_for(raftinfer::CublasLtMatmulShape shape, RaftInferDataType dtype,
           std::size_t workspace_budget = 4U * 1024U * 1024U) {
  return raftinfer::CublasLtMatmulConfig{
      .shape = shape,
      .input_dtype = dtype,
      .weight_dtype = dtype,
      .output_dtype = dtype,
      .input_order = raftinfer::CublasLtMatrixOrder::RowMajor,
      .weight_order = raftinfer::CublasLtMatrixOrder::RowMajor,
      .output_order = raftinfer::CublasLtMatrixOrder::RowMajor,
      .workspace_budget_bytes = workspace_budget,
  };
}

template <typename T> void check_shape(raftinfer::CublasLtMatmulShape shape) {
  const auto logical_input = sequence(shape.m * shape.k, 0.007F, 0.03F);
  const auto logical_weight = sequence(shape.k * shape.n, -0.005F, 0.01F);
  const auto physical_input = shape.transpose_input
                                  ? transpose(logical_input, shape.m, shape.k)
                                  : logical_input;
  const auto physical_weight = shape.transpose_weight
                                   ? transpose(logical_weight, shape.k, shape.n)
                                   : logical_weight;
  const auto input = encode<T>(physical_input);
  const auto weight = encode<T>(physical_weight);
  const auto expected = reference_matmul(logical_input, logical_weight, shape);

  auto plan =
      raftinfer::CublasLtMatmulPlan::create(config_for(shape, DTypeTraits<T>::dtype));
  assert(plan != nullptr);
  assert(plan->algorithm_id() >= 0);
  assert(plan->input_bytes() == input.size() * sizeof(T));
  assert(plan->weight_bytes() == weight.size() * sizeof(T));
  assert(plan->output_bytes() == expected.size() * sizeof(T));
  assert(plan->workspace_bytes() <= 4U * 1024U * 1024U);

  Stream stream;
  auto input_device = upload<T>(input, stream.get());
  auto weight_device = upload<T>(weight, stream.get());
  DeviceBuffer output_device{plan->output_bytes()};
  DeviceBuffer workspace_device{plan->workspace_bytes()};
  void *workspace =
      plan->workspace_bytes() == 0 ? nullptr : workspace_device.data();

  const int algorithm_before = plan->algorithm_id();
  for (int iteration = 0; iteration < 2; ++iteration) {
    plan->run(stream.get(), input_device.data(), input_device.bytes(),
              weight_device.data(), weight_device.bytes(), output_device.data(),
              output_device.bytes(), workspace, plan->workspace_bytes());
    const auto output =
        download<T>(output_device, expected.size(), stream.get());
    for (std::size_t i = 0; i < output.size(); ++i) {
      assert(close_enough(DTypeTraits<T>::decode(output[i]), expected[i]));
    }
    // Repeated run must retain the initialization-time heuristic selection.
    assert(plan->algorithm_id() == algorithm_before);
  }
}

template <typename Input>
void check_f32_output_shape(raftinfer::CublasLtMatmulShape shape) {
  const auto logical_input = sequence(shape.m * shape.k, 0.007F, 0.03F);
  const auto logical_weight = sequence(shape.k * shape.n, -0.005F, 0.01F);
  const auto physical_input = shape.transpose_input
                                  ? transpose(logical_input, shape.m, shape.k)
                                  : logical_input;
  const auto physical_weight = shape.transpose_weight
                                   ? transpose(logical_weight, shape.k, shape.n)
                                   : logical_weight;
  const auto input = encode<Input>(physical_input);
  const auto weight = encode<Input>(physical_weight);
  const auto expected = reference_matmul(logical_input, logical_weight, shape);
  auto config = config_for(shape, DTypeTraits<Input>::dtype);
  config.output_dtype = RAFTINFER_DTYPE_F32;
  auto plan = raftinfer::CublasLtMatmulPlan::create(config);
  assert(plan->input_bytes() == input.size() * sizeof(Input));
  assert(plan->weight_bytes() == weight.size() * sizeof(Input));
  assert(plan->output_bytes() == expected.size() * sizeof(float));

  Stream stream;
  auto input_device = upload<Input>(input, stream.get());
  auto weight_device = upload<Input>(weight, stream.get());
  DeviceBuffer output_device{plan->output_bytes()};
  DeviceBuffer workspace_device{plan->workspace_bytes()};
  plan->run(stream.get(), input_device.data(), input_device.bytes(),
            weight_device.data(), weight_device.bytes(), output_device.data(),
            output_device.bytes(),
            plan->workspace_bytes() == 0 ? nullptr : workspace_device.data(),
            plan->workspace_bytes());
  const auto output =
      download<float>(output_device, expected.size(), stream.get());
  for (std::size_t i = 0; i < output.size(); ++i) {
    assert(close_enough(output[i], expected[i]));
  }
}

void check_validation() {
  const auto valid_shape = raftinfer::CublasLtMatmulShape{
      .m = 17,
      .n = 13,
      .k = 19,
      .transpose_input = false,
      .transpose_weight = true,
  };
  for (const auto shape : {
           raftinfer::CublasLtMatmulShape{0, 1, 1, false, true},
           raftinfer::CublasLtMatmulShape{1, 0, 1, false, true},
           raftinfer::CublasLtMatmulShape{1, 1, 0, false, true},
       }) {
    expect_matmul_error([&] {
      (void)raftinfer::CublasLtMatmulPlan::create(config_for(shape, RAFTINFER_DTYPE_F16));
    });
  }

  auto bad_dtype = config_for(valid_shape, RAFTINFER_DTYPE_F16);
  bad_dtype.weight_dtype = RAFTINFER_DTYPE_Q4_K;
  expect_matmul_error(
      [&] { (void)raftinfer::CublasLtMatmulPlan::create(bad_dtype); });

  auto mixed_output = config_for(valid_shape, RAFTINFER_DTYPE_F16);
  mixed_output.output_dtype = RAFTINFER_DTYPE_BF16;
  expect_matmul_error(
      [&] { (void)raftinfer::CublasLtMatmulPlan::create(mixed_output); });

  auto mixed_weight = config_for(valid_shape, RAFTINFER_DTYPE_F16);
  mixed_weight.weight_dtype = RAFTINFER_DTYPE_BF16;
  expect_matmul_error(
      [&] { (void)raftinfer::CublasLtMatmulPlan::create(mixed_weight); });

  auto bad_layout = config_for(valid_shape, RAFTINFER_DTYPE_F16);
  bad_layout.weight_order = static_cast<raftinfer::CublasLtMatrixOrder>(99);
  expect_matmul_error(
      [&] { (void)raftinfer::CublasLtMatmulPlan::create(bad_layout); });

  auto overflowing = config_for(valid_shape, RAFTINFER_DTYPE_F16);
  overflowing.shape.m = std::numeric_limits<std::size_t>::max();
  expect_matmul_error(
      [&] { (void)raftinfer::CublasLtMatmulPlan::create(overflowing); });

  const auto descriptor_limit =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  raftinfer::detail::validate_cublaslt_shape(
      raftinfer::CublasLtMatmulShape{descriptor_limit, 1, 1, false, true});
  expect_matmul_error([&] {
    raftinfer::detail::validate_cublaslt_shape(
        raftinfer::CublasLtMatmulShape{descriptor_limit + 1, 1, 1, false, true});
  });
  expect_matmul_error([&] {
    raftinfer::detail::validate_cublaslt_shape(
        raftinfer::CublasLtMatmulShape{1, descriptor_limit + 1, 1, false, true});
  });
  expect_matmul_error([&] {
    raftinfer::detail::validate_cublaslt_shape(
        raftinfer::CublasLtMatmulShape{1, 1, descriptor_limit + 1, false, true});
  });
}

void check_run_validation() {
  const auto shape = raftinfer::CublasLtMatmulShape{
      .m = 17,
      .n = 13,
      .k = 19,
      .transpose_input = false,
      .transpose_weight = true,
  };
  auto plan = raftinfer::CublasLtMatmulPlan::create(config_for(shape, RAFTINFER_DTYPE_F16));
  DeviceBuffer input{plan->input_bytes()};
  DeviceBuffer weight{plan->weight_bytes()};
  DeviceBuffer output{plan->output_bytes()};
  DeviceBuffer workspace{plan->workspace_bytes()};
  void *valid_workspace =
      plan->workspace_bytes() == 0 ? nullptr : workspace.data();

  auto launch = [&](const void *input_ptr, std::size_t input_bytes,
                    const void *weight_ptr, std::size_t weight_bytes,
                    void *output_ptr, std::size_t output_bytes,
                    void *workspace_ptr, std::size_t workspace_bytes) {
    plan->run(nullptr, input_ptr, input_bytes, weight_ptr, weight_bytes,
              output_ptr, output_bytes, workspace_ptr, workspace_bytes);
  };
  expect_matmul_error([&] {
    launch(nullptr, plan->input_bytes(), weight.data(), plan->weight_bytes(),
           output.data(), plan->output_bytes(), valid_workspace,
           plan->workspace_bytes());
  });
  expect_matmul_error([&] {
    launch(input.data(), plan->input_bytes() - 1, weight.data(),
           plan->weight_bytes(), output.data(), plan->output_bytes(),
           valid_workspace, plan->workspace_bytes());
  });
  expect_matmul_error([&] {
    launch(input.data(), plan->input_bytes(), weight.data(),
           plan->weight_bytes() - 1, output.data(), plan->output_bytes(),
           valid_workspace, plan->workspace_bytes());
  });
  expect_matmul_error([&] {
    launch(input.data(), plan->input_bytes(), weight.data(),
           plan->weight_bytes(), output.data(), plan->output_bytes() - 1,
           valid_workspace, plan->workspace_bytes());
  });
  if (plan->workspace_bytes() > 0) {
    expect_matmul_error([&] {
      launch(input.data(), plan->input_bytes(), weight.data(),
             plan->weight_bytes(), output.data(), plan->output_bytes(),
             workspace.data(), plan->workspace_bytes() - 1);
    });
    expect_matmul_error([&] {
      launch(input.data(), plan->input_bytes(), weight.data(),
             plan->weight_bytes(), output.data(), plan->output_bytes(), nullptr,
             plan->workspace_bytes());
    });
  }
}

void check_deterministic_workspace_validation() {
  const auto buffers = raftinfer::detail::CublasLtRunBuffers{
      .input = reinterpret_cast<const void *>(0x1000),
      .input_bytes = 128,
      .weight = reinterpret_cast<const void *>(0x2000),
      .weight_bytes = 256,
      .output = reinterpret_cast<void *>(0x3000),
      .output_bytes = 64,
      .workspace = reinterpret_cast<void *>(0x4000),
      .workspace_bytes = 4096,
  };
  const auto requirements = raftinfer::detail::CublasLtBufferRequirements{
      .input_bytes = 128,
      .weight_bytes = 256,
      .output_bytes = 64,
      .workspace_bytes = 4096,
  };
  raftinfer::detail::validate_cublaslt_run_buffers(buffers, requirements);

  auto undersized = buffers;
  undersized.workspace_bytes = requirements.workspace_bytes - 1;
  expect_matmul_error([&] {
    raftinfer::detail::validate_cublaslt_run_buffers(undersized, requirements);
  });

  auto missing = buffers;
  missing.workspace = nullptr;
  expect_matmul_error([&] {
    raftinfer::detail::validate_cublaslt_run_buffers(missing, requirements);
  });
}

template <typename T> void check_candidate_selection_shape() {
  const auto shape = raftinfer::CublasLtMatmulShape{
      .m = 128,
      .n = 96,
      .k = 64,
      .transpose_input = false,
      .transpose_weight = true,
  };
  auto config = config_for(shape, DTypeTraits<T>::dtype);
  const auto candidates = raftinfer::enumerate_cublaslt_candidates(config, 16);
  const auto oversized_request =
      raftinfer::enumerate_cublaslt_candidates(config, 128);
  assert(!candidates.empty());
  assert(candidates.size() <= 16);
  assert(!oversized_request.empty());
  assert(oversized_request.size() <= 16);
  for (const auto &candidate : candidates) {
    assert(candidate.algorithm_id >= 0);
    assert(candidate.workspace_bytes <= config.workspace_budget_bytes);
  }

  const auto logical_input = sequence(shape.m * shape.k, 0.007F, 0.03F);
  const auto logical_weight = sequence(shape.k * shape.n, -0.005F, 0.01F);
  const auto input = encode<T>(logical_input);
  const auto weight = encode<T>(transpose(logical_weight, shape.k, shape.n));

  auto first_heuristic = raftinfer::CublasLtMatmulPlan::create(config);
  auto tuned = raftinfer::CublasLtMatmulPlan::create(config);
  assert(first_heuristic->algorithm_id() == candidates.front().algorithm_id);

  Stream stream;
  auto input_device = upload<T>(input, stream.get());
  auto weight_device = upload<T>(weight, stream.get());
  DeviceBuffer first_output{first_heuristic->output_bytes()};
  DeviceBuffer tuned_output{tuned->output_bytes()};
  DeviceBuffer workspace{config.workspace_budget_bytes};

  first_heuristic->run(stream.get(), input_device.data(), input_device.bytes(),
                       weight_device.data(), weight_device.bytes(),
                       first_output.data(), first_output.bytes(),
                       workspace.data(), workspace.bytes());
  tuned->select_fastest(stream.get(), input_device.data(), weight_device.data(),
                        tuned_output.data(), workspace.data(),
                        workspace.bytes(), 1, 2);

  const auto first = download<T>(first_output, shape.m * shape.n, stream.get());
  const auto selected =
      download<T>(tuned_output, shape.m * shape.n, stream.get());
  for (std::size_t index = 0; index < first.size(); ++index) {
    assert(close_enough(DTypeTraits<T>::decode(selected[index]),
                        DTypeTraits<T>::decode(first[index])));
  }
}

} // namespace

int main() {
  int device{};
  assert(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp properties{};
  assert(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
  assert(properties.major == 12 && properties.minor == 0);

  for (const auto shape : {
           raftinfer::CublasLtMatmulShape{1, 1, 1, false, true},
           raftinfer::CublasLtMatmulShape{3, 5, 7, false, true},
           raftinfer::CublasLtMatmulShape{17, 13, 19, false, true},
           raftinfer::CublasLtMatmulShape{5, 7, 3, true, false},
       }) {
    check_shape<__half>(shape);
    check_shape<__nv_bfloat16>(shape);
    check_f32_output_shape<__half>(shape);
    check_f32_output_shape<__nv_bfloat16>(shape);
  }
  check_validation();
  check_run_validation();
  check_deterministic_workspace_validation();
  check_candidate_selection_shape<__half>();
  check_candidate_selection_shape<__nv_bfloat16>();
}
