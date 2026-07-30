#include "cublaslt_matmul.hpp"

#include <cublasLt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace raftinfer {
namespace {

[[noreturn]] void throw_status(const char *operation, cublasStatus_t status) {
  throw CublasLtMatmulError(std::string(operation) +
                            " failed with cuBLAS status " +
                            std::to_string(static_cast<int>(status)));
}

void check_status(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw_status(operation, status);
}

void require(bool condition, const char *message) {
  if (!condition)
    throw CublasLtMatmulError(message);
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char *message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw CublasLtMatmulError(message);
  }
  return lhs * rhs;
}

std::size_t dtype_size(RaftInferDataType dtype) {
  switch (dtype) {
  case RAFTINFER_DTYPE_F32:
    return 4;
  case RAFTINFER_DTYPE_F16:
  case RAFTINFER_DTYPE_BF16:
    return 2;
  default:
    throw CublasLtMatmulError(
        "cuBLASLt matmul supports only F32, F16, and BF16 tensors");
  }
}

cudaDataType_t cuda_dtype(RaftInferDataType dtype) {
  switch (dtype) {
  case RAFTINFER_DTYPE_F32:
    return CUDA_R_32F;
  case RAFTINFER_DTYPE_F16:
    return CUDA_R_16F;
  case RAFTINFER_DTYPE_BF16:
    return CUDA_R_16BF;
  default:
    throw CublasLtMatmulError(
        "cuBLASLt matmul supports only F32, F16, and BF16 tensors");
  }
}

void validate_order(CublasLtMatrixOrder order) {
  require(order == CublasLtMatrixOrder::RowMajor,
          "cuBLASLt matmul requires contiguous row-major matrices");
}

void require_aligned(const void *pointer, std::size_t alignment,
                     const char *message) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  require(address % alignment == 0, message);
}

class BestEffortDeviceGuard {
public:
  explicit BestEffortDeviceGuard(int target_device) noexcept {
    if (target_device < 0 || cudaGetDevice(&previous_device_) != cudaSuccess)
      return;
    if (previous_device_ == target_device) {
      selected_target_ = true;
      return;
    }
    if (cudaSetDevice(target_device) != cudaSuccess)
      return;
    selected_target_ = true;
    switched_device_ = true;
  }

  ~BestEffortDeviceGuard() noexcept {
    if (switched_device_)
      (void)cudaSetDevice(previous_device_);
  }

  BestEffortDeviceGuard(const BestEffortDeviceGuard &) = delete;
  BestEffortDeviceGuard &operator=(const BestEffortDeviceGuard &) = delete;

  bool selected_target() const noexcept { return selected_target_; }

private:
  int previous_device_{-1};
  bool selected_target_{};
  bool switched_device_{};
};

struct PhysicalShape {
  std::uint64_t input_rows;
  std::uint64_t input_cols;
  std::uint64_t weight_rows;
  std::uint64_t weight_cols;
  std::uint64_t output_rows;
  std::uint64_t output_cols;
};

PhysicalShape physical_shape(const CublasLtMatmulShape &shape) {
  detail::validate_cublaslt_shape(shape);

  return PhysicalShape{
      .input_rows =
          static_cast<std::uint64_t>(shape.transpose_input ? shape.k : shape.m),
      .input_cols =
          static_cast<std::uint64_t>(shape.transpose_input ? shape.m : shape.k),
      .weight_rows = static_cast<std::uint64_t>(
          shape.transpose_weight ? shape.n : shape.k),
      .weight_cols = static_cast<std::uint64_t>(
          shape.transpose_weight ? shape.k : shape.n),
      .output_rows = static_cast<std::uint64_t>(shape.m),
      .output_cols = static_cast<std::uint64_t>(shape.n),
  };
}

void destroy_layout(cublasLtMatrixLayout_t &layout) noexcept {
  if (layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(layout);
    layout = nullptr;
  }
}

void set_row_major(cublasLtMatrixLayout_t layout) {
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  check_status(cublasLtMatrixLayoutSetAttribute(
                   layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
               "cublasLtMatrixLayoutSetAttribute");
}

void set_minimum_alignment(cublasLtMatmulPreference_t preference,
                           cublasLtMatmulPreferenceAttributes_t attribute,
                           std::uint32_t bytes) {
  check_status(cublasLtMatmulPreferenceSetAttribute(preference, attribute,
                                                    &bytes, sizeof(bytes)),
               "cublasLtMatmulPreferenceSetAttribute");
}

constexpr std::size_t kMaxCublasLtCandidates = 16;

int cublaslt_algorithm_config_id(const cublasLtMatmulAlgo_t &algorithm) {
  int id = -1;
  std::size_t written = 0;
  check_status(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,
                                                    CUBLASLT_ALGO_CONFIG_ID,
                                                    &id, sizeof(id), &written),
               "cublasLtMatmulAlgoConfigGetAttribute");
  require(written == sizeof(id) && id >= 0,
          "cuBLASLt returned an invalid algorithm id");
  return id;
}

class CudaEvent {
public:
  CudaEvent() {
    require(cudaEventCreateWithFlags(&event_, cudaEventDefault) == cudaSuccess,
            "cuBLASLt tuning event creation failed");
  }

  ~CudaEvent() noexcept {
    if (event_ != nullptr)
      (void)cudaEventDestroy(event_);
  }

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;

  cudaEvent_t get() const noexcept { return event_; }

private:
  cudaEvent_t event_{};
};

} // namespace

void detail::validate_cublaslt_shape(const CublasLtMatmulShape &shape) {
  require(shape.m != 0 && shape.n != 0 && shape.k != 0,
          "cuBLASLt matmul dimensions must be non-zero");
  const auto descriptor_limit =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  require(shape.m <= descriptor_limit && shape.n <= descriptor_limit &&
              shape.k <= descriptor_limit,
          "cuBLASLt matmul dimensions exceed int32 descriptor range");
}

void detail::validate_cublaslt_run_buffers(
    const CublasLtRunBuffers &buffers,
    const CublasLtBufferRequirements &requirements) {
  require(buffers.input != nullptr, "cuBLASLt matmul input pointer is null");
  require(buffers.weight != nullptr, "cuBLASLt matmul weight pointer is null");
  require(buffers.output != nullptr, "cuBLASLt matmul output pointer is null");
  require(buffers.input_bytes == requirements.input_bytes,
          "cuBLASLt matmul input byte size mismatch");
  require(buffers.weight_bytes == requirements.weight_bytes,
          "cuBLASLt matmul weight byte size mismatch");
  require(buffers.output_bytes == requirements.output_bytes,
          "cuBLASLt matmul output byte size mismatch");
  require(buffers.workspace_bytes >= requirements.workspace_bytes,
          "cuBLASLt matmul workspace is too small");
  require(requirements.workspace_bytes == 0 || buffers.workspace != nullptr,
          "cuBLASLt matmul workspace pointer is null");
  constexpr std::size_t kBufferAlignment = 16;
  require_aligned(buffers.input, kBufferAlignment,
                  "cuBLASLt matmul input is not 16-byte aligned");
  require_aligned(buffers.weight, kBufferAlignment,
                  "cuBLASLt matmul weight is not 16-byte aligned");
  require_aligned(buffers.output, kBufferAlignment,
                  "cuBLASLt matmul output is not 16-byte aligned");
  if (requirements.workspace_bytes != 0) {
    constexpr std::size_t kWorkspaceAlignment = 256;
    require_aligned(buffers.workspace, kWorkspaceAlignment,
                    "cuBLASLt matmul workspace is not 256-byte aligned");
  }
}

class CublasLtMatmulPlan::Impl {
public:
  ~Impl() noexcept {
    if (output_layout == nullptr && weight_layout == nullptr &&
        input_layout == nullptr && operation == nullptr && handle == nullptr) {
      return;
    }
    BestEffortDeviceGuard device_guard{device_id};
    if (!device_guard.selected_target())
      return;
    destroy_layout(output_layout);
    destroy_layout(weight_layout);
    destroy_layout(input_layout);
    if (operation != nullptr) {
      (void)cublasLtMatmulDescDestroy(operation);
      operation = nullptr;
    }
    if (handle != nullptr) {
      (void)cublasLtDestroy(handle);
      handle = nullptr;
    }
  }

  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t input_layout{};
  cublasLtMatrixLayout_t weight_layout{};
  cublasLtMatrixLayout_t output_layout{};
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t input_bytes{};
  std::size_t weight_bytes{};
  std::size_t output_bytes{};
  std::size_t workspace_bytes{};
  int algorithm_id{-1};
  int device_id{-1};
  CublasLtMatmulConfig config{};
};

CublasLtMatmulPlan::CublasLtMatmulPlan(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

CublasLtMatmulPlan::~CublasLtMatmulPlan() noexcept = default;

std::unique_ptr<CublasLtMatmulPlan>
CublasLtMatmulPlan::create(const CublasLtMatmulConfig &config) {
  validate_order(config.input_order);
  validate_order(config.weight_order);
  validate_order(config.output_order);
  require(config.weight_dtype == config.input_dtype,
          "cuBLASLt matmul input and weight dtypes must match");
  require(config.output_dtype == config.input_dtype ||
              config.output_dtype == RAFTINFER_DTYPE_F32,
          "cuBLASLt matmul output must match inputs or be F32");
  const std::size_t input_element_bytes = dtype_size(config.input_dtype);
  const std::size_t weight_element_bytes = dtype_size(config.weight_dtype);
  (void)cuda_dtype(config.output_dtype);
  const PhysicalShape physical = physical_shape(config.shape);

  auto impl = std::make_unique<Impl>();
  impl->input_bytes =
      checked_mul(checked_mul(config.shape.m, config.shape.k,
                              "cuBLASLt input element count overflow"),
                  input_element_bytes, "cuBLASLt input byte size overflow");
  impl->weight_bytes =
      checked_mul(checked_mul(config.shape.k, config.shape.n,
                              "cuBLASLt weight element count overflow"),
                  weight_element_bytes, "cuBLASLt weight byte size overflow");
  impl->output_bytes = checked_mul(
      checked_mul(config.shape.m, config.shape.n,
                  "cuBLASLt output element count overflow"),
      dtype_size(config.output_dtype), "cuBLASLt output byte size overflow");
  impl->config = config;

  const cudaError_t device_error = cudaGetDevice(&impl->device_id);
  require(device_error == cudaSuccess,
          "cuBLASLt matmul could not query the current CUDA device");
  cudaDeviceProp properties{};
  const cudaError_t properties_error =
      cudaGetDeviceProperties(&properties, impl->device_id);
  require(properties_error == cudaSuccess,
          "cuBLASLt matmul could not query CUDA device properties");
  require(properties.major == 12 && properties.minor == 0,
          "cuBLASLt matmul requires an RTX 50-class sm_120 device");

  check_status(cublasLtCreate(&impl->handle), "cublasLtCreate");
  check_status(cublasLtMatmulDescCreate(&impl->operation, CUBLAS_COMPUTE_32F,
                                        CUDA_R_32F),
               "cublasLtMatmulDescCreate");
  const cublasOperation_t transpose_input =
      config.shape.transpose_input ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transpose_weight =
      config.shape.transpose_weight ? CUBLAS_OP_T : CUBLAS_OP_N;
  check_status(cublasLtMatmulDescSetAttribute(
                   impl->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                   &transpose_input, sizeof(transpose_input)),
               "cublasLtMatmulDescSetAttribute(TRANSA)");
  check_status(cublasLtMatmulDescSetAttribute(
                   impl->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                   &transpose_weight, sizeof(transpose_weight)),
               "cublasLtMatmulDescSetAttribute(TRANSB)");

  check_status(cublasLtMatrixLayoutCreate(
                   &impl->input_layout, cuda_dtype(config.input_dtype),
                   physical.input_rows, physical.input_cols,
                   static_cast<std::int64_t>(physical.input_cols)),
               "cublasLtMatrixLayoutCreate(input)");
  check_status(cublasLtMatrixLayoutCreate(
                   &impl->weight_layout, cuda_dtype(config.weight_dtype),
                   physical.weight_rows, physical.weight_cols,
                   static_cast<std::int64_t>(physical.weight_cols)),
               "cublasLtMatrixLayoutCreate(weight)");
  check_status(cublasLtMatrixLayoutCreate(
                   &impl->output_layout, cuda_dtype(config.output_dtype),
                   physical.output_rows, physical.output_cols,
                   static_cast<std::int64_t>(physical.output_cols)),
               "cublasLtMatrixLayoutCreate(output)");
  set_row_major(impl->input_layout);
  set_row_major(impl->weight_layout);
  set_row_major(impl->output_layout);

  cublasLtMatmulPreference_t preference{};
  check_status(cublasLtMatmulPreferenceCreate(&preference),
               "cublasLtMatmulPreferenceCreate");
  try {
    const std::uint64_t workspace_budget = config.workspace_budget_bytes;
    check_status(cublasLtMatmulPreferenceSetAttribute(
                     preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_budget, sizeof(workspace_budget)),
                 "cublasLtMatmulPreferenceSetAttribute(workspace)");
    constexpr std::uint32_t kBufferAlignment = 16;
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                          kBufferAlignment);
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                          kBufferAlignment);
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                          kBufferAlignment);
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                          kBufferAlignment);

    std::vector<cublasLtMatmulHeuristicResult_t> results(1);
    int returned_results = 0;
    check_status(cublasLtMatmulAlgoGetHeuristic(
                     impl->handle, impl->operation, impl->input_layout,
                     impl->weight_layout, impl->output_layout,
                     impl->output_layout, preference,
                     static_cast<int>(results.size()), results.data(),
                     &returned_results),
                 "cublasLtMatmulAlgoGetHeuristic");
    require(returned_results == 1 &&
                results.front().state == CUBLAS_STATUS_SUCCESS,
            "cuBLASLt found no matmul algorithm within workspace budget");
    require(results.front().workspaceSize <= config.workspace_budget_bytes,
            "cuBLASLt heuristic exceeded the workspace budget");
    impl->algorithm = results.front().algo;
    impl->workspace_bytes = results.front().workspaceSize;
  } catch (...) {
    (void)cublasLtMatmulPreferenceDestroy(preference);
    throw;
  }
  check_status(cublasLtMatmulPreferenceDestroy(preference),
               "cublasLtMatmulPreferenceDestroy");

  impl->algorithm_id = cublaslt_algorithm_config_id(impl->algorithm);

  return std::unique_ptr<CublasLtMatmulPlan>(
      new CublasLtMatmulPlan(std::move(impl)));
}

std::vector<CublasLtCandidate>
enumerate_cublaslt_candidates(const CublasLtMatmulConfig &config,
                              std::size_t maximum) {
  require(maximum > 0, "cuBLASLt candidate maximum must be non-zero");
  auto plan = CublasLtMatmulPlan::create(config);

  cublasLtMatmulPreference_t preference{};
  check_status(cublasLtMatmulPreferenceCreate(&preference),
               "cublasLtMatmulPreferenceCreate");
  try {
    const std::uint64_t workspace_budget = config.workspace_budget_bytes;
    check_status(cublasLtMatmulPreferenceSetAttribute(
                     preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_budget, sizeof(workspace_budget)),
                 "cublasLtMatmulPreferenceSetAttribute(workspace)");
    constexpr std::uint32_t kBufferAlignment = 16;
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                          kBufferAlignment);
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                          kBufferAlignment);
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                          kBufferAlignment);
    set_minimum_alignment(preference,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                          kBufferAlignment);

    const std::size_t bounded_maximum =
        std::min<std::size_t>(maximum, kMaxCublasLtCandidates);
    std::vector<cublasLtMatmulHeuristicResult_t> results(bounded_maximum);
    int returned_results = 0;
    check_status(cublasLtMatmulAlgoGetHeuristic(
                     plan->impl_->handle, plan->impl_->operation,
                     plan->impl_->input_layout, plan->impl_->weight_layout,
                     plan->impl_->output_layout, plan->impl_->output_layout,
                     preference, static_cast<int>(results.size()),
                     results.data(), &returned_results),
                 "cublasLtMatmulAlgoGetHeuristic");
    std::vector<CublasLtCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(returned_results));
    for (int index = 0; index < returned_results; ++index) {
      const auto &result = results[static_cast<std::size_t>(index)];
      if (result.state != CUBLAS_STATUS_SUCCESS ||
          result.workspaceSize > config.workspace_budget_bytes) {
        continue;
      }
      candidates.push_back(CublasLtCandidate{
          .algorithm = result.algo,
          .algorithm_id = cublaslt_algorithm_config_id(result.algo),
          .workspace_bytes = result.workspaceSize,
      });
    }
    check_status(cublasLtMatmulPreferenceDestroy(preference),
                 "cublasLtMatmulPreferenceDestroy");
    return candidates;
  } catch (...) {
    (void)cublasLtMatmulPreferenceDestroy(preference);
    throw;
  }
}

std::size_t CublasLtMatmulPlan::input_bytes() const noexcept {
  return impl_->input_bytes;
}

std::size_t CublasLtMatmulPlan::weight_bytes() const noexcept {
  return impl_->weight_bytes;
}

std::size_t CublasLtMatmulPlan::output_bytes() const noexcept {
  return impl_->output_bytes;
}

std::size_t CublasLtMatmulPlan::workspace_bytes() const noexcept {
  return impl_->workspace_bytes;
}

int CublasLtMatmulPlan::algorithm_id() const noexcept {
  return impl_->algorithm_id;
}

void CublasLtMatmulPlan::run(cudaStream_t stream, const void *input,
                             std::size_t input_bytes, const void *weight,
                             std::size_t weight_bytes, void *output,
                             std::size_t output_bytes, void *workspace,
                             std::size_t workspace_bytes) const {
  detail::validate_cublaslt_run_buffers(
      detail::CublasLtRunBuffers{
          .input = input,
          .input_bytes = input_bytes,
          .weight = weight,
          .weight_bytes = weight_bytes,
          .output = output,
          .output_bytes = output_bytes,
          .workspace = workspace,
          .workspace_bytes = workspace_bytes,
      },
      detail::CublasLtBufferRequirements{
          .input_bytes = impl_->input_bytes,
          .weight_bytes = impl_->weight_bytes,
          .output_bytes = impl_->output_bytes,
          .workspace_bytes = impl_->workspace_bytes,
      });

  int current_device = -1;
  require(cudaGetDevice(&current_device) == cudaSuccess &&
              current_device == impl_->device_id,
          "cuBLASLt matmul plan used on a different CUDA device");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check_status(cublasLtMatmul(impl_->handle, impl_->operation, &alpha, input,
                              impl_->input_layout, weight, impl_->weight_layout,
                              &beta, output, impl_->output_layout, output,
                              impl_->output_layout, &impl_->algorithm,
                              workspace, impl_->workspace_bytes, stream),
               "cublasLtMatmul");
}

void CublasLtMatmulPlan::select_fastest(cudaStream_t stream, const void *input,
                                        const void *weight, void *output,
                                        void *workspace,
                                        std::size_t workspace_bytes,
                                        std::uint32_t warmups,
                                        std::uint32_t measurements) {
  require(measurements > 0,
          "cuBLASLt tuning requires at least one measurement");
  detail::validate_cublaslt_run_buffers(
      detail::CublasLtRunBuffers{
          .input = input,
          .input_bytes = impl_->input_bytes,
          .weight = weight,
          .weight_bytes = impl_->weight_bytes,
          .output = output,
          .output_bytes = impl_->output_bytes,
          .workspace = workspace,
          .workspace_bytes = workspace_bytes,
      },
      detail::CublasLtBufferRequirements{
          .input_bytes = impl_->input_bytes,
          .weight_bytes = impl_->weight_bytes,
          .output_bytes = impl_->output_bytes,
          .workspace_bytes = workspace_bytes,
      });

  int current_device = -1;
  require(cudaGetDevice(&current_device) == cudaSuccess &&
              current_device == impl_->device_id,
          "cuBLASLt matmul plan used on a different CUDA device");

  const auto candidates = enumerate_cublaslt_candidates(impl_->config, 16);
  require(!candidates.empty(),
          "cuBLASLt found no matmul tuning candidates within workspace budget");
  CudaEvent start;
  CudaEvent stop;
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  std::vector<float> timings;
  timings.reserve(measurements);
  CublasLtCandidate best = candidates.front();
  float best_median = std::numeric_limits<float>::infinity();

  for (const auto &candidate : candidates) {
    if (candidate.workspace_bytes > workspace_bytes)
      continue;
    for (std::uint32_t iteration = 0; iteration < warmups; ++iteration) {
      check_status(cublasLtMatmul(impl_->handle, impl_->operation, &alpha,
                                  input, impl_->input_layout, weight,
                                  impl_->weight_layout, &beta, output,
                                  impl_->output_layout, output,
                                  impl_->output_layout, &candidate.algorithm,
                                  workspace, candidate.workspace_bytes, stream),
                   "cublasLtMatmul(tuning warmup)");
    }
    timings.clear();
    for (std::uint32_t iteration = 0; iteration < measurements; ++iteration) {
      require(cudaEventRecord(start.get(), stream) == cudaSuccess,
              "cuBLASLt tuning start event record failed");
      check_status(cublasLtMatmul(impl_->handle, impl_->operation, &alpha,
                                  input, impl_->input_layout, weight,
                                  impl_->weight_layout, &beta, output,
                                  impl_->output_layout, output,
                                  impl_->output_layout, &candidate.algorithm,
                                  workspace, candidate.workspace_bytes, stream),
                   "cublasLtMatmul(tuning measurement)");
      require(cudaEventRecord(stop.get(), stream) == cudaSuccess,
              "cuBLASLt tuning stop event record failed");
      require(cudaEventSynchronize(stop.get()) == cudaSuccess,
              "cuBLASLt tuning event synchronization failed");
      float milliseconds = 0.0F;
      require(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()) ==
                  cudaSuccess,
              "cuBLASLt tuning event timing failed");
      timings.push_back(milliseconds);
    }
    std::sort(timings.begin(), timings.end());
    const float median = timings[timings.size() / 2];
    if (median < best_median) {
      best_median = median;
      best = candidate;
    }
  }
  require(best_median < std::numeric_limits<float>::infinity(),
          "cuBLASLt tuning candidates exceeded caller workspace");
  impl_->algorithm = best.algorithm;
  impl_->algorithm_id = best.algorithm_id;
  impl_->workspace_bytes = best.workspace_bytes;
}

} // namespace raftinfer
