#pragma once

#include <brt/tensor.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brt::kernels {

class Qwen35DeltaError : public std::runtime_error {
public:
  explicit Qwen35DeltaError(const std::string &message)
      : std::runtime_error(message) {}
};

struct GatedDeltaShape {
  std::size_t tokens;
  std::size_t hidden_size;
  std::size_t key_heads;
  std::size_t value_heads;
  std::size_t key_dim;
  std::size_t value_dim;
  std::size_t conv_width;
  float epsilon;
};

enum class GatedDeltaSchedule : std::uint8_t {
  register_resident_current,
  register_resident_prefill_sm120,
  register_resident_decode_sm120,
};

struct GatedDeltaLaunchPolicy {
  GatedDeltaSchedule schedule{GatedDeltaSchedule::register_resident_current};
  std::uint32_t warps_per_block{4};
  bool transposed_boundary_state{false};
};

struct GatedDeltaScheduleDiagnostic {
  std::size_t bucket_tokens{};
  std::size_t key_dim{};
  std::size_t value_dim{};
  GatedDeltaSchedule candidate_schedule{
      GatedDeltaSchedule::register_resident_current};
  GatedDeltaSchedule schedule{GatedDeltaSchedule::register_resident_current};
  std::uint32_t warps_per_block{};
  bool transposed_boundary_state{};
  bool candidate_accepted{};
  bool correctness_passed{};
  float current_median_ms{};
  float candidate_median_ms{};
  std::string rejection_reason;

  bool operator==(const GatedDeltaScheduleDiagnostic &) const = default;
};

// Returns the caller-owned FP32 scratch size needed by
// `qwen35_gated_delta`. The workspace holds one convolved q/k/v token and one
// recurrent output token, and may be reused across calls on the same stream.
std::size_t qwen35_gated_delta_workspace_bytes(const GatedDeltaShape &shape);

// Construction-time policy selection hook. Until target-side benchmarking
// proves an sm_120 candidate is both correct and meaningfully faster, the
// project-native register-resident current path remains selected.
GatedDeltaLaunchPolicy
qwen35_gated_delta_select_policy(const GatedDeltaShape &shape,
                                 std::size_t bucket_tokens) noexcept;

GatedDeltaScheduleDiagnostic
qwen35_gated_delta_schedule_diagnostic(const GatedDeltaShape &shape,
                                       std::size_t bucket_tokens) noexcept;

// Input token layout is `[q, k, v, beta, dt, gate]`, matching the FP32
// reference. Input and output use `dtype`; auxiliary weights use
// `weight_dtype`, which may match `dtype` or be FP32. Convolution and recurrent
// session state remain FP32. State and workspace must already be allocated.
// Every typed pointer must satisfy its dtype alignment; FP32 state and
// workspace pointers must satisfy `alignof(float)`.
//
// The call enqueues work only on `stream`. It does not allocate memory, create
// a stream, or synchronize. Tokens are applied in causal order, so the same
// entry point supports prefill, continued prefill, and one-token decode.
void qwen35_gated_delta(const void *input, const void *conv_weight,
                        const void *recurrent_a, const void *dt_bias,
                        const void *output_norm_weight, void *output,
                        float *convolution_state, float *recurrent_state,
                        void *workspace, std::size_t workspace_bytes,
                        GatedDeltaShape shape, BrtDataType dtype,
                        BrtDataType weight_dtype, cudaStream_t stream);

void qwen35_gated_delta(const void *input, const void *conv_weight,
                        const void *recurrent_a, const void *dt_bias,
                        const void *output_norm_weight, void *output,
                        float *convolution_state, float *recurrent_state,
                        void *workspace, std::size_t workspace_bytes,
                        GatedDeltaShape shape, GatedDeltaLaunchPolicy policy,
                        BrtDataType dtype, BrtDataType weight_dtype,
                        cudaStream_t stream);

} // namespace brt::kernels
