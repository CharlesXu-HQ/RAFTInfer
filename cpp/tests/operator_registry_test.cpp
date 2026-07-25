#include "../registry/operator_registry.hpp"

#include <brt/tensor.h>

#include "assert_enabled.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

brt::TensorSignature tensor(
    BrtDataType dtype = BRT_DTYPE_F32,
    BrtQuantFormat quant = BRT_QUANT_NONE,
    std::uint32_t rank = 2,
    std::size_t alignment = 16,
    std::vector<std::int64_t> shape = {4, 16}) {
  return brt::TensorSignature{dtype, quant, rank, alignment, shape};
}

brt::TensorConstraint constraint(
    BrtDataType dtype = BRT_DTYPE_F32,
    BrtQuantFormat quant = BRT_QUANT_NONE,
    std::uint32_t rank = 2,
    std::size_t alignment = 16,
    std::vector<std::int64_t> min_shape = {1, 1},
    std::vector<std::int64_t> max_shape = {4096, 4096}) {
  return brt::TensorConstraint{
      dtype, quant, rank, alignment, std::move(min_shape), std::move(max_shape)};
}

brt::OperatorSignature signature(
    bool graph_capture,
    bool deterministic,
    std::size_t workspace_bytes,
    brt::OperatorKind op = brt::OperatorKind::Matmul,
    brt::CudaArchitecture architecture = brt::CudaArchitecture::Sm120a,
    brt::ShapeBucket shape_bucket = brt::ShapeBucket::Decode,
    BrtDataType dtype = BRT_DTYPE_F32,
    BrtQuantFormat quant = BRT_QUANT_NONE,
    std::uint32_t rank = 2,
    std::size_t alignment = 16,
    std::vector<std::int64_t> shape = {4, 16}) {
  return brt::OperatorSignature{
      op,
      brt::ExecutionRegime::HostDispatch,
      architecture,
      shape_bucket,
      graph_capture,
      deterministic,
      workspace_bytes,
      {tensor(dtype, quant, rank, alignment, shape)}};
}

brt::OperatorSignature two_input_signature(
    BrtDataType second_dtype = BRT_DTYPE_F32,
    BrtQuantFormat second_quant = BRT_QUANT_NONE,
    std::uint32_t second_rank = 2,
    std::size_t second_alignment = 16,
    std::vector<std::int64_t> second_shape = {4, 16}) {
  brt::OperatorSignature result = signature(false, true, 0);
  result.inputs.push_back(
      tensor(second_dtype, second_quant, second_rank, second_alignment, second_shape));
  return result;
}

brt::KernelCapability capability(
    std::string name,
    int priority,
    bool graph_safe,
    bool deterministic,
    std::size_t workspace_bytes,
    BrtDataType dtype = BRT_DTYPE_F32,
    BrtQuantFormat quant = BRT_QUANT_NONE,
    std::uint32_t rank = 2,
    std::size_t alignment = 16,
    std::vector<std::int64_t> min_shape = {1, 1},
    std::vector<std::int64_t> max_shape = {4096, 4096},
    brt::OperatorKind op = brt::OperatorKind::Matmul,
    brt::CudaArchitecture architecture = brt::CudaArchitecture::Sm120a,
    brt::ShapeBucket shape_bucket = brt::ShapeBucket::Decode,
    brt::KernelProvenance provenance = brt::KernelProvenance::ProjectNative) {
  return brt::KernelCapability{
      std::move(name),
      op,
      brt::ExecutionRegime::HostDispatch,
      architecture,
      shape_bucket,
      provenance,
      {constraint(dtype, quant, rank, alignment, std::move(min_shape), std::move(max_shape))},
      graph_safe,
      deterministic,
      workspace_bytes,
      priority};
}

brt::KernelCapability two_input_capability(
    std::string name,
    int priority,
    brt::TensorConstraint second_constraint) {
  return brt::KernelCapability{
      std::move(name),
      brt::OperatorKind::Matmul,
      brt::ExecutionRegime::HostDispatch,
      brt::CudaArchitecture::Sm120a,
      brt::ShapeBucket::Decode,
      brt::KernelProvenance::ProjectNative,
      {constraint(), std::move(second_constraint)},
      true,
      true,
      0,
      priority};
}

std::string reasons_for(const brt::DispatchError& error) {
  std::string joined;
  for (const auto& rejection : error.rejections()) {
    joined += rejection.kernel_name;
    joined += ":";
    joined += rejection.reason;
    joined += "\n";
  }
  return joined;
}

void expect_contains(const std::string& text, const std::string& needle) {
  assert(text.find(needle) != std::string::npos);
}

}  // namespace

int main() {
  brt::OperatorRegistry registry;
  registry.register_kernel(capability("fast_ordinary", 100, false, false, 4096));
  registry.register_kernel(capability("graph_safe", 50, true, false, 4096));
  registry.register_kernel(capability("no_workspace_fallback", 10, true, true, 0));

  const auto ordinary = registry.resolve(signature(false, false, 4096));
  assert(ordinary.registration->get().name == "fast_ordinary");

  const auto graph = registry.resolve(signature(true, false, 4096));
  assert(graph.registration->get().name == "graph_safe");

  const auto deterministic = registry.resolve(signature(false, true, 0));
  assert(deterministic.registration->get().name == "no_workspace_fallback");

  assert(&registry.resolve(signature(false, false, 4096)).registration.value().get() ==
         &ordinary.registration.value().get());

  brt::OperatorSignature same = signature(false, true, 0);
  brt::OperatorSignature different_alignment = signature(false, true, 0);
  different_alignment.inputs[0].alignment = 32;
  brt::OperatorSignature different_bucket = same;
  different_bucket.shape_bucket = brt::ShapeBucket::Prefill4;
  brt::OperatorSignature different_architecture = same;
  different_architecture.architecture = brt::CudaArchitecture::Sm90;
  assert(same == signature(false, true, 0));
  assert(!(same == different_alignment));
  assert(!(same == different_bucket));
  assert(!(same == different_architecture));

  brt::OperatorRegistry bucket_cache_registry;
  bucket_cache_registry.register_kernel(
      capability("decode_bucket", 10, true, true, 0));
  bucket_cache_registry.register_kernel(capability(
      "prefill_bucket", 10, true, true, 0, BRT_DTYPE_F32, BRT_QUANT_NONE, 2,
      16, {1, 1}, {4096, 4096}, brt::OperatorKind::Matmul,
      brt::CudaArchitecture::Sm120a, brt::ShapeBucket::Prefill4));
  assert(bucket_cache_registry.resolve(signature(false, true, 0))
             .registration->get()
             .name == "decode_bucket");
  assert(bucket_cache_registry
             .resolve(signature(false, true, 0, brt::OperatorKind::Matmul,
                                brt::CudaArchitecture::Sm120a,
                                brt::ShapeBucket::Prefill4))
             .registration->get()
             .name == "prefill_bucket");
  assert(bucket_cache_registry.resolve(signature(false, true, 0))
             .registration->get()
             .name == "decode_bucket");

  bool duplicate_failed = false;
  try {
    registry.register_kernel(capability("graph_safe", 40, true, true, 0));
  } catch (const std::invalid_argument&) {
    duplicate_failed = true;
  }
  assert(duplicate_failed);

  const auto expect_invalid_capability =
      [](brt::KernelCapability invalid, const std::string& expected_reason) {
        brt::OperatorRegistry invalid_registry;
        bool rejected = false;
        try {
          invalid_registry.register_kernel(std::move(invalid));
        } catch (const std::invalid_argument& error) {
          rejected = true;
          expect_contains(error.what(), expected_reason);
        }
        assert(rejected);
      };

  brt::KernelCapability missing_operator =
      capability("missing_operator", 0, true, true, 0);
  missing_operator.op = brt::OperatorKind::Unspecified;
  expect_invalid_capability(std::move(missing_operator), "operator kind");

  brt::KernelCapability missing_architecture =
      capability("missing_architecture", 0, true, true, 0);
  missing_architecture.architecture = brt::CudaArchitecture::Unspecified;
  expect_invalid_capability(std::move(missing_architecture), "architecture");

  brt::KernelCapability non_rtx50_architecture =
      capability("non_rtx50_architecture", 0, true, true, 0);
  non_rtx50_architecture.architecture = brt::CudaArchitecture::Sm90;
  expect_invalid_capability(std::move(non_rtx50_architecture), "sm_120a");

  brt::KernelCapability missing_bucket =
      capability("missing_bucket", 0, true, true, 0);
  missing_bucket.shape_bucket = brt::ShapeBucket::Unspecified;
  expect_invalid_capability(std::move(missing_bucket), "shape bucket");

  brt::KernelCapability missing_provenance =
      capability("missing_provenance", 0, true, true, 0);
  missing_provenance.provenance = brt::KernelProvenance::Unspecified;
  expect_invalid_capability(std::move(missing_provenance), "provenance");

  brt::OperatorRegistry nondeterministic_only;
  nondeterministic_only.register_kernel(capability("fast_ordinary", 100, false, false, 4096));
  nondeterministic_only.register_kernel(capability("graph_safe", 50, true, false, 4096));

  bool deterministic_failed = false;
  try {
    (void)nondeterministic_only.resolve(signature(false, true, 4096));
  } catch (const brt::DispatchError& error) {
    deterministic_failed = true;
    const std::string reasons = reasons_for(error);
    expect_contains(reasons, "fast_ordinary:");
    expect_contains(reasons, "not deterministic");
    expect_contains(reasons, "graph_safe:");
    expect_contains(reasons, "not deterministic");
  }
  assert(deterministic_failed);

  bool unsupported_failed = false;
  try {
    (void)registry.resolve(
        signature(false, false, 4096, brt::OperatorKind::EmbeddingLookup,
                  brt::CudaArchitecture::Sm90, brt::ShapeBucket::Prefill17,
                  BRT_DTYPE_Q4_K, BRT_QUANT_Q4_K, 3, 32, {4, 16, 2}));
  } catch (const brt::DispatchError& error) {
    unsupported_failed = true;
    const std::string reasons = reasons_for(error);
    expect_contains(reasons, "operator kind");
    expect_contains(reasons, "architecture unsupported");
    expect_contains(reasons, "shape bucket unsupported");
    expect_contains(reasons, "dtype");
    expect_contains(reasons, "quantization");
    expect_contains(reasons, "rank");
    expect_contains(reasons, "alignment");
    expect_contains(reasons, "shape");
  }
  assert(unsupported_failed);

  brt::OperatorRegistry stable_order;
  stable_order.register_kernel(capability("z_kernel", 10, true, true, 0));
  stable_order.register_kernel(capability("a_kernel", 10, true, true, 0));
  assert(stable_order.resolve(signature(false, true, 0)).registration->get().name ==
         "a_kernel");

  brt::OperatorRegistry two_input_registry;
  two_input_registry.register_kernel(
      two_input_capability("q4_second_input", 100,
                           constraint(BRT_DTYPE_Q4_K, BRT_QUANT_Q4_K, 2, 32)));
  two_input_registry.register_kernel(
      two_input_capability("f32_second_input", 50, constraint()));
  assert(two_input_registry.resolve(two_input_signature()).registration->get().name ==
         "f32_second_input");

  bool second_input_rejected = false;
  try {
    (void)two_input_registry.resolve(two_input_signature(BRT_DTYPE_BF16));
  } catch (const brt::DispatchError& error) {
    second_input_rejected = true;
    const std::string reasons = reasons_for(error);
    expect_contains(reasons, "q4_second_input:");
    expect_contains(reasons, "input 1 dtype");
    expect_contains(reasons, "f32_second_input:");
    expect_contains(reasons, "input 1 dtype");
  }
  assert(second_input_rejected);

  brt::OperatorRegistry input_count_registry;
  input_count_registry.register_kernel(capability("one_input_only", 10, true, true, 0));
  bool input_count_rejected = false;
  try {
    (void)input_count_registry.resolve(two_input_signature());
  } catch (const brt::DispatchError& error) {
    input_count_rejected = true;
    expect_contains(reasons_for(error), "input count");
  }
  assert(input_count_rejected);

  brt::OperatorRegistry provenance_competition;
  provenance_competition.register_kernel(capability(
      "project_native_baseline", 10, true, true, 0));
  provenance_competition.register_kernel(capability(
      "validated_bw24_candidate", 20, true, true, 0, BRT_DTYPE_F32,
      BRT_QUANT_NONE, 2, 16, {1, 1}, {4096, 4096},
      brt::OperatorKind::Matmul, brt::CudaArchitecture::Sm120a,
      brt::ShapeBucket::Decode, brt::KernelProvenance::UpstreamBw24));
  const auto provenance_winner =
      provenance_competition.resolve(signature(false, true, 0));
  assert(provenance_winner.registration->get().name ==
         "validated_bw24_candidate");
  assert(provenance_winner.registration->get().capability.provenance ==
         brt::KernelProvenance::UpstreamBw24);

  const std::vector<brt::OperatorKind> qwen35_operator_kinds = {
      brt::OperatorKind::EmbeddingLookup,
      brt::OperatorKind::RmsNorm,
      brt::OperatorKind::ResidualAdd,
      brt::OperatorKind::PartialRope,
      brt::OperatorKind::QkNormalize,
      brt::OperatorKind::QkNormRope,
      brt::OperatorKind::SigmoidGate,
      brt::OperatorKind::SwiGlu,
      brt::OperatorKind::CausalAttention,
      brt::OperatorKind::ConvolutionShiftUpdate,
      brt::OperatorKind::RecurrentDeltaUpdate,
      brt::OperatorKind::DeltaOutputNormGate,
      brt::OperatorKind::GatedDeltaNet,
      brt::OperatorKind::Argmax,
  };
  for (std::size_t index = 0; index < qwen35_operator_kinds.size(); ++index) {
    brt::OperatorRegistry qwen35_registry;
    const std::string kernel_name = "qwen35_kernel_" + std::to_string(index);
    qwen35_registry.register_kernel(capability(
        kernel_name, 0, true, true, 0, BRT_DTYPE_BF16, BRT_QUANT_NONE, 2, 16,
        {1, 1}, {17, 4096}, qwen35_operator_kinds[index],
        brt::CudaArchitecture::Sm120a, brt::ShapeBucket::Prefill4,
        brt::KernelProvenance::ProjectNative));
    const auto selected = qwen35_registry.resolve(
        signature(true, true, 0, qwen35_operator_kinds[index],
                  brt::CudaArchitecture::Sm120a, brt::ShapeBucket::Prefill4,
                  BRT_DTYPE_BF16));
    assert(selected.registration->get().name == kernel_name);
    assert(selected.registration->get().capability.provenance ==
           brt::KernelProvenance::ProjectNative);
  }
}
