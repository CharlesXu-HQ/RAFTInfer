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
    int arch_major = 10,
    int arch_minor = 0,
    BrtDataType dtype = BRT_DTYPE_F32,
    BrtQuantFormat quant = BRT_QUANT_NONE,
    std::uint32_t rank = 2,
    std::size_t alignment = 16,
    std::vector<std::int64_t> shape = {4, 16}) {
  return brt::OperatorSignature{
      brt::OperatorKind::Matmul,
      brt::ExecutionRegime::HostDispatch,
      arch_major,
      arch_minor,
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
    std::vector<std::int64_t> max_shape = {4096, 4096}) {
  return brt::KernelCapability{
      std::move(name),
      brt::OperatorKind::Matmul,
      brt::ExecutionRegime::HostDispatch,
      10,
      0,
      10,
      0,
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
      10,
      0,
      10,
      0,
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
  std::hash<brt::OperatorSignature> hash;
  assert(same == signature(false, true, 0));
  assert(!(same == different_alignment));
  assert(hash(same) != hash(different_alignment));

  bool duplicate_failed = false;
  try {
    registry.register_kernel(capability("graph_safe", 40, true, true, 0));
  } catch (const std::invalid_argument&) {
    duplicate_failed = true;
  }
  assert(duplicate_failed);

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
    (void)registry.resolve(signature(false, false, 4096, 9, 0, BRT_DTYPE_Q4_K,
                                     BRT_QUANT_Q4_K, 3, 32, {4, 16, 2}));
  } catch (const brt::DispatchError& error) {
    unsupported_failed = true;
    const std::string reasons = reasons_for(error);
    expect_contains(reasons, "architecture");
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
}
