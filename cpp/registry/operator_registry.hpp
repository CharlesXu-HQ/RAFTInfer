#pragma once

#include <brt/tensor.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brt {

enum class OperatorKind {
  Matmul,
};

enum class ExecutionRegime {
  HostDispatch,
  CudaGraph,
};

struct TensorSignature {
  BrtDataType dtype{};
  BrtQuantFormat quant{};
  std::uint32_t rank{};
  std::size_t alignment{};
  std::vector<std::int64_t> shape;

  bool operator==(const TensorSignature&) const = default;
};

struct TensorConstraint {
  BrtDataType dtype{};
  BrtQuantFormat quant{};
  std::uint32_t rank{};
  std::size_t alignment{};
  std::vector<std::int64_t> min_shape;
  std::vector<std::int64_t> max_shape;
};

struct OperatorSignature {
  OperatorKind op{};
  ExecutionRegime regime{};
  int arch_major{};
  int arch_minor{};
  bool graph_capture{};
  bool deterministic{};
  std::size_t workspace_bytes{};
  std::vector<TensorSignature> inputs;

  bool operator==(const OperatorSignature&) const = default;
};

}  // namespace brt

template <>
struct std::hash<brt::TensorSignature> {
  std::size_t operator()(const brt::TensorSignature& signature) const noexcept;
};

template <>
struct std::hash<brt::OperatorSignature> {
  std::size_t operator()(const brt::OperatorSignature& signature) const noexcept;
};

namespace brt {

struct KernelCapability {
  std::string name;
  OperatorKind op{};
  ExecutionRegime regime{};
  int min_arch_major{};
  int min_arch_minor{};
  int max_arch_major{};
  int max_arch_minor{};
  std::vector<TensorConstraint> inputs;
  bool graph_safe{};
  bool deterministic{};
  std::size_t workspace_bytes{};
  int priority{};

  std::optional<std::string> rejection_reason(const OperatorSignature& signature) const;
  bool matches(const OperatorSignature& signature) const {
    return !rejection_reason(signature).has_value();
  }
};

struct KernelRegistration {
  std::string name;
  KernelCapability capability;
};

struct KernelRejection {
  std::string kernel_name;
  std::string reason;
};

class DispatchError : public std::runtime_error {
 public:
  explicit DispatchError(std::vector<KernelRejection> rejections);

  const std::vector<KernelRejection>& rejections() const noexcept { return rejections_; }

 private:
  std::vector<KernelRejection> rejections_;
};

struct DispatchResult {
  std::optional<std::reference_wrapper<const KernelRegistration>> registration;
};

class OperatorRegistry {
 public:
  const KernelRegistration& register_kernel(KernelCapability capability);
  DispatchResult resolve(const OperatorSignature& signature) const;

 private:
  std::deque<KernelRegistration> registrations_;
  std::unordered_set<std::string> names_;
  mutable std::unordered_map<OperatorSignature, const KernelRegistration*> cache_;
};

}  // namespace brt
