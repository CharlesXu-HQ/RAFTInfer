#include "operator_registry.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

namespace {

template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

bool architecture_less(int lhs_major, int lhs_minor, int rhs_major, int rhs_minor) {
  return lhs_major < rhs_major || (lhs_major == rhs_major && lhs_minor < rhs_minor);
}

bool architecture_greater(int lhs_major, int lhs_minor, int rhs_major, int rhs_minor) {
  return lhs_major > rhs_major || (lhs_major == rhs_major && lhs_minor > rhs_minor);
}

void append_reason(std::vector<std::string>& reasons, std::string reason) {
  reasons.push_back(std::move(reason));
}

std::string join_reasons(const std::vector<std::string>& reasons) {
  std::string joined;
  for (std::size_t i = 0; i < reasons.size(); ++i) {
    if (i != 0) {
      joined += "; ";
    }
    joined += reasons[i];
  }
  return joined;
}

std::string build_dispatch_message(const std::vector<brt::KernelRejection>& rejections) {
  std::ostringstream out;
  out << "no matching kernel";
  for (const auto& rejection : rejections) {
    out << "\n" << rejection.kernel_name << ": " << rejection.reason;
  }
  return out.str();
}

}  // namespace

namespace brt {

std::optional<std::string> KernelCapability::rejection_reason(
    const OperatorSignature& signature) const {
  std::vector<std::string> reasons;

  if (signature.op != op) {
    append_reason(reasons, "operator kind mismatch");
  }
  if (signature.regime != regime) {
    append_reason(reasons, "execution regime mismatch");
  }
  if (architecture_less(signature.arch_major, signature.arch_minor, min_arch_major,
                        min_arch_minor) ||
      architecture_greater(signature.arch_major, signature.arch_minor, max_arch_major,
                           max_arch_minor)) {
    append_reason(reasons, "architecture unsupported");
  }
  if (signature.graph_capture && !graph_safe) {
    append_reason(reasons, "not graph safe");
  }
  if (signature.deterministic && !deterministic) {
    append_reason(reasons, "not deterministic");
  }
  if (signature.workspace_bytes < workspace_bytes) {
    append_reason(reasons, "workspace capacity insufficient");
  }
  if (signature.inputs.size() != inputs.size()) {
    append_reason(reasons, "input count mismatch");
  } else {
    for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
      const TensorSignature& input = signature.inputs[input_index];
      const TensorConstraint& constraint = inputs[input_index];
      const std::string prefix = "input " + std::to_string(input_index) + " ";

      if (input.dtype != constraint.dtype) {
        append_reason(reasons, prefix + "dtype unsupported");
      }
      if (input.quant != constraint.quant) {
        append_reason(reasons, prefix + "quantization unsupported");
      }
      if (input.rank != constraint.rank) {
        append_reason(reasons, prefix + "rank unsupported");
      }
      if (input.alignment != constraint.alignment) {
        append_reason(reasons, prefix + "alignment unsupported");
      }
      if (input.shape.size() != constraint.rank ||
          constraint.min_shape.size() != constraint.rank ||
          constraint.max_shape.size() != constraint.rank) {
        append_reason(reasons, prefix + "shape rank unsupported");
      } else {
        for (std::size_t dim = 0; dim < input.shape.size(); ++dim) {
          if (input.shape[dim] < constraint.min_shape[dim] ||
              input.shape[dim] > constraint.max_shape[dim]) {
            append_reason(reasons, prefix + "shape bounds unsupported");
            break;
          }
        }
      }
    }
  }

  if (reasons.empty()) {
    return std::nullopt;
  }
  return join_reasons(reasons);
}

DispatchError::DispatchError(std::vector<KernelRejection> rejections)
    : std::runtime_error(build_dispatch_message(rejections)), rejections_(std::move(rejections)) {}

const KernelRegistration& OperatorRegistry::register_kernel(KernelCapability capability) {
  const auto [_, inserted] = names_.insert(capability.name);
  if (!inserted) {
    throw std::invalid_argument("duplicate kernel registration: " + capability.name);
  }

  registrations_.push_back(KernelRegistration{capability.name, std::move(capability)});
  cache_.clear();
  return registrations_.back();
}

DispatchResult OperatorRegistry::resolve(const OperatorSignature& signature) const {
  if (const auto cached = cache_.find(signature); cached != cache_.end()) {
    return DispatchResult{std::cref(*cached->second)};
  }

  const KernelRegistration* best = nullptr;
  std::vector<KernelRejection> rejections;
  for (const auto& registration : registrations_) {
    if (const auto reason = registration.capability.rejection_reason(signature)) {
      rejections.push_back(KernelRejection{registration.name, *reason});
      continue;
    }
    if (best == nullptr ||
        registration.capability.priority > best->capability.priority ||
        (registration.capability.priority == best->capability.priority &&
         registration.name < best->name)) {
      best = &registration;
    }
  }

  if (best == nullptr) {
    throw DispatchError(std::move(rejections));
  }

  cache_.emplace(signature, best);
  return DispatchResult{std::cref(*best)};
}

}  // namespace brt

std::size_t std::hash<brt::TensorSignature>::operator()(
    const brt::TensorSignature& signature) const noexcept {
  std::size_t seed = 0;
  hash_combine(seed, static_cast<int>(signature.dtype));
  hash_combine(seed, static_cast<int>(signature.quant));
  hash_combine(seed, signature.rank);
  hash_combine(seed, signature.alignment);
  hash_combine(seed, signature.shape.size());
  for (const auto dim : signature.shape) {
    hash_combine(seed, dim);
  }
  return seed;
}

std::size_t std::hash<brt::OperatorSignature>::operator()(
    const brt::OperatorSignature& signature) const noexcept {
  std::size_t seed = 0;
  hash_combine(seed, static_cast<int>(signature.op));
  hash_combine(seed, static_cast<int>(signature.regime));
  hash_combine(seed, signature.arch_major);
  hash_combine(seed, signature.arch_minor);
  hash_combine(seed, signature.graph_capture);
  hash_combine(seed, signature.deterministic);
  hash_combine(seed, signature.workspace_bytes);
  hash_combine(seed, signature.inputs.size());
  for (const auto& input : signature.inputs) {
    hash_combine(seed, std::hash<brt::TensorSignature>{}(input));
  }
  return seed;
}
