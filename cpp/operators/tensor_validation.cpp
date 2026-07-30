#include "tensor_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

bool is_known_dtype(RaftInferDataType dtype) {
  switch (dtype) {
    case RAFTINFER_DTYPE_F32:
    case RAFTINFER_DTYPE_F16:
    case RAFTINFER_DTYPE_BF16:
    case RAFTINFER_DTYPE_Q4_K:
      return true;
  }
  return false;
}

bool is_known_quant(RaftInferQuantFormat quant) {
  switch (quant) {
    case RAFTINFER_QUANT_NONE:
    case RAFTINFER_QUANT_Q4_K:
      return true;
  }
  return false;
}

bool is_known_memory(RaftInferMemoryType memory) {
  switch (memory) {
    case RAFTINFER_MEMORY_HOST:
    case RAFTINFER_MEMORY_CUDA_DEVICE:
      return true;
  }
  return false;
}

std::size_t element_size(RaftInferDataType dtype) {
  switch (dtype) {
    case RAFTINFER_DTYPE_F32:
      return 4;
    case RAFTINFER_DTYPE_F16:
    case RAFTINFER_DTYPE_BF16:
      return 2;
    case RAFTINFER_DTYPE_Q4_K:
      break;
  }
  throw std::invalid_argument("dtype has no unquantized element size");
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::invalid_argument(message);
  }
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::invalid_argument(message);
  }
  return lhs * rhs;
}

void validate_common_metadata(const RaftInferTensorDesc& desc) {
  if (desc.data == nullptr) {
    throw std::invalid_argument("tensor data is required");
  }
  if (desc.rank == 0 || desc.rank > 4) {
    throw std::invalid_argument("tensor rank must be between 1 and 4");
  }
  if (!is_known_dtype(desc.dtype)) {
    throw std::invalid_argument("unknown tensor dtype");
  }
  if (!is_known_quant(desc.quant)) {
    throw std::invalid_argument("unknown tensor quantization format");
  }
  if (!is_known_memory(desc.memory)) {
    throw std::invalid_argument("unknown tensor memory type");
  }
  for (std::uint32_t index = 0; index < desc.rank; ++index) {
    if (desc.shape[index] <= 0) {
      throw std::invalid_argument("tensor dimensions must be positive");
    }
  }
}

void validate_unquantized_tensor(const RaftInferTensorDesc& desc) {
  if (desc.quant != RAFTINFER_QUANT_NONE) {
    throw std::invalid_argument("unquantized dtype requires RAFTINFER_QUANT_NONE");
  }
  const std::size_t bytes_per_element = element_size(desc.dtype);
  std::size_t required_bytes = bytes_per_element;
  for (std::uint32_t index = 0; index < desc.rank; ++index) {
    if (desc.strides[index] <= 0) {
      throw std::invalid_argument("unquantized tensor strides must be positive");
    }
    const auto extent = static_cast<std::size_t>(desc.shape[index] - 1);
    const auto stride = static_cast<std::size_t>(desc.strides[index]);
    const auto dim_bytes = checked_mul(extent, stride, "tensor byte range overflow");
    required_bytes = checked_add(required_bytes, dim_bytes, "tensor byte range overflow");
  }
  if (desc.byte_size < required_bytes) {
    throw std::invalid_argument("tensor byte_size is smaller than shape and strides require");
  }
}

void validate_quantized_tensor(const RaftInferTensorDesc& desc) {
  if (desc.scales == nullptr) {
    throw std::invalid_argument("quantized tensor scales are required");
  }
  if (desc.byte_size == 0) {
    throw std::invalid_argument("quantized tensor byte_size must be non-zero");
  }
  if (desc.dtype != RAFTINFER_DTYPE_Q4_K || desc.quant != RAFTINFER_QUANT_Q4_K) {
    throw std::invalid_argument("unsupported quantized tensor format");
  }
}

}  // namespace

namespace raftinfer {

void validate_tensor_desc(const RaftInferTensorDesc& desc) {
  validate_common_metadata(desc);
  if (desc.quant == RAFTINFER_QUANT_NONE) {
    validate_unquantized_tensor(desc);
    return;
  }
  validate_quantized_tensor(desc);
}

}  // namespace raftinfer
