#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RaftInferDataType {
  RAFTINFER_DTYPE_F32 = 1,
  RAFTINFER_DTYPE_F16 = 2,
  RAFTINFER_DTYPE_BF16 = 3,
  RAFTINFER_DTYPE_Q4_K = 100
} RaftInferDataType;

typedef enum RaftInferQuantFormat {
  RAFTINFER_QUANT_NONE = 0,
  RAFTINFER_QUANT_Q4_K = 1
} RaftInferQuantFormat;

typedef enum RaftInferMemoryType {
  RAFTINFER_MEMORY_HOST = 0,
  RAFTINFER_MEMORY_CUDA_DEVICE = 1
} RaftInferMemoryType;

typedef struct RaftInferTensorDesc {
  void* data;
  const void* scales;
  const void* zero_points;
  size_t byte_size;
  int64_t shape[4];
  int64_t strides[4];
  uint32_t rank;
  RaftInferDataType dtype;
  RaftInferQuantFormat quant;
  RaftInferMemoryType memory;
} RaftInferTensorDesc;

#ifdef __cplusplus
}
#endif
