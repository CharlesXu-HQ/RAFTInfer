#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BrtDataType {
  BRT_DTYPE_F32 = 1,
  BRT_DTYPE_F16 = 2,
  BRT_DTYPE_BF16 = 3,
  BRT_DTYPE_Q4_K = 100
} BrtDataType;

typedef enum BrtQuantFormat {
  BRT_QUANT_NONE = 0,
  BRT_QUANT_Q4_K = 1
} BrtQuantFormat;

typedef enum BrtMemoryType {
  BRT_MEMORY_HOST = 0,
  BRT_MEMORY_CUDA_DEVICE = 1
} BrtMemoryType;

typedef struct BrtTensorDesc {
  void* data;
  const void* scales;
  const void* zero_points;
  size_t byte_size;
  int64_t shape[4];
  int64_t strides[4];
  uint32_t rank;
  BrtDataType dtype;
  BrtQuantFormat quant;
  BrtMemoryType memory;
} BrtTensorDesc;

#ifdef __cplusplus
}
#endif
