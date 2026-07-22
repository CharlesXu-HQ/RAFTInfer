#include <cuda_runtime.h>

#include <cstdint>

__global__ void brt_smoke_kernel(uint32_t* values, uint32_t count) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] = index;
}

void brt_launch_smoke(uint32_t* values, uint32_t count, cudaStream_t stream) {
  brt_smoke_kernel<<<(count + 255) / 256, 256, 0, stream>>>(values, count);
}
