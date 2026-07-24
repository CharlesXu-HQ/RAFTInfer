#include "../operators/tensor_validation.hpp"

#include <cassert>
#include <stdexcept>

int main() {
  float values[6]{};
  BrtTensorDesc valid{
      values, nullptr, nullptr, 24, {2, 3, 0, 0}, {12, 4, 0, 0},
      2, BRT_DTYPE_F32, BRT_QUANT_NONE, BRT_MEMORY_HOST};
  brt::validate_tensor_desc(valid);

  auto invalid = valid;
  invalid.rank = 5;
  try { brt::validate_tensor_desc(invalid); assert(false); }
  catch (const std::invalid_argument&) {}

  invalid = valid;
  invalid.shape[1] = 0;
  try { brt::validate_tensor_desc(invalid); assert(false); }
  catch (const std::invalid_argument&) {}

  invalid = valid;
  invalid.byte_size = 20;
  try { brt::validate_tensor_desc(invalid); assert(false); }
  catch (const std::invalid_argument&) {}

  unsigned char packed[32]{};
  float scales[2]{};
  BrtTensorDesc quant{
      packed, scales, nullptr, sizeof(packed), {2, 32, 0, 0}, {16, 0, 0, 0},
      2, BRT_DTYPE_Q4_K, BRT_QUANT_Q4_K, BRT_MEMORY_HOST};
  brt::validate_tensor_desc(quant);
}
