#pragma once

#include <bit>
#include <cstdint>

namespace brt::reference {

struct bf16_t {
  std::uint16_t bits{};
};

inline float bf16_to_float(bf16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value.bits) << 16U);
}

inline bf16_t float_to_bf16(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7F800000U) == 0x7F800000U) {
    const std::uint16_t sign_and_exp = static_cast<std::uint16_t>((bits >> 16U) & 0xFF80U);
    if ((bits & 0x007FFFFFU) == 0) {
      return bf16_t{sign_and_exp};
    }
    std::uint16_t payload = static_cast<std::uint16_t>((bits >> 16U) & 0x007FU);
    if (payload == 0) {
      payload = 1;
    }
    return bf16_t{static_cast<std::uint16_t>(sign_and_exp | payload)};
  }
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  const std::uint32_t rounded = bits + 0x7FFFU + lsb;
  return bf16_t{static_cast<std::uint16_t>(rounded >> 16U)};
}

}  // namespace brt::reference
