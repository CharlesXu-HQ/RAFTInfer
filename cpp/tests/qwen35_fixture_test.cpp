#include "../reference/qwen35.hpp"

#include "assert_enabled.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'R', 'I', 'F', 'Q',
                                             '3', '5', 'F', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;

struct Fixture {
  raftinfer::reference::FullAttentionReferenceArgs full{};
  raftinfer::reference::GatedDeltaReferenceArgs delta{};
  std::vector<float> full_input;
  std::vector<float> full_query_norm_weight;
  std::vector<float> full_key_norm_weight;
  std::vector<float> full_output_weight;
  std::vector<float> full_expected_output;
  std::vector<float> delta_input;
  std::vector<float> delta_conv_weight;
  std::vector<float> delta_recurrent_a;
  std::vector<float> delta_dt_bias;
  std::vector<float> delta_output_norm_weight;
  std::vector<float> delta_initial_convolution;
  std::vector<float> delta_initial_recurrent;
  std::vector<float> delta_expected_output;
  std::vector<float> delta_expected_convolution;
  std::vector<float> delta_expected_recurrent;
};

class Reader {
public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  template <class T> T scalar() {
    static_assert(std::is_same_v<T, std::uint32_t> ||
                  std::is_same_v<T, std::uint64_t> || std::is_same_v<T, float>);
    require(sizeof(T));
    std::array<std::uint8_t, sizeof(T)> storage{};
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                sizeof(T), storage.begin());
    offset_ += sizeof(T);
    if constexpr (std::endian::native == std::endian::big) {
      std::reverse(storage.begin(), storage.end());
    }
    T value{};
    std::memcpy(&value, storage.data(), sizeof(T));
    return value;
  }

  std::array<std::uint8_t, 8> magic() {
    require(kMagic.size());
    std::array<std::uint8_t, 8> value{};
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                value.size(), value.begin());
    offset_ += value.size();
    return value;
  }

  std::vector<float> floats() {
    const auto count = scalar<std::uint64_t>();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::invalid_argument("Qwen3.5 fixture vector length overflow");
    }
    const auto byte_count = static_cast<std::size_t>(count) * sizeof(float);
    require(byte_count);
    std::vector<float> values(static_cast<std::size_t>(count));
    for (float &value : values) {
      value = scalar<float>();
    }
    return values;
  }

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
  void require(std::size_t byte_count) const {
    if (byte_count > bytes_.size() - std::min(offset_, bytes_.size())) {
      throw std::invalid_argument("truncated Qwen3.5 fixture");
    }
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t offset_{};
};

Fixture parse_fixture(std::span<const std::uint8_t> bytes) {
  Reader reader(bytes);
  if (reader.magic() != kMagic) {
    throw std::invalid_argument("invalid Qwen3.5 fixture magic");
  }
  if (reader.scalar<std::uint32_t>() != kVersion) {
    throw std::invalid_argument("unsupported Qwen3.5 fixture version");
  }
  if (reader.scalar<std::uint32_t>() != kEndianMarker) {
    throw std::invalid_argument("invalid Qwen3.5 fixture endian marker");
  }
  const auto declared_size = reader.scalar<std::uint64_t>();
  if (declared_size != bytes.size()) {
    throw std::invalid_argument("Qwen3.5 fixture length mismatch");
  }

  Fixture fixture;
  fixture.full.tokens = reader.scalar<std::uint32_t>();
  fixture.full.hidden_size = reader.scalar<std::uint32_t>();
  fixture.full.query_heads = reader.scalar<std::uint32_t>();
  fixture.full.kv_heads = reader.scalar<std::uint32_t>();
  fixture.full.head_dim = reader.scalar<std::uint32_t>();
  fixture.full.rotary_dim = reader.scalar<std::uint32_t>();
  fixture.full.position_offset = reader.scalar<std::uint32_t>();
  fixture.full.rope_base = reader.scalar<float>();
  fixture.full.epsilon = reader.scalar<float>();

  fixture.delta.tokens = reader.scalar<std::uint32_t>();
  fixture.delta.hidden_size = reader.scalar<std::uint32_t>();
  fixture.delta.key_heads = reader.scalar<std::uint32_t>();
  fixture.delta.value_heads = reader.scalar<std::uint32_t>();
  fixture.delta.key_dim = reader.scalar<std::uint32_t>();
  fixture.delta.value_dim = reader.scalar<std::uint32_t>();
  fixture.delta.conv_width = reader.scalar<std::uint32_t>();
  fixture.delta.epsilon = reader.scalar<float>();

  fixture.full_input = reader.floats();
  fixture.full_query_norm_weight = reader.floats();
  fixture.full_key_norm_weight = reader.floats();
  fixture.full_output_weight = reader.floats();
  fixture.full_expected_output = reader.floats();
  fixture.delta_input = reader.floats();
  fixture.delta_conv_weight = reader.floats();
  fixture.delta_recurrent_a = reader.floats();
  fixture.delta_dt_bias = reader.floats();
  fixture.delta_output_norm_weight = reader.floats();
  fixture.delta_initial_convolution = reader.floats();
  fixture.delta_initial_recurrent = reader.floats();
  fixture.delta_expected_output = reader.floats();
  fixture.delta_expected_convolution = reader.floats();
  fixture.delta_expected_recurrent = reader.floats();

  if (reader.offset() != bytes.size()) {
    throw std::invalid_argument("Qwen3.5 fixture has trailing bytes");
  }
  return fixture;
}

std::vector<std::uint8_t> read_fixture_file() {
  std::ifstream stream(RAFTINFER_QWEN35_FIXTURE_PATH, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(std::string("missing Qwen3.5 fixture: ") +
                             RAFTINFER_QWEN35_FIXTURE_PATH);
  }
  stream.seekg(0, std::ios::end);
  const auto end = stream.tellg();
  if (end < 0) {
    throw std::runtime_error("failed to size Qwen3.5 fixture");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream) {
    throw std::runtime_error("failed to read Qwen3.5 fixture");
  }
  return bytes;
}

template <class Fn> void expect_invalid(Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    thrown = true;
  }
  assert(thrown);
}

void store_u32(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint32_t value) {
  assert(offset + sizeof(value) <= bytes.size());
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    bytes[offset + byte] =
        static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU);
  }
}

void store_u64(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint64_t value) {
  assert(offset + sizeof(value) <= bytes.size());
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    bytes[offset + byte] =
        static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU);
  }
}

void expect_near(float actual, float expected, std::size_t index,
                 const char *field) {
  constexpr float tolerance = 1.0e-5F;
  const float absolute = std::fabs(actual - expected);
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  if (absolute > tolerance && absolute > tolerance * scale) {
    std::cerr << field << '[' << index << "] expected " << expected
              << " but got " << actual << ", abs diff " << absolute << '\n';
    assert(false);
  }
}

void expect_vector_near(std::span<const float> actual,
                        std::span<const float> expected, const char *field) {
  assert(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    expect_near(actual[index], expected[index], index, field);
  }
}

} // namespace

int main() {
  const auto bytes = read_fixture_file();
  const Fixture fixture = parse_fixture(bytes);

  {
    auto invalid = bytes;
    invalid.front() ^= 0xffU;
    expect_invalid([&] { (void)parse_fixture(invalid); });
  }
  {
    auto invalid = bytes;
    store_u32(invalid, 8, kVersion + 1);
    expect_invalid([&] { (void)parse_fixture(invalid); });
  }
  {
    auto invalid = bytes;
    store_u64(invalid, 16, bytes.size() + 1);
    expect_invalid([&] { (void)parse_fixture(invalid); });
  }
  {
    auto invalid = bytes;
    invalid.pop_back();
    expect_invalid([&] { (void)parse_fixture(invalid); });
  }

  std::vector<float> full_output(fixture.full.tokens *
                                 fixture.full.hidden_size);
  raftinfer::reference::qwen35_gated_full_attention(
      fixture.full_input, full_output,
      raftinfer::reference::FullAttentionReferenceWeights{
          .query_norm_weight = fixture.full_query_norm_weight,
          .key_norm_weight = fixture.full_key_norm_weight,
          .output_weight = fixture.full_output_weight},
      fixture.full);
  expect_vector_near(full_output, fixture.full_expected_output,
                     "full_expected_output");

  raftinfer::reference::GatedDeltaReferenceState delta_state(fixture.delta);
  assert(delta_state.convolution.size() ==
         fixture.delta_initial_convolution.size());
  assert(delta_state.recurrent.size() ==
         fixture.delta_initial_recurrent.size());
  delta_state.convolution = fixture.delta_initial_convolution;
  delta_state.recurrent = fixture.delta_initial_recurrent;
  std::vector<float> delta_output(fixture.delta.tokens *
                                  fixture.delta.hidden_size);
  raftinfer::reference::qwen35_gated_delta_prefill(
      fixture.delta_input, delta_output,
      raftinfer::reference::GatedDeltaReferenceWeights{
          .conv_weight = fixture.delta_conv_weight,
          .recurrent_a = fixture.delta_recurrent_a,
          .dt_bias = fixture.delta_dt_bias,
          .output_norm_weight = fixture.delta_output_norm_weight},
      fixture.delta, delta_state);
  expect_vector_near(delta_output, fixture.delta_expected_output,
                     "delta_expected_output");
  expect_vector_near(delta_state.convolution,
                     fixture.delta_expected_convolution,
                     "delta_expected_convolution");
  expect_vector_near(delta_state.recurrent, fixture.delta_expected_recurrent,
                     "delta_expected_recurrent");
}
