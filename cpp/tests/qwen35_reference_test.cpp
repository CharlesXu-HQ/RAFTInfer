#include "../reference/qwen35.hpp"

#include "assert_enabled.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void expect_near(float actual, float expected, float tolerance = 1.0e-5F) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "expected " << expected << " but got " << actual
              << " with tolerance " << tolerance << "\n";
    assert(false);
  }
}

void expect_vector_near(std::span<const float> actual,
                        std::span<const float> expected,
                        float tolerance = 1.0e-5F) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    expect_near(actual[i], expected[i], tolerance);
  }
}

template <class Fn> void expect_invalid_argument(Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    thrown = true;
  }
  assert(thrown);
}

} // namespace

int main() {
  {
    const std::array<float, 6> input{3.0F, 4.0F, 0.0F, 1.0F, -2.0F, 2.0F};
    // llama.cpp's Qwen3.5 GGUF conversion stores ordinary RMSNorm weights as
    // final multiplicative scales (the HF +1 offset is applied at conversion).
    const std::array<float, 3> weight{1.0F, 2.0F, 0.5F};
    std::array<float, 6> output{};

    raftinfer::reference::qwen35_rms_norm(input, weight, output, 2, 3, 0.0F);

    expect_vector_near(output, std::array<float, 6>{1.0392305F, 2.7712812F,
                                                    0.0F, 0.5773503F,
                                                    -2.3094010F, 0.5773503F});
  }

  {
    const raftinfer::reference::FullAttentionReferenceArgs args{.tokens = 2,
                                                          .hidden_size = 4,
                                                          .query_heads = 2,
                                                          .kv_heads = 1,
                                                          .head_dim = 2,
                                                          .rotary_dim = 2,
                                                          .position_offset = 0,
                                                          .rope_base = 10000.0F,
                                                          .epsilon = 1.0e-6F};
    const std::array<float, 24> input{
        1.0F,  2.0F,  3.0F,  4.0F,   2.0F,   -1.0F, 0.5F,  -0.5F,
        -2.0F, 0.25F, 0.5F,  -0.25F, 1.5F,   -0.5F, -1.0F, 0.75F,
        0.25F, -0.5F, 1.25F, -1.5F,  -0.75F, 0.5F,  1.0F,  -0.25F};
    const std::array<float, 2> query_norm_weight{1.25F, 0.5F};
    const std::array<float, 2> key_norm_weight{0.75F, 1.75F};
    const std::array<float, 16> output_weight{
        1.0F, -0.5F, 0.25F,  0.75F, -1.0F, 0.5F,  1.5F, -0.25F,
        0.5F, 1.0F,  -0.75F, 0.25F, 1.25F, -1.5F, 0.5F, 1.0F};
    std::array<float, 8> output{};

    raftinfer::reference::qwen35_gated_full_attention(
        input, output,
        raftinfer::reference::FullAttentionReferenceWeights{
            .query_norm_weight = query_norm_weight,
            .key_norm_weight = key_norm_weight,
            .output_weight = output_weight},
        args);

    expect_vector_near(output,
                       std::array<float, 8>{0.2226649F, 0.4692524F, -0.7496101F,
                                            -0.0261312F, 0.9759887F, 0.4197111F,
                                            -1.6327481F, 0.2551961F},
                       1.0e-5F);
  }

  {
    const raftinfer::reference::GatedDeltaReferenceArgs args{.tokens = 3,
                                                       .hidden_size = 4,
                                                       .key_heads = 1,
                                                       .value_heads = 2,
                                                       .key_dim = 2,
                                                       .value_dim = 2,
                                                       .conv_width = 4,
                                                       .epsilon = 1.0e-6F};
    const std::array<float, 48> input{
        1.0F,  -2.0F, 0.5F,   1.5F,  0.25F, -0.75F, 1.25F, -1.5F, 0.1F,  0.2F,
        -0.3F, 0.4F,  0.7F,   -0.2F, 0.3F,  -0.6F,  -1.0F, 0.5F,  2.0F,  -0.5F,
        0.75F, 1.0F,  -1.25F, 0.5F,  -0.6F, 0.3F,   0.8F,  -0.2F, -0.1F, 0.9F,
        -0.4F, 0.2F,  2.0F,   -1.0F, 0.0F,  1.0F,   -0.5F, 1.5F,  0.25F, -0.75F,
        0.5F,  -0.4F, 0.6F,   -0.7F, 0.8F,  -0.3F,  0.2F,  -0.5F};
    const std::array<float, 32> conv_weight{
        0.2F, -0.1F,  0.05F, 0.3F,  -0.25F, 0.15F, 0.1F,   -0.05F,
        0.4F, -0.2F,  0.3F,  0.25F, -0.15F, 0.35F, -0.05F, 0.2F,
        0.1F, 0.25F,  -0.3F, 0.15F, -0.2F,  0.05F, 0.45F,  -0.1F,
        0.3F, -0.25F, 0.2F,  0.05F, -0.35F, 0.1F,  -0.15F, 0.4F};
    // GGUF stores the already transformed coefficient -exp(A_log).
    const std::array<float, 2> recurrent_a{-1.1051702F, -0.8187308F};
    const std::array<float, 2> dt_bias{0.2F, -0.4F};
    const std::array<float, 2> output_norm_weight{1.25F, 0.75F};
    std::array<float, 12> prefill_output{};
    raftinfer::reference::GatedDeltaReferenceState prefill_state(args);

    raftinfer::reference::qwen35_gated_delta_prefill(
        input, prefill_output,
        raftinfer::reference::GatedDeltaReferenceWeights{.conv_weight = conv_weight,
                                                   .recurrent_a = recurrent_a,
                                                   .dt_bias = dt_bias,
                                                   .output_norm_weight =
                                                       output_norm_weight},
        args, prefill_state);

    expect_vector_near(prefill_output,
                       std::array<float, 12>{
                           0.3607474F, -0.0848603F, 0.0456384F, 0.2228722F,
                           0.0154146F, 0.6658783F, 0.2578419F, -0.0484879F,
                           -0.8969909F, -0.0499238F, 0.0453625F, -0.1946568F},
                       1.0e-5F);

    std::array<float, 12> stepped_output{};
    raftinfer::reference::GatedDeltaReferenceState stepped_state(args);
    for (std::size_t token = 0; token < args.tokens; ++token) {
      raftinfer::reference::qwen35_gated_delta_step(
          std::span<const float>(input).subspan(token * 16, 16),
          std::span<float>(stepped_output)
              .subspan(token * args.hidden_size, args.hidden_size),
          raftinfer::reference::GatedDeltaReferenceWeights{.conv_weight = conv_weight,
                                                     .recurrent_a = recurrent_a,
                                                     .dt_bias = dt_bias,
                                                     .output_norm_weight =
                                                         output_norm_weight},
          args, stepped_state);
    }

    expect_vector_near(stepped_output, prefill_output);
    expect_vector_near(stepped_state.convolution, prefill_state.convolution);
    expect_vector_near(stepped_state.recurrent, prefill_state.recurrent);
  }

  {
    std::array<float, 3> output{};
    expect_invalid_argument([&] {
      raftinfer::reference::qwen35_rms_norm(
          std::array<float, 4>{}, std::array<float, 2>{}, output, 2, 2, -1.0F);
    });
    expect_invalid_argument([&] {
      raftinfer::reference::qwen35_rms_norm(std::span<const float>{},
                                      std::array<float, 2>{},
                                      std::span<float>{}, 0, 2, 0.0F);
    });
  }
}
