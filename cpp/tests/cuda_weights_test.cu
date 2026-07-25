#include "../execution/execution_context.hpp"
#include "../foundation/device_context.hpp"
#include "../model/cuda_weights.hpp"
#include "../model/gguf_reader.hpp"
#include "../model/model.hpp"
#include "../model/qwen35_manifest.hpp"

#include "assert_enabled.hpp"
#include "qwen35_gguf_fixture.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path write_fixture(std::vector<std::uint8_t> bytes,
                                    const std::string &suffix) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("brt_cuda_weights_" + suffix + ".gguf");
  std::ofstream output{path, std::ios::binary};
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  assert(output);
  return path;
}

std::vector<std::uint8_t> filled_fixture(std::uint32_t tensor_type = 1,
                                         bool f32_auxiliary_tensors = false) {
  auto bytes =
      brt::test::make_qwen35_gguf_fixture(tensor_type, f32_auxiliary_tensors);
  const auto catalog = brt::gguf::read_catalog(std::span{bytes});
  for (std::size_t tensor_index = 0; tensor_index < catalog.tensors.size();
       ++tensor_index) {
    const auto &tensor = catalog.tensors[tensor_index];
    const std::size_t begin =
        static_cast<std::size_t>(catalog.tensor_data_offset + tensor.offset);
    if (tensor.type == 0) {
      assert(tensor.byte_size % sizeof(float) == 0);
      const std::size_t elements = tensor.byte_size / sizeof(float);
      for (std::size_t element = 0; element < elements; ++element) {
        const float value =
            static_cast<float>(tensor_index + 1) +
            static_cast<float>(element + 1) / 128.0F;
        std::memcpy(bytes.data() + begin + element * sizeof(float), &value,
                    sizeof(value));
      }
    } else {
      for (std::size_t byte = 0; byte < tensor.byte_size; ++byte) {
        bytes[begin + byte] =
            static_cast<std::uint8_t>((tensor_index * 17 + byte) & 0xff);
      }
    }
  }
  return bytes;
}

std::vector<std::uint8_t> read_device_bytes(const void *device,
                                            std::size_t size) {
  int previous_device = 0;
  const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
  assert(cudaSetDevice(0) == cudaSuccess);
  std::vector<std::uint8_t> bytes(size);
  const auto error =
      cudaMemcpy(bytes.data(), device, size, cudaMemcpyDeviceToHost);
  if (restore_device) {
    assert(cudaSetDevice(previous_device) == cudaSuccess);
  }
  assert(error == cudaSuccess);
  return bytes;
}

bool device_bytes_equal(const brt::model::CudaTensorView &view,
                        std::span<const std::uint8_t> expected) {
  const auto actual = read_device_bytes(view.device_data, view.bytes);
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

std::uint16_t f32_to_f16_bits(float value) {
  const __half converted = __float2half_rn(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &converted, sizeof(bits));
  return bits;
}

std::uint16_t f32_to_bf16_bits(float value) {
  const __nv_bfloat16 converted = __float2bfloat16_rn(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &converted, sizeof(bits));
  return bits;
}

std::vector<std::uint8_t>
converted_f32_payload(const brt::model::Model &model,
                      const brt::gguf::TensorInfo &tensor,
                      brt::model::CudaWeightType target_type) {
  const auto payload = model.tensor_payload(tensor);
  assert(payload.size() % sizeof(float) == 0);
  std::vector<std::uint8_t> converted(payload.size() / sizeof(float) *
                                      sizeof(std::uint16_t));
  for (std::size_t element = 0; element < payload.size() / sizeof(float);
       ++element) {
    float value = 0.0F;
    std::memcpy(&value, payload.data() + element * sizeof(float),
                sizeof(value));
    const std::uint16_t bits =
        target_type == brt::model::CudaWeightType::f16
            ? f32_to_f16_bits(value)
            : f32_to_bf16_bits(value);
    converted[element * sizeof(std::uint16_t)] =
        static_cast<std::uint8_t>(bits & 0xffU);
    converted[element * sizeof(std::uint16_t) + 1] =
        static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
  }
  return converted;
}

void expect_weight_error(auto &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const brt::model::CudaWeightError &) {
    thrown = true;
  }
  assert(thrown);
}

void expect_view_matches(const brt::model::CudaTensorView &view,
                         const brt::gguf::TensorInfo &tensor,
                         const brt::model::Model &model,
                         brt::model::CudaWeightType expected_type) {
  assert(view.device_data != nullptr);
  assert(view.bytes == tensor.byte_size);
  assert(view.type == expected_type);
  assert(device_bytes_equal(view, model.tensor_payload(tensor)));
}

void expect_f32_aux_converted(const brt::model::CudaTensorView &view,
                              const brt::gguf::TensorInfo &tensor,
                              const brt::model::Model &model,
                              brt::model::CudaWeightType expected_type) {
  assert(tensor.type == 0);
  assert(view.device_data != nullptr);
  assert(view.bytes == tensor.byte_size / 2);
  assert(view.type == expected_type);
  assert(device_bytes_equal(view,
                            converted_f32_payload(model, tensor,
                                                  expected_type)));
}

void expect_common_matches(
    const brt::model::Qwen35CudaCommonLayerWeights &weights,
    const brt::model::Qwen35CommonLayerTensors &tensors,
    const brt::model::Model &model, brt::model::CudaWeightType expected_type) {
  expect_view_matches(weights.input_norm, *tensors.input_norm, model,
                      expected_type);
  expect_view_matches(weights.post_attention_norm, *tensors.post_attention_norm,
                      model, expected_type);
  expect_view_matches(weights.ffn_gate, *tensors.ffn_gate, model,
                      expected_type);
  expect_view_matches(weights.ffn_down, *tensors.ffn_down, model,
                      expected_type);
  expect_view_matches(weights.ffn_up, *tensors.ffn_up, model, expected_type);
}

void expect_linear_matches(
    const brt::model::Qwen35CudaLinearAttentionWeights &weights,
    const brt::model::Qwen35LinearAttentionTensors &tensors,
    const brt::model::Model &model, brt::model::CudaWeightType expected_type) {
  expect_view_matches(weights.qkv, *tensors.qkv, model, expected_type);
  expect_view_matches(weights.gate, *tensors.gate, model, expected_type);
  expect_view_matches(weights.convolution, *tensors.convolution, model,
                      expected_type);
  expect_view_matches(weights.time_step_bias, *tensors.time_step_bias, model,
                      expected_type);
  expect_view_matches(weights.recurrent_a, *tensors.recurrent_a, model,
                      expected_type);
  expect_view_matches(weights.beta, *tensors.beta, model, expected_type);
  expect_view_matches(weights.alpha, *tensors.alpha, model, expected_type);
  expect_view_matches(weights.output_norm, *tensors.output_norm, model,
                      expected_type);
  expect_view_matches(weights.output, *tensors.output, model, expected_type);
}

void expect_full_matches(
    const brt::model::Qwen35CudaFullAttentionWeights &weights,
    const brt::model::Qwen35FullAttentionTensors &tensors,
    const brt::model::Model &model, brt::model::CudaWeightType expected_type) {
  expect_view_matches(weights.query, *tensors.query, model, expected_type);
  expect_view_matches(weights.key, *tensors.key, model, expected_type);
  expect_view_matches(weights.value, *tensors.value, model, expected_type);
  expect_view_matches(weights.output, *tensors.output, model, expected_type);
  expect_view_matches(weights.query_norm, *tensors.query_norm, model,
                      expected_type);
  expect_view_matches(weights.key_norm, *tensors.key_norm, model,
                      expected_type);
}

std::vector<std::uint8_t>
fixture_with_tensor_type(std::vector<std::uint8_t> bytes,
                         std::uint32_t tensor_type) {
  auto catalog = brt::gguf::read_catalog(std::span{bytes});
  for (const auto &tensor : catalog.tensors) {
    auto name_position = std::search(bytes.begin(), bytes.end(),
                                     tensor.name.begin(), tensor.name.end());
    assert(name_position != bytes.end());
    std::size_t position =
        static_cast<std::size_t>(std::distance(bytes.begin(), name_position));
    position += tensor.name.size();
    position += sizeof(std::uint32_t);
    position += tensor.dimensions.size() * sizeof(std::uint64_t);
    auto *type_bytes = bytes.data() + position;
    type_bytes[0] = static_cast<std::uint8_t>(tensor_type);
    type_bytes[1] = static_cast<std::uint8_t>(tensor_type >> 8);
    type_bytes[2] = static_cast<std::uint8_t>(tensor_type >> 16);
    type_bytes[3] = static_cast<std::uint8_t>(tensor_type >> 24);
  }
  return bytes;
}

void run_mixed_f32_auxiliary_case(std::uint32_t primary_tensor_type,
                                  brt::model::CudaWeightType expected_type,
                                  brt::DeviceContext &device) {
  const auto path = write_fixture(
      filled_fixture(primary_tensor_type, true),
      expected_type == brt::model::CudaWeightType::f16 ? "mixed_f16_aux_f32"
                                                       : "mixed_bf16_aux_f32");
  brt::model::Model model{path.string()};
  const auto plan = device.upload_qwen35_weights(model);
  const auto &manifest = model.qwen35_manifest();

  expect_view_matches(plan->token_embedding(), *manifest.token_embedding,
                      model, expected_type);
  expect_f32_aux_converted(plan->output_norm(), *manifest.output_norm, model,
                           expected_type);
  expect_view_matches(plan->output(), *manifest.output, model, expected_type);
  assert(&plan->tensor(*manifest.output_norm) == &plan->output_norm());

  for (std::size_t layer_index = 0; layer_index < manifest.layers.size();
       ++layer_index) {
    const auto &layer = plan->layer(layer_index);
    const auto &expected = manifest.layers[layer_index];
    expect_f32_aux_converted(layer.common.input_norm,
                             *expected.common.input_norm, model,
                             expected_type);
    expect_f32_aux_converted(layer.common.post_attention_norm,
                             *expected.common.post_attention_norm, model,
                             expected_type);
    expect_view_matches(layer.common.ffn_gate, *expected.common.ffn_gate,
                        model, expected_type);
    expect_view_matches(layer.common.ffn_down, *expected.common.ffn_down,
                        model, expected_type);
    expect_view_matches(layer.common.ffn_up, *expected.common.ffn_up, model,
                        expected_type);
    if (expected.full_attention.has_value()) {
      const auto &full = *expected.full_attention;
      expect_view_matches(layer.full_attention->query, *full.query, model,
                          expected_type);
      expect_view_matches(layer.full_attention->key, *full.key, model,
                          expected_type);
      expect_view_matches(layer.full_attention->value, *full.value, model,
                          expected_type);
      expect_view_matches(layer.full_attention->output, *full.output, model,
                          expected_type);
      expect_f32_aux_converted(layer.full_attention->query_norm,
                               *full.query_norm, model, expected_type);
      expect_f32_aux_converted(layer.full_attention->key_norm, *full.key_norm,
                               model, expected_type);
    } else {
      const auto &linear = *expected.linear_attention;
      expect_view_matches(layer.linear_attention->qkv, *linear.qkv, model,
                          expected_type);
      expect_view_matches(layer.linear_attention->gate, *linear.gate, model,
                          expected_type);
      expect_f32_aux_converted(layer.linear_attention->convolution,
                               *linear.convolution, model, expected_type);
      expect_f32_aux_converted(layer.linear_attention->time_step_bias,
                               *linear.time_step_bias, model, expected_type);
      expect_f32_aux_converted(layer.linear_attention->recurrent_a,
                               *linear.recurrent_a, model, expected_type);
      expect_view_matches(layer.linear_attention->beta, *linear.beta, model,
                          expected_type);
      expect_view_matches(layer.linear_attention->alpha, *linear.alpha, model,
                          expected_type);
      expect_f32_aux_converted(layer.linear_attention->output_norm,
                               *linear.output_norm, model, expected_type);
      expect_view_matches(layer.linear_attention->output, *linear.output,
                          model, expected_type);
    }
  }
}

} // namespace

int main() {
  brt::DeviceContext device{0, 64 * 1024 * 1024};

  {
    const auto fixture = filled_fixture();
    const auto path = write_fixture(fixture, "f16");
    brt::model::Model model{path.string()};
    const auto plan = device.upload_qwen35_weights(model);
    assert(plan->tensor_count() == 56);
    assert(plan->layer_count() == 4);
    const auto &token_embedding = plan->token_embedding();
    assert(token_embedding.type == brt::model::CudaWeightType::f16);
    assert(token_embedding.bytes == 256);
    assert(token_embedding.device_data != nullptr);
    assert(device_bytes_equal(
        token_embedding,
        model.tensor_payload(*model.qwen35_manifest().token_embedding)));
    const void *first_address = token_embedding.device_data;
    assert(plan->token_embedding().device_data == first_address);

    const auto &manifest = model.qwen35_manifest();
    assert(&plan->tensor(0) == &plan->token_embedding());
    assert(&plan->tensor(1) == &plan->output_norm());
    assert(&plan->tensor(2) == &plan->output());
    assert(&plan->tensor(*manifest.token_embedding) ==
           &plan->token_embedding());
    assert(&plan->tensor(std::string_view{"output_norm.weight"}) ==
           &plan->output_norm());

    for (std::size_t layer_index = 0; layer_index < manifest.layers.size();
         ++layer_index) {
      const auto &layer = plan->layer(layer_index);
      const auto &expected = manifest.layers[layer_index];
      assert(layer.index == expected.index);
      expect_common_matches(layer.common, expected.common, model,
                            brt::model::CudaWeightType::f16);
      if (layer_index == 3) {
        assert(layer.full_attention.has_value());
        assert(!layer.linear_attention.has_value());
        expect_full_matches(*layer.full_attention, *expected.full_attention,
                            model, brt::model::CudaWeightType::f16);
      } else {
        assert(!layer.full_attention.has_value());
        assert(layer.linear_attention.has_value());
        expect_linear_matches(*layer.linear_attention,
                              *expected.linear_attention, model,
                              brt::model::CudaWeightType::f16);
      }
    }

    assert(&plan->tensor(3) == &plan->layer(0).common.input_norm);
    assert(&plan->tensor(16) == &plan->layer(0).linear_attention->output);
    assert(&plan->tensor(50) == &plan->layer(3).full_attention->query);
    assert(&plan->tensor(55) == &plan->layer(3).full_attention->key_norm);

    auto foreign = *manifest.token_embedding;
    foreign.name = "unknown.weight";
    expect_weight_error([&] { (void)plan->tensor(foreign); });
    expect_weight_error(
        [&] { (void)plan->tensor(std::string_view{"unknown.weight"}); });
    foreign = *manifest.token_embedding;
    foreign.offset += 32;
    expect_weight_error([&] { (void)plan->tensor(foreign); });
  }

  {
    const auto path =
        write_fixture(fixture_with_tensor_type(filled_fixture(), 30), "bf16");
    brt::model::Model model{path.string()};
    const auto plan = device.upload_qwen35_weights(model);
    assert(plan->token_embedding().type == brt::model::CudaWeightType::bf16);
    assert(device_bytes_equal(
        plan->output_norm(),
        model.tensor_payload(*model.qwen35_manifest().output_norm)));
    const auto &manifest = model.qwen35_manifest();
    expect_view_matches(plan->token_embedding(), *manifest.token_embedding,
                        model, brt::model::CudaWeightType::bf16);
    expect_view_matches(plan->output_norm(), *manifest.output_norm, model,
                        brt::model::CudaWeightType::bf16);
    expect_view_matches(plan->output(), *manifest.output, model,
                        brt::model::CudaWeightType::bf16);
    for (std::size_t layer_index = 0; layer_index < manifest.layers.size();
         ++layer_index) {
      const auto &layer = plan->layer(layer_index);
      const auto &expected = manifest.layers[layer_index];
      expect_common_matches(layer.common, expected.common, model,
                            brt::model::CudaWeightType::bf16);
      if (expected.full_attention.has_value()) {
        expect_full_matches(*layer.full_attention, *expected.full_attention,
                            model, brt::model::CudaWeightType::bf16);
      } else {
        expect_linear_matches(*layer.linear_attention,
                              *expected.linear_attention, model,
                              brt::model::CudaWeightType::bf16);
      }
    }
  }

  run_mixed_f32_auxiliary_case(1, brt::model::CudaWeightType::f16, device);
  run_mixed_f32_auxiliary_case(30, brt::model::CudaWeightType::bf16, device);

  {
    brt::model::Model model{
        write_fixture(filled_fixture(0), "f32_primary").string()};
    expect_weight_error([&] { (void)device.upload_qwen35_weights(model); });
  }

  {
    brt::model::Model model{
        write_fixture(filled_fixture(), "fake_f32_primary").string()};
    auto fake_manifest = model.qwen35_manifest();
    auto f32_token_embedding = *fake_manifest.token_embedding;
    f32_token_embedding.type = 0;
    fake_manifest.token_embedding = &f32_token_embedding;
    expect_weight_error([&] {
      (void)device.upload_qwen35_weights_for_tests(model, fake_manifest);
    });
  }

  {
    brt::model::Model model{
        write_fixture(filled_fixture(), "duplicate_name").string()};
    auto fake_manifest = model.qwen35_manifest();
    fake_manifest.output = fake_manifest.token_embedding;
    expect_weight_error([&] {
      (void)device.upload_qwen35_weights_for_tests(model, fake_manifest);
    });
  }

  {
    brt::model::Model model{
        write_fixture(filled_fixture(), "branch_mismatch").string()};
    auto fake_manifest = model.qwen35_manifest();
    fake_manifest.layers[0].full_attention =
        fake_manifest.layers[3].full_attention;
    expect_weight_error([&] {
      (void)device.upload_qwen35_weights_for_tests(model, fake_manifest);
    });
  }

  {
    brt::model::Model model{
        write_fixture(filled_fixture(), "descriptor").string()};
    auto foreign = *model.qwen35_manifest().token_embedding;
    foreign.offset += 32;
    auto fake_manifest = model.qwen35_manifest();
    fake_manifest.token_embedding = &foreign;
    expect_weight_error([&] {
      (void)device.upload_qwen35_weights_for_tests(model, fake_manifest);
    });
  }

  {
    const auto path = write_fixture(filled_fixture(), "outlive");
    brt::model::Model model{path.string()};
    std::unique_ptr<brt::model::CudaWeightPlan> plan;
    std::vector<std::uint8_t> expected;
    {
      brt::DeviceContext scoped_device{0, 64 * 1024 * 1024};
      plan = scoped_device.upload_qwen35_weights(model);
      expected.assign(
          model.tensor_payload(*model.qwen35_manifest().output).begin(),
          model.tensor_payload(*model.qwen35_manifest().output).end());
    }
    assert(plan != nullptr);
    assert(device_bytes_equal(plan->output(), expected));
    plan.reset();
  }
}
