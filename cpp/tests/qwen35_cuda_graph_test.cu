#include "../execution/cuda_graph_decode.hpp"
#include "../execution/qwen35_executor.hpp"
#include "../kernels/qwen35_primitives.cuh"
#include "../model/model.hpp"
#include "../foundation/device_context.hpp"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include "qwen35_nonzero_fixture.hpp"

namespace {

__global__ void add_kernel(const std::int32_t *input, std::int32_t *output) {
  *output = *input + 3;
}

__global__ void multiply_kernel(std::int32_t *value) { *value *= 7; }

__global__ void increment_position_kernel(std::uint32_t *position) {
  ++*position;
}

void require_cuda(cudaError_t status) { assert(status == cudaSuccess); }

void run_graph_raii_tests() {
  require_cuda(cudaSetDevice(0));
  cudaStream_t stream{};
  require_cuda(cudaStreamCreate(&stream));

  std::int32_t *device_input{};
  std::int32_t *device_output{};
  require_cuda(cudaMalloc(&device_input, sizeof(*device_input)));
  require_cuda(cudaMalloc(&device_output, sizeof(*device_output)));
  std::int32_t *host_input{};
  std::int32_t *host_output{};
  require_cuda(cudaHostAlloc(&host_input, sizeof(*host_input), cudaHostAllocDefault));
  require_cuda(cudaHostAlloc(&host_output, sizeof(*host_output), cudaHostAllocDefault));

  brt::CudaGraphDecode graph{0, stream};
  graph.capture([&] {
    require_cuda(cudaMemcpyAsync(device_input, host_input, sizeof(*host_input),
                                 cudaMemcpyHostToDevice, stream));
    add_kernel<<<1, 1, 0, stream>>>(device_input, device_output);
    multiply_kernel<<<1, 1, 0, stream>>>(device_output);
    require_cuda(cudaMemcpyAsync(host_output, device_output,
                                 sizeof(*host_output), cudaMemcpyDeviceToHost,
                                 stream));
  });
  assert(graph.captured());
  *host_input = 2;
  graph.replay();
  require_cuda(cudaStreamSynchronize(stream));
  assert(*host_output == 35);
  *host_input = 5;
  graph.replay();
  require_cuda(cudaStreamSynchronize(stream));
  assert(*host_output == 56);

  graph.reset();
  assert(!graph.captured());
  graph.capture([&] {
    require_cuda(cudaMemcpyAsync(device_input, host_input, sizeof(*host_input),
                                 cudaMemcpyHostToDevice, stream));
    add_kernel<<<1, 1, 0, stream>>>(device_input, device_output);
    require_cuda(cudaMemcpyAsync(host_output, device_output,
                                 sizeof(*host_output), cudaMemcpyDeviceToHost,
                                 stream));
  });
  *host_input = 8;
  graph.replay();
  require_cuda(cudaStreamSynchronize(stream));
  assert(*host_output == 11);

  graph.reset();
  bool thrown = false;
  try {
    graph.capture([&] { throw 1; });
  } catch (...) {
    thrown = true;
  }
  assert(thrown);
  assert(!graph.captured());
  graph.capture([&] {
    require_cuda(cudaMemcpyAsync(device_input, host_input, sizeof(*host_input),
                                 cudaMemcpyHostToDevice, stream));
    add_kernel<<<1, 1, 0, stream>>>(device_input, device_output);
    require_cuda(cudaMemcpyAsync(host_output, device_output,
                                 sizeof(*host_output), cudaMemcpyDeviceToHost,
                                 stream));
  });
  *host_input = 9;
  graph.replay();
  require_cuda(cudaStreamSynchronize(stream));
  assert(*host_output == 12);

  graph.reset();
  require_cuda(cudaFreeHost(host_output));
  require_cuda(cudaFreeHost(host_input));
  require_cuda(cudaFree(device_output));
  require_cuda(cudaFree(device_input));
  require_cuda(cudaStreamDestroy(stream));
}

void run_device_position_rope_graph_tests() {
  require_cuda(cudaSetDevice(0));
  cudaStream_t stream{};
  require_cuda(cudaStreamCreate(&stream));
  constexpr std::size_t head_dim = 4;
  const std::vector<float> input{1.0F, -2.0F, 0.5F, 3.0F};
  const std::vector<float> weight{1.0F, 1.0F, 1.0F, 1.0F};
  float *device_input{};
  float *device_weight{};
  float *device_output{};
  std::uint32_t *device_position{};
  float *host_graph_output{};
  require_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)));
  require_cuda(cudaMalloc(&device_weight, weight.size() * sizeof(float)));
  require_cuda(cudaMalloc(&device_output, input.size() * sizeof(float)));
  require_cuda(cudaMalloc(&device_position, sizeof(*device_position)));
  require_cuda(cudaHostAlloc(&host_graph_output, input.size() * sizeof(float),
                             cudaHostAllocDefault));
  require_cuda(cudaMemcpyAsync(device_input, input.data(),
                               input.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaMemcpyAsync(device_weight, weight.data(),
                               weight.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  std::uint32_t position = 128;
  require_cuda(cudaMemcpyAsync(device_position, &position, sizeof(position),
                               cudaMemcpyHostToDevice, stream));
  require_cuda(cudaStreamSynchronize(stream));

  const auto shape = brt::kernels::QkNormRopeShape{
      .tokens = 1,
      .heads = 1,
      .head_dim = head_dim,
      .rotary_dim = 2,
      .position_offset = 0,
      .rope_base = 10'000.0F,
  };
  brt::CudaGraphDecode graph{0, stream};
  graph.capture([&] {
    brt::kernels::qwen35_qk_norm_rope(
        device_input, device_weight, device_output, shape, 1.0e-6F,
        BRT_DTYPE_F32, BRT_DTYPE_F32, stream, device_position);
    require_cuda(cudaMemcpyAsync(host_graph_output, device_output,
                                 input.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    increment_position_kernel<<<1, 1, 0, stream>>>(device_position);
  });

  for (const std::uint32_t expected_position : {128U, 129U}) {
    graph.replay();
    require_cuda(cudaStreamSynchronize(stream));
    std::vector<float> host_reference(input.size());
    auto host_shape = shape;
    host_shape.position_offset = expected_position;
    brt::kernels::qwen35_qk_norm_rope(
        device_input, device_weight, device_output, host_shape, 1.0e-6F,
        BRT_DTYPE_F32, BRT_DTYPE_F32, stream);
    require_cuda(cudaMemcpyAsync(host_reference.data(), device_output,
                                 host_reference.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    for (std::size_t index = 0; index < input.size(); ++index)
      assert(std::fabs(host_graph_output[index] - host_reference[index]) <
             1.0e-5F);
  }

  graph.reset();
  require_cuda(cudaFreeHost(host_graph_output));
  require_cuda(cudaFree(device_position));
  require_cuda(cudaFree(device_output));
  require_cuda(cudaFree(device_weight));
  require_cuda(cudaFree(device_input));
  require_cuda(cudaStreamDestroy(stream));
}

std::filesystem::path write_fixture(std::vector<std::uint8_t> bytes) {
  const auto path = std::filesystem::temp_directory_path() /
                    "brt_qwen35_cuda_graph.gguf";
  std::ofstream output{path, std::ios::binary};
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  assert(output);
  return path;
}

void assert_equal_executor_state(
    brt::Qwen35Executor &ordinary, brt::Qwen35Executor &graph,
    std::vector<float> &ordinary_logits, std::vector<float> &graph_logits) {
  ordinary.copy_last_logits(ordinary_logits);
  graph.copy_last_logits(graph_logits);
  assert(ordinary_logits == graph_logits);
  const auto ordinary_state = ordinary.state_snapshot_for_tests();
  const auto graph_state = graph.state_snapshot_for_tests();
  assert(ordinary_state.full_kv_cache == graph_state.full_kv_cache);
  assert(ordinary_state.linear_convolution == graph_state.linear_convolution);
  assert(ordinary_state.linear_recurrent == graph_state.linear_recurrent);
}

void run_executor_graph_equivalence_tests() {
  require_cuda(cudaSetDevice(0));
  const brt::test::Qwen35GgufFixtureOptions fixture_options{
      .hidden_size = 4096,
      .full_head_count = 16,
      .full_kv_head_count = 4,
      .full_head_dimension = 256,
      .linear_key_head_count = 4,
      .linear_value_head_count = 16,
      .linear_head_dimension = 256,
      .rotary_dimension = 64,
  };
  const auto path = write_fixture(
      brt::test::make_qwen35_nonzero_bf16_gguf_fixture(fixture_options));
  brt::model::Model model{path.string()};
  constexpr std::size_t max_context = 64;
  auto ordinary_policy = brt::Qwen35ExecutionPolicy{};
  ordinary_policy.decode_graph = false;
  auto graph_policy = ordinary_policy;
  graph_policy.decode_graph = true;
  const auto workspace_bytes = brt::Qwen35Executor::workspace_bytes(
      model.qwen35_config(), max_context, graph_policy);

  brt::DeviceContext device{0, 1024U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);
  auto ordinary_owner = device.create_execution_owner(workspace_bytes);
  auto graph_owner = device.create_execution_owner(workspace_bytes);
  auto ordinary_context = ordinary_owner->execution_context();
  auto graph_context = graph_owner->execution_context();
  brt::Qwen35Executor ordinary{ordinary_context, model.qwen35_config(),
                               *weights, max_context, ordinary_policy};
  brt::Qwen35Executor graph{graph_context, model.qwen35_config(), *weights,
                            max_context, graph_policy};
  assert(graph.diagnostics().attention ==
         brt::Qwen35AttentionImplementation::online_tiled);

  const std::vector<std::int32_t> prompt{1, 2, 3, 4};
  std::vector<float> ordinary_logits(model.qwen35_config().vocabulary_size);
  std::vector<float> graph_logits(model.qwen35_config().vocabulary_size);
  const auto run_sequence = [&] {
    const auto ordinary_prefill = ordinary.prefill(prompt);
    const auto graph_prefill = graph.prefill(prompt);
    assert(ordinary_prefill.token == graph_prefill.token);
    assert(ordinary_prefill.position == graph_prefill.position);
    assert(ordinary.position() == graph.position());
    assert_equal_executor_state(ordinary, graph, ordinary_logits, graph_logits);
    for (std::size_t step = 0; step < 8; ++step) {
      const auto token = static_cast<std::int32_t>((step + 5) % 15 + 1);
      const auto ordinary_result = ordinary.decode(token);
      const auto graph_result = graph.decode(token);
      assert(ordinary_result.token == graph_result.token);
      assert(ordinary_result.position == graph_result.position);
      assert(ordinary.position() == graph.position());
      assert_equal_executor_state(ordinary, graph, ordinary_logits,
                                  graph_logits);
      const auto diagnostics = graph.diagnostics();
      assert(diagnostics.decode_graph_captured);
      if (step == 0)
        assert(!diagnostics.decode_graph_replayed);
      else
        assert(diagnostics.decode_graph_replayed);
    }
  };
  run_sequence();
  ordinary.reset();
  graph.reset();
  assert(ordinary.position() == 0);
  assert(graph.position() == 0);
  run_sequence();
}

} // namespace

int main() {
  const char *opt_in = std::getenv("BRT_RUN_GPU_TESTS");
  if (opt_in == nullptr || std::string{opt_in} != "1")
    return 77;
  run_graph_raii_tests();
  run_device_position_rope_graph_tests();
  run_executor_graph_equivalence_tests();
}
