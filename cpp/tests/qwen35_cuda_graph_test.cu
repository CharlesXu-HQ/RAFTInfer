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
#include <cstring>
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

  raftinfer::CudaGraphDecode graph{0, stream};
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
  graph.replay_on_current_device();
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

  const auto shape = raftinfer::kernels::QkNormRopeShape{
      .tokens = 1,
      .heads = 1,
      .head_dim = head_dim,
      .rotary_dim = 2,
      .position_offset = 0,
      .rope_base = 10'000.0F,
  };
  raftinfer::CudaGraphDecode graph{0, stream};
  graph.capture([&] {
    raftinfer::kernels::qwen35_qk_norm_rope(
        device_input, device_weight, device_output, shape, 1.0e-6F,
        RAFTINFER_DTYPE_F32, RAFTINFER_DTYPE_F32, stream, device_position);
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
    raftinfer::kernels::qwen35_qk_norm_rope(
        device_input, device_weight, device_output, host_shape, 1.0e-6F,
        RAFTINFER_DTYPE_F32, RAFTINFER_DTYPE_F32, stream);
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
                    "raftinfer_qwen35_cuda_graph.gguf";
  std::ofstream output{path, std::ios::binary};
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  assert(output);
  return path;
}

struct ExecutorObservation {
  raftinfer::Qwen35ExecutorResult result;
  std::vector<float> logits;
  raftinfer::test::Qwen35ExecutorStateSnapshot state;
};

ExecutorObservation observe_executor(raftinfer::Qwen35Executor &executor,
                                     raftinfer::Qwen35ExecutorResult result,
                                     std::size_t vocabulary_size) {
  ExecutorObservation observation{
      .result = result,
      .logits = std::vector<float>(vocabulary_size),
      .state = executor.state_snapshot_for_tests(),
  };
  executor.copy_last_logits(observation.logits);
  return observation;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_logits_bits(const std::vector<float> &actual,
                      const std::vector<float> &expected) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (float_bits(actual[index]) != float_bits(expected[index]))
      return false;
  }
  return true;
}

void assert_same_observation(const ExecutorObservation &actual,
                             const ExecutorObservation &expected) {
  assert(actual.result.token == expected.result.token);
  assert(actual.result.position == expected.result.position);
  assert(same_logits_bits(actual.logits, expected.logits));
  assert(actual.state.full_kv_cache == expected.state.full_kv_cache);
  assert(actual.state.linear_convolution == expected.state.linear_convolution);
  assert(actual.state.linear_recurrent == expected.state.linear_recurrent);
}

void run_executor_graph_equivalence_tests() {
  require_cuda(cudaSetDevice(0));
  const raftinfer::test::Qwen35GgufFixtureOptions fixture_options{
      .hidden_size = 4096,
      .context_length = 1030,
      .full_head_count = 16,
      .full_kv_head_count = 4,
      .full_head_dimension = 256,
      .linear_key_head_count = 4,
      .linear_value_head_count = 16,
      .linear_head_dimension = 256,
      .rotary_dimension = 64,
  };
  const auto path = write_fixture(
      raftinfer::test::make_qwen35_nonzero_bf16_gguf_fixture(fixture_options));
  raftinfer::model::Model model{path.string()};
  constexpr std::size_t max_context = 64;
  auto graph_policy = raftinfer::Qwen35ExecutionPolicy{};
  graph_policy.decode_graph = true;
  const auto workspace_bytes = raftinfer::Qwen35Executor::workspace_bytes(
      model.qwen35_config(), max_context, graph_policy);

  raftinfer::DeviceContext device{0, 1024U * 1024U * 1024U};
  auto weights = device.upload_qwen35_weights(model);
  auto graph_owner = device.create_execution_owner(workspace_bytes);
  auto graph_context = graph_owner->execution_context();
  raftinfer::Qwen35Executor graph{graph_context, model.qwen35_config(), *weights,
                            max_context, graph_policy};
  const auto construction_diagnostics = graph.diagnostics();
  assert(construction_diagnostics.attention ==
         raftinfer::Qwen35AttentionImplementation::online_tiled);
  assert(!construction_diagnostics.cublaslt_plans.empty());

  const std::vector<std::int32_t> prompt{1, 2, 3, 4};
  const auto run_sequence =
      [&](bool expect_first_decode_replay,
          const std::vector<ExecutorObservation> *expected_observations) {
    std::vector<ExecutorObservation> graph_observations;
    const auto graph_prefill = graph.prefill(prompt);
    graph_observations.push_back(observe_executor(
        graph, graph_prefill, model.qwen35_config().vocabulary_size));
    if (expected_observations != nullptr) {
      assert_same_observation(graph_observations.back(),
                              expected_observations->front());
    }
    for (std::size_t step = 0; step < 8; ++step) {
      const auto token = static_cast<std::int32_t>((step + 5) % 15 + 1);
      const auto graph_result = graph.decode(token);
      graph_observations.push_back(observe_executor(
          graph, graph_result, model.qwen35_config().vocabulary_size));
      if (expected_observations != nullptr) {
        assert_same_observation(
            graph_observations.back(),
            (*expected_observations)[graph_observations.size() - 1]);
      }
      const auto diagnostics = graph.diagnostics();
      assert(diagnostics.cublaslt_algorithm_ids ==
             construction_diagnostics.cublaslt_algorithm_ids);
      assert(diagnostics.cublaslt_plans ==
             construction_diagnostics.cublaslt_plans);
      assert(diagnostics.decode_graph_captured);
      if (step == 0)
        assert(diagnostics.decode_graph_replayed == expect_first_decode_replay);
      else
        assert(diagnostics.decode_graph_replayed);
    }
    return graph_observations;
  };
  const auto captured_observations = run_sequence(false, nullptr);
  graph.reset();
  assert(graph.position() == 0);
  const auto replayed_observations =
      run_sequence(true, &captured_observations);
  assert(replayed_observations.size() == captured_observations.size());

  constexpr std::size_t boundary_max_context = 1030;
  auto boundary_graph_policy = raftinfer::Qwen35ExecutionPolicy{};
  boundary_graph_policy.decode_graph = true;
  boundary_graph_policy.decode_attention =
      raftinfer::Qwen35DecodeAttentionMode::split_k_256;
  const auto boundary_graph_workspace =
      raftinfer::Qwen35Executor::workspace_bytes(
          model.qwen35_config(), boundary_max_context, boundary_graph_policy);
  auto boundary_graph_owner =
      device.create_execution_owner(boundary_graph_workspace);
  auto boundary_graph_context = boundary_graph_owner->execution_context();
  raftinfer::Qwen35Executor boundary_graph{
      boundary_graph_context, model.qwen35_config(), *weights,
      boundary_max_context, boundary_graph_policy};

  for (const std::size_t start_position : {254U, 255U, 511U, 1023U}) {
    boundary_graph.set_decode_graph_enabled_for_tests(true);
    boundary_graph.reset();
    std::vector<std::int32_t> repeated_prompt(start_position, 1);
    const auto graph_prefill_observation = observe_executor(
        boundary_graph, boundary_graph.prefill(repeated_prompt),
        model.qwen35_config().vocabulary_size);
    std::vector<ExecutorObservation> graph_decode_observations;
    for (const std::int32_t token : {2, 3}) {
      const auto graph_result = boundary_graph.decode(token);
      graph_decode_observations.push_back(observe_executor(
          boundary_graph, graph_result,
          model.qwen35_config().vocabulary_size));
    }
    const auto graph_diagnostics = boundary_graph.diagnostics();

    boundary_graph.set_decode_graph_enabled_for_tests(false);
    boundary_graph.reset();
    assert_same_observation(
        graph_prefill_observation,
        observe_executor(boundary_graph,
                         boundary_graph.prefill(repeated_prompt),
                         model.qwen35_config().vocabulary_size));
    std::size_t step = 0;
    for (const std::int32_t token : {2, 3}) {
      const auto stream_observation = observe_executor(
          boundary_graph, boundary_graph.decode(token),
          model.qwen35_config().vocabulary_size);
      assert_same_observation(graph_decode_observations[step],
                              stream_observation);
      ++step;
    }
    if (start_position == 511) {
      assert(graph_diagnostics.decode_graph_replayed);
      assert(graph_diagnostics.decode_attention.implementation ==
             raftinfer::Qwen35DecodeAttentionImplementation::split_k);
      assert(graph_diagnostics.decode_attention.last_context_bucket_tokens ==
             1024);
      assert(graph_diagnostics.decode_attention.split_k_graph_captured);
    }
  }
}

} // namespace

int main() {
  const char *opt_in = std::getenv("RAFTINFER_RUN_GPU_TESTS");
  if (opt_in == nullptr || std::string{opt_in} != "1")
    return 77;
  run_graph_raii_tests();
  run_device_position_rope_graph_tests();
  run_executor_graph_equivalence_tests();
}
