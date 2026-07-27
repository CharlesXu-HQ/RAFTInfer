# SDD ledger — plan: docs/superpowers/plans/2026-07-26-m4-bf16-performance.md

Baseline: 1d0f0c7
Workspace: /Users/charles/Documents/bw24 二次开发/.worktrees/codex-m4-bf16-performance
Branch: codex/m4-bf16-performance
Baseline verification: cargo build --workspace --locked and scripts/local-check.sh passed (17/17 host CTest, 49 Rust tests, shell fixtures passed).
Task 1: target verification deferred — isolated source sync succeeded, but GPU preflight refused because `sglang::scheduler` occupied GPU 0; no process was interrupted.
Task 1: complete (commits 1d0f0c7..a0c68b2, review clean; local suite passed, target CUDA deferred for environmental contention).
Task 2: plan ruling confirmed by user — online/materialized selection occurs before allocation; direct online dispatch rejects unsupported signature drift and never retains reference logits workspace.
Task 2: fix round 1/5 (1 addressed, 0 open — online dispatch now rejects non-null or nonzero workspace; commits 37c90af..6bd91dc).
Task 2: target verification passed at 6bd91dc — NVCC 13.2.78 build/link succeeded and focused `brt_qwen35_cuda_attention_test` passed 1/1 on idle RTX 5090 after two successful GPU preflights.
Task 2: complete (commits a0c68b2..6bd91dc, review clean after fix round 1).
Task 3: pre-review target compile fix — explicit `std::span<const uint32_t>` upload committed as 8979835 after NVCC identified mutable-span deduction failure.
Task 3: minor (deferred): add direct host-position/device-position agreement cases at context 128 and above 128; final reviewer must triage.
Task 3: target verification passed at 8979835 — NVCC build/link succeeded and focused `brt_qwen35_cuda_attention_test` passed 1/1 on idle RTX 5090 after two successful GPU preflights.
Task 3: complete (commits 6bd91dc..8979835, review approved with 1 deferred minor).
Task 4: minor (deferred): add a constructed unsupported-online fallback test that asserts the full resolved diagnostics tuple (`materialized_reference`, `f32`, `token_major`); final reviewer must triage.
Task 4: target verification passed at 6ef2b53 — NVCC build/link succeeded, `brt_workspace_layout_test` passed, and opt-in `brt_qwen35_executor_test` passed 1/1 on idle RTX 5090 after two successful GPU preflights.
Task 4: complete (commits 8979835..6ef2b53, review approved with 1 deferred minor).
Task 5: target compile fix round 1 — NVCC exposed C `assert` macro parsing of inline vector initializer commas; replaced them with named expected vectors in a2beb0a.
Task 5: target compile fix round 2 — NVCC exposed invalid `std::vector<float>::size_bytes()` in the grouped-logits parity assertion; replaced it with the exact vector byte count in d77d92e.
Task 5: target compile passed at d77d92e — NVCC 13.2.78 compiled and linked `brt_qwen35_executor_test` for `sm_120a` in isolated build `/home/charles/brt-validation/task5-a2beb0a/build`.
Task 5: pending — focused GPU CTest and four-prompt exact-token parity require the externally approved shared-GPU execution step; no GPU command was run after the approval service rejected the combined request.
Task 5: independent review approved at d77d92e — spec compliant; no Critical, Important, or Minor findings.
Task 5: target focused GPU CTest passed at d77d92e — `brt_qwen35_executor_test` passed 1/1 in 1.11 seconds after a successful idle-GPU preflight.
Task 5: parity environment blocker — the fixed `brt-dev:26.06-cuda13` image has `curl` and `flock` but no `jq`; the parity script exited during dependency validation before loading either model. A read-only host `jq`/library bind-mount probe was rejected by the external approval service and was not executed.
Task 5: parity environment resolved with explicit user approval — host `jq`, `libjq.so.1`, and `libonig.so.5` were mounted read-only into the ephemeral validation container; a no-GPU probe returned `{"jq_ok":true}`.
Task 5: target parity passed at d77d92e — pinned BF16 artifact SHA-256 `5e2d54b1b54df02cf1797e6a5e1465255ed68a9547bfd0ab0bde1357347d65e8`; 4/4 fixed prompts passed exact greedy-token parity with 32 generated tokens each. Evidence: `/home/charles/brt-validation/task5-a2beb0a/qwen35-task5-parity.jsonl`.
Task 5: target post-verification preflight passed — GPU 0 returned idle with 31972 MiB free and 0% utilization.
Task 5: late architecture-review Important resolved by target evidence — grouped casts are the default policy, and the exact four-prompt/32-token real-model parity run above exercised that default path successfully.
Task 5: minor (deferred): test-only grouped-cast launch accounting uses translation-unit global mutable state and assumes the current serial CUDA test harness; final reviewer must triage thread-safety/documentation before merge.
Task 5: complete (commits 6ef2b53..d77d92e, review clean; target NVCC build, focused GPU CTest, and four-prompt exact-token parity passed).
Task 6: target compile passed at 608e2bf — NVCC 13.2.78 compiled and linked `brt_qwen35_cuda_graph_test` and `brt_qwen35_executor_test` for `sm_120a` in isolated build `/home/charles/brt-validation/task6-608e2bf/build`.
Task 6: review requested changes — capture path reads an unmaterialized pinned result and pinned host scalars are not exception-safe across later constructor failures; fix round 1 dispatched to the original implementer.
Task 6: minor (deferred): generic graph failure coverage exercises a throwing capture body but does not force a CUDA API invalidation/instantiation failure; final reviewer must triage.
Task 6: fix round 1/5 (2 addressed, 0 original findings open — capture avoids pending D2H dereference; pinned scalars use constructor-unwind-safe RAII; commits 608e2bf..53f8114).
Task 6: target fix-round compile passed at 53f8114 — NVCC 13.2.78 compiled and linked graph/executor tests for `sm_120a`; `brt_qwen35_executor_test` passed, while `brt_qwen35_cuda_graph_test` exposed a new test-only reset/replay assertion error.
Task 6: fix round 2 root cause — the strengthened assertion assumed every sequence's first decode captures, but executor `reset()` intentionally preserves captured topology, so the repeated sequence's first decode replays and correctly sets `decode_graph_replayed=true`.
Task 6: fix round 2/5 (1 addressed, 0 open — first sequence requires capture/no replay; post-reset sequence requires immediate replay; commit 6404df4; scoped review approved).
Task 6: target resync blocked after 6404df4 — external approval service rejected the rsync request because its own review stream disconnected; no workaround attempted. A fresh explicit user approval is required before target rebuild/retest/parity.
Task 6: target verification passed at 6404df4 — NVCC 13.2.78 compiled graph/executor/parity targets for `sm_120a`; `brt_qwen35_executor_test` and `brt_qwen35_cuda_graph_test` passed 2/2 on idle RTX 5090.
Task 6: decode allocation gate passed — warmed online graph replay remained at zero current/peak/total RMM allocation bytes and counts under the executor test statistics adaptor.
Task 6: target parity passed at 6404df4 — 4/4 fixed prompts retained exact greedy-token parity with 32 generated tokens each. Evidence: `/home/charles/brt-validation/task6-6404df4/qwen35-task6-parity.jsonl`.
Task 6: target post-verification preflight passed — GPU 0 returned idle with 31972 MiB free and 0% utilization.
Task 6: minor (deferred): `qwen35_executor_source_test.cmake` is an implementation-text change detector rather than a behavioral test; final reviewer must decide whether to delete it after the CUDA graph behavioral suite has proven the same contracts.
Task 6: complete (commits d77d92e..6404df4, review clean after fix rounds 1–2; target NVCC build, graph/executor GPU CTest, zero decode allocation gate, and four-prompt exact-token parity passed).
Task 7: in progress — construction-time cuBLASLt candidate enumeration/median selection for token buckets 1, 128, and 512 dispatched from base 6404df4; measured prefill/decode selection is forbidden.
Task 7: target compile failed at 9ccf79e — NVCC 13.2.78 reported unqualified `algorithm_id(...)` lookup collision in `cublaslt_matmul.cu:398`; no GPU test executed.
Task 7: review requested changes — CUDA compile blocker; enumeration not hard-capped at 16; selected algorithm IDs absent from production diagnostics; graph-capture test uses materialized policy that cannot capture.
Task 7: minor (deferred): test-only `execution_calls` remains zero because no real selection/enumeration entry point increments it, so it cannot independently detect a future runtime-selection regression; final reviewer must triage after production diagnostics coverage is added.
Task 7: fix round 1 target compile passed at fde7ec2 — CUDA 13.2.78 compiled and linked cuBLASLt, executor, and graph tests for `sm_120a`.
Task 7: fix round 1 target tests exposed two regressions — `brt_qwen35_executor_test` rejects the new context-length-4 fixture paired with `max_context=128`; `brt_qwen35_cuda_graph_test` intermittently fails exact cross-executor state/logit equality.
Task 7: graph failure root cause confirmed with a target-only diagnostic assertion — repeated runs eventually showed independently constructed ordinary/graph executors selecting different `cublaslt_algorithm_ids`; exact cross-executor equality is therefore not a valid graph-replay oracle under construction-time autotuning.
Task 7: fix round 2 local complete at 5a161c5 — release fixture/context now cover tuned buckets 1/128/512, production diagnostics expose bucket/shape/tuned selected-plan metadata, graph oracle checks exact same-executor capture/replay with tolerant independent-executor parity, and construction diagnostics remain stable across prefill/decode/capture/replay; local host suite and diff checks passed, target NVCC/GPU rerun delegated to controller.
Task 7: fix round 2 target compile and focused smoke passed at 5a161c5 — CUDA 13.2.78 built all three `sm_120a` test targets; cuBLASLt, executor, and graph GPU tests passed 3/3 once.
Task 7: fix round 2 stability gate failed — `brt_qwen35_cuda_graph_test` failed on repeat 5/12 at the new 2% cross-executor tolerance assertion; the exact same-executor capture/replay oracle did not fail. Cross-executor comparison remains invalid under independent construction-time autotuning and is assigned to fix round 3 for removal rather than arbitrary tolerance widening.
Task 7: fix round 3 local complete at 243e5bc — removed the independent ordinary executor and cross-executor tolerance oracle from the graph test, preserved exact same-graph-executor capture/reset/replay observation checks and graph diagnostics, and kept construction-selected cuBLASLt diagnostics stable during decode; local host suite and diff checks passed, target NVCC/GPU repeat gate delegated to controller.
Task 7: complete at 243e5bc — CUDA 13.2 `sm_120a` compile passed; focused cuBLASLt/executor/graph GPU tests passed 3/3; graph repeat `ctest --repeat until-fail:12` passed 12/12 after Fix Round 3; independent review approved; real Qwen3.5-9B BF16 parity passed 4/4 exact greedy-token prompts with 32 generated tokens each (evidence `/home/charles/brt-validation/task7-243e5bc/qwen35-task7-parity.jsonl`); post-verification preflight idle with 31972 MiB free, 0% utilization, 35C.
Task 8: in progress from base 575445c — retain the existing optimized register-resident Gated DeltaNet path as baseline; add correctness-gated `sm_120a` prefill/decode candidates and promote only a measurably faster schedule for exact release dimensions/buckets.
Task 8: local implementation complete — added policy/diagnostic interfaces, explicit 64/128 sm120 candidate recurrent schedules, construction-time immutable executor diagnostics, and CUDA behavioral tests for 17/128/512 prefill, continued prefill, and eight decodes; current schedule remains the selected fallback pending controller RTX5090 benchmark evidence. Local `git diff --check` and `scripts/local-check.sh` passed; target CUDA compile/tests/parity/microbench delegated to controller.
