# Licences

- **The recipe's original code and documentation** are licensed under the Apache License 2.0 (`LICENSE`), unless a
  file or directory says otherwise.
- **Third-party code keeps its licence.** Files modified through `patches/` keep the upstream SPDX and copyright headers
  and the upstream licence (vLLM, b12x and the sparse-MLA plugin: Apache-2.0; NCCL: `nccl-LICENSE.txt`). Files adapted
  from third-party code keep that code's licence and are listed in `NOTICE`.
- **Served files with their own header** keep it. Several files written in the lab carry an Apache-2.0 SPDX header or
  were declared Apache-2.0 by the lab work that produced them (mHC prefill helper, KDA checkpoint modules, KDA state
  store, prefill RDMA package); they stay Apache-2.0. The recipe's own files without a header are covered by the root
  licence, Apache-2.0. Four of the recipe's own served files, which the lab plans serve with the SPDX line of the
  recipe's earlier licence, are served with the line "SPDX-License-Identifier: Apache-2.0" instead
  (`glm53_speedup/adaptive_chunk.py` of both profiles, and the TP2 profile's `b12x/moe/fused_moe/_tp2_cluster.py` and
  `vllm/model_executor/kernels/mhc/tp2_geometry.py`): a change of that comment line only, recorded per file in
  `sources/installed-files.json` (`lab_served_sha256`, `recipe_change`).
- **The base image's stage inputs** in `docker/base` keep the licences the lab declared for them, except that the
  lab's own files carry the repository licence: upstream files and patches of vLLM, b12x, FlashInfer and the
  sparse-MLA plugin stay Apache-2.0 with their headers; the lab's installers, tests and manifests are Apache-2.0, and
  their SPDX lines say so (`NOTICE`, "Base image stages"; `docker/base/README.md`). The images built from
  `docker/Dockerfile` contain much more third-party software, under its own terms (`docs/redistribution.md`).
- The served bytes are pinned by sha256, so a header is changed only by a new plan that is staged and served; the SPDX
  lines of the four files above are the one exception, recorded per file. The modified third-party files are
  identified by their patch headers, `sources/installed-files.json` and this table. Those of both profiles carry the
  recipe's "Modified by the GLM-5.3 RiNGSiDE recipe (othexmr)" notice inside the file. `NOTICE` carries the provenance
  that a header lacks (for example SparkRing for the lean all-reduce files).

Texts: `Apache-2.0.txt`, `nccl-LICENSE.txt`, `MIT-zai-org-GLM-5.3-Flash.txt` (the chat template),
`MIT-fla-org-flash-linear-attention.txt` (the flash-linear-attention kernel code in the TP4 o_norm FP8 kernel),
`MIT-MiaAI-Lab-GLM-5.3-Flash-EXL3-2x-DGX-Sparks.txt` (the MIT notice MiaAI-Lab keeps for its contributions before
2026-09-07, among them the chat template's thinking gate), `BSD-3-Clause-NVIDIA-CUTLASS.txt` (two vendored CUTLASS-derived headers of the FP8 GEMM build), the
`NOTICE` files of SparkRing (`sparkring-NOTICE.txt`) and switchless-nccl (`switchless-nccl-NOTICE.txt`), and the plugin
and vendored licences inside `src/tp4/glm53_lean_allreduce/LICENSE` and `build/glm53_fp8_gemm/vendor/LICENSE-vllm`.
`components.json` maps every attribution key to its upstream, licence and texts.

Models and weights are not in this repository; their licences are those of their Hugging Face repositories
(RedHatAI/GLM-5.3-Flash-NVFP4 states MIT, and so does nvidia/GLM-5.3-Flash-NVFP4, whose tokenizer.json the profiles
read; zai-org/GLM-5.3-Flash is MIT, Copyright (c) 2026 Z.AI Co., Ltd; the drafter incoai/GLM-5.3-Flash-DFlash2 states CC
BY-NC-ND 4.0 and is released for research and evaluation).

The optional weightless steering (`launch/options/weightless/prepare.py`, Apache-2.0 like the recipe's own code): the
edits it applies to `model.py` are, like every other patch to that vLLM file, under the file's Apache-2.0 licence;
msuiche/weightless code is fetched by the operator from its repository and is not part of this repository or its
licence.

## Every served file

Generated from `sources/installed-files.json` (`licence` and `attribution` of each entry). Native artefacts and the
Humming package are listed in `release/assets.json`.

| served path | profiles | kind | licence | attribution keys (NOTICE, components.json) |
|---|---|---|---|---|
| `/chat_template_mm.jinja` | tp2, tp4 | template | MIT (zai-org/GLM-5.3-Flash chat template, revision 690b705, with Raymond Lucke's thinking gate from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks eaf90d0, 2026-08-27, MIT period) | glm-template, miaai-lab-chat-template |
| `/opt/glm53-speedup/runtime/` | tp2, tp4 | config-dir | Apache-2.0 (repository licence) | - |
| `/opt/switchless-nccl/` | tp4 | binary-dir | NCCL LICENSE.txt (Apache-2.0 with BSD portions) | nccl, sparkring-nccl, switchless-nccl |
| `b12x/moe/_shared/kernels/dynamic.py` | tp2, tp4 | patch | Apache-2.0 (b12x file) | - |
| `b12x/moe/fused_moe/_impl.py` | tp2, tp4 | patch | Apache-2.0 (b12x file) | - |
| `b12x/moe/fused_moe/_tp2_cluster.py` | tp2 | src | Apache-2.0 (SPDX header) | - |
| `b12x/moe/fused_moe/_tp4_cluster_candidate.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `b12x/moe/fused_moe/_tp4_tile_major.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `deterministic_sparse_topk.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_greedy_verify.py` | tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_indexer_rowsplit.py` | tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_kda_ckpt.py` | tp4 | src | Apache-2.0 (SPDX header with the vLLM copyright line; contains b12x code) | b12x, vllm, sparkring-kda-ckpt |
| `glm53_kda_ckpt_plan.py` | tp4 | src | Apache-2.0 (declared by the lab work that wrote it) | sparkring-kda-ckpt |
| `glm53_kda_conv.py` | tp4 | src | Apache-2.0 (SPDX header with the vLLM copyright line; contains b12x's KDA prefill prepare kernel) | b12x, vllm |
| `glm53_lean_allreduce/` | tp4 | src-dir | Apache-2.0 (LICENSE in the package directory) | b12x-roce, sparkring-roce |
| `glm53_prefill_rdma/` | tp4 | src-dir | Apache-2.0 (SPDX headers) | - |
| `glm53_replay_boundary.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_sparse_mla/backend.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `glm53_speedup/adaptive_chunk.py` | tp2, tp4 | src | Apache-2.0 (SPDX header) | - |
| `glm53_speedup/cadence.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_speedup/fair_prefill.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_speedup/geometry.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_speedup/graphs.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_speedup/policy.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_speedup/scheduler.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_speedup/telemetry.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `glm53_utf8_guard.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `humming/` | tp4 | third-party-dir | Apache-2.0 (humming-kernels) | humming |
| `humming_kernels-0.1.12.dist-info/` | tp4 | third-party-dir | Apache-2.0 (humming-kernels) | humming |
| `nvfp4_topk_rowspread/` | tp4 | src-dir | Apache-2.0 (_kernel.so: built from vLLM's persistent top-k sources with the hunks of vllm-project/vllm pull request 55314 by Dovis01, after sgl-project/sglang#37625 by ormandj and bold84, and the rowspread change, build/topk_rowspread) and the repository licence (__init__.py, the loader, no SPDX header) | vllm |
| `vllm/config/vllm.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/distributed/device_communicators/cuda_communicator.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/entrypoints/launchers/api_server/entry.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/model_executor/kernels/linear/nvfp4/b12x.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/model_executor/kernels/mhc/tilelang.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/model_executor/kernels/mhc/tp2_geometry.py` | tp2 | src | Apache-2.0 (SPDX header) | - |
| `vllm/model_executor/layers/attention/mla_attention.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/model_executor/layers/fused_moe/b12x.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-mhc |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-mhc |
| `vllm/model_executor/layers/linear.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-mhc |
| `vllm/model_executor/layers/mla.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-mhc |
| `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` | tp2, tp4 | src | Apache-2.0 (repository licence; the file carries no SPDX header) | - |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/models/common/ops/glm53_qnorm_fp8.py` | tp4 | src | Apache-2.0 (SPDX header with the vLLM copyright line; adapted from vLLM's fused q/kv RMSNorm kernel, vllm/models/common/ops/fused_qk_rmsnorm.py) | vllm |
| `vllm/models/glm5next/nvidia/attention.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained; its kpool tail ring size is a port of vllm-project/vllm#58454 (mmastrac) as Morrowmake/vllm-cmp170hx 1d4b59e ported it) | sparkring-mhc, vllm-pr-58454, morrowmake-vllm-cmp170hx |
| `vllm/models/glm5next/nvidia/kda.py` | tp2 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-mhc |
| `vllm/models/glm5next/nvidia/kda.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-mhc, sparkring-kda-ckpt |
| `vllm/models/glm5next/nvidia/kda_fp8_handoff.py` | tp4 | src | Apache-2.0 (SPDX header) | - |
| `vllm/models/glm5next/nvidia/kda_state_store.py` | tp4 | src | Apache-2.0 (SPDX header with the vLLM project copyright line) | vllm |
| `vllm/models/glm5next/nvidia/mhc_prefill_sharding.py` | tp4 | src | Apache-2.0 (SPDX header) | sparkring-mhc |
| `vllm/models/glm5next/nvidia/model.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained; its DFlash2 auxiliary hidden-state capture is adapted from vLLM's DeepseekV4Model.forward in vllm/models/deepseek_v4/nvidia/model.py) | sparkring-mhc, vllm |
| `vllm/models/glm5next/nvidia/model.py` | tp2 | patch | Apache-2.0 (upstream file; SPDX header retained; its DFlash2 auxiliary hidden-state capture is adapted from vLLM's DeepseekV4Model.forward in vllm/models/deepseek_v4/nvidia/model.py) | vllm |
| `vllm/models/glm5next/nvidia/ops/glm53_onorm_fp8.py` | tp4 | src | Apache-2.0 (SPDX header; contains code copied from flash-linear-attention, MIT, with its notice) | vllm, flash-linear-attention |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained; its kpool tail ring is a port of vllm-project/vllm#58454 (mmastrac) as Morrowmake/vllm-cmp170hx 1d4b59e ported it) | vllm-pr-58454, morrowmake-vllm-cmp170hx |
| `vllm/models/glm5next/nvidia/ops/recoverssm.py` | tp2, tp4 | src | Apache-2.0 (SPDX header; adapted from vLLM's Kimi-K3 RecoverSSM) | vllm |
| `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py` | tp2 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/core/kv_cache_utils.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained; its DFlash2 drafter KV-cache group is built on vLLM's own KV-cache group mechanisms: is_eagle_group, UniformTypeKVCacheSpecs and the shared KV-cache tensors of the hybrid allocator) | vllm |
| `vllm/v1/core/sched/scheduler.py` | tp2 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/core/sched/scheduler.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-kda-ckpt |
| `vllm/v1/core/single_type_kv_cache_manager.py` | tp2 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/core/single_type_kv_cache_manager.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-kda-ckpt |
| `vllm/v1/sample/ops/topk_topp_triton.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/worker/gpu/async_utils.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/worker/gpu/model_runner.py` | tp2 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/worker/gpu/model_runner.py` | tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | sparkring-kda-ckpt |
| `vllm/v1/worker/gpu/sample/gumbel.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained; its vocabulary clamp is a port of vllm-project/vllm#50843 (alexbi29)) | vllm-pr-50843 |
| `vllm/v1/worker/gpu/sample/sampler.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained; its vocabulary clamp is a port of vllm-project/vllm#50843 (alexbi29)) | vllm-pr-50843 |
| `vllm/v1/worker/gpu_worker.py` | tp2, tp4 | patch | Apache-2.0 (upstream file; SPDX header retained) | - |
