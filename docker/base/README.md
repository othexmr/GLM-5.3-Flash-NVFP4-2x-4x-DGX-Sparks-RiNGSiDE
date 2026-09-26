# Base image stages

The inputs of the lab's image stages, from which `docker/Dockerfile` rebuilds the base image that both profiles were
measured on (image ID `sha256:017fd0ba...`, `docs/image.md`), byte for byte except the files named below. The lab built that image in stages from
the vLLM source image of ZJY0516/vllm `4500c80c`; the stage inputs come from the lab's earlier two-Spark recipe
repository and its research archive, neither of which is public. `docs/build-provenance.md` records where each input
comes from and how the recipe checked it.

`SHA256SUMS` lists every file of this directory; the Dockerfile checks it before it uses any of them. Each stage also
keeps its own hash gates (the preimage and result sha256 of every file it changes), so a stage refuses to run on a
different parent. The lab's per-stage Dockerfiles are not kept: `docker/Dockerfile` runs the same commands, one build
step per stage. Files that differ from the lab's copies:
- the docstring of `09-drafter/patch_vllm_dflash2_edge_scale.py` no longer names a reviewer; its code and the text it
  inserts into vLLM are unchanged;
- the SPDX line of every file the lab wrote names the repository licence, Apache-2.0 (2026-09-26), also in the lab's
  integration scripts, whose line named vLLM's Apache-2.0 as well; with it the hashes that pin those files:
  `SHA256SUMS`, the three GLM-5.3 parser files in `12-reasoning/source-manifest.json` and in the port's
  `source-port-manifest.json` and `vllm-tree.sha256` (the port patch adds the same three files), and the parser pins
  of `thinking-gate/check_preimage.py` and `thinking-gate/thinking_contract.py`. `thinking-gate/prepare-receipt.json`
  is the lab's receipt and keeps the lab's hashes. A rebuilt base therefore also differs from the measured image in
  the SPDX line of `vllm/parser/glm53_moe.py`, `vllm/reasoning/glm53_moe_reasoning_parser.py` and
  `vllm/tool_parsers/glm53_moe_tool_parser.py` and of the tests in `/opt/glm53-reasoning-fix`;
- the DFlash2 auxiliary hidden-state capture and drafter KV-cache group are the recipe's, the code that the TP4
  profile serves (`NOTICE`), in place of the lab's earlier versions: stage 02's
  `glm_release_4500c80c_glm5next_dflash_aux.patch` (the capture in `model.py`),
  `glm_release_4500c80c_glm5next_dflash_group.patch` (the drafter group in `_get_kv_cache_groups_glm5_next`, with its
  exact-fit block size) and `glm_release_4500c80c_dflash_exact_fit_layout.patch` (the tensor-layout recognizer and its
  three consumers), the stage oracle `02-kernels/test_glm5_dflash_group.py`, and the `model.py` and
  `kv_cache_utils.py` sections of `v029-port/v029-glm53-local.patch` with their entries in `source-port-manifest.json`
  and `vllm-tree.sha256`. The
  hash gates after them change with them: `08-router/SOURCE.SHA256SUMS` and the stage-02 lines of `docker/Dockerfile`.
  `glm_release_4500c80c_kv_layout_dump.patch` (the lab's own layout record) is unchanged and still applies without
  fuzz. A base rebuilt from these inputs therefore differs from the measured image in
  `vllm/models/glm5next/nvidia/model.py` (`03143971...` instead of `7cddb9d1...`) and `vllm/v1/core/kv_cache_utils.py`
  (`2614f957...` instead of `7ef72953...`), the preimages of the profiles' patches of those files. It has not been
  built yet (`docs/build-provenance.md`). Two binary data files of stage 04
(`b12x/policy/_profiles/data/*.json.gz`, b12x policy profiles) are stored as base64 text (`*.json.gz.b64`) and decoded
by the Dockerfile; stage 04 checks the decoded bytes against the lab's sums. `04-b12x-kda/README.md` is the recipe's
note on where that stage's b12x files come from, not a stage input.

`12-reasoning` holds the revision of stage 12 that the lab built into the measured image's chain (2026-09-08): the
measured image carries its `source-manifest.json` (`27dca3bb...`) and parser copy `vllm/parser/glm53_moe.py`
(`db19a7f7...`, also the parser of the port) in `/opt/glm53-reasoning-fix`; the copies here differ from those only in
the SPDX line (above). The lab changed that directory again on 2026-09-13 for the thinking gate: the gated parser
(`a0d5d6e9...` in the lab, here with the same SPDX change), its manifest and seven of the eight tests, rewritten for
the port. Those later files are the inputs of `thinking-gate/`, which copies its tests over stage 12's in
`/opt/glm53-reasoning-fix`, as the lab's thinking gate did. They are not stage 12's: the later Responses identity test
expects the `response.incomplete` event of stage 19's Responses repair and fails on the stage-12 image (the first
build of this Dockerfile on 2026-09-26 stopped there while this directory held the later revision).

## Order and effect

The chain is `00-source`, `01` to `17`, `19`, `v029-port`, `21`, `thinking-gate`, `mixed-prefill-repair`. The
vLLM v0.29.0 port reinstalls the whole `vllm` package, so the vLLM file changes of stages 01 to 19 do not reach the
base image; they run because every later stage checks its parent's bytes. What survives into the base image:

| Stage | Changes | In the base image |
|---|---|---|
| `00-source` | `distributions.txt`: the Python distributions of the lab's vLLM source build (2026-09-03) | the Python environment (docker/tools/pin_source.py pins a rebuild to it) |
| `01-compat` | legacy `vllm._C` import shim | replaced by the port |
| `02-kernels` | b12x 1.3.0 with its CUDA Python and CUTLASS DSL wheels (`../base/wheels.SHA256SUMS`); the recipe's DFlash2 capture and drafter KV-cache group and the lab's KV-layout record for the glm-release vLLM; builds the `glm53_sparse_mla` plugin (Libertai/vllm-sparse-mla-blackwell `b1f20638`) and patches its backend | b12x, the wheels, the plugin (its backend changed again by 14, 19, 21); the vLLM patches are replaced |
| `03-flashkda` | FlashKDA prefill option of the GLM-5.3 KDA layer | replaced by the port |
| `04-b12x-kda` | the b12x KDA prefill `policy` and `sequence` packages (b12x's KDA prefill work at `4b389912`, three files differing, see `04-b12x-kda/README.md`) and their vLLM integration | the b12x packages |
| `05-layout` to `10-collectives` | DFlash XQA layout, dual FP8 dense path, FP8 MoE oracle, router dedup, DFlash2 edge scale, TP2 all-reduce switch (vLLM files) | replaced by the port |
| `11-moe` | b12x decode fast path (`b12x/moe/fused_moe/_impl.py`, `b12x/moe/_shared/kernels/dynamic.py`) | yes (changed again by 13, 19) |
| `12-reasoning` | GLM-5.3 reasoning and tool parser fixes; their tests in `/opt/glm53-reasoning-fix` | `/opt/glm53-reasoning-fix` (stages 17 and 19 run its tests; the thinking gate replaces them); the vLLM files are replaced |
| `13-deterministic-fastpath` | b12x deterministic decode fast path | yes |
| `14-attention-tail` | full padded KPool support of the sparse-MLA backend | yes (changed again by 19, 21) |
| `15-topk-repair` | the `nvfp4_topk_pr55314` top-k extension (hunks of vllm-project/vllm pull request 55314), built in the stage | the extension; its vLLM callsite is replaced |
| `16-deterministic-topk` | `deterministic_sparse_topk.py` | yes (changed again by 19; both profiles serve their own copy) |
| `17-tool-call-truncation` | parser engine | replaced by the port |
| `19-runtime-bugfixes` | runtime repairs of vLLM, b12x `_impl.py`, the sparse-MLA backend and the top-k helper; the drafter batch fix of FlashInfer's XQA sources (`drafter-batch-fix`) | the b12x, sparse-MLA, top-k and FlashInfer files; the vLLM files are replaced |
| `v029-port` | vLLM v0.29.0 (`98dff2a8`) with the GLM-5.3 backport and the lab's Python changes (`v029-glm53-local.patch`; its `model.py` and `kv_cache_utils.py` carry the recipe's DFlash2 capture and KV-cache group); `source-port-manifest.json` (118 changed files), `vllm-tree.sha256` (every file of the ported `vllm/` tree), `assemble-wheel.py` | the `vllm` package: official v0.29.0 wheel natives, ported Python tree, rebuilt Rust artifacts |
| `21-atomic-prefill` | `vllm/model_executor/layers/fused_moe/b12x.py` and the sparse-MLA backend | yes: both are patch preimages of the profiles |
| `thinking-gate` | the thinking-gated `vllm/parser/glm53_moe.py`, its tests and template contract, and the served chat template (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks's `files/chat_template.jinja`) | yes |
| `mixed-prefill-repair` | `vllm/outputs.py`, `vllm/v1/core/sched/scheduler.py` (a patch preimage of the profiles), `vllm/v1/worker/gpu/cudagraph_utils.py`, the first `glm53_speedup` package | yes |

## Licences

The files keep the licences the lab declared for them in its earlier recipe repository, except that the lab's own
files carry the repository licence: installers, tests, manifests and checksum lists written in the lab are Apache-2.0,
and their SPDX lines say so; files and patches taken from or changing vLLM, b12x, FlashInfer and the sparse-MLA plugin
keep their upstream Apache-2.0 licence and headers, with the lab's changes. `15-topk-repair/LICENSE` is the Apache-2.0
text that bundle carries. `NOTICE` ("Base image stages") and `licenses/components.json` (`vllm`, `b12x`,
`b12x-kda-prefill`, `flashinfer`, `sparse-mla`) list the upstream projects. `thinking-gate/chat_template_mm.jinja` is
the served chat template: zai-org's template (MIT) with Raymond Lucke's thinking gate from
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks eaf90d0 (MIT), credited in `NOTICE` and `licenses/components.json`
(`glm-template`, `miaai-lab-chat-template`).
