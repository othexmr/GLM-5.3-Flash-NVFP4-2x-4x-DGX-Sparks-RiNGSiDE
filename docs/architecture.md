# Architecture

## What a launch is

A profile launches one container per rank from the base image. The command sets the environment and the vLLM server
arguments (`docs/configuration.md`) and bind-mounts, read-only, every file that differs from the image: 65 paths for
TP4 and 44 for TP2, plus five site paths (weights, drafter, caches, profiler output). `launch/render.py` turns the
profile's argv template and a site file into those commands.

The self-built profile image of `docker/Dockerfile` carries the same served files at the same paths instead of
mounting them: its build rebuilds the base image from pinned sources and the lab's stage inputs, builds the native
artefacts, runs `sources/apply.py` against the rebuilt base and bakes the result in. `launch/compose.py` launches it
with the same command minus the overlay mounts, one Docker Compose project per rank (`docker/README.md`).

## Where each served file comes from

| Kind | Where it lives | How it is checked |
|---|---|---|
| Modified upstream file (vLLM, b12x, sparse-MLA plugin) | `patches/<component>/<profile>/*.patch` | preimage sha256 (the image file) before `git apply`, served sha256 after |
| The recipe's own modules and new files | `src/<profile>/<install path>` | served sha256 |
| Runtime override, chat template | `launch/profiles/<profile>/` | served sha256 |
| Native artefacts (NCCL, sparse-MLA, FP8 GEMM and rowspread top-k extensions, prefill RDMA library) | not in Git; `build/` has the recipes, `docker/Dockerfile` builds them | sha256 of the measured bytes; a rebuild is reported as rebuilt |
| Humming 0.1.15 | not in Git; vllm-project/humming at a pinned commit, built by `docker/Dockerfile` | per-file sha256 |
| Base image | the lab's image, or `docker/Dockerfile`'s rebuild of it from `docker/base` and pinned sources | image ID; the rebuild checks every patch preimage (`docs/build-provenance.md`) |

`sources/installed-files.json` lists every served path of both profiles with its sha256 and source. The overlay
patches are independent single-file patches: each applies to one image file, and a profile applies its own directory.

## The recipe's main changes

- **Scheduler** (`glm53_speedup`, vLLM scheduler and runner patches): fair prefill cap of 4,608 tokens in mixed steps,
  split cadence, adaptive fair chunk, saturation-aware speculation length (K up to 7) with a full (requests, K) CUDA
  graph grid, prefix-cache retention, startup warm-up, the replay boundary (one complete hybrid state at the latest
  position a DFlash2 replay or extension can hit), the first-token repay (a request that has just produced its first
  token gets one decode-only turn before the next prefill-only chunk) and (TP4) KDA checkpoints saved inside a prefill
  chunk. The verification-length policy follows the adaptive verification length of
  MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (`GLM53_ADAPTIVE_K`: a running average of accepted drafts per request
  picks one verified prefix for the whole batch from the candidate lengths 2, 4 and 7, with a FULL CUDA graph captured
  for every length). The recipe's implementation was written from that project's documentation; its cost mode,
  censored updates, probes and saturation growth (the verified prefix grows from 4 up to 7, one step at a time, while
  every request of the batch keeps accepting its full proposal) are the recipe's own.
- **Transport** (TP4): NCCL Ring over the switchless cycle with both PCIe functions of each port, a neighbour-only
  RDMA all-reduce for small messages, and an RDMA ring for the token-sharded mHC prefill collectives.
- **Kernels**: tile-major NVFP4 MoE weights, atomic and split MoE paths, dense FP8 and Humming paths, sparse-MLA index
  pipeline and split-KV, fused KDA convolution and output paths, fused mHC stages, RecoverSSM (TP4: the fused commit
  with the ordering fix of the RiNGSiDE release), and (TP4) the KDA recurrent state stored in fp16 with fp32 slow channels and
  greedy verification of temperature-0 batches from per-rank argmax pairs.

The lab measured each change against a same-session reference before it entered a profile; the per-change evidence
lives in the lab archive, and `bench/results/README.md` reports the composed profiles. The exceptions are the changes
of the TP4 RiNGSiDE release plan: the change notices, the DFlash2 capture and drafter KV-cache group and the served
fixes (the fused RecoverSSM commit back on, with its ordering fix and the exact-landing column
(`GLM53_RECOVERSSM_FUSED_COMMIT=1`); the KDA FP8 handoff through the tuned FP8 GEMM configurations
(`GLM53_KDA_FP8_HANDOFF_TUNED=1`); the MLA q_a norm with the q_b_proj FP8 quantization fused in
(`GLM53_MLA_QNORM_FP8=2`); the rowspread indexer top-k with pool-width decode logits (`GLM53_TOPK_ROWSPREAD=1`,
`GLM53_INDEXER_POOL_LOGITS=1`); the exact top-p fallback and the UTF-8 continuation guard, mask only
(`GLM53_TOPP_FIX=1`, `GLM53_UTF8_GUARD=1`); the kpool tail ring sized for speculative decoding
(vllm-project/vllm#58454); the sampler kernels' tile argmax clamped to the vocabulary (vllm-project/vllm#50843); the
kpool tail group left out of the generic slot mapping (an out-of-bounds read at long contexts); the
RedHatAI/GLM-5.3-Flash-NVFP4 checkpoint (revision `18d55bfd`) with NVIDIA's `tokenizer.json` and top-p 0.95 as its
generation default; the prefill indexer's query rows split across the ranks (`GLM53_INDEXER_ROW_SPLIT=1`)), which were
checked for correctness on their own and then measured together in the release window (`CURRENT.md`). The TP2
profile's release changes (change notices, the DFlash2 capture and drafter KV-cache group and the correctness fixes)
are verified by GPU leaves and CPU checks and not yet served in a TP2 window.

## Profiles are independent

TP4 and TP2 serve different versions of several files. Each profile has its own patch and `src/` directories, so
one profile can be re-imported without touching the other (`docs/updating.md`).
