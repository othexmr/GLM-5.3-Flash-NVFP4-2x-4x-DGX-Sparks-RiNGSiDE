# Build provenance

Audit of every input of the base image and of the native artefacts, 2026-09-25. It was done on a CPU from this
repository, the lab's records and the lab's saved copies of public inputs, without network access and without
building an image: nothing below was built or run on a GPU for this audit. `docker/Dockerfile` implements the
result; `docker/README.md` has the commands.

## Verdict

- **Rebuildable from public sources plus this repository.** With the lab's stage inputs now in `docker/base`, every
  input of the base image and of the native artefacts is either public (pinned by commit, Git tree, sha256 or image
  digest) or in this repository. No input is missing.
- **The Python payload follows from the records, checked offline.** The vLLM v0.29.0 port patch, applied to the public
  v0.29.0 source, reproduces the lab's ported `vllm/` tree file for file (2,948 files), except `model.py` and
  `kv_cache_utils.py`, which carry the recipe's DFlash2 code, and the SPDX line of the lab's three GLM-5.3 parser
  files (`docker/base/README.md`). All 54 patch preimages of both profiles follow from the recorded stage results
  (`tests/cpu/test_easy_build.py` re-checks the chain on every CPU run), and the b12x and sparse-MLA chains were
  replayed from the public b12x 1.3.0 wheel and the public plugin source with the stage inputs (below).
- **A rebuild is a new image, not the measured one.** Native binaries are compiled again and differ from the measured
  bytes (the builds are not bit-reproducible); Ubuntu, CUDA and Rocky Linux packages are resolved at build time; the
  image ID always differs. A self-built image needs its own qualification before its results can stand next to the
  measured ones (`docker/README.md`, "Qualify a self-built image").
- **Not yet built.** `docker/Dockerfile` was written from the lab's build records and checked on a CPU only. Its first
  build on a DGX Spark is the test of the file itself.

## Blockers

Before this change the base image could not be rebuilt from public sources plus this repository. The missing inputs
were:

1. **The lab's image stages** (stages 01 to 21, the vLLM v0.29.0 port patch and its manifests, the thinking gate, the
   mixed-prefill repair). They existed only in the lab's earlier two-Spark recipe repository and its research archive,
   neither of which is public. Now in `docker/base`, byte for byte except one docstring, the recipe's DFlash2 capture
   and KV-cache group and the SPDX line of the lab's own files (`docker/base/README.md`), with
   `docker/base/SHA256SUMS`.
2. **The b12x KDA prefill packages** (`b12x/policy`, `b12x/sequence`) that stage 04 installs. They are b12x's public
   KDA prefill work: 19 of their 22 files are byte-identical (Git blob) to local-inference-lab/b12x commit `4b389912`
   (Luke Alonso, 2026-09-04, "fix(kda-prefill): order prepare before recurrence"; checked on GitHub on 2026-09-25).
   The lab took them from its local rewrite `c083e1d4` (2026-09-05, not public) on the b12x commit `287be2e4`.
   Three files differ from `4b389912`: `b12x/sequence/kda_prefill/_cute_kernels.py`, the lab's combination of the
   upstream fixes `b794879d` (Martin Vit) and `4b389912` (Luke Alonso), and the two policy profile data files
   (`b12x/policy/_profiles/data/*.json.gz`), which are those of `287be2e4` (in the lab's clone of b12x). Now in
   `docker/base/04-b12x-kda` under their Apache-2.0 licence; its `README.md` has the details.
3. **The Python environment of the vLLM source build.** The upstream Dockerfile resolves several requirements when it
   runs. Now recorded: `docker/base/00-source/distributions.txt` (228 distributions of the lab's build of
   2026-09-03); `docker/tools/pin_source.py` pins a rebuild back to them.

**No blocker remains for the documented chain.** What remains are risks, each of which makes the build stop rather
than produce a different image silently where a check exists:

| Risk | Where the build stops |
|---|---|
| The glm-release commit `4500c80c` is fetched from the fork ZJY0516/vllm (the lab's record); a fork commit can disappear. Whether vllm-project/vllm also carries the commit was not checked here (no network). | the `vllm-source` build (Git fetch of the commit) |
| The two base images are pinned by the digests the lab's build resolved. NVIDIA removes images of old CUDA releases from Docker Hub; a removed digest cannot be pulled. | the `vllm-source` build (image pull) |
| PyPI, the PyTorch and FlashInfer package indexes, the GitHub release asset, rustup, crates.io and the Ubuntu, NVIDIA CUDA, deadsnakes and Rocky Linux package repositories must be reachable. | the step that downloads |
| The vLLM source build resolves requirements at build time: the lab's rebuild of 2026-09-08 already installed newer releases of 15 third-party distributions than its build of 2026-09-03 (among them openai, anthropic, mcp, fastsafetensors, boto3). | stage `source`: `pin_source.py` re-pins them (default `SOURCE_PIN=exact`) or fails (`check`) |
| Local builds inside the vLLM source image (vLLM 4500c80c itself, DeepEP) cannot be re-pinned. | `pin_source.py` (vllm must carry `+g4500c80c`, deep-ep `2.0.0+local`) |

## Inputs

| Input | Upstream and version | Obtained | Pinned by | Change set | Status |
|---|---|---|---|---|---|
| NVIDIA base container | `nvidia/cuda:13.0.3-base-ubuntu24.04`, resolved by the lab's source build to `@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1` | Docker Hub, as `FINAL_BASE_IMAGE` of the vLLM source build | image digest | none | public |
| Build base of the vLLM source build and of the Rust stage | `pytorch/manylinuxaarch64-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359@sha256:f91599c49f526c77d01b68286f2bf943a5fd6a432d7e3f0afcc5784825908fe9` | Docker Hub, as `BUILD_BASE_IMAGE` | image digest | the lab's only change to the upstream Dockerfile (`arm64.patch`, sha256 `fa792300...`: the default of this build argument); passed as a build argument instead | public |
| vLLM source image | ZJY0516/vllm `4500c80c080328dfe62435d083f4063e00d987df` (branch glm-release; merge base with vllm-project/vllm main `ee17d0d8`), its `docker/Dockerfile` (sha256 `cc2f7297...` in the lab's record), target `vllm-openai`, build arguments `CUDA_VERSION=13.0.3`, `PYTHON_VERSION=3.12`, `torch_cuda_arch_list=12.1a`, `FLASHINFER_VERSION=0.6.18`, `max_jobs=16`, `nvcc_threads=2` | Git context of the compose service `vllm-source` (`launch/compose.py`) | commit | none (upstream build) | public; resolves packages at build time (above) |
| Python distributions of the source image | 228 distributions, among them torch 2.13.0+cu130, torchvision 0.28.0+cu130, torchaudio 2.11.0+cu130, transformers 5.16.1, flashinfer-python and flashinfer-cubin 0.6.18, flashinfer-jit-cache 0.6.18+cu130, triton 3.7.1, nvidia-cutlass-dsl 4.6.2, tilelang 0.1.12, apache-tvm-ffi 0.1.11, humming-kernels 0.1.12, instanttensor 0.1.9, nvidia-nccl-cu13 2.30.7 (the stock NCCL of the TP2 profile) | PyPI, download.pytorch.org (cu130), flashinfer.ai (cu130) | name and version (`docker/base/00-source/distributions.txt`; hashes were not recorded) | none | public |
| Stage 02 wheels | b12x 1.3.0, cuda-python and cuda-bindings 13.3.1, cuda-core 1.0.1, cuda-pathfinder 1.8.1, nvidia-cutlass-dsl, -libs-base, -libs-core, -libs-cu13 4.6.2, apache-tvm-ffi 0.1.13.post3 (downloaded, not installed) | PyPI (`docker/tools/fetch_wheels.py`) | sha256 (`docker/base/wheels.SHA256SUMS`) | none | public |
| vLLM | vllm-project/vllm v0.29.0 `98dff2a81d747d1dba01a47f939f48c3526d4206` (Git tree `d105bc35...`); the official ARM64 CUDA 13.0 wheel `vllm-0.29.0-cp38-abi3-manylinux_2_28_aarch64.whl` (310,033,787 bytes) | Git fetch by commit; the GitHub release asset | tree hash; wheel sha256 `e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b0` | the port below | public |
| GLM-5.3 backport and the lab's vLLM changes | the GLM-5.3 sources of `4500c80c` ported onto v0.29.0, and 33 lab Python changes: 118 changed files | `docker/base/v029-port/v029-glm53-local.patch` | patch sha256, `source-port-manifest.json` (every changed file) and `vllm-tree.sha256` (every file of the ported `vllm/` tree) | in the repository | reproduced offline (below) |
| vLLM Rust artifacts | `vllm-rs` and `_rust_tool_parser` of the ported source; rustc 1.95.0 (the lab's build resolved the source's channel "1.95" to it), crates by `Cargo.lock`, setuptools-rust 1.13.0, setuptools-scm 10.2.3, vcs-versioning 2.3.4, semantic-version 2.10.0, wheel 0.48.0, setuptools from `'setuptools>=77.0.3,<81.0.0'` | built in stage `v029-rust` (the ported source's `rust-build-cache` recipe without its PyTorch install) | toolchain version, lock file, tool versions | none | rebuilt, not reproduced (the lab's were rebuilt too) |
| Lab stages 01 to 19 | the lab's stage inputs, in the revisions the lab built (stage 12 that of 2026-09-08; its directory's revision of 2026-09-13 is the thinking gate's input); the vLLM files they change are replaced by the port, the b12x, sparse-MLA, FlashInfer, top-k and test files remain (`docker/base/README.md`) | `docker/base/01-*` to `19-*` | `docker/base/SHA256SUMS` and each stage's own preimage and result hashes | in the repository | the lab's fresh build of stages 00 to 16 on 2026-09-08 matched its tested image in the compared payload (10,796 of 10,806 files; the other 10 were native libraries, a build receipt and the version stamp) |
| b12x | local-inference-lab/b12x 1.3.0 (the patched files equal commit `3a437ab5`, `sources.lock.json`), with b12x's KDA prefill packages at `4b389912` (three files differ, stage 04), the lab's decode fast path (11), deterministic fast path (13) and runtime repair (19) | PyPI wheel; stage inputs | wheel sha256 `c97d8863...`; stage hashes | in the repository | replayed offline (below) |
| Sparse-MLA plugin | Libertai/vllm-sparse-mla-blackwell `b1f20638d7f82f880e4bcb2136ee2f19c06f38a4` (tree `f72c01f9...`), built in stage 02, with the lab's backend changes (02, 14, 19, 21) | Git fetch by commit; stage inputs | tree hash; the plugin's `SOURCE.SHA256SUMS`; stage hashes | in the repository | replayed offline (below) |
| FlashInfer | flashinfer-python 0.6.18 of the source image, with the lab's XQA drafter batch fix (stage 19: `flashinfer/jit/xqa.py`, `flashinfer/data/csrc/xqa/{mha.cu,mha.h,xqa_wrapper.cu}`) | PyPI; stage inputs | version; stage hashes | in the repository | public plus repository |
| The lab's four base layers | the v0.29.0 port (above), the atomic-prefill stage 21 (`vllm/model_executor/layers/fused_moe/b12x.py`, the sparse-MLA backend), the thinking gate (`vllm/parser/glm53_moe.py`, 2026-09-13) and the mixed-prefill repair (`vllm/outputs.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu/cudagraph_utils.py`, the first `glm53_speedup` package, 2026-09-15) | `docker/base/v029-port`, `21-atomic-prefill`, `thinking-gate`, `mixed-prefill-repair` | stage hashes | in the repository | public plus repository |
| NCCL and the switchless patches (TP4) | NVIDIA/nccl `73cf112295c33aee2b895f329f592f2a9b4b0f97` (v2.30.7-1, tree `3e7de6f9...`) with `patches/nccl/0001` and `0002` (trees `560ba01b...`, `9edccf06...`) | Git fetch by commit | tree hashes after each patch | `patches/nccl` | rebuilt: the measured library came from tree `16e997d8`, the same series before its change-notice comments (`build/nccl/README.md`) |
| Sparse-MLA extensions | the plugin at `b1f20638` with `patches/sparse-mla/build/tp4-so-a1c.patch` (tree `442d83de...`) or `tp2-so-m7e.patch` (tree `ce997603...`) | Git fetch by commit | tree hashes | `patches/sparse-mla/build` | rebuilt (the measured builds came from the trees before the notice comments) |
| FP8 GEMM extension (TP4) | `build/glm53_fp8_gemm` with the CUTLASS v4.5.0 headers that flashinfer-python 0.6.18 ships | this repository | file hashes of the repository | in the repository | rebuilt; `docker/tools/build_fp8_gemm.py` calls the lab's `build_probe.build()`; the GPU-less build links against the CUDA driver stub |
| Prefill RDMA library (TP4) | `src/tp4/glm53_prefill_rdma/_prb.cu` (sha256 `7ffd99abacd3e746...`) | this repository | file hash | in the repository | rebuilt (the measured nodes already served two different builds) |
| Rowspread top-k extension (TP4) | `build/topk_rowspread` (the rowspread `persistent_topk.cuh` and binding) with the top-k sources of stage 15 (`docker/base/15-topk-repair/src`) | this repository | file hashes checked by `build/topk_rowspread/build.py` | in the repository | rebuilt, with its own build receipt (the loader accepts only the library its receipt names; `build/topk_rowspread/README.md`) |
| Humming (TP4) | vllm-project/humming `a74973b5079e42ef861720b62f847ce9d33447f5`, humming-kernels 0.1.15 | Git fetch by commit; build tools by `build/humming/constraints.txt` | commit, constraints | none | rebuilt; installed from the directory the lab installed from, so its dist-info can match (`build/humming/README.md`) |

Build-only stages also take tools from package repositories at build time: git and ca-certificates (apt, stages
`fetch` and `overlay-tools`), the CUDA driver stub package if the image lacks it (apt, stages `fp8-gemm` and `topk-rowspread`), git, curl,
make and perl-core (dnf, stage `v029-rust`) and the rustup installer that `build_rust.sh` runs. None of them is in
the profile images; `tests/cpu/test_easy_build.py` fails if a package manager runs in a stage the profile images
derive from.

## The lab's chain

The measured base image, `sha256:017fd0ba...`, was built on 2026-09-15 (125 layers, 23,932,673,369 bytes, the lab's
build record) as the last of these images. The Dockerfile rebuilds the same chain:

| Step | The lab's image | Recipe stage |
|---|---|---|
| vLLM source build of 4500c80c (2026-09-03, 3,251 s on a DGX Spark) | `sha256:cfeeeca2...` | `vllm-source` (compose service), then `source` |
| stages 01 to 17, 19 (stage 18, a rejected top-k variant, is not in the chain) | stage 19 `sha256:ed546433...` | `s01-compat` to `s19-runtime-bugfixes` |
| vLLM v0.29.0 port (2026-09-10) | `sha256:5c89be8e...` | `v029-wheel`, `v029-rust`, `src-vllm-v029`, `v029-assemble`, `v029-port` |
| atomic-prefill stage 21 (2026-09-11) | `sha256:fbbbe381...` | `s21-atomic-prefill` |
| thinking gate (2026-09-13) | `sha256:092cf1ef...` | `thinking-gate` |
| mixed-prefill repair (2026-09-15) | `sha256:017fd0ba...` | `mixed-prefill-repair`, `base` (checks every preimage) |

The lab's build record of 2026-09-15 tagged the image `local/glm53-speedup:mixed-repair-20260915`; the recipe
records it under `local/glm53-v029:ported-20260910`.

## Checked offline for this audit

With the lab's saved copies of the public inputs, on a CPU:

- The saved vLLM v0.29.0 release archive has sha256 `e9cc3f16...` (the lab's record); its files form Git tree
  `d105bc35...`, the base tree in `sources.lock.json`.
- `git apply` of `v029-glm53-local.patch` on that tree gives all 118 changed files of `source-port-manifest.json`, and
  the `vllm/` tree equals the recorded tree (2,948 files, `vllm-tree.sha256`): the lab's ported source with the
  recipe's `model.py` (`03143971...`) and `kv_cache_utils.py` (`2614f957...`) in place of the lab's `7cddb9d1...` and
  `7ef72953...`, and with the Apache-2.0 SPDX line in the three GLM-5.3 parser files (checked again on 2026-09-26: the
  archive's files form tree `d105bc35...`; `verify_port.py` passes for the current patch with the current records). Of
  the 36 vLLM patch preimages, 21 are files of the port and 11 are unchanged v0.29.0 files, as `sources.lock.json`
  records.
- Stage 02's DFlash2 patches apply without fuzz to the glm-release files (`model.py` `4436911e...`,
  `kv_cache_utils.py` `7d299419...`, the lab's feature preimages), the kept layout-record patch applies after them,
  the stage oracle passes on the result in a CPU module environment (not the installed source image), and stage 08's
  router script turns the stage-02 `model.py` (`388d38b1...`) into the port's (`03143971...`).
- Stage 21 turns the port's `fused_moe/b12x.py` into its preimage; the mixed-prefill repair turns the port's
  `v1/core/sched/scheduler.py` into its preimage; the thinking gate's preimage check equals the port's parser.
- The b12x 1.3.0 wheel (sha256 `c97d8863...`) with stage 04's packages, stage 11's patch script, stage 13's patch and
  stage 19's file gives `dynamic.py` `2ebe82bd...` and `_impl.py` `6984ff84...`, the preimages of both profiles.
- The plugin's `backend.py` at `b1f20638` (`3dcf4769...`) with the patches of stages 02 and 14 and the files of
  stages 19 and 21 gives `8bb06348...`, the preimage of both profiles.

Not checked here: anything that needs a build, a GPU or the network (the availability of the public sources, the
non-Python content of the vLLM source image, the native builds).

## Measured and estimated

Measured by the lab: the vLLM source build took 3,251 s on 2026-09-03; the fresh build of stages 00 to 16 took
56 minutes on 2026-09-08 (its source stage about 33 minutes); the Rust stage of the port took 395 s on 2026-09-10; the
base image is 23.9 GB. Not measured: a build of `docker/Dockerfile` (none has run). `docker/README.md` gives an
estimate, marked as one.
