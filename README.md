# GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE

Serving recipe for **GLM-5.3-Flash NVFP4** on **four (TP4) or two (TP2) NVIDIA DGX Sparks cabled as a ring, with no
switch**. It is the configuration the lab measured, except that the TP2 profile's release changes are not yet served
in a TP2 window; its numbers follow (Status, below): one launch profile per topology, every file the launch mounts
over the base image (as a patch against the image's file, or in full where the file is the recipe's own), the NCCL
patches, build sources for the native extensions and the base image, and the results.

- Model: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (revision `18d55bfd`),
  quantized from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). The profiles read the
  `tokenizer.json` of [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (revision
  `423acf37`) in place of its own; the two files differ only in a truncation block.
- Also served: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (revision `423acf37`),
  with `SWITCHLESS_MODEL_DIR` pointing at it (the served files are the same). On the NVIDIA checkpoint, TP4 was measured
  before the prefill indexer's query rows split across the ranks (`GLM53_INDEXER_ROW_SPLIT=1`): the release plan's run
  on the NVIDIA checkpoint (lab plan `31978478`, RigMark with 3 runs per cell and llama-benchy) is an additional arm of
  the results. The TP2 results were measured on the NVIDIA checkpoint (lab plan `69ad74fc`, an earlier TP2 profile).
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (revision `7d74cdd8`),
  Inco AI's DFlash 2 on the DFlash method of [z-lab/dflash](https://github.com/z-lab/dflash) (Jian Chen, Yesheng
  Liang, Zhijian Liu); CC BY-NC-ND 4.0, released for research and evaluation per its model card; not redistributed
  here).
- Engine: vLLM v0.29.0 with the GLM-5.3 backport, b12x kernels and the sparse-MLA plugin, in one base image.

## Quick start

**Before you start**
- Hardware: four DGX Sparks (TP4) or two (TP2), cabled and addressed as [fabric/README.md](fabric/README.md) describes
  (TP4: a ring over both ConnectX-7 ports, all four RoCE functions in the four cable subnets; TP2: one cable on the
  `f1` port). On every TP4 node, `python3 fabric/preflight_tp4.py --rank N` must exit 0.
- On every node: Docker Engine with the NVIDIA Container Toolkit and the Compose plugin, the weights at the pinned
  revisions ([docs/operations.md](docs/operations.md), "Preflight sequence", step 3), and ssh access from the machine
  that runs the launcher.

**Steps** (TP4 shown; use `tp2` for two Sparks)

1. Clone this repository on every node, at the same path, and write a site file:

   ```sh
   git clone https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE.git
   cd GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE
   cp site.env.example site.env        # addresses, interfaces, paths, ssh targets
   ```

2. Render one Docker Compose project per rank: `python3 -B launch/compose.py --profile tp4 --site site.env`
   (writes `deploy/tp4/rank<N>/`).
3. Build the profile image once, on rank 0's node:
   `cd deploy/tp4/rank0 && docker compose --profile build build`. **Estimate, not measured:** 1.5 to 3 hours for the
   first build on one DGX Spark ([docker/README.md](docker/README.md)).
4. Share the image with the other nodes: `docker save glm53-switchless:tp4 | ssh PEER docker load`, over the fabric
   where the build node reaches its peers there ([docker/README.md](docker/README.md)).
5. Start: `launch/up.sh --profile tp4 --site site.env` prints every command (a dry run); add `--go` to run them. It
   starts the workers before rank 0 and waits for `/health` and `/v1/models`. With `--build` it also does steps 3
   and 4.
6. Check that it answers: `curl -s http://HEAD_NODE:8888/v1/models` lists `nvidia/GLM-5.3-Flash-NVFP4`, and the
   example completion of [docs/operations.md](docs/operations.md) returns a full answer (`HEAD_NODE` is rank 0's
   address).
7. Client example: an oh-my-pi (omp) provider configuration for this server is in
   [examples/omp/](examples/omp/README.md).
8. Stop: `launch/down.sh --profile tp4 --site site.env --go`.

The first boot takes much longer than later ones: every rank compiles kernels into its cache directory, and rank 0 runs
the startup warm-up (TP4: 17 prompts, TP2: six) before the API listens. The API has no authentication and no TLS: keep
it on a trusted network or behind an authenticating TLS proxy. A self-built image serves the profile's Python bytes with
rebuilt native code; it is not the measured image until it is qualified ([docker/README.md](docker/README.md)). What
else differs from the measured setup: [docs/limitations.md](docs/limitations.md).

## TP4: trying a 512K context limit

The default TP4 profile uses a 262,144-token context limit. A local long-context check on 2026-09-26 used
release lab plan `89dfef84` with only the context limit and KV-cache budget changed (derived plan `6b38d997`).
With `--max-model-len 1048576` and **24 GiB of KV cache per GPU**, it completed a 520,014-token cold prefill in
116.0 seconds, a 1,030,014-token cold prefill in 358.3 seconds, and retrieved all three needles from a
1,000,083-token prompt. These are single-run lab measurements, not a long-context quality evaluation or
qualification of a freshly rebuilt Docker image. Earlier larger-KV configurations failed the memory-margin check.

For a **512K local experiment**, use these vLLM arguments on **every rank**:

| Argument | Published default | 512K setting |
|---|---:|---:|
| `--max-model-len` | `262144` | `524288` |
| `--kv-cache-memory` | `34359738368` (32 GiB) | `25769803776` (24 GiB per GPU) |

The 512K setting is derived from the successful 1M-limit lab configuration; it has not been separately qualified
on the rebuilt image. The context limit includes **input plus generated tokens**. These argument changes need
no image rebuild, but require restarting all four ranks.

For a one-off test with an already built, working TP4 image:

1. Stop the current TP4 service with `launch/down.sh --profile tp4 --site site.env --go`.
2. Generate the rank projects with `python3 -B launch/compose.py --profile tp4 --site site.env`.
3. In each generated `deploy/tp4/rank<N>/compose.yaml`, change the values following the two flags above in the
   `glm53` service's `command` list. Keep the checked-in profile and its provenance manifests unchanged.
4. Copy each rank's edited `compose.yaml` and accompanying `.env` to that rank's existing project directory on
   its node. Start ranks **3, 2, 1, then 0**, running the following from each rank's project directory:

   ```sh
   docker compose up -d --no-build --pull never glm53
   ```

5. Wait for rank 0's `/health` and check `/v1/models` reports `max_model_len: 524288`. On **every node**, check
   `MemAvailable` in `/proc/meminfo` after startup; retain at least **16 GiB (16,777,216 kB)** before a long request.
   Stop the experiment if a rank fails or falls below that margin. Allow for the longer cold-prefill time in the
   client's timeout and test one long request before adding concurrent requests.

**Do not run `launch/up.sh` after these generated-file edits:** it regenerates the Compose files from the default
profile and overwrites the experiment's settings. To restore the default, stop all ranks with `launch/down.sh`
and launch normally with `launch/up.sh --profile tp4 --site site.env --go`. For a maintained profile, import a
reviewed derived plan through the workflow in [docs/updating.md](docs/updating.md).

## Status (2026-09-25)

| | TP4 (four Sparks) | TP2 (two Sparks) |
|---|---|---|
| Profile | `launch/profiles/tp4`: the RiNGSiDE release plan, lab plan `89dfef84`: lab plan `6f490797` (the composed lab winners, KDA checkpoints saved inside a prefill chunk, the fp16 KDA state with fp32 slow channels, greedy argmax verification, the replay boundary and first-token repay) with the release changes (change notices in the served files, the DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4 code) and 10 served changes: the fused RecoverSSM commit back on, with its ordering fix and the exact-landing column (`GLM53_RECOVERSSM_FUSED_COMMIT=1`); the KDA FP8 handoff through the tuned FP8 GEMM configurations (`GLM53_KDA_FP8_HANDOFF_TUNED=1`); the MLA q_a norm with the q_b_proj FP8 quantization fused in (`GLM53_MLA_QNORM_FP8=2`); the rowspread indexer top-k with pool-width decode logits (`GLM53_TOPK_ROWSPREAD=1`, `GLM53_INDEXER_POOL_LOGITS=1`); the exact top-p fallback and the UTF-8 continuation guard, mask only (`GLM53_TOPP_FIX=1`, `GLM53_UTF8_GUARD=1`); the kpool tail ring sized for speculative decoding (vllm-project/vllm#58454); the sampler kernels' tile argmax clamped to the vocabulary (vllm-project/vllm#50843); the kpool tail group left out of the generic slot mapping (an out-of-bounds read at long contexts); the RedHatAI/GLM-5.3-Flash-NVFP4 checkpoint (revision `18d55bfd`) with NVIDIA's `tokenizer.json` and top-p 0.95 as its generation default; the prefill indexer's query rows split across the ranks (`GLM53_INDEXER_ROW_SPLIT=1`); measured in the release window of this plan (Results) | `launch/profiles/tp2`: the RiNGSiDE release plan for TP2, lab plan `b80911ec`: lab plan `69ad74fc` (TP2 record plan, startup warm-up, the weak-spot fixes (prefix-cache retention 4,608, the prefix-cache contract, the MoE mixed-batch split), the replay boundary and first-token repay) with change notices in the served files and the DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4 code, and 5 correctness fixes: the exact-landing column of the RecoverSSM state write (the fused commit stays off); the kpool tail ring sized for speculative decoding (vllm-project/vllm#58454); the kpool tail group left out of the generic slot mapping (an out-of-bounds read at long contexts); the exact top-p fallback and the UTF-8 continuation guard, mask only (`GLM53_TOPP_FIX=1`, `GLM53_UTF8_GUARD=1`); the sampler kernels' tile argmax clamped to the vocabulary (vllm-project/vllm#50843); served with the RedHatAI/GLM-5.3-Flash-NVFP4 checkpoint (revision `18d55bfd`) with NVIDIA's `tokenizer.json` and top-p 0.95 as its generation default and a KV cache pool 1 GiB smaller per rank (`--kv-cache-memory 11274289152`, 10.5 GiB) as the RedHat checkpoint's memory margin; verified by GPU leaves and CPU checks, not yet served in a TP2 window; its TP2 window follows the release (TP2 numbers follow) |
| Determinism | **NON-DETERMINISTIC**: MoE launches of 10 or more rows (prefill and decode) accumulate with atomic adds | **NON-DETERMINISTIC** prefill: prefill MoE launches accumulate with atomic adds; decode launches stay deterministic |
| Results | the release window of this plan (2026-09-26, 00:36-01:50): RigMark with 2 runs per cell, prefill 8K to 256K, 1-16 streams, staggered arrivals 2-16; tool-eval-bench (2 rounds) and sparkDash (2 rounds) in the same boot; the Korean prompt 1 x 12 at width 6 (12 of 12 runs without a long-generation break); the emoji stress test and llama-benchy did not run on this plan (llama-benchy ran on the same release plan with the NVIDIA checkpoint: an additional arm) ([bench/results](bench/results/README.md)) | the final TP2 run of lab plan `69ad74fc`, before the release's changes and fixes and on the NVIDIA checkpoint (the TP2 profile's own numbers follow): RigMark with 2 runs per cell, prefill 8K to 64K, 1-6 streams, staggered arrivals 2-6, and sparkDash in a separate window of the same plan; one window without a fresh reference run, compared with the previous profile's stored windows as information only ([bench/results](bench/results/README.md)) |

<!-- chart:start -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/results/chart-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="bench/results/chart-light.svg">
  <img alt="RiNGSiDE measured results. TP4 (RedHatAI checkpoint, lab plan 89dfef84): single-stream decode 105.3 code, 61.7 prose, 145.4 structured tok/s; cold prefill 1.7 s at 8K to 53.6 s at 256K; short code aggregate 77.0 tok/s at 1 stream to 345.8 tok/s at 16 streams; sparkDash code, 1 stream 151.3 tok/s; tool-eval-bench 96 in all 2 rounds (133/138 points each). TP2 (NVIDIA checkpoint, lab plan 69ad74fc): single-stream decode 56.5 code, 33.0 prose, 83.4 structured tok/s; cold prefill 3.6 s at 8K to 25.7 s at 64K; short code aggregate 44.0 tok/s at 1 stream to 97.3 tok/s at 6 streams; sparkDash code, 1 stream 85.7 tok/s." src="bench/results/chart-light.svg" width="880">
</picture>

RiNGSiDE's own measured results only, rendered by `bench/results/chart.py` from `bench/results/results.json`
and the records in `bench/results/evidence/`; the full tables, the caveats and the comparison recipes are in
[bench/results](bench/results/README.md).
<!-- chart:end -->

**What you can reproduce from this repository alone**
- Every served source file byte for byte, given the base image: 54 single-file patches (30 for TP4, 24 for TP2;
  checked against the image file's sha256 before and the served file's sha256 after), 55 files carried in full under
  `src/` (38 for TP4, 17 for TP2), the runtime overrides and the chat template. The one exception is the third-party
  Humming package that TP4 mounts, which comes from vllm-project/humming at a pinned commit
  (`build/humming/README.md`).
- The base image and the native artefacts, from pinned public sources plus the lab's stage inputs in `docker/base`:
  `docker/Dockerfile` rebuilds the base image the lab measured on, with the recipe's DFlash2 code in `model.py` and
  `kv_cache_utils.py` ([docs/image.md](docs/image.md)), and checks every patch preimage, builds the native
  artefacts from `build/`, and bakes each profile's overlay into one image per profile
  ([docs/build-provenance.md](docs/build-provenance.md)). Python files come out byte for byte as the lab served them;
  native code is compiled again.
- The exact `docker run` command of every rank from a site file (`launch/render.py`), verified token for token against
  the lab plans (mount order aside, which does not matter here). `launch/profiles/*/canonical-launch.json` shows the
  rendered commands for a documentation site; `sources/verify.py` fails if a flag, mount or served file changes
  without a new plan. `launch/compose.py` renders the same command, without the overlay mounts, as one Docker Compose
  project per rank for the self-built image.
- The NCCL source tree of the TP4 library and the source trees of the sparse-MLA and FP8 GEMM extensions (the NCCL
  and sparse-MLA trees carry change-notice comments that the measured builds did not have; `build/`).

**What still needs something outside this repository**
- **Network access for the build.** `docker/Dockerfile` fetches the pinned public sources, wheels and base images
  ([docs/build-provenance.md](docs/build-provenance.md) lists them, and what is resolved at build time).
- **The measured binaries.** The measured base image `sha256:017fd0ba...` and the measured native artefacts are not
  published; RiNGSiDE distributes source and patches only ([docs/redistribution.md](docs/redistribution.md)). A
  self-built image has the profile's Python bytes and rebuilt native code, so it needs its own qualification
  ([docker/README.md](docker/README.md)).
- **Hardware.** DGX Sparks cabled as described in [fabric/README.md](fabric/README.md); the TP4 NCCL build and the
  lean all-reduce assume that cabling and addressing.
- **Weights** from Hugging Face (see above).

## The measured launch form

The measured runs used the base image with each profile's overlay mounted from every node, as `launch/render.py`
renders it. Python 3.11+, Git. Nothing here contacts a node; you run the rendered commands yourself.

```sh
# 1. Checks (CPU only)
python3 -B sources/verify.py
python3 -B tests/run_cpu.py

# 2. Build the overlay for a profile from the base image's files and the native artefacts
python3 -B sources/apply.py --profile tp4 --image-root IMAGE_ROOT --artefacts ARTEFACTS \
    --third-party HUMMING_SITE_PACKAGES --rank 0 --out overlay-tp4

# 3. Render the per-rank docker commands from your site file
cp site.env.example site.env    # fill in addresses, interfaces and paths
python3 -B launch/render.py --profile tp4 --site site.env
```

Start the worker ranks before rank 0. The first boot compiles kernels into the per-rank cache directories, and every
boot runs the startup warm-up prompts before the API listens (TP4: 17 prompts, TP2: six).
[docs/operations.md](docs/operations.md) has the prerequisites, the preflight sequence (including
`fabric/preflight_tp4.py`), an example request and troubleshooting.

## Optional: weightless GLP-44 steering (TP4, off by default)

[msuiche/weightless](https://github.com/msuiche/weightless) steers the model with a projective activation vector:
GLP-44, alpha 2.0, applied to the post-layer stream of every layer, which suppresses refusals. The lab served it on
the TP4 profile in one verification window (2026-09-26, 02:05-02:18 UTC+7): every rank logged the steering line and a
request completed. RigMark code-full, one round: cold prefill 6 to 7 % slower than the TP4 row at every depth (1.81 s
at 8K to 57.02 s at 256K); single-stream decode 104.7 code, 57.2 prose and 139.1 structured tok/s (the TP4 row: 105.3,
61.7 and 145.4, medians of 2 rounds), which one round does not separate from run-to-run variation. tool-eval-bench 92
(127/138 points, one round) against 96 in both rounds of the release window: in this bench the option lowers the
model's resistance to prompt injection; its extra misses are a sleeper prompt injection that it followed (TC-60), a
prompt injection whose attacker content it reproduced (TC-58), a multi-turn cancellation (TC-49), a multi-turn
information reveal (TC-50) and a search-read-calculate pipeline step (TC-55)
(`bench/results/evidence/weightless-tp4.json`). The TP2 profile does not offer it.

weightless states no licence, so this repository contains none of its code. `launch/options/weightless/prepare.py`
fetches the weightless GLM-5.3 patcher at a pinned commit and the GLP-44 vector from Hugging Face at a pinned revision
(MIT; the repository is gated: request access there and set `HF_TOKEN` or run `huggingface-cli login`), checks both by
sha256, rebuilds the TP4 profile's `model.py` from this repository, applies the patcher's replacements and the
recipe's own fail-closed guards, and refuses any result but the pinned steered file (sha256 `8cbfea91...`), which is
the file the lab's verification window served (`78693488...`) with the author note of one comment left out:

```sh
python3 -B launch/options/weightless/prepare.py --out /srv/switchless/weightless
# copy the directory to the same path on every node, then add to the TP4 site file:
SWITCHLESS_WEIGHTLESS_DIR=/srv/switchless/weightless
```

`launch/render.py` and `launch/compose.py` then mount the steered `model.py` and the vector and set the four
`WEIGHTLESS_*` variables; without the site line both produce the default profile unchanged.
[docs/operations.md](docs/operations.md) has the steps and the log line to check.

The edits `launch/options/weightless/prepare.py` applies to `model.py` are, like every other patch to that vLLM file,
under the file's Apache-2.0 licence; msuiche/weightless code is fetched by the operator from its repository and is not
part of this repository or its licence.

## Layout

| Path | Contents |
|---|---|
| `launch/profiles/{tp4,tp2}` | argv template, mount table, runtime override, chat template |
| `launch/` | `render.py` (docker run per rank), `compose.py` (a Compose project per rank), `up.sh` and `down.sh`; `options/weightless/prepare.py` (the optional steering) |
| `docker/` | `Dockerfile` (base image, native artefacts, profile images), `base/` (the lab's image stages), `tools/` |
| `patches/{vllm,b12x,sparse-mla}/{tp4,tp2}` | single-file patches against the base image's files |
| `patches/nccl`, `patches/sparse-mla/build` | Git patch series for the NCCL library and the sparse-MLA extensions |
| `src/{tp4,tp2}` | files served in full: the recipe's own modules and new files (install paths below each directory) |
| `sources/` | `installed-files.json` (every served byte), `apply.py`, `verify.py`, `import_plan.py` |
| `build/` | build recipes for the native artefacts |
| `bench/results` | results page and data |
| `examples/` | client configuration examples |
| `docs/` | [configuration](docs/configuration.md), [image](docs/image.md), [build provenance](docs/build-provenance.md), [architecture](docs/architecture.md), [operations](docs/operations.md), [updating a profile](docs/updating.md), [limitations](docs/limitations.md), [redistribution](docs/redistribution.md), [credits](docs/credits.md) |

## Licence

Original code and documentation: Apache License 2.0 ([LICENSE](LICENSE)). Patched and derived third-party code keeps
its own licence and notices ([NOTICE](NOTICE), [licenses/](licenses/README.md)); the served chat template is MIT. The
recipe builds on vLLM and Humming (Jinzhen Lin; both vllm-project), b12x, the sparse-MLA plugin, NCCL, SparkRing,
switchless-nccl, and zai-org's chat template with the thinking gate by Raymond Lucke, as carried in
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks; [docs/credits.md](docs/credits.md) credits them.
