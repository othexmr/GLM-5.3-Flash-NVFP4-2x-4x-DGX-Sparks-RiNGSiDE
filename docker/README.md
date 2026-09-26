# Build the images

`Dockerfile` is the single build definition of RiNGSiDE. It rebuilds the lab's base image from pinned sources and
the lab's stage inputs (`base/`), builds the native artefacts from their recipes in `../build/`, and bakes each
profile's overlay into one ready-to-run image per profile. RiNGSiDE publishes no images and no binaries: you build
them. `../docs/build-provenance.md` records every input; `tools/` holds the build's helper scripts.

| Target | Result |
|---|---|
| `profile-tp4` | the TP4 image: base + switchless NCCL, sparse-MLA A1c, FP8 GEMM, prefill RDMA library, rowspread top-k extension, Humming 0.1.15 and the TP4 overlay at the served paths |
| `profile-tp2` | the TP2 image: base + sparse-MLA M7e and the TP2 overlay |
| `base` | the base image alone (the rebuilt counterpart of `sha256:017fd0ba...`) |

**Not built yet; its first build check follows the release.** The Dockerfile was written from the lab's build records
and checked on a CPU; its first build on a DGX Spark is also the test of the file. A self-built image is a new image:
its native code is compiled again and its Ubuntu and CUDA packages are resolved at build time, so it needs the
qualification below before its numbers can stand next to the measured ones.

## Requirements

- One DGX Spark (linux/arm64) to build on. The build needs no GPU.
- Docker Engine with BuildKit (its built-in Dockerfile frontend: `RUN --mount`, heredocs, `ADD --checksum`) and the
  Compose plugin. The one-command build uses a `service:` reference in `additional_contexts`; if your Compose release
  rejects it, use the two commands under "Without Compose".
- Network access during the build: GitHub (sources and the vLLM release wheel), Docker Hub, PyPI,
  download.pytorch.org, flashinfer.ai, NVIDIA's CUDA package repository, the Ubuntu and Rocky Linux archives, the
  deadsnakes PPA, rustup and crates.io.
- Disk: plan for about 150 GB free (estimate): the base image alone is 23.9 GB (measured), plus the build cache.
- Building downloads third-party software under its own terms: `../docs/redistribution.md`.

## Build with Docker Compose

Render the compose projects from your site file, then build in rank 0's project:

```sh
cp site.env.example site.env        # fill in addresses, interfaces and paths
python3 -B launch/compose.py --profile tp4 --site site.env      # writes deploy/tp4/rank<N>/{compose.yaml,.env}
cd deploy/tp4/rank0
docker compose --profile build build  # the vLLM source image, then docker/Dockerfile's profile-tp4
```

Compose builds the service `vllm-source` first (ZJY0516/vllm's own `docker/Dockerfile` at `4500c80c`, with the
lab's build arguments) and then the service `glm53`, whose Dockerfile starts from it. The result is tagged
`glm53-switchless:tp4` (`SWITCHLESS_BUILD_IMAGE` changes it). `launch/up.sh --build` runs the same build on rank 0's
node and shares the image.

**Build time (estimate, not measured):** 1.5 to 3 hours for the first build of one profile on one DGX Spark. The
measured parts: the lab's vLLM source build took 54 minutes (2026-09-03), its fresh build of stages 00 to 16
56 minutes (2026-09-08, the source stage about 33 minutes of it), its Rust stage of the port 395 s (2026-09-10). The
native builds and the export of the 24 GB image come on top. A second profile reuses the base and takes minutes.

## Without Compose

```sh
docker buildx build --load --target vllm-openai -t glm53-switchless/vllm-glm-release:4500c80c \
  --build-arg CUDA_VERSION=13.0.3 --build-arg PYTHON_VERSION=3.12 --build-arg torch_cuda_arch_list=12.1a \
  --build-arg FLASHINFER_VERSION=0.6.18 --build-arg max_jobs=16 --build-arg nvcc_threads=2 \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359@sha256:f91599c49f526c77d01b68286f2bf943a5fd6a432d7e3f0afcc5784825908fe9 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1 \
  --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
  -f docker/Dockerfile 'https://github.com/ZJY0516/vllm.git#4500c80c080328dfe62435d083f4063e00d987df'

docker buildx build --load --target profile-tp4 -t glm53-switchless:tp4 -f docker/Dockerfile \
  --build-context vllm-source=docker-image://glm53-switchless/vllm-glm-release:4500c80c .
```

Run the second command from the repository root. The first builds from the upstream repository, so its
`-f docker/Dockerfile` is upstream's file.

Build arguments of `docker/Dockerfile`:

| Argument | Default | Meaning |
|---|---|---|
| `SOURCE_PIN` | `exact` | `exact` re-pins the source image's Python distributions to the lab's build (`tools/pin_source.py`); `check` fails on any difference; `report` records differences and continues (the result is then a different image) |
| `NCCL_JOBS` | `4` | parallel jobs of the NCCL build (the recorded build used 4) |

## Share the image with the other nodes

Every rank runs the same image. Build once and copy it:

```sh
docker save glm53-switchless:tp4 | ssh PEER docker load        # on the build node, once per other node
docker image inspect --format '{{.Id}}' glm53-switchless:tp4   # on every node: the same ID
```

Use each peer's fabric address for `PEER` where the build node reaches it over the fabric (the fastest path), or a
registry you run on your network (tag, push, and pull by digest on the other nodes). `launch/up.sh --build` copies
the image with `docker save | docker load`: directly from the build node when `SWITCHLESS_SHARE_TARGETS` names the
peers as the build node reaches them, otherwise through the machine that runs `up.sh`. It then compares the image
IDs of all nodes.

## Compare with the measured bytes

1. Repository: `python3 -B sources/verify.py` checks that every served byte is pinned as the lab plan serves it.
2. Image: the build ran this already and fails on a source file that differs; to see the result again:

   ```sh
   docker run --rm --entrypoint python3 glm53-switchless:tp4 -B \
     /opt/glm53-switchless/recipe/docker/tools/check_served.py --profile tp4 --rank 0 \
     --manifest /opt/glm53-switchless/recipe/sources/installed-files.json
   ```

   Every patched, repository and template file is `measured` (the sha256 the lab served); the native artefacts and
   the Humming files are `rebuilt` (their bytes differ from the lab's builds, as expected).
3. Receipts in the image: `/opt/glm53-switchless/source-distributions.json` (Python distributions against the lab's
   source build), `/opt/glm53-switchless/base-check.json` (the 42 patch preimages and the package versions of
   `docs/image.md`), `/opt/glm53-switchless/<profile>/apply-receipt.json` (every served file: patched, copied or
   rebuilt artefact) and `baked.json`.
4. Against the measured base image, where you have it: list the sha256 of every file below
   `/usr/local/lib/python3.12/dist-packages` in both images and compare. Expect differences in native libraries,
   bytecode caches, version stamps and build receipts, in `vllm/models/glm5next/nvidia/model.py` and
   `vllm/v1/core/kv_cache_utils.py` (the recipe's DFlash2 capture and KV-cache group), and in the SPDX line of
   `vllm/parser/glm53_moe.py`, `vllm/reasoning/glm53_moe_reasoning_parser.py` and
   `vllm/tool_parsers/glm53_moe_tool_parser.py` (`docker/base/README.md`); any other Python source that differs is a
   finding.

## Qualify a self-built image

The measured results belong to the lab's image and artefacts. A self-built image serves the same Python bytes with
rebuilt native code, so qualify it before quoting it:

1. Boot every rank (`launch/up.sh`) and check the log markers of `../docs/operations.md` ("Launching"), including the
   startup warm-up and, for TP4, the non-deterministic MoE path.
2. Check function: the example request of `../docs/operations.md`, a tool call, a long-context request, and a request
   with thinking off.
3. Measure with the protocol of `../bench/protocols/README.md` on the same hardware and compare with
   `../bench/results/README.md`; a matched run of the measured setup is the fair reference where you have it.
4. Keep the image ID, the `check_served.py` output, the receipts above and the results together. Until then, the
   self-built image is unqualified.

## Stages

`Dockerfile` runs, in order: `source` (the vLLM source image with its Python distributions pinned), the lab's
stages `s01-compat` to `s19-runtime-bugfixes`, the vLLM v0.29.0 port (`v029-*`), `s21-atomic-prefill`,
`thinking-gate`, `mixed-prefill-repair` and `base`; then the native artefacts (`nccl`, `sparse-mla-tp4`,
`sparse-mla-tp2`, `fp8-gemm`, `prefill-rdma`, `topk-rowspread`, `humming`), the overlays (`overlay-tp4`, `overlay-tp2`: `sources/apply.py`
against the rebuilt base, with `--allow-rebuilt`) and the profile images. Every stage checks what it starts from: the
sums of `base/`, Git commits and trees, the wheel sha256, each lab stage's preimage and result hashes, the ported tree
(`tools/verify_port.py`), the patch preimages (`tools/check_base.py`) and the served files (`tools/check_served.py`).
`base/README.md` says what each lab stage changes.
