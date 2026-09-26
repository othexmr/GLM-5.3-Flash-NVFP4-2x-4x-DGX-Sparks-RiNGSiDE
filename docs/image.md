# Base image

Both profiles run on one local image:

| | |
|---|---|
| Image ID | `sha256:017fd0ba0a265dd81b78b352b14d33df03e9996f8f45870d4c3df84e08b1ebcb` |
| Tag | `local/glm53-v029:ported-20260910` |
| Platform | linux/arm64 (DGX Spark, GB10, SM 12.1), CUDA 13.0.3, Python 3.12 |
| Packages | vllm 0.29.0+glm53.local, torch 2.13.0+cu130, transformers 5.16.1, flashinfer-python 0.6.18, triton 3.7.1, nvidia-cutlass-dsl 4.6.2, b12x 1.3.0 |
| Public reference | none: RiNGSiDE publishes no images; `docker/Dockerfile` rebuilds it (below) |

## What is in it

- **vLLM** v0.29.0 (vllm-project/vllm `98dff2a81d747d1dba01a47f939f48c3526d4206`) with the GLM-5.3 model sources of the
  `glm-release` branch (`4500c80c`; the lab's builds fetched that commit from ZJY0516/vllm) backported onto it: the
  official v0.29.0 ARM64 wheel's native extensions with the ported Python tree and rebuilt Rust artifacts.
- **b12x** 1.3.0 (local-inference-lab/b12x; its files equal commit `3a437ab5` for the patched paths) with the lab's
  earlier MoE changes (deterministic output, atomic prefill, decode fast path).
- **glm53_sparse_mla**, the sparse-MLA plugin of Libertai/vllm-sparse-mla-blackwell (`b1f20638`), with the lab's
  atomic-prefill and NoPE backend changes.
- The lab's earlier layers: the thinking-gated GLM-5.3 parser, the deterministic sparse top-k module, a first version
  of the `glm53_speedup` scheduler package and a mixed-prefill repair of `vllm/outputs.py`,
  `vllm/v1/core/sched/scheduler.py` and `vllm/v1/worker/gpu/cudagraph_utils.py`.

The image was built in four stages (vLLM port, atomic-prefill stage, thinking gate, mixed-prefill repair) on top of
the lab's earlier image stages over the vLLM source image of `4500c80c`. `docker/Dockerfile` rebuilds that chain from
pinned sources and the lab's stage inputs (`docker/base`) and checks every patch preimage;
[build-provenance.md](build-provenance.md) records every input and the lab's intermediate images. A rebuilt image has
the same Python files as far as the records reach, except `vllm/models/glm5next/nvidia/model.py` and
`vllm/v1/core/kv_cache_utils.py`, which carry the recipe's DFlash2 capture and KV-cache group in place of the lab's
earlier versions, and the lab's three GLM-5.3 parser files, whose SPDX line names the repository licence
(`docker/base/README.md`); it also has rebuilt native code and a different image ID, so it is not the measured image.
For every file the profiles replace, the repository records the sha256 of the image's file (the patch preimage) and
how that file relates to upstream:

| Relation of the image file to upstream | Files |
|---|---|
| identical to vllm-project/vllm `98dff2a8` | 12 |
| GLM-5.3 backport file (absent from or different in `98dff2a8`) | 14 |
| earlier lab change baked into the image | 5 (vLLM `fused_moe/b12x.py` and `v1/core/sched/scheduler.py`, b12x `dynamic.py` and `_impl.py`, sparse-MLA `backend.py`) |

`sources.lock.json` holds the relation of each patch (`preimage_upstream`) and the evidence for each preimage hash
(`sources/installed-files.json`, `preimage_evidence`).

## Getting the files the patches need

`docker/Dockerfile` runs `sources/apply.py` inside the rebuilt image. The patches of `model.py` (both profiles) and
`kv_cache_utils.py` (TP4) are recorded against the rebuilt base's files; the measured image has other files there
(`7cddb9d1...`, `7ef72953...`), so `apply.py` stops at those three on the measured image. For the other files an
overlay can be built from the measured image, whose files `sources/apply.py` reads from a directory tree, for example:

```sh
cid=$(docker create sha256:017fd0ba0a265dd81b78b352b14d33df03e9996f8f45870d4c3df84e08b1ebcb)
mkdir -p IMAGE_ROOT/usr/local/lib/python3.12
docker cp "$cid":/usr/local/lib/python3.12/dist-packages IMAGE_ROOT/usr/local/lib/python3.12/
docker rm "$cid"
```

`apply.py` checks the sha256 of the image files that the patches replace (the preimages: `preimage_sha256` in
`sources/installed-files.json`), not the whole image. An image that differs in one of those files fails that
check instead of producing different served files; differences in any other file of the image, including the
packages the patched files import, are not detected by it. Only the image ID (`docker image inspect`) identifies the
whole image.
