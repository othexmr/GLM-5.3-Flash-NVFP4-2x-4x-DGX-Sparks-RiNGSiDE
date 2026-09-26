# Native artefacts

The profiles mount six native artefacts and one third-party Python package. None of them is in Git. Each is
identified by the sha256 of the bytes that were measured (`release/assets.json`), with the source and the command that
built it. Rebuilding from the same source is expected to give different bytes (compilers embed paths and build
details; the lab's own two builds of the prefill RDMA library differ), so a rebuild is a new artefact: give it its own
identity and measure it before serving it. `sources/apply.py --allow-rebuilt` accepts such artefacts and reports them.

| Artefact | Profile | Measured sha256 | Source | Recipe |
|---|---|---|---|---|
| NCCL 2.30.7 switchless dual-PF | TP4 | `5d5c6550...` | NVIDIA/nccl `73cf1122` + `patches/nccl` | [nccl/](nccl/README.md) |
| sparse-MLA extension (A1c) | TP4 | `1f940e75...` | Libertai plugin `b1f20638` + `patches/sparse-mla/build/tp4-so-a1c.patch` | [sparse-mla/](sparse-mla/README.md) |
| sparse-MLA extension (M7e) | TP2 | `221fdbaa...` | Libertai plugin `b1f20638` + `patches/sparse-mla/build/tp2-so-m7e.patch` | [sparse-mla/](sparse-mla/README.md) |
| FP8 GEMM extension | TP4 | `db120c32...` | `build/glm53_fp8_gemm/` | [glm53_fp8_gemm/](glm53_fp8_gemm/README.md) |
| prefill RDMA library | TP4 | per rank, `9f410623...` / `9b54e41b...` | `src/tp4/glm53_prefill_rdma/_prb.cu` | [prefill_rdma/](prefill_rdma/README.md) |
| rowspread top-k extension and its build receipt | TP4 | `3fefebaa...` (receipt `b500393f...`) | `build/topk_rowspread/` with the stage-15 top-k sources of `docker/base` | [topk_rowspread/](topk_rowspread/README.md) |
| humming-kernels 0.1.15 | TP4 | per file (`sources/installed-files.json`) | vllm-project/humming `a74973b5` | [humming/](humming/README.md) |

All builds ran inside the base image (CUDA 13.0, SM 12.1) on a DGX Spark. `toolchain.lock.json` records the toolchain.
`docker/Dockerfile` builds every one of them from these recipes inside the rebuilt base image (stages `nccl`,
`sparse-mla-tp4`, `sparse-mla-tp2`, `fp8-gemm`, `prefill-rdma`, `topk-rowspread`, `humming`); those builds are rebuilt artefacts in the
sense above.
