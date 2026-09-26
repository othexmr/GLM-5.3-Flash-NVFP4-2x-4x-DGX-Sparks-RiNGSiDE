# FP8 GEMM extension (TP4)

SM 12.x FP8 W8A8 GEMM configurations with vLLM's epilogue, loaded by the dense FP8 path
(`GLM53_DENSE_FP8_GEMM`, `GLM53_FP8_GEMM_SO`). The kernel family is vLLM v0.29.0's SM120 `scaled_mm`
(`glm53_fp8_gemm.cuh`); `vendor/` holds three verbatim vLLM headers (`vendor/README.txt`).
Licence: Apache-2.0 (derived from vLLM; `vendor/LICENSE-vllm`), except the two `broadcast_load_epilogue` headers in
`vendor/`, which are NVIDIA CUTLASS code under BSD-3-Clause (their header; `licenses/BSD-3-Clause-NVIDIA-CUTLASS.txt`).
A binary built from this directory carries both licences.

Build inside the base image with PyTorch's extension loader (`build_probe.py` builds and then compares every
configuration against vLLM's `cutlass_scaled_mm` bit for bit):

- sources: `bind.cpp` and the six `cfg_*.cu`;
- includes: the image's CUTLASS v4.5.0 (`flashinfer/data/cutlass`: `include`, `tools/util/include`), `vendor/`, this
  directory;
- flags: `-O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG
  -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1`, `TORCH_CUDA_ARCH_LIST=12.1a`, link `-lcuda`.

The measured extension: `glm53_fp8_gemm.so`, 1,002,696 bytes, sha256
`db120c32a467aa53be745ede8b20b08ff2a473db2d74382416da7c97aab31de0`, mounted at `/opt/glm53_fp8_gemm/glm53_fp8_gemm.so`.
