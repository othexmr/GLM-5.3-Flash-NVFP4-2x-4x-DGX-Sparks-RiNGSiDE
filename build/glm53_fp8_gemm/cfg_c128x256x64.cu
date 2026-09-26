// dense-fp8-kernels agent 2026-09-24: config c128x256x64 (see glm53_fp8_gemm.cuh).
#include "glm53_fp8_gemm.cuh"

GLM53_FP8_CONFIG(c128x256x64, 128, 256, 64, cutlass::gemm::KernelTmaWarpSpecializedCooperative, 0)
