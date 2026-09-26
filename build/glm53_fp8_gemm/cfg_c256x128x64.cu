// dense-fp8-kernels agent 2026-09-24: config c256x128x64 (see glm53_fp8_gemm.cuh).
#include "glm53_fp8_gemm.cuh"

GLM53_FP8_CONFIG(c256x128x64, 256, 128, 64, cutlass::gemm::KernelTmaWarpSpecializedCooperative, 0)
