// dense-fp8-kernels agent 2026-09-24: config p64x128x128 (see glm53_fp8_gemm.cuh).
#include "glm53_fp8_gemm.cuh"

GLM53_FP8_CONFIG(p64x128x128, 64, 128, 128, cutlass::gemm::KernelTmaWarpSpecializedPingpong, 0)
