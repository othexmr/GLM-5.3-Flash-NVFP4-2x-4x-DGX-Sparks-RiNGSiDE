// dense-fp8-kernels agent 2026-09-24: config p128x128x64 (see glm53_fp8_gemm.cuh).
#include "glm53_fp8_gemm.cuh"

GLM53_FP8_CONFIG(p128x128x64, 128, 128, 64, cutlass::gemm::KernelTmaWarpSpecializedPingpong, 0)
