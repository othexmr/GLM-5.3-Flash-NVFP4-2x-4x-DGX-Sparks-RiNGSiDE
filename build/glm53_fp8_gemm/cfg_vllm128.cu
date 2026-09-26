// dense-fp8-kernels agent 2026-09-24: config vllm128 (see glm53_fp8_gemm.cuh).
#include "glm53_fp8_gemm.cuh"

GLM53_FP8_CONFIG(vllm128, 128, 128, 128, cutlass::gemm::collective::KernelScheduleAuto, 0)
