// dense-fp8-kernels agent 2026-09-24: Python binding of the SM120 FP8 W8A8 GEMM configs (glm53_fp8_gemm.cuh).
// gemm(out, a, b, a_scales, b_scales, config, swizzle, raster): out [M, N] bf16 row-major (a row slice of a
// contiguous [*, N] tensor is fine), a [M, K] e4m3 row-major, b [K, N] e4m3 column-major (weight.t() of a contiguous
// [N, K] weight), a_scales fp32 [M] / [M, 1] or one element, b_scales fp32 [N] / [N, 1] or one element; the same
// operand contract as vLLM's cutlass_scaled_mm without bias.
#include <torch/extension.h>

#include <string>
#include <vector>

namespace glm53 {
void run_vllm128(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&,
            torch::Tensor const&, int64_t, int64_t);
std::string describe_vllm128();
void run_c128x128x64(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&,
            torch::Tensor const&, int64_t, int64_t);
std::string describe_c128x128x64();
void run_c128x256x64(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&,
            torch::Tensor const&, int64_t, int64_t);
std::string describe_c128x256x64();
void run_c256x128x64(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&,
            torch::Tensor const&, int64_t, int64_t);
std::string describe_c256x128x64();
void run_p128x128x64(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&,
            torch::Tensor const&, int64_t, int64_t);
std::string describe_p128x128x64();
void run_p64x128x128(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&,
            torch::Tensor const&, int64_t, int64_t);
std::string describe_p64x128x128();
}  // namespace glm53

static const std::vector<std::string> kNames = {"vllm128", "c128x128x64", "c128x256x64", "c256x128x64", "p128x128x64", "p64x128x128"};

void gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b, torch::Tensor a_scales, torch::Tensor b_scales,
          int64_t config, int64_t swizzle, int64_t raster) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda() && a_scales.is_cuda() && b_scales.is_cuda(), "cuda tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device() && a.device() == a_scales.device() &&
              a.device() == b_scales.device(), "one device");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn && b.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "e4m3 operands");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "bf16 output");
  TORCH_CHECK(a_scales.scalar_type() == at::ScalarType::Float && b_scales.scalar_type() == at::ScalarType::Float,
              "fp32 scales");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2, "2-D operands");
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(b.size(0) == K && out.size(0) == M && out.size(1) == N, "shapes");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == K, "a row-major contiguous");
  TORCH_CHECK(b.stride(0) == 1 && b.stride(1) == K, "b column-major contiguous ([N, K] weight transposed)");
  TORCH_CHECK(out.stride(1) == 1 && out.stride(0) == N, "out row-major with row stride N");
  TORCH_CHECK(K % 16 == 0 && N % 8 == 0 && M > 0, "alignment: K % 16 == 0, N % 8 == 0");
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous(), "contiguous scales");
  TORCH_CHECK(a_scales.numel() == M || a_scales.numel() == 1, "a_scales per token or per tensor");
  TORCH_CHECK(b_scales.numel() == N || b_scales.numel() == 1, "b_scales per channel or per tensor");
  TORCH_CHECK(swizzle >= 0 && swizzle <= 8 && raster >= 0 && raster <= 2, "swizzle 0..8, raster 0 (heuristic) / 1 (M) / 2 (N)");
  switch (config) {
    case 0: glm53::run_vllm128(out, a, b, a_scales, b_scales, swizzle, raster); break;
    case 1: glm53::run_c128x128x64(out, a, b, a_scales, b_scales, swizzle, raster); break;
    case 2: glm53::run_c128x256x64(out, a, b, a_scales, b_scales, swizzle, raster); break;
    case 3: glm53::run_c256x128x64(out, a, b, a_scales, b_scales, swizzle, raster); break;
    case 4: glm53::run_p128x128x64(out, a, b, a_scales, b_scales, swizzle, raster); break;
    case 5: glm53::run_p64x128x128(out, a, b, a_scales, b_scales, swizzle, raster); break;
    default: TORCH_CHECK(false, "unknown config ", config);
  }
}

std::vector<std::string> configs() { return kNames; }
std::vector<std::string> describe() { return {glm53::describe_vllm128(), glm53::describe_c128x128x64(), glm53::describe_c128x256x64(), glm53::describe_c256x128x64(), glm53::describe_p128x128x64(), glm53::describe_p64x128x128()}; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm, "SM120 FP8 W8A8 GEMM with vLLM's ScaledEpilogue (no bias)");
  m.def("configs", &configs);
  m.def("describe", &describe);
}
