// SM120/SM121 FP8 W8A8 GEMM with the served scale layout (dense-fp8-kernels agent, 2026-09-24).
//
// out[M, N] (BF16) = bf16_rn(a_scale[m] * (b_scale[n] * sum_k a[m, k] * b[k, n])), a [M, K] e4m3 row-major,
// b [K, N] e4m3 column-major (the [N, K] row-major weight, transposed view), FP32 per-token and per-channel scales.
//
// This is vLLM v0.29.0's SM120 FP8 kernel family (csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm.cuh,
// cutlass_3x_gemm_sm120, and cutlass_gemm_caller.cuh) with its epilogue tree taken verbatim (vendor/: ScaledEpilogue,
// Sm90ColOrScalarBroadcast x (Sm90RowOrScalarBroadcast x acc), round-to-nearest multiplies), the same mainloop
// builder (Sm120 TMA warp-specialized, SM120_16x8x32_TN e4m3 MMA atom, 1x1x1 cluster) and the same kernel wrapper
// (enable_sm120_family, GemmUniversal with the default tile scheduler). What a config may change: the CTA tile, the
// cooperative/pingpong schedule, the pipeline stage count, and at run time the scheduler's swizzle and raster order.
// None of these changes the per-element arithmetic: every config issues the same MMA atom over k = 0..K-1 in
// ascending order into one FP32 accumulator (no split-K, no stream-K) and applies the same epilogue. The leaf checks
// bit identity against the served kernel.
#pragma once

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace glm53 {

using namespace cute;

// Verbatim from vLLM v0.29.0 csrc/libtorch_stable/cutlass_extensions/common.hpp.
// SM12x family includes SM120 (RTX 5090) and SM121 (DGX Spark GB10)
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120f.\n");
    asm("trap;");
  #endif
#endif
  }
};

// vLLM's cutlass_3x_gemm_sm120 for InType e4m3, OutType bf16, Epilogue = ScaledEpilogue. Stages == 0 is vLLM's
// StageCountAutoCarveout (the served config); Stages > 0 fixes the mainloop stage count.
template <typename TileShape_, typename KernelSchedule_, int Stages>
struct Sm120Fp8Gemm {
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;
  using ElementAB = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::bfloat16_t;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ElementAcc = float;
  using Epilogue = vllm::c3x::ScaledEpilogue<ElementAcc, ElementD, TileShape>;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using EVTCompute = typename Epilogue::EVTCompute;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  using StageCountType = std::conditional_t<
      (Stages > 0), cutlass::gemm::collective::StageCount<(Stages > 0 ? Stages : 1)>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentA, ElementAB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape, StageCountType,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;
};

// Compile-time description of a config (reported by the module, checked by the probe/leaf).
template <typename G>
std::string describe(const char* name) {
  using TS = typename G::TileShape;
  constexpr int stages = G::CollectiveMainloop::DispatchPolicy::Stages;
  constexpr bool coop = cute::is_base_of_v<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                           typename G::CollectiveMainloop::DispatchPolicy::Schedule>;
  return std::string(name) + " tile " + std::to_string(int(size<0>(TS{}))) + "x" +
         std::to_string(int(size<1>(TS{}))) + "x" + std::to_string(int(size<2>(TS{}))) + " " +
         (coop ? "cooperative" : "pingpong") + " stages " + std::to_string(stages) + " smem " +
         std::to_string(int(sizeof(typename G::GemmKernel::SharedStorage))) + " threads " +
         std::to_string(int(G::GemmKernel::MaxThreadsPerBlock));
}

// vLLM's cutlass_gemm_caller (cutlass_gemm_caller.cuh) with the regular torch API; the tile scheduler arguments
// (swizzle, raster) are the only additions. swizzle <= 1 and raster 0 are vLLM's defaults ({}: no swizzle,
// heuristic raster).
template <typename G>
void run(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& a_scales,
         torch::Tensor const& b_scales, int64_t swizzle, int64_t raster) {
  using GemmKernel = typename G::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = StrideC;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  StrideD d_stride = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<cutlass::float_e4m3_t*>(a.data_ptr());
  auto b_ptr = static_cast<cutlass::float_e4m3_t*>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr, b_stride};

  auto c_ptr = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      G::Epilogue::prepare_args(a_scales, b_scales), c_ptr, c_stride, c_ptr, d_stride};

  const c10::cuda::CUDAGuard device_guard(a.device());
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::TileSchedulerArguments scheduler{};
  using Raster = typename GemmKernel::TileScheduler::RasterOrderOptions;
  scheduler.max_swizzle_size = static_cast<int>(swizzle);
  scheduler.raster_order = raster == 1 ? Raster::AlongM : (raster == 2 ? Raster::AlongN : Raster::Heuristic);
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args,
                                      epilogue_args, hw_info, scheduler};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess, "glm53_fp8_gemm: cannot implement");

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "glm53_fp8_gemm: run failed ",
              cutlassGetStatusString(status));
}

}  // namespace glm53

// One translation unit per config (parallel compile): GLM53_FP8_CONFIG(name, tile M, N, K, schedule, stages).
#define GLM53_FP8_CONFIG(NAME, TM, TN, TK, SCHEDULE, STAGES)                                                   \
  namespace glm53 {                                                                                          \
  using G_##NAME = Sm120Fp8Gemm<Shape<Int<TM>, Int<TN>, Int<TK>>, SCHEDULE, STAGES>;                           \
  void run_##NAME(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,                         \
                  torch::Tensor const& sa, torch::Tensor const& sb, int64_t swizzle, int64_t raster) {        \
    run<G_##NAME>(out, a, b, sa, sb, swizzle, raster);                                                       \
  }                                                                                                          \
  std::string describe_##NAME() { return describe<G_##NAME>(#NAME); }                                        \
  }
