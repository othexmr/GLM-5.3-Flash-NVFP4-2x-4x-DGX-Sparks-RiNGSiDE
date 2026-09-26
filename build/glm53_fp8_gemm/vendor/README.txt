Verbatim copies (no edits) of three vLLM v0.29.0 headers (vllm-project/vllm 98dff2a81d747d1dba01a47f939f48c3526d4206):
  cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp          <- csrc/libtorch_stable/cutlass_extensions/epilogue/
      Apache-2.0 (LICENSE-vllm)
  cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp       <- csrc/cutlass_extensions/epilogue/
  cutlass_extensions/epilogue/broadcast_load_epilogue_array_c3x.hpp <- csrc/cutlass_extensions/epilogue/
      BSD-3-Clause, Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES: modified excerpts of NVIDIA CUTLASS
      v3.5.0 that vLLM vendors; the licence is in their header (and ../../../licenses/BSD-3-Clause-NVIDIA-CUTLASS.txt)
They define the served epilogue tree (ScaledEpilogue: D = bf16_rn(scale_a[m] * (scale_b[n] * acc)) through
Sm90ColOrScalarBroadcast / Sm90RowOrScalarBroadcast), used unchanged by ../glm53_fp8_gemm.cuh.
