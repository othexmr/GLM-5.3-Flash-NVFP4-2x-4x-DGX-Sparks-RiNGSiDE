// SPDX-License-Identifier: Apache-2.0
// ctx1m-topk rowspread build (2026-09-25): the image's pinned PR55314 topk.cu (2d0288fa) with the rowspread
// persistent_topk.cuh, isolated in its own C++ namespace and torch library (nvfp4_topk_rowspread) so that it loads
// next to the served nvfp4_topk_pr55314 build. Same op schema as the served binding.
#include "csrc/libtorch_stable/torch_utils.h"
#include <torch/csrc/stable/library.h>

#define vllm nvfp4_topk_rowspread_impl
#define persistent_topk nvfp4_persistent_topk_rowspread
#include "csrc/libtorch_stable/topk.cu"
#undef persistent_topk
#undef vllm

STABLE_TORCH_LIBRARY_FRAGMENT(nvfp4_topk_rowspread, ops) {
  ops.def("persistent_topk(Tensor logits, Tensor lengths, Tensor(a!) output, "
          "Tensor(b!) workspace, int k, int max_seq_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(nvfp4_topk_rowspread, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&nvfp4_persistent_topk_rowspread));
}
