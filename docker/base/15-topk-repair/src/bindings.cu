// SPDX-License-Identifier: Apache-2.0
// Standalone validation binding for the pinned vLLM PR55314 kernel repair.
// Keep upstream source bytes intact; isolate their C++/CUDA symbols from _C.
#include "csrc/libtorch_stable/torch_utils.h"
#include <torch/csrc/stable/library.h>

#define vllm nvfp4_topk_pr55314_impl
#define persistent_topk nvfp4_persistent_topk_pr55314
#include "csrc/libtorch_stable/topk.cu"
#undef persistent_topk
#undef vllm

STABLE_TORCH_LIBRARY_FRAGMENT(nvfp4_topk_pr55314, ops) {
  // Workspace is explicitly mutable here because topk.cu stream-orders a memset
  // and the kernel updates its row state. Argument order/dtypes match the pin.
  ops.def("persistent_topk(Tensor logits, Tensor lengths, Tensor(a!) output, "
          "Tensor(b!) workspace, int k, int max_seq_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(nvfp4_topk_pr55314, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&nvfp4_persistent_topk_pr55314));
}
