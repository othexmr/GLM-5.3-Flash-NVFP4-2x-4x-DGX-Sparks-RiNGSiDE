# PR55314 top-k repair validation bundle

This bundle is CPU-prepared and has not been compiled or executed by its reviewer. The lead owns compilation, CUDA tests, image packaging and model qualification.

The frozen vLLM source is `4500c80c080328dfe62435d083f4063e00d987df`. Only the `persistent_topk.cuh` and `topk_histogram_4096.cuh` hunks from [vLLM PR55314](https://github.com/vllm-project/vllm/pull/55314), observed open at head `a68f6649976720b85c5e2df50af6aadd293cf5ca`, are applied. Both hunks passed dry-run application against the pin, then were applied to this scratch copy. Original source bytes, the complete upstream diff, source URLs and before/after hashes are retained. The upstream Apache-2.0 license is included. The integration binding uses the same license.

From this directory, the CPU integrity check is:

```sh
python3 build.py --verify-only
```

For lead-owned compilation in the exact candidate CUDA/PyTorch toolchain:

```sh
python3 build.py
```

The script compiles one CUDA translation unit for GB10 sm121, loads its registration without running a GPU kernel, and creates `runtime/nvfp4_topk_pr55314`. It does not install or modify vLLM. Compilation needs the installed PyTorch stable C-shim headers/libraries, CUDA runtime and cuBLAS headers, CUB/CCCL, a C++20 host compiler, nvcc and Ninja. It does not need FlashInfer, CUTLASS, DeepGEMM, NCCL, KDA or MoE source. It preserves the pin's `USE_CUDA` and `TORCH_TARGET_VERSION=0x020B000000000000ULL` definitions. A new Torch operator namespace plus private C++/CUDA namespace substitution, hidden visibility and local symbol binding isolate it from the existing `_C_stable_libtorch` operator.

Run the test process with `runtime/` available on its Python import path, then import `nvfp4_topk_pr55314` **before** CUDA graph capture. The import validates the source pin, PR head, source-manifest digest, exact PyTorch/CUDA versions and binary digest, then loads the prebuilt operator once. It has no compile-on-demand or fallback. Its callable is:

```python
from nvfp4_topk_pr55314 import persistent_topk
persistent_topk(logits, lengths, indices, workspace, 512, max_seq_len)
```

The original `_C.persistent_topk` remains callable for the differential gate. Do not import the runtime package inside the same process that ran `build.py`, because that process already loaded the registration; start a fresh test process. The standalone schema marks both output and workspace mutable because the source mutates both; argument order/dtypes match the original callable. A void fake implementation supports tracing, but actual V2 compilation and replay still need validation.

Start with the retained unequal-cluster fixture and the unique spaced-score controls. Broaden to k=512, row counts 1/5/10/20/30, lengths 0/1/511/512/513, 8191/8192/8193, 9407/17802/32768/32769, and up to 65536 completed pools. Use mixed/padded rows and nontrivial row stride; for lengths below k require all valid indices exactly once followed by -1 padding. For longer unique-score rows require in-range unique indices and exact sorted score-value equality. For exact ties validate the selected values and uniqueness; do not demand a particular index order. Reuse allocations while changing scores and lengths, including clustered-to-random transitions, and capture/replay active 5/10/20 shapes after warm-up. Record the loaded binary and build receipt on both ranks. A successful kernel gate does not establish real-logit causality or whole-model reliability.

The proposed production route is an import at module initialization and replacement of the single CUDA `persistent_topk` call in `sparse_attn_indexer_kpool.py`; see `production-callsite.proposed.patch`. It retains prefill and other platform calls. The lead must package the runtime module at a deterministic import path, preserve startup hash verification and build-before-capture ordering, and run model correctness/graph gates under the resulting image identity. A full native `_C_stable_libtorch` rebuild is an alternative when retaining the original namespace is necessary, but would rebuild unrelated native kernels. There is no automatic namespace override here.

Prefill uses the separate `top_k_per_row_prefill` callable implemented in pinned `sampler.cu`; it does not include the two headers patched here. This bundle repairs only the mapped decode persistent-top-k dependency. In `sampler.cu:281–284`, candidates enter its 2048-item stash only when the complete threshold bin fits; `:422–456` otherwise descends further FP32 key bits, and `:295–299` clips only after all 32 bits match. The specific unequal-value coarse-bin clipping mechanism is therefore absent from that inspected prefill helper. This is source-level negative evidence, not comprehensive prefill GPU qualification.

The build loader uses [PyTorch's documented extension loader](https://docs.pytorch.org/docs/2.14/cpp_extension.html) with `is_python_module=False`; the local installed version remains the consuming API authority. Source-only verification cannot prove compilation support or graph behavior.
