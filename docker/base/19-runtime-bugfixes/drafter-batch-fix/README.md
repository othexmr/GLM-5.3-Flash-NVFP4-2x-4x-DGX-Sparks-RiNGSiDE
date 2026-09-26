# ND2: fixed per-request DFlash2 sliding-window XQA partitions

The candidate removes the confirmed batch-composition dependence in the SM121
DFlash2 attention kernel. Its 80-case active-window GPU matrix is bitwise exact
across batch composition and padding; the retained baseline mismatches in 47 of
those cases. The fixed kernel also passes the independent FP32 attention reference
and repeated CUDA graph replay in every active-window case. These are standalone
component results, not a TP2 serving or promotion qualification.

## Exact failure and corrected call path

`outputs/2026-09-08-nvfp4-fresh-k7-k4-results/workflows/w2/REPORT.md:69` records
ND2 as an XQA split-count issue. The report points at `mha.cu:3015`, but the actual
FlashInfer wrapper calls **`launchMHAFlashInfer` at original `mha.cu:3123`**.
`XQA_NB_SUB_SEQ` is read only by the other `launchMHA` entry point; setting that
environment variable does not repair this serving path.

The live stage-17 sources were checked against the retained runtime-source tree
and match the manifest's original hashes. The call chain is:

1. `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` prepares a fixed number
   of queries per request, selects FULL CUDA graphs, and rebuilds draft metadata.
2. `vllm/v1/attention/backends/flashinfer.py` routes the SM12x draft group to the
   dedicated XQA decode API, including non-causal draft blocks.
3. `flashinfer/decode.py:xqa_batch_decode_with_kv_cache` and `flashinfer/xqa.py`
   invoke `xqa_wrapper.cu`, which calls `launchMHAFlashInfer`.
4. Its old launch policy computes
   `min(max(1, SM_count / (batch_size * KV_heads)), ceil(table_capacity / 256))`.
   At the retained TP2 geometry, long-enough table capacity gives 12, 6, 4, 3,
   and 2 partitions for batch sizes 1, 2, 3, 4, and 6.
5. The kernel visits this request's tiles beginning at
   `nbSkipLeadingTiles + partition_index`, stepping by that partition count.
   Partial softmax/output values are merged in partition order. Thus changing
   batch size changes the floating point expression even though each individual
   launch is repeatable. This is not an XQA-versus-fallback dispatch finding.

The batch-dependent variable called `maxSeqLen` is recomputed from page-table
width in `flashinfer/xqa.py`; it is capacity, not necessarily the batch's observed
maximum sequence length. The per-request window start uses the request's own
sequence length and actual query width. Fixing the partition count therefore
also removes page-table/capture-capacity dependence from the partition policy.

## Implementation

Four source files change; `source-manifest.json` pins every before/after hash.
`drafter-batch-fix.patch` is the same change in reviewable unified-diff form.

- `mha.cu` uses
  `min(max(1, SM_count / KV_heads), ceil(sliding_window / 256))` for this policy.
  It is independent of batch size, batch query maximum, page-table capacity, and
  graph padding. The validated geometry is BF16 Q16/KV4, head size 128, page size 64,
  sliding window 2048, giving eight partitions. The existing masks, merge order,
  SM121 memory fence, target attention, deterministic sparse top-k, and routing
  canonicalization are unchanged.
- `xqa_wrapper.cu` calls a host bounds check before the GPU launch. `mha.cu` uses
  the kernel's exact scratch types and alignment to validate available scratch,
  semaphore storage, and 32-bit indexing. It fails rather than choosing fewer
  partitions when scratch is insufficient.
- `mha.h` declares that bounds check.
- `jit/xqa.py` enables the policy for sliding-window modules whose compilation
  target set includes SM12x, adds its compile define, and appends the distinct
  `_batch_invariant_sliding_v1` URI suffix. Existing cached/prebuilt binaries
  cannot satisfy this new module identity. Non-windowed module identities are
  unchanged.

The qualified target set is GB10 SM121 only. A mixed compilation target list
containing SM12x also compiles this policy into the generic kernel for the other
listed architectures; that mixed-architecture artifact is not qualified here.
This candidate makes no invariance claim for the unwindowed diagnostic path,
which deliberately retains the original policy.

The model's retained draft config has five sliding-attention layers,
Q32/KV8 globally, head dimension 128, sliding_window 2048, and block_size 8. Tensor
parallelism halves the heads. K4 normally uses five draft input positions;
q1, q5, and q8 exercise single-token and speculative specializations.

## Verification and cost

`python3 test_cpu.py` passes 11 tests, including compiled execution of the actual
candidate's C++ host split/bounds logic, source/JIT identity tests, exact workspace
boundary and overflow checks, and all-file installer prevalidation/idempotence.
CPU tests do not stand in for CUDA compilation or GPU correctness.

Parent-owned GPU receipts are adjacent in `../`:

- `xqa-baseline-full.json`: 80 active-window cases, 47 batch mismatches,
  zero same-shape replay failures, zero FP32-reference failures.
- `xqa-fixed-full.json`: 80 active-window cases, zero batch mismatches,
  zero same-shape replay failures, zero FP32-reference failures.
- Both have 80 unwindowed diagnostic cases, preserving 47 batch mismatches.
  They are outside the active-window acceptance gate and remain explicit negatives.

Both matrices use the same Q/K/V bytes for the compared request. Cases cover
q1/5/8; sequence lengths 513/4097/32769/131073; batch sizes 1/2/3/4/6; short/long
companions; wider page tables; and zero-query graph padding via ragged offsets.
Uniform nonzero queries with zero KV length are invalid padding inputs and are
not used. Masks use the actual uint32 packed words viewed as uint16, including
zero high words. Each case checks six poisoned-output CUDA graph replays.
The independent FP32 reference uses per-query sliding boundaries and a complete
non-causal draft block; its tolerance is separate from the strict BF16 raw-bit
comparison. Maximum FP32 absolute error over active cases: baseline 0.00219518,
fixed 0.00183794.

The parent built and tested:
`local/glm53-nvfp4:19-drafter-batch-review1-20260909`, image ID
`sha256:b597d336d4bd8c2228fb9f194eab8dda91920cdd33e86927a15e13aecab51f17`.

Fixed splitting costs more work at concurrency. Mean graph time over the four
sequence lengths for q5/B6 is 24.91 us baseline versus 41.20 us fixed; q8/B6 is 26.72 us
versus 41.69 us. Solo q5/q8 is about 13.92 us baseline versus 15.5 us fixed. These are
standalone kernel timings, not service throughput. Preserve this measured cost
and qualify matched TP2 acceptance/TTFT/ITL before promotion. Eight partitions
are the reviewed candidate; alternate partition tuning is a separate experiment.

## Install and GPU recipe

`install.py --check` validates the entire source closure without changing it.
`install.py` checks every source before writing any member, accepts only its pinned
original/fixed hashes, and is idempotent. It does not clear shared JIT caches.

For the build, use this directory as the Docker build context and the exact
reviewed stage-19 image as `BASE_IMAGE`. The Dockerfile installs the four-file
closure. The parent owns all image integration, remote transfer, lifecycle changes,
and GPU scheduling; this packet did not change a live service or run GPU work.

On each Spark, in the parent-controlled service-down/exclusive-GPU interval,
run the retained `../gpu-gates/xqa_batch_oracle.py` in the candidate image:

```sh
python3 /gate/xqa_batch_oracle.py --output /receipts/xqa-fixed-full.json --require-exact
```

Mount the parent fixture at `/gate` and a fresh receipt directory at `/receipts`.
The script checks active-window exactness and records the unwindowed negative
separately. Preserve both baseline and candidate JSON plus image/source hashes.
No environment split override is needed or effective on the original entry point.

Additional bounds gate at Q16/KV4, D128, BF16, q5 or q8, batch 6, window_left 2047:

- Via `flashinfer.decode.xqa_batch_decode_with_kv_cache`, allocate exactly
  `8*1024*1024 + 1622016` uint8 workspace bytes: must launch and match the reference.
- Allocate one byte less: must raise `workspace is too small` before a GPU launch;
  a subsequent valid call must still pass.
- Via low-level `flashinfer.xqa.xqa`, the minimum semaphore is 96 bytes and scratch
  is 1,622,016 bytes. One-byte-short semaphore must be rejected before launch.

After the standalone tests, TP2 serving qualification still requires exact
same-request token IDs and logprobs across concurrency/capture padding, actual
candidate JIT URI on both ranks, unchanged target/top-k source hashes, acceptance
and latency comparisons, and healthy graph replay. Cache/NVMe and async scheduling
remain off under the parent-owned K4 profile. Component proof does not establish model-wide batch invariance or full service
promotion.
