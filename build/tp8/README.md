# Local TP8 evidence

`local-evidence.json` records maintainer checks of the experimental profile.
The target/source plan and actual image/native hashes identify the tested bytes.
It cannot be used as an eight-rank tester-kit admission receipt.

* ARM64/SM121 profile image built; 44 installed source/config pins matched and
  both native artifacts were rebuilt and identified.
* 193 CPU tests passed, including topology faults, unsupported flags, stale or
  incomplete receipts, cleanup timeouts, export privacy and unchanged TP2/TP4
  release files/manifest objects.
* Sparse MLA, top-k, eight-head KDA, Red Hat NVFP4 MoE and dense FP8 leaves passed
  on one GB10. Dense: 370 real-weight cases with fresh-input graph equality.
  MoE: real installed rank loader on logical shards 0–7, with bytewise slice
  checks, installed compressed-tensors conversion and a dequantized reference.
* The separate NCCL library passed three paired comparisons (six communicator
  boots) on four ranks. Actual Ring/IB routes, both roots and both neighbors
  were verified. The wall timings are noisy; this is correctness evidence and
  provides no performance promotion.

Failed harness runs and their partial results are retained in the maintainer's
lab evidence. They exposed old checkpoint field names/scale conventions, an
h32 KDA window assumption, shared-versus-dense MLP geometry, zero rotary width,
and a test constructor's default dtype. No numerical limit was relaxed.

Eight-rank transport, whole-model load, drafter state, graph/concurrency serving,
memory capacity and performance remain external-test requirements. The kit
stages these checks and exports a structured report. It does not promote a
candidate based on these local receipts.
