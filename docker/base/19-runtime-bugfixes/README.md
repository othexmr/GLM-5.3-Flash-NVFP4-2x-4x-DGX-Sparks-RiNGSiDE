# Runtime bug fixes

This registered stage follows stage 17 directly. The rejected stage-18 top-k algorithm remains excluded. The installer verifies every installed source preimage and candidate hash before editing, then compiles the installed Python. The drafter installer independently gates its four source files and versions the sliding-window JIT cache.

The stage repairs visible-content log-probability alignment and tool truncation status, graph-captured request-ID buffer ownership, top-k JIT specialization, forced swap_ab dispatch, retained KDA boundary accounting, cacheable-group estimation and DFlash2 sliding-window batch invariance. Prefix retention zero keeps the original target prefill chunk geometry. Ambiguous or non-token-aligned content spans omit log-probability metadata rather than attaching shifted entries.

## Evidence so far

The review7 image (`sha256:ed5464335dbb16df89cc55c292d9b03b2efd37ad14a383aeb95a1810cc69a5d1`) passed its CPU integration tests and inherited parser matrices. The a164 pair ran that exact image on both ranks, with source hashes and loaded drafter JIT mappings checked. Its 32,769-token cold fixture reproduced all 32 saved output token IDs and log probabilities exactly (maximum delta zero), with all prompt tokens computed locally and no prefix hits. Native Responses full/stream checks passed aligned probabilities and valid-JSON truncated function calls marked incomplete. Native Chat truncation passed on a163, whose Chat repair bytes are unchanged.

Standalone GPU checks passed the actual graph-buffer regression, 24 sparse top-k cases, and forced-layout fallback across nine route arms and 42 graph bindings. The drafter matrix changed from 47/80 batch-composition mismatches to 0/80, passed six graph replays per case and an independent FP32 reference, and correctly rejected a scratch buffer one byte too small. Full-attention behavior was intentionally retained; this fix qualifies the tested SM12x sliding-window path.

Two 131,073-token cold replays also passed exact token IDs and log probabilities, including equality with the saved a162 fixture. The complete a164 baseline battery passed strict counters for 36 decode cells and 12 prefill cells, with zero request preemption. Median aggregate client decode rates for C1 through C6 were 43.131, 57.513, 55.407, 74.723, 82.047 and 91.615 tokens/s. These are one-boot baseline measurements, not a matched claim of improvement. System paging occurred during scored intervals; three collector tail windows were incompletely bracketed. See [the retained receipts](../../benchmarks/runtime-bugfixes/README.md). This stage does not include the four experimental speed changes or enable NVMe. The a164 profiler allocated 500,213 KV tokens versus a163's 580,461; rank 1's lower profiled free KV budget explains the pair's lower allocation, but the underlying boot-to-boot memory variation remains a qualification consideration. No host settings were changed.

## Retained failed attempts

The first scheduler integration changed chunk geometry even with prefix retention zero. On a163, the cold fixture retained exact output IDs but log probabilities changed. The correction restores the old zero-retention geometry and passes a164's strict cold comparison. An earlier content-span matcher rejected a hidden trailing EOS token and returned empty requested probabilities; the revised matcher validates the consumed prefix and passes the live API check. Raw failed receipts remain in the laboratory review directory.

GPU check scripts here require an explicitly owned idle GPU. Do not run them alongside a serving process. NVMe worker/source repairs are a separate source deliverable and have not qualified physical external NVMe or CUDA DMA recovery.
