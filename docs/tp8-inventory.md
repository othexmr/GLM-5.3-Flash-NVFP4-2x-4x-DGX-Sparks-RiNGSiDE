# TP8 source and shape inventory

This baseline derives from corrected TP2 plan `b80911ec`, preserving required
imports and disabling specialized paths at their dispatch switches. It does
not remove complete files merely because they contain a TP2/TP4 optimization.
The target's 64 attention/KDA heads divide by eight. Residual width and indexer
heads remain replicated. Shapes below are logical unless explicitly packed.

| Operation / owner | Global dimensions | Per TP8 rank | Baseline implementation and gate |
|---|---|---|---|
| Residual stream and mHC | hidden 4096; four mHC streams | hidden 4096, four streams | Existing generic TileLang mHC; TP2/TP4 geometry switches rejected by startup guard |
| MLA query heads | 64 × 256 | 8 × 256 | q_b projection 2048 × 1536; native sparse-MLA sees eight heads × latent width 512 |
| Sparse MLA cache/op | latent width 512; top-k 2048 plus tail/padding | eight heads, width unchanged | Existing M7e eight-head dispatch pads MMA rows to 16; split-KV off; actual eight-head CUDA leaf required |
| KDA | 64 heads × 128 | 8 × 128 | B12X prefill binding and existing recurrent path; FP32 state; fused RecoverSSM commit off |
| KDA fused input projection | q/k/v each 8192, b 64; f_a/g_a each 128 | q/k/v each 1024, b 8; f_a/g_a each 128; total 3336 × 4096 | Actual mixed sharded/replicated loader; f_a/g_a never divide by TP |
| Indexer | 32 heads × 128; q_lora 1536 | all 32 heads; wq_b 4096 × 1536 | Replicated scorer; row split off. Standard indexer and canonical top-k remain; no fictitious four-head build |
| Dense layers 0–2 gate/up | intermediate 12288; hidden 4096 | concatenated gate/up 3072 × 4096 | Default installed FP8 dense path; Humming and TP4 tuned GEMMs absent |
| Dense down | 4096 × 12288 | 4096 × 1536 | Default FP8 dense, then NCCL reduction |
| Shared expert in routed layers | intermediate 2048; hidden 4096 | gate/up 512 × 4096; down 4096 × 256 | Standard TP while sequence parallelism is disabled; distinct from layers 0–2 and from routed-expert NVFP4 |
| Routed experts | 288, top-8; intermediate 2048 | every expert has gate/up 512 × 4096 and down 4096 × 256 | Generic B12X NVFP4 adapter; tile-major, cluster-map, atomic prefill and decode fastpath off |
| Expert packed bytes | 2 FP4 values per byte; scales per 16 values | gate/up U8 [288,512,2048], scales [288,512,256]; down U8 [288,4096,128], scales [288,4096,16] | Real checkpoint slices and installed packing must be checked on all logical ranks 0–7; scalar input/global scales retained |
| Input embedding | vocab 154880 × 4096 | logical 19360 rows × 4096 | Sharded input embedding (`GLM53_EMBED_REPLICATE=0`); physical padding is the loader's responsibility |
| LM head | vocab 154880 × 4096 | logical 19360 × 4096 | Separate LM-head loader and padding; default FP8 path; all-gather uses NCCL |
| DFlash2, later stage | hidden 4096, 32 query heads, eight KV heads, five layers | four query heads, one KV head | Pinned revision `7d74cdd8`; fixed K=4, target-only stage has no drafter mount |

MLA's fused low-rank projection is replicated at 2048 × 4096 (1536 query plus
512 latent channels; this checkpoint has zero rotary channels). MLA's output projection is 4096 × 2048
per rank; KDA's output projection is 4096 × 1024.

The drafter captures target layers [5,14,24,33,42] and retains residual width
4096. Its divisibility is separate from the target's. TP8 MLA q_b and indexer
wq_b are different shapes, so a TP4 optimization relying on their shape equality
cannot be carried across silently.

| Class | Included source / dispatch | TP8 decision |
|---|---|---|
| Generic correctness | TP8 vLLM patches for sampler vocabulary clamp, exact top-p, UTF-8 mask, kpool tail/slot mapping, exact-landing state write, DFlash capture/KV grouping | Retained; source pins identify each postimage |
| Generic infrastructure | `launch/render.py`, `compose.py`, `sources/import_plan.py`, `apply.py`, `verify.py`, Docker profile stages | Eight rank import/render/apply/manifest support; frozen TP2/TP4 tests |
| Native transport setting | `patches/nccl-tp8`, `fabric/ring.py`, `fabric/preflight.py` | Configured four/eight cable networks; exact four local GIDs; separate tree and library |
| TP-specific geometry | `src/tp8/glm53_speedup/{scheduler,graphs}.py`, mHC geometry modules, B12X tile/cluster support modules | Retain imports, disable dispatch; enabling a forbidden switch fails before target allocation |
| TP4 specialized transport | lean all-reduce, prefill RDMA, deferred reduction | Not installed by this TP2-derived baseline; both all-reduce selectors explicitly choose NCCL |
| Scheduler policies | adaptive chunk/cadence/fair split, first-token repay, geometry, runtime overrides | Off. Fixed-K and graph variants use ordinary vLLM scheduling; future policies need matched TP8 controls |
| Optional builds | Humming, tuned FP8 GEMM, rowspread, TP4 prefill RDMA | Absent; no performance claim or automatic fallback into an unbuilt extension |

`sources/installed-files.json` is the complete file inventory. The TP8 runtime
guard is `src/tp8/glm53_tp8.py`; its configuration digest is compared on the CPU
process group before target allocation. Every rank must report the same source,
image, model/tokenizer and native-library identities through the tester kit.

The allocation ledger starts with an explicit 4 GiB KV cache. Sharded weights,
replicated weights/state, packed copies, communication buffers and graph
workspaces need separate observed values; they are not estimated as a flat 45%
reduction from TP4. The kit records sampled MemAvailable across the full load
interval, which includes packing/warm-up/capture, and through serving. These
samples do not claim exact sub-phase peaks or catch transients shorter than the
sampling interval. Exact allocator peaks require additional engine traces.
