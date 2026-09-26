# Results

Measured on four (TP4) and two (TP2) NVIDIA DGX Spark systems (GB10, 128 GB unified memory each) cabled as a
switchless ring, 2026-09-21 to 2026-09-26. **The RiNGSiDE TP4 row's RigMark cells are medians of 2 runs** (the release
window of the TP4 profile's plan, one boot; receipts in `runs/2026-09-26-ringside-redhat-rowsplit-tp4-1/`), **the
RiNGSiDE TP2 row's of 2 runs** (its final run, one boot), the final TP4 run before the release (lab plan `387b3ff3`,
an additional arm) holds medians of 2 runs (receipts in `runs/2026-09-25-tp4-final-2/`), the release plan's run on the
NVIDIA checkpoint (lab plan `31978478`, an additional arm) holds medians of 3 runs (receipts in
`runs/2026-09-25-ringside-combined-tp4-1/`), the previous profiles' rows (additional arms) hold medians of 5 runs
(TP4, receipts in `runs/2026-09-24-rigmark5-tp4-1/`) and of 3 staggered rounds (TP2), and **every other cell is a
single run**; the runs column of each table says which. `results.json` holds every number with the lab window it comes
from, `evidence/` the records behind the derived figures, the protocol of the replay matrix, the tool-eval-bench runs
and a manifest of every measured arm, and `runs/` the RigMark receipts of the four multi-run TP4 windows, and the
llama-benchy receipts of three of them (not the release window's: llama-benchy did not run on this plan).

> **NON-DETERMINISTIC outputs.** Both profiles accumulate NVFP4 MoE outputs with atomic adds in prefill (eager
> prefill launches and the prefill rows of mixed prefill and decode steps), and the TP4 profile also in decode launches
> of 10 or more rows (`B12X_GLM53_ATOMIC_DECODE_MIN_ROWS=10`), so the order varies from run to run and outputs are not
> bit-reproducible. TP4 decode launches below 10 rows and all TP2 decode launches stay deterministic.

## What is compared

Each row is a **whole recipe** as it ran on this hardware: its own model checkpoint, quantisation, speculative
decoding, image, runtime, parallel layout and admission limit. A difference between two rows is the difference
between two recipes, not between two kernels or two settings. The comparison recipes are independent projects with
other design goals (EXL3 quantization, 1M-token KV); they are named by GitHub owner and repository with the commit
measured, and the configuration column and the manifests below say how each was set up. They admit fewer concurrent
requests than the RiNGSiDE TP4 profile, so the tables stop at 6 streams.

## How it was measured

- **RigMark** ([alexellis/rigmark](https://github.com/alexellis/rigmark)), in the fork
  [othexmr/rigmark](https://github.com/othexmr/rigmark) (branch `feat/staggered-arrival`, commit
  [`40fabcaf`](https://github.com/othexmr/rigmark/commit/40fabcaf6e963ac27ab347f61d54a225dbd07419), parent
  alexellis/rigmark `e748928d`). The windows record the local revision `6e0e24a6`; `40fabcaf` was
  published with the identical tree (`3ba4ae72`). Protocol 1.1.0, prompts 1.0.0, `reasoning_effort` low in
  the chat template arguments. Cells: cold time to first token at 8K, 32K and 64K prompt tokens; single-stream
  decode rate on code, prose and structured prompts; aggregate end-to-end throughput of short code requests at 1 to 6
  concurrent streams (8, 12 and 16 for the RiNGSiDE TP4 profile).
- **Staggered arrivals** (the fork's section). *Prefill-first*: one 32,768-token request starts, and while it is in its
  prefill L-1 short requests arrive; the cell is the median time to first token of those short newcomers.
  *Decode-first*: L-1 short streams are decoding when a 32,768-token request arrives; the cell is that request's time
  to first token. L = 2, 4, 6 (also 8, 12, 16 for the RiNGSiDE TP4 profile). "-" means no valid cell (the round's
  newcomer only started after every incumbent had finished).
- **sparkDash**: the lab's reproduction, in Python, of the decode benchmark of MiaAI-Lab/sparkDash (its DecodeBench
  protocol: the same prompts, temperature 0, 400 tokens, thinking off, one warm-up stream); code prompts, one stream.
- **Replay matrix** (RiNGSiDE profiles only): a replay of authored multi-user scenarios with a 16-task quality set and
  an arrival-rate sweep against a per-request latency target. Protocol: `../protocols/README.md`; record:
  `evidence/replay-matrix.json`.
- Model `RedHatAI/GLM-5.3-Flash-NVFP4` revision `18d55bfd` for the RiNGSiDE TP4 row and `nvidia/GLM-5.3-Flash-NVFP4`
  revision `423acf37` for the TP2 row, drafter `incoai/GLM-5.3-Flash-DFlash2` revision `7d74cdd8` for both.
- **The RiNGSiDE TP4 row** is the release window of the TP4 profile's own plan, lab plan `89dfef84`: one boot on
  2026-09-26 with, in this order, 2 runs of every RigMark cell (00:39-01:09; the median is shown), sparkDash (a Python
  reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol, 2 rounds; each cell is the median of its rounds),
  tool-eval-bench and the correctness checks below. The emoji and CJK wall stress test and llama-benchy were not run
  on this plan; the llama-benchy results below are the release plan's run on the NVIDIA checkpoint (lab plan
  `31978478`). The replay matrix was not run on it. The final TP4 run of lab plan `387b3ff3` (the profile before the
  release's changes) is listed with the additional arms, and so is the release plan's RigMark run on the NVIDIA
  checkpoint (lab plan `31978478`).
- **The previous TP4 profile** (lab plan `50e80d44`, 2026-09-24) is listed with the additional arms: RigMark and the
  staggered arrivals from one boot with 5 runs of every cell, and sparkDash and the replay matrix from the final run of
  lab plan `9e4f793a` (the same profile without the KDA checkpoints), single runs, marked with that plan. A single-run
  window of the same code is listed there too: lab plan `594faefb`, whose copy of `mhc_prefill_sharding.py`
  differs only in its header comment.
- **The RiNGSiDE TP2 row** is the final TP2 run: lab plan `69ad74fc` (the base of the TP2 profile, before the
  release's changes and fixes), one boot on 2026-09-25 with 2 runs of every RigMark cell (02:27 to 02:52; prefill at
  8K, 32K and 64K). It was one window without a fresh reference window (the owner's decision), so no rule judged it;
  the TP2 section compares it, as information only, with the stored windows of the previous TP2 profile. sparkDash (a
  Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) ran once in a separate window of the same
  plan on 2026-09-25; the replay matrix was not run on it.
- **The previous TP2 profile** (lab plan `32dd2ab8`, 2026-09-24) is listed with the additional arms: RigMark,
  staggered arrivals over three rounds and the replay matrix swept at 0.05, 0.1, 0.15 req/s only, with sparkDash from
  the final run of the TP2 base before it (lab plan `f1771e0a`), marked with that plan. **The owner promoted it
  although three guard clauses of the lab's rule failed**; the TP2 section lists every clause.

## Caveats

- **Single runs.** Apart from the RiNGSiDE TP4 row (medians of 2 runs), the RiNGSiDE TP2 row and the final TP4 run
  before the release (medians of 2 runs), the release plan's run on the NVIDIA checkpoint (medians of 3 runs), the
  previous TP4 profile's row (medians of 5 runs) and the previous TP2 profile's staggered cells (medians of 3 rounds),
  no cell has repetitions; small differences between single runs are not significant. With 2 runs, the median of a
  cell is the mean of the two and shows nothing of their spread.
- **Room temperature.** Room temperature was not logged. Two windows of the same TP4 plan measured 2.5 hours apart
  differed by +0.47 % in median decode step time at fixed (streams, K) and by -4.22 to +2.33 % in single cells.
  Windows whose code changes the lab judged not to slow decode moved by up to +1.21 % in median decode step time when
  measured 2.1 hours apart (single cells up to +3.24 %), which the lab attributes to the room temperature. The rows
  below come from different days and times of day. Record: `evidence/temperature-drift.json`.
- **Throughput depends on the prompt text.** Speculative decoding accepts more or fewer drafted tokens depending on
  what is generated, so decode rates and aggregate throughput depend on the prompt text as well as on the recipe;
  the cells hold for the benchmarks' own prompts.
- **Request caps.** The comparison recipes ran with at most 4 (MiaAI-Lab, their `MAX_NUM_SEQS=4`; 2 in their
  long-coding example) or 6 (tonyd2wild) concurrent requests. For tonyd2wild's TP4 recipe the 6 slots at 262,144
  context were the lab's choice (the published launch uses 64 slots at 500,000); the TP2 recipe ships 6. The RiNGSiDE
  TP2 profile also admits 6, its TP4 profile 16. Up to 6 streams every recipe ran every cell (a recipe that admits
  fewer queues the rest); streams beyond a recipe's cap were not run for the comparison recipes.

## Ranking

Rows are ordered by mean rank (best first). For each of the 15 ranked columns (cold TTFT 8K / 32K / 64K;
decode code / prose / structured; short code aggregate C1 and C6; staggered prefill-first and decode-first at L2, L4
and L6; sparkDash code x1) the rows with a value are ranked from 1 (best: the lowest time, the highest throughput);
equal values share the mean of their positions. A row's mean rank is the mean over the columns where it has a value.
A missing cell ("-", "not run") is left out of that row's mean: it neither counts against the row nor for it, which
can favour a row whose missing cell would have ranked low. The last column gives the mean rank and the number of
columns it is taken over. Cells marked with another lab plan count like the others. The admission limit and the replay
matrix are not ranked.

## TP4

> Decode and aggregate throughput depend on the prompt text through speculative acceptance (Caveats).

| recipe | configuration | admits | runs | cold TTFT 8K / 32K / 64K (s) | decode code / prose / structured (tok/s) | short code aggregate C1 / C6 (tok/s) | staggered prefill-first newcomer L2 / L4 / L6 (s) | staggered decode-first newcomer L2 / L4 / L6 (s) | sparkDash code x1 (tok/s) | replay matrix | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | release window of the TP4 profile, lab plan 89dfef84 (the RiNGSiDE release plan); RigMark cells are medians of 2 runs; sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) the median of 2 rounds in the same boot; replay matrix not run on it | 16 | 2 | 1.71 / 6.46 / 12.87 | 105.3 / 61.7 / 145.4 | 77.0 / 191.9 | 2.42 / 2.47 / 2.55 | 7.66 / 7.85 / 9.41 | 151.3 | not run | 1.00 (15 of 15) |
| tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark | commit 2ac4e8dd, the recipe's image at 262,144 context and 6 request slots (the lab's choice; the published launch uses 500,000 and 64), on this ring's NCCL transport (the recipe's optional all-peer RoCE route needs connectivity a ring does not have); sparkDash from a separate window of the same launch plan | 6 | 1 | 4.52 / 14.16 / 28.39 | 82.5 / 35.8 / 127.4 | 58.3 / 92.4 | 7.93 / 8.05 / 8.16 | 14.58 / 14.71 / 14.90 | 120.8 | not run | 2.33 (15 of 15) |
| MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks | commit 7cded8e, their default-off speed options on except KDA_BF16_LARGE_M and DRAFT_KV_COMPACT; image built by their Dockerfile with one self-test skipped; TP4 on this ring's NCCL transport | 4 | 1 | 10.53 / 14.72 / 29.65 | 70.6 / 36.9 / 80.8 | 49.7 / 107.7 | 6.35 / 6.39 / 6.48 | 42.02 / 51.34 / - | 81.6 | not run | 2.64 (14 of 15) |

### The RiNGSiDE TP4 profile at 8 to 16 streams (release window, medians of 2 runs)

| streams | short code aggregate (tok/s) | prose aggregate (tok/s) | staggered prefill-first newcomer (s) | staggered decode-first newcomer (s) |
|---|---|---|---|---|
| 8 | 232.6 | 154.0 | 2.53 | 9.51 |
| 12 | 283.2 | 194.4 | 2.67 | 9.82 |
| 16 | 345.8 | 225.6 | 2.80 | 10.06 |

Not run for the comparison recipes: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks admits at most 4 concurrent requests;
tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark admits at most 6 concurrent requests.

## TP2

> Decode and aggregate throughput depend on the prompt text through speculative acceptance (Caveats).

| recipe | configuration | admits | runs | cold TTFT 8K / 32K / 64K (s) | decode code / prose / structured (tok/s) | short code aggregate C1 / C6 (tok/s) | staggered prefill-first newcomer L2 / L4 / L6 (s) | staggered decode-first newcomer L2 / L4 / L6 (s) | sparkDash code x1 (tok/s) | replay matrix | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | final TP2 run, lab plan 69ad74fc (the TP2 profile's base, before the release's changes and fixes); RigMark cells are medians of 2 runs; one window, no fresh reference; sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) from a separate window of the same plan; replay matrix not run on it | 6 | 2 | 3.56 / 12.94 / 25.66 | 56.5 / 33.0 / 83.4 | 44.0 / 97.3 | 4.70 / 5.15 / 4.90 | 14.38 / 14.81 / 15.42 | 85.7 | not run | 1.00 (15 of 15) |
| MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks | commit 7cded8e, their default-off speed options and compact draft pages on; image built by their Dockerfile with one self-test skipped | 4 | 1 | 5.18 / 19.79 / 39.05 | 48.6 / 25.0 / 75.3 | 37.5 / 67.3 | 5.66 / 5.76 / 5.77 | 54.87 / 69.06 / - | 73.0 | not run | 2.21 (14 of 15) |
| tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark | commit 9fea48c with five recorded changes: this pair's direct link and addresses, a pinned drafter, the recipe's RoCE all-reduce off (it needs another image from the same author), the chat template of the author's TP4 recipe, the author's documented prefix-cache repair | 6 | 1 | 6.29 / 28.56 / 57.40 | 41.7 / 15.8 / 63.5 | 24.3 / 84.7 | 17.85 / 11.33 / 22.75 | 19.80 / 26.50 / 20.24 | 56.6 | not run | 2.73 (15 of 15) |

Lab plan `69ad74fc`, the base of the TP2 profile, is the previous TP2 profile (`32dd2ab8`) plus the startup warm-up v3
(six prompts), the replay boundary v3 with the admit fix (at most 6 states kept) and P1 first-token repay, as in the
TP4 profile; TP2 has neither the fp16 KDA state nor the greedy argmax verification. **Its final run had no fresh
reference window** (the owner's decision), so no rule judged it. As information only, the lab compared it with the
stored windows of the previous profile (different boots, times of day and run counts; a ratio per cell, above 0 % =
the final run is faster, geometric means per group): overall +15.8 % over 21 cells | cold +5.5 % replay +145.7 % code
+1.0 % prose +2.6 % staggered +0.6 %. The replay-boundary cache check of the final run found the predicted hits with 3
differences from fresh prefills and 1 near-tie, and no hit was refused. Record: `evidence/tp2-row-69ad74fc.json`. The
TP2 profile of this repository, lab plan `b80911ec`, was not benchmarked: it is the TP2 release plan `33a8cc9e` (lab
plan `69ad74fc` with change notices and the DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4
code) with 5 correctness fixes, each verified by GPU leaves and CPU checks (`CURRENT.md`), served on the checkpoint
`RedHatAI/GLM-5.3-Flash-NVFP4` revision `18d55bfd` with `--override-generation-config {"top_p":0.95}` and
`--kv-cache-memory 11274289152` (was `12348030976`); the TP2 numbers of this profile follow.

The previous TP2 profile (lab plan `32dd2ab8`), on which the current one builds, is the TP2 base before it
(`f1771e0a`) plus prefix-cache retention every 4,608 tokens, the prefix-cache contract and the MoE mixed-batch split.
It was measured against a fresh window of that base on the same node pair, back to back, and judged by the lab's rule,
fixed before the data and amended before any serving data. **Three clauses failed, and the owner promoted it
anyway:**

| clause | what it checks | measured | result |
|---|---|---|---|
| 1 | decode guard: fixed-(streams, K) decode step-time median change within +-1.0 %, and no cell above +3.0 % (cells with at least 30 steps in both windows) | 19 cells: median -0.62 %, worst cell 5/k4 +6.29 % (435 / 625 steps) | **failed** |
| 2' | warm replay: warm-replay TTFT change: 8,192 <= -25 % (timing sanity bound); 32,768 and 65,536 each <= +2.0 % | -55.35 % / -3.68 % / -43.25 % (8K / 32K / 64K) | passed |
| 3 | cold TTFT: no cold TTFT cell (8,192 / 32,768 / 65,536) above +3.0 % | -0.85 % / +0.17 % / -2.34 % (8K / 32K / 64K) | passed |
| 4' | load test: replay-sweep SLO attainment of the candidate >= the fresh reference at 0.05 and 0.1 req/s, valid measurements in both windows (the absolute >= 0.9 of the 13:09 rule is information) | 0.05 req/s: 0.83 against 0.33; 0.1 req/s: 0.50 against 0.07 | passed |
| 5 | staggered arrivals: geometric mean change over 9 cells (L2/L4/L6 x decode-first newcomer TTFT, prefill-first newcomer median TTFT, prefill-first long TTFT; each the median of 3 rounds) <= +1.0 %, and 3 valid rounds per cell in both windows | geometric mean -2.03 %; slowest cell L6 prefill-first newcomer +14.53 % | passed |
| 6 | host memory: candidate MemAvailable minimum >= 307 MiB (0.3 GiB) on both hosts during the campaign (sampled every 30 s), and no CUDA error / out of memory / EngineDeadError marker in either window | lowest MemAvailable 306 MiB on the rank 0 host (the first sample of the run; reference 365 MiB) | **failed** |
| 7' | correctness and engagement: cached-prefix check: the candidate's differences outside near-ties <= the reference's; no candidate first-difference top-2 gap above 2.0 (a difference without a recorded gap counts as above 2.0); predicted replay hits in both windows; engagement markers present in the candidate and absent in the reference; completed-answer checks pass; acceptance >= reference - 0.0083 | 2 differences outside near-ties (8K replay against cold: token 5, gap 2.125; 32K replay against cold: token 15, gap 0.875) | **failed** |

Not gated by the rule, and slower against the reference: the short-code aggregate at 6 streams moved -9.97 % (95.2 to
85.7 tok/s, single runs), and the L6 prefill-first newcomer +14.53 % (median of three rounds: 5.49 / 4.77 / 6.97 s).
Still open: whether the cached 4,608-token state gives the same next tokens as a fresh prefill, and the memory
headroom of the rank 0 host. Record: `evidence/tp2-weakspots-promotion.json`.

### Prefill by prompt length (release window, medians of 2 runs)

RigMark's prefill suite in the same boot: time to first token of a prompt new to the cache (cold) and of the same
prompt sent again right away (warm replay, served from the prefix cache). The largest prompt, 262,136 tokens, is 8
tokens under the 262,144-token context.

| prompt tokens | cold TTFT (s) | cold prefill (tok/s) | warm replay TTFT (s) | warm replay (tok/s) |
|---:|---:|---:|---:|---:|
| 8,192 | 1.71 | 4,785 | 0.43 | 19,197 |
| 32,768 | 6.46 | 5,069 | 0.54 | 60,711 |
| 65,536 | 12.87 | 5,093 | 0.70 | 93,607 |
| 131,072 | 25.98 | 5,044 | 0.71 | 183,307 |
| 262,136 | 53.57 | 4,893 | 0.77 | 341,343 |

### llama-benchy (the release plan on the NVIDIA checkpoint)

[eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0 did not run on the TP4 profile's plan (lab plan
`89dfef84`). The results below are the release plan's run on the NVIDIA checkpoint: lab plan `31978478` with
`nvidia/GLM-5.3-Flash-NVFP4` revision `423acf37` (the TP4 profile is that plan with the checkpoint switch), window
2026-09-25-ringside-combined-tp4-1 (2026-09-25 21:49-21:58, UTC+7), in the same boot as that arm's RigMark cells, on
llama-benchy's own prompts (English prose from its default book) with thinking on. Its numbers measure different
prompts, request shapes and statistics than RigMark's and do not compare cell for cell with the tables above; no
comparison recipe has been run with it here. Medians of 3 runs; arguments and files:
`runs/2026-09-25-ringside-combined-tp4-1/README.md` (the final TP4 run's are in `runs/2026-09-25-tp4-final-2/`, the
previous TP4 profile's llama-benchy results are in `runs/2026-09-24-rigmark5-tp4-1/`).

| context depth (tokens) | concurrency | prompt processing (t/s) | generation (t/s) | per request: prompt / generation (t/s) | TTFT (ms) | runs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 2,531.9 | 82.9 | 2,531.9 / 82.9 | 900 | 3 |
| 0 | 2 | 2,611.8 | 100.4 | 1,644.8 / 50.7 | 1,367 | 3 |
| 0 | 4 | 3,086.2 | 101.1 | 801.2 / 36.5 | 2,647 | 3 |
| 4,096 | 1 | 4,498.3 | 66.6 | 4,498.3 / 66.6 | 1,457 | 3 |
| 4,096 | 2 | 3,604.0 | 56.3 | 3,126.0 / 39.4 | 2,405 | 3 |
| 4,096 | 4 | 3,875.3 | 59.9 | 994.0 / 35.3 | 6,273 | 3 |
| 8,192 | 1 | 4,845.6 | 73.8 | 4,845.6 / 73.8 | 2,205 | 3 |
| 8,192 | 2 | 4,001.4 | 44.9 | 3,309.1 / 31.3 | 3,694 | 3 |
| 8,192 | 4 | 3,879.1 | 41.7 | 1,053.9 / 30.5 | 9,807 | 3 |
| 32,768 | 1 | 4,813.8 | 77.3 | 4,813.8 / 77.3 | 7,316 | 3 |
| 32,768 | 2 | 4,183.3 | 37.9 | 2,406.8 / 33.3 | 14,750 | 3 |
| 32,768 | 4 | 4,177.5 | 31.6 | 1,084.3 / 31.4 | 32,196 | 3 |

### tool-eval-bench (TP4)

[SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) commit `5381d589` (MIT): 69
tool-calling scenarios (tool selection, parameter precision, multi-step chains, refusal, error recovery, instruction
following, prompt-injection safety, structured output, planning), pass 2 / partial 1 / fail 0 points, at temperature 0
with seed 42 and thinking on (both recipes think by default), 4 scenarios in parallel. One run per window: the release
window's boot after RigMark and sparkDash, the final TP4 run's boot after llama-benchy, the TP4 profile's own check
window, and the MiaAI-Lab TP4 recipe with the launch plan of its TP4 row; no tonyd2wild recipe was run with it. The
median turn time depends on each recipe's speed and on how long its model thinks. Record:
`evidence/tool-eval-bench.json`.

| recipe | run | score | points | pass / partial / fail | failed scenarios | median turn (s) | runs |
|---|---|---:|---:|---|---|---:|---:|
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | release window (lab plan 89dfef84), after RigMark and sparkDash in the same boot (round 1 of 2) | 96 | 133 / 138 | 66 / 1 / 2 | TC-43, TC-68 | 5.4 | 1 |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | release window (lab plan 89dfef84), round 2: an extra round in the same boot at the owner's request, after the Korean check | 96 | 133 / 138 | 66 / 1 / 2 | TC-43, TC-68 | 5.3 | 1 |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | final TP4 run (lab plan 387b3ff3), after RigMark and llama-benchy in the same boot | 93 | 129 / 138 | 63 / 3 / 3 | TC-43, TC-51, TC-68 | 5.0 | 1 |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | check window of lab plan 6f490797 (the replay-boundary admit fix), tool-eval-bench and the cache check only | 93 | 129 / 138 | 63 / 3 / 3 | TC-43, TC-51, TC-68 | 5.4 | 1 |
| MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks | commit 7cded8e on TP4 with the launch plan of its TP4 results row, tool-eval-bench only | 94 | 130 / 138 | 64 / 2 / 3 | TC-43, TC-51, TC-68 | 8.7 | 1 |

## Correctness checks (TP4 release window)

In the release window's boot, after the speed runs: a Korean long-generation prompt (sampled with the server's
defaults, the test that showed long-generation breaks in earlier windows). The emoji and CJK wall stress test was not
run on this plan. A break follows the lab's rule: a derailed run, a runaway, a failed request, or a garbled run with
at least 5 flagged characters (U+FFFD or Cyrillic, Arabic, Thai or kana characters); fewer flagged characters are an
isolated glitch. Record: `evidence/release-window-tp4.json`.

| check | what ran | result |
|---|---|---|
| Korean long generation | the Korean prompt 1 x 12 at width 6 | 12 of 12 runs complete without a long-generation break; no isolated flagged character |

## Additional measured arms

Not the headline rows: the final TP4 run before the release (lab plan `387b3ff3`, 2 runs), the release plan's run on
the NVIDIA checkpoint (lab plan `31978478`, 3 runs), the previous TP4 profile (5 runs) and a single-run window of its
code, the earlier TP4 base on its own and with the optional
[msuiche/weightless](https://github.com/msuiche/weightless) GLP-44 steering, the previous TP2 profile (staggered cells
over three rounds), the earlier TP2 base, and a second MiaAI-Lab configuration (their long-coding example, measured on
the other node pair). All single runs except the NVIDIA-checkpoint run's, the final TP4 run's and the previous
profiles' repeated cells.

| recipe | configuration | admits | runs | cold TTFT 8K / 32K / 64K (s) | decode code / prose / structured (tok/s) | short code aggregate C1 / C6 (tok/s) | staggered prefill-first newcomer L2 / L4 / L6 (s) | staggered decode-first newcomer L2 / L4 / L6 (s) | sparkDash code x1 (tok/s) | replay matrix | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | the RiNGSiDE release plan on the NVIDIA checkpoint (nvidia/GLM-5.3-Flash-NVFP4 423acf37), lab plan 31978478; RigMark cells are medians of 3 runs in one boot, then llama-benchy; tool-eval-bench, sparkDash and the correctness checks did not run (the window stopped there); replay matrix not run on it | 16 | 3 | 1.69 / 6.50 / 13.13 | 103.8 / 59.8 / 147.4 | 79.8 / 197.2 | 2.37 / 2.43 / 2.47 | 7.66 / 7.81 / 9.32 | - | not run | 1.86 (14 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | final TP4 run before the release, lab plan 387b3ff3: lab plan 6f490797 without the replay-boundary admit fix; RigMark cells are medians of 2 runs; sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) from a separate window of lab plan 6f490797; replay matrix not run on it | 16 | 2 | 1.71 / 6.52 / 13.20 | 108.9 / 60.4 / 154.1 | 77.4 / 199.7 | 2.39 / 2.44 / 2.48 | 7.69 / 7.82 / 9.43 | 150.4 (lab plan 6f490797) | not run | 2.67 (15 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | previous TP4 profile, lab plan 50e80d44 (before A2b, greedy argmax verification, the replay boundary and P1); RigMark cells are medians of 5 runs; sparkDash and replay matrix single runs of lab plan 9e4f793a | 16 | 5 | 1.81 / 6.55 / 13.26 | 102.8 / 59.5 / 147.0 | 78.5 / 185.2 | 2.40 / 2.43 / 2.49 | 7.58 / 7.78 / 9.29 | 150.7 (lab plan 9e4f793a) | valid, quality 15/16, meets the latency target up to 0.2 req/s (lab plan 9e4f793a) | 2.97 (15 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | lab plan 594faefb, the same TP4 code as the previous profile 50e80d44; single-run window of 2026-09-24 13:09-13:30 | 16 | 1 | 1.88 / 6.57 / 13.28 | 101.6 / 57.4 / 145.0 | 79.1 / 175.7 | 2.38 / 2.45 / 2.48 | 7.65 / 7.88 / 9.36 | - | not run | 4.07 (14 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | earlier TP4 base (lab plan 9e4f793a, without the KDA checkpoints); NON-DETERMINISTIC decode | 16 | 1 | 2.06 / 6.90 / 13.59 | 101.7 / 59.8 / 149.7 | 82.0 / 186.5 | 2.40 / 3.51 / 2.50 | 7.85 / 8.01 / 9.40 | 150.7 | valid, quality 15/16, meets the latency target up to 0.2 req/s | 4.10 (15 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE + msuiche/weightless | earlier TP4 base with the optional weightless GLP-44 steering; NON-DETERMINISTIC decode | 16 | 1 | 2.19 / 7.33 / 14.48 | 101.6 / 58.5 / 136.7 | 84.3 / 174.8 | 2.60 / 2.66 / 2.71 | 8.27 / 8.41 / 9.97 | 148.9 | valid, quality 16/16, meets the latency target up to 0.2 req/s | 5.40 (15 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | previous TP2 profile, lab plan 32dd2ab8; staggered cells are the median of three rounds; sparkDash from lab plan f1771e0a; promoted although three guard clauses failed (TP2 section) | 6 | 1 (staggered: 3 rounds) | 4.08 / 13.11 / 25.93 | 54.8 / 33.6 / 77.0 | 41.8 / 85.7 | 4.65 / 4.74 / 5.49 | 14.99 / 14.99 / 15.26 | 84.8 (lab plan f1771e0a) | valid, quality 16/16, no swept rate meets the latency target | 7.30 (15 of 15) |
| othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE | earlier TP2 base (lab plan f1771e0a, before the weak-spot fixes) | 6 | 1 | 4.38 / 13.56 / 25.72 | 54.6 / 33.0 / 81.5 | 42.8 / 99.6 | 5.89 / 11.96 / 4.79 | 15.63 / 16.03 / 16.68 | 84.8 | valid, quality 16/16, no swept rate meets the latency target | 7.57 (15 of 15) |
| MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks | commit 71962cd, their examples/tp2-long-coding.env, other node pair, automatic KV pool (12.2 / 12.0 GiB per rank) | 2 | 1 | 6.04 / 22.74 / 45.30 | 51.2 / 20.7 / 76.6 | 38.5 / 52.9 | 1.82 / 16.88 / 29.96 | 69.64 / - / - | 75.7 | not run | 8.23 (13 of 15) |

## Arm manifests

What each row ran, from `evidence/comparison-arms.json` (every field there names its source; fields the lab did
not record are marked "not recorded").

| row | repository and commit | checkpoint | quantisation | speculative decoding | runtime | TP | context | admits | KV cache per rank |
|---|---|---|---|---|---|---|---|---|---|
| tp4-ringside | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `89dfef84` | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) `18d55bfd` | RedHatAI NVFP4 checkpoint (compressed-tensors), KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp4-tonyd2wild-2ac4e8dd | tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark `2ac4e8dd` | derived from [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVFP4 experts and offline-converted NVFP4 dense projections, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM v0.1.dev20051+g487ecf187 (their image) | 4 | 262144 | --max-num-seqs 6 | 24.0 GiB |
| tp4-miaai-7cded8e | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks `7cded8e2` | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) `5ab363a8` | EXL3 4 bpw (their kit), KV fp8 | dflash, incoai/GLM-5.3-Flash-DFlash2 `dc77ff1c`, K 4 | vLLM v0.1.dev20051+g487ecf187 (their image) | 4 | 524288 | MAX_NUM_SEQS 4 | 14.0 GiB |
| tp2-ringside | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `69ad74fc` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 2 | 262144 | --max-num-seqs 6 | 11.5 GiB |
| tp2-miaai-7cded8e | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks `7cded8e2` | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) `5ab363a8` | EXL3 4 bpw (their kit), KV fp8 | dflash, incoai/GLM-5.3-Flash-DFlash2 `dc77ff1c`, K 7 | vLLM v0.1.dev20051+g487ecf187 (their image) | 2 | 524288 | MAX_NUM_SEQS 4 | 14.0 GiB |
| tp2-tonyd2wild-9fea48c | tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark `9fea48c1` | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) `18d55bfd` | compressed-tensors NVFP4 (the checkpoint the recipe ships with), KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM v0.1.dev20051+g487ecf187 (their image) | 2 | 262144 | --max-num-seqs 6 | 8.0 GiB |
| tp4-release-nvidia | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `31978478` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp4-previous-run | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `387b3ff3` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp4-previous-profile | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `50e80d44` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp4-ringside-single-run | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `594faefb` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp4-previous-base | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `9e4f793a` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp4-previous-base-weightless | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `13ec0d86` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 4 | 262144 | --max-num-seqs 16 | 32.0 GiB |
| tp2-previous-profile | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `32dd2ab8` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 2 | 262144 | --max-num-seqs 6 | 11.5 GiB |
| tp2-previous-base | othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE `f1771e0a` | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) `423acf37` | NVIDIA NVFP4 checkpoint, KV fp8_e4m3 | dflash, incoai/GLM-5.3-Flash-DFlash2 `7d74cdd8`, K 7 | vLLM 0.29.0+glm53.local (vLLM v0.29.0 with the GLM-5.3 backport, b12x 1.3.0) | 2 | 262144 | --max-num-seqs 6 | 11.5 GiB |
| tp2-miaai-71962cd-long-coding | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks `71962cd1` | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) `5ab363a8` | EXL3 4 bpw (their kit), KV fp8 | dflash, incoai/GLM-5.3-Flash-DFlash2 `dc77ff1c`, K 7 | vLLM v0.1.dev20051+g487ecf187 (their image) | 2 | 262144 | MAX_NUM_SEQS 2 | automatic (GPU_MEM_UTIL 0.865) |

## KDA checkpoints against the TP4 base before them

Measured on 2026-09-24 between 12:48 and 13:30: the TP4 base without them (lab plan `9e4f793a`) and then the
KDA-checkpoint plan (lab plan `594faefb`, the code of the previous TP4 profile `50e80d44`), back to back in one session
(single samples). The final TP4 profile keeps the KDA checkpoints.

| measurement | without (lab plan 9e4f793a) | with KDA checkpoints (lab plan 594faefb) | change |
|---|---:|---:|---:|
| cold TTFT 8K | 2.062 s | 1.884 s | -8.63 % |
| cold TTFT 32K | 6.919 s | 6.569 s | -5.06 % |
| cold TTFT 64K | 13.631 s | 13.284 s | -2.55 % |
| replay TTFT 8K / 32K / 64K | 0.556 / 0.880 / 1.104 s | 0.551 / 0.729 / 0.919 s | -0.90 % / -17.16 % / -16.76 % |
| decode step time, 28 fixed (streams, K) cells | | | median -0.10 %, range -2.07 to +2.69 % |
| speculative acceptance (window-wide) | 0.3968 | 0.4008 | +0.0040 |

A cached-prefix check compared replayed and extended prompts with fresh ones token by token (a difference whose top-2
logprob gap in the reference is below 0.5 counts as a near-tie). Differences outside near-ties: with KDA checkpoints 1
(8K replay against cold, token 3, gap 0.875); without 3 (8K replay against cold, token 5, gap 1.75; 32K replay against
cold, token 13, gap 0.625; 64K extend against fresh, token 0, gap 1.25). The cause is not established. Under the lab's
rule as fixed before the window, the KDA-checkpoint plan did not pass clause 4 (cached-prefix regression: no mismatch
outside near-ties); it passed a revised clause (no more mismatches than the fresh reference, no gap above 2.0) that
the lab fixed before seeing that plan's cache-check data. Record: `evidence/kda-checkpoints-tp4.json`.

## Records

| file | what it holds |
|---|---|
| `results.json` | every number of the tables, with the lab window (`lab_receipt`), lab plan and number of runs of each row |
| `runs/2026-09-26-ringside-redhat-rowsplit-tp4-1/` | the release window: RigMark's JSON receipts and cards (2 runs; RigMark did not render the `code-full` card, the folder's `README.md` says why); llama-benchy did not run on this plan |
| `runs/2026-09-25-tp4-final-2/` | the final TP4 run before the release (lab plan `387b3ff3`): RigMark's JSON receipts and cards (2 runs), and llama-benchy's results |
| `runs/2026-09-25-ringside-combined-tp4-1/` | the release plan on the NVIDIA checkpoint (lab plan `31978478`): RigMark's JSON receipts and cards (3 runs; RigMark did not render the `code-full` card, the folder's `README.md` says why), and llama-benchy's results (shown on this page) |
| `evidence/release-window-tp4.json` | the release window: plan, context, times, the switch, served-byte and marker receipts, and the correctness checks |
| `runs/2026-09-24-rigmark5-tp4-1/` | the previous TP4 profile's 5-run window: RigMark's JSON receipts and cards, and llama-benchy's results |
| `evidence/tool-eval-bench.json` | tool-eval-bench: per run the score, scenario outcomes, median turn time and settings |
| `evidence/comparison-arms.json` | per row: repository and commit, checkpoint, quantisation, speculative decoding, image, runtime, parallel layout, context, admission limit, KV cache and every recorded deviation, each with its lab source |
| `evidence/kda-checkpoints-tp4.json` | the KDA-checkpoint comparison: per-cell decode step times, TTFT, acceptance, the cached-prefix check and the rule verdicts |
| `evidence/tp2-weakspots-promotion.json` | the TP2 promotion: every clause of the rule with its threshold, measured value and result |
| `evidence/tp2-row-69ad74fc.json` | the cells of the RiNGSiDE TP2 row (2 runs) and the information comparison with the previous profile's stored windows |
| `evidence/tp2-row-32dd2ab8.json` | the cells of the previous TP2 profile's row, with all three staggered rounds |
| `evidence/replay-matrix.json` | the replay matrix: fixture, commands, latency target, sweep and per-window results (protocol text: `../protocols/README.md`) |
| `evidence/temperature-drift.json` | same-plan repeat windows and the window shifts behind the room-temperature caveat |
