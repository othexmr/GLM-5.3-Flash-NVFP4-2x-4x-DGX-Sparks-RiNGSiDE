# Current state

Updated 2026-09-26. Published as the RiNGSiDE release; the TP2 profile's window and the first build check of
`docker/Dockerfile` follow as updates of this repository.

## Profiles

| | TP4 | TP2 |
|---|---|---|
| Lab plan (sha256) | `89dfef8440c6318f0d34a3d767cb2ce11a1946715085ac60cf890802b38b9b76` | `b80911ecfd59c25fa8a711d974f1734cd990fc5e26ec3ea0d355f1d95397f29d` |
| What it is | the RiNGSiDE release plan (next section): the release plan `23ebce61` (lab plan `6f490797` with change notices and comment fixes in the served files and the DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4 code) with 10 served changes. Lab plan `6f490797` is the previous TP4 profile (lab plan `50e80d44`: the composed lab changes and KDA checkpoints saved inside a prefill chunk) plus the KDA recurrent state stored in fp16 with each head's 8 slowest key channels in fp32 (A2b, a numerics change of the stored state; compute stays fp32) and its replay fix, greedy argmax verification of temperature-0 batches, the startup warm-up v3 (17 prompts), the replay boundary v3 (one complete hybrid state at the latest position a DFlash2 replay can hit, at most 16 kept) with the admit fix, and P1 first-token repay | the RiNGSiDE release plan for TP2 (next sections): lab plan `69ad74fc` with change notices in the served files and the DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4 code, and 5 correctness fixes: the exact-landing column of the RecoverSSM state write (the fused commit stays off); the kpool tail ring sized for speculative decoding (vllm-project/vllm#58454); the kpool tail group left out of the generic slot mapping (an out-of-bounds read at long contexts); the exact top-p fallback and the UTF-8 continuation guard, mask only (`GLM53_TOPP_FIX=1`, `GLM53_UTF8_GUARD=1`); the sampler kernels' tile argmax clamped to the vocabulary (vllm-project/vllm#50843); served with the RedHatAI/GLM-5.3-Flash-NVFP4 checkpoint (revision `18d55bfd`) with NVIDIA's `tokenizer.json` and top-p 0.95 as its generation default and a KV cache pool 1 GiB smaller per rank (`--kv-cache-memory 11274289152`, 10.5 GiB) as the RedHat checkpoint's memory margin; none of them served in a TP2 window yet; its TP2 window follows the release. Lab plan `69ad74fc` is the previous TP2 profile (lab plan `32dd2ab8`: the TP2 record plan with tile-major MoE, the TP2 ports of the TP4 kernels and RecoverSSM, the startup warm-up, and the weak-spot fixes ported from TP4) plus the startup warm-up v3 (six prompts), the replay boundary v3 with the admit fix (at most 6 states kept) and P1 first-token repay; neither the fp16 KDA state nor the greedy argmax verification |
| Ranks, slots, context | 4 ranks, 16 requests, 262,144 tokens | 2 ranks, 6 requests, 262,144 tokens |
| Determinism | NON-DETERMINISTIC (atomic MoE accumulation in prefill and in decode launches from 10 rows) | NON-DETERMINISTIC prefill (atomic MoE accumulation in prefill; decode deterministic) |
| Served files | 30 patches, 38 files in `src/tp4`, 8 native or third-party artefacts (65 mounted paths) | 24 patches, 17 files in `src/tp2`, 1 native artefact (44 mounted paths) |
| Results | the release window of this plan (2026-09-26 00:36-01:50: RigMark with 2 runs per cell, prefill 8K-256K, 1-16 streams, staggered arrivals 2-16), tool-eval-bench (2 rounds) and sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol, 2 rounds) in the same boot, then the Korean prompt 1 x 12 at width 6 (12 of 12 runs without a long-generation break, 0 with isolated flagged characters); the emoji stress test and llama-benchy did not run on this plan (llama-benchy ran on the same release plan with the NVIDIA checkpoint: an additional arm); the final TP4 run of lab plan `387b3ff3` is an additional arm, and so is the release plan's run on the NVIDIA checkpoint. The replay matrix was not run on this plan | the final TP2 run of lab plan `69ad74fc`, before the release's changes and fixes (2026-09-25 02:27-02:52: RigMark with 2 runs per cell, prefill 8K-64K, 1-6 streams, staggered arrivals 2-6), one window **without a fresh reference run** (owner's decision, 02:1x), compared with the previous profile's stored windows as information only. sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) in a separate window of the same plan (02:55-03:05); the replay matrix was not run on it. No TP2 window has served the release plan |

Both profiles run on the same base image, `sha256:017fd0ba0a265dd81b78b352b14d33df03e9996f8f45870d4c3df84e08b1ebcb`.
`docker/Dockerfile` rebuilds it with the recipe's DFlash2 code in `model.py` and `kv_cache_utils.py` (Open items).
Four of the recipe's own files are served with the SPDX line "Apache-2.0" where the lab plans serve the line of the
recipe's earlier licence (`glm53_speedup/adaptive_chunk.py` of both profiles, the TP2 profile's `_tp2_cluster.py` and
`tp2_geometry.py`): a comment-only change, recorded in `sources/installed-files.json` and included in both launch
digests (`licenses/README.md`).

## TP4 RiNGSiDE release plan (2026-09-25)

- Lab plan `89dfef84` is the release plan `23ebce61` with 10 changes, on every rank, and nothing else
  (`launch/profiles/tp4/profile.json`, `source_plan.derived_from`, one step per change):
  1. The fused RecoverSSM commit back on (GLM53_RECOVERSSM_FUSED_COMMIT=1: the accepted KDA state is committed by the
     next verify step) with its ordering fix (the commit's rule 1 at commit time, its two launches skipped when the
     host proves that no row is within 8 tokens of a block end; rule 2 at step start, a flush of the requests absent
     from the step) and the exact-landing column of the state write (recoverssm.py, kda.py, model_runner.py).
  2. The KDA FP8 handoff projects through the tuned SM120 FP8 GEMM configurations of the dense FP8 module instead of
     the generic scaled matrix multiply (GLM53_KDA_FP8_HANDOFF_TUNED=1, from 1,536 rows per call).
  3. The MLA q_a/kv_a RMSNorm writes q_b_proj's per-token FP8 quantization itself and q_b_proj hands it on to the
     indexer (GLM53_MLA_QNORM_FP8=2 with GLM53_DENSE_SHARED_QUANT; a new module,
     vllm/models/common/ops/glm53_qnorm_fp8.py, adapted from vLLM's fused q/kv RMSNorm kernel); a layer that misses
     the hand-over raises instead of recomputing.
  4. The sparse-attention indexer's decode top-k through a rowspread build of the persistent top-k kernel
     (GLM53_TOPK_ROWSPREAD=1; a new native extension, nvfp4_topk_rowspread, in which every row at or below the radix
     threshold is taken by one CTA of the whole grid) and decode logits cdiv(max_model_len, 4) columns wide instead of
     max_model_len (GLM53_INDEXER_POOL_LOGITS=1). Credit: vllm-project/vllm#55314 by Dovis01 (the persistent top-k
     hunks the extension is built from; concept from sgl-project/sglang#37625 by ormandj and bold84).
  5. An exact top-p fallback in the Triton top-k/top-p sampler for every row whose estimated normaliser, search exit
     or reconstructed cutoff cannot guarantee the cut (GLM53_TOPP_FIX=1), and a UTF-8 continuation guard for sampled
     requests that masks every token which cannot continue an incomplete UTF-8 character (GLM53_UTF8_GUARD=1 with the
     model's tokenizer; mask only: GLM53_UTF8_GUARD_FOLD, GLM53_UTF8_GUARD_GREEDY and GLM53_SAMPLER_AUDIT unset).
  6. The indexer's kpool tail ring sized for speculative decoding: index_kpool x cdiv(index_kpool + K, index_kpool)
     slots per request (12 at K 7) instead of index_kpool, so the rows a verify step stashes no longer overwrite the
     open pool's committed keys (a port of vllm-project/vllm#58454, always on). Credit: vllm-project/vllm#58454 by
     mmastrac, building on vllm-project/vllm#55219 by ivanium; Morrowmake/vllm-cmp170hx 1d4b59e, the port this
     follows.
  7. The tile argmax of the gumbel, greedy-statistics and resampling kernels clamped to the vocabulary (a port of
     vllm-project/vllm#50843). Credit: vllm-project/vllm#50843 by alexbi29.
  8. The model runner leaves the kpool tail group (KpoolTailSpec) out of the generic slot mapping, as it does for
     circular-buffer groups, so the generic kernel no longer reads past the group's 32-column block table (up to about
     350 KB past it at 1,048,576 tokens with the 12-slot ring); the kpool tail metadata builder writes the tail slots
     itself, as before (model_runner.py).
  9. The checkpoint RedHatAI/GLM-5.3-Flash-NVFP4 (revision 18d55bfd, NVFP4 weights and activations in the
     compressed-tensors format) in place of nvidia/GLM-5.3-Flash-NVFP4 (SWITCHLESS_MODEL_DIR), with NVIDIA's
     tokenizer.json (revision 423acf37) mounted over its own (SWITCHLESS_TOKENIZER_FILE; the two files differ only in
     a truncation block of at most 2,048 tokens) and --override-generation-config {"top_p":0.95}, NVIDIA's generation
     default that RedHat's generation_config.json lacks; the API name stays nvidia/GLM-5.3-Flash-NVFP4.
  10. The prefill indexer's query rows split across the four ranks (GLM53_INDEXER_ROW_SPLIT=1): for a prefill chunk of
     at least 1,024 query rows over at least 4,096 compressed key positions, each rank scores and selects the top-k
     pools of its own contiguous 64-row tiles with the served kernels and the ranks all-gather the int32 pool ids,
     where every rank used to run the whole chunk; the gathered selection is the full-row selection byte for byte
     (eager prefill only; the recipe's own module glm53_indexer_rowsplit.py, newly mounted;
     sparse_attn_indexer_kpool.py takes the new branch only with the switch on). Credit: the idea comes from rhys101's
     SG18 native prefill TP split (rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8), adapted to TP4 by knapcio
     (knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4); glm53_indexer_rowsplit.py is the recipe's own implementation, no
     code copied.
- The release plan `23ebce61` is lab plan `6f490797` (next section) with the RiNGSiDE change notices and comment fixes
  in 38 served files and directories (the recipe's Apache-2.0 4(b) notice in every modified third-party file, comments
  that named an AI model or the lab's agents reworded; comment lines and docstring text only) and the DFlash2
  auxiliary hidden-state capture (`model.py`, adapted from vLLM's `DeepseekV4Model.forward`) and drafter KV-cache
  group (`vllm/v1/core/kv_cache_utils.py`, mounted over the image's file), written from vLLM's DeepSeek-V4 code
  (`source_plan.derived_from.base_derived_from`). Its third change, `GLM53_RECOVERSSM_FUSED_COMMIT=0`, is superseded
  by the fused commit's fix (change 1 above).
- Checks (`source_plan.qualification`): staging: every bind source of every rank read back with the planned sha256 on
  its host, the package directories file by file (2026-09-26 00:30-00:31); static checks of the combined build (CPU)
  of the plan 31978478 that this variant derives from: 23 of 23 pass: the plan rebuilds byte for byte; every
  component's own patches give its tested bytes; the three-way merges onto the release bytes reproduce every served
  file; the served code is token-equal to the tested code (docstring text aside); every file compiles; the plan
  differs from the release plan only in the declared mounts and environment edits (2026-09-25 20:09); static checks of
  the redhat-rowsplit on redhat variant (CPU): 6 of 6 pass: the plan rebuilds byte for byte, and its declared edits
  undo to the plan 31978478 on every rank (2026-09-26 00:28); GPU leaf, the real kernels, TP4 geometry: PASS: 26 cases
  at the 2,304-token block, strict raw-bit equality of the committed states with poisoned record tails; the fix with
  its host skip (A2 + B) equal to the fix without it (A + B) at every step of every case (2026-09-25 16:54-16:59 and
  18:47-18:54); serving window of lab plan 6f490797 with the ordering fix and the exact-landing column (patch A + B,
  fused commit on): Korean prompt 1 x 72 at width 6: 72 of 72 runs complete without a long-generation break, 3 of them
  with isolated flagged characters (2026-09-25, window ending 18:17); GPU leaf, one GB10: PASS: 30 of 30 comparisons
  bitwise equal to the served path at 20 row counts, every call routed as expected (2026-09-25 18:22); GPU leaf, one
  GB10: PASS in mode 2: 30 of 30 comparisons bitwise equal to the served path at 9 row counts (the kv output, the
  handed FP8 values and scales, both GEMMs), hand-over and relay exact (2026-09-25 18:22); GPU leaf, one GB10: PASS:
  the canonical top-k output equal to the served selector's in 322 cases, with the pool-width logits too; 28 to 74 %
  less time per call from 4 rows up at the kernel level (a leaf timing, not a serving measurement) (2026-09-25
  18:18-18:22); overlay leaf of the candidate plan's files in the serving image: exited 0 (2026-09-25 18:24); GPU
  leaf, top-p fix: PASS: exact in every case of 38 case families (1,000 and more rows): every row exact or a boundary
  decision within 3.2e-6 of mass; where the image's kernel was right the fixed masks equal its masks in every row
  (2026-09-25 18:25-18:26); GPU leaves, UTF-8 guard: PASS: the continuation table equal to Python's strict UTF-8
  decoder in all 11,151,360 token-by-state pairs; in verification with drafts 0 invalid tokens at rows inside a
  character with the guard on, against 5 of 6 without it; with the switches unset the hooked sampler files are
  bit-identical to the image's over 12 verify steps (2026-09-25 18:18 and 18:25); GPU leaf, the served kernels through
  the image's own slot-mapping code: PASS: 84 case runs at K 1 to 7, half with the old ring and half with the fixed
  one, each with a second turn through the prefix cache: 19,950 wrong pooled keys with the old ring (in 31 cases;
  4,525 of them inherited through the prefix cache), 0 with the fixed ring; in a case shaped like the Korean test the
  old ring gave a wrong key to 63 to 70 % of the decode-built pools, the fixed ring to none; every control clean
  (2026-09-25 18:57); GPU leaf (the kpool ring leaf's sampler rows): PASS: the old kernels emitted the
  out-of-vocabulary tile-local id 154,880 46 times on rows with NaN in the last tile, the clamped kernels never
  (2026-09-25 18:57); CPU proof, the image's slot-mapping kernel source over a guarded table: PASS (5 of 5): 76
  batches with positions up to 1,048,575: the old mapping reads past the table in every batch kind, the new one never;
  the tail group's final mapping and the other six groups' mappings identical (2026-09-25 19:5x); GPU leaf, the
  image's own slot-mapping kernel with a sentinel after the tail table: PASS: 320 batches with positions up to
  1,048,575: 492,150 tail slots computed from memory past the table with the old mapping, 0 with the new; the final
  tail mapping and the other six groups identical (2026-09-25 20:07); GPU leaf, the served kernels on one GB10 (four
  threads as four ranks): PASS: in all 10 cases (context 0 to 248K, a 4,608-row chunk, ties, a ragged three-request
  chunk, short rows) every rank's gathered pool ids and the expanded token indices equal the full-row selection byte
  for byte; each rank engaged once and check mode counted 0 differing rows; on the long single-request cases the
  busiest rank's slice took 0.25 to 0.27 of the full-row indexer time (the rule: at most 0.40) (2026-09-26
  00:06-00:07); static checks (CPU): 16 of 16 pass: the overlay rebuilds from the pinned base; the overlay minus its
  additions is the served file; partition properties over 5 world sizes, 3 tile sizes and about 300 row counts; four
  simulated ranks give the full-row selection byte for byte, and a wrong gather order or rebased key bounds are caught
  (2026-09-26 00:09); static proofs (CPU): each changed file keeps every original line in order, a rewritten line in
  its place, and every inserted line is a comment; Python token streams without comments equal the base file's, except
  the docstring text of the 12 files listed in the derivation (there the syntax trees without docstrings are equal and
  nothing reads __doc__); the C and CUDA files' comment-stripped tokens are equal; every file compiles; the plan
  builder rebuilds the plan byte for byte and 14 of 14 CPU tests pass (2026-09-25 15:07); CPU test vectors: all pass:
  the 17 KV-cache grouping cases and the two served layout records (F3, F4); the capabilities, the layer ids and 19 of
  19 capture replays (F2) (2026-09-25 12:37); GPU leaf, one GB10, the TP4 plan's container and configuration: PASS:
  3,358 comparisons of the captured auxiliary hidden states and final hidden states between the earlier and the served
  model.py at tolerance 0, eager and CUDA-graph replays, 8 token counts, 3 layer-id sets, 0 failed; the KV-cache
  groups and layout of all 15 TP4 cases equal between the image's kv_cache_utils.py and the served one, including the
  served layout record over four workers (2026-09-25 15:48-15:56); ownership leaf on four nodes, TP4: PASS on all four
  ranks: 567 comparisons per rank at tolerance 0, 0 failed, with the token-sharded mHC prefill ownership and its
  gathers (2026-09-25 15:56-15:57); release window of this plan (one boot, four nodes): Korean prompt 1 x 12 at width
  6: 12 of 12 runs complete without a long-generation break, no isolated flagged character; switch, served-byte and
  marker receipts ok (2026-09-26 00:36-01:50); serving window of the same files with another switch set (folding,
  greedy rows and the sampler audit on), four nodes: not the served configuration: its scorer's runtime failures were
  the boot import tracebacks that every window of this image logs; the emoji test's malformed flags were answers cut
  at the 2,000-token cap; the 12 U+FFFD of its CJK stress test were literal replacement characters that the fold wrote
  into words (a window without the guard had none), which supports serving the guard without the fold; its Korean
  smoke test gave 36 of 36 clean runs. The served setting is measured by the release window (2026-09-25, ending
  19:57).
- Not established: the speed cells are medians of one boot's runs, with no matched reference window (earlier windows
  are other boots and times of day); the emoji/CJK wall stress test and llama-benchy were not run on this plan
  (llama-benchy ran on the same release plan with the NVIDIA checkpoint, lab plan 31978478, in window
  2026-09-25-ringside-combined-tp4-1: the additional arm); the fused commit's host skip is exact only under
  synchronous scheduling without adaptive verification (the profile runs --no-async-scheduling); the bit identity of
  the MLA q_a norm hand-over depends on the Triton layout choice proven for Triton 3.7.1 in this image.

## TP4 base plan `6f490797` (2026-09-25)

- Lineage (each step the owner's decision on the lab's measurements): `50e80d44` + A2b (`GLM53_KDA_STATE=fp16-slow32`)
  and greedy argmax verification (`GLM53_GREEDY_ARGMAX_VERIFY=1`), + the startup warm-up v3 (`entry.py` `9dd85a4c`,
  `GLM53_STARTUP_PREFILL_WARMUP=17` with a list of prompt lengths) and the replay boundary v3
  (`GLM53_REPLAY_BOUNDARY=1`, `GLM53_REPLAY_BOUNDARY_MAX=16`), + the A2b replay fix (`kda.py` `11c6fde6`,
  `kda_state_store.py` `662df244`) and P1 first-token repay (`glm53_speedup/cadence.py` `c510028b`,
  `GLM53_SPEEDUP_FIRST_TOKEN_REPAY=1`): lab plan `387b3ff3`, measured in the final TP4 run; + the replay-boundary admit
  fix (`glm53_replay_boundary.py` `d41662b1` instead of `144c7d34`): lab plan `6f490797`, the TP4 profile until the
  release plan.
- Served files that changed: three new files in `src/tp4` (`glm53_greedy_verify.py`, `glm53_replay_boundary.py`,
  `vllm/models/glm5next/nvidia/kda_state_store.py`), five patches (`entry.py`, `kda.py`, `v1/core/sched/scheduler.py`,
  `v1/core/single_type_kv_cache_manager.py`, `v1/worker/gpu/model_runner.py`) and three files in full
  (`glm53_speedup/cadence.py`, `glm53_kda_ckpt_plan.py`, `ops/recoverssm.py`). No native artefact changed. Every served
  sha256 was read back on all four hosts by the lab's staging job of this plan (2026-09-25 01:40).
- Results (`bench/results/README.md`): the RiNGSiDE TP4 row is the final TP4 run, lab plan `387b3ff3`, with 2 runs per
  RigMark cell (the previous profile's row, with 5 runs, is now an additional arm). The admit fix changes only the
  handling of a prefix hit where the image's own cache entry and a replay boundary share a hash: in the final run the
  module refused five valid hits at the 1,152-token boundary during tool-eval-bench (none during RigMark) and those
  requests were recomputed. The lab did not re-run RigMark for the fix, judging that RigMark's prompts do not reach that
  path; its check window (2026-09-25 01:41-01:54) logged no refused hit on any rank, the cache check found the predicted
  hits with 2 differences from fresh prefills, and tool-eval-bench scored 93 as in the final run. The row's sparkDash
  cells come from a separate window of lab plan `6f490797` (2026-09-25 03:40-03:48, code at one stream
  150.4 tok/s), marked with that plan.
- tool-eval-bench (SeraphimSerapis/tool-eval-bench `5381d589`, 69 scenarios, temperature 0, seed 42, single runs):
  the final TP4 run 93 (129/138, median turn 5.0 s), the check window of `6f490797` 93 (129/138, 5.4 s),
  MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks `7cded8e` on TP4 94 (130/138, 8.7 s); the same three scenarios failed in
  every run.

## TP2 RiNGSiDE release plan (2026-09-25)

- Lab plan `b80911ec` is the TP2 release plan `33a8cc9e` with 7 changes (5 correctness fixes, the checkpoint and its
  argument change), on both ranks, and nothing else (`launch/profiles/tp2/profile.json`, `source_plan.derived_from`):
  1. The RecoverSSM commit's final column and the image helper's running column are cdiv(nc', block) - 1, so a verify
     window accepted in full that ends exactly on a 4,608-token block boundary keeps its state in the completed block,
     where the earlier code wrote it to an unallocated column; the fused commit stays off
     (GLM53_RECOVERSSM_FUSED_COMMIT unset; recoverssm.py, kda.py).
  2. The indexer's kpool tail ring sized for speculative decoding: index_kpool x cdiv(index_kpool + K, index_kpool)
     slots per request (12 at K 7) instead of index_kpool, so the rows a verify step stashes no longer overwrite the
     open pool's committed keys (a port of vllm-project/vllm#58454, always on; the TP2 attention.py and
     kpool_compress.py). Credit: vllm-project/vllm#58454 by mmastrac, building on vllm-project/vllm#55219 by ivanium;
     Morrowmake/vllm-cmp170hx 1d4b59e, the port this follows.
  3. The model runner leaves the kpool tail group (KpoolTailSpec) out of the generic slot mapping, as it does for
     circular-buffer groups, so the generic kernel no longer reads past the group's 32-column block table on the steps
     past 32 x ring tokens; the kpool tail metadata builder writes the tail slots itself, as before (model_runner.py).
  4. An exact top-p fallback in the Triton top-k/top-p sampler for every row whose estimated normaliser, search exit
     or reconstructed cutoff cannot guarantee the cut (GLM53_TOPP_FIX=1), and a UTF-8 continuation guard for sampled
     requests that masks every token which cannot continue an incomplete UTF-8 character (GLM53_UTF8_GUARD=1 with the
     model's tokenizer; mask only: GLM53_UTF8_GUARD_FOLD, GLM53_UTF8_GUARD_GREEDY and GLM53_SAMPLER_AUDIT unset).
  5. The tile argmax of the gumbel, greedy-statistics and resampling kernels clamped to the vocabulary (a port of
     vllm-project/vllm#50843). Credit: vllm-project/vllm#50843 by alexbi29.
  6. The checkpoint RedHatAI/GLM-5.3-Flash-NVFP4 (revision 18d55bfd, NVFP4 weights and activations in the
     compressed-tensors format) in place of nvidia/GLM-5.3-Flash-NVFP4 (SWITCHLESS_MODEL_DIR), with NVIDIA's
     tokenizer.json (revision 423acf37) mounted over its own (SWITCHLESS_TOKENIZER_FILE; the two files differ only in
     a truncation block of at most 2,048 tokens) and --override-generation-config {"top_p":0.95}, NVIDIA's generation
     default that RedHat's generation_config.json lacks; the API name stays nvidia/GLM-5.3-Flash-NVFP4.
  7. --kv-cache-memory 11274289152 on both ranks (was 12348030976): a KV cache pool 1 GiB smaller per rank (10.5 GiB
     instead of 11.5 GiB), as the memory margin of the RedHat checkpoint, whose dense layers 0-2 are estimated to take
     0.2-0.3 GB more per rank at TP2; with the NVIDIA checkpoint the TP2 nodes kept 0.3-1.0 GiB of host memory
     available while serving.
- The TP2 release plan `33a8cc9e` is lab plan `69ad74fc` (next section) with the RiNGSiDE change notices and comment
  fixes in the served files (the recipe's Apache-2.0 4(b) notice in every modified third-party file, comments that
  named an AI model or the lab's agents reworded; comment lines and docstring text only) and the DFlash2 capture
  (`model.py`) and drafter KV-cache group (`kv_cache_utils.py`, now mounted over the image's file) of the TP4 profile.
  TP2 already writes RecoverSSM's accepted state after sampling (`GLM53_RECOVERSSM_FUSED_COMMIT` unset).
- Checks (`source_plan.qualification`): staging: every bind source of every rank read back with the planned sha256 on
  its host, the package directories file by file (2026-09-26 03:19-03:20); static checks of the combined build (CPU)
  of the plan 3938cc08 that this variant derives from: 11 of 11 pass: the plan rebuilds byte for byte; every
  component's own patches give its tested bytes; the three-way merges onto the release bytes reproduce every served
  file; the served code is token-equal to the tested code (docstring text aside); every file compiles; the plan
  differs from the release plan only in the declared mounts and environment edits (2026-09-25 21:10); static checks of
  the redhat variant (CPU): 6 of 6 pass: the plan rebuilds byte for byte, and its declared edits undo to the plan
  3938cc08 on every rank (2026-09-26 03:18); GPU leaf, the real kernels, TP2 geometry (32 heads, fp32 state,
  4,608-token block, at most 6 rows): PASS: the fixed state write never breaks continuity in 62,749 checks, every
  exact landing, the next pre-copy and the next verify included; the earlier code breaks it after exact landings
  (2026-09-25 18:24-18:26); GPU leaf, the served kernels through the image's own slot-mapping code: PASS: 84 case runs
  at K 1 to 7, half with the old ring and half with the fixed one, each with a second turn through the prefix cache:
  19,950 wrong pooled keys with the old ring (in 31 cases; 4,525 of them inherited through the prefix cache), 0 with
  the fixed ring; in a case shaped like the Korean test the old ring gave a wrong key to 63 to 70 % of the
  decode-built pools, the fixed ring to none; every control clean (2026-09-25 18:57); CPU proof, the image's
  slot-mapping kernel source over a guarded table: PASS (5 of 5): 76 batches with positions up to 1,048,575: the old
  mapping reads past the table in every batch kind, the new one never; the tail group's final mapping and the other
  six groups' mappings identical (2026-09-25 19:5x); GPU leaf, the image's own slot-mapping kernel with a sentinel
  after the tail table: PASS: 320 batches with positions up to 1,048,575: 492,150 tail slots computed from memory past
  the table with the old mapping, 0 with the new; the final tail mapping and the other six groups identical
  (2026-09-25 20:07); GPU leaf, top-p fix: PASS: exact in every case of 38 case families (1,000 and more rows): every
  row exact or a boundary decision within 3.2e-6 of mass; where the image's kernel was right the fixed masks equal its
  masks in every row (2026-09-25 18:25-18:26); GPU leaves, UTF-8 guard: PASS: the continuation table equal to Python's
  strict UTF-8 decoder in all 11,151,360 token-by-state pairs; in verification with drafts 0 invalid tokens at rows
  inside a character with the guard on, against 5 of 6 without it; with the switches unset the hooked sampler files
  are bit-identical to the image's over 12 verify steps (2026-09-25 18:18 and 18:25); GPU leaf (the kpool ring leaf's
  sampler rows): PASS: the old kernels emitted the out-of-vocabulary tile-local id 154,880 46 times on rows with NaN
  in the last tile, the clamped kernels never (2026-09-25 18:57); static proofs (CPU): each changed file keeps every
  original line in order and every inserted line is a comment; token streams without comments equal the base file's
  (docstring text aside); every file compiles; the release lane's CPU checks 14 of 14 pass for this plan (2026-09-25
  15:07); CPU test vectors: all pass for both profiles' cases: the KV-cache grouping cases, the served layout records
  and the capture replays (2026-09-25 12:37); GPU comparison with the earlier code in the TP2 configuration (one DGX
  Spark): 3,358 of 3,358 comparisons with the earlier code equal (the captured auxiliary hidden states, the final
  hidden states, eager and CUDA-graph replays, bit for bit); the KV-cache groups and layout equal; the served model.py
  and kv_cache_utils.py bytes checked inside the container; lab job 266d4c (2026-09-26 02:49-02:57).
- Not established: no serving window of this plan: the TP2 results are those of lab plan 69ad74fc, measured before
  these changes; each fix is verified by GPU leaves and CPU checks only.

## TP2 base plan `69ad74fc` (2026-09-25)

- Lineage: `32dd2ab8` (promoted by the owner on 2026-09-24 although three guard clauses of the lab's rule failed) + the
  startup warm-up v3 (`GLM53_STARTUP_PREFILL_WARMUP=6`, lab plan `558eae7a`) + the replay boundary v3
  (`GLM53_REPLAY_BOUNDARY=1`, `GLM53_REPLAY_BOUNDARY_MAX=6`, lab plan `fe5fefca`) + P1 first-token repay (lab plan
  `008475b3`) + the replay-boundary admit fix: lab plan `69ad74fc`, the TP2 profile until the release plan, measured directly in its final run.
- Served files that changed: one new file in `src/tp2` (`glm53_replay_boundary.py`, the same bytes as in TP4), one new
  patch (`v1/core/single_type_kv_cache_manager.py`), two changed patches (`entry.py`, `v1/core/sched/scheduler.py`) and
  `glm53_speedup/cadence.py` (the TP4 bytes). No native artefact changed. Every served sha256 was read back on both
  hosts by the lab's staging job of this plan (2026-09-25 01:54).
- Results (`bench/results/README.md`): the RiNGSiDE TP2 row is the final TP2 run (2026-09-25 02:21-02:55, one boot, 2
  runs per RigMark cell), with the sparkDash cells of a separate window of the same plan (2026-09-25 02:55-03:05, every
  stream ok); the previous profile's row, with its three staggered rounds and replay matrix, is now an additional arm.
  **The final TP2 run had no fresh-reference comparison**: the owner decided at 02:1x not to run a reference window back
  to back. The lab's information-only comparison with the previous profile's stored windows (other boots, times of day
  and run counts; speed ratio per cell, geometric means): overall +15.8 % over 21 cells, cold TTFT +5.5 %, warm replay
  +145.7 %, code aggregate +1.0 %, prose aggregate +2.6 %, staggered prefill-first newcomer +0.6 %. No rule and no
  verdict go with it. Replay-boundary and P1 markers were present with no refused hit; the cache check found the
  predicted hits with 3 differences from fresh prefills and 1 near-tie.

## Change notices (Apache-2.0 section 4(b))

Both profiles carry the recipe's notice inside every modified third-party file they serve ("Modified by the GLM-5.3
RiNGSiDE recipe (othexmr): ..."), one line per change, below the file's SPDX lines and any earlier notice. The served
bytes are pinned, so a notice needs a new plan that is staged and served.

The lab's own "Modified by the GLM-5.3 lab" comments stay in the TP4 patches of `kda.py`, `v1/core/sched/scheduler.py`,
`v1/core/single_type_kv_cache_manager.py` and `v1/worker/gpu/model_runner.py` and in `glm53_kda_ckpt.py` and
`glm53_kda_ckpt_plan.py` (followed there by the recipe's notice), and in the TP2 patches of `v1/core/sched/scheduler.py`
and `v1/core/single_type_kv_cache_manager.py`; they name the lab rather than the recipe. For the GLM-5.3 backport files
of the image, a notice describes the served patch only, not the lab's earlier port changes baked into the image file.
`NOTICE` states the modification for all of them.

## Open items

- The DFlash2 auxiliary hidden-state capture and the drafter KV-cache group: both profiles serve the recipe's code,
  written from vLLM's DeepSeek-V4 code (the release sections above), and the base-image stage inputs carry it too
  (`docker/base`: stage 02's three DFlash2 patches and their oracle, the `model.py` and `kv_cache_utils.py` sections
  of the v0.29.0 port patch; `docker/base/README.md`). It was compared with the lab's earlier code on a GPU in both
  configurations. A rebuilt base differs from the measured image `017fd0ba` in those two files, so the patches of
  `model.py` and `kv_cache_utils.py` apply to the rebuilt base and not to the measured image. The replacement in
  `docker/base` was checked on a CPU only; its first build on a DGX Spark is open with the rest of
  `docker/Dockerfile`.
- The TP2 profile's release plan was not served in a TP2 window: its results are those of lab plan `69ad74fc`,
  measured before the release's changes and on the NVIDIA checkpoint; its fixes are verified by GPU leaves and CPU
  checks (above); its TP2 window (RigMark, sparkDash, the Korean check) follows the release.
- RiNGSiDE distributes no binaries. `docker/Dockerfile` rebuilds the base image and the native artefacts from pinned
  sources and the lab's stage inputs (`docker/base`, `docs/build-provenance.md`) and bakes one image per profile;
  `launch/compose.py`, `launch/up.sh` and `launch/down.sh` launch it with Docker Compose. The Dockerfile has not been
  built yet: its first build on a DGX Spark and the qualification of the self-built images are open
  (`docker/README.md`).
- The TP4 results are 2 runs of one boot (the release window), the TP2 results 2 runs of one boot (the final TP2 run
  of `69ad74fc`); the previous profiles' rows are kept as additional arms. The replay matrix was not run on the final
  profiles; sparkDash ran in the release window (TP4) and in a separate window of `69ad74fc` (TP2). The final TP2 run
  has no fresh-reference comparison (above).
- Replay boundary, final TP4 run: the cache check found the predicted hits with 1 difference from fresh prefills and 4
  near-ties. P1's own marker check counted the three startup import tracebacks that a fresh reference window also
  logs; measured alone against a fresh reference earlier, P1 missed one median clause of its rule (+0.07 % against at
  most 0.0 %), and the owner made it the default.
- Served files carry comments with lab archive paths ("outputs/..."); they are part of the served bytes.
- The sparse-MLA build patches carry the comment fix of `setup.py` (result trees `442d83de`, `ce997603`); the measured
  extensions are builds of the earlier trees, which differ only in comments, and the patches' change notices keep the
  recipe's earlier name (`build/sparse-mla/README.md`).
- The drafter, incoai/GLM-5.3-Flash-DFlash2, states CC BY-NC-ND 4.0 and is released for research and evaluation
  (`docs/limitations.md`, `NOTICE`).
