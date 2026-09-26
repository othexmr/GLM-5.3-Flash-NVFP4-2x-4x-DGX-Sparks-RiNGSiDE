# Replay matrix (modified RigMark application replay)

The RiNGSiDE profiles only. Every figure is a single run. The records behind this section are in `replay-matrix.json`.

## Client

- othexmr/rigmark, branch `feat/serving-interference-and-replay`, commit `c1b26ecf0724563dcf8a20009b6e169aa87215fd` (protocol `rigmark-application-replay.1`): the same tree (`bf3ce13b`) as the measured commit `78759b10` that every window's replay report records. For each window the measured commit was exported with `git archive` into a fresh directory.
- The headline RigMark cells and staggered arrivals use another client: othexmr/rigmark `40fabcaf` (branch `feat/staggered-arrival`), published with the identical tree (`3ba4ae72`) to the revision the windows record (`6e0e24a6`).

## Requests

- **Authored scenarios, not recorded production traffic.** The request text is built from five files: one code file, one document and three long-context files. The files are the lab's; their hashes identify them:

| role | file | bytes | sha256 |
|---|---|---:|---|
| code | `mhc_prefill_pipeline.py` | 22,440 | `feb8e7b9aa900df4` |
| document | `README.md` | 10,244 | `d5cba53f74209f27` |
| context | `model.py` | 50,715 | `467dc36110df71b7` |
| context | `kda.py` | 49,478 | `e574b6b4c0357a16` |
| context | `recoverssm.py` | 37,876 | `58ad39dcd449f06e` |

- **Quality set:** `examples/replay/quality-16.json` of the client commit (trace sha256 `5a32f113bc7bb00a`): 16 deterministic-answer tasks (arithmetic, code tracing, conversions, counting), arriving 0.25 s apart, 2,048 output tokens each.
- Every window replayed byte-identical traces (same trace sha256 per run).

## Commands

Site values are placeholders: `${RIGMARK}` a checkout of the client commit, `${API_URL}` the server, `${MODEL}` the served model id (read from `/v1/models`), `${OUT}` a fresh output directory, `${NS}` a fresh run namespace, `${CODE}` `${DOC}` `${CTX1..3}` the input files, `${DIRECTION}` one of `short-first`, `long-first`, `sessions`, `${RUN}` one run directory, `${RATES}` `0.05,0.1,0.15,0.2,0.3,0.4` (`0.05,0.1,0.15` in the two TP2 promotion windows).

```sh
# quality set
${RIGMARK}/rigmark replay --trace ${RIGMARK}/examples/replay/quality-16.json --output ${OUT}/quality-16 --run-id ${NS}-quality --delivery-tokens usage --base-url ${API_URL} --model ${MODEL} --identity ${OUT}/identity.json --extra-body '{"chat_template_kwargs": {"reasoning_effort": "low"}}' --timeout 900 --run
# prepare one directional trace (DIRECTION = short-first, long-first, sessions)
${RIGMARK}/rigmark prepare-replay --code ${CODE} --document ${DOC} --context ${CTX1} --context ${CTX2} --context ${CTX3} --users 6 --direction ${DIRECTION} --delay 1 --max-tokens 4096 --long-max-tokens 8192 --followup-max-tokens 2048 --output ${OUT}/${DIRECTION}.json
# run it
${RIGMARK}/rigmark replay --trace ${OUT}/${DIRECTION}.json --output ${OUT}/${DIRECTION} --run-id ${NS}-${DIRECTION} --delivery-tokens usage --base-url ${API_URL} --model ${MODEL} --identity ${OUT}/identity.json --extra-body '{"chat_template_kwargs": {"reasoning_effort": "low"}}' --timeout 900 --run
# arrival-rate sweep
${RIGMARK}/rigmark replay-sweep --code ${CODE} --document ${DOC} --context ${CTX1} --context ${CTX2} --context ${CTX3} --rates ${RATES} --duration 120 --seed 1 --output ${OUT}/sweep --max-tokens 4096 --long-max-tokens 8192 --base-url ${API_URL} --model ${MODEL} --identity ${OUT}/identity.json --extra-body '{"chat_template_kwargs": {"reasoning_effort": "low"}}' --timeout 900 --delivery-tokens usage --slo-visible 15 --slo-gap 3 --slo-total 300 --slo-basis output --target 0.9
# re-verify every run directory from its receipts
${RIGMARK}/rigmark compare-replay ${RUN} ${RUN}
```

## Latency target

- A request meets the target only if it completed, its **first output of any kind** (reasoning or answer text) arrived within **15 s** of its scheduled arrival (client dispatch lag included), **no gap** between two output deltas exceeded **3 s**, and it **finished within 300 s** of its scheduled arrival.
- There is no percentile: each request passes or fails. **Attainment** = passing requests / all planned requests at that rate; failed, truncated and errored requests stay in the denominator.
- A rate **meets the latency target** when attainment is at least **0.9** and the measurement is valid.

## Arrival-rate sweep

- Open loop: a seeded Poisson process (seed 1), one unit-rate exponential sequence scaled by 1/rate over a 120 s window, so every rate replays the same prompts in the same order with the clock compressed.
- Requests per rate: 0.05 req/s: 6, 0.1 req/s: 14, 0.15 req/s: 18, 0.2 req/s: 27, 0.3 req/s: 41, 0.4 req/s: 51.
- Single-turn arrivals cycle through four tasks on the code file and the document; every 6th arrival is the long-context review. Output budgets 4,096 tokens (8,192 for the review). `reasoning_effort` low.
- Rates run in ascending order; the sweep would stop after a rate whose completion fraction fell below 0.5 (none did).
- **Capacity** (the page cell) is the highest swept rate that meets the target with every lower rate also meeting it.

## Quality set

- A task passes only if its request completed and its final line is exactly `ANSWER: <value>`. Duplicate or conflicting ANSWER lines, and truncated or failed requests, count as misses. The page prints passed/16.
- It catches gross output breakage between matched runs; it is not an accuracy benchmark.

## Valid

"Valid" means all three hold for every run of the window: exact token delivery (every finished stream delivered exactly the completion tokens the server reported), a valid client schedule (every request dispatched within 0.05 s of its scheduled time), and receipts that re-verify (`compare-replay`). Speed is compared only between valid runs.

## Results

| window | profile | valid | quality | attainment by rate (req/s) | meets the target up to |
|---|---|---|---|---|---|
| `2026-09-24-final-replay-ours-tp4-1` | TP4 previous base (lab plan 9e4f793a); the page shows it for the TP4 rows | yes | 15/16 | 0.05: 1.00, 0.1: 0.93, 0.15: 1.00, 0.2: 1.00, 0.3: 0.61, 0.4: 0.35 | 0.2 req/s |
| `2026-09-24-final-replay-ours-tp4-weightless-1` | TP4 previous base with the optional weightless GLP-44 steering | yes | 16/16 | 0.05: 1.00, 0.1: 0.93, 0.15: 1.00, 0.2: 0.96, 0.3: 0.63, 0.4: 0.31 | 0.2 req/s |
| `2026-09-24-final-replay-ours-tp2-1` | TP2 profile f1771e0a (the TP2 row until 2026-09-24 16:3x) | yes | 16/16 | 0.05: 0.33, 0.1: 0.07, 0.15: 0.06, 0.2: 0.00, 0.3: 0.00, 0.4: 0.00 | none of the swept rates |
| `2026-09-24-tp2ws-ref-1` | TP2 profile f1771e0a, fresh reference of the 32dd2ab8 promotion (sweep 0.05-0.15 only) | yes | 16/16 | 0.05: 0.33, 0.1: 0.07, 0.15: 0.06 | none of the swept rates |
| `2026-09-24-tp2ws-cand-1` | TP2 profile 32dd2ab8 (sweep 0.05-0.15 only) | yes | 16/16 | 0.05: 0.83, 0.1: 0.50, 0.15: 0.33 | none of the swept rates |

## Caveats

- The TP4 replay window scored quality 15/16: task q05 invalid_answer (expected value present: True, ANSWER line present: False).
- No swept rate meets the latency target in any TP2 window (f1771e0a: attainment 0.33 at 0.05 req/s; 32dd2ab8: 0.83).
- The TP2 32dd2ab8 window swept 0.05-0.15 req/s only, the final matrix 0.05-0.4 req/s.
- The replay matrix was run on the RiNGSiDE profiles only; the comparison recipes have no replay cells.
- The TP4 row shows the replay matrix of the previous base 9e4f793a, not of the final profile 594faefb.
