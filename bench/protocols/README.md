# Benchmark protocol

One campaign per arm on a freshly booted server, with the nodes otherwise idle. Every cell is a single run unless
stated. The records behind the results page are in `../results/evidence/`, and `../results/results.json` names the
lab window of every row.

## RigMark

[alexellis/rigmark](https://github.com/alexellis/rigmark) in the fork
[othexmr/rigmark](https://github.com/othexmr/rigmark), branch `feat/staggered-arrival`, commit
`40fabcaf6e963ac27ab347f61d54a225dbd07419` (parent alexellis/rigmark
`e748928d`). The windows record the local revision `6e0e24a6ff5061721b83a482d1967c6b4e07517f`, which was published as
`40fabcaf` with the identical tree (`3ba4ae72`). Protocol 1.1.0, prompts 1.0.0 (sha256 `0c3ac401...`), temperature 0,
seed 20260905, and `chat_template_kwargs: {"reasoning_effort": "low"}` in every request.

1. **Code-full**: cold prefill of 8K, 32K and 64K-token prompts (time to first token; the RiNGSiDE TP4 multi-run
   windows also 128K and 262,136 tokens), warm replay of the same prompts (served from the prefix cache), single-stream
   decode on code, prose and structured prompts (4,096 output tokens), and short code requests (256 output tokens) at 1
   to 6 concurrent streams, plus 8, 12 and 16 for the RiNGSiDE TP4 profile (aggregate end-to-end tokens per second).
2. **Prose concurrency with staggered arrivals** (the fork's section): prose requests at the same stream counts, and
   staggered arrivals at L = 2, 4 and 6 (8, 12 and 16 for the RiNGSiDE TP4 profile):
   - *prefill-first*: one 32,768-token request starts, and while it is in its prefill L-1 short requests arrive; the
     cell is the median time to first token of those short newcomers;
   - *decode-first*: L-1 short streams are decoding when a 32,768-token request arrives; the cell is that request's
     time to first token.
   A round in which the newcomer only starts after every incumbent has finished is invalid ("-" on the page). The
   windows ran one run per cell, except the final TP4 and TP2 runs (2 runs of every cell, every suite), the
   previous TP4 profile's window (5 runs) and the previous TP2 profile's window (3 staggered rounds); a cell is then
   the median (with 2 runs, the mean of the two).
3. **sparkDash**: the lab's Python reproduction of the decode benchmark of
   [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) (its DecodeBench protocol: the same prompt texts,
   temperature 0, 400 tokens with `min_tokens` 400, thinking off in the request, one 32-token warm-up stream): decode
   throughput waves of prose, structured, code and JSON prompts at 1 to 4 streams. The page shows code at one stream.
   The reproduction is not part of this repository.

4. **llama-benchy** ([eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0, the RiNGSiDE TP4 profiles only,
   in the boot of the final TP4 run and in that of the previous profile's 5-run window): `--pp 2048 --tg 128`, context
   depths 0, 4096, 8192 and 32768, concurrency 1, 2 and 4, 3 runs, its default book as prompt text, thinking on; the
   arguments and seeds are in `../results/runs/2026-09-25-tp4-final-2/README.md`.

5. **tool-eval-bench** ([SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) commit
   `5381d589`, MIT): its 69 tool-calling scenarios (15 categories, from tool selection and parameter precision to
   prompt-injection safety and planning), scored pass 2 / partial 1 / fail 0 points, at temperature 0 with seed 42,
   thinking on, 4 scenarios in parallel, a 300 s request timeout and at most 8 turns. One run each on the final TP4 run
   (after llama-benchy, same boot), on the TP4 profile's check window and on the MiaAI-Lab TP4 recipe with the launch
   plan of its TP4 row; the summary is `../results/evidence/tool-eval-bench.json`.

## Replay matrix (RiNGSiDE profiles only)

The full protocol, with the commands, is `../results/evidence/replay-matrix.md`; the per-window results are in
`../results/evidence/replay-matrix.json`. In short:

- **Client**: [othexmr/rigmark](https://github.com/othexmr/rigmark) `feat/serving-interference-and-replay` @
  [`c1b26ecf`](https://github.com/othexmr/rigmark/commit/c1b26ecf0724563dcf8a20009b6e169aa87215fd), same tree as the
  measured `78759b10`; a different commit from the headline cells' client.
- **Requests**: authored multi-user scenarios, not recorded production traffic, built from five input files named by
  their hashes; a 16-task quality set whose tasks end in an exact `ANSWER:` line.
- **Latency target**: a request meets it only if it completes, its first output of any kind arrives within 15 s of its
  scheduled arrival, no gap between output deltas exceeds 3 s, and it finishes within 300 s. Attainment at a rate is
  the passing share of all planned requests; a rate meets the target at attainment 0.9 or more in a valid run.
- **Sweep**: an open-loop seeded Poisson arrival process over 120 s at 0.05, 0.1, 0.15, 0.2, 0.3 and 0.4 req/s (0.05
  to 0.15 in the two TP2 windows of the `32dd2ab8` promotion); the page's cell is the highest rate that meets the
  target with every lower rate also meeting it.
- **Valid**: exact token delivery, a valid client schedule and receipts that re-verify, for every run of the window.

## Comparison recipes

Comparison recipes were measured with the same tools on the same nodes. Each ran as a whole recipe: its own checkpoint,
quantisation, speculative decoding, image, runtime and admission limit, with the changes the lab recorded to make it
run on this ring (`../results/evidence/comparison-arms.json`, per row and field, each with its source). Up to 6 streams
every recipe ran every cell (a recipe that admits fewer requests queues the rest); streams beyond a recipe's request cap
(8 to 16) were not run, so the results page lists those cells for the RiNGSiDE TP4 profile only.
