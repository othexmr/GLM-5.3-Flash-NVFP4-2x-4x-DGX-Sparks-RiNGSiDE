# Window 2026-09-25-ringside-combined-tp4-1: the release plan on the NVIDIA checkpoint

One boot of lab plan `31978478` (the RiNGSiDE release plan with nvidia/GLM-5.3-Flash-NVFP4 revision `423acf37`), on
four DGX Sparks, 2026-09-25 (UTC+7): RigMark 3 rounds (21:04-21:49), then llama-benchy (passes A and B:
`llama-benchy/`, llama-benchy's own results). The window stopped there by the owner's choice: tool-eval-bench,
sparkDash and the correctness checks did not run. The TP4 profile then moved to another checkpoint; this run is an
additional arm of the results page.

`rigmark/code-full.card.txt` is not included: RigMark did not render it (its log: "could not render terminal report:
invalid receipt: prefill.262136.warm_replay.runs[1].decode_tokens_per_second does not match tokens/time"); its JSON
receipt `rigmark/code-full.json` is.

## llama-benchy

[eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0 (MIT), run through the lab's seeding wrapper, with the
arguments of the earlier TP4 windows (`../2026-09-25-tp4-final-2/README.md`): pass A `--pp 2048 --tg 128 --depth 0
4096 8192 --concurrency 1 2 4 --runs 3 --latency-mode generation`, pass B the same at `--depth 32768`.
`passA.json` and `passB.json` are llama-benchy's own results; `summary.json` is the lab's table of their medians,
shown on the results page as the release plan's llama-benchy run on the NVIDIA checkpoint.
