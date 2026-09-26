# Window 2026-09-25-tp4-final-2: final TP4 run, RigMark 2 rounds and llama-benchy

One boot of lab plan `387b3ff3` on four DGX Sparks, 2026-09-25 00:30 to 01:20 (UTC+7), with nothing else running:
RigMark 00:30 to 00:59, llama-benchy 00:59 to 01:07, then tool-eval-bench (its summary is in
`../../evidence/tool-eval-bench.json`). Lab plan `387b3ff3` is the TP4 profile (`launch/profiles/tp4`, lab plan
`6f490797`) without the replay-boundary admit fix: the profile serves `glm53_replay_boundary.py` `d41662b1` where
this run served `144c7d34`; every other served file and every flag are the same. The results page (`../../README.md`)
shows these runs as the TP4 row, medians of 2 runs.

## RigMark

[othexmr/rigmark](https://github.com/othexmr/rigmark) branch `feat/staggered-arrival`, commit `40fabcaf` (the runs record the local revision
`6e0e24a6`, published as `40fabcaf` with the identical tree), protocol 1.1.0, prompts 1.0.0, 2 runs of every
suite. Site values are placeholders:

```sh
./rigmark run --base-url ${API_URL} --model nvidia/GLM-5.3-Flash-NVFP4 --metadata metadata.json \
  --comparison-id tp4-winners-2round-20260924 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8,12,16 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label tp4-winners-2round-20260924-code-full --prefill-depths 8192,32768,65536,131072,262136 --output code-full.json
./rigmark run --base-url ${API_URL} --model nvidia/GLM-5.3-Flash-NVFP4 --metadata metadata.json \
  --comparison-id tp4-winners-2round-20260924 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8,12,16 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label tp4-winners-2round-20260924-prose-concurrency --concurrency-workload prose --skip-prefill \
  --staggered 2,4,6,8,12,16 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 \
  --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose --output prose-concurrency.json
```

The prefill suite covers 8,192, 32,768, 65,536, 131,072 and 262,136 tokens (8 tokens under the 262,144-token
context), each cold and as an immediate warm replay. The staggered-arrival section is the
fork's modification of RigMark. Each card names the sha256 of its JSON receipt: `code-full.json` is included unedited,
and the prose receipt's hash is the one recorded below. With 2 runs, RigMark's median of a cell is the mean of its two
runs.

## llama-benchy

[eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0 (MIT), run through the lab's seeding wrapper (it seeds llama-benchy's prompt sampling and refuses to
run without the pinned tokenizer and book): pass A `--pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3
--latency-mode generation` with seed 2701, pass B the same at `--depth 32768` with seed 2702; the model's own tokenizer
(its `/tokenize` counts matched the server's), llama-benchy's default book (Project Gutenberg text 1661, sha256
`8a2f79a2f460...`, 606,662 bytes) and `chat_template_kwargs` `{"enable_thinking": true}`. `passA.json` and
`passB.json` are llama-benchy's own results; `summary.json` is the lab's table of their medians.

## Files

Included (copied byte for byte, except the extract):

| file | status | sha256 | bytes |
|---|---|---|---:|
| `rigmark/code-full.json` | unedited | `075dbcd5c2a2bfb6` | 301,177 |
| `rigmark/code-full.card.txt` | unedited | `f8a22018242f5b2d` | 3,692 |
| `rigmark/prose-concurrency.card.txt` | unedited | `d0c248561bda838f` | 3,876 |
| `rigmark/metadata.json` | unedited | `901182f4423acb01` | 1,449 |
| `llama-benchy/passA.json` | unedited | `153444b157f65ea1` | 25,331 |
| `llama-benchy/passB.json` | unedited | `cfb21ac436815ecc` | 8,645 |
| `llama-benchy/summary.json` | unedited | `882f9a367229eef8` | 3,716 |
| `rigmark/prose-concurrency.rounds.json` | extract | `d52d3a1f7c63c21d` | 66,775 |

Not included, identified by the sha256 of the lab's file:

| file in the window | sha256 | bytes | why |
|---|---|---:|---|
| `native/prose-concurrency.json` | `607f45bdb85564d4` | 2,866,863 | too large for this repository (2.9 MB); rigmark/prose-concurrency.rounds.json keeps every summary statistic and the per-round metrics |
| `native/code-full-command.json` | `664aef45031bfb73` | 875 | names the lab host and paths; the command is shown above with placeholders |
| `native/prose-concurrency-command.json` | `c16c7a1098333485` | 1,175 | names the lab host and paths; the command is shown above with placeholders |
| `llama-benchy/passA.command.json` | `ca770a23de5e00e0` | 1,637 | names the lab host and paths; the arguments are listed above |
| `llama-benchy/passB.command.json` | `bc01a5630f644f04` | 1,621 | names the lab host and paths; the arguments are listed above |

No included file was edited: the address of the server appears only in the command and receipt files left out above.
`manifest.json` lists the same with full hashes.
