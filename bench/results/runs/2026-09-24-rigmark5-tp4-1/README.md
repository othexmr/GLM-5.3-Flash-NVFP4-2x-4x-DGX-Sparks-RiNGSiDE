# Window 2026-09-24-rigmark5-tp4-1: TP4 profile, RigMark 5 rounds and llama-benchy

One boot of the published TP4 profile (`launch/profiles/tp4`, lab plan `50e80d44`) on four DGX Sparks, 2026-09-24
16:30 to 17:59 (UTC+7), with nothing else running. The results page (`../../README.md`) shows these runs as the TP4
5-run medians.

## RigMark

[othexmr/rigmark](https://github.com/othexmr/rigmark) branch `feat/staggered-arrival`, commit `40fabcaf` (the runs record the local revision
`6e0e24a6`, published as `40fabcaf` with the identical tree), protocol 1.1.0, prompts 1.0.0, 5 runs of every
suite. Site values are placeholders:

```sh
./rigmark run --base-url ${API_URL} --model nvidia/GLM-5.3-Flash-NVFP4 --metadata metadata.json \
  --comparison-id tp4-final-5round-20260924 --runs 5 --prefill-runs 5 --concurrency 1,2,3,4,5,6,8,12,16 \
  --concurrency-runs 5 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label tp4-final-5round-20260924-code-full --prefill-depths 8192,32768,65536,131072,262136 --output code-full.json
./rigmark run --base-url ${API_URL} --model nvidia/GLM-5.3-Flash-NVFP4 --metadata metadata.json \
  --comparison-id tp4-final-5round-20260924 --runs 5 --prefill-runs 5 --concurrency 1,2,3,4,5,6,8,12,16 \
  --concurrency-runs 5 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label tp4-final-5round-20260924-prose-concurrency --concurrency-workload prose --skip-prefill \
  --staggered 2,4,6,8,12,16 --staggered-runs 5 --staggered-depth 32768 --staggered-incumbent-tokens 1024 \
  --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose --output prose-concurrency.json
```

The prefill suite covers 8,192, 32,768, 65,536, 131,072 and 262,136 tokens (8 tokens under the 262,144-token
context), each cold and as an immediate warm replay. The staggered-arrival section is the
fork's modification of RigMark. Each card names the sha256 of its JSON receipt: `code-full.json` is included unedited,
and the prose receipt's hash is the one recorded below.

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
| `rigmark/code-full.json` | unedited | `6ea47a2a16ab6741` | 732,050 |
| `rigmark/code-full.card.txt` | unedited | `537854c09fdd6c4f` | 3,692 |
| `rigmark/prose-concurrency.card.txt` | unedited | `e2295fbc349f24b5` | 3,876 |
| `rigmark/metadata.json` | unedited | `15f7ab907bc359dd` | 1,430 |
| `llama-benchy/passA.json` | unedited | `ed19cbd27fd2bf23` | 25,315 |
| `llama-benchy/passB.json` | unedited | `49eff65d4ce31192` | 8,600 |
| `llama-benchy/summary.json` | unedited | `88a480a52e5c9aee` | 3,706 |
| `rigmark/prose-concurrency.rounds.json` | extract | `595f87319d643f62` | 137,014 |

Not included, identified by the sha256 of the lab's file:

| file in the window | sha256 | bytes | why |
|---|---|---:|---|
| `native/prose-concurrency.json` | `c106f69b0daa86c0` | 7,146,506 | too large for this repository (7 MB); rigmark/prose-concurrency.rounds.json keeps every summary statistic and the per-round metrics |
| `native/code-full-command.json` | `ec35761b1d7b73ed` | 877 | names the lab host and paths; the command is shown above with placeholders |
| `native/prose-concurrency-command.json` | `2388673cce2950b2` | 1,177 | names the lab host and paths; the command is shown above with placeholders |
| `llama-benchy/passA.command.json` | `72392928d4d4a634` | 1,643 | names the lab host and paths; the arguments are listed above |
| `llama-benchy/passB.command.json` | `d2c0b50e75edc261` | 1,627 | names the lab host and paths; the arguments are listed above |

No included file was edited: the address of the server appears only in the command and receipt files left out above.
`manifest.json` lists the same with full hashes.
