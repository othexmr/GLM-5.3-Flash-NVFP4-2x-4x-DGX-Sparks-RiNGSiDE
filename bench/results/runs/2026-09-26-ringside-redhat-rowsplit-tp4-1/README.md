# Window 2026-09-26-ringside-redhat-rowsplit-tp4-1: the release window of the TP4 profile

One boot of lab plan `89dfef84`, the TP4 profile of this repository (`launch/profiles/tp4`), on four DGX Sparks,
2026-09-26 (UTC+7), in this order: the switch and served-byte receipts, the trap probe, RigMark 2 rounds
(00:39-01:09), sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) 2 rounds
(01:09-01:12), tool-eval-bench (01:12-01:18; its summary is in `../../evidence/tool-eval-bench.json`), the Korean
check (`../../evidence/release-window-tp4.json`) and the engagement markers. The emoji/CJK wall stress test and
llama-benchy were not run on this plan; the llama-benchy results on the results page are the release plan's run on the
NVIDIA checkpoint (lab plan `31978478`, `../2026-09-25-ringside-combined-tp4-1/`). The results page
(`../../README.md`) shows these runs as the TP4 row, medians of 2 runs.

## RigMark

[othexmr/rigmark](https://github.com/othexmr/rigmark) branch `feat/staggered-arrival`, commit `40fabcaf` (the runs record the local revision
`6e0e24a6`, published as `40fabcaf` with the identical tree), protocol 1.1.0, prompts 1.0.0,
2 runs of every suite. Site values are placeholders:

```sh
./rigmark run --base-url ${API_URL} --model nvidia/GLM-5.3-Flash-NVFP4 --metadata metadata.json --comparison-id \
  ringside-redhat-rowsplit-20260926 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8,12,16 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' --label \
  ringside-redhat-rowsplit-20260926-code-full --prefill-depths 8192,32768,65536,131072,262136 --output \
  code-full.json
./rigmark run --base-url ${API_URL} --model nvidia/GLM-5.3-Flash-NVFP4 --metadata metadata.json --comparison-id \
  ringside-redhat-rowsplit-20260926 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8,12,16 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' --label \
  ringside-redhat-rowsplit-20260926-prose-concurrency --concurrency-workload prose --skip-prefill --staggered \
  2,4,6,8,12,16 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 \
  --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose --output \
  prose-concurrency.json
```

Each card names the sha256 of its JSON receipt: `code-full.json` is included with the lab's model directory replaced by `${SWITCHLESS_MODEL_DIR}` (the card names the receipt as RigMark wrote it: `original_sha256` in `manifest.json`), and the prose receipt's hash is
the one recorded below.
`code-full.card.txt` is not included: RigMark did not render it (its log: "could not render terminal report: invalid
receipt: prefill.262136.cold.runs[1].decode_tokens_per_second does not match tokens/time;
prefill.262136.cold.runs[2].decode_tokens_per_second does not match tokens/time;
prefill.262136.warm_replay.runs[2].decode_tokens_per_second does not match tokens/time"); its JSON receipt
`code-full.json` is.

## Files

Included (copied byte for byte, except the extract and the files marked redacted, in which the lab's model directory is replaced by `${SWITCHLESS_MODEL_DIR}`):

| file | status | sha256 | bytes |
|---|---|---|---:|
| `rigmark/code-full.json` | redacted | `13f42edede7d10a2` | 301,773 |
| `rigmark/prose-concurrency.card.txt` | unedited | `1feb93c3cd3492cf` | 3,876 |
| `rigmark/metadata.json` | redacted | `02538b52217d7316` | 1,718 |
| `rigmark/prose-concurrency.rounds.json` | extract | `ad86493e0e4ccb4b` | 67,106 |

Not included, identified by the sha256 of the lab's file:

| file in the window | sha256 | bytes | why |
|---|---|---:|---|
| `native/prose-concurrency.json` | `9482dab6efa43d8f` | 2,872,618 | too large for this repository; rigmark/prose-concurrency.rounds.json keeps every summary statistic and the per-round metrics |
| `native/code-full-command.json` | `9e367e14f47ea0e4` | 925 | names the lab host and paths; the command is shown above with placeholders |
| `native/prose-concurrency-command.json` | `7825c06d2d63b571` | 1,225 | names the lab host and paths; the command is shown above with placeholders |

`manifest.json` lists the same with full hashes.
