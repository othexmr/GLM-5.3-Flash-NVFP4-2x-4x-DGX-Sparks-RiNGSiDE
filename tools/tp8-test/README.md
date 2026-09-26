# TP8 tester kit

Use an exclusive hardware window. In the maintainer's lab, every command that
touches a Spark must be submitted through `hw.py`. The kit does not configure
networking, download weights, read credentials or publish a report. Commands
without `--go` only check inputs and print a plan.

1. Build `docker/Dockerfile` target `profile-tp8` and install the identical image
   on eight nodes. Follow `docker/README.md` for its pinned build context.
2. Copy `site.env.example` to ignored `site-tp8.env`, fill eight management/SSH
   destinations and model/tokenizer paths, and set the TP8 values documented in
   `docs/tp8.md`. Prepare a topology matching the actual cabling:

   ```sh
   python3 -B fabric/make_topology.py --ranks 8 > topology.json
   ```

3. Check every installed consumer, full target shard identity, pinned tokenizer,
   local fabric and PCI roots; then exchange endpoint probes and execute real
   all-reduce/all-gather with the exact library, before loading weights:

   ```sh
   python3 -B tools/tp8-test/kit.py transport --site site-tp8.env --topology topology.json
   python3 -B tools/tp8-test/kit.py transport --site site-tp8.env --topology topology.json --out work/tp8-transport-1 --go
   ```

4. Run target-only eager bring-up, continuous memory sampling, deterministic
   arithmetic/JSON/Korean probes, longer prompts and exact prompt replays:

   ```sh
   python3 -B tools/tp8-test/kit.py run-stage --site site-tp8.env --topology topology.json --stage target --receipt work/tp8-transport-1/receipt.json --out work/tp8-target-1 --go
   ```

5. Only with a passing target receipt, run `--stage fixed-k` and its receipt as
   the input to `--stage graphs`. Give each run a new output directory. The
   importer derives both candidates (`sources/import_plan.py --tp8-variants`);
   neither changes the baseline or enables adaptive policies. K=4 is fixed.
   The graph stage adds bounded C4 and both staggered-arrival directions.

`run-stage` stops its own serving containers when finished or failed. `launch`
starts the admitted stage for a manual session; stop that session with
`launch/down.sh --profile tp8 --site site-tp8.env --go`. It does not grant a
passing memory or smoke receipt. Direct `up.sh --profile tp8 --go` refuses
launches without the kit's transport checks.

Each receipt names the exact image, source, target, tokenizer, library, topology
and checker bytes, plus the site/rendered-launch digest and raw evidence files. Missing ranks, changed hashes,
failed tests, a reboot, a stale receipt, unsupported settings or a memory reserve
below 16 GiB stop advancement. A changed checker requires a new transport run.
Changing model/tokenizer mounts or other site settings requires a new transport
receipt. Configure the intended drafter path before the first transport run.
For drafting, the kit checks the selected snapshot, its geometry and identical
bytes on all ranks. Both accepted and rejected draft tokens must be observed.

These are **bring-up tests**, not exhaustive state correctness or performance
qualification. Task-level prompt replay and draft counters do not prove internal
KDA/cache byte equality. Record such state leaves separately. Memory sampling
is every 200 ms and spans before load through serving; it reports sampled
headroom, not an exact allocator peak. HTTP TTFT/decode measurements include
client and network costs; no engine-only prefill speed is fabricated. The
context labels select approximate prompt sizes; `prompt_tokens` records the
actual tokenizer count. Short smoke answers do not measure sustained decode.

`collective.py` also supports four ranks for a maintainer regression, separately
from eight-rank admission. It verifies exact integer-valued data in int32/fp32
all-reduce and all-gather, from 4 bytes to 8 MiB. Actual rank/channel and
HCA/GID records must agree with the topology. Rank zero must log Ring selections
for both collective operations. Both PCI roots and cable peers must appear.
An environment variable alone does not satisfy this gate.

Export a report with:

```sh
python3 -B tools/tp8-test/kit.py bundle --receipt work/tp8-target-1/receipt.json --out work/tp8-report.json
```

The exporter uses a typed allowlist. It preserves hashes, numeric metrics and
rank relationships, consistently replaces endpoint addresses, and excludes raw
logs, paths, prompts/outputs, environments and headers. Inspect the resulting
JSON before attaching it to the TP8 issue template. Retain the full local
evidence, including failures, for diagnosis.

Optimization promotion additionally needs fresh same-runtime TP8 controls,
paired repetitions, fixed quality/non-regression rules, complete operation
timings, replay/state tests and memory. The kit never promotes a variant from a
single smoke run or infers TP8 speed from a four-rank or single-GPU result.
