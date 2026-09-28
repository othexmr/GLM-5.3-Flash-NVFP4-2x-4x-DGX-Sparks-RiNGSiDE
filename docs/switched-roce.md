# Experimental switched-RoCE option (TP4 only)

This is an explicit, **off-by-default, CPU/source-verified candidate**. It keeps
both measured profiles, their canonical source, manifests, image pins and launch
fixtures unchanged. It is not a new measured lab plan or a serving qualification.

## Why a separate option

Profile promotion requires `sources/import_plan.py`, a complete derived plan and
its actual served files/native artefact identities (`docs/updating.md`). This
candidate changes the prefill CUDA source, so the canonical prefill SO cannot be
relabelled as its binary. No newly built/qualified SO is supplied here. Rather
than invent that identity or a lab qualification, this follows the existing
`launch/options/weightless/prepare.py` pattern: pinned inputs, separate generated
files and explicit optional mounts through `launch/render.py` and Compose.
Future promotion into a profile must use the importer, not edits to `src/` or
profile manifests.

`changes.json` records generated exact-input edits and actual output hashes for
six files. Preparation refuses drift or replacement of different output bytes.
The canonical source tree is never patched in place. Authorship/licence notices
are retained; no model weights, binaries or third-party downloads are included.

## Fabric contract and discovery

- Four ranks form the same *logical* neighbour ring, now over an Ethernet switch.
  Two F1 rails must be active/reachable on every rank. Do not rewire or change
  addresses to make numeric GID indices agree.
- The option retains the upstream HCA naming contract. Inspect each node's RDMA
  devices, mapped netdevs, active ports and GID tables (`rdma link show`,
  `show_gids`, or the read-only `/sys/class/infiniband` GID/type/netdev files).
  Arbitrarily renamed adapters are not automatically supported.
- Native device-list enumeration indices are not protocol lane indices. Devices
  are opened by name; lean's two logical slots and prefill's F1 slots pair lane 0
  with the peer's lane 0, and lane 1 with lane 1. The same rail ordering is required
  across peers. Opposite-rank shortcuts and invalid neighbour/lane inputs are not
  introduced. Unused F0 ports need not be active or opened in this option.
- Each selected HCA independently scans its own port's extended GID table for a
  nonzero IPv4-mapped RoCEv2 entry. Exactly one is required. Missing, ambiguous or
  unrepresentable indices, and unexpected table-query errors, fail clearly.
  Duplicate eligible entries are ambiguous even if their IPv4 address is equal.
  There is no first-match selection or silent IPv6/GID-index fallback. A single
  eligible GID does not itself prove correct routing: operators must still verify
  that each selected netdev/address belongs to the intended rail.
- Different per-HCA/per-node GID indices are normal. Each QP uses its local HCA's
  selected source index and the peer's advertised GID. Numeric index equality is
  not required. Existing active-port, registration, peer and completion checks
  remain; none are disabled to make a link appear healthy.

## Prepare without launching

Source-only preparation is standard-library Python and offline:

```sh
python3 -B launch/options/switched-roce/prepare.py --out deploy/switched
```

It produces the six sources and `switched-roce-receipt.json`. A null
`native_sha256` means **not ready to enable**: the matching prefill SO is missing.
The lean proxy remains source-keyed and builds through its existing mechanism.
Both native libraries require compatible libibverbs extended-GID headers/runtime.

Build the new prefill SO only in a compatible ARM64 CUDA development environment
for the selected immutable image. `--build-native` uses the canonical Dockerfile's
NVCC architecture/link flags and records the actual SO hash; it does not run GPU
or transport checks. For example, with `PINNED_IMAGE` replaced locally by
`registry.example/image@sha256:<64-hex-digest>`:

```sh
docker run --rm --network none \
  -v "$PWD:/recipe:ro" -v "$PWD/deploy/switched:/candidate" \
  --entrypoint python3 "$PINNED_IMAGE" \
  /recipe/launch/options/switched-roce/prepare.py --out /candidate --build-native
```

These are operator instructions, not a statement that the candidate was built.
Do not substitute the canonical SO or copy a library from another source hash.
Distribute identical prepared source/SO bytes to all ranks and independently
check the receipt hashes, dependencies and ABI before any serving test.

In the ignored site file, replace `OPTIONDIR` with that prepared directory's
absolute local path (same layout on every rank):

```text
GLM53_RDMA_SWITCHED=1
SWITCHLESS_SWITCHED_ROCE_DIR=OPTIONDIR
```

`GLM53_RDMA_SWITCHED=1` without the prepared directory, invalid flag values, and
TP2 activation are rejected by the renderer. Review the existing renderer/Compose
output before executing it. The option adds only seven read-only file mounts and
the opt-in flag, and removes two direct-cable assumptions:
`NCCL_SWITCHLESS_RING_ONLY` and `NCCL_IB_ADDR_RANGE`. All other canonical settings,
including custom collectives, model/lean graphs and eager prefill policy, remain.
Setting the flag to `0` or leaving it unset returns the canonical launch unchanged.
This is not permission to alter a live generation.

## Checks and qualification limits

```sh
python3 -B sources/verify.py
python3 -B tests/run_cpu.py
python3 -B tests/cpu/test_switched_roce.py
```

The focused suite exercises canonical/default identity, Python/native mapping
parity, per-HCA index differences, absent IPv4, ambiguity, invalid lanes/edges,
unexpected queries, source-pin mismatches, preparation and renderer/Compose
behaviour. It compiles the actual C/CUDA **host-side functions** against an
in-memory verbs-table fixture; it does not compile all CUDA or perform RDMA.

The experimental ancestor `3339fa860e7c21f207ca2c648c0b205192dd18c3` was attested
separately. This candidate rebases that mapping onto public upstream
`6606f7638aff044a8170e414818477679934a1c3` and adds fail-closed selection and tests.
Those are different bytes. No old GPU, graph, serving or throughput result is
claimed for this candidate. No performance numbers are published here.
Fresh native build, all-rank collective correctness, model graph capture/replay
and serving validation on the exact candidate fingerprint remain required.
