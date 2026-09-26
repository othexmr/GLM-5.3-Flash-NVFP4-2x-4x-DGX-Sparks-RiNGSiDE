# Experimental TP8

The `tp8` profile is a candidate for eight DGX Sparks on a closed, alternating
two-port ring. Eight-rank loading, correctness, memory capacity and performance
have not been validated by the maintainer. The four-Spark lab can check source,
single-rank kernels and four-rank transport; those checks do not qualify TP8.

The initial profile uses the corrected TP2 release sources with TP8 sharding,
target-only eager execution, one request, an 8,192-token limit, FP32 KDA state,
and an explicit 4 GiB KV cache per rank. The target is
`RedHatAI/GLM-5.3-Flash-NVFP4`, revision
`18d55bfd5a2194887738da73753975c9d3842f46`, with the pinned NVIDIA tokenizer
override recorded in `release/assets.json` and top-p 0.95. The primary API model
name is `RedHatAI/GLM-5.3-Flash-NVFP4`, identifying Red Hat's quantization.
`nvidia/GLM-5.3-Flash-NVFP4` is also accepted for compatibility with TP2/TP4
clients; both names select the same Red Hat weights. The profile is text-only.

TP2/TP4 geometry, lean and pair all-reduce, prefill RDMA, row-split indexer,
atomic MoE prefill, fused state commit and adaptive scheduling are disabled.
The support modules remain present where imports require them. `glm53_tp8.py`
rejects unsupported settings and compares its configuration digest across all
ranks before target allocation. Default quantized backends still require their
TP8-shaped kernel checks; a switch being disabled does not prove a fallback.

NCCL has a separate `nccl-tp8` source lock, Docker stage and installation under
`/opt/switchless-nccl-tp8`. The TP4 library, profile and served files remain pinned
to the release. Both profiles' bytes and manifest objects are regression tested.

Build with Dockerfile target `profile-tp8`; supply the same pinned `vllm-source`
build context described in `docker/README.md`. The stage builds its native
artefacts from source. A rebuilt image and library get their own identities and
must pass the tester gates. The initial native ARM64/SM121 compile receipt is in
`build/nccl-tp8/compile-receipt.json`; it is not a collective test.

Generate a topology with:

```sh
python3 -B fabric/make_topology.py --ranks 8 --subnet-base 10.100.224.0/24 > topology.json
```

Edit the generated management and endpoint addresses to match the actual
cabling. This command writes a plan; it does not configure a network. Validate
the global topology and each node's sysfs with `fabric/preflight.py`. NCCL needs
eight comma-separated /24 cable networks in `SWITCHLESS_CABLE_SUBNETS`; there
are still four local IPv4 GIDs per Spark, across two cables and two PCI roots.
The local preflight alone does not prove peer reachability or collective routes.

Bring-up proceeds through transport, target-only, fixed-K=4 drafting, then
graphs and bounded concurrency. Each stage needs successful evidence from the
previous stage on the exact image, source, target, tokenizer and topology.
Target-only and fixed-K timings are bring-up diagnostics. Report performance
only after the graphs stage passes; speed comparisons still require repeated,
matched controls on the exact runtime and workload.
Adaptive policies and optional speed paths need a new, matched TP8 control.
Do not infer speed from the four-rank transport check or single-GPU kernels.

Keep at least 16 GiB host MemAvailable on every rank throughout load, packing,
warm-up, graph capture and serving. Sample the full lifecycle; an after-ready
reading cannot establish peak headroom. Increase the explicit cache in steps
no larger than 16 GiB, using smaller steps near the reserve. The initial guard
allows at most 8 GiB and 8K context; larger capacity needs a reviewed new plan.

TP6 is unsupported by this implementation's equal-head TP layout: 64 attention
and KDA heads do not divide by six. The indexer's 32 heads are replicated and
are not an additional TP divisor. Six Sparks can form three independent TP2
replicas, or TP4 plus TP2, with suitable independent cabling, ports and caches.
A six-node ring does not automatically supply those independent fabrics.

TP8 remains experimental until the eight-rank gates pass for the exact image,
assets and topology being tested.
