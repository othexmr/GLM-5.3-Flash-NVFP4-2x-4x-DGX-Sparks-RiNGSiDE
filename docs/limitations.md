# Limitations

- **The API name is the NVIDIA checkpoint's.** The profiles serve RedHatAI/GLM-5.3-Flash-NVFP4 under the model name
  `nvidia/GLM-5.3-Flash-NVFP4`, as the lab measured them; clients select the model by that name.
- **The TP2 profile's release changes are not served yet.** Its results (`bench/results/README.md`) are those of lab
  plan `69ad74fc`, measured before them and on the NVIDIA checkpoint: the change notices, the DFlash2 capture and
  drafter KV-cache group and 5 correctness fixes, and the checkpoint switch with its KV cache margin. Each fix and the
  DFlash2 code's TP2 case are verified by GPU leaves and CPU checks; no TP2 window has served them (`CURRENT.md`); the
  TP2 profile's window follows the release, and its numbers with it.
- **The drafter is licensed for non-commercial use.** incoai/GLM-5.3-Flash-DFlash2, which both profiles load,
  states CC BY-NC-ND 4.0 and is released for research and evaluation (its model card at revision `7d74cdd8`).
  The recipe does not distribute it; a deployment downloads it and uses it under that licence.
- **No binaries are distributed.** The measured base image and native artefacts are not published; you build the
  images with `docker/Dockerfile` (`docker/README.md`).
- **A self-built image is not the measured image.** `docker/Dockerfile` rebuilds the base image from pinned sources
  and the lab's stage inputs and checks every patch preimage and served file, but it compiles the native code again
  (the NCCL library, the sparse-MLA and FP8 GEMM extensions, the prefill RDMA library, the vLLM Rust artifacts and the
  extensions of the lab's image stages), and Ubuntu, CUDA and some build tools are resolved when it runs
  (`docs/build-provenance.md`). The lab's own builds of the same source already differ (the prefill RDMA library
  between nodes). Its base's `model.py` and `kv_cache_utils.py` carry the recipe's DFlash2 capture and KV-cache group
  in place of the lab's earlier versions; both profiles serve their own copies of both. A self-built image needs its
  own qualification before its numbers stand next to the measured ones.
- **The Dockerfile has not been built yet (its first build check follows the release).** It was written from the lab's
  build records and checked on a CPU; its first build on a DGX Spark is also its test. The one-command Compose build
  needs a Compose release that accepts a `service:` reference in `additional_contexts` (`docker/README.md` has the
  two-command fallback).
- **Build inputs can disappear.** The glm-release commit is fetched from a fork (ZJY0516/vllm), and the two base
  images are pinned by digests that their registries may remove; the build then stops at that input
  (`docs/build-provenance.md`, "Blockers").
- **Topology-specific.** The TP4 NCCL build accepts only the four cable subnets 10.100.224.0/24 to 10.100.227.0/24, and
  the lean all-reduce assumes the cabling in `fabric/README.md`. Other topologies need source changes and new
  measurements.
- **NON-DETERMINISTIC outputs.** Both profiles accumulate MoE outputs with atomic adds in prefill (and TP4 also in
  decode launches of 10 or more rows), so outputs are not bit-reproducible from run to run.
- **Most results are single runs** from different days and times of day (the RiNGSiDE TP4 row: medians of 2 runs in
  one boot; the TP2 row: medians of 2 runs in one boot; the previous TP4 profile: 5 runs); room temperature was not
  logged (`bench/results/README.md`, Caveats).
- **The TP4 profile has one release window.** Lab plan `89dfef84` was measured in one boot (2026-09-26): its RigMark
  cells are medians of 2 runs of that boot, with no matched reference window (the earlier TP4 rows are other boots and
  times of day); each of its changes was checked for correctness on its own before (`CURRENT.md`).
- **No CI on hardware.** The CPU checks validate metadata, patches and rendering, not numerics or serving.
- **No authentication on the API.** The server listens on all interfaces without authentication or TLS
  (`docs/operations.md`, API trust boundary).
- **Fixed TP4 device names.** The TP4 transports open four named RDMA functions and GID index 3
  (`fabric/README.md`); other hardware naming needs source changes.
- **Carried-over settings.** A few environment variables of the measured plans have no reader in the served files or in
  the available image sources (`docs/configuration.md`); they are kept so the launch matches what was measured.
- **TP2 final run without a fresh reference.** The TP2 profile's final run was one window; the lab compared it with
  the previous profile's stored windows (other boots, times of day and run counts) as information only, with no rule
  (`bench/results/README.md`).
- **TP2 arrival tails.** The TP2 staggered-arrival cells vary between rounds (the L6 prefill-first newcomer took 5.49,
  4.77 and 6.97 s in the three rounds of the previous profile's window), and an earlier TP2 tail event (a 10.98 s
  incumbent freeze at L6, 2026-09-22) was not diagnosed.
- **TP2 built on a profile promoted over failed guard clauses.** Three clauses of the lab's rule failed for the
  previous TP2 profile `32dd2ab8`, which the current one extends (a decode cell 6.29 % slower, host memory 1 MiB under
  the floor, one cached-prefix difference above the gap bound); the owner promoted it anyway (`bench/results/README.md`).
