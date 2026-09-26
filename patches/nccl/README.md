# NCCL patches

Upstream: [NVIDIA/nccl](https://github.com/NVIDIA/nccl), base `73cf112295c33aee2b895f329f592f2a9b4b0f97` (v2.30.7-1).

A Git series for the TP4 NCCL library (`build/nccl/README.md`). Apply in `series` order with `git apply --index`;
the tree after each patch is recorded in `sources.lock.json`.

| patch | contents | authors | tree after |
|---|---|---|---|
| `0001-switchless-ring-setup-and-listener-hardening.patch` | skip Tree/PAT connection setup and advertise both listener GIDs (SparkRing), with the opt-in parsing, listener validation and diagnostics of alexellis/switchless-nccl | SparkRing contributors; Alex Ellis / OpenFaaS Ltd | `560ba01b9becbc7d3daa1677f0216503fc3be631` |
| `0002-dual-pf-four-gid-routing.patch` | four-GID listener handles, PCI-root-preserving routes, strict two-cable/two-root validation, route diagnostics, CPU tests; change notices in `connect.cc` | the GLM-5.3 Switchless recipe lab (othexmr) | `9edccf0677e35c9b451eb2f5e037dfb9e6677a93` |

Existing NVIDIA copyright notices and `LICENSE.txt` stay in the tree. `connect.cc`, the one NVIDIA file both patches
change, keeps the switchless-nccl change notice of `0001` and adds this recipe's notice after it; the files `0002`
creates are new. The measured library was built from tree `16e997d8cbb8c67afffceb974e1513b4f47c5a2f`, the same series
before those notice comments were added (they are the only difference; `build/nccl/README.md`). The TP2 profile uses
the image's stock NCCL.
