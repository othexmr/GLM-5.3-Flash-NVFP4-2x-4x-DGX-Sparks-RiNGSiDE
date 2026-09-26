# NCCL 2.30.7 switchless dual-PF (TP4)

Source: NVIDIA/nccl `73cf112295c33aee2b895f329f592f2a9b4b0f97` (v2.30.7-1) with `patches/nccl/0001` and `0002` applied
in order. The resulting source tree is Git tree `9edccf0677e35c9b451eb2f5e037dfb9e6677a93`.

The measured library below was built from tree `16e997d8cbb8c67afffceb974e1513b4f47c5a2f`: the same series before the
change-notice comments were added to `src/transport/net_ib/connect.cc`. Those three comment lines are the only
difference, but NCCL's log macros record source line numbers, so a build from the current tree differs in the line
numbers of some log messages (and a rebuild is not bit-reproducible in any case).

```sh
git clone https://github.com/NVIDIA/nccl.git && cd nccl
git checkout 73cf112295c33aee2b895f329f592f2a9b4b0f97
git apply --index ../patches/nccl/0001-switchless-ring-setup-and-listener-hardening.patch
git apply --index ../patches/nccl/0002-dual-pf-four-gid-routing.patch
git write-tree    # expect 9edccf0677e35c9b451eb2f5e037dfb9e6677a93
make -j4 src.build CUDA_HOME=/usr/local/cuda-13.0 NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
sha256sum build/lib/libnccl.so.2.30.7
```

The measured library is `libnccl.so.2.30.7`, 61,427,896 bytes, sha256
`5d5c65502cd3d1336dd73b9dd3f1d584fc1289c79fd595f9605b8a6ac3caab7f`, built with CUDA 13.0 (`nvcc` V13.0.88) on Linux
AArch64. Install it as `/opt/switchless-nccl/libnccl.so.2.30.7` with the symlink `libnccl.so.2` in the same
directory (`sources/apply.py` does this from an artefacts directory).

The build accepts only the fabric plan of `fabric/README.md` and must run with `NCCL_ALGO=Ring`. Two limits of this
exact source are known and kept: an alias-only `NCCL_SKIP_TREE_CONNECT` can bypass the strict listener validation,
and turning off both the switchless and the extended-GID options does not restore upstream's selected-device-only
publication. The TP2 profile uses the image's stock NCCL.
