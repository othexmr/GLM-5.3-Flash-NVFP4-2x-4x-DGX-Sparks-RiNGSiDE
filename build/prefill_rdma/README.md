# Prefill RDMA library (TP4)

`src/tp4/glm53_prefill_rdma/ring.py` loads `libprb-<first 16 hex of the source sha256>.so` from its package
directory. Build it from `_prb.cu` on each node, inside the base image (its `nvcc` and `libibverbs`):

```sh
cd src/tp4/glm53_prefill_rdma
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121,code=sm_121 -Xcompiler -fPIC -shared \
    -o libprb-7ffd99abacd3e746.so _prb.cu -libverbs -lpthread
```

The measured libraries were built from the earlier `_prb.cu` (sha256 `7a0bd2d1227e142c...`, output name `libprb-7a0bd2d1227e142c.so`), which differs from the served source only in the comment on line 2; the same bytes are served under the current name. The measured nodes served two different builds of that source: ranks 0, 1 and 2
`9f4106239b7fa8ef6949691f2638a9d8ebf10103bb10e3941cd4ad9073d11b73`, rank 3
`9b54e41bf46f09939f10a60d40194d6fd2d7398f092a090ba9a6686fd4a18e57`. `sources/apply.py --rank N` selects the rank's
recorded hash.
