# Sparse-MLA extensions

The plugin `glm53_sparse_mla` (Libertai/vllm-sparse-mla-blackwell) ships a compiled extension
`_C.cpython-312-aarch64-linux-gnu.so`. The profiles mount the lab's builds of it:

| Profile | Patch against `b1f20638d7f82f880e4bcb2136ee2f19c06f38a4` | Result tree | Measured sha256 | Bytes |
|---|---|---|---|---|
| TP4 | `patches/sparse-mla/build/tp4-so-a1c.patch` | `442d83ded51b5a0226bd13fbb2e3711d45d64699` (measured build: `acb5d5d0`) | `1f940e75297105155cb0f253a351a4c62354054da63b39cd84628731feaf7b99` | 9,050,840 |
| TP2 | `patches/sparse-mla/build/tp2-so-m7e.patch` | `ce9976030ba1b0d87566c78b8269ad73f4e25c01` (measured build: `50c50d5b`) | `221fdbaa6a25d377118093d30761322c22db7105312b396d2aecd8262e2c002b` | 5,825,912 |

Build inside the base image, in the patched plugin tree:

```sh
git clone https://github.com/Libertai/vllm-sparse-mla-blackwell.git && cd vllm-sparse-mla-blackwell
git checkout b1f20638d7f82f880e4bcb2136ee2f19c06f38a4
git apply ../patches/sparse-mla/build/tp4-so-a1c.patch
GLM53_ARCHS=120a,121a python3 setup.py build_ext --inplace
sha256sum glm53_sparse_mla/_C.cpython-312-aarch64-linux-gnu.so
```

The patch changes `csrc/sparse_mla.cu` (the lab's kernels), `setup.py` (serving flags, `-DGLM53_M7E`) and
`glm53_sparse_mla/backend.py` (the image's backend), and adds a "Modified by the GLM-5.3 Switchless recipe" notice to
each of them. The measured extensions were built from the trees before those notice comments were added (the
"measured build" trees above); the comments, and one comment line of `setup.py` reworded since (the comment fix of
2026-09-25), are the only difference, but the build passes `-lineinfo` and the kernel's
checks record source line numbers, so a build from the current trees differs in those (and a rebuild is not
bit-reproducible in any case). The served `backend.py` of the TP4 profile is a further patch
against the image file (`patches/sparse-mla/tp4/backend.py.patch`); the TP2 profile serves the image's `backend.py`.
