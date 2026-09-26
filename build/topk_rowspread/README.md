# Rowspread top-k extension (TP4)

The TP4 profile's sparse-attention indexer selects its decode top-k through `nvfp4_topk_rowspread`
(`GLM53_TOPK_ROWSPREAD=1`, read by the patched `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`). The
package directory holds three files: the loader `__init__.py` (served from `src/tp4/nvfp4_topk_rowspread/`), the
compiled library `_kernel.so` and its `build-receipt.json`. The loader checks the PyTorch and CUDA versions and the
library's sha256 against the receipt and refuses anything else, so the library and the receipt always come from the
same build (`release/assets.json`: `nvfp4-topk-rowspread`, `nvfp4-topk-rowspread-receipt`).

## Source

The base image's stage 15 (`docker/base/15-topk-repair`) builds vLLM's persistent top-k kernel with the hunks of
vllm-project/vllm pull request 55314 as `nvfp4_topk_pr55314`. This extension is the same translation unit with two
files replaced:

| file | sha256 | what it is |
|---|---|---|
| `persistent_topk.cuh` | `7c23293a...` | the rowspread version of stage 15's `csrc/libtorch_stable/persistent_topk.cuh` (`d098711c...`): every row at or below `RADIX_THRESHOLD` is taken by one CTA of the whole grid (row r by CTA r mod the grid) in a first pass, instead of by CTA 0 of its group only; the radix rows follow in the original loop, with the same groups, round-robin and barriers |
| `bindings.cu` | `fca0bd59...` | the operator binding: stage 15's `topk.cu` in its own C++ namespace and torch library (`nvfp4_topk_rowspread`), so that it loads next to `nvfp4_topk_pr55314`; the same op schema |

`topk.cu` (`2d0288fa...`), `topk_histogram_4096.cuh` (`994188b0...`) and `torch_utils.h` (`f619e194...`) are stage
15's files, unchanged. The measured library was built from `persistent_topk.cuh` `2f7f4fe2...`, which is this
directory's file without its first two lines (the recipe's change notice).

## Build

Inside the base image (its CUDA 13.0, PyTorch 2.13.0+cu130 and `/opt/topk-repair/src`):

```sh
python3 -B build/topk_rowspread/build.py --src /opt/topk-repair/src --work /tmp/rowspread --out OUT
```

`OUT/_kernel.so` and `OUT/build-receipt.json` go into the package directory next to the loader.
`docker/Dockerfile` runs the same command in its `topk-rowspread` stage. Flags: `-O3 -std=c++20 -DUSE_CUDA
-DTORCH_TARGET_VERSION=0x020B000000000000ULL`, hidden visibility, `-gencode=arch=compute_121,code=sm_121`, linked with
`-Wl,-Bsymbolic` (the measured build's, `measured-build-receipt.json`). A rebuild gets its own receipt and is a rebuilt
artefact (`../README.md`).

## The measured build

`measured-build-receipt.json` is the receipt served with the measured library (`_kernel.so` sha256 `3fefebaa...`,
5,285,424 bytes): built in the served image on one DGX Spark on 2026-09-25 in 21.3 s, and staged unchanged on all
four nodes.
