# Sparse-MLA plugin patches

Upstream: [Libertai/vllm-sparse-mla-blackwell](https://github.com/Libertai/vllm-sparse-mla-blackwell), reference base
`b1f20638d7f82f880e4bcb2136ee2f19c06f38a4`.

Single-file overlay patches against files of the base image (`docs/image.md`), one directory per profile. Each patch
starts with a header naming its target, the image file's sha256 (preimage), the served file's sha256 (postimage), the
licence and any attribution, followed by a Git diff with full blob ids. `sources/apply.py` applies them with
`git apply` (no fuzz, no three-way merge) and checks both hashes. Patches are independent of each other; `series`
lists them in a fixed order.

`build/` holds two patches against the plugin repository itself, from which the compiled extensions were built (`build/sparse-mla/README.md`); they are not overlay patches and are not in `series`.

| profile | patch | target | preimage (image file) | upstream relation |
|---|---|---|---|---|
| tp2 | `tp2/backend.py.patch` | `glm53_sparse_mla/backend.py` | `8bb063484493` | image-baked lab change (atomic-prefill and NoPE backend stages) on Libertai/vllm-sparse-mla-blackwell@b1f20638 |
| tp4 | `tp4/backend.py.patch` | `glm53_sparse_mla/backend.py` | `8bb063484493` | image-baked lab change (atomic-prefill and NoPE backend stages) on Libertai/vllm-sparse-mla-blackwell@b1f20638 |
