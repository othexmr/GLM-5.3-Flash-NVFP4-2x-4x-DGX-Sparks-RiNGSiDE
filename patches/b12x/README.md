# b12x patches

Upstream: [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), reference base
`3a437ab5168060e4d625f05e1625c04089f1ba37 (b12x 1.3.0 for these files)`.

Single-file overlay patches against files of the base image (`docs/image.md`), one directory per profile. Each patch
starts with a header naming its target, the image file's sha256 (preimage), the served file's sha256 (postimage), the
licence and any attribution, followed by a Git diff with full blob ids. `sources/apply.py` applies them with
`git apply` (no fuzz, no three-way merge) and checks both hashes. Patches are independent of each other; `series`
lists them in a fixed order.

The image files already carry earlier lab MoE changes on top of b12x 1.3.0.

| profile | patch | target | preimage (image file) | upstream relation |
|---|---|---|---|---|
| tp2 | `tp2/moe-_shared-kernels-dynamic.py.patch` | `b12x/moe/_shared/kernels/dynamic.py` | `2ebe82bd221a` | image-baked lab change (deterministic output, atomic prefill, decode fast path) on b12x 1.3.0, whose file equals local-inference-lab/b12x@3a437ab5 |
| tp2 | `tp2/moe-fused_moe-_impl.py.patch` | `b12x/moe/fused_moe/_impl.py` | `6984ff844351` | image-baked lab change (deterministic output, atomic prefill, decode fast path) on b12x 1.3.0, whose file equals local-inference-lab/b12x@3a437ab5 |
| tp4 | `tp4/moe-_shared-kernels-dynamic.py.patch` | `b12x/moe/_shared/kernels/dynamic.py` | `2ebe82bd221a` | image-baked lab change (deterministic output, atomic prefill, decode fast path) on b12x 1.3.0, whose file equals local-inference-lab/b12x@3a437ab5 |
| tp4 | `tp4/moe-fused_moe-_impl.py.patch` | `b12x/moe/fused_moe/_impl.py` | `6984ff844351` | image-baked lab change (deterministic output, atomic prefill, decode fast path) on b12x 1.3.0, whose file equals local-inference-lab/b12x@3a437ab5 |
