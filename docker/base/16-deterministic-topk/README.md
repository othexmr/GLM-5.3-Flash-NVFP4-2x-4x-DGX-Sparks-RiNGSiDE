# Deterministic sparse top-k

The native prefill selector and repaired persistent decode selector return mathematically valid score sets in nondeterministic order. Equal scores can also select different token indices. Earlier selected-value tests allowed both behaviors, which change downstream attention reductions and can change continuations.

This layer uses the existing exact selection only to obtain the kth score. Two Triton scans select every greater score, choose lower token indices at the threshold, and emit ascending relative indices. It preserves the top-k score multiset while fixing both order and tie-breaking. Masked rows remain `-1`; no answer values or prompt-specific behavior are introduced.

`install.py` requires the exact stage-15 source hash and verifies both installed payload hashes. `test_install.py` checks reconstruction and rejection of modified inputs. `test_original_order.py` reproduces the original order/tie variation. Run `test_gpu.py` in the built image on an otherwise idle GPU: it checks 24 combinations through 65,536 pools, empty/short rows, nonzero prefill starts, padded strides, changed inputs, poisoned outputs, and eager/CUDA-graph equality against an independent stable CPU ordering.

The two-Spark service tests passed exact token-ID and log-probability equality for all four original hybrid-boundary cases. Separate single-rank and dual-rank cache corruption tests passed complete-boundary recomputation with identical output. These results are recorded in `benchmarks/deterministic-topk/`; they do not establish general model factual reliability or external-device qualification.
