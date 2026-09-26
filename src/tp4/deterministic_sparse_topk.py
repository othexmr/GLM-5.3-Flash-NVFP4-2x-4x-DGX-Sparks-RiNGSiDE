"""Canonicalize an exact top-k selection: lower indices win ties, ascending output.

The existing selector supplies only the kth score. Two fixed-order scans recover
the selected indices without relying on atomic arrival order. Prefill indices
are relative to each row's start, matching vLLM's native operation.

2026-09-23, on the installed module (sha 268a4923: the 05465edb helper with
SS/OS/N/T as runtime arguments, so a new context width never recompiles): above
SMALL_ROWS rows the row threshold (the minimum score over the selector's K picks)
is computed once per row by _threshold instead of by every (row, tile) program of
_counts. _counts re-gathered all K selected scores in each of its cdiv(n, 1024)
tiles -- K random loads per tile, 27-32x the row's own scan at 32K context and 128x
at 128K; 7.9 % of a 128K prefill. Same reduction (tl.min over the same K values,
same K and num_warps), same counts, same scatter: identical output (leaf:
outputs/2026-09-23-topk-threshold). At <= SMALL_ROWS rows (C1-C2 decode) the extra
launch costs more than the redundant gathers, so the installed fused kernel runs
unchanged there.

2026-09-23 23:35, tile skip above SMALL_ROWS only (outputs/2026-09-23-topk-tiles-large): the tile skip of
3618fe4d (outputs/2026-09-23-topk-tiles) cut C9-C16 decode steps by 1.67 and 1.38 % in two campaigns, but C1/K5
(6 rows) was 3.45 and 2.87 % slower in both, with no kernel-level explanation (the leaf timed the small path faster).
Here the skip runs only on the path above SMALL_ROWS rows: _counts_thr skips as in 3618fe4d, and _scatter_skip is
3618fe4d's _scatter. At <= SMALL_ROWS rows, 60cd8ccc's _counts and _scatter run unchanged (same source, same names),
so C1-C3 decode runs exactly the promoted kernels. Identical output on every path.
"""
import os
import sys

import torch
import triton as tr
import triton.language as tl

SMALL_ROWS = 16
# Top-k threshold path at small row counts (2026-09-24, outputs/2026-09-24-topk-small-threshold; default off):
# GLM53_TOPK_SMALL_ROWS=<n> overrides SMALL_ROWS, the largest row count that keeps the fused small path (_counts +
# _scatter); above it the threshold path runs (_threshold once per row, _counts_thr and _scatter_skip, which return at
# once for tiles past the row's end). At decode the logits are max_model_len wide (262,144 = 256 tiles of 1,024), and
# the fused _counts re-gathers the row's K selected scores in every one of the 256 tiles while _scatter reads all 256
# tiles' counts. Same threshold (min over the same K values), same counts, same scatter order: identical output.
_SMALL_ROWS_ENV = os.environ.get("GLM53_TOPK_SMALL_ROWS", "").strip()
if _SMALL_ROWS_ENV:
    SMALL_ROWS = int(_SMALL_ROWS_ENV)
    if not 0 <= SMALL_ROWS <= 16:
        raise ValueError("GLM53_TOPK_SMALL_ROWS must be 0..16")
_SMALL_LOGGED = [not _SMALL_ROWS_ENV]


@tr.jit(do_not_specialize=["SS", "OS", "N", "T"])
def _counts(S, O, Starts, Ends, Counts, Thresholds,
            SS, OS, N,
            K: tl.constexpr, T, HAS_START: tl.constexpr,
            B: tl.constexpr):
    r, tile = tl.program_id(0), tl.program_id(1)
    start = tl.load(Starts + r) if HAS_START else 0
    end = tl.load(Ends + r)
    ks = tl.arange(0, K)
    ids = tl.load(O + r * OS + ks)
    selected = tl.load(S + r * SS + start + ids,
                       (ids >= 0) & (start + ids < end), other=float('inf'))
    threshold = tl.min(selected, 0)
    if tile == 0:
        tl.store(Thresholds + r, threshold)
    cols = tile * B + tl.arange(0, B)
    valid = (cols < N) & (cols >= start) & (cols < end)
    scores = tl.load(S + r * SS + cols, valid, other=0.)
    greater = tl.sum((valid & (scores > threshold)).to(tl.int32), 0)
    equal = tl.sum((valid & (scores == threshold)).to(tl.int32), 0)
    tl.store(Counts + (r * T + tile) * 2, greater)
    tl.store(Counts + (r * T + tile) * 2 + 1, equal)


@tr.jit(do_not_specialize=["SS", "OS"])
def _threshold(S, O, Starts, Ends, Thresholds,
               SS, OS, K: tl.constexpr, HAS_START: tl.constexpr):
    r = tl.program_id(0)
    start = tl.load(Starts + r) if HAS_START else 0
    end = tl.load(Ends + r)
    ks = tl.arange(0, K)
    ids = tl.load(O + r * OS + ks)
    selected = tl.load(S + r * SS + start + ids,
                       (ids >= 0) & (start + ids < end), other=float('inf'))
    tl.store(Thresholds + r, tl.min(selected, 0))


@tr.jit(do_not_specialize=["SS", "N", "T"])
def _counts_thr(S, Starts, Ends, Counts, Thresholds,
                SS, N, T, HAS_START: tl.constexpr, B: tl.constexpr):
    r, tile = tl.program_id(0), tl.program_id(1)
    end = tl.load(Ends + r)
    if (tile > 0) & (tile * B >= end):
        return
    start = tl.load(Starts + r) if HAS_START else 0
    threshold = tl.load(Thresholds + r)
    cols = tile * B + tl.arange(0, B)
    valid = (cols < N) & (cols >= start) & (cols < end)
    scores = tl.load(S + r * SS + cols, valid, other=0.)
    greater = tl.sum((valid & (scores > threshold)).to(tl.int32), 0)
    equal = tl.sum((valid & (scores == threshold)).to(tl.int32), 0)
    tl.store(Counts + (r * T + tile) * 2, greater)
    tl.store(Counts + (r * T + tile) * 2 + 1, equal)


@tr.jit(do_not_specialize=["SS", "OS", "N", "T"])
def _scatter(S, O, Starts, Ends, Counts, Thresholds,
             SS, OS, N,
             K: tl.constexpr, T, TP: tl.constexpr,
             HAS_START: tl.constexpr, B: tl.constexpr):
    r, tile = tl.program_id(0), tl.program_id(1)
    start = tl.load(Starts + r) if HAS_START else 0
    end = tl.load(Ends + r)
    threshold = tl.load(Thresholds + r)
    tiles = tl.arange(0, TP)
    gt = tl.load(Counts + (r*T + tiles)*2, tiles < T, other=0)
    eq = tl.load(Counts + (r*T + tiles)*2+1, tiles < T, other=0)
    gt_total = tl.sum(gt, 0)
    gt_before = tl.sum(tl.where(tiles < tile, gt, 0), 0)
    eq_before = tl.sum(tl.where(tiles < tile, eq, 0), 0)
    tie_slots = tl.maximum(0, K - gt_total)
    cols = tile * B + tl.arange(0, B)
    valid = (cols < N) & (cols >= start) & (cols < end)
    scores = tl.load(S + r*SS + cols, valid, other=0.)
    equal = valid & (scores == threshold)
    tie_rank = eq_before + tl.cumsum(equal.to(tl.int32), 0)
    keep = valid & ((scores > threshold) | (equal & (tie_rank <= tie_slots)))
    offset = gt_before + tl.minimum(eq_before, tie_slots)
    dest = offset + tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.store(O + r*OS + dest, cols-start, keep & (dest < K))
    if tile == 0:
        ks = tl.arange(0, K)
        tl.store(O+r*OS+ks, -1, ks >= tl.minimum(K, end-start))


@tr.jit(do_not_specialize=["SS", "OS", "N", "T"])
def _scatter_skip(S, O, Starts, Ends, Counts, Thresholds,
                  SS, OS, N,
                  K: tl.constexpr, T, TP: tl.constexpr,
                  HAS_START: tl.constexpr, B: tl.constexpr):
    r, tile = tl.program_id(0), tl.program_id(1)
    end = tl.load(Ends + r)
    if (tile > 0) & (tile * B >= end):
        return
    start = tl.load(Starts + r) if HAS_START else 0
    threshold = tl.load(Thresholds + r)
    tiles = tl.arange(0, TP)
    live = tl.maximum(tl.minimum((end + B - 1) // B, T), 1)   # tiles that ran (tile 0 always runs)
    gt = tl.load(Counts + (r*T + tiles)*2, tiles < live, other=0)
    eq = tl.load(Counts + (r*T + tiles)*2+1, tiles < live, other=0)
    gt_total = tl.sum(gt, 0)
    gt_before = tl.sum(tl.where(tiles < tile, gt, 0), 0)
    eq_before = tl.sum(tl.where(tiles < tile, eq, 0), 0)
    tie_slots = tl.maximum(0, K - gt_total)
    cols = tile * B + tl.arange(0, B)
    valid = (cols < N) & (cols >= start) & (cols < end)
    scores = tl.load(S + r*SS + cols, valid, other=0.)
    equal = valid & (scores == threshold)
    tie_rank = eq_before + tl.cumsum(equal.to(tl.int32), 0)
    keep = valid & ((scores > threshold) | (equal & (tie_rank <= tie_slots)))
    offset = gt_before + tl.minimum(eq_before, tie_slots)
    dest = offset + tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.store(O + r*OS + dest, cols-start, keep & (dest < K))
    if tile == 0:
        ks = tl.arange(0, K)
        tl.store(O+r*OS+ks, -1, ks >= tl.minimum(K, end-start))


def canonicalize_topk(scores, starts, ends, output):
    """In-place, CUDA-graph-compatible canonicalization of exact selected values."""
    rows, n = scores.shape
    if rows == 0:
        return
    if not _SMALL_LOGGED[0]:
        _SMALL_LOGGED[0] = True
        print(f"GLM53_TOPK_SMALL_ROWS engaged: fused small path at <= {SMALL_ROWS} rows, threshold path above",
              file=sys.stderr, flush=True)
    k = output.shape[1]
    assert scores.stride(1) == output.stride(1) == 1
    assert k in (512, 1024, 2048)
    tiles = tr.cdiv(n, 1024)
    counts = torch.empty((rows, tiles, 2), device=scores.device, dtype=torch.int32)
    thresholds = torch.empty((rows,), device=scores.device, dtype=torch.float32)
    st = starts if starts is not None else ends
    args = (scores, output, st, ends, counts, thresholds, scores.stride(0), output.stride(0), n, k, tiles)
    if rows <= SMALL_ROWS:
        _counts[(rows, tiles)](*args, starts is not None, 1024, num_warps=4)
    else:
        _threshold[(rows,)](scores, output, st, ends, thresholds, scores.stride(0), output.stride(0), k,
                            starts is not None, num_warps=4)
        _counts_thr[(rows, tiles)](scores, st, ends, counts, thresholds, scores.stride(0), n, tiles,
                                   starts is not None, 1024, num_warps=4)
    scatter = _scatter if rows <= SMALL_ROWS else _scatter_skip
    scatter[(rows, tiles)](*args, tr.next_power_of_2(tiles), starts is not None, 1024, num_warps=4)
