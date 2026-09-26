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
"""
import torch
import triton as tr
import triton.language as tl

SMALL_ROWS = 16


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
    start = tl.load(Starts + r) if HAS_START else 0
    end = tl.load(Ends + r)
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


def canonicalize_topk(scores, starts, ends, output):
    """In-place, CUDA-graph-compatible canonicalization of exact selected values."""
    rows, n = scores.shape
    if rows == 0:
        return
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
    _scatter[(rows, tiles)](*args, tr.next_power_of_2(tiles), starts is not None, 1024, num_warps=4)
