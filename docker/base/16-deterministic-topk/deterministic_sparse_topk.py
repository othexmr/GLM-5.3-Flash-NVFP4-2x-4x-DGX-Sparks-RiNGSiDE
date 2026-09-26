"""Canonicalize an exact top-k selection: lower indices win ties, ascending output.

The existing selector supplies only the kth score. Two fixed-order scans recover
the selected indices without relying on atomic arrival order. Prefill indices
are relative to each row's start, matching vLLM's native operation.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _counts(S, O, Starts, Ends, Counts, Thresholds,
            SS: tl.constexpr, OS: tl.constexpr, N: tl.constexpr,
            K: tl.constexpr, T: tl.constexpr, HAS_START: tl.constexpr,
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


@tr.jit
def _scatter(S, O, Starts, Ends, Counts, Thresholds,
             SS: tl.constexpr, OS: tl.constexpr, N: tl.constexpr,
             K: tl.constexpr, T: tl.constexpr, TP: tl.constexpr,
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
    args = (scores, output, starts if starts is not None else ends, ends,
            counts, thresholds, scores.stride(0), output.stride(0), n, k, tiles)
    _counts[(rows, tiles)](*args, starts is not None, 1024, num_warps=4)
    _scatter[(rows, tiles)](*args, tr.next_power_of_2(tiles), starts is not None, 1024, num_warps=4)
