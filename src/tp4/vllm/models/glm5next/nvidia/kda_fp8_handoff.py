# SPDX-License-Identifier: Apache-2.0
# KDA FP8 handoff through the mHC prefill-ownership all-gather (mhc-fusion lane, 2026-09-24). Default off.
# + GLM53_KDA_FP8_HANDOFF_RING (rdma-prefill lane, 2026-09-24, outputs/2026-09-24-ring-fp8-gather). Default off.
# + GLM53_KDA_FP8_HANDOFF_TUNED (prefill-levers lane, 2026-09-25, outputs/2026-09-25-prefill-levers). Default off.
"""Quantize a KDA layer's owner-row mHC pre output once, all-gather FP8 bytes + per-token scales, project from them.

Under TP4 prefill ownership (mhc_prefill_sharding, every eager forward of >= GLM53_MHC_PREFILL_MIN_ROWS rows) the mHC
pre runs on the owner quarter of the rows and its BF16 `layer_input` is all-gathered before attention. On a KDA layer
the only reader of that tensor is `in_proj_qkvbfg_a` (family kda_in of glm53_dual_fp8_dense), whose W8A8 prefill path
quantizes it per token (`ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)`) before `cutlass_scaled_mm`. With
GLM53_KDA_FP8_HANDOFF=1 the owner rows are quantized with that same kernel, the FP8 bytes and FP32 scales are
all-gathered instead of the BF16 rows (half the bytes), and the projection runs the exact calls of
`DualPathFp8DenseLinearMethod.apply`'s W8A8 branch on the gathered pair (same chunking: `dual_chunk_rows`, and
`dual_chunk_min_rows` when a chunk override sets it). The quant is row-local and the gather copies bytes, so the GEMM
inputs and outputs are bitwise identical to today's.

GLM53_KDA_FP8_HANDOFF_GATHER: `block` (default: one all-gather of a flat buffer per rank, [owner rows x H FP8][owner
rows x 4 scale bytes] padded to 16 bytes; then one GEMM call per owner block, whose rows are contiguous) or `two`
(the FP8 bytes and the scales as two all-gathers, then `apply`'s own chunk grid). A row-interleaved packing is not
offered: `cutlass_scaled_mm` accepts a row-strided FP8 A but does not return the contiguous result (ceiling-0139).
`block` changes the GEMM's row chunking to the owner blocks; the per-row result does not depend on the chunk (checked
bitwise in the leaf against `apply`). Engages only when the layer's in-projection is the dual-path FP8 method in w8a8
prefill mode with its FP8 weight present and no bias; otherwise the caller keeps the BF16 all-gather.

GLM53_KDA_FP8_HANDOFF_RING=1 (default 0): when the owned forward's collectives run on the two-direction RDMA prefill
ring (the ownership helper's `owner.rdma`, set by GLM53_MHC_PREFILL_RDMA for forwards of at least its threshold), the
gather rides that ring instead of PyNccl, whatever GLM53_KDA_FP8_HANDOFF_GATHER says. The ring moves only BF16
[rows, 4096] tensors, i.e. whole 8,192-byte rows, and its all-gather copies bytes in rank order, so each rank sends the
`block` buffer [owner rows x H FP8][owner rows x 4 scale bytes] padded to a whole number of ring rows (any owner row
count, odd included; at most 8,191 pad bytes, never read) and the projection runs `block`'s one GEMM per owner block.
One ring call per KDA layer carries the FP8 bytes and the scales. The gathered bytes equal PyNccl's gather of the same
buffers; the per-row GEMM result does not depend on the chunking (the handoff's leaf, and this lever's leaf, check it
bitwise against `two`). GLM53_KDA_FP8_HANDOFF_RING_MIN_ROWS (default 4,608 full rows): smaller owned forwards keep the
configured PyNccl gather, where 4 owner blocks of fewer than ~1,152 rows make the per-block GEMMs cost what the ring
saves (leaf 2 of outputs/2026-09-24-ring-fp8-gather). Fallback to the configured PyNccl gather, per forward: the
selector off, fewer rows than that threshold, no ring on this forward (`owner.rdma` None: below the ring's own
threshold, or a helper without the ring), or a buffer larger than the ring's maximum (not reachable at H = 4,096 and
owner rows <= 3,584).

GLM53_KDA_FP8_HANDOFF_TUNED=1 (default 0): `project` sends each of its GEMM calls through the dense module's own tuned
SM120 configuration choice (`glm53_dual_fp8_dense._fp8_gemm_choice`, the GLM53_DENSE_FP8_GEMM table with
GLM53_FP8_GEMM_SO), exactly as `DualPathFp8DenseLinearMethod.apply` does for the family's calls of the same row count:
per owner block for `block` and the ring, per chunk and for the unchunked call for `two`. Unset, every call runs
`_C.cutlass_scaled_mm` as before. The configurations keep vLLM's MMA atom, ascending k order and epilogue, so the
product is bitwise the served kernel's (outputs/2026-09-24-dense-fp8-kernels, and this lever's leaf at the handoff's
own block and chunk sizes). A call with no matching rule (no table, or at most 128 rows) runs the served kernel.
GLM53_KDA_FP8_HANDOFF_TUNED_MIN_ROWS (default 1,536): a call of fewer rows keeps the served kernel. Leaf H round 1
(outputs/2026-09-25-prefill-levers) measured the table's p128x128x64 choice slower than the served kernel on the
handoff's 1,152-1,281-row owner blocks (4,608-5,121-row forwards: 1.011-1.042) and faster from 1,728 rows on.
"""
from __future__ import annotations

import logging
import os

import torch

_LOG = logging.getLogger(__name__)
SELECTOR = "GLM53_KDA_FP8_HANDOFF"
GATHER_ENV = "GLM53_KDA_FP8_HANDOFF_GATHER"
_RAW = os.environ.get(SELECTOR, "0")
GATHER = os.environ.get(GATHER_ENV, "block")
if _RAW not in ("0", "1") or GATHER not in ("two", "block"):
    raise RuntimeError(f"invalid {SELECTOR}={_RAW!r} / {GATHER_ENV}={GATHER!r}")
ENABLED = _RAW == "1"
MIN_ROWS_ENV = "GLM53_KDA_FP8_HANDOFF_MIN_ROWS"
MIN_ROWS = int(os.environ.get(MIN_ROWS_ENV, "0"))   # full rows of the owned forward; 0 = every owned forward
if MIN_ROWS < 0:
    raise RuntimeError(f"invalid {MIN_ROWS_ENV}={MIN_ROWS}")
BLOCK_ALIGN = 16         # each rank's gathered block starts 16-byte aligned
_REPORTED: set = set()
if ENABLED:
    _LOG.warning("GLM53_KDA_FP8_HANDOFF_READY gather=%s min_rows=%d default_off=1", GATHER, MIN_ROWS)
RING_ENV = "GLM53_KDA_FP8_HANDOFF_RING"
_RING_RAW = os.environ.get(RING_ENV, "0")
if _RING_RAW not in ("0", "1"):
    raise RuntimeError(f"invalid {RING_ENV}={_RING_RAW!r}")
RING = _RING_RAW == "1"
RING_ROW_BYTES = 8192    # one ring row: the prefill ring moves BF16 [rows, 4096] only
RING_MIN_ROWS_ENV = "GLM53_KDA_FP8_HANDOFF_RING_MIN_ROWS"
RING_MIN_ROWS = int(os.environ.get(RING_MIN_ROWS_ENV, "4608"))   # full rows of the owned forward
if RING_MIN_ROWS < 0:
    raise RuntimeError(f"invalid {RING_MIN_ROWS_ENV}={RING_MIN_ROWS}")
if ENABLED and RING:
    _LOG.warning("GLM53_KDA_FP8_HANDOFF_RING_READY layout=block%d min_rows=%d default_off=1", RING_ROW_BYTES,
                 RING_MIN_ROWS)
TUNED_ENV = "GLM53_KDA_FP8_HANDOFF_TUNED"
_TUNED_RAW = os.environ.get(TUNED_ENV, "0")
if _TUNED_RAW not in ("0", "1"):
    raise RuntimeError(f"invalid {TUNED_ENV}={_TUNED_RAW!r}")
TUNED = _TUNED_RAW == "1"
TUNED_MIN_ROWS_ENV = "GLM53_KDA_FP8_HANDOFF_TUNED_MIN_ROWS"
TUNED_MIN_ROWS = int(os.environ.get(TUNED_MIN_ROWS_ENV, "1536"))   # rows of one GEMM call
if TUNED_MIN_ROWS < 129:
    raise RuntimeError(f"invalid {TUNED_MIN_ROWS_ENV}={TUNED_MIN_ROWS} (the table starts above 128 rows)")
_TUNED_REPORTED: set = set()
if ENABLED and TUNED:
    _LOG.warning("GLM53_KDA_FP8_HANDOFF_TUNED_READY rules=GLM53_DENSE_FP8_GEMM min_rows=%d default_off=1", TUNED_MIN_ROWS)


def eligible(attn) -> bool:
    """True when the KDA attention's in-projection is the dual-path FP8 method in its W8A8 prefill form (cached)."""
    cached = getattr(attn, "_glm53_kda_fp8_handoff_ok", None)
    if cached is not None:
        return cached
    ok = False
    try:
        layer = attn.in_proj_qkvbfg_a
        method = layer.quant_method
        ok = (type(method).__name__ == "DualPathFp8DenseLinearMethod" and getattr(method, "family", None) == "kda_in"
              and getattr(method, "prefill_mode", None) == "w8a8" and getattr(layer, "dual_w8", None) is not None
              and getattr(layer, "dual_w8_scale", None) is not None and layer.bias is None
              and not getattr(layer, "gather_output", False) and layer.dual_w8.dtype == torch.float8_e4m3fn)
    except AttributeError:
        ok = False
    attn._glm53_kda_fp8_handoff_ok = ok
    return ok


def engage(attn, owner) -> bool:
    """The caller's gate: selector on, an owned forward of >= MIN_ROWS rows, an eligible KDA in-projection."""
    return ENABLED and owner.rows >= MIN_ROWS and eligible(attn)


def gather(owner, x: torch.Tensor):
    """Owner-row BF16 [owner_rows, H] -> the full rows' FP8 input for `project`, one accounted gather: `two` returns
    (FP8 [rows, H], FP32 scales [rows, 1]); `block` returns ([(first row, FP8 block, scale block), ...], rows)."""
    from vllm import _custom_ops as ops

    if not owner.comm.available or owner.comm.disabled:
        raise RuntimeError("mHC communicator became unavailable; no local fallback")
    q_rows, hidden = x.shape
    if (q_rows != owner.owner_rows or x.device != owner.comm.device or x.dtype != torch.bfloat16
            or not x.is_contiguous()):
        raise RuntimeError("KDA FP8 handoff requires contiguous BF16 owner rows")
    stream = torch.cuda.current_stream(x.device) if x.is_cuda else None     # CPU only in the unit tests
    ring, ring_rows = (getattr(owner, "rdma", None) if RING and owner.rows >= RING_MIN_ROWS else None), 0
    if ring is not None:
        ring_rows = -(-(q_rows * hidden + 4 * q_rows) // RING_ROW_BYTES)
        if 4 * ring_rows > ring.max_rows:                  # the ring's region cannot hold it: PyNccl for this forward
            ring = None
    transport = "ring" if ring is not None else "pynccl"
    if ring is not None:
        # the `block` buffer padded to whole ring rows, viewed as BF16 [ring rows, 4096]; one ring all-gather
        fp8_bytes, used, blk = q_rows * hidden, q_rows * hidden + 4 * q_rows, ring_rows * RING_ROW_BYTES
        send = torch.empty(blk, dtype=torch.uint8, device=x.device)
        q, s = ops.scaled_fp8_quant(x, output=send[:fp8_bytes].view(q_rows, hidden).view(torch.float8_e4m3fn),
                                    use_per_token_if_dynamic=True)
        send[fp8_bytes:used].copy_(s.view(torch.uint8).view(-1))
        recv = torch.empty(4 * blk, dtype=torch.uint8, device=x.device)
        width = RING_ROW_BYTES // 2
        ring.all_gather(recv.view(torch.bfloat16).view(4 * ring_rows, width), send.view(torch.bfloat16).view(ring_rows, width))
        blocks = []
        for b in range(4):
            lo, base = b * q_rows, b * blk
            n = min(q_rows, owner.rows - lo)
            if n <= 0:
                break
            qb = recv[base:base + fp8_bytes].view(q_rows, hidden)[:n].view(torch.float8_e4m3fn)
            sb = recv[base + fp8_bytes:base + used].view(torch.float32).view(q_rows, 1)[:n]
            blocks.append((lo, qb, sb))
        q_full, s_full = blocks, owner.rows
    elif GATHER == "two":
        q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        q_out = torch.empty((owner.padded_rows, hidden), dtype=torch.uint8, device=x.device)
        s_out = torch.empty((owner.padded_rows, 1), dtype=torch.float32, device=x.device)
        owner.comm.all_gather(q_out, q.view(torch.uint8), stream=stream)
        owner.comm.all_gather(s_out, s, stream=stream)
        for t in (q, s, q_out, s_out) if stream is not None else ():
            t.record_stream(stream)
        q_full, s_full = q_out[: owner.rows].view(torch.float8_e4m3fn), s_out[: owner.rows]
    else:
        fp8_bytes = q_rows * hidden
        blk = -(-(fp8_bytes + 4 * q_rows) // BLOCK_ALIGN) * BLOCK_ALIGN
        send = torch.empty(blk, dtype=torch.uint8, device=x.device)
        q, s = ops.scaled_fp8_quant(x, output=send[:fp8_bytes].view(q_rows, hidden).view(torch.float8_e4m3fn),
                                    use_per_token_if_dynamic=True)
        send[fp8_bytes:fp8_bytes + 4 * q_rows].copy_(s.view(torch.uint8).view(-1))
        recv = torch.empty(4 * blk, dtype=torch.uint8, device=x.device)
        owner.comm.all_gather(recv, send, stream=stream)
        for t in (s, send, recv) if stream is not None else ():
            t.record_stream(stream)
        blocks = []
        for b in range(4):
            lo, base = b * q_rows, b * blk
            n = min(q_rows, owner.rows - lo)
            if n <= 0:
                break
            qb = recv[base:base + fp8_bytes].view(q_rows, hidden)[:n].view(torch.float8_e4m3fn)
            sb = recv[base + fp8_bytes:base + fp8_bytes + 4 * q_rows].view(torch.float32).view(q_rows, 1)[:n]
            blocks.append((lo, qb, sb))
        q_full, s_full = blocks, owner.rows
    owner.ag_count += 1          # the layer's one input gather (PrefillOwnership.finish checks the count)
    if RING:
        key = (owner.rows, transport)
        if key not in _REPORTED and len(_REPORTED) < 64:
            _REPORTED.add(key)
            _LOG.warning("GLM53_KDA_FP8_HANDOFF rows=%d owner_rows=%d gather=%s transport=%s ring_rows=%d", owner.rows,
                         owner.owner_rows, "ring-block" if transport == "ring" else GATHER, transport, ring_rows)
        return q_full, s_full
    key = owner.rows
    if key not in _REPORTED and len(_REPORTED) < 64:
        _REPORTED.add(key)
        _LOG.warning("GLM53_KDA_FP8_HANDOFF rows=%d owner_rows=%d gather=%s", owner.rows, owner.owner_rows, GATHER)
    return q_full, s_full


def _tuned(layer, rows: int):
    """GLM53_KDA_FP8_HANDOFF_TUNED: (extension module, config index, swizzle, raster) as the dense module's rule table
    picks them for a W8A8 prefill GEMM of this layer's family with `rows` rows (`_fp8_gemm_choice`, the call `apply`
    makes), or None: the call then runs the served kernel (also below TUNED_MIN_ROWS rows)."""
    from vllm.model_executor.layers.quantization import glm53_dual_fp8_dense as dense

    if rows < TUNED_MIN_ROWS or not dense.FP8_GEMM_RULES:
        return None
    cfg = dense._fp8_gemm_choice(layer.quant_method.family, rows, layer.dual_fp8_N, layer.dual_fp8_K, None)
    if cfg is None:
        return None
    if rows not in _TUNED_REPORTED and len(_TUNED_REPORTED) < 64:
        _TUNED_REPORTED.add(rows)
        name = next((n for n, i in dense._FP8_GEMM[1].items() if i == cfg[0]), str(cfg[0]))
        _LOG.warning("GLM53_KDA_FP8_HANDOFF_TUNED engaged rows=%d config=%s swizzle=%d raster=%d", rows, name, cfg[1],
                     cfg[2])
    return (dense._FP8_GEMM[0], *cfg)


def project(layer, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """DualPathFp8DenseLinearMethod.apply's W8A8 prefill branch (bias None) on a pre-quantized input: for `two` the same
    chunk rule, ops and order; for `block` the same op per owner block. With GLM53_KDA_FP8_HANDOFF_TUNED each call
    takes apply's GEMM configuration choice for its row count (`_tuned`)."""
    from vllm import _custom_ops as ops

    N = layer.dual_fp8_N
    w8t = layer.dual_w8.t()
    if isinstance(q, list):                 # `block`: one call per gathered owner block (contiguous rows)
        out = torch.empty((s, N), dtype=torch.bfloat16, device=q[0][1].device)
        for lo, qb, sb in q:
            t = _tuned(layer, qb.shape[0]) if TUNED else None
            if t is not None:
                t[0].gemm(out[lo : lo + qb.shape[0]], qb, w8t, sb, layer.dual_w8_scale, *t[1:])
            else:
                torch.ops._C.cutlass_scaled_mm(out[lo : lo + qb.shape[0]], qb, w8t, sb, layer.dual_w8_scale, None)
        return out
    M = q.shape[0]
    chunk = layer.dual_chunk_rows
    if chunk and M > chunk and M >= getattr(layer, "dual_chunk_min_rows", 0):
        out = torch.empty((M, N), dtype=torch.bfloat16, device=q.device)
        for i in range(0, M, chunk):
            qs = q[i : i + chunk]
            t = _tuned(layer, qs.shape[0]) if TUNED else None
            if t is not None:
                t[0].gemm(out[i : i + qs.shape[0]], qs, w8t, s[i : i + chunk], layer.dual_w8_scale, *t[1:])
            else:
                torch.ops._C.cutlass_scaled_mm(out[i : i + qs.shape[0]], qs, w8t, s[i : i + chunk], layer.dual_w8_scale, None)
        return out
    t = _tuned(layer, M) if TUNED else None
    if t is not None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=q.device)
        t[0].gemm(out, q, w8t, s, layer.dual_w8_scale, *t[1:])
        return out
    return ops.cutlass_scaled_mm(q, w8t, s, layer.dual_w8_scale, torch.bfloat16, None)
