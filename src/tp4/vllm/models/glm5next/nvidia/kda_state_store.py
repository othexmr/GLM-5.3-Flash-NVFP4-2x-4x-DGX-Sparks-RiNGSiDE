# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow storage of the GLM-5.3-Flash KDA recurrent state (kda-bf16-state lane, levers A2/A2b, 2026-09-24;
outputs/2026-09-24-kda-fp16-sr-state).

GLM53_KDA_STATE=fp16-slow32 (kda.py) makes the recurrent-state pool fp16 with each head's SLOW_SLOTS slowest key channels
kept exact in fp32 side columns (ops/recoverssm.py: side_layout, slow_slot_table). Storage only: every kernel loads the
state into fp32 registers, computes in fp32 and rounds only where it stores (the slow channels not at all). b12x's KDA
prefill accepts only an fp32 state pool, so each b12x prefill call runs on an fp32 staging copy of its rows:
Stage.stage_in converts the rows' states into staging slots 1..n (slot 0 is the null slot; with the side layout the
slow channels come from the side), b12x runs unchanged on the staging pool, and Stage.stage_out stores the final states
back (main columns rounded to nearest, slow channels exact). For the conv fold (glm53_kda_conv), whose conv-state
kernels index the conv pool with the same binding indices, the conv rows are staged too (a plain copy, bit-exact). One
staging pool per process and geometry, shared by every KDA layer: layers run one after another on one stream, and
prefill runs eager.

Composition with the KDA prefill checkpoints (kda-fp16 lane, 2026-09-24; outputs/2026-09-24-kda-fp16-compose): with
export_rows > 0 the staging pool also holds rows export_base .. export_base + export_rows - 1 for the states a checkpoint
recurrence (glm53_kda_ckpt) exports inside a chunk. export_checkpoints maps a call's checkpoint slots (pool blocks) to
those rows, one row per (sequence, checkpoint) entry, and stage_out_exports stores them into their pool blocks exactly as
stage_out stores a final state. With export_rows = 0 every launch is the one above.

Fused staging (a2b-replay-fix lane, 2026-09-24; outputs/2026-09-24-a2b-replay-fix). A b12x prefill call on the side
layout used to add 2 to 7 small launches per KDA layer (state rows in and out, conv rows in and out, the staged
checkpoint map, the exports out) on top of the 5 index launches every b12x call makes (initial indices copy, the
negated mask, the masked fill, the two counters). Short prefill forwards, such as a replay's recompute, are
launch-bound, so each of them paid it on all 34 KDA layers. stage_in_prefill does the state rows, the conv rows, the
b12x metadata buffers (initial indices, num_seqs, num_tokens) and the staged checkpoint map in ONE launch;
stage_out_prefill stores the final rows, their conv rows and the exported states in ONE launch. Every element takes
exactly the path it took before (the same _state_load / _state_store, plain copies for the conv rows, the same int32
index values), so the staging rows, the b12x inputs, the stored state bytes and the outputs are bit-identical to the
unfused launches. GLM53_KDA_STATE_FUSED_STAGING=0 selects the unfused launches (kda.py then runs its previous code).
"""

import os
import sys

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.models.glm5next.nvidia.ops.recoverssm import SLOW_SLOTS, _state_load, _state_store, side_layout

_BLOCK = 2048
_BV = 32
_MARKED: set[str] = set()
_STAGES: dict[tuple, "Stage"] = {}
_FUSED = os.environ.get("GLM53_KDA_STATE_FUSED_STAGING", "1").strip() != "0"


def mark(kind: str, detail="") -> None:
    """One stderr line per process and kind: GLM53_KDA_STATE_<kind> <detail> (serving engagement evidence). detail may
    be a callable, evaluated only for the first line (hot paths skip formatting it)."""
    if kind not in _MARKED:
        _MARKED.add(kind)
        detail = detail() if callable(detail) else detail
        print(f"GLM53_KDA_STATE_{kind} {detail}".rstrip(), file=sys.stderr, flush=True)


@triton.jit
def _stage_rows_kernel(
    src_ptr,
    dst_ptr,
    idx_ptr,
    stride_idx,
    src_row_stride,
    dst_row_stride,
    row_elems,
    null_block_id,
    TO_STAGING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat rows. TO_STAGING: pool[idx[r]] -> staging[r + 1]; else staging[r + 1] -> pool[idx[r]], skipping the null
    block. The store converts to the destination's element type (exact when widening, round to nearest when narrowing)."""
    r = tl.program_id(0)
    chunk = tl.program_id(1)
    blk = tl.load(idx_ptr + r * stride_idx).to(tl.int64)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_elems
    if TO_STAGING:
        x = tl.load(src_ptr + blk * src_row_stride + offs, mask=mask)
        tl.store(dst_ptr + (r + 1) * dst_row_stride + offs, x, mask=mask)
    else:
        if blk > null_block_id:
            x = tl.load(src_ptr + (r + 1) * src_row_stride + offs, mask=mask)
            tl.store(dst_ptr + blk * dst_row_stride + offs, x, mask=mask)


@triton.jit
def _stage_side_kernel(
    pool_ptr,
    stage_ptr,
    idx_ptr,
    slot_ptr,
    stride_idx,
    pool_block_stride,
    pool_head_stride,
    pool_v_stride,
    side_stride_v,
    stage_block_stride,
    null_block_id,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    TO_STAGING: tl.constexpr,
):
    """Side layout, one (row, head, value tile) per program. TO_STAGING: the fp16 main columns (slow channels from the
    fp32 side) -> staging[r + 1] (fp32 [H, V, K]); else staging[r + 1] -> the pool: main columns rounded to nearest,
    slow channels exact, skipping the null block."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    blk = tl.load(idx_ptr + r * stride_idx).to(tl.int64)
    offs_v = vb * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, K)
    mask_v = offs_v < V
    mask_k = offs_k < K
    mask = mask_v[:, None] & mask_k[None, :]
    row = pool_ptr + blk * pool_block_stride + h * pool_head_stride
    srow = stage_ptr + (r + 1) * stage_block_stride + h * (V * K) + offs_v[:, None] * K + offs_k[None, :]
    if TO_STAGING:
        x = _state_load(row, offs_v, offs_k, mask_v, mask_k, pool_v_stride, 1, slot_ptr + h * K, side_stride_v, K, True)
        tl.store(srow, x, mask=mask)
    else:
        if blk > null_block_id:
            x = tl.load(srow, mask=mask, other=0.0)
            _state_store(row, x, offs_v, offs_k, mask, mask_k, pool_v_stride, 1, slot_ptr + h * K, side_stride_v, K, True)


@triton.jit(do_not_specialize=["n_rows", "num_tokens", "n_exports"],
            do_not_specialize_on_alignment=["idx_ptr", "has_init_ptr", "ckpt_ptr"])
def _stage_in_fused_kernel(
    pool_ptr,
    stage_ptr,
    idx_ptr,
    slot_ptr,
    conv_pool_ptr,
    conv_stage_ptr,
    has_init_ptr,
    init_out_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    ckpt_ptr,
    export_out_ptr,
    stride_idx,
    stride_has,
    pool_block_stride,
    pool_head_stride,
    pool_v_stride,
    side_stride_v,
    stage_block_stride,
    conv_pool_stride,
    conv_stage_stride,
    conv_elems,
    n_rows,
    num_tokens,
    n_exports,
    export_base,
    null_block_id,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    NVB: tl.constexpr,
    H: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
    EXP_BLOCK: tl.constexpr,
):
    """One program per (row r, part p). p < H * NVB: _stage_side_kernel's TO_STAGING tile (h, vb) = divmod(p, NVB);
    p >= H * NVB: _stage_rows_kernel's TO_STAGING conv chunk p - H * NVB (the grid has such programs only when the call
    stages conv rows). Program (r, 0) also writes the b12x initial index of row r (its staged row r + 1, or the null
    block without an initial state: the old copy_ + masked_fill_); program (0, 0) writes num_seqs, num_tokens (the old
    fill_ launches) and the staged checkpoint map (export_checkpoints' torch.where: a real slot i becomes export_base + i,
    the null block stays)."""
    r = tl.program_id(0)
    p = tl.program_id(1)
    blk = tl.load(idx_ptr + r * stride_idx).to(tl.int64)
    if p < H * NVB:
        h = p // NVB
        vb = p % NVB
        offs_v = vb * BV + tl.arange(0, BV)
        offs_k = tl.arange(0, K)
        mask_v = offs_v < V
        mask_k = offs_k < K
        mask = mask_v[:, None] & mask_k[None, :]
        row = pool_ptr + blk * pool_block_stride + h * pool_head_stride
        srow = stage_ptr + (r + 1) * stage_block_stride + h * (V * K) + offs_v[:, None] * K + offs_k[None, :]
        x = _state_load(row, offs_v, offs_k, mask_v, mask_k, pool_v_stride, 1, slot_ptr + h * K, side_stride_v, K, True)
        tl.store(srow, x, mask=mask)
    else:
        offs = (p - H * NVB) * CONV_BLOCK + tl.arange(0, CONV_BLOCK)
        cmask = offs < conv_elems
        xc = tl.load(conv_pool_ptr + blk * conv_pool_stride + offs, mask=cmask)
        tl.store(conv_stage_ptr + (r + 1) * conv_stage_stride + offs, xc, mask=cmask)
    if p == 0:
        has = tl.load(has_init_ptr + r * stride_has)
        tl.store(init_out_ptr + r, tl.where(has != 0, r + 1, null_block_id))
        if r == 0:
            tl.store(num_seqs_ptr, n_rows)
            tl.store(num_tokens_ptr, num_tokens)
            if n_exports > 0:
                offs_e = tl.arange(0, EXP_BLOCK)
                me = offs_e < n_exports
                s = tl.load(ckpt_ptr + offs_e, mask=me, other=0)
                tl.store(export_out_ptr + offs_e, tl.where(s != null_block_id, export_base + offs_e, s), mask=me)


@triton.jit(do_not_specialize=["n_rows", "n_exports"],
            do_not_specialize_on_alignment=["idx_ptr", "ckpt_ptr"])
def _stage_out_fused_kernel(
    pool_ptr,
    stage_ptr,
    idx_ptr,
    slot_ptr,
    conv_pool_ptr,
    conv_stage_ptr,
    ckpt_ptr,
    stride_idx,
    pool_block_stride,
    pool_head_stride,
    pool_v_stride,
    side_stride_v,
    stage_block_stride,
    conv_pool_stride,
    conv_stage_stride,
    conv_elems,
    n_rows,
    n_exports,
    export_base,
    null_block_id,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    NVB: tl.constexpr,
    H: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
    EXP_BLOCK: tl.constexpr,
):
    """One program per (row r, part p). Rows r < n_rows are the call's final rows (stage_out): staging row r + 1 ->
    pool block idx[r]; rows n_rows + e are the exports (stage_out_exports): staging row export_base + e -> pool block
    ckpt[e]. A null (or negative) block is skipped. p < H * NVB: _stage_side_kernel's store tile (main columns rounded
    to nearest, slow channels exact); p >= H * NVB: _stage_rows_kernel's conv chunk of a final row. The unfused order
    wrote the exports after the finals, so a final block that also appears among the exports keeps the export (the
    checkpoint contract makes this impossible; the check keeps the order's result anyway)."""
    r = tl.program_id(0)
    p = tl.program_id(1)
    if r < n_rows:
        blk = tl.load(idx_ptr + r * stride_idx).to(tl.int64)
        srow_i = r + 1
    else:
        blk = tl.load(ckpt_ptr + (r - n_rows)).to(tl.int64)
        srow_i = export_base + (r - n_rows)
    if p < H * NVB:
        write = blk > null_block_id
        if r < n_rows:
            if n_exports > 0:
                offs_e = tl.arange(0, EXP_BLOCK)
                ex = tl.load(ckpt_ptr + offs_e, mask=offs_e < n_exports, other=null_block_id).to(tl.int64)
                write = write & (tl.sum((ex == blk).to(tl.int32), axis=0) == 0)
        if write:
            h = p // NVB
            vb = p % NVB
            offs_v = vb * BV + tl.arange(0, BV)
            offs_k = tl.arange(0, K)
            mask_v = offs_v < V
            mask_k = offs_k < K
            mask = mask_v[:, None] & mask_k[None, :]
            row = pool_ptr + blk * pool_block_stride + h * pool_head_stride
            srow = stage_ptr + srow_i * stage_block_stride + h * (V * K) + offs_v[:, None] * K + offs_k[None, :]
            x = tl.load(srow, mask=mask, other=0.0)
            _state_store(row, x, offs_v, offs_k, mask, mask_k, pool_v_stride, 1, slot_ptr + h * K, side_stride_v, K, True)
    else:
        if r < n_rows:
            if blk > null_block_id:
                offs = (p - H * NVB) * CONV_BLOCK + tl.arange(0, CONV_BLOCK)
                cmask = offs < conv_elems
                xc = tl.load(conv_stage_ptr + (r + 1) * conv_stage_stride + offs, mask=cmask)
                tl.store(conv_pool_ptr + blk * conv_pool_stride + offs, xc, mask=cmask)


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _row_elems(t: torch.Tensor) -> int:
    """Elements of one block of a pool view; every block must be contiguous inside."""
    if t.ndim < 2 or not t[0].is_contiguous():
        raise ValueError(f"KDA state staging needs rows contiguous inside a block, got strides {t.stride()}")
    return t[0].numel()


class Stage:
    """fp32 staging pool [max_seqs + 1 + export_rows, H, V, K] (+ conv rows for the max_seqs + 1 call rows) for b12x
    prefill on a narrow state pool."""

    def __init__(self, state_pool: torch.Tensor, max_seqs: int, conv_pool: torch.Tensor | None = None,
                 key_dim: int | None = None, export_rows: int = 0) -> None:
        device = state_pool.device
        self.max_seqs = int(max_seqs)
        self.seq_slots = self.max_seqs + 1              # null slot 0 + the call's rows 1..max_seqs
        self.export_rows = int(export_rows)
        self.export_base = self.seq_slots               # checkpoint exports: rows export_base .. slots - 1
        self.slots = self.seq_slots + self.export_rows  # the b12x plan's max_state_slots
        if NULL_BLOCK_ID != 0:
            raise ValueError("KDA state staging assumes the null block is block 0")
        if self.export_rows < 0:
            raise ValueError(f"KDA state staging: export_rows must be >= 0, got {export_rows}")
        self.key_dim = int(key_dim or state_pool.shape[-1])
        self.side = side_layout(state_pool, self.key_dim)
        heads, value_dim = state_pool.shape[1], state_pool.shape[2]
        self.state = torch.zeros((self.slots, heads, value_dim, self.key_dim), dtype=torch.float32, device=device)
        self.conv = (
            None
            if conv_pool is None
            else torch.zeros((self.seq_slots, *conv_pool.shape[1:]), dtype=conv_pool.dtype, device=device)
        )
        self.rows = torch.arange(1, self.seq_slots, dtype=torch.int32, device=device)
        self.export_ids = torch.arange(self.export_base, self.slots, dtype=torch.int32, device=device)
        # fused staging (stage_in_prefill / stage_out_prefill): the side layout only; the staged checkpoint map lives in
        # export_slots (the recurrence of the same layer call reads it on the same stream before the next call writes
        # it); absent conv rows / checkpoints get 1-element stand-ins of the same dtypes (one compiled kernel variant)
        self.fused = bool(self.side and _FUSED)
        self.export_slots = torch.zeros(max(1, self.export_rows), dtype=torch.int32, device=device)
        self._exp_block = _next_pow2(max(16, self.export_rows))   # the vector width of the map (masked to the entries)
        self._nvb = triton.cdiv(value_dim, _BV)
        self._state_programs = heads * self._nvb
        self._none_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        self._none_conv = torch.zeros(1, dtype=torch.bfloat16 if conv_pool is None else conv_pool.dtype, device=device)
        self._geom: dict[tuple, tuple] = {}

    def _check(self, indices: torch.Tensor, capacity: int | None = None) -> int:
        n = int(indices.shape[0])
        cap = self.max_seqs if capacity is None else int(capacity)
        if n > cap:
            raise ValueError(f"KDA state staging holds {cap} rows, got {n}")
        return n

    def _copy_flat(self, pool: torch.Tensor, staging: torch.Tensor, indices: torch.Tensor, to_staging: bool,
                   capacity: int | None = None) -> None:
        n = self._check(indices, capacity)
        if n == 0:
            return
        elems = _row_elems(pool)
        if staging[0].numel() != elems or pool.device != staging.device or indices.device != staging.device:
            raise ValueError("KDA state staging rows do not match the pool")
        src, dst = (pool, staging) if to_staging else (staging, pool)
        _stage_rows_kernel[(n, triton.cdiv(elems, _BLOCK))](
            src, dst, indices, indices.stride(0), src.stride(0), dst.stride(0), elems, NULL_BLOCK_ID,
            TO_STAGING=to_staging, BLOCK=_BLOCK, num_warps=4,
        )

    def _copy_state(self, pool: torch.Tensor, indices: torch.Tensor, to_staging: bool,
                    slot: torch.Tensor | None, staging: torch.Tensor | None = None,
                    capacity: int | None = None) -> None:
        """Rows of ``pool`` at ``indices`` to/from staging rows 1..n of ``staging`` (default: the staging pool; the export
        path passes a view whose row 1 is export_base)."""
        staging = self.state if staging is None else staging
        if not self.side:
            self._copy_flat(pool, staging, indices, to_staging, capacity)
            return
        n = self._check(indices, capacity)
        if n == 0:
            return
        if slot is None or tuple(slot.shape) != (pool.shape[1], self.key_dim) or slot.dtype != torch.int32:
            raise ValueError("KDA state staging on the side layout needs the layer's [H, K] int32 slot table")
        heads, value_dim = pool.shape[1], pool.shape[2]
        if pool.stride(3) != 1 or pool.stride(2) != self.key_dim + 2 * SLOW_SLOTS:
            raise ValueError(f"KDA state staging: unexpected side-layout strides {pool.stride()}")
        _stage_side_kernel[(n, heads, triton.cdiv(value_dim, _BV))](
            pool, staging, indices, slot, indices.stride(0), pool.stride(0), pool.stride(1), pool.stride(2),
            pool.stride(2) // 2, staging.stride(0), NULL_BLOCK_ID,
            V=value_dim, K=self.key_dim, BV=_BV, TO_STAGING=to_staging, num_warps=4,
        )

    def stage_in(
        self, state_pool: torch.Tensor, indices: torch.Tensor, conv_pool: torch.Tensor | None = None,
        slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows' states (and conv rows) into staging slots 1..n; returns (staging pool, staged indices 1..n)."""
        self._copy_state(state_pool, indices, True, slot)
        if conv_pool is not None:
            if self.conv is None:
                raise ValueError("KDA state staging was created without conv rows")
            self._copy_flat(conv_pool, self.conv, indices, True)
        return self.state, self.rows[: int(indices.shape[0])]

    def stage_out(
        self, state_pool: torch.Tensor, indices: torch.Tensor, conv_pool: torch.Tensor | None = None,
        slot: torch.Tensor | None = None,
    ) -> None:
        """Staging slots 1..n back into the rows' blocks: states rounded to the pool dtype (slow channels exact with
        the side layout), conv rows copied."""
        self._copy_state(state_pool, indices, False, slot)
        if conv_pool is not None:
            self._copy_flat(conv_pool, self.conv, indices, False)

    def export_checkpoints(self, ckpt, keep_conv: bool):
        """The call's checkpoint plan (glm53_kda_ckpt.Checkpoints; slots [seqs, m] = pool blocks or the null block) with
        every real slot replaced by its own staging export row, export_base + (s * m + c); null entries stay null, so
        the recurrence skips them as before. keep_conv keeps the conv-history fields (the unfused path writes the conv
        history into the conv pool, which it does not stage); without it they are dropped, and the caller writes the
        conv history into the conv pool after the call (the fused path stages the conv rows of the call only)."""
        slots = ckpt.slots
        n = int(slots.numel())
        if n > self.export_rows:
            raise ValueError(f"KDA state staging holds {self.export_rows} export rows, the call plans {n} entries")
        if not slots.is_contiguous() or slots.device != self.state.device:
            raise ValueError("KDA checkpoint slots must be contiguous and on the staging device")
        staged = torch.where(slots != NULL_BLOCK_ID, self.export_ids[:n].view(slots.shape).to(slots.dtype), slots)
        return type(ckpt)(slots=staged.contiguous(), offsets=ckpt.offsets, max_checkpoints=ckpt.max_checkpoints,
                          conv_rows=ckpt.conv_rows if keep_conv else None,
                          conv_slots=ckpt.conv_slots if keep_conv else None,
                          conv_cols=ckpt.conv_cols if keep_conv else None)

    def stage_out_exports(self, state_pool: torch.Tensor, ckpt, slot: torch.Tensor | None = None) -> None:
        """Store the exported states: staging row export_base + i -> pool block ckpt.slots.view(-1)[i] (null entries
        skipped by the kernels), as stage_out stores a final state (main columns rounded to nearest, slow channels
        exact with the side layout)."""
        flat = ckpt.slots.reshape(-1)
        if flat.numel() == 0:
            return
        view = self.state[self.export_base - 1:]       # its row r + 1 is staging row export_base + r
        self._copy_state(state_pool, flat, False, slot, staging=view, capacity=self.export_rows)

    # ---- fused staging (a2b-replay-fix) ----
    def _pool_geometry(self, pool: torch.Tensor, slot: torch.Tensor, conv_pool: torch.Tensor | None) -> tuple:
        """Validated launch constants of one layer's pools (the checks _copy_state / _copy_flat make on every call),
        cached per (pool, slot table, conv pool); the cache holds the tensors, so their ids stay theirs."""
        key = (id(pool), id(slot), id(conv_pool))
        hit = self._geom.get(key)
        if hit is not None:
            return hit[3]
        if not self.fused:
            raise ValueError("KDA state staging: the fused launches need the side layout")
        if slot is None or tuple(slot.shape) != (pool.shape[1], self.key_dim) or slot.dtype != torch.int32:
            raise ValueError("KDA state staging on the side layout needs the layer's [H, K] int32 slot table")
        if pool.device != self.state.device or tuple(pool.shape[1:3]) != tuple(self.state.shape[1:3]):
            raise ValueError("KDA state staging rows do not match the pool")
        if pool.stride(3) != 1 or pool.stride(2) != self.key_dim + 2 * SLOW_SLOTS:
            raise ValueError(f"KDA state staging: unexpected side-layout strides {pool.stride()}")
        conv = (self._none_conv, self._none_conv, 0, 0, 0, 0)
        if conv_pool is not None:
            if self.conv is None:
                raise ValueError("KDA state staging was created without conv rows")
            elems = _row_elems(conv_pool)
            if (self.conv[0].numel() != elems or conv_pool.device != self.conv.device
                    or conv_pool.dtype != self.conv.dtype):
                raise ValueError("KDA state staging rows do not match the pool")
            conv = (conv_pool, self.conv, conv_pool.stride(0), self.conv.stride(0), elems, triton.cdiv(elems, _BLOCK))
        geom = (pool.stride(0), pool.stride(1), pool.stride(2), pool.stride(2) // 2, conv)
        if len(self._geom) >= 1024:                   # serving passes the layers' persistent pools (34 entries)
            self._geom.clear()
        self._geom[key] = (pool, slot, conv_pool, geom)
        return geom

    def stage_in_prefill(
        self, state_pool: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor,
        initial_indices: torch.Tensor, num_seqs: torch.Tensor, num_tokens_buf: torch.Tensor, num_tokens: int,
        slot: torch.Tensor, conv_pool: torch.Tensor | None = None, ckpt=None, keep_conv: bool = True,
    ):
        """One launch for a b12x prefill call on the side layout: the rows' states (and conv rows) into staging rows
        1..n exactly as stage_in, initial_indices[r] = r + 1 with an initial state else the null block (the old
        copy_ + masked_fill_ of the staged indices), num_seqs = n and num_tokens (the old fill_ launches), and with a
        checkpoint plan the staged map export_checkpoints builds. Returns (staged indices 1..n, the staged plan or
        None); the plan keeps its conv fields with keep_conv, as export_checkpoints does."""
        n = self._check(indices)
        if n == 0:
            raise ValueError("KDA state staging: a prefill call needs at least one row")
        if not (has_initial_state.dtype == torch.bool and initial_indices.dtype == torch.int32
                and num_seqs.dtype == torch.int32 and num_tokens_buf.dtype == torch.int32
                and int(initial_indices.shape[0]) >= n and int(has_initial_state.shape[0]) >= n
                and indices.device == self.state.device):
            raise ValueError("KDA state staging: unexpected b12x metadata buffers")
        pbs, phs, pvs, ssv, conv = self._pool_geometry(state_pool, slot, conv_pool)
        cpool, cstage, cps, css, celems, cprog = conv
        n_exports, ckpt_slots = 0, self._none_i32
        if ckpt is not None:
            slots = ckpt.slots
            n_exports = int(slots.numel())
            if n_exports > self.export_rows:
                raise ValueError(f"KDA state staging holds {self.export_rows} export rows, the call plans {n_exports} entries")
            if not slots.is_contiguous() or slots.device != self.state.device or slots.dtype != torch.int32:
                raise ValueError("KDA checkpoint slots must be contiguous int32 on the staging device")
            ckpt_slots = slots
        _stage_in_fused_kernel[(n, self._state_programs + cprog)](
            state_pool, self.state, indices, slot, cpool, cstage,
            has_initial_state.view(torch.uint8), initial_indices, num_seqs, num_tokens_buf, ckpt_slots, self.export_slots,
            indices.stride(0), has_initial_state.stride(0), pbs, phs, pvs, ssv, self.state.stride(0), cps, css, celems,
            n, int(num_tokens), n_exports, self.export_base, NULL_BLOCK_ID,
            V=int(self.state.shape[2]), K=self.key_dim, BV=_BV, NVB=self._nvb, H=int(self.state.shape[1]),
            CONV_BLOCK=_BLOCK, EXP_BLOCK=self._exp_block, num_warps=4,
        )
        run_ckpt = None
        if ckpt is not None:
            run_ckpt = type(ckpt)(slots=self.export_slots[:n_exports].view(ckpt.slots.shape), offsets=ckpt.offsets,
                                  max_checkpoints=ckpt.max_checkpoints,
                                  conv_rows=ckpt.conv_rows if keep_conv else None,
                                  conv_slots=ckpt.conv_slots if keep_conv else None,
                                  conv_cols=ckpt.conv_cols if keep_conv else None)
        return self.rows[:n], run_ckpt

    def stage_out_prefill(self, state_pool: torch.Tensor, indices: torch.Tensor, slot: torch.Tensor,
                          conv_pool: torch.Tensor | None = None, ckpt=None) -> None:
        """One launch: stage_out (the final rows, and their conv rows with conv_pool) and stage_out_exports (the call's
        exported states into ckpt.slots' blocks), element by element as those two did."""
        n = self._check(indices)
        pbs, phs, pvs, ssv, conv = self._pool_geometry(state_pool, slot, conv_pool)
        cpool, cstage, cps, css, celems, cprog = conv
        n_exports, ckpt_slots = 0, self._none_i32
        if ckpt is not None:
            n_exports = int(ckpt.slots.numel())
            if n_exports > self.export_rows or not ckpt.slots.is_contiguous():
                raise ValueError("KDA state staging: the checkpoint plan does not fit the export rows")
            ckpt_slots = ckpt.slots
        if n + n_exports == 0:
            return
        _stage_out_fused_kernel[(n + n_exports, self._state_programs + cprog)](
            state_pool, self.state, indices, slot, cpool, cstage, ckpt_slots,
            indices.stride(0), pbs, phs, pvs, ssv, self.state.stride(0), cps, css, celems,
            n, n_exports, self.export_base, NULL_BLOCK_ID,
            V=int(self.state.shape[2]), K=self.key_dim, BV=_BV, NVB=self._nvb, H=int(self.state.shape[1]),
            CONV_BLOCK=_BLOCK, EXP_BLOCK=self._exp_block, num_warps=4,
        )


def get_stage(state_pool: torch.Tensor, max_seqs: int, conv_pool: torch.Tensor | None = None,
              key_dim: int | None = None, export_rows: int = 0) -> Stage:
    """The process-wide staging pool for this geometry (created on first use, then shared by every layer)."""
    key = (
        str(state_pool.device),
        int(max_seqs),
        tuple(state_pool.shape[1:]),
        state_pool.dtype,
        None if conv_pool is None else (tuple(conv_pool.shape[1:]), conv_pool.dtype),
        key_dim,
        int(export_rows),
    )
    stage = _STAGES.get(key)
    if stage is None:
        stage = _STAGES[key] = Stage(state_pool, max_seqs, conv_pool, key_dim, export_rows)
    return stage


__all__ = ["Stage", "get_stage", "mark"]
