# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): ported to GLM-5.3 with the per-token records in per-layer buffers
# indexed by spec-batch position instead of the KV page, an opt-in fused commit applied by the next verify
# (GLM53_RECOVERSSM_FUSED_COMMIT), and kernels that load and store fp16 or bf16 state pools with fp32 slow channels on
# the side (GLM53_KDA_STATE).
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): fused-commit ordering fix with host skip, exact-landing column.
"""GLM-5.3-Flash KDA RecoverSSM: Kimi-K3's speculative verify and accepted-state recovery
(vllm/models/kimi_k3/nvidia/ops/recoverssm.py in this image), adapted by the GLM-5.3 lab, 2026-09-23.

One change: the per-token records (v-correction fp32 [V], key + raw gate [2K]) live in per-layer buffers indexed by the
request's position in the spec batch, not in the mamba KV page indexed by its state block. GLM's mamba page (conv +
state, 1,171,456 B) is 8 KB under its attention page at block size 2304; the records (128 KB at 8 spec tokens) would
push vLLM to 4608-token blocks. Verify and commit of one step see the same spec batch order (both come from the step's
metadata), so the batch position addresses the same records in both. Everything else is Kimi's code."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# 2026-09-23 (outputs/2026-09-23-recoverssm-fused-commit): with GLM53_RECOVERSSM_FUSED_COMMIT=1 the accepted-state
# commit is not written after sampling; the next verify applies it in a prologue (the commit's arithmetic, verbatim),
# stores the committed state and verifies from registers, so each checkpoint is read once per step instead of twice.
# Pending commits are keyed by the block the commit targets, which is the block the request's next verify reads;
# anything a step's spec batch does not consume is flushed by the standalone commit before that step's forward.
FUSED_COMMIT = os.environ.get("GLM53_RECOVERSSM_FUSED_COMMIT", "0").strip() == "1"

# FP16 state with fp32 slow channels (kda-bf16-state lane, lever A2b, 2026-09-24; outputs/2026-09-24-kda-fp16-sr-state).
# GLM53_KDA_STATE=fp16-slow32 (kda.py) gives an fp16 state pool with SLOW_SLOTS extra fp32 values per value row: the state
# tensor is fp16 [H, V, K + 2 * SLOW_SLOTS], and the columns after K hold, as fp32, the exact values of each head's
# SLOW_SLOTS slowest key channels (static: the largest decays at raw gate 0, slow_slot_table). Every kernel here loads the
# fp16 main columns into fp32 registers, takes the slow channels from the fp32 side, computes in fp32, and stores the
# main columns rounded to nearest and the slow channels exactly. A2 measured that round-to-nearest stagnation in exactly
# those channels caused the fp16/bf16 drift. fp32 pools and narrow pools without the side: unchanged.
SLOW_SLOTS = 8
_SLOW_SLOT_CACHE: dict = {}


def side_layout(state: torch.Tensor, key_dim: int) -> bool:
    """True for the fp16 + fp32-side layout: an fp16 pool whose last dimension is key_dim + 2 * SLOW_SLOTS."""
    return state.dtype == torch.float16 and state.shape[-1] == key_dim + 2 * SLOW_SLOTS


def slow_slot_table(A_log, dt_bias, num_heads, head_dim, lower_bound, slots=SLOW_SLOTS):
    """[H, K] int32: the side slot (0..slots-1) of each head's `slots` slowest key channels, -1 elsewhere. Slowest = the
    largest decay at raw gate 0, exp(lower_bound * sigmoid(exp(A_log) * dt_bias)) with lower_bound < 0, ranked through
    its argument exp(A_log) * dt_bias (smallest first): the same order without the decay's saturation at 1.0 in fp32,
    so ties cannot make the set depend on the top-k implementation. Built once per layer (kda.py builds it at
    bind_kv_cache, before any CUDA graph capture) and cached by the parameters' data pointers (parameters live for the
    process)."""
    if lower_bound is None:
        raise ValueError("the fp32 slow channels need the bounded KDA gate")
    key = (A_log.data_ptr(), dt_bias.data_ptr(), int(num_heads), int(head_dim), float(lower_bound), int(slots))
    table = _SLOW_SLOT_CACHE.get(key)
    if table is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("slow_slot_table must be built before CUDA graph capture")
        if float(lower_bound) >= 0:
            raise ValueError("the fp32 slow channels need a negative KDA gate lower bound")
        A = A_log.detach().float().reshape(num_heads).exp()
        db = dt_bias.detach().float().reshape(num_heads, head_dim)
        top = torch.sort(torch.topk(-(A[:, None] * db), slots, dim=1).indices, dim=1).values
        table = torch.full((num_heads, head_dim), -1, dtype=torch.int32, device=A_log.device)
        table.scatter_(1, top, torch.arange(slots, dtype=torch.int32, device=A_log.device).expand(num_heads, slots).contiguous())
        _SLOW_SLOT_CACHE[key] = table
    return table


@triton.jit
def _state_load(row_ptr, offs_v, offs_k, mask_v, mask_k, stride_v, stride_k, slot_row_ptr, side_stride_v,
                K: tl.constexpr, SIDE: tl.constexpr):
    """A [BV, BK] fp32 tile of one (block, head) state at row_ptr. SIDE (A2b): the slow key channels come exact from the
    fp32 side columns that follow the K main columns of each value row."""
    mask = mask_v[:, None] & mask_k[None, :]
    x = tl.load(row_ptr + offs_v[:, None] * stride_v + offs_k[None, :] * stride_k, mask=mask, other=0.0).to(tl.float32)
    if SIDE:
        slot = tl.load(slot_row_ptr + offs_k, mask=mask_k, other=-1)
        slow = slot >= 0
        side = (row_ptr + K).to(tl.pointer_type(tl.float32))
        xs = tl.load(side + offs_v[:, None] * side_stride_v + tl.maximum(slot, 0)[None, :], mask=mask & slow[None, :],
                     other=0.0)
        x = tl.where(slow[None, :], xs, x)
    return x


@triton.jit
def _state_store(row_ptr, x, offs_v, offs_k, mask, mask_k, stride_v, stride_k, slot_row_ptr, side_stride_v,
                 K: tl.constexpr, SIDE: tl.constexpr):
    """Store a [BV, BK] fp32 tile under mask: the main columns through the pool's element type (round to nearest), and
    with SIDE the slow key channels exact into the fp32 side columns."""
    tl.store(row_ptr + offs_v[:, None] * stride_v + offs_k[None, :] * stride_k, x, mask=mask)
    if SIDE:
        slot = tl.load(slot_row_ptr + offs_k, mask=mask_k, other=-1)
        slow = slot >= 0
        side = (row_ptr + K).to(tl.pointer_type(tl.float32))
        tl.store(side + offs_v[:, None] * side_stride_v + tl.maximum(slot, 0)[None, :], x, mask=mask & slow[None, :])


@triton.jit
def _state_as_stored(x, slot_row_ptr, offs_k, mask_k, DST: tl.constexpr, SIDE: tl.constexpr):
    """The fp32 value that _state_store keeps for x on a DST pool: rounded to nearest, the slow channels exact (SIDE)."""
    r = x.to(DST).to(tl.float32)
    if SIDE:
        slot = tl.load(slot_row_ptr + offs_k, mask=mask_k, other=-1)
        r = tl.where((slot >= 0)[None, :], x, r)
    return r


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        return lower_bound * tl.sigmoid(A * gate_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        tl.log(1.0 + tl.exp(gate_input)),
    )
    return -A * softplus_gate


@triton.jit
def _kda_recurrent_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )

    state *= tl.exp(gate)[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    correction *= tl.sigmoid(raw_beta)
    return state + correction[:, None] * normalized_k[None, :], correction


@triton.jit
def _kda_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    slot_ptr,
    side_stride_v,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SIDE: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    state = _state_load(
        state_ptr + state_idx * stride_state_block + pid_h * stride_state_head,
        offs_v, offs_k, mask_v, mask_k, stride_state_v, stride_state_k, slot_ptr + pid_h * K, side_stride_v, K, SIDE,
    )
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))

    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=token_valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)

        q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
        dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        updated_state, correction = _kda_recurrent_step(
            state,
            k,
            v,
            raw_g,
            raw_beta,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        state = tl.where(token_valid, updated_state, state)

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=token_valid & mask_v,
        )

        correction_ptr = (
            correction_cache_ptr
            + pid_b * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask=token_valid & mask_v,
        )
        if pid_v == 0:
            kg_ptr = (
                kg_cache_ptr
                + pid_b * stride_kg_block
                + pid_h * stride_kg_head
                + token_offset * stride_kg_pos
            )
            tl.store(
                kg_ptr + offs_k * stride_kg_dim,
                k,
                mask=token_valid & mask_k,
            )
            tl.store(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                raw_g,
                mask=token_valid & mask_k,
            )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        # Exact-landing fix (outputs/2026-09-25-fused-commit-fix, section 4a): the state after (p + 1) * block_size
        # tokens belongs in column p, the image's align invariant, so an exact landing's final state goes to the
        # boundary block and kda.py hands the image helper num_computed - 1, which keeps the running column there.
        # Column final_num_computed // mamba_block_size is not allocated when every token of an untrimmed window is
        # accepted and the window ends on a boundary (the scheduler allocates cdiv(num_computed + 1 + K, block_size)).
        final_state_col = tl.minimum(
            (final_num_computed - 1) // mamba_block_size, block_table_width - 1
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_HISTORY)
    mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < conv_history_len)
    final_values = tl.load(
        source_ptr
        + offs_d[:, None] * dim_stride
        + (commit_len - 1 + offs_h[None, :]) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr
            + offs_d[:, None] * dim_stride
            + (boundary_recovery_len - 1 + offs_h[None, :]) * token_stride,
            mask=mask & (boundary_state_idx > null_block_id),
        )
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride
        tl.store(
            boundary_ptr
            + offs_d[:, None] * dim_stride
            + offs_h[None, :] * token_stride,
            boundary_values,
            mask=mask & (boundary_state_idx > null_block_id),
        )
    tl.store(
        final_ptr + offs_d[:, None] * dim_stride + offs_h[None, :] * token_stride,
        final_values,
        mask=mask,
    )


@triton.jit
def _commit_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    slot_ptr,
    side_stride_v,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SIDE: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        pid_b * correction_cache_block_stride + pid_h * stride_correction_cache_head
    )
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(
        tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty)
    )
    kg_cache_ptr += pid_b * kg_cache_block_stride + pid_h * stride_kg_cache_head

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    slot_row_ptr = slot_ptr + (pid_l * NUM_HEADS + pid_h) * K
    initial_state = _state_load(
        source_state_ptr, offs_v, offs_k, mask_v, mask_k, stride_state_v, stride_state_k, slot_row_ptr, side_stride_v,
        K, SIDE,
    )
    A = tl.exp(
        tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(
            tl.float32
        )
    )

    dt_bias = tl.load(
        dt_bias_ptr
        + pid_l * stride_dt_bias_layer
        + pid_h * stride_dt_bias_head
        + offs_k * stride_dt_bias_dim,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    final_decay = tl.full([BK], 1.0, tl.float32)
    final_correction = tl.zeros([BV, BK], tl.float32)
    boundary_decay = tl.full([BK], 1.0, tl.float32)
    boundary_correction = tl.zeros([BV, BK], tl.float32)

    for reverse_offset in range(commit_len):
        token_offset = commit_len - reverse_offset - 1
        correction_ptr = (
            correction_cache_ptr + token_offset * stride_correction_cache_pos
        )
        kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
        k = tl.load(
            kg_ptr + offs_k * stride_kg_cache_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        correction = tl.load(
            correction_ptr + offs_v * stride_correction_cache_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            kg_ptr + (K + offs_k) * stride_kg_cache_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        gate = _kda_gate(
            raw_g,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        update = correction[:, None] * normalized_k[None, :]
        decay = tl.exp(gate)
        final_correction += update * final_decay[None, :]
        final_decay *= decay
        if ALIGN_MODE:
            before_boundary = token_offset < boundary_recovery_len
            boundary_correction += tl.where(
                before_boundary,
                update * boundary_decay[None, :],
                0.0,
            )
            boundary_decay *= tl.where(before_boundary, decay, 1.0)

    state = initial_state * final_decay[None, :] + final_correction
    if ALIGN_MODE:
        boundary_state = initial_state * boundary_decay[None, :] + boundary_correction
        _state_store(
            state_ptr + boundary_state_idx * state_block_stride + pid_h * stride_state_head,
            boundary_state,
            offs_v, offs_k, mask_state & (boundary_state_idx > null_block_id), mask_k, stride_state_v, stride_state_k,
            slot_row_ptr, side_stride_v, K, SIDE,
        )

    _state_store(
        state_ptr + final_state_idx * state_block_stride + pid_h * stride_state_head,
        state,
        offs_v, offs_k, mask_state, mask_k, stride_state_v, stride_state_k, slot_row_ptr, side_stride_v, K, SIDE,
    )


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    # A2b: an fp16 pool with fp32 slow channels carries them in 2 * SLOW_SLOTS extra columns after the key columns.
    side = side_layout(checkpoint_state, key_dim)
    slot_table, side_stride_v = checkpoint_state, 0
    if side:
        slot_table = slow_slot_table(A_log, dt_bias, num_heads, key_dim, lower_bound)
        side_stride_v = checkpoint_state.stride(2) // 2
        checkpoint_state = checkpoint_state[..., :key_dim]
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    # Records are indexed by spec batch position: one row per request.
    num_rows = correction_cache.shape[0]
    expected_correction_shape = (num_rows, num_heads, spec_query_len, value_dim)
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(
            f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}"
        )
    expected_kg_shape = (num_rows, num_heads, spec_query_len, 2 * key_dim)
    if kg_cache.shape != expected_kg_shape:
        raise ValueError(
            f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}"
        )
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype:
        raise ValueError("KDA RecoverSSM key/gate buffer must match activation dtype")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if batch > num_rows:
        raise ValueError("KDA RecoverSSM spec batch exceeds its record rows")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_v), batch, num_heads)
    _kda_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        out,
        query_start_loc,
        state_indices,
        lower_bound or 0.0,
        NULL_BLOCK_ID,
        slot_table,
        side_stride_v,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        kg_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        SPEC_QUERY_LEN=spec_query_len,
        USE_LOWER_BOUND=lower_bound is not None,
        SIDE=side,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _kda_recoverssm_verify_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction2_ptr,
    kg2_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    wbuf_ptr,
    pend_entry_ptr,
    pend_src_ptr,
    pend_len_ptr,
    pend_final_ptr,
    pend_bound_ptr,
    pend_rec_ptr,
    lower_bound,
    null_block_id,
    slot_ptr,
    side_stride_v,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_buf,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_buf,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    ROUND_STATE: tl.constexpr,
    SIDE: tl.constexpr,
):
    """_kda_recoverssm_verify_kernel with a prologue that applies the previous step's pending commit (the arithmetic of
    _commit_kda_state_kernel, verbatim) for rows that have one, stores the committed state (and the boundary state in
    align mode), and verifies from it. Records are double-buffered: this step writes buffer wbuf, the pending records
    are in 1 - wbuf."""
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    wbuf = tl.load(wbuf_ptr).to(tl.int64)
    rbuf = 1 - wbuf
    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
        tl.float32
    )
    entry = tl.load(pend_entry_ptr + pid_b).to(tl.int64)
    if entry >= 0:
        # ---- the pending commit of the previous step (as _commit_kda_state_kernel) -----------------------------------
        source_state_idx = tl.load(pend_src_ptr + entry).to(tl.int64)
        commit_len = tl.load(pend_len_ptr + entry)
        final_state_idx = tl.load(pend_final_ptr + entry).to(tl.int64)
        boundary_state_idx = tl.load(pend_bound_ptr + entry).to(tl.int64)
        boundary_recovery_len = tl.load(pend_rec_ptr + entry)
        initial_state = _state_load(
            state_ptr + source_state_idx * stride_state_block + pid_h * stride_state_head,
            offs_v, offs_k, mask_v, mask_k, stride_state_v, stride_state_k, slot_ptr + pid_h * K, side_stride_v, K, SIDE,
        )
        rcorr_ptr = (
            correction2_ptr
            + rbuf * stride_correction_buf
            + entry * stride_correction_block
            + pid_h * stride_correction_head
        )
        rkg_ptr = kg2_ptr + rbuf * stride_kg_buf + entry * stride_kg_block + pid_h * stride_kg_head
        final_decay = tl.full([BK], 1.0, tl.float32)
        final_correction = tl.zeros([BV, BK], tl.float32)
        boundary_decay = tl.full([BK], 1.0, tl.float32)
        boundary_correction = tl.zeros([BV, BK], tl.float32)
        for reverse_offset in range(commit_len):
            token_offset = commit_len - reverse_offset - 1
            correction_ptr = rcorr_ptr + token_offset * stride_correction_pos
            kg_ptr = rkg_ptr + token_offset * stride_kg_pos
            k = tl.load(
                kg_ptr + offs_k * stride_kg_dim,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            correction = tl.load(
                correction_ptr + offs_v * stride_correction_dim,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            raw_g = tl.load(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
            gate = _kda_gate(
                raw_g,
                dt_bias,
                A,
                lower_bound,
                USE_LOWER_BOUND,
            )
            update = correction[:, None] * normalized_k[None, :]
            decay = tl.exp(gate)
            final_correction += update * final_decay[None, :]
            final_decay *= decay
            if ALIGN_MODE:
                before_boundary = token_offset < boundary_recovery_len
                boundary_correction += tl.where(
                    before_boundary,
                    update * boundary_decay[None, :],
                    0.0,
                )
                boundary_decay *= tl.where(before_boundary, decay, 1.0)
        state = initial_state * final_decay[None, :] + final_correction
        if ALIGN_MODE:
            boundary_state = initial_state * boundary_decay[None, :] + boundary_correction
            _state_store(
                state_ptr + boundary_state_idx * stride_state_block + pid_h * stride_state_head,
                boundary_state,
                offs_v, offs_k, mask_state & (boundary_state_idx > null_block_id), mask_k, stride_state_v,
                stride_state_k, slot_ptr + pid_h * K, side_stride_v, K, SIDE,
            )
        _state_store(
            state_ptr + final_state_idx * stride_state_block + pid_h * stride_state_head,
            state,
            offs_v, offs_k, mask_state, mask_k, stride_state_v, stride_state_k, slot_ptr + pid_h * K, side_stride_v,
            K, SIDE,
        )
        if SIDE:
            # A2b: verify from exactly the stored representation (main columns rounded, slow channels exact), which is
            # what the unfused path's next verify reads back; fused and unfused stay bit-identical.
            state = _state_as_stored(state, slot_ptr + pid_h * K, offs_k, mask_k, state_ptr.dtype.element_ty, SIDE)
        elif ROUND_STATE:
            # GLM53_KDA_STATE_BF16 (kda-bf16-state, 2026-09-24): on a narrower pool, verify from exactly the state just
            # stored (the unfused path's next verify reads it back from the pool), so the fused and unfused paths stay
            # bit-identical and this step's records match the state the next commit applies them to.
            state = state.to(state_ptr.dtype.element_ty).to(tl.float32)
    else:
        state = _state_load(
            state_ptr + state_idx * stride_state_block + pid_h * stride_state_head,
            offs_v, offs_k, mask_v, mask_k, stride_state_v, stride_state_k, slot_ptr + pid_h * K, side_stride_v, K, SIDE,
        )

    # ---- the verify (as _kda_recoverssm_verify_kernel), records into buffer wbuf ----------------------------------------
    wcorr_ptr = (
        correction2_ptr
        + wbuf * stride_correction_buf
        + pid_b * stride_correction_block
        + pid_h * stride_correction_head
    )
    wkg_ptr = kg2_ptr + wbuf * stride_kg_buf + pid_b * stride_kg_block + pid_h * stride_kg_head
    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=token_valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)

        q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
        updated_state, correction = _kda_recurrent_step(
            state,
            k,
            v,
            raw_g,
            raw_beta,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        state = tl.where(token_valid, updated_state, state)

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=token_valid & mask_v,
        )
        tl.store(
            wcorr_ptr + token_offset * stride_correction_pos + offs_v * stride_correction_dim,
            correction,
            mask=token_valid & mask_v,
        )
        if pid_v == 0:
            kg_ptr = wkg_ptr + token_offset * stride_kg_pos
            tl.store(
                kg_ptr + offs_k * stride_kg_dim,
                k,
                mask=token_valid & mask_k,
            )
            tl.store(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                raw_g,
                mask=token_valid & mask_k,
            )


@triton.jit
def _fused_prepare_kernel(
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    state_indices_ptr,
    pend_entry_ptr,
    flush_lens_ptr,
    null_block_id,
    num_rows,
    stride_state_indices,
):
    """Entry e of the previous commit is handed to the row of this step's spec batch whose state index is e's final
    block (pend_entry[row] = e), provided the commit writes no block-boundary state. Every other entry is flushed now,
    before the forward (flush_lens[e] = its commit length). That covers requests that left the batch or skipped the
    step, and commits that crossed a block boundary: their completed block may be read this step by another request's
    prefix-cache hit, and the runner's align postprocess may already have copied into it."""
    entry = tl.program_id(0)
    commit_len = tl.load(commit_lens_ptr + entry)
    if commit_len == 0:
        tl.store(flush_lens_ptr + entry, 0)
        return
    final_state_idx = tl.load(final_state_indices_ptr + entry)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + entry)
    found = -1
    for row in range(num_rows):
        idx = tl.load(state_indices_ptr + row * stride_state_indices)
        found = tl.where(idx == final_state_idx, row, found)
    if (found >= 0) & (boundary_state_idx <= null_block_id):
        tl.store(pend_entry_ptr + found, entry)
        tl.store(flush_lens_ptr + entry, 0)
    else:
        tl.store(flush_lens_ptr + entry, commit_len)


# Fused-commit ordering fix (2026-09-25, outputs/2026-09-25-fused-commit-fix). The served runner touches KDA state blocks
# between a commit and the next metadata build: the image's align postprocess copies the running column back into the
# completed block when a step lands exactly on a block boundary (v1/worker/mamba_utils.py:460-502), the next step
# pre-copies the running column into a new block before the build (model_runner.py:1609-1618, mamba_utils.py:505-633),
# prefix-cache hits pre-copy completed blocks, and the scheduler frees a finished or preempted request's blocks and hands
# its uncached running block out first (v1/core/block_pool.py:723-747). A deferred commit that any of these reach is read
# stale or lands in a block that has changed owner. Two rules keep the deferral only where nothing else can touch it:
#   rule 1 (commit time): an entry that writes a boundary state, or whose request's next verify window could reach the
#     next block, is committed at once, before the image helper and the align postprocess; the rest is deferred, and
#     its final block is the block the request's next verify reads in place;
#   rule 2 (step start): before any GPU work of the next step, the deferred entries of requests not scheduled in it
#     (finished, preempted, aborted, skipped) are committed (flush_absent_requests, called by the runner).
# Entries that remain are handed to their rows by prepare_fused as before.
# A2, the rule-1 host skip: in a pure spec-decode step (every active row verifies, request_indices None) kda.py knows
# every row's num_computed and query length on the host (exact under synchronous scheduling). When every row satisfies
# num_computed % block_size + query_len <= block_size - spec_query_len, no entry of the step's commit can cross a
# boundary or end within spec_query_len - 1 tokens of one, so rule 1 would commit nothing at once: kda.py then sets
# rule1_host_skip for this commit only, and the commit launches neither the split kernel nor the rule-1 commit. The
# entries' batch rows are then 0..n-1 (the split kernel would have stored exactly that in pend_row), so rule 2 reads
# row_ids instead of pend_row (pend_map). Same entries deferred, same bytes; two launches fewer in most decode steps.
@triton.heuristics({"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None})
@triton.jit
def _fused_split_kernel(
    commit_lens_ptr,
    boundary_state_indices_ptr,
    request_indices_ptr,
    num_computed_ptr,
    now_lens_ptr,
    pend_row_ptr,
    null_block_id,
    mamba_block_size,
    stride_request_indices,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    """Rule 1 for entry e of this step's commit: now_lens[e] = its commit length if it is committed now (and
    commit_lens[e] = 0, so nothing hands or flushes it again), else 0; pend_row[e] = the entry's batch row."""
    e = tl.program_id(0)
    row = e
    if HAS_REQUEST_INDICES:
        row = tl.load(request_indices_ptr + e * stride_request_indices).to(tl.int32)
    tl.store(pend_row_ptr + e, row)
    commit_len = tl.load(commit_lens_ptr + e)
    now = commit_len < 0
    if ALIGN_MODE:
        final_num_computed = tl.load(num_computed_ptr + row * stride_num_computed).to(tl.int32) + commit_len
        crosses = tl.load(boundary_state_indices_ptr + e) > null_block_id
        near_end = final_num_computed % mamba_block_size + SPEC_QUERY_LEN > mamba_block_size
        now = (commit_len > 0) & (crosses | near_end)
    tl.store(now_lens_ptr + e, tl.where(now, commit_len, 0))
    tl.store(commit_lens_ptr + e, tl.where(now, 0, commit_len))


@triton.jit
def _fused_absent_kernel(commit_lens_ptr, pend_row_ptr, keep_rows_ptr, flush_lens_ptr):
    """Rule 2 for entry e of the deferred commit: flush_lens[e] = its commit length (and commit_lens[e] = 0) if its
    request is not scheduled in the new step (keep_rows[pend_row[e]] == 0), else 0."""
    e = tl.program_id(0)
    commit_len = tl.load(commit_lens_ptr + e)
    keep = tl.load(keep_rows_ptr + tl.load(pend_row_ptr + e))
    flush = (commit_len > 0) & (keep == 0)
    tl.store(flush_lens_ptr + e, tl.where(flush, commit_len, 0))
    tl.store(commit_lens_ptr + e, tl.where(flush, 0, commit_len))


# The fused contexts of this process, for the runner's step-start flush (rule 2).
FUSED_CONTEXTS: list = []
_ORDER_LOGGED: set = set()


def _order_log_once(key: str, msg: str, *args) -> None:
    """One log line per process and key: the engagement markers of the ordering fix (switch receipts count them)."""
    if key not in _ORDER_LOGGED:
        _ORDER_LOGGED.add(key)
        from vllm.logger import init_logger

        init_logger(__name__).info(msg, *args)


def flush_absent_requests(prev_req_ids, scheduled_req_ids, new_req_ids) -> None:
    """Rule 2, called by the runner at the start of execute_model, before any GPU work of the step: commit the deferred
    entries whose request (prev_req_ids[row], the batch order of the step that made them) is not a continuing request
    scheduled in this step. Nothing is launched when every request of that step continues."""
    if not prev_req_ids:
        return
    contexts = [c for c in FUSED_CONTEXTS if c.pending_active]
    if not contexts:
        return
    keep = [int(r in scheduled_req_ids and r not in new_req_ids) for r in prev_req_ids]
    if all(keep):
        return
    keep_rows = torch.tensor(keep, dtype=torch.int32).pin_memory()
    for context in contexts:
        context.flush_absent(keep_rows.to(context.pend_row.device, non_blocking=True))
    _order_log_once("flush", "GLM53_RECOVERSSM_FUSED_ORDER flush_absent engaged: %d of %d rows absent, %d contexts",
                    len(keep) - sum(keep), len(keep), len(contexts))


@dataclass
class KDARecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    correction_caches: tuple[torch.Tensor, ...]
    correction_cache_base_addrs: torch.Tensor
    correction_cache_block_strides: torch.Tensor
    kg_caches: tuple[torch.Tensor, ...]
    kg_cache_base_addrs: torch.Tensor
    kg_cache_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float | None
    spec_query_len: int
    # Fused commit (FUSED_COMMIT): records are [2, rows, ...] per layer, buffer `write_buf` is written by the next
    # verify; the pending commit of the last step lives in the plan arrays above plus pend_src (its source blocks).
    fused: bool = False
    align_mode: bool = False
    records2: tuple | None = None
    record_base_addrs: tuple | None = None
    pend_entry: torch.Tensor | None = None
    pend_src: torch.Tensor | None = None
    pend_row: torch.Tensor | None = None
    row_ids: torch.Tensor | None = None       # A2: 0..max_num_reqs-1, the entry -> row map of a skipped split
    pend_map: torch.Tensor | None = None      # A2: the entry -> row map of the last commit (pend_row or row_ids)
    rule1_host_skip: bool = False             # A2: set by kda.py around one commit (no entry can need rule 1)
    flush_lens: torch.Tensor | None = None
    wbuf: torch.Tensor | None = None
    write_buf: int = 0
    pending_active: bool = False
    n_prev: int = 0
    # A2b: fp16 pool with fp32 slow channels (side_layout); the stacked [layers, H, K] slot tables; the key dimension
    # (the pool's last dimension is larger with the side); the side's value-row stride in fp32 elements.
    side: bool = False
    slow_slots: torch.Tensor | None = None
    key_dim: int = 0
    side_stride_v: int = 0

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        align_mode: bool = False,
    ) -> "KDARecoverSSMCommitContext":
        if not layers:
            raise ValueError("KDA RecoverSSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 2 for layer in layers):
            raise ValueError("GLM KDA RecoverSSM pages hold conv and state only")
        if any(len(layer.recoverssm_records) != 2 for layer in layers):
            raise ValueError("GLM KDA RecoverSSM layers need correction and key/gate records")

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        fused = layers[0].recoverssm_records[0].ndim == 5
        records2 = [tuple(layer.recoverssm_records) for layer in layers] if fused else None
        correction_caches = [
            layer.recoverssm_records[0][0] if fused else layer.recoverssm_records[0] for layer in layers
        ]
        kg_caches = [layer.recoverssm_records[1][0] if fused else layer.recoverssm_records[1] for layer in layers]
        A_log = [layer.A_log for layer in layers]
        dt_bias = [
            layer.dt_bias.view(layer.local_num_heads, layer.head_dim)
            for layer in layers
        ]
        lower_bounds = {layer.gate_lower_bound for layer in layers}
        if len(lower_bounds) != 1:
            raise ValueError("KDA RecoverSSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError("KDA RecoverSSM checkpoint must be four-dimensional")
        num_blocks, num_heads, value_dim, key_dim = state_ref.shape
        side = side_layout(state_ref, int(layers[0].head_dim))  # A2b: the key columns are the first head_dim
        if side:
            key_dim = int(layers[0].head_dim)
        for state in checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError(
                    "KDA RecoverSSM layers need matching checkpoint layout"
                )
        expected_correction_shape = (
            max_num_reqs,
            num_heads,
            spec_query_len,
            value_dim,
        )
        correction_ref = correction_caches[0]
        for correction_cache in correction_caches:
            if (
                correction_cache.shape != expected_correction_shape
                or correction_cache.dtype != torch.float32
                or correction_cache.device != state_ref.device
                or correction_cache.stride()[1:] != correction_ref.stride()[1:]
            ):
                raise ValueError(
                    "KDA RecoverSSM correction buffers need float32 shape "
                    f"{expected_correction_shape}"
                )
        expected_kg_shape = (max_num_reqs, num_heads, spec_query_len, 2 * key_dim)
        kg_ref = kg_caches[0]
        for kg_cache in kg_caches:
            if (
                kg_cache.shape != expected_kg_shape
                or kg_cache.dtype != kg_ref.dtype
                or kg_cache.device != state_ref.device
                or kg_cache.stride()[1:] != kg_ref.stride()[1:]
            ):
                raise ValueError(
                    f"KDA RecoverSSM key/gate buffers need shape {expected_kg_shape}"
                )
        if any(param.shape != (num_heads,) for param in A_log):
            raise ValueError("KDA RecoverSSM A_log shape is incompatible")
        if any(param.shape != (num_heads, key_dim) for param in dt_bias):
            raise ValueError("KDA RecoverSSM dt_bias shape is incompatible")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("KDA RecoverSSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != state_ref.device
                or conv_state.shape[0] != num_blocks
            ):
                raise ValueError("KDA RecoverSSM layers need matching conv state")

        device = state_ref.device

        def _base_addrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        def _block_strides(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.stride(0) for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        fused_fields = {}
        if fused:
            if any(r[0].shape[0] != 2 or r[1].shape[0] != 2 for r in records2):
                raise ValueError("fused KDA RecoverSSM records need two buffers")
            fused_fields = dict(
                fused=True,
                align_mode=align_mode,
                records2=tuple(records2),
                record_base_addrs=tuple(
                    (_base_addrs([r[0][b] for r in records2]), _base_addrs([r[1][b] for r in records2]))
                    for b in (0, 1)
                ),
                pend_entry=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
                pend_src=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
                pend_row=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
                row_ids=torch.arange(max_num_reqs, dtype=torch.int32, device=device),
                flush_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
                wbuf=torch.zeros(1, dtype=torch.int32, device=device),
            )
        side_fields = dict(side=side, key_dim=int(key_dim), side_stride_v=int(state_ref.stride(2) // 2) if side else 0)
        if side:
            lb = next(iter(lower_bounds))
            side_fields["slow_slots"] = torch.stack(
                [slow_slot_table(layer.A_log, layer.dt_bias, num_heads, key_dim, lb) for layer in layers]
            ).contiguous()
        context = cls(
            **fused_fields,
            **side_fields,
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states),
            conv_state_block_strides=_block_strides(conv_states),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints),
            state_block_strides=_block_strides(checkpoints),
            correction_caches=tuple(correction_caches),
            correction_cache_base_addrs=_base_addrs(correction_caches),
            correction_cache_block_strides=_block_strides(correction_caches),
            kg_caches=tuple(kg_caches),
            kg_cache_base_addrs=_base_addrs(kg_caches),
            kg_cache_block_strides=_block_strides(kg_caches),
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_recovery_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            A_log=torch.stack(tuple(A_log)).contiguous(),
            dt_bias=torch.stack(tuple(dt_bias)).contiguous(),
            lower_bound=lower_bounds.pop(),
            spec_query_len=spec_query_len,
        )
        if fused:
            FUSED_CONTEXTS.append(context)
            _order_log_once("on", "GLM53_RECOVERSSM_FUSED_ORDER on: boundary and near-end commits at once, absent "
                            "requests flushed at step start (outputs/2026-09-25-fused-commit-fix)")
        return context

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold accepted KDA and convolution inputs into every layer."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(
            arg is None for arg in align_args
        ):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "KDA RecoverSSM align block size must cover one speculative window"
            )
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA RecoverSSM block table must be two-dimensional")
        device = self.checkpoints[0].device
        if (
            any(
                tensor.device != device
                for tensor in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                )
            )
            or (request_indices is not None and request_indices.device != device)
            or (block_table is not None and block_table.device != device)
            or (
                num_computed_tokens is not None and num_computed_tokens.device != device
            )
        ):
            raise ValueError("KDA RecoverSSM commit inputs must be on the same device")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = (
            0 if num_computed_tokens is None else num_computed_tokens.stride(0)
        )

        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_stride,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=block_history,
            ALIGN_MODE=block_table is not None,
            num_warps=4,
        )

        if self.fused:
            # Defer the state commit: the next build's prepare_fused() hands each entry to the row that verifies from
            # its final block (the fused verify applies it) or flushes it. The verify of this step wrote its records
            # into buffer write_buf; the next verify writes the other one.
            if block_table is not None and not self.align_mode:
                raise ValueError("fused KDA RecoverSSM context was created without align mode")
            if self.rule1_host_skip and block_table is not None and request_indices is None:
                # A2: kda.py proved on the host that no entry can need rule 1 (see the fix's comment above).
                self.pend_map = self.row_ids
                _order_log_once("skip", "GLM53_RECOVERSSM_FUSED_ORDER rule-1 host skip engaged")
            else:
                # Rule 1 (fused-commit fix): commit now every entry that writes a boundary state or whose next verify
                # window could reach the next block; record every entry's batch row for rule 2.
                _fused_split_kernel[(batch,)](
                    self.commit_lens,
                    self.boundary_state_indices,
                    request_indices,
                    num_computed_tokens if num_computed_tokens is not None else self.commit_lens,
                    self.flush_lens,
                    self.pend_row,
                    NULL_BLOCK_ID,
                    mamba_block_size or 1,
                    request_indices.stride(0) if request_indices is not None else 0,
                    num_computed_stride,
                    SPEC_QUERY_LEN=self.spec_query_len,
                    ALIGN_MODE=block_table is not None,
                    num_warps=1,
                )
                if block_table is not None:
                    corr_addrs, kg_addrs = self.record_base_addrs[self.write_buf]
                    self._commit_state(batch, state_indices, True, self.flush_lens, corr_addrs, kg_addrs)
                self.pend_map = self.pend_row
            self.pend_src[:batch].copy_(state_indices[:batch])
            self.n_prev = batch
            self.pending_active = True
            self.write_buf = 1 - self.write_buf
            self.wbuf.fill_(self.write_buf)
            return

        self._commit_state(batch, state_indices, block_table is not None, self.commit_lens,
                           self.correction_cache_base_addrs, self.kg_cache_base_addrs)

    def _commit_state(self, batch, state_indices, align_mode, commit_lens, correction_base_addrs, kg_base_addrs):
        num_layers = len(self.checkpoints)
        state_ref = self.checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        key_dim = self.key_dim or key_dim  # A2b: the side columns follow the key columns
        block_k = triton.next_power_of_2(key_dim)
        block_v = min(triton.next_power_of_2(value_dim), 32)
        grid = (
            triton.cdiv(value_dim, block_v),
            batch,
            num_layers * num_heads,
        )
        _commit_kda_state_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.correction_caches[0],
            correction_base_addrs,
            self.correction_cache_block_strides,
            self.kg_caches[0],
            kg_base_addrs,
            self.kg_cache_block_strides,
            self.A_log,
            self.dt_bias,
            state_indices,
            commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.slow_slots if self.side else state_ref,
            self.side_stride_v,
            self.lower_bound or 0.0,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            self.correction_caches[0].stride(1),
            self.correction_caches[0].stride(2),
            self.correction_caches[0].stride(3),
            self.kg_caches[0].stride(1),
            self.kg_caches[0].stride(2),
            self.kg_caches[0].stride(3),
            self.A_log.stride(0),
            self.A_log.stride(1),
            self.dt_bias.stride(0),
            self.dt_bias.stride(1),
            self.dt_bias.stride(2),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_heads,
            USE_LOWER_BOUND=self.lower_bound is not None,
            ALIGN_MODE=align_mode,
            SIDE=self.side,
            num_warps=4,
            num_stages=2,
        )

    def flush_absent(self, keep_rows: torch.Tensor) -> None:
        """Rule 2 (fused-commit fix): commit the deferred entries whose batch row is 0 in keep_rows (the request is not
        scheduled in the new step), before any GPU work of that step."""
        if not self.pending_active:
            return
        n = self.n_prev
        pend_map = self.pend_map if self.pend_map is not None else self.pend_row
        _fused_absent_kernel[(n,)](self.commit_lens, pend_map, keep_rows, self.flush_lens, num_warps=1)
        corr_addrs, kg_addrs = self.record_base_addrs[1 - self.write_buf]
        self._commit_state(n, self.pend_src, self.align_mode, self.flush_lens, corr_addrs, kg_addrs)

    def prepare_fused(self, state_indices: torch.Tensor | None, num_rows: int) -> None:
        """Before a step's forward (the metadata build): map the last commit's entries to this step's spec rows
        (pend_entry) and flush every entry the fused verify will not apply."""
        self.pend_entry.fill_(-1)
        if not self.pending_active:
            return
        n = self.n_prev
        rows = state_indices if (state_indices is not None and num_rows > 0) else self.pend_src
        _fused_prepare_kernel[(n,)](
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            rows,
            self.pend_entry,
            self.flush_lens,
            NULL_BLOCK_ID,
            num_rows if state_indices is not None else 0,
            rows.stride(0),
            num_warps=1,
        )
        read_buf = 1 - self.write_buf
        corr_addrs, kg_addrs = self.record_base_addrs[read_buf]
        self._commit_state(n, self.pend_src, self.align_mode, self.flush_lens, corr_addrs, kg_addrs)
        self.pending_active = False


def kda_recoverssm_verify_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction2: torch.Tensor,
    kg2: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    context: "KDARecoverSSMCommitContext",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """kda_recoverssm_verify for a fused-commit context: applies the rows' pending commits (context.pend_entry) and
    writes this step's records into buffer context.wbuf of the [2, rows, ...] records."""
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if correction2.ndim != 5 or kg2.ndim != 5 or correction2.shape[0] != 2 or kg2.shape[0] != 2:
        raise ValueError("fused KDA RecoverSSM records need shape [2, rows, heads, spec, dim]")
    if correction2.shape[1:] != (context.pend_entry.shape[0], num_heads, spec_query_len, value_dim):
        raise ValueError("fused KDA RecoverSSM correction records are incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)) or v.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM q, k, v and gate heads must be contiguous")
    batch = state_indices.shape[0]
    if out is None:
        out = torch.empty_like(v)
    if total_tokens == 0:
        return out
    side = side_layout(checkpoint_state, key_dim)  # A2b
    slot_table, side_stride_v = checkpoint_state, 0
    if side:
        if not context.side:
            raise ValueError("KDA RecoverSSM: a side-layout pool needs a side-layout commit context")
        slot_table = slow_slot_table(A_log, dt_bias, num_heads, key_dim, lower_bound)
        side_stride_v = checkpoint_state.stride(2) // 2
        checkpoint_state = checkpoint_state[..., :key_dim]
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_v), batch, num_heads)
    _kda_recoverssm_verify_fused_kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction2,
        kg2,
        out,
        query_start_loc,
        state_indices,
        context.wbuf,
        context.pend_entry,
        context.pend_src,
        context.commit_lens,
        context.final_state_indices,
        context.boundary_state_indices,
        context.boundary_recovery_lens,
        lower_bound or 0.0,
        NULL_BLOCK_ID,
        slot_table,
        side_stride_v,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction2.stride(0),
        correction2.stride(1),
        correction2.stride(2),
        correction2.stride(3),
        correction2.stride(4),
        kg2.stride(0),
        kg2.stride(1),
        kg2.stride(2),
        kg2.stride(3),
        kg2.stride(4),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        SPEC_QUERY_LEN=spec_query_len,
        USE_LOWER_BOUND=lower_bound is not None,
        ALIGN_MODE=False,
        ROUND_STATE=checkpoint_state.dtype != torch.float32,
        SIDE=side,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["FUSED_COMMIT", "FUSED_CONTEXTS", "SLOW_SLOTS", "KDARecoverSSMCommitContext", "flush_absent_requests",
           "kda_recoverssm_verify", "kda_recoverssm_verify_fused", "side_layout", "slow_slot_table"]
