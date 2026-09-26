# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Contains code from b12x (local-inference-lab/b12x, Apache-2.0): _PrepareConvKernel, _prepare_conv_key,
# _compile_prepare_conv and run_prefill_conv_fused derive from b12x/sequence/kda_prefill/_cute_kernels.py.
# run_kda_prefill_conv_fused follows the b12x KDA prefill launcher (_run_b12x_kda_prefill) of the GLM-5.3 backport's
# vllm/models/glm5next/nvidia/kda.py (vllm-project/vllm, Apache-2.0).
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): the short causal conv folded into b12x's KDA prepare kernel, plus a
# conv-state update kernel and conv-fused prefill entry points.
"""KDA prefill with the short causal conv folded into b12x's prepare pass (flashkda lane, 2026-09-24).

Today (kda.py, pure-prefill step): ``causal_conv1d_fn`` reads the q|k|v columns of the merged projection
(``[T, 6144]``, row stride 6416), writes a new ``[T, 6144]`` activation, updates the conv state, and b12x's prepare
kernel reads q and k from it while the recurrence kernel reads v. Here the prepare kernel (one CTA per 16-token tile
and head, as b12x's) stages the pre-conv q, k and v rows of its tile plus a three-row halo (the previous rows, or the
conv state for a sequence's first tile, or zeros), computes the depthwise 4-tap conv and SiLU with the conv kernel's
exact instruction sequence, feeds q and k into b12x's unchanged prepare math, and writes post-conv v into a
``[T, H*128]`` buffer that b12x's unchanged recurrence kernel reads. A small kernel then updates the conv state
exactly as the conv kernel does (the last three input rows of each sequence, shifted with the old state for chunks
shorter than three tokens; columns 3.. untouched). The full-size post-conv q/k round trip disappears; v keeps one
write and one read.

Everything outside the staging and conv block of ``_PrepareConvKernel.kernel`` is b12x 1.3.0's prepare kernel
(``b12x/sequence/kda_prefill/_cute_kernels.py`` sha256 1580ca67), unchanged; the prologue and recurrence launches are
b12x's own. Compile keys carry this file's sha256, so a changed kernel never loads a stale cubin from the persistent
b12x compile cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_trans_b16,
    shared_ptr_to_u32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from b12x.sequence.kda_prefill import _cute_kernels as K
from b12x.sequence.kda_prefill._policy import WorkspaceRecord as REC

_HEAD_DIM = 128
_CHUNK = 16
_THREADS = 128
_WIDTH = 4
_HALO = _WIDTH - 1
_XROWS = _CHUNK + _HALO
_LOG2E = 1.4426950408889634
_SOURCE_TAG = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

_PREPARE_CONV_CACHE: dict[tuple, object] = {}
_STATE_CACHE: dict[tuple, object] = {}


@dsl_user_op
def _conv_silu(a0: Float32, a1: Float32, a2: Float32, a3: Float32, w0: Float32, w1: Float32, w2: Float32,
               w3: Float32, *, loc=None, ip=None) -> Float32:
    """One output of the conv kernel: fp32 taps oldest first, then silu = acc / (1 + exp(-acc)).

    The instruction sequence mirrors the Triton ``_causal_conv1d_fwd_kernel`` PTX (see the README's receipt), so the
    bf16 result is bit-identical: fma taps in order, exp as ex2.approx of acc * -log2(e), full-range fp32 division."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, w0, w1, w2, w3)],
            _CONV_SILU_PTX,
            "=f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Filled from the Triton conv kernel's PTX (ceiling job receipt); see README section 3.
_CONV_SILU_PTX = (
    "{ .reg .f32 acc, t, e, d;"
    " fma.rn.f32 acc, $1, $5, 0f00000000;"
    " fma.rn.f32 acc, $2, $6, acc;"
    " fma.rn.f32 acc, $3, $7, acc;"
    " fma.rn.f32 acc, $4, $8, acc;"
    " mul.f32 t, acc, 0fBFB8AA3B;"
    " ex2.approx.f32 e, t;"
    " add.f32 d, e, 0f3F800000;"
    " div.full.f32 $0, acc, d; }"
)


class _PrepareConvKernel:
    """b12x's prepare kernel with the conv of q, k and v folded into its staging."""

    def __init__(self, *, heads, tiles_capacity, window_tiles, qk_l2norm, a_log_type, dt_bias_type, index_type,
                 null_state_index):
        self.heads = int(heads)
        self.tiles_capacity = int(tiles_capacity)
        self.window_tiles = int(window_tiles)
        self.qk_l2norm = bool(qk_l2norm)
        self.a_log_type = a_log_type
        self.dt_bias_type = dt_bias_type
        self.index_type = index_type
        self.has_null = null_state_index is not None
        self.null_state_index = 0 if null_state_index is None else int(null_state_index)
        self.proj = self.heads * _HEAD_DIM

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        conv_w: cute.Pointer,
        conv_state: cute.Pointer,
        init_indices: cute.Pointer,
        v_out: cute.Pointer,
        raw_g: cute.Pointer,
        raw_beta: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        cu_seqlens: cute.Pointer,
        pos_seq: cute.Pointer,
        pos_local: cute.Pointer,
        error_code: cute.Pointer,
        ready: cute.Pointer,
        ws_bf16: cute.Pointer,
        ws_f32: cute.Pointer,
        x_stride: Int64,
        cs_slot_stride: Int64,
        cs_tok_stride: Int64,
        v_out_stride: Int64,
        g_stride: Int64,
        beta_token_stride: Int64,
        beta_head_stride: Int64,
        scale: Float32,
        gate_scale: Float32,
        eps: Float32,
        window: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x, conv_w, conv_state, init_indices, v_out, raw_g, raw_beta, A_log, dt_bias, cu_seqlens, pos_seq,
            pos_local, error_code, ready, ws_bf16, ws_f32, x_stride, cs_slot_stride, cs_tok_stride, v_out_stride,
            g_stride, beta_token_stride, beta_head_stride, scale, gate_scale, eps, window,
        ).launch(grid=(self.window_tiles, self.heads, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        conv_w: cute.Pointer,
        conv_state: cute.Pointer,
        init_indices: cute.Pointer,
        v_out: cute.Pointer,
        raw_g: cute.Pointer,
        raw_beta: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        cu_seqlens: cute.Pointer,
        pos_seq: cute.Pointer,
        pos_local: cute.Pointer,
        error_code: cute.Pointer,
        ready: cute.Pointer,
        ws_bf16: cute.Pointer,
        ws_f32: cute.Pointer,
        x_stride: Int64,
        cs_slot_stride: Int64,
        cs_tok_stride: Int64,
        v_out_stride: Int64,
        g_stride: Int64,
        beta_token_stride: Int64,
        beta_head_stride: Int64,
        scale: Float32,
        gate_scale: Float32,
        eps: Float32,
        window: Int32,
    ):
        local_tile, head, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        local_tile = Int32(local_tile)
        tile = window * Int32(self.window_tiles) + local_tile
        head = Int32(head)
        column = Int32(thread)
        warp = column // Int32(32)
        lane = Int32(cute.arch.lane_idx())
        error = error_code[Int32(0)].to(Int32)
        seq = Int32(-1)
        local = Int32(0)
        if tile < Int32(self.tiles_capacity):
            seq = pos_seq[tile].to(Int32)
            local = pos_local[tile].to(Int32)
        if (error == Int32(0)) & (seq >= Int32(0)):
            allocator = cutlass.utils.SmemAllocator()
            tile_elements = _CHUNK * _HEAD_DIM
            s_part = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((2 * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_beta = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_raw = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((3 * tile_elements,), stride=(1,)),
                byte_alignment=128,
            )
            s_squares = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((4 * _CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            # Pre-conv q, k, v rows of the tile plus the three-row halo: [tensor][row 0..18][128].
            s_x = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((3 * _XROWS * _HEAD_DIM,), stride=(1,)),
                byte_alignment=128,
            )
            square_layout = cute.make_layout((_CHUNK * _CHUNK,), stride=(1,))
            squares = s_squares.iterator
            s_p = cute.make_tensor(squares, square_layout)
            s_p2 = cute.make_tensor(squares + _CHUNK * _CHUNK, square_layout)
            s_inv = cute.make_tensor(squares + 2 * _CHUNK * _CHUNK, square_layout)
            s_inv2 = cute.make_tensor(squares + 3 * _CHUNK * _CHUNK, square_layout)

            start = cu_seqlens[seq].to(Int32) + local * Int32(_CHUNK)
            end = cu_seqlens[seq + Int32(1)].to(Int32)
            rows = cutlass.min(Int32(_CHUNK), end - start)
            head_elements = head.to(Int64) * Int64(_HEAD_DIM)

            raw_addr = shared_ptr_to_u32(s_raw.iterator)
            x_addr = shared_ptr_to_u32(s_x.iterator)
            # Raw g rows into the third region of s_raw, as b12x stages them.
            for item in cutlass.range_constexpr(2):
                g_chunk = column + Int32(item * _THREADS)
                g_row = g_chunk // Int32(16)
                g_col = g_chunk % Int32(16)
                g_live = cutlass.min(g_row, cutlass.max(rows - Int32(1), Int32(0)))
                g_bytes = Int32(16)
                if g_row >= rows:
                    g_bytes = Int32(0)
                K._cp_async_16_zfill(
                    raw_addr + (Int32(512) + g_chunk) * Int32(16),
                    K._pointer_address(raw_g, (start + g_live).to(Int64) * g_stride + head_elements + (g_col * Int32(8)).to(Int64)),
                    g_bytes,
                )
            # Pre-conv rows: tile rows t = row - 3 from x (zero past the tail); halo rows from x, or from the conv
            # state (rows 0..2 = x[-3..-1]) for a sequence's first tile when it has an initial state, else zero.
            init_slot = init_indices[seq].to(Int64)
            has_init = cutlass.Boolean(True)
            if cutlass.const_expr(self.has_null):
                has_init = init_slot != Int64(self.null_state_index)
            for item in cutlass.range_constexpr((3 * _XROWS * 16 + _THREADS - 1) // _THREADS):
                x_chunk = column + Int32(item * _THREADS)
                if x_chunk < Int32(3 * _XROWS * 16):
                    x_tensor = x_chunk // Int32(_XROWS * 16)
                    x_local = x_chunk % Int32(_XROWS * 16)
                    x_row = x_local // Int32(16)
                    x_col = x_local % Int32(16)
                    x_channel = (x_tensor * Int32(self.proj)).to(Int64) + head_elements + (x_col * Int32(8)).to(Int64)
                    x_t = x_row - Int32(_HALO)
                    x_live = cutlass.max(cutlass.min(x_t, cutlass.max(rows - Int32(1), Int32(0))), Int32(0))
                    x_address = K._pointer_address(x, (start + x_live).to(Int64) * x_stride + x_channel)
                    x_bytes = Int32(16)
                    if x_t >= rows:
                        x_bytes = Int32(0)
                    if x_t < Int32(0):
                        if local > Int32(0):
                            x_address = K._pointer_address(x, (start + x_t).to(Int64) * x_stride + x_channel)
                        else:
                            x_bytes = Int32(0)
                            if has_init:
                                x_address = K._pointer_address(
                                    conv_state, init_slot * cs_slot_stride + x_row.to(Int64) * cs_tok_stride + x_channel
                                )
                                x_bytes = Int32(16)
                    K._cp_async_16_zfill(x_addr + x_chunk * Int32(16), x_address, x_bytes)
            cute.arch.cp_async_commit_group()
            # Parameter, beta and conv-weight loads overlap the staging copies.
            rate = cute.math.exp(Float32(A_log[head]), fastmath=False)
            bias = Float32(dt_bias[head * Int32(_HEAD_DIM) + column])
            beta_raw = Float32(0.0)
            if column < rows:
                beta_offset = (
                    (start + column).to(Int64) * beta_token_stride
                    + head.to(Int64) * beta_head_stride
                )
                beta_raw = Float32(raw_beta[beta_offset])
            weights = cute.make_rmem_tensor((3 * _WIDTH,), Float32)
            for w_tensor in cutlass.range_constexpr(3):
                w_row = (Int32(w_tensor * self.proj) + head * Int32(_HEAD_DIM) + column).to(Int64) * Int64(_WIDTH)
                for w_tap in cutlass.range_constexpr(_WIDTH):
                    weights[w_tensor * _WIDTH + w_tap] = Float32(conv_w[w_row + Int64(w_tap)])
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Conv + SiLU down this thread's column: q and k into the s_raw rows b12x reads (zero past the tail, as
            # b12x's zero-filled staging), v into the post-conv buffer the recurrence reads.
            for conv_tensor in cutlass.range_constexpr(3):
                x_base = conv_tensor * _XROWS * _HEAD_DIM
                c0 = Float32(s_x[Int32(x_base) + column])
                c1 = Float32(s_x[Int32(x_base + _HEAD_DIM) + column])
                c2 = Float32(s_x[Int32(x_base + 2 * _HEAD_DIM) + column])
                for ct in cutlass.range_constexpr(_CHUNK):
                    c3 = Float32(s_x[Int32(x_base + (ct + _HALO) * _HEAD_DIM) + column])
                    conv_y = BFloat16(
                        _conv_silu(
                            c0, c1, c2, c3,
                            weights[conv_tensor * _WIDTH], weights[conv_tensor * _WIDTH + 1],
                            weights[conv_tensor * _WIDTH + 2], weights[conv_tensor * _WIDTH + 3],
                        )
                    )
                    if cutlass.const_expr(conv_tensor < 2):
                        conv_value = conv_y
                        if Int32(ct) >= rows:
                            conv_value = BFloat16(0.0)
                        s_raw[Int32(conv_tensor * tile_elements + ct * _HEAD_DIM) + column] = conv_value
                    else:
                        if Int32(ct) < rows:
                            v_out[(start + Int32(ct)).to(Int64) * v_out_stride + head_elements + column.to(Int64)] = conv_y
                    c0 = c1
                    c1 = c2
                    c2 = c3
            cute.arch.sync_threads()

            # ---- from here on: b12x 1.3.0's prepare kernel, unchanged ----
            # Row sums of squares: eight lanes per row, sixteen strided elements each.
            sum_row = column >> Int32(3)
            sum_part = column & Int32(7)
            q_sq = Float32(0.0)
            k_sq = Float32(0.0)
            for item in cutlass.range_constexpr(_HEAD_DIM // 8):
                element = sum_row * Int32(_HEAD_DIM) + Int32(item * 8) + sum_part
                q_value = Float32(s_raw[element])
                k_value = Float32(s_raw[Int32(tile_elements) + element])
                q_sq += q_value * q_value
                k_sq += k_value * k_value
            q_sq = warp_reduce(q_sq, K._add, 8)
            k_sq = warp_reduce(k_sq, K._add, 8)
            if sum_part == Int32(0):
                s_part[sum_row] = q_sq
                s_part[Int32(_CHUNK) + sum_row] = k_sq

            q_values = cute.make_rmem_tensor((_CHUNK,), Float32)
            k_values = cute.make_rmem_tensor((_CHUNK,), Float32)
            g_cum = cute.make_rmem_tensor((_CHUNK,), Float32)
            running = Float32(0.0)
            for t in cutlass.range_constexpr(_CHUNK):
                q_value = Float32(0.0)
                k_value = Float32(0.0)
                g2 = Float32(0.0)
                if Int32(t) < rows:
                    q_value = Float32(s_raw[Int32(t * _HEAD_DIM) + column])
                    k_value = Float32(s_raw[Int32(tile_elements + t * _HEAD_DIM) + column])
                    g_value = Float32(s_raw[Int32(2 * tile_elements + t * _HEAD_DIM) + column])
                    z = rate * (g_value + bias)
                    sigmoid = cute.arch.rcp_approx(
                        Float32(1.0) + K._exp2_approx_ftz_f32(-z * Float32(_LOG2E))
                    )
                    g2 = gate_scale * sigmoid
                q_values[t] = q_value
                k_values[t] = k_value
                running += g2
                g_cum[t] = running
            if column < Int32(_CHUNK):
                beta = Float32(0.0)
                if column < rows:
                    beta = cute.arch.rcp_approx(
                        Float32(1.0) + K._exp2_approx_ftz_f32(-beta_raw * Float32(_LOG2E))
                    )
                s_beta[column] = beta
            cute.arch.sync_threads()

            ring_index = (window & Int32(1)) * Int32(self.window_tiles) + local_tile
            record = ring_index.to(Int64) * Int64(self.heads) + head.to(Int64)
            rec_bf16 = record * Int64(REC.BYTES // 2)
            rec_f32 = record * Int64(REC.BYTES // 4)
            q_base = rec_bf16 + Int64(REC.Q_TILDE // 2)
            k_base = rec_bf16 + Int64(REC.K_TILDE // 2)
            kr_base = rec_bf16 + Int64(REC.K_R // 2)
            last = g_cum[_CHUNK - 1]
            lambda_c = K._exp2_approx_ftz_f32(last)
            ws_f32[rec_f32 + Int64(REC.LAMBDA_C // 4) + column.to(Int64)] = lambda_c
            if column < Int32(_CHUNK):
                ws_f32[rec_f32 + Int64(REC.BETA // 4) + column.to(Int64)] = s_beta[column]
            for t in cutlass.range_constexpr(_CHUNK):
                rinv_q = Float32(1.0)
                rinv_k = Float32(1.0)
                if cutlass.const_expr(self.qk_l2norm):
                    rinv_q = cute.math.rsqrt(s_part[Int32(t)] + eps, fastmath=False)
                    rinv_k = cute.math.rsqrt(s_part[Int32(_CHUNK + t)] + eps, fastmath=False)
                lam = K._exp2_approx_ftz_f32(g_cum[t])
                lam_inv = K._exp2_approx_ftz_f32(-g_cum[t])
                lam_r = K._exp2_approx_ftz_f32(last - g_cum[t])
                q_tilde = BFloat16(q_values[t] * rinv_q * lam * scale)
                k_tilde = BFloat16(k_values[t] * rinv_k * lam)
                k_inv = BFloat16(k_values[t] * rinv_k * lam_inv)
                k_r = BFloat16(k_values[t] * rinv_k * lam_r)
                physical = (
                    Int32(t * _HEAD_DIM)
                    + ((((column >> Int32(3)) ^ Int32(t & 7)) << Int32(3)) | (column & Int32(7)))
                )
                s_raw[physical] = q_tilde
                s_raw[Int32(tile_elements) + physical] = k_tilde
                s_raw[Int32(2 * tile_elements) + column * Int32(_CHUNK) + Int32(t)] = k_inv
                ws_bf16[q_base + physical.to(Int64)] = q_tilde
                ws_bf16[k_base + physical.to(Int64)] = k_tilde
                ws_bf16[kr_base + physical.to(Int64)] = k_r
            cute.arch.sync_threads()

            if warp < Int32(2):
                a_base = raw_addr + Int32(tile_elements * 2)
                if warp == Int32(1):
                    a_base = raw_addr
                b_base = raw_addr + Int32(2 * tile_elements * 2)
                gid = lane >> Int32(2)
                tid = lane & Int32(3)
                matrix = lane >> Int32(3)
                matrix_row = lane & Int32(7)
                a_row = (matrix & Int32(1)) * Int32(8) + matrix_row
                prod = cute.make_rmem_tensor((2, 4), Float32)
                for half in cutlass.range_constexpr(2):
                    for item in cutlass.range_constexpr(4):
                        prod[half, item] = Float32(0.0)
                for kb in cutlass.range_constexpr(8):
                    a_chunk = (Int32(kb * 2) + (matrix >> Int32(1))) ^ (a_row & Int32(7))
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + a_row * Int32(256) + a_chunk * Int32(16))
                    b_row = Int32(kb * 16) + (matrix & Int32(1)) * Int32(8) + matrix_row
                    b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                        b_base + b_row * Int32(32) + (matrix >> Int32(1)) * Int32(16)
                    )
                    prod[0, 0], prod[0, 1], prod[0, 2], prod[0, 3] = bf16_mma_m16n8k16_f32(
                        prod[0, 0], prod[0, 1], prod[0, 2], prod[0, 3], a0, a1, a2, a3, b0, b1
                    )
                    prod[1, 0], prod[1, 1], prod[1, 2], prod[1, 3] = bf16_mma_m16n8k16_f32(
                        prod[1, 0], prod[1, 1], prod[1, 2], prod[1, 3], a0, a1, a2, a3, b2, b3
                    )
                for half in cutlass.range_constexpr(2):
                    for item in cutlass.range_constexpr(4):
                        row_i = gid + Int32((item >> 1) * 8)
                        col_j = Int32(half * 8) + tid * Int32(2) + Int32(item & 1)
                        index = row_i * Int32(_CHUNK) + col_j
                        value = prod[half, item]
                        if warp == Int32(0):
                            lower = Float32(0.0)
                            if col_j < row_i:
                                lower = s_beta[row_i] * value
                            identity = Float32(0.0)
                            if col_j == row_i:
                                identity = Float32(1.0)
                            s_p[index] = lower
                            s_inv[index] = identity - lower
                        else:
                            mqk = Float32(0.0)
                            if col_j <= row_i:
                                mqk = value
                            ws_bf16[rec_bf16 + Int64(REC.MQK // 2) + index.to(Int64)] = BFloat16(mqk)
            cute.arch.sync_threads()

            for _step in cutlass.range_constexpr(3):
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _THREADS)
                    row = index // Int32(_CHUNK)
                    col = index % Int32(_CHUNK)
                    acc = Float32(0.0)
                    for j in cutlass.range_constexpr(_CHUNK):
                        acc += s_p[row * Int32(_CHUNK) + Int32(j)] * s_p[Int32(j * _CHUNK) + col]
                    s_p2[index] = acc
                cute.arch.sync_threads()
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _THREADS)
                    row = index // Int32(_CHUNK)
                    col = index % Int32(_CHUNK)
                    acc = s_inv[index]
                    for j in cutlass.range_constexpr(_CHUNK):
                        acc += s_inv[row * Int32(_CHUNK) + Int32(j)] * s_p2[Int32(j * _CHUNK) + col]
                    s_inv2[index] = acc
                cute.arch.sync_threads()
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _THREADS)
                    s_p[index] = s_p2[index]
                    s_inv[index] = s_inv2[index]
                cute.arch.sync_threads()
            for entry in cutlass.range_constexpr(2):
                index = column + Int32(entry * _THREADS)
                ws_bf16[rec_bf16 + Int64(REC.INV // 2) + index.to(Int64)] = BFloat16(s_inv[index])
            cute.arch.sync_threads()
            if column == Int32(0):
                cute.arch.fence_acq_rel_gpu()
                K._st_release_gpu_i32(K._pointer_address(ready, record), window + Int32(1))


class _ConvStateKernel:
    """conv_state[slot][0..2] <- the last three rows of [old state rows 0..2 (zeros without one), the chunk's rows],
    per sequence and channel; a null slot is skipped. The conv kernel's state update, as a separate launch that runs
    after every prepare read its halo."""

    def __init__(self, *, conv_dim, index_type, null_state_index):
        self.conv_dim = int(conv_dim)
        self.index_type = index_type
        self.has_null = null_state_index is not None
        self.null_state_index = 0 if null_state_index is None else int(null_state_index)

    @cute.jit
    def __call__(self, x: cute.Pointer, conv_state: cute.Pointer, cu_seqlens: cute.Pointer, state_indices: cute.Pointer,
                 init_indices: cute.Pointer, num_seqs: cute.Pointer, x_stride: Int64, cs_slot_stride: Int64,
                 cs_tok_stride: Int64, state_index_stride: Int64, seq_capacity: Int32, stream: cuda.CUstream):
        self.kernel(x, conv_state, cu_seqlens, state_indices, init_indices, num_seqs, x_stride, cs_slot_stride,
                    cs_tok_stride, state_index_stride).launch(
            grid=(seq_capacity, self.conv_dim // _THREADS, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, conv_state: cute.Pointer, cu_seqlens: cute.Pointer, state_indices: cute.Pointer,
               init_indices: cute.Pointer, num_seqs: cute.Pointer, x_stride: Int64, cs_slot_stride: Int64,
               cs_tok_stride: Int64, state_index_stride: Int64):
        seq, block, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        seq = Int32(seq)
        channel = (Int32(block) * Int32(_THREADS) + Int32(thread)).to(Int64)
        if seq < num_seqs[Int32(0)].to(Int32):
            slot = state_indices[seq.to(Int64) * state_index_stride].to(Int64)
            live = cutlass.Boolean(True)
            if cutlass.const_expr(self.has_null):
                live = slot != Int64(self.null_state_index)
            if live:
                start = cu_seqlens[seq].to(Int32)
                length = cu_seqlens[seq + Int32(1)].to(Int32) - start
                init_slot = init_indices[seq].to(Int64)
                has_init = cutlass.Boolean(True)
                if cutlass.const_expr(self.has_null):
                    has_init = init_slot != Int64(self.null_state_index)
                old = cute.make_rmem_tensor((_HALO,), BFloat16)
                for j in cutlass.range_constexpr(_HALO):
                    value = BFloat16(0.0)
                    if has_init:
                        value = conv_state[init_slot * cs_slot_stride + Int64(j) * cs_tok_stride + channel]
                    old[j] = value
                new = cute.make_rmem_tensor((_HALO,), BFloat16)
                for j in cutlass.range_constexpr(_HALO):
                    position = length + Int32(j)            # index into [old(3), x(length)]
                    value = BFloat16(0.0)
                    if position >= Int32(_HALO):
                        value = x[(start + position - Int32(_HALO)).to(Int64) * x_stride + channel]
                    else:
                        for i in cutlass.range_constexpr(_HALO):
                            if position == Int32(i):
                                value = old[i]
                    new[j] = value
                for j in cutlass.range_constexpr(_HALO):
                    conv_state[slot * cs_slot_stride + Int64(j) * cs_tok_stride + channel] = new[j]


def _prepare_conv_key(binding) -> tuple:
    caps = binding.plan.caps
    return ("prepare_conv_fused", _SOURCE_TAG, binding.output.device.index, caps.heads, caps.tiles_capacity,
            binding.plan.window_tiles, caps.qk_l2norm, binding.A_log.dtype, binding.dt_bias.dtype,
            binding.initial_state_indices.dtype, caps.null_state_index)


def _compile_prepare_conv(binding):
    key = _prepare_conv_key(binding)
    cached = _PREPARE_CONV_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding.plan.caps
    a_log_type = K._numeric_type(binding.A_log.dtype)
    dt_bias_type = K._numeric_type(binding.dt_bias.dtype)
    index_type = K._numeric_type(binding.initial_state_indices.dtype)
    kernel = _PrepareConvKernel(heads=caps.heads, tiles_capacity=caps.tiles_capacity,
                                window_tiles=binding.plan.window_tiles, qk_l2norm=caps.qk_l2norm, a_log_type=a_log_type,
                                dt_bias_type=dt_bias_type, index_type=index_type, null_state_index=caps.null_state_index)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        K._fake_pointer(BFloat16), K._fake_pointer(Float32), K._fake_pointer(BFloat16), K._fake_pointer(index_type),
        K._fake_pointer(BFloat16), K._fake_pointer(BFloat16), K._fake_pointer(BFloat16), K._fake_pointer(a_log_type),
        K._fake_pointer(dt_bias_type), K._fake_pointer(Int32), K._fake_pointer(Int32), K._fake_pointer(Int32),
        K._fake_pointer(Int32), K._fake_pointer(Int32), K._fake_pointer(BFloat16), K._fake_pointer(Float32),
        Int64(1), Int64(1), Int64(1), Int64(1), Int64(1), Int64(1), Int64(1),
        Float32(1.0), Float32(1.0), Float32(1.0), Int32(0), current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("glm53.kda_prefill.prepare_conv", 1, key),
    )

    def launch(active, conv, scale: float, gate_scale: float, eps: float, window: int) -> None:
        if _prepare_conv_key(active) != key:
            raise ValueError("compiled conv-fused KDA prepare kernel does not match the binding")
        raw(
            K._pointer(conv.x, BFloat16), K._pointer(conv.weight, Float32), K._pointer(conv.state_sd, BFloat16),
            K._pointer(active.initial_state_indices, index_type), K._pointer(conv.v_out, BFloat16),
            K._pointer(active.raw_g, BFloat16), K._pointer(active.raw_beta, BFloat16),
            K._pointer(active.A_log, a_log_type), K._pointer(active.dt_bias, dt_bias_type),
            K._pointer(active.cu_seqlens, Int32), K._pointer(active.pos_seq, Int32), K._pointer(active.pos_local, Int32),
            K._pointer(active.error_code, Int32), K._pointer(active.ready_flags, Int32),
            K._pointer(active.ws.view(torch.bfloat16), BFloat16), K._pointer(active.ws.view(torch.float32), Float32),
            int(conv.x.stride(0)), int(conv.state_sd.stride(0)), int(conv.state_sd.stride(1)), int(conv.v_out.stride(0)),
            int(active.raw_g.stride(0)), int(active.raw_beta.stride(0)), int(active.raw_beta.stride(1)),
            float(scale), float(gate_scale), float(eps), int(window), current_cuda_stream(),
        )

    _PREPARE_CONV_CACHE[key] = launch
    return key, launch


def _state_key(binding, conv) -> tuple:
    caps = binding.plan.caps
    return ("conv_state_update", _SOURCE_TAG, binding.output.device.index, int(conv.x.shape[1]),
            binding.initial_state_indices.dtype, binding.final_state_indices.dtype, caps.null_state_index)


def _compile_state(binding, conv):
    key = _state_key(binding, conv)
    cached = _STATE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding.plan.caps
    index_type = K._numeric_type(binding.initial_state_indices.dtype)
    final_type = K._numeric_type(binding.final_state_indices.dtype)
    if final_type is not index_type:
        raise TypeError("state index tensors must share one dtype")
    kernel = _ConvStateKernel(conv_dim=int(conv.x.shape[1]), index_type=index_type, null_state_index=caps.null_state_index)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        K._fake_pointer(BFloat16), K._fake_pointer(BFloat16), K._fake_pointer(Int32), K._fake_pointer(index_type),
        K._fake_pointer(index_type), K._fake_pointer(Int32), Int64(1), Int64(1), Int64(1), Int64(1), Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("glm53.kda_prefill.conv_state", 1, key),
    )

    def launch(active, conv_args) -> None:
        raw(
            K._pointer(conv_args.x, BFloat16), K._pointer(conv_args.state_sd, BFloat16), K._pointer(active.cu_seqlens, Int32),
            K._pointer(active.final_state_indices, index_type), K._pointer(active.initial_state_indices, index_type),
            K._pointer(active.num_seqs, Int32), int(conv_args.x.stride(0)), int(conv_args.state_sd.stride(0)),
            int(conv_args.state_sd.stride(1)), int(active.final_state_indices.stride(0)), int(active.seq_capacity),
            current_cuda_stream(),
        )

    _STATE_CACHE[key] = launch
    return key, launch


class ConvArgs:
    """The conv operands of one fused prefill: x [T, 3*H*128] pre-conv (row stride free, channels dense), weight
    [3*H*128, 4] fp32 contiguous, state_sd [slots, >=3, 3*H*128] bf16 with dense channels (the SD layout), v_out
    [T, H*128] bf16 (row stride free)."""

    def __init__(self, x: torch.Tensor, weight: torch.Tensor, state_sd: torch.Tensor, v_out: torch.Tensor):
        dim = int(x.shape[1])
        if x.dtype != torch.bfloat16 or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
            raise ValueError("x must be bf16 with dense channels, 16-byte aligned rows")
        if weight.dtype != torch.float32 or tuple(weight.shape) != (dim, _WIDTH) or not weight.is_contiguous():
            raise ValueError("conv weight must be a contiguous fp32 [dim, 4] tensor")
        if (state_sd.dtype != torch.bfloat16 or state_sd.shape[2] != dim or state_sd.shape[1] < _HALO
                or state_sd.stride(2) != 1 or state_sd.stride(1) % 8 or state_sd.stride(0) % 8 or state_sd.data_ptr() % 16):
            raise ValueError("conv state must be bf16 [slots, >=3, dim] with dense channels (SD layout)")
        if v_out.dtype != torch.bfloat16 or v_out.stride(1) != 1 or v_out.shape[1] * 3 != dim:
            raise ValueError("v_out must be bf16 [T, dim / 3] with dense channels")
        self.x, self.weight, self.state_sd, self.v_out = x, weight, state_sd, v_out


def run_prefill_conv_fused(binding, conv: ConvArgs, *, lower_bound: float, scale: float, eps: float,
                           windows: int) -> None:
    """b12x's window pipeline with the conv-fused prepare: prologue, per window the fused prepare on the side stream
    and b12x's recurrence (reading ``binding.v`` = post-conv v) on the main stream, then the conv-state update.

    ``binding`` must be bound with ``v`` = a ``[T, H, 128]`` view of ``conv.v_out``; its ``q``/``k`` are not read."""
    device = binding.output.device
    plan = binding.plan
    launched = int(windows)
    if launched < 1 or launched > plan.max_windows:
        raise ValueError(f"windows must be in 1..{plan.max_windows}, got {launched}")
    if binding.v.data_ptr() != conv.v_out.data_ptr():
        raise ValueError("binding.v must view conv.v_out")
    with torch.cuda.device(device):
        main = torch.cuda.current_stream(device)
        resources = K._side_resources(device, launched)
        side = resources.stream
        fork = resources.events[2 * launched]
        prepared = resources.events[:launched]
        consumed = resources.events[launched:2 * launched]
        K.run_prologue(binding, windows=launched)
        fork.record(main)
        side.wait_event(fork)

        def enqueue_prepare(window: int) -> None:
            with torch.cuda.stream(side):
                if window >= 2:
                    side.wait_event(consumed[window - 2])
                K._launch_stage(
                    lambda b: (_prepare_conv_key(b), _PREPARE_CONV_CACHE.get(_prepare_conv_key(b))),
                    _compile_prepare_conv,
                    binding,
                    conv,
                    float(scale),
                    float(lower_bound) * _LOG2E,
                    float(eps),
                    int(window),
                )
                prepared[window].record(side)

        enqueue_prepare(0)
        for window in range(launched):
            main.wait_event(prepared[window])
            K.run_recurrence(binding, window=window)
            consumed[window].record(main)
            if window + 1 < launched:
                enqueue_prepare(window + 1)
        K._launch_stage(
            lambda b: (_state_key(b, conv), _STATE_CACHE.get(_state_key(b, conv))),
            lambda b: _compile_state(b, conv),
            binding,
            conv,
        )


def run_kda_prefill_conv_fused(layer, *, qkv: torch.Tensor, conv_weight: torch.Tensor, conv_state_sd: torch.Tensor,
                               raw_g: torch.Tensor, raw_beta: torch.Tensor, cu_seqlens: torch.Tensor,
                               state_indices: torch.Tensor, has_initial_state: torch.Tensor,
                               recurrent_state: torch.Tensor, scratch, output: torch.Tensor,
                               v_out: torch.Tensor) -> None:
    """kda.py's ``_run_b12x_kda_prefill`` with the conv folded in: same plan, metadata buffers, index masking and run
    scalars (``api.run``'s scale = 128 ** -0.5 and eps = 1e-6), with ``qkv`` the pre-conv ``[T, 3*H*128]`` view, the
    conv weight ``[3*H*128, 4]``, the SD conv state ``[slots, width, 3*H*128]``, ``output`` the ``[T, H, 128]`` layer
    output rows and ``v_out`` a ``[>=T, H*128]`` bf16 buffer for post-conv v."""
    api = layer._b12x_prefill_api
    plan = layer._b12x_prefill_plan
    if api is None or plan is None:
        raise RuntimeError("b12x KDA prefill KV cache was not bound before inference")
    num_tokens = int(qkv.shape[0])
    num_requests = int(state_indices.shape[0])
    if num_tokens > layer._b12x_prefill_max_tokens or num_requests > layer._b12x_prefill_max_seqs:
        raise ValueError(
            "b12x KDA prefill capacity exceeded: "
            f"tokens={num_tokens}/{layer._b12x_prefill_max_tokens}, "
            f"requests={num_requests}/{layer._b12x_prefill_max_seqs}"
        )
    null = plan.caps.null_state_index
    initial_indices = layer._b12x_prefill_initial_indices[:num_requests]
    initial_indices.copy_(state_indices)
    initial_indices.masked_fill_(~has_initial_state[:num_requests], null)
    null_indices = layer._b12x_prefill_null_indices[:num_requests]
    zero_offsets = layer._b12x_prefill_zero_offsets[:num_requests]
    layer._b12x_prefill_num_seqs.fill_(num_requests)
    layer._b12x_prefill_num_tokens.fill_(num_tokens)
    heads, head_dim = layer.local_num_heads, layer.head_dim
    proj = heads * head_dim
    v_rows = v_out[:num_tokens]
    binding = api.bind(
        plan,
        scratch=scratch,
        q=qkv[:, :proj].view(num_tokens, heads, head_dim),
        k=qkv[:, proj:2 * proj].view(num_tokens, heads, head_dim),
        v=v_rows.view(num_tokens, heads, head_dim),
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=layer.A_log.view(-1),
        dt_bias=layer.dt_bias.view(-1, head_dim),
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens[: num_requests + 1],
        initial_state_indices=initial_indices,
        final_state_indices=state_indices,
        checkpoint_state_indices=null_indices,
        checkpoint_offsets=zero_offsets,
        num_seqs=layer._b12x_prefill_num_seqs,
        num_tokens=layer._b12x_prefill_num_tokens,
        output=output,
    )
    run_prefill_conv_fused(
        binding,
        ConvArgs(qkv, conv_weight, conv_state_sd, v_rows),
        lower_bound=float(layer.kda_lower_bound),
        scale=float(head_dim) ** -0.5,
        eps=1e-6,
        windows=plan.launched_windows(num_tokens, num_requests),
    )


def prewarm(binding, conv: ConvArgs) -> None:
    """Compile the fused prepare and the state update for ``binding`` (the prologue and recurrence are b12x's)."""
    with torch.cuda.device(binding.output.device):
        _compile_prepare_conv(binding)
        _compile_state(binding, conv)


__all__ = ["ConvArgs", "prewarm", "run_kda_prefill_conv_fused", "run_prefill_conv_fused"]
