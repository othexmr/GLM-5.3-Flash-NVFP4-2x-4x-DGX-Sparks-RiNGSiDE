# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# glm53_gated_rmsnorm_fp8_kernel contains code copied from the flash-linear-attention project, via vLLM's
# vllm/third_party/flash_linear_attention/ops/kda.py (layer_norm_gated_fwd_kernel).
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): the body of flash-linear-attention's layer_norm_gated_fwd_kernel,
# specialised to RMS norm with a weight and a sigmoid gate, fused with a per-token FP8 quantization of its BF16 output
# (glm53_gated_rmsnorm_fp8_kernel, GLM53_KDA_ONORM_FP8).
"""GLM KDA o_norm with the o_proj's per-token FP8 quantization fused in (dense-fp8-kernels lane, 2026-09-24; used only
behind GLM53_KDA_ONORM_FP8 in kda.py, default off).

The served prefill path runs two kernels between the KDA core and the o_proj GEMM:
  1. FLA `layer_norm_gated_fwd_kernel` (vllm/third_party/flash_linear_attention/ops/kda.py, the FusedRMSNormGated used
     as KDA's o_norm): per head, RMS norm with weight, times sigmoid(gate), written in place in BF16;
  2. vLLM `dynamic_per_token_scaled_fp8_quant_kernel_strided` inside the o_proj's W8A8 dispatch: per token (16 heads x
     128 = 2,048 values per rank) absmax, scale = max(absmax / 448, 1 / (448 * 512)), q = e4m3(clamp(x / scale)).
At 13,824 rows the quantization re-reads the 56.6 MB BF16 output (364 us per KDA layer in the cold 32K trace).

`glm53_gated_rmsnorm_fp8_kernel` is kernel 1 with the same code, block shape (BT = 16 rows = one token's 16 heads,
BD = 128) and warp count, plus kernel 2's arithmetic on the BF16-rounded values of that tile: the same absmax, IEEE
round-to-nearest divisions (`tl.div_rn`, as nvcc compiles vLLM's `/` without fast math), the same clamp and a
round-to-nearest-even satfinite e4m3 conversion. The intent is bit identity with kernels 1 + 2; the leaf checks it.
WRITE_Y=False skips the BF16 store (only valid when the o_proj consumes the FP8 copy; kda.py enforces that)."""
import torch
import triton
import triton.language as tl

# Triton 3.7 reads a global inside @jit only if it is a tl.constexpr instance (an annotation is not enough: leaf job
# 20260924-020304 stopped at this NameError at compile time, before any judged data).
FP8_MAX = tl.constexpr(448.0)
MIN_SCALE = tl.constexpr(1.0 / (448.0 * 512.0))
SERVED_BT = 16


@triton.jit
def glm53_gated_rmsnorm_fp8_kernel(
    x,  # [T, D] BF16 input (T = tokens * H); y may alias it (in place, as the served kernel)
    g,  # gate, rows addressed as (t // H) * g_stride_n + (t % H) * D
    y,  # [T, D] BF16 output
    w,  # [D] norm weight
    q,  # [T // H, H * D] e4m3 output
    s,  # [T // H] fp32 per-token scale
    eps,
    T,
    H: tl.constexpr,
    g_stride_n: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    WRITE_Y: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    # --- kernel 1: the FLA layer_norm_gated_fwd_kernel body (IS_RMS_NORM, HAS_WEIGHT, sigmoid gate) ---
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_xbar = tl.where(m_d[None, :], b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    b_x_hat = b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :]

    o_t = i_t * BT + tl.arange(0, BT)
    o_g = (o_t // H) * g_stride_n + (o_t % H) * D
    b_g = tl.load(
        g + o_g[:, None] + o_d[None, :],
        mask=(o_t[:, None] < T) & m_d[None, :],
        other=0.0,
    ).to(tl.float32)
    b_y = b_y * tl.sigmoid(b_g)

    b_yb = b_y.to(y.dtype.element_ty)
    if WRITE_Y:
        p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_y, b_yb, boundary_check=(0, 1))

    # --- kernel 2: per-token FP8 on the BF16 values (BT == H: this program is exactly one token) ---
    b_v = tl.where(m_d[None, :], b_yb.to(tl.float32), 0.0)
    amax = tl.max(tl.max(tl.abs(b_v), axis=1), axis=0)
    scale = tl.maximum(tl.div_rn(amax, FP8_MAX), MIN_SCALE)
    b_q = tl.div_rn(b_v, scale)
    b_q = tl.minimum(tl.maximum(b_q, -FP8_MAX), FP8_MAX)
    o_r = tl.arange(0, BT)
    tl.store(q + i_t * (BT * D) + o_r[:, None] * D + o_d[None, :], b_q.to(q.dtype.element_ty), mask=m_d[None, :])
    tl.store(s + i_t, scale)


def gated_rmsnorm_fp8(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float, write_y: bool = True):
    """x: the KDA core output [1, M, H, D] (or [M, H, D]) BF16; g: the gate g2 [M, H, D]; weight: o_norm.weight [D].
    Returns (x2, q, s): x2 = the [M, H * D] view the served code hands to o_proj (normalized in place when write_y),
    q [M, H * D] e4m3 and s [M, 1] fp32, the per-token quantization of x2."""
    D = x.shape[-1]
    H = g.shape[-2]
    xt = x.contiguous().reshape(-1, D)            # as rms_norm_gated: x.contiguous().reshape(-1, D), y = x in place
    gv = g.view(-1, H, D)
    T = xt.shape[0]
    # The served launch (layer_norm_gated_fwd, D <= 512): BT = 16 rows, BD = next_power_of_2(D), 8 warps, grid
    # cdiv(T, 16). With 16 heads a tile is exactly one token, the unit of the per-token quantization.
    if H != SERVED_BT or T % H or D > 512 or weight is None or weight.shape != (D,) or x.dtype != torch.bfloat16:
        raise ValueError(f'gated_rmsnorm_fp8: unsupported geometry T={T} H={H} D={D} dtype={x.dtype}')
    m = T // H
    q = torch.empty((m, H * D), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    glm53_gated_rmsnorm_fp8_kernel[(m,)](
        xt, gv, xt, weight, q, s, eps, T,
        H=H, g_stride_n=gv.stride(0), D=D, BT=SERVED_BT, BD=triton.next_power_of_2(D), WRITE_Y=write_y, num_warps=8)
    return xt.view(m, H * D), q, s
