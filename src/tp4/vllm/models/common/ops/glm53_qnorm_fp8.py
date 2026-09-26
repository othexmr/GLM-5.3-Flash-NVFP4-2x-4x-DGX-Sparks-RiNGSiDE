# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): vLLM's fused q/kv RMSNorm kernel with per-token FP8 quantization.
"""MLA q_a + kv_a RMSNorm with the q_b_proj's per-token FP8 quantization fused in (prefill-levers lane, 2026-09-25;
used only behind GLM53_MLA_QNORM_FP8 in vllm/model_executor/layers/mla.py, default off).

The served prefill path of a GLM-5.3 MLA layer (mla.py 80038f1a, fuse_qkv_rmsnorm=True) runs two kernels between the
fused q_a/kv_a projection and the q_b_proj GEMM:
  1. `_fused_q_kv_rmsnorm_kernel` (vllm/models/common/ops/fused_qk_rmsnorm.py in the image, sha256 abc078b1): one
     program per (token, task); task 0 normalizes the token's q_c row (1,536 values), task 1 its kv_c row (512), in
     fp32 with the weight, one BF16 store;
  2. vLLM `dynamic_per_token_scaled_fp8_quant` inside q_b_proj's W8A8 dispatch (glm53_dual_fp8_dense.py): per token
     absmax, scale = max(absmax / 448, 1 / (448 * 512)), q = e4m3(clamp(x / scale)). The DSA indexer's wq_b reuses it
     through the dense module's shared-quant slot (GLM53_DENSE_SHARED_QUANT=mla_qb>indexer), so it runs once per MLA
     layer: about 267 us per 13,824-row call (outputs/2026-09-23-dense-prefill), re-reading the BF16 q_c.

`_glm53_fused_q_kv_rmsnorm_fp8_kernel` is kernel 1 statement for statement (the same launch: grid (tokens, 2),
BLOCK_SIZE = next_power_of_2(max(Q, KV)), default warps, the PDL wait/launch), then, in task 0, kernel 2's arithmetic
on the BF16-rounded values of the row it already holds: the same absmax, IEEE round-to-nearest divisions (`tl.div_rn`,
as nvcc compiles vLLM's `/` without fast math), the same clamp and a round-to-nearest-even satfinite e4m3 conversion
(the o_norm lever's recipe, outputs/2026-09-24-dense-fp8-kernels, bit-exact in its leaf). The intent is bit identity
with kernels 1 + 2; the leaf checks it. WRITE_Q=False skips the BF16 q_c store (valid only when every reader of q_c
takes the FP8 copy; mla.py enforces that and raises otherwise).
v2 (2026-09-25 18:3x, after leaf Q round 1): the FP8 store's offsets carry a contiguity of 8 (see the store), so the
compiled kernel keeps the served kernel's single 8-per-thread layout."""
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Triton 3.7 reads a global inside @jit only if it is a tl.constexpr instance (outputs/2026-09-24-dense-fp8-kernels).
FP8_MAX = tl.constexpr(448.0)
MIN_SCALE = tl.constexpr(1.0 / (448.0 * 512.0))


@triton.jit
def _glm53_fused_q_kv_rmsnorm_fp8_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    q8_ptr,
    q_scale_ptr,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WRITE_Q: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    # --- kernel 1, the served _fused_q_kv_rmsnorm_kernel body, statement for statement ---
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    # --- the served kernel stores y.to(row_out.dtype.element_ty) here, for both tasks ---
    yb = y.to(row_out.dtype.element_ty)
    if pid_task == 0:
        if WRITE_Q:
            tl.store(row_out + block, yb, mask=mask)
        # --- kernel 2: per-token FP8 of this q_c row, on its BF16 values ---
        v = tl.where(mask, yb.to(tl.float32), 0.0)
        amax = tl.max(tl.abs(v), axis=0)
        scale = tl.maximum(tl.div_rn(amax, FP8_MAX), MIN_SCALE)
        qv = tl.div_rn(v, scale)
        qv = tl.minimum(tl.maximum(qv, -FP8_MAX), FP8_MAX)
        # v2: the FP8 row is stored 8 bytes per thread. With its natural 16-byte vectors the coalescer gives the row's
        # loads the store's 16-per-thread layout (it takes the widest access of the slice), which regroups the RMS
        # reduction and changed a few BF16 results in leaf Q round 1; capping this store's contiguity at 8 keeps every
        # access, and the reduction, in the served kernel's layout (sizePerThread 8).
        q_offs = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), 8)
        tl.store(q8_ptr + token_idx * Q_SIZE + q_offs, qv.to(q8_ptr.dtype.element_ty), mask=mask)
        tl.store(q_scale_ptr + token_idx, scale)
    else:
        tl.store(row_out + block, yb, mask=mask)


def fused_q_kv_rmsnorm_fp8(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    write_q: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The served `fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)` (same checks, packed outputs, grid, block and
    launch) plus the per-token FP8 quantization of the normalized q rows. Returns (qr_out, kv_out, q8, q_scale): qr_out
    and kv_out as the served function returns them (qr_out allocated but not written when write_q is False), q8 [T, Q]
    e4m3 and q_scale [T, 1] fp32 = `ops.scaled_fp8_quant(qr_out, use_per_token_if_dynamic=True)` of the served qr_out."""
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    q8 = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    q_scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=qr.device)
    if num_tokens == 0:
        return qr_out, kv_out, q8, q_scale

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    _glm53_fused_q_kv_rmsnorm_fp8_kernel[(num_tokens, 2)](
        qr,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        q8,
        q_scale,
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK_SIZE=block_size,
        WRITE_Q=write_q,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return qr_out, kv_out, q8, q_scale
