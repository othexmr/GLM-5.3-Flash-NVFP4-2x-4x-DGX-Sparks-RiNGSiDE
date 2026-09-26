"""Dual-path online FP8 for the BF16 dense projections of GLM-5.3 Flash (NVFP4 lane).

Installed as ``vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py``.

Why: on the two GB10s the decode verification step (1 + K tokens) spends about
51 ms of a 115 ms step in cuBLAS BF16 GEMMs over 8.8 GB of dense weights per
rank, at the memory-bandwidth floor. Halving the bytes is the only dense lever.
Marlin FP8 W8A16 reads half the bytes at the same bandwidth (18.9 vs 37.8 ms
for the per-step dense set at M=8, 2026-09-05 probe) but is 2.2 to 2.4x slower
than BF16 at prefill M, while the FP8 tensor-core GEMM (W8A8, per-token
activation scales) is 1.7 to 2.0x faster than BF16 at M=2,048 and on most
shapes at M=14,336. So: one online per-channel FP8 copy of each selected weight
serves both regimes, dispatched on M.

How: the method creates the same BF16 ``ModelWeightParameter`` as
``UnquantizedLinearMethod`` (so every existing loader and stacked mapping is
untouched), quantizes after loading (per-output-channel absmax / 448 scale),
builds a Marlin-repacked copy for M <= MARLIN_MAX_M, keeps the plain FP8 copy
for W8A8 (or on-the-fly dequantize) above it, and frees the BF16 tensor.

Selection: ``VLLM_GLM53_DUAL_FP8_DENSE`` is a comma list of families (below) or
``off``; only prefixes under ``language_model.model.layers.`` are touched, so the
DFlash drafter and any vision tower are never quantized. ``VLLM_GLM53_DUAL_FP8_PREFILL``
is ``w8a8`` (default) or ``dequant``.
"""
from __future__ import annotations

import os
import re

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

FAMILIES: dict[str, str] = {
    "kda_in": r"\.self_attn\.in_proj_qkvbfg_a$",
    "kda_fg": r"\.self_attn\.(f_b_proj|g_b_proj)$",
    "attn_out": r"\.self_attn\.o_proj$",
    "mla_qb": r"\.self_attn\.q_b_proj$",
    "mla_a": r"\.self_attn\.fused_qkv_a_proj$",
    # shared-expert down_proj (4,096 x 2,048 per rank, 42 layers): the 2026-09-05
    # "Marlin loses" verdict was an L2 artefact; on the L2-safe ring FP8 Marlin is
    # 1.45x. Not in DEFAULT_FAMILIES so the a92 identity is unchanged; select it.
    "shared": r"\.mlp\.shared_experts\.gate_up_proj$",
    "shared_down": r"\.mlp\.shared_experts\.down_proj$",
    "dense": r"\.layers\.[0-2]\.mlp\.(gate_up_proj|down_proj)$",
    # DSA indexer query up-projection (4,096 x 1,536 replicated, 1.35x on the L2-safe
    # ring); the 64-row weights projection loses (0.31x) and stays BF16.
    "indexer": r"\.self_attn\.indexer\.wq_b$",
    # MLA kv_b (8,192 x 512 per rank, 11 layers, 1.21x FP8 on the L2-safe ring).
    "mla_kvb": r"\.self_attn\.kv_b_proj$",
    # The output head (77,440 x 4,096 per rank, 634 MB BF16, 2.8 ms per step)
    # goes through the embedding hook below; it is not under the layer scope.
    # The DFlash drafter shares it by module (llm_base_proposer._maybe_share_lm_head),
    # so the drafter's logits read the same FP8 copy.
    "head": r"^language_model\.lm_head$",
    # The DFlash2 drafter (qwen3_dflash2, 5 Qwen3 layers at model.layers.45..49,
    # 1.22 GB BF16 per rank per step: qkv/o/gate_up/down, the two grouped-conv
    # kernel projections, and the replicated 4,096 x 20,480 feature fc). The
    # target lives under language_model.model.layers, so ^model\. is the drafter.
    # The grouped-conv kernel projections (1,024 x 4,096, 8 MB, ten per step) are
    # left BF16: 0.1 ms per step at M=8 and the FP8 prefill path loses there.
    "draft": r"^model\.(fc|layers\.\d+\.(self_attn\.(qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj)))$",
}
# Families whose regex is anchored at the module root; they are matched before
# the language-model layer scope filter.
ROOT_FAMILIES = ("head", "draft")
DEFAULT_FAMILIES = "kda_in,attn_out,mla_qb,mla_a,shared,dense"
SCOPE_PREFIX = "language_model.model.layers."
# 2026-09-23: decode-regime ceiling of the Marlin copy, env-configurable with the unchanged default 64.
# Above it the W8A8 path runs (per-token FP8 activation quant + CUTLASS scaled_mm, twice the NVFP4 weight bytes).
# A 16-request C16 x K7 verify step is 128 rows; C9+ at K7 and C13+ at K4 exceed 64 today.
MARLIN_MAX_M = int(os.environ.get("VLLM_GLM53_DUAL_MARLIN_MAX_M", "64"))
if not 1 <= MARLIN_MAX_M <= 256:
    raise ValueError("VLLM_GLM53_DUAL_MARLIN_MAX_M must be 1..256")
# cuBLASLt/CUTLASS FP8 picks a slow algorithm at M=14,336 when N*K is large;
# chunking the rows to 2,048 restores 1.3x to 1.5x over BF16 there.
CHUNK_ROWS = 2048
# 24 M covers the drafter's 4,096 x 6,144 down projection (0.83x unchunked at 14,336 rows).
CHUNK_WHEN_ELEMENTS_AT_LEAST = 24 * 1024 * 1024
FP8_MAX = 448.0


def _selected_families() -> list[str]:
    raw = os.environ.get("VLLM_GLM53_DUAL_FP8_DENSE", "off").strip()
    if raw in ("", "0", "off", "none"):
        return []
    families: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # `default` (or `1`/`on`) expands to the retained family set and may be
        # combined with extras, e.g. `default,head`; duplicates are dropped.
        expansion = DEFAULT_FAMILIES.split(",") if token in ("1", "default", "on") else [token]
        for fam in expansion:
            if fam not in families:
                families.append(fam)
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"VLLM_GLM53_DUAL_FP8_DENSE has unknown families {unknown}; known {sorted(FAMILIES)}")
    return families


def _decode_mode() -> str:
    """Weight format of the M<=64 (verification-step) Marlin copy: fp8 (W8A16, 2.7%
    weight error) or nvfp4 (W4A16, e2m1 with per-16-block e4m3 scales, the
    checkpoint's own expert format, about 4x fewer weight bytes than BF16)."""
    mode = os.environ.get("VLLM_GLM53_DUAL_FP8_DECODE", "fp8").strip().lower()
    if mode not in ("fp8", "nvfp4"):
        raise ValueError("VLLM_GLM53_DUAL_FP8_DECODE must be fp8 or nvfp4")
    return mode


def _fp8_override_families() -> set[str]:
    """Families that keep the FP8 W8A16 Marlin copy even when the decode mode is
    nvfp4 (a98 showed the 4-bit indexer and shared-expert-down copies cost about
    four acceptance points on sampled prose for 1.4 ms of step)."""
    raw = os.environ.get("VLLM_GLM53_DUAL_FP8_DECODE_FP8_FAMILIES", "").strip()
    return {f.strip() for f in raw.split(",") if f.strip()}



# --- Humming dense candidate (default OFF) -------------------------------------
# Routes the M<=MARLIN_MAX_M decode path of selected families through the pinned
# Humming 0.1.15 dense kernel instead of Marlin. Leaf evidence for shared_down
# (outputs/2026-09-22-humming-shared-down-rank{1,2,3}, rank 0 in
# 2026-09-22-humming-dense-v2-rank0): selected-row medians 1.377-1.389 on four
# ranks, zero losing shapes, inside the tolerance contract. Per-call ratios have
# shrunk substantially in serving every time this session; this exists to measure
# that, not to assume it.
def _humming_families() -> set[str]:
    raw = os.environ.get("GLM53_HUMMING_DENSE", "").strip()
    if raw in ("", "0", "off", "none"):
        return set()
    return {f.strip() for f in raw.split(",") if f.strip()}


_HUMMING_SHARED: dict = {}


def _humming_layer_config(N: int, K: int, device):
    """One config per (N,K,device). The 42 shared_down projections share a local
    shape, so the config and its per-M kernel choices are built once."""
    key = (N, K, str(device))
    if key in _HUMMING_SHARED:
        return _HUMMING_SHARED[key]
    from humming import dtypes
    from humming.config import InputQuantizationMode, WeightScaleType
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config
    from humming.tune import get_heuristics_config

    config = prepare_layer_config(
        shape_n=N, shape_k=K,
        weight_schema=HummingWeightSchema(b_dtype=dtypes.float8e4m3, bs_dtype=dtypes.bfloat16,
                                          weight_scale_type=WeightScaleType.CHANNEL),
        input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16,
                                        input_quant_mode=InputQuantizationMode.Disabled),
        torch_dtype=torch.bfloat16, device=device, pad_n_to_multiple=64)
    if config.input_quant_mode.should_quantize:
        raise ValueError("Humming config would quantize activations; the leaf contract forbids it")
    # Precompute every M we can meet at or below the Marlin cutoff. Doing this
    # lazily would call the tuner inside a CUDA graph capture.
    kernels = {m: get_heuristics_config(config, shape_m=m, gemm_type="dense", use_f16_accum=False)
               for m in range(1, MARLIN_MAX_M + 1)}
    _HUMMING_SHARED[key] = (config, kernels)
    return _HUMMING_SHARED[key]

# --- Humming NVFP4 above the Marlin cutoff (2026-09-23, default OFF) ------------
# GLM53_HUMMING_NVFP4_M128=<families>: for 64 < M <= HUMMING_NVFP4_MAX_M those families run Humming's dense kernel
# on the same online NVFP4 bytes the Marlin copy is built from (e2m1, E4M3 scales per 16, tensor scale), instead of
# W8A8 FP8 CUTLASS, which reads twice the bytes. Leaf: outputs/2026-09-23-humming-m128. Above the cutoff the weight
# precision is then the NVFP4 the M <= 64 steps already use, not FP8.
HUMMING_NVFP4_MAX_M = 128


def _humming_nvfp4_families() -> set[str]:
    raw = os.environ.get("GLM53_HUMMING_NVFP4_M128", "").strip()
    if raw in ("", "0", "off", "none"):
        return set()
    return {f.strip() for f in raw.split(",") if f.strip()}


_HUMMING_NVFP4_SHARED: dict = {}


def _humming_nvfp4_layer_config(N: int, K: int, device):
    """One config per (N,K,device) with per-M kernels for MARLIN_MAX_M < M <= HUMMING_NVFP4_MAX_M, built at load
    time (never inside a CUDA graph capture)."""
    key = (N, K, str(device))
    if key in _HUMMING_NVFP4_SHARED:
        return _HUMMING_NVFP4_SHARED[key]
    from humming import dtypes
    from humming.config import InputQuantizationMode, WeightScale2Type
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config
    from humming.tune import get_heuristics_config

    config = prepare_layer_config(
        shape_n=N, shape_k=K,
        weight_schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3,
                                          weight_scale_group_size=16, weight_scale_2_type=WeightScale2Type.TENSOR),
        input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16,
                                        input_quant_mode=InputQuantizationMode.Disabled),
        torch_dtype=torch.bfloat16, device=device, pad_n_to_multiple=64)
    if config.input_quant_mode.should_quantize:
        raise ValueError("Humming NVFP4 config would quantize activations")
    kernels = {m: get_heuristics_config(config, shape_m=m, gemm_type="dense", use_f16_accum=False)
               for m in range(MARLIN_MAX_M + 1, HUMMING_NVFP4_MAX_M + 1)}
    _HUMMING_NVFP4_SHARED[key] = (config, kernels)
    return _HUMMING_NVFP4_SHARED[key]

_E2M1_CODES = ((0.0, 0), (0.5, 1), (1.0, 2), (1.5, 3), (2.0, 4), (3.0, 5), (4.0, 6), (6.0, 7))


def quantize_nvfp4_for_marlin(w: torch.Tensor):
    """BF16 [N, K] -> (packed e2m1 uint8 [N, K/2], e4m3 block scales [N, K/16],
    global scale fp32 [1]) in the layout `prepare_fp4_layer_for_marlin` consumes.
    Convention (the one compressed-tensors hands Marlin after inverting the
    checkpoint's 448*6/amax): dequant = code * block_scale * global_scale,
    global_scale = max block scale / 448 so every block scale fits e4m3."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import cast_to_fp4

    N, K = w.shape
    assert K % 16 == 0, (N, K)
    wf = w.float().view(N, K // 16, 16)
    scales = wf.abs().amax(-1, keepdim=True) / 6.0                      # [N, K/16, 1]
    global_scale = (scales.max() / 448.0).clamp(min=1e-12)
    s_e4m3 = (scales / global_scale).to(torch.float8_e4m3fn)
    s_eff = (s_e4m3.float() * global_scale).clamp(min=1e-30)
    q = cast_to_fp4((wf / s_eff).clamp(-6.0, 6.0))                     # values in {0, +-0.5, ..., +-6}
    mag = q.abs().view(N, K)
    codes = torch.zeros(N, K, dtype=torch.uint8, device=w.device)
    for value, code in _E2M1_CODES:
        codes[mag == value] = code
    codes |= (q.view(N, K) < 0).to(torch.uint8) << 3
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()    # element 2i in the low nibble
    return packed, s_e4m3.view(N, K // 16).contiguous(), global_scale.float().reshape(1)


def _prefill_mode() -> str:
    mode = os.environ.get("VLLM_GLM53_DUAL_FP8_PREFILL", "w8a8").strip().lower()
    if mode not in ("w8a8", "dequant"):
        raise ValueError("VLLM_GLM53_DUAL_FP8_PREFILL must be w8a8 or dequant")
    return mode


def family_for_prefix(prefix: str) -> str | None:
    selected = _selected_families()
    for fam in ROOT_FAMILIES:
        if fam in selected and re.search(FAMILIES[fam], prefix):
            return fam
    if not prefix.startswith(SCOPE_PREFIX):
        return None
    for fam in selected:
        if fam in ROOT_FAMILIES:
            continue
        if re.search(FAMILIES[fam], prefix):
            return fam
    return None


def maybe_dual_fp8_head_method(layer: torch.nn.Module, prefix: str):
    """Hook called from VocabParallelEmbedding.__init__ for ParallelLMHead only."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

    if not isinstance(layer, ParallelLMHead):
        return None
    # 2026-09-23: the NVFP4 checkpoint lists lm_head in exclude_modules, and ModelOpt's get_quant_method hands
    # an excluded ParallelLMHead an UnquantizedLinearMethod (not UnquantizedEmbeddingMethod). The old check therefore
    # declined silently and the head stayed BF16 on this lane (serving logs never show "family head selected"). Both
    # are the plain BF16 method; accept either.
    if not isinstance(getattr(layer, "quant_method", None), (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)):
        return None
    if family_for_prefix(prefix) != "head":
        return None
    logger.info_once("GLM53 dual-path FP8 dense: family head selected for %s", prefix, scope="global")
    return DualPathFp8DenseLinearMethod("head")


def maybe_dual_fp8_dense_method(layer: torch.nn.Module, prefix: str):
    """Hook called from LinearBase.__init__ after the stock quant method is chosen.
    Returns a DualPathFp8DenseLinearMethod for selected unquantized linears, else None."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod):
        return None
    fam = family_for_prefix(prefix)
    if fam is None:
        return None
    decode = "fp8" if fam in _fp8_override_families() else _decode_mode()
    logger.info_once("GLM53 dual-path FP8 dense: family %s selected (prefill mode %s, decode marlin %s for M<=%d)", fam, _prefill_mode(), decode, MARLIN_MAX_M, scope="global")
    return DualPathFp8DenseLinearMethod(fam)


class DualPathFp8DenseLinearMethod(LinearMethodBase):
    def __init__(self, family: str) -> None:
        self.family = family
        self.prefill_mode = _prefill_mode()
        self.decode_mode = "fp8" if family in _fp8_override_families() else _decode_mode()

    # Identical to UnquantizedLinearMethod.create_weights so loading is unchanged.
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs):
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.dual_fp8_N = sum(output_partition_sizes)
        layer.dual_fp8_K = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin

        w = layer.weight.data
        assert w.dim() == 2 and w.dtype == torch.bfloat16, (w.shape, w.dtype)
        N, K = w.shape
        scale = (w.abs().amax(dim=1).float() / FP8_MAX).clamp(min=1e-12)  # [N]
        w8 = (w.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()  # [N, K]
        # W8A8 / dequant copy (plain layout)
        layer.dual_w8 = Parameter(w8, requires_grad=False)
        layer.dual_w8_scale = Parameter(scale.reshape(N, 1).contiguous(), requires_grad=False)  # [N, 1]
        # Marlin copy for the decode/verification regime (FP8 W8A16 or NVFP4 W4A16)
        holder = torch.nn.Module()
        holder.input_size_per_partition = K
        holder.output_size_per_partition = N
        holder.orig_dtype = torch.bfloat16
        holder.params_dtype = torch.bfloat16
        nvfp4_canonical = None
        if self.decode_mode == "nvfp4":
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin

            packed, block_scale, global_scale = quantize_nvfp4_for_marlin(w)
            nvfp4_canonical = None
            if self.family in _humming_nvfp4_families():
                # The bytes Marlin is about to repack (prepare_fp4_layer_for_marlin replaces the holder's tensors).
                nvfp4_canonical = dict(weight=packed.clone().contiguous().view(torch.int32),
                                       weight_scale=block_scale.clone(), weight_scale_2=global_scale.clone())
            holder.weight = Parameter(packed, requires_grad=False)
            holder.weight_scale = Parameter(block_scale, requires_grad=False)
            holder.weight_global_scale = Parameter(global_scale, requires_grad=False)
            prepare_fp4_layer_for_marlin(holder)
            layer.dual_marlin_global_scale = holder.weight_global_scale
        else:
            holder.weight = Parameter(w8.clone(), requires_grad=False)
            holder.weight_scale = Parameter(scale.clone().contiguous(), requires_grad=False)
            prepare_fp8_layer_for_marlin(holder, size_k_first=False)
            layer.dual_marlin_global_scale = None
        layer.dual_marlin_weight = holder.weight
        layer.dual_marlin_scale = holder.weight_scale
        layer.dual_marlin_workspace = holder.workspace
        layer.dual_chunk_rows = CHUNK_ROWS if (N * K) >= CHUNK_WHEN_ELEMENTS_AT_LEAST else 0
        layer.dual_humming_nvfp4 = None
        if self.decode_mode == "nvfp4" and nvfp4_canonical is not None:
            import json as _json

            from humming.transform import transform_humming_tensors

            config, kernels = _humming_nvfp4_layer_config(N, K, w.device)
            layer.dual_humming_nvfp4 = dict(
                config=config, kernels=kernels, tensors=transform_humming_tensors(config, nvfp4_canonical),
                locks=torch.zeros(1024, device=w.device, dtype=torch.int32),
                compute=_json.dumps(dict(gemm_type="dense", use_f16_accum=False)), N=N)
            logger.info_once("GLM53 Humming NVFP4: family %s engaged for %d<M<=%d (N=%d K=%d)",
                             self.family, MARLIN_MAX_M, HUMMING_NVFP4_MAX_M, N, K, scope="global")
        layer.dual_humming = None
        if self.family in _humming_families():
            if self.decode_mode != "fp8":
                raise ValueError("GLM53_HUMMING_DENSE requires VLLM_GLM53_DUAL_FP8_DECODE=fp8; "
                                 f"family {self.family} is in {self.decode_mode} mode")
            import json as _json

            from humming.transform import transform_humming_tensors

            config, kernels = _humming_layer_config(N, K, w8.device)
            # Same canonical bytes the native path keeps: the FP8 weight and the
            # per-channel scale Marlin rounds to BF16. No second quantization.
            canonical = dict(weight=layer.dual_w8.detach().contiguous().view(torch.int32),
                             weight_scale=layer.dual_w8_scale.detach().to(torch.bfloat16).contiguous())
            tensors = transform_humming_tensors(config, canonical)
            layer.dual_humming = dict(
                config=config, kernels=kernels, tensors=tensors,
                locks=torch.zeros(1024, device=w8.device, dtype=torch.int32),
                compute=_json.dumps(dict(gemm_type="dense", use_f16_accum=False)), N=N)
            logger.info_once("GLM53 Humming dense: family %s engaged for M<=%d (N=%d K=%d)",
                             self.family, MARLIN_MAX_M, N, K, scope="global")
        # Free the BF16 tensor; forward never reads layer.weight again.
        layer.register_parameter("weight", None)
        torch.cuda.empty_cache()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear

        N, K = layer.dual_fp8_N, layer.dual_fp8_K
        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        if M <= MARLIN_MAX_M:
            hs = getattr(layer, "dual_humming", None)
            if hs is not None and M in hs["kernels"]:
                import json as _json

                from humming.forward import humming_forward

                out = humming_forward(hs["config"], inputs=x2, locks=hs["locks"],
                                      compute_config=hs["compute"],
                                      tuning_config=_json.dumps(hs["kernels"][M]),
                                      **hs["tensors"])[..., : hs["N"]]
                if bias is not None:
                    out = out + bias
                return out.view(*x.shape[:-1], N)
            if self.decode_mode == "nvfp4":
                from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

                out = apply_fp4_marlin_linear(input=x2, weight=layer.dual_marlin_weight, weight_scale=layer.dual_marlin_scale,
                                              weight_global_scale=layer.dual_marlin_global_scale, workspace=layer.dual_marlin_workspace,
                                              size_n=N, size_k=K, bias=bias)
            else:
                out = apply_fp8_marlin_linear(x2, layer.dual_marlin_weight, layer.dual_marlin_scale, layer.dual_marlin_workspace, N, K, bias)
            return out.view(*x.shape[:-1], N)
        hn = getattr(layer, "dual_humming_nvfp4", None)
        if hn is not None and M in hn["kernels"]:
            import json as _json

            from humming.forward import humming_forward

            out = humming_forward(hn["config"], inputs=x2, locks=hn["locks"], compute_config=hn["compute"],
                                  tuning_config=_json.dumps(hn["kernels"][M]), **hn["tensors"])[..., : hn["N"]]
            if bias is not None:
                out = out + bias
            return out.view(*x.shape[:-1], N)
        if self.prefill_mode == "dequant":
            w = (layer.dual_w8.to(torch.float32) * layer.dual_w8_scale).to(x2.dtype)
            out = torch.nn.functional.linear(x2, w, bias)
            return out.view(*x.shape[:-1], N)
        w8t = layer.dual_w8.t()  # [K, N] column-major view for the scaled GEMM
        chunk = layer.dual_chunk_rows
        if chunk and M > chunk:
            # Row chunks write straight into one preallocated output through the
            # underlying op (its first argument is the destination), so no
            # concatenation copy: the a93 prefill capture charged that copy 1.9%.
            out = torch.empty((M, N), dtype=x2.dtype, device=x2.device)
            for i in range(0, M, chunk):
                xs = x2[i : i + chunk]
                q, s = ops.scaled_fp8_quant(xs, use_per_token_if_dynamic=True)
                torch.ops._C.cutlass_scaled_mm(out[i : i + xs.shape[0]], q, w8t, s, layer.dual_w8_scale, bias)
        else:
            q, s = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
            out = ops.cutlass_scaled_mm(q, w8t, s, layer.dual_w8_scale, x2.dtype, bias)
        return out.view(*x.shape[:-1], N)
