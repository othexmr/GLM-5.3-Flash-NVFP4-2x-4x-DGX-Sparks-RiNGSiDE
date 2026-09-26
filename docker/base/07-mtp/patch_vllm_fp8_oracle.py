#!/usr/bin/env python3
"""Anchor-checked patch: let an FP8 MoE layer fall back to a vLLM kernel when the
runner's MoE backend is b12x (the B12X library serves NVFP4/W4A8/W6A8 experts only).

GLM-5.3 Flash NVFP4 (RedHatAI) keeps its MTP layer 45 experts in block FP8, so
`--speculative-config '{"method":"mtp"}'` under `--moe-backend b12x` failed in
`map_fp8_backend` ("moe_backend='b12x' is not supported for FP8 MoE"). The target's
45 NVFP4 MoE layers keep B12X; only FP8 MoE layers take the fallback, chosen by
VLLM_GLM53_FP8_MOE_FALLBACK (triton by default; marlin; empty restores the error).
Inert for DFlash/ngram boots, which construct no FP8 MoE layer.
"""
from pathlib import Path

ORACLE = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/fp8.py")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_oracle_source(source: str) -> str:
    return replace_once(
        source,
        '''    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for FP8 MoE. "
''',
        '''    if backend := mapping.get(runner_backend):
        return backend
    # GLM-5.3 Flash NVFP4 lane: B12X serves no FP8 experts; an FP8 MoE layer
    # (the checkpoint's MTP layer 45) falls back to a vLLM kernel.
    if runner_backend in ("b12x", "flashinfer_b12x"):
        import os

        fallback = os.environ.get("VLLM_GLM53_FP8_MOE_FALLBACK", "triton").strip().lower()
        if fallback and fallback in mapping:
            logger.info_once(
                "FP8 MoE layer under moe_backend=%s: using the %s kernel "
                "(VLLM_GLM53_FP8_MOE_FALLBACK)",
                runner_backend,
                fallback,
            )
            return mapping[fallback]
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for FP8 MoE. "
''',
    )


def main() -> None:
    source = patch_oracle_source(ORACLE.read_text())
    if "logger = " not in source and "logger=" not in source:
        raise RuntimeError("oracle/fp8.py has no module logger to reuse")
    ORACLE.write_text(source)
    compile(source, str(ORACLE), "exec")
    print("FP8 MoE oracle b12x fallback installed")


if __name__ == "__main__":
    main()
