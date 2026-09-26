#!/usr/bin/env python3
"""Anchor-checked patch: stop evaluating the GLM-5.3 Flash router gate twice per MoE layer.

`Glm5NextMoE.forward` computes `router_logits = self.gate(hidden_states)` and hands them to
`self.experts`, but the MoE runner also owns the gate (`gate=self.gate` in FusedMoEFactory)
and recomputes the logits after launching the shared experts (moe_runner.py, "If the Runner
holds the gate, apply it after the stream sync"); the a100 trace shows two [8,3,8] cuBLAS
GEMMs plus two split-K reductions before every one of the 42 routed-MoE kernels. With
VLLM_GLM53_ROUTER_DEDUP=1 the model passes an uninitialised placeholder of the right shape
and dtype instead of the first evaluation; the runner's own evaluation (the one that
overlaps the shared-expert stream) is the only one left. The custom op's fake
implementation derives shapes from hidden_states only, so the placeholder is never read.
Default off (unset or 0) keeps the original code path.
"""
from pathlib import Path

MODEL = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_model_source(source: str) -> str:
    source = replace_once(
        source,
        '''        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
''',
        '''        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        # GLM-5.3 Flash NVFP4 lane: the runner owns this gate and recomputes the
        # logits itself (overlapped with the shared experts), so with
        # VLLM_GLM53_ROUTER_DEDUP=1 the evaluation here is skipped and a
        # placeholder of the same shape and dtype is passed instead.
        if _GLM53_ROUTER_DEDUP and self.experts.gate is not None:
            router_logits = hidden_states.new_empty(
                (hidden_states.shape[0], self.gate.weight.shape[0]),
                dtype=self.gate.out_dtype or hidden_states.dtype,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
''',
    )
    # module-level switch, read once at import
    source = replace_once(
        source,
        "\nfrom vllm.model_executor.layers.fused_moe import (",
        '\nimport os as _glm53_os\n\n_GLM53_ROUTER_DEDUP = _glm53_os.environ.get("VLLM_GLM53_ROUTER_DEDUP", "0").strip() == "1"\n\nfrom vllm.model_executor.layers.fused_moe import (',
    )
    return source


def main() -> None:
    source = patch_model_source(MODEL.read_text())
    MODEL.write_text(source)
    compile(source, str(MODEL), "exec")
    print("GLM-5.3 router dedup switch installed")


if __name__ == "__main__":
    main()
