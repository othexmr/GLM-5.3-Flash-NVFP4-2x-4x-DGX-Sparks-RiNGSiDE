#!/usr/bin/env python3
"""Install the dual-path FP8 head hook in VocabParallelEmbedding.__init__ (anchor-checked).
Only ParallelLMHead instances with the `head` family selected are affected."""
from pathlib import Path

EMB = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_embedding_source(source: str) -> str:
    return replace_once(
        source,
        '''        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
''',
        '''        self.quant_method: QuantizeMethodBase = quant_method
        # GLM-5.3 Flash NVFP4 lane: online dual-path FP8 for the output head
        # (VLLM_GLM53_DUAL_FP8_DENSE contains `head`); a no-op otherwise.
        from vllm.model_executor.layers.quantization.glm53_dual_fp8_dense import (
            maybe_dual_fp8_head_method,
        )

        _dual_head = maybe_dual_fp8_head_method(self, prefix)
        if _dual_head is not None:
            self.quant_method = _dual_head

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
''',
    )


def main() -> None:
    source = patch_embedding_source(EMB.read_text())
    EMB.write_text(source)
    compile(source, str(EMB), "exec")
    print("VocabParallelEmbedding dual-path FP8 head hook installed")


if __name__ == "__main__":
    main()
