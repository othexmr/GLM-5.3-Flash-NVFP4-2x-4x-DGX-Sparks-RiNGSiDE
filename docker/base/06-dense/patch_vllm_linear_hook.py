#!/usr/bin/env python3
"""Install the dual-path FP8 dense hook in LinearBase.__init__ (anchor-checked)."""
from pathlib import Path

LINEAR = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_linear_source(source: str) -> str:
    return replace_once(
        source,
        '''        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
            self.quant_method = quant_method
        else:
            raise ValueError("All linear layers should support quant method.")
''',
        '''        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
            self.quant_method = quant_method
        else:
            raise ValueError("All linear layers should support quant method.")
        # GLM-5.3 Flash NVFP4 lane: online dual-path FP8 for selected BF16 dense
        # projections (VLLM_GLM53_DUAL_FP8_DENSE); a no-op unless selected.
        from vllm.model_executor.layers.quantization.glm53_dual_fp8_dense import (
            maybe_dual_fp8_dense_method,
        )

        _dual = maybe_dual_fp8_dense_method(self, prefix)
        if _dual is not None:
            self.quant_method = _dual
''',
    )


def main() -> None:
    source = patch_linear_source(LINEAR.read_text())
    LINEAR.write_text(source)
    compile(source, str(LINEAR), "exec")
    print("LinearBase dual-path FP8 hook installed")


if __name__ == "__main__":
    main()
