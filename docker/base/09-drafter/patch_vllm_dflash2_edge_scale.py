#!/usr/bin/env python3
"""Anchor-checked patch: a frozen scale on the DFlash2 candidate selector's edge term.

`_score_edges` in qwen3_dflash2.py scores candidate c at position i as
unary_logits[i, c] + <P[p] * W h_i, S[c]> (the pairwise edge term against the previously
selected candidate p). VLLM_GLM53_DFLASH2_EDGE_SCALE (float, default 1.0) multiplies the
edge term so its weight against the unary logits can be calibrated without retraining.
At 1.0 (or unset) the expression is unchanged.
"""
from pathlib import Path

DRAFT = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_draft_source(source: str) -> str:
    source = replace_once(
        source,
        '''    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )
''',
        '''    predecessors = predecessor_table[predecessor_ids]
    edge = torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )
    if _GLM53_EDGE_SCALE != 1.0:
        # GLM-5.3 Flash NVFP4 lane: frozen edge-term calibration
        # (VLLM_GLM53_DFLASH2_EDGE_SCALE); unchanged at 1.0.
        edge = edge * _GLM53_EDGE_SCALE
    return unary_logits[:, :, None] + edge
''',
    )
    source = replace_once(
        source,
        "\n\ndef _score_edges(\n",
        '\n\nimport os as _glm53_os\n\n_GLM53_EDGE_SCALE = float(_glm53_os.environ.get("VLLM_GLM53_DFLASH2_EDGE_SCALE", "1.0") or "1.0")\n\n\ndef _score_edges(\n',
    )
    return source


def main() -> None:
    source = patch_draft_source(DRAFT.read_text())
    DRAFT.write_text(source)
    compile(source, str(DRAFT), "exec")
    print("DFlash2 edge-scale switch installed")


if __name__ == "__main__":
    main()
