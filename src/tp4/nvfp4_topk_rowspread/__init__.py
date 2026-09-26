"""ctx1m-topk rowspread top-k build (2026-09-25): the image's pinned PR55314 persistent top-k with short rows spread over
every CTA of the grid (persistent_topk.cuh rowspread). Prebuilt and hash-checked; no lazy compile, no fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_RECEIPT = json.loads((_ROOT / "build-receipt.json").read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("rowspread top-k identity failure: " + message)


_require(str(torch.__version__) == _RECEIPT["torch_version"], "PyTorch version")
_require(torch.version.cuda == _RECEIPT["torch_cuda_version"], "PyTorch CUDA version")
_BINARY = _ROOT / "_kernel.so"
_require(hashlib.sha256(_BINARY.read_bytes()).hexdigest() == _RECEIPT["binary_sha256"], "compiled binary digest")

torch.ops.load_library(str(_BINARY))


@torch.library.register_fake("nvfp4_topk_rowspread::persistent_topk")
def _fake(logits, lengths, output, workspace, k, max_seq_len):
    return None


persistent_topk = torch.ops.nvfp4_topk_rowspread.persistent_topk.default
build_receipt = dict(_RECEIPT)
