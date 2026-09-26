"""Prebuilt, hash-checked PR55314 top-k repair. No lazy compile or fallback."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_RECEIPT = json.loads((_ROOT / "build-receipt.json").read_text())
_SOURCE_BYTES = (_ROOT / "source-manifest.json").read_bytes()
_SOURCE = json.loads(_SOURCE_BYTES)
_EXPECTED_VLLM = "4500c80c080328dfe62435d083f4063e00d987df"
_EXPECTED_PR = "a68f6649976720b85c5e2df50af6aadd293cf5ca"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("nvfp4 top-k repair identity failure: " + message)


_require(_SOURCE["vllm_commit"] == _EXPECTED_VLLM, "vLLM source pin")
_require(_SOURCE["pr_head_commit"] == _EXPECTED_PR, "upstream PR head")
_require(
    hashlib.sha256(_SOURCE_BYTES).hexdigest() == _RECEIPT["source_manifest_sha256"],
    "source manifest digest",
)
_require(str(torch.__version__) == _RECEIPT["torch_version"], "PyTorch version")
_require(torch.version.cuda == _RECEIPT["torch_cuda_version"], "PyTorch CUDA version")
_BINARY = _ROOT / "_kernel.so"
_require(
    hashlib.sha256(_BINARY.read_bytes()).hexdigest() == _RECEIPT["binary_sha256"],
    "compiled binary digest",
)

# Import-time only. The production indexer imports this module before profiling
# or capture; each worker independently verifies the exact prebuilt library.
torch.ops.load_library(str(_BINARY))


@torch.library.register_fake("nvfp4_topk_pr55314::persistent_topk")
def _fake(logits, lengths, output, workspace, k, max_seq_len):
    return None


persistent_topk = torch.ops.nvfp4_topk_pr55314.persistent_topk.default
build_receipt = dict(_RECEIPT)
logging.getLogger(__name__).info(
    "Loaded nvfp4 PR55314 top-k binary=%s source_manifest=%s",
    _RECEIPT["binary_sha256"],
    _RECEIPT["source_manifest_sha256"],
)
