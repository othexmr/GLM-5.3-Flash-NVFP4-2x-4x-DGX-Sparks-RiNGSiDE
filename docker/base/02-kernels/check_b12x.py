"""Build-time check that the derived image can select B12X NVFP4 kernels.

Runs inside the image without a GPU: it only proves the imports resolve and the
vLLM in the base image has the kernel-selection modules that know about B12X.
The live proof is the kernel-evidence capture after health.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import torch

major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 12):
    sys.exit(f"b12x needs torch >= 2.12, image has {torch.__version__}")

tvm_ffi_version = importlib.metadata.version("apache-tvm-ffi")
tilelang_version = importlib.metadata.version("tilelang")
if (tvm_ffi_version, tilelang_version) != ("0.1.11", "0.1.12"):
    sys.exit(
        "source-pinned TileLang pair changed: "
        f"apache-tvm-ffi={tvm_ffi_version} tilelang={tilelang_version}"
    )

b12x = importlib.import_module("b12x")
linear = importlib.import_module("vllm.model_executor.kernels.linear.nvfp4.b12x")
oracle = importlib.import_module("vllm.model_executor.layers.fused_moe.oracle.nvfp4")
glm5next = importlib.import_module("vllm.models.glm5next.nvidia.model")
interfaces = importlib.import_module("vllm.model_executor.models.interfaces")
backends = {member.name for member in oracle.NvFp4MoeBackend}
missing = {"B12X", "FLASHINFER_B12X", "MARLIN"} - backends
if missing:
    sys.exit(f"vLLM NVFP4 MoE oracle lacks {sorted(missing)}; has {sorted(backends)}")
if not interfaces.supports_eagle3(glm5next.Glm5NextForConditionalGeneration):
    sys.exit("GLM5Next conditional model lacks the DFlash2 auxiliary-state interface")
print(
    "b12x import ok; torch",
    torch.__version__,
    "; b12x",
    getattr(b12x, "__version__", "unknown"),
    "; apache-tvm-ffi/tilelang",
    f"{tvm_ffi_version}/{tilelang_version}",
    "; linear kernel classes",
    [name for name in dir(linear) if name.endswith("LinearKernel")],
    "; glm5next dflash aux interface ok",
)
