#!/usr/bin/env python3
"""Verify this bundle, or compile only its standalone CUDA extension.

Default build is lead-owned and must run in the exact runtime toolchain.
--verify-only is CPU-only and does not import torch or start compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIN = "4500c80c080328dfe62435d083f4063e00d987df"
PR = "a68f6649976720b85c5e2df50af6aadd293cf5ca"
DIFF_SHA = "e09d108e2c0f880552092b471622628ba25d6a27ab13ced83ca586e8421886fe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("bundle verification failure: " + message)


def verify() -> tuple[dict, str]:
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    require(manifest["vllm_commit"] == PIN, "vLLM source pin")
    require(manifest["pr_head_commit"] == PR, "upstream PR head")
    require(sha(ROOT / "upstream-pr55314.diff") == DIFF_SHA == manifest["pr_diff_sha256"],
            "upstream PR diff digest")
    require(sha(ROOT / "LICENSE") == manifest["license_sha256"], "LICENSE digest")
    for kind, prefix in (("original_files", "original"), ("patched_files", "src")):
        for row in manifest[kind]:
            require(sha(ROOT / prefix / row["path"]) == row["sha256"],
                    prefix + "/" + row["path"])
    if (ROOT / "bundle-files.sha256.json").exists():
        for relative, expected in json.loads((ROOT / "bundle-files.sha256.json").read_text()).items():
            require(sha(ROOT / relative) == expected, relative)
    return manifest, sha(ROOT / "source-manifest.json")


def main() -> None:
    require(not sys.flags.optimize,
            "hash gates require assertions enabled; re-run without -O / PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--cuda-arch", default="121", choices=("121",))
    args = parser.parse_args()
    manifest, manifest_sha = verify()
    if args.verify_only:
        print(json.dumps({"verified": True, "source_manifest_sha256": manifest_sha,
                          "compiled": False, "gpu_executed": False}, indent=2))
        return

    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("The exact CUDA runtime build toolchain is required")
    compiler = Path(CUDA_HOME) / "bin" / "nvcc"
    nvcc_version = subprocess.run([str(compiler), "--version"], check=True,
                                  text=True, capture_output=True).stdout
    common = ["-O3", "-std=c++20", "-DUSE_CUDA",
              "-DTORCH_TARGET_VERSION=0x020B000000000000ULL"]
    cflags = common + ["-fvisibility=hidden"]
    cuda_flags = common + ["--expt-relaxed-constexpr", "--expt-extended-lambda",
                          "-Xcompiler=-fvisibility=hidden",
                          "-gencode=arch=compute_121,code=sm_121"]
    includes = [str(ROOT / "src")]
    cccl = Path(CUDA_HOME) / "include" / "cccl"
    if cccl.is_dir():
        includes.append(str(cccl))
    identity = {"source_manifest_sha256": manifest_sha,
                "binding_sha256": sha(ROOT / "src" / "bindings.cu"),
                "loader_sha256": sha(ROOT / "module_template.py"),
                "build_script_sha256": sha(Path(__file__).resolve()),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": torch.version.cuda,
                "cuda_arch": args.cuda_arch, "nvcc_version": nvcc_version,
                "cflags": cflags, "cuda_flags": cuda_flags}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    name = "nvfp4_topk_pr55314_" + identity_sha[:16]
    build_dir = ROOT / ".build" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    load(name=name, sources=[str(ROOT / "src" / "bindings.cu")],
         extra_include_paths=includes, extra_cflags=cflags,
         extra_cuda_cflags=cuda_flags, extra_ldflags=["-Wl,-Bsymbolic"],
         build_directory=str(build_dir), with_cuda=True,
         is_python_module=False, verbose=True)
    binaries = list(build_dir.glob(name + "*.so"))
    if len(binaries) != 1:
        raise RuntimeError("Expected one standalone binary, found " + repr(binaries))
    # Copy a self-contained runtime package; do not install or modify vLLM.
    runtime = ROOT / "runtime" / "nvfp4_topk_pr55314"
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binaries[0], runtime / "_kernel.so")
    shutil.copy2(ROOT / "module_template.py", runtime / "__init__.py")
    shutil.copy2(ROOT / "source-manifest.json", runtime / "source-manifest.json")
    shutil.copy2(ROOT / "LICENSE", runtime / "LICENSE")
    receipt = identity | {"built_utc": datetime.now(timezone.utc).isoformat(),
                          "binary_sha256": sha(runtime / "_kernel.so"),
                          "binary_name": binaries[0].name,
                          "build_identity_sha256": identity_sha,
                          "scope": "compiled standalone operator only; GPU/model qualification required"}
    encoded = json.dumps(receipt, indent=2) + "\n"
    (runtime / "build-receipt.json").write_text(encoded)
    (ROOT / "build-receipt.json").write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
