# SPDX-License-Identifier: Apache-2.0
"""Build the FP8 GEMM extension (build/glm53_fp8_gemm) without a GPU and copy it to OUT/glm53_fp8_gemm.so.

    python3 -B build_fp8_gemm.py --src build/glm53_fp8_gemm --work DIR --out DIR

Calls build() of build/glm53_fp8_gemm/build_probe.py, the function the lab built the measured extension with
(PyTorch's extension loader, TORCH_CUDA_ARCH_LIST=12.1a, the image's CUTLASS v4.5.0, the recorded flags). The loader
imports the module after linking it; in a GPU-less docker build the caller provides the CUDA driver stub as
libcuda.so.1 on LD_LIBRARY_PATH for that import (docker/Dockerfile, stage fp8-gemm). The GPU probe of build_probe.py
is not run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('build_probe', a.src / 'build_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    result = {}
    module = probe.build(a.src.resolve(), a.work.resolve(), result)
    so = Path(module.__file__)
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(so, a.out / 'glm53_fp8_gemm.so')
    result['build']['out_sha256'] = hashlib.sha256((a.out / 'glm53_fp8_gemm.so').read_bytes()).hexdigest()
    (a.out / 'glm53_fp8_gemm.build.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print('build_fp8_gemm: ' + result['build']['out_sha256'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
