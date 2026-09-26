# SPDX-License-Identifier: Apache-2.0
"""Build the rowspread top-k extension of the TP4 profile (nvfp4_topk_rowspread) inside the base image.

    python3 -B build/topk_rowspread/build.py --src /opt/topk-repair/src --work DIR --out DIR

--src is the persistent top-k source tree of the base image's stage 15 (docker/base/15-topk-repair/src, copied to
/opt/topk-repair/src by that stage): vLLM's top-k sources at 4500c80c with the hunks of vllm-project/vllm pull request
55314. The build takes its topk.cu, topk_histogram_4096.cuh and torch_utils.h (sha256 checked), replaces
persistent_topk.cuh and the operator binding with this directory's rowspread versions, and compiles one CUDA
translation unit for GB10 (sm_121) with the flags of the measured build (measured-build-receipt.json).

Writes OUT/_kernel.so and OUT/build-receipt.json. The served loader (src/tp4/nvfp4_topk_rowspread/__init__.py) checks
the PyTorch and CUDA versions and the binary's sha256 against build-receipt.json and refuses anything else, so both
files always come from the same build. A rebuild is not byte-identical to the measured library; sources/apply.py
reports both as rebuilt artefacts.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the stage-15 sources this build takes unchanged (the measured build used the same bytes)
STAGE15 = {'csrc/libtorch_stable/topk.cu': '2d0288facc79a50569cbd76e4445e8ebf800ed1a495296aa5ac6dba9ad7e8f60',
           'csrc/libtorch_stable/topk_histogram_4096.cuh': '994188b08d722f92a91bbb19401846144a08df1bce54f1decbaec3adb1a927e7',
           'csrc/libtorch_stable/torch_utils.h': 'f619e1943039a67a3254f7e0679e1f3e0faa9d8fccae795f3b20df9e91a8a429'}
# this directory's files: the rowspread persistent_topk.cuh (the measured build's source, 2f7f4fe2..., plus the
# change-notice comment lines at its top) and the binding (the measured build's source)
OWN = {'persistent_topk.cuh': '7c23293a633675b14056d45f44300e2ad3e9175c5fc691c7a0a3a230cc1c7f3b',
       'bindings.cu': 'fca0bd595e1862d9341b266553b0c1dfc66e64d87db5881df981a7ba5878ca19'}
COMMON = ['-O3', '-std=c++20', '-DUSE_CUDA', '-DTORCH_TARGET_VERSION=0x020B000000000000ULL']
CFLAGS = COMMON + ['-fvisibility=hidden']
CUDA_FLAGS = COMMON + ['--expt-relaxed-constexpr', '--expt-extended-lambda', '-Xcompiler=-fvisibility=hidden',
                       '-gencode=arch=compute_121,code=sm_121']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    t0 = time.monotonic()
    tree = a.work / 'src'
    (tree / 'csrc/libtorch_stable').mkdir(parents=True, exist_ok=True)
    for rel, want in STAGE15.items():
        if sha(a.src / rel) != want:
            raise SystemExit(f'{a.src / rel} is not the stage-15 source {want[:16]}')
        shutil.copyfile(a.src / rel, tree / rel)
    for name, want in OWN.items():
        if sha(HERE / name) != want:
            raise SystemExit(f'{name} differs from the recorded source {want[:16]}')
    shutil.copyfile(HERE / 'persistent_topk.cuh', tree / 'csrc/libtorch_stable/persistent_topk.cuh')
    shutil.copyfile(HERE / 'bindings.cu', tree / 'bindings.cu')
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load
    if CUDA_HOME is None:
        raise SystemExit('CUDA toolchain missing')
    nvcc = subprocess.run([str(Path(CUDA_HOME) / 'bin/nvcc'), '--version'], check=True, text=True,
                          capture_output=True).stdout
    includes = [str(tree)]
    if (Path(CUDA_HOME) / 'include/cccl').is_dir():
        includes.append(str(Path(CUDA_HOME) / 'include/cccl'))
    identity = dict(persistent_topk_sha256=sha(tree / 'csrc/libtorch_stable/persistent_topk.cuh'),
                    binding_sha256=sha(tree / 'bindings.cu'), topk_cu_sha256=sha(tree / 'csrc/libtorch_stable/topk.cu'),
                    build_script_sha256=sha(Path(__file__)), torch_version=str(torch.__version__),
                    torch_cuda_version=torch.version.cuda, cuda_arch='121', nvcc_version=nvcc, cflags=CFLAGS,
                    cuda_flags=CUDA_FLAGS)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    name = 'nvfp4_topk_rowspread_' + identity_sha[:16]
    build_dir = a.work / 'build' / name
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '4')
    load(name=name, sources=[str(tree / 'bindings.cu')], extra_include_paths=includes, extra_cflags=CFLAGS,
         extra_cuda_cflags=CUDA_FLAGS, extra_ldflags=['-Wl,-Bsymbolic'], build_directory=str(build_dir),
         with_cuda=True, is_python_module=False, verbose=True)
    binaries = list(build_dir.glob(name + '*.so'))
    if len(binaries) != 1:
        raise SystemExit(f'expected one binary, found {binaries}')
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binaries[0], a.out / '_kernel.so')
    receipt = dict(identity, built_utc=datetime.now(timezone.utc).isoformat(), binary_sha256=sha(a.out / '_kernel.so'),
                   binary_name=binaries[0].name, build_identity_sha256=identity_sha,
                   build_seconds=round(time.monotonic() - t0, 1))
    (a.out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('nvcc_version', 'cflags', 'cuda_flags')}))


if __name__ == '__main__':
    main()
