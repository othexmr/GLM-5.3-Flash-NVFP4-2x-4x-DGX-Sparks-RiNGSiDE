# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-License-Identifier: Apache-2.0
"""Install only the prebuilt top-k module and the exact KPool CUDA callsite."""
import hashlib,json,py_compile,shutil,subprocess,sys
from pathlib import Path
def _require(condition,message):
    if not condition:raise RuntimeError(message)
_require(not sys.flags.optimize,'hash gates require assertions enabled; re-run without -O / PYTHONOPTIMIZE')
D=Path(__file__).resolve().parent
root=Path('/usr/local/lib/python3.12/dist-packages')
m=json.loads((D/'callsite-manifest.json').read_text())
for rel,row in m['files'].items():
    _require(hashlib.sha256((root/rel).read_bytes()).hexdigest()==row['before'],'Unexpected source preimage: '+rel)
patch=D/'production-callsite.proposed.patch'
_require(hashlib.sha256(patch.read_bytes()).hexdigest()==m['patch_sha256'],'Patch hash differs')
target=root/'nvfp4_topk_pr55314'
_require(not target.exists(),'Unexpected existing payload: '+str(target))
runtime=D/'runtime/nvfp4_topk_pr55314'
receipt=json.loads((runtime/'build-receipt.json').read_text())
_require(hashlib.sha256((runtime/'_kernel.so').read_bytes()).hexdigest()==receipt['binary_sha256'],'Built binary differs from its receipt')
shutil.copytree(runtime,target)
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch)],cwd=root,check=True)
for rel,row in m['files'].items():
    _require(hashlib.sha256((root/rel).read_bytes()).hexdigest()==row['after'],'Installed payload differs: '+rel)
    py_compile.compile(str(root/rel),doraise=True)
print('Dedicated TopK module and KPool callsite installed and verified')
