# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
"""Apply the exact tool-call truncation repair to the two parser engine files; verify preimages and results."""
import hashlib,json,py_compile,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent

def install(root):
    manifest=json.loads((D/'source-manifest.json').read_text())
    for rel,row in manifest['files'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=row['before']:raise RuntimeError('Unexpected source preimage: '+rel)
    patch=D/'tool-call-truncation.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=manifest['patch_sha256']:raise RuntimeError('Patch hash differs')
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch)],cwd=root,check=True)
    for rel,row in manifest['files'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=row['after']:raise RuntimeError('Installed payload differs: '+rel)
        py_compile.compile(str(root/rel),doraise=True)
    return manifest

if __name__=='__main__':
    install(Path('/usr/local/lib/python3.12/dist-packages'))
    print('Tool-call truncation repair installed and verified')
