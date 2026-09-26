# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-License-Identifier: Apache-2.0
import hashlib,json,py_compile,subprocess,sys
from pathlib import Path
def _require(condition,message):
    if not condition:raise RuntimeError(message)
_require(not sys.flags.optimize,'hash gates require assertions enabled; re-run without -O / PYTHONOPTIMIZE')
D=Path(__file__).resolve().parent
root=Path('/usr/local/lib/python3.12/dist-packages')
m=json.loads((D/'source-manifest.json').read_text())
for rel,row in m['files'].items():
    _require(hashlib.sha256((root/rel).read_bytes()).hexdigest()==row['before'],'Unexpected source preimage: '+rel)
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(D/'attention-tail.patch')],cwd=root,check=True)
for rel,row in m['files'].items():
    _require(hashlib.sha256((root/rel).read_bytes()).hexdigest()==row['after'],'Installed payload differs: '+rel)
    py_compile.compile(str(root/rel),doraise=True)
print('Full KPool attention support source hash verified')
