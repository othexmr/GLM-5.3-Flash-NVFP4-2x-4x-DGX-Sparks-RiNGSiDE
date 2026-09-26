# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-License-Identifier: Apache-2.0
"""CPU-only source, actual import, schema and fake-tracing installation check."""
import hashlib,json
from pathlib import Path
import torch
import nvfp4_topk_pr55314 as repair
D=Path(__file__).resolve().parent
root=Path('/usr/local/lib/python3.12/dist-packages')
m=json.loads((D/'callsite-manifest.json').read_text())
for rel,row in m['files'].items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==row['after'],rel
assert Path(repair.__file__).resolve().parent==root/'nvfp4_topk_pr55314'
schema=str(repair.persistent_topk._schema)
assert 'Tensor(a!) output' in schema and 'Tensor(b!) workspace' in schema, schema
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    logits=torch.empty((5,5000),device='cuda',dtype=torch.float32)
    lengths=torch.empty((5,),device='cuda',dtype=torch.int32)
    output=torch.empty((5,512),device='cuda',dtype=torch.int32)
    workspace=torch.empty((1048576,),device='cuda',dtype=torch.uint8)
    assert repair.persistent_topk(logits,lengths,output,workspace,512,20000) is None
print(json.dumps({'pass':True,'binary_sha256':repair.build_receipt['binary_sha256'],'schema':schema,'fake_trace':True,'gpu_execution':False},indent=2))
