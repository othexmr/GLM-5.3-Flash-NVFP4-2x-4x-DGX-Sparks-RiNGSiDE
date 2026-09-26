# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-License-Identifier: Apache-2.0
"""GPU oracle for pinned NoPE full KPool support; no model inference."""
import ast
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import glm53_sparse_mla as op
import glm53_sparse_mla.backend as backend

assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
original = backend.Glm53SparseMLAImpl.forward_mqa
source = Path(backend.__file__).read_text()
anchor = '''        cap = int(attn_metadata.topk_tokens)
        if topk_indices.shape[1] > cap:
            topk_indices = topk_indices[:, :cap].contiguous()
'''
assert anchor not in source
assert hashlib.sha256(source.encode()).hexdigest() == json.loads((Path(__file__).parent/'source-manifest.json').read_text())['files']['glm53_sparse_mla/backend.py']['after']
candidate = original
old_source = source.replace('        head_dim = kv_c_and_k_pe_cache.shape[-1]', anchor + '        head_dim = kv_c_and_k_pe_cache.shape[-1]', 1)
tree = ast.parse(old_source)
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Glm53SparseMLAImpl')
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward_mqa')
namespace = vars(backend).copy()
exec(compile(ast.Module(body=[method], type_ignores=[]), '<truncated-negative-control>', 'exec'), namespace)
original = namespace['forward_mqa']

output = Path('/gate/nope-tail-graph-metadata.json')
assert not output.exists()
report = {
    'scope': 'Actual pinned backend/converter/CUDA op; old vs full2176-column support; independent FP32 attention over independently mapped dequantized BF16 KV. No full-model qualification.',
    'installed_backend_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'negative_control_function_sha256': hashlib.sha256(ast.unparse(method).encode()).hexdigest(),
    'gpu': torch.cuda.get_device_name(),
    'cases': [],
}


def reference(q, logical_cache, table, indices, request_ids):
    result = torch.zeros_like(q, dtype=torch.float32)
    block_size = logical_cache.shape[1]
    for row in range(q.shape[0]):
        selected = indices[row]
        selected = selected[selected >= 0].long()
        if not len(selected):
            continue
        blocks = table[request_ids[row].long(), selected // block_size].long()
        values = logical_cache[blocks, selected % block_size].float()
        scores = torch.matmul(q[row].float(), values.T) / math.sqrt(512)
        result[row] = torch.matmul(torch.softmax(scores, dim=-1), values)
    return result


def compare(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected).abs()
    denominator = max(float(torch.linalg.vector_norm(expected)), 1e-6)
    return {'max_abs': float(error.max()), 'relative_l2': float(torch.linalg.vector_norm(error)) / denominator}


block_size, stride_rows, rows, requests = 4608, 4640, 20, 3
scale = 0.3
torch.manual_seed(9104)
table = torch.arange(6, device='cuda', dtype=torch.int32).reshape(3, 2)
request_ids = torch.arange(rows, device='cuda', dtype=torch.int32) % requests
indices = torch.full((rows, 2176), -1, device='cuda', dtype=torch.int32)
q = torch.empty(rows, 32, 512, device='cuda', dtype=torch.bfloat16)
storage = torch.empty(6, stride_rows, 512, device='cuda', dtype=torch.float8_e4m3fn)
cache = storage[:, :block_size]
metadata = SimpleNamespace(topk_tokens=2048, block_size=block_size, req_id_per_token=request_ids, block_table=table)
impl = SimpleNamespace(kv_cache_dtype='fp8_e4m3', topk_indices_buffer=indices, num_heads=32, kv_lora_rank=512, softmax_scale=1/math.sqrt(512))
layer = SimpleNamespace(_k_scale_float=scale)
def refresh(replay):
    table.copy_(torch.randperm(6, device='cuda').reshape(3,2).int())
    request_ids.copy_((torch.arange(rows, device='cuda', dtype=torch.int32) + replay) % 3)
    indices.fill_(-1)
    for row in range(15):
        length = [4609,4610,4611,9213,9214,9215][(row+replay)%6]
        indices[row,:2048] = torch.randperm(length-length%4, device='cuda')[:2048].int()
        indices[row,2048:2048+length%4] = torch.arange(length-length%4,length,device='cuda').int()
    q.copy_((torch.randn_like(q.float())*.4).bfloat16())
    values = torch.randn(6,stride_rows,512,device='cuda')*.12
    values[:,block_size:].fill_(float('nan'))
    storage.copy_((values/scale).to(storage.dtype))
refresh(0)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3): candidate(impl,q,cache,metadata,layer)
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    result,_ = candidate(impl,q,cache,metadata,layer)
pointers = [t.data_ptr() for t in (q,storage,indices,table,request_ids,result)]
for replay in range(6):
    refresh(replay+1)
    result.fill_(float('nan'))
    graph.replay()
    eager,_ = candidate(impl,q,cache,metadata,layer)
    torch.cuda.synchronize()
    assert torch.equal(result,eager)
    ref = reference(q,(cache.float()*scale).bfloat16().float(),table,indices,request_ids)
    error = compare(result,ref)
    assert error['max_abs'] < .01 and error['relative_l2'] < .012, error
    assert torch.count_nonzero(result[15:]) == 0
    flat,stride = backend.flat_kv_row_view(cache,block_size)
    converted = backend.triton_convert_req_index_to_global_index(request_ids,table,indices,BLOCK_SIZE=block_size,NUM_TOPK_TOKENS=2176,BLOCK_STRIDE_ROWS=stride)
    cpu_indices, cpu_table, cpu_ids = indices.cpu(),table.cpu(),request_ids.cpu()
    expected = torch.full_like(cpu_indices,-1)
    for row in range(rows):
        selected = cpu_indices[row]>=0
        tokens = cpu_indices[row,selected].long()
        expected[row,selected] = (cpu_table[cpu_ids[row].long(),tokens//block_size]*stride_rows+tokens%block_size).int()
    assert torch.equal(converted.cpu(),expected)
    assert pointers == [t.data_ptr() for t in (q,storage,indices,table,request_ids,result)]
    record = {'replay':replay,'active_rows':15,'requests':3,'padded_rows':5,'error':error,'graph_eager_bitexact':True,'converter_exact':True,'stable_bindings':True}
    report['cases'].append(record)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(record),flush=True)
report['pass'] = True
output.write_text(json.dumps(report,indent=2)+'\n')
print('NOPE_METADATA_GRAPH_ADDENDUM_PASS',flush=True)
