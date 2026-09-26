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

output = Path('/gate/nope-tail-gpu.json')
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


for block_size in (64, 4608):
    for padded in (False, True):
        for dtype_name in ('bf16', 'fp8'):
            for rows in (5, 10, 20):
                torch.manual_seed(731 + rows + block_size)
                scale = 1.0 if dtype_name == 'bf16' else 0.25
                dtype = torch.bfloat16 if dtype_name == 'bf16' else torch.float8_e4m3fn
                max_length = 9216 if block_size == 4608 else 4096
                blocks_per_request = math.ceil(max_length / block_size)
                physical_blocks = 2 * blocks_per_request
                stride_rows = block_size + (32 if padded else 0)
                table_cpu = torch.randperm(physical_blocks).reshape(2, blocks_per_request)
                table = table_cpu.to(device='cuda', dtype=torch.int32)
                request_ids = torch.arange(rows, device='cuda', dtype=torch.int32) % 2
                lengths = [2048, 2049, 2050, 2051, 2052, 4093, 4094, 4095, 4096]
                if block_size == 4608:
                    lengths += [4609, 4610, 4611, 4612, 9213, 9214, 9215, 9216]
                indices_cpu = torch.full((rows, 2176), -1, dtype=torch.int32)
                tails = []
                active_rows = min(rows, 15)
                for row in range(active_rows):
                    length = lengths[row % len(lengths)]
                    tail_count = length % 4
                    indices_cpu[row, :2048] = torch.arange(2048, dtype=torch.int32)
                    if tail_count:
                        indices_cpu[row, 2048:2048 + tail_count] = torch.arange(length - tail_count, length, dtype=torch.int32)
                    tails.append(tail_count)
                # One explicitly masked row even in the smallest shape.
                indices_cpu[active_rows - 1].fill_(-1)
                tails[active_rows - 1] = 0
                indices = indices_cpu.to('cuda')
                storage_float = torch.randn(physical_blocks, stride_rows, 512, device='cuda') * 0.025
                # Poison physical gaps: any stride bug reaches NaNs.
                if padded:
                    storage_float[:, block_size:].fill_(float('nan'))
                for row in range(active_rows):
                    req = row % 2
                    for token in indices_cpu[row, 2048:].tolist():
                        if token < 0:
                            continue
                        physical = int(table_cpu[req, token // block_size])
                        storage_float[physical, token % block_size].fill_(4.0)
                storage = (storage_float / scale).to(dtype)
                cache = storage[:, :block_size]
                q = torch.ones(rows, 32, 512, device='cuda', dtype=torch.bfloat16)
                metadata = SimpleNamespace(topk_tokens=2048, block_size=block_size, req_id_per_token=request_ids, block_table=table)
                impl = SimpleNamespace(kv_cache_dtype='fp8_e4m3' if dtype_name == 'fp8' else 'bfloat16', topk_indices_buffer=indices, num_heads=32, kv_lora_rank=512, softmax_scale=1 / math.sqrt(512))
                layer = SimpleNamespace(_k_scale_float=scale)

                # Check the real converter against a separate CPU page mapping.
                flat, observed_stride = backend.flat_kv_row_view(cache, block_size)
                expected_cpu = torch.full_like(indices_cpu, -1)
                for row in range(rows):
                    for column, token in enumerate(indices_cpu[row].tolist()):
                        if token >= 0:
                            expected_cpu[row, column] = int(table_cpu[row % 2, token // block_size]) * stride_rows + token % block_size
                converted = backend.triton_convert_req_index_to_global_index(request_ids, table, indices, BLOCK_SIZE=block_size, NUM_TOPK_TOKENS=2176, BLOCK_STRIDE_ROWS=observed_stride)
                assert torch.equal(converted.cpu(), expected_cpu)
                assert observed_stride == stride_rows

                # Match actual FP8->BF16 dequantization before the FP32 oracle.
                dequantized = (cache.float() * scale).to(torch.bfloat16).float()
                expected = reference(q, dequantized, table, indices, request_ids)
                old, _ = original(impl, q, cache, metadata, layer)
                new, _ = candidate(impl, q, cache, metadata, layer)
                torch.cuda.synchronize()
                old_error = compare(old, expected)
                new_error = compare(new, expected)
                assert new_error['max_abs'] < 0.035 and new_error['relative_l2'] < 0.012, new_error
                tail_rows = [row for row, count in enumerate(tails) if count]
                assert tail_rows
                assert float((old[tail_rows].float() - expected[tail_rows]).abs().max()) > 1.0, old_error
                masked = (indices >= 0).sum(1) == 0
                assert torch.count_nonzero(new[masked]) == 0
                poisoned = torch.full_like(q, float('nan'))
                op.sparse_fwd(q, flat, converted, impl.softmax_scale, scale, poisoned)
                assert torch.equal(poisoned, new)

                # Warm all shape-specific paths before capture, then change values
                # in the bound input storage and compare three graph replays.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        candidate(impl, q, cache, metadata, layer)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output, _ = candidate(impl, q, cache, metadata, layer)
                graph_max = 0.0
                for replay in range(3):
                    q.copy_((torch.randn_like(q.float()) * 0.4).to(torch.bfloat16))
                    # Replace logical KV values while preserving all storage identities.
                    replacement = torch.randn_like(storage_float) * 0.1
                    if padded:
                        replacement[:, block_size:].fill_(float('nan'))
                    storage.copy_((replacement / scale).to(dtype))
                    graph_output.fill_(float('nan'))
                    graph.replay()
                    eager, _ = candidate(impl, q, cache, metadata, layer)
                    torch.cuda.synchronize()
                    assert torch.equal(graph_output, eager)
                    ref = reference(q, (cache.float() * scale).to(torch.bfloat16).float(), table, indices, request_ids)
                    error = compare(eager, ref)
                    assert error['max_abs'] < 0.01 and error['relative_l2'] < 0.012, error
                    graph_max = max(graph_max, error['max_abs'])
                    assert torch.count_nonzero(eager[masked]) == 0
                row = {'block_size': block_size, 'padded_stride': padded, 'kv_dtype': dtype_name, 'rows': rows, 'old_error': old_error, 'full_tail_error': new_error, 'random_graph_max_abs': graph_max, 'graph_replays': 3, 'converter_exact': True, 'poison_output_exact': True, 'tail_counts': tails}
                report['cases'].append(row)
                output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)
                del graph, graph_output, old, new, expected, eager, storage, cache, storage_float, flat, poisoned, dequantized
                torch.cuda.empty_cache()
report['pass'] = len(report['cases']) == 24
output.write_text(json.dumps(report, indent=2) + '\n')
assert report['pass']
print('NOPE_FULL_TAIL_GPU_ORACLE_PASS', flush=True)
