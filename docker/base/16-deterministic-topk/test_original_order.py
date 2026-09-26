"""Repeat identical inputs to distinguish selection accuracy from order stability."""
import json
import torch
import vllm._custom_ops
from nvfp4_topk_pr55314 import persistent_topk

torch.manual_seed(51321)
rows, n, k = 10, 8192, 512
lengths = torch.full((rows,), n, dtype=torch.int32, device='cuda')
starts = torch.zeros_like(lengths)
out = torch.empty((rows,k),dtype=torch.int32,device='cuda')
workspace = torch.empty(1048576,dtype=torch.uint8,device='cuda')
results=[]
for family in ('unique', 'ties'):
    scores = torch.stack([torch.randperm(n, device='cuda').float() for _ in range(rows)])
    if family == 'ties':
        scores.remainder_(7)
    for name in ('prefill', 'decode'):
        snapshots=[]
        for rep in range(8):
            out.fill_(-999)
            if name == 'prefill':
                torch.ops._C.top_k_per_row_prefill(scores, starts, lengths, out, rows, scores.stride(0), 1, k)
            else:
                persistent_topk(scores,lengths,out,workspace,k,n*4)
            torch.cuda.synchronize()
            snapshots.append(out.clone())
        selected=[torch.gather(scores,1,o.long()).sort(1).values for o in snapshots]
        expected=scores.topk(k,dim=1).values.sort(1).values
        results.append({'family':family,'path':name,
                        'selected_values_correct':all(torch.equal(s,expected)for s in selected),
                        'exact_order_stable':all(torch.equal(snapshots[0],o)for o in snapshots),
                        'selected_ids_stable':all(torch.equal(snapshots[0].sort(1).values,o.sort(1).values)for o in snapshots),
                        'different_positions':[int((snapshots[0]!=o).sum())for o in snapshots]})
print(json.dumps(results,indent=2))
