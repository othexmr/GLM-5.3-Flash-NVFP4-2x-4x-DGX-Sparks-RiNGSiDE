import json
import torch
import vllm._custom_ops
from nvfp4_topk_pr55314 import persistent_topk
from deterministic_sparse_topk import canonicalize_topk

torch.manual_seed(51321)
results = []
for n in (2048, 8193, 32769, 65536):
    rows, k = 8, 512
    for family in ('unique', 'clustered', 'ties'):
        store = torch.full((rows,n+7),float('nan'),device='cuda')
        scores = store[:,:n]
        for r in range(rows):
            v = torch.randperm(n,device='cuda')
            if family == 'clustered':
                v = (v.to(torch.int32)+0x3f800000).view(torch.float32)
            elif family == 'ties':
                v = v.remainder(7).float()
            scores[r] = v.float()
        for path in ('prefill', 'decode'):
            start = torch.tensor([0,0,0,0,0,31,127,511],device='cuda',dtype=torch.int32)
            if path == 'decode': start.zero_()
            lengths = torch.tensor([0,1,511,512,513,n-31,n-127,n-511],device='cuda',dtype=torch.int32)
            ends = start + lengths
            out_store = torch.full((rows,k+3),-777,device='cuda',dtype=torch.int32)
            out = out_store[:,:k]
            workspace = torch.empty(1048576,device='cuda',dtype=torch.uint8)
            # Native operators assume dense output; retain a dense output for
            # selection, then verify the canonicalizer's separate padded stride.
            native = torch.empty((rows,k),device='cuda',dtype=torch.int32)
            def run():
                if path == 'prefill':
                    torch.ops._C.top_k_per_row_prefill(scores,start,ends,native,rows,scores.stride(0),1,k)
                else:
                    persistent_topk(scores,ends,native,workspace,k,n*4)
                out.copy_(native)
                canonicalize_topk(scores,start if path=='prefill' else None,ends,out)
            for _ in range(2):run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            for rep in range(3):
                if rep: scores.copy_(scores.roll(17,dims=1))
                reference=torch.full((rows,k),-1,dtype=torch.int32)
                cpu=scores.cpu()
                for r,(a,z) in enumerate(zip(start.cpu().tolist(),ends.cpu().tolist())):
                    ids=sorted(range(z-a),key=lambda i:(-float(cpu[r,a+i]),i))[:k]
                    reference[r,:len(ids)]=torch.tensor(sorted(ids),dtype=torch.int32)
                out_store.fill_(-777)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out.cpu(),reference),(n,family,path,rep,'graph')
                assert (out_store[:,k:]==-777).all(), 'padding overwritten'
                run();torch.cuda.synchronize()
                assert torch.equal(out.cpu(),reference),(n,family,path,rep,'eager')
            results.append({'columns':n,'family':family,'path':path,'pass':True})
            print(json.dumps(results[-1]),flush=True)
            del graph
print('DETERMINISTIC_TOPK_PASS',len(results))
