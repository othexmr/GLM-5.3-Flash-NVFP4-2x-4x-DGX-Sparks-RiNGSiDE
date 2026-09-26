"""Independent value-set oracle for dedicated pinned TopK backport."""
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import torch
import vllm._custom_ops
parser=argparse.ArgumentParser()
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--runtime-root', type=Path, required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
assert not a.output.exists()
import sys
sys.path.insert(0,str(a.runtime_root))
import nvfp4_topk_pr55314 as repair
assert a.library.resolve() == (Path(repair.__file__).parent/'_kernel.so').resolve()
run=repair.persistent_topk
K=512
report={'scope':'Dedicated compiled TopK backport against independent torch.topk selected-value sets. Tied-score index order is not required deterministic. No model-logit attribution.', 'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(),'cases':[]}
def check(scores,lengths,out):
    torch.cuda.synchronize()
    for row,n in enumerate(lengths.cpu().tolist()):
        ids=out[row]; valid=ids[ids>=0].long()
        assert valid.numel()==min(K,n),(row,n,valid.numel())
        assert int((ids==-1).sum())==K-min(K,n),(row,'padding')
        assert valid.unique().numel()==valid.numel(),(row,'duplicates')
        if n:
            assert int(valid.max())<n,(row,n,'out of bounds')
            actual=scores[row,valid].sort().values
            expected=torch.topk(scores[row,:n],min(K,n)).values.sort().values
            assert torch.equal(actual,expected),(row,n,int((actual!=expected).sum()))
    return True
# Paths around histogram limits plus stride/vector and row dispatch boundaries.
shapes=[(1,5000),(5,5000),(10,5000),(20,5000),(32,5000),(40,5000),
        (3,4095),(3,4096),(3,4097),(5,8191),(5,8192),(5,8193),(10,16383),(10,16384),(10,16385),
        (20,32768),(20,32769),(20,49152),(20,65536)]
for rows,n in shapes:
    for family in ('spread_unique','unequal_coarse_bin','equal_ties'):
        torch.manual_seed(8221+rows+n)
        # Change physical row stride independently of valid length; NaN gaps
        # expose reading beyond a row's declared range.
        pad=3 if n%4==1 else 0
        store=torch.empty(rows,n+pad,device='cuda',dtype=torch.float32)
        scores=store[:,:n]
        lengths=torch.full((rows,),n,device='cuda',dtype=torch.int32)
        output=torch.full((rows,K),-999,device='cuda',dtype=torch.int32)
        workspace=torch.empty(1048576,device='cuda',dtype=torch.uint8)
        def refresh(rep):
            store.fill_(float('nan'))
            for row in range(rows):
                lengths[row]=n if row<max(1,rows-5) else [0,1,511,512,513][(row+rep)%5]
                m=int(lengths[row])
                if family=='spread_unique': v=torch.linspace(-4,4,max(m,1),device='cuda')[:m]
                elif family=='unequal_coarse_bin': v=(0x3f800000+torch.arange(m,device='cuda',dtype=torch.int32)).view(torch.float32)
                else: v=torch.arange(m,device='cuda',dtype=torch.int32).remainder(7).float()
                scores[row,:m]=v[torch.randperm(m,device='cuda')]
        refresh(0)
        # max_seq_len is token-granular at the active KPool callsite.
        run(scores,lengths,output,workspace,K,n*4)
        check(scores,lengths,output)
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):run(scores,lengths,output,workspace,K,n*4)
        torch.cuda.current_stream().wait_stream(side)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run(scores,lengths,output,workspace,K,n*4)
        signatures=[]
        for rep in range(3):
            refresh(rep+1)
            output.fill_(-999);workspace.fill_(0xA5)
            graph.replay();check(scores,lengths,output)
            # Eager and graph can differ in tied index choices/order; each
            # independently must select the exact mathematically valid values.
            saved=output.clone()
            output.fill_(-999);workspace.fill_(0x5A)
            run(scores,lengths,output,workspace,K,n*4);check(scores,lengths,output)
            signatures.append(hashlib.sha256(saved.cpu().numpy().tobytes()).hexdigest())
        case={'rows':rows,'columns':n,'row_stride':scores.stride(0),'family':family,'graph_replays':3,'input_and_length_changes':True,'poison_overwritten':True,'selected_values_exact':True,'signatures':signatures}
        report['cases'].append(case);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(case),flush=True)
        del graph
report['pass']=len(report['cases'])==len(shapes)*3
report['finished_unix']=time.time()
a.output.write_text(json.dumps(report,indent=2)+'\n')
print('TOPK_REPAIR_GPU_ORACLE_PASS',flush=True)
