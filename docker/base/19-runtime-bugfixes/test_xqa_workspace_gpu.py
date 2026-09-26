import json,torch
from flashinfer.decode import xqa_batch_decode_with_kv_cache as xqa
q=torch.randn(30,16,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(8,4,64,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
bt=torch.arange(8,device='cuda',dtype=torch.int32).repeat(6,1)
seq=torch.full((6,),512,device='cuda',dtype=torch.int32)
mask=torch.full((30,1),31,device='cuda',dtype=torch.uint32).view(torch.uint16)
def run(n):
 w=torch.zeros(8*1024*1024+n,device='cuda',dtype=torch.uint8)
 out=xqa(q,(k,v),w,bt,seq,512,bmm1_scale=128**-0.5,window_left=2047,kv_layout='HND',q_len_per_req=5,mask=mask)
 torch.cuda.synchronize();return out
try:run(1622015)
except RuntimeError as e:
 assert 'workspace is too small' in str(e),str(e)
 print(json.dumps({'short_scratch_rejected':True,'error':str(e)}),flush=True)
else:raise RuntimeError('one-byte-short scratch accepted')
out=run(1622016);assert torch.isfinite(out).all()
reference=run(128*1024*1024)
assert torch.equal(out.view(torch.int16),reference.view(torch.int16))
print(json.dumps({'exact_boundary_pass':True,'same_as_large_workspace':True,'scratch_bytes':1622016}),flush=True)
