"""Bitwise XQA invariance under co-scheduling, padding and page-table width.
Standalone GPU test, owned model pair must be stopped. Baseline divergence is
reported, not confused with a numerical reference failure. Same Q/K/V in row0.
"""
import argparse,json,os,time
from pathlib import Path
import torch
from flashinfer.decode import xqa_batch_decode_with_kv_cache as xqa
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--require-exact',action='store_true');ap.add_argument('--quick',action='store_true');a=ap.parse_args()
torch.manual_seed(92026);dev='cuda';dtype=torch.bfloat16;page=64;qh=16;kh=4;dim=128
maxlen=131136;pages=maxlen//page
k=torch.randn(pages,kh,page,dim,device=dev,dtype=dtype);v=torch.randn_like(k)
work=torch.zeros(128*1024*1024,device=dev,dtype=torch.uint8)
rows=[];doc={'device':torch.cuda.get_device_name(),'torch':torch.__version__,'XQA_NB_SUB_SEQ':os.getenv('XQA_NB_SUB_SEQ'),'rows':rows}
def save():Path(a.output).write_text(json.dumps(doc,indent=2)+'\n')
lengths=(4097,)if a.quick else(513,4097,32769,131073)
for qlen in ((5,)if a.quick else(1,5,8)):
 for window in ((2047,)if a.quick else(2047,-1)):
  qbase=torch.randn(qlen,qh,dim,device=dev,dtype=dtype)
  for length in lengths:
   base_out=None;reference=None
   if window>=0:
    begin=max(0,length-qlen+1-(window+1))
    kk=k.permute(0,2,1,3).reshape(-1,kh,dim)[begin:length].float().repeat_interleave(qh//kh,dim=1)
    vv=v.permute(0,2,1,3).reshape(-1,kh,dim)[begin:length].float().repeat_interleave(qh//kh,dim=1)
    scores=torch.einsum('qhd,khd->qhk',qbase.float(),kk)*(dim**-0.5)
    positions=torch.arange(begin,length,device=dev)
    row_beg=torch.arange(qlen,device=dev)+length-qlen+1-(window+1)
    scores.masked_fill_(positions[None,None,:]<row_beg[:,None,None],float('-inf'))
    reference=torch.einsum('qhk,khd->qhd',scores.softmax(-1),vv).cpu()
   variants=[('solo',1,length,False),('two_same',2,length,False),('three_long',3,maxlen,False),('four_short',4,max(qlen,513),False),('six_long',6,maxlen,False),('six_padding',6,0,True),('solo_wide_table',1,maxlen,False)]
   for name,batch,other,padding in variants:
    q=qbase.repeat(1 if padding else batch,1,1).contiguous();width=(max(length,other)+page-1)//page
    bt=torch.arange(width,device=dev,dtype=torch.int32).repeat(batch,1).contiguous()
    seq=torch.tensor([length]+[other]*(batch-1),device=dev,dtype=torch.int32)
    out=torch.empty_like(q)
    # All draft rows see every draft token (DFlash's bidirectional block).
    mask=torch.full((q.shape[0],1),(1<<qlen)-1,device=dev,dtype=torch.uint32).view(torch.uint16)if qlen>1 else None
    cu=torch.tensor([0]+[qlen]*batch,device=dev,dtype=torch.int32)if padding and qlen>1 else None
    if padding and qlen==1:continue
    def run():
     return xqa(q,(k,v),work,bt,seq,width*page,bmm1_scale=dim**-0.5,window_left=window,out=out,kv_layout='HND',q_len_per_req=qlen,mask=mask,q_cu_seq_lens=cu)
    for _ in range(3):run()
    torch.cuda.synchronize();initial=out[:qlen].clone()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    repeated=True
    for _ in range(6):
     out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
     repeated=repeated and torch.equal(initial.view(torch.int16),out[:qlen].view(torch.int16))
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(30):graph.replay()
    end.record();end.synchronize();micros=start.elapsed_time(end)*1000/30
    actual=initial.cpu()
    if base_out is None:base_out=actual.clone()
    bitdiff=int((actual.view(torch.int16)!=base_out.view(torch.int16)).sum())
    row={'q_len':qlen,'window_left':window,'seq_len':length,'variant':name,'batch':batch,'table_width':width,'companion_length':other,'finite':bool(torch.isfinite(actual).all()),'repeat_exact':repeated,'same_request_bitwise_exact':bitdiff==0,'different_elements':bitdiff,'max_abs_difference':float((actual.float()-base_out.float()).abs().max()),'graph_us':micros}
    if reference is not None:
     error=(actual.float()-reference).abs();row['fp32_reference_max_abs_error']=float(error.max());row['fp32_reference_pass']=bool((error<=0.003+0.025*reference.abs()).all())
    rows.append(row);save();print(json.dumps(row),flush=True);del graph

doc['active_window_pass']=all(r['finite']and r['repeat_exact']and r['same_request_bitwise_exact']and r.get('fp32_reference_pass',True)for r in rows if r['window_left']==2047)
doc['pass']=doc['active_window_pass'];save()
print('XQA_BATCH_EXACT',doc['pass'],flush=True)
if a.require_exact and not doc['pass']:raise SystemExit(1)
