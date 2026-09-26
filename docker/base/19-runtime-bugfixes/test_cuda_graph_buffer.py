"""Run actual metadata-builder methods with CUDA memory and CUDA-graph replay."""
import ast,json,sys,sysconfig
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
source=Path(sys.argv[1]) if len(sys.argv)>1 else Path(sysconfig.get_paths()['purelib'])/'glm53_sparse_mla/backend.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='build' for f in n.body))
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','build')]
scope={'np':np,'torch':torch,'Glm53SparseMLAMetadata':lambda **kw:NS(**kw)}
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),scope)
builder=NS();scope['__init__'](builder,NS(block_size=4608),[],NS(model_config=NS(hf_config=NS(index_topk=2)),scheduler_config=NS(max_num_batched_tokens=4),compilation_config=NS(cudagraph_capture_sizes=[8])),torch.device('cuda'))
def build(starts):
 metadata=NS(num_actual_tokens=starts[-1],query_start_loc_cpu=starts,num_reqs=len(starts)-1,max_query_len=max(np.diff(starts)),max_seq_len=starts[-1],query_start_loc=starts,slot_mapping=None,block_table_tensor=None)
 return scope['build'](builder,0,metadata)
metadata=build([0,2,4]);stable=builder.req_id_per_token_buffer;out=torch.empty(4,dtype=torch.int32,device='cuda');torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):out.copy_(metadata.req_id_per_token)
build([0,9,10]);build([0,1,4]);torch.cuda.synchronize();graph.replay();torch.cuda.synchronize()
receipt={'stable_capture_pointer':stable.data_ptr()==builder.req_id_per_token_buffer.data_ptr(),'replayed_ids':out.cpu().tolist(),'expected':[0,1,1,1]}
receipt['pass']=receipt['stable_capture_pointer'] and receipt['replayed_ids']==receipt['expected'];print(json.dumps(receipt),flush=True)
if not receipt['pass']:raise SystemExit(1)
