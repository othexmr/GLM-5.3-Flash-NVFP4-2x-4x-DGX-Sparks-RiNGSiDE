import ast, os, unittest
from pathlib import Path
from types import SimpleNamespace as NS
D=Path(__file__).resolve().parent/'runtime-candidate'
def lift(path, cls, methods, env=None):
 tree=ast.parse((D/path).read_text()); nodes=tree.body
 if cls:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
 chosen=[n for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in methods]
 if len(chosen)!=len(methods):raise ValueError((cls,methods))
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+chosen,type_ignores=[])
 scope={'os':os};scope.update(env or {});exec(compile(ast.fix_missing_locations(mod),str(D/path),'exec'),scope);return scope
class RuntimeRepairs(unittest.TestCase):
 def test_layout_selector_uses_actual_override(self):
  scope=lift('b12x/moe/fused_moe/_impl.py',None,['_resolved_dynamic_swap_ab','_glm53_decode_fastpath_selected'],{
   '_normalize_quant_mode':lambda x:x,'_is_w4a8_quant_mode':lambda x:x=='w4a8_mx','_W6A8_QUANT_MODES':('w6a8_mx',),
   '_get_activation_kernel_spec':lambda *a,**k:NS(is_gated=True),'_GLM53_DECODE_FASTPATH':True,'_GLM53_DECODE_FASTPATH_MAX_ROWS':64,'_dynamic_work_source':lambda:'materialized_queue'})
  kw=dict(quant_mode='nvfp4',activation='silu',routed_rows=40,n=1024,direct_routing=False,deterministic_output=True,w4a8_repacked=False,selected_tile_m=16)
  old=os.environ.pop('B12X_DYNAMIC_SWAP_AB',None)
  try:
   self.assertTrue(scope['_glm53_decode_fastpath_selected'](**kw))
   os.environ['B12X_DYNAMIC_SWAP_AB']='1';self.assertFalse(scope['_glm53_decode_fastpath_selected'](**kw));self.assertTrue(scope['_resolved_dynamic_swap_ab']('nvfp4','silu',1024))
   os.environ['B12X_DYNAMIC_SWAP_AB']='0';self.assertTrue(scope['_glm53_decode_fastpath_selected'](**kw))
  finally:
   os.environ.pop('B12X_DYNAMIC_SWAP_AB',None)
   if old is not None:os.environ['B12X_DYNAMIC_SWAP_AB']=old
 def test_facade_propagates_truncation(self):
  p=lift('vllm/parser/abstract_parser.py','Parser',['tool_call_truncated'])['tool_call_truncated']
  for flag in (False,True):self.assertEqual(p.fget(NS(_tool_parser=NS(tool_call_truncated=flag))),flag)
  self.assertFalse(p.fget(NS(_tool_parser=None)))
 def test_stream_item_completion_status(self):
  state_type=NS(TOOL_CALL=1)
  fn=lift('vllm/entrypoints/openai/responses/streaming_events.py','SimpleStreamingEventProcessor',['close_current'],{'_StateType':state_type})['close_current']
  for flag in (False,True):
   ev=NS(type='response.output_item.done',item=NS(status='completed'))
   obj=NS(state=NS(current_state=1),_STATE_HANDLERS={1:NS(done_fn=lambda _: [ev])})
   out=fn(obj,tool_call_truncated=flag)
   self.assertEqual(out[0].item.status,'incomplete' if flag else 'completed')
 def test_content_logprobs_match_only_content_tokens(self):
  scope={};exec((D/'vllm/entrypoints/openai/responses/content_logprobs.py').read_text(),scope);f=scope['content_logprob_span']
  entries=[NS(token=x) for x in ['<think>','secret','</think>','Hello',' world','<tool_call>']]
  raw=''.join(e.token for e in entries)
  selected,end=f(raw,'Hello world',entries)
  self.assertEqual(selected,entries[3:5]);self.assertEqual(raw[:end],'<think>secret</think>Hello world')
  self.assertEqual(f(raw,'ello',entries)[0],[])
  duplicate=[NS(token='same'),NS(token='|'),NS(token='same')]
  self.assertEqual(f('same|same','same',duplicate)[0],[])
  self.assertEqual(f('same|same','same',duplicate,5)[0],duplicate[2:])
  # A buffered boundary can return content from an earlier engine delta.
  self.assertEqual(f(raw,'Hello',entries)[0],entries[3:4])
  self.assertEqual(f(raw,' world',entries,raw.index(' world'))[0],entries[4:5])
  self.assertEqual(f('Hello','Hello',[NS(token='token_id:123')])[0],[])
 def test_displayed_logprobs_allow_hidden_end_token_only_after_span(self):
  scope={};exec((D/'vllm/entrypoints/openai/responses/content_logprobs.py').read_text(),scope);f=scope['content_logprob_span']
  entries=[NS(token=x)for x in ['V','iolet',' silver',' amber',' ','42','.','<|user|>']]
  self.assertEqual(f('Violet silver amber 42.','Violet silver amber 42.',entries)[0],entries[:-1])
  # Equal prefix lengths alone are insufficient; hidden/mismatched prefix must not shift attribution.
  bad=[NS(token=x)for x in ['xxxx','Hello','<|user|>']]
  self.assertEqual(f('goodHello','Hello',bad)[0],[])
 def test_retention_off_preserves_qualified_prefill_chunks(self):
  f=lift('vllm/v1/core/sched/scheduler.py','Scheduler',['_mamba_block_aligned_split'])['_mamba_block_aligned_split']
  def obj(retention,enabled=True):
   return NS(block_size=4608,cache_config=NS(block_size=1152,enable_prefix_caching=enabled,prefix_cache_retention_interval=retention),use_eagle_block_drop=True,mamba_has_prefill_checkpoint_blocks=False,mamba_partial_cache_hit=True,hash_block_size=1152,max_num_scheduled_tokens=14336,scheduler_config=NS(long_prefill_token_threshold=0))
  def chunks(o):
   req=NS(num_prompt_tokens=32769,num_tokens=32769,num_computed_tokens=0,shared_prefix_boundary=0);result=[]
   while req.num_computed_tokens<req.num_prompt_tokens:
    n=f(o,req,min(14336,req.num_prompt_tokens-req.num_computed_tokens));self.assertGreater(n,0);result.append(n);req.num_computed_tokens+=n
   return result
  original=[13824,13824,3456,1152,513]
  self.assertEqual(chunks(obj(0)),original)
  self.assertEqual(chunks(obj(None,False)),original)
  for retention in(None,4608,13824):
   self.assertEqual(chunks(obj(retention)),[13824,13824,4608,513])
   req=NS(num_prompt_tokens=32769,num_tokens=32769,num_computed_tokens=0,shared_prefix_boundary=0)
   self.assertEqual(f(obj(retention),req,10368),9216)
 def test_cache_estimate_skips_noncacheable_tail(self):
  f=lift('vllm/v1/core/kv_cache_manager.py','KVCacheManager',['estimate_cached_tokens'])['estimate_cached_tokens']
  specs=[NS(kv_cache_spec=NS(prefix_cacheable=x)) for x in [True,False,True]]
  blocks=[[NS(block_hash_num_tokens=9216)],[NS(block_hash_num_tokens=0)],[NS(block_hash_num_tokens=4608)]]
  obj=NS(kv_cache_config=NS(kv_cache_groups=specs),get_blocks=lambda _:NS(blocks=blocks))
  self.assertEqual(f(obj,NS(request_id='r')),4608)
 def test_graph_buffer_survives_large_eager_prefill(self):
  import numpy as np
  class Buf:
   def __init__(self,size):self.data=np.empty(size,dtype=np.int32)
   @property
   def shape(self):return self.data.shape
   def fill_(self,n):self.data.fill(n)
   def __getitem__(self,x):
    b=Buf(0);b.data=self.data[x];return b
   def copy_(self,x,**kw):self.data[:]=x
  torch=NS(empty=lambda size,**kw:Buf(size[0]),int32=None,from_numpy=lambda x:x)
  tree=ast.parse((D/'glm53_sparse_mla/backend.py').read_text());cls=next(n.name for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='build' for f in n.body))
  scope=lift('glm53_sparse_mla/backend.py',cls,['__init__','build'],{'torch':torch,'np':np,'Glm53SparseMLAMetadata':lambda **kw:NS(**kw)})
  obj=NS();scope['__init__'](obj,NS(block_size=4608),[],NS(model_config=NS(hf_config=NS(index_topk=2)),scheduler_config=NS(max_num_batched_tokens=4),compilation_config=NS(cudagraph_capture_sizes=[8])),None)
  stable=obj.req_id_per_token_buffer
  def run(starts):
   count=starts[-1];md=NS(num_actual_tokens=count,query_start_loc_cpu=starts,num_reqs=len(starts)-1,max_query_len=max(np.diff(starts)),max_seq_len=count,query_start_loc=starts,slot_mapping=None,block_table_tensor=None)
   return scope['build'](obj,0,md)
  run([0,2,4]);run([0,9,10]);self.assertIs(stable,obj.req_id_per_token_buffer)
  out=run([0,1,4]);self.assertEqual(out.req_id_per_token.data.tolist(),[0,1,1,1]);self.assertEqual(stable.data[:4].tolist(),[0,1,1,1])
if __name__=='__main__':unittest.main()
