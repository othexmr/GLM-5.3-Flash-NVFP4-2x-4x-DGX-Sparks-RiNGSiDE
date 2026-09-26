"""CPU contracts for actual phase metadata, cached plans and graph fallback."""
import ast,sys,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
D=Path(__file__).parent/'source'
def extract(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names:nodes.append(n)
  if isinstance(n,ast.ClassDef):nodes.extend(x for x in n.body if isinstance(x,ast.FunctionDef) and x.name in names)
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
class PolicyTests(unittest.TestCase):
 def setUp(self):
  self.ctx=NS(cudagraph_runtime_mode=0,attn_metadata={'a':NS(glm53_moe_all_prefill=True),'b':NS(glm53_moe_all_prefill=True)})
  self.capturing=False;self.available=True
  sys.modules['vllm.config']=NS(CUDAGraphMode=NS(NONE=0))
  self.ns={'logger':NS(info=lambda *a:None),'_GLM53_PREFILL_POLICY_LOGGED':set(),'_GLM53_ATOMIC_PREFILL':True,'is_forward_context_available':lambda:self.available,'get_forward_context':lambda:self.ctx,'_is_current_stream_capturing':lambda:self.capturing,'Any':object,'MoEActivation':object,'CommonAttentionMetadata':object,'torch':torch}
  extract(D/'vllm/model_executor/layers/fused_moe/b12x.py',{'_glm53_use_atomic_prefill','_plan'},self.ns)
  extract(D/'glm53_sparse_mla/backend.py',{'_glm53_all_requests_prefilling'},self.ns)
 def use(self,t=8192,q='nvfp4'):return self.ns['_glm53_use_atomic_prefill'](t,q)
 def test_pure_prefill_threshold(self):
  self.assertTrue(self.use());self.assertTrue(self.use(128));self.assertFalse(self.use(127))
 def test_decode_and_mixed(self):
  for a,b in [(False,False),(True,False),(False,True)]:
   self.ctx.attn_metadata={'a':NS(glm53_moe_all_prefill=a),'b':NS(glm53_moe_all_prefill=b)};self.assertFalse(self.use())
 def test_unknown_metadata(self):
  for m in [None,{},[],[{'a':NS(glm53_moe_all_prefill=True)}],{'a':None},{'a':NS(glm53_moe_all_prefill=1)}]:
   self.ctx.attn_metadata=m;self.assertFalse(self.use())
 def test_target_phase_with_dense_drafter_metadata(self):
  self.ctx.attn_metadata={'target':NS(glm53_moe_all_prefill=True),'draft':NS(num_prefills=0)}
  self.assertTrue(self.use())
  self.ctx.attn_metadata['target'].glm53_moe_all_prefill=False
  self.assertFalse(self.use())
  self.ctx.attn_metadata={'draft':NS(num_prefills=1)}
  self.assertFalse(self.use())
 def test_capture_and_replay_policy(self):
  self.capturing=True;self.assertFalse(self.use());self.capturing=False
  for mode in [1,2,3]:self.ctx.cudagraph_runtime_mode=mode;self.assertFalse(self.use())
 def test_opt_in_and_quant_scope(self):
  for q in ['w4a16','w4a8_nvfp4','mxfp4']:self.assertFalse(self.use(q=q))
  self.available=False;self.assertFalse(self.use());self.available=True
  self.ns['_GLM53_ATOMIC_PREFILL']=False;self.assertFalse(self.use())
 def test_real_cpu_scheduler_flags(self):
  f=self.ns['_glm53_all_requests_prefilling']
  for flags,want in [([True],True),([True,True],True),([True,False],False),([False],False),([],False)]:
   self.assertEqual(f(NS(is_prefilling=torch.tensor(flags,dtype=torch.bool),num_reqs=len(flags))),want)
  self.assertFalse(f(NS(is_prefilling=None,num_reqs=1)))
  self.assertFalse(f(NS(is_prefilling=torch.tensor([1]),num_reqs=1)))
  self.assertFalse(f(NS(is_prefilling=torch.tensor([[True]]),num_reqs=1)))
  self.assertFalse(f(NS(is_prefilling=torch.tensor([True]),num_reqs=2)))
  # Guard must short-circuit before any device read/synchronization.
  self.assertFalse(f(NS(is_prefilling=NS(device=NS(type='cuda')),num_reqs=1)))
 def test_installed_plan_api_contract(self):
  import sysconfig
  root=Path(sysconfig.get_paths()['purelib'])/'b12x/moe/fused_moe'
  tree=ast.parse((root/'_impl.py').read_text())
  scratch=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TPMoEScratchPlan')
  fields={n.target.id for n in scratch.body if isinstance(n,ast.AnnAssign)}
  self.assertIn('launch_plan',fields)
  launch=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TPMoEPlan')
  self.assertIn('deterministic_output',{n.name for n in launch.body if isinstance(n,ast.FunctionDef)})
  self.assertIn('Plan = TPMoEScratchPlan',(root/'_vllm_compat.py').read_text())
 def test_plan_cache_same_shape_phase_switch(self):
  calls=[]
  def plan(caps):
   calls.append(caps);return NS(launch_plan=NS(deterministic_output=caps.deterministic_output is not False))
  fused=NS(Caps=lambda **kw:NS(**kw),plan=plan)
  self.ns.update(_require_b12x_fused_moe=lambda:fused,_GLM53_PREFILL_POLICY_LOGGED=set(),logger=NS(info=lambda *a:None))
  obj=NS(_quant_mode='nvfp4',_plans={},_swiglu_params=lambda a:(10.,None,None),_prepared=lambda:NS(w1_fp4=NS(device='cuda'),plan='weights'))
  run=lambda:self.ns['_plan'](obj,tokens=8192,topk=8,activation='silu')
  self.ctx.attn_metadata=None;baseline=run();self.assertTrue(baseline.launch_plan.deterministic_output)
  self.ctx.attn_metadata={'a':NS(glm53_moe_all_prefill=True)};atomic=run();self.assertFalse(atomic.launch_plan.deterministic_output)
  self.assertIs(run(),atomic);self.assertIsNot(atomic,baseline)
  self.capturing=True;self.assertIs(run(),baseline)
  self.assertEqual(len(calls),2);self.assertEqual(len(obj._plans),2)
if __name__=='__main__':unittest.main(verbosity=2)
