"""Exercise the actual installed vLLM parser/registry; no GPU/model request."""
import json
from vllm.parser.glm53_moe import Glm53MoeParser
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.parser import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
SPECIAL=['<think>','</think>','<tool_call>','</tool_call>','<arg_key>','</arg_key>','<arg_value>','</arg_value>']
class Tokenizer:
 all_special_tokens=SPECIAL
 all_special_ids=list(range(1000,1000+len(SPECIAL)))
 def get_vocab(self):return {**{chr(i):i for i in range(256)},**dict(zip(self.all_special_tokens,self.all_special_ids))}
 def encode(self,text,**kwargs):
  result=[]
  while text:
   special=next((s for s in SPECIAL if text.startswith(s)),None)
   if special:result.append(self.get_vocab()[special]);text=text[len(special):]
   else:result.append(ord(text[0]));text=text[1:]
  return result
 def decode(self,ids,**kwargs):return ''.join(dict(zip(self.all_special_ids,SPECIAL)).get(i,chr(i))for i in ids)
tok=Tokenizer();req=ChatCompletionRequest(model='glm53',messages=[{'role':'user','content':'Test'}],stream=True)
# Thinking-ON combinations. The base class ORs the two flags and defaults on,
# which is exactly what the gated template does, so these must extract.
flags=[{}, {'thinking':True,'enable_thinking':False},{'thinking':False,'enable_thinking':True}]
count=0
for kw in flags:
 original=kw.copy();parser=Glm53MoeParser(tok,chat_template_kwargs=kw);assert kw==original and parser.thinking_enabled
 for raw,reason,content in [('draft</think>answer','draft','answer'),('<think>draft</think>answer','draft','answer'),('unfinished', 'unfinished',None)]:
  assert parser.extract_reasoning(raw,req)==(reason,content)
  for split in range(len(raw)+1):
   p=Glm53MoeParser(tok,chat_template_kwargs=kw);ds=[]
   for idx,chunk in enumerate((raw[:split],raw[split:])):
    d=p.parse_delta(chunk,[],req,prompt_token_ids=tok.encode('<think>'),finished=idx==1)
    if d:ds.append(d)
   assert ''.join(d.content or ''for d in ds)==(content or ''),(kw,split,ds)
   assert ''.join(d.reasoning or ''for d in ds)==(reason or ''),(kw,split,ds)
   count+=1
 p=Glm53MoeParser(tok,chat_template_kwargs=kw)
 assert not p.is_reasoning_end(tok.encode('<think>draft'))
 assert p.is_reasoning_end(tok.encode('<think>draft</think>answer'))
 assert p.extract_content_ids(tok.encode('draft</think>answer'))==tok.encode('answer')
 # Both managers use the same engine, including the worker-side reasoning adapter without model_config.
 r=ReasoningParserManager.get_reasoning_parser('glm53')(tok,chat_template_kwargs=kw)
 assert r.extract_reasoning('draft</think>answer',req)==('draft','answer')
 ToolParserManager.get_tool_parser('glm53')(tok,chat_template_kwargs=kw)
 combined=ParserManager.get_parser(tool_parser_name='glm53',reasoning_parser_name='glm53',enable_auto_tools=True)(tok,chat_template_kwargs=kw)
 assert combined.reasoning_parser is not None
assert Glm47MoeParser(tok,chat_template_kwargs={'enable_thinking':False}).extract_reasoning('direct answer',req)==(None,'direct answer')
# Thinking-OFF is now honoured rather than suppressed: no reasoning is extracted
# and every token is content. Pairs with the gated template's <think></think>;
# the always-open template must not be served by this parser.
off_flags=[{'enable_thinking':False},{'thinking':False},{'thinking':False,'enable_thinking':False}]
off_count=0
for kw in off_flags:
 original=kw.copy();p=Glm53MoeParser(tok,chat_template_kwargs=kw)
 assert kw==original,(kw,original)
 assert not p.thinking_enabled,kw
 for raw in ('draft</think>answer','plain answer','unfinished'):
  assert p.extract_reasoning(raw,req)==(None,raw),(kw,raw,p.extract_reasoning(raw,req))
  for split in range(len(raw)+1):
   stream_parser=Glm53MoeParser(tok,chat_template_kwargs=kw);deltas=[]
   for idx,chunk in enumerate((raw[:split],raw[split:])):
    delta=stream_parser.parse_delta(chunk,[],req,prompt_token_ids=tok.encode('<think></think>'),finished=idx==1)
    if delta:deltas.append(delta)
   # The GLM-4.7 base class consumes a stray </think> in the stream even with
   # thinking off (verified identical on Glm47MoeParser), and the gated template
   # never opens one, so the token is out of contract here. Assert that no text
   # is lost apart from that control token, and that nothing is called reasoning.
   assert ''.join(d.content or ''for d in deltas)==raw.replace('</think>',''),(kw,raw,split,deltas)
   assert not any(d.reasoning for d in deltas),(kw,raw,split,deltas)
   off_count+=1
 r=ReasoningParserManager.get_reasoning_parser('glm53')(tok,chat_template_kwargs=kw)
 assert r.extract_reasoning('draft</think>answer',req)==(None,'draft</think>answer'),kw
print(json.dumps({'pass':True,'thinking_on_stream_splits_checked':count,'thinking_off_stream_splits_checked':off_count,'flag_combinations':len(flags)+len(off_flags),'thinking_off_honoured':True,'registry_and_worker_reasoner':True,'legacy_glm47_off_unchanged':True,'scope':'Real installed vLLM parser with synthetic tokenizer; real tokenizer/live tools/grammar remain separate'}))
# Tool transitions and suppressing visible reasoning share the real installed engine.
tool={'type':'function','function':{'name':'multiply','description':'Multiply integers','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}
req_tool=ChatCompletionRequest(model='glm53',messages=[{'role':'user','content':'Multiply 3 and 7.'}],tools=[tool],tool_choice='auto',stream=True)
raw='Use a tool.</think><tool_call>multiply<arg_key>a</arg_key><arg_value>3</arg_value><arg_key>b</arg_key><arg_value>7</arg_value></tool_call>'
for split in range(len(raw)+1):
 p=Glm53MoeParser(tok,tools=req_tool.tools,chat_template_kwargs={});ds=[]
 for idx,chunk in enumerate((raw[:split],raw[split:])):
  d=p.parse_delta(chunk,[],req_tool,prompt_token_ids=tok.encode('<think>'),finished=idx==1)
  if d:ds.append(d)
 assert not ''.join(d.content or ''for d in ds),(split,ds)
 calls=[t for d in ds for t in(d.tool_calls or [])]
 names=''.join(t.function.name or ''for t in calls if t.function)
 args=''.join(t.function.arguments or ''for t in calls if t.function)
 assert names=='multiply'and json.loads(args)=={'a':3,'b':7},(split,names,args)
# Thinking-off on the tool path: the scratchpad text becomes content (no reasoning
# is claimed) and the tool call is still extracted intact.
for split in range(len(raw)+1):
 p=Glm53MoeParser(tok,tools=req_tool.tools,chat_template_kwargs={'enable_thinking':False});ds=[]
 for idx,chunk in enumerate((raw[:split],raw[split:])):
  d=p.parse_delta(chunk,[],req_tool,prompt_token_ids=tok.encode('<think></think>'),finished=idx==1)
  if d:ds.append(d)
 assert not ''.join(d.reasoning or ''for d in ds),(split,ds)
 # Same base-class rule as above: the stray </think> is consumed even with
 # thinking off, so the visible scratchpad text survives without it.
 assert ''.join(d.content or ''for d in ds)=='Use a tool.',(split,ds)
 calls=[t2 for d in ds for t2 in(d.tool_calls or [])]
 names=''.join(c.function.name or ''for c in calls if c.function)
 args=''.join(c.function.arguments or ''for c in calls if c.function)
 assert names=='multiply'and json.loads(args)=={'a':3,'b':7},(split,names,args)
req_hidden=ChatCompletionRequest(model='glm53',messages=[{'role':'user','content':'Test'}],stream=True,include_reasoning=False)
p=Glm53MoeParser(tok,chat_template_kwargs={});ds=[]
for i,chunk in enumerate(('private','</think>','public')):
 d=p.parse_delta(chunk,tok.encode(chunk),req_hidden,prompt_token_ids=tok.encode('<think>'),finished=i==2)
 if d:ds.append(d)
assert ''.join(d.content or ''for d in ds)=='public'and not any(d.reasoning for d in ds)
print(json.dumps({'tools_pass':True,'thinking_on_tool_stream_splits_checked':len(raw)+1,'thinking_off_tool_stream_splits_checked':len(raw)+1,'include_reasoning_false_hides_scratchpad':True,'special_token_id_stream':True}))
