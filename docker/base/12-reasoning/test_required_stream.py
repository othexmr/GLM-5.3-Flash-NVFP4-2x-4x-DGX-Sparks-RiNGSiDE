"""Real composite parser regression for required-tool JSON across chunk boundaries."""
import ast,json,os
from pathlib import Path
s=Path('/opt/glm53-reasoning-fix/test_parser.py').read_text();tree=ast.parse(s)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef))or(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SPECIAL'for t in n.targets))],type_ignores=[]),'test-fixture','exec'))

tok=Tokenizer();tool={'type':'function','function':{'name':'multiply','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}
request=ChatCompletionRequest(model='glm53',messages=[{'role':'user','content':'Multiply'}],tools=[tool],tool_choice='required',stream=True)
cls=ParserManager.get_parser(tool_parser_name='glm53',reasoning_parser_name='glm53',enable_auto_tools=True)
raw='Let me calculate.</think>[{"name":"multiply","parameters":{"a":3,"b":7}}]'
passed=0;failed=[]
for chunks in [[raw[:i],raw[i:]]for i in range(len(raw)+1)]+[list(raw)]:
 p=cls(tok,tools=request.tools,chat_template_kwargs={'enable_thinking':False});calls={};reason=[]
 try:
  for i,chunk in enumerate(chunks):
   d=p.parse_delta(chunk,[],request,prompt_token_ids=tok.encode('<think>'),finished=i==len(chunks)-1)
   if not d:continue
   reason.append(d.reasoning or'')
   assert not d.content,repr(d.content)
   for t in d.tool_calls or[]:
    dst=calls.setdefault(t.index,{'name':'','arguments':''})
    if t.function:
     dst['name']+=t.function.name or'';dst['arguments']+=t.function.arguments or''
  assert ''.join(reason)=='Let me calculate.',reason
  assert len(calls)==1,calls;c=next(iter(calls.values()));assert c['name']=='multiply',c;assert json.loads(c['arguments'])=={'a':3,'b':7},c
  passed+=1
 except Exception as e:failed.append({'split':len(chunks[0]),'chunks':len(chunks),'error':type(e).__name__+': '+str(e)})
print(json.dumps({'patch':os.getenv('PATCH_REQUIRED'),'passed':passed,'failed':failed},indent=2))
assert not failed
