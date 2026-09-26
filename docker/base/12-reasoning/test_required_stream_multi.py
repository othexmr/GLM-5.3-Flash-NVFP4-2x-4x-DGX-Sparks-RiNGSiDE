"""Broader required-tool parser regression; CPU only, actual installed parser."""
import ast,json,random
from pathlib import Path
s=Path('/opt/glm53-reasoning-fix/test_parser.py').read_text();tree=ast.parse(s)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef))or(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SPECIAL'for t in n.targets))],type_ignores=[]),'test-fixture','exec'))
tok=Tokenizer();tool={'type':'function','function':{'name':'inspect','parameters':{'type':'object','properties':{'data':{}},'required':['data']}}}
request=ChatCompletionRequest(model='glm53',messages=[{'role':'user','content':'Inspect'}],tools=[tool],tool_choice='required',stream=True)
cls=ParserManager.get_parser(tool_parser_name='glm53',reasoning_parser_name='glm53',enable_auto_tools=True)
passed=0;failed=[]
for data in ({'nested':[1,{'text':'quote " slash \\ line\n ไทย'}],'null':None}, {'unicode':'日本語 😀','bool':True}):
 expected=[{'name':'inspect','parameters':{'data':data}}, {'name':'inspect','parameters':{'data':{'second':'ok'}}}]
 raw='Inspecting.</think>'+json.dumps(expected,ensure_ascii=False)
 layouts=[[raw[:i],raw[i:]] for i in range(len(raw)+1)]+[list(raw)]
 rng=random.Random(7)
 for _ in range(20):
  chunks=[];pos=0
  while pos<len(raw):n=rng.randint(1,19);chunks.append(raw[pos:pos+n]);pos+=n
  layouts.append(chunks)
 for chunks in layouts:
  p=cls(tok,tools=request.tools,chat_template_kwargs={'enable_thinking':False});calls={};reason=[]
  try:
   for i,chunk in enumerate(chunks):
    d=p.parse_delta(chunk,[],request,prompt_token_ids=tok.encode('<think>'),finished=i==len(chunks)-1)
    if not d:continue
    reason.append(d.reasoning or'');assert not d.content,repr(d.content)
    for t in d.tool_calls or[]:
     dst=calls.setdefault(t.index,{'name':'','arguments':''})
     if t.function:dst['name']+=t.function.name or'';dst['arguments']+=t.function.arguments or''
   assert ''.join(reason)=='Inspecting.';assert len(calls)==2,calls
   assert [{'name':c['name'],'parameters':json.loads(c['arguments'])}for c in calls.values()]==expected,calls
   passed+=1
  except Exception as e:failed.append({'first_chunk':chunks[0],'chunks':len(chunks),'error':str(e)})
print(json.dumps({'passed':passed,'failed':failed},ensure_ascii=False,indent=2));assert not failed
