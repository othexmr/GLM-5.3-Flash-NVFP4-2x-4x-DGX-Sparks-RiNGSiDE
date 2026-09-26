"""CPU-only installed-parser/Responses event regression. No model/API requests.

Covers existing-call tails, post-tool content, new calls in a compound delta,
GLM53 auto/XML production of that shape, and rejection of nameless orphans.
"""
import json
from types import SimpleNamespace
from pydantic import ValidationError
from vllm.entrypoints.generate.base.protocol import DeltaFunctionCall,DeltaMessage,DeltaToolCall
from vllm.entrypoints.openai.responses.streaming_events import SimpleStreamingEventProcessor,split_delta
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager

SPECIAL=['<think>','</think>','<tool_call>','</tool_call>','<arg_key>','</arg_key>','<arg_value>','</arg_value>']
class Tokenizer:
    all_special_tokens=SPECIAL
    all_special_ids=list(range(1000,1000+len(SPECIAL)))
    def get_vocab(self):
        return {**{chr(i):i for i in range(256)},**dict(zip(self.all_special_tokens,self.all_special_ids))}
    def encode(self,text,**kwargs):
        ids=[]
        while text:
            special=next((s for s in SPECIAL if text.startswith(s)),None)
            if special:ids.append(self.get_vocab()[special]);text=text[len(special):]
            else:ids.append(ord(text[0]));text=text[1:]
        return ids
    def decode(self,ids,**kwargs):
        mapping=dict(zip(self.all_special_ids,SPECIAL))
        return ''.join(mapping.get(i,chr(i)) for i in ids)


def tc(index,name=None,args=''):
    return DeltaToolCall(index=index,id=f'fixture-{index}'if name else None,
        type='function'if name else None,function=DeltaFunctionCall(name=name,arguments=args))


def consume(deltas):
    processor=SimpleStreamingEventProcessor()
    events=[]
    for delta in deltas:
        if delta is None:continue
        for dm in split_delta(delta):
            state,call=processor.resolve_target_state(dm)
            if processor.needs_transition(state,call):
                events.extend(processor.close_current())
                events.extend(processor.open(state,call))
            events.extend(processor.emit_delta(dm,SimpleNamespace()))
    events.extend(processor.close_current())
    return events


def completed_calls(events):
    return [e.item for e in events if e.type=='response.output_item.done'
            and e.item.type=='function_call']

first=DeltaMessage(tool_calls=[tc(0,'inspect','{"x":"fir')])
compound=DeltaMessage(content='Done.',tool_calls=[tc(0,args='st"}'),tc(1,'inspect','{"x":"second"}')])
parts=split_delta(compound)
assert len(parts)==3 and parts[0].tool_calls[0].index==0
assert parts[1].content=='Done.' and parts[2].tool_calls[0].index==1
calls=completed_calls(consume([first,compound]))
assert [c.name for c in calls]==['inspect','inspect']
assert [json.loads(c.arguments)for c in calls]==[{'x':'first'},{'x':'second'}]
assert len({c.id for c in calls})==2 and len({c.call_id for c in calls})==2

# Existing-call tail + reasoning + content + new named call preserves all fields.
with_reasoning=DeltaMessage(reasoning='Next.',content='Done.',tool_calls=[tc(0,args='st"}'),tc(1,'inspect','{"x":"second"}')])
parts=split_delta(with_reasoning)
assert len(parts)==4 and parts[0].tool_calls[0].index==0
assert parts[1].reasoning=='Next.' and parts[2].content=='Done.'
assert parts[3].tool_calls[0].index==1
assert len(completed_calls(consume([first,with_reasoning])))==2

# A brand-new named call still follows preceding content/reasoning.
new=DeltaMessage(reasoning='First.',content='Before.',tool_calls=[tc(0,'inspect','{}')])
parts=split_delta(new)
assert [bool(d.tool_calls) for d in parts]==[False,False,True]

# Nameless orphan calls must never become a fabricated 'unknown' tool or borrow
# the name of a previously closed call.
for prefix in ([],[DeltaMessage(tool_calls=[tc(0,'inspect','{}')]),DeltaMessage(content='Closed.')]):
    try:consume(prefix+[DeltaMessage(tool_calls=[tc(7,args='{}')])])
    except (ValueError,ValidationError):pass
    else:raise AssertionError('Orphan nameless tool was accepted')

# Exercise the actual GLM53 composite parser, whose auto/XML branch is inherited
# from GLM47. Its second chunk contains the active tail and a new named call.
tool={'type':'function','function':{'name':'inspect','parameters':{'type':'object','properties':{'x':{'type':'string'}},'required':['x']}}}
req=ChatCompletionRequest(model='glm53',messages=[{'role':'user','content':'Inspect twice.'}],tools=[tool],tool_choice='auto',stream=True)
cls=ParserManager.get_parser(tool_parser_name='glm53',reasoning_parser_name='glm53',enable_auto_tools=True)
tok=Tokenizer()
for use_ids in (False,True):
    parser=cls(tok,tools=req.tools,chat_template_kwargs={'enable_thinking':False})
    chunks=['Working.</think><tool_call>inspect<arg_key>x</arg_key><arg_value>fir',
            'st</arg_value></tool_call>Done.<tool_call>inspect<arg_key>x</arg_key><arg_value>second</arg_value></tool_call>']
    deltas=[parser.parse_delta(chunk,tok.encode(chunk)if use_ids else[],req,
        prompt_token_ids=tok.encode('<think>'),finished=i==len(chunks)-1)
        for i,chunk in enumerate(chunks)]
    assert deltas[1] and deltas[1].content=='Done.'
    assert len(deltas[1].tool_calls)==2 and deltas[1].tool_calls[0].function.name is None
    events=consume(deltas);calls=completed_calls(events)
    assert [json.loads(c.arguments)for c in calls]==[{'x':'first'},{'x':'second'}]
    text=''.join(e.delta for e in events if e.type=='response.output_text.delta')
    assert text=='Done.',text
print(json.dumps({'pass':True,'compound_tail_content_new_call':True,
    'reasoning_order':True,'orphan_rejected':True,'glm53_auto_text_and_token_ids':True,
    'scope':'actual installed parser/event processor; synthetic tokenizer; CPU only'}))
