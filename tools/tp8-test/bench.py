#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP bring-up probes. Client wall timings are not engine prefill timings or promotion evidence."""
import concurrent.futures
import json
import re
import statistics
import time
import urllib.request

MODEL='RedHatAI/GLM-5.3-Flash-NVFP4'


def completion(endpoint, prompt, max_tokens=256):
    body=dict(model=MODEL,messages=[{'role':'user','content':prompt}],max_tokens=max_tokens,temperature=0,
              stream=True,stream_options={'include_usage':True},chat_template_kwargs={'enable_thinking':False})
    req=urllib.request.Request(endpoint.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),
                               {'Content-Type':'application/json'})
    begin=time.perf_counter();first=None;text=[];usage=None;finish=None
    with urllib.request.urlopen(req,timeout=180) as response:
        for raw in response:
            line=raw.decode().strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':break
            event=json.loads(data)
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{});part=delta.get('content') or ''
                if part or delta.get('reasoning_content') or delta.get('reasoning'):
                    first=first or time.perf_counter()
                if part:
                    text.append(part)
                finish=choice.get('finish_reason') or finish
    end=time.perf_counter()
    if first is None or finish!='stop' or not usage or usage.get('completion_tokens',0)<=0:
        raise ValueError('completion, stop reason or actual usage missing')
    return dict(text=''.join(text),finish_reason=finish,ttft_ms=(first-begin)*1000,elapsed_ms=(end-begin)*1000,
        completion_tokens=usage['completion_tokens'],prompt_tokens=usage['prompt_tokens'],
        cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0),
        decode_tps=(usage['completion_tokens']-1)/max(end-first,1e-9),
        timing_scope='client stream wall time including transport',usage=usage)


def spec_counts(endpoint):
    with urllib.request.urlopen(endpoint.rstrip('/')+'/metrics',timeout=10) as response: text=response.read().decode()
    totals={}
    for line in text.splitlines():
        m=re.match(r'(vllm:spec_decode_num_(?:draft|accepted)_tokens_total)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$',line)
        if m:totals[m[1]]=totals.get(m[1],0.)+float(m[2])
    return totals


def run(endpoint,stage,nonce,result=None):
    if result is None:result={}
    result.update(ok=False,completion_gate=False,speculation_gate=False,metrics=[],probes=[],stage=stage)
    before=spec_counts(endpoint) if stage!='target' else {}
    prompts=[('arithmetic','Reply with only the decimal integer: 17 + 25.',lambda s:s.strip()=='42'),
             ('korean','다음 두 숫자를 더하세요. 숫자만 답하세요: 17 + 25.',lambda s:s.strip()=='42'),
             ('json','Return only a JSON object with key answer and integer value 42, without markdown.',
              lambda s:json.loads(s)=={'answer':42}),
             ('sequence','Return only a JSON array containing the integers from 1 through 32 in order, without markdown.',
              lambda s:json.loads(s)==list(range(1,33)))]
    for name,prompt,gate in prompts:
        row=completion(endpoint,prompt)
        try:passed=bool(gate(row['text']))
        except (ValueError,TypeError):passed=False
        result['probes'].append(dict(name=name,passed=passed,**row))
        if not passed:raise ValueError('quality probe failed: '+name)
    # Cold prefixes differ by run, while exact replay pairs keep identical prompts.
    for context in (512,2048,4096):
        prompt=('Record '+nonce+' size '+str(context)+'.\n'+('The marker is 42.\n'*(context//6))+
                '\nReply with only the marker number.')
        for repetition in range(2):
            row=completion(endpoint,prompt,256)
            if row['text'].strip()!='42':raise ValueError('long prompt/replay quality failed')
            result['metrics'].append(dict(case='cold' if repetition==0 else 'replay',
                                           context=context,repetition=repetition,concurrency=1,**row))
    if stage=='graphs':
        for _ in range(3):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                rows=list(pool.map(lambda _:completion(endpoint,prompts[0][1]),range(4)))
            if any(r['text'].strip()!='42' for r in rows):raise ValueError('concurrent graph completion failed')
            result['metrics'] += [dict(case='concurrent',concurrency=4,**row) for row in rows]
        # Both arrival directions, with the long request starting before/after a short one.
        long_prompt='Record '+nonce+' stagger.\n'+'Marker 42.\n'*512+'Reply only with the marker number.'
        for label,first,second in (('long-first',long_prompt,prompts[0][1]),('short-first',prompts[0][1],long_prompt)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a=pool.submit(completion,endpoint,first);time.sleep(.25);b=pool.submit(completion,endpoint,second)
                rows=[a.result(),b.result()]
            if any(r['text'].strip()!='42' for r in rows):raise ValueError('staggered completion failed')
            result['metrics'] += [dict(case=label,concurrency=2,**row) for row in rows]
    if stage!='target':
        after=spec_counts(endpoint)
        draft=after.get('vllm:spec_decode_num_draft_tokens_total',0)-before.get('vllm:spec_decode_num_draft_tokens_total',0)
        accepted=after.get('vllm:spec_decode_num_accepted_tokens_total',0)-before.get('vllm:spec_decode_num_accepted_tokens_total',0)
        result['speculation_counts']=dict(draft=draft,accepted=accepted,rejected=draft-accepted)
        # This only demonstrates executed accept/reject paths and task-level replay; it is not a KDA byte oracle.
        result['speculation_gate']=draft>accepted>0
        if not result['speculation_gate']:raise ValueError('both speculative accept and reject paths must execute')
    result.update(ok=True,completion_gate=True)
    return result
