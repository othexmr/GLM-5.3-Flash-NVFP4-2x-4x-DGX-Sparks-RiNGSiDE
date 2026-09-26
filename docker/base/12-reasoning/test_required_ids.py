"""CPU-only regression on the installed composite parser; no model/API request.

Run in the repaired image with VLLM_ENFORCE_STRICT_TOOL_CALLING=0. Covers the
Kimi deterministic-ID path and ordinary GLM random IDs, with actual registry
adapters and synthetic tokenization. Includes history across request boundaries.
"""
import json
import os
from types import SimpleNamespace

# Must precede vLLM imports: ToolParser.__init_subclass__ consumes this flag.
os.environ['VLLM_ENFORCE_STRICT_TOOL_CALLING'] = '0'
from vllm.parser import ParserManager
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

class Tokenizer:
    all_special_tokens = []
    all_special_ids = []
    def get_vocab(self):
        return {chr(i): i for i in range(256)}
    def encode(self, text, **kwargs):
        return list(map(ord, text))
    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))

tool = {'type': 'function', 'function': {
    'name': 'inspect', 'parameters': {'type': 'object', 'properties': {
        'value': {'type': 'integer'}}, 'required': ['value']}}}
objects = [{'name': 'inspect', 'parameters': {'value': i}} for i in range(3)]
raw = json.dumps(objects, separators=(',', ':'))

# Dedicated entrypoint tests the changed generic branch, with no reasoning to
# obscure the history-counter invariant. The GLM reasoner is tested separately.
kimi_cls = ParserManager.get_parser(tool_parser_name='kimi_k2', enable_auto_tools=True)
glm_cls = ParserManager.get_parser(tool_parser_name='glm53', enable_auto_tools=True)
kimi_cfg = SimpleNamespace(hf_overrides={},
    hf_config=SimpleNamespace(model_type='kimi_k2'), hf_text_config=None)


def run(cls, model_config, chunks, history):
    request = ChatCompletionRequest(model='fixture', messages=history + [
        {'role': 'user', 'content': 'Inspect.'}], tools=[tool],
        tool_choice='required', stream=True)
    parser = cls(Tokenizer(), tools=request.tools, model_config=model_config)
    assert parser.tool_parser.supports_required_and_named
    calls = []
    for i, chunk in enumerate(chunks):
        delta = parser.parse_delta(chunk, [], request, prompt_token_ids=[],
            finished=i == len(chunks) - 1)
        if delta:
            assert not delta.content and not delta.reasoning
            calls.extend(delta.tool_calls or [])
    assert [c.index for c in calls] == list(range(len(calls))), calls
    assert all(c.id and c.type == 'function' and c.function.name == 'inspect'
               for c in calls), calls
    assert len({c.id for c in calls}) == len(calls), calls
    return parser, calls


def history_for(calls):
    return [{'role': 'assistant', 'tool_calls': [
        {'id': c.id, 'type': 'function', 'function': {
            'name': c.function.name, 'arguments': c.function.arguments}}
        for c in calls]}] + [{'role': 'tool', 'tool_call_id': c.id, 'content': 'ok'}
                            for c in calls]

passed = 0
histories = [[], [{'role': 'assistant', 'tool_calls': [
    {'id': f'functions.inspect:{i}', 'type': 'function', 'function': {
        'name': 'inspect', 'arguments': '{}'}} for i in range(5)]}]]
# Full compound emission, every two-way split, and one-character deltas exercise
# 3-at-once, 2+1, 1+2, and 1+1+1 allocations without assuming chunk boundaries.
layouts = [[raw]] + [[raw[:i], raw[i:]] for i in range(len(raw) + 1)] + [list(raw)]
for history in histories:
    offset = len(history[0]['tool_calls']) if history else 0
    for chunks in layouts:
        parser, calls = run(kimi_cls, kimi_cfg, chunks, history)
        assert len(calls) == len(objects), calls
        assert [json.loads(c.function.arguments) for c in calls] == [
            o['parameters'] for o in objects]
        assert [c.id for c in calls] == [f'functions.inspect:{offset+i}'
                                       for i in range(len(objects))], calls
        assert parser._stream_state.history_tool_call_cnt == offset + len(objects)
        next_history = history + history_for(calls)
        following, next_calls = run(kimi_cls, kimi_cfg,
            ['[{"name":"inspect","parameters":{"value":99}}]'], next_history)
        assert next_calls[0].id == f'functions.inspect:{offset+len(objects)}'
        assert not {c.id for c in calls}.intersection(c.id for c in next_calls)
        assert following._stream_state.history_tool_call_cnt == offset + len(objects) + 1
        passed += 1
for chunks in layouts:
    parser, calls = run(glm_cls, None, chunks, [])
    assert len(calls) == len(objects)
    assert all(c.id.startswith('chatcmpl-tool-') for c in calls)
    assert parser._stream_state.history_tool_call_cnt == len(objects)
    passed += 1
print(json.dumps({'passed': passed, 'deterministic_ids_across_turns': True,
    'compound_emission_counter': True, 'random_glm_ids': True,
    'scope': 'actual installed parser; synthetic tokenizer; CPU only'}))
