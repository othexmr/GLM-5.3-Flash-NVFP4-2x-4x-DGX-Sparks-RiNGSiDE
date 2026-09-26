# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
"""CPU regressions for the installed Responses serving and stored-history paths.

The engine and renderer are absent. Real vLLM contexts, event processing,
protocol models and serving methods consume deterministic parser deltas.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from openai.types.responses import response_text_delta_event
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall, DeltaMessage, DeltaToolCall, FunctionCall,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


def tool(index, name=None, arguments=''):
    return DeltaToolCall(
        index=index, id=f'parser-id-{index}' if name else None,
        type='function' if name else None,
        function=DeltaFunctionCall(name=name, arguments=arguments),
    )


class ScriptedParser:
    def __init__(self, deltas, *, allow_full_parse=False):
        self.deltas = iter(deltas)
        self.allow_full_parse = allow_full_parse
        self.full_parse_count = 0

    def parse_delta(self, *, request, **kwargs):
        delta = next(self.deltas)
        if delta is not None and not request.include_reasoning:
            delta = delta.model_copy(update={'reasoning': None})
        return delta

    def parse(self, *args, **kwargs):
        self.full_parse_count += 1
        if not self.allow_full_parse:
            raise AssertionError('Streaming finalization reparsed raw output')
        return None, 'Full output.', [
            FunctionCall(id='full-parser-id', name='inspect', arguments='{}')]

    def count_reasoning_tokens(self, token_ids):
        return 2


class ServingHarness(OpenAIServingResponses):
    def __init__(self, parser, stream_tokens=None):
        # Do not instantiate an engine, tokenizer, model config or renderer.
        self.use_harmony = False
        self.parser = type(parser) if parser else None
        self.enable_auto_tools = True
        self.enable_log_outputs = True
        self.request_logger = Mock()
        self.tool_server = None
        self.response_store = {}
        self.response_store_lock = asyncio.Lock()
        self.msg_store = {}
        self.chat_template = None
        self.chat_template_content_format = 'auto'
        self.chat_template_kwargs = {}
        self.online_renderer = SimpleNamespace(
            exclude_tools_when_tool_choice_none=False,
            preprocess_chat=AsyncMock(return_value=(None, ['rendered-input'])),
        )
        self._create_stream_response_logprobs = Mock(side_effect=lambda **kwargs: [
            response_text_delta_event.Logprob(
                token=(stream_tokens[kwargs['token_ids'][0] - 200]
                       if stream_tokens is not None else 'visible'), logprob=-0.25,
                top_logprobs=[{'token': 'other', 'logprob': -1.5}],
            )
        ])


def request(**kwargs):
    return ResponsesRequest(
        model='fixture', input='Inspect twice.', store=True,
        tools=[{'type': 'function', 'name': 'inspect', 'parameters': {
            'type': 'object', 'properties': {'x': {'type': 'string'}},
        }}], **kwargs,
    )


async def outputs(context, request_id, chunks, finish_reason='stop'):
    for index, chunk in enumerate(chunks):
        last = index == len(chunks) - 1
        context.append_output(RequestOutput(
            request_id=request_id, prompt='fixture prompt',
            prompt_token_ids=[101, 102, 103], prompt_logprobs=None,
            outputs=[CompletionOutput(
                index=0, text=chunk, token_ids=[index + 200],
                cumulative_logprob=0.0, logprobs=None,
                finish_reason=finish_reason if last else None,
            )],
            finished=last, num_cached_tokens=1,
        ))
        yield context


class ResponsesIdentity(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_items_are_final_stored_and_continued_without_reparse(self):
        for finish_reason in ('stop', 'length'):
            with self.subTest(finish_reason=finish_reason):
                req = request(stream=True, include_reasoning=True,
                              include=['message.output_text.logprobs'])
                parser = ScriptedParser([
                    DeltaMessage(reasoning='Reason.'),
                    DeltaMessage(content='Before.'),
                    DeltaMessage(tool_calls=[tool(0, 'inspect', '{"x":"fir')]),
                    DeltaMessage(content='Between.', tool_calls=[
                        tool(0, arguments='st"}'),
                        tool(1, 'inspect', '{"x":"second"}'),
                    ]),
                    DeltaMessage(content='After.'),
                ])
                context = SimpleContext(response_parser=parser)
                # Metadata is retained only for exact raw-text/token spans.
                # ScriptedParser supplies tool deltas independently; the raw
                # chunks and mocked logprob tokens must match visible text.
                chunks = ['Reason.', 'Before.', 'Tool.', 'Between.', 'After.']
                serving = ServingHarness(parser, stream_tokens=chunks)
                previous_input = [{'role': 'user', 'content': 'Inspect twice.'}]
                serving.msg_store[req.request_id] = previous_input
                events = [event async for event in serving.responses_stream_generator(
                    req, SamplingParams(max_tokens=20),
                    outputs(context, req.request_id, chunks, finish_reason),
                    context, 'fixture', None,
                    RequestResponseMetadata(request_id=req.request_id), created_time=123,
                )]
                self.assertEqual([e.sequence_number for e in events], list(range(len(events))))
                self.assertEqual(events[-1].type, 'response.incomplete'
                                 if finish_reason == 'length' else 'response.completed')
                final = events[-1].response
                done_events = [e for e in events if e.type == 'response.output_item.done']
                added_events = [e for e in events if e.type == 'response.output_item.added']
                self.assertEqual([e.output_index for e in done_events], list(range(6)))
                self.assertEqual([item.type for item in final.output], [
                    'reasoning', 'message', 'function_call', 'message', 'function_call', 'message'])
                self.assertEqual([e.item.model_dump() for e in done_events],
                                 [item.model_dump() for item in final.output])
                self.assertEqual([e.item.id for e in added_events],
                                 [item.id for item in final.output])
                self.assertEqual(len({item.id for item in final.output}), len(final.output))
                calls = [item for item in final.output if item.type == 'function_call']
                self.assertEqual([json.loads(c.arguments) for c in calls],
                                 [{'x': 'first'}, {'x': 'second'}])
                self.assertEqual([e.item.call_id for e in added_events
                                  if e.item.type == 'function_call'], [c.call_id for c in calls])
                self.assertEqual(len({c.call_id for c in calls}), 2)
                argument_done = [e for e in events if e.type == 'response.function_call_arguments.done']
                self.assertEqual([(e.item_id, e.arguments) for e in argument_done],
                                 [(c.id, c.arguments) for c in calls])
                for message in (item for item in final.output if item.type == 'message'):
                    self.assertEqual(len(message.content[0].logprobs), 1)
                    logprob = message.content[0].logprobs[0]
                    self.assertEqual(logprob.logprob, -0.25)
                    self.assertEqual(logprob.token, message.content[0].text)
                    self.assertEqual(logprob.bytes, list(message.content[0].text.encode()))
                    self.assertEqual(logprob.top_logprobs[0].bytes, list(b'other'))
                self.assertEqual(final.status, 'incomplete' if finish_reason == 'length' else 'completed')
                self.assertEqual(final.usage.input_tokens, 3)
                self.assertEqual(final.usage.output_tokens, 5)
                self.assertEqual(final.usage.input_tokens_details.cached_tokens, 1)
                self.assertEqual(final.usage.output_tokens_details.reasoning_tokens, 2)
                self.assertEqual(parser.full_parse_count, 0)
                serving.request_logger.log_outputs.assert_called_once()

                stored = await serving.retrieve_responses(final.id, None, False)
                self.assertEqual(stored.model_dump(), final.model_dump())
                followup = request(stream=False, previous_response_id=final.id)
                followup.input = [
                    {'type': 'function_call_output', 'call_id': c.call_id, 'output': 'ok'}
                    for c in calls
                ]
                messages, engine_inputs = await serving._make_request(followup, stored)
                self.assertEqual(engine_inputs, ['rendered-input'])
                historic_calls = [c for m in messages for c in m.get('tool_calls', [])]
                self.assertEqual([c['id'] for c in historic_calls], [c.call_id for c in calls])
                self.assertEqual([c['function']['arguments'] for c in historic_calls],
                                 [c.arguments for c in calls])
                self.assertEqual([m['tool_call_id'] for m in messages if m['role'] == 'tool'],
                                 [c.call_id for c in calls])
                self.assertEqual(previous_input, [{'role': 'user', 'content': 'Inspect twice.'}])
                self.assertFalse(any('reasoning' in m for m in messages))

    async def test_hidden_reasoning_and_empty_output_are_authoritative(self):
        for visible in (None, 'Visible.'):
            with self.subTest(visible=visible):
                req = request(stream=True, include_reasoning=False,
                              include=['message.output_text.logprobs'])
                parser = ScriptedParser([DeltaMessage(reasoning='Private.'),
                                         DeltaMessage(content=visible)])
                context = SimpleContext(response_parser=parser)
                serving = ServingHarness(parser)
                events = [event async for event in serving.responses_stream_generator(
                    req, SamplingParams(max_tokens=20), outputs(context, req.request_id, ['r', 'c']),
                    context, 'fixture', None, RequestResponseMetadata(request_id=req.request_id),
                )]
                final = events[-1].response
                self.assertEqual(len(final.output), int(visible is not None))
                self.assertFalse(any('reasoning' in e.type for e in events))
                self.assertEqual(final.usage.output_tokens_details.reasoning_tokens, 2)
                self.assertEqual(parser.full_parse_count, 0)
                serving._create_stream_response_logprobs.assert_not_called()
                if visible is not None:
                    self.assertEqual(final.output[0].content[0].text, visible)
                    self.assertEqual(final.output[0].content[0].logprobs, [])

    async def test_nonstreaming_still_parses_and_preserves_cancelled_store(self):
        req = request(stream=False)
        parser = ScriptedParser([], allow_full_parse=True)
        context = SimpleContext(response_parser=parser)
        serving = ServingHarness(parser)
        final = await serving.responses_full_generator(
            req, SamplingParams(max_tokens=20), outputs(context, req.request_id, ['raw']),
            context, 'fixture', None, RequestResponseMetadata(request_id=req.request_id),
        )
        self.assertEqual(parser.full_parse_count, 1)
        call = next(item for item in final.output if item.type == 'function_call')
        self.assertEqual(call.call_id, 'full-parser-id')
        cancelled = final.model_copy(update={'status': 'cancelled'})
        serving.response_store[final.id] = cancelled

        async def empty():
            if False:
                yield

        await serving.responses_full_generator(
            req, SamplingParams(max_tokens=20), empty(), context,
            'fixture', None, RequestResponseMetadata(request_id=req.request_id),
            streamed_output_items=final.output,
        )
        self.assertIs(serving.response_store[final.id], cancelled)
        self.assertEqual(parser.full_parse_count, 1)


if __name__ == '__main__':
    unittest.main()
