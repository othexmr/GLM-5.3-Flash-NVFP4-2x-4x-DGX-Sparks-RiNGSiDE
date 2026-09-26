# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
"""Actual request/grammar/worker-gate regression; CPU only, no model requests."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser import ParserManager
from vllm.parser.glm53_moe import Glm53MoeParser, Glm53MoeParserToolAdapter
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.tool_parsers.utils import get_json_schema_from_tools
from vllm.v1.structured_output import StructuredOutputManager
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string


SPECIAL = ["<think>", "</think>", "<tool_call>", "</tool_call>",
           "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]


class Tokenizer:
    all_special_tokens = SPECIAL
    all_special_ids = list(range(1000, 1000 + len(SPECIAL)))

    def get_vocab(self):
        return {**{chr(i): i for i in range(256)},
                **dict(zip(self.all_special_tokens, self.all_special_ids))}

    def encode(self, text, **kwargs):
        result = []
        while text:
            special = next((s for s in SPECIAL if text.startswith(s)), None)
            if special:
                result.append(self.get_vocab()[special])
                text = text[len(special):]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        inverse = dict(zip(self.all_special_ids, SPECIAL))
        return "".join(inverse.get(i, chr(i)) for i in ids)


SCHEMA = {"type": "object", "properties": {
    "label": {"type": "string", "enum": ["alpha", "beta"]},
    "count": {"type": "integer"}},
    "required": ["label", "count"], "additionalProperties": False}
# Thinking-on grammar transitions must explicitly keep reasoning enabled.
FLAGS = {"thinking": True, "enable_thinking": True, "reasoning_effort": "low"}
tok = Tokenizer()
parser_cls = ParserManager.get_parser(tool_parser_name="glm53",
    reasoning_parser_name="glm53", enable_auto_tools=True)
assert ToolParserManager.get_tool_parser("glm53") is Glm53MoeParserToolAdapter
assert Glm53MoeParser.tool_parser_cls is Glm53MoeParserToolAdapter
assert Glm53MoeParserToolAdapter.supports_required_and_named is True


def request_for(api, choice, strict, *, tools=True):
    function = {"name": "inspect", "parameters": deepcopy(SCHEMA)}
    if strict is not None:
        function["strict"] = strict
    if api == "chat":
        tool_choice = ({"type": "function", "function": {"name": "inspect"}}
                       if choice == "named" else choice)
        tool_args = {"tools": [{"type": "function", "function": function}],
                     "tool_choice": tool_choice} if tools else {}
        return ChatCompletionRequest(model="glm53", messages=[
            {"role": "user", "content": "Inspect."}],
            response_format={"type": "json_object"},
            chat_template_kwargs=dict(FLAGS), **tool_args)
    tool_choice = ({"type": "function", "name": "inspect"}
                   if choice == "named" else choice)
    tool_args = {"tools": [{"type": "function", **function}],
                 "tool_choice": tool_choice} if tools else {}
    return ResponsesRequest(model="glm53", input="Inspect.",
        text={"format": {"type": "json_object"}},
        chat_template_kwargs=dict(FLAGS), **tool_args)


def parser_for(request):
    return parser_cls(tok, tools=request.tools,
                      chat_template_kwargs=request.chat_template_kwargs)


def native_call(label="alpha", count="2", name="inspect", extra=""):
    return (f"<tool_call>{name}<arg_key>label</arg_key>"
            f"<arg_value>{label}</arg_value><arg_key>count</arg_key>"
            f"<arg_value>{count}</arg_value>{extra}</tool_call>")


adjusted_cases = 0
strict_requests = {}
for api in ("chat", "responses"):
    for enabled in (False, True):
        with patch.object(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", enabled):
            for choice in ("auto", "required", "named"):
                for strict in (None, False, True):
                    request = request_for(api, choice, strict)
                    expected_json = get_json_schema_from_tools(
                        request.tool_choice, request.tools)
                    parser = parser_for(request)
                    assert parser.reasoning_parser._parser_engine.thinking_enabled
                    assert parser.tool_parser._parser_engine.thinking_enabled
                    assert parser.tool_parser.supports_required_and_named is True
                    assert parser.adjust_request(request) is request
                    assert request.skip_special_tokens is False
                    assert request.chat_template_kwargs == FLAGS
                    params = request.structured_outputs
                    native_tag = None if params is None else params.structural_tag
                    if choice == "auto" and strict is True and enabled:
                        assert native_tag is not None, (api, enabled, choice, strict)
                        expected = get_model_structural_tag(model="glm_4_7",
                            tools=request.tools, tool_choice="auto", reasoning=False)
                        assert json.loads(native_tag) == expected.model_dump()
                        assert params.json is None
                        if api == "chat":
                            assert request.response_format is None
                        else:
                            assert request.text is None
                            assert request.extract_structured_outputs() is params
                        strict_requests[api] = request
                    else:
                        assert native_tag is None, (api, enabled, choice, strict)
                        if choice == "auto":
                            assert expected_json is None and params is None
                            if api == "chat":
                                assert request.response_format.type == "json_object"
                            else:
                                assert request.text.format.type == "json_object"
                        else:
                            if api == "chat":
                                assert params.json == expected_json
                                assert request.response_format is None
                            else:
                                assert params is None
                                assert request.text.format.type == "json_schema"
                                assert request.extract_structured_outputs().json == expected_json
                            # Required/named must still parse the generic JSON wire
                            # format when strict enforcement is enabled or disabled.
                            args = {"label": "alpha", "count": 2}
                            body = json.dumps(args) if choice == "named" else json.dumps([
                                {"name": "inspect", "parameters": args},
                                {"name": "inspect", "parameters": args}])
                            reason, content, calls = parser.parse(
                                "private</think>" + body, request, enable_auto_tools=True)
                            assert reason == "private" and content is None
                            assert len(calls) == (1 if choice == "named" else 2)
                            assert all(c.name == "inspect" and json.loads(c.arguments) == args
                                       for c in calls)
                    adjusted_cases += 1
            for choice, with_tools in (("none", True), ("auto", False)):
                request = request_for(api, choice, True, tools=with_tools)
                parser_for(request).adjust_request(request)
                assert request.structured_outputs is None
                adjusted_cases += 1


grammar_cases = 0
for api, request in strict_requests.items():
    grammar = Grammar.from_structural_tag(request.structured_outputs.structural_tag)
    xml = native_call() + native_call("beta", "3")
    for body in ("Ordinary answer.", xml, "Checking. " + xml + " Done."):
        assert _is_grammar_accept_string(grammar, body), (api, body)
        grammar_cases += 1
    invalid = [
        native_call(name="unknown"),
        native_call(label="wrong-enum"),
        native_call(count="not-an-integer"),
        "<tool_call>inspect<arg_key>label</arg_key><arg_value>alpha</arg_value></tool_call>",
        native_call(extra="<arg_key>extra</arg_key><arg_value>1</arg_value>"),
        "private</think>" + xml,
    ]
    for body in invalid:
        assert not _is_grammar_accept_string(grammar, body), (api, body)
        grammar_cases += 1
    reason, content, calls = parser_for(request).parse(
        "private</think>" + xml, request, enable_auto_tools=True)
    assert reason == "private" and not content
    assert [c.name for c in calls] == ["inspect", "inspect"]
    assert [json.loads(c.arguments) for c in calls] == [
        {"label": "alpha", "count": 2}, {"label": "beta", "count": 3}]

    # Exercise the installed worker-side methods without constructing an engine.
    # The same delta contains the close marker and grammar text, as in spec decode.
    manager = object.__new__(StructuredOutputManager)
    manager.reasoner_cls = ReasoningParserManager.get_reasoning_parser("glm53")
    manager.tokenizer = tok
    manager.enable_in_reasoning = False
    prompt = tok.encode("<think>")
    worker = SimpleNamespace(use_structured_output=True, prompt_token_ids=prompt,
        all_token_ids=list(prompt), structured_output_request=SimpleNamespace(
            reasoner=None, reasoning_ended=None, reasoning_end_token_index=None,
            reasoning_parser_kwargs={"chat_template_kwargs": dict(FLAGS)}))
    assert manager.should_fill_bitmask(worker) is False
    private = tok.encode("private")
    worker.all_token_ids.extend(private)
    assert manager.should_advance(worker, private) is False
    assert manager.should_fill_bitmask(worker) is False
    end_and_xml = tok.encode("</think>" + xml)
    worker.all_token_ids.extend(end_and_xml)
    assert manager.should_advance(worker, end_and_xml) is True
    assert manager.should_fill_bitmask(worker) is True
    suffix = manager.trim_reasoning_for_advance(worker, end_and_xml)
    assert suffix == tok.encode(xml)
    assert _is_grammar_accept_string(grammar, tok.decode(suffix))

off_cases = 0
for api in ("chat", "responses"):
    for choice in ("auto", "required", "named"):
        request = request_for(api, choice, True)
        request.chat_template_kwargs = {"thinking": False, "enable_thinking": False}
        parser = parser_for(request)
        assert not parser.reasoning_parser._parser_engine.thinking_enabled
        # ParserManager gives the tool adapter its own engine and forwards
        # chat_template_kwargs only to the reasoning one. The tool split is
        # driven by prompt_token_ids (<think></think>), so this internal flag
        # is not load-bearing; the reasoning engine above is the contract.
        args = {"label": "alpha", "count": 2}
        body = (native_call() if choice == "auto" else json.dumps(args) if choice == "named"
                else json.dumps([{"name": "inspect", "parameters": args}]))
        reason, content, calls = parser.parse(body, request, enable_auto_tools=True)
        assert not reason and not content, (api, choice, reason, content)
        assert len(calls) == 1 and calls[0].name == "inspect"
        assert json.loads(calls[0].arguments) == args
        off_cases += 1

print(json.dumps({"passed": True, "request_adjustment_cases": adjusted_cases,
    "thinking_off_tool_cases": off_cases,
    "grammar_accept_reject_cases": grammar_cases,
    "chat_and_responses_strict_auto": True, "generic_required_named_preserved": True,
    "thinking_on_worker_gate": True,
    "scope": "actual installed requests, parser registry, xgrammar and worker methods; CPU only"}))
