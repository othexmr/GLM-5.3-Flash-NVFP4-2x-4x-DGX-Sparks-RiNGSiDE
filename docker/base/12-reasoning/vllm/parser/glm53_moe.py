# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Local NVFP4 recipe contributors
"""Explicit GLM-5.3 parser: reasoning effort sizes an always-open think block.

Select with both --reasoning-parser glm53 and --tool-call-parser glm53.
Requires the official GLM-5.3 prompt contract, ending in <think>. Legacy
thinking/enable_thinking flags do not disable extraction. GLM-4.7 is unchanged.
"""
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.parser.engine.adapters import make_adapters

class Glm53MoeParser(Glm47MoeParser):
    def __init__(self, tokenizer, tools=None, **kwargs):
        kwargs = dict(kwargs)
        chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        chat_kwargs.update(thinking=True, enable_thinking=True)
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, tools, **kwargs)

Glm53MoeParserReasoningAdapter, _Glm53MoeParserToolAdapter = make_adapters(Glm53MoeParser)


class Glm53MoeParserToolAdapter(_Glm53MoeParserToolAdapter):
    # GLM-5.3 retains the GLM-4.7/5 native XML format for auto tool calls.
    structural_tag_model = "glm_4_7"

    def get_structural_tag(self, request, *, reasoning=False):
        if request.tool_choice != "auto":
            return None
        return super().get_structural_tag(request, reasoning=reasoning)


# ToolParser's subclass hook assumes a native grammar also replaces the
# required/named JSON paths. This adapter deliberately constrains only auto;
# retain the repaired generic JSON contract for required and named choices.
Glm53MoeParserToolAdapter.supports_required_and_named = True
Glm53MoeParser.tool_parser_cls = Glm53MoeParserToolAdapter
