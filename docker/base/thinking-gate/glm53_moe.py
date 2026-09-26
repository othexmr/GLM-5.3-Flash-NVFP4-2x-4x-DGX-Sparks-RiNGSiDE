# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Local NVFP4 recipe contributors
"""Explicit GLM-5.3 parser: reasoning effort sizes the think block.

Select with both --reasoning-parser glm53 and --tool-call-parser glm53.

Thinking flags are honoured by the GLM-4.7 base class and are NOT overridden
here. That makes this parser correct only against a template that gates the
think block on the same flags -- the thinking-gated template, sha256
7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb. The launcher
pins that hash; do not pair this parser with the always-open official template
(0c4099f3...), because a client sending enable_thinking=false would stop
extraction while the template still opened <think>, leaking raw reasoning into
content. GLM-4.7 is unchanged.
"""
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.parser.engine.adapters import make_adapters

class Glm53MoeParser(Glm47MoeParser):
    # Inherits GLM-4.7 flag handling verbatim: thinking / enable_thinking
    # disable extraction, matching the gated template's <think></think>.
    pass

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
