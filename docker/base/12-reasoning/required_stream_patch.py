# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow adaptation of vLLM issue49216/PR49228: accumulate required-tool content.

Upstream proposal head349165a736e183b8666e180ead0c31005057b1e9 is not merged.
This implementation starts accumulation at tool-phase entry, preserving the
incremental reasoning path and leaving named/auto/native-tool parsing unchanged.
"""
ANCHOR = '        # Tool call extraction\n        if self._in_tool_call_phase(state):\n            if not state.tool_call_text_started:\n'
REPLACEMENT = '        # Tool call extraction\n        if self._in_tool_call_phase(state):\n            if not state.tool_call_text_started:\n                # Engine parsers consume deltas, but the generic required-tool\n                # JSON parser needs the complete content prefix across chunks.\n                # Begin accumulation here so earlier reasoning is not retained.\n                if (\n                    request.tool_choice == "required"\n                    and self._tool_parser is not None\n                    and self._tool_parser.supports_required_and_named\n                ):\n                    state.engine_based = False\n'
COUNTER_ANCHOR = '''            if (
                delta_message
                and delta_message.tool_calls
                and delta_message.tool_calls[0].id is not None
            ):
                state.history_tool_call_cnt += 1
'''
COUNTER_REPLACEMENT = '''            if delta_message and delta_message.tool_calls:
                # Each announced ID consumes one history index, including
                # several complete required-tool objects in the same delta.
                state.history_tool_call_cnt += sum(
                    tc.id is not None for tc in delta_message.tool_calls
                )
'''
def transform(source):
    if source.count(ANCHOR)!=1:raise RuntimeError('Required-tool anchor mismatch')
    if source.count(COUNTER_ANCHOR)!=1:raise RuntimeError('Tool ID counter anchor mismatch')
    return source.replace(ANCHOR,REPLACEMENT).replace(COUNTER_ANCHOR,COUNTER_REPLACEMENT)
