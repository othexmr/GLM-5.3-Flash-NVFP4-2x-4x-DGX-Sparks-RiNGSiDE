# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep active tool tails before transitions and retain emitted text logprobs."""

REPLACEMENTS = [
    (
        '''    """Decompose a DeltaMessage with multiple fields into atomic deltas.

    The Responses API emits typed SSE events (one type per event), so a
    compound DeltaMessage must be split before entering the state machine.
    Order: reasoning -> content -> tool_calls (grouped by index).
''',
        '''    """Decompose compound deltas without closing an active tool before its tail.

    Nameless tool groups continue an already-open call. Emit those before
    post-tool reasoning/content, then emit groups that introduce new calls.
    A nameless orphan is still rejected by the event processor; never invent
    a function name or attach it to a different call.
''',
    ),
    (
        '''    deltas: list[DeltaMessage] = []
    if has_reasoning:
        deltas.append(DeltaMessage(reasoning=delta.reasoning))
    if has_content:
        deltas.append(DeltaMessage(content=delta.content))
    if has_tools:
        groups: dict[int | None, list[DeltaToolCall]] = {}
        for tc in delta.tool_calls:
            groups.setdefault(tc.index, []).append(tc)
        for tcs in groups.values():
            deltas.append(DeltaMessage(tool_calls=tcs))
    return deltas or [delta]
''',
        '''    continuations: list[DeltaMessage] = []
    new_calls: list[DeltaMessage] = []
    if has_tools:
        groups: dict[int | None, list[DeltaToolCall]] = {}
        for tc in delta.tool_calls:
            groups.setdefault(tc.index, []).append(tc)
        for tcs in groups.values():
            target = (
                new_calls
                if any(
                    tc.function is not None and tc.function.name is not None
                    for tc in tcs
                )
                else continuations
            )
            target.append(DeltaMessage(tool_calls=tcs))

    deltas = continuations
    if has_reasoning:
        deltas.append(DeltaMessage(reasoning=delta.reasoning))
    if has_content:
        deltas.append(DeltaMessage(content=delta.content))
    deltas.extend(new_calls)
    return deltas or [delta]
''',
    ),
    (
        '''    accumulated_text: str = ""
    tool_call_id: str = ""
''',
        '''    accumulated_text: str = ""
    accumulated_logprobs: list[response_text_delta_event.Logprob] = field(
        default_factory=list
    )
    tool_call_id: str = ""
''',
    ),
    (
        '''    state.current_state = _StateType.CONTENT
    state.current_item_id = random_uuid()
    state.content_index = 0
    state.accumulated_text = ""
''',
        '''    state.current_state = _StateType.CONTENT
    state.current_item_id = random_uuid()
    state.content_index = 0
    state.accumulated_text = ""
    state.accumulated_logprobs = []
''',
    ),
    (
        '''    state.accumulated_text += delta
    return [
        ResponseTextDeltaEvent(
''',
        '''    state.accumulated_text += delta
    state.accumulated_logprobs.extend(logprobs or [])
    return [
        ResponseTextDeltaEvent(
''',
    ),
    (
        '''    part = ResponseOutputText(
        type="output_text",
        text=state.accumulated_text,
        annotations=[],
    )
''',
        '''    part = ResponseOutputText(
        type="output_text",
        text=state.accumulated_text,
        annotations=[],
        logprobs=[
            dict(
                logprob.model_dump(),
                bytes=list(logprob.token.encode("utf-8", errors="replace")),
                top_logprobs=[
                    dict(
                        top.model_dump(),
                        bytes=list(top.token.encode("utf-8", errors="replace")),
                    )
                    for top in logprob.top_logprobs or []
                ],
            )
            for logprob in state.accumulated_logprobs
        ],
    )
''',
    ),
    (
        '''            text=state.accumulated_text,
            logprobs=[],
            item_id=state.current_item_id,
''',
        '''            text=state.accumulated_text,
            logprobs=[logprob.model_dump() for logprob in state.accumulated_logprobs],
            item_id=state.current_item_id,
''',
    ),
]


def transform(source):
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise RuntimeError('Responses streaming anchor mismatch: ' + before[:80])
        source = source.replace(before, after)
    return source
