# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain completed function calls when continuing a stored Responses turn."""

BEFORE = '''    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )
'''
AFTER = '''    if prev_response_output is not None:
        # Build this turn separately so tool merging cannot mutate prior turns.
        output_messages: list[ChatCompletionMessageParam] = []
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    output_messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )
            elif isinstance(output_item, ResponseFunctionToolCall):
                message = _construct_message_from_response_item(
                    output_item,
                    prev_msg=output_messages[-1] if output_messages else None,
                )
                if message is not None:
                    output_messages.append(message)
        messages.extend(output_messages)
'''


def transform(source):
    if source.count(BEFORE) != 1:
        raise RuntimeError('Responses stored-history anchor mismatch')
    return source.replace(BEFORE, AFTER)
