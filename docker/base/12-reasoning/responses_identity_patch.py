# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse SimpleContext streaming output items in final and stored responses."""

REPLACEMENTS = [
    (
        '''        created_time: int | None = None,
    ) -> ErrorResponse | ResponsesResponse:
''',
        '''        created_time: int | None = None,
        *,
        streamed_output_items: list[ResponseOutputItem] | None = None,
    ) -> ErrorResponse | ResponsesResponse:
''',
    ),
    (
        '''            # TODO: Build final response items from the accumulated streaming
            # parser results instead of reparsing the complete output.
''',
        '''            # Completed streaming items retain their emitted IDs, arguments,
            # and order. Non-streaming requests still parse the complete output.
''',
    ),
    (
        '''                parser=context.response_parser,
            )
''',
        '''                parser=context.response_parser,
                streamed_output_items=streamed_output_items,
            )
''',
    ),
    (
        '''        parser: Parser | None = None,
    ) -> list[ResponseOutputItem]:
''',
        '''        parser: Parser | None = None,
        *,
        streamed_output_items: list[ResponseOutputItem] | None = None,
    ) -> list[ResponseOutputItem]:
''',
    ),
    (
        '''        # Compute logprobs if requested
        logprobs = None
''',
        '''        # Reuse exactly the completed items delivered to the client. An
        # empty streamed output is authoritative too; it must not be reparsed.
        if streamed_output_items is not None:
            return streamed_output_items

        # Compute logprobs if requested
        logprobs = None
''',
    ),
    (
        '''            try:
                async for event_data in processor(
''',
        '''            streamed_output_items: list[ResponseOutputItem] | None = (
                [] if isinstance(context, SimpleContext) else None
            )
            try:
                async for event_data in processor(
''',
    ),
    (
        '''                ):
                    yield event_data
            except GenerationError as e:
''',
        '''                ):
                    if (
                        streamed_output_items is not None
                        and event_data.type == "response.output_item.done"
                    ):
                        if event_data.output_index != len(streamed_output_items):
                            raise ValueError("Non-sequential completed output item")
                        streamed_output_items.append(event_data.item.model_copy(deep=True))
                    yield event_data
            except GenerationError as e:
''',
    ),
    (
        '''                created_time=created_time,
            )
            yield _increment_sequence_number_and_return(
                ResponseCompletedEvent(
''',
        '''                created_time=created_time,
                streamed_output_items=streamed_output_items,
            )
            yield _increment_sequence_number_and_return(
                ResponseCompletedEvent(
''',
    ),
]


def transform(source):
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise RuntimeError('Responses identity anchor mismatch: ' + before[:80])
        source = source.replace(before, after)
    return source
