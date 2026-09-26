# SPDX-License-Identifier: Apache-2.0
"""Select original logprob entries only when their text span is unambiguous.

Never re-tokenize parsed text: that can change BPE boundaries and assign a
probability from a different generated token. Omit metadata for ambiguous or
partial-token spans instead of publishing shifted probabilities.
"""

def content_logprob_span(raw_text, content, entries, start=0):
    if not content or not entries:
        return [], start
    pos = raw_text.find(content, start)
    if pos < 0 or raw_text.find(content, pos + 1) >= 0:
        return [], start
    end = pos + len(content)
    offset = 0
    lo = hi = None
    prefix = []
    for index, entry in enumerate(entries):
        if offset == pos and lo is None:
            lo = index
        prefix.append(entry.token)
        offset += len(entry.token)
        if offset == end:
            hi = index + 1
            break
    if lo is None or hi is None:
        return [], end
    # EOS/stop-token logprobs may follow displayed text. Verify the exact
    # consumed prefix, while retaining original token entries and probabilities.
    if "".join(prefix) != raw_text[:end]:
        return [], end
    selected = entries[lo:hi]
    if "".join(entry.token for entry in selected) != content:
        return [], end
    return selected, end
