# SPDX-License-Identifier: Apache-2.0
"""Block-aligned fair chunks. TP2 never emits a sub-4608 long chunk."""
from .geometry import Geometry


def chunk_cap(configured_cap, decoders, enabled, *, max_requests=6, block_size=2304):
    Geometry(max_requests, 4)
    if type(decoders) is not int or not 0 <= decoders <= max_requests:
        raise ValueError('decoder count outside configured slot geometry')
    if type(enabled) is not bool:
        raise ValueError('selector must be bool')
    if not enabled:
        return configured_cap
    if type(block_size) is not int or block_size not in (2304, 4608):
        raise ValueError('unqualified hybrid block size')
    # A fixed 4608 TP2 cap is already one block: safe no-op, not a speed claim.
    if type(configured_cap) is not int or configured_cap not in (block_size, 2 * block_size):
        raise ValueError('adaptive cap must be one or two hybrid blocks')
    return block_size if decoders >= 5 else configured_cap
