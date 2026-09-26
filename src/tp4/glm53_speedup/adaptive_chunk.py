# SPDX-License-Identifier: Apache-2.0
"""Count actual running decoders, not level labels or queued arrivals."""
from .geometry import Geometry

def chunk_cap(configured_cap,decoders,enabled,*,max_requests=6):
    Geometry(max_requests, 4)
    if type(decoders) is not int or not 0<=decoders<=max_requests:raise ValueError('decoder count outside configured slot geometry')
    if type(enabled) is not bool:raise ValueError('selector must be bool')
    if not enabled:return configured_cap
    if configured_cap!=4608:raise ValueError('adaptive chunk requires fixed fair4608 base')
    # Level 6 has FIVE incumbent decoders plus one prefill. A six-decoder-only
    # threshold could never improve that arrival under the six-slot limit.
    return 2304 if decoders>=5 else 4608
