# SPDX-License-Identifier: Apache-2.0
"""Boot-time tuning map for full-width TP2 mHC; no change when disabled."""
import json
import logging

_seen = set()
_log = logging.getLogger('vllm.mhc.tp2_geometry')
_log.setLevel(logging.INFO)


def parse_map(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('duplicate mHC shape')
            result[k] = v
        return result
    data = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(data, dict):
        raise ValueError('mHC tuning must be an object')
    for k, v in data.items():
        if (not k.isascii() or not k.isdecimal() or str(int(k)) != k or not 1 <= int(k) <= 16
            or not isinstance(v, list) or len(v) != 2 or any(type(x) is not int for x in v)
            or v[0] not in (2, 3, 4, 6, 8, 12) or v[1] not in (1, 2, 4, 8)):
            raise ValueError('mHC map requires M1..16: [tile_n, splits] with legal small-kernel divisors')
    return {int(k): tuple(v) for k, v in data.items()}


def select_geometry(mapping, tokens, hidden_size, streams, normalized, tp_size):
    if tp_size != 2 or hidden_size != 4096 or streams != 4 or not normalized:
        return None
    value = mapping.get(tokens)
    if value is not None:
        tile_n, splits = value
        if hidden_size % (splits * 256) or (streams * streams + 2 * streams) % tile_n:
            raise ValueError('unsafe mHC reduction geometry')
        key = (tokens, tile_n, splits)
        if key not in _seen:
            _log.info('GLM53_TP2_MHC selected=True M=%d tile_n=%d splits=%d', *key)
            _seen.add(key)
    return value
