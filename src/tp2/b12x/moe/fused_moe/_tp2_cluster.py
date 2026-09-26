# SPDX-License-Identifier: Apache-2.0
"""Explicit N1024 grid tuning. Does not reuse the N512/TP4 map."""
import json
import logging

_seen = set()
_log = logging.getLogger('vllm.b12x.tp2_cluster')
_log.setLevel(logging.INFO)


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('duplicate TP2 tuning entry')
        result[k] = v
    return result


def parse_map(raw):
    if raw in ('', '0', None):
        return {}
    data = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(data, dict) or any(
        not k.isascii() or not k.isdecimal() or str(int(k)) != k or not 1 <= int(k) <= 128
        or type(v) is not int or not 1 <= v <= 48 for k, v in data.items()
    ):
        raise ValueError('TP2 map requires M1..128 and integer CTA counts 1..48')
    return data


def select_clusters(raw, *, experts, tokens, hidden, intermediate, topk,
                    quant_mode, activation, deterministic, requested, hardware_limit):
    mapping = parse_map(raw)
    if not mapping:
        return requested
    if (experts, hidden, intermediate, topk, quant_mode, activation, deterministic) != (
            288, 4096, 1024, 8, 'nvfp4', 'silu', True):
        return requested
    selected = mapping.get(str(tokens), requested)
    if selected is not None and selected > hardware_limit:
        raise ValueError('TP2 CTA count exceeds device capacity')
    key = (tokens, selected)
    if str(tokens) in mapping and key not in _seen:
        _log.info('B12X_TP2_CLUSTER selected=True M=%d N=1024 ctAs=%d', tokens, selected)
        _seen.add(key)
    return selected
