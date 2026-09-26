"""Default-off TP4-only resident-grid geometry selection.

Changing the number of resident CTAs changes scheduling of the same packed
expert weight reads. It does not quantize, prune, remap or approximate weights.
This module is copied beside an isolated B12X overlay by build_overlay.py.
"""
import json


def select_clusters(raw, *, experts, tokens, hidden, intermediate, topk,
                    quant_mode, activation, deterministic, requested, hardware_limit):
    if raw in ('', '0', None): return requested
    if (experts,hidden,intermediate,topk,quant_mode,activation,deterministic) != (
            288,4096,512,8,'nvfp4','silu',True):
        return requested
    if tokens not in (1,5,10,15,20,25,30): return requested
    config = json.loads(raw)
    if not isinstance(config,dict) or set(config)-{'1','5','10','15','20','25','30'}:
        raise ValueError('TP4 MoE cluster map has unsupported token geometries')
    if any(type(v) is not int or not 1 <= v <= 48 for v in config.values()):
        raise ValueError('TP4 MoE cluster count must be an integer in 1..48')
    selected = config.get(str(tokens))
    if selected is None: return requested
    if selected > hardware_limit: raise ValueError('cluster count exceeds hardware limit')
    return selected
