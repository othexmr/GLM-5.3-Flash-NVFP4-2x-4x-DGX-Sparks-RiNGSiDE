"""One immutable engine-derived geometry shared by policy and graph capture.

Wider geometry is CPU-qualified, not a promise of GPU capacity or throughput.
No runtime override can resize already captured graphs or drafting buffers.
"""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Geometry:
    max_requests: int = 6
    max_k: int = 4

    def __post_init__(self):
        if type(self.max_requests) is not int or not 1 <= self.max_requests <= 16:
            raise ValueError('geometry max_requests must be an integer in 1..16')
        if type(self.max_k) is not int or not 1 <= self.max_k <= 8:
            raise ValueError('geometry max_k must be an integer in 1..8')

    @property
    def ks(self):return tuple(range(1, self.max_k + 1))

    @property
    def query_len(self):return self.max_k + 1

    @classmethod
    def runtime(cls, max_requests, max_k):
        geometry=cls(max_requests, max_k)
        selector=os.environ.get('GLM53_SPEEDUP_GEOMETRY', '0')
        if selector not in ('0', '1'):
            raise ValueError('invalid parametric geometry selector')
        if selector != '1' and geometry != cls():
            raise ValueError('wider geometry requires GLM53_SPEEDUP_GEOMETRY=1')
        return geometry


def capture_axes(max_requests, max_k, requests=None, ks=None):
    """Restrict adaptive shapes, while retaining native full-K for every C."""
    geometry=Geometry(max_requests, max_k)
    def axis(values, limit, label):
        if values is None:return tuple(range(1, limit + 1))
        if not isinstance(values, tuple) or not values:
            raise ValueError(label+' must be a nonempty tuple')
        if any(type(x) is not int or not 1 <= x <= limit for x in values) or tuple(sorted(set(values))) != values:
            raise ValueError(label+' must be sorted, unique and within runtime geometry')
        return values
    requests=axis(requests, max_requests, 'capture requests')
    ks=axis(ks, max_k, 'capture K')
    if max_k not in ks:raise ValueError('capture K must include native proposal K')
    return requests, ks


def capture_axes_from_env(max_requests, max_k):
    def read(key):
        value=os.environ.get(key)
        if value is None:return None
        if not value or any(not s.isdecimal() or str(int(s)) != s for s in value.split(',')):
            raise ValueError(key+' requires canonical comma-separated integers')
        return tuple(int(s) for s in value.split(','))
    return capture_axes(max_requests, max_k, read('GLM53_SPEEDUP_CAPTURE_REQUESTS'), read('GLM53_SPEEDUP_CAPTURE_KS'))
