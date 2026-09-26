# SPDX-License-Identifier: Apache-2.0
"""Prefill ring collectives (reduce-scatter / all-gather) over our own RDMA transport for the TP4 Spark ring, both ways
round the ring with one PCIe function per direction (version bidir-1, 2026-09-24)."""
from .protocol import KIND_AG, KIND_RS, WORLD  # noqa: F401

__all__ = ["PrefillRing"]


def __getattr__(name):
    if name == "PrefillRing":
        from .ring import PrefillRing
        return PrefillRing
    raise AttributeError(name)
