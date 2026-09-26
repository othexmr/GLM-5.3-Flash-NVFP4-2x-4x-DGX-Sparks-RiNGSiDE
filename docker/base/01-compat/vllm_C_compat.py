"""Compatibility import for the legacy ``vllm._C`` module name.

CUDA extensions at the pinned glm-release commit are intentionally built as
``_C_stable_libtorch``.  Importing that binary registers the same torch ops;
this module preserves the explicit legacy import capability gate without
duplicating or rebuilding any CUDA kernels.
"""

from . import _C_stable_libtorch as _extension

__all__ = ["_extension"]
