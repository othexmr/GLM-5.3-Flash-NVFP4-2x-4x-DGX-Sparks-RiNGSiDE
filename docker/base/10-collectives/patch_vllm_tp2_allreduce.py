#!/usr/bin/env python3
"""Anchor-checked patch: replace the two-rank small-message TP all-reduce with one exchange.

Every decode step of GLM-5.3 Flash NVFP4 under TP2 issues 102 all-reduces of 64 KB bf16
(45 layers x 2, drafter 5 x 2, 2 vocab embeddings), each a `RING_LL` ring all-reduce that
NCCL executes as two dependent network steps (reduce-scatter half, all-gather half) and that
costs 34-44 us on the 200 GbE RoCE rail while the wire time is about 2.6 us. With two ranks
the same result is one exchange: each rank sends its partial sum to the peer, receives the
peer's partial, and adds locally (one network step + one add kernel). This patch installs
that exchange behind `VLLM_GLM53_TP2_ALLREDUCE`:

    nccl  (default, unset)  unchanged: pynccl ncclAllReduce on the current stream
    p2p                     ncclGroupStart / ncclSend(own) / ncclRecv(peer) / ncclGroupEnd + add
    ag                      ncclAllGather of both partials into a persistent [2, ...] buffer + add

It applies only when the group has exactly two ranks, the tensor is contiguous CUDA
bf16/fp16/fp32, and its byte size is <= VLLM_GLM53_TP2_ALLREDUCE_MAX_BYTES (default
262144 = 256 KB, i.e. up to 32 decode tokens at hidden 4,096); larger tensors (prefill
chunks of up to 14,336 tokens = 117 MB) keep the NCCL all-reduce. The decision is a pure
function of (tensor shape, dtype, environment) so both ranks always agree on which NCCL
call they issue on the shared communicator.

CUDA graphs: the exchange runs on the current stream through the same PyNcclCommunicator
that vLLM already captures inside FULL decode graphs; NCCL send/recv and all-gather are
graph-capturable like all-reduce (NCCL >= 2.9 captures the kernel and re-issues the proxy
work through a host node at every replay). Receive buffers are persistent per
(shape, dtype, stream) so replays see stable addresses; the output is a fresh tensor (like
pynccl's out-of-place all-reduce), never the persistent buffer. The first exchange for a
group must run eagerly (NCCL sets up the point-to-point connections lazily on first use,
which is a blocking host-side operation); vLLM's profile run and pre-capture warm-up do
that, and if the first call ever lands inside a capture the code falls back to the NCCL
all-reduce for that call on both ranks.

Anchors are written from vLLM's `vllm/distributed/device_communicators/cuda_communicator.py`
at the 2025-2026 layout (CudaCommunicator.all_reduce: symm-mem, quick-reduce, custom
all-reduce, then `pynccl_comm.all_reduce(input_)`). TODO markers name what must be
confirmed against the image file before building; `--check` prints, inside the image,
whether each anchor matches and shows the neighbouring lines when it does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

# TODO(anchor): confirm the path inside the a118 image:
#   docker run --rm --network none --entrypoint find local/glm53-redhat-b12x:4500c80c-a118-dflash2-edge-scale \
#     /usr/local/lib/python3.12/dist-packages/vllm -name cuda_communicator.py
COMMUNICATOR = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py"
)

# TODO(anchor): the tail of CudaCommunicator.all_reduce. Confirm with
#   grep -n -B2 -A6 "pynccl_comm.all_reduce(input_)" <COMMUNICATOR>
# and adjust the three lines below (indentation included) if the image differs.
PYNCCL_ANCHOR = """        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_)
"""

PYNCCL_REPLACEMENT = """        assert pynccl_comm is not None
        # GLM-5.3 Flash NVFP4 lane: VLLM_GLM53_TP2_ALLREDUCE=p2p|ag replaces the two-rank
        # all-reduce of small messages with one send/recv pair (or one all-gather) and a
        # local add on the same pynccl communicator and stream. Default nccl = unchanged.
        _glm53_tp2 = self._glm53_tp2_exchange(pynccl_comm)
        if _glm53_tp2 is not None and _glm53_tp2.applicable(input_):
            return _glm53_tp2.all_reduce(input_)
        out = pynccl_comm.all_reduce(input_)
"""

# TODO(anchor): the class statement. Confirm with grep -n "^class CudaCommunicator" <COMMUNICATOR>
CLASS_ANCHOR = "class CudaCommunicator(DeviceCommunicatorBase):\n"

CLASS_REPLACEMENT = '''class CudaCommunicator(DeviceCommunicatorBase):
    def _glm53_tp2_exchange(self, pynccl_comm):
        """GLM-5.3 lane: lazily built two-rank exchange (None when inactive)."""
        state = self.__dict__.get("_glm53_tp2_state", None)
        if state is None:
            try:
                state = _Glm53Tp2Exchange.create(pynccl_comm) or False
            except Exception:
                _glm53_logger.exception("GLM53_TP2_ALLREDUCE disabled after an error at setup")
                state = False
            self.__dict__["_glm53_tp2_state"] = state
        return state or None

'''

HELPER = '''
# --- GLM-5.3 Flash NVFP4 lane: TP2 small-message all-reduce replacement ---------------
import logging as _glm53_logging
import os as _glm53_os

_GLM53_TP2_MODE = (
    _glm53_os.environ.get("VLLM_GLM53_TP2_ALLREDUCE", "nccl").strip().lower() or "nccl"
)
if _GLM53_TP2_MODE not in ("nccl", "p2p", "ag"):
    raise ValueError(
        "VLLM_GLM53_TP2_ALLREDUCE must be nccl, p2p, or ag, got %r" % _GLM53_TP2_MODE
    )
_GLM53_TP2_MAX_BYTES = int(
    _glm53_os.environ.get("VLLM_GLM53_TP2_ALLREDUCE_MAX_BYTES", "262144")
)
_GLM53_TP2_LOG_CALLS = int(_glm53_os.environ.get("VLLM_GLM53_TP2_ALLREDUCE_LOG_CALLS", "4"))
_glm53_logger = _glm53_logging.getLogger(__name__)


class _Glm53Tp2Exchange:
    """Two-rank replacement for a small all-reduce on a PyNcclCommunicator.

    p2p: group(send own, recv peer) then out = own + peer.
    ag:  all_gather both partials into a persistent [2, *shape] buffer, out = g[0] + g[1].
    Both ranks compute own + peer in the same IEEE order, so the two residual streams stay
    bit-identical across ranks (as with NCCL's ring). Receive buffers are persistent per
    (shape, dtype, stream); the output is always a fresh tensor.
    """

    def __init__(self, pynccl_comm, mode, max_bytes):
        self.comm = pynccl_comm
        self.mode = mode
        self.max_bytes = max_bytes
        self.rank = int(pynccl_comm.rank)
        self.peer = 1 - self.rank
        self.buffers = {}
        self.warm = False
        self.calls = 0
        self.capture_fallbacks = 0

    @classmethod
    def create(cls, pynccl_comm):
        mode = _GLM53_TP2_MODE
        if mode == "nccl":
            return None
        if pynccl_comm is None or getattr(pynccl_comm, "disabled", True):
            _glm53_logger.warning("GLM53_TP2_ALLREDUCE disabled: pynccl communicator unavailable")
            return None
        world_size = int(getattr(pynccl_comm, "world_size", 0))
        if world_size != 2:
            _glm53_logger.info("GLM53_TP2_ALLREDUCE inactive for a %d-rank group", world_size)
            return None
        import inspect as _inspect

        def has(method, names=()):
            fn = getattr(pynccl_comm, method, None)
            if fn is None or not callable(fn):
                return False
            if not names:
                return True
            try:
                params = _inspect.signature(fn).parameters
            except (TypeError, ValueError):
                return False
            return all(name in params for name in names)

        # TODO(pynccl): confirm these names in vllm/distributed/device_communicators/pynccl.py
        # (send(tensor, dst, stream=None), recv(tensor, src, stream=None), group_start(),
        # group_end(), all_gather(output_tensor, input_tensor, stream=None)).
        if mode == "p2p":
            ok = (
                has("send", ("tensor", "dst"))
                and has("recv", ("tensor", "src"))
                and has("group_start")
                and has("group_end")
            )
        else:
            ok = has("all_gather", ("output_tensor", "input_tensor"))
        if not ok:
            _glm53_logger.warning(
                "GLM53_TP2_ALLREDUCE disabled: PyNcclCommunicator lacks the %s methods", mode
            )
            return None
        _glm53_logger.info(
            "GLM53_TP2_ALLREDUCE active mode=%s max_bytes=%d rank=%d peer=%d",
            mode, _GLM53_TP2_MAX_BYTES, int(pynccl_comm.rank), 1 - int(pynccl_comm.rank),
        )
        return cls(pynccl_comm, mode, _GLM53_TP2_MAX_BYTES)

    def applicable(self, tensor):
        if not tensor.is_cuda or not tensor.is_contiguous():
            return False
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes == 0 or nbytes > self.max_bytes:
            return False
        if not self.warm and torch.cuda.is_current_stream_capturing():
            # First use must be eager (NCCL connects send/recv peers lazily, on the host);
            # deterministic on both ranks, so both issue the same NCCL all-reduce instead.
            self.capture_fallbacks += 1
            return False
        return True

    def _buffer(self, tensor, lead):
        key = (lead, tuple(tensor.shape), tensor.dtype, torch.cuda.current_stream().cuda_stream)
        buffer = self.buffers.get(key)
        if buffer is None:
            shape = ((lead,) + tuple(tensor.shape)) if lead else tuple(tensor.shape)
            buffer = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
            self.buffers[key] = buffer
        return buffer

    def all_reduce(self, tensor):
        comm = self.comm
        if self.mode == "p2p":
            received = self._buffer(tensor, 0)
            comm.group_start()
            comm.send(tensor=tensor, dst=self.peer)
            comm.recv(tensor=received, src=self.peer)
            comm.group_end()
            out = torch.add(tensor, received)
        else:
            gathered = self._buffer(tensor, 2)
            comm.all_gather(output_tensor=gathered, input_tensor=tensor)
            out = torch.add(gathered[0], gathered[1])
        self.warm = True
        self.calls += 1
        if self.calls <= _GLM53_TP2_LOG_CALLS:
            _glm53_logger.info(
                "GLM53_TP2_ALLREDUCE call %d mode=%s shape=%s dtype=%s capturing=%s",
                self.calls, self.mode, tuple(tensor.shape), tensor.dtype,
                torch.cuda.is_current_stream_capturing(),
            )
        return out
# --- end GLM-5.3 lane block -------------------------------------------------------------


'''


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_source(source: str) -> str:
    if "_Glm53Tp2Exchange" in source:
        raise RuntimeError("source already carries the GLM-5.3 TP2 exchange block")
    source = replace_once(source, PYNCCL_ANCHOR, PYNCCL_REPLACEMENT)
    source = replace_once(source, CLASS_ANCHOR, HELPER + CLASS_REPLACEMENT)
    return source


def check_anchors(source: str) -> bool:
    """Report each anchor; on a miss print the closest lines so the anchor can be fixed."""
    ok = True
    for label, anchor, needle in (
        ("pynccl all_reduce tail", PYNCCL_ANCHOR, "pynccl_comm.all_reduce("),
        ("class statement", CLASS_ANCHOR, "class CudaCommunicator"),
    ):
        count = source.count(anchor)
        print(f"{label}: {'OK' if count == 1 else 'MISSING'} (count={count})")
        if count != 1:
            ok = False
            for number, line in enumerate(source.splitlines(), 1):
                if needle in line:
                    lines = source.splitlines()
                    lo, hi = max(0, number - 4), min(len(lines), number + 4)
                    print(f"  candidate near line {number}:")
                    for k in range(lo, hi):
                        print(f"  {k + 1:5d}| {lines[k]}")
    if "_Glm53Tp2Exchange" in source:
        print("already patched: the GLM-5.3 TP2 block is present")
        ok = False
    return ok


def main(argv: list[str]) -> int:
    if "--check" in argv:
        return 0 if check_anchors(COMMUNICATOR.read_text()) else 1
    source = patch_source(COMMUNICATOR.read_text())
    compile(source, str(COMMUNICATOR), "exec")
    if "--dry-run" in argv:
        sys.stdout.write(source)
        return 0
    COMMUNICATOR.write_text(source)
    print("GLM-5.3 TP2 all-reduce exchange switch installed (VLLM_GLM53_TP2_ALLREDUCE, default nccl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
